#!/usr/bin/env python
"""Merge per-host shard outputs into unified rank0 + summary.

Each shard ran eval_k_sweep.py / eval_v5_pareto.py with --idx_offset/--idx_stride
and a unique --output_name suffix like 'ksweep_full58k_sh{N}'. Each shard
produced rank files: {base}_sh{N}_rank{R}.json (per local GPU).

This script:
  1. Globs all matching shard rank files.
  2. Concatenates per-key sample lists.
  3. Writes:
     - {base}_rank0.json  (concatenated, used by score_rouge1_sac_aligned.py)
     - {base}_summary.json (per-k or per-label aggregate)
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def merge_shards(exp_dir: str, base: str) -> None:
    pattern = os.path.join(exp_dir, "output", f"{base}_sh*_rank*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"no shard files match {pattern}")

    print(f"[merge] {base}: found {len(files)} shard rank files")
    merged: dict[str, list] = defaultdict(list)
    # Dedup on (key, sample id) -- stale rank files from a previous run with a
    # different world_size/shard split would otherwise be silently
    # double-counted (2026-06-13, F64).
    seen_ids: set = set()
    n_dups = 0
    for fp in files:
        d = json.load(open(fp))
        for k, samples in d.items():
            for s in samples:
                sid = (k, s.get("id"), s.get("dataset"))
                if sid[1] is not None and sid in seen_ids:
                    n_dups += 1
                    continue
                seen_ids.add(sid)
                merged[k].append(s)
    if n_dups:
        print(f"[merge][WARN] {base}: dropped {n_dups} duplicate (key,id) samples "
              f"-- stale shard files likely present, check the glob inputs!")

    n_per_key = {k: len(v) for k, v in merged.items()}
    print(f"[merge] {base}: per-key counts: {n_per_key}")
    if len(set(n_per_key.values())) > 1:
        print(f"[merge][WARN] {base}: per-key counts differ across keys "
              f"-- incomplete shard set for some keys?")

    rank0_path = os.path.join(exp_dir, "output", f"{base}_rank0.json")
    with open(rank0_path, "w") as f:
        json.dump(dict(merged), f, ensure_ascii=False)
    print(f"[merge] wrote {rank0_path}")

    summary = {"overall": {}, "per_dataset_per_k": {}, "merged_from": files}
    for key, rs in merged.items():
        if not rs:
            continue
        f1s = [r.get("f1", 0.0) for r in rs]
        ems = [r.get("em", 0.0) for r in rs]
        summary["overall"][key] = {
            "f1": float(np.mean(f1s)),
            "em": float(np.mean(ems)),
            "count": len(rs),
        }
        per_ds_f1 = defaultdict(list)
        per_ds_em = defaultdict(list)
        for r in rs:
            ds = r.get("dataset", "?")
            per_ds_f1[ds].append(r.get("f1", 0.0))
            per_ds_em[ds].append(r.get("em", 0.0))
        summary["per_dataset_per_k"][key] = {
            ds: {
                "f1": float(np.mean(per_ds_f1[ds])),
                "em": float(np.mean(per_ds_em[ds])),
                "n": len(per_ds_f1[ds]),
            }
            for ds in per_ds_f1
        }

    summary_path = os.path.join(exp_dir, "output", f"{base}_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[merge] wrote {summary_path}")

    print(f"[merge] {base} done")
    print("-" * 70)
    print(f"{'key':>16} {'N':>8} {'F1':>8} {'EM':>8}")
    for key, m in summary["overall"].items():
        print(f"{key:>16} {m['count']:>8} {m['f1']:>8.4f} {m['em']:>8.4f}")
    print("-" * 70)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--bases", required=True,
                   help="comma-separated base output names, e.g. 'ksweep_full58k,pareto_full58k_eps0.10'")
    args = p.parse_args()

    for base in args.bases.split(","):
        base = base.strip()
        if not base:
            continue
        merge_shards(args.exp_dir, base)


if __name__ == "__main__":
    main()
