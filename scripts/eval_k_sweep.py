"""K-sweep evaluation for MANTA.

For each sample in a (possibly subsampled) eval set, generate answers at each
k in K_VALUES and compute token-F1 / EM against reference answers. Splits samples
across visible GPUs (mp.spawn). Saves per-rank per-k JSON + aggregate summary.
"""
import argparse
import json
import os
import random
import re
import string
import sys
import time
from collections import Counter

import numpy as np
import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modeling import get_model, load_adapter


# ---- SQuAD-normalized metrics ----
def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    s = " ".join(s.split())
    return s


def token_f1(pred: str, gold: str) -> float:
    pred_tokens = normalize_answer(pred).split()
    gold_tokens = normalize_answer(gold).split()
    if not gold_tokens:
        return float(not pred_tokens)
    if not pred_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    nc = sum(common.values())
    if nc == 0:
        return 0.0
    p = nc / len(pred_tokens)
    r = nc / len(gold_tokens)
    return 2 * p * r / (p + r)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True,
                   help="Path to experiment dir containing config.json + output/instruction_adapter.pt")
    p.add_argument("--eval_data_path", required=True,
                   help="qa_eval.pt path (eval examples)")
    p.add_argument("--eval_meta_path", required=True,
                   help="qa_eval_meta.json path (eval metadata with answers)")
    p.add_argument("--ks", type=str, default="2,4,8,16,32,34",
                   help="Comma-separated k values to sweep")
    p.add_argument("--num_samples", type=int, default=2000,
                   help="Number of samples to evaluate (subset for speed). -1 = all")
    p.add_argument("--idx_offset", type=int, default=0, help="shard offset for cross-host eval")
    p.add_argument("--idx_stride", type=int, default=1, help="shard stride for cross-host eval")
    p.add_argument("--idx_start", type=int, default=-1, help="shard start index (inclusive); -1=disabled")
    p.add_argument("--idx_end", type=int, default=-1, help="shard end index (exclusive); -1=disabled")
    p.add_argument("--seed", type=int, default=42, help="Subset selection seed")
    p.add_argument("--max_gen", type=int, default=128)
    p.add_argument("--adapter_name", type=str, default="instruction_adapter.pt")
    p.add_argument("--output_name", type=str, default="k_sweep_results")
    p.add_argument("--gate_threshold", type=float, default=None,
                   help="If set, pass to lm_inference. Use 0.0 to disable gate soft-multiply (keeps all compressed tokens without applying the gate). None = default behavior (soft gate).")
    return p.parse_args()


def evaluate_worker(rank, args, world_size, idx_list, ks):
    with open(os.path.join(args.work_dir, "output", "config.json")) as f:
        config = json.load(f)
    training_config = config["sft_training_config"]
    task_config = config["sft_task_config"]

    # [RANK_RESUME=1] idempotent relaunch on flaky nodes: skip this rank if its
    # output file already exists and is complete (guard BEFORE the heavy
    # torch.load). Opt-in via env so normal FORCE-style reruns are unaffected.
    if os.environ.get("RANK_RESUME") == "1":
        _total = len(idx_list)
        _per = _total // world_size
        _start = rank * _per
        _end = _start + _per if rank < world_size - 1 else _total
        _expected = _end - _start
        _out = os.path.join(args.work_dir, "output", f"{args.output_name}_rank{rank}.json")
        if os.path.exists(_out):
            try:
                with open(_out) as f:
                    _d = json.load(f)
                if all(len(_d.get(str(k), [])) == _expected for k in ks):
                    print(f"[GPU {rank}] RANK_RESUME: {_out} complete ({_expected} samples) — skip")
                    return
                print(f"[GPU {rank}] RANK_RESUME: {_out} incomplete — regenerating")
            except Exception:
                print(f"[GPU {rank}] RANK_RESUME: {_out} unreadable — regenerating")

    # Each worker loads the eval dataset itself — avoids pickling 67K+ tensors
    # through mp.spawn args (every shared tensor consumes one mmap region;
    # vm.max_map_count caps that out on full-eval scale). Mirrors eval_v5_pareto.py.
    eval_examples = torch.load(args.eval_data_path)
    with open(args.eval_meta_path) as f:
        eval_meta = json.load(f)

    # Per-rank slice (by idx_list — main proc owns the subsampling decision)
    total = len(idx_list)
    per_gpu = total // world_size
    start = rank * per_gpu
    end = start + per_gpu if rank < world_size - 1 else total
    my_idx = idx_list[start:end]
    my_ex = [eval_examples[i] for i in my_idx]
    my_meta = [eval_meta[i] for i in my_idx]

    print(f"[GPU {rank}] {len(my_ex)} samples × {len(ks)} k values")

    model = get_model(training_config["model_id"], task_config, rank)
    adapter_path = os.path.join(args.work_dir, "output", args.adapter_name)
    model = load_adapter(model, save_path_and_name=adapter_path)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(training_config["model_id"])

    # results[k] = list of {id, f1, em, generated}
    results = {k: [] for k in ks}

    with torch.no_grad():
        for ex, meta in tqdm(zip(my_ex, my_meta), total=len(my_ex),
                             desc=f"GPU{rank}", disable=(rank != 0)):
            inputs = {
                "input_ids": ex["input_ids"].unsqueeze(0).to(f"cuda:{rank}"),
                "lm_targets": ex["lm_targets"].unsqueeze(0).to(f"cuda:{rank}"),
            }
            for k in ks:
                # Note: lm_inference supports truncate_k kwarg
                gen_ids = model.lm_inference(inputs, generate_num=args.max_gen,
                                             truncate_k=(None if k is None else int(k)),
                                             gate_threshold=args.gate_threshold)
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                best_f1 = max(token_f1(gen_text, ans) for ans in meta["answers"])
                best_em = max(exact_match(gen_text, ans) for ans in meta["answers"])
                results[k].append({
                    "id": meta["id"],
                    "dataset": meta["dataset"],
                    "f1": best_f1,
                    "em": best_em,
                    "generated": gen_text,
                })

    out_path = os.path.join(args.work_dir, "output", f"{args.output_name}_rank{rank}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[GPU {rank}] saved {out_path}")


def main():
    args = parse_args()
    world_size = torch.cuda.device_count()
    assert world_size > 0, "no GPU"

    print(f"[INFO] world_size={world_size}")

    # Main proc loads ONLY the meta (small JSON). Each worker loads the .pt
    # of tensors and slices by idx_list[start:end]. Keeps mp.spawn args tiny
    # (a list of ints), avoiding vm.max_map_count blow-up on 67K+ tensors.
    with open(args.eval_meta_path) as f:
        eval_meta = json.load(f)
    n_total = len(eval_meta)
    print(f"[INFO] {n_total} samples total")

    # Subsample (deterministic by seed). num_samples <= 0 OR >= n_total = full set.
    if 0 < args.num_samples < n_total:
        rng = random.Random(args.seed)
        idx_list = sorted(rng.sample(range(n_total), args.num_samples))
        print(f"[INFO] subsampled to {len(idx_list)} samples (seed={args.seed})")
        with open(os.path.join(args.work_dir, "output", f"{args.output_name}_selected_ids.json"), "w") as f:
            json.dump([eval_meta[i]["id"] for i in idx_list], f)
    else:
        idx_list = list(range(n_total))

    ks = [int(k) for k in args.ks.split(",")]
    print(f"[INFO] k values: {ks}")
    if args.idx_start >= 0 and args.idx_end >= 0:
        idx_list = idx_list[args.idx_start:args.idx_end]
        print(f"[INFO] sharded range: [{args.idx_start}, {args.idx_end}) -> {len(idx_list)} samples")
    elif args.idx_stride > 1:
        idx_list = idx_list[args.idx_offset::args.idx_stride]
        print(f"[INFO] sharded: off={args.idx_offset} stride={args.idx_stride} -> {len(idx_list)} samples")
    print(f"[INFO] evaluating {len(idx_list)} samples")

    t0 = time.time()
    if world_size == 1:
        # Single-GPU: skip mp.spawn to avoid SHM/FD pressure from pickling tensors.
        evaluate_worker(0, args, 1, idx_list, ks)
    else:
        mp.spawn(evaluate_worker,
                 args=(args, world_size, idx_list, ks),
                 nprocs=world_size, join=True)
    print(f"[INFO] eval done in {(time.time() - t0)/60:.1f} min")

    # Aggregate across ranks
    merged = {k: [] for k in ks}
    for rank in range(world_size):
        path = os.path.join(args.work_dir, "output", f"{args.output_name}_rank{rank}.json")
        with open(path) as f:
            d = json.load(f)
        for k in ks:
            merged[int(k)].extend(d[str(k)])

    summary = {"overall": {}, "per_dataset_per_k": {}}
    print("\n" + "=" * 60)
    print(f"{'k':>4} {'N':>6} {'F1':>8} {'EM':>8}")
    print("-" * 60)
    for k in ks:
        rs = merged[k]
        f1s = [r["f1"] for r in rs]
        ems = [r["em"] for r in rs]
        summary["overall"][k] = {
            "f1": float(np.mean(f1s)) if f1s else 0.0,
            "em": float(np.mean(ems)) if ems else 0.0,
            "count": len(rs),
        }
        print(f"{k:>4} {len(rs):>6} {summary['overall'][k]['f1']:>8.4f} {summary['overall'][k]['em']:>8.4f}")
        ds_metrics = {}
        for r in rs:
            ds = r.get("dataset", "?")
            ds_metrics.setdefault(ds, {"f1": [], "em": [], "count": 0})
            ds_metrics[ds]["f1"].append(r["f1"])
            ds_metrics[ds]["em"].append(r["em"])
            ds_metrics[ds]["count"] += 1
        summary["per_dataset_per_k"][k] = {
            ds: {"f1": float(np.mean(m["f1"])), "em": float(np.mean(m["em"])), "n": m["count"]}
            for ds, m in ds_metrics.items()
        }
    print("=" * 60)

    out_summary = os.path.join(args.work_dir, "output", f"{args.output_name}_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] saved {out_summary}")


if __name__ == "__main__":
    main()
