#!/usr/bin/env python
"""§3 Motivation probe — dump per-chunk loss-vs-k matrix.

For 1000 MRQA held-out samples (across 12 datasets for diversity), compress each
sample with Matryoshka encoder, then for each chunk c iterate k ∈ K_set holding
the OTHER chunks at K_max, and record the decoder LM loss (per-sample, masked
over targets). This gives the per-chunk loss-vs-k curve used in:
  - Figure 2(a): 50-chunk spaghetti plot
  - Figure 2(b): "min k s.t. L_k - L_K_max < 0.05" histogram
  - §3.3 Lagrangian / oracle-vs-fixed-k motivation
  - Figure 1 right panel (oracle vs fixed k on average budget)

Output: analysis/motivation_probe/per_chunk_loss.json with one record per
chunk: {sample_id, source, dataset, chunk_idx, n_chunks_in_sample, losses: {k: float}}

8-GPU data parallel via mp.spawn. Adapter taken from main PACT ckpt
(experiment/pact_v5_mrqa_allk_k34up by default).
"""
import argparse
import json
import os
import random
import sys
import time

import numpy as np
import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_descriptor")
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modeling import (
    get_model, load_adapter, truncate_single_chunk_kv,
)


# K_SET override via env (e.g. PACT_K_SET=1,2,4,8,16,32 for k≤32 ckpts like dyneps_k32)
K_SET = [int(x) for x in os.environ.get("PACT_K_SET", "1,2,4,8,16,34").split(",")]
K_MAX = max(K_SET)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True,
                   help="experiment dir containing config.json + output/instruction_adapter.pt")
    p.add_argument("--mrqa_eval_pt",
                   default="data/mrqa_eval_full/qa_eval.pt")
    p.add_argument("--mrqa_eval_meta",
                   default="data/mrqa_eval_full/qa_eval_meta.json")
    p.add_argument("--mrqa_samples", type=int, default=1000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adapter_name", default="instruction_adapter.pt")
    p.add_argument("--out_dir", default="analysis/motivation_probe")
    p.add_argument("--max_chunks_per_sample", type=int, default=12,
                   help="cap on chunks per sample (large samples truncated)")
    return p.parse_args()


def build_index(args):
    """Stratified sample: aim for ~equal coverage across 12 MRQA datasets so the
    loss-vs-k matrix reflects task diversity rather than just NQ/HotpotQA bulk."""
    rng = random.Random(args.seed)
    mrqa_meta = json.load(open(args.mrqa_eval_meta))
    by_ds = {}
    for i, m in enumerate(mrqa_meta):
        by_ds.setdefault(m["dataset"], []).append(i)
    plan = []
    n_datasets = len(by_ds)
    per_ds_quota = max(1, args.mrqa_samples // n_datasets)
    for ds, idxs in by_ds.items():
        rng.shuffle(idxs)
        chosen = idxs[: per_ds_quota]
        for i in chosen:
            plan.append(("mrqa", i, mrqa_meta[i]["id"], ds))
    rng.shuffle(plan)
    return plan[: args.mrqa_samples]


def worker(rank, args, world_size, plan):
    with open(os.path.join(args.work_dir, "output", "config.json")) as f:
        config = json.load(f)
    # Use SFT config (matches trained adapter)
    training_config = config["sft_training_config"]
    task_config = config["sft_task_config"]

    # Load eval data inside worker to avoid pickling
    mrqa_data = torch.load(args.mrqa_eval_pt, weights_only=False)

    # Per-rank slice
    total = len(plan)
    per_gpu = total // world_size
    start = rank * per_gpu
    end = start + per_gpu if rank < world_size - 1 else total
    my_plan = plan[start:end]
    if rank == 0:
        print(f"[GPU {rank}] {len(my_plan)} samples (of {total})")

    model = get_model(training_config["model_id"], task_config, rank)
    adapter_path = os.path.join(args.work_dir, "output", args.adapter_name)
    model = load_adapter(model, save_path_and_name=adapter_path)
    model.eval()
    dev = f"cuda:{rank}"

    # MRQA qa_eval.pt only stores lm_targets (no instruction_target). Switch the
    # loss path to pretrain-LM mode so _decoder_lm_loss_for_pkv reads lm_targets.
    # Encoder/compress is unaffected at inference because we don't backprop.
    model.task_config = dict(model.task_config)  # break shared dict
    model.task_config["is_sft"] = False
    model.task_config["is_pretrain"] = True

    records = []
    with torch.no_grad():
        for src, src_idx, sample_id, dataset in tqdm(my_plan, disable=(rank != 0)):
            try:
                ex = mrqa_data[src_idx]
                inputs = {k: v.unsqueeze(0).to(dev) for k, v in ex.items()
                          if torch.is_tensor(v)}

                (_ctids, _ctok, end_idx, _mlh, full_pkv, _ems,
                 per_chunk_mem_sizes) = model.compress(inputs, truncate_k=None)
                N = len(per_chunk_mem_sizes)
                if N == 0:
                    continue
                if N > args.max_chunks_per_sample:
                    # truncate to first chunks (keep prefix continuity)
                    N_use = args.max_chunks_per_sample
                else:
                    N_use = N

                # L_full = loss with all chunks at K_max
                L_full = model._decoder_lm_loss_for_pkv(
                    inputs, full_pkv, end_idx, reduction='mean'
                ).item()

                # Per-chunk per-k loss curves
                for c in range(N_use):
                    losses_k = {K_MAX: L_full}
                    for k in K_SET:
                        if k == K_MAX:
                            continue
                        pkv_c = truncate_single_chunk_kv(
                            full_pkv, per_chunk_mem_sizes, c, k
                        )
                        L_ck = model._decoder_lm_loss_for_pkv(
                            inputs, pkv_c, end_idx, reduction='mean'
                        ).item()
                        losses_k[k] = L_ck

                    records.append({
                        "sample_id": str(sample_id),
                        "source": src,
                        "dataset": dataset,
                        "chunk_idx": c,
                        "n_chunks_in_sample": N,
                        "L_full": L_full,
                        "losses": {str(k): losses_k[k] for k in K_SET},
                    })
            except Exception as e:
                print(f"[GPU {rank}] sample {sample_id} failed: {e}")
                continue

    out_path = os.path.join(args.work_dir, "..", "..",
                            args.out_dir, f"per_chunk_loss_rank{rank}.json")
    out_path = os.path.normpath(out_path)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    json.dump(records, open(out_path, "w"))
    print(f"[GPU {rank}] saved {len(records)} chunk records → {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    plan = build_index(args)
    print(f"[main] {len(plan)} samples planned (mrqa+slimpajama)")
    ws = torch.cuda.device_count()
    print(f"[main] launching mp.spawn with world_size={ws}")
    mp.spawn(worker, args=(args, ws, plan), nprocs=ws, join=True)

    # merge per-rank
    records = []
    for r in range(ws):
        p = os.path.join(args.out_dir, f"per_chunk_loss_rank{r}.json")
        if os.path.exists(p):
            records.extend(json.load(open(p)))
    print(f"[main] merged {len(records)} chunk records")
    final = os.path.join(args.out_dir, "per_chunk_loss.json")
    json.dump(records, open(final, "w"))
    print(f"[main] saved {final}")


if __name__ == "__main__":
    main()
