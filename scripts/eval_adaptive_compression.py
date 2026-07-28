"""Adaptive compression evaluator (PACT v4).

For each sample, runs inference under the model's learned gate (no external
truncate_k) and measures:
  - F1 / EM vs gold answers
  - per-chunk effective k (kept anchors)
  - per-sample mean compression rate = sum(kept_anchors) / sum(max_anchors)

Supports multiple gate_modes:
  - cumprod          : measures sum(g > 0.1) per chunk
  - cross_chunk_softmax : measures sum(g) per chunk (soft) AND threshold-rounded
  - content_topk     : k is determined by truncate_k parameter (per-chunk top-k)

Output: summary JSON with overall F1/EM/mean_kept_per_chunk/mean_compression_rate.
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
    p.add_argument("--work_dir", required=True)
    p.add_argument("--eval_data_path", required=True)
    p.add_argument("--eval_meta_path", required=True)
    p.add_argument("--num_samples", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_gen", type=int, default=128)
    p.add_argument("--adapter_name", type=str, default="instruction_adapter.pt")
    p.add_argument("--output_name", type=str, default="adaptive_eval")
    # adaptive mode: how to run the gate
    p.add_argument("--mode", default="adaptive",
                   choices=["adaptive", "topk_sweep"],
                   help="adaptive = soft gate; topk_sweep = hard top-k at given k values")
    p.add_argument("--topk_values", default="2,4,8,12,16,24,34",
                   help="for topk_sweep mode (content_topk gate)")
    p.add_argument("--target_avg_override", type=float, default=None,
                   help="override model.gate_predictor.target_avg at inference (cross_chunk_softmax)")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="adaptive-mode soft-gate keep threshold")
    return p.parse_args()


def measure_chunk_keep_rate(model, inputs, mode="adaptive", topk=None, threshold=0.5):
    """Run model and return (generated_tokens, kept_per_chunk:list[int], total_anchors:int).

    For 'adaptive' mode:
      - cumprod: count anchors with g > threshold
      - cross_chunk_softmax: count anchors with g > threshold (default 0.5)
      - content_topk: requires topk param to be set; hard top-k
    For 'topk_sweep' mode (any gate, treats it like content_topk):
      use the gate's score to pick top-k anchors per chunk; if gate is cumprod,
      use g; if cross_chunk_softmax, use g.
    """
    # mirror lm_inference logic but record kept anchors
    use_content_topk = (model.use_gate and model.gate_mode == "content_topk")
    compress_truncate_k = None if use_content_topk else None  # always full

    # For non-gate models with topk_sweep: use positional truncation (Matryoshka).
    # For gate models: pass None and let gate decide.
    compress_truncate_k = topk if (mode == "topk_sweep" and topk is not None
                                    and not (model.use_gate and model.gate_predictor is not None)) else None
    (compress_token_ids, compress_token, end_idx,
     mem_last_hidden, encoder_past_key_values, encoder_mem_size,
     per_chunk_mem_sizes) = model.compress(inputs, truncate_k=compress_truncate_k)

    kept_per_chunk = []

    if model.use_gate and model.gate_predictor is not None:
        gate_g, gate_info_raw = model.gate_predictor(mem_last_hidden, per_chunk_mem_sizes)

        if mode == "topk_sweep" and topk is not None:
            # Hard top-k by score (content_topk) or g (cumprod / ccs).
            if model.gate_mode == "content_topk":
                score = gate_info_raw["score"][0]
            else:
                score = gate_g[0]
            keep_indices = []
            _off = 0
            for m in per_chunk_mem_sizes:
                seg = score[_off:_off + m]
                k_eff = max(1, min(int(topk), m))
                topk_idx = torch.topk(seg, k_eff, largest=True).indices.sort().values
                keep_indices.append(topk_idx + _off)
                kept_per_chunk.append(int(topk_idx.numel()))
                _off += m
            keep_idx = torch.cat(keep_indices, dim=0)
            encoder_past_key_values = tuple(
                (k.index_select(2, keep_idx), v.index_select(2, keep_idx))
                for (k, v) in encoder_past_key_values
            )
            compress_token_ids = compress_token_ids.index_select(1, keep_idx)

        elif mode == "adaptive":
            if model.gate_mode == "content_topk":
                # No adaptive natural mode; fall back to threshold on score sigmoid.
                score = gate_info_raw["score"][0]
                g_sig = torch.sigmoid(score)
                keep_mask = g_sig > threshold
            else:
                keep_mask = (gate_g[0] > threshold)

            # Per-chunk hard-drop low-g anchors
            keep_indices = []
            _off = 0
            for m in per_chunk_mem_sizes:
                seg_mask = keep_mask[_off:_off + m]
                if not seg_mask.any():
                    seg_mask = torch.zeros_like(seg_mask)
                    seg_mask[0] = True  # keep at least one
                local_idx = seg_mask.nonzero(as_tuple=False).squeeze(-1)
                keep_indices.append(local_idx + _off)
                kept_per_chunk.append(int(local_idx.numel()))
                _off += m
            keep_idx = torch.cat(keep_indices, dim=0)
            encoder_past_key_values = tuple(
                (k.index_select(2, keep_idx), v.index_select(2, keep_idx))
                for (k, v) in encoder_past_key_values
            )
            compress_token_ids = compress_token_ids.index_select(1, keep_idx)
    else:
        kept_per_chunk = list(per_chunk_mem_sizes)

    # Decode
    lm_target_emb = model.decoder.model.embed_tokens(inputs['lm_targets'])
    bsz, seq_len, emb_size = lm_target_emb.size()
    expand_lm_token = model.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)
    lm_emb = torch.cat([expand_lm_token, lm_target_emb], dim=1)
    latter_position_ids = torch.arange(end_idx, end_idx + seq_len + 1,
                                       device=lm_target_emb.device).unsqueeze(0)

    generate_text = []
    past_key_values = encoder_past_key_values
    next_inputs_embeds = lm_emb.clone()
    next_position_ids = latter_position_ids.clone()
    for _ in range(128):
        if model.task_config["use_pe"]:
            out = model.decoder(position_ids=next_position_ids,
                                inputs_embeds=next_inputs_embeds,
                                past_key_values=past_key_values, use_cache=True)
        else:
            out = model.decoder(inputs_embeds=next_inputs_embeds,
                                past_key_values=past_key_values, use_cache=True)
        logit = out.logits[:, -1]
        past_key_values = out.past_key_values
        next_token_id = torch.argmax(logit, dim=-1)
        next_inputs_embeds = model.decoder.model.embed_tokens(next_token_id).unsqueeze(1).to(lm_target_emb.device)
        next_position_ids = next_position_ids[:, -1:] + 1
        generate_text.append(next_token_id.item())
        if next_token_id.item() == model.tokenizer.eos_token_id:
            break

    total_anchors = sum(per_chunk_mem_sizes)
    return generate_text, kept_per_chunk, total_anchors


def evaluate_worker(rank, args, world_size, examples_subset, meta_subset, topks):
    with open(os.path.join(args.work_dir, "output", "config.json")) as f:
        config = json.load(f)
    training_config = config["sft_training_config"]
    task_config = config["sft_task_config"]

    # Per-rank slice
    total = len(examples_subset)
    per_gpu = total // world_size
    start = rank * per_gpu
    end = start + per_gpu if rank < world_size - 1 else total
    my_ex = examples_subset[start:end]
    my_meta = meta_subset[start:end]

    print(f"[GPU {rank}] {len(my_ex)} samples")

    model = get_model(training_config["model_id"], task_config, rank)
    adapter_path = os.path.join(args.work_dir, "output", args.adapter_name)
    model = load_adapter(model, save_path_and_name=adapter_path)
    model.eval()
    if args.target_avg_override is not None and hasattr(model, "gate_predictor") and model.gate_predictor is not None and hasattr(model.gate_predictor, "target_avg"):
        old_ta = model.gate_predictor.target_avg
        model.gate_predictor.target_avg = float(args.target_avg_override)
        if rank == 0:
            print(f"[gate] target_avg override: {old_ta} -> {model.gate_predictor.target_avg}")

    tokenizer = AutoTokenizer.from_pretrained(training_config["model_id"])

    if args.mode == "adaptive":
        results = {"adaptive": []}
        modes_to_run = [("adaptive", None)]
    else:
        results = {f"k={k}": [] for k in topks}
        modes_to_run = [(f"k={k}", k) for k in topks]

    with torch.no_grad():
        for ex, meta in tqdm(zip(my_ex, my_meta), total=len(my_ex),
                             desc=f"GPU{rank}", disable=(rank != 0)):
            inputs = {
                "input_ids": ex["input_ids"].unsqueeze(0).to(f"cuda:{rank}"),
                "lm_targets": ex["lm_targets"].unsqueeze(0).to(f"cuda:{rank}"),
            }
            for label, kk in modes_to_run:
                gen_ids, kept_per_chunk, total_anchors = measure_chunk_keep_rate(
                    model, inputs, mode=args.mode, topk=kk, threshold=args.threshold
                )
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
                best_f1 = max(token_f1(gen_text, ans) for ans in meta["answers"])
                best_em = max(exact_match(gen_text, ans) for ans in meta["answers"])
                total_kept = sum(kept_per_chunk)
                results[label].append({
                    "id": meta["id"],
                    "dataset": meta["dataset"],
                    "f1": best_f1,
                    "em": best_em,
                    "generated": gen_text,
                    "kept_per_chunk": kept_per_chunk,
                    "total_kept": total_kept,
                    "total_anchors": total_anchors,
                    "compression_rate": total_kept / max(1, total_anchors),
                    "mean_k_per_chunk": total_kept / max(1, len(kept_per_chunk)),
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
    eval_examples = torch.load(args.eval_data_path)
    with open(args.eval_meta_path) as f:
        eval_meta = json.load(f)
    assert len(eval_examples) == len(eval_meta)
    print(f"[INFO] {len(eval_examples)} total samples")

    if args.num_samples > 0 and args.num_samples < len(eval_examples):
        rng = random.Random(args.seed)
        idx = rng.sample(range(len(eval_examples)), args.num_samples)
        idx.sort()
        examples_subset = [eval_examples[i] for i in idx]
        meta_subset = [eval_meta[i] for i in idx]
    else:
        examples_subset = eval_examples
        meta_subset = eval_meta

    topks = [int(k) for k in args.topk_values.split(",")] if args.mode == "topk_sweep" else []

    t0 = time.time()
    if world_size == 1:
        evaluate_worker(0, args, 1, examples_subset, meta_subset, topks)
    else:
        mp.spawn(evaluate_worker,
                 args=(args, world_size, examples_subset, meta_subset, topks),
                 nprocs=world_size, join=True)
    print(f"[INFO] eval done in {(time.time()-t0)/60:.1f} min")

    # Aggregate
    keys = ["adaptive"] if args.mode == "adaptive" else [f"k={k}" for k in topks]
    merged = {k: [] for k in keys}
    for rank in range(world_size):
        path = os.path.join(args.work_dir, "output", f"{args.output_name}_rank{rank}.json")
        with open(path) as f:
            d = json.load(f)
        for k in keys:
            merged[k].extend(d.get(k, []))

    summary = {"args": vars(args), "per_label": {}}
    print("\n" + "=" * 80)
    print(f"{'label':>10} {'N':>6} {'F1':>8} {'EM':>8} {'mean_k':>8} {'comp_rate':>10} {'k_std':>8}")
    print("-" * 80)
    for label in keys:
        rs = merged[label]
        if not rs:
            continue
        f1s = [r["f1"] for r in rs]
        ems = [r["em"] for r in rs]
        kpc = [r["mean_k_per_chunk"] for r in rs]
        cps = [r["compression_rate"] for r in rs]
        # per-sample k variance from per-chunk lists
        sample_k_std = []
        for r in rs:
            kpc_list = r.get("kept_per_chunk") or []
            if len(kpc_list) > 1:
                sample_k_std.append(float(np.std(kpc_list)))
        k_std_mean = float(np.mean(sample_k_std)) if sample_k_std else 0.0
        row = {
            "n": len(rs),
            "f1": float(np.mean(f1s)),
            "em": float(np.mean(ems)),
            "mean_k_per_chunk": float(np.mean(kpc)),
            "compression_rate": float(np.mean(cps)),
            "within_sample_k_std": k_std_mean,
        }
        summary["per_label"][label] = row
        print(f"{label:>10} {row['n']:>6} {row['f1']:>8.4f} {row['em']:>8.4f} "
              f"{row['mean_k_per_chunk']:>8.2f} {row['compression_rate']:>10.4f} "
              f"{row['within_sample_k_std']:>8.2f}")
    print("=" * 80)

    out_summary = os.path.join(args.work_dir, "output", f"{args.output_name}_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] saved {out_summary}")


if __name__ == "__main__":
    main()
