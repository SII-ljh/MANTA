#!/usr/bin/env python
"""P1-EFF — efficiency benchmark: latency / peak GPU memory / KV-cache bytes /
KPredictor overhead, for the paper's efficiency table.

Conditions per sample (single GPU, single process):
  full      : no compression — raw context prefilled into the decoder (upper anchor)
  k=32/8/1  : PACT-M fixed-k (Matryoshka truncation inside compress())
  adaptive  : PACT-Adaptive (compress full -> KPredictor -> per-chunk truncation),
              KPredictor forward timed separately (its true overhead)

Per (sample, condition) we record:
  ctx_tokens        : raw context length
  compress_s        : encoder compress() wall time (full = decoder prefill time)
  kpred_s           : KPredictor predict_k wall time (adaptive only)
  decode_s          : generation loop wall time
  gen_tokens        : tokens generated (greedy, EOS-stop)
  kv_bytes          : KV cache bytes entering the decode loop
  peak_mem_bytes    : torch.cuda.max_memory_allocated (reset per condition)

Output: reports/benchmark_efficiency_<exp>.json + printed summary table.

Usage (1 GPU, ~30-60 min @ 100 samples):
  python scripts/benchmark_efficiency.py \
      --work_dir experiment/pact_v5_mrqa_dyneps_k32 \
      --eval_data_path data/mrqa_eval_subset/qa_eval.pt \
      --eval_meta_path data/mrqa_eval_subset/qa_eval_meta.json \
      --num_samples 100 --max_gen 64 --eps 0.05
"""
import argparse
import json
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modeling import (  # noqa: E402
    get_model, load_adapter, extract_full_kv, _build_keep_indices,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    p.add_argument("--eval_data_path", default="data/mrqa_eval_subset/qa_eval.pt")
    p.add_argument("--eval_meta_path", default="data/mrqa_eval_subset/qa_eval_meta.json")
    p.add_argument("--num_samples", type=int, default=100)
    p.add_argument("--max_gen", type=int, default=64)
    p.add_argument("--ks", default="32,8,1")
    p.add_argument("--eps", type=float, default=0.05)
    p.add_argument("--adapter_name", default="instruction_adapter.pt")
    p.add_argument("--device", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3, help="warmup samples (not recorded)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=None)
    return p.parse_args()


def kv_bytes(past_key_values):
    return sum(k.numel() * k.element_size() + v.numel() * v.element_size()
               for (k, v) in past_key_values)


def sync():
    torch.cuda.synchronize()


def decode_loop(model, encoder_past_key_values, lm_targets, end_idx, max_gen):
    """Replicates lm_inference's decode loop; returns (gen_token_count, decode_s)."""
    lm_target_emb = model.decoder.model.embed_tokens(lm_targets)
    bsz, seq_len, emb_size = lm_target_emb.size()
    expand_lm_token = model.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)
    lm_emb = torch.cat([expand_lm_token, lm_target_emb], dim=1)
    next_position_ids = torch.arange(
        end_idx, end_idx + seq_len + 1, device=lm_target_emb.device).unsqueeze(0)
    past_key_values = encoder_past_key_values
    next_inputs_embeds = lm_emb
    n_gen = 0
    sync(); t0 = time.perf_counter()
    for _ in range(max_gen):
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
        next_inputs_embeds = model.decoder.model.embed_tokens(next_token_id).unsqueeze(1)
        next_position_ids = next_position_ids[:, -1:] + 1
        n_gen += 1
        if next_token_id.item() == model.tokenizer.eos_token_id:
            break
    sync(); t1 = time.perf_counter()
    return n_gen, t1 - t0


@torch.no_grad()
def run_condition(model, inputs, cond, eps, max_gen):
    """cond in {'full', 'k<int>', 'adaptive'}. Returns metrics dict."""
    device = inputs["input_ids"].device
    torch.cuda.reset_peak_memory_stats(device)
    rec = {"cond": cond, "ctx_tokens": int(inputs["input_ids"].size(1))}

    if cond == "full":
        # uncompressed anchor: raw context prefilled straight into the decoder
        emb = model.decoder.model.embed_tokens(inputs["input_ids"])
        L = emb.size(1)
        pos = torch.arange(0, L, device=device).unsqueeze(0)
        sync(); t0 = time.perf_counter()
        if model.task_config["use_pe"]:
            out = model.decoder(position_ids=pos, inputs_embeds=emb, use_cache=True)
        else:
            out = model.decoder(inputs_embeds=emb, use_cache=True)
        sync(); rec["compress_s"] = time.perf_counter() - t0
        past = out.past_key_values
        rec["kv_bytes"] = kv_bytes(past)
        rec["kpred_s"] = 0.0
        n_gen, dec_s = decode_loop(model, past, inputs["lm_targets"], L, max_gen)
    else:
        truncate_k = None if cond == "adaptive" else int(cond[1:])
        sync(); t0 = time.perf_counter()
        (compress_token_ids, _ct, end_idx, _hid, enc_pkv, _ems,
         per_chunk_mem_sizes) = model.compress(inputs, truncate_k=truncate_k)
        sync(); rec["compress_s"] = time.perf_counter() - t0

        if cond == "adaptive":
            sync(); t0 = time.perf_counter()
            eps_t = torch.tensor([float(eps)], device=enc_pkv[0][0].device)
            anchor_kv = extract_full_kv(enc_pkv)
            _, k_per_chunk = model.k_predictor.predict_k(
                anchor_kv, per_chunk_mem_sizes, eps_t, threshold=None)
            keep_idx = _build_keep_indices(per_chunk_mem_sizes, k_per_chunk,
                                           enc_pkv[0][0].device)
            enc_pkv = tuple((k.index_select(2, keep_idx), v.index_select(2, keep_idx))
                            for (k, v) in enc_pkv)
            sync(); rec["kpred_s"] = time.perf_counter() - t0
            rec["mean_k"] = float(sum(k_per_chunk)) / max(1, len(k_per_chunk))
        else:
            rec["kpred_s"] = 0.0

        rec["kv_bytes"] = kv_bytes(enc_pkv)
        n_gen, dec_s = decode_loop(model, enc_pkv, inputs["lm_targets"], end_idx, max_gen)

    rec["gen_tokens"] = n_gen
    rec["decode_s"] = dec_s
    rec["decode_tok_per_s"] = n_gen / dec_s if dec_s > 0 else 0.0
    rec["peak_mem_bytes"] = int(torch.cuda.max_memory_allocated(device))
    return rec


def main():
    args = parse_args()
    device = f"cuda:{args.device}"
    with open(os.path.join(args.work_dir, "output", "config.json")) as f:
        config = json.load(f)
    training_config = config["sft_training_config"]
    task_config = config["sft_task_config"]

    eval_examples = torch.load(args.eval_data_path)
    with open(args.eval_meta_path) as f:
        eval_meta = json.load(f)

    import random
    rng = random.Random(args.seed)
    idx = list(range(len(eval_examples)))
    rng.shuffle(idx)
    idx = idx[: args.num_samples + args.warmup]

    model = get_model(training_config["model_id"], task_config, args.device)
    model = load_adapter(model, save_path_and_name=os.path.join(
        args.work_dir, "output", args.adapter_name))
    model.eval()

    has_predictor = getattr(model, "use_k_predictor", False) and model.k_predictor is not None
    conds = ["full"] + [f"k{k}" for k in args.ks.split(",")]
    if has_predictor:
        conds.append("adaptive")
    print(f"[bench] conditions: {conds}  samples={args.num_samples} warmup={args.warmup}")

    records = []
    for i, sample_i in enumerate(idx):
        ex = eval_examples[sample_i]
        inputs = {
            "input_ids": ex["input_ids"].unsqueeze(0).to(device),
            "lm_targets": ex["lm_targets"].unsqueeze(0).to(device),
        }
        for cond in conds:
            try:
                rec = run_condition(model, inputs, cond, args.eps, args.max_gen)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                rec = {"cond": cond, "oom": True,
                       "ctx_tokens": int(inputs["input_ids"].size(1))}
            rec["sample_idx"] = sample_i
            rec["dataset"] = eval_meta[sample_i].get("dataset")
            rec["warmup"] = i < args.warmup
            records.append(rec)
        if i == args.warmup - 1:
            print("[bench] warmup done")

    # aggregate (non-warmup, non-oom)
    agg = {}
    for cond in conds:
        rs = [r for r in records if r["cond"] == cond and not r.get("warmup")
              and not r.get("oom")]
        if not rs:
            agg[cond] = {"n": 0}
            continue
        mean = lambda key: sum(r[key] for r in rs) / len(rs)  # noqa: E731
        agg[cond] = {
            "n": len(rs),
            "ctx_tokens_mean": mean("ctx_tokens"),
            "compress_s_mean": mean("compress_s"),
            "kpred_s_mean": mean("kpred_s"),
            "decode_tok_per_s_mean": mean("decode_tok_per_s"),
            "kv_mb_mean": mean("kv_bytes") / 1e6,
            "peak_mem_gb_mean": mean("peak_mem_bytes") / 1e9,
            "oom_count": sum(1 for r in records
                             if r["cond"] == cond and r.get("oom")),
        }

    exp_name = os.path.basename(os.path.normpath(args.work_dir))
    out = args.out or f"reports/benchmark_efficiency_{exp_name}.json"
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump({"args": vars(args), "aggregate": agg, "records": records}, f,
                  ensure_ascii=False, indent=1)

    print(f"\n{'cond':<10}{'n':>4}{'ctx':>8}{'compress_s':>12}{'kpred_s':>10}"
          f"{'dec tok/s':>11}{'KV MB':>9}{'peak GB':>9}")
    for cond, a in agg.items():
        if a["n"] == 0:
            print(f"{cond:<10}{0:>4}  (no data)")
            continue
        print(f"{cond:<10}{a['n']:>4}{a['ctx_tokens_mean']:>8.0f}"
              f"{a['compress_s_mean']:>12.3f}{a['kpred_s_mean']:>10.4f}"
              f"{a['decode_tok_per_s_mean']:>11.1f}{a['kv_mb_mean']:>9.1f}"
              f"{a['peak_mem_gb_mean']:>9.2f}")
    print(f"\n[saved] {out}")


if __name__ == "__main__":
    main()
