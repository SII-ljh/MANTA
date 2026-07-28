"""Per-chunk 500xCompressor -> PACT eval bridge (SAC-aligned ROUGE-1).

Generate answers with the trained per-chunk 500xC model on PACT's EXACT 2000
MRQA eval examples (data/mrqa_eval_subset/qa_eval.pt + qa_eval_meta.json), at the
chosen compression ratio, and emit the same per-rank schema that
scripts/score_rouge1_sac_aligned.py ingests (zero scorer drift). PACT and 500xC
share the Llama-3.1-8B tokenizer and qa_eval.pt holds pre-tokenized prompt
input_ids positionally aligned to qa_eval_meta.json.

PACT-comparable regime (matches the sibling ICAE bridge + PACT lm_inference):
  * input_ids   = CONTEXT tokens  -> compress per-chunk (num_mem KV per 512 window).
  * lm_targets  = question block ("\\n### Question:..\\n### Answer:\\n") fed to the
                  decoder UNCOMPRESSED (real tokens against the compressed-context
                  past_key_values), then the answer is greedy-generated.
                  The model SEES the question.
  * realized compression_rate = num_context_tokens / (num_chunks * num_mem)
    (context-only -- the uncompressed question is not counted).

Generation matches PACT (greedy, max_new_tokens=128, do_sample=False, decode &
strip, NO newline-as-eos).

Output per rank: {output_dir}/{output_name}_rank{rank}.json
  = { "<ratio>": [ {"id","dataset","generated","compression_rate"}, ... ] }
"""
import argparse
import json
import os
import sys
import time

import torch
import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")
from tqdm import tqdm

THIS = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, THIS)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
from c500x_perchunk import C500xPerChunk, default_lora_config


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--llama_path", required=True, help="base Llama-3.1-8B dir")
    p.add_argument("--trainable_path", default=None,
                   help="trained LoRA+memory_embeddings .pt (omit for untrained smoke)")
    p.add_argument("--eval_data_path", required=True)
    p.add_argument("--eval_meta_path", required=True)
    p.add_argument("--ratio", type=int, default=16,
                   help="compression ratio -> num_mem = chunk_size // ratio")
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--max_gen", type=int, default=128)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_name", type=str, default="c500x_mrqa_gen")
    p.add_argument("--cpu", action="store_true")
    p.add_argument("--lora_r", type=int, default=128,
                   help="must match training (launcher default 128)")
    p.add_argument("--lora_alpha", type=int, default=256,
                   help="must match training (launcher default 256)")
    return p.parse_args()


def build_model(args, device):
    num_mem = args.chunk_size // args.ratio
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = C500xPerChunk(args.llama_path, num_mem=num_mem,
                          chunk_size=args.chunk_size,
                          lora_config=default_lora_config(args.lora_r,
                                                          args.lora_alpha),
                          device=torch.device(device), dtype=dtype)
    if args.trainable_path and os.path.isfile(args.trainable_path):
        model.load_trainable(args.trainable_path)
        print(f"[load] trained params from {args.trainable_path}", flush=True)
    else:
        print("[warn] no trainable_path -> UNTRAINED model (plumbing smoke only)",
              flush=True)
    model.to(device)
    model.eval()
    return model, num_mem


def evaluate_worker(rank, args, world_size, idx_list):
    dev = "cpu" if args.cpu else f"cuda:{rank}"

    # [RANK_RESUME=1] idempotent relaunch on flaky nodes: skip this rank if its
    # output file already exists and is complete. Opt-in via env.
    if os.environ.get("RANK_RESUME") == "1":
        _total = len(idx_list)
        _per = _total // world_size
        _start = rank * _per
        _end = _start + _per if rank < world_size - 1 else _total
        _expected = _end - _start
        _out = os.path.join(args.output_dir, f"{args.output_name}_rank{rank}.json")
        if os.path.exists(_out):
            try:
                with open(_out) as f:
                    _d = json.load(f)
                if len(_d.get(str(args.ratio), [])) == _expected:
                    print(f"[{dev}] RANK_RESUME: {_out} complete ({_expected} samples) — skip", flush=True)
                    return
                print(f"[{dev}] RANK_RESUME: {_out} incomplete — regenerating", flush=True)
            except Exception:
                print(f"[{dev}] RANK_RESUME: {_out} unreadable — regenerating", flush=True)

    eval_examples = torch.load(args.eval_data_path)
    with open(args.eval_meta_path) as f:
        eval_meta = json.load(f)

    total = len(idx_list)
    per = total // world_size
    start = rank * per
    end = start + per if rank < world_size - 1 else total
    my_idx = idx_list[start:end]
    print(f"[{dev}] {len(my_idx)} samples @ ratio {args.ratio}", flush=True)

    model, num_mem = build_model(args, dev)
    ratio_key = str(args.ratio)
    results = {ratio_key: []}

    with torch.no_grad():
        for j in tqdm(my_idx, disable=(rank != 0)):
            ex, meta = eval_examples[j], eval_meta[j]
            ids = ex["input_ids"].unsqueeze(0).to(dev)            # CONTEXT -> compress
            lm_targets = ex["lm_targets"].unsqueeze(0).to(dev)    # question block
            prompt_len = ids.shape[1]
            gen, num_chunks, kv_total = model.lm_inference(
                ids, lm_targets=lm_targets, max_new_tokens=args.max_gen)
            gen_text = model.tokenizer.decode(gen, skip_special_tokens=True).strip()
            comp = prompt_len / kv_total if kv_total > 0 else 1.0
            results[ratio_key].append({
                "id": meta["id"],
                "dataset": meta["dataset"],
                "generated": gen_text,
                "compression_rate": comp,
            })

    out_path = os.path.join(args.output_dir, f"{args.output_name}_rank{rank}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[{dev}] saved {out_path}", flush=True)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    world_size = 1 if args.cpu else torch.cuda.device_count()
    assert world_size > 0, "no GPU (use --cpu for smoke)"

    with open(args.eval_meta_path) as f:
        n_total = len(json.load(f))
    idx_list = list(range(n_total))
    if 0 < args.num_samples < n_total:
        idx_list = idx_list[: args.num_samples]
    print(f"[INFO] world_size={world_size} samples={len(idx_list)} "
          f"ratio={args.ratio} num_mem={args.chunk_size // args.ratio}", flush=True)

    t0 = time.time()
    if world_size == 1:
        evaluate_worker(0, args, 1, idx_list)
    else:
        mp.spawn(evaluate_worker, args=(args, world_size, idx_list),
                 nprocs=world_size, join=True)
    print(f"[INFO] generation done in {(time.time()-t0)/60:.1f} min", flush=True)

    ratio_key = str(args.ratio)
    merged = {ratio_key: []}
    for rank in range(world_size):
        p = os.path.join(args.output_dir, f"{args.output_name}_rank{rank}.json")
        with open(p) as f:
            merged[ratio_key].extend(json.load(f)[ratio_key])
    comps = [s["compression_rate"] for s in merged[ratio_key]]
    mc = sum(comps) / len(comps) if comps else 0.0
    print(f"[INFO] ratio {args.ratio}x nominal -> {mc:.2f}x realized (n={len(comps)})")
    combined = os.path.join(args.output_dir, f"{args.output_name}_merged.json")
    with open(combined, "w") as f:
        json.dump(merged, f, ensure_ascii=False)
    print(f"[INFO] merged: {combined}\n[INFO] rescore: "
          f"scripts/score_rouge1_sac_aligned.py --meta {args.eval_meta_path} "
          f"--sources c500x={args.output_dir}/{args.output_name}_rank*.json")


if __name__ == "__main__":
    main()
