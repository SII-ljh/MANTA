"""Per-chunk ICAE -> PACT eval bridge (SAC-aligned ROUGE-1).

Generate answers with a (trained) per-chunk ICAE Llama-3.1-8B on PACT's EXACT
2000 MRQA eval examples (data/mrqa_eval_subset/qa_eval.pt + qa_eval_meta.json),
then save generations in PACT's per-rank schema so
scripts/score_rouge1_sac_aligned.py rescores them with the identical protocol.
Zero tokenizer/scorer drift: ICAE and PACT share the Llama-3.1-8B tokenizer and
qa_eval.pt already holds the pre-tokenized prompt input_ids.

Per sample (mirrors PACT eval_k_sweep / lm_inference regime):
  * input_ids   = CONTEXT tokens  -> compress per-chunk (num_mem per 512 window).
  * lm_targets  = question block ("\\n### Question:..\\n### Answer:\\n") appended
                  UNCOMPRESSED to the decoder, then greedy-generate the answer.
  * realized compression_rate = num_prompt_tokens / (num_chunks * num_mem).

Output per rank: {output_dir}/{output_name}_rank{rank}.json
  = { "<ratio>": [ {"id","dataset","generated","compression_rate"}, ... ] }
which is exactly the {key:[samples]} schema score_rouge1_sac_aligned.py expects.

Generation matches PACT (greedy, max_new_tokens=128, decode & strip, NO
newline-as-eos). Multi-GPU via mp.spawn; --cpu single-process for plumbing smoke.
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
from peft import LoraConfig

ICAE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ICAE_DIR)
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from icae_perchunk import ICAEPerChunk


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True,
                   help="Base Llama-3.1-8B dir (backbone).")
    p.add_argument("--trainable_path", default=None,
                   help="Trained icae_trainable.pt (LoRA+mem+ae). "
                        "Omit for an untrained structural smoke.")
    p.add_argument("--eval_data_path", required=True)
    p.add_argument("--eval_meta_path", required=True)
    p.add_argument("--ratio", type=int, default=16,
                   help="chunk_size/num_mem. num_mem = chunk_size//ratio.")
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--num_samples", type=int, default=-1)
    p.add_argument("--max_gen", type=int, default=128)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--output_name", type=str, default="icae_mrqa_gen")
    p.add_argument("--cpu", action="store_true")
    return p.parse_args()


def build_model(args, device):
    num_mem = args.chunk_size // args.ratio
    lora = LoraConfig(r=args.lora_rank, lora_alpha=args.lora_alpha,
                      lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
                      target_modules=["q_proj", "v_proj"])
    dtype = torch.float32 if args.cpu else torch.bfloat16
    model = ICAEPerChunk(
        llama_path=args.model_path, lora_config=lora, num_mem=num_mem,
        chunk_size=args.chunk_size, torch_dtype=dtype,
        attn_impl=("eager" if args.cpu else "sdpa"),
    )
    if args.trainable_path and os.path.exists(args.trainable_path):
        model.load_trainable(args.trainable_path, map_location="cpu")
        print(f"[load] {args.trainable_path}", flush=True)
    else:
        print("[warn] no trainable_path -> UNTRAINED weights (smoke only)", flush=True)
    model.to(device)
    model.eval()
    return model, num_mem


def evaluate_worker(rank, args, world_size, idx_list):
    dev = "cpu" if args.cpu else f"cuda:{rank}"
    eval_examples = torch.load(args.eval_data_path)
    with open(args.eval_meta_path) as f:
        eval_meta = json.load(f)

    total = len(idx_list)
    per = total // world_size
    start = rank * per
    end = start + per if rank < world_size - 1 else total
    my_idx = idx_list[start:end]
    my_ex = [eval_examples[i] for i in my_idx]
    my_meta = [eval_meta[i] for i in my_idx]
    print(f"[GPU {rank}] {len(my_ex)} samples", flush=True)

    model, num_mem = build_model(args, dev)
    key = str(args.ratio)
    results = {key: []}

    with torch.no_grad():
        for ex, meta in tqdm(zip(my_ex, my_meta), total=len(my_ex),
                             desc=f"R{rank}", disable=(rank != 0)):
            inputs = {
                "input_ids": ex["input_ids"].unsqueeze(0).to(dev),
                "lm_targets": ex["lm_targets"].unsqueeze(0).to(dev),
            }
            prompt_len = ex["input_ids"].shape[0]
            gen_ids, n_chunks = model.lm_inference(inputs, generate_num=args.max_gen)
            gen_text = model.tokenizer.decode(
                gen_ids, skip_special_tokens=True).strip()
            comp = prompt_len / (n_chunks * num_mem) if n_chunks * num_mem > 0 else 1.0
            results[key].append({
                "id": meta["id"],
                "dataset": meta["dataset"],
                "generated": gen_text,
                "compression_rate": comp,
            })

    out_path = os.path.join(args.output_dir, f"{args.output_name}_rank{rank}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[GPU {rank}] saved {out_path}", flush=True)


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    world_size = 1 if args.cpu else torch.cuda.device_count()
    assert world_size > 0, "no GPU (use --cpu for a CPU smoke)"

    with open(args.eval_meta_path) as f:
        n_total = len(json.load(f))
    idx_list = list(range(n_total))
    if 0 < args.num_samples < n_total:
        idx_list = idx_list[:args.num_samples]
    print(f"[INFO] world_size={world_size} samples={len(idx_list)} ratio={args.ratio}",
          flush=True)

    t0 = time.time()
    if world_size == 1:
        evaluate_worker(0, args, 1, idx_list)
    else:
        mp.spawn(evaluate_worker, args=(args, world_size, idx_list),
                 nprocs=world_size, join=True)
    print(f"[INFO] generation done in {(time.time()-t0)/60:.1f} min", flush=True)

    key = str(args.ratio)
    merged = {key: []}
    for rank in range(world_size):
        p = os.path.join(args.output_dir, f"{args.output_name}_rank{rank}.json")
        with open(p) as f:
            d = json.load(f)
        merged[key].extend(d[key])
    combined = os.path.join(args.output_dir, f"{args.output_name}_merged.json")
    with open(combined, "w") as f:
        json.dump(merged, f, ensure_ascii=False)
    comps = [s["compression_rate"] for s in merged[key]]
    mc = sum(comps) / len(comps) if comps else 0.0
    print(f"[INFO] ratio {args.ratio}x nominal -> {mc:.2f}x realized "
          f"(n={len(comps)})  merged={combined}", flush=True)


if __name__ == "__main__":
    main()
