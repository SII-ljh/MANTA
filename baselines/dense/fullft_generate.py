#!/usr/bin/env python
"""Full-FT topline generation on data/mrqa_eval_full (n=67854).

Feeds the IDENTICAL full prompt every compression method receives
(input_ids + lm_targets = [BOS]### Context:\n<ctx>\n### Question:\n<q>\n### Answer:\n)
to the full-FT model and greedily decodes the answer. No compression.

Multi-GPU sharded (torchrun): each rank handles a strided slice, writes
<output_name>_rank<R>.json with schema {output_name: [{id, generated}, ...]}
so scripts/score_rouge1_sac_aligned.py + scripts/rescore_tokenf1_from_generated.py
(dataset/answers taken from qa_eval_meta.json by id) pick it up unchanged.
"""
import argparse
import json
import os

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model_dir", required=True)   # the full-FT /final dir
    p.add_argument("--eval_pt", required=True)
    p.add_argument("--meta", required=True,
                   help="qa_eval_meta.json — for per-record dataset tag (token-F1 rescorer needs it)")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--output_name", default="fullft_full58k")
    p.add_argument("--max_gen", type=int, default=128)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--num_samples", type=int, default=0)  # 0 = all
    return p.parse_args()


def _to_list(x):
    return x.tolist() if torch.is_tensor(x) else list(x)


def main():
    a = parse()
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    dev = f"cuda:{local_rank}"

    tok = AutoTokenizer.from_pretrained(a.model_dir)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        a.model_dir, torch_dtype=torch.bfloat16).to(dev).eval()
    model.config.use_cache = True

    data = torch.load(a.eval_pt, map_location="cpu")
    with open(a.meta) as f:
        meta = json.load(f)
    if a.num_samples > 0:
        data = data[:a.num_samples]
    my = list(range(len(data)))[rank::world]
    if rank == 0:
        print(f"[gen] {len(data)} total, ~{len(my)} on rank0, batch={a.batch}", flush=True)

    def prompt_ids(ex):
        return _to_list(ex["input_ids"]) + _to_list(ex["lm_targets"])

    results = []
    for bs in range(0, len(my), a.batch):
        chunk = my[bs:bs + a.batch]
        seqs = [prompt_ids(data[i]) for i in chunk]
        maxlen = max(len(s) for s in seqs)
        input_ids, attn = [], []
        for s in seqs:
            pad = maxlen - len(s)
            input_ids.append([pad_id] * pad + s)     # left pad for batched gen
            attn.append([0] * pad + [1] * len(s))
        input_ids = torch.tensor(input_ids, device=dev)
        attn = torch.tensor(attn, device=dev)
        with torch.no_grad():
            out = model.generate(input_ids=input_ids, attention_mask=attn,
                                 max_new_tokens=a.max_gen, do_sample=False,
                                 num_beams=1, pad_token_id=pad_id)
        gen = out[:, input_ids.shape[1]:]
        for j, i in enumerate(chunk):
            text = tok.decode(gen[j], skip_special_tokens=True).strip()
            results.append({"id": meta[i]["id"], "dataset": meta[i]["dataset"],
                            "generated": text})
        if rank == 0 and (bs // a.batch) % 50 == 0:
            print(f"[gen rank0] {bs + len(chunk)}/{len(my)}", flush=True)

    os.makedirs(a.out_dir, exist_ok=True)
    outpath = os.path.join(a.out_dir, f"{a.output_name}_rank{rank}.json")
    with open(outpath, "w") as f:
        json.dump({a.output_name: results}, f)
    print(f"[gen rank{rank}] wrote {len(results)} -> {outpath}", flush=True)


if __name__ == "__main__":
    main()
