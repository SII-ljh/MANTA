#!/usr/bin/env python
"""Tokenize long_context_eval needle-QA sets (4K/8K/16K/32K) into SAC eval format.

Source: data/long_context_eval/long_{L}_test_instruction_dataset.json
        list[{context, question, answers, subset, needle_position, ...}] (500/length)
Output: data/longctx_eval/qa_eval_{L}.pt        list[{input_ids, lm_targets}]
        data/longctx_eval/qa_eval_{L}_meta.json list[{id, dataset, question, answers}]

Schema + prompt format mirror scripts/prepare_longbench_eval.py exactly
(### Context / ### Question / ### Answer, BOS, middle truncation) so the
existing eval_k_sweep.py / eval_v5_pareto.py drivers work unchanged.
"""
import argparse, hashlib, json, os
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id",
                   default=os.environ.get("MANTA_MODEL_ID", "meta-llama/Llama-3.1-8B"))
    p.add_argument("--raw_dir", default="data/long_context_eval")
    p.add_argument("--out_dir", default="data/longctx_eval")
    p.add_argument("--lengths", nargs="+", default=["4096", "8192", "16384", "32768"])
    p.add_argument("--max_ctx_tokens", type=int, default=32000)
    return p.parse_args()

def middle_truncate(ids, cap):
    if len(ids) <= cap:
        return ids
    half = cap // 2
    return ids[:half] + ids[-half:]

def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_id)
    bos = [tok.bos_token_id]
    ctx_header = tok("### Context:\n", add_special_tokens=False)["input_ids"]
    q_header = tok("\n### Question:\n", add_special_tokens=False)["input_ids"]
    a_header = tok("\n### Answer:\n", add_special_tokens=False)["input_ids"]
    for L in args.lengths:
        src = os.path.join(args.raw_dir, f"long_{L}_test_instruction_dataset.json")
        data = json.load(open(src))
        pt, meta = [], []
        for i, ex in enumerate(tqdm(data, desc=f"L={L}")):
            raw_ctx = tok(ex["context"], add_special_tokens=False)["input_ids"]
            raw_q = tok(ex["question"], add_special_tokens=False)["input_ids"]
            overhead = len(bos) + len(ctx_header) + len(q_header) + len(raw_q) + len(a_header) + 4
            budget = max(0, args.max_ctx_tokens - overhead)
            ctx_trunc = middle_truncate(raw_ctx, budget)
            input_ids = bos + ctx_header + ctx_trunc
            prompt = q_header + raw_q + a_header
            uid = hashlib.sha224("{}_{}_{}".format(L, i, ex["question"]).encode()).hexdigest()[:48]
            pt.append({"input_ids": torch.LongTensor(input_ids),
                       "lm_targets": torch.LongTensor(prompt)})
            meta.append({"id": uid, "dataset": ex.get("subset", f"long_{L}"),
                         "question": ex["question"],
                         "answers": [str(a) for a in ex.get("answers", [])]})
        torch.save(pt, os.path.join(args.out_dir, f"qa_eval_{L}.pt"))
        json.dump(meta, open(os.path.join(args.out_dir, f"qa_eval_{L}_meta.json"), "w"))
        lens = [len(x["input_ids"]) for x in pt]
        print(f"[L={L}] {len(pt)} samples  input tokens min/mean/max = "
              f"{min(lens)}/{sum(lens)//len(lens)}/{max(lens)}")
    print("[done]", args.out_dir)

if __name__ == "__main__":
    main()
