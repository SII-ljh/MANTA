"""Build synthetic long-context evaluation set from unified data.

Strategy: pick a question with a "needle" (its original context contains the
answer), then pad the context with random distractor contexts from other
unified samples until reaching a target token length. Output uses the same
schema as unified_*_instruction_dataset.json so downstream tools (precompute,
evaluator) work without modification.

CPU-only. Uses Llama tokenizer for accurate token counts.

Usage (2222 server):
    python scripts/build_long_context_eval.py \
        --source unified --split test \
        --target_lengths 4096 8192 16384 32768 \
        --samples_per_length 500 \
        --out_dir data/long_context_eval
"""

import argparse
import json
import os
import random
import sys

from transformers import AutoTokenizer

BASE = os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_ID = os.environ.get("MANTA_MODEL_ID", "meta-llama/Llama-3.1-8B")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="unified",
                   choices=["unified", "extended_qa", "mrqa", "r500x"])
    p.add_argument("--split", default="test", choices=["train", "test"])
    p.add_argument("--target_lengths", type=int, nargs="+",
                   default=[4096, 8192, 16384, 32768])
    p.add_argument("--samples_per_length", type=int, default=500)
    p.add_argument("--out_dir", default=os.path.join(BASE, "data", "long_context_eval"))
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--needle_subset",
                   default="BioASQ,SQuAD,HotpotQA,SearchQA,NewsQA,NaturalQuestionsShort,WikiHop,2WikiMQA,MuSiQue",
                   help="comma-separated subsets eligible as the needle question")
    p.add_argument("--max_distractors", type=int, default=200,
                   help="cap on number of distractor contexts to concatenate")
    return p.parse_args()


def load_source(source, split):
    fname = f"{source}_{split}_instruction_dataset.json"
    path = os.path.join(BASE, "sft", "output", fname)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    args = parse_args()
    random.seed(args.seed)

    print(f"[build] loading {args.source}/{args.split}", flush=True)
    data = load_source(args.source, args.split)
    print(f"[build] {len(data)} samples loaded", flush=True)

    print(f"[build] tokenizer: {MODEL_ID}", flush=True)
    tok = AutoTokenizer.from_pretrained(MODEL_ID)

    # Pre-tokenize all contexts to estimate lengths
    print("[build] pre-tokenizing contexts (this may take a few minutes)...", flush=True)
    cached_ctx_ids = []
    for i, ex in enumerate(data):
        ids = tok(ex["context"], add_special_tokens=False)["input_ids"]
        cached_ctx_ids.append(ids)
        if (i + 1) % 5000 == 0:
            print(f"  tokenized {i+1}/{len(data)}", flush=True)

    needle_subsets = set(args.needle_subset.split(","))
    eligible_needle_idxes = [
        i for i, ex in enumerate(data)
        if ex.get("subset", ex.get("dataset")) in needle_subsets
        and len(cached_ctx_ids[i]) >= 100  # need a non-trivial original context
        and len(cached_ctx_ids[i]) <= 4000  # small enough to fit in shortest length
    ]
    print(f"[build] {len(eligible_needle_idxes)} eligible needle samples", flush=True)

    distractor_pool = list(range(len(data)))

    os.makedirs(args.out_dir, exist_ok=True)

    for target_len in args.target_lengths:
        rng = random.Random(args.seed + target_len)
        out_path = os.path.join(args.out_dir,
                                f"long_{target_len}_{args.split}_instruction_dataset.json")
        out_samples = []
        attempts = 0
        rng.shuffle(eligible_needle_idxes)
        for needle_idx in eligible_needle_idxes:
            if len(out_samples) >= args.samples_per_length:
                break
            needle = data[needle_idx]
            needle_ctx_ids = cached_ctx_ids[needle_idx]
            if len(needle_ctx_ids) > target_len - 200:
                continue  # leave room for distractors

            distractors_used = []
            tokens_so_far = len(needle_ctx_ids)
            random_pool = rng.sample(distractor_pool, min(len(distractor_pool), args.max_distractors * 2))
            for d_idx in random_pool:
                if d_idx == needle_idx:
                    continue
                d_ids = cached_ctx_ids[d_idx]
                if len(d_ids) < 50:
                    continue
                if tokens_so_far + len(d_ids) > target_len + 500:
                    continue
                distractors_used.append(d_idx)
                tokens_so_far += len(d_ids)
                if tokens_so_far >= target_len - 200:
                    break

            if tokens_so_far < target_len * 0.85:
                attempts += 1
                continue

            insert_pos = rng.randint(0, len(distractors_used))
            ordered_idxes = (
                distractors_used[:insert_pos]
                + [needle_idx]
                + distractors_used[insert_pos:]
            )

            joined_context = "\n\n".join(data[j]["context"] for j in ordered_idxes)

            out_samples.append({
                "context": joined_context,
                "question": needle["question"],
                "answers": needle["answers"],
                "subset": f"long_{target_len}_{needle.get('subset', needle.get('dataset', 'unknown'))}",
                "needle_position": insert_pos,
                "num_distractors": len(distractors_used),
                "approx_tokens": tokens_so_far,
                "source_sample_idx": needle_idx,
            })

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(out_samples, f, ensure_ascii=False, indent=2)
        print(f"[build] wrote {len(out_samples)} samples to {out_path} "
              f"(skipped {attempts} due to length mismatch)", flush=True)


if __name__ == "__main__":
    main()
