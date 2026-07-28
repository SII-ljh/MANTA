"""
Convert user's phase2 QA data to SAC SFT format.
Train: {input_ids, lm_targets, instruction_target}
Eval:  {input_ids, lm_targets, meta} (meta for F1 computation)
"""
import os
import sys
import json
import random
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


DATA_ROOT = os.path.join(os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw", "phase2")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=os.environ.get("MANTA_MODEL_ID", "meta-llama/Llama-3.1-8B"))
    parser.add_argument("--output_dir", type=str, default="../experiment/sac_8b/output")
    parser.add_argument("--max_context_tokens", type=int, default=3060,
                        help="Max context tokens (6 chunks * 510)")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def load_jsonl(path):
    data = []
    with open(path) as f:
        for line in f:
            data.append(json.loads(line))
    return data


def process_example(example, tokenizer, max_ctx, split):
    """Convert one QA example to SAC format."""
    context_ids = [tokenizer.bos_token_id] + tokenizer(
        "### Context:\n" + example["context"], add_special_tokens=False
    )["input_ids"]
    
    question_ids = tokenizer(
        "\n### Question:\n" + example["question"] + "\n### Answer:\n",
        add_special_tokens=False
    )["input_ids"]
    
    answer_text = example["answer"]
    answer_ids = tokenizer(answer_text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
    
    # Truncate context if too long
    if len(context_ids) > max_ctx:
        context_ids = context_ids[:max_ctx]
    
    input_ids = torch.LongTensor(context_ids)
    
    if split == "train":
        lm_targets = torch.LongTensor(question_ids + answer_ids)
        # instruction_target: mask question tokens with -100, keep answer tokens
        instruction_target = [-100] * len(question_ids) + list(answer_ids)
        instruction_target = instruction_target[1:]  # Drop first (consistent with SAC)
        instruction_target = torch.LongTensor(instruction_target)
        return {
            "input_ids": input_ids,
            "lm_targets": lm_targets,
            "instruction_target": instruction_target,
        }
    else:
        # Eval: only question as lm_targets, no answer
        lm_targets = torch.LongTensor(question_ids)
        return {
            "input_ids": input_ids,
            "lm_targets": lm_targets,
        }


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    random.seed(args.seed)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    
    train_out = os.path.join(args.output_dir, "qa_train.pt")
    eval_out = os.path.join(args.output_dir, "qa_eval.pt")
    eval_meta_out = os.path.join(args.output_dir, "qa_eval_meta.json")
    
    if os.path.exists(train_out) and os.path.exists(eval_out):
        print(f"[INFO] Data already exists: {train_out}")
        return
    
    # Load train data
    print("[INFO] Loading train data...")
    train_raw = load_jsonl(os.path.join(DATA_ROOT, "train", "all_train_v2.jsonl"))
    print(f"[INFO] Loaded {len(train_raw)} train examples")
    
    # Load eval data  
    print("[INFO] Loading eval data...")
    eval_raw = load_jsonl(os.path.join(DATA_ROOT, "eval", "all_eval.jsonl"))
    print(f"[INFO] Loaded {len(eval_raw)} eval examples")
    
    # Process train
    print("[INFO] Processing train examples...")
    train_examples = []
    for ex in tqdm(train_raw):
        train_examples.append(process_example(ex, tokenizer, args.max_context_tokens, "train"))
    
    # Process eval
    print("[INFO] Processing eval examples...")
    eval_examples = []
    eval_meta = []
    for ex in tqdm(eval_raw):
        eval_examples.append(process_example(ex, tokenizer, args.max_context_tokens, "eval"))
        eval_meta.append({
            "id": ex.get("id", ""),
            "dataset": ex["dataset"],
            "question": ex["question"],
            "answer": ex["answer"],
            "answers": ex.get("answers", [ex["answer"]]),
            "task_type": ex.get("task_type", ""),
        })
    
    # Save
    print(f"[INFO] Saving {len(train_examples)} train, {len(eval_examples)} eval")
    torch.save(train_examples, train_out)
    torch.save(eval_examples, eval_out)
    with open(eval_meta_out, "w") as f:
        json.dump(eval_meta, f, ensure_ascii=False, indent=2)
    
    print(f"[INFO] Train saved: {train_out}")
    print(f"[INFO] Eval saved: {eval_out}")
    print(f"[INFO] Eval meta saved: {eval_meta_out}")
    
    # Print stats
    ctx_lens = [len(ex["input_ids"]) for ex in train_examples[:5000]]
    import numpy as np
    print(f"\n[STATS] Context tokens: mean={np.mean(ctx_lens):.0f}, max={max(ctx_lens)}")


if __name__ == "__main__":
    main()
