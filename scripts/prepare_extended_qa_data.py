"""Extended QA data preparation for SAC — full migration from R500xCompressor.

Reads pre-truncated JSONL from R500xCompressor's ctx_curriculum/ directory
(TARGET_CTX_TOKENS=12000) and converts to SAC's tokenized .pt format.

Covers ALL datasets from R500x:
  - Original QA: squad, hotpotqa, race, triviaqa, narrativeqa, qasper,
                  2wikimqa, musique, drop, msc, niah(babilong)
  - MRQA/short:  newsqa, searchqa, naturalquestions, bioasq, textbookqa,
                  duorc, wikihop
  - LongBench:   longbench_hotpotqa, longbench_musique, longbench_narrativeqa,
                  longbench_2wikimqa, longbench_qasper, longbench_triviaqa,
                  longbench_multifieldqa_en, longbench_passage_retrieval_en
  - LongBenchV2: longbenchv2
  - VTCBench:    vtcbench_memory, vtcbench_retrieval, vtcbench_reasoning,
                  vtcbench_paperqa
  - Others:      scbench, quality

Outputs (in sft/output/):
  - r500x_train_instruction_dataset.json  (intermediate JSON, subset field)
  - r500x_test_instruction_dataset.json   (intermediate JSON, subset field)
  - r500x_train_{model}_{N}samples_instruction.pt
  - r500x_eval_{model}_{N}samples_instruction.pt

Run on 2222 (CPU, internet access):
    conda activate manta
    python scripts/prepare_extended_qa_data.py
"""

import json
import os
import random
import sys
from collections import defaultdict

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

# ---------------------------------------------------------------------------
# Paths & Config
# ---------------------------------------------------------------------------
SAC_BASE = os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
R500X_CTX = os.path.join(os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw", "ctx_curriculum")
SFT_OUTPUT = os.path.join(SAC_BASE, "sft", "output")
LLAMA_PATH = os.environ.get("MANTA_MODEL_ID", "meta-llama/Llama-3.1-8B")

SEED = 42
random.seed(SEED)

# Max train samples per dataset (matching R500x training config)
# Datasets not listed here use all available samples
TRAIN_SAMPLE_CAPS = {
    "squad": 49,           # very short context, only 49 available
    "race": 1913,          # short context, use all available
    "drop": 1901,          # short context, use all available
    "newsqa": 50000,
    "searchqa": 50000,
    "naturalquestions": 50000,
    "duorc": 50000,
    "hotpotqa": 40000,
    "triviaqa": 40000,
    "2wikimqa": 40000,
    "musique": 20000,
    "narrativeqa": 15000,
    "msc": 10000,
    "qasper": 5000,
    "quality": 5000,
    "wikihop": 20000,
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def read_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    return data


def write_json(data, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"  Wrote {len(data)} samples to {path}")


def dataset_to_subset(dataset_name):
    """Map R500x dataset name to SAC subset name for evaluation grouping."""
    mapping = {
        "squad": "SQuAD",
        "hotpotqa": "HotpotQA",
        "race": "RACE",
        "triviaqa": "TriviaQA",
        "narrativeqa": "NarrativeQA",
        "qasper": "Qasper",
        "2wikimqa": "2WikiMQA",
        "musique": "MuSiQue",
        "drop": "DROP",
        "msc": "MSC",
        "babilong": "BABILong",
        "newsqa": "NewsQA",
        "searchqa": "SearchQA",
        "naturalquestions": "NaturalQuestions",
        "bioasq": "BioASQ",
        "textbookqa": "TextbookQA",
        "duorc": "DuoRC",
        "wikihop": "WikiHop",
        "quality": "QuALITY",
        "scbench": "SCBench",
        "longbench_hotpotqa": "LB_HotpotQA",
        "longbench_musique": "LB_MuSiQue",
        "longbench_narrativeqa": "LB_NarrativeQA",
        "longbench_2wikimqa": "LB_2WikiMQA",
        "longbench_qasper": "LB_Qasper",
        "longbench_triviaqa": "LB_TriviaQA",
        "longbench_multifieldqa_en": "LB_MultifieldQA",
        "longbench_passage_retrieval_en": "LB_PassageRetrieval",
        "longbenchv2": "LongBenchV2",
        "vtcbench_memory": "VTC_Memory",
        "vtcbench_retrieval": "VTC_Retrieval",
        "vtcbench_reasoning": "VTC_Reasoning",
        "vtcbench_paperqa": "VTC_PaperQA",
    }
    return mapping.get(dataset_name, dataset_name)


# ---------------------------------------------------------------------------
# Load all R500x ctx_curriculum data
# ---------------------------------------------------------------------------
def load_r500x_data():
    """Load all JSONL files from R500x ctx_curriculum, split into train/eval."""
    eval_dir = os.path.join(R500X_CTX, "eval")
    train_dir = os.path.join(R500X_CTX, "train")

    # --- Eval data ---
    print("\n=== Loading R500x eval data ===")
    eval_data = []
    eval_files = sorted([
        f for f in os.listdir(eval_dir)
        if f.endswith(".jsonl") and f != "all_eval.jsonl"
    ])
    for fname in eval_files:
        ds_name = fname.replace(".jsonl", "")
        path = os.path.join(eval_dir, fname)
        items = read_jsonl(path)
        # Add subset field for SAC evaluation
        for item in items:
            item["subset"] = dataset_to_subset(item.get("dataset", ds_name))
        eval_data.extend(items)
        print(f"  {ds_name}: {len(items)} samples")

    # --- Train data ---
    print("\n=== Loading R500x train data ===")
    train_data = []
    train_files = sorted([
        f for f in os.listdir(train_dir)
        if f.endswith(".jsonl") and "all_train" not in f
    ])
    for fname in train_files:
        ds_name = fname.replace(".jsonl", "")
        # Handle babilong split files
        base_ds = ds_name.split("_train_")[0] if "_train_" in ds_name else ds_name
        path = os.path.join(train_dir, fname)
        items = read_jsonl(path)
        for item in items:
            item["subset"] = dataset_to_subset(item.get("dataset", base_ds))

        # Apply sample cap
        cap = TRAIN_SAMPLE_CAPS.get(base_ds)
        if cap and len(items) > cap:
            random.shuffle(items)
            items = items[:cap]

        train_data.extend(items)
        print(f"  {ds_name}: {len(items)} samples")

    random.shuffle(train_data)
    return train_data, eval_data


# ---------------------------------------------------------------------------
# Tokenize to SAC .pt format
# ---------------------------------------------------------------------------
def tokenize_examples(examples, tokenizer, split):
    """Convert JSON examples to SAC's tokenized tensor format.

    Train: {input_ids, lm_targets, instruction_target}
    Eval:  {input_ids, lm_targets}
    """
    result = []
    for ex in tqdm(examples, desc=f"Tokenizing {split}"):
        context_ids = [tokenizer.bos_token_id] + tokenizer(
            "### Context:\n" + ex["context"], add_special_tokens=False
        )["input_ids"]

        question_ids = tokenizer(
            "\n### Question:\n" + ex["question"] + "\n### Answer:\n",
            add_special_tokens=False
        )["input_ids"]

        answer_text = ex.get("answer", ex.get("answers", [""])[0])
        answer_ids = tokenizer(
            answer_text, add_special_tokens=False
        )["input_ids"] + [tokenizer.eos_token_id]

        input_ids = torch.LongTensor(context_ids)

        if split == "train":
            lm_targets = torch.LongTensor(question_ids + answer_ids)
            instruction_target = [-100] * len(question_ids) + list(answer_ids)
            instruction_target = instruction_target[1:]
            instruction_target = torch.LongTensor(instruction_target)
            result.append({
                "input_ids": input_ids,
                "lm_targets": lm_targets,
                "instruction_target": instruction_target,
            })
        else:
            lm_targets = torch.LongTensor(question_ids)
            result.append({
                "input_ids": input_ids,
                "lm_targets": lm_targets,
            })

    return result


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    os.makedirs(SFT_OUTPUT, exist_ok=True)

    # Check if already generated
    json_train_path = os.path.join(SFT_OUTPUT, "r500x_train_instruction_dataset.json")
    json_test_path = os.path.join(SFT_OUTPUT, "r500x_test_instruction_dataset.json")

    if os.path.exists(json_train_path) and os.path.exists(json_test_path):
        print(f"[INFO] JSON caches already exist, loading...")
        with open(json_train_path, "r") as f:
            train_data = json.load(f)
        with open(json_test_path, "r") as f:
            eval_data = json.load(f)
    else:
        train_data, eval_data = load_r500x_data()

        # Save intermediate JSON (for evaluator backward compat)
        print("\n=== Saving intermediate JSON ===")
        write_json(train_data, json_train_path)
        write_json(eval_data, json_test_path)

    # Print summary
    print(f"\n=== Data Summary ===")
    print(f"Train: {len(train_data)} samples")
    print(f"Eval:  {len(eval_data)} samples")

    train_subsets = defaultdict(int)
    eval_subsets = defaultdict(int)
    for ex in train_data:
        train_subsets[ex["subset"]] += 1
    for ex in eval_data:
        eval_subsets[ex["subset"]] += 1

    print(f"\nTrain per subset:")
    for s in sorted(train_subsets):
        print(f"  {s}: {train_subsets[s]}")
    print(f"\nEval per subset:")
    for s in sorted(eval_subsets):
        print(f"  {s}: {eval_subsets[s]}")

    # Tokenize
    print("\n=== Tokenizing ===")
    tokenizer = AutoTokenizer.from_pretrained(LLAMA_PATH)

    train_samples_num = len(train_data)
    model_name = LLAMA_PATH.split("/")[-1]

    pt_train_path = os.path.join(
        SFT_OUTPUT,
        f"r500x_train_{model_name}_{train_samples_num}samples_instruction.pt"
    )
    pt_eval_path = os.path.join(
        SFT_OUTPUT,
        f"r500x_eval_{model_name}_{train_samples_num}samples_instruction.pt"
    )

    if os.path.exists(pt_train_path) and os.path.exists(pt_eval_path):
        print(f"[INFO] .pt files already exist, skipping tokenization")
        print(f"  Train: {pt_train_path}")
        print(f"  Eval:  {pt_eval_path}")
    else:
        train_tensors = tokenize_examples(train_data, tokenizer, "train")
        eval_tensors = tokenize_examples(eval_data, tokenizer, "eval")

        print(f"\n=== Saving .pt files ===")
        torch.save(train_tensors, pt_train_path)
        torch.save(eval_tensors, pt_eval_path)
        print(f"  Train: {pt_train_path} ({len(train_tensors)} samples)")
        print(f"  Eval:  {pt_eval_path} ({len(eval_tensors)} samples)")

        # Print token length stats
        train_ctx_lens = [len(ex["input_ids"]) for ex in train_tensors[:10000]]
        eval_ctx_lens = [len(ex["input_ids"]) for ex in eval_tensors]
        print(f"\n=== Token Length Stats ===")
        print(f"  Train (first 10k): mean={np.mean(train_ctx_lens):.0f}, "
              f"median={np.median(train_ctx_lens):.0f}, "
              f"min={min(train_ctx_lens)}, max={max(train_ctx_lens)}")
        print(f"  Eval: mean={np.mean(eval_ctx_lens):.0f}, "
              f"median={np.median(eval_ctx_lens):.0f}, "
              f"min={min(eval_ctx_lens)}, max={max(eval_ctx_lens)}")

    print("\nDone.")


if __name__ == "__main__":
    main()
