"""Prepare merged training & test dataset: MRQA IID + R500x (non-overlap).

Design:
  Training:
    - MRQA IID (6 datasets, MRQA format, capped) -> matches MRQA test format
    - R500x (non-MRQA-overlap, non-MRQA-OOD) -> matches R500x test format
    - R500x TriviaQA kept separately (different from MRQA TriviaQA-web)

  Test:
    - MRQA full test (6 IID + 6 OOD, 67,854 samples)
    - R500x-unique test subsets (not in MRQA)

  Removed from training:
    - R500x SQuAD/NewsQA/SearchQA/HotpotQA/NQ (replaced by MRQA versions)
    - R500x DROP/DuoRC/RACE (MRQA OOD, eval-only)

Usage:
    conda activate manta
    python scripts/prepare_merged_data.py
"""

import json
import os
import random
from collections import Counter

SEED = 42
random.seed(SEED)

BASE = os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SFT_OUTPUT = os.path.join(BASE, "sft", "output")

# --- MRQA IID training caps ---
MRQA_IID_CAPS = {
    "SQuAD": 40000,
    "NewsQA": 30000,
    "SearchQA": 40000,
    "HotpotQA": 40000,
    "NaturalQuestionsShort": 40000,
    "TriviaQA-web": 40000,
}

MRQA_IID_SUBSETS = set(MRQA_IID_CAPS.keys())

# R500x subsets to REMOVE from training:
#   1. Overlapping with MRQA IID (replaced by MRQA format versions)
#   2. MRQA OOD datasets (eval-only)
R500X_REMOVE_TRAIN = {
    # MRQA IID overlap (replaced by MRQA versions)
    "SQuAD", "NewsQA", "SearchQA", "HotpotQA", "NaturalQuestions",
    # MRQA OOD (eval-only)
    "DROP", "DuoRC", "RACE",
}

# R500x test subsets that overlap with MRQA test (skip to avoid duplication)
# Note: TriviaQA (R500x) != TriviaQA-web (MRQA), so we keep TriviaQA
MRQA_TEST_SUBSETS = {
    "SQuAD", "NewsQA", "SearchQA", "HotpotQA",
    "NaturalQuestionsShort", "TriviaQA-web",
    "BioASQ", "DROP", "DuoRC.ParaphraseRC", "RACE",
    "RelationExtraction", "TextbookQA",
}

# R500x test subset names that overlap with MRQA (skip these)
R500X_TEST_SKIP = {
    "SQuAD", "NewsQA", "SearchQA", "HotpotQA",
    "NaturalQuestions",  # overlaps with NaturalQuestionsShort
    "BioASQ", "DROP", "DuoRC", "RACE", "TextbookQA",
}


def load_json(filename):
    path = os.path.join(SFT_OUTPUT, filename)
    print(f"Loading {path} ...")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  -> {len(data)} samples")
    return data


def save_json(data, filename):
    path = os.path.join(SFT_OUTPUT, filename)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    print(f"Saved {len(data)} samples to {path}")


def normalize_sample(sample):
    """Normalize to common format: {context, question, answers, subset}."""
    return {
        "context": sample["context"],
        "question": sample["question"],
        "answers": sample["answers"],
        "subset": sample.get("subset", "unknown"),
    }


def build_train():
    print("\n" + "=" * 60)
    print("Building MERGED training set")
    print("=" * 60)

    # --- Part 1: MRQA IID ---
    mrqa_train = load_json("mrqa_train_instruction_dataset.json")

    # Group MRQA by subset
    mrqa_by_subset = {}
    for s in mrqa_train:
        sub = s.get("subset", "unknown")
        if sub not in mrqa_by_subset:
            mrqa_by_subset[sub] = []
        mrqa_by_subset[sub].append(s)

    mrqa_selected = []
    print("\nMRQA IID selection:")
    for subset, cap in MRQA_IID_CAPS.items():
        samples = mrqa_by_subset.get(subset, [])
        if len(samples) > cap:
            random.shuffle(samples)
            samples = samples[:cap]
        mrqa_selected.extend([normalize_sample(s) for s in samples])
        print(f"  {subset}: {len(samples)} (cap={cap}, available={len(mrqa_by_subset.get(subset, []))})")
    print(f"  MRQA IID total: {len(mrqa_selected)}")

    # --- Part 2: R500x (filtered) ---
    r500x_train = load_json("r500x_train_instruction_dataset.json")

    r500x_selected = []
    r500x_removed = Counter()
    r500x_kept = Counter()
    for s in r500x_train:
        sub = s.get("subset", "unknown")
        if sub in R500X_REMOVE_TRAIN:
            r500x_removed[sub] += 1
        else:
            r500x_selected.append(normalize_sample(s))
            r500x_kept[sub] += 1

    print("\nR500x kept:")
    for k, v in sorted(r500x_kept.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print(f"  R500x kept total: {len(r500x_selected)}")

    print("\nR500x removed:")
    for k, v in sorted(r500x_removed.items(), key=lambda x: -x[1]):
        reason = "MRQA IID overlap" if k not in {"DROP", "DuoRC", "RACE"} else "MRQA OOD"
        print(f"  {k}: {v} ({reason})")

    # --- Merge ---
    merged_train = mrqa_selected + r500x_selected
    random.shuffle(merged_train)

    print(f"\nMerged training total: {len(merged_train)}")
    tc = Counter(s["subset"] for s in merged_train)
    print("Per subset:")
    for k, v in sorted(tc.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    return merged_train


def build_test():
    print("\n" + "=" * 60)
    print("Building MERGED test set")
    print("=" * 60)

    # --- Part 1: MRQA test (all 12 subsets) ---
    mrqa_test = load_json("mrqa_test_instruction_dataset.json")
    mrqa_normalized = [normalize_sample(s) for s in mrqa_test]
    print(f"\nMRQA test: {len(mrqa_normalized)} samples")
    tc = Counter(s["subset"] for s in mrqa_normalized)
    for k, v in sorted(tc.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # --- Part 2: R500x test (unique subsets only) ---
    r500x_test = load_json("r500x_test_instruction_dataset.json")

    r500x_unique = []
    r500x_skipped = Counter()
    r500x_kept = Counter()
    for s in r500x_test:
        sub = s.get("subset", "unknown")
        if sub in R500X_TEST_SKIP:
            r500x_skipped[sub] += 1
        else:
            r500x_unique.append(normalize_sample(s))
            r500x_kept[sub] += 1

    print(f"\nR500x test kept (unique subsets): {len(r500x_unique)} samples")
    for k, v in sorted(r500x_kept.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    print(f"\nR500x test skipped (overlap with MRQA): {sum(r500x_skipped.values())}")
    for k, v in sorted(r500x_skipped.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    # --- Merge ---
    merged_test = mrqa_normalized + r500x_unique
    print(f"\nMerged test total: {len(merged_test)}")
    tc = Counter(s["subset"] for s in merged_test)
    print("Per subset:")
    for k, v in sorted(tc.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")

    return merged_test


def main():
    train = build_train()
    test = build_test()

    save_json(train, "merged_train_instruction_dataset.json")
    save_json(test, "merged_test_instruction_dataset.json")

    # --- Verification ---
    print("\n" + "=" * 60)
    print("Verification")
    print("=" * 60)

    train_subsets = set(s["subset"] for s in train)
    test_subsets = set(s["subset"] for s in test)

    mrqa_ood = {"BioASQ", "DROP", "DuoRC.ParaphraseRC", "RACE", "RelationExtraction", "TextbookQA"}
    mrqa_iid = set(MRQA_IID_CAPS.keys())

    print("\nMRQA IID in training:", mrqa_iid & train_subsets)
    print("MRQA OOD in training:", mrqa_ood & train_subsets)
    if mrqa_ood & train_subsets:
        print("  WARNING: MRQA OOD datasets found in training!")
    else:
        print("  OK: No MRQA OOD in training")

    print("\nTest subsets not in training (OOD):")
    ood_test = test_subsets - train_subsets
    for sub in sorted(ood_test):
        count = sum(1 for s in test if s["subset"] == sub)
        print(f"  {sub}: {count}")

    print("\nDone.")


if __name__ == "__main__":
    main()
