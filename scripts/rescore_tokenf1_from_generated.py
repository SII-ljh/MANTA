#!/usr/bin/env python3
"""Rescore SQuAD-normalized token-F1/EM (second metric, unified standard v2)
from bridge-style rank files that carry only `generated` text (ICAE / c500x
bridges), using the gold answers in qa_eval_meta.json.

Added 2026-06-13 (review F64): the baseline bridges emit
  { "<key>": [ {"id","dataset","generated",...}, ... ] }
without per-sample f1/em, so aggregate_tokenf1.py cannot consume them (and used
to silently emit all-zero reports). This script computes max-over-references
token-F1 + normalized EM (identical functions to eval_k_sweep.py) and writes
the same compact report schema as aggregate_tokenf1.py.

Usage:
  python scripts/rescore_tokenf1_from_generated.py \
      --meta data/mrqa_eval_subset/qa_eval_meta.json \
      --sources "LABEL=GLOB" [--sources ...] [--out OUT.json]
"""
import argparse
import glob
import json
import re
import statistics as st
import string
from collections import Counter, defaultdict

IID_PREFIX = ("SQuAD", "NewsQA", "TriviaQA", "SearchQA", "HotpotQA",
              "NaturalQuestions")


def is_iid(ds):
    return any(ds.startswith(p) for p in IID_PREFIX)


# ---- identical to scripts/eval_k_sweep.py ----
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", required=True, help="qa_eval_meta.json (answers)")
    ap.add_argument("--sources", action="append", required=True,
                    help="LABEL=GLOB (repeatable)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    meta = json.load(open(args.meta))
    by_id = {}
    for m in meta:
        key = (m["id"], m["dataset"])
        if key in by_id:
            raise ValueError(f"duplicate (id,dataset) in meta: {key}")
        by_id[key] = m.get("answers") or [m.get("answer", "")]

    report = {}
    for src in args.sources:
        label, pat = src.split("=", 1)
        files = sorted(glob.glob(pat))
        if not files:
            print(f"[warn] no files for {pat}")
            continue
        per_key_f1 = defaultdict(lambda: defaultdict(list))
        per_key_em = defaultdict(lambda: defaultdict(list))
        n_missing = 0
        for f in files:
            j = json.load(open(f))
            for key, samples in j.items():
                if not isinstance(samples, list):
                    continue
                for s in samples:
                    refs = by_id.get((s["id"], s["dataset"]))
                    if refs is None:
                        n_missing += 1
                        continue
                    gen = s.get("generated", "")
                    f1 = max(token_f1(gen, r) for r in refs)
                    em = max(exact_match(gen, r) for r in refs)
                    per_key_f1[key][s["dataset"]].append(f1)
                    per_key_em[key][s["dataset"]].append(em)
        if n_missing:
            print(f"[warn] {label}: {n_missing} samples had no meta match "
                  f"(wrong --meta for this rank file?)")
        for key, ds_map in per_key_f1.items():
            iid = [st.mean(v) for d, v in ds_map.items() if is_iid(d)]
            ood = [st.mean(v) for d, v in ds_map.items() if not is_iid(d)]
            em_map = per_key_em[key]
            iid_em = [st.mean(v) for d, v in em_map.items() if is_iid(d)]
            ood_em = [st.mean(v) for d, v in em_map.items() if not is_iid(d)]
            all_f1 = [x for v in ds_map.values() for x in v]
            report[f"{label} [{key}]"] = {
                "tok_iid_f1": round(100 * st.mean(iid), 2) if iid else None,
                "tok_ood_f1": round(100 * st.mean(ood), 2) if ood else None,
                "tok_iid_em": round(100 * st.mean(iid_em), 2) if iid_em else None,
                "tok_ood_em": round(100 * st.mean(ood_em), 2) if ood_em else None,
                "tok_overall_f1": round(100 * st.mean(all_f1), 2),
                "n": len(all_f1),
                "n_datasets_iid": len(iid), "n_datasets_ood": len(ood),
            }

    txt = json.dumps(report, indent=1, ensure_ascii=False)
    print(txt)
    if args.out:
        open(args.out, "w").write(txt)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
