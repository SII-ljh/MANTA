#!/usr/bin/env python3
"""Aggregate token-F1/EM reports from per-rank evaluator output.

Scores are averaged per dataset, then macro-averaged over the IID and OOD
splits. Each input sample must contain `f1`, `em`, and `dataset` fields.
"""
import argparse, glob, json, statistics as st
from collections import defaultdict

IID_PREFIX = ("SQuAD", "NewsQA", "TriviaQA", "SearchQA", "HotpotQA", "NaturalQuestions")

def is_iid(ds): return any(ds.startswith(p) for p in IID_PREFIX)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", action="append", required=True, help="LABEL=GLOB (repeatable)")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    report = {}
    for src in args.sources:
        label, pat = src.split("=", 1)
        per_key = defaultdict(lambda: defaultdict(list))  # key -> dataset -> [f1]
        per_key_em = defaultdict(lambda: defaultdict(list))
        extras = defaultdict(dict)
        files = sorted(glob.glob(pat))
        if not files:
            print(f"[warn] no files for {pat}"); continue
        for f in files:
            j = json.load(open(f))
            for key, samples in j.items():
                if not isinstance(samples, list):
                    continue
                for s in samples:
                    # Require scored records; missing fields indicate a bad run.
                    if "f1" not in s or "em" not in s:
                        raise KeyError(
                            f"{f}: sample id={s.get('id','?')} has no 'f1'/'em' "
                            f"field. Check the evaluator output.")
                    per_key[key][s["dataset"]].append(s["f1"])
                    per_key_em[key][s["dataset"]].append(s["em"])
                    for ek in ("mean_k", "mean_compression"):
                        if ek in s: extras[key][ek] = None  # mark presence
        for key, ds_map in per_key.items():
            iid_means = [st.mean(v) for d, v in ds_map.items() if is_iid(d)]
            ood_means = [st.mean(v) for d, v in ds_map.items() if not is_iid(d)]
            all_f1 = [x for v in ds_map.values() for x in v]
            em_map = per_key_em[key]
            iid_em = [st.mean(v) for d, v in em_map.items() if is_iid(d)]
            ood_em = [st.mean(v) for d, v in em_map.items() if not is_iid(d)]
            report[f"{label} [{key}]"] = {
                "tok_iid_f1": round(100 * st.mean(iid_means), 2) if iid_means else None,
                "tok_ood_f1": round(100 * st.mean(ood_means), 2) if ood_means else None,
                "tok_iid_em": round(100 * st.mean(iid_em), 2) if iid_em else None,
                "tok_ood_em": round(100 * st.mean(ood_em), 2) if ood_em else None,
                "tok_overall_f1": round(100 * st.mean(all_f1), 2),
                "n": len(all_f1),
                "n_datasets_iid": len(iid_means), "n_datasets_ood": len(ood_means),
            }
    txt = json.dumps(report, indent=1, ensure_ascii=False)
    print(txt)
    if args.out:
        open(args.out, "w").write(txt)
        print(f"[saved] {args.out}")

if __name__ == "__main__":
    main()
