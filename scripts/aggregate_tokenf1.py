#!/usr/bin/env python3
"""Aggregate SQuAD-normalized token-F1 (second metric, unified standard v2 2026-06-11)
from eval rank*.json files into a compact report.

Per key (k or eps setting): per-dataset mean -> IID/OOD macro (mirrors SAC-aligned
ROUGE protocol) + overall sample mean. IID = 6 MRQA in-domain datasets (prefix match).

Usage:
  python scripts/aggregate_tokenf1.py --sources "LABEL=GLOB" [--out OUT.json]
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
                    # FAIL LOUD on missing f1/em (2026-06-13, F64): bridge rank
                    # files (ICAE/c500x) carry only `generated` -- silently
                    # defaulting to 0.0 produced plausible-looking all-zero
                    # token-F1 reports. Use scripts/rescore_tokenf1_from_generated.py
                    # for those files instead.
                    if "f1" not in s or "em" not in s:
                        raise KeyError(
                            f"{f}: sample id={s.get('id','?')} has no 'f1'/'em' "
                            f"field. This rank file was not produced by "
                            f"eval_k_sweep.py; rescore it with "
                            f"scripts/rescore_tokenf1_from_generated.py.")
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
