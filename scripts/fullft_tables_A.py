#!/usr/bin/env python3
"""Phase-A CPU-only tables for the full-FT (no-LoRA) 1B variant, full-58K.

Produces, from already-computed eval summaries (no GPU):
  - Table 1 analog: per-dataset token-F1/EM at the ε=0.05 operating point
    (full-FT vs v3-LoRA vs dense full-FT topline), macro IID(6)/OOD(6).
  - Table 4a analog: fixed-vs-adaptive at MATCHED mean_k for the full-FT ckpt
    (adaptive ε point minus linear-interpolated fixed-k curve at same mean_k),
    replicating the paper's F68 "training-anchor linear interpolation" recipe.

Metric convention (see memory eval-iid-ood-macro-average-convention):
  macro = equal-weight mean over the 6 datasets in a split (NOT micro).
  compressor summaries store f1/em as FRACTIONS (0-1) -> x100 here.
  topline per-dataset report stores PERCENTAGES -> left as-is (guarded).
"""
import os
import json, sys

ROOT = os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IID = ["SQuAD", "NewsQA", "TriviaQA-web", "SearchQA", "HotpotQA", "NaturalQuestionsShort"]
OOD = ["BioASQ", "DROP", "DuoRC.ParaphraseRC", "RACE", "RelationExtraction", "TextbookQA"]
FF  = "experiment/pact_v5_mrqa_fullft_1B/output"
V3  = "experiment/pact_v5_mrqa_dyneps_k32_1B_v3/output"


def L(path):
    return json.load(open(f"{ROOT}/{path}"))


def pct(x):
    """fraction(0-1) -> percent; already-percent left as-is."""
    return x * 100.0 if x <= 1.5 else x


def macro(per_ds, split, key, normalize=True):
    """Equal-weight mean over ALL 6 datasets in split. Fail loudly if any missing
    (silently averaging <6 would misreport the macro)."""
    missing = [d for d in split
               if not (isinstance(per_ds.get(d), dict) and key in per_ds[d])]
    if missing:
        raise ValueError(f"macro: missing '{key}' for {missing}; need all 6 of {split}")
    vals = [per_ds[d][key] for d in split]
    if normalize:
        vals = [pct(v) for v in vals]
    return sum(vals) / len(vals)


# ---------------------------------------------------------------- Table 1
def table1():
    ff = L(f"{FF}/pareto_full58k_dyneps_summary.json")
    v3 = L(f"{V3}/pareto_full58k_dyneps_summary.json")
    top = L("reports/tokenf1_perdataset_fullft_1b_full58k.json")
    # topline json is a flat {dataset:{f1,em,n}}
    lab = "eps=0.05"
    ff_pd = ff["per_label_per_dataset"][lab]
    v3_pd = v3["per_label_per_dataset"][lab]
    ff_mk = ff["per_label"][lab]["mean_k_per_chunk"]
    v3_mk = v3["per_label"][lab]["mean_k_per_chunk"]
    ff_cr = ff["per_label"][lab]["compression_rate"]
    v3_cr = v3["per_label"][lab]["compression_rate"]
    print("=" * 100)
    print(f"# TABLE 1 (per-dataset, operating point {lab})  token-F1 / EM  [x100]")
    print(f"#   full-FT mean_k={ff_mk:.2f} comp_rate={ff_cr:.3f}   |   "
          f"v3-LoRA mean_k={v3_mk:.2f} comp_rate={v3_cr:.3f}")
    print("=" * 100)
    hdr = f"{'dataset':>22} | {'fullFT F1/EM':>14} | {'v3LoRA F1/EM':>14} | {'topline F1/EM':>14} | {'ΔF1(FF-v3)':>10}"
    for split, name in [(IID, "IID"), (OOD, "OOD")]:
        print(f"\n--- {name} ---")
        print(hdr)
        print("-" * len(hdr))
        for d in split:
            fF1 = pct(ff_pd[d]["f1"]); fEM = pct(ff_pd[d]["em"])   # compressor: fraction->pct
            vF1 = pct(v3_pd[d]["f1"]); vEM = pct(v3_pd[d]["em"])
            tF1 = top[d]["f1"];        tEM = top[d]["em"]          # topline: already percent
            print(f"{d:>22} | {fF1:6.2f}/{fEM:6.2f} | {vF1:6.2f}/{vEM:6.2f} | "
                  f"{tF1:6.2f}/{tEM:6.2f} | {fF1-vF1:+9.2f}")
        mfF1 = macro(ff_pd, split, "f1"); mfEM = macro(ff_pd, split, "em")
        mvF1 = macro(v3_pd, split, "f1"); mvEM = macro(v3_pd, split, "em")
        mtF1 = macro(top, split, "f1", normalize=False)  # already percent
        mtEM = macro(top, split, "em", normalize=False)
        print("-" * len(hdr))
        print(f"{'MACRO '+name:>22} | {mfF1:6.2f}/{mfEM:6.2f} | {mvF1:6.2f}/{mvEM:6.2f} | "
              f"{mtF1:6.2f}/{mtEM:6.2f} | {mfF1-mvF1:+9.2f}")


# ---------------------------------------------------------------- Table 4a
def interp_fixed(kv, curve, mk):
    """linear interpolation of fixed-k macro curve at mean_k=mk; clamp at ends."""
    if mk <= kv[0]:
        return curve[0]
    if mk >= kv[-1]:
        return curve[-1]
    for i in range(len(kv) - 1):
        a, b = kv[i], kv[i + 1]
        if a <= mk <= b:
            t = (mk - a) / (b - a)
            return curve[i] + t * (curve[i + 1] - curve[i])
    return curve[-1]


def table4a(tag, exp):
    ks = L(f"{exp}/ksweep_full58k_summary.json")["per_dataset_per_k"]
    pa = L(f"{exp}/pareto_full58k_dyneps_summary.json")
    kv = sorted(int(k) for k in ks)
    cI = [macro(ks[str(k)], IID, "f1") for k in kv]  # already percent (macro applies pct)
    cO = [macro(ks[str(k)], OOD, "f1") for k in kv]
    print("\n" + "=" * 100)
    print(f"# TABLE 4a ({tag}) fixed-vs-adaptive @ MATCHED mean_k  (Δ = adaptive - fixed-k interp)  token-F1")
    print("=" * 100)
    print(f"{'eps':>8} | {'mean_k':>6} | {'adaptive F1 ID/OOD':>19} | {'fixed@mk F1 ID/OOD':>19} | {'Δ ID/OOD':>13}")
    print("-" * 80)
    for lab in pa["per_label"]:
        mk = pa["per_label"][lab]["mean_k_per_chunk"]
        pd = pa["per_label_per_dataset"][lab]
        aI = macro(pd, IID, "f1"); aO = macro(pd, OOD, "f1")  # percent
        fI = interp_fixed(kv, cI, mk); fO = interp_fixed(kv, cO, mk)  # percent
        print(f"{lab:>8} | {mk:6.2f} | {aI:8.2f}/{aO:8.2f}    | {fI:8.2f}/{fO:8.2f}    | {aI-fI:+6.2f}/{aO-fO:+6.2f}")


if __name__ == "__main__":
    table1()
    table4a("full-FT", FF)
    table4a("v3-LoRA", V3)
