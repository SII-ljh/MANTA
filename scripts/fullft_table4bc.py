#!/usr/bin/env python3
"""Tables 4b / 4c for the full-FT (no-LoRA) 1B variant, full-58K (n=67854).

  Table 4b (architecture decomposition):  SAC -> +Matryoshka -> +eps-Gate
      +Matryoshka  = pact_v5_mrqa_fullft_matonly_1B  (use_k_predictor=False, fixed k)
      +eps-Gate    = pact_v5_mrqa_fullft_1B          (full MANTA, adaptive)
      SAC arm      = SAC-1B dedicated ckpts (LoRA-trained baseline, unchanged reference;
                     flagged in the output because it is NOT a full-FT model)
  Table 4c (allocation mechanism):        naive global allocation vs eps-Gate
      naive        = pact_v5_mrqa_fullft_naive_1B    (gate_mode=naive, global budget 7.37)

Metric convention (memory: eval-iid-ood-macro-average-convention):
  macro = equal-weight mean over the 6 datasets of a split (NOT micro / sample mean).
  Eval summaries store f1/em as FRACTIONS -> x100 here. SAC reports are already percent.

Compression-matched comparison follows the F68/F112 recipe: the arm with a free
knob is linearly interpolated onto the other arm's mean_k, so both sides sit at the
same average per-chunk allocation. Fails loudly rather than extrapolating.

Usage:  python scripts/fullft_table4bc.py
"""
import json
import os
import sys

ROOT = os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IID = ["SQuAD", "NewsQA", "TriviaQA-web", "SearchQA", "HotpotQA", "NaturalQuestionsShort"]
OOD = ["BioASQ", "DROP", "DuoRC.ParaphraseRC", "RACE", "RelationExtraction", "TextbookQA"]

MATONLY = f"{ROOT}/experiment/pact_v5_mrqa_fullft_matonly_1B/output/ksweep_full58k_abl_summary.json"
NAIVE = f"{ROOT}/experiment/pact_v5_mrqa_fullft_naive_1B/output/pareto_full58k_naive_summary.json"
MANTA_AD = f"{ROOT}/experiment/pact_v5_mrqa_fullft_1B/output/pareto_full58k_dyneps_summary.json"
MANTA_KS = f"{ROOT}/experiment/pact_v5_mrqa_fullft_1B/output/ksweep_full58k_summary.json"
SAC = {64: f"{ROOT}/reports/tokenf1_sac_1b_r64_full58k.json",
       128: f"{ROOT}/reports/tokenf1_sac_1b_r128_full58k.json"}


EXPECT_N = 67854


def load(path, what):
    if not os.path.exists(path):
        print(f"[MISSING] {what}: {path}")
        return None
    return json.load(open(path))


def assert_complete(summ, what, kind):
    """Every label must carry the full eval set. A summary built from a partial rank
    set is structurally valid and its macro looks plausible -- the count is the only
    tell, so refuse to table anything that fails it."""
    if summ is None:
        return None
    ns = ({k: v["count"] for k, v in summ["overall"].items()} if kind == "ksweep"
          else {k: v["n"] for k, v in summ["per_label"].items()})
    if not ns:
        sys.exit(f"[FATAL] {what}: no labels carrying a sample count")
    bad = {k: n for k, n in ns.items() if n != EXPECT_N}
    if bad:
        sys.exit(f"[FATAL] {what}: expected n={EXPECT_N} for every label, got {bad}")
    return summ


def only(d, what):
    """Single-entry accessor. `list(x.values())[0]` on a file that grew a second entry
    silently picks whichever came first in JSON insertion order."""
    if len(d) != 1:
        sys.exit(f"[FATAL] {what}: expected exactly 1 entry, found {len(d)}: {sorted(d)}")
    return next(iter(d.items()))


def macro(per_ds, split, key):
    """Equal-weight mean over ALL 6 datasets; loud on a missing one (averaging <6
    silently would misreport the macro)."""
    missing = [d for d in split if not (isinstance(per_ds.get(d), dict) and key in per_ds[d])]
    if missing:
        raise ValueError(f"macro: missing '{key}' for {missing}")
    vals = [per_ds[d][key] for d in split]
    return 100.0 * sum(vals) / len(vals)


def curve_from_ksweep(summ):
    """{k: {'mean_k': k, 'iid_f1':…}} from a ksweep summary (per_dataset_per_k)."""
    out = {}
    for kstr, per_ds in summ["per_dataset_per_k"].items():
        k = float(kstr)
        out[k] = dict(mean_k=k,
                      iid_f1=macro(per_ds, IID, "f1"), ood_f1=macro(per_ds, OOD, "f1"),
                      iid_em=macro(per_ds, IID, "em"), ood_em=macro(per_ds, OOD, "em"))
    return out


def curve_from_pareto(summ):
    """{label: {...}} from a pareto summary (per_label + per_label_per_dataset)."""
    out = {}
    for lab, v in summ["per_label"].items():
        per_ds = summ["per_label_per_dataset"][lab]
        out[lab] = dict(mean_k=v["mean_k_per_chunk"],
                        comp=v.get("compression_rate"),
                        k_std=v.get("within_sample_k_std"),
                        iid_f1=macro(per_ds, IID, "f1"), ood_f1=macro(per_ds, OOD, "f1"),
                        iid_em=macro(per_ds, IID, "em"), ood_em=macro(per_ds, OOD, "em"))
    return out


def interp(curve, target_mk, fields=("iid_f1", "ood_f1", "iid_em", "ood_em")):
    """Linear interpolation of `curve` (dict of points with mean_k) at target_mk.
    Returns None if target_mk is outside the curve's range (no extrapolation)."""
    pts = sorted(curve.values(), key=lambda p: p["mean_k"])
    mks = [p["mean_k"] for p in pts]
    if target_mk < mks[0] - 1e-9 or target_mk > mks[-1] + 1e-9:
        return None
    for a, b in zip(pts, pts[1:]):
        if a["mean_k"] - 1e-9 <= target_mk <= b["mean_k"] + 1e-9:
            span = b["mean_k"] - a["mean_k"]
            w = 0.0 if span < 1e-9 else (target_mk - a["mean_k"]) / span
            return {f: a[f] + w * (b[f] - a[f]) for f in fields}
    return None


def fmt(p):
    if p is None:
        return f"{'n/a':>29}"
    return f"{p['iid_f1']:6.2f}/{p['iid_em']:6.2f}   {p['ood_f1']:6.2f}/{p['ood_em']:6.2f}"


HDR = f"{'arm':<44} {'mean_k':>7}  {'IID F1/EM':>13}   {'OOD F1/EM':>13}"


def row(name, mk, p):
    mks = f"{mk:7.2f}" if mk is not None else f"{'—':>7}"
    print(f"{name:<44} {mks}  {fmt(p)}")


def main():
    mat_s = assert_complete(load(MATONLY, "matonly ksweep (table 4b)"),
                            "matonly ksweep", "ksweep")
    ad_s = assert_complete(load(MANTA_AD, "MANTA full-FT adaptive"),
                           "MANTA adaptive", "pareto")
    ks_s = assert_complete(load(MANTA_KS, "MANTA full-FT fixed-k"),
                           "MANTA fixed-k", "ksweep")
    nv_s = assert_complete(load(NAIVE, "naive-gate pareto (table 4c)"),
                           "naive-gate pareto", "pareto")
    if ad_s is None or ks_s is None:
        sys.exit("[FATAL] MANTA full-FT reference summaries missing — cannot build either table")

    manta_ad = curve_from_pareto(ad_s)
    manta_ks = curve_from_ksweep(ks_s)
    print(f"\nfull-58K n={EXPECT_N} (verified for every label of every summary used here)"
          f"   macro IID(6)/OOD(6) token-F1 / EM  [x100]\n")

    # ---------------- Table 4b ----------------
    print("=" * 96)
    print("# TABLE 4b — architecture decomposition (full-FT variant)")
    print("=" * 96)
    if mat_s is None:
        print("matonly ksweep not available yet — skipping 4b\n")
    else:
        mat = curve_from_ksweep(mat_s)
        print(HDR)
        print("-" * 96)
        # anchors: k=8 (nominal 64x) and k=4 (nominal 128x)
        for ratio, k in ((64, 8.0), (128, 4.0)):
            print(f"--- nominal {ratio}x (fixed k={int(k)}) ---")
            sac = load(SAC[ratio], f"SAC-1B r{ratio}")
            if sac:
                _, v = only(sac, f"SAC-1B r{ratio} report")
                row(f"SAC-1B r{ratio}  [LoRA-trained baseline, not full-FT]", k,
                    dict(iid_f1=v["tok_iid_f1"], ood_f1=v["tok_ood_f1"],
                         iid_em=v["tok_iid_em"], ood_em=v["tok_ood_em"]))
            if k in mat:
                row("+Matryoshka (full-FT matonly, fixed k)", k, mat[k])
            if k in manta_ks:
                row("  [ref] full MANTA full-FT at same fixed k", k, manta_ks[k])
            # +eps-Gate: the adaptive point closest to this anchor, plus the strict
            # compression-matched delta (matonly interpolated onto its mean_k).
            lab = min(manta_ad, key=lambda l: abs(manta_ad[l]["mean_k"] - k))
            g = manta_ad[lab]
            row(f"+eps-Gate (full MANTA adaptive, {lab})", g["mean_k"], g)
            mi = interp(mat, g["mean_k"])
            if mi:
                row("  matonly interpolated @ that mean_k", g["mean_k"], mi)
                print(f"{'  => eps-Gate − Matryoshka (compression-matched)':<44} {'':>7}  "
                      f"{g['iid_f1']-mi['iid_f1']:+6.2f}/{g['iid_em']-mi['iid_em']:+6.2f}   "
                      f"{g['ood_f1']-mi['ood_f1']:+6.2f}/{g['ood_em']-mi['ood_em']:+6.2f}")
            else:
                print("  [warn] matonly curve does not cover that mean_k — no matched delta")
            print()

    # ---------------- Table 4c ----------------
    print("=" * 96)
    print("# TABLE 4c — allocation mechanism: naive global vs eps-Gate (full-FT variant)")
    print("=" * 96)
    if nv_s is None:
        print("naive-gate pareto not available yet — skipping 4c\n")
        return
    nv = curve_from_pareto(nv_s)
    lab_nv, v = only(nv, "naive-gate pareto summary")
    print(HDR)
    print("-" * 96)
    row(f"naive global allocation ({lab_nv})", v["mean_k"], v)
    gi = interp(manta_ad, v["mean_k"])
    if gi:
        row("eps-Gate interpolated @ same mean_k", v["mean_k"], gi)
        print(f"{'  => eps-Gate − naive (compression-matched)':<44} {'':>7}  "
              f"{gi['iid_f1']-v['iid_f1']:+6.2f}/{gi['iid_em']-v['iid_em']:+6.2f}   "
              f"{gi['ood_f1']-v['ood_f1']:+6.2f}/{gi['ood_em']-v['ood_em']:+6.2f}")
    else:
        print(f"  [warn] MANTA adaptive curve (mean_k "
              f"{min(p['mean_k'] for p in manta_ad.values()):.2f}–"
              f"{max(p['mean_k'] for p in manta_ad.values()):.2f}) does not cover naive mean_k "
              f"{v['mean_k']:.2f}; compare against the fixed-k curve instead")
        mi = interp(manta_ks, v["mean_k"])
        if mi:
            row("MANTA fixed-k interpolated @ same mean_k", v["mean_k"], mi)
    mi2 = interp(curve_from_ksweep(mat_s), v["mean_k"]) if mat_s else None
    if mi2:
        row("matonly interpolated @ same mean_k", v["mean_k"], mi2)
    # mechanism evidence: within-sample allocation spread at matched mean_k
    print(f"\nwithin-sample k std (allocation elasticity):  naive={v['k_std']}")
    for lab, g in sorted(manta_ad.items(), key=lambda kv: kv[1]["mean_k"]):
        print(f"    eps-Gate {lab:<12} mean_k={g['mean_k']:6.2f}  k_std={g['k_std']}")

    # ---- paper-shaped three-arm block (tab:threearm): 64x tier, one recipe,
    # only the allocation policy differs. Raw rows, no interpolation -- the paper's
    # convention for this table is recipe-matched, and the eps-Gate row happens to
    # sit at HARDER compression than the no-gate anchor, so the comparison is
    # conservative in our favour rather than needing a matched-ratio caveat.
    print("\n" + "=" * 96)
    print("# TABLE 4c (paper shape, tab:threearm) — 64x tier, identical recipe, allocation policy only")
    print("=" * 96)
    if mat_s is None:
        print("matonly ksweep missing — cannot print the no-gate arm")
        return
    mat = curve_from_ksweep(mat_s)
    print(HDR + f"  {'k_std':>6}")
    print("-" * 104)
    rows = []
    if 8.0 in mat:
        rows.append(("no gate (fixed k=8)", 8.0, mat[8.0], None))
    rows.append((f"naive gate (global allocation)", v["mean_k"], v, v["k_std"]))
    lab05 = min(manta_ad, key=lambda l: abs(manta_ad[l]["mean_k"] - v["mean_k"]))
    g = manta_ad[lab05]
    rows.append((f"eps-Gate ({lab05})", g["mean_k"], g, g["k_std"]))
    for name, mk, p, ks in rows:
        kss = f"{ks:6.2f}" if isinstance(ks, (int, float)) else f"{'—':>6}"
        print(f"{name:<44} {mk:7.2f}  {fmt(p)}  {kss}")
    if 8.0 in mat:
        d1 = v["iid_f1"] - mat[8.0]["iid_f1"]
        d2 = g["iid_f1"] - v["iid_f1"]
        print(f"\n  naive − no-gate  = {d1:+.2f} ID token-F1   "
              f"(paper's LoRA sibling: +1.1)")
        print(f"  eps-Gate − naive = {d2:+.2f} ID token-F1   "
              f"(paper's LoRA sibling: +1.63)")


if __name__ == "__main__":
    main()
