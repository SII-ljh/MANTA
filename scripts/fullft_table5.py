#!/usr/bin/env python3
"""Table 5 (ε controllability + extrapolation + mid-k smoothness) for the full-FT
(no-LoRA) 1B variant, full-58K macro token-F1. Reuses codex-reviewed helpers from
fullft_tables_A.py (macro / interp_fixed / L / IID / OOD / FF / V3).

Three blocks:
  5-CTRL : trained-grid ∪ unseen-ε adaptive points, sorted by ε → shows mean_k
           decreases monotonically with ε (controllability) and unseen ε points
           fall smoothly on the curve (extrapolation).
  5a-EXTRAP : each UNSEEN-ε point vs v3-LoRA fixed-k curve interpolated at the SAME
              mean_k (matched-mean_k, == F112/F113 recipe) + recovery vs dense
              full-FT topline (macro IID 69.55 / OOD 44.50, F91).
  5b-MIDK : fixed-k {3,6,12,24} macro F1/EM vs v3 fixed-k interpolated at same k —
            shows a single ckpt fills the intermediate anchors smoothly.
"""
from fullft_tables_A import macro, interp_fixed, L, IID, OOD, FF, V3

TOP_I, TOP_O = 69.55, 44.50            # dense full-FT topline macro token-F1 (F91)
TRAINED_EPS = {0.001, 0.01, 0.05, 0.1, 0.2}   # Stage-2 training ε grid


def fixed_curve(exp):
    ks = L(f"{exp}/ksweep_full58k_summary.json")["per_dataset_per_k"]
    kv = sorted(int(k) for k in ks)
    return kv, [macro(ks[str(k)], IID, "f1") for k in kv], [macro(ks[str(k)], OOD, "f1") for k in kv]


def adaptive_points(exp, fname):
    """Return list of (eps_float, label, mean_k, comp, f1I, f1O, emI, emO) from a pareto summary."""
    d = L(f"{exp}/{fname}")
    out = []
    for lab in d["per_label"]:
        pl = d["per_label"][lab]; pd = d["per_label_per_dataset"][lab]
        eps = float(lab.split("=")[1])
        out.append((eps, lab, pl["mean_k_per_chunk"], pl["compression_rate"],
                    macro(pd, IID, "f1"), macro(pd, OOD, "f1"),
                    macro(pd, IID, "em"), macro(pd, OOD, "em")))
    return out


v3kv, v3fI, v3fO = fixed_curve(V3)

# ---- 5-CTRL: full controllability curve (trained grid + unseen extrap), sorted by eps ----
pts = adaptive_points(FF, "pareto_full58k_dyneps_summary.json") + \
      adaptive_points(FF, "pareto_full58k_extrap_summary.json")
pts.sort(key=lambda r: r[0])
print("=" * 120)
print("# TABLE 5-CTRL  full-FT ε controllability (trained-grid ∪ UNSEEN-ε), full-58K macro token-F1  [x100]")
print("#   monotone check: mean_k should DECREASE as ε increases")
print("=" * 120)
h = f"{'eps':>7} {'seen?':>6} | {'mean_k':>7} {'comp':>6} | {'F1 IID/OOD':>14} | {'EM IID/OOD':>14}"
print(h); print("-" * len(h))
prev_mk = None; mono = True
for eps, lab, mk, comp, f1I, f1O, emI, emO in pts:
    seen = "grid" if any(abs(eps - t) < 1e-9 for t in TRAINED_EPS) else "UNSEEN"
    if prev_mk is not None and mk > prev_mk + 1e-6:
        mono = False
    prev_mk = mk
    print(f"{eps:>7.3f} {seen:>6} | {mk:7.2f} {comp:6.3f} | {f1I:6.2f}/{f1O:6.2f} | {emI:6.2f}/{emO:6.2f}")
print(f"\n  monotone (mean_k non-increasing as ε↑): {mono}")

# ---- 5a-EXTRAP: unseen-ε vs v3 fixed-k @ matched mean_k + topline recovery ----
print("\n" + "=" * 120)
print("# TABLE 5a-EXTRAP  UNSEEN-ε full-FT adaptive vs v3-LoRA fixed-k @ matched mean_k + topline recovery")
print("=" * 120)
h = (f"{'eps':>7} | {'mean_k':>7} {'comp':>6} | {'fullFT F1 I/O':>14} {'EM I/O':>13} | "
     f"{'v3fix@mk I/O':>13} {'Δ I/O':>13} | {'recov% I/O':>11}")
print(h); print("-" * len(h))
for eps, lab, mk, comp, f1I, f1O, emI, emO in sorted(adaptive_points(FF, "pareto_full58k_extrap_summary.json")):
    vI = interp_fixed(v3kv, v3fI, mk); vO = interp_fixed(v3kv, v3fO, mk)
    print(f"{eps:>7.3f} | {mk:7.2f} {comp:6.3f} | {f1I:6.2f}/{f1O:6.2f} {emI:6.2f}/{emO:6.2f} | "
          f"{vI:6.2f}/{vO:6.2f} {f1I-vI:+6.2f}/{f1O-vO:+5.2f} | {100*f1I/TOP_I:5.1f}/{100*f1O/TOP_O:5.1f}")

# ---- 5b-MIDK: mid-k fixed points vs v3 fixed-k interp at same k ----
midk = L(f"{FF}/ksweep_full58k_midk_summary.json")["per_dataset_per_k"]
print("\n" + "=" * 120)
print("# TABLE 5b-MIDK  full-FT fixed mid-k {3,6,12,24} vs v3-LoRA fixed-k interp @ same k, macro token-F1  [x100]")
print("=" * 120)
h = f"{'k':>4} | {'fullFT F1 I/O':>14} {'EM I/O':>13} | {'v3fix@k I/O':>13} {'ΔF1 I/O':>13}"
print(h); print("-" * len(h))
for k in sorted(int(x) for x in midk):
    d = midk[str(k)]
    fI = macro(d, IID, "f1"); fO = macro(d, OOD, "f1")
    eI = macro(d, IID, "em"); eO = macro(d, OOD, "em")
    vI = interp_fixed(v3kv, v3fI, k); vO = interp_fixed(v3kv, v3fO, k)
    print(f"{k:>4} | {fI:6.2f}/{fO:6.2f} {eI:6.2f}/{eO:6.2f} | {vI:6.2f}/{vO:6.2f} {fI-vI:+6.2f}/{fO-vO:+5.2f}")
