#!/usr/bin/env python3
"""Table 2 (strictly-harder multi-ratio) for the full-FT (no-LoRA) variant, full-58K macro token-F1.
Reuses codex-reviewed helpers from fullft_tables_A.py (macro / interp_fixed / L / IID / OOD / FF / V3).
Compares full-FT adaptive at 4 increasingly-hard eps vs v3-LoRA at MATCHED mean_k, two ways:
  (a) v3 fixed-k curve interp (brackets all mean_k, no extrapolation; == F112 method), and
  (b) v3 adaptive curve interp (both-adaptive; flagged OOR when mean_k below v3's adaptive range).
Also reports recovery vs dense full-FT topline (macro IID 69.55 / OOD 44.50, F91)."""
import sys
from fullft_tables_A import macro, interp_fixed, L, IID, OOD, FF, V3

TOP_I, TOP_O = 69.55, 44.50  # dense full-FT topline macro token-F1 (F91 / F110)


def fixed_curve(exp):
    ks = L(f"{exp}/ksweep_full58k_summary.json")["per_dataset_per_k"]
    kv = sorted(int(k) for k in ks)
    return kv, [macro(ks[str(k)], IID, "f1") for k in kv], [macro(ks[str(k)], OOD, "f1") for k in kv]


def adaptive_curve(exp):
    """v3 adaptive (mean_k, macroF1) curve, sorted ascending by mean_k."""
    pa = L(f"{exp}/pareto_full58k_dyneps_summary.json")
    rows = []
    for lab in pa["per_label"]:
        mk = pa["per_label"][lab]["mean_k_per_chunk"]
        pd = pa["per_label_per_dataset"][lab]
        rows.append((mk, macro(pd, IID, "f1"), macro(pd, OOD, "f1")))
    rows.sort()
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def point(summary, lab):
    pl = summary["per_label"][lab]; pd = summary["per_label_per_dataset"][lab]
    return dict(mean_k=pl["mean_k_per_chunk"], comp=pl["compression_rate"],
                f1I=macro(pd, IID, "f1"), f1O=macro(pd, OOD, "f1"),
                emI=macro(pd, IID, "em"), emO=macro(pd, OOD, "em"))


ff_par = L(f"{FF}/pareto_full58k_dyneps_summary.json")     # has eps=0.05
ff_hard = L(f"{FF}/pareto_full58k_harder_summary.json")    # eps=0.12,0.22,0.34
v3kv, v3fI, v3fO = fixed_curve(V3)
v3ak, v3aI, v3aO = adaptive_curve(V3)
v3a_lo = v3ak[0]  # lowest mean_k v3 adaptive reaches (below this = OOR extrapolation)

POINTS = [("0.05", ff_par), ("0.12", ff_hard), ("0.22", ff_hard), ("0.34", ff_hard)]
print("=" * 128)
print("# TABLE 2 strictly-harder — full-FT adaptive vs v3-LoRA @ matched mean_k, full-58K macro token-F1")
print(f"#   v3 adaptive curve mean_k range = [{v3a_lo:.2f}, {v3ak[-1]:.2f}]; below {v3a_lo:.2f} -> v3-adaptive extrapolated (OOR flag)")
print("=" * 128)
hdr = (f"{'eps':>5} | {'ff mk':>6} {'comp':>5} | {'fullFT F1 I/O':>14} | "
       f"{'v3fix@mk I/O':>13} {'Δfix':>13} | {'v3adp@mk I/O':>13} {'Δadp':>13} | {'recov% I/O':>11}")
print(hdr); print("-" * len(hdr))
for eps, summ in POINTS:
    p = point(summ, f"eps={eps}")
    mk = p["mean_k"]
    vfI = interp_fixed(v3kv, v3fI, mk); vfO = interp_fixed(v3kv, v3fO, mk)
    vaI = interp_fixed(v3ak, v3aI, mk); vaO = interp_fixed(v3ak, v3aO, mk)
    oor = " *OOR" if mk < v3a_lo else ""
    print(f"{eps:>5} | {mk:6.2f} {p['comp']:5.3f} | {p['f1I']:6.2f}/{p['f1O']:6.2f} | "
          f"{vfI:6.2f}/{vfO:6.2f} {p['f1I']-vfI:+6.2f}/{p['f1O']-vfO:+5.2f} | "
          f"{vaI:6.2f}/{vaO:6.2f}{oor} {p['f1I']-vaI:+6.2f}/{p['f1O']-vaO:+5.2f} | "
          f"{100*p['f1I']/TOP_I:5.1f}/{100*p['f1O']/TOP_O:5.1f}")
print("\n(EM shown separately)")
print(f"{'eps':>5} | {'fullFT EM I/O':>14}")
for eps, summ in POINTS:
    p = point(summ, f"eps={eps}")
    print(f"{eps:>5} | {p['emI']:6.2f}/{p['emO']:6.2f}")
