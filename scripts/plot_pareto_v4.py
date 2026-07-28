#!/usr/bin/env python
"""PACT Pareto figures (full-58K token-F1), ICLR camera-ready style.

Produces two figures under analysis/fig_pareto_v4/:
  (A) fig_pareto_singleckpt  — single-ckpt Pareto: PACT-M fixed-k sweep + PACT-Adaptive
      (one ckpt -> whole curve) vs SAC / 500xC / ICAE dedicated-ckpt points (one ckpt/rate).
      Answers reviewer "compare to SAC, not only to yourself".
  (B) fig_pareto_strictharder — conservative budget-matched: PACT operates at STRICTLY
      higher nominal compression than the baseline bracket it beats, yet wins both metrics
      (the single tie pt4 vs SAC-OOD is annotated).

All numbers are full-58K (n=67854) token-F1, IID/OOD, from PACT/reports/tokenf1_*_full58k*.json
(extracted 2026-06-25). Two panels (IID, OOD), x = nominal compression rate (log).
CPU only. Writes PNG (300dpi) + PDF (vector).
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

ROOT = os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = f"{ROOT}/analysis/fig_pareto_v4"
os.makedirs(OUT, exist_ok=True)

OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#7F7F7F", black="#222222")

plt.rcParams.update({
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans", "Arial"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9, "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5, "legend.frameon": False,
    "lines.linewidth": 1.8, "lines.markersize": 6, "grid.linewidth": 0.5, "grid.alpha": 0.30,
    "axes.grid": True, "axes.axisbelow": True, "pdf.fonttype": 42, "ps.fonttype": 42,
})

# ---- DATA (full-58K token-F1; source report files in parens) -----------------
# PACT-M fixed-k sweep, v4@20k (tokenf1_v4eval_step19999_full58k_ksweep.json). x=512/k.
PACT_KSWEEP = [  # (k, comp, iid, ood)
    (1, 512.0, 44.96, 32.80), (2, 256.0, 47.57, 35.49), (4, 128.0, 50.79, 38.27),
    (8, 64.0, 55.04, 40.31), (16, 32.0, 58.34, 43.59), (32, 16.0, 60.04, 45.22)]
# PACT-Adaptive, v4@20k (tokenf1_v4eval_step19999_full58k_adaptive.json). x=512/mean_k (mean_k from Table 5).
PACT_ADAPT = [  # (eps, mean_k, comp, iid, ood)
    (0.01, 9.43, 54.3, 57.83, 43.19), (0.05, 7.84, 65.3, 56.85, 42.08),
    (0.10, 6.14, 83.4, 55.21, 40.44), (0.20, 4.09, 125.2, 52.40, 38.45),
    (0.30, 3.10, 165.2, 50.28, 36.53)]
# SAC-1B dedicated (tokenf1_sac_1b_r{64,85,128,170}_full58k.json). comp=512/k.
SAC = [  # (k, comp, iid, ood)
    (8, 64.0, 52.01, 40.10), (6, 85.3, 50.65, 37.68), (4, 128.0, 48.50, 36.82), (3, 170.7, 48.26, 36.39)]
# 500xCompressor dedicated (tokenf1_c500x_1b_r*_full58k.json). comp=nominal r.
C500 = [(64.0, 51.04, 37.02), (85.0, 50.37, 36.93), (128.0, 49.46, 36.08), (170.0, 47.97, 34.73)]
# ICAE dedicated (tokenf1_icae_1b_r*_full58k.json).
ICAE = [(64.0, 41.18, 31.00), (85.0, 43.98, 33.11), (128.0, 42.70, 32.66), (170.0, 41.76, 32.25)]
# Full-FT topline, no compression (tokenf1_fullft_1b_full58k.json).
TOPLINE = dict(iid=69.56, ood=44.50)
# Strictly-harder PACT operating points (tokenf1_v4eval_newcomp + grid eps=0.05). PACT compresses HARDER.
PACT_HARD = [  # (eps, comp_nominal, iid, ood)
    (0.05, 65.0, 56.85, 42.08), (0.12, 91.0, 54.65, 39.94),
    (0.22, 133.6, 51.92, 37.73), (0.34, 180.4, 49.75, 36.05)]


def style_ax(ax, ylabel, title):
    ax.set_xscale("log")
    ax.set_xlabel("compression rate  (× , log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, loc="left")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:g}×"))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())
    ax.set_xticks([16, 32, 64, 128, 256, 512])


# ===================== Figure A: single-ckpt Pareto =====================
def fig_singleckpt():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    for ax, key, ttl in [(axes[0], "iid", "(a) MRQA in-domain"),
                         (axes[1], "ood", "(b) MRQA out-of-domain")]:
        ix = 2 if key == "iid" else 3
        # topline
        ax.axhline(TOPLINE[key], color=OI["grey"], ls=":", lw=1.2)
        ax.text(500, TOPLINE[key] + 0.3, f"Full-FT topline (1×) {TOPLINE[key]:.1f}",
                color=OI["grey"], fontsize=7, ha="right", va="bottom")
        # PACT-M fixed-k (one ckpt -> whole curve)
        xs = [r[1] for r in PACT_KSWEEP]; ys = [r[ix] for r in PACT_KSWEEP]
        ax.plot(xs, ys, "-o", color=OI["blue"], label="PACT-M fixed-$k$ (1 ckpt)", zorder=5)
        # PACT-Adaptive (same ckpt, continuous knob)
        xa = [r[2] for r in PACT_ADAPT]; ya = [r[ix + 1] for r in PACT_ADAPT]
        ax.plot(xa, ya, "--s", color=OI["sky"], label="PACT-Adaptive ($\\epsilon$, same ckpt)", zorder=4)
        # SAC dedicated points (one ckpt per rate -> discrete, no connecting curve)
        xse = [r[1] for r in SAC]; yse = [r[ix] for r in SAC]
        ax.plot(xse, yse, "D", color=OI["verm"], label="SAC-1B (dedicated ckpt/rate)",
                ls="none", ms=6.5, zorder=6)
        # 500xC + ICAE dedicated points
        ax.plot([r[0] for r in C500], [r[ix - 1] for r in C500], "^", color=OI["green"],
                label="500xC (dedicated)", ls="none", ms=6, zorder=3)
        ax.plot([r[0] for r in ICAE], [r[ix - 1] for r in ICAE], "x", color=OI["purple"],
                label="ICAE (dedicated)", ls="none", ms=6, mew=1.6, zorder=3)
        style_ax(ax, "token-F1", ttl)
    axes[0].legend(loc="lower left", ncol=1)
    axes[0].text(0.5, 0.04,
                 "PACT: one checkpoint $\\to$ entire curve.\nBaselines: one checkpoint per point.",
                 transform=axes[0].transAxes, fontsize=7, color=OI["black"], va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_pareto_singleckpt.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_pareto_singleckpt")


# ===================== Figure B: strictly-harder budget-matched =====================
def fig_strictharder():
    fig, axes = plt.subplots(1, 2, figsize=(9.4, 3.7))
    series = [("PACT (ours)", PACT_HARD, OI["blue"], "-o", 1),
              ("SAC-1B", [(0, r[1], r[2], r[3]) for r in SAC], OI["verm"], "--D", 1),
              ("500xC", [(0, r[0], r[1], r[2]) for r in C500], OI["green"], "--^", 0),
              ("ICAE", [(0, r[0], r[1], r[2]) for r in ICAE], OI["purple"], "--x", 0)]
    for ax, key, ttl, ix in [(axes[0], "iid", "(a) MRQA in-domain", 2),
                             (axes[1], "ood", "(b) MRQA out-of-domain", 3)]:
        for name, data, col, sty, _ in series:
            xs = [d[1] for d in data]; ys = [d[ix] for d in data]
            ax.plot(xs, ys, sty, color=col, label=name, mew=1.6 if "x" in sty else 1)
        style_ax(ax, "token-F1", ttl)
    # annotate the single tie: pt4 (PACT 180.4×) vs SAC k3 (170.7×) on OOD
    axes[1].annotate("pt4: PACT 180× vs SAC 171×\nOOD $\\approx$ tie (−0.34)",
                     xy=(180.4, 36.05), xytext=(60, 33.0), fontsize=7, color=OI["black"],
                     arrowprops=dict(arrowstyle="->", color=OI["grey"], lw=0.8))
    axes[0].legend(loc="lower left")
    axes[0].text(0.5, 0.05,
                 "Each PACT point sits at STRICTLY higher\ncompression than the baseline it beats.",
                 transform=axes[0].transAxes, fontsize=7, color=OI["black"], va="bottom")
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{OUT}/fig_pareto_strictharder.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("wrote fig_pareto_strictharder")


if __name__ == "__main__":
    fig_singleckpt()
    fig_strictharder()
    print("OUT:", OUT)
