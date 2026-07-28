#!/usr/bin/env python
"""Render the two oracle figures (F92) in ICLR camera-ready style.

Fig A (motivation, §3.1/§3.2): 3 panels —
  (a) oracle joint-budget allocation Pareto-dominates fixed-k (loss vs avg budget)
  (b) per-chunk min-sufficient-k distribution (heterogeneous sufficiency)
  (c) per-dataset oracle mean budget (task-aligned heterogeneity, IID vs OOD)
Fig B (oracle skyline, §5.4.1 / Fig 4): token-F1 vs compression rate —
  PACT-Adaptive (learned) vs selfref oracle upper bound, + Full-FT 1x topline.

CPU only; reads the saved JSONs, writes PDF (vector) + PNG (300dpi) to figures/.
"""
import os
import json
import glob
import re
import string
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

ROOT = os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ORACLE_EVAL = f"{ROOT}/analysis/oracle_v4_eval/summary.json"
PARETO = f"{ROOT}/experiment/v4eval_step19999/output/pareto_oracle_v4_20k_summary.json"
SUB_META = f"{ROOT}/data/mrqa_eval_subset/qa_eval_meta.json"
GEN_GLOB = f"{ROOT}/experiment/fullft_1b_mrqa/eval_full58k/fullft_full58k_rank*.json"
OUT = f"{ROOT}/figures"

# ---- Okabe-Ito colorblind-safe palette (matches the paper's other figures) ----
OI = dict(orange="#E69F00", sky="#56B4E9", green="#009E73", yellow="#F0E442",
          blue="#0072B2", verm="#D55E00", purple="#CC79A7", grey="#7F7F7F",
          black="#222222")

plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 300,
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
    "font.size": 9,
    "axes.titlesize": 9.5,
    "axes.labelsize": 9,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "legend.frameon": False,
    "lines.linewidth": 1.8,
    "lines.markersize": 5.5,
    "grid.linewidth": 0.5,
    "grid.alpha": 0.30,
    "axes.grid": True,
    "axes.axisbelow": True,
    "pdf.fonttype": 42,   # editable text in PDF
    "ps.fonttype": 42,
})

IID = {"SQuAD", "NewsQA", "TriviaQA", "TriviaQA-web", "SearchQA", "HotpotQA",
       "NQ", "NaturalQuestionsShort"}


def norm(s):
    s = s.lower()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return " ".join(s.split())


def tok_f1(p, g):
    pt, gt = norm(p).split(), norm(g).split()
    if not pt or not gt:
        return float(pt == gt)
    c = Counter(pt) & Counter(gt)
    ns = sum(c.values())
    if ns == 0:
        return 0.0
    pr, rc = ns / len(pt), ns / len(gt)
    return 2 * pr * rc / (pr + rc)


def fullft_subset_f1():
    sub = json.load(open(SUB_META))
    gens = {}
    for fp in glob.glob(GEN_GLOB):
        for _k, rows in json.load(open(fp)).items():
            for r in rows:
                gens[r["id"]] = r["generated"]
    fs = []
    for m in sub:
        g = gens.get(m["id"], "")
        fs.append(max(tok_f1(g, a) for a in m["answers"]))
    return 100.0 * sum(fs) / len(fs)


def fig_motivation(oe):
    f1c = oe["figure1_curves"]
    hist = oe["figure2b_min_k_hist"]
    t6 = oe["table6_per_dataset"]

    fig, ax = plt.subplots(1, 3, figsize=(9.2, 2.85))

    # (a) oracle vs fixed-k loss Pareto
    a = ax[0]
    orc = f1c["oracle"]
    ox = [orc[k]["achieved_mean_k"] for k in ["1", "2", "4", "8", "16", "32"]]
    oy = [orc[k]["mean_chunk_loss"] for k in ["1", "2", "4", "8", "16", "32"]]
    fx = [1, 2, 4, 8, 16, 32]
    fy = [f1c["fixed_per_k"][str(k)] for k in fx]
    a.plot(fx, fy, "--s", color=OI["verm"], label="fixed-$k$ (uniform)",
           markerfacecolor="white", markeredgewidth=1.3)
    a.plot(ox, oy, "-o", color=OI["blue"], label="oracle (per-chunk)")
    a.fill_between(ox, oy, [max(fy)] * len(oy), color=OI["blue"], alpha=0.06)
    a.set_xscale("log", base=2)
    a.set_xticks([1, 2, 4, 8, 16, 32])
    a.set_xticklabels([1, 2, 4, 8, 16, 32])
    a.set_xlabel("average budget  $\\bar{k}$ (slots/chunk)")
    a.set_ylabel("decoder LM loss  (↓)")
    a.set_title("(a) oracle Pareto-dominates fixed-$k$", fontweight="bold")
    a.legend(loc="upper left")
    a.annotate("up to $-0.13$ nats\nat matched budget", xy=(8, 5.356),
               xytext=(1.4, 5.345), fontsize=7.3, color=OI["blue"],
               arrowprops=dict(arrowstyle="->", color=OI["blue"], lw=0.9))

    # (b) per-chunk min-k distribution
    b = ax[1]
    ks = ["1", "2", "4", "8", "16", "32"]
    fr = [100 * hist["min_k_frac"][k] for k in ks]
    bars = b.bar(range(len(ks)), fr, color=OI["sky"], edgecolor=OI["black"],
                 linewidth=0.6, width=0.72)
    bars[0].set_color(OI["green"])
    bars[-1].set_color(OI["orange"])
    b.set_xticks(range(len(ks)))
    b.set_xticklabels(ks)
    b.set_xlabel("min sufficient $k$  (per chunk, $\\varepsilon{=}0.05$)")
    b.set_ylabel("% of chunks")
    b.set_title("(b) heterogeneous sufficiency", fontweight="bold")
    b.text(0, fr[0] + 1.5, "67.6%\nneed $\\leq$1", ha="center", fontsize=7.3,
           color=OI["green"])
    b.text(5, fr[-1] + 1.5, "8.1%\nneed full 32", ha="center", fontsize=7.3,
           color=OI["orange"])
    b.set_ylim(0, max(fr) + 11)

    # (c) per-dataset oracle mean budget
    c = ax[2]
    items = sorted(t6.items(), key=lambda kv: kv[1]["mean_k"])
    names = [k for k, _ in items]
    mk = [v["mean_k"] for _, v in items]
    cols = [OI["blue"] if n in IID else OI["orange"] for n in names]
    c.barh(range(len(names)), mk, color=cols, edgecolor=OI["black"],
           linewidth=0.5, height=0.72)
    c.set_yticks(range(len(names)))
    c.set_yticklabels(names, fontsize=7)
    c.set_xlabel("oracle mean budget  $\\bar{k}$  (@$\\bar{k}{=}8$)")
    c.set_title("(c) task-aligned heterogeneity", fontweight="bold")
    mean_mk = sum(mk) / len(mk)
    c.axvline(mean_mk, color=OI["grey"], ls=":", lw=1.0)
    c.text(mean_mk + 0.05, len(names) - 0.4, "mean", fontsize=6.8,
           color=OI["grey"], ha="left", va="center")
    from matplotlib.patches import Patch
    c.legend(handles=[Patch(color=OI["blue"], label="IID"),
                      Patch(color=OI["orange"], label="OOD")],
             loc="lower right", title="1.8$\\times$ spread", title_fontsize=7.3)

    fig.tight_layout(w_pad=1.4)
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_oracle_motivation.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("[saved] fig_oracle_motivation.{pdf,png}")


def fig_skyline(pl, topline):
    eps = ["0.01", "0.05", "0.1", "0.2", "0.3"]
    def comp(label):  # compression rate = 512 / mean_k
        return [512.0 / pl[f"{label}{e}"]["mean_k_per_chunk"] for e in eps]
    def f1(label):
        return [100.0 * pl[f"{label}{e}"]["f1"] for e in eps]

    ac_x, ac_y = comp("eps="), f1("eps=")
    or_x, or_y = comp("oracle_eps="), f1("oracle_eps=")

    fig, ax = plt.subplots(figsize=(4.35, 3.5))
    # headroom band (between the two curves over the shared x-range)
    import numpy as np
    xs = np.geomspace(min(ac_x + or_x), max(ac_x + or_x), 100)
    ac_i = np.interp(xs, ac_x[::-1], ac_y[::-1])
    or_i = np.interp(xs, or_x[::-1], or_y[::-1])
    ax.fill_between(xs, ac_i, or_i, color=OI["green"], alpha=0.10, lw=0)

    ax.axhline(topline, color=OI["grey"], ls=(0, (5, 2)), lw=1.3)
    ax.text(ac_x[-1] * 0.97, topline + 0.5, "Full-FT topline (1$\\times$, no compression)",
            ha="right", va="bottom", fontsize=7.3, color=OI["grey"])

    ax.plot(or_x, or_y, "-D", color=OI["green"], label="oracle upper bound (selfref)",
            markerfacecolor="white", markeredgewidth=1.4)
    ax.plot(ac_x, ac_y, "-o", color=OI["blue"], label="PACT-Adaptive (learned, ours)")

    # matched-eps headroom arrows at two points
    for i in (1, 3):
        ax.annotate("", xy=(ac_x[i], or_y[i]), xytext=(ac_x[i], ac_y[i]),
                    arrowprops=dict(arrowstyle="<->", color=OI["green"], lw=0.9))
    ax.text(ac_x[1] * 1.05, (ac_y[1] + or_y[1]) / 2,
            "$+5$–$6$ F1\nheadroom", fontsize=7.3, color=OI["green"], va="center")

    ax.set_xscale("log")
    ax.set_xlabel("compression rate  (512 / $\\bar{k}$,  $\\rightarrow$ more compression)")
    ax.set_ylabel("token-F1")
    ax.set_title("Oracle skyline: PACT-Adaptive vs. upper bound",
                 fontweight="bold", fontsize=9)
    ax.legend(loc="lower left")
    ax.set_xticks([50, 70, 100, 150])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    ax.text(0.985, 0.86, "MRQA 2000-subset\n(v4@20k, token-F1)",
            transform=ax.transAxes, ha="right", va="top", fontsize=6.6,
            color=OI["grey"])

    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{OUT}/fig_oracle_skyline.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("[saved] fig_oracle_skyline.{pdf,png}")


if __name__ == "__main__":
    import matplotlib.ticker  # noqa
    oe = json.load(open(ORACLE_EVAL))
    pl = json.load(open(PARETO))["per_label"]
    topline = fullft_subset_f1()
    print(f"[info] Full-FT 2000-subset token-F1 = {topline:.2f}")
    fig_motivation(oe)
    fig_skyline(pl, topline)
    print("[done]")
