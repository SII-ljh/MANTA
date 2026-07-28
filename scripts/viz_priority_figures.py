#!/usr/bin/env python
"""PACT paper figures exp19 / exp20 / exp21 from a priority+attention dump.

Reusable: --dump <json from dump_priority_attention.py>, --outroot <analysis>.
Produces:
  exp19_gate_viz/      per-sample KPredictor priority heatmap (chunk x anchor
                       keep-prob), selected-k annotated; + multi-sample panel.
  exp20_attention/     decoder attention to memory per chunk, compressed vs full.
  exp21_case_study/    one rich annotated case (k_c, keep-prob, retain, attn).

Colorblind-safe (Okabe-Ito + viridis), 150+ dpi, tight layout.
"""
import argparse, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
      "#F0E442", "#000000", "#999999"]
plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "figure.dpi": 150, "savefig.dpi": 150, "axes.axisbelow": True,
})


def short_label(s):
    return f"{s['dataset']}#{s['id']}"


# --------------------------------------------------------------------------
# EXP 19 — KPredictor per-chunk priority heatmap
# --------------------------------------------------------------------------
def exp19(samples, outdir):
    os.makedirs(outdir, exist_ok=True)
    # one heatmap per sample
    for s in samples:
        nch = s["n_chunks"]
        mem = s["per_chunk_mem_sizes"]
        maxm = max(mem)
        # matrix [chunk, anchor-slot] = keep-prob (NaN past chunk length)
        M = np.full((nch, maxm), np.nan)
        for ci, kp in enumerate(s["per_chunk_keepprob"]):
            M[ci, :len(kp)] = kp
        fig, ax = plt.subplots(figsize=(min(12, 1.0 + maxm * 0.28),
                                        1.4 + nch * 0.62))
        im = ax.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        # draw the selected-k boundary per chunk
        for ci, k in enumerate(s["k_per_chunk"]):
            k_eff = min(k, mem[ci])
            ax.axvline(k_eff - 0.5, ymin=1 - (ci + 1) / nch, ymax=1 - ci / nch,
                       color="#D55E00", lw=2.4)
            ax.text(maxm - 0.3, ci, f"k={k_eff}/{mem[ci]}", va="center",
                    ha="right", fontsize=9, color="white",
                    bbox=dict(boxstyle="round,pad=0.15", fc="#D55E00",
                              ec="none", alpha=0.85))
        ax.set_xlabel("anchor slot index (within chunk, Matryoshka order)")
        ax.set_ylabel("chunk index")
        ax.set_yticks(range(nch))
        ax.set_title(f"{short_label(s)} ({s['dataset']}) — KPredictor priority "
                     f"(keep-prob $p_i$), $\\varepsilon$={s.get('eps','?')}\n"
                     f"selected $k$/chunk = {s['k_per_chunk']}  "
                     f"(retain {s['retain_rate']*100:.0f}%, mean $k$={s['mean_k']:.1f})",
                     fontsize=11)
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
        cb.set_label("keep-prob $p_i$ (priority)")
        # red line legend
        ax.plot([], [], color="#D55E00", lw=2.4, label="selected-$k$ cutoff")
        ax.legend(loc="upper right", frameon=True, fontsize=8)
        fig.tight_layout()
        p = os.path.join(outdir, f"exp19_gate_heatmap_{s['dataset']}_{s['id']}.png")
        fig.savefig(p, bbox_inches="tight"); plt.close(fig)
        print(f"[exp19] {p}")

    # cross-sample summary: mean keep-prob curve per dataset + chosen-k bar
    fig, (a0, a1) = plt.subplots(1, 2, figsize=(12, 4.2))
    by_ds = {}
    for s in samples:
        by_ds.setdefault(s["dataset"], []).append(s)
    for i, (ds, ss) in enumerate(sorted(by_ds.items())):
        # mean keep-prob over slot index across that dataset's chunks
        curves = [kp for s in ss for kp in s["per_chunk_keepprob"]]
        L = max(len(c) for c in curves)
        mat = np.full((len(curves), L), np.nan)
        for j, c in enumerate(curves):
            mat[j, :len(c)] = c
        mean_curve = np.nanmean(mat, axis=0)
        a0.plot(range(L), mean_curve, color=CB[i % len(CB)], lw=2,
                marker="o", ms=3, label=ds)
    a0.axhline(0.8, color="gray", ls="--", lw=1, label="calib. threshold")
    a0.set_xlabel("anchor slot index"); a0.set_ylabel("mean keep-prob $p_i$")
    a0.set_title("Priority decay within chunk, by dataset"); a0.legend(fontsize=8)
    a0.grid(alpha=0.3)
    # chosen-k per sample, grouped
    labels = [short_label(s) for s in samples]
    meanks = [s["mean_k"] for s in samples]
    cols = [CB[sorted(by_ds).index(s["dataset"]) % len(CB)] for s in samples]
    a1.bar(range(len(samples)), meanks, color=cols)
    a1.set_xticks(range(len(samples)))
    a1.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    a1.set_ylabel("mean selected $k$ / chunk")
    a1.set_title("Realized per-sample budget (priority-aware)")
    a1.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    p = os.path.join(outdir, "exp19_gate_summary_panel.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[exp19] {p}")


# --------------------------------------------------------------------------
# EXP 20 — decoder attention to memory: compressed vs full
# --------------------------------------------------------------------------
def exp20(samples, outdir):
    samples = [s for s in samples if "attn_compressed" in s]
    if not samples:
        print("[exp20] no attention dumps; skipping")
        return
    os.makedirs(outdir, exist_ok=True)
    # per-sample grouped bar: per-chunk attention mass full vs compressed
    n = len(samples)
    fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.8), sharey=True)
    if n == 1:
        axes = [axes]
    for ax, s in zip(axes, samples):
        full = s["attn_full"]["per_chunk_attn_mass"]
        comp = s["attn_compressed"]["per_chunk_attn_mass"]
        x = np.arange(len(full))
        w = 0.38
        ax.bar(x - w/2, full, w, color=CB[0], label="full KV")
        ax.bar(x + w/2, comp, w, color=CB[3], label="compressed KV")
        # annotate selected k per chunk
        for ci, k in enumerate(s["k_per_chunk"]):
            ax.text(ci, max(full[ci], comp[ci]) + 0.003, f"k={k}",
                    ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x); ax.set_xticklabels([f"c{ci}" for ci in x])
        ax.set_xlabel("chunk")
        ax.set_title(f"{short_label(s)}", fontsize=10)
        ax.grid(axis="y", alpha=0.3)
    axes[0].set_ylabel("decoder→memory attention mass\n(mean over layers/heads/queries)")
    axes[0].legend(fontsize=8, frameon=True)
    fig.suptitle("Decoder attention to compressed vs full KV per chunk — "
                 "compression preserves attention to salient (high-$k$) chunks",
                 y=1.04, fontsize=12)
    fig.tight_layout()
    p = os.path.join(outdir, "exp20_attention_per_chunk_full_vs_compressed.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[exp20] {p}")

    # retention-of-attention scatter: per chunk, fraction of full attention kept
    fig, ax = plt.subplots(figsize=(6.4, 4.6))
    for i, s in enumerate(samples):
        full = np.array(s["attn_full"]["per_chunk_attn_mass"])
        comp = np.array(s["attn_compressed"]["per_chunk_attn_mass"])
        retain = np.array([min(k, m) / m for k, m in
                           zip(s["k_per_chunk"], s["per_chunk_mem_sizes"])])
        attn_kept = comp / np.clip(full, 1e-9, None)
        ax.scatter(retain * 100, attn_kept * 100, color=CB[i % len(CB)],
                   s=70, label=short_label(s), edgecolor="white", zorder=3)
    lim = [0, 105]
    ax.plot(lim, lim, ls="--", color="gray", lw=1, label="y=x (proportional)")
    ax.set_xlabel("anchors retained per chunk (%)")
    ax.set_ylabel("decoder attention mass retained (%)")
    ax.set_title("Attention is retained super-proportionally to anchors\n"
                 "(points above y=x: compression keeps salient attention)")
    ax.legend(fontsize=8, frameon=True); ax.grid(alpha=0.3)
    ax.set_xlim(lim); ax.set_ylim(0, max(120, ax.get_ylim()[1]))
    fig.tight_layout()
    p = os.path.join(outdir, "exp20_attention_retention_scatter.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[exp20] {p}")


# --------------------------------------------------------------------------
# EXP 21 — single rich case study
# --------------------------------------------------------------------------
def exp21(samples, outdir, case_id, min_display_mem=8):
    os.makedirs(outdir, exist_ok=True)
    s = next((x for x in samples if x["id"] == case_id), samples[0])
    nch = s["n_chunks"]; mem = s["per_chunk_mem_sizes"]
    has_attn = "attn_compressed" in s

    # Drop tiny tail chunks (partial last chunk with too few anchor slots to be
    # meaningful, e.g. mem<min_display_mem) from the visualization only — they
    # carry almost no budget yet read as a misleading 100% on an absolute axis.
    disp = [ci for ci in range(nch) if mem[ci] >= min_display_mem]
    nd = len(disp)
    maxm = max(mem[ci] for ci in disp)
    labels = [f"c{ci}" for ci in disp]

    fig = plt.figure(figsize=(13, 3.0 + nd * 0.7))
    gs = fig.add_gridspec(1, 3, width_ratios=[2.6, 1.0, 1.0], wspace=0.32)

    # (a) keep-prob heatmap with selected-k cutoff
    ax0 = fig.add_subplot(gs[0, 0])
    M = np.full((nd, maxm), np.nan)
    for row, ci in enumerate(disp):
        kp = s["per_chunk_keepprob"][ci]
        M[row, :len(kp)] = kp
    im = ax0.imshow(M, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    for row, ci in enumerate(disp):
        ke = min(s["k_per_chunk"][ci], mem[ci])
        ax0.axvline(ke - 0.5, ymin=1 - (row + 1) / nd, ymax=1 - row / nd,
                    color="#D55E00", lw=2.6)
        ax0.text(maxm - 0.3, row, f"k={ke}/{mem[ci]}", va="center", ha="right",
                 fontsize=9, color="white",
                 bbox=dict(boxstyle="round,pad=0.15", fc="#D55E00", ec="none"))
    ax0.set_yticks(range(nd)); ax0.set_yticklabels(labels); ax0.set_ylabel("chunk")
    ax0.set_xlabel("anchor slot (Matryoshka order)")
    ax0.set_title("(a) KPredictor priority $p_i$ + selected $k$")
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.02, label="keep-prob")

    # (b) per-chunk budget relative to the most-funded chunk (max k = 100%)
    ax1 = fig.add_subplot(gs[0, 1])
    ks = [min(s["k_per_chunk"][ci], mem[ci]) for ci in disp]
    maxk = max(ks)
    rel = [k / maxk * 100 for k in ks]
    ax1.barh(range(nd), rel, color=CB[2])
    ax1.set_yticks(range(nd)); ax1.set_yticklabels(labels)
    ax1.invert_yaxis()
    ax1.set_xlabel("budget vs. max chunk (%)"); ax1.set_title("(b) relative budget")
    for i, r in enumerate(rel):
        ax1.text(r + 1.5, i, f"{r:.0f}%", va="center", fontsize=8)
    ax1.set_xlim(0, 112); ax1.grid(axis="x", alpha=0.3)

    # (c) attention full vs compressed
    ax2 = fig.add_subplot(gs[0, 2])
    if has_attn:
        full = [s["attn_full"]["per_chunk_attn_mass"][ci] for ci in disp]
        comp = [s["attn_compressed"]["per_chunk_attn_mass"][ci] for ci in disp]
        y = np.arange(nd); h = 0.38
        ax2.barh(y - h/2, full, h, color=CB[0], label="full")
        ax2.barh(y + h/2, comp, h, color=CB[3], label="compressed")
        ax2.set_yticks(y); ax2.set_yticklabels(labels)
        ax2.invert_yaxis(); ax2.legend(fontsize=8)
        ax2.set_xlabel("attn mass"); ax2.set_title("(c) decoder attention")
        ax2.grid(axis="x", alpha=0.3)
    else:
        ax2.axis("off"); ax2.text(0.5, 0.5, "attn n/a", ha="center")

    ans = s["answers"][0] if s["answers"] else "?"
    fig.suptitle(f"PACT case study — {s['dataset']} #{s['id']} "
                 f"(answer: “{ans}”)  |  "
                 f"mean $k$={s['mean_k']:.1f}, retain {s['retain_rate']*100:.0f}% "
                 f"(≈{1/max(1e-9,s['retain_rate']):.0f}×), "
                 f"$\\varepsilon$={s.get('eps','?')}", y=1.02, fontsize=12)
    fig.tight_layout()
    p = os.path.join(outdir, f"exp21_case_study_{s['dataset']}_{s['id']}.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    print(f"[exp21] {p}")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--case_id", type=int, default=52189)
    ap.add_argument("--min_display_mem", type=int, default=8,
                    help="drop chunks with fewer anchor slots than this from the "
                         "exp21 case-study panels (tiny partial tail chunks)")
    args = ap.parse_args()
    d = json.load(open(args.dump))
    eps = d.get("eps")
    for s in d["samples"]:
        s["eps"] = eps
    exp19(d["samples"], os.path.join(args.outroot, "exp19_gate_viz"))
    exp20(d["samples"], os.path.join(args.outroot, "exp20_attention"))
    exp21(d["samples"], os.path.join(args.outroot, "exp21_case_study"),
          args.case_id, args.min_display_mem)


if __name__ == "__main__":
    main()
