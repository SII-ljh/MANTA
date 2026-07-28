#!/usr/bin/env python
"""
PACT viz: realized-compression distribution (exp17) + per-dataset selected-k
distribution (exp18) from per-sample adaptive eval shards.

Reusable: parameterized by --glob (rank-shard glob), --eps (ε for exp18),
--outdir-root, and --tag. Re-run for T4-3B / other ckpts by pointing --glob
at their pareto_* shards.

Input schema (per shard JSON): {"eps=<v>": [ {id, dataset, k_per_chunk:[...],
total_kept, total_anchors, compression_rate (=kept/anchors retain ratio),
mean_k_per_chunk, f1, em}, ... ], ...}

Definitions:
  retain ratio r   = total_kept / total_anchors  (field: compression_rate)
  compression Xfold = 1 / r
We report/plot retain ratio (bounded, unambiguous) and annotate the
mean fold-compression too.
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito colorblind-safe palette
CB = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9",
      "#F0E442", "#000000", "#999999", "#882255", "#44AA99", "#332288"]

IID = ["SearchQA", "NaturalQuestionsShort", "SQuAD", "TriviaQA-web",
       "HotpotQA", "NewsQA"]
OOD = ["BioASQ", "DROP", "DuoRC.ParaphraseRC", "RACE",
       "RelationExtraction", "TextbookQA"]

plt.rcParams.update({
    "font.size": 12, "axes.titlesize": 13, "axes.labelsize": 12,
    "legend.fontsize": 10, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "figure.dpi": 150, "savefig.dpi": 150, "axes.grid": True,
    "grid.alpha": 0.3, "axes.axisbelow": True,
})


def load_records(pattern):
    """Return {eps_str: [records...]} merged across shards."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise SystemExit(f"No files match {pattern}")
    merged = {}
    for f in files:
        d = json.load(open(f))
        for k, recs in d.items():
            merged.setdefault(k, []).extend(recs)
    print(f"[load] {len(files)} shards; eps keys={list(merged)}; "
          f"n/eps={ {k: len(v) for k, v in merged.items()} }")
    return merged


# ----------------------------------------------------------------------------
def exp17(merged, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    eps_keys = sorted(merged.keys(), key=lambda s: float(s.split("=")[1]))
    stats = {}
    fig, axes = plt.subplots(1, len(eps_keys),
                             figsize=(3.4 * len(eps_keys), 3.6),
                             sharey=True)
    if len(eps_keys) == 1:
        axes = [axes]
    bins = np.linspace(0, 1, 41)
    for i, ek in enumerate(eps_keys):
        ev = float(ek.split("=")[1])
        r = np.array([rec["compression_rate"] for rec in merged[ek]],
                     dtype=float)
        fold = 1.0 / np.clip(r, 1e-9, None)
        m, sd = float(r.mean()), float(r.std())
        stats[ek] = {
            "eps": ev, "n": int(r.size),
            "retain_mean": m, "retain_std": sd,
            "retain_p10": float(np.percentile(r, 10)),
            "retain_p50": float(np.percentile(r, 50)),
            "retain_p90": float(np.percentile(r, 90)),
            "fold_mean": float(fold.mean()), "fold_std": float(fold.std()),
            "fold_p50": float(np.percentile(fold, 50)),
        }
        ax = axes[i]
        ax.hist(r, bins=bins, color=CB[i % len(CB)], alpha=0.85,
                edgecolor="white", linewidth=0.3)
        ax.axvline(m, color="black", ls="--", lw=1.4)
        ax.set_title(rf"$\varepsilon$={ev:g}")
        ax.set_xlabel("retain ratio (kept/anchors)")
        ax.set_xlim(0, 1)
        ax.text(0.97, 0.96,
                f"mean={m:.3f}\nstd={sd:.3f}\n~{stats[ek]['fold_mean']:.0f}×",
                transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray",
                          alpha=0.85))
        if i == 0:
            ax.set_ylabel("# samples")
    fig.suptitle("PACT realized per-sample compression (retain ratio) "
                 r"by elastic-budget $\varepsilon$", y=1.02)
    fig.tight_layout()
    png = os.path.join(outdir, f"exp17_realized_rate_hist_{tag}.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    # overlaid step-hist variant
    fig2, ax = plt.subplots(figsize=(6.2, 4.2))
    for i, ek in enumerate(eps_keys):
        r = np.array([rec["compression_rate"] for rec in merged[ek]])
        ax.hist(r, bins=bins, histtype="step", lw=1.8,
                color=CB[i % len(CB)],
                label=rf"$\varepsilon$={float(ek.split('=')[1]):g} "
                      rf"($\mu$={stats[ek]['retain_mean']:.2f})")
    ax.set_xlabel("retain ratio (kept/anchors)")
    ax.set_ylabel("# samples")
    ax.set_xlim(0, 1)
    ax.legend(title="elastic budget", frameon=False)
    ax.set_title("Realized compression spread per ε (overlaid)")
    fig2.tight_layout()
    png2 = os.path.join(outdir, f"exp17_realized_rate_overlay_{tag}.png")
    fig2.savefig(png2, bbox_inches="tight")
    plt.close(fig2)

    json.dump(stats, open(os.path.join(outdir, f"exp17_stats_{tag}.json"), "w"),
              indent=2)
    # csv
    with open(os.path.join(outdir, f"exp17_stats_{tag}.csv"), "w") as fh:
        fh.write("eps,n,retain_mean,retain_std,retain_p10,retain_p50,"
                 "retain_p90,fold_mean,fold_std,fold_p50\n")
        for ek in eps_keys:
            s = stats[ek]
            fh.write(f"{s['eps']:g},{s['n']},{s['retain_mean']:.4f},"
                     f"{s['retain_std']:.4f},{s['retain_p10']:.4f},"
                     f"{s['retain_p50']:.4f},{s['retain_p90']:.4f},"
                     f"{s['fold_mean']:.3f},{s['fold_std']:.3f},"
                     f"{s['fold_p50']:.3f}\n")
    print(f"[exp17] wrote {png}, {png2}, stats json+csv")
    return stats


# ----------------------------------------------------------------------------
def exp18(merged, eps_str, outdir, tag):
    os.makedirs(outdir, exist_ok=True)
    if eps_str not in merged:
        raise SystemExit(f"eps key {eps_str} not in {list(merged)}")
    recs = merged[eps_str]
    ev = float(eps_str.split("=")[1])

    # collect per-chunk k per dataset
    by_ds = {}
    for r in recs:
        ds = r["dataset"]
        by_ds.setdefault(ds, []).extend(r["k_per_chunk"])
    # mean-k per dataset (per-chunk) for ordering
    ds_meank = {ds: float(np.mean(v)) for ds, v in by_ds.items()}
    order = sorted(ds_meank, key=lambda d: ds_meank[d])  # easiest->hardest? low k = easy

    # canonical matryoshka grid for binning display
    grid = [1, 2, 4, 8, 16, 32]
    edges = [0, 1.5, 3, 6, 12, 24, 33]  # bucket per-chunk k into {1,2,4,8,16,32}
    labels = ["1", "2", "4", "8", "16", "32"]

    stats = {"eps": ev, "datasets": {}}
    for ds in by_ds:
        arr = np.array(by_ds[ds], dtype=float)
        hist, _ = np.histogram(arr, bins=edges)
        frac = (hist / hist.sum()).tolist()
        stats["datasets"][ds] = {
            "n_chunks": int(arr.size),
            "mean_k": float(arr.mean()),
            "std_k": float(arr.std()),
            "median_k": float(np.median(arr)),
            "k_bucket_frac": dict(zip(labels, frac)),
            "split": "IID" if ds in IID else ("OOD" if ds in OOD else "?"),
        }
    stats["dataset_order_by_mean_k"] = order

    # ---- Figure: stacked-bar k distribution per dataset, ordered by mean k ----
    fig, ax = plt.subplots(figsize=(11, 5.2))
    bottoms = np.zeros(len(order))
    fracs = np.array([[stats["datasets"][ds]["k_bucket_frac"][l]
                       for l in labels] for ds in order])  # (nds, nbins)
    x = np.arange(len(order))
    for j, l in enumerate(labels):
        ax.bar(x, fracs[:, j], bottom=bottoms, width=0.78,
               color=CB[j], edgecolor="white", linewidth=0.3, label=f"k={l}")
        bottoms += fracs[:, j]
    ax.set_xticks(x)
    xtl = [f"{ds}\n({'I' if stats['datasets'][ds]['split']=='IID' else 'O'})"
           for ds in order]
    ax.set_xticklabels(xtl, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("fraction of chunks")
    ax.set_ylim(0, 1.10)
    ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    fig.suptitle(rf"PACT per-chunk selected-$k$ distribution by dataset "
                 rf"($\varepsilon$={ev:g})  --  ordered by mean $k$ "
                 rf"(left = lowest budget); number above bar = mean $k$",
                 y=1.00, fontsize=12)
    # annotate mean-k above each bar
    for xi, ds in zip(x, order):
        ax.text(xi, 1.015, f"{stats['datasets'][ds]['mean_k']:.1f}",
                ha="center", va="bottom", fontsize=8.5, fontweight="bold")
    ax.legend(title="per-chunk k", ncol=6, frameon=False,
              loc="upper center", bbox_to_anchor=(0.5, -0.28))
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    png = os.path.join(outdir, f"exp18_k_dist_per_dataset_{tag}.png")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)

    # ---- companion: mean-k bar (IID vs OOD colored) ----
    fig2, ax = plt.subplots(figsize=(9, 4.4))
    meanks = [stats["datasets"][ds]["mean_k"] for ds in order]
    cols = ["#0072B2" if stats["datasets"][ds]["split"] == "IID"
            else "#D55E00" for ds in order]
    ax.bar(x, meanks, color=cols, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=40, ha="right", fontsize=9)
    ax.set_ylabel("mean per-chunk k")
    ax.axhline(float(np.mean([k for ds in order
                              for k in by_ds[ds]])), color="gray", ls="--",
               lw=1, label="global mean")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#0072B2", label="IID"),
                       Patch(color="#D55E00", label="OOD")], frameon=False)
    ax.set_title(rf"Mean per-chunk budget by dataset ($\varepsilon$={ev:g})")
    fig2.tight_layout()
    png2 = os.path.join(outdir, f"exp18_meank_per_dataset_{tag}.png")
    fig2.savefig(png2, bbox_inches="tight")
    plt.close(fig2)

    json.dump(stats, open(os.path.join(outdir, f"exp18_stats_{tag}.json"), "w"),
              indent=2)
    with open(os.path.join(outdir, f"exp18_stats_{tag}.csv"), "w") as fh:
        fh.write("dataset,split,n_chunks,mean_k,median_k,std_k,"
                 + ",".join(f"frac_k{l}" for l in labels) + "\n")
        for ds in order:
            s = stats["datasets"][ds]
            fh.write(f"{ds},{s['split']},{s['n_chunks']},{s['mean_k']:.3f},"
                     f"{s['median_k']:.1f},{s['std_k']:.3f},"
                     + ",".join(f"{s['k_bucket_frac'][l]:.4f}" for l in labels)
                     + "\n")
    print(f"[exp18] wrote {png}, {png2}, stats json+csv")
    print("[exp18] mean-k order (low->high): "
          + ", ".join(f"{ds}={ds_meank[ds]:.2f}" for ds in order))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", required=True)
    ap.add_argument("--eps", default="eps=0.05",
                    help="eps key for exp18 (e.g. eps=0.05)")
    ap.add_argument("--outdir-root", required=True)
    ap.add_argument("--tag", default="t1_1b_v3")
    ap.add_argument("--only", choices=["17", "18", "both"], default="both")
    args = ap.parse_args()
    merged = load_records(args.glob)
    if args.only in ("17", "both"):
        exp17(merged, os.path.join(args.outdir_root,
                                   "exp17_realized_rate_hist"), args.tag)
    if args.only in ("18", "both"):
        exp18(merged, args.eps,
              os.path.join(args.outdir_root, "exp18_k_dist_per_dataset"),
              args.tag)


if __name__ == "__main__":
    main()
