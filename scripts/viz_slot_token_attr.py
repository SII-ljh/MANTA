"""Aggregate per-token NLL attribution shards -> correlations, concentration, plot, table.

Tests the residual-code claim at TOKEN resolution:
  H: adding the canonical next slot reduces loss specifically on tokens that were HARD
     (high NLL) under the previous prefix. -> positive corr(base_nll, delta), strong for
     canonical, weak/null for reverse/shuffled/random.

Metrics per policy (pooled over all sample,s,token):
  * Pearson & Spearman corr(base_nll, delta)            -> "is reduction targeted at hard tokens?"
  * mean delta (total reduction per token)              -> "how much does the slot buy?"
  * concentration: share of TOTAL positive reduction captured by the hardest-quartile
    tokens (top 25% by base_nll). Uniform baseline = 0.25; >0.25 = front-loaded on hard tokens.
  * per-s Pearson (early vs late slots).
"""
import argparse, glob, json, os
import numpy as np
from scipy.stats import pearsonr, spearmanr

POLICIES = ["canonical", "reverse", "shuffled", "random"]


def load(in_dir):
    out = {p: {k: [] for k in ["base_nll", "delta", "s", "sample", "tok"]} for p in POLICIES}
    n_used = 0
    files = sorted(glob.glob(os.path.join(in_dir, "tokattr_shard*.npz")))
    assert files, f"no shards in {in_dir}"
    for f in files:
        z = np.load(f)
        n_used += int(z["meta__n_used"][0])
        for p in POLICIES:
            for k in out[p]:
                out[p][k].append(z[f"{p}__{k}"])
    for p in POLICIES:
        for k in out[p]:
            out[p][k] = np.concatenate(out[p][k])
    return out, n_used, files


def concentration(base_nll, delta, q=0.75):
    """Share of total POSITIVE reduction on the top (1-q) hardest tokens (by base_nll)."""
    pos = np.clip(delta, 0, None)
    tot = pos.sum()
    if tot <= 0:
        return float("nan")
    thr = np.quantile(base_nll, q)
    hard = base_nll >= thr
    return float(pos[hard].sum() / tot)


def summarize(data):
    rows = {}
    for p in POLICIES:
        b, d, s = data[p]["base_nll"], data[p]["delta"], data[p]["s"]
        m = np.isfinite(b) & np.isfinite(d)
        b, d, s = b[m], d[m], s[m]
        pr = pearsonr(b, d)
        sr = spearmanr(b, d)
        # per-s pearson for the first few add-steps
        per_s = {}
        for sv in sorted(set(s.tolist()))[:8]:
            mm = s == sv
            if mm.sum() > 20:
                per_s[int(sv)] = float(pearsonr(b[mm], d[mm])[0])
        rows[p] = {
            "n_pairs": int(b.size),
            "pearson_r": float(pr[0]), "pearson_p": float(pr[1]),
            "spearman_r": float(sr[0]),
            "mean_delta": float(d.mean()),
            "frac_delta_positive": float((d > 0).mean()),
            "concentration_top25": concentration(b, d, q=0.75),
            "concentration_top10": concentration(b, d, q=0.90),
            "pearson_per_s": per_s,
        }
    return rows


def make_plot(data, rows, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))

    # (0,0) hexbin canonical: base_nll vs delta
    ax = axes[0, 0]
    b, d = data["canonical"]["base_nll"], data["canonical"]["delta"]
    hb = ax.hexbin(b, d, gridsize=45, bins="log", cmap="viridis")
    ax.axhline(0, color="r", lw=0.7, ls="--")
    ax.set_title(f"canonical: per-token base NLL vs ΔNLL\nPearson r={rows['canonical']['pearson_r']:.3f}")
    ax.set_xlabel("base NLL (prefix k-1)"); ax.set_ylabel("ΔNLL (+ = improvement)")
    fig.colorbar(hb, ax=ax, label="log10(count)")

    # (0,1) binned mean delta vs base-nll decile, all policies
    ax = axes[0, 1]
    for p in POLICIES:
        b, d = data[p]["base_nll"], data[p]["delta"]
        order = np.argsort(b)
        b, d = b[order], d[order]
        edges = np.quantile(b, np.linspace(0, 1, 11))
        xs, ys = [], []
        for i in range(10):
            mm = (b >= edges[i]) & (b <= edges[i + 1])
            if mm.sum():
                xs.append(b[mm].mean()); ys.append(d[mm].mean())
        ax.plot(xs, ys, marker="o", label=p)
    ax.axhline(0, color="k", lw=0.6, ls=":")
    ax.set_title("mean ΔNLL vs base-NLL decile"); ax.set_xlabel("base NLL (decile mean)")
    ax.set_ylabel("mean ΔNLL"); ax.legend()

    # (0,2) correlation bars
    ax = axes[0, 2]
    x = np.arange(len(POLICIES))
    ax.bar(x - 0.2, [rows[p]["pearson_r"] for p in POLICIES], 0.4, label="Pearson")
    ax.bar(x + 0.2, [rows[p]["spearman_r"] for p in POLICIES], 0.4, label="Spearman")
    ax.set_xticks(x); ax.set_xticklabels(POLICIES, rotation=20)
    ax.set_title("corr(base NLL, ΔNLL)"); ax.axhline(0, color="k", lw=0.6); ax.legend()

    # (1,0) concentration bars
    ax = axes[1, 0]
    ax.bar(x - 0.2, [rows[p]["concentration_top25"] for p in POLICIES], 0.4, label="top25% hard")
    ax.bar(x + 0.2, [rows[p]["concentration_top10"] for p in POLICIES], 0.4, label="top10% hard")
    ax.axhline(0.25, color="r", ls="--", lw=0.8, label="uniform top25%")
    ax.axhline(0.10, color="orange", ls="--", lw=0.8, label="uniform top10%")
    ax.set_xticks(x); ax.set_xticklabels(POLICIES, rotation=20)
    ax.set_title("share of total reduction on hardest tokens"); ax.legend(fontsize=8)

    # (1,1) mean delta bars
    ax = axes[1, 1]
    ax.bar(x, [rows[p]["mean_delta"] for p in POLICIES], color="slateblue")
    ax.set_xticks(x); ax.set_xticklabels(POLICIES, rotation=20)
    ax.set_title("mean ΔNLL per token (per added slot)"); ax.axhline(0, color="k", lw=0.6)

    # (1,2) per-s pearson (canonical vs controls)
    ax = axes[1, 2]
    for p in POLICIES:
        ps = rows[p]["pearson_per_s"]
        if ps:
            ks = sorted(ps); ax.plot(ks, [ps[k] for k in ks], marker="o", label=p)
    ax.set_title("Pearson(base NLL, ΔNLL) by add-step s"); ax.set_xlabel("add-step s")
    ax.set_ylabel("Pearson r"); ax.axhline(0, color="k", lw=0.6); ax.legend()

    fig.tight_layout()
    fig.savefig(out_png, dpi=130)
    fig.savefig(out_png.replace(".png", ".pdf"))
    print(f"[INFO] wrote {out_png}")


def write_table(rows, n_used, files, out_md):
    L = ["# Per-token NLL attribution — slot residual code (token resolution)", "",
         f"- samples used: **{n_used}**  | shards: {len(files)}",
         "- delta = NLL(canonical prefix k-1) - NLL(prefix k-1 + 1 slot); + = improvement",
         "- base prefix held FIXED (canonical k-1) across all arms; only the ADDED slot varies",
         "- concentration uniform baselines: top25%→0.25, top10%→0.10", "",
         "| policy | n_pairs | Pearson r | (p) | Spearman r | mean ΔNLL | %Δ>0 | conc.top25 | conc.top10 |",
         "|---|--:|--:|--:|--:|--:|--:|--:|--:|"]
    for p in POLICIES:
        r = rows[p]
        L.append(f"| {p} | {r['n_pairs']} | {r['pearson_r']:.3f} | {r['pearson_p']:.1e} | "
                 f"{r['spearman_r']:.3f} | {r['mean_delta']:.4f} | {r['frac_delta_positive']*100:.1f}% | "
                 f"{r['concentration_top25']:.3f} | {r['concentration_top10']:.3f} |")
    L += ["", "## Pearson(base NLL, ΔNLL) by add-step s (first slots)", "",
          "| s | " + " | ".join(POLICIES) + " |", "|--:|" + "---|" * len(POLICIES)]
    all_s = sorted({s for p in POLICIES for s in rows[p]["pearson_per_s"]})
    for s in all_s:
        L.append(f"| {s} | " + " | ".join(
            f"{rows[p]['pearson_per_s'].get(s, float('nan')):.3f}" for p in POLICIES) + " |")
    open(out_md, "w").write("\n".join(L) + "\n")
    print(f"[INFO] wrote {out_md}")
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    data, n_used, files = load(a.in_dir)
    rows = summarize(data)
    json.dump(rows, open(os.path.join(a.out_dir, "summary.json"), "w"), indent=2)
    make_plot(data, rows, os.path.join(a.out_dir, "token_attribution.png"))
    write_table(rows, n_used, files, os.path.join(a.out_dir, "summary_table.md"))


if __name__ == "__main__":
    main()
