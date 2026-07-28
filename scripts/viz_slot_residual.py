"""Build the 4-panel "ordered residual code" figure + summary table from
slot_residual_analysis.py functional records.

Panels:
  (a) order-ablation: answer-NLL vs budget k for canonical / reverse / shuffled-null
      / random-band (mean +/- 95% CI).  -> order matters (residual, not a decorrelated bag)
  (b) order-sensitivity bar: mean(random - canonical) NLL gap; PACT (+ SAC if --sac given)
  (c) marginal-gain decay g_k = NLL(k-1)-NLL(k) on canonical fine ladder (log-y),
      canonical vs shuffled-null.  -> steep monotone diminishing residual
  (d) residual alignment cos(D_k, r_k) + residual-norm ||r_k|| vs k, canonical vs null.

Also writes summary_table.md with order-sensitivity / align-index / marginal-gain
Spearman / monotonic-fraction per condition.
"""
import argparse, glob, json, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

KS = [1, 2, 4, 8, 16, 32]


def load(out_dir):
    recs = []
    for f in glob.glob(os.path.join(out_dir, "functional_shard*.json")):
        recs += json.load(open(f))
    return recs


def ci95(a, axis=0):
    a = np.asarray(a, float)
    return 1.96 * a.std(axis) / np.sqrt(a.shape[axis])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--fig_dir", required=True)
    ap.add_argument("--label", default="PACT-1B v4")
    ap.add_argument("--sac_out_dir", default=None, help="optional SAC records dir for panel (b)")
    args = ap.parse_args()
    os.makedirs(args.fig_dir, exist_ok=True)
    recs = load(args.out_dir)
    full = [r for r in recs if len(r["ks"]) == 6]
    n = len(full)
    print(f"[viz] {len(recs)} records, {n} full-32 ladder")

    canon = np.array([r["nll_canonical"] for r in full])
    rev = np.array([r["nll_reverse"] for r in full])
    shuf = np.array([r["nll_shuffled"] for r in full])
    rand = np.array([np.array(r["nll_random_runs"]).mean(0) for r in full])

    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))

    # (a) order-ablation curves
    a = ax[0, 0]
    for arr, lab, c, m in [(canon, "canonical (PACT order)", "#1f77b4", "o"),
                            (rand, "random k-subset", "#7f7f7f", "s"),
                            (shuf, "shuffled-null", "#ff7f0e", "^"),
                            (rev, "reverse", "#d62728", "v")]:
        mu = arr.mean(0); e = ci95(arr)
        a.plot(KS, mu, marker=m, color=c, label=lab)
        a.fill_between(KS, mu - e, mu + e, color=c, alpha=0.15)
    a.set_xscale("log", base=2); a.set_xticks(KS); a.set_xticklabels(KS)
    a.set_xlabel("budget k (slots kept per chunk)"); a.set_ylabel("answer NLL")
    a.set_title("(a) Order ablation — order carries the information")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # (b) order-sensitivity bar — computed over the COMMON k-subrange {1,2,4,8}
    # so PACT (32-slot ladder) and SAC (8-slot, k_choices=[8]) are apples-to-apples.
    b = ax[0, 1]
    common = [1, 2, 4, 8]
    cidx = [KS.index(k) for k in common]
    os_pact = float((rand[:, cidx] - canon[:, cidx]).mean())
    labels, vals = [args.label], [os_pact]
    if args.sac_out_dir:
        s = load(args.sac_out_dir)
        sf = [r for r in s if r["ks"][:4] == common]
        if sf:
            sc = np.array([r["nll_canonical"][:4] for r in sf])
            sr = np.array([np.array(r["nll_random_runs"])[:, :4].mean(0) for r in sf])
            labels.append("SAC-1B (r64)"); vals.append(float((sr - sc).mean()))
            print(f"[viz] SAC n={len(sf)} order-sensitivity(k<=8)={vals[-1]:.3f}")
    b.bar(labels, vals, color=["#1f77b4", "#2ca02c"][:len(labels)])
    b.set_ylabel("order-sensitivity (k<=8)\nmean(random - canonical) NLL")
    b.set_title("(b) Order-sensitivity: PACT learned order vs SAC")
    b.grid(alpha=0.3, axis="y")

    # (c) marginal-gain decay (canonical fine ladder)
    c = ax[1, 0]
    def fine_gain(key):
        gs = []
        for r in full:
            nf = np.array(r[key]); gs.append(-np.diff(nf))
        # pad to common length (max m); all full -> 32, so len 31
        L = min(len(g) for g in gs)
        return np.array([g[:L] for g in gs])
    gc = fine_gain("nll_canonical_fine"); gs = fine_gain("nll_shuffled_fine")
    kk = np.arange(1, gc.shape[1] + 1)
    c.plot(kk, gc.mean(0), color="#1f77b4", marker="o", ms=3, label="canonical")
    c.plot(kk, gs.mean(0), color="#ff7f0e", marker="^", ms=3, label="shuffled-null")
    c.axhline(0, color="k", lw=0.6)
    c.set_xlabel("slot index k"); c.set_ylabel("marginal gain  NLL(k-1)-NLL(k)")
    c.set_title("(c) Diminishing residual — front-loaded gains")
    c.legend(fontsize=8); c.grid(alpha=0.3)

    # (d) residual alignment + residual norm
    d = ax[1, 1]
    def stack_align(key):
        cos = [np.array(r[key]["cos"]) for r in recs]
        rn = [np.array(r[key]["rnorm"]) for r in recs]
        L = min(min(len(x) for x in cos), min(len(x) for x in rn))
        return np.array([x[:L] for x in cos]), np.array([x[:L] for x in rn])
    cc, rnc = stack_align("align_canonical")
    cs, rns = stack_align("align_shuffled")
    kx = np.arange(1, cc.shape[1] + 1)
    d.plot(kx, cc.mean(0), color="#1f77b4", marker="o", ms=3, label="cos(Δ,r) canonical")
    d.plot(kx, cs.mean(0), color="#ff7f0e", marker="^", ms=3, label="cos(Δ,r) shuffled")
    d.set_xlabel("slot index k"); d.set_ylabel("residual alignment  cos(Δ_k, r_k)")
    d.set_title("(d) Each slot points along the prefix residual")
    d2 = d.twinx()
    d2.plot(np.arange(1, rnc.shape[1] + 1), rnc.mean(0), color="#1f77b4", ls="--", alpha=0.6)
    d2.plot(np.arange(1, rns.shape[1] + 1), rns.mean(0), color="#ff7f0e", ls="--", alpha=0.6)
    d2.set_ylabel("residual norm ||r_k|| (dashed)")
    d.legend(fontsize=8, loc="upper right"); d.grid(alpha=0.3)

    fig.suptitle(f"PACT anchor slots form an ordered residual code  (n={n}, {args.label})", y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    for ext in ["png", "pdf"]:
        fig.savefig(os.path.join(args.fig_dir, f"slot_residual.{ext}"), dpi=150)
    print("[viz] saved figure ->", args.fig_dir)

    # ---- summary table ----
    from scipy.stats import spearmanr
    def gain_stats(g):
        mu = g.mean(0)
        rho = spearmanr(np.arange(len(mu)), mu).correlation
        mono = float((np.diff(mu) <= 1e-6).mean())  # frac of steps where gain decreases
        return rho, mono
    rho_c, mono_c = gain_stats(gc)
    rho_s, mono_s = gain_stats(gs)
    lines = [
        f"# Slot ordered-residual summary ({args.label}, n={n})", "",
        "| condition | order-sensitivity (random-canonical NLL) | align-index mean cos(Δ,r) | marginal-gain Spearman(k,gain) | monotone-frac |",
        "|---|---|---|---|---|",
        f"| canonical | {os_pact:.3f} | {cc.mean():.3f} | {rho_c:.3f} | {mono_c:.2f} |",
        f"| shuffled-null | 0 (ref) | {cs.mean():.3f} | {rho_s:.3f} | {mono_s:.2f} |",
        "",
        "NLL by k (mean):", "",
        "| policy | " + " | ".join(f"k={k}" for k in KS) + " |",
        "|" + "---|" * (len(KS) + 1),
        "| canonical | " + " | ".join(f"{v:.3f}" for v in canon.mean(0)) + " |",
        "| random | " + " | ".join(f"{v:.3f}" for v in rand.mean(0)) + " |",
        "| shuffled | " + " | ".join(f"{v:.3f}" for v in shuf.mean(0)) + " |",
        "| reverse | " + " | ".join(f"{v:.3f}" for v in rev.mean(0)) + " |",
    ]
    with open(os.path.join(args.fig_dir, "summary_table.md"), "w") as f:
        f.write("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
