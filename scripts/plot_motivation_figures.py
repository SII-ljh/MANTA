#!/usr/bin/env python
"""Generate Figure 1 right panel + Figure 2(a)(b) from motivation_probe output.

Inputs:
  - analysis/motivation_probe/per_chunk_loss.json  (per-chunk loss matrix)
  - analysis/oracle_eval/summary.json              (oracle vs fixed-k curves)

Outputs (PDF):
  - figures/figure1_right.pdf  — oracle vs fixed-k mean LM loss vs avg budget
  - figures/figure2a.pdf       — 50-chunk loss-vs-k spaghetti plot
  - figures/figure2b.pdf       — "min k to reach K_max within tol" histogram
"""
import argparse
import json
import os
import random

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


K_SET = [1, 2, 4, 8, 16, 34]
K_MAX = max(K_SET)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per_chunk_loss",
                   default="analysis/motivation_probe/per_chunk_loss.json")
    p.add_argument("--oracle_summary",
                   default="analysis/oracle_eval/summary.json")
    p.add_argument("--out_dir", default="figures")
    p.add_argument("--n_spaghetti", type=int, default=50)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tolerance", type=float, default=0.05)
    return p.parse_args()


def figure2a(records, n_spaghetti, seed, out_path):
    rng = random.Random(seed)
    chosen = rng.sample(records, k=min(n_spaghetti, len(records)))
    fig, ax = plt.subplots(figsize=(5, 3.5))
    for r in chosen:
        ys = [float(r["losses"][str(k)]) for k in K_SET]
        ax.plot(K_SET, ys, color="C0", alpha=0.35, lw=0.9)
    ax.set_xscale("log")
    ax.set_xticks(K_SET); ax.set_xticklabels([str(k) for k in K_SET])
    ax.set_xlabel("k (anchor slots per chunk)")
    ax.set_ylabel("Decoder LM loss")
    ax.set_title("(a) Per-chunk loss-vs-k is heterogeneous")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def figure2b(records, tolerance, out_path):
    min_k_counts = {k: 0 for k in K_SET}
    for r in records:
        L_full = float(r["losses"][str(K_MAX)])
        for k in K_SET:
            if float(r["losses"][str(k)]) - L_full <= tolerance:
                min_k_counts[k] += 1
                break
    fig, ax = plt.subplots(figsize=(5, 3.5))
    xs = list(range(len(K_SET)))
    ys = [min_k_counts[k] / max(1, len(records)) for k in K_SET]
    ax.bar(xs, ys, color="C2", edgecolor="black")
    ax.set_xticks(xs); ax.set_xticklabels([str(k) for k in K_SET])
    ax.set_xlabel("min k s.t. L_k − L_K_max ≤ {:.2f}".format(tolerance))
    ax.set_ylabel("Fraction of chunks")
    ax.set_title("(b) Min-k distribution spans the entire grid")
    for x, y in zip(xs, ys):
        ax.text(x, y + 0.005, f"{100 * y:.1f}%", ha="center", fontsize=8)
    ax.set_ylim(0, max(ys) * 1.2)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def figure1_right(oracle_summary, out_path):
    fig1 = oracle_summary["figure1_curves"]
    fixed_per_k = {int(k): v for k, v in fig1["fixed_per_k"].items()}
    oracle = {int(k): v["mean_chunk_loss"] for k, v in fig1["oracle"].items()}
    achieved_mk = {int(k): v["achieved_mean_k"] for k, v in fig1["oracle"].items()}

    fig, ax = plt.subplots(figsize=(5, 3.5))
    xs_fixed = sorted(fixed_per_k.keys())
    ys_fixed = [fixed_per_k[k] for k in xs_fixed]
    ax.plot(xs_fixed, ys_fixed, "o-", color="C1", label="Fixed k", lw=1.6, ms=6)

    xs_oracle = sorted(oracle.keys())
    ys_oracle = [oracle[k] for k in xs_oracle]
    xs_oracle_actual = [achieved_mk[k] for k in xs_oracle]
    ax.plot(xs_oracle_actual, ys_oracle, "s--", color="C0",
            label="Oracle (joint budget)", lw=1.6, ms=6)

    ax.set_xscale("log")
    ax.set_xlabel("Average k (slots per chunk)")
    ax.set_ylabel("LM loss")
    ax.set_title("Oracle per-chunk allocation Pareto-dominates fixed k")
    ax.legend(loc="upper right")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, format="pdf", bbox_inches="tight")
    plt.close()
    print(f"[saved] {out_path}")


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    records = json.load(open(args.per_chunk_loss))
    print(f"[plot] {len(records)} chunk records")

    figure2a(records, args.n_spaghetti, args.seed,
             os.path.join(args.out_dir, "figure2a_spaghetti.pdf"))
    figure2b(records, args.tolerance,
             os.path.join(args.out_dir, "figure2b_min_k_hist.pdf"))

    if os.path.exists(args.oracle_summary):
        oracle_summary = json.load(open(args.oracle_summary))
        figure1_right(oracle_summary,
                      os.path.join(args.out_dir, "figure1_right_oracle.pdf"))
    else:
        print(f"[skip] {args.oracle_summary} not found — skipping Figure 1 right")


if __name__ == "__main__":
    main()
