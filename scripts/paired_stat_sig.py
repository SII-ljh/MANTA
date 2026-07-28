"""Paired bootstrap test: PACT adaptive vs PACT fixed-k at matched compression.

Tests the T-elastic claim: per-sample (same input), adaptive ≥ fixed-k.
Bootstrap pairs across samples; CI on Δ F1; one-sided p-value Pr(Δ ≤ 0).

Output: reports/iclr_paper/table_paired_elastic_vs_fixed.md
"""

import os
import json
from pathlib import Path
from collections import defaultdict

import numpy as np

ROOT = Path(
    os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
EXP = ROOT / "experiment" / "pact_v4a_comi" / "output"


def load_mode(mode: str, nranks: int = 8):
    """Load per-sample records keyed by (id, dataset) to allow pairing."""
    records = {}
    for r in range(nranks):
        p = EXP / f"comi_{mode}_rank{r}.json"
        if not p.exists():
            # k16/k32 sometimes used 4 ranks
            continue
        with open(p) as f:
            d = json.load(f)
        for _, lst in d.items():
            for rec in lst:
                key = (rec["id"], rec["dataset"])
                records[key] = rec
    return records


def paired_bootstrap(deltas: np.ndarray, n_resample: int = 1000, seed: int = 42):
    rng = np.random.default_rng(seed)
    n = len(deltas)
    boots = np.empty(n_resample)
    for i in range(n_resample):
        idx = rng.integers(0, n, n)
        boots[i] = deltas[idx].mean()
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    # one-sided p: Pr(mean_delta ≤ 0)
    p_one_sided = float((boots <= 0).mean())
    return lo, hi, p_one_sided, float(boots.std(ddof=1))


def compare_pair(label_a: str, mode_a: str, n_a: int,
                  label_b: str, mode_b: str, n_b: int):
    a = load_mode(mode_a, n_a)
    b = load_mode(mode_b, n_b)
    common_keys = sorted(set(a) & set(b))
    if not common_keys:
        return None
    f1_a = np.array([a[k]["f1"] * 100 for k in common_keys])
    f1_b = np.array([b[k]["f1"] * 100 for k in common_keys])
    deltas = f1_a - f1_b
    lo, hi, p, se = paired_bootstrap(deltas)
    return {
        "label_a": label_a, "label_b": label_b,
        "n": len(common_keys),
        "mean_a": float(f1_a.mean()), "mean_b": float(f1_b.mean()),
        "delta": float(deltas.mean()),
        "ci_lo": lo, "ci_hi": hi,
        "p_one_sided": p, "se": se,
    }


def main():
    pairs = [
        # (adaptive_label, mode, ranks, fixed_label, mode, ranks)
        ("adaptive t=4",  "adaptive_t4",  8, "fixed k=4",  "k4",  8),
        ("adaptive t=8",  "adaptive_t8",  8, "fixed k=8",  "k8",  8),
        ("adaptive t=16", "adaptive_t16", 8, "fixed k=16", "k16", 4),
        ("adaptive t=32", "adaptive_t32", 8, "fixed k=32", "k32", 4),
    ]

    lines = ["# Paired bootstrap — Elastic (adaptive) vs Fixed-k (PACT v4a_comi)\n"]
    lines.append(
        "Same model checkpoint; per-sample paired Δ F1. 1000-resample bootstrap on Δ. "
        "Null: adaptive ≤ fixed (p = Pr(mean Δ ≤ 0)).\n"
    )
    lines.append("| Pair | n | adaptive F1 | fixed F1 | Δ F1 | 95% CI | one-sided p |")
    lines.append("|---|---:|---:|---:|---:|---|---:|")
    for la, ma, nra, lb, mb, nrb in pairs:
        r = compare_pair(la, ma, nra, lb, mb, nrb)
        if r is None:
            lines.append(f"| {la} vs {lb} | – | – | – | – | – | – |")
            continue
        sig = "*" if r["p_one_sided"] < 0.05 else ""
        lines.append(
            f"| {la} vs {lb} | {r['n']} | {r['mean_a']:.2f} | "
            f"{r['mean_b']:.2f} | {r['delta']:+.2f} | "
            f"[{r['ci_lo']:+.2f}, {r['ci_hi']:+.2f}] | {r['p_one_sided']:.3f}{sig} |"
        )

    out = ROOT / "reports" / "iclr_paper" / "table_paired_elastic_vs_fixed.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
