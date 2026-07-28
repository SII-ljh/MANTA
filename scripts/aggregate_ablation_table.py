"""P0-B ablation table — merges clean / Matryoshka-only / gate-only / full PACT results.

Pulls from each experiment's comi_*_summary.json files and writes
reports/iclr_paper/table3_ablation.md.
"""
import os
import json
from pathlib import Path

ROOT = Path(
    os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

EXPS = [
    ("Clean (no Mat, no gate)",      "clean_baseline_comi",       ["k34"]),
    ("Mat-only (no gate)",           "pact_matryoshka_only_comi", ["k4", "k8", "k16", "k32", "k34"]),
    ("Gate-only (no Mat)",           "pact_gate_only_comi",       ["adaptive_t4", "adaptive_t8", "adaptive_t12", "adaptive_t16", "adaptive_t24"]),
    ("Full PACT (v4a_comi)",         "pact_v4a_comi",             ["k4", "k8", "k16", "k32", "adaptive"]),
]


def load_summary(exp, mode_label):
    p = ROOT / "experiment" / exp / "output" / f"comi_{mode_label}_summary.json"
    if not p.exists():
        return None
    with open(p) as f:
        d = json.load(f)
    pl = d.get("per_label", {})
    if not pl:
        return None
    key = next(iter(pl))
    return pl[key]


def main():
    md = ["# P0-B Ablation table — comi_eval 1000\n"]
    md.append("All variants trained with same COMI 100K SFT recipe (12500 steps, 8x H100).\n")
    md.append("| Variant | mode | n | mean k/chunk | comp_rate | F1 (%) | EM (%) |")
    md.append("|---|---|---:|---:|---:|---:|---:|")

    for variant, exp, modes in EXPS:
        first = True
        for m in modes:
            s = load_summary(exp, m)
            if s is None:
                md.append(f"| {variant if first else ''} | {m} | – | – | – | – | – |")
            else:
                md.append(
                    f"| {variant if first else ''} | {m} | {s['n']} | "
                    f"{s['mean_k_per_chunk']:.2f} | {s['compression_rate']:.3f} | "
                    f"{s['f1']*100:.2f} | {s['em']*100:.2f} |"
                )
            first = False

    md.append("")
    md.append("Mode legend:")
    md.append("- `k{N}`     : fixed top-k inference (k anchors per chunk).")
    md.append("- `adaptive` / `adaptive_t{N}` : soft-gate inference with target_avg=N (gate-driven Pareto).")

    out = ROOT / "reports/iclr_paper/table3_ablation.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
