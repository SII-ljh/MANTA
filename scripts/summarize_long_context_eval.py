"""Summarise long-context eval results across (ckpt, length) combinations.

Reads eval_summary.json from each experiment/<src>_eval_long_<L>/output/
and prints a compact F1/EM table.

CPU only.
"""

import json
import os
import sys

BASE = os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOURCES = ["sac_8b_cross_chunk_unified", "sac_8b_perceiver_io_unified"]
LENGTHS = [4096, 8192, 16384, 32768]


def load_summary(src, L):
    p = os.path.join(BASE, "experiment", f"{src}_eval_long_{L}",
                     "output", "eval_summary.json")
    if not os.path.isfile(p):
        return None
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main():
    rows = []
    for src in SOURCES:
        for L in LENGTHS:
            s = load_summary(src, L)
            if s is None:
                rows.append((src, L, None, None, None, 0))
                continue
            rows.append((
                src, L,
                s["avg_rouge1_f1"], s["avg_exact_match"],
                s.get("avg_bleu4"), s["num_samples"],
            ))

    print(f"{'experiment':<48s} {'len':>6s} {'F1':>8s} {'EM':>8s} {'BLEU4':>8s} {'n':>6s}")
    for src, L, f1, em, bleu, n in rows:
        if f1 is None:
            print(f"{src:<48s} {L:>6d}    pending")
            continue
        print(f"{src:<48s} {L:>6d} {f1*100:>7.2f}% {em*100:>7.2f}% {bleu*100:>7.2f}% {n:>6d}")

    out_path = os.path.join(BASE, "logs", "long_context_eval_summary.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump([
            {"src": src, "length": L,
             "f1": f1, "em": em, "bleu4": bleu, "n": n}
            for src, L, f1, em, bleu, n in rows
        ], f, indent=2)
    print(f"\nWritten to {out_path}")


if __name__ == "__main__":
    main()
