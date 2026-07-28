#!/usr/bin/env python3
"""Fail loudly if an eval artifact was built from an incomplete set of samples or labels.

A summary aggregated from a partial rank set, or one that silently lost a k / eps label,
is structurally valid JSON and looks fine in a table -- only its sample count and label
set give it away. Every headline eval passes through this gate.

Kinds:
  ksweep  eval_k_sweep.py summary        -> overall.<k>.count
  pareto  eval_v5_pareto.py summary      -> per_label.<eps>.n
  report  aggregate_tokenf1.py report    -> <label>.n   (this one GLOBS rank files while the
          summary reads range(world_size), so leftover higher-index rank files from a run
          with more GPUs land here and nowhere else -- checking n is what catches them)

Usage:
  check_eval_n.py <file.json> <ksweep|pareto|report> [--n 67854] [--labels 2,4,8,16 | --count 4]
"""
import argparse
import json
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("kind", choices=["ksweep", "pareto", "report"])
    ap.add_argument("--n", type=int, default=67854, help="expected samples per label")
    ap.add_argument("--labels", default=None,
                    help="comma-separated label set that MUST be present exactly")
    ap.add_argument("--count", type=int, default=None,
                    help="expected number of labels (use when label names are not fixed)")
    a = ap.parse_args()

    s = json.load(open(a.path))
    if a.kind == "ksweep":
        ns = {str(k): v["count"] for k, v in s["overall"].items()}
    elif a.kind == "pareto":
        ns = {str(k): v["n"] for k, v in s["per_label"].items()}
    else:
        ns = {str(k): v["n"] for k, v in s.items() if isinstance(v, dict) and "n" in v}

    print(f"[n-check] {a.path} ({a.kind}): {ns}")

    # An empty artifact must never pass vacuously.
    if not ns:
        sys.exit(f"[n-check] FAIL: no labels carrying a sample count in {a.path}")

    bad = {k: n for k, n in ns.items() if n != a.n}
    if bad:
        sys.exit(f"[n-check] FAIL: expected n={a.n} for every label, got {bad}")

    if a.labels:
        want = set(a.labels.split(","))
        got = set(ns)
        if want != got:
            sys.exit(f"[n-check] FAIL: label set mismatch — missing {sorted(want - got)}, "
                     f"unexpected {sorted(got - want)}")
    if a.count is not None and len(ns) != a.count:
        sys.exit(f"[n-check] FAIL: expected {a.count} labels, got {len(ns)}: {sorted(ns)}")

    print(f"[n-check] OK ({len(ns)} labels, all n={a.n})")


if __name__ == "__main__":
    main()
