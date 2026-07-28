"""Aggregate per-experiment eval_by_length_and_dataset.json into one comparison report.

Two tables:
  1. F1 by length bucket   (columns = experiments, rows = length buckets + TOTAL)
  2. F1 by dataset         (columns = experiments, rows = datasets sorted by n)

Both also report delta vs a chosen baseline column.

Usage:
    python util/compare_eval_reports.py \\
        --baseline experiment/sac_8b_unified \\
        --label-baseline base \\
        --exp experiment/sac_8b_cross_chunk_inst:V1 \\
        --exp experiment/sac_8b_cross_chunk_v3_inst:V3 \\
        --exp experiment/sac_8b_cross_chunk_v4a2_4gpu_inst:V4A2 \\
        --output reports/eval_compare.md
"""

import argparse
import json
import os

LENGTH_BUCKET_ORDER = [
    "short_0-512",
    "medium_512-1k",
    "medium_1k-2k",
    "long_2k-4k",
    "long_4k-8k",
    "very_long_8k+",
]


def load_report(path):
    with open(os.path.join(path, "output", "eval_by_length_and_dataset.json")) as f:
        return json.load(f)


def parse_exp(spec):
    if ":" in spec:
        path, label = spec.split(":", 1)
    else:
        path = spec
        label = os.path.basename(os.path.normpath(spec))
    return path, label


def fmt_cell(v):
    return f"{v*100:.2f}" if v is not None else "—"


def fmt_delta(v, base):
    if v is None or base is None:
        return "—"
    d = (v - base) * 100
    sign = "+" if d > 0 else ""
    return f"{sign}{d:.2f}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="baseline work_dir (forms 'base' column)")
    p.add_argument("--label-baseline", default="base")
    p.add_argument("--exp", action="append", default=[],
                   help="extra exp specs: path[:label]; repeat the flag")
    p.add_argument("--output", default="reports/eval_compare.md")
    args = p.parse_args()

    base_report = load_report(args.baseline)
    columns = [(args.label_baseline, base_report)]
    for spec in args.exp:
        path, label = parse_exp(spec)
        columns.append((label, load_report(path)))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    out = []
    out.append("# Cross-experiment eval comparison")
    out.append("")
    out.append(f"Baseline column: **{args.label_baseline}** = `{args.baseline}`")
    out.append("")
    out.append("Each cell shows ROUGE-1 F1 (%). For non-baseline columns the "
               "second number is delta vs baseline.")
    out.append("")

    # ----- 1. Per length bucket -----
    out.append("## 1. F1 by length bucket")
    out.append("")
    header = ["bucket", "n"] + [f"{lab} F1" for lab, _ in columns]
    for lab, _ in columns[1:]:
        header.append(f"Δ {lab}")
    out.append("| " + " | ".join(header) + " |")
    out.append("|" + "|".join(["---"] + ["---:"] * (len(header) - 1)) + "|")

    def n_of(report, key, bucket):
        return report[key].get(bucket, {}).get("n", 0)

    def f1_of(report, key, bucket):
        return report[key].get(bucket, {}).get("f1")

    for b in LENGTH_BUCKET_ORDER:
        n = n_of(base_report, "by_length_bucket", b)
        if n == 0:
            continue
        base_f1 = f1_of(base_report, "by_length_bucket", b)
        cells = [b, str(n)]
        for lab, rep in columns:
            cells.append(fmt_cell(f1_of(rep, "by_length_bucket", b)))
        for lab, rep in columns[1:]:
            cells.append(fmt_delta(f1_of(rep, "by_length_bucket", b), base_f1))
        out.append("| " + " | ".join(cells) + " |")

    # TOTAL row
    total_n = base_report["overall"]["n"]
    total_base = base_report["overall"]["f1"]
    cells = ["**TOTAL**", f"**{total_n}**"]
    for lab, rep in columns:
        cells.append(f"**{rep['overall']['f1']*100:.2f}**")
    for lab, rep in columns[1:]:
        cells.append(f"**{fmt_delta(rep['overall']['f1'], total_base)}**")
    out.append("| " + " | ".join(cells) + " |")
    out.append("")

    # ----- 2. Per dataset -----
    out.append("## 2. F1 by dataset")
    out.append("")
    out.append("| dataset | n | " + " | ".join(f"{lab} F1" for lab, _ in columns)
               + " | " + " | ".join(f"Δ {lab}" for lab, _ in columns[1:]) + " |")
    out.append("|" + "|".join(["---"] + ["---:"] * (len(columns) * 2 - 1)) + "|")

    datasets = sorted(base_report["by_dataset"].keys(),
                      key=lambda d: -base_report["by_dataset"][d]["n"])
    for ds in datasets:
        n = base_report["by_dataset"][ds]["n"]
        base_f1 = base_report["by_dataset"][ds]["f1"]
        cells = [ds, str(n)]
        for lab, rep in columns:
            v = rep["by_dataset"].get(ds, {}).get("f1")
            cells.append(fmt_cell(v))
        for lab, rep in columns[1:]:
            v = rep["by_dataset"].get(ds, {}).get("f1")
            cells.append(fmt_delta(v, base_f1))
        out.append("| " + " | ".join(cells) + " |")
    out.append("")

    with open(args.output, "w") as f:
        f.write("\n".join(out))
    print(f"Wrote {args.output} ({len(columns)} columns, {len(datasets)} datasets)")


if __name__ == "__main__":
    main()
