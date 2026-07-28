"""Evaluate inference results along two orthogonal axes: length bucket and dataset.

Replaces the older util/evaluate_by_length.py (length-only) and
util/evaluate_extended.py (dataset-group-only) reports. Produces three views:

    A. Per length bucket   (6 rows, n / F1 / EM / BLEU4)
    B. Per dataset subset  (~35 rows, n / F1 / EM / BLEU4)
    C. Length x Dataset matrix of F1 (cells with n < MIN_CELL_N are blanked)

Outputs:
    output/eval_by_length_and_dataset.json    structured JSON
    output/eval_report.md                     human-readable Markdown report
    stdout                                    pretty tables

Inputs:
    {work_dir}/output/instruction_inference_results.json
        Per-sample list with fields: sample_idx, subset, rouge-f1, exact_match, bleu4
    sft/output/unified_test_instruction_dataset.json
        Per-sample list with fields: subset, length_bucket, token_length

Join key:
    sample_idx -> index into the test dataset.

Usage:
    python util/evaluate_by_length_and_dataset.py --work_dir experiment/sac_8b_cross_chunk_v4a2_4gpu_inst
"""

import argparse
import json
import os
import sys
from collections import defaultdict

import numpy as np

BASE = os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_DATA_PATH = os.path.join(BASE, "sft", "output", "unified_test_instruction_dataset.json")

LENGTH_BUCKET_ORDER = [
    "short_0-512",
    "medium_512-1k",
    "medium_1k-2k",
    "long_2k-4k",
    "long_4k-8k",
    "very_long_8k+",
]

# Pretty-print labels for length buckets (kept short for table widths)
LENGTH_BUCKET_LABEL = {
    "short_0-512": "0-512",
    "medium_512-1k": "512-1k",
    "medium_1k-2k": "1k-2k",
    "long_2k-4k": "2k-4k",
    "long_4k-8k": "4k-8k",
    "very_long_8k+": "8k+",
}

# Minimum cell sample count for the length x dataset matrix; below this F1 is too noisy
MIN_CELL_N = 10

METRIC_KEYS = ["rouge-f1", "exact_match", "bleu4"]


def load_inputs(work_dir, test_data_path):
    results_path = os.path.join(work_dir, "output", "instruction_inference_results.json")
    with open(results_path, "r") as f:
        results = json.load(f)
    with open(test_data_path, "r") as f:
        test_data = json.load(f)
    return results, test_data


def join_by_sample_idx(results, test_data):
    """Return per-sample rows enriched with `subset` and `length_bucket`.

    Two input formats are supported:
      (a) cross_chunk evaluator output: each item has `sample_idx`; we look up
          subset / length_bucket from the parallel test_data list.
      (b) sft instruction_evaluator output: each item already carries `subset`
          and `length_bucket` directly; sample_idx is absent.
    """
    joined = []
    missing = 0
    for res in results:
        idx = res.get("sample_idx")
        if idx is not None and idx < len(test_data):
            td = test_data[idx]
            subset = td.get("subset", res.get("subset", "unknown"))
            length_bucket = td.get("length_bucket", res.get("length_bucket", "unknown"))
        elif "length_bucket" in res:
            subset = res.get("subset", "unknown")
            length_bucket = res.get("length_bucket", "unknown")
        else:
            missing += 1
            continue
        joined.append(
            {
                "subset": subset,
                "length_bucket": length_bucket,
                "rouge-f1": res.get("rouge-f1", 0.0),
                "exact_match": res.get("exact_match", 0.0),
                "bleu4": res.get("bleu4", 0.0),
            }
        )
    if missing:
        print(f"WARNING: dropped {missing}/{len(results)} samples (no sample_idx and no length_bucket)")
    return joined


def aggregate(rows, key):
    """Group `rows` by `key` and return {value: {n, f1, em, bleu4}}."""
    buckets = defaultdict(lambda: {m: [] for m in METRIC_KEYS})
    for r in rows:
        buckets[r[key]]["rouge-f1"].append(r["rouge-f1"])
        buckets[r[key]]["exact_match"].append(r["exact_match"])
        buckets[r[key]]["bleu4"].append(r["bleu4"])
    out = {}
    for k, m in buckets.items():
        n = len(m["rouge-f1"])
        out[k] = {
            "n": n,
            "f1": float(np.mean(m["rouge-f1"])),
            "em": float(np.mean(m["exact_match"])),
            "bleu4": float(np.mean(m["bleu4"])),
        }
    return out


def aggregate_cross(rows):
    """Return {(length_bucket, subset): {n, f1, em, bleu4}}."""
    cells = defaultdict(lambda: {m: [] for m in METRIC_KEYS})
    for r in rows:
        cells[(r["length_bucket"], r["subset"])]["rouge-f1"].append(r["rouge-f1"])
        cells[(r["length_bucket"], r["subset"])]["exact_match"].append(r["exact_match"])
        cells[(r["length_bucket"], r["subset"])]["bleu4"].append(r["bleu4"])
    out = {}
    for key, m in cells.items():
        n = len(m["rouge-f1"])
        out[key] = {
            "n": n,
            "f1": float(np.mean(m["rouge-f1"])),
            "em": float(np.mean(m["exact_match"])),
            "bleu4": float(np.mean(m["bleu4"])),
        }
    return out


def overall(rows):
    return {
        "n": len(rows),
        "f1": float(np.mean([r["rouge-f1"] for r in rows])),
        "em": float(np.mean([r["exact_match"] for r in rows])),
        "bleu4": float(np.mean([r["bleu4"] for r in rows])),
    }


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------
def print_table(title, header, rows):
    print()
    print(title)
    print(header)
    sep = "-" * len(header)
    print(sep)
    for r in rows:
        print(r)
    print(sep)


def fmt_row(label, n, f1, em, bleu4):
    return f"{label:<22} {n:>7} {f1*100:>8.2f} {em*100:>8.2f} {bleu4:>8.4f}"


def header_row():
    return f"{'bucket':<22} {'n':>7} {'F1(%)':>8} {'EM(%)':>8} {'BLEU4':>8}"


def render_text_report(by_length, by_dataset, ov):
    lines = []
    lines.append("=== A. Per length bucket ===")
    lines.append(header_row())
    lines.append("-" * len(header_row()))
    for b in LENGTH_BUCKET_ORDER:
        if b in by_length:
            m = by_length[b]
            lines.append(fmt_row(b, m["n"], m["f1"], m["em"], m["bleu4"]))
    # any unknown bucket
    for b in by_length:
        if b not in LENGTH_BUCKET_ORDER:
            m = by_length[b]
            lines.append(fmt_row(b, m["n"], m["f1"], m["em"], m["bleu4"]))
    lines.append("-" * len(header_row()))
    lines.append(fmt_row("TOTAL", ov["n"], ov["f1"], ov["em"], ov["bleu4"]))

    lines.append("")
    lines.append("=== B. Per dataset (sorted by n desc) ===")
    lines.append(header_row())
    lines.append("-" * len(header_row()))
    sorted_ds = sorted(by_dataset.items(), key=lambda kv: -kv[1]["n"])
    for ds, m in sorted_ds:
        lines.append(fmt_row(ds, m["n"], m["f1"], m["em"], m["bleu4"]))
    lines.append("-" * len(header_row()))
    lines.append(fmt_row("TOTAL", ov["n"], ov["f1"], ov["em"], ov["bleu4"]))
    return "\n".join(lines)


def render_cross_matrix(cross, by_dataset):
    """Format length-bucket x dataset F1 matrix as a fixed-width table."""
    datasets = sorted(by_dataset.keys(), key=lambda d: -by_dataset[d]["n"])
    col_w = 8
    label_w = max(22, max(len(d) for d in datasets) + 2)
    head = f"{'dataset':<{label_w}}" + "".join(
        f"{LENGTH_BUCKET_LABEL[b]:>{col_w}}" for b in LENGTH_BUCKET_ORDER
    ) + f"{'all':>{col_w}}"
    sep = "-" * len(head)
    title = f"=== C. F1(%) matrix: dataset x length bucket (cells with n<{MIN_CELL_N} blanked) ==="
    out = [title, head, sep]
    for ds in datasets:
        row = f"{ds:<{label_w}}"
        for b in LENGTH_BUCKET_ORDER:
            cell = cross.get((b, ds))
            if cell and cell["n"] >= MIN_CELL_N:
                row += f"{cell['f1']*100:>{col_w}.2f}"
            else:
                row += f"{'-':>{col_w}}"
        m = by_dataset[ds]
        row += f"{m['f1']*100:>{col_w}.2f}"
        out.append(row)
    out.append(sep)
    return "\n".join(out)


def render_markdown(work_dir, by_length, by_dataset, cross, ov):
    name = os.path.basename(os.path.normpath(work_dir))
    md = [f"# Eval Report: `{name}`", ""]
    md.append(f"- Total samples: **{ov['n']:,}**")
    md.append(f"- Overall F1: **{ov['f1']*100:.2f}%**")
    md.append(f"- Overall EM: **{ov['em']*100:.2f}%**")
    md.append(f"- Overall BLEU-4: **{ov['bleu4']:.4f}**")
    md.append("")

    md.append("## A. Per length bucket")
    md.append("")
    md.append("| Length bucket | n | F1 (%) | EM (%) | BLEU-4 |")
    md.append("|---|---:|---:|---:|---:|")
    for b in LENGTH_BUCKET_ORDER:
        if b in by_length:
            m = by_length[b]
            md.append(f"| {b} | {m['n']:,} | {m['f1']*100:.2f} | {m['em']*100:.2f} | {m['bleu4']:.4f} |")
    md.append(f"| **TOTAL** | **{ov['n']:,}** | **{ov['f1']*100:.2f}** | **{ov['em']*100:.2f}** | **{ov['bleu4']:.4f}** |")
    md.append("")

    md.append("## B. Per dataset")
    md.append("")
    md.append("| Dataset | n | F1 (%) | EM (%) | BLEU-4 |")
    md.append("|---|---:|---:|---:|---:|")
    for ds, m in sorted(by_dataset.items(), key=lambda kv: -kv[1]["n"]):
        md.append(f"| {ds} | {m['n']:,} | {m['f1']*100:.2f} | {m['em']*100:.2f} | {m['bleu4']:.4f} |")
    md.append(f"| **TOTAL** | **{ov['n']:,}** | **{ov['f1']*100:.2f}** | **{ov['em']*100:.2f}** | **{ov['bleu4']:.4f}** |")
    md.append("")

    md.append(f"## C. F1 (%) matrix: dataset × length bucket  (cells with n<{MIN_CELL_N} blanked)")
    md.append("")
    cols = " | ".join(LENGTH_BUCKET_LABEL[b] for b in LENGTH_BUCKET_ORDER)
    md.append(f"| Dataset | {cols} | all |")
    md.append("|---|" + "---:|" * (len(LENGTH_BUCKET_ORDER) + 1))
    for ds in sorted(by_dataset.keys(), key=lambda d: -by_dataset[d]["n"]):
        cells = []
        for b in LENGTH_BUCKET_ORDER:
            cell = cross.get((b, ds))
            cells.append(f"{cell['f1']*100:.2f}" if cell and cell["n"] >= MIN_CELL_N else "—")
        m = by_dataset[ds]
        md.append(f"| {ds} | " + " | ".join(cells) + f" | {m['f1']*100:.2f} |")
    md.append("")
    return "\n".join(md)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--work_dir", type=str, required=True)
    parser.add_argument(
        "--test_data",
        type=str,
        default=TEST_DATA_PATH,
        help="Path to unified test dataset (provides length_bucket per sample_idx)",
    )
    args = parser.parse_args()
    sys.setrecursionlimit(10000)

    output_dir = os.path.join(args.work_dir, "output")
    json_path = os.path.join(output_dir, "eval_by_length_and_dataset.json")
    md_path = os.path.join(output_dir, "eval_report.md")

    results, test_data = load_inputs(args.work_dir, args.test_data)
    rows = join_by_sample_idx(results, test_data)

    by_length = aggregate(rows, "length_bucket")
    by_dataset = aggregate(rows, "subset")
    cross = aggregate_cross(rows)
    ov = overall(rows)

    print(render_text_report(by_length, by_dataset, ov))
    print()
    print(render_cross_matrix(cross, by_dataset))

    cross_json = {f"{b}__{d}": v for (b, d), v in cross.items()}
    payload = {
        "work_dir": args.work_dir,
        "overall": ov,
        "by_length_bucket": by_length,
        "by_dataset": by_dataset,
        "by_length_x_dataset": cross_json,
        "schema": {
            "length_bucket_order": LENGTH_BUCKET_ORDER,
            "min_cell_n_for_matrix": MIN_CELL_N,
        },
    }
    with open(json_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nJSON saved to {json_path}")

    md = render_markdown(args.work_dir, by_length, by_dataset, cross, ov)
    with open(md_path, "w") as f:
        f.write(md)
    print(f"Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
