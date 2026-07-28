"""Create per-(ckpt, length) eval work_dirs for the long-context evaluation.

For each source experiment (sac_8b_cross_chunk_unified, sac_8b_perceiver_io_unified)
and each target length (4096/8192/16384/32768), this script:

  1. Mkdir   experiment/<ckpt>_eval_long_<L>/output/
  2. Copy    config.json with `dataset` overridden to `long_<L>`
  3. Symlink compressor.pt to the source checkpoint

The SAC baseline (sac_8b_unified) uses sft/instruction_evaluator.py with a
different schema, so it is skipped here — see scripts/eval_sac_long_context.py
for the baseline driver.

CPU only.
"""

import argparse
import json
import os
import shutil

BASE = os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

SOURCE_EXPERIMENTS = [
    "sac_8b_cross_chunk_unified",
    "sac_8b_perceiver_io_unified",
]
TARGET_LENGTHS = [4096, 8192, 16384, 32768]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sources", nargs="+", default=SOURCE_EXPERIMENTS)
    p.add_argument("--lengths", type=int, nargs="+", default=TARGET_LENGTHS)
    p.add_argument("--dry_run", action="store_true")
    return p.parse_args()


def setup_one(source, length, dry_run):
    src_dir = os.path.join(BASE, "experiment", source)
    src_config = os.path.join(src_dir, "output", "config.json")
    src_compressor = os.path.join(src_dir, "output", "compressor.pt")
    if not os.path.isfile(src_config):
        print(f"[skip] missing {src_config}")
        return
    if not os.path.isfile(src_compressor):
        print(f"[skip] missing {src_compressor}")
        return

    dst_name = f"{source}_eval_long_{length}"
    dst_dir = os.path.join(BASE, "experiment", dst_name)
    dst_out = os.path.join(dst_dir, "output")

    with open(src_config, "r", encoding="utf-8") as f:
        config = json.load(f)
    new_config = {
        "compressor_config": dict(config["compressor_config"]),
        "training_config": dict(config["training_config"]),
    }
    new_config["training_config"]["dataset"] = f"long_{length}"

    print(f"[setup] {dst_name}  dataset=long_{length}")
    if dry_run:
        return

    os.makedirs(dst_out, exist_ok=True)
    with open(os.path.join(dst_out, "config.json"), "w", encoding="utf-8") as f:
        json.dump(new_config, f, indent=2)

    link_path = os.path.join(dst_out, "compressor.pt")
    if os.path.islink(link_path) or os.path.exists(link_path):
        os.remove(link_path)
    os.symlink(src_compressor, link_path)


def main():
    args = parse_args()
    for src in args.sources:
        for L in args.lengths:
            setup_one(src, L, args.dry_run)


if __name__ == "__main__":
    main()
