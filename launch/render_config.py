#!/usr/bin/env python3
"""Render a config template from ``configs/`` into a training work_dir.

The trainer (``sft/matryoshka_trainer.py``) reads ``<work_dir>/config.json``.
This script takes one of the templates in ``configs/``, applies the settings
that depend on the machine (how many GPUs) or on the run (step count,
initialization checkpoint, backbone), and writes the result.

Example
-------
    python launch/render_config.py \
        --template configs/manta_1b_stage1.json \
        --work_dir experiment/manta_1b_stage1 \
        --model_id meta-llama/Llama-3.2-1B \
        --device_count 8 --total_steps 20000 --init SCRATCH

Arbitrary fields can be overridden with dotted keys, which is how the
ablation arms are produced, e.g.::

    --set matryoshka_training_config.use_k_predictor=false
    --set matryoshka_training_config.eps_grid='[0.05]'
"""
import argparse
import json
import os


def set_dotted(cfg, dotted, raw):
    keys = dotted.split(".")
    node = cfg
    for k in keys[:-1]:
        if k not in node or not isinstance(node[k], dict):
            raise SystemExit("[render_config] no such config section: %s" % k)
        node = node[k]
    if keys[-1] not in node:
        print("[render_config] note: adding new key %s" % dotted)
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        value = raw  # plain string, e.g. a path
    node[keys[-1]] = value


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--template", required=True, help="a JSON file from configs/")
    p.add_argument("--work_dir", required=True,
                   help="experiment directory; config.json is written here")
    p.add_argument("--model_id", default=None,
                   help="HF id or local path of the backbone (both stages)")
    p.add_argument("--device_count", type=int, default=None,
                   help="number of GPUs; gradient accumulation is derived from it")
    p.add_argument("--total_steps", type=int, default=None)
    p.add_argument("--init", default=None,
                   help="init_adapter_path: 'SCRATCH' or a checkpoint .pt")
    p.add_argument("--save_step", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--lr_warmup_steps", type=int, default=None)
    p.add_argument("--ckpt_step_offset", type=int, default=None,
                   help="steps already completed, for resumed runs")
    p.add_argument("--set", action="append", default=[], metavar="DOTTED=VALUE",
                   help="override any config field (repeatable)")
    args = p.parse_args()

    with open(args.template, encoding="utf-8") as f:
        cfg = json.load(f)

    train_sections = ["pretrain_training_config", "sft_training_config"]

    if args.model_id:
        for s in train_sections:
            cfg[s]["model_id"] = args.model_id
        cfg["data_config"]["model_id"] = args.model_id

    for flag, section, key in (
        (args.total_steps, "matryoshka_training_config", "total_steps"),
        (args.init, "matryoshka_training_config", "init_adapter_path"),
        (args.ckpt_step_offset, "matryoshka_training_config", "ckpt_step_offset"),
    ):
        if flag is not None:
            cfg[section][key] = flag

    for flag, key in ((args.save_step, "save_step"),
                      (args.lr, "learning_rate"),
                      (args.lr_warmup_steps, "lr_warmup_steps")):
        if flag is not None:
            for s in train_sections:
                cfg[s][key] = flag

    overridden = set()
    for item in args.set:
        if "=" not in item:
            raise SystemExit("[render_config] --set expects DOTTED=VALUE, got %r"
                             % item)
        dotted, raw = item.split("=", 1)
        set_dotted(cfg, dotted, raw)
        overridden.add(dotted)

    # Gradient accumulation is derived last, from the final batch-size fields, so
    # that a --set on total_batch_size / batch_size_per_device cannot leave a
    # stale accumulation value behind (that would silently change the effective
    # batch size and the number of optimizer steps).
    if args.device_count is not None:
        if args.device_count < 1:
            raise SystemExit("[render_config] --device_count must be >= 1")
        for s in train_sections:
            if "%s.gradient_accumulation_steps" % s in overridden:
                print("[render_config] note: %s.gradient_accumulation_steps set "
                      "explicitly, not derived" % s)
                cfg[s]["device_count"] = args.device_count
                continue
            total = cfg[s]["total_batch_size"]
            per_dev = cfg[s].get("batch_size_per_device", 1)
            if total % (args.device_count * per_dev) != 0:
                raise SystemExit(
                    "[render_config] total_batch_size %d is not divisible by "
                    "device_count %d x batch_size_per_device %d"
                    % (total, args.device_count, per_dev))
            cfg[s]["device_count"] = args.device_count
            cfg[s]["gradient_accumulation_steps"] = \
                total // (args.device_count * per_dev)

    os.makedirs(os.path.join(args.work_dir, "output"), exist_ok=True)
    out = os.path.join(args.work_dir, "config.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2)

    m = cfg["matryoshka_training_config"]
    sec = ("pretrain_training_config" if m["data_mode"] == "pretrain"
           else "sft_training_config")
    print("[render_config] wrote %s" % out)
    print("[render_config]   stage=%s data_mode=%s steps=%s lr=%s "
          "devices=%s accum=%s init=%s"
          % (m["stage"], m["data_mode"], m["total_steps"],
             cfg[sec]["learning_rate"], cfg[sec]["device_count"],
             cfg[sec]["gradient_accumulation_steps"], m["init_adapter_path"]))


if __name__ == "__main__":
    main()
