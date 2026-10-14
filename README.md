# MANTA

MANTA (MAtryoshka Nesting with Tolerance-conditioned Allocation) is a context-compression model with nested Matryoshka tokens and a tolerance-conditioned Elastic Allocator. The allocator predicts a per-chunk Matryoshka token count at inference time, so one checkpoint supports a continuous compression–quality trade-off.

The release contains the main LoRA recipe and its evaluation path. Experimental baselines, ablations, figures, and analysis scripts are kept out of this repository.

## Main model

- Matryoshka training exposes prefixes with `k ∈ {1, 2, 4, 8, 16, 32}`.
- The Elastic Allocator predicts per-chunk retention from compressed key/value states and the requested tolerance.
- The decoder is frozen; the encoder uses rank-128 LoRA (`q_proj`, `v_proj`, alpha 256).
- Stage learning rates are `1e-4` and `5e-5`.

## Installation

```bash
conda create -n manta python=3.10 -y
conda activate manta
pip install -r requirements.txt
```

Install a PyTorch build matching the CUDA runtime before installing the remaining requirements. The exact versions used for the reference run are listed in `requirements-freeze.txt`.

## Train

Prepare the raw-text and MRQA tensors expected by the two config files, then run the two-stage launcher:

```bash
DRYRUN=1 bash launch/train_manta.sh
bash launch/train_manta.sh
```

Set `MODEL_ID`, `EXP_TAG`, `EXP_ROOT`, or `CUDA_VISIBLE_DEVICES` to select a backbone, output directory, or GPU set. The launcher resumes from the newest step-tagged checkpoint and writes the final adapter to `<work_dir>/output/instruction_adapter.pt`.

## Evaluate

```bash
EXP_DIR=experiment/manta_1b_stage2 \
EVAL_PT=data/mrqa_eval_full/qa_eval.pt \
EVAL_META=data/mrqa_eval_full/qa_eval_meta.json \
TAG=manta_1b EXPECT_N=67854 \
bash launch/eval_manta.sh
```

The evaluator runs a fixed Matryoshka-token sweep and an adaptive tolerance sweep, checks sample counts, and writes SQuAD-normalized token-F1/EM reports. IID and OOD scores are macro-averages over the six datasets in each split.

## Layout

```
model/        MANTA model, LoRA layer, Matryoshka compression, Elastic Allocator
sft/          main two-stage trainer and MRQA data pipeline
pretrain/     raw-text data pipeline used by stage 1
configs/      reference stage configurations
launch/       training, evaluation, and config-rendering entry points
scripts/      fixed-k, adaptive, and metric aggregation evaluators
tests/        CPU tests for the allocator and data ordering
util/          distributed setup and training helpers
docs/         concise reproduction notes
```

## Tests

```bash
python -m pytest tests/test_k_predictor.py tests/test_dataloader_order.py -q
python tests/test_k_predictor_smoke.py
python -m compileall -q -f .
bash -n launch/train_manta.sh launch/eval_manta.sh
```

The tests run without a checkpoint or GPU. The optional end-to-end smoke test requires a local backbone and adapter; see `tests/test_manta_v5_cpu_e2e.py`.

## License

Released under the Apache License 2.0. See `LICENSE` and `NOTICE` for third-party attributions.
