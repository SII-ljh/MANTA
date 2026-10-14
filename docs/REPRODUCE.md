# Reproduce MANTA

This guide covers the main training and evaluation path. Run commands from the repository root. Set `MANTA_ROOT` when data and experiment outputs live elsewhere.

```bash
export MANTA_ROOT=/path/to/MANTA
```

## Data

The stage-1 config expects tokenized raw-text examples. The stage-2 config expects tokenized MRQA training examples and an evaluation tensor with answer metadata. Put these files in the paths configured by `configs/manta_1b_stage1.json` and `configs/manta_1b_stage2.json`.

The data preparation modules are `pretrain/pre_prepare_data.py` and `sft/instruction_prepare_data.py`; both cache their tensors under the selected work directory.

## Training

The reference recipe uses two stages:

| Setting | Stage 1 | Stage 2 |
|---|---:|---:|
| steps | 20,000 | 20,000 |
| total batch size | 16 | 16 |
| learning rate | 1e-4 | 5e-5 |
| warmup steps | 1,000 | 1,000 |
| chunk size | 512 | 512 |
| Matryoshka tokens per chunk | 32 | 32 |
| ladder | 1, 2, 4, 8, 16, 32 | 1, 2, 4, 8, 16, 32 |

The encoder uses LoRA rank 128 on `q_proj` and `v_proj`; the decoder remains frozen. The launcher passes the stage-specific settings to `launch/render_config.py` and resumes from `instruction_adapter_step*.pt` when available.

```bash
DRYRUN=1 bash launch/train_manta.sh
bash launch/train_manta.sh
```

Keep gradient checkpointing disabled because compression requires the encoder KV cache. Preserve the dataloader order; `tests/test_dataloader_order.py` checks this invariant.

## Evaluation

```bash
EXP_DIR=experiment/manta_1b_stage2 \
EVAL_PT=data/mrqa_eval_full/qa_eval.pt \
EVAL_META=data/mrqa_eval_full/qa_eval_meta.json \
TAG=manta_1b EXPECT_N=67854 \
bash launch/eval_manta.sh
```

The script runs:

1. fixed-k evaluation with `scripts/eval_k_sweep.py`;
2. tolerance-conditioned allocation with `scripts/eval_v5_pareto.py`;
3. sample-count checks with `scripts/check_eval_n.py`;
4. token-F1/EM aggregation with `scripts/aggregate_tokenf1.py`.

Set `EXPECT_N` to the expected number of examples. The evaluator stores rank outputs and reuses them only when the manifest matches the current adapter, data, sweep grids, and sample count.

Metrics use SQuAD normalization, maximum over references, and macro averages over the six IID and six OOD datasets. The generated reports are diagnostic artifacts; no baseline or ablation code is required to run the main model.

## Tests

```bash
python -m pytest tests/test_k_predictor.py tests/test_dataloader_order.py -q
python tests/test_k_predictor_smoke.py
python -m compileall -q -f .
bash -n launch/train_manta.sh launch/eval_manta.sh
```

For the optional end-to-end smoke test:

```bash
MANTA_MODEL_ID=/path/to/backbone \
MANTA_SAC_ADAPTER=/path/to/adapter.pt \
python tests/test_manta_v5_cpu_e2e.py
```
