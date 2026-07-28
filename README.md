# MANTA: Adaptive Context Compression via Matryoshka Anchors and Tolerance-Conditioned Allocation

Reference implementation, training recipes, and evaluation harness for the paper
*MANTA: Adaptive Context Compression via Matryoshka Anchors and
Tolerance-Conditioned Allocation* (under review).

Context compression lets a language model read long inputs cheaply by encoding
the context into a compact, reusable compressed representation. Existing methods
hand every context of a given length the same amount of memory, even though the
information they carry differs enormously. MANTA makes compression
*information-adaptive*: every part of the context receives its minimal
sufficient memory — the least memory at which its loss stays within a single
global tolerance of keeping everything — so how much each part keeps is not
imposed in advance but emerges from the text itself.

Two components realize this:

- **Matryoshka anchor representation.** The compressed key–value representation
  is nested: every prefix of a context's memory is itself a valid compressed
  representation. A single trained model therefore operates across a wide range
  of compression ratios instead of one fixed ratio.
- **Elastic allocator.** A lightweight head (0.42M parameters) decides how much
  memory each span keeps from its content and from the tolerance requested at
  inference time. The tolerance is one continuous knob: turning it moves the
  whole model along its compression–quality curve, with no retraining.

## Results

Aggregate MRQA accuracy (token-F1 / EM, full evaluation pool, *n* = 67,854).
Every MANTA row is **the same single model**, operated by choosing a tolerance;
each baseline row is a separate model trained for that one ratio. The MANTA rows
realize a *strictly higher* compression ratio than the baselines they are
compared against.

| Method | Ratio | ID F1 | ID EM | OOD F1 | OOD EM |
|---|---|---|---|---|---|
| No compression (fine-tuned reference) | 1× | 81.73 | 72.01 | 66.59 | 56.21 |
| ICAE | 64× | 45.70 | 34.85 | 34.26 | 25.78 |
| 500xCompressor | 64× | 51.04 | 40.25 | 37.02 | 28.94 |
| SAC | 64× | 52.01 | 40.93 | 40.10 | 30.59 |
| **MANTA** (ε=0.05) | **66×** | **60.47** | **49.39** | **43.44** | **34.30** |
| ICAE | 85× | 43.98 | 33.15 | 33.11 | 24.25 |
| 500xCompressor | 85× | 50.37 | 39.45 | 36.93 | 28.24 |
| SAC | 85× | 50.65 | 39.94 | 37.68 | 28.60 |
| **MANTA** (ε=0.1) | **87×** | **58.73** | **47.87** | **42.06** | **33.26** |
| ICAE | 128× | 42.70 | 32.14 | 32.66 | 23.80 |
| 500xCompressor | 128× | 49.46 | 38.82 | 36.08 | 27.30 |
| SAC | 128× | 48.50 | 38.13 | 36.82 | 28.09 |
| **MANTA** (ε=0.2) | **132×** | **55.88** | **45.43** | **39.87** | **31.29** |
| ICAE | 171× | 41.76 | 30.79 | 32.25 | 22.98 |
| 500xCompressor | 171× | 47.97 | 37.52 | 34.73 | 26.12 |
| SAC | 171× | 48.26 | 37.93 | 36.39 | 27.70 |
| **MANTA** (ε=0.34) | **181×** | **53.90** | **43.63** | **38.75** | **30.29** |

Backbone: Llama-3.2-1B. ID / OOD are macro-averages over the six in-domain and
six out-of-domain MRQA datasets (equal weight per dataset). Scale robustness
(1B / 3B / 8B), a Qwen2.5-1.5B backbone, long-context needle-QA, and the
ablations are reported in the paper and reproducible with the scripts here.

## Repository layout

```
model/                 encoder–decoder compression model
  modeling.py            Matryoshka anchors, compressed-KV assembly, elastic
                         allocator (class KPredictor), inference paths
  lora.py                LoRA wrapper used by the adapter-harness variant
sft/                   stage-2 instruction tuning
  matryoshka_trainer.py  main trainer: all-k objective, tolerance sampling,
                         oracle labeling, DDP, step-tagged checkpointing
  instruction_*.py       data preparation / dataloader / evaluator
pretrain/              stage-1 training on raw long text
  pre_*.py               trainer, dataloader, evaluator, data preparation
  chunk_mae/             chunk-level MAE pretraining of the compressor
util/                  metrics and report helpers (token-F1 / EM, splits)
scripts/               data preparation, evaluation, analysis, tables, figures
                       (see docs/SCRIPTS.md for the index)
launch/                portable entry points
  train_manta.sh         two-stage training with resume
  eval_manta.sh          fixed-k sweep + adaptive sweep + scoring
  render_config.py       fills a configs/ template into a run directory
configs/               training configurations (paper hyperparameters)
tests/                 CPU unit tests for the ladder, allocator, dataloader
docs/                  reproduction, data, baselines, script index
```

A note on naming: the method is **MANTA**; the implementation, checkpoint
directories, and some identifiers keep the project's internal code name
**PACT** (for example `pact_v5_*` experiment directories and the
`use_k_predictor` / `KPredictor` identifiers for the elastic allocator). The two
names refer to the same system.

## Installation

```bash
conda create -n manta python=3.10 -y
conda activate manta
# install the torch build matching your CUDA runtime first, then:
pip install -r requirements.txt
```

The pinned versions used for every number in the paper are `torch==2.2.0`,
`transformers==4.43.1`, `tokenizers==0.19.1`; `requirements-freeze.txt` is the
exact environment snapshot. Training the 1B model needs 8 GPUs with ≥40GB each
(≈16h wall clock for both stages); evaluation runs on any number of GPUs and
shards across them.

## Quick start

```bash
# 0. prepare data (see docs/DATA.md — this is the step that needs the corpora)

# 1. train: stage 1 on raw long text, stage 2 on MRQA
DRYRUN=1 bash launch/train_manta.sh          # 50+50 step smoke test first
bash launch/train_manta.sh                   # full run

# 2. evaluate the same checkpoint in both modes
EXP_DIR=experiment/manta_1b_stage2 \
EVAL_PT=data/mrqa_eval_full/qa_eval.pt \
EVAL_META=data/mrqa_eval_full/qa_eval_meta.json \
TAG=manta_1b EXPECT_N=67854 \
bash launch/eval_manta.sh
```

`docs/REPRODUCE.md` walks through the full pipeline, the ablation arms, the
long-context evaluation, and which script produces each table and figure of the
paper.

## Checkpoints

Trained checkpoints are not distributed in this repository. Each run writes
`<work_dir>/output/instruction_adapter.pt` (the trained encoder plus the
allocator; the decoder is frozen and comes from the public backbone) together
with the `config.json` needed to load it. Checkpoint release is planned
alongside publication.

## License and third-party code

This repository is released under the Apache License 2.0 (see `LICENSE`). It
builds on the public SAC codebase and evaluates against the ICAE and
500xCompressor baselines; backbones (Llama 3.x, Qwen2.5) and datasets (MRQA,
PG-19, SlimPajama) remain under their own licenses. See `NOTICE` for
attributions and `docs/BASELINES.md` for how the baselines were trained.

## Citation

```bibtex
@inproceedings{manta2026,
  title     = {MANTA: Adaptive Context Compression via Matryoshka Anchors and
               Tolerance-Conditioned Allocation},
  author    = {Anonymous},
  year      = {2026},
  note      = {Under review. Author list and venue will be filled in on publication.}
}
```
