# Baselines

Every compression baseline in the paper was trained by us on the same data, the
same step count and the same backbone as MANTA, so the comparison is
compression-matched rather than copied from published numbers. This document
gives the harness and settings for each.

`baselines/` contains our own implementations (HF `transformers` + `peft` only,
no vendored third-party code). Upstream references are listed in `NOTICE`.

## SAC (dedicated per-ratio models)

SAC trains with the vanilla two-stage path already in this repository — the same
trainers, with no Matryoshka ladder and no allocator. Write a config with only
the `pretrain_*` / `sft_*` sections (i.e. drop `matryoshka_training_config`) and:

```bash
cd pretrain && python pre_trainer.py         --work_dir <run_dir> --port 14941
cd ../sft    && python instruction_trainer.py --work_dir <run_dir> --port 14942
```

The compression ratio is set by the two task-config fields, under the constraint
`mem_size × compress_ratio == chunk_size`:

| Nominal ratio | `mem_size` | `compress_ratio` | `chunk_size` |
|---|---|---|---|
| 64× | 8 | 64 | 512 |
| 85× | 6 | 85 | 510 |
| 128× | 4 | 128 | 512 |
| 171× | 3 | 170 | 510 |

The 85× and 171× tiers use a 510-token span because the product must match the
span exactly; a mismatched `mem_size`/span pair fails at the anchor construction
step rather than silently training something else.

Other settings match MANTA: 20,000 steps per stage, total batch 16, lr 1e-4
(stage 1) and 5e-5 (stage 2), 320,000 samples, MRQA for stage 2. If you change
the total batch size, scale the learning rate by `sqrt(batch/16)`, which is what
the reported runs did.

## ICAE

Per-span ICAE with a LoRA-adapted frozen backbone, trained in two stages
(autoencoding, then QA):

```bash
python baselines/icae/train_icae.py --stage ae \
    --model_path <backbone> --train_data <regen data> \
    --output_dir <run>/ae --ratio 64 --chunk_size 512 \
    --lora_rank 128 --lora_alpha 32 --learning_rate 1e-4 --warmup_steps 300

python baselines/icae/train_icae.py --stage qa \
    --model_path <backbone> --train_data <mrqa data> \
    --init_trainable <run>/ae/trainable.pt --output_dir <run>/qa --ratio 64

python baselines/icae/eval_icae_mrqa_bridge.py \
    --model_path <backbone> --trainable_path <run>/qa/trainable.pt \
    --eval_data_path data/mrqa_eval_full/qa_eval.pt \
    --eval_meta_path data/mrqa_eval_full/qa_eval_meta.json \
    --ratio 64 --output_dir <run>/qa --output_name icae_r64
```

`--ratio` selects the tier (64 / 85 / 128 / 171); memory slots per span follow
from `chunk_size / ratio`. The eval bridge writes rank JSONs in the same format
as our own evaluators, so `scripts/rescore_tokenf1_from_generated.py` scores
baseline and MANTA generations with one scorer.

## 500xCompressor

Same shape, with the number of memory slots given directly:

```bash
python baselines/c500x/train_c500x.py --stage regen \
    --llama_path <backbone> --data_path <regen data> --output_dir <run>/regen \
    --num_mem 8 --chunk_size 512 --lr 1e-4 --bf16

python baselines/c500x/train_c500x.py --stage qa \
    --llama_path <backbone> --data_path <mrqa data> \
    --init_trainable <run>/regen/trainable.pt --output_dir <run>/qa --num_mem 8

python baselines/c500x/eval_c500x_mrqa_bridge.py --help
```

`--num_mem` 8 / 6 / 4 / 3 corresponds to the 64× / 85× / 128× / 171× tiers.

## No-compression reference

The uncompressed row is a dense full-parameter SFT of the same backbone on the
same MRQA data:

```bash
python baselines/dense/fullft_sft.py --model <backbone> \
    --train_pt sft/output/mrqa_train_<backbone>_320000samples_instruction.pt \
    --out_dir <run> --steps 20000 --lr 1e-5 --warmup 200 --max_len 6200

python baselines/dense/fullft_generate.py --model_dir <run>/final \
    --eval_pt data/mrqa_eval_full/qa_eval.pt \
    --meta data/mrqa_eval_full/qa_eval_meta.json \
    --out_dir <run> --output_name dense_full58k
```

Use `--lr 1e-5`. The reference rows in the paper were retrained at this learning
rate; 5e-5 (the value that suits the compression models) leaves the dense model
undertrained and understates the uncompressed ceiling by 11–18 token-F1.

## LongLLMLingua

Token-level prompt compression, evaluation only, no training:

```bash
pip install llmlingua
python scripts/eval_longllmlingua.py --target_token 250 --output_name lingua_16x
```

## Fairness notes

- Data, step count, batch size and backbone are matched across all methods.
- Baselines are dedicated per-ratio models: four trained checkpoints cover the
  four tiers, whereas the MANTA rows are one checkpoint operated at four
  tolerances.
- Nominal ratios are used for grouping, as is conventional. Realized ratios
  differ per method; `scripts/measured_realized_64x.py` and
  `scripts/calc_real_compression.py` compute them, and the MANTA operating
  points realize strictly higher compression than the baselines they are
  compared against.
- Where a baseline run was visibly under-optimized, it was retrained and the
  better result reported.
