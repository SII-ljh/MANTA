# Data

Datasets are not redistributed here. This document states exactly what each
stage consumes, in what format, and which script produces it, so the pipeline
can be rebuilt from the public sources.

Expected layout (none of it is tracked in git):

```
data/
  raw/                      corpora you stage yourself (plain text / JSONL)
  mrqa_eval_full/           qa_eval.pt + qa_eval_meta.json   (the 67,854 pool)
  mrqa_eval_subset/         a small subset used by some analysis scripts
  long_context_eval/        synthesized needle-QA sets
pretrain/output/            stage-1 tokenized data and the long_text.json cache
sft/output/                 stage-2 tokenized data
experiment/                 run directories: config, checkpoints, eval JSONs
reports/                    aggregated metric JSONs
```

## Stage 1 — raw long text

`data_config.dataset_repo` is handed to
`datasets.load_dataset(repo, split="train", streaming=True)`, and the `text`
field of each example is collected. The paper uses PG-19 for the 1B, 3B and Qwen
models, and SlimPajama-6B (`DKYoon/SlimPajama-6B`) for the earlier 8B runs.

The traversal is cached, which also gives you a way to bypass the HF path
entirely: write a JSON list of raw strings to `pretrain/output/long_text.json`
and the `dataset_repo` name is never used. That is what `pg19_local` means in the
shipped configs — a locally staged PG-19 dump. `scripts/prepare_pretrain_data.py`
builds the same cache from a plain text file (`data/raw/pretrain_train.txt` by
default).

Tokenization (done once, on first use, by the trainer) writes
`pretrain/output/train_<backbone>_<N>samples_<min>-<max>len.pt` and its eval
counterpart. Settings: 320,000 samples, `min_len` 510, `max_len` 2040 tokens,
span size 512.

## Stage 2 — MRQA instruction data

`data_config.instruction_dataset_repo: "mrqa"` refers to the HF `mrqa` dataset.
`sft/instruction_prepare_data.py` caches the raw traversal as
`sft/output/mrqa_<split>_instruction_dataset.json` and writes

```
sft/output/mrqa_train_<backbone>_<N>samples_instruction.pt
sft/output/mrqa_eval_<backbone>_<N>samples_instruction.pt
```

with 320,000 training samples and contexts truncated to `max_context_tokens`
6,000. The eval `.pt` is in the same format the evaluators expect and can be
passed to `launch/eval_manta.sh` as `EVAL_PT`.

Sample order must be preserved end to end — see the dataloader-order note in
`docs/REPRODUCE.md`.

## Evaluation pool

Full MRQA is the complete evaluation pool, *n* = 67,854, formed from the test and
validation splits:

| in-domain | *n* | out-of-domain | *n* |
|---|---|---|---|
| SQuAD | 10,507 | BioASQ | 1,504 |
| NewsQA | 4,212 | DROP | 1,503 |
| TriviaQA-web | 7,785 | DuoRC (ParaphraseRC) | 1,501 |
| SearchQA | 16,980 | RACE | 674 |
| HotpotQA | 5,901 | RelationExtraction | 2,948 |
| NaturalQuestions-Short | 12,836 | TextbookQA | 1,503 |
| **total** | **58,221** | **total** | **9,633** |

Evaluation needs two index-aligned files:

- `qa_eval.pt` — tokenized examples (a list; entry *i* pairs with metadata *i*).
- `qa_eval_meta.json` — a JSON list whose entry *i* carries at least `id`,
  `answers` (list of reference strings) and `dataset` (used for the per-dataset
  and macro ID/OOD splits).

`scripts/prepare_qa_data.py --output_dir data/mrqa_eval_full` writes both from
JSONL dumps of the pool. Reported metrics are SQuAD-normalized token-F1 and EM,
max over references; ID and OOD figures are macro-averages over the six datasets
in each group.

## Long-context needle-QA

```bash
python scripts/build_long_context_eval.py     # synthesize 4K/8K/16K/32K, 500 each
python scripts/prepare_longctx_eval.py        # tokenize into the eval format
python scripts/setup_long_context_eval.py     # one work_dir per (checkpoint, length)
```

Decoding for these sets uses 64 new tokens instead of 128.

## Optional: pre-computed KV caches

`scripts/precompute_kv.py` (and the batched `scripts/precompute_kv_fast.py`)
pre-compute per-span compressed KV caches for a QA set. They are not needed for
the main pipeline; they speed up repeated analyses and are how the baseline
harnesses were fed.
