# Reproducing the paper

Everything below assumes the repository root as working directory and the
environment from the README. Absolute paths are never baked in: scripts resolve
the repository root from their own location, and `MANTA_ROOT` overrides it if you
keep data or experiments elsewhere.

```bash
export MANTA_ROOT=/path/to/this/repo        # optional
export MANTA_MODEL_ID=meta-llama/Llama-3.1-8B   # optional, backbone for analysis scripts
```

## 1. Data

See `docs/DATA.md`. You need three things: a raw long-text corpus for stage 1,
the MRQA training split tokenized for stage 2, and the MRQA evaluation pool with
its answer metadata. The long-context and analysis sections need two more small
sets, both built by scripts in this repository.

## 2. Training

The main line trains one model in two stages. The encoder (compressor) is fully
fine-tuned; the decoder LLM stays frozen throughout; the elastic allocator is
trained jointly.

```bash
DRYRUN=1 bash launch/train_manta.sh    # 50 + 50 steps, verifies the whole path
bash launch/train_manta.sh             # the real run
```

Hyperparameters, as used for every 1B number in the paper (`configs/manta_1b_stage1.json`,
`configs/manta_1b_stage2.json`):

| | Stage 1 (raw text) | Stage 2 (MRQA) |
|---|---|---|
| steps | 20,000 | 20,000 |
| total batch size | 16 (1 per device × 8 GPUs × 2 accum) | same |
| learning rate | 2e-5 | 1e-5 |
| lr warmup | 1,000 | 1,000 |
| gradient clip | 2.0 | 2.0 |
| span (chunk) size | 512 | 512 |
| memory slots per span | 32 | 32 |
| Matryoshka ladder | k ∈ {1,2,4,8,16,32}, all-k per step | same |
| ladder weights | [2,1,1,1,1,2] | same |
| tolerance grid | ε ∈ {0.001,0.01,0.05,0.1,0.2}, sampled per step | same |
| allocator warmup skip / ramp | 7,000 / 5,000 | 0 / 3,000 |
| oracle | relative excess loss, floor 1e-3, self-referential context | same |
| precision | bf16 forward, fp32 master weights, fp32 allocator | same |
| init | from scratch | stage-1 checkpoint |

Other scales and backbones use the identical recipe with only the backbone
swapped:

```bash
MODEL_ID=meta-llama/Llama-3.2-3B EXP_TAG=manta_3b bash launch/train_manta.sh
MODEL_ID=Qwen/Qwen2.5-1.5B       EXP_TAG=manta_qwen bash launch/train_manta.sh
```

The LoRA-adapted variant (the appendix table that puts the architecture under the
baselines' adapter harness) turns off full fine-tuning and raises the learning
rates to the values LoRA needs:

```bash
S1_EXTRA='--set matryoshka_training_config.full_finetune=false' \
S2_EXTRA='--set matryoshka_training_config.full_finetune=false' \
EXP_TAG=manta_1b_lora bash launch/train_manta.sh   # then --lr 1e-4 / 5e-5
```

### Ablation arms

Same recipe, one flag changed (Table "allocation-policy counterfactual" and the
architecture ablations):

```bash
# Matryoshka ladder only — no allocator, uniform k
EXP_TAG=manta_1b_matonly \
S1_EXTRA='--set matryoshka_training_config.use_k_predictor=false' \
S2_EXTRA='--set matryoshka_training_config.use_k_predictor=false' \
bash launch/train_manta.sh

# naive gate — same scoring head, no tolerance conditioning, one global allocation
EXP_TAG=manta_1b_naive \
S1_EXTRA='--set matryoshka_training_config.predictor_gate_mode=naive --set matryoshka_training_config.naive_global_budget=7.37' \
S2_EXTRA='--set matryoshka_training_config.predictor_gate_mode=naive --set matryoshka_training_config.naive_global_budget=7.37' \
bash launch/train_manta.sh
```

`scripts/make_ablation_configs.py` writes the same variants as standalone config
files if you prefer that route.

### What resuming does and does not restore

Each stage is idempotent at stage granularity and resumes mid-stage from the
newest step-tagged checkpoint (`instruction_adapter_step*.pt`, written every
`SAVE_STEP` steps). Resuming restores **weights only**: the optimizer state is
re-initialized and the dataloader restarts from the beginning of the stream, so
a resumed run replays the data prefix it already saw. Runs that complete without
interruption are unaffected. Two consequences worth knowing:

- The trainer stops at `min(len(data), total_steps)`, so asking for more steps
  than the prepared data provides ends early. `train_manta.sh` compares the step
  count actually reached against the target and refuses to mark such a stage
  done, instead of silently carrying an under-trained checkpoint forward. More
  than one epoch needs a second stage invocation, not a larger `total_steps`.
- Killing a stage halfway means that stage restarts from its newest tagged
  checkpoint and overwrites the partial outputs; it does not corrupt the
  previous stage.

### Two implementation constraints

- **Gradient checkpointing must stay off.** Compression reads the encoder's
  KV-cache output; checkpointing forces `use_cache=False`, which hands
  `compress()` a `None` cache and crashes the run.
- **Do not sort the SFT batches by length.** The dataloader must preserve the
  original sample order. Length-sorted batching pushes the learning-rate decay
  phase onto the shortest sequences and costs 6–12 token-F1.
  `scripts/smoke_dataloader_order.py` and `tests/test_dataloader_order.py` guard
  this.

## 3. Evaluation

One checkpoint, two modes, one script:

```bash
EXP_DIR=experiment/manta_1b_stage2 \
EVAL_PT=data/mrqa_eval_full/qa_eval.pt \
EVAL_META=data/mrqa_eval_full/qa_eval_meta.json \
TAG=manta_1b EXPECT_N=67854 \
bash launch/eval_manta.sh
```

It runs the fixed-k sweep (`scripts/eval_k_sweep.py`, k ∈ {1,2,4,8,16,32}), the
adaptive sweep (`scripts/eval_v5_pareto.py`, ε ∈ {0.001,0.01,0.05,0.1,0.2}),
a completeness gate on the number of scored samples
(`scripts/check_eval_n.py`), and the token-F1 / EM aggregation
(`scripts/aggregate_tokenf1.py`).

`TAG` names every output file and defaults to the experiment directory, so two
checkpoints never write to the same report. Existing outputs are reused only when
the sweep grids, sample count, adapter and eval set match a manifest written
alongside them; changing any of those under the same tag stops the run rather
than reprinting stale numbers. Set `EXPECT_N` to have the aggregated report
checked as well — that is what catches leftover rank files from an earlier run on
more GPUs, which the glob would otherwise fold into the metric.

Protocol details that matter for comparability: span size 512, greedy decoding,
128 new tokens for MRQA and 64 for needle-QA, fixed seed, SQuAD-normalized
token-F1 and EM with max over references. **The paper's ID / OOD numbers are
macro-averages over the six datasets in each group**, not per-example means —
recomputing them as a micro-average inflates every number by roughly five
points. `scripts/compare_subset_fullft.py` and `scripts/topline_gap_fullft.py`
apply the macro convention.

Generations are saved per rank, so any metric can be recomputed on CPU without
re-running the model (`scripts/rescore_tokenf1_from_generated.py`).

Tolerance points beyond the training grid (for the extrapolation row) are just
other values passed to `EPS_LIST`; `scripts/meank_sweep.py` reports the
tolerance-to-compression mapping alone, and `scripts/calc_real_compression.py`
and `scripts/measured_realized_64x.py` compute realized (rather than nominal)
compression ratios.

### Long context

```bash
python scripts/build_long_context_eval.py      # synthesize needle-QA at 4K/8K/16K/32K
python scripts/prepare_longctx_eval.py         # tokenize into the eval format
python scripts/setup_long_context_eval.py      # one work_dir per (checkpoint, length)
# evaluate each work_dir with launch/eval_manta.sh, then:
python scripts/summarize_long_context_eval.py
```

### Serving cost

`scripts/benchmark_efficiency.py` measures latency, peak GPU memory and
KV-cache bytes for the numbers in the serving-cost discussion.

## 4. From evaluation outputs to paper artifacts

The aggregation scripts read the eval JSONs produced above. They reference the
experiment directory names of our runs near the top of each file (for example
`experiment/pact_v5_mrqa_fullft_1B/output`) — either point `S1_DIR`/`S2_DIR` at
those names when training or edit that one dict.

| Paper artifact | Script |
|---|---|
| Main per-dataset table at the 64× tier | `scripts/fullft_tables_A.py` (LoRA-era counterpart: `scripts/maintable_perdataset_v4_20k_tokenf1_icaefix.py`) |
| Four-tier aggregate table | `scripts/fullft_table2_harder.py` |
| Adaptive vs fixed-k at matched compression | `scripts/fullft_tables_A.py`, significance via `scripts/paired_stat_sig.py` |
| Architecture / allocation-policy ablations | `scripts/fullft_table4bc.py`, `scripts/aggregate_ablation_table.py` |
| Tolerance-to-compression mapping, extrapolation | `scripts/fullft_table5.py`, `scripts/meank_sweep.py` |
| Paired bootstrap confidence intervals | `scripts/bootstrap_ci_margins.py` |
| Scale and cross-backbone rows | `launch/eval_manta.sh` per checkpoint, then `scripts/compare_subset_fullft.py`, `scripts/topline_gap_fullft.py` |
| Long-context table | `scripts/summarize_long_context_eval.py` |
| Allocator fidelity table | `scripts/analysis_predictor_fidelity.py`, `scripts/plot_predictor_fidelity.py` |
| Slot redundancy / ordered-residual tables | `scripts/slot_redundancy_analysis.py`, `scripts/slot_residual_analysis.py`, `scripts/viz_slot_residual.py`, `scripts/slot_token_attribution.py` |
| Causal grounding of the allocation | `scripts/analysis_causal_chunk_grounding.py`, `scripts/recompute_causal_corr.py` |
| Motivation figures (loss-vs-k heterogeneity) | `scripts/motivation_probe.py` → `scripts/plot_motivation_figures.py` |
| Oracle allocation figures | `scripts/oracle_eval.py` → `scripts/viz_oracle_figures.py` |
| Compression–quality curves | `scripts/plot_pareto_v4.py` |
| Allocation distribution figures | `scripts/viz_kdist_realized.py` |
| Attention / case-study figures | `scripts/dump_priority_attention.py` → `scripts/viz_priority_figures.py` |
| Token-level compression baseline | `scripts/eval_longllmlingua.py` |

## 5. Tests

```bash
python -m pytest tests -q               # 59 tests: allocator, ladder truncation, data order
python tests/test_k_predictor_smoke.py  # allocator shapes at production sizes
python scripts/test_slot_select.py      # slot-selection tensor logic
```

All three are CPU-only and pass without a GPU, a checkpoint or any data.

`tests/test_pact_v5_cpu_e2e.py` is the heavier smoke: it instantiates a real
backbone on CPU in fp32, loads a checkpoint, and runs one forward/backward
through the allocator path. Run it explicitly and point it at what it needs:

```bash
MANTA_MODEL_ID=/path/to/backbone MANTA_SAC_ADAPTER=/path/to/adapter.pt \
  python tests/test_pact_v5_cpu_e2e.py
```

## 6. Compute

The 1B main model takes about 16.2 h wall clock on 8×RTX 4090 (8.1 h per stage),
roughly 130 GPU-hours. Training one MANTA model costs about 1.5× the four-ratio
dedicated baseline sweep it replaces, because the all-k objective forwards six
prefixes per step and the encoder is trained in full. The oracle labeling adds
`N · |K|` no-gradient decoder forwards per step, reusing the already-encoded
compressed KV; that cost is included in the figure above. Full-MRQA evaluation
of one checkpoint (both modes, 67,854 samples) is the dominant evaluation cost
and shards linearly across GPUs.
