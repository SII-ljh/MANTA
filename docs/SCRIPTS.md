# Script index

`scripts/` is kept flat and with the original file names, so that what is
described here is exactly what produced the reported numbers. Grouped by purpose:

## Data preparation

| Script | Purpose |
|---|---|
| `prepare_pretrain_data.py` | build the stage-1 long-text cache from a plain text corpus |
| `prepare_qa_data.py` | tokenize a QA pool into `qa_train.pt` / `qa_eval.pt` + `qa_eval_meta.json` |
| `prepare_merged_data.py` | merge MRQA in-domain with additional QA sources, de-overlapped |
| `prepare_extended_qa_data.py` | extended QA preparation with a context-length curriculum |
| `precompute_kv.py`, `precompute_kv_fast.py` | pre-compute per-span compressed KV caches (batched variant) |
| `build_long_context_eval.py` | synthesize the needle-QA long-context sets |
| `prepare_longctx_eval.py` | tokenize the needle-QA sets into the eval format |
| `setup_long_context_eval.py` | create one eval work_dir per (checkpoint, length) |

## Training entry points

| Script | Purpose |
|---|---|
| `train_pretrain.py` | stage-1 launcher (the vanilla, non-Matryoshka path) |
| `train_sft.py` | stage-2 launcher (the vanilla path) |
| `make_ablation_configs.py` | emit configs for the ablation arms as standalone files |

The MANTA main line trains through `sft/matryoshka_trainer.py`; use
`launch/train_manta.sh`.

## Evaluation

| Script | Purpose |
|---|---|
| `eval_k_sweep.py` | fixed-k sweep over the Matryoshka ladder |
| `eval_v5_pareto.py` | adaptive evaluation: allocator at a list of tolerances (optionally the oracle upper bound) |
| `eval_adaptive_compression.py` | earlier adaptive evaluator (top-k / threshold modes) |
| `eval_zero_shot.py` | uncompressed zero-shot backbone reference |
| `eval_longllmlingua.py` | token-level compression baseline |
| `merge_eval_shards.py` | merge per-host shards into one rank-0 summary |
| `check_eval_n.py` | fail loudly on incomplete eval artifacts (sample counts, label sets) |
| `rescore_tokenf1_from_generated.py` | recompute token-F1 / EM from saved generations, CPU only |
| `aggregate_tokenf1.py` | per-dataset and macro ID/OOD aggregation |
| `meank_sweep.py` | tolerance-to-mean-allocation mapping, no generation |
| `calc_real_compression.py`, `measured_realized_64x.py` | realized (not nominal) compression ratios |
| `summarize_long_context_eval.py` | collect long-context results across lengths |
| `benchmark_efficiency.py` | latency, peak memory, KV-cache bytes |

## Motivation and oracle analysis

| Script | Purpose |
|---|---|
| `motivation_probe.py` | dump the per-span loss-vs-k matrix that motivates adaptive allocation |
| `plot_motivation_figures.py` | render the motivation figures from that dump |
| `oracle_eval.py` | oracle per-span allocation: sufficiency distribution and skyline |
| `viz_oracle_figures.py` | render the oracle figures |

## Allocation and representation analysis

| Script | Purpose |
|---|---|
| `analysis_predictor_fidelity.py`, `plot_predictor_fidelity.py` | allocator fidelity in-domain vs out-of-domain |
| `analysis_causal_chunk_grounding.py`, `recompute_causal_corr.py` | does the allocation track causal span importance |
| `slot_redundancy_analysis.py` | redundancy between memory slots |
| `slot_residual_analysis.py`, `slot_select.py`, `viz_slot_residual.py` | ordered-residual-code behaviour of the ladder |
| `slot_token_attribution.py`, `viz_slot_token_attr.py` | per-token NLL attribution across slots |
| `dump_priority_attention.py`, `viz_priority_figures.py` | attention and case-study figures |
| `viz_kdist_realized.py` | distribution of realized allocations, per dataset |
| `selfcheck_kpredictor.py` | sanity checks on a trained allocator, no retraining |
| `smoke_dataloader_order.py` | CPU check that the dataloaders preserve sample order |
| `test_slot_select.py` | CPU unit test for the slot-selection logic |

## Statistics, tables, figures

| Script | Purpose |
|---|---|
| `fullft_tables_A.py` | per-dataset main table and the matched-compression fixed-vs-adaptive table |
| `fullft_table2_harder.py` | four-tier aggregate table |
| `fullft_table4bc.py` | architecture and allocation-policy ablation tables |
| `fullft_table5.py` | tolerance controllability, extrapolation, mid-ladder smoothness |
| `maintable_perdataset_v4_20k_tokenf1_icaefix.py` | per-dataset table for the LoRA-harness line |
| `compare_subset_fullft.py`, `topline_gap_fullft.py` | macro ID/OOD splits and the gap to the uncompressed reference |
| `aggregate_ablation_table.py` | merge ablation arms into one table |
| `bootstrap_ci_margins.py` | paired bootstrap confidence intervals on margins |
| `paired_stat_sig.py` | paired test, adaptive vs fixed-k at matched compression |
| `plot_pareto_v4.py` | compression–quality curves |

File names carry some project history (`v5`, `v4_20k`, `fullft`, `icaefix`).
They are kept as-is: these are the exact scripts behind the numbers, and
`docs/REPRODUCE.md` maps each paper artifact to its script.
