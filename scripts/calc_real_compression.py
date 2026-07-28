"""Compute the TRUE token-level compression rate for PACT adaptive mode.

For each sample:
    real_compression = input_token_count / total_kept_anchors

Then aggregate per-dataset and overall.
"""
import json
import os
from collections import defaultdict

import torch

PACT_DIR = os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVAL_PT = f"{PACT_DIR}/sft/output/comi_eval_Llama-3.1-8B_100000samples_instruction.pt"
EXP_DIR = f"{PACT_DIR}/experiment/pact_v4a_comi/output"

# Load tokenized eval data — list of dicts with 'input_ids' tensor
eval_data = torch.load(EVAL_PT)
print(f"loaded {len(eval_data)} eval samples")
input_lens = [len(s["input_ids"]) for s in eval_data]
print(f"input token lengths: min={min(input_lens)} median={sorted(input_lens)[len(input_lens)//2]} max={max(input_lens)} mean={sum(input_lens)/len(input_lens):.0f}")

# Load meta for dataset labels
META = f"{PACT_DIR}/data/unified_eval_subset/comi_eval_meta.json"
with open(META) as f:
    meta = json.load(f)
assert len(meta) == len(eval_data), f"meta {len(meta)} vs data {len(eval_data)}"

for label in ["adaptive", "k4", "k8"]:
    rank_files = [f"{EXP_DIR}/comi_{label}_rank{r}.json" for r in range(8)]
    by_id = {}
    for f in rank_files:
        if not os.path.exists(f):
            continue
        with open(f) as fp:
            d = json.load(fp)
        for key, arr in d.items():
            for s in arr:
                by_id[int(s["id"])] = s

    if not by_id:
        print(f"\n=== {label}: no data ===")
        continue

    print(f"\n=== {label}: n={len(by_id)} ===")
    per_ds_lens = defaultdict(list)
    per_ds_kept = defaultdict(list)
    per_ds_anchors = defaultdict(list)

    for sid, s in by_id.items():
        ds = s["dataset"]
        tk = int(s["total_kept"])
        ta = int(s["total_anchors"])
        in_tokens = input_lens[sid]
        per_ds_lens[ds].append(in_tokens)
        per_ds_kept[ds].append(tk)
        per_ds_anchors[ds].append(ta)

    print(f"{'dataset':<14}{'N':>5}{'input_tok mean':>16}{'kept_anch mean':>16}{'real comp':>12}{'anchor comp':>14}")
    print("-" * 80)
    grand_tok = 0
    grand_kept = 0
    for ds in ["nq", "hotpotqa", "2wikimqa", "narrativeqa", "multi_news"]:
        if ds not in per_ds_lens:
            continue
        n = len(per_ds_lens[ds])
        tot_tok = sum(per_ds_lens[ds])
        tot_kept = sum(per_ds_kept[ds])
        tot_anchors = sum(per_ds_anchors[ds])
        real_comp = tot_tok / tot_kept if tot_kept else 0
        anchor_comp = tot_anchors / tot_kept if tot_kept else 0
        print(f"{ds:<14}{n:>5}{tot_tok/n:>16.0f}{tot_kept/n:>16.1f}{real_comp:>12.2f}{anchor_comp:>14.2f}")
        grand_tok += tot_tok
        grand_kept += tot_kept

    overall_comp = grand_tok / grand_kept if grand_kept else 0
    print(f"{'OVERALL':<14}{len(by_id):>5}{grand_tok/len(by_id):>16.0f}{grand_kept/len(by_id):>16.1f}{overall_comp:>12.2f}")
