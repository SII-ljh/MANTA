"""Regression test for the dataloader length-sort bug.

MAIN (allk_k34up) was trained/31 with the OLD dataloaders that fed
examples in their original (seed-0 shuffled, mixed-length) order. On
both dataloaders were patched to length-sort the whole shard descending to
support batch_size>1. At the bs_per_device=1 used by every real run this sort
is the SOLE behavioural change vs MAIN, and it makes the WSD low-LR decay phase
train only on the shortest sequences -> ~6pt/stage  regression.

Fix: at batch_size==1 the dataloader MUST preserve input order (no sort), so
the training example stream is identical to MAIN. batch_size>1 keeps sorting.
"""
import os
import sys
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "pretrain"))
sys.path.insert(0, os.path.join(ROOT, "sft"))

import pre_dataloader            # noqa: E402
import instruction_dataloader    # noqa: E402

# lengths deliberately NON-monotonic: a length-descending sort reorders the
# markers to [1, 4, 2, 0, 3], so order preservation is observable.
_LENS = [3, 9, 5, 2, 7]


def test_pretrain_bs1_preserves_input_order():
    examples = [{
        "inputs":    torch.full((L,), i, dtype=torch.long),
        "ae_target": torch.full((L,), i, dtype=torch.long),
        "lm_target": torch.full((L,), i, dtype=torch.long),
    } for i, L in enumerate(_LENS)]
    ds = pre_dataloader.CompressDataset(examples, batch_size=1)
    markers = [int(b["input_ids"][0, 0].item()) for b in ds]
    assert markers == [0, 1, 2, 3, 4], f"bs=1 must preserve order, got {markers}"


def test_instruction_bs1_preserves_input_order():
    examples = [{
        "input_ids":          torch.full((L,), i, dtype=torch.long),
        "lm_targets":         torch.full((L,), i, dtype=torch.long),
        "instruction_target": torch.full((L,), i, dtype=torch.long),
    } for i, L in enumerate(_LENS)]
    ds = instruction_dataloader.CompressDataset(examples, batch_size=1)
    markers = [int(b["input_ids"][0, 0].item()) for b in ds]
    assert markers == [0, 1, 2, 3, 4], f"bs=1 must preserve order, got {markers}"


def test_pretrain_bs2_still_batches_and_pads():
    # bs>1 path is intentionally kept: equal-length stacked batches.
    examples = [{
        "inputs":    torch.full((L,), i, dtype=torch.long),
        "ae_target": torch.full((L,), i, dtype=torch.long),
        "lm_target": torch.full((L,), i, dtype=torch.long),
    } for i, L in enumerate(_LENS)]
    batches = list(pre_dataloader.CompressDataset(examples, batch_size=2))
    assert len(batches) == 2  # 5 examples, bs=2 -> 2 full batches, drop trailing
    for b in batches:
        assert b["input_ids"].size(0) == 2
        # within a batch all rows padded to equal length
        assert b["input_ids"].size(1) == b["lm_targets"].size(1)


def test_instruction_bs2_still_batches_and_pads():
    examples = [{
        "input_ids":          torch.full((L,), i, dtype=torch.long),
        "lm_targets":         torch.full((L,), i, dtype=torch.long),
        "instruction_target": torch.full((L,), i, dtype=torch.long),
    } for i, L in enumerate(_LENS)]
    batches = list(instruction_dataloader.CompressDataset(examples, batch_size=2))
    assert len(batches) == 2
    for b in batches:
        assert b["input_ids"].size(0) == 2
