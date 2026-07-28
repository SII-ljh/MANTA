"""CPU smoke: drive the patched dataloaders exactly as matryoshka_trainer does
(get_dataset -> DataLoader(batch_size=None)) and confirm the bs==1 training
stream preserves MAIN's example order (no length-sort), while bs>1 still pads.
No model weights / GPU needed. Run: python scripts/smoke_dataloader_order.py
"""
import os, sys
import torch
from torch.utils.data import DataLoader

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "pretrain"))
sys.path.insert(0, os.path.join(ROOT, "sft"))
import pre_dataloader, instruction_dataloader


def make_pretrain_examples(n, seed=0):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        L = int(torch.randint(64, 512, (1,), generator=g).item())
        out.append({"inputs": torch.full((L,), i, dtype=torch.long),
                    "ae_target": torch.full((L,), i, dtype=torch.long),
                    "lm_target": torch.full((L,), i, dtype=torch.long)})
    return out


def make_sft_examples(n, seed=1):
    g = torch.Generator().manual_seed(seed)
    out = []
    for i in range(n):
        L = int(torch.randint(64, 512, (1,), generator=g).item())
        out.append({"input_ids": torch.full((L,), i, dtype=torch.long),
                    "lm_targets": torch.full((L,), i, dtype=torch.long),
                    "instruction_target": torch.full((L,), i, dtype=torch.long)})
    return out


def shard(examples, rank, device_count):
    per = len(examples) // device_count
    return examples[rank * per: rank * per + per]


def drive(get_dataset, examples, bs):
    ds = get_dataset("Compress", examples, bs)
    loader = DataLoader(ds, batch_size=None)
    return list(loader)


def marker_seq(batches):
    return [int(b["input_ids"][0, 0].item()) for b in batches]


def main():
    print("=== PRETRAIN dataloader, bs=1, device_count=8 shard rank0 ===")
    ex = make_pretrain_examples(320)
    sh = shard(ex, rank=0, device_count=8)
    batches = drive(pre_dataloader.get_dataset, sh, bs=1)
    seq = marker_seq(batches)
    expected = [int(e["inputs"][0].item()) for e in sh]
    assert len(batches) == len(sh), len(batches)
    assert seq == expected, "bs=1 must preserve input order"
    lens = [b["input_ids"].size(1) for b in batches]
    print(f"  steps={len(batches)} order_preserved={seq == expected}")
    print(f"  first-5 lens={lens[:5]}  last-5 (WSD decay tail) lens={lens[-5:]}")
    assert lens != sorted(lens, reverse=True), "tail must NOT be length-sorted (the bug)"
    print("  -> decay tail is NOT the shortest sequences (regression gone)")

    print("=== SFT dataloader, bs=1, shard rank3 ===")
    ex = make_sft_examples(320)
    sh = shard(ex, rank=3, device_count=8)
    batches = drive(instruction_dataloader.get_dataset, sh, bs=1)
    seq = marker_seq(batches)
    expected = [int(e["input_ids"][0].item()) for e in sh]
    assert seq == expected, "bs=1 must preserve input order"
    for b in batches:
        assert b["input_ids"].size(0) == 1 and torch.isfinite(b["input_ids"].float()).all()
    print(f"  steps={len(batches)} order_preserved=True all-finite=True")

    print("=== bs=2 still length-sorts + pads (feature retained) ===")
    ex = make_pretrain_examples(40)
    batches = drive(pre_dataloader.get_dataset, ex, bs=2)
    assert all(b["input_ids"].size(0) == 2 for b in batches)
    assert all(b["input_ids"].size(1) == b["lm_targets"].size(1) for b in batches)
    print(f"  bs=2 batches={len(batches)} all rows equal-length-padded=True")

    print("\n[smoke] ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
