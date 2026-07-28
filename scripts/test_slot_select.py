"""CPU unit test for scripts/slot_select.py — pure tensor logic, no model load.

Validates:
  1. canonical-set selection == legacy first-k prefix reference (numerically identical).
  2. global index mapping respects per-chunk offsets.
  3. reverse keeps the LAST k; shuffled prefix is a consistent sub-order; random is a
     valid k-subset; sizes are correct including ragged final chunk (m < 32).
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from slot_select import (
    build_keep_indices_from_sets, select_kv,
    policy_canonical, policy_reverse, policy_shuffled, policy_random, make_perms,
)

dev = torch.device("cpu")
torch.manual_seed(0)

# Synthetic full per-chunk KV: 3 layers, B=1, H=2, head_dim=4. Chunks: [32,32,20].
mems = [32, 32, 20]
total_M = sum(mems)
L, B, H, Dh = 3, 1, 2, 4
pkv = tuple(
    (torch.randn(B, H, total_M, Dh), torch.randn(B, H, total_M, Dh)) for _ in range(L)
)


def legacy_prefix_keep_idx(mems, k):
    """Reference impl of model/modeling.py _build_keep_indices (first-k prefix)."""
    parts, off = [], 0
    for m in mems:
        ke = min(int(k), int(m))
        parts.append(torch.arange(off, off + ke))
        off += m
    return torch.cat(parts)


fails = []

# --- Test 1: canonical == legacy prefix, for every k, on actual tensors ---
for k in [1, 2, 4, 8, 16, 32]:
    sets = policy_canonical(mems, k, dev)
    keep = build_keep_indices_from_sets(mems, sets, dev)
    ref = legacy_prefix_keep_idx(mems, k)
    if not torch.equal(keep, ref):
        fails.append(f"[T1] canonical keep_idx != legacy prefix at k={k}: {keep.tolist()} vs {ref.tolist()}")
    sel = select_kv(pkv, keep)
    ref_sel = tuple((K.index_select(2, ref), V.index_select(2, ref)) for (K, V) in pkv)
    for li in range(L):
        if not (torch.equal(sel[li][0], ref_sel[li][0]) and torch.equal(sel[li][1], ref_sel[li][1])):
            fails.append(f"[T1] select_kv mismatch at k={k} layer={li}")

# --- Test 2: offsets — chunk c local idx j maps to global sum(mems[:c]) + j ---
sets = [torch.tensor([0, 5]), torch.tensor([3]), torch.tensor([0, 19])]
keep = build_keep_indices_from_sets(mems, sets, dev)
expected = torch.tensor([0, 5, 32 + 3, 64 + 0, 64 + 19])
if not torch.equal(keep, expected):
    fails.append(f"[T2] offset mapping wrong: {keep.tolist()} vs {expected.tolist()}")

# --- Test 3: reverse keeps LAST k (ragged chunk m=20) ---
for k in [1, 4, 8]:
    sets = policy_reverse(mems, k, dev)
    for m, s in zip(mems, sets):
        keep_n = min(k, m)
        if s.numel() != keep_n:
            fails.append(f"[T3] reverse size wrong k={k} m={m}: {s.numel()}")
        if int(s.min()) != m - keep_n or int(s.max()) != m - 1:
            fails.append(f"[T3] reverse not last-k k={k} m={m}: {s.tolist()}")

# --- Test 4: shuffled is a consistent sub-order (prefix nesting across k) ---
g = torch.Generator().manual_seed(123)
perms = make_perms(mems, g, dev)
s4 = policy_shuffled(mems, 4, dev, perms)
s8 = policy_shuffled(mems, 8, dev, perms)
for c in range(len(mems)):
    if not torch.equal(s8[c][:4], s4[c]):
        fails.append(f"[T4] shuffled not nested at chunk {c}: {s4[c].tolist()} vs {s8[c][:4].tolist()}")
    if len(set(perms[c].tolist())) != mems[c]:
        fails.append(f"[T4] perm not a permutation at chunk {c}")

# --- Test 5: random is a valid k-subset (unique, in range, right size) ---
g2 = torch.Generator().manual_seed(7)
for k in [3, 8]:
    sets = policy_random(mems, k, dev, g2)
    for m, s in zip(mems, sets):
        keep_n = min(k, m)
        if s.numel() != keep_n or len(set(s.tolist())) != keep_n:
            fails.append(f"[T5] random not unique/sized k={k} m={m}: {s.tolist()}")
        if s.numel() and (int(s.min()) < 0 or int(s.max()) >= m):
            fails.append(f"[T5] random out of range k={k} m={m}")

if fails:
    print("FAILED:")
    for f in fails:
        print("  ", f)
    sys.exit(1)
print("ALL SLOT_SELECT UNIT TESTS PASSED")
