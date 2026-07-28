"""Standalone arbitrary slot-selection for PACT "ordered residual code" analysis.

Per-chunk anchor slots live concatenated on a [total_M] axis where chunk c occupies
[offset_c, offset_c + m_c). The legacy path (_build_keep_indices in model/modeling.py)
keeps the FIRST k of each chunk (Matryoshka prefix). This module generalises that to
ARBITRARY per-chunk local index sets, so we can compare slot-selection POLICIES:
  canonical prefix / reverse / random subset / shuffled-order null / greedy oracle.

No edits to modeling.py: callers run model.compress(truncate_k=None) to get the full
per-chunk KV, then apply select_kv() with indices built here. select_kv mirrors the
exact op modeling.py uses (index_select on dim=2 of each (K,V) layer tensor), so a
canonical-prefix selection reproduces the model's own truncate path numerically.
"""
import torch


def build_keep_indices_from_sets(per_chunk_mem_sizes, idx_sets_per_chunk, device):
    """Map per-chunk LOCAL anchor indices -> flat GLOBAL indices into [total_M].

    per_chunk_mem_sizes: list[int] m_c (full, untruncated anchor count per chunk)
    idx_sets_per_chunk:  list (len N_chunks) of 1D long indices / lists, LOCAL (0..m_c-1)
    Returns: 1D LongTensor of global indices (kept order = concat of per-chunk order).
    """
    assert len(per_chunk_mem_sizes) == len(idx_sets_per_chunk), \
        f"{len(per_chunk_mem_sizes)} chunks vs {len(idx_sets_per_chunk)} idx-sets"
    parts = []
    offset = 0
    for m, local in zip(per_chunk_mem_sizes, idx_sets_per_chunk):
        local_t = torch.as_tensor(local, dtype=torch.long, device=device).reshape(-1)
        if local_t.numel():
            assert int(local_t.min()) >= 0 and int(local_t.max()) < m, \
                f"local idx out of [0,{m}) : {local_t.tolist()}"
        parts.append(local_t + offset)
        offset += int(m)
    if not parts:
        return torch.empty(0, dtype=torch.long, device=device)
    return torch.cat(parts, dim=0)


def select_kv(encoder_past_key_values, keep_idx):
    """Select anchors (dim=2) on every layer (K,V). Mirrors modeling.py truncate op."""
    return tuple(
        (K.index_select(2, keep_idx), V.index_select(2, keep_idx))
        for (K, V) in encoder_past_key_values
    )


# ---- Policy index-set generators (return list of per-chunk LOCAL index tensors) ----

def policy_canonical(per_chunk_mem_sizes, k, device):
    """First min(k,m_c) anchors per chunk (PACT's learned Matryoshka order)."""
    return [torch.arange(min(int(k), int(m)), device=device) for m in per_chunk_mem_sizes]


def policy_reverse(per_chunk_mem_sizes, k, device):
    """LAST min(k,m_c) anchors per chunk (reverse prefix)."""
    out = []
    for m in per_chunk_mem_sizes:
        keep = min(int(k), int(m))
        out.append(torch.arange(int(m) - keep, int(m), device=device))
    return out


def policy_shuffled(per_chunk_mem_sizes, k, device, perms):
    """Prefix of length k of a FIXED per-chunk permutation `perms` (the null control).

    perms: list (len N_chunks) of 1D long permutations of range(m_c). Same perms must be
    reused across all k so the null is a single consistent ordering, not a fresh draw.
    """
    out = []
    for m, perm in zip(per_chunk_mem_sizes, perms):
        keep = min(int(k), int(m))
        out.append(torch.as_tensor(perm, dtype=torch.long, device=device)[:keep])
    return out


def policy_random(per_chunk_mem_sizes, k, device, generator):
    """A random k-subset per chunk (fresh draw; call repeatedly for a band)."""
    out = []
    for m in per_chunk_mem_sizes:
        keep = min(int(k), int(m))
        perm = torch.randperm(int(m), generator=generator, device=device)
        out.append(perm[:keep])
    return out


def make_perms(per_chunk_mem_sizes, generator, device):
    """One fixed permutation per chunk, for the shuffled-order null."""
    return [torch.randperm(int(m), generator=generator, device=device)
            for m in per_chunk_mem_sizes]
