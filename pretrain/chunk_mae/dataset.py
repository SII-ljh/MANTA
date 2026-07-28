"""Streaming dataset that yields per-chunk KV with a random subset masked.

Each yielded sample is a dict consumed by ChunkMAEModel.forward:
    {
        "sample_idx":  int,
        "chunk_kvs":   list[C] of L tuples (K, V), each [1, H, mem_c, D],
                       UNMODIFIED — model substitutes V at masked chunks.
        "real_mems":   list[int] length C,
        "mask_chunk_indices": sorted list[int] of masked chunk positions,
    }

Shards on disk are produced by SAC stage-1 precompute (sft/precompute_kv.py).
"""

import glob as glob_module
import os
import random

import torch
from torch.utils.data import IterableDataset


class ChunkMAEDataset(IterableDataset):
    """Stream precomputed KV shards and pick chunks to mask.

    Args:
        shard_dir: directory containing shard_*.pt files.
        mask_ratio: fraction of chunks to mask (rounded, at least 1).
        min_chunks: drop samples with fewer chunks than this (need enough
                    visible chunks for fusion to be meaningful).
        max_chunks: drop samples with more chunks (compressor's chunk pos
                    embedding is bounded; default safe for unified data).
        dtype: cast K, V to this dtype on yield. CPU bf16 is slow, prefer
               torch.float32 for CPU smoke runs.
        seed: RNG seed for shard shuffling and mask selection.
        max_samples: optional cap on samples yielded (after filtering).
        cycle: if True, never naturally exhaust (mirrors ShardedKVDataset
               behavior; trainer terminates on step counter).
    """

    def __init__(
        self,
        shard_dir=None,
        shard_files=None,
        mask_ratio=0.25,
        min_chunks=4,
        max_chunks=32,
        dtype=torch.float32,
        seed=42,
        max_samples=None,
        cycle=False,
        shuffle_shards=True,
    ):
        super().__init__()
        if not (0.0 < mask_ratio < 1.0):
            raise ValueError(f"mask_ratio must be in (0, 1), got {mask_ratio}")
        if min_chunks < 2:
            raise ValueError(f"min_chunks must be >= 2, got {min_chunks}")
        if (shard_dir is None) == (shard_files is None):
            raise ValueError("provide exactly one of shard_dir or shard_files")

        if shard_files is not None:
            self.shard_files = list(shard_files)
        else:
            self.shard_files = sorted(
                glob_module.glob(os.path.join(shard_dir, "shard_*.pt"))
            )
        if not self.shard_files:
            raise FileNotFoundError(
                f"no shards available (shard_dir={shard_dir}, shard_files={shard_files})"
            )
        if shuffle_shards:
            random.Random(seed).shuffle(self.shard_files)

        self.mask_ratio = mask_ratio
        self.min_chunks = min_chunks
        self.max_chunks = max_chunks
        self.dtype = dtype
        self.seed = seed
        self.max_samples = max_samples
        self.cycle = cycle

    def __iter__(self):
        """Iteration semantics:

            cycle=False, max_samples=None  -> single pass, natural exhaustion
            cycle=False, max_samples=N     -> single pass, capped at N
            cycle=True,  max_samples=None  -> infinite re-iteration over all shards
            cycle=True,  max_samples=N     -> infinite re-iteration over the FIRST N
                                              filtered samples (caches them in RAM)

        Mask selection is re-randomized every iteration of every sample so a
        cycled sample is not seen with the same mask twice.
        """
        rng = random.Random(self.seed)

        # Cache-and-cycle path: load shards once, keep first N filtered samples
        # in CPU RAM, then loop forever. Avoids re-reading shards from disk.
        if self.cycle and self.max_samples is not None:
            cached = self._collect_first_n(self.max_samples)
            while True:
                for entry in cached:
                    yield self._make_yield(entry, rng)
            return

        emitted = 0
        first_pass = True
        while first_pass or self.cycle:
            first_pass = False
            for shard_file in self.shard_files:
                shard = torch.load(shard_file, map_location="cpu", weights_only=False)
                for entry in shard:
                    if self.max_samples is not None and emitted >= self.max_samples:
                        del shard
                        return

                    nc = entry["num_chunks"]
                    if nc < self.min_chunks or nc > self.max_chunks:
                        continue

                    yield self._make_yield(entry, rng)
                    emitted += 1
                del shard

    def _collect_first_n(self, n):
        """Read shards in order until we have n filtered+cast samples cached."""
        cached = []
        for shard_file in self.shard_files:
            if len(cached) >= n:
                break
            shard = torch.load(shard_file, map_location="cpu", weights_only=False)
            for entry in shard:
                if len(cached) >= n:
                    break
                nc = entry["num_chunks"]
                if nc < self.min_chunks or nc > self.max_chunks:
                    continue
                cached.append(self._cast_entry(entry))
            del shard
        return cached

    def _cast_entry(self, entry):
        """Return a copy of entry with chunk_kvs cast to self.dtype."""
        chunk_kvs = entry["chunk_kvs"]
        if self.dtype is not None:
            chunk_kvs = [
                tuple((k.to(self.dtype), v.to(self.dtype)) for (k, v) in layers)
                for layers in chunk_kvs
            ]
        return {
            "sample_idx": entry["sample_idx"],
            "num_chunks": entry["num_chunks"],
            "chunk_kvs": chunk_kvs,
        }

    def _make_yield(self, entry, rng):
        """Apply dtype cast (if needed) and pick a fresh mask index set."""
        # Entries from the cache are already cast; raw shard entries are not.
        if "chunk_kvs" in entry and self.dtype is not None and not isinstance(
            entry["chunk_kvs"][0][0][0], torch.Tensor
        ):
            entry = self._cast_entry(entry)
        elif "chunk_kvs" in entry and self.dtype is not None:
            # Cast lazily if needed; cached entries are pre-cast (idempotent cast).
            sample_dtype = entry["chunk_kvs"][0][0][0].dtype
            if sample_dtype != self.dtype:
                entry = self._cast_entry(entry)

        nc = entry["num_chunks"]
        chunk_kvs = entry["chunk_kvs"]
        real_mems = [chunk_kvs[c][0][0].shape[2] for c in range(nc)]
        num_mask = max(1, int(round(self.mask_ratio * nc)))
        num_mask = min(num_mask, nc - 1)
        mask_idx = sorted(rng.sample(range(nc), num_mask))
        return {
            "sample_idx": entry["sample_idx"],
            "chunk_kvs": chunk_kvs,
            "real_mems": real_mems,
            "mask_chunk_indices": mask_idx,
        }
