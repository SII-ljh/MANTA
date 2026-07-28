"""Chunk-MAE unsupervised pre-training for cross-chunk fusion (V3 Stage 2).

Task: random mask k chunks' V, reconstruct from remaining chunks via the
CrossChunkKVCompressor. Reuses precomputed KV shards from stage 1 SAC encoder
so no new precomputation is needed.
"""

from .dataset import ChunkMAEDataset
from .model import ChunkMAEModel

__all__ = ["ChunkMAEDataset", "ChunkMAEModel"]
