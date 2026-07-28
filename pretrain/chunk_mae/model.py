"""Chunk-MAE wrapper around CrossChunkKVCompressor.

Replaces the input V at masked chunks with a learnable mask_token, runs the
compressor, and returns reconstruction MSE loss between V_fused and the
original V at masked positions.

Why this works:
    The fusion module has no other channel from masked chunk to itself —
    the only path that recovers the masked V is cross-chunk attention to
    visible chunks. Self-supervised by construction; no QA labels needed.

K is left untouched (cross-chunk fusion is V-only — preserves RoPE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChunkMAEModel(nn.Module):
    """Chunk-masked autoencoder over a CrossChunkKVCompressor.

    Args:
        compressor: an instance of CrossChunkKVCompressor.
        mask_token_init_std: stddev for mask_token init (Normal).
    """

    def __init__(self, compressor, mask_token_init_std=0.02):
        super().__init__()
        self.compressor = compressor
        # Per-layer mask token shared across all chunks and mem positions.
        # Shape mirrors a single (1, H, 1, D) so it broadcasts to mem_c.
        self.mask_token = nn.Parameter(
            torch.zeros(
                compressor.num_layers,
                compressor.num_kv_heads,
                1,
                compressor.head_dim,
            )
        )
        nn.init.normal_(self.mask_token, std=mask_token_init_std)

    @staticmethod
    def _chunk_offsets(real_mems):
        """Return list of (start, end) indices in the un-padded flat seq."""
        offsets = []
        cursor = 0
        for rm in real_mems:
            offsets.append((cursor, cursor + rm))
            cursor += rm
        return offsets

    def _build_masked_input(self, chunk_kvs, real_mems, mask_set):
        """Substitute V with broadcast mask_token at masked chunk positions.

        Returns a new list of L-tuples; original chunk_kvs unchanged
        (immutability — see ~/.claude/rules/common/coding-style.md).
        """
        L = self.compressor.num_layers
        H = self.compressor.num_kv_heads
        D = self.compressor.head_dim

        masked = []
        for c in range(len(chunk_kvs)):
            rm = real_mems[c]
            new_layers = []
            for l in range(L):
                k = chunk_kvs[c][l][0]
                v = chunk_kvs[c][l][1]
                if c in mask_set:
                    mt = self.mask_token[l].view(1, H, 1, D).expand(1, H, rm, D)
                    v = mt.to(dtype=v.dtype, device=v.device)
                new_layers.append((k, v))
            masked.append(tuple(new_layers))
        return masked

    def forward(self, sample, return_metrics=False):
        """Run masked reconstruction and return scalar loss.

        Args:
            sample: dict from ChunkMAEDataset.
            return_metrics: if True, also return a dict of diagnostics.

        Returns:
            loss (and metrics dict if requested).
        """
        chunk_kvs = sample["chunk_kvs"]
        real_mems = sample["real_mems"]
        mask_set = set(sample["mask_chunk_indices"])
        nc = len(chunk_kvs)
        L = self.compressor.num_layers

        # Original V per chunk per layer — target for reconstruction.
        target_V = [
            [chunk_kvs[c][l][1] for c in range(nc)]
            for l in range(L)
        ]

        masked_chunk_kvs = self._build_masked_input(chunk_kvs, real_mems, mask_set)
        fused = self.compressor(masked_chunk_kvs)

        offsets = self._chunk_offsets(real_mems)

        # Loss: per-layer MSE averaged over masked chunks; outer mean over L.
        layer_losses = []
        for l in range(L):
            v_fused_l = fused[l][1]  # [1, H, S_orig, D]
            chunk_losses = []
            for c in mask_set:
                s, e = offsets[c]
                pred = v_fused_l[:, :, s:e, :].float()
                tgt = target_V[l][c].to(device=pred.device).float()
                chunk_losses.append(F.mse_loss(pred, tgt))
            if chunk_losses:
                layer_losses.append(torch.stack(chunk_losses).mean())

        if not layer_losses:
            # Should not happen given dataset guarantees ≥1 mask.
            zero = torch.zeros((), device=self.mask_token.device, requires_grad=True)
            return (zero, {}) if return_metrics else zero

        loss = torch.stack(layer_losses).mean()

        if return_metrics:
            with torch.no_grad():
                # Baseline: predicting all-zeros vs. original V (sanity reference).
                baseline = []
                for l in range(L):
                    for c in mask_set:
                        s, e = offsets[c]
                        tgt = target_V[l][c].to(device=loss.device).float()
                        baseline.append((tgt ** 2).mean())
                baseline_mse = torch.stack(baseline).mean().item()
            metrics = {
                "loss": loss.item(),
                "baseline_zero_mse": baseline_mse,
                "num_masked_chunks": len(mask_set),
                "num_chunks": nc,
            }
            return loss, metrics

        return loss
