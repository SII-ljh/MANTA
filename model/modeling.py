import sys
import os
import random
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_config import BASE_PATH
sys.path.append(BASE_PATH)
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import torch.nn.functional as F
from torch import nn
import math
from model.lora import LinearLoraLayer


def _patch_qwen2_rope_full_cache():
    """Backbone-integration fix (2026-06-13): make Qwen2's index-gather RoPE
    tolerate PACT's non-0-indexed position ids.

    PACT's compress()/decoder branches use 1-indexed / kv-offset position ids
    (e.g. arange(start_idx+1, end_idx+1), or arange(end_idx, ...) over a tiny
    mem KV cache). Llama's RoPE (transformers >=4.40) computes cos/sin DIRECTLY
    from the position-id *values* (matmul), so any position value works. Qwen2's
    older RoPE instead returns cos_cached[:seq_len] and apply_rotary_pos_emb does
    `cos[position_ids]` — a GATHER that assumes position_ids in [0, seq_len).
    With PACT's 1-indexed ids the max position == seq_len -> "index out of bounds"
    (IndexKernel.cu assert / SIGABRT on the very first forward).

    Fix: return the FULL (un-sliced) cos/sin cache from the rotary forward so the
    gather indexes the full [0, max_seq_len_cached) table. Mathematically identical
    to Llama's value-based RoPE; cache is grown if a position id exceeds it.
    No-op for Llama (different rotary signature/class). Idempotent.
    """
    try:
        import transformers.models.qwen2.modeling_qwen2 as _q
    except Exception:
        return
    RE = getattr(_q, "Qwen2RotaryEmbedding", None)
    if RE is None or getattr(RE, "_pact_full_cache_patched", False):
        return
    # Only old-style cached rotary (forward(x, seq_len) + cos_cached) needs this.
    if not (hasattr(RE, "_set_cos_sin_cache") and hasattr(RE, "forward")):
        return

    def _full_cache_forward(self, x, seq_len=None):
        if seq_len is not None and int(seq_len) > self.max_seq_len_cached:
            self._set_cos_sin_cache(seq_len=int(seq_len), device=x.device, dtype=x.dtype)
        return (self.cos_cached.to(dtype=x.dtype), self.sin_cached.to(dtype=x.dtype))

    RE.forward = _full_cache_forward
    RE._pact_full_cache_patched = True


# Apply at import so every model built in this process (encoder, decoder,
# eval-time loads) gets the fix regardless of construction path.
_patch_qwen2_rope_full_cache()


class GatePredictor(nn.Module):
    """Multi-mode gate predictor.

    gate_mode:

      "cumprod" (v3 default)
        Monotonic per-anchor gate within chunk:
            g_i = prod_{j<=i} sigmoid(MLP(h_j))
        Truncation-only; cross-chunk elastic only via aggregated lm_loss.

      "cross_chunk_softmax" (v4a — ICLR novelty: global budget conservation)
        Pool each chunk to a score; cross-chunk softmax allocates fixed total
        budget B_total = N_chunks * target_avg among chunks.
            s_c    = MLP(mean_pool(anchor_hidden_c))           # [num_chunks, B]
            α_c    = softmax(s_c, dim=chunks)
            b_c    = N_chunks * target_avg * α_c               # sum_c b_c = B_total
            g_{c,i} = sigmoid((b_c - i - 0.5) / tau)           # soft within-chunk gate
        Chunks COMPETE for fixed global budget — true elastic redistribution.
        No sparsity loss needed; budget conservation is structural.

      "content_topk" (v4b — ICLR novelty: content-aware anchor selection)
        Per-anchor score s_i. At training, with matryoshka k sampled, output a
        differentiable soft top-k mask using Gumbel-Sigmoid + L0 hard concrete:
            s_i + Gumbel noise → ranked → soft top-k via sigmoid((rank - k) * τ⁻¹)
        At inference, hard top-k by score. Replaces matryoshka's first-k
        positional truncation with content-driven selection.

    Returns:
      g       [B, total_M]
      info    {"log_h", "h" (per-anchor), "b_chunks" (cross_chunk_softmax),
               "score" (content_topk), "gate_mode"}
    """

    def __init__(self, hidden_size: int, max_mem_size: int = 34,
                 init_bias: float = 5.0, gate_mode: str = "cumprod",
                 elastic_tau: float = 1.0, target_avg: float = 12.0):
        super().__init__()
        self.gate_mode = gate_mode
        self.max_mem_size = int(max_mem_size)
        self.elastic_tau = float(elastic_tau)
        self.target_avg = float(target_avg)
        h = max(hidden_size // 4, 16)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, h),
            nn.GELU(),
            nn.Linear(h, 1),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, init_bias)

    def forward(self, anchor_hidden: torch.Tensor, per_chunk_mem_sizes):
        B, total_M, D = anchor_hidden.shape
        device = anchor_hidden.device

        if self.gate_mode == "cross_chunk_softmax":
            # 1) score per chunk via mean pool
            pooled = []
            offset = 0
            for m in per_chunk_mem_sizes:
                seg = anchor_hidden[:, offset:offset + m, :].float().mean(dim=1)  # [B, D]
                pooled.append(seg)
                offset += m
            pooled_stack = torch.stack(pooled, dim=0)                              # [N_c, B, D]
            scores = self.mlp(pooled_stack).squeeze(-1)                            # [N_c, B]
            N_c = scores.shape[0]
            # 2) cross-chunk softmax → α_c
            alpha = F.softmax(scores, dim=0)                                       # [N_c, B]
            # 3) per-chunk budget; sum = N_c * target_avg
            B_total = float(N_c) * self.target_avg
            b_chunks = B_total * alpha                                             # [N_c, B]
            b_chunks = b_chunks.clamp(max=float(self.max_mem_size))
            # 4) within-chunk soft positional gate
            parts_g = []
            for ci, m in enumerate(per_chunk_mem_sizes):
                b = b_chunks[ci]                                                   # [B]
                pos = torch.arange(m, device=device, dtype=torch.float32)
                logits = (b.unsqueeze(-1) - pos.unsqueeze(0) - 0.5) / self.elastic_tau
                parts_g.append(torch.sigmoid(logits))
            g = torch.cat(parts_g, dim=-1)
            return g, {"b_chunks": b_chunks, "log_h": None, "h": None,
                       "score": None, "alpha": alpha, "gate_mode": self.gate_mode}

        if self.gate_mode == "content_topk":
            # Per-anchor score; soft top-k mask within chunk via Gumbel-Sigmoid.
            score = self.mlp(anchor_hidden.float()).squeeze(-1)                    # [B, total_M]
            return g_passthrough(score), {"log_h": None, "h": None,
                                          "b_chunks": None, "score": score,
                                          "gate_mode": self.gate_mode}

        # Per-anchor MLP path (cumprod).
        logits = self.mlp(anchor_hidden.float()).squeeze(-1)
        log_h = F.logsigmoid(logits)
        h = log_h.exp()
        parts = []
        offset = 0
        for m in per_chunk_mem_sizes:
            parts.append(torch.cumsum(log_h[:, offset:offset + m], dim=-1))
            offset += m
        log_g = torch.cat(parts, dim=-1)
        g = log_g.exp()
        return g, {"log_h": log_h, "h": h, "b_chunks": None,
                   "score": None, "gate_mode": "cumprod"}


def g_passthrough(score: torch.Tensor) -> torch.Tensor:
    """Placeholder gate for content_topk — the actual mask is built downstream
    using sample_topk_mask_gumbel() after we know k (matryoshka-controlled).
    Returns sigmoid(score) for inspection but it is NOT applied to KV directly."""
    return torch.sigmoid(score)


def sample_topk_mask_gumbel(score: torch.Tensor, k: int,
                            tau: float = 1.0, training: bool = True,
                            hard_at_eval: bool = True) -> torch.Tensor:
    """Differentiable top-k mask for a 1-D score vector.

    Training: relax via soft-rank using Gumbel noise + temperature.
    Inference: hard top-k.

    score: [m]
    k:     int, 1 <= k <= m
    Returns mask [m] in [0, 1]; expected sum ≈ k at low τ.
    """
    m = score.shape[-1]
    k = max(1, min(int(k), m))
    if not training and hard_at_eval:
        idx = torch.topk(score, k, largest=True).indices
        mask = torch.zeros_like(score)
        mask[idx] = 1.0
        return mask
    # Gumbel-perturbed scores; soft "is rank ≥ m-k" via sigmoid of margin to k-th score.
    g = -torch.log(-torch.log(torch.rand_like(score).clamp(min=1e-9))).clamp(max=10.0)
    s = (score + g) / max(tau, 1e-3)
    sorted_s, _ = torch.sort(s, descending=True)
    threshold = sorted_s[k - 1].detach()  # value at top-k boundary
    return torch.sigmoid((s - threshold) * (1.0 / max(tau, 1e-3)))


# ============================================================================
# PACT v5 — K Predictor (r2, 2026-05-25)
#
# Per-chunk integer k prediction with ε-conditioning. Replaces v4a gate which
# F13 ablation showed not to add F1 on top of Matryoshka. Architecture:
#   - Per-layer dim reduction (Linear(d_kv, d_layer_low=4)) keeps full 32-layer
#     KV signal at <1M params (vs 67M for r1 single-linear flatten).
#   - Monotone survival head: per slot the head emits a "continue" hazard logit
#     a_i; keep-prob p_i = ∏_{j<=i} sigmoid(a_j) is non-increasing by
#     construction. Training BCE acts on the SAME monotone p_i used at inference
#     (no more train/infer mismatch from the old per-slot-BCE + cummin scheme),
#     so a single noisy intermediate slot can no longer collapse the whole tail.
# Supervision: per-chunk oracle k = min{k : L(chunk_c at k, others at K_max)
#                                            − L_full < ε}.
# ε sampled discretely from {0.01, 0.05, 0.1, 0.2}, matching the evaluation grid.
# ============================================================================


K_CHOICES = [1, 2, 4, 8, 16, 34]

_LOG_HALF = math.log(0.5)


def log1mexp(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(1 - exp(x)) for x <= 0.

    Two-branch trick (Mächler 2012): use log(-expm1(x)) when x is close to 0
    and log1p(-exp(x)) when x is very negative. We clamp x just below 0 so that
    x == 0 (p_i == 1) does not produce -inf.
    """
    x = x.clamp(max=-1e-7)
    return torch.where(
        x > _LOG_HALF,
        torch.log(-torch.expm1(x)),
        torch.log1p(-torch.exp(x)),
    )


def monotone_keep_logprob(hazard_logit: torch.Tensor):
    """Survival/cumprod construction shared by loss and inference.

    hazard_logit: [..., m] raw "continue" logits a_i.
        h_i = sigmoid(a_i) in (0, 1)       # per-slot continue prob
        p_i = ∏_{j<=i} h_j                 # keep prob, monotone non-increasing
    Returns (log_p, log1m_p), each same shape, numerically stable:
        log_p   = cumsum(logsigmoid(a))    # <= 0, decreasing along last dim
        log1m_p = log(1 - exp(log_p))
    """
    log_h = F.logsigmoid(hazard_logit)               # [..., m], <= 0
    log_p = torch.cumsum(log_h, dim=-1)              # monotone decreasing
    log1m_p = log1mexp(log_p)
    return log_p, log1m_p


def independent_keep_logprob(hazard_logit: torch.Tensor):
    """Independent (non-monotone) ablation of `monotone_keep_logprob`.

    Each slot decides independently: p_i = sigmoid(a_i), with NO cumulative
    product across slots — the keep-probs are not forced to be non-increasing.
    Returns (log_p, log1m_p) = (logsigmoid(a_i), logsigmoid(-a_i)), elementwise.
    Used by the A3 ablation (predictor_head_mode="independent") to test whether
    hard-coding the ordered/residual structure (cumprod) helps vs letting it
    emerge from the (still prefix-structured) supervision labels.
    """
    log_p = F.logsigmoid(hazard_logit)
    log1m_p = F.logsigmoid(-hazard_logit)
    return log_p, log1m_p


def keep_logprob(hazard_logit: torch.Tensor, head_mode: str = "monotone"):
    """Dispatch the keep-prob parameterisation shared by loss and inference.

    head_mode="monotone"    -> monotone_keep_logprob  (cumprod survival, default)
    head_mode="independent" -> independent_keep_logprob (per-slot sigmoid)
    """
    if head_mode == "independent":
        return independent_keep_logprob(hazard_logit)
    return monotone_keep_logprob(hazard_logit)


class KPredictor(nn.Module):
    """Predict per-chunk retention via a monotone survival head.

    Input  : anchor KVs across all layers, [B, L=32, 2, total_M, d_kv=4096]
    forward: list of per-chunk hazard logits a_i, each [B, m_c] (raw, no sigmoid).
             The keep-prob is p_i = ∏_{j<=i} sigmoid(a_j) — built in
             `monotone_keep_logprob`, used identically by loss and inference.
    predict_k: count the prefix with p_i > threshold → integer k per chunk
             (p_i is already monotone, so no cummin projection is needed).
    """

    def __init__(self,
                 num_layers: int = 32,
                 d_kv: int = 4096,
                 d_layer_low: int = 4,
                 max_slots: int = 34,
                 d_hidden: int = 256,
                 attn_heads: int = 4,
                 eps_embed_dim: int = 64,
                 pos_weight: float = 4.0,
                 head_mode: str = "monotone",
                 num_kv_heads: int = None,
                 head_dim: int = None,
                 feat_mode: str = "linear4",
                 use_residual_energy: bool = False,
                 slot_mixer: str = "mha",
                 eps_mode: str = "add",
                 eps_monotone: bool = False):
        super().__init__()
        self.num_layers = num_layers
        self.d_layer_low = d_layer_low
        self.max_slots = max_slots
        # Keep-prob parameterisation: "monotone" (cumprod survival, default) or
        # "independent" (per-slot sigmoid; A3 ablation). Shared by loss+predict_k.
        self.head_mode = str(head_mode)
        # --- KV-aware redesign knobs (2026-06-29). All default to the original
        # behaviour so an unflagged KPredictor is bit-identical to the v4 module.
        #   feat_mode           : "linear4" (orig) | "kv_headstats" (Idea2)
        #   use_residual_energy : append per-slot residual-energy channel (Idea1)
        #   slot_mixer          : "mha" (orig) | "gru" (Idea3)
        #   eps_mode            : "add" (orig) | "film" (Idea4)
        #   eps_monotone        : route ε ONLY through a non-negative subtractive
        #                         term so k is monotone-decreasing in ε by
        #                         construction (Idea4); overrides eps_mode path.
        self.feat_mode = str(feat_mode)
        self.use_residual_energy = bool(use_residual_energy)
        self.slot_mixer = str(slot_mixer)
        self.eps_mode = str(eps_mode)
        self.eps_monotone = bool(eps_monotone)
        # pos_weight up-weights the "keep" term in the BCE (predictor_loss). With
        # weighted BCE the per-slot optimum shifts to p* = w·q / (1−q + w·q),
        # so the oracle boundary q>0.5 maps to p > w/(1+w), NOT p > 0.5. We store
        # this calibrated threshold and use it as predict_k's default so the
        # ε→budget calibration stays consistent between train and inference.
        # pos_weight == 4.0 default (class-balanced for ~5:1 imbalance); 1.0 → threshold 0.5 no-op.
        self.pos_weight = float(pos_weight)
        self.keep_threshold = self.pos_weight / (1.0 + self.pos_weight)
        # --- Featurization ---
        if self.feat_mode == "kv_headstats":
            # Idea2: respect (num_kv_heads, head_dim) geometry; K and V get
            # SEPARATE per-head pooled stats (norm/mean/var over head_dim) — K is
            # addressing/position, V is content (Table 4d), so they must not share
            # a projection. Per-layer in-dim = 2 (K,V) * num_kv_heads * 3 stats.
            assert num_kv_heads and head_dim, "kv_headstats needs num_kv_heads/head_dim"
            assert num_kv_heads * head_dim == d_kv, "num_kv_heads*head_dim must == d_kv"
            self.num_kv_heads = int(num_kv_heads)
            self.head_dim = int(head_dim)
            self.n_stats = 3  # L2-norm, mean, var over head_dim
            per_layer_in = 2 * self.num_kv_heads * self.n_stats
            self.layer_proj = nn.ModuleList([
                nn.Linear(per_layer_in, d_layer_low) for _ in range(num_layers)
            ])
            reduced_dim = num_layers * d_layer_low
        else:
            # "linear4" (original): one shared Linear(d_kv→d_layer_low) per layer,
            # applied to both K and V; concat across layers ×2.
            self.layer_proj = nn.ModuleList([
                nn.Linear(d_kv, d_layer_low) for _ in range(num_layers)
            ])
            reduced_dim = num_layers * 2 * d_layer_low
        self._reduced_dim = reduced_dim
        # After per-layer reduce + concat (+ optional residual-energy channel).
        n_energy = 1 if self.use_residual_energy else 0
        self.input_proj = nn.Linear(reduced_dim + n_energy, d_hidden)
        # ε-conditional embedding. "film" emits (γ, β) → 2·d_hidden; "add" → d_hidden.
        eps_out = 2 * d_hidden if self.eps_mode == "film" else d_hidden
        self.eps_proj = nn.Sequential(
            nn.Linear(1, eps_embed_dim),
            nn.GELU(),
            nn.Linear(eps_embed_dim, eps_out),
        )
        if self.eps_monotone:
            # logit_i ← logit_i − softplus(scale)·ε  : non-negative slope in ε so
            # keep-prob (hence k) is monotone non-increasing in ε by construction.
            self.eps_mono_scale = nn.Parameter(torch.zeros(1))
        # Learnable position encoding per slot index within chunk
        self.pos_emb = nn.Parameter(torch.zeros(max_slots, d_hidden))
        nn.init.normal_(self.pos_emb, std=0.02)
        # Slot mixer over the ordered slots within a chunk.
        if self.slot_mixer == "gru":
            # Idea3: causal recurrence — each slot's hazard sees a running summary
            # of the kept prefix, matching the nested/residual-code semantics.
            self.slot_gru = nn.GRU(d_hidden, d_hidden, batch_first=True)
        else:
            # Original: single-layer multi-head self-attention over slots.
            self.attn = nn.MultiheadAttention(d_hidden, num_heads=attn_heads, batch_first=True)
        self.norm = nn.LayerNorm(d_hidden)
        # Per-slot MLP head → raw "continue" hazard logit a_i.
        # Small-std weight + LARGE POSITIVE bias on the last layer: at init each
        # hazard logit ≈ +bias, so sigmoid(a_i) ≈ 1 and the cumulative keep-prob
        # p_i = ∏ sigmoid(a_j) stays high across all slots → predict_k starts
        # near max_slots ("keep everything, let training prune"). This is the
        # right prior for the survival head and avoids the early collapse to
        # tiny k seen with zero-bias init (where p_i = 0.5^i decays instantly).
        # Small-std (not zero) weight keeps grad flowing upstream from step 1.
        self.head = nn.Sequential(
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, 1),
        )
        nn.init.normal_(self.head[-1].weight, std=1e-3)
        # bias=4.0 → sigmoid≈0.982 → p_{34}≈0.982^34≈0.54 > 0.5: all slots kept
        # at init, so initial k ≈ max_slots for every chunk.
        nn.init.constant_(self.head[-1].bias, 4.0)

    def _reduce_kv(self, anchor_kv):
        """anchor_kv: [B, L, 2, total_M, d_kv] (any dtype)
           → [B, total_M, reduced_dim] in float32 (predictor is fp32 internally).

        feat_mode="linear4"      → reduced_dim = L*2*d_layer_low (original)
        feat_mode="kv_headstats" → reduced_dim = L*d_layer_low; each layer feeds
            per-head (norm/mean/var over head_dim) for K and V separately into a
            Linear, so the (num_kv_heads, head_dim) geometry is preserved instead
            of being flattened away."""
        anchor_kv = anchor_kv.float()
        B, L, KV, total_M, d_kv = anchor_kv.shape
        if self.feat_mode == "kv_headstats":
            H, hd = self.num_kv_heads, self.head_dim
            reduced = []
            for l in range(L):
                kv_l = anchor_kv[:, l]                          # [B, 2, total_M, d_kv]
                kv_l = kv_l.permute(0, 2, 1, 3).contiguous()    # [B, total_M, 2, d_kv]
                kv_l = kv_l.reshape(B, total_M, 2, H, hd)       # split heads
                norm = kv_l.norm(dim=-1)                        # [B, total_M, 2, H]
                mean = kv_l.mean(dim=-1)
                var = kv_l.var(dim=-1, unbiased=False)
                stats = torch.stack([norm, mean, var], dim=-1)  # [B, total_M, 2, H, 3]
                stats = stats.reshape(B, total_M, 2 * H * self.n_stats)
                reduced.append(self.layer_proj[l](stats))       # [B, total_M, d_layer_low]
            x = torch.stack(reduced, dim=2)                      # [B, total_M, L, d_layer_low]
            x = x.reshape(B, total_M, L * self.d_layer_low)
            return x
        reduced = []
        for l in range(L):
            kv_l = anchor_kv[:, l]                              # [B, 2, total_M, d_kv]
            kv_l = kv_l.permute(0, 2, 1, 3).contiguous()        # [B, total_M, 2, d_kv]
            kv_l = self.layer_proj[l](kv_l)                     # [B, total_M, 2, d_layer_low]
            reduced.append(kv_l)
        x = torch.stack(reduced, dim=2)                          # [B, total_M, L, 2, d_layer_low]
        x = x.reshape(B, total_M, L * 2 * self.d_layer_low)
        return x

    @staticmethod
    def _residual_energy(anchor_kv_chunk):
        """Per-slot residual / marginal-utility curve for ONE chunk's anchors.

        anchor_kv_chunk: [B, L, 2, m_c, d_kv] raw KV slice for the chunk.
        Returns: [B, m_c, 1] in [0, 1] — r_i = ‖A_i − P_{<i}A_i‖² / ‖A_i‖², the
        fraction of slot i's energy NOT already spanned by the kept prefix. This
        mirrors the oracle's marginal-loss definition (slot i is worth keeping iff
        it adds energy the prefix lacks) and is a pure detached data statistic
        (scale-robust, no learned params) → good OOD behaviour.

        Modified Gram–Schmidt over the flattened per-slot vectors; m_c ≤ 34 and
        B is typically 1, so the double loop is negligible vs the decoder passes.
        """
        x = anchor_kv_chunk.detach().float()                    # [B, L, 2, m_c, d_kv]
        B, L, KV, m_c, d_kv = x.shape
        v = x.permute(0, 3, 1, 2, 4).reshape(B, m_c, L * KV * d_kv)  # [B, m_c, D]
        out = torch.zeros(B, m_c, 1, device=v.device, dtype=v.dtype)
        for b in range(B):
            basis = []                                          # orthonormal prefix basis
            for i in range(m_c):
                vi = v[b, i]
                vi_norm = vi.norm() + 1e-8
                r = vi.clone()
                for q in basis:
                    r = r - torch.dot(q, r) * q
                rn = r.norm()
                out[b, i, 0] = (rn * rn) / (vi_norm * vi_norm)
                if rn > 1e-6:
                    basis.append(r / rn)
        return out

    def forward(self, anchor_kv, per_chunk_mem_sizes, epsilon):
        """anchor_kv: [B, L, 2, total_M, d_kv]
        per_chunk_mem_sizes: list[int] per-chunk slot counts
        epsilon: scalar or [B] tensor
        Returns: list of [B, m_c] raw hazard logits a_i (no sigmoid, no cumprod).
                 The monotone keep-prob p_i is derived downstream by both the
                 loss and predict_k via `monotone_keep_logprob`.
        """
        B = anchor_kv.shape[0]
        x_red = self._reduce_kv(anchor_kv)                      # [B, total_M, reduced_dim] (fp32)

        if epsilon.dim() == 0:
            epsilon = epsilon.expand(B)
        eps_col = epsilon.view(B, 1).float()
        if not self.eps_monotone:
            cond = self.eps_proj(eps_col)                       # [B, d_hidden] or [B, 2*d_hidden]

        logit_list = []
        offset = 0
        for m_c in per_chunk_mem_sizes:
            xr = x_red[:, offset:offset + m_c, :]              # [B, m_c, reduced_dim]
            if self.use_residual_energy:
                e = self._residual_energy(
                    anchor_kv[:, :, :, offset:offset + m_c, :]
                )                                              # [B, m_c, 1]
                xr = torch.cat([xr, e], dim=-1)
            chunk_x = self.input_proj(xr)                      # [B, m_c, d_hidden]
            # ε conditioning
            if not self.eps_monotone:
                if self.eps_mode == "film":
                    gamma, beta = cond.chunk(2, dim=-1)        # each [B, d_hidden]
                    chunk_x = chunk_x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
                else:
                    chunk_x = chunk_x + cond.unsqueeze(1)
            chunk_x = chunk_x + self.pos_emb[:m_c].unsqueeze(0)
            # slot mixer
            if self.slot_mixer == "gru":
                mixed, _ = self.slot_gru(chunk_x)
            else:
                mixed, _ = self.attn(chunk_x, chunk_x, chunk_x)
            chunk_x = self.norm(chunk_x + mixed)
            logit = self.head(chunk_x).squeeze(-1)             # [B, m_c]
            if self.eps_monotone:
                # subtractive, non-negative slope in ε → k monotone-decreasing in ε
                logit = logit - F.softplus(self.eps_mono_scale) * eps_col
            logit_list.append(logit)
            offset += m_c
        return logit_list

    def predict_k(self, anchor_kv, per_chunk_mem_sizes, epsilon, threshold: float = None,
                  min_k: int = 1):
        """Inference: count the monotone-keep-prob prefix above threshold.

        k_c = #{i : p_i > threshold}, where p_i = ∏_{j<=i} sigmoid(a_j) is
        non-increasing by construction — so this is exactly the leading run of
        "keep" slots, with no cummin projection needed (train/infer aligned).

        threshold=None (default) uses the pos_weight-calibrated boundary
        w/(1+w), which matches the oracle q>0.5 decision under the weighted BCE
        used in training. Pass an explicit float only to override calibration.

        Returns:
            ks: list[list[int]], outer len = N_chunks, inner len = B.
                Each value in [min_k, m_c].
            ks_b0: list[int] convenience — first-batch view (one int per chunk),
                use this when B==1.
        """
        if threshold is None:
            threshold = self.keep_threshold
        logit_list = self.forward(anchor_kv, per_chunk_mem_sizes, epsilon)
        ks = []
        for logit in logit_list:
            log_p, _ = keep_logprob(logit, self.head_mode)      # [B, m_c]
            p = log_p.exp()
            # Leading run of "keep" slots: k = length of the prefix above
            # threshold. For monotone p (non-increasing) this equals
            # (p>threshold).sum() exactly (bit-identical to the original); for
            # the independent head it stops at the first below-threshold slot,
            # preserving the prefix-truncation semantics of compress().
            above = (p > threshold).long()                      # [B, m_c]
            ks_chunk = torch.cumprod(above, dim=-1).sum(dim=-1)  # [B]; prefix length
            ks_chunk = ks_chunk.clamp(min=min_k).cpu().tolist()
            ks.append(ks_chunk)
        ks_b0 = [row[0] for row in ks]
        return ks, ks_b0

    def predict_k_naive(self, anchor_kv, per_chunk_mem_sizes, global_budget,
                        k_choices, min_k: int = 1):
        """Naive-Gate inference: score all slots, spend ONE global fixed budget
        greedily top-k, snap per-chunk counts to the K_set ladder.

        Score = monotone keep-prob p_i (same head as the eps path), but with a
        NEUTRAL epsilon (0.0) so the eps-conditioning embedding contributes a
        constant offset and the allocation is driven purely by content scores +
        the fixed global budget. No per-chunk eps tolerance.
        """
        B = anchor_kv.shape[0]
        eps0 = torch.zeros(B, device=anchor_kv.device, dtype=torch.float32)
        logit_list = self.forward(anchor_kv, per_chunk_mem_sizes, eps0)
        scores_per_chunk = []
        for logit in logit_list:
            log_p, _ = keep_logprob(logit, self.head_mode)   # [B, m_c]
            scores_per_chunk.append(log_p.exp())             # keep-prob as score
        return naive_global_topk_k_per_chunk(
            scores_per_chunk, per_chunk_mem_sizes, global_budget, k_choices,
            min_k=min_k,
        )


def extract_full_kv(encoder_past_key_values):
    """encoder_past_key_values: tuple[(K_l, V_l)] for L layers, each
        K_l, V_l: [B, num_heads, total_M, head_dim]
    Returns: [B, L, 2, total_M, num_heads * head_dim]
    """
    stacked = []
    for K_l, V_l in encoder_past_key_values:
        B, NH, M, HD = K_l.shape
        K_flat = K_l.transpose(1, 2).reshape(B, M, NH * HD)
        V_flat = V_l.transpose(1, 2).reshape(B, M, NH * HD)
        KV = torch.stack([K_flat, V_flat], dim=1)               # [B, 2, M, NH*HD]
        stacked.append(KV)
    return torch.stack(stacked, dim=1)                          # [B, L, 2, M, NH*HD]


def _build_keep_indices(per_chunk_mem_sizes, k_per_chunk, device):
    """Build a flat index tensor selecting the first k_c anchors per chunk
    from a concatenated [total_M] axis.
    """
    keep_idx_parts = []
    offset = 0
    for m, k in zip(per_chunk_mem_sizes, k_per_chunk):
        k_eff = min(int(k), int(m))
        keep_idx_parts.append(torch.arange(offset, offset + k_eff, device=device))
        offset += m
    return torch.cat(keep_idx_parts, dim=0)


def truncate_kv_to_k(encoder_past_key_values, k, per_chunk_mem_sizes):
    """Truncate per-chunk anchors. `k` can be scalar (applies to all chunks)
    or list[int] (per-chunk).
    Returns tuple of (K, V) tensors with total kept dim across chunks.
    """
    if isinstance(k, int):
        k_per_chunk = [k] * len(per_chunk_mem_sizes)
    else:
        k_per_chunk = list(k)
        assert len(k_per_chunk) == len(per_chunk_mem_sizes)
    device = encoder_past_key_values[0][0].device
    keep_idx = _build_keep_indices(per_chunk_mem_sizes, k_per_chunk, device)
    return tuple(
        (K.index_select(2, keep_idx), V.index_select(2, keep_idx))
        for (K, V) in encoder_past_key_values
    )


def truncate_single_chunk_kv(encoder_past_key_values, per_chunk_mem_sizes,
                              target_chunk: int, k: int):
    """Truncate ONLY `target_chunk` to first k anchors; other chunks kept whole.
    Used for per-chunk oracle measurement.
    """
    k_per_chunk = list(per_chunk_mem_sizes)
    k_per_chunk[target_chunk] = min(int(k), int(per_chunk_mem_sizes[target_chunk]))
    return truncate_kv_to_k(encoder_past_key_values, k_per_chunk, per_chunk_mem_sizes)


def naive_global_topk_k_per_chunk(scores_per_chunk, per_chunk_mem_sizes,
                                  global_budget, k_choices, min_k: int = 1):
    """Naive-Gate allocation rule (ablation variant, gate_mode="naive").

    Unlike the eps-Gate (per-chunk learned budget with an epsilon tolerance),
    the naive gate scores every anchor slot and spends a SINGLE GLOBAL FIXED
    budget greedily top-k across all chunks, then snaps each chunk's raw count
    to the nearest k in the K_set ladder.

    Concretely:
      B_total = round(global_budget * N_chunks)          # total slots to keep
      rank all slots (across chunks) by score desc, keep the top B_total
      raw_k_c = #(kept slots in chunk c)                  # >=0, <= m_c
      k_c = snap raw_k_c to the NEAREST k in k_choices (tie -> larger),
            clamped to [min_k, m_c]; 0 -> min_k.
    Nearest-snapping keeps the realized mean_k tracking global_budget with
    small near-unbiased error (snap-UP would over-spend +1..+3); snapping to the
    ladder keeps the kept-set on the same {1,2,4,8,16,32} grid the matryoshka
    backbone was trained on (so the naive truncation reuses the front-loaded
    anchors exactly like the eps path).

    scores_per_chunk: list len N_chunks of [B, m_c] float score tensors
                      (higher = more important; here = monotone keep-prob p_i).
    per_chunk_mem_sizes: list[int] m_c per chunk.
    global_budget: float, target mean slots-per-chunk (e.g. 7.37 for ~64x).
    k_choices: sorted K_set ladder, e.g. [1,2,4,8,16,32].
    Returns:
      ks: list[list[int]] outer len N_chunks, inner len B.
      ks_b0: list[int] first-batch view (one int per chunk), use when B==1.
    """
    N = len(per_chunk_mem_sizes)
    B = scores_per_chunk[0].shape[0]
    device = scores_per_chunk[0].device
    sorted_ladder = sorted(int(k) for k in k_choices)
    B_total = int(round(float(global_budget) * N))
    B_total = max(N * min_k, B_total)  # at least min_k per chunk on aggregate

    def snap_near(raw, m):
        # Snap raw count to the NEAREST ladder rung (tie -> larger). Nearest
        # (vs snap-up) keeps realized mean_k tracking global_budget with small,
        # near-unbiased error (~|0.8| over the ladder), which is what the T8
        # "mean compression matched to T1" requirement needs; snap-up would
        # systematically over-spend by +1 to +3.
        raw = int(raw)
        if raw <= min_k:
            return min_k
        best = min(sorted_ladder, key=lambda k: (abs(k - raw), -k))
        return min(best, int(m))

    ks = [[0] * B for _ in range(N)]
    for b in range(B):
        # Flatten (chunk, slot)->score; rank desc, take top B_total.
        flat_scores = []
        flat_chunk = []
        for c, m in enumerate(per_chunk_mem_sizes):
            s = scores_per_chunk[c][b]                       # [m_c]
            flat_scores.append(s)
            flat_chunk.append(torch.full((int(m),), c, device=device, dtype=torch.long))
        flat_scores = torch.cat(flat_scores, dim=0)          # [total_M]
        flat_chunk = torch.cat(flat_chunk, dim=0)
        total_M = flat_scores.numel()
        n_keep = min(B_total, total_M)
        if n_keep > 0:
            top_idx = torch.topk(flat_scores, n_keep, largest=True).indices
            kept_chunks = flat_chunk.index_select(0, top_idx)
            raw_counts = torch.bincount(kept_chunks, minlength=N).cpu().tolist()
        else:
            raw_counts = [0] * N
        for c, m in enumerate(per_chunk_mem_sizes):
            ks[c][b] = snap_near(raw_counts[c], m)
    ks_b0 = [row[0] for row in ks]
    return ks, ks_b0


def compute_oracle_k(losses_per_k, L_full, epsilon, k_choices=K_CHOICES,
                      relative: bool = False, eps_floor: float = 1e-3):
    """Per-sample oracle k = min{k ∈ k_choices : criterion(k) < ε}, default K_max.

    losses_per_k: dict {k_int: [B] tensor}
    L_full:       [B] reference loss
    epsilon:      [B] tensor
    relative:     if True, criterion = (L_k - L_full) / max(L_full, eps_floor);
                  otherwise criterion = L_k - L_full (absolute, original).
    eps_floor:    floor for L_full denominator under relative mode (prevents
                  divide-by-near-zero on samples the model already aces).
    Returns:      [B] long tensor
    """
    B = L_full.shape[0]
    K_max = max(k_choices)
    oracle = torch.full((B,), K_max, device=L_full.device, dtype=torch.long)
    if relative:
        denom = L_full.clamp(min=eps_floor)
    for k in sorted(k_choices):
        if k not in losses_per_k:
            continue
        if relative:
            criterion = (losses_per_k[k] - L_full) / denom
        else:
            criterion = losses_per_k[k] - L_full
        satisfied = criterion < epsilon
        not_yet = (oracle == K_max)
        update = satisfied & not_yet
        oracle = torch.where(update, torch.full_like(oracle, k), oracle)
    return oracle


def compute_oracle_k_per_chunk(losses_per_chunk_per_k, L_full, epsilon,
                                k_choices=K_CHOICES,
                                relative: bool = False, eps_floor: float = 1e-3):
    """Per-chunk oracle. Forwards relative/eps_floor to compute_oracle_k.
    losses_per_chunk_per_k: outer list len=N_chunks, each entry is
                             dict {k_int: [B] tensor}
    Returns: list of [B] long tensors, length N_chunks.
    """
    N_chunks = len(losses_per_chunk_per_k)
    out = []
    for c in range(N_chunks):
        out.append(compute_oracle_k(
            losses_per_chunk_per_k[c], L_full, epsilon, k_choices,
            relative=relative, eps_floor=eps_floor,
        ))
    return out


def oracle_to_per_slot_labels(oracle_k, max_slots: int):
    """oracle_k: [B] long; max_slots: m_c
    Returns: [B, m_c] float labels, y[b, i] = 1 if oracle_k[b] >= i+1 else 0.
    """
    idx = torch.arange(max_slots, device=oracle_k.device).unsqueeze(0) + 1   # [1, m_c]
    return (oracle_k.unsqueeze(-1) >= idx).float()


def _within_sample_rank_loss(pred_k, oracle_k, mode="corr"):
    """Within-sample (across-chunk) ranking loss between the predictor's
    continuous expected-k and the oracle k. pred_k, oracle_k: [N_chunks] for one
    batch element. Scale/mean-INVARIANT, so it optimises the per-chunk RELATIVE
    allocation (the elastic-budget ordering) without touching mean-k — the
    complement to the pointwise BCE, which only fixes calibration.

    mode="corr"     : 1 − Pearson(pred_k, oracle_k)            (listwise)
    mode="pairwise" : mean softplus(−(Δpred)·sign(Δoracle))    over chunk pairs
    Returns a scalar tensor (0 if <2 chunks or no oracle variation).
    """
    N = pred_k.shape[0]
    if N < 2:
        return pred_k.sum() * 0.0
    if mode == "pairwise":
        di = pred_k.unsqueeze(0) - pred_k.unsqueeze(1)          # [N,N] pred_j - pred_i
        do = oracle_k.unsqueeze(0) - oracle_k.unsqueeze(1)
        mask = (do != 0)
        if mask.sum() == 0:
            return pred_k.sum() * 0.0
        # encourage sign(Δpred) to match sign(Δoracle)
        loss = F.softplus(-torch.sign(do) * di)
        return loss[mask].mean()
    # default: correlation loss
    po = oracle_k - oracle_k.mean()
    pp = pred_k - pred_k.mean()
    denom = (pp.norm() * po.norm())
    if denom < 1e-8:
        return pred_k.sum() * 0.0     # no variation → undefined, skip
    corr = (pp * po).sum() / denom
    return 1.0 - corr


def predictor_loss(logit_list, per_chunk_mem_sizes, oracle_per_chunk,
                   pos_weight: float = 4.0, head_mode: str = "monotone",
                   rank_weight: float = 0.0, rank_mode: str = "corr"):
    """Weighted BCE on the MONOTONE keep-prob p_i (the same quantity predict_k
    thresholds), so training and inference optimise one target.

    For each slot: loss = -(pos_weight * y * log p_i + (1 - y) * log(1 - p_i)),
    with step labels y_i = 1[oracle_k >= i+1] from `oracle_to_per_slot_labels`.

    logit_list: list of [B, m_c] raw hazard logits (KPredictor.forward output)
    per_chunk_mem_sizes: list[int] m_c per chunk
    oracle_per_chunk: list of [B] integer oracle k per chunk
    pos_weight: up-weight on the positive (keep) term. Oracle k collapses small,
        so keep-labels are the minority — pos_weight > 1 counteracts the bias
        toward predicting tiny k, especially at high slot indices.
    rank_weight: weight on an added WITHIN-SAMPLE (across-chunk) ranking loss
        (default 0.0 == original pointwise BCE only). The pointwise BCE has no
        cross-chunk term, so with detached features it collapses toward the per-ε
        mean-k and loses per-chunk discrimination (F101/F102). The ranking loss
        directly optimises the relative per-chunk allocation (within-sample
        Spearman) and is mean/scale-invariant, so it complements the BCE.
    rank_mode: "corr" (1−Pearson, listwise) | "pairwise" (margin over pairs).
    """
    total_loss = 0.0
    total_slots = 0
    pred_k_chunks = []     # continuous expected-k per chunk: [B] each
    for logit, m_c, oracle_c in zip(logit_list, per_chunk_mem_sizes, oracle_per_chunk):
        target = oracle_to_per_slot_labels(oracle_c, m_c).to(logit.device)   # [B, m_c]
        log_p, log1m_p = keep_logprob(logit, head_mode)
        loss = -(pos_weight * target * log_p + (1.0 - target) * log1m_p)
        total_loss = total_loss + loss.sum()
        total_slots += logit.numel()
        if rank_weight > 0.0:
            pred_k_chunks.append(log_p.exp().sum(dim=-1))   # E[#kept] = Σ p_i, [B]
    bce = total_loss / max(total_slots, 1)
    if rank_weight <= 0.0 or len(pred_k_chunks) < 2:
        return bce
    pred_k = torch.stack(pred_k_chunks, dim=0)              # [N_chunks, B]
    oracle_k = torch.stack(oracle_per_chunk, dim=0).to(pred_k.dtype)  # [N_chunks, B]
    B = pred_k.shape[1]
    rl = sum(_within_sample_rank_loss(pred_k[:, b], oracle_k[:, b], rank_mode)
             for b in range(B)) / max(B, 1)
    return bce + rank_weight * rl


class CompressLLM(torch.nn.Module):
    def __init__(self, model_id, mem_size, compress_ratio, device_rank, task_config):
        super().__init__()
        self.loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # CPU debugging support: pass device_rank="cpu" to load on CPU in fp32
        # (bf16 on CPU is slow/unreliable on x86).
        is_cpu = (str(device_rank) == "cpu")
        device_map = "cpu" if is_cpu else f"cuda:{device_rank}"
        dtype = torch.float32 if is_cpu else torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
        self.decoder = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=dtype,
            device_map=device_map,
        )
        freeze_decoder(self.decoder)
        # Optional gradient checkpointing on the encoder (saves ~30-40% peak
        # activation memory; ~20% slowdown). Decoder is frozen, so no need.
        if bool(task_config.get("gradient_checkpointing", False)):
            self.model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": False}
            )
        self.device = device_map
        self.task_config = task_config
        config = self.model.config
        self.vocab_size = config.vocab_size
        self.chunk_size = task_config["chunk_size"]
        self.role_tokens = nn.Parameter(self.model.model.embed_tokens.weight.new_zeros((mem_size, config.hidden_size)),requires_grad=True)
        self.special_tokens = nn.Parameter(self.model.model.embed_tokens.weight.new_zeros((2, config.hidden_size)), requires_grad=True)
        self.compress_ratio = compress_ratio
        self.mem_size = mem_size

        mean = torch.mean(self.model.model.embed_tokens.weight).item()
        std = torch.std(self.model.model.embed_tokens.weight).item()
        nn.init.normal_(self.role_tokens, mean=0.0, std=0.02)
        nn.init.normal_(self.special_tokens, mean=mean, std=std)

        self.use_gate = bool(task_config.get("use_gate", False))
        self.gate_mode = str(task_config.get("gate_mode", "cumprod"))
        if self.use_gate:
            # cross_chunk_softmax: init_bias is irrelevant (softmax normalizes); use 0.
            # content_topk: init_bias=0 → uniform priors before learning.
            # cumprod: 5.0 to keep gate ≈1 early in training (legacy v3).
            init_bias_default = {
                "cumprod": 5.0,
                "cross_chunk_softmax": 0.0,
                "content_topk": 0.0,
            }.get(self.gate_mode, 5.0)
            self.gate_predictor = GatePredictor(
                config.hidden_size,
                max_mem_size=int(task_config.get("mem_size", 34)),
                init_bias=float(task_config.get("gate_init_bias", init_bias_default)),
                gate_mode=self.gate_mode,
                elastic_tau=float(task_config.get("elastic_tau", 1.0)),
                target_avg=float(task_config.get("target_avg", 12.0)),
            ).to(self.device)
        else:
            self.gate_predictor = None
        # Gumbel temperature for content_topk; annealed externally via task_config.
        self._gumbel_tau = float(task_config.get("gumbel_tau", 1.0))

        # PACT v5 K-Predictor (r2). Not mutually exclusive with gate, but typically
        # use_gate=False when use_k_predictor=True (gate has been ablation-falsified).
        self.use_k_predictor = bool(task_config.get("use_k_predictor", False))
        # PACT naive-Gate ablation: "epsilon" (default eps-tolerance learned
        # per-chunk budget) vs "naive" (per-chunk score head + ONE global
        # fixed-budget top-k allocation, no eps tolerance). T8 ablation.
        self.predictor_gate_mode = str(task_config.get("predictor_gate_mode", "epsilon"))
        self.naive_global_budget = task_config.get("naive_global_budget", None)
        if self.use_k_predictor:
            mcfg = self.model.config
            num_kv_heads = getattr(mcfg, "num_key_value_heads", mcfg.num_attention_heads)
            head_dim = mcfg.hidden_size // mcfg.num_attention_heads
            d_kv = num_kv_heads * head_dim
            # KV-aware redesign presets (2026-06-29): predictor_arch is a
            # convenience that expands to the composable flags below; explicit
            # flags still override. "kv_marginal" = Idea1+Idea2 (recommended).
            _arch = str(task_config.get("predictor_arch", "default"))
            _preset = {
                "default":     {},
                "kv_marginal": {"feat_mode": "kv_headstats", "use_residual_energy": True},
            }.get(_arch, {})
            _feat_cfg = task_config.get("predictor_feat_mode", None)
            _feat_mode = str(_feat_cfg) if _feat_cfg is not None \
                else _preset.get("feat_mode", "linear4")
            _res_cfg = task_config.get("predictor_use_residual_energy", None)
            _use_res = bool(_res_cfg) if _res_cfg is not None \
                else _preset.get("use_residual_energy", False)
            self.k_predictor = KPredictor(
                num_layers=mcfg.num_hidden_layers,
                d_kv=d_kv,
                d_layer_low=int(task_config.get("predictor_d_layer_low", 4)),
                max_slots=int(task_config.get("mem_size", mem_size)),
                d_hidden=int(task_config.get("predictor_hidden", 256)),
                attn_heads=int(task_config.get("predictor_attn_heads", 4)),
                eps_embed_dim=int(task_config.get("predictor_eps_embed_dim", 64)),
                pos_weight=float(task_config.get("predictor_pos_weight", 4.0)),
                head_mode=str(task_config.get("predictor_head_mode", "monotone")),
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                feat_mode=_feat_mode,
                use_residual_energy=_use_res,
                slot_mixer=str(task_config.get("predictor_slot_mixer", "mha")),
                eps_mode=str(task_config.get("predictor_eps_mode", "add")),
                eps_monotone=bool(task_config.get("predictor_eps_monotone", False)),
            ).to(self.device)   # fp32 (small ~1M params)
        else:
            self.k_predictor = None
        # Trainer sets this each step for warmup gating.
        self._global_step = 0

    def _sample_matryoshka_k(self):
        if not (self.task_config.get("use_matryoshka", False) and self.training):
            return None
        ks = self.task_config.get("matryoshka_ks", [2, 4, 8, 16, 32, self.mem_size])
        weights = self.task_config.get("matryoshka_weights", [4.0, 3.0, 2.0, 1.5, 1.0, 1.0])
        return random.choices(ks, weights=weights, k=1)[0]

    def forward(self,inputs):
        # PACT v5: dispatch to dedicated branch when KPredictor is active.
        if self.use_k_predictor:
            return self._forward_k_predictor(inputs)

        tot_loss = 0
        tot_task = 0
        loss_info = {}

        # Matryoshka: sample per-chunk truncation k per step (training only).
        truncate_k = self._sample_matryoshka_k()
        if truncate_k is not None:
            loss_info["matryoshka_k"] = truncate_k

        # For content_topk gate: keep ALL anchors in compress; gate's top-k mask
        # (driven by matryoshka_k) controls how many survive — content-aware,
        # not positional. For other modes, compress() does first-k truncation
        # (matryoshka's positional front-loading prior).
        use_content_topk = (self.use_gate and self.gate_mode == "content_topk")
        compress_truncate_k = None if use_content_topk else truncate_k

        # context position ids:[1,......,end_idx]
        (compress_token_ids, compress_token, end_idx,
         mem_last_hidden, encoder_past_key_values, encoder_mem_size,
         per_chunk_mem_sizes) = self.compress(inputs, truncate_k=compress_truncate_k)

        # Gate: soft-multiply K and V by g (per anchor, per chunk).
        gate_g = None
        gate_info = None
        if self.use_gate and self.gate_predictor is not None:
            gate_g, gate_info = self.gate_predictor(mem_last_hidden, per_chunk_mem_sizes)

            if self.gate_mode == "content_topk":
                # Build per-chunk soft top-k mask using gate score + matryoshka k.
                # k per chunk = min(truncate_k or mem_size, m). The gate's *score*
                # decides WHICH anchors survive (not first-k).
                score_full = gate_info["score"][0]  # [total_M] (B=1 in this path)
                mask_parts = []
                _off = 0
                k_target = int(truncate_k) if truncate_k is not None else self.mem_size
                for m in per_chunk_mem_sizes:
                    seg_score = score_full[_off:_off + m]
                    k_eff = max(1, min(k_target, m))
                    seg_mask = sample_topk_mask_gumbel(
                        seg_score, k_eff, tau=self._gumbel_tau,
                        training=self.training, hard_at_eval=True
                    )
                    mask_parts.append(seg_mask)
                    _off += m
                gate_g = torch.cat(mask_parts, dim=-1).unsqueeze(0)  # [1, total_M]
                gate_info["mask_mean"] = float(gate_g.mean().detach().cpu())

            g_for_kv = gate_g.to(encoder_past_key_values[0][0].dtype)
            g_kv = g_for_kv[:, None, :, None]
            encoder_past_key_values = tuple(
                (k * g_kv, v * g_kv) for (k, v) in encoder_past_key_values
            )
            sum_g_chunks = []
            _off = 0
            for m in per_chunk_mem_sizes:
                sum_g_chunks.append(gate_g[:, _off:_off + m].sum(dim=-1))
                _off += m
            sum_g_stack = torch.stack(sum_g_chunks, dim=0)
            mean_sum_g = sum_g_stack.mean()
            loss_info["gate_E_sum_g"] = float(mean_sum_g.detach().cpu())
            loss_info["gate_N_kept"] = float((gate_g > 0.1).float().sum(dim=-1).mean().detach().cpu())
            if gate_info.get("h") is not None:
                loss_info["gate_mean_h"] = float(gate_info["h"].mean().detach().cpu())
            if gate_info.get("b_chunks") is not None:
                loss_info["gate_b_mean"] = float(gate_info["b_chunks"].mean().detach().cpu())
                loss_info["gate_b_std"] = float(gate_info["b_chunks"].std().detach().cpu())
            if gate_info.get("alpha") is not None:
                loss_info["gate_alpha_entropy"] = float(
                    -(gate_info["alpha"].clamp(min=1e-9) * gate_info["alpha"].clamp(min=1e-9).log()).sum(dim=0).mean().detach().cpu()
                )

##########################################################AE Task########################################################################

        if self.task_config["is_pretrain"] and self.task_config["use_ae_loss"]:
            # print("AE Task")
            inputs_embeds = self.decoder.model.embed_tokens(inputs['ae_targets'])  
            bsz, seq_len, emb_size = inputs_embeds.size()
            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_ae_token = self.special_tokens[0:1].unsqueeze(0).expand(bsz, 1, emb_size)
            # [B,mem_size,E];     [B,1,E];      [B,seq_len-1,E]
            ae_emb = torch.cat([expand_ae_token, inputs_embeds[:, :-1, :]], dim=1)

            # ae_pids:[0], ae_target[:, :-1] pids:[1,......,end_idx] because drop the last token to predict [1,...,end_idx+1] the last one is <eos>.
            position_ids = torch.arange(0, inputs_embeds.size(1), device=inputs_embeds.device).unsqueeze(0)
            ae_position_ids = position_ids
            # print(f"ae_position_ids:{ae_position_ids}")
            if self.task_config["use_pe"]:
                outputs = self.decoder(position_ids=ae_position_ids, inputs_embeds=ae_emb,past_key_values=encoder_past_key_values)
            else:
                outputs = self.decoder(inputs_embeds=ae_emb)
            # [B,mem_size+S,V] -> [B,S,V]
            logits = outputs.logits
            inputs['ae_targets'] = inputs['ae_targets'].contiguous().view(-1).to(logits.device)
            ae_loss = self.loss_fct(logits.contiguous().view(-1, self.vocab_size), inputs['ae_targets'])  # [ae]+context[:-1] -> context[:]
            loss_info["ae_loss"] = ae_loss.item()

            if "ae_weight" not in self.task_config:
                tot_loss += ae_loss
                tot_task += 1
            else:
                # print(f"ae_weight:{self.task_config['ae_weight']}; tot_task:{tot_task}")
                tot_loss += ae_loss*self.task_config["ae_weight"]
                tot_task += self.task_config["ae_weight"]
                # print(f"ae_weight:{self.task_config['ae_weight']}; tot_task:{tot_task}")
        
#######################################################LM Task################################################################################

        if self.task_config["is_pretrain"] and self.task_config["use_lm_loss"]:
            # print("LM Task")
            lm_target_emb = self.decoder.model.embed_tokens(inputs['lm_targets'][:, :-1])
            bsz, seq_len, emb_size = lm_target_emb.size()
            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)
            lm_emb = torch.cat([expand_lm_token, lm_target_emb],dim=1)

            # context pids:[1,......,end_idx] 
            # lm_pids:[end_idx], lm_target_pids:[end_idx+1,......]
            latter_position_ids = torch.arange(end_idx,end_idx+seq_len+1,device=lm_target_emb.device).unsqueeze(0)
            lm_position_ids = latter_position_ids
            # print(f"lm_position_ids:{lm_position_ids}")
            if self.task_config["use_pe"]:
                outputs = self.decoder(inputs_embeds=lm_emb, position_ids=lm_position_ids,past_key_values=encoder_past_key_values)
            else:
                outputs = self.decoder(inputs_embeds=lm_emb)
            # [B,mem_size+S,V] -> [B,S,V]
            logits = outputs.logits
            logits = logits.contiguous().view(-1, self.vocab_size)
            inputs['lm_targets'] = inputs['lm_targets'].contiguous().view(-1).to(logits.device)
            lm_loss = self.loss_fct(logits, inputs['lm_targets'])
            loss_info["lm_loss"] = lm_loss.item()

            if "lm_weight" not in self.task_config:
                tot_loss += lm_loss
                tot_task += 1
            else:
                # print(f"lm_weight:{self.task_config['lm_weight']}; tot_task:{tot_task}")
                tot_loss += lm_loss*self.task_config["lm_weight"]
                tot_task += self.task_config["lm_weight"]           
                # print(f"lm_weight:{self.task_config['lm_weight']}; tot_task:{tot_task}")


######################################################QA Task####################################################################
        # LM loss
        if self.task_config["is_sft"] and self.task_config["use_lm_loss"]:
            # print("QA Task")
            lm_target_emb = self.decoder.model.embed_tokens(inputs['lm_targets'][:, :-1])
            bsz, seq_len, emb_size = lm_target_emb.size()
            # [1,E] -> [1,1,E] -> [B,1,E]
            expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)
            lm_emb = torch.cat([expand_lm_token,lm_target_emb],dim=1)
            # context position ids:[1,......,end_idx];
            #                                         [LM] position ids:[end_idx];  QA position ids:[end_idx+1,.......]
            latter_position_ids = torch.arange(end_idx,end_idx+seq_len+1,device=lm_target_emb.device).unsqueeze(0)
            lm_position_ids = latter_position_ids
            # print(f"lm_position_ids:{lm_position_ids}")
            if self.task_config["use_pe"]:
                outputs = self.decoder(inputs_embeds=lm_emb, position_ids=lm_position_ids,past_key_values=encoder_past_key_values)
            else:
                outputs = self.decoder(inputs_embeds=lm_emb)
            # [B,mem_size+S,V] -> [B,S,V]
            logits = outputs.logits

            #  in prepare_data.py, we drop the fisrt -100, so here we drop the [LM]'s logits which is used to predict the fisrt -100.
            #  but it's no influence because -100 are not used to calculate the loss.
            logits = logits[:, 1:]    
            logits = logits.contiguous().view(-1, self.vocab_size)
            inputs["instruction_target"] = inputs["instruction_target"].contiguous().view(-1).to(logits.device)
            lm_loss = self.loss_fct(logits, inputs["instruction_target"])
            loss_info["lm_loss"] = lm_loss.item()
            tot_loss += lm_loss
            tot_task += 1


        loss = tot_loss / tot_task

        # Optional sparsity loss. cross_chunk_softmax does NOT need it
        # (budget is structurally constrained by softmax). content_topk also
        # does not need it (k controlled by matryoshka). Cumprod path keeps
        # backward compat — no extra loss by default.
        sparsity_weight = float(self.task_config.get("sparsity_weight", 0.0))
        if self.use_gate and gate_info is not None and sparsity_weight > 0.0:
            if gate_info.get("b_chunks") is not None:
                norm = float(self.gate_predictor.max_mem_size)
                sp = (gate_info["b_chunks"].float() / norm).mean()
            elif gate_info.get("score") is not None:
                sp = torch.sigmoid(gate_info["score"].float()).mean()
            else:
                sp = None
            if sp is not None:
                loss = loss + sparsity_weight * sp
                loss_info["sparsity_loss"] = float(sp.detach().cpu())

        # Entropy penalty (kept for backward compat with v3 cumprod path).
        entropy_weight = float(self.task_config.get("entropy_weight", 0.0))
        if self.use_gate and gate_info is not None and gate_info.get("log_h") is not None and entropy_weight > 0.0:
            h = gate_info["log_h"].exp().clamp(min=1e-6, max=1 - 1e-6)
            ent = -(h * h.log() + (1 - h) * (1 - h).log()).mean()
            loss = loss + entropy_weight * ent
            loss_info["entropy"] = float(ent.detach().cpu())

        loss_info["loss"] = float(loss.detach().cpu())
        return {"loss": loss, "loss_info": loss_info}

    def compute_num_chunks(self, total_length):
        assert total_length > 0
        num_chunks = math.ceil(total_length / self.chunk_size)
        return num_chunks

    def get_uniform_position_ids(self, x_1, x_n, ratio):
        start = (x_1 + (ratio - 1) // 2)
        end = x_n
        if start >= end: # a length can arrive as 513 > 512; the real length is 512, so start at 511
            start = x_1
        return torch.arange(start, end, step=ratio, device=self.device).unsqueeze(0)

    def compress(self, inputs, truncate_k=None, truncate_k_per_chunk=None):
        """Compress context tokens into per-chunk anchor KV cache.

        truncate_k: int | list[int] | None
            - int: same k applied to all chunks (legacy Matryoshka behavior).
            - list[int]: per-chunk k (PACT v5; e.g. KPredictor.predict_k output).
            - None: keep all anchors.
        truncate_k_per_chunk: explicit list form (preferred over scalar when both set).

        Returns:
            compress_token_ids: [1, total_M] anchor position ids (after truncation)
            compress_token:     [B, total_M, D] embedding-layer hidden at anchor positions
            end_idx:            position id after last context token (unchanged by k)
            mem_last_hidden:    [B, total_M, D] LAST-layer hidden at anchor positions (for gate)
            encoder_past_key_values: tuple of (K, V) per layer, mem-dim = total_M
            encoder_mem_size:   total_M = sum(per_chunk_mem_sizes)
            per_chunk_mem_sizes: list[int] kept anchors per chunk (post-truncation)
        """
        bsz, total_length = inputs['input_ids'].size()
        num_chunks = self.compute_num_chunks(total_length)
        chunk_size = self.chunk_size

        # Normalize truncation spec to either None or a per-chunk list of ints.
        if truncate_k_per_chunk is not None:
            k_list = list(truncate_k_per_chunk)
            assert len(k_list) == num_chunks, (
                f"truncate_k_per_chunk length {len(k_list)} != num_chunks {num_chunks}"
            )
        elif isinstance(truncate_k, (list, tuple)):
            k_list = list(truncate_k)
            assert len(k_list) == num_chunks, (
                f"truncate_k (list) length {len(k_list)} != num_chunks {num_chunks}"
            )
        elif truncate_k is None:
            k_list = None
        else:
            k_list = [int(truncate_k)] * num_chunks

        end_idx = 0
        encoder_mem_size = 0
        compress_token = None
        compress_token_ids = None
        all_mem_last_hidden = []
        per_chunk_mem_sizes = []
        all_trimmed_past_key_values = []

        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min((chunk_idx + 1) * chunk_size, total_length)
            chunk_input_ids = inputs['input_ids'][:, start_idx:end_idx]
            inputs_embeds = self.model.model.embed_tokens(chunk_input_ids)

            bsz, seq_len, emb_size = inputs_embeds.size()
            position_ids = torch.arange(start_idx + 1, end_idx + 1, device=inputs_embeds.device).unsqueeze(0)
            mem_position_ids = self.get_uniform_position_ids(x_1=start_idx + 1, x_n=end_idx + 1, ratio=self.compress_ratio)
            current_mem_size = mem_position_ids.size(1)

            # role tokens use original (un-truncated) anchor count so spacing is preserved
            encode_inputs_embeds = inputs_embeds.clone()
            role_embeds = (self.role_tokens[:current_mem_size, :]).unsqueeze(0).expand(bsz, current_mem_size, emb_size)
            mem_real_idx = mem_position_ids.squeeze(0) - 1 - start_idx
            encode_inputs_embeds.index_add_(1, mem_real_idx, role_embeds)
            attention_mask = self.build_attention_mask_full_bidirectional(seq_len).unsqueeze(0).unsqueeze(1).to(inputs_embeds.device).to(inputs_embeds.dtype)

            if self.task_config["use_pe"]:
                outputs = self.model(position_ids=position_ids, inputs_embeds=encode_inputs_embeds,
                                     output_hidden_states=True, attention_mask=attention_mask)
            else:
                outputs = self.model(inputs_embeds=encode_inputs_embeds, output_hidden_states=True)

            # Matryoshka truncation: keep first k anchors of this chunk
            chunk_k = None if k_list is None else int(k_list[chunk_idx])
            if chunk_k is not None and chunk_k < current_mem_size:
                kept = chunk_k
                kept_mem_real_idx = mem_real_idx[:kept]
                kept_mem_position_ids = mem_position_ids[:, :kept]
            else:
                kept = current_mem_size
                kept_mem_real_idx = mem_real_idx
                kept_mem_position_ids = mem_position_ids

            per_chunk_mem_sizes.append(kept)
            encoder_mem_size += kept

            if compress_token_ids is None:
                compress_token_ids = kept_mem_position_ids
            else:
                compress_token_ids = torch.cat((compress_token_ids, kept_mem_position_ids), dim=1)

            embed_hidden = outputs.hidden_states[0]
            mem_hidden = embed_hidden[:, kept_mem_real_idx]
            if compress_token is None:
                compress_token = mem_hidden
            else:
                compress_token = torch.cat((compress_token, mem_hidden), dim=1)

            last_hidden = outputs.hidden_states[-1]
            all_mem_last_hidden.append(last_hidden[:, kept_mem_real_idx])

            past_key_values = outputs.past_key_values
            trimmed_past_key_values = tuple(
                (layer_key[:, :, kept_mem_real_idx, :], layer_value[:, :, kept_mem_real_idx, :])
                for layer_key, layer_value in past_key_values
            )
            all_trimmed_past_key_values.append(trimmed_past_key_values)

        mem_last_hidden = torch.cat(all_mem_last_hidden, dim=1)
        encoder_past_key_values = self.concat_past_key_values_by_layer(all_trimmed_past_key_values)

        return (compress_token_ids, compress_token, end_idx,
                mem_last_hidden, encoder_past_key_values, encoder_mem_size,
                per_chunk_mem_sizes)

    # ========================================================================
    # PACT v5 — Stage 2/3 training branch (use_k_predictor=True)
    # ========================================================================

    def _decoder_lm_loss_for_pkv(self, inputs, past_key_values, end_idx,
                                  reduction: str = 'mean'):
        """Run decoder forward with given past_key_values and compute LM loss.

        Supports both pretrain LM (PG-19) and SFT (COMI) modes via task_config flags.
        reduction='mean' → scalar; reduction='none' → [B] per-sample (masked over targets).
        """
        is_sft = bool(self.task_config.get("is_sft", False))
        targets_key = "instruction_target" if is_sft else "lm_targets"
        targets = inputs[targets_key]
        skip_first_logit = is_sft   # SFT path drops the [LM]-position logit

        lm_target_emb = self.decoder.model.embed_tokens(inputs['lm_targets'][:, :-1])
        bsz, seq_len, emb_size = lm_target_emb.size()
        expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)
        lm_emb = torch.cat([expand_lm_token, lm_target_emb], dim=1)
        latter_position_ids = torch.arange(
            end_idx, end_idx + seq_len + 1, device=lm_target_emb.device
        ).unsqueeze(0)

        if self.task_config["use_pe"]:
            outputs = self.decoder(
                inputs_embeds=lm_emb,
                position_ids=latter_position_ids,
                past_key_values=past_key_values,
            )
        else:
            outputs = self.decoder(
                inputs_embeds=lm_emb,
                past_key_values=past_key_values,
            )
        logits = outputs.logits
        if skip_first_logit:
            logits = logits[:, 1:]

        if reduction == 'none':
            ce_none = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            tgt_flat = targets.reshape(-1).to(logits.device)
            per_token = ce_none(
                logits.reshape(-1, self.vocab_size).contiguous(),
                tgt_flat.contiguous(),
            ).view(targets.shape)                                # [B, S]
            mask = (targets != -100).float().to(logits.device)
            per_sample = (per_token * mask).sum(dim=-1) / mask.sum(dim=-1).clamp(min=1)
            return per_sample                                    # [B]
        return self.loss_fct(
            logits.contiguous().view(-1, self.vocab_size),
            targets.contiguous().view(-1).to(logits.device),
        )

    def _forward_k_predictor(self, inputs):
        """PACT v5 Stage 2/3 forward: matryoshka backbone + (warmup-gated)
        per-chunk oracle + KPredictor BCE."""
        loss_info = {}

        # 1) Compress once (full KV; no truncation)
        (_, _, end_idx, _, encoder_past_key_values_full, _,
         per_chunk_mem_sizes) = self.compress(inputs, truncate_k=None)
        N_chunks = len(per_chunk_mem_sizes)
        B = inputs['input_ids'].size(0)
        device = self.device

        # Hyperparameters from task_config
        K_choices = list(self.task_config.get("k_choices", K_CHOICES))
        K_max = max(K_choices)
        eps_grid = torch.tensor(
            self.task_config.get("eps_grid", [0.01, 0.05, 0.1, 0.2]),
            device=device, dtype=torch.float32,
        )
        predictor_warmup_skip = int(self.task_config.get("predictor_warmup_skip", 5000))
        ramp_steps = int(self.task_config.get("ramp_steps", 5000))
        alpha_max = float(self.task_config.get("alpha_max", 1.0))
        global_step = int(getattr(self, "_global_step", 0))

        # 2) Backbone matryoshka: sample one k with priority-aware weights so
        #    small-k truncations are seen more often, pushing important context
        #    toward the front anchors. Weights are configurable via
        #    matryoshka_weights (default [4,3,2,1.5,1,1] matches K_CHOICES len).
        mks_weights = self.task_config.get(
            "matryoshka_weights", [4.0, 3.0, 2.0, 1.5, 1.0, 1.0]
        )
        if len(mks_weights) != len(K_choices):
            # Length mismatch -> fall back to uniform to stay safe.
            mks_weights = [1.0] * len(K_choices)
        # Two modes:
        #   - sampling (default): one k per microbatch via priority-aware weights.
        #   - all_k (matryoshka_all_k=True): every k contributes this step,
        #     combined as a weighted sum (weights renormalised to sum=1 so the
        #     gradient scale matches the sampled path on average).
        if bool(self.task_config.get("matryoshka_all_k", False)):
            w_sum = float(sum(mks_weights)) or 1.0
            m_loss = None
            for k_c, w_c in zip(K_choices, mks_weights):
                pkv_k = truncate_kv_to_k(
                    encoder_past_key_values_full, k_c, per_chunk_mem_sizes
                )
                l_k = self._decoder_lm_loss_for_pkv(
                    inputs, pkv_k, end_idx, reduction='mean'
                )
                loss_info[f"m_loss_k{k_c}"] = float(l_k.detach().cpu())
                term = (w_c / w_sum) * l_k
                m_loss = term if m_loss is None else m_loss + term
            loss_info["matryoshka_k"] = -1  # sentinel: all-k mode
        else:
            k_sample = random.choices(K_choices, weights=mks_weights, k=1)[0]
            loss_info["matryoshka_k"] = k_sample
            pkv_sample = truncate_kv_to_k(
                encoder_past_key_values_full, k_sample, per_chunk_mem_sizes
            )
            m_loss = self._decoder_lm_loss_for_pkv(inputs, pkv_sample, end_idx, reduction='mean')
        loss_info["m_loss"] = float(m_loss.detach().cpu())

        warmup_active = global_step < predictor_warmup_skip
        loss_info["warmup_phase"] = bool(warmup_active)

        if warmup_active:
            loss_info["alpha"] = 0.0
            loss_info["loss"] = float(m_loss.detach().cpu())
            return {"loss": m_loss, "loss_info": loss_info}

        # 3) Reference L_full (no grad)
        with torch.no_grad():
            L_full = self._decoder_lm_loss_for_pkv(
                inputs, encoder_past_key_values_full, end_idx, reduction='none'
            )                                                    # [B]
        loss_info["L_full_mean"] = float(L_full.mean().cpu())

        # 4) Either fixed ε (predictor_fixed_eps set) or discrete sampling.
        fixed_eps = self.task_config.get("predictor_fixed_eps", None)
        if fixed_eps is not None:
            epsilon = torch.full(
                (B,), float(fixed_eps), device=device, dtype=torch.float32
            )
        else:
            eps_idx = torch.randint(0, len(eps_grid), (B,), device=device)
            epsilon = eps_grid[eps_idx]                          # [B]
        loss_info["eps_mean"] = float(epsilon.mean().cpu())

        # 4-naive) PACT naive-Gate ablation (predictor_gate_mode == "naive"):
        # no eps tolerance. eps is pinned to 0 (neutral); the per-chunk TARGET k
        # is the GLOBAL fixed-budget top-k allocation of the predictor's own
        # current scores (snapped to K_set). The survival head is then trained
        # (same weighted-BCE / per-slot survival labels as the eps path) to
        # reproduce that global-budget allocation. Single-variable vs eps path:
        # only the oracle/target differs (global-fixed-budget vs per-chunk eps).
        if self.predictor_gate_mode == "naive":
            global_budget = float(self.task_config.get("naive_global_budget", 8.0))
            epsilon = torch.zeros(B, device=device, dtype=torch.float32)
            loss_info["eps_mean"] = 0.0
            loss_info["naive_global_budget"] = global_budget
            anchor_kv = extract_full_kv(encoder_past_key_values_full).detach()
            # Target k per chunk = global-topk allocation of CURRENT scores.
            with torch.no_grad():
                _, k_target_b0 = self.k_predictor.predict_k_naive(
                    anchor_kv, per_chunk_mem_sizes, global_budget, K_choices
                )
            loss_info["pred_mean_k"] = (
                float(sum(k_target_b0) / N_chunks) if N_chunks > 0 else 0.0
            )
            oracle_per_chunk = [
                torch.full((B,), int(k_target_b0[c]), device=device, dtype=torch.long)
                for c in range(N_chunks)
            ]
            logit_list = self.k_predictor(anchor_kv, per_chunk_mem_sizes, epsilon)
            pos_weight = float(self.task_config.get("predictor_pos_weight", 4.0))
            p_loss = predictor_loss(
                logit_list, per_chunk_mem_sizes, oracle_per_chunk, pos_weight=pos_weight,
                head_mode=self.k_predictor.head_mode
            )
            loss_info["p_loss"] = float(p_loss.detach().cpu())
            ramp_step = max(0, global_step - predictor_warmup_skip)
            alpha = min(alpha_max, ramp_step / max(ramp_steps, 1))
            loss = m_loss + alpha * p_loss
            loss_info["alpha"] = alpha
            oracle_stack = torch.stack(oracle_per_chunk, dim=0).float()
            loss_info["oracle_mean_k"] = float(oracle_stack.mean().cpu())
            loss_info["oracle_per_chunk_var"] = (
                float(oracle_stack.var(dim=0, unbiased=False).mean().cpu())
                if oracle_stack.size(0) >= 2 else 0.0
            )
            loss_info["loss"] = float(loss.detach().cpu())
            return {"loss": loss, "loss_info": loss_info}

        # 4b) Predictor's current per-chunk k (no grad). Used as the "other
        #     chunks" context for the oracle below — replaces the previous
        #     K_max baseline so that the oracle reflects the joint compression
        #     state of the model, not an idealised everyone-at-K_max world.
        #     Reference L_full (all K_max) is unchanged; the predictor receives
        #     a self-correcting signal: aggressive other-chunk predictions push
        #     L_ck[k] up, raising the oracle for the target chunk.
        assert B == 1, (
            "PACT v5 KPredictor oracle path assumes batch_size_per_device=1; "
            f"got B={B}"
        )
        anchor_kv = extract_full_kv(encoder_past_key_values_full).detach()
        with torch.no_grad():
            _, k_pred_b0 = self.k_predictor.predict_k(
                anchor_kv, per_chunk_mem_sizes, epsilon
            )                                                    # list[int], len N_chunks
        if N_chunks > 0:
            loss_info["pred_mean_k"] = float(sum(k_pred_b0) / N_chunks)

        # Oracle other-chunk context. "selfref" (default, current behaviour):
        # other chunks held at their own predicted k -> self-referential moving
        # target. "kmax": other chunks held at K_max -> stationary oracle target
        # decoupled from the predictor's own output. Diagnostic lever 2026-06-13.
        oracle_ctx = str(self.task_config.get("oracle_context_mode", "selfref"))
        base_k_list = list(k_pred_b0) if oracle_ctx == "selfref" else [K_max] * N_chunks
        loss_info["oracle_ctx_kmax"] = (oracle_ctx != "selfref")

        # 5) Per-chunk per-k decoder forwards (no grad). Target chunk c uses
        #    candidate k; other chunks use base_k_list (selfref=predicted k,
        #    kmax=K_max).
        losses_per_chunk_per_k = []
        with torch.no_grad():
            for c in range(N_chunks):
                L_ck = {}
                for k in K_choices:
                    k_list = list(base_k_list)
                    k_list[c] = k
                    pkv = truncate_kv_to_k(
                        encoder_past_key_values_full, k_list, per_chunk_mem_sizes
                    )
                    L_ck[k] = self._decoder_lm_loss_for_pkv(
                        inputs, pkv, end_idx, reduction='none'
                    )
                losses_per_chunk_per_k.append(L_ck)

        # 6) Per-chunk oracle (already detached). Reference L_full = all K_max.
        oracle_relative = bool(self.task_config.get("oracle_relative_eps", False))
        oracle_floor = float(self.task_config.get("oracle_eps_floor", 1e-3))
        oracle_per_chunk = compute_oracle_k_per_chunk(
            losses_per_chunk_per_k, L_full, epsilon, K_choices,
            relative=oracle_relative, eps_floor=oracle_floor,
        )
        loss_info["oracle_relative"] = oracle_relative
        loss_info["oracle_floor"] = oracle_floor

        # 7) Predictor forward (with grad on predictor params; encoder detached)
        logit_list = self.k_predictor(anchor_kv, per_chunk_mem_sizes, epsilon)
        pos_weight = float(self.task_config.get("predictor_pos_weight", 4.0))
        p_loss = predictor_loss(
            logit_list, per_chunk_mem_sizes, oracle_per_chunk, pos_weight=pos_weight,
            head_mode=self.k_predictor.head_mode,
            rank_weight=float(self.task_config.get("predictor_rank_loss_weight", 0.0)),
            rank_mode=str(self.task_config.get("predictor_rank_loss_mode", "corr")),
        )
        loss_info["p_loss"] = float(p_loss.detach().cpu())

        # 8) Alpha ramp + total
        ramp_step = max(0, global_step - predictor_warmup_skip)
        alpha = min(alpha_max, ramp_step / max(ramp_steps, 1))
        loss = m_loss + alpha * p_loss
        loss_info["alpha"] = alpha

        # Wandb-friendly oracle stats
        oracle_stack = torch.stack(oracle_per_chunk, dim=0).float()  # [N_chunks, B]
        loss_info["oracle_mean_k"] = float(oracle_stack.mean().cpu())
        if oracle_stack.size(0) >= 2:
            loss_info["oracle_per_chunk_var"] = float(
                oracle_stack.var(dim=0, unbiased=False).mean().cpu()
            )
        else:
            loss_info["oracle_per_chunk_var"] = 0.0

        loss_info["loss"] = float(loss.detach().cpu())
        return {"loss": loss, "loss_info": loss_info}

    def lm_inference(self, inputs, generate_num=1024, truncate_k: int = None,
                      gate_threshold: float = None, epsilon: float = None,
                      k_predictor_threshold: float = None):
        """v5 dispatch: if self.use_k_predictor and epsilon is provided, use the
        trained KPredictor to pick per-chunk k. Otherwise fall back to v3/v4
        gate/positional behavior.
        """
        # PACT v5 inference branch (KPredictor active + epsilon given).
        use_v5_predictor = (
            self.use_k_predictor
            and self.k_predictor is not None
            and epsilon is not None
        )

        # For content_topk: do NOT positionally truncate in compress();
        # gate's hard top-k by score determines surviving anchors per chunk.
        use_content_topk = (self.use_gate and self.gate_mode == "content_topk")
        # In v5, compress() runs with full KV first; truncation happens after
        # predict_k via index_select on encoder_past_key_values.
        compress_truncate_k = (
            None if (use_content_topk or use_v5_predictor) else truncate_k
        )

        (compress_token_ids, compress_token, end_idx,
         mem_last_hidden, encoder_past_key_values, encoder_mem_size,
         per_chunk_mem_sizes) = self.compress(inputs, truncate_k=compress_truncate_k)

        if use_v5_predictor:
            device = encoder_past_key_values[0][0].device
            B = encoder_past_key_values[0][0].size(0)
            eps_t = torch.tensor([float(epsilon)] * B, device=device)
            with torch.no_grad():
                anchor_kv = extract_full_kv(encoder_past_key_values)
                if self.predictor_gate_mode == "naive":
                    # Naive-Gate: ignore epsilon; allocate one global fixed
                    # budget. `epsilon` arg is repurposed as the global budget
                    # ONLY if naive_global_budget is unset (eval override hook).
                    gb = self.naive_global_budget
                    if gb is None:
                        gb = float(epsilon)
                    K_choices = list(self.task_config.get("k_choices", K_CHOICES))
                    _, k_per_chunk_b0 = self.k_predictor.predict_k_naive(
                        anchor_kv, per_chunk_mem_sizes, float(gb), K_choices
                    )
                else:
                    _, k_per_chunk_b0 = self.k_predictor.predict_k(
                        anchor_kv, per_chunk_mem_sizes, eps_t,
                        threshold=k_predictor_threshold,
                    )
            # Apply per-chunk first-k truncation to encoder_past_key_values.
            keep_idx = _build_keep_indices(per_chunk_mem_sizes, k_per_chunk_b0, device)
            encoder_past_key_values = tuple(
                (k.index_select(2, keep_idx), v.index_select(2, keep_idx))
                for (k, v) in encoder_past_key_values
            )
            compress_token_ids = compress_token_ids.index_select(1, keep_idx)
            # Stash chosen k for callers / eval scripts.
            self._last_v5_k_per_chunk = list(k_per_chunk_b0)

        if self.use_gate and self.gate_predictor is not None:
            gate_g, gate_info_raw = self.gate_predictor(mem_last_hidden, per_chunk_mem_sizes)

            if use_content_topk:
                # Hard top-k by score per chunk; truncate_k controls k per chunk.
                score_full = gate_info_raw["score"][0]
                keep_indices = []
                _off = 0
                k_target = int(truncate_k) if truncate_k is not None else self.mem_size
                for m in per_chunk_mem_sizes:
                    seg_score = score_full[_off:_off + m]
                    k_eff = max(1, min(k_target, m))
                    topk = torch.topk(seg_score, k_eff, largest=True).indices
                    idx_local = topk.sort().values
                    keep_indices.append(idx_local + _off)
                    _off += m
                keep_idx = torch.cat(keep_indices, dim=0)
                encoder_past_key_values = tuple(
                    (k.index_select(2, keep_idx), v.index_select(2, keep_idx))
                    for (k, v) in encoder_past_key_values
                )
                compress_token_ids = compress_token_ids.index_select(1, keep_idx)

            elif gate_threshold is not None and gate_threshold > 0.0:
                keep_mask = (gate_g > gate_threshold)[0]
                if not keep_mask.all():
                    keep_idx = keep_mask.nonzero(as_tuple=False).squeeze(-1)
                    if keep_idx.numel() == 0:
                        keep_idx = torch.tensor([0], device=keep_mask.device)
                    encoder_past_key_values = tuple(
                        (k.index_select(2, keep_idx), v.index_select(2, keep_idx))
                        for (k, v) in encoder_past_key_values
                    )
                    compress_token_ids = compress_token_ids.index_select(1, keep_idx)
            elif gate_threshold is None:
                # cross_chunk_softmax & cumprod: apply soft gate
                g_for_kv = gate_g.to(encoder_past_key_values[0][0].dtype)
                g_kv = g_for_kv[:, None, :, None]
                encoder_past_key_values = tuple(
                    (k * g_kv, v * g_kv) for (k, v) in encoder_past_key_values
                )
        lm_target_emb = self.decoder.model.embed_tokens(inputs['lm_targets'])
        bsz, seq_len, emb_size = lm_target_emb.size()
        expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)

        lm_emb = torch.cat([expand_lm_token, lm_target_emb], dim=1)
        # context position ids:[1,......,end_idx]
        # [LM] position ids:[end_idx];  QA position ids:[end_idx+1,.......]
        latter_position_ids = torch.arange(end_idx, end_idx + seq_len + 1, device=lm_target_emb.device).unsqueeze(0)
        lm_position_ids = latter_position_ids

        generate_text = []
        past_key_values = encoder_past_key_values
        next_inputs_embeds = lm_emb.clone()
        next_position_ids = lm_position_ids.clone()
        for i in range(generate_num):
            if self.task_config["use_pe"]:
                out = self.decoder(position_ids=next_position_ids, inputs_embeds=next_inputs_embeds, past_key_values=past_key_values, use_cache=True)
            else:
                out = self.decoder(inputs_embeds=next_inputs_embeds, past_key_values=past_key_values, use_cache=True)
            # [B,S,V] -> [B,V]
            logit = out.logits[:, -1]
            past_key_values = out.past_key_values
            # [B,V]->[B]
            next_token_id = torch.argmax(logit, dim=-1)
            # [B]->[B,E]->[B,1,E]
            next_inputs_embeds = self.decoder.model.embed_tokens(next_token_id).unsqueeze(1).to(lm_target_emb.device)  
            # [1, seq_len]/[1,1] -> [1,1]
            next_position_ids = next_position_ids[:,-1:]+1
            generate_text.append(next_token_id.item())
            if next_token_id.item() == self.tokenizer.eos_token_id:
                return generate_text
        return generate_text



    def lm_inference_diag(self, inputs, generate_num=128, strategy="gate_soft",
                          target_n_per_chunk=None, seed=42):
        """Diagnostic inference supporting multiple keep-strategies on stage3 model.

        strategy in:
          - 'all34'       : no gate, keep all anchors (baseline at full budget)
          - 'gate_soft'   : multiply pkv by trained gate g (no hard drop)
          - 'gate_hard'   : keep top-N anchors per chunk by g (hard select)
          - 'first'       : keep first N anchors per chunk (positional baseline)
          - 'random'      : keep random N anchors per chunk (seed=seed + sample_hash)
          - 'uniform'     : multiply all pkv by (N / M_chunk), soft uniform attenuation

        target_n_per_chunk : int. Required for gate_hard/first/random/uniform.
        Returns: (generated_token_ids, info_dict). info contains gate stats.
        """
        (compress_token_ids, compress_token, end_idx,
         mem_last_hidden, encoder_past_key_values, encoder_mem_size,
         per_chunk_mem_sizes) = self.compress(inputs, truncate_k=None)

        device = encoder_past_key_values[0][0].device
        kv_dtype = encoder_past_key_values[0][0].dtype

        info = {"per_chunk_mem_sizes": list(per_chunk_mem_sizes), "strategy": strategy}

        gate_g = None
        gate_info_raw = None
        if self.use_gate and self.gate_predictor is not None:
            gate_g, gate_info_raw = self.gate_predictor(mem_last_hidden, per_chunk_mem_sizes)
            info["gate_mode"] = self.gate_mode
            info["gate_g_mean"] = float(gate_g.mean().detach().cpu())
            info["gate_g_std"] = float(gate_g.std().detach().cpu())
            if gate_info_raw.get("b_chunks") is not None:
                bc = gate_info_raw["b_chunks"]
                info["b_chunks_mean"] = float(bc.mean().detach().cpu())
                info["b_chunks_std"] = float(bc.std().detach().cpu())
                info["b_chunks_list"] = bc.squeeze(-1).detach().cpu().tolist() if bc.dim() == 2 else bc.detach().cpu().tolist()
            # per-chunk N_kept(>0.1) and sum_g
            n_kept_per_chunk = []
            sum_g_per_chunk = []
            off = 0
            for m in per_chunk_mem_sizes:
                seg = gate_g[0, off:off+m]
                n_kept_per_chunk.append(int((seg > 0.1).sum().item()))
                sum_g_per_chunk.append(float(seg.sum().item()))
                off += m
            info["n_kept_per_chunk"] = n_kept_per_chunk
            info["sum_g_per_chunk"] = sum_g_per_chunk
            # monotonicity: g[0] vs g[-1] per chunk
            head_g, tail_g = [], []
            off = 0
            for m in per_chunk_mem_sizes:
                head_g.append(float(gate_g[0, off].item()))
                tail_g.append(float(gate_g[0, off+m-1].item()))
                off += m
            info["g_head_mean"] = float(sum(head_g)/len(head_g))
            info["g_tail_mean"] = float(sum(tail_g)/len(tail_g))

        # --- Strategy dispatch ---
        if strategy == "all34":
            pass  # leave pkv as-is, no gate

        elif strategy == "gate_soft":
            assert gate_g is not None, "gate_soft requires use_gate=True"
            g_kv = gate_g.to(kv_dtype)[:, None, :, None]
            encoder_past_key_values = tuple(
                (k * g_kv, v * g_kv) for (k, v) in encoder_past_key_values
            )

        elif strategy in ("gate_hard", "first", "random"):
            assert target_n_per_chunk is not None, f"{strategy} requires target_n_per_chunk"
            import random as _rnd
            keep_indices = []
            off = 0
            for ci, m in enumerate(per_chunk_mem_sizes):
                n_keep = min(target_n_per_chunk, m)
                if strategy == "gate_hard":
                    assert gate_g is not None
                    seg = gate_g[0, off:off+m]
                    topk = torch.topk(seg, n_keep, largest=True).indices
                    idx_local = topk.sort().values  # keep positional order
                elif strategy == "first":
                    idx_local = torch.arange(n_keep, device=device)
                elif strategy == "random":
                    rng = _rnd.Random(seed + ci * 7919)
                    chosen = sorted(rng.sample(range(m), n_keep))
                    idx_local = torch.tensor(chosen, device=device, dtype=torch.long)
                keep_indices.append(idx_local + off)
                off += m
            keep_idx = torch.cat(keep_indices, dim=0)
            info["kept_total"] = int(keep_idx.numel())
            encoder_past_key_values = tuple(
                (k.index_select(2, keep_idx), v.index_select(2, keep_idx))
                for (k, v) in encoder_past_key_values
            )
            compress_token_ids = compress_token_ids.index_select(1, keep_idx)

        elif strategy == "adaptive":
            # v4 adaptive inference: use the learned gate to decide per-chunk k* directly.
            # - elastic_budget: keep first round(b_chunk) anchors per chunk
            # - sparse_sigmoid: keep anchors with g > threshold (default 0.5)
            # - cumprod / others: fallback to keep all anchors with g > 0.5
            assert gate_g is not None, "adaptive requires use_gate=True"
            threshold = float(target_n_per_chunk) if target_n_per_chunk is not None else 0.5
            keep_indices = []
            kept_per_chunk = []
            off = 0
            for ci, m in enumerate(per_chunk_mem_sizes):
                seg = gate_g[0, off:off + m]
                if self.gate_mode == "elastic_budget" and gate_info_raw is not None and gate_info_raw.get("b_chunks") is not None:
                    b = float(gate_info_raw["b_chunks"][ci, 0].item())
                    n_keep = max(1, min(m, int(round(b))))
                    # Keep first n_keep anchors (matryoshka front-loads info).
                    idx_local = torch.arange(n_keep, device=device)
                else:
                    keep_mask = seg > threshold
                    if not keep_mask.any():
                        idx_local = torch.tensor([0], device=device, dtype=torch.long)
                    else:
                        idx_local = keep_mask.nonzero(as_tuple=False).squeeze(-1)
                keep_indices.append(idx_local + off)
                kept_per_chunk.append(int(idx_local.numel()))
                off += m
            keep_idx = torch.cat(keep_indices, dim=0)
            info["kept_per_chunk"] = kept_per_chunk
            info["kept_total"] = int(keep_idx.numel())
            info["mean_kept"] = float(sum(kept_per_chunk) / max(1, len(kept_per_chunk)))
            encoder_past_key_values = tuple(
                (k.index_select(2, keep_idx), v.index_select(2, keep_idx))
                for (k, v) in encoder_past_key_values
            )
            compress_token_ids = compress_token_ids.index_select(1, keep_idx)

        elif strategy == "k_predictor":
            # PACT v5: use trained KPredictor to pick per-chunk k.
            # target_n_per_chunk repurposed as epsilon (must be in eps_grid).
            assert self.use_k_predictor and self.k_predictor is not None, (
                "k_predictor strategy requires use_k_predictor=True"
            )
            assert target_n_per_chunk is not None, (
                "k_predictor strategy uses target_n_per_chunk as epsilon"
            )
            B = encoder_past_key_values[0][0].size(0)
            eps_t = torch.tensor([float(target_n_per_chunk)] * B, device=device)
            with torch.no_grad():
                anchor_kv = extract_full_kv(encoder_past_key_values)
                if self.predictor_gate_mode == "naive":
                    gb = self.naive_global_budget
                    if gb is None:
                        gb = float(target_n_per_chunk)
                    K_choices = list(self.task_config.get("k_choices", K_CHOICES))
                    _, k_per_chunk_b0 = self.k_predictor.predict_k_naive(
                        anchor_kv, per_chunk_mem_sizes, float(gb), K_choices
                    )
                else:
                    _, k_per_chunk_b0 = self.k_predictor.predict_k(
                        anchor_kv, per_chunk_mem_sizes, eps_t, threshold=None,
                    )
            keep_idx = _build_keep_indices(per_chunk_mem_sizes, k_per_chunk_b0, device)
            info["kept_per_chunk"] = list(k_per_chunk_b0)
            info["kept_total"] = int(keep_idx.numel())
            info["mean_kept"] = float(sum(k_per_chunk_b0) / max(1, len(k_per_chunk_b0)))
            info["epsilon"] = float(target_n_per_chunk)
            encoder_past_key_values = tuple(
                (k.index_select(2, keep_idx), v.index_select(2, keep_idx))
                for (k, v) in encoder_past_key_values
            )
            compress_token_ids = compress_token_ids.index_select(1, keep_idx)

        elif strategy == "uniform":
            assert target_n_per_chunk is not None, "uniform requires target_n_per_chunk"
            # per-chunk scale = target_n / m
            scale_vec = torch.zeros(sum(per_chunk_mem_sizes), device=device, dtype=kv_dtype)
            off = 0
            for m in per_chunk_mem_sizes:
                s = min(target_n_per_chunk, m) / m
                scale_vec[off:off+m] = s
                off += m
            s_kv = scale_vec[None, None, :, None]
            encoder_past_key_values = tuple(
                (k * s_kv, v * s_kv) for (k, v) in encoder_past_key_values
            )

        else:
            raise ValueError(f"unknown strategy: {strategy}")

        # --- Decode loop (same as lm_inference) ---
        lm_target_emb = self.decoder.model.embed_tokens(inputs['lm_targets'])
        bsz, seq_len, emb_size = lm_target_emb.size()
        expand_lm_token = self.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)
        lm_emb = torch.cat([expand_lm_token, lm_target_emb], dim=1)
        latter_position_ids = torch.arange(end_idx, end_idx + seq_len + 1, device=lm_target_emb.device).unsqueeze(0)
        lm_position_ids = latter_position_ids

        generate_text = []
        past_key_values = encoder_past_key_values
        next_inputs_embeds = lm_emb.clone()
        next_position_ids = lm_position_ids.clone()
        for i in range(generate_num):
            if self.task_config["use_pe"]:
                out = self.decoder(position_ids=next_position_ids, inputs_embeds=next_inputs_embeds,
                                   past_key_values=past_key_values, use_cache=True)
            else:
                out = self.decoder(inputs_embeds=next_inputs_embeds,
                                   past_key_values=past_key_values, use_cache=True)
            logit = out.logits[:, -1]
            past_key_values = out.past_key_values
            next_token_id = torch.argmax(logit, dim=-1)
            next_inputs_embeds = self.decoder.model.embed_tokens(next_token_id).unsqueeze(1).to(lm_target_emb.device)
            next_position_ids = next_position_ids[:, -1:] + 1
            generate_text.append(next_token_id.item())
            if next_token_id.item() == self.tokenizer.eos_token_id:
                return generate_text, info
        return generate_text, info


    def ae_inference(self, inputs):
        (compress_token_ids, compress_token, end_idx,
         mem_last_hidden, encoder_past_key_values, encoder_mem_size,
         per_chunk_mem_sizes) = self.compress(inputs)
        bsz, tot_mem_size, emb_size = compress_token.size()
        # [1,E] -> [1,1,E] -> [B,1,E]
        expand_ae_token = self.special_tokens[0:1].unsqueeze(0).expand(bsz, 1, emb_size)

        # [B,tot_mem_size,E];   [B,1,E]
        ae_emb = expand_ae_token

        # ae_pid:[0]  shape:[1]->[1,1]->[B,1]
        position_ids = torch.arange(0, 1, device=compress_token.device).unsqueeze(0).expand(bsz, 1)
        ae_position_ids = position_ids

        generate_text = []
        past_key_values = encoder_past_key_values
        next_inputs_embeds = ae_emb.clone()
        next_position_ids = ae_position_ids.clone()

        for i in range(inputs['input_ids'].size(-1)+20):
            # print(f"next_pids:{next_position_ids}")
            if self.task_config["use_pe"]:
                out = self.decoder(position_ids=next_position_ids, inputs_embeds=next_inputs_embeds, past_key_values=past_key_values, use_cache=True)
            else:
                out = self.decoder(inputs_embeds=next_inputs_embeds, past_key_values=past_key_values, use_cache=True)
            # [B,S,V] -> [B,V]
            logit = out.logits[:, -1]
            past_key_values = out.past_key_values
            # [B,V]->[B]
            next_token_id = torch.argmax(logit, dim=-1)
            # [B]->[B,E]->[B,1,E]
            next_inputs_embeds = self.decoder.model.embed_tokens(next_token_id).unsqueeze(1).to(compress_token.device)
            # next_position_ids:[B,S] -> [B,1]
            next_position_ids = next_position_ids[:,-1:]+1
            generate_text.append(next_token_id.item())
            if next_token_id.item() == self.tokenizer.eos_token_id:
                return generate_text
        return generate_text



    # def forward(self, inputs):
    #     loss_info = {}
    #     inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"])
    #     lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'][:, :-1])
    #     encode_inputs_embeds = torch.cat([inputs_embeds, lm_target_emb], dim=1)
    #     outputs = self.model(
    #         inputs_embeds=encode_inputs_embeds,
    #     )
    #     # [B,mem_size+S,V] -> [B,S,V]
    #     logits = outputs.logits[:,inputs_embeds.size(1):]
    #     logits = logits.contiguous().view(-1, self.vocab_size)
    #     inputs["instruction_target"] = inputs["instruction_target"].contiguous().view(-1).to(logits.device)
    #     lm_loss = self.loss_fct(logits, inputs["instruction_target"])
    #     loss_info["lm_loss"] = lm_loss.item()
    #     loss = lm_loss
    #     return {"loss": loss, "loss_info": loss_info}



    def vanilla_llama_inference(self, inputs):
        inputs_embeds = self.model.model.embed_tokens(inputs["input_ids"])
        lm_target_emb = self.model.model.embed_tokens(inputs['lm_targets'])
        encode_inputs_embeds = torch.cat([inputs_embeds, lm_target_emb], dim=1)

        generate_text = []
        past_key_values = None
        next_inputs_embeds = encode_inputs_embeds.clone()
        for i in range(1024):
            # print(f"next_position_ids:{next_position_ids}")
            out = self.model(inputs_embeds=next_inputs_embeds,
                                     past_key_values=past_key_values,
                                     use_cache=True)
            # [B,S,V] -> [B,V]
            logit = out.logits[:, -1]
            past_key_values = out.past_key_values
            # [B,V]->[B]
            next_token_id = torch.argmax(logit, dim=-1)
            next_inputs_embeds = self.model.model.embed_tokens(next_token_id).unsqueeze(1).to(inputs_embeds.device)
            generate_text.append(next_token_id.item())
            if next_token_id.item() == self.tokenizer.eos_token_id:
                return generate_text
        return generate_text

    def build_attention_mask_full_bidirectional(self, num_input):
        total_len = num_input
        mask = torch.zeros((total_len, total_len))  # all zeros means fully visible
        return mask

    def concat_past_key_values_by_layer(self, all_trimmed_past_key_values):
        num_layers = len(all_trimmed_past_key_values[0])
        merged_past_key_values = []

        for layer_idx in range(num_layers):
            keys = []
            values = []
            for trimmed in all_trimmed_past_key_values:
                layer_key, layer_value = trimmed[layer_idx]
                keys.append(layer_key)
                values.append(layer_value)
            merged_key = torch.cat(keys, dim=2)
            merged_value = torch.cat(values, dim=2)
            merged_past_key_values.append((merged_key, merged_value))

        return tuple(merged_past_key_values)


def freeze_encoder(model):
    for name, param in model.named_parameters():
        if name == "role_tokens" or name == "special_tokens":
            continue
        if name.startswith("gate_predictor."):
            continue
        # PACT v5: KPredictor must remain trainable.
        if name.startswith("k_predictor."):
            continue
        param.requires_grad = False

def freeze_decoder(model):
    for name, param in model.named_parameters():
        param.requires_grad = False


def load_adapter(model, save_path_and_name='adapter.pt', log=False, exclude_prefixes=()):
    adapter_state_dict = torch.load(save_path_and_name, map_location='cpu')  # load onto CPU first

    # move the adapter weights onto the model's device
    adapter_state_dict = {k: v.to(model.device) for k, v in adapter_state_dict.items()}

    # Drop keys under any excluded prefix (e.g. "k_predictor." for predictor-only
    # retrains where the new predictor arch differs from the loaded ckpt — keeps
    # its fresh init instead of crashing on a shape mismatch). load_state_dict's
    # strict=False ignores MISSING/UNEXPECTED keys but still raises on a shape
    # mismatch of a PRESENT key, so we must filter rather than rely on strict.
    if exclude_prefixes:
        adapter_state_dict = {
            k: v for k, v in adapter_state_dict.items()
            if not any(k.startswith(p) for p in exclude_prefixes)
        }
        if log:
            print(f"[load_adapter] excluded prefixes {exclude_prefixes}")

    model.load_state_dict(adapter_state_dict, strict=False)
    return model

def load_model_with_adapter(model_id, task_config, rank, save_path_and_name='adapter.pt', log=False):
    model = get_model(model_id, task_config, rank)
    load_adapter(model, save_path_and_name, log)
    return model

def get_model_for_compress(model_id, task_config, rank):

    def add_compress_lora(model, task_config):
        for name, module in model.named_children():
            if isinstance(module, nn.Linear) and ((name == "q_proj") or (name == "v_proj")):
                setattr(model, name, LinearLoraLayer(module.in_features, module.out_features, r=128,
                                                     weight=module.weight.data.clone()))
            else:
                # Recursively apply this function to submodules
                add_compress_lora(module, task_config)

    model = CompressLLM(
        model_id,
        mem_size=task_config["mem_size"],
        compress_ratio=task_config["compress_ratio"],
        device_rank=rank,
        task_config=task_config
    )

    # freeze all the model except mem tokens and special tokens
    freeze_encoder(model)
    if bool(task_config.get("full_finetune", False)):
        # Full fine-tuning variant (no LoRA): train the ENTIRE encoder base
        # (self.model) directly instead of injecting LoRA adapters. Decoder stays
        # frozen (compress-for-a-frozen-LLM setup); mem/special tokens and the
        # k_predictor were already left trainable by freeze_encoder above.
        for _p in model.model.parameters():
            _p.requires_grad = True
    else:
        # only add lora to encoder, don't add lora to model.decoder
        add_compress_lora(model.model, task_config)
    return model

def get_model(model_id, task_config, rank):
    if task_config["task_type"] == "Compress":
        return get_model_for_compress(model_id, task_config, rank)
    raise Exception("Don't exist [{task_type}] task.")

def save_adapter(model, save_path_and_name='adapter.pt', log=False):
    adapter_name = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            if log:
                print("[Save Adapter]", name)
            adapter_name.add(name)

    state_dict = model.state_dict()
    adapter_state_dict = {name: param for name, param in state_dict.items() if name in adapter_name}
    torch.save(adapter_state_dict, save_path_and_name)
