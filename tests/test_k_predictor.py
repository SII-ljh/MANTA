"""Unit tests for PACT v5 K Predictor (r2).

Tests the new KPredictor + per-chunk oracle + utility functions in isolation.
Uses small dims (d_kv=64, num_layers=4) so all tests run on CPU in <30s.
"""
import os
import sys
import math
import pytest
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modeling import (
    KPredictor,
    extract_full_kv,
    truncate_kv_to_k,
    truncate_single_chunk_kv,
    compute_oracle_k,
    compute_oracle_k_per_chunk,
    oracle_to_per_slot_labels,
    predictor_loss,
    K_CHOICES,
)


# ---- Small dims for fast CPU tests ----
NUM_LAYERS = 4
NUM_HEADS = 4
HEAD_DIM = 16
D_KV = NUM_HEADS * HEAD_DIM      # 64
MAX_SLOTS = 8                    # smaller than prod 34
D_LAYER_LOW = 4
D_HIDDEN = 32
ATTN_HEADS = 2


def _mock_encoder_pkv(batch, total_m, num_layers=NUM_LAYERS,
                     num_heads=NUM_HEADS, head_dim=HEAD_DIM, seed=0):
    """Mock encoder_past_key_values: tuple of (K, V) tensors per layer.
    Each K, V: [B, num_heads, total_M, head_dim].
    """
    g = torch.Generator().manual_seed(seed)
    layers = []
    for _ in range(num_layers):
        K = torch.randn(batch, num_heads, total_m, head_dim, generator=g)
        V = torch.randn(batch, num_heads, total_m, head_dim, generator=g)
        layers.append((K, V))
    return tuple(layers)


def _make_predictor():
    return KPredictor(
        num_layers=NUM_LAYERS,
        d_kv=D_KV,
        d_layer_low=D_LAYER_LOW,
        max_slots=MAX_SLOTS,
        d_hidden=D_HIDDEN,
        attn_heads=ATTN_HEADS,
        eps_embed_dim=16,
    )


# =====================================================================
# extract_full_kv
# =====================================================================

def test_extract_full_kv_shape():
    """extract_full_kv: tuple[L (K,V)] with K=[B,NH,M,HD] → [B, L, 2, M, NH*HD]"""
    B, total_M = 2, 6
    pkv = _mock_encoder_pkv(B, total_M)
    out = extract_full_kv(pkv)
    assert out.shape == (B, NUM_LAYERS, 2, total_M, NUM_HEADS * HEAD_DIM)
    # K is first
    K0_flat = pkv[0][0].transpose(1, 2).reshape(B, total_M, NUM_HEADS * HEAD_DIM)
    assert torch.allclose(out[:, 0, 0], K0_flat)
    V0_flat = pkv[0][1].transpose(1, 2).reshape(B, total_M, NUM_HEADS * HEAD_DIM)
    assert torch.allclose(out[:, 0, 1], V0_flat)


# =====================================================================
# truncate helpers
# =====================================================================

def test_truncate_kv_to_k_per_layer_shape():
    B, total_M = 1, 10
    per_chunk = [4, 6]   # sum = 10
    pkv = _mock_encoder_pkv(B, total_M)
    pkv_trunc = truncate_kv_to_k(pkv, k=2, per_chunk_mem_sizes=per_chunk)
    # k=2 applied to each chunk → kept count = 2+2 = 4
    for (K, V) in pkv_trunc:
        assert K.shape == (B, NUM_HEADS, 4, HEAD_DIM)
        assert V.shape == (B, NUM_HEADS, 4, HEAD_DIM)


def test_truncate_kv_to_k_keeps_first_k_per_chunk():
    """Verify it keeps the first k anchors in each chunk."""
    B = 1
    per_chunk = [3, 3]
    pkv = _mock_encoder_pkv(B, 6)
    pkv_trunc = truncate_kv_to_k(pkv, k=2, per_chunk_mem_sizes=per_chunk)
    K0, V0 = pkv_trunc[0]
    # Should be indices [0, 1] (chunk0) and [3, 4] (chunk1)
    expected_idx = torch.tensor([0, 1, 3, 4])
    assert torch.allclose(K0, pkv[0][0][:, :, expected_idx, :])
    assert torch.allclose(V0, pkv[0][1][:, :, expected_idx, :])


def test_truncate_kv_to_k_above_max_keeps_all():
    """k larger than chunk size: keep all anchors in that chunk."""
    pkv = _mock_encoder_pkv(1, 6)
    per_chunk = [3, 3]
    pkv_trunc = truncate_kv_to_k(pkv, k=100, per_chunk_mem_sizes=per_chunk)
    for (K, _) in pkv_trunc:
        assert K.shape[2] == 6   # all kept


def test_truncate_kv_to_k_accepts_list():
    """truncate_kv_to_k should also accept per-chunk list of k values."""
    pkv = _mock_encoder_pkv(1, 6)
    per_chunk = [3, 3]
    pkv_trunc = truncate_kv_to_k(pkv, k=[1, 2], per_chunk_mem_sizes=per_chunk)
    # chunk 0: keep 1, chunk 1: keep 2 → kept = 3 total
    K0 = pkv_trunc[0][0]
    assert K0.shape[2] == 3
    expected_idx = torch.tensor([0, 3, 4])   # chunk0 idx 0; chunk1 idx 3,4
    assert torch.allclose(K0, pkv[0][0][:, :, expected_idx, :])


def test_truncate_single_chunk_kv_affects_only_target():
    """truncate_single_chunk_kv: chunk c → k anchors; others kept at K_max."""
    pkv = _mock_encoder_pkv(1, 9)
    per_chunk = [3, 3, 3]
    pkv_trunc = truncate_single_chunk_kv(pkv, per_chunk, target_chunk=1, k=1)
    K_trunc = pkv_trunc[0][0]
    # Kept: chunk0 full (3), chunk1 first 1, chunk2 full (3) = 7
    assert K_trunc.shape[2] == 7
    expected_idx = torch.tensor([0, 1, 2,    # chunk 0 all
                                  3,           # chunk 1 keep first 1
                                  6, 7, 8])    # chunk 2 all
    assert torch.allclose(K_trunc, pkv[0][0][:, :, expected_idx, :])


# =====================================================================
# Oracle K
# =====================================================================

def test_compute_oracle_k_eps_zero_picks_kmax():
    """ε=0: no k satisfies (L_k - L_full < 0 only if L_k < L_full, rare).
    Should default to K_max for unsatisfied samples."""
    L_full = torch.tensor([1.0, 1.0])
    losses_per_k = {
        1: torch.tensor([2.0, 2.0]),
        2: torch.tensor([1.5, 1.5]),
        4: torch.tensor([1.2, 1.2]),
        8: torch.tensor([1.05, 1.05]),
        16: torch.tensor([1.01, 1.01]),
        34: L_full,
    }
    epsilon = torch.tensor([0.0, 0.0])
    oracle = compute_oracle_k(losses_per_k, L_full, epsilon, K_CHOICES)
    # all > L_full, none satisfy < 0 → defaults to K_max = 34
    assert (oracle == 34).all()


def test_compute_oracle_k_eps_huge_picks_smallest():
    """ε very large: smallest k satisfies → all oracle = min(K_CHOICES)."""
    L_full = torch.tensor([0.5, 0.5])
    losses_per_k = {
        1: torch.tensor([100.0, 100.0]),
        2: torch.tensor([50.0, 50.0]),
        4: torch.tensor([20.0, 20.0]),
        8: torch.tensor([5.0, 5.0]),
        16: torch.tensor([1.0, 1.0]),
        34: L_full,
    }
    epsilon = torch.tensor([1000.0, 1000.0])
    oracle = compute_oracle_k(losses_per_k, L_full, epsilon, K_CHOICES)
    assert (oracle == 1).all()


def test_compute_oracle_k_intermediate_eps():
    """Intermediate ε: picks smallest k satisfying constraint."""
    L_full = torch.tensor([0.0])
    losses_per_k = {
        1: torch.tensor([1.0]),   # delta = 1.0
        2: torch.tensor([0.5]),   # delta = 0.5
        4: torch.tensor([0.2]),   # delta = 0.2
        8: torch.tensor([0.05]),  # delta = 0.05
        16: torch.tensor([0.01]),
        34: torch.tensor([0.0]),
    }
    # ε = 0.1: k=4 (delta=0.2 fails), k=8 (delta=0.05 < 0.1 passes) → oracle=8
    oracle = compute_oracle_k(losses_per_k, L_full, torch.tensor([0.1]), K_CHOICES)
    assert oracle.item() == 8
    # ε = 0.3: k=4 satisfies → oracle=4
    oracle = compute_oracle_k(losses_per_k, L_full, torch.tensor([0.3]), K_CHOICES)
    assert oracle.item() == 4


def test_compute_oracle_k_non_monotonic_no_error():
    """Loss not monotone in k (rare but possible): should not throw."""
    L_full = torch.tensor([0.0])
    losses_per_k = {
        1: torch.tensor([1.0]),
        2: torch.tensor([0.3]),
        4: torch.tensor([0.6]),  # back up — non-monotone
        8: torch.tensor([0.05]),
        16: torch.tensor([0.01]),
        34: torch.tensor([0.0]),
    }
    # ε=0.1: k=1 fail, k=2 fail (0.3 > 0.1), k=4 fail (0.6 > 0.1), k=8 pass (0.05<0.1) → 8
    oracle = compute_oracle_k(losses_per_k, L_full, torch.tensor([0.1]), K_CHOICES)
    assert oracle.item() == 8


def test_compute_oracle_k_per_chunk_independence():
    """Per-chunk oracle: chunks should be independent.
    Build a 2-chunk case where chunk-0 wants k=2 and chunk-1 wants k=8."""
    L_full = torch.tensor([0.0])
    losses_c0 = {
        1: torch.tensor([10.0]),
        2: torch.tensor([0.05]),    # passes ε=0.1
        4: torch.tensor([0.04]),
        8: torch.tensor([0.03]),
        16: torch.tensor([0.02]),
        34: L_full,
    }
    losses_c1 = {
        1: torch.tensor([10.0]),
        2: torch.tensor([5.0]),    # fails
        4: torch.tensor([1.0]),    # fails
        8: torch.tensor([0.05]),   # passes
        16: torch.tensor([0.03]),
        34: L_full,
    }
    out = compute_oracle_k_per_chunk(
        [losses_c0, losses_c1], L_full, torch.tensor([0.1]), K_CHOICES
    )
    assert len(out) == 2
    assert out[0].item() == 2
    assert out[1].item() == 8


# =====================================================================
# oracle_to_per_slot_labels
# =====================================================================

def test_oracle_to_per_slot_labels_basic():
    """oracle_k=3, m_c=5 → [1, 1, 1, 0, 0] (slots 0,1,2 should be kept)."""
    oracle_k = torch.tensor([3])
    y = oracle_to_per_slot_labels(oracle_k, max_slots=5)
    assert y.shape == (1, 5)
    assert torch.allclose(y, torch.tensor([[1., 1., 1., 0., 0.]]))


def test_oracle_to_per_slot_labels_monotone():
    """Labels must be monotone non-increasing along slot dim."""
    oracle_k = torch.tensor([0, 1, 4, 8])
    y = oracle_to_per_slot_labels(oracle_k, max_slots=8)
    diffs = y[:, 1:] - y[:, :-1]
    assert (diffs <= 0).all()
    # oracle_k=0 → all zeros
    assert (y[0] == 0).all()
    # oracle_k=8 → all ones
    assert (y[3] == 1).all()


# =====================================================================
# KPredictor
# =====================================================================

def test_kpredictor_param_count_in_range():
    """After per-layer reduction, predictor params should be roughly 1M
    (in tiny test config, just a few hundred K). For production dims (32 layers,
    d_kv=4096, d_layer_low=4) the count is ~944K. Here just sanity check finite,
    and confirm scale: well below 10M, and dominated by layer_proj."""
    p = _make_predictor()
    total = sum(t.numel() for t in p.parameters())
    layer_proj = sum(t.numel() for t in p.layer_proj.parameters())
    # In production sizing this would be checked at >800K; in mini test config
    # we just ensure layer_proj is the dominant chunk
    assert layer_proj > 0
    assert total > 0
    # Layer_proj contributes a significant fraction (at minimum, not negligible)
    assert layer_proj / total > 0.05


def test_kpredictor_forward_returns_logit_list():
    """KPredictor.forward returns list of [B, m_c] LOGIT tensors (no sigmoid)."""
    p = _make_predictor()
    B, total_M = 2, 7
    per_chunk = [3, 4]
    anchor_kv = torch.randn(B, NUM_LAYERS, 2, total_M, D_KV)
    epsilon = torch.tensor([0.05, 0.1])
    out = p(anchor_kv, per_chunk, epsilon)
    assert isinstance(out, list)
    assert len(out) == 2
    assert out[0].shape == (B, 3)
    assert out[1].shape == (B, 4)
    # Logits can be any real number (not [0,1])
    assert torch.isfinite(out[0]).all()


def test_kpredictor_keep_everything_init():
    """r2 init: head last-layer has small-std weight + bias=+4 so every slot's
    'continue' logit ≈ +4 at init → high cumulative keep-prob → predict_k starts
    near max_slots ('keep everything, let training prune'). (Updated 2026-06-29:
    the old near-zero-logit expectation predated the bias=4.0 survival prior.)"""
    p = _make_predictor()
    anchor_kv = torch.randn(1, NUM_LAYERS, 2, MAX_SLOTS, D_KV)
    epsilon = torch.tensor([0.05])
    logits = p(anchor_kv, [MAX_SLOTS], epsilon)[0]
    assert (logits > 2.0).all(), f"init logits should be ≈+4, got {logits}"
    _, ks_b0 = p.predict_k(anchor_kv, [MAX_SLOTS], epsilon)
    assert ks_b0[0] >= MAX_SLOTS - 1   # near-max k at init


def test_kpredictor_eps_conditional():
    """Different ε produces different logits (after head warmup)."""
    p = _make_predictor()
    # Train the head a tiny bit so it's not all-zero
    for layer in p.head.modules():
        if isinstance(layer, torch.nn.Linear):
            torch.nn.init.normal_(layer.weight, std=0.1)
            torch.nn.init.normal_(layer.bias, std=0.1)
    anchor_kv = torch.randn(1, NUM_LAYERS, 2, MAX_SLOTS, D_KV)
    out_low = p(anchor_kv, [MAX_SLOTS], torch.tensor([0.01]))[0]
    out_high = p(anchor_kv, [MAX_SLOTS], torch.tensor([0.2]))[0]
    # Should produce different outputs (head is the same, only eps embedding changed)
    assert not torch.allclose(out_low, out_high, atol=1e-4)


def test_kpredictor_predict_k_returns_ints_in_range():
    """predict_k returns (list[list[int]], list[int]); ks_b0 ∈ [min_k, m_c]."""
    p = _make_predictor()
    p.eval()
    per_chunk = [3, 5, 4]
    total_M = sum(per_chunk)
    anchor_kv = torch.randn(1, NUM_LAYERS, 2, total_M, D_KV)
    ks_full, ks_b0 = p.predict_k(anchor_kv, per_chunk, torch.tensor([0.05]))
    assert isinstance(ks_full, list) and isinstance(ks_b0, list)
    assert len(ks_full) == 3 and len(ks_b0) == 3
    for row, k0, m in zip(ks_full, ks_b0, per_chunk):
        assert isinstance(row, list) and len(row) == 1   # B=1
        assert isinstance(k0, int)
        assert 1 <= k0 <= m   # min_k=1 by default


def test_kpredictor_predict_k_cummin_is_monotone():
    """After cummin projection, q_chain must be monotone non-increasing.
    Construct a case where raw sigmoid(logit) is NOT monotone, verify cummin
    inside predict_k handles it (k_pred should equal the cummin-monotone count)."""
    p = _make_predictor()
    p.eval()
    # Manually replace head with one that produces non-monotone outputs
    # Use a known sequence by patching head output:
    # We force a specific logit pattern via monkey-patching .head
    class FixedHead(torch.nn.Module):
        def __init__(self, target):
            super().__init__()
            self.target = target
        def forward(self, x):
            # x: [B, m, d_hidden] → return [B, m, 1] mapping per-slot
            B, m, _ = x.shape
            return self.target[None, :m, None].expand(B, m, 1).clone()
    # Logits chosen so sigmoid = [0.9, 0.3, 0.5, 0.2, 0.1] — NOT monotone (0.3<0.5)
    # cummin should yield [0.9, 0.3, 0.3, 0.2, 0.1] — predict_k > 0.5 → 1
    logit_vals = torch.tensor([2.2, -0.85, 0.0, -1.39, -2.20])  # ≈ [0.9, 0.3, 0.5, 0.2, 0.1]
    p.head = FixedHead(logit_vals)
    anchor_kv = torch.randn(1, NUM_LAYERS, 2, 5, D_KV)
    _, ks_b0 = p.predict_k(anchor_kv, [5], torch.tensor([0.05]))
    # cummin: [0.9, 0.3, 0.3, 0.2, 0.1]; > 0.5 → only slot 0 → k=1
    assert ks_b0[0] == 1


def test_kpredictor_grad_flow_to_layer_proj():
    """BCE loss must produce gradients on KPredictor params (incl. layer_proj),
    but oracle labels must be detached (no grad to them)."""
    p = _make_predictor()
    B, total_M = 1, MAX_SLOTS
    anchor_kv = torch.randn(B, NUM_LAYERS, 2, total_M, D_KV)
    epsilon = torch.tensor([0.05])
    logit_list = p(anchor_kv, [total_M], epsilon)
    # Oracle: arbitrary integer; build target tensor
    oracle_per_chunk = [torch.tensor([3])]
    loss = predictor_loss(logit_list, [total_M], oracle_per_chunk)
    loss.backward()
    # Check grads on layer_proj
    layer_proj_grads = [layer.weight.grad for layer in p.layer_proj]
    assert all(g is not None and g.abs().sum() > 0 for g in layer_proj_grads), \
        "layer_proj should have non-zero gradients"
    # Head must have grad
    head_last = list(p.head.modules())[-1]
    if hasattr(head_last, "weight"):
        assert head_last.weight.grad is not None
    # Oracle labels should not require grad
    assert not oracle_per_chunk[0].requires_grad


# =====================================================================
# predictor_loss
# =====================================================================

def test_predictor_loss_zero_when_logits_match_targets():
    """If logits are large positive on '1'-slots and large negative on '0'-slots,
    BCE should be near zero."""
    # oracle_k=3, m_c=5 → target=[1,1,1,0,0]
    big = 20.0
    logit = torch.tensor([[big, big, big, -big, -big]])
    loss = predictor_loss([logit], [5], [torch.tensor([3])])
    assert loss.item() < 1e-7


def test_predictor_loss_weighted_monotone_survival():
    """predictor_loss is the weighted BCE on the MONOTONE keep-prob p_i (survival
    head), NOT a plain per-slot bce_with_logits. Confirm against the same
    monotone_keep_logprob the head/predict_k use. (Updated 2026-06-29: the old
    plain-BCE expectation predated the survival-head + pos_weight redesign.)"""
    from model.modeling import monotone_keep_logprob
    logit = torch.tensor([[1.0, -1.0]])
    oracle_k = torch.tensor([1])
    pos_weight = 4.0
    # per-slot labels y_i = 1[oracle_k >= i+1] = [1, 0]
    y = torch.tensor([[1.0, 0.0]])
    log_p, log1m_p = monotone_keep_logprob(logit)
    expected = -(pos_weight * y * log_p + (1.0 - y) * log1m_p).sum() / 2
    got = predictor_loss([logit], [2], [oracle_k], pos_weight=pos_weight)
    assert torch.allclose(got, expected, atol=1e-6)


# =====================================================================
# K_CHOICES sanity
# =====================================================================

def test_k_choices_sorted_ascending():
    """K_CHOICES must be ascending; max must be 34 (or the production max_slots).
    r2 spec: [1, 2, 4, 8, 16, 34]."""
    assert K_CHOICES == sorted(K_CHOICES)
    assert K_CHOICES[0] >= 1
    assert max(K_CHOICES) == 34
    # r2 spec
    assert K_CHOICES == [1, 2, 4, 8, 16, 34]


# =====================================================================
# KV-aware redesign variants (2026-06-29): Idea1 residual-energy,
# Idea2 kv-headstats, Idea3 gru mixer, Idea4 eps-FiLM / eps-monotone.
# =====================================================================

def _make_variant(**kw):
    """KPredictor with the small test dims + arbitrary variant flags."""
    base = dict(
        num_layers=NUM_LAYERS, d_kv=D_KV, d_layer_low=D_LAYER_LOW,
        max_slots=MAX_SLOTS, d_hidden=D_HIDDEN, attn_heads=ATTN_HEADS,
        eps_embed_dim=16, num_kv_heads=NUM_HEADS, head_dim=HEAD_DIM,
    )
    base.update(kw)
    return KPredictor(**base)


# Each variant flag set: (label, kwargs)
_VARIANTS = [
    ("linear4_default", {}),
    ("kv_headstats",    {"feat_mode": "kv_headstats"}),
    ("residual_energy", {"use_residual_energy": True}),
    ("gru_mixer",       {"slot_mixer": "gru"}),
    ("eps_film",        {"eps_mode": "film"}),
    ("eps_monotone",    {"eps_monotone": True}),
    ("kv_marginal",     {"feat_mode": "kv_headstats", "use_residual_energy": True}),
    ("full_combo",      {"feat_mode": "kv_headstats", "use_residual_energy": True,
                         "slot_mixer": "gru", "eps_mode": "film"}),
]


@pytest.mark.parametrize("label,kw", _VARIANTS)
def test_variant_forward_shapes(label, kw):
    """Every variant returns a per-chunk list of [B, m_c] finite logits."""
    p = _make_variant(**kw)
    B, per_chunk = 1, [3, 5]
    total_M = sum(per_chunk)
    anchor_kv = torch.randn(B, NUM_LAYERS, 2, total_M, D_KV)
    out = p(anchor_kv, per_chunk, torch.tensor([0.05]))
    assert [o.shape for o in out] == [(B, 3), (B, 5)]
    assert all(torch.isfinite(o).all() for o in out)


@pytest.mark.parametrize("label,kw", _VARIANTS)
def test_variant_monotone_keep_prob_non_increasing(label, kw):
    """Survival head keeps p_i = ∏ sigmoid(a_j) non-increasing regardless of
    featurization / mixer / eps-conditioning (head_mode='monotone')."""
    from model.modeling import keep_logprob
    p = _make_variant(**kw)
    anchor_kv = torch.randn(1, NUM_LAYERS, 2, MAX_SLOTS, D_KV)
    logits = p(anchor_kv, [MAX_SLOTS], torch.tensor([0.05]))[0]
    log_p, _ = keep_logprob(logits, "monotone")
    p_keep = log_p.exp()[0]
    diffs = p_keep[1:] - p_keep[:-1]
    assert (diffs <= 1e-5).all(), f"{label}: keep-prob not non-increasing: {p_keep}"


@pytest.mark.parametrize("label,kw", _VARIANTS)
def test_variant_predict_k_in_range(label, kw):
    """predict_k stays in [1, m_c] for every variant."""
    p = _make_variant(**kw)
    p.eval()
    per_chunk = [3, 5, 4]
    anchor_kv = torch.randn(1, NUM_LAYERS, 2, sum(per_chunk), D_KV)
    _, ks_b0 = p.predict_k(anchor_kv, per_chunk, torch.tensor([0.05]))
    assert all(1 <= k <= m for k, m in zip(ks_b0, per_chunk))


def test_eps_monotone_k_non_increasing_in_eps():
    """Idea4: with eps_monotone, predicted k is monotone non-increasing in ε
    by construction (ε enters only via a non-negative subtractive term)."""
    p = _make_variant(eps_monotone=True)
    p.eval()
    # Make the head non-trivial so logits aren't all identical.
    for m in p.head.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.normal_(m.weight, std=0.3)
    # Force a clearly-positive slope so the effect is visible at test scale.
    with torch.no_grad():
        p.eps_mono_scale.fill_(2.0)
    per_chunk = [MAX_SLOTS, MAX_SLOTS]
    anchor_kv = torch.randn(1, NUM_LAYERS, 2, sum(per_chunk), D_KV)
    prev = None
    for eps in [0.0, 0.05, 0.1, 0.2, 0.5, 1.0]:
        _, ks = p.predict_k(anchor_kv, per_chunk, torch.tensor([eps]))
        tot = sum(ks)
        if prev is not None:
            assert tot <= prev, f"k increased with ε: {tot} > {prev} at ε={eps}"
        prev = tot


def test_kv_headstats_reduced_dim_and_no_shared_layer_proj():
    """Idea2: kv_headstats reduces to L*d_layer_low (K/V share a per-layer Linear
    over their concatenated per-head stats, not a single d_kv flatten)."""
    p = _make_variant(feat_mode="kv_headstats")
    assert p._reduced_dim == NUM_LAYERS * D_LAYER_LOW
    # per-layer Linear in-features = 2 (K,V) * num_kv_heads * 3 stats
    assert p.layer_proj[0].in_features == 2 * NUM_HEADS * 3
    anchor_kv = torch.randn(1, NUM_LAYERS, 2, 6, D_KV)
    red = p._reduce_kv(anchor_kv)
    assert red.shape == (1, 6, NUM_LAYERS * D_LAYER_LOW)


def test_residual_energy_in_unit_range_and_first_slot_full():
    """Idea1: residual-energy ratio r_i ∈ [0,1]; the first slot is fully novel
    (r_0 == 1, empty prefix)."""
    anchor_kv = torch.randn(1, NUM_LAYERS, 2, 5, D_KV)
    e = KPredictor._residual_energy(anchor_kv)
    assert e.shape == (1, 5, 1)
    assert (e >= -1e-5).all() and (e <= 1.0 + 1e-5).all()
    assert abs(e[0, 0, 0].item() - 1.0) < 1e-4   # first slot: empty prefix


def test_default_unflagged_matches_original_path():
    """An unflagged KPredictor must keep the original layer_proj(d_kv→low) and
    reduced_dim = L*2*low (no behavioural drift for the v4 main config)."""
    p = _make_variant()
    assert p.feat_mode == "linear4"
    assert p.use_residual_energy is False
    assert p.slot_mixer == "mha"
    assert p.eps_mode == "add"
    assert p._reduced_dim == NUM_LAYERS * 2 * D_LAYER_LOW
    assert p.layer_proj[0].in_features == D_KV
    assert hasattr(p, "attn") and not hasattr(p, "slot_gru")


# =====================================================================
# Within-sample ranking loss (2026-06-30): objective fix for the
# pointwise-BCE mean-k collapse (F101/F102).
# =====================================================================

def test_rank_loss_corr_perfect_and_anti():
    from model.modeling import _within_sample_rank_loss
    pred = torch.tensor([1.0, 2.0, 3.0, 4.0])
    orac = torch.tensor([2.0, 4.0, 6.0, 8.0])     # perfectly correlated
    assert _within_sample_rank_loss(pred, orac, "corr").item() < 1e-5     # 1-1=0
    anti = torch.tensor([8.0, 6.0, 4.0, 2.0])      # anti-correlated
    assert abs(_within_sample_rank_loss(pred, anti, "corr").item() - 2.0) < 1e-4  # 1-(-1)=2


def test_rank_loss_degenerate_zero():
    from model.modeling import _within_sample_rank_loss
    # <2 chunks → 0
    assert _within_sample_rank_loss(torch.tensor([3.0]), torch.tensor([5.0]), "corr").item() == 0.0
    # no oracle variation → 0 (undefined corr, skipped)
    flat = torch.tensor([2.0, 2.0, 2.0])
    assert _within_sample_rank_loss(torch.tensor([1.0, 2.0, 3.0]), flat, "corr").item() == 0.0


def test_rank_loss_pairwise_finite_and_sign():
    from model.modeling import _within_sample_rank_loss
    pred = torch.tensor([1.0, 3.0, 2.0]); orac = torch.tensor([1.0, 3.0, 2.0])
    good = _within_sample_rank_loss(pred, orac, "pairwise").item()
    bad = _within_sample_rank_loss(torch.tensor([3.0, 1.0, 2.0]), orac, "pairwise").item()
    assert good < bad and good >= 0.0


def test_predictor_loss_rank_weight_changes_loss_and_default_noop():
    """rank_weight=0 == original BCE; rank_weight>0 adds the ranking term."""
    # 3 chunks, B=1, distinct oracle k so there IS within-sample variation
    logits = [torch.randn(1, MAX_SLOTS) for _ in range(3)]
    per_chunk = [MAX_SLOTS]*3
    oracle = [torch.tensor([1]), torch.tensor([4]), torch.tensor([7])]
    base = predictor_loss(logits, per_chunk, oracle, pos_weight=2.0)
    base0 = predictor_loss(logits, per_chunk, oracle, pos_weight=2.0, rank_weight=0.0)
    assert torch.allclose(base, base0)                       # default no-op
    ranked = predictor_loss(logits, per_chunk, oracle, pos_weight=2.0, rank_weight=1.0)
    assert not torch.allclose(base, ranked)                  # rank term active
    assert torch.isfinite(ranked)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
