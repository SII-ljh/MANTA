"""Production-size KPredictor smoke test (CPU, no LLM load).

Instantiates KPredictor with Llama-3.1-8B KV dims (num_kv_heads=8 GQA, head_dim=128
→ d_kv=1024) and runs a single forward+backward at realistic sizes. Catches
shape errors, dtype issues, OOM at production sizing without needing GPU.
"""
import os
import sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modeling import (
    KPredictor,
    extract_full_kv,
    truncate_kv_to_k,
    truncate_single_chunk_kv,
    compute_oracle_k_per_chunk,
    predictor_loss,
    K_CHOICES,
)

# Llama-3.1-8B (GQA): 32 layers, 8 KV heads, head_dim 128 → d_kv 1024
NUM_LAYERS = 32
NUM_KV_HEADS = 8
HEAD_DIM = 128
D_KV = NUM_KV_HEADS * HEAD_DIM   # 1024
MEM_SIZE = 34
B = 1
N_CHUNKS = 4
TOTAL_M = N_CHUNKS * MEM_SIZE


def main():
    torch.manual_seed(0)
    p = KPredictor(
        num_layers=NUM_LAYERS,
        d_kv=D_KV,
        d_layer_low=4,
        max_slots=MEM_SIZE,
        d_hidden=256,
        attn_heads=4,
        eps_embed_dim=64,
    )
    total = sum(t.numel() for t in p.parameters())
    print(f"[smoke] KPredictor params: {total:,} ({total/1e6:.3f}M)")
    assert 400_000 < total < 1_200_000, f"params out of expected range: {total}"

    # Mock encoder_past_key_values (bf16 like real KV cache)
    pkv = tuple(
        (torch.randn(B, NUM_KV_HEADS, TOTAL_M, HEAD_DIM, dtype=torch.bfloat16),
         torch.randn(B, NUM_KV_HEADS, TOTAL_M, HEAD_DIM, dtype=torch.bfloat16))
        for _ in range(NUM_LAYERS)
    )
    per_chunk = [MEM_SIZE] * N_CHUNKS

    # 1) extract_full_kv at production size + dtype
    anchor_kv = extract_full_kv(pkv)
    assert anchor_kv.shape == (B, NUM_LAYERS, 2, TOTAL_M, D_KV)
    assert anchor_kv.dtype == torch.bfloat16
    print(f"[smoke] anchor_kv shape={tuple(anchor_kv.shape)} dtype={anchor_kv.dtype}")

    # 2) Predictor forward (handles bf16 → fp32 internally)
    epsilon = torch.tensor([0.05])
    logit_list = p(anchor_kv, per_chunk, epsilon)
    assert len(logit_list) == N_CHUNKS
    for lg, m in zip(logit_list, per_chunk):
        assert lg.shape == (B, m)
    print(f"[smoke] forward OK; logits shape per chunk = {tuple(logit_list[0].shape)}")

    # 3) predict_k integer output
    p.eval()
    _ks_full, ks = p.predict_k(anchor_kv, per_chunk, epsilon)
    print(f"[smoke] predict_k = {ks}")
    assert all(1 <= k <= m for k, m in zip(ks, per_chunk))

    # 4) BCE backward — gradient flows to layer_proj
    p.train()
    oracle = [torch.tensor([5]) for _ in range(N_CHUNKS)]
    logit_list = p(anchor_kv, per_chunk, epsilon)
    loss = predictor_loss(logit_list, per_chunk, oracle)
    loss.backward()
    grad_total = sum(
        (l.weight.grad.abs().sum().item() if l.weight.grad is not None else 0.0)
        for l in p.layer_proj
    )
    print(f"[smoke] BCE backward OK; sum |layer_proj grad| = {grad_total:.4f}")
    assert grad_total > 0

    # 5) truncate helpers at production size
    pkv_t = truncate_kv_to_k(pkv, k=4, per_chunk_mem_sizes=per_chunk)
    assert pkv_t[0][0].shape == (B, NUM_KV_HEADS, 4 * N_CHUNKS, HEAD_DIM)
    pkv_s = truncate_single_chunk_kv(pkv, per_chunk, target_chunk=1, k=2)
    expected = MEM_SIZE * N_CHUNKS - (MEM_SIZE - 2)
    assert pkv_s[0][0].shape == (B, NUM_KV_HEADS, expected, HEAD_DIM)
    print(f"[smoke] truncate helpers OK at production shapes")

    print("[smoke] ALL CHECKS PASS")


if __name__ == "__main__":
    main()
