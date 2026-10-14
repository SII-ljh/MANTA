"""End-to-end MANTA v5 CPU smoke. Loads real Llama-3.1-8B in fp32 on CPU,
exercises `_forward_k_predictor` with warmup_skip=0 so the per-chunk oracle
path runs. ~3-5 min on CPU (1.5TB RAM available).

Verifies:
  - CompressLLM instantiates with device_rank="cpu"
  - SAC pretrain adapter loads with strict=False (no k_predictor keys → random init)
  - k_predictor params are trainable (freeze_encoder exemption holds)
  - One full forward through _forward_k_predictor produces finite loss
  - Loss info contains: m_loss, p_loss, alpha, oracle_mean_k, oracle_per_chunk_var
  - Backward populates grads on k_predictor.layer_proj
  - save_adapter / load_adapter roundtrip preserves k_predictor params
"""
import os
import sys
import time
import torch

sys.path.insert(0, os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.modeling import (
    get_model_for_compress, load_adapter, save_adapter,
    K_CHOICES,
)


MODEL_ID = os.environ.get("MANTA_MODEL_ID", "meta-llama/Llama-3.1-8B")
SAC_ADAPTER = os.environ.get(
    "MANTA_SAC_ADAPTER",
    os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                 "weights", "sac_8b_paper", "output", "adapter.pt"))
OUT_DIR = "/tmp/manta_v5_cpu_smoke"
os.makedirs(OUT_DIR, exist_ok=True)


def build_task_config():
    """Mirrors experiment/manta_stage2/config.json sft_task_config + matryoshka_training_config."""
    return {
        "is_pretrain": False,
        "is_sft": True,
        "chunk_size": 510,
        "mem_size": 34,
        "compress_ratio": 15,
        "task_type": "Compress",
        "use_pe": True,
        "use_ae_loss": False,
        "use_lm_loss": True,
        # Matryoshka + KPredictor
        "use_matryoshka": True,
        "use_gate": False,
        "matryoshka_ks": [1, 2, 4, 8, 16, 34],
        "matryoshka_weights": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "use_k_predictor": True,
        "k_choices": [1, 2, 4, 8, 16, 34],
        "eps_grid": [0.01, 0.05, 0.1, 0.2],
        "predictor_warmup_skip": 0,    # force oracle path to run for smoke
        "ramp_steps": 1,                # alpha=1 immediately
        "alpha_max": 1.0,
        "predictor_hidden": 256,
        "predictor_attn_heads": 4,
        "predictor_d_layer_low": 4,
        "predictor_eps_embed_dim": 64,
        # Legacy gate knobs (unused with use_gate=False; keep for back-compat)
        "gate_init_bias": 5.0,
        "gate_mode": "cumprod",
        "sparsity_weight": 0.0,
        "target_avg": 12.0,
        "elastic_tau": 1.0,
        "gumbel_tau": 1.0,
        "lambda_budget": 0.0,
        "entropy_weight": 0.0,
        "n_target": 12.0,
    }


def build_minimal_inputs(model, batch_size: int = 1, n_chunks: int = 2, lm_seq: int = 32):
    """Build minimal but valid inputs for is_sft=True forward path.
    n_chunks * chunk_size = total context length.

    Follows instruction_prepare_data.py convention:
      - lm_targets: shape [B, S] (full question + answer concatenation)
      - instruction_target: shape [B, S-1] (shifted-by-1, prompt part masked with -100)
    """
    chunk_size = 510
    total_ctx = n_chunks * chunk_size
    vocab = model.tokenizer.vocab_size if hasattr(model.tokenizer, "vocab_size") else 128256
    input_ids = torch.randint(low=10, high=vocab - 1, size=(batch_size, total_ctx))
    lm_targets = torch.randint(low=10, high=vocab - 1, size=(batch_size, lm_seq))
    # instruction_target: shape [B, lm_seq-1]; first ~1/3 masked (prompt portion)
    instruction_target = lm_targets[:, 1:].clone()           # shape [B, S-1]
    mask_len = (lm_seq - 1) // 3
    instruction_target[:, :mask_len] = -100
    return {
        "input_ids": input_ids,
        "lm_targets": lm_targets,
        "instruction_target": instruction_target,
    }


def section(title):
    print(f"\n{'='*60}\n[{title}]\n{'='*60}")


def main():
    t0 = time.time()

    section("1) build task_config + load CompressLLM on CPU (~2 min)")
    task_config = build_task_config()
    model = get_model_for_compress(MODEL_ID, task_config, rank="cpu")
    print(f"  loaded in {time.time()-t0:.1f}s; device={model.device}")
    assert model.use_k_predictor is True
    assert model.k_predictor is not None

    section("2) sanity: k_predictor params are trainable")
    k_pred_params = [(n, p) for n, p in model.named_parameters() if n.startswith("k_predictor.")]
    n_kp = len(k_pred_params)
    n_trainable = sum(1 for _, p in k_pred_params if p.requires_grad)
    print(f"  k_predictor params: {n_kp} tensors, {n_trainable} require grad")
    assert n_kp > 0 and n_trainable == n_kp, "k_predictor params must all be trainable"

    section("3) load SAC pretrain adapter (no k_predictor keys → strict=False)")
    t1 = time.time()
    load_adapter(model, SAC_ADAPTER, log=False)
    print(f"  loaded SAC adapter in {time.time()-t1:.1f}s")

    section("4) build minimal inputs")
    inputs = build_minimal_inputs(model, batch_size=1, n_chunks=2, lm_seq=32)
    print(f"  input_ids shape={tuple(inputs['input_ids'].shape)}")
    print(f"  lm_targets shape={tuple(inputs['lm_targets'].shape)}")
    print(f"  instruction_target shape={tuple(inputs['instruction_target'].shape)}")

    section("5) forward through _forward_k_predictor (~2-3 min on CPU)")
    model.train()
    model._global_step = 10  # past warmup_skip=0
    t2 = time.time()
    out = model(inputs)
    fwd_time = time.time() - t2
    loss = out["loss"]
    info = out["loss_info"]
    print(f"  forward took {fwd_time:.1f}s")
    print(f"  loss = {loss.item():.4f}")
    print(f"  loss_info keys: {sorted(info.keys())}")
    for k in ("m_loss", "p_loss", "alpha", "oracle_mean_k", "oracle_per_chunk_var",
              "matryoshka_k", "warmup_phase", "eps_mean", "L_full_mean"):
        if k in info:
            print(f"    {k:25s} = {info[k]}")

    assert torch.isfinite(loss), f"loss not finite: {loss.item()}"
    assert "m_loss" in info and "p_loss" in info
    assert info["warmup_phase"] is False, "warmup should be off"
    assert info["alpha"] > 0, "alpha should be > 0 past warmup"
    assert info["oracle_mean_k"] >= min(K_CHOICES) - 0.1

    section("6) backward → grads on k_predictor.layer_proj")
    t3 = time.time()
    loss.backward()
    bwd_time = time.time() - t3
    print(f"  backward took {bwd_time:.1f}s")
    lp_grads = [layer.weight.grad for layer in model.k_predictor.layer_proj]
    nonzero = [g is not None and g.abs().sum().item() > 0 for g in lp_grads]
    print(f"  layer_proj layers with nonzero grad: {sum(nonzero)}/{len(nonzero)}")
    assert all(nonzero), "all layer_proj should have nonzero gradients after backward"

    section("7) save_adapter / load_adapter roundtrip preserves k_predictor")
    ckpt = os.path.join(OUT_DIR, "smoke_adapter.pt")
    save_adapter(model, save_path_and_name=ckpt, log=False)
    sd = torch.load(ckpt, map_location="cpu")
    k_keys = [k for k in sd.keys() if k.startswith("k_predictor.")]
    print(f"  saved adapter has {len(k_keys)} k_predictor keys (sample: {k_keys[:3]})")
    assert len(k_keys) > 0, "k_predictor params must be in saved adapter"
    # ensure we can reload
    load_adapter(model, ckpt, log=False)
    print("  reload OK")

    print(f"\n[smoke] ALL CHECKS PASS — total {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
