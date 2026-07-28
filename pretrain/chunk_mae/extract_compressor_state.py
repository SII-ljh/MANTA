"""Extract a CrossChunkKVCompressor-only state_dict from a ChunkMAE checkpoint.

The pre-train ckpt contains both compressor weights (keys prefixed with
"compressor.") and the MAE-only ``mask_token``. The downstream SFT trainer
only needs the compressor weights — this script strips ``mask_token`` and
re-saves a flat dict whose keys are exactly ``compressor.<...>`` (so it can
be loaded directly into ``CrossChunkCompressLLM`` via
``model.load_state_dict(state, strict=False)``).

Usage:
    python -m pretrain.chunk_mae.extract_compressor_state \\
        --ckpt pretrain/output/chunk_mae_*/ckpt_final.pt \\
        --out  pretrain/output/chunk_mae_*/compressor_init.pt

The output also stores the pretrain ``args`` so the SFT side can verify
architecture compatibility when loading.
"""

import argparse
import os
import sys

import torch

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def extract_compressor_state(
    ckpt_path: str,
    out_path: str | None = None,
) -> dict:
    """Load a ChunkMAE ckpt, drop ``mask_token``, return + optionally save the
    compressor-only state.

    Returns:
        dict with keys:
            "state":      flat dict, keys "compressor.<...>" (compressor params).
            "pretrain_args": the original pretrain args (for verification).
            "step":        training step at which the ckpt was saved.
    """
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(ckpt_path)

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model" not in raw:
        raise ValueError(
            f"{ckpt_path} is not a ChunkMAE ckpt (missing 'model' key); keys={list(raw)}"
        )

    full_state = raw["model"]
    compressor_state = {
        k: v for k, v in full_state.items() if k.startswith("compressor.")
    }
    dropped = sorted(k for k in full_state if not k.startswith("compressor."))

    payload = {
        "state": compressor_state,
        "pretrain_args": raw.get("args"),
        "step": raw.get("step"),
        "dropped_keys": dropped,
    }

    if out_path is not None:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        torch.save(payload, out_path)

    return payload


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="ChunkMAE ckpt path (ckpt_final.pt)")
    p.add_argument("--out", required=True, help="Output compressor-init path")
    args = p.parse_args()

    payload = extract_compressor_state(args.ckpt, args.out)
    n_kept = len(payload["state"])
    print(f"[extract] kept {n_kept} compressor params, dropped: {payload['dropped_keys']}")
    print(f"[extract] saved to {args.out} (pretrain step={payload['step']})")


if __name__ == "__main__":
    main()
