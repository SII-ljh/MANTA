"""Chunk-MAE pre-training entry point.

Single-process training loop (DDP can be added later for GPU). On CPU this
is enough for smoke tests + small-scale ablation.

Usage:
    python -m pretrain.chunk_mae.train \
        --shard_dir data/precomputed_kv/unified_inst/train \
        --output_dir pretrain/output/chunk_mae_smoke \
        --num_steps 50 --latent_dim 128 --num_fusion_layers 1 \
        --max_samples 16 --device cpu
"""

import argparse
import json
import os
import sys
import time

import torch
from torch.utils.data import DataLoader

# Make the project root importable when run as a module or as a script.
_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from cross_chunk.compressor import CrossChunkKVCompressor  # noqa: E402

from pretrain.chunk_mae.dataset import ChunkMAEDataset  # noqa: E402
from pretrain.chunk_mae.model import ChunkMAEModel  # noqa: E402


def _identity_collate(batch):
    """IterableDataset already yields one sample at a time; bypass collation."""
    return batch[0]


def build_argparser():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--shard_dir", required=True)
    p.add_argument("--mask_ratio", type=float, default=0.25)
    p.add_argument("--min_chunks", type=int, default=4)
    p.add_argument("--max_chunks", type=int, default=32)
    p.add_argument("--max_samples", type=int, default=None,
                   help="Cap on samples (after filtering). None = unlimited.")
    p.add_argument("--cycle", action="store_true",
                   help="Cycle shards forever; trainer terminates on --num_steps.")
    # Compressor architecture (defaults match V3-Both-Additive but slimmer).
    p.add_argument("--num_layers", type=int, default=32)
    p.add_argument("--num_kv_heads", type=int, default=8)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=512)
    p.add_argument("--num_fusion_layers", type=int, default=4)
    p.add_argument("--num_fusion_heads", type=int, default=8)
    p.add_argument("--mem_size", type=int, default=34)
    p.add_argument("--max_chunks_arch", type=int, default=64,
                   help="Compressor max_chunks (chunk pos embedding size).")
    p.add_argument("--attention_window", type=int, default=None)
    p.add_argument("--pos_encoding_mode", default="intra_chunk")
    p.add_argument("--summary_pool_mode", default="mean")
    p.add_argument("--num_summary_layers", type=int, default=2)
    p.add_argument("--gate_init", type=float, default=1.0,
                   help="High gate init for pretrain — fusion needs full V_delta capacity.")
    p.add_argument("--fusion_strategy", default="additive")
    p.add_argument("--local_stream_enabled", type=int, default=1,
                   help="1 = enabled (V3 default), 0 = SummaryOnly (E21).")
    # Training
    p.add_argument("--num_steps", type=int, default=1000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--log_every", type=int, default=10)
    p.add_argument("--save_every", type=int, default=0,
                   help="0 = save once at end only.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dtype", default="float32",
                   choices=["float32", "bfloat16"],
                   help="Compute / data dtype. CPU should use float32.")
    p.add_argument("--output_dir", required=True)
    return p


def build_compressor(args):
    return CrossChunkKVCompressor(
        num_layers=args.num_layers,
        num_kv_heads=args.num_kv_heads,
        head_dim=args.head_dim,
        latent_dim=args.latent_dim,
        num_fusion_layers=args.num_fusion_layers,
        num_fusion_heads=args.num_fusion_heads,
        mem_size=args.mem_size,
        max_chunks=args.max_chunks_arch,
        attention_window=args.attention_window,
        pos_encoding_mode=args.pos_encoding_mode,
        summary_pool_mode=args.summary_pool_mode,
        num_summary_layers=args.num_summary_layers,
        gate_init=args.gate_init,
        fusion_strategy=args.fusion_strategy,
        local_stream_enabled=bool(args.local_stream_enabled),
    )


def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device(args.device)

    dataset = ChunkMAEDataset(
        shard_dir=args.shard_dir,
        mask_ratio=args.mask_ratio,
        min_chunks=args.min_chunks,
        max_chunks=args.max_chunks,
        dtype=dtype,
        seed=args.seed,
        max_samples=args.max_samples,
        cycle=args.cycle,
    )
    loader = DataLoader(
        dataset, batch_size=1, num_workers=0, collate_fn=_identity_collate,
    )

    compressor = build_compressor(args)
    model = ChunkMAEModel(compressor).to(device)
    model.train()

    optim = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    log_path = os.path.join(args.output_dir, "train_log.jsonl")
    log_f = open(log_path, "w", encoding="utf-8")

    step = 0
    t_start = time.time()
    losses = []

    for sample in loader:
        # Move tensors to device.
        sample["chunk_kvs"] = [
            tuple(
                (k.to(device), v.to(device))
                for (k, v) in layers
            )
            for layers in sample["chunk_kvs"]
        ]

        loss, metrics = model(sample, return_metrics=True)
        optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), args.grad_clip,
        ).item()
        optim.step()

        step += 1
        losses.append(loss.item())

        if step % args.log_every == 0 or step == 1:
            elapsed = time.time() - t_start
            line = {
                "step": step,
                "loss": loss.item(),
                "loss_ma10": sum(losses[-10:]) / min(len(losses), 10),
                "baseline_zero_mse": metrics["baseline_zero_mse"],
                "grad_norm": grad_norm,
                "num_chunks": metrics["num_chunks"],
                "num_masked_chunks": metrics["num_masked_chunks"],
                "elapsed_s": round(elapsed, 1),
            }
            print(json.dumps(line), flush=True)
            log_f.write(json.dumps(line) + "\n")
            log_f.flush()

        if args.save_every and step % args.save_every == 0:
            ckpt = os.path.join(args.output_dir, f"ckpt_step{step}.pt")
            torch.save({"step": step, "model": model.state_dict()}, ckpt)

        if step >= args.num_steps:
            break

    # Final save
    ckpt = os.path.join(args.output_dir, "ckpt_final.pt")
    torch.save({"step": step, "model": model.state_dict(), "args": vars(args)}, ckpt)
    log_f.close()

    summary = {
        "steps_run": step,
        "first_loss": losses[0] if losses else None,
        "last_loss": losses[-1] if losses else None,
        "min_loss": min(losses) if losses else None,
        "elapsed_s": round(time.time() - t_start, 1),
        "ckpt": ckpt,
    }
    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("=== summary ===", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
