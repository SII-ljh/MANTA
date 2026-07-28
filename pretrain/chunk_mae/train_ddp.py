"""DDP Chunk-MAE pre-training for multi-GPU H100.

Spawned via mp.spawn (matches cross_chunk/trainer.py convention). Each rank
takes a disjoint slice of train shards (cycle=True per F17 NCCL fix); rank 0
handles validation, logging, and checkpointing.

Usage:
    PYTHONUNBUFFERED=1 python -m pretrain.chunk_mae.train_ddp \\
        --shard_dir data/precomputed_kv/unified_inst/train \\
        --output_dir pretrain/output/chunk_mae_4xh100 \\
        --num_steps 50000 --num_val_shards 8 --eval_every 5000 \\
        --warmup_steps 1000 --save_every 10000

NCCL hang prevention (FINDINGS F17):
    - cycle=True on the train dataset so per-rank exhaustion is impossible;
      the step counter is the sole termination criterion.
    - dist.barrier() before save & destroy_process_group.
"""

import argparse
import datetime
import glob as glob_module
import json
import math
import os
import random
import sys
import time

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from pretrain.chunk_mae.dataset import ChunkMAEDataset  # noqa: E402
from pretrain.chunk_mae.model import ChunkMAEModel  # noqa: E402
from pretrain.chunk_mae.train import (  # noqa: E402
    _identity_collate,
    build_compressor,
)


def _setup_dist(rank, world_size, port, nccl_timeout_sec):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    dist.init_process_group(
        "nccl",
        rank=rank,
        world_size=world_size,
        timeout=datetime.timedelta(seconds=nccl_timeout_sec),
    )
    torch.cuda.set_device(rank)


def _cleanup_dist():
    if dist.is_initialized():
        dist.destroy_process_group()


def _lr_schedule(step, warmup_steps, num_steps, lr_min_ratio):
    """Linear warmup -> cosine decay to lr_min_ratio of peak."""
    if warmup_steps > 0 and step < warmup_steps:
        return step / max(warmup_steps, 1)
    if num_steps <= warmup_steps:
        return 1.0
    progress = (step - warmup_steps) / max(num_steps - warmup_steps, 1)
    progress = min(max(progress, 0.0), 1.0)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return lr_min_ratio + (1 - lr_min_ratio) * cosine


def _move_sample(sample, device):
    sample["chunk_kvs"] = [
        tuple((k.to(device, non_blocking=True), v.to(device, non_blocking=True))
              for (k, v) in layers)
        for layers in sample["chunk_kvs"]
    ]
    return sample


@torch.no_grad()
def _run_eval(unwrapped_model, val_loader, device):
    if val_loader is None:
        return None
    was_training = unwrapped_model.training
    unwrapped_model.eval()
    losses, baselines = [], []
    for sample in val_loader:
        _move_sample(sample, device)
        loss, metrics = unwrapped_model(sample, return_metrics=True)
        losses.append(metrics["loss"])
        baselines.append(metrics["baseline_zero_mse"])
    if was_training:
        unwrapped_model.train()
    if not losses:
        return None
    return {
        "val_loss": sum(losses) / len(losses),
        "val_baseline_zero_mse": sum(baselines) / len(baselines),
        "val_n": len(losses),
    }


def _split_shards(shard_dir, num_val_shards, seed):
    all_shards = sorted(glob_module.glob(os.path.join(shard_dir, "shard_*.pt")))
    if not all_shards:
        raise FileNotFoundError(f"no shard_*.pt in {shard_dir}")
    if num_val_shards <= 0:
        train, val = all_shards, []
    elif num_val_shards >= len(all_shards):
        raise ValueError(
            f"num_val_shards={num_val_shards} >= total shards={len(all_shards)}"
        )
    else:
        train = all_shards[:-num_val_shards]
        val = all_shards[-num_val_shards:]
    random.Random(seed).shuffle(train)
    return train, val


def train_worker(rank, args, world_size):
    _setup_dist(rank, world_size, args.port, args.nccl_timeout_sec)
    torch.manual_seed(args.seed + rank)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    device = torch.device(f"cuda:{rank}")
    is_main = rank == 0

    if is_main:
        os.makedirs(args.output_dir, exist_ok=True)

    train_shards_global, val_shards = _split_shards(
        args.shard_dir, args.num_val_shards, args.seed,
    )
    # Round-robin shard slicing per rank (matches cross_chunk/trainer.py).
    train_shards_rank = train_shards_global[rank::world_size]

    if is_main:
        print(
            f"[setup] world_size={world_size}, train_shards={len(train_shards_global)} "
            f"(~{len(train_shards_rank)}/rank), val_shards={len(val_shards)}",
            flush=True,
        )

    train_ds = ChunkMAEDataset(
        shard_files=train_shards_rank,
        mask_ratio=args.mask_ratio,
        min_chunks=args.min_chunks,
        max_chunks=args.max_chunks,
        dtype=dtype,
        seed=args.seed + rank,
        max_samples=None,
        cycle=True,  # F17 — never naturally exhaust; step counter governs stop.
        shuffle_shards=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=1, num_workers=0, collate_fn=_identity_collate,
    )

    val_loader = None
    if is_main and val_shards and args.eval_every > 0:
        val_ds = ChunkMAEDataset(
            shard_files=val_shards,
            mask_ratio=args.mask_ratio,
            min_chunks=args.min_chunks,
            max_chunks=args.max_chunks,
            dtype=dtype,
            seed=args.seed + 99991,
            max_samples=args.val_max_samples,
            cycle=False,
            shuffle_shards=False,
        )
        val_loader = DataLoader(
            val_ds, batch_size=1, num_workers=0, collate_fn=_identity_collate,
        )

    compressor = build_compressor(args)
    model = ChunkMAEModel(compressor)
    if dtype == torch.bfloat16:
        model = model.to(device=device, dtype=dtype)
    else:
        model = model.to(device=device)
    model.train()

    if is_main:
        n_params = sum(p.numel() for p in model.parameters())
        print(
            f"[setup] model params: {n_params/1e6:.2f}M, dtype={dtype}",
            flush=True,
        )

    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=False)
    optim = torch.optim.AdamW(
        ddp_model.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )
    sched = torch.optim.lr_scheduler.LambdaLR(
        optim, lambda s: _lr_schedule(
            s, args.warmup_steps, args.num_steps, args.lr_min_ratio,
        ),
    )

    log_f = None
    if is_main:
        log_f = open(
            os.path.join(args.output_dir, "train_log.jsonl"), "w", encoding="utf-8",
        )

    step = 0
    t_start = time.time()
    losses_local = []   # rank-local per-step loss (for moving avg display)
    eval_history = []

    for sample in train_loader:
        _move_sample(sample, device)

        loss, metrics = ddp_model(sample, return_metrics=True)
        optim.zero_grad()
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            ddp_model.parameters(), args.grad_clip,
        ).item()
        optim.step()
        sched.step()
        step += 1

        # Cross-rank avg of this step's loss (for honest training curve).
        loss_t = torch.tensor(loss.item(), device=device, dtype=torch.float32)
        dist.all_reduce(loss_t, op=dist.ReduceOp.SUM)
        loss_avg = (loss_t / world_size).item()
        losses_local.append(loss_avg)

        if is_main and (step % args.log_every == 0 or step == 1):
            elapsed = time.time() - t_start
            line = {
                "event": "train",
                "step": step,
                "loss": loss_avg,
                "loss_ma50": sum(losses_local[-50:]) / min(len(losses_local), 50),
                "baseline_zero_mse_rank0": metrics["baseline_zero_mse"],
                "grad_norm": grad_norm,
                "lr": optim.param_groups[0]["lr"],
                "num_chunks_rank0": metrics["num_chunks"],
                "num_masked_chunks_rank0": metrics["num_masked_chunks"],
                "elapsed_s": round(elapsed, 1),
                "step_per_s": round(step / max(elapsed, 1e-6), 2),
            }
            print(json.dumps(line), flush=True)
            log_f.write(json.dumps(line) + "\n")
            log_f.flush()

        # Periodic eval — rank 0 runs val pass, others wait at barrier.
        if args.eval_every and step % args.eval_every == 0:
            if is_main:
                em = _run_eval(ddp_model.module, val_loader, device)
                if em:
                    em["step"] = step
                    eval_history.append(em)
                    print("[eval] " + json.dumps(em), flush=True)
                    log_f.write(json.dumps({"event": "eval", **em}) + "\n")
                    log_f.flush()
            dist.barrier()

        # Periodic checkpoint — rank 0, others barrier.
        if args.save_every and step % args.save_every == 0:
            if is_main:
                ckpt_path = os.path.join(args.output_dir, f"ckpt_step{step}.pt")
                torch.save(
                    {"step": step, "model": ddp_model.module.state_dict()},
                    ckpt_path,
                )
                print(f"[save] {ckpt_path}", flush=True)
            dist.barrier()

        if step >= args.num_steps:
            break

    # Final eval + save (rank 0 only).
    if is_main:
        if val_loader is not None:
            em = _run_eval(ddp_model.module, val_loader, device)
            if em:
                em["step"] = step
                eval_history.append(em)
                print("[final eval] " + json.dumps(em), flush=True)
                log_f.write(json.dumps({"event": "eval_final", **em}) + "\n")

        ckpt_path = os.path.join(args.output_dir, "ckpt_final.pt")
        torch.save(
            {
                "step": step,
                "model": ddp_model.module.state_dict(),
                "args": vars(args),
            },
            ckpt_path,
        )
        log_f.close()

        summary = {
            "world_size": world_size,
            "steps_run": step,
            "first_loss": losses_local[0] if losses_local else None,
            "last_loss": losses_local[-1] if losses_local else None,
            "min_loss": min(losses_local) if losses_local else None,
            "loss_ma50_final": (
                sum(losses_local[-50:]) / min(len(losses_local), 50)
                if losses_local else None
            ),
            "elapsed_s": round(time.time() - t_start, 1),
            "step_per_s": round(step / max(time.time() - t_start, 1e-6), 2),
            "eval_history": eval_history,
            "ckpt": ckpt_path,
        }
        with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)
        print("=== summary ===", flush=True)
        print(json.dumps(summary, indent=2), flush=True)

    # Two barriers (F17): make sure all ranks reach end before destroy.
    dist.barrier()
    dist.barrier()
    _cleanup_dist()


def build_argparser():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--shard_dir", required=True)
    p.add_argument("--mask_ratio", type=float, default=0.25)
    p.add_argument("--min_chunks", type=int, default=4)
    p.add_argument("--max_chunks", type=int, default=32)
    p.add_argument("--num_val_shards", type=int, default=8)
    p.add_argument("--val_max_samples", type=int, default=400)
    # Compressor architecture (defaults match V3-Both-Additive).
    p.add_argument("--num_layers", type=int, default=32)
    p.add_argument("--num_kv_heads", type=int, default=8)
    p.add_argument("--head_dim", type=int, default=128)
    p.add_argument("--latent_dim", type=int, default=512)
    p.add_argument("--num_fusion_layers", type=int, default=4)
    p.add_argument("--num_fusion_heads", type=int, default=8)
    p.add_argument("--mem_size", type=int, default=34)
    p.add_argument("--max_chunks_arch", type=int, default=64)
    p.add_argument("--attention_window", type=int, default=None)
    p.add_argument("--pos_encoding_mode", default="intra_chunk")
    p.add_argument("--summary_pool_mode", default="mean")
    p.add_argument("--num_summary_layers", type=int, default=2)
    p.add_argument("--gate_init", type=float, default=1.0)
    p.add_argument("--fusion_strategy", default="additive")
    p.add_argument("--local_stream_enabled", type=int, default=1)
    # Training
    p.add_argument("--num_steps", type=int, default=50000)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight_decay", type=float, default=0.01)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=1000)
    p.add_argument("--lr_min_ratio", type=float, default=0.1)
    p.add_argument("--log_every", type=int, default=100)
    p.add_argument("--eval_every", type=int, default=5000)
    p.add_argument("--save_every", type=int, default=10000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dtype", default="bfloat16",
                   choices=["float32", "bfloat16"])
    p.add_argument("--output_dir", required=True)
    # DDP
    p.add_argument("--port", type=int, default=29500)
    p.add_argument("--nccl_timeout_sec", type=int, default=1800)
    return p


def main():
    args = build_argparser().parse_args()
    world_size = torch.cuda.device_count()
    if world_size < 1:
        raise RuntimeError("No CUDA devices visible. Set CUDA_VISIBLE_DEVICES.")
    print(f"[main] launching DDP across {world_size} GPUs", flush=True)
    mp.spawn(train_worker, args=(args, world_size), nprocs=world_size, join=True)


if __name__ == "__main__":
    main()
