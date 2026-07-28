"""Fast pre-compute per-chunk KV caches with batched chunk processing.

Batches full-size chunks (510 tokens) across multiple samples into a single
forward pass for significant speedup over sequential chunk processing.

Usage (GPU server 4444):
    conda activate manta
    python scripts/precompute_kv_fast.py --dataset mrqa --split train --batch_chunks 32
"""

import argparse
import json
import math
import os
import sys
import time

import torch
import torch.multiprocessing as mp
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modeling import get_model, load_adapter

BASE = os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ADAPTER_PATH = os.path.join(BASE, "experiment/sac_8b_paper/output/adapter.pt")
MODEL_ID = os.environ.get("MANTA_MODEL_ID", "meta-llama/Llama-3.1-8B")
TASK_CONFIG = {
    "chunk_size": 510, "mem_size": 34, "compress_ratio": 15,
    "task_type": "Compress", "use_pe": True,
    "use_ae_loss": False, "use_lm_loss": True,
    "is_pretrain": True, "is_sft": False,
}

DATASET_FILES = {
    "extended_qa": "extended_qa_{split}_instruction_dataset.json",
    "mrqa": "mrqa_{split}_instruction_dataset.json",
    "r500x": "r500x_{split}_instruction_dataset.json",
    "unified": "unified_{split}_instruction_dataset.json",
    "long_4096": "long_4096_{split}_instruction_dataset.json",
    "long_8192": "long_8192_{split}_instruction_dataset.json",
    "long_16384": "long_16384_{split}_instruction_dataset.json",
    "long_32768": "long_32768_{split}_instruction_dataset.json",
}

# Per-dataset source directory relative to BASE. Defaults to sft/output/.
DATASET_DIRS = {
    "long_4096": os.path.join(BASE, "data", "long_context_eval"),
    "long_8192": os.path.join(BASE, "data", "long_context_eval"),
    "long_16384": os.path.join(BASE, "data", "long_context_eval"),
    "long_32768": os.path.join(BASE, "data", "long_context_eval"),
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=list(DATASET_FILES.keys()), required=True)
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--shard_size", type=int, default=500)
    p.add_argument("--batch_chunks", type=int, default=32,
                   help="Max full-size chunks per forward pass (tune for GPU mem)")
    p.add_argument("--num_workers", type=int, default=0, help="0=auto-detect GPUs")
    p.add_argument("--output_dir", default=os.path.join(BASE, "data", "precomputed_kv"))
    p.add_argument("--adapter_path", default=None,
                   help="Override default ADAPTER_PATH; for testing F11 with "
                        "instruction_adapter.pt")
    p.add_argument("--output_subdir", default=None,
                   help="Override the dataset subdir under output_dir; useful "
                        "for keeping multiple adapter variants side-by-side")
    return p.parse_args()


def load_data(dataset, split):
    filename = DATASET_FILES[dataset].format(split=split)
    src_dir = DATASET_DIRS.get(dataset, os.path.join(BASE, "sft", "output"))
    path = os.path.join(src_dir, filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_context(example, tokenizer):
    context = tokenizer(example["context"], add_special_tokens=False)["input_ids"]
    context_ids = ([tokenizer.bos_token_id]
                   + tokenizer("### Context:\n", add_special_tokens=False)["input_ids"]
                   + context)
    return torch.LongTensor(context_ids).unsqueeze(0)


def flush_full_batch(model, batch_embeds, batch_pos, batch_meta, batch_targets,
                     pending, chunk_size, device):
    """Run batched forward pass for full-size chunks and distribute results."""
    if not batch_embeds:
        return

    B = len(batch_embeds)
    batched_embeds = torch.cat(batch_embeds, dim=0)
    batched_pos = torch.cat(batch_pos, dim=0)

    attn_mask = model.build_attention_mask_full_bidirectional(chunk_size)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(1).to(device).to(torch.bfloat16)

    with torch.no_grad():
        outputs = model.model(
            position_ids=batched_pos,
            inputs_embeds=batched_embeds,
            output_hidden_states=False,
            attention_mask=attn_mask,
        )

    past_kv = outputs.past_key_values

    for i in range(B):
        start_idx, mem_pos = batch_meta[i]
        entry_key, chunk_idx = batch_targets[i]
        mem_real_idx = mem_pos.squeeze(0) - 1 - start_idx
        trimmed_kv = tuple(
            (k[i:i+1, :, mem_real_idx, :].cpu(),
             v[i:i+1, :, mem_real_idx, :].cpu())
            for k, v in past_kv
        )
        pending[entry_key]["chunk_results"][chunk_idx] = {
            "kv": trimmed_kv,
            "mem_position_ids": mem_pos.cpu(),
            "mem_size": mem_pos.size(1),
        }

    del outputs, past_kv, batched_embeds, batched_pos

    batch_embeds.clear()
    batch_pos.clear()
    batch_meta.clear()
    batch_targets.clear()


def process_short_chunk(model, embeds, pos_ids, mem_pos, start_idx, seq_len, device):
    """Process a single short (last) chunk."""
    attn_mask = model.build_attention_mask_full_bidirectional(seq_len)
    attn_mask = attn_mask.unsqueeze(0).unsqueeze(1).to(device).to(torch.bfloat16)

    with torch.no_grad():
        outputs = model.model(
            position_ids=pos_ids,
            inputs_embeds=embeds,
            output_hidden_states=False,
            attention_mask=attn_mask,
        )

    past_kv = outputs.past_key_values
    mem_real_idx = mem_pos.squeeze(0) - 1 - start_idx
    trimmed_kv = tuple(
        (k[:, :, mem_real_idx, :].cpu(), v[:, :, mem_real_idx, :].cpu())
        for k, v in past_kv
    )
    result = {
        "kv": trimmed_kv,
        "mem_position_ids": mem_pos.cpu(),
        "mem_size": mem_pos.size(1),
    }
    del outputs, past_kv
    return result


def finalize_sample(entry):
    """Convert pending entry to shard-ready format."""
    chunk_results = entry["chunk_results"]
    return {
        "sample_idx": entry["sample_idx"],
        "num_chunks": entry["num_chunks"],
        "total_mem_size": sum(r["mem_size"] for r in chunk_results),
        "end_idx": entry["end_idx"],
        "chunk_kvs": [r["kv"] for r in chunk_results],
        "chunk_mem_position_ids": [r["mem_position_ids"] for r in chunk_results],
    }


def try_collect_completed(pending, completed_up_to, shard_buffer):
    """Collect completed samples in order into shard_buffer."""
    while completed_up_to + 1 in pending:
        key = completed_up_to + 1
        entry = pending[key]
        if not all(r is not None for r in entry["chunk_results"]):
            break
        shard_buffer.append(finalize_sample(entry))
        del pending[key]
        completed_up_to = key
    return completed_up_to


def worker(rank, num_gpus, args, data, shard_output_dir):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = get_model(MODEL_ID, TASK_CONFIG, rank)
    adapter_path = args.adapter_path or ADAPTER_PATH
    if rank == 0:
        print(f"[precompute] adapter: {adapter_path}", flush=True)
    model = load_adapter(model, save_path_and_name=adapter_path, log=(rank == 0))
    model.eval()

    chunk_size = model.chunk_size
    compress_ratio = model.compress_ratio
    batch_limit = args.batch_chunks

    per_gpu = len(data) // num_gpus
    start_i = rank * per_gpu
    end_i = start_i + per_gpu if rank < num_gpus - 1 else len(data)
    my_indices = list(range(start_i, end_i))

    shard_buffer = []
    shard_count = 0
    t0 = time.time()

    # Batching buffers
    full_batch_embeds = []
    full_batch_pos = []
    full_batch_meta = []
    full_batch_targets = []

    pending = {}
    completed_up_to = -1

    for i, global_idx in enumerate(my_indices):
        example = data[global_idx]
        input_ids = tokenize_context(example, tokenizer).to(device)
        total_length = input_ids.size(1)
        num_chunks = math.ceil(total_length / chunk_size)

        pending[i] = {
            "sample_idx": global_idx,
            "num_chunks": num_chunks,
            "end_idx": min(num_chunks * chunk_size, total_length),
            "chunk_results": [None] * num_chunks,
        }

        for chunk_idx in range(num_chunks):
            si = chunk_idx * chunk_size
            ei = min((chunk_idx + 1) * chunk_size, total_length)
            seq_len = ei - si

            chunk_input = input_ids[:, si:ei]
            embeds = model.model.model.embed_tokens(chunk_input)
            pos_ids = torch.arange(si + 1, ei + 1, device=device).unsqueeze(0)
            mem_pos = model.get_uniform_position_ids(
                x_1=si + 1, x_n=ei + 1, ratio=compress_ratio,
            )
            mem_real_idx = mem_pos.squeeze(0) - 1 - si

            enc_embeds = embeds.clone()
            role_embeds = model.role_tokens[:mem_pos.size(1), :].unsqueeze(0)
            enc_embeds.index_add_(1, mem_real_idx, role_embeds)

            if seq_len == chunk_size:
                full_batch_embeds.append(enc_embeds)
                full_batch_pos.append(pos_ids)
                full_batch_meta.append((si, mem_pos))
                full_batch_targets.append((i, chunk_idx))

                if len(full_batch_embeds) >= batch_limit:
                    flush_full_batch(
                        model, full_batch_embeds, full_batch_pos,
                        full_batch_meta, full_batch_targets,
                        pending, chunk_size, device,
                    )
            else:
                result = process_short_chunk(
                    model, enc_embeds, pos_ids, mem_pos, si, seq_len, device,
                )
                pending[i]["chunk_results"][chunk_idx] = result

        del input_ids

        # Collect completed samples in order
        completed_up_to = try_collect_completed(
            pending, completed_up_to, shard_buffer,
        )

        # Write shard when buffer is full
        while len(shard_buffer) >= args.shard_size:
            shard_path = os.path.join(
                shard_output_dir,
                f"shard_{rank:02d}_{shard_count:05d}.pt",
            )
            torch.save(shard_buffer[:args.shard_size], shard_path)
            shard_buffer = shard_buffer[args.shard_size:]
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (len(my_indices) - i - 1) / speed if speed > 0 else 0
            if rank == 0:
                mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                print(
                    f"  [GPU{rank}] {i+1}/{len(my_indices)} "
                    f"| {speed:.1f} s/s | ETA {eta/60:.0f}min "
                    f"| peak {mem_gb:.1f}GB "
                    f"| saved shard_{rank:02d}_{shard_count:05d}.pt",
                    flush=True,
                )
            shard_count += 1

    # Final flush of remaining batched chunks
    flush_full_batch(
        model, full_batch_embeds, full_batch_pos,
        full_batch_meta, full_batch_targets,
        pending, chunk_size, device,
    )

    # Finalize all remaining pending samples
    for key in sorted(pending.keys()):
        entry = pending[key]
        if all(r is not None for r in entry["chunk_results"]):
            shard_buffer.append(finalize_sample(entry))

    # Write remaining shard
    if shard_buffer:
        shard_path = os.path.join(
            shard_output_dir,
            f"shard_{rank:02d}_{shard_count:05d}.pt",
        )
        torch.save(shard_buffer, shard_path)

    elapsed = time.time() - t0
    print(
        f"  [GPU{rank}] Done: {len(my_indices)} samples in {elapsed/60:.1f}min",
        flush=True,
    )


def main():
    args = parse_args()
    data = load_data(args.dataset, args.split)
    num_gpus = (
        torch.cuda.device_count() if args.num_workers == 0 else args.num_workers
    )
    print(
        f"Dataset: {args.dataset}, Split: {args.split}, "
        f"Samples: {len(data)}, GPUs: {num_gpus}, "
        f"batch_chunks: {args.batch_chunks}",
        flush=True,
    )

    subdir = args.output_subdir if args.output_subdir else args.dataset
    shard_output_dir = os.path.join(args.output_dir, subdir, args.split)
    os.makedirs(shard_output_dir, exist_ok=True)

    if num_gpus == 1:
        worker(0, 1, args, data, shard_output_dir)
    else:
        mp.spawn(
            worker, args=(num_gpus, args, data, shard_output_dir),
            nprocs=num_gpus, join=True,
        )


if __name__ == "__main__":
    main()
