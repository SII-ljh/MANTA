"""Pre-compute per-chunk KV caches for all QA samples using pretrained SAC encoder.

Replicates SAC compress() logic exactly but stores per-chunk KV (not concatenated)
for second-stage cross-chunk fusion experiments.

Usage (GPU server 4444):
    conda activate manta
    # 2-GPU, extended_qa (default)
    python scripts/precompute_kv.py --split train --num_workers 2
    # 2-GPU, mrqa
    python scripts/precompute_kv.py --dataset mrqa --split train --num_workers 2
    # 2-GPU, r500x
    python scripts/precompute_kv.py --dataset r500x --split train --num_workers 2
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
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", choices=['extended_qa', 'mrqa', 'r500x', 'unified'], default="extended_qa")
    p.add_argument("--split", choices=["train", "test"], required=True)
    p.add_argument("--shard_size", type=int, default=500)
    p.add_argument("--num_workers", type=int, default=0, help="0=auto-detect GPUs")
    p.add_argument("--output_dir", default=os.path.join(BASE, "data", "precomputed_kv"))
    return p.parse_args()


def load_data(dataset, split):
    filename = DATASET_FILES[dataset].format(split=split)
    path = os.path.join(BASE, "sft", "output", filename)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def tokenize_context(example, tokenizer):
    context = tokenizer(example["context"], add_special_tokens=False)["input_ids"]
    context_ids = ([tokenizer.bos_token_id]
                   + tokenizer("### Context:\n", add_special_tokens=False)["input_ids"]
                   + context)
    return torch.LongTensor(context_ids).unsqueeze(0)


def compress_per_chunk(model, input_ids, device):
    input_ids = input_ids.to(device)
    bsz, total_length = input_ids.size()
    chunk_size = model.chunk_size
    compress_ratio = model.compress_ratio
    num_chunks = math.ceil(total_length / chunk_size)
    per_chunk_results = []

    for chunk_idx in range(num_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, total_length)
        chunk_input_ids = input_ids[:, start_idx:end_idx]
        inputs_embeds = model.model.model.embed_tokens(chunk_input_ids)
        bsz_c, seq_len, emb_size = inputs_embeds.size()
        position_ids = torch.arange(start_idx + 1, end_idx + 1, device=device).unsqueeze(0)
        mem_position_ids = model.get_uniform_position_ids(
            x_1=start_idx + 1, x_n=end_idx + 1, ratio=compress_ratio
        )
        current_mem_size = mem_position_ids.size(1)
        encode_inputs_embeds = inputs_embeds.clone()
        role_embeds = model.role_tokens[:current_mem_size, :].unsqueeze(0).expand(bsz_c, current_mem_size, emb_size)
        mem_real_idx = mem_position_ids.squeeze(0) - 1 - start_idx
        encode_inputs_embeds.index_add_(1, mem_real_idx, role_embeds)
        attention_mask = model.build_attention_mask_full_bidirectional(seq_len)
        attention_mask = attention_mask.unsqueeze(0).unsqueeze(1).to(device).to(torch.bfloat16)

        with torch.no_grad():
            outputs = model.model(
                position_ids=position_ids,
                inputs_embeds=encode_inputs_embeds,
                output_hidden_states=False,
                attention_mask=attention_mask,
            )

        past_key_values = outputs.past_key_values
        trimmed_kv = tuple(
            (k[:, :, mem_real_idx, :].cpu(), v[:, :, mem_real_idx, :].cpu())
            for k, v in past_key_values
        )
        per_chunk_results.append({
            "kv": trimmed_kv,
            "mem_position_ids": mem_position_ids.cpu(),
            "mem_size": current_mem_size,
        })

    return per_chunk_results, num_chunks, per_chunk_results[-1] and (min((num_chunks) * chunk_size, total_length))


def worker(rank, num_gpus, args, data, shard_output_dir):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = get_model(MODEL_ID, TASK_CONFIG, rank)
    model = load_adapter(model, save_path_and_name=ADAPTER_PATH, log=(rank == 0))
    model.eval()

    per_gpu = len(data) // num_gpus
    start = rank * per_gpu
    end = start + per_gpu if rank < num_gpus - 1 else len(data)
    my_indices = list(range(start, end))

    shard_buffer = []
    shard_count = 0
    t0 = time.time()

    for i, global_idx in enumerate(my_indices):
        example = data[global_idx]
        input_ids = tokenize_context(example, tokenizer)
        per_chunk_results, num_chunks, end_idx = compress_per_chunk(model, input_ids, device)

        shard_buffer.append({
            "sample_idx": global_idx,
            "num_chunks": num_chunks,
            "total_mem_size": sum(r["mem_size"] for r in per_chunk_results),
            "end_idx": end_idx,
            "chunk_kvs": [r["kv"] for r in per_chunk_results],
            "chunk_mem_position_ids": [r["mem_position_ids"] for r in per_chunk_results],
        })

        if len(shard_buffer) >= args.shard_size:
            shard_path = os.path.join(shard_output_dir, f"shard_{rank:02d}_{shard_count:05d}.pt")
            torch.save(shard_buffer, shard_path)
            elapsed = time.time() - t0
            speed = (i + 1) / elapsed
            eta = (len(my_indices) - i - 1) / speed if speed > 0 else 0
            if rank == 0:
                print(f"  [GPU{rank}] {i+1}/{len(my_indices)} | {speed:.1f} s/s | ETA {eta/60:.0f}min | saved {shard_path}")
            shard_buffer = []
            shard_count += 1

    if shard_buffer:
        shard_path = os.path.join(shard_output_dir, f"shard_{rank:02d}_{shard_count:05d}.pt")
        torch.save(shard_buffer, shard_path)

    elapsed = time.time() - t0
    print(f"  [GPU{rank}] Done: {len(my_indices)} samples in {elapsed/60:.1f}min")


def main():
    args = parse_args()
    data = load_data(args.dataset, args.split)
    num_gpus = torch.cuda.device_count() if args.num_workers == 0 else args.num_workers
    print(f"Dataset: {args.dataset}, Split: {args.split}, Samples: {len(data)}, GPUs: {num_gpus}")

    shard_output_dir = os.path.join(args.output_dir, args.dataset, args.split)
    os.makedirs(shard_output_dir, exist_ok=True)

    if num_gpus == 1:
        worker(0, 1, args, data, shard_output_dir)
    else:
        mp.spawn(worker, args=(num_gpus, args, data, shard_output_dir), nprocs=num_gpus, join=True)


if __name__ == "__main__":
    main()
