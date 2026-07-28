"""
Prepare pretrain data for SAC from PG-19 text file (optimized).
Only reads as much text as needed for samples_num + 1000 examples.
"""
import os
import random
import argparse
import torch
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, default=os.environ.get("MANTA_MODEL_ID", "meta-llama/Llama-3.1-8B"))
    parser.add_argument("--text_file", type=str,
                        default=os.path.join(os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "data", "raw", "pretrain_train.txt"))
    parser.add_argument("--output_dir", type=str, default="../experiment/sac_8b/output")
    parser.add_argument("--samples_num", type=int, default=320000)
    parser.add_argument("--min_len", type=int, default=510)
    parser.add_argument("--max_len", type=int, default=2040)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model_id.split("/")[-1]
    train_path = os.path.join(args.output_dir, f"train_{model_name}_{args.samples_num}samples_{args.min_len}-{args.max_len}len.pt")
    eval_path = os.path.join(args.output_dir, f"eval_{model_name}_{args.samples_num}samples_{args.min_len}-{args.max_len}len.pt")

    if os.path.exists(train_path):
        print(f"[INFO] Data already exists: {train_path}")
        return

    random.seed(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)

    target_total = args.samples_num + 1000
    # Estimate: avg ~1275 tokens per sample, so need ~400M tokens
    # Avg ~4 chars/token -> read ~1.6GB of text should be enough
    # Read 2GB to be safe
    max_bytes = 2 * 1024 * 1024 * 1024
    chunk_bytes = 10 * 1024 * 1024  # 10MB chunks for tokenization

    print(f"[INFO] Tokenizing PG-19 (up to {max_bytes // (1024**3)}GB)...")
    all_ids = []
    bytes_read = 0

    with open(args.text_file, "r") as f:
        while bytes_read < max_bytes:
            text_chunk = f.read(chunk_bytes)
            if not text_chunk:
                break
            ids = tokenizer(text_chunk, add_special_tokens=False)["input_ids"]
            all_ids.extend(ids)
            bytes_read += len(text_chunk.encode("utf-8"))
            
            # Check if we have enough tokens
            needed_tokens = target_total * args.max_len  # worst case
            if len(all_ids) >= needed_tokens:
                print(f"  Got enough tokens ({len(all_ids):,}), stopping early")
                break
            
            if bytes_read % (100 * 1024 * 1024) < chunk_bytes:
                print(f"  Read {bytes_read // (1024*1024)}MB, {len(all_ids):,} tokens")

    print(f"[INFO] Total tokens: {len(all_ids):,}")

    # Create examples with random window sizes
    examples = []
    pos = 0
    print(f"[INFO] Creating {target_total} examples...")
    
    pbar = tqdm(total=target_total)
    while len(examples) < target_total and pos + args.min_len < len(all_ids):
        win_len = random.randint(args.min_len, min(args.max_len, len(all_ids) - pos))
        ids = all_ids[pos:pos + win_len]
        half = len(ids) // 2

        inputs = torch.LongTensor([tokenizer.bos_token_id] + ids[:half])
        ae_target = torch.LongTensor([tokenizer.bos_token_id] + ids[:half] + [tokenizer.eos_token_id])
        lm_target = torch.LongTensor(ids[half:] + [tokenizer.eos_token_id])
        examples.append({"inputs": inputs, "ae_target": ae_target, "lm_target": lm_target})

        pos += win_len
        pbar.update(1)
    pbar.close()

    print(f"[INFO] Created {len(examples)} examples")
    random.shuffle(examples)

    eval_examples = examples[:1000]
    train_examples = examples[1000:args.samples_num + 1000]

    print(f"[INFO] Saving {len(train_examples)} train, {len(eval_examples)} eval")
    torch.save(train_examples, train_path)
    torch.save(eval_examples, eval_path)
    print(f"[INFO] Done: {train_path}")


if __name__ == "__main__":
    main()
