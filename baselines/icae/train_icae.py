"""Per-chunk ICAE trainer (HF Trainer + plain DDP, NO deepspeed).

Stage 1 (mode=ae) : autoencoding pretrain on PACT's 320K PG-19 passages
                    ($WORK/.../pg19_pact_pretrain_320k_hf, column `input_ids`).
                    Compress each passage per-chunk -> regenerate it.
Stage 2 (mode=qa) : QA SFT on mrqa_train_ab_320k.json (AB "conversations" format).
                    Compress the CONTEXT (### Context..) per-chunk, append the
                    question block (### Question..### Answer:) UNCOMPRESSED, CE on
                    the answer tokens only.

Labels are built so they align with the decoder input [compressed(M) ; ae(1) ; prompt].
Next-token convention: the token at decoder position p predicts label[p+1]; with the
ae token sitting just before the first prompt token, label index (M + prompt_offset - 1)
holds the first supervised target (matches native train_ICAE.py offset of M+prompt_len-1).
"""
import argparse
import json
import os

import torch
from datasets import load_from_disk
from peft import LoraConfig
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from icae_perchunk import ICAEPerChunk

EOS = 128001  # Llama-3.1 <|end_of_text|>


# ============================ datasets ============================
class AEDataset(Dataset):
    """Autoencoding: compress a passage, regenerate it (+eos)."""

    def __init__(self, hf_dir, num_mem, chunk_size, max_len):
        self.ds = load_from_disk(hf_dir)
        self.num_mem = num_mem
        self.chunk_size = chunk_size
        self.max_len = max_len

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ids = self.ds[idx]["input_ids"]
        if self.max_len is not None and len(ids) > self.max_len:
            ids = ids[: self.max_len]
        return {"input_ids": ids}


class QADataset(Dataset):
    """AB conversations -> (context_ids, question_block_ids, answer_ids)."""

    def __init__(self, json_path, tokenizer, max_ctx_len, max_qa_len):
        with open(json_path) as f:
            self.data = json.load(f)
        self.tok = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_qa_len = max_qa_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conv = self.data[idx]["conversations"]
        user = next(t["content"] for t in conv if t["role"] == "user")
        ans = next(t["content"] for t in conv if t["role"] == "assistant")
        # user content = "### Context:\n{ctx}\n### Question:\n{q}\n### Answer:"
        marker = "\n### Question:"
        ci = user.find(marker)
        context = user[len("### Context:\n"):ci] if ci != -1 else user
        qblock = user[ci:] if ci != -1 else ""  # "\n### Question:\n{q}\n### Answer:"

        ctx_ids = self.tok(context, add_special_tokens=False).input_ids
        if len(ctx_ids) > self.max_ctx_len:
            ctx_ids = ctx_ids[: self.max_ctx_len]
        q_ids = self.tok(qblock, add_special_tokens=False).input_ids
        a_ids = self.tok(ans, add_special_tokens=False).input_ids
        return {"ctx_ids": ctx_ids, "q_ids": q_ids, "a_ids": a_ids}


# ============================ collators ============================
class AECollator:
    """Pad passages to batch max; labels regenerate the passage (+eos)."""

    def __init__(self, num_mem, chunk_size, pad_id=EOS):
        self.num_mem = num_mem
        self.chunk_size = chunk_size
        self.pad_id = pad_id

    def _n_chunks(self, T):
        return (T + self.chunk_size - 1) // self.chunk_size

    def __call__(self, batch):
        maxT = max(len(b["input_ids"]) for b in batch)
        B = len(batch)
        input_ids = torch.full((B, maxT), self.pad_id, dtype=torch.long)
        # M is fixed per batch by maxT (all rows padded to same length -> same n_chunks).
        M = self._n_chunks(maxT) * self.num_mem
        # decoder seq = [M compressed ; 1 ae ; maxT text].  labels length == decoder seq.
        labels = torch.full((B, M + 1 + maxT), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            ids = b["input_ids"]
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long)
            # supervise the L passage tokens + 1 eos, starting at offset M (ae token
            # at position M predicts the first text token -> label index M).
            tgt = ids + [EOS]
            labels[i, M : M + len(tgt)] = torch.tensor(tgt, dtype=torch.long)
        return {"input_ids": input_ids, "labels": labels, "mode": "ae"}


class QACollator:
    """Pad context to batch max (-> fixed M); pad (q||a) to batch max qa len.

    Decoder seq = [M ; 1 ae ; ctx? NO — q||a] ... actually decoder prompt = q||a.
    The model compresses ctx (len = padded ctx), appends ae + (q||a) embeddings.
    labels length = M + 1 + qa_len; supervise only the answer (+eos) span.
    """

    def __init__(self, num_mem, chunk_size, pad_id=EOS):
        self.num_mem = num_mem
        self.chunk_size = chunk_size
        self.pad_id = pad_id

    def _n_chunks(self, T):
        return (T + self.chunk_size - 1) // self.chunk_size

    def __call__(self, batch):
        ctxT = max(len(b["ctx_ids"]) for b in batch)
        qaT = max(len(b["q_ids"]) + len(b["a_ids"]) + 1 for b in batch)  # +1 eos
        B = len(batch)
        ctx = torch.full((B, ctxT), self.pad_id, dtype=torch.long)
        qa = torch.full((B, qaT), self.pad_id, dtype=torch.long)
        M = self._n_chunks(ctxT) * self.num_mem
        labels = torch.full((B, M + 1 + qaT), -100, dtype=torch.long)
        for i, b in enumerate(batch):
            cids, qids, aids = b["ctx_ids"], b["q_ids"], b["a_ids"]
            ctx[i, : len(cids)] = torch.tensor(cids, dtype=torch.long)
            seq = qids + aids + [EOS]
            qa[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            # decoder positions: [M compressed][1 ae][qa...]. The ae token (idx M)
            # predicts qa[0]; qa token j (decoder idx M+1+j) predicts qa[j+1].
            # We want to supervise ONLY the answer span (a_ids + eos), i.e. tokens
            # qa[len(qids) : len(qids)+len(aids)+1].  Those occupy label indices
            # M + len(qids) .. M + len(qids)+len(aids)  (shifted by -1 from input pos).
            qlen, alen = len(qids), len(aids)
            start = M + qlen           # label index predicting first answer token
            tgt = aids + [EOS]
            labels[i, start : start + len(tgt)] = torch.tensor(tgt, dtype=torch.long)
        # input_ids for the model = ctx || qa (model splits at ctx_len).
        input_ids = torch.cat([ctx, qa], dim=1)
        return {"input_ids": input_ids, "labels": labels, "mode": "qa", "ctx_len": ctxT}


# ============================ trainer wrapper ============================
class ICAETrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        mode = inputs.pop("mode")
        out = model(**inputs, mode=mode)
        return (out["loss"], out) if return_outputs else out["loss"]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--stage", choices=["ae", "qa"], required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--model_path", required=True)
    p.add_argument("--train_data", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--init_trainable", default=None,
                   help="path to a Stage1 trainable .pt to warm-start (Stage2).")
    p.add_argument("--ratio", type=int, default=16)
    p.add_argument("--chunk_size", type=int, default=512)
    p.add_argument("--lora_rank", type=int, default=128)
    p.add_argument("--lora_alpha", type=int, default=32)
    p.add_argument("--lora_dropout", type=float, default=0.05)
    p.add_argument("--max_ae_len", type=int, default=2048)
    p.add_argument("--max_ctx_len", type=int, default=512)
    p.add_argument("--max_qa_len", type=int, default=128)
    p.add_argument("--per_device_train_batch_size", type=int, default=2)
    p.add_argument("--gradient_accumulation_steps", type=int, default=1)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--num_train_epochs", type=float, default=1.0)
    p.add_argument("--warmup_steps", type=int, default=300)
    p.add_argument("--logging_steps", type=int, default=50)
    p.add_argument("--save_strategy", default="epoch")
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--max_steps", type=int, default=-1)
    p.add_argument("--bf16", action="store_true")
    p.add_argument("--attn_impl", default="sdpa")
    p.add_argument("--cpu_smoke", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    num_mem = args.chunk_size // args.ratio
    assert num_mem >= 1, f"ratio {args.ratio} too large for chunk {args.chunk_size}"

    lora_config = LoraConfig(
        r=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],  # ICAE default: q,v
    )

    dtype = torch.float32 if args.cpu_smoke else (torch.bfloat16 if args.bf16 else torch.float32)
    model = ICAEPerChunk(
        llama_path=args.model_path,
        lora_config=lora_config,
        num_mem=num_mem,
        chunk_size=args.chunk_size,
        torch_dtype=dtype,
        attn_impl=("eager" if args.cpu_smoke else args.attn_impl),
    )
    if args.init_trainable and os.path.exists(args.init_trainable):
        print(f"[init] warm-starting trainable params from {args.init_trainable}")
        model.load_trainable(args.init_trainable)

    rank = int(os.environ.get("RANK", 0))
    if rank == 0:
        print(f"[INFO] stage={args.stage} ratio={args.ratio} num_mem={num_mem} "
              f"chunk={args.chunk_size}")
        print(f"[INFO] trainable params = {model.num_trainable():,}")

    tok = model.tokenizer
    if args.stage == "ae":
        ds = AEDataset(args.train_data, num_mem, args.chunk_size, args.max_ae_len)
        collator = AECollator(num_mem, args.chunk_size)
    else:
        ds = QADataset(args.train_data, tok, args.max_ctx_len, args.max_qa_len)
        collator = QACollator(num_mem, args.chunk_size)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        seed=args.seed,
        overwrite_output_dir=False,
        num_train_epochs=args.num_train_epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16 and not args.cpu_smoke,
        report_to=[],
        remove_unused_columns=False,
        ddp_find_unused_parameters=False,
        dataloader_num_workers=2,
        use_cpu=args.cpu_smoke,
    )

    trainer = ICAETrainer(
        model=model, args=targs, train_dataset=ds, data_collator=collator
    )
    trainer.train()

    # Save trainable params only (LoRA + memory + ae) — small, portable.
    if rank == 0:
        out = os.path.join(args.output_dir, "icae_trainable.pt")
        model.save_trainable(out)
        print(f"[SAVE] trainable params -> {out}")


if __name__ == "__main__":
    main()
