"""Train entry point for the per-chunk 500xCompressor PACT baseline.

Stage 1 (regeneration pretrain): pre-tokenized PG-19 HF dataset (column
  `input_ids` = [bos]+passage). Compress per 512-chunk, decode [BOS;text],
  CE over text+EOS.

Stage 2 (QA SFT): AB conversations json. The user turn is
  "### Context:\n{ctx}\n### Question:\n{q}\n### Answer:".  We SPLIT it at
  "\n### Question:" -> the CONTEXT part ("### Context:\n{ctx}") is compressed
  per-chunk (query-agnostic compression); the QUESTION block ("\n### Question:\n
  {q}\n### Answer:\n") is fed to the decoder UNCOMPRESSED (real tokens against the
  compressed-context KV), followed by the teacher-forced assistant answer.  CE is
  taken on the ANSWER tokens (+EOS) ONLY (context and question masked to -100).
  This matches the eval bridge regime exactly: the model SEES the question.

Plain HF Trainer + DDP (NO deepspeed). Saves only LoRA + memory_embeddings.
"""
import argparse
import json
import os

import torch
from datasets import load_from_disk
from torch.utils.data import Dataset
from transformers import AutoTokenizer, Trainer, TrainingArguments

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from c500x_perchunk import C500xPerChunk, default_lora_config

EOS = 128001
QUESTION_MARKER = "\n### Question:"


# ----------------------------- datasets ---------------------------------- #
class RegenDataset(Dataset):
    """Stage-1: pre-tokenized passages. Truncate to max_length; build labels."""
    def __init__(self, hf_dir, max_length):
        self.ds = load_from_disk(hf_dir)
        self.max_length = max_length

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, idx):
        ids = self.ds[idx]["input_ids"][: self.max_length]
        return {"input_ids": ids}


class QADataset(Dataset):
    """Stage-2: AB conversations json -> (context_ids, question_ids, answer_ids).

    The user turn "### Context:\n{ctx}\n### Question:\n{q}\n### Answer:" is split
    at "\n### Question:" so that:
      * context part  = "### Context:\n{ctx}"          -> compressed per-chunk
      * question block= "\n### Question:\n{q}\n### Answer:" (+trailing "\n" to
        match the eval lm_targets format) -> fed UNCOMPRESSED to the decoder
    The assistant content is the answer (teacher-forced, +EOS).
    """
    def __init__(self, json_path, tokenizer, max_ctx_len, max_q_len, max_ans_len):
        with open(json_path) as f:
            self.data = json.load(f)
        self.tok = tokenizer
        self.max_ctx_len = max_ctx_len
        self.max_q_len = max_q_len
        self.max_ans_len = max_ans_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        conv = self.data[idx]["conversations"]
        user = next(c["content"] for c in conv if c["role"] == "user")
        asst = next(c["content"] for c in conv if c["role"] == "assistant")

        ci = user.find(QUESTION_MARKER)
        if ci != -1:
            context = user[:ci]                 # "### Context:\n{ctx}"
            qblock = user[ci:]                  # "\n### Question:\n{q}\n### Answer:"
        else:
            context = user
            qblock = ""
        # Match the eval lm_targets, which ends with "### Answer:\n" (trailing nl).
        if not qblock.endswith("\n"):
            qblock = qblock + "\n"

        # Context compressed with a leading BOS so chunk-0 matches eval input_ids
        # (qa_eval.pt input_ids begin with <|begin_of_text|>).
        ctx_ids = [self.tok.bos_token_id] + self.tok.encode(
            context, add_special_tokens=False)
        ctx_ids = ctx_ids[: self.max_ctx_len]
        q_ids = self.tok.encode(qblock, add_special_tokens=False)[: self.max_q_len]
        ans_ids = self.tok.encode(asst, add_special_tokens=False)[: self.max_ans_len]
        return {"ctx_ids": ctx_ids, "q_ids": q_ids, "ans_ids": ans_ids}


# ----------------------------- collators --------------------------------- #
class RegenCollator:
    """Left content, right pad. labels = passage+EOS (pad -> -100).

    The pretokenized passages are [bos]+passage. The BOS stays on the COMPRESS
    side (input_ids, chunk-0 convention matches Stage2/eval contexts), but the
    decoder strips it and re-adds its own single BOS (see
    C500xPerChunk._forward_regen), so the decode targets are passage+EOS --
    NOT bos+passage+EOS (2026-06-13 review fix, F64). Decoder input length is
    then 1+(T-1) = T, so labels width = T.
    """
    def __init__(self, pad_id, bos_id=None):
        self.pad_id = pad_id
        self.bos_id = bos_id

    def __call__(self, feats):
        seqs = [f["input_ids"] for f in feats]
        B = len(seqs)
        T = max(len(s) for s in seqs)
        # mirror the model's global strip condition: ALL rows start with BOS
        strip = self.bos_id is not None and all(s and s[0] == self.bos_id
                                                for s in seqs)
        input_ids = torch.full((B, T), self.pad_id, dtype=torch.long)
        attn = torch.zeros((B, T), dtype=torch.long)
        lab_w = T if strip else T + 1
        labels = torch.full((B, lab_w), -100, dtype=torch.long)
        for i, s in enumerate(seqs):
            L = len(s)
            input_ids[i, :L] = torch.tensor(s, dtype=torch.long)
            attn[i, :L] = 1
            tgt = (s[1:] if strip else s) + [EOS]
            labels[i, : len(tgt)] = torch.tensor(tgt, dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attn,
                "labels": labels, "mode": "regen"}


class QACollator:
    """Build the QA batch consistent with the eval / model QA regime.

    Per sample: compress the CONTEXT (right-padded to batch max ctx len) into the
    KV cache; the decoder real tokens are a SINGLE right-padded sequence
    qa = [q_ids ; ans_ids ; EOS].  The question block is UNCOMPRESSED.

    Loss / shift convention (matches 500xC's no-internal-shift + ICAE):
      logit at decoder position p is compared to labels[p] directly.  We want the
      LAST question token (decoder index len(q)-1) to predict the FIRST answer
      token, so the answer (+EOS) labels are placed starting at offset len(q)-1:
      labels[len(q)-1 : len(q)-1+len(ans)+1] = ans + [EOS]; everything else (the
      question positions and trailing pad) is -100.
    """
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, feats):
        B = len(feats)
        Tc = max(len(f["ctx_ids"]) for f in feats)
        # qa real-token sequence length = question + answer + EOS
        S = max(len(f["q_ids"]) + len(f["ans_ids"]) + 1 for f in feats)

        ctx = torch.full((B, Tc), self.pad_id, dtype=torch.long)
        cattn = torch.zeros((B, Tc), dtype=torch.long)
        qa = torch.full((B, S), self.pad_id, dtype=torch.long)
        labels = torch.full((B, S), -100, dtype=torch.long)

        for i, f in enumerate(feats):
            c, q, a = f["ctx_ids"], f["q_ids"], f["ans_ids"]
            ctx[i, : len(c)] = torch.tensor(c, dtype=torch.long)
            cattn[i, : len(c)] = 1
            seq = q + a + [EOS]
            qa[i, : len(seq)] = torch.tensor(seq, dtype=torch.long)
            # answer (+EOS) labels start at the LAST question position (len(q)-1),
            # so that position's logit is supervised to emit the first answer token.
            start = len(q) - 1
            tgt = a + [EOS]
            labels[i, start : start + len(tgt)] = torch.tensor(tgt, dtype=torch.long)

        return {"input_ids": ctx, "attention_mask": cattn,
                "qa_input_ids": qa, "labels": labels, "mode": "qa"}


# --------------------------- Trainer glue -------------------------------- #
class C500xTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kw):
        out = model(**inputs)
        loss = out["loss"]
        return (loss, out) if return_outputs else loss

    def _save(self, output_dir=None, state_dict=None):
        """Save ONLY LoRA + memory_embeddings (~105MB), NOT the 32GB frozen base.
        The base is the unmodified Llama-3.1-8B at --llama_path; eval reloads it
        and applies trainable.pt. Avoids 32GB/checkpoint disk blowup."""
        output_dir = output_dir or self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        _unwrap(self.model).save_trainable(os.path.join(output_dir, "trainable.pt"))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))


def _unwrap(m):
    return m.module if hasattr(m, "module") else m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["regen", "qa"], required=True)
    ap.add_argument("--llama_path", required=True)
    ap.add_argument("--data_path", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--num_mem", type=int, required=True)
    ap.add_argument("--chunk_size", type=int, default=512)
    ap.add_argument("--max_length", type=int, default=2048)        # regen passage cap
    ap.add_argument("--max_prompt_len", type=int, default=2048)    # qa context cap
    ap.add_argument("--max_q_len", type=int, default=128)          # qa question-block cap
    ap.add_argument("--max_ans_len", type=int, default=64)
    ap.add_argument("--init_trainable", default=None,
                    help="path to trainable.pt to init LoRA+mem (Stage2 from Stage1)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--per_device_train_batch_size", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=1)
    ap.add_argument("--warmup_steps", type=int, default=300)
    ap.add_argument("--logging_steps", type=int, default=50)
    ap.add_argument("--save_steps", type=int, default=1000)
    ap.add_argument("--save_strategy", default="steps")
    ap.add_argument("--max_steps", type=int, default=-1)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lora_r", type=int, default=64,
                    help="LoRA rank (vendor default 64; SAC-paper baseline 128)")
    ap.add_argument("--lora_alpha", type=int, default=32,
                    help="LoRA alpha (vendor default 32; SAC-paper baseline 256)")
    args = ap.parse_args()

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    model = C500xPerChunk(args.llama_path, num_mem=args.num_mem,
                          chunk_size=args.chunk_size,
                          lora_config=default_lora_config(args.lora_r,
                                                          args.lora_alpha),
                          device=device, dtype=dtype)
    if args.init_trainable and os.path.isfile(args.init_trainable):
        model.load_trainable(args.init_trainable)
        if local_rank == 0:
            print(f"[init] loaded trainable from {args.init_trainable}", flush=True)
    model.to(device)

    tok = model.tokenizer
    if args.stage == "regen":
        ds = RegenDataset(args.data_path, args.max_length)
        coll = RegenCollator(pad_id=tok.pad_token_id or EOS,
                             bos_id=tok.bos_token_id)
    else:
        ds = QADataset(args.data_path, tok, args.max_prompt_len,
                       args.max_q_len, args.max_ans_len)
        coll = QACollator(pad_id=tok.pad_token_id or EOS)

    targs = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=False,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="constant_with_warmup",
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_strategy=args.save_strategy,
        save_steps=args.save_steps,
        save_total_limit=2,
        bf16=args.bf16,
        save_only_model=True,
        # True required under multi-GPU DDP: last-layer LoRA params get no
        # grad when only K/V caches are harvested (deterministic, grads are
        # truly zero). False crashed all >1-GPU launches (2026-06-13).
        ddp_find_unused_parameters=True,
        remove_unused_columns=False,
        report_to=[],
        dataloader_num_workers=2,
    )

    trainer = C500xTrainer(model=model, args=targs, train_dataset=ds,
                           data_collator=coll)
    trainer.train()
    if trainer.is_world_process_zero:
        _unwrap(model).save_trainable(os.path.join(args.output_dir, "trainable_final.pt"))
        print("[done] saved trainable_final.pt", flush=True)


if __name__ == "__main__":
    main()
