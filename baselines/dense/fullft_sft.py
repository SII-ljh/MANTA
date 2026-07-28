#!/usr/bin/env python
"""Full-FT topline baseline for the PACT main table (Llama-3.2-1B).

Full-PARAMETER SFT of the backbone on MRQA 320K with the SAME prompt template
and SAME tokenized data PACT uses (no compression, no freeze, no LoRA). This is
the uncompressed (1x) ceiling, NOT a budget-matched baseline.

Training sequence per example (reusing the cached instruction .pt):
    tokens = input_ids (= [BOS]### Context:\n<ctx>)  +  lm_targets (= \n### Question:\n<q>\n### Answer:\n<ans>[EOS])
    labels = [-100]*len(input_ids)                   +  instruction_target (= [-100]*question + answer + EOS)
so loss is on the answer span only, exactly like the compression methods' decoder loss.

DDP via torchrun --nproc_per_node=8. bf16. save_safetensors=False (Llama-3.2-1B has
tie_word_embeddings=True -> safetensors refuses shared tensors; torch.save handles them).
"""
import argparse
import glob
import os

import torch
from torch.utils.data import Dataset
from transformers import (AutoModelForCausalLM, AutoTokenizer, Trainer,
                          TrainingArguments)

# accelerate's extract_model_from_parallel does `import deepspeed` whenever the
# package is installed. On these net-less GPU nodes deepspeed's import probes
# CUDA_HOME (no toolkit present) and raises MissingCUDAException, killing every
# rank at _wrap_model. We never use deepspeed -> force it unavailable.
import accelerate.utils.imports as _ai      # noqa: E402
import accelerate.utils.other as _ao        # noqa: E402
_ai.is_deepspeed_available = lambda: False
_ao.is_deepspeed_available = lambda: False


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--train_pt", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--per_device_batch", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--save_steps", type=int, default=2000)
    p.add_argument("--max_len", type=int, default=6200)
    p.add_argument("--optim", default="adamw_torch",
                   help="adamw_torch (default) or adafactor (low-mem, for shared nodes)")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def _to_list(x):
    return x.tolist() if torch.is_tensor(x) else list(x)


class SFTDataset(Dataset):
    def __init__(self, data, max_len):
        self.data = data
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        ex = self.data[i]
        ctx = _to_list(ex["input_ids"])          # [BOS]### Context:\n<ctx>
        dec = _to_list(ex["lm_targets"])          # \n### Question:\n<q>\n### Answer:\n<ans><eos>
        # input_ids and labels MUST be equal length (HF Trainer shifts internally).
        # Supervise only the answer span (= the trailing answer+eos tokens of dec);
        # mask context + question. instruction_target is the *shifted* next-token
        # label whose non-(-100) entries are exactly the answer+eos tokens, so
        # n_ans = #non-masked in instruction_target gives the answer length in dec.
        it = ex.get("instruction_target")
        if it is not None:
            it = _to_list(it)
            n_ans = sum(1 for t in it if t != -100)
        else:
            n_ans = len(dec)  # fallback: supervise full decoder span
        n_ans = max(1, min(n_ans, len(dec)))
        input_ids = ctx + dec
        labels = [-100] * (len(input_ids) - n_ans) + dec[len(dec) - n_ans:]
        # left-truncate the CONTEXT (preserve BOS + question + answer) if over budget
        if len(input_ids) > self.max_len:
            overflow = len(input_ids) - self.max_len
            input_ids = input_ids[:1] + input_ids[1 + overflow:]
            labels = labels[:1] + labels[1 + overflow:]
        return {"input_ids": input_ids, "labels": labels}


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        maxlen = max(len(b["input_ids"]) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids, lab = b["input_ids"], b["labels"]
            pad = maxlen - len(ids)
            input_ids.append(ids + [self.pad_id] * pad)
            labels.append(lab + [-100] * pad)
            attn.append([1] * len(ids) + [0] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def main():
    a = parse()
    tok = AutoTokenizer.from_pretrained(a.model)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id

    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[fullft] loading train data {a.train_pt}", flush=True)
    data = torch.load(a.train_pt, map_location="cpu")
    ds = SFTDataset(data, a.max_len)
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[fullft] {len(ds)} train examples; pad_id={pad_id}", flush=True)

    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16)
    model.config.use_cache = False

    targs = TrainingArguments(
        output_dir=a.out_dir,
        max_steps=a.steps,
        per_device_train_batch_size=a.per_device_batch,
        gradient_accumulation_steps=a.grad_accum,
        learning_rate=a.lr,
        warmup_steps=a.warmup,
        lr_scheduler_type="cosine",
        optim=a.optim,
        bf16=True,
        logging_steps=50,
        save_steps=a.save_steps,
        save_total_limit=2,
        save_safetensors=False,          # tied embeddings -> torch.save, not safetensors
        report_to=[],
        ddp_find_unused_parameters=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        seed=a.seed,
        dataloader_num_workers=2,
        remove_unused_columns=False,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=ds,
                      data_collator=PadCollator(pad_id))
    # Auto-resume from the latest HF checkpoint if one exists in out_dir. Makes
    # long runs on flaky/shared nodes crash-safe: a supervised wrapper can just
    # relaunch and training continues from the last save_steps checkpoint. With
    # no checkpoint present this is a no-op (identical to a fresh run).
    # newest-first; pick the newest COMPLETE checkpoint. Guards against a
    # checkpoint dir left half-written by a mid-save crash (which would
    # otherwise trap every relaunch on a corrupt "latest"): trainer_state.json
    # is written at the end of _save_checkpoint, so its presence marks a
    # complete save. Falls back to an older valid checkpoint, or a fresh run.
    ckpts = sorted(
        glob.glob(os.path.join(a.out_dir, "checkpoint-*")),
        key=lambda p: int(p.rsplit("-", 1)[-1]) if p.rsplit("-", 1)[-1].isdigit() else -1,
        reverse=True,
    )
    resume = None
    for c in ckpts:
        if os.path.isdir(c) and os.path.exists(os.path.join(c, "trainer_state.json")):
            resume = c
            break
    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[fullft] resume_from_checkpoint={resume}", flush=True)
    trainer.train(resume_from_checkpoint=resume)
    final = os.path.join(a.out_dir, "final")
    trainer.save_model(final)
    if trainer.is_world_process_zero():
        tok.save_pretrained(final)
        print(f"[fullft] saved final model to {final}", flush=True)


if __name__ == "__main__":
    main()
