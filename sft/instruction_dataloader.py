from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch

# Llama-3 pad id; both Llama-3.1-8B and Llama-3.2-1B configs report this.
LLAMA3_PAD_ID = 128004
LABEL_IGNORE = -100  # CrossEntropyLoss(ignore_index=-100) skips these positions

# Per-key pad value. The model uses `lm_targets[:, :-1]` as teacher-forcing
# embedding input (see model/modeling.py SFT path → embed_tokens), so it MUST
# be padded with a valid token id (LLAMA3_PAD_ID), not -100 (CUDA assert on
# negative index lookup). Only `instruction_target` (pure CE label) is padded
# with -100 so loss skips pad positions.
_PAD_BY_KEY = {
    "input_ids":  LLAMA3_PAD_ID,
    "lm_targets": LLAMA3_PAD_ID,
    "instruction_target": LABEL_IGNORE,
}


def _pad_to_length(t: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
    cur = t.size(0)
    if cur == length:
        return t
    if cur > length:
        return t[:length]
    return torch.cat([t, t.new_full((length - cur,), pad_value)], dim=0)


class CompressDataset(IterableDataset):
    """Yields padded batches for SAC/PACT SFT (instruction tuning).

    Patched 2026-06-01: removed the hard `assert batch_size == 1` so we can
    actually use the GPU memory we paid for. Each batch is padded per-key —
    input_ids with pad_token_id, every other key with -100.

    Note: per-example dicts may contain None for some keys (e.g. test split
    has no `instruction_target`); we preserve that contract by emitting None
    for any key whose first value in the batch is None.
    """

    def __init__(self, examples, batch_size):
        super(CompressDataset).__init__()
        # Length-sort by input_ids so within-batch padding is minimal and
        # peak memory hits early.
        # bs==1 (every real training run) preserves input order so the example
        # stream is identical to MAIN (allk_k34up). Only bs>1 sorts by length to
        # minimise within-batch padding. Sorting at bs==1 was the 2026-06-01
        # regression that pushed the WSD decay phase onto the shortest sequences.
        if batch_size > 1:
            print("[WARN] batch_size_per_device > 1: GLOBAL descending length-sort"
                  " of the training stream is active (known -12 F1 WSD-decay"
                  " curriculum regression, see F64/F32). Use bs=1 + grad-accum"
                  " for any run whose numbers will be compared.", flush=True)
        self.examples = examples if batch_size == 1 else sorted(
            examples, key=lambda e: e["input_ids"].size(0), reverse=True)
        self.batch_size = batch_size

    def __iter__(self):
        bs = self.batch_size
        keys = list(self.examples[0].keys())
        bucket = {k: [] for k in keys}
        count = 0
        for example in self.examples:
            for k in keys:
                bucket[k].append(example.get(k))
            count += 1
            if count == bs:
                yield self._make_batch(bucket, keys)
                bucket = {k: [] for k in keys}
                count = 0
        # drop trailing partial batch

    @staticmethod
    def _make_batch(bucket, keys):
        out = {}
        for k in keys:
            vals = bucket[k]
            if vals[0] is None:
                out[k] = None
                continue
            pad_val = _PAD_BY_KEY.get(k, LABEL_IGNORE)
            max_len = max(v.size(0) for v in vals)
            out[k] = torch.stack([_pad_to_length(v, max_len, pad_val) for v in vals])
        return out


def get_dataset(task_type, examples, batch_size):
    if task_type == "Compress":
        return CompressDataset(examples, batch_size)

    raise Exception("Don't exist [{task_type}] task.")
