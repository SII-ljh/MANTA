from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch

# Llama-3 pad id.
LLAMA3_PAD_ID = 128004
LABEL_IGNORE = -100  # CrossEntropyLoss(ignore_index=-100) skips these positions

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
    """Yield padded instruction-tuning batches."""

    def __init__(self, examples, batch_size):
        super(CompressDataset).__init__()
        # Preserve order for single-sample training; sort larger batches.
        if batch_size > 1:
            print("[WARN] batch_size_per_device > 1: sorting by length", flush=True)
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
        # Drop a trailing partial batch.

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
