from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch

# Llama-3 pad token id; the dataloader does not receive the tokenizer.
LLAMA3_PAD_ID = 128004
LABEL_IGNORE = -100  # CrossEntropyLoss(ignore_index=-100) skips these positions


def _pad_to_length(t: torch.Tensor, length: int, pad_value: int) -> torch.Tensor:
    """Right-pad a 1-D LongTensor to `length` with `pad_value`. No-op if already long enough."""
    cur = t.size(0)
    if cur == length:
        return t
    if cur > length:
        return t[:length]
    return torch.cat([t, t.new_full((length - cur,), pad_value)], dim=0)


class CompressDataset(IterableDataset):
    """Yield padded pretraining batches."""

    def __init__(self, examples, batch_size):
        super(CompressDataset).__init__()
        # Preserve order for single-sample training; sort larger batches.
        if batch_size > 1:
            print("[WARN] batch_size_per_device > 1: sorting by length", flush=True)
        self.examples = examples if batch_size == 1 else sorted(
            examples, key=lambda e: e["inputs"].size(0), reverse=True)
        self.batch_size = batch_size

    def __iter__(self):
        bs = self.batch_size
        bucket_in, bucket_ae, bucket_lm = [], [], []
        for example in self.examples:
            bucket_in.append(example["inputs"])
            bucket_ae.append(example["ae_target"])
            bucket_lm.append(example["lm_target"])
            if len(bucket_in) == bs:
                yield self._make_batch(bucket_in, bucket_ae, bucket_lm)
                bucket_in, bucket_ae, bucket_lm = [], [], []
        # Drop a trailing partial batch.

    @staticmethod
    def _make_batch(bucket_in, bucket_ae, bucket_lm):
        max_in = max(t.size(0) for t in bucket_in)
        max_ae = max(t.size(0) for t in bucket_ae)
        max_lm = max(t.size(0) for t in bucket_lm)
        return {
            "input_ids":  torch.stack([_pad_to_length(t, max_in, LLAMA3_PAD_ID) for t in bucket_in]),
            "ae_targets": torch.stack([_pad_to_length(t, max_ae, LLAMA3_PAD_ID) for t in bucket_ae]),
            "lm_targets": torch.stack([_pad_to_length(t, max_lm, LLAMA3_PAD_ID) for t in bucket_lm]),
        }


def get_dataset(task_type, examples, batch_size):
    if task_type == "Compress":
        return CompressDataset(examples, batch_size)

    raise Exception("Don't exist [{task_type}] task.")
