from torch.utils.data import DataLoader, Dataset, IterableDataset
import torch

# Llama-3 / Llama-3.2 pad token id. Both Llama-3.1-8B and Llama-3.2-1B use 128004
# (verified in their config.json `pad_token_id`). Hardcoded here because the
# dataloader doesn't receive the tokenizer.
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
    """Yields padded batches for SAC/PACT pretrain.

    Patched 2026-06-01 to support batch_size > 1:
      * length-sorts examples by `inputs` size (descending) so each yielded
        batch has near-uniform length and padding overhead stays small
      * right-pads every tensor to its batch max length with pad_token_id
        (NOT -100). The model uses lm_targets/ae_targets as BOTH the decoder
        teacher-forcing input (`embed_tokens(lm_targets[:,:-1])`) AND the CE
        label. Padding with -100 crashes embedding lookup with a device-side
        assert (vocab index OOB). pad_token_id is a valid token id; the CE
        loss will count pad positions, but length-sorting keeps pad ratio
        small (typically <5% within a batch), so the noise is negligible.
    """

    def __init__(self, examples, batch_size):
        super(CompressDataset).__init__()
        # Sort once at init by input length so within-batch padding is minimal.
        # Descending so the first batch carries the longest sample (memory
        # peak hits early — fail-fast on OOM instead of crashing mid-epoch).
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
        # drop trailing partial batch (matches original behaviour)

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
