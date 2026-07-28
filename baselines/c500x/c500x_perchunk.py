"""Per-chunk 500xCompressor (Li et al., ACL 2025) for the PACT project.

A controlled reproduction of 500xC adapted to PACT's per-chunk regime so it is
directly comparable to the SAC / Activation-Beacon / PACT baselines (all chunk
context into fixed 512-token windows).

Native 500xC compresses a whole input (<= max_length) into `num_mem` learnable
memory tokens, taking the K/V cache of those memory positions across ALL layers
as the compressed representation; a frozen copy of the same LLM (LoRA disabled)
then regenerates / answers from that KV cache.  We keep that exactly, but apply
it PER 512-token chunk:

    input token sequence
        -> split into consecutive chunks of `chunk_size` (=512) tokens
        -> for each chunk: encoder (Llama + LoRA) over [chunk_embeds ; memory_embeds]
                           keep the K/V of the last `num_mem` positions, all layers
        -> concatenate the per-chunk trimmed K/V along the SEQUENCE dim, per layer
        -> decoder (same Llama, LoRA disabled) consumes that concatenated KV as
           past_key_values and generates text / answer.

    ratio = chunk_size / num_mem   (16x default -> num_mem = 512/16 = 32)

Only the LoRA params (q_proj, v_proj) and `memory_embeddings` are trainable; the
base model is frozen and the decoder runs with `self.llama.disable_adapter()`.

PACT-comparability note (important): PACT's eval feeds `input_ids`=context (to be
compressed) and `lm_targets`=the uncompressed question block ("\\n### Question:..
\\n### Answer:\\n"), then generates the answer.  We mirror that EXACTLY (and the
sibling ICAE baseline): the CONTEXT is compressed per-chunk into the KV cache,
then the question block is fed to the decoder UNCOMPRESSED (as real tokens
processed against `past_key_values` = the compressed-context KV), and the answer
is greedy-generated.  This keeps compression query-agnostic (context compressed
without seeing the question) yet the model DOES see the question at decode time.
The decoder sequence is: past_key_values = compressed-context KV (length
n_chunks*num_mem); then real tokens = [question_block (+answer during training)].
transformers (4.43.1) auto-derives cache_position = arange(past_len, past_len+
new_len) when given inputs_embeds + a prefilled Cache, so the question tokens get
positions continuing after the compressed KV and attend to the context.

Three modes:
  * forward(mode="regen")  : Stage-1 regeneration pretrain (predict the text).
  * forward(mode="qa")     : Stage-2 QA SFT.  Compress CONTEXT per-chunk, feed
                             the (uncompressed) question block + teacher-forced
                             answer to the decoder; CE on the ANSWER tokens only.
  * lm_inference(...)      : compress context, append the uncompressed question
                             block, greedy-decode the answer from the compressed KV.

Dims are read from `model.config` (hidden_size, num_hidden_layers,
num_key_value_heads, head_dim) so GQA (Llama-3.1-8B: 8 KV heads) is handled
correctly -- the trimmed_past_key_values slices the SEQUENCE dim only, the KV
head dim is untouched (the reference's `layer_key[:, :, -num_mem:, :]`).
"""

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

EOS_FALLBACK = 128001  # Llama-3 <|end_of_text|>; overridden by tokenizer.eos_token_id when available


def default_lora_config(r=64, alpha=32):
    """500xC LoRA config. Vendor reference default = r=64, alpha=32 (q/v proj).
    The SAC-paper baseline config (which produced the reference 500xC numbers)
    is r=128, alpha=256 -- pass those for paper-parity runs (launcher default)."""
    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "v_proj"],
    )


class C500xPerChunk(nn.Module):
    def __init__(
        self,
        llama_path,
        num_mem,
        chunk_size=512,
        lora_config=None,
        device=None,
        dtype=torch.bfloat16,
        _tiny_config=None,
    ):
        """
        Args:
            llama_path (str): path / HF id for the base LLM (local Llama-3.1-8B).
            num_mem (int): compressed tokens per chunk (= chunk_size / ratio).
            chunk_size (int): window size to chunk the input (512, PACT-matched).
            lora_config (LoraConfig): defaults to the 500xC reference config.
            device, dtype: placement.
            _tiny_config: (debug only) a LlamaConfig to build a random tiny model
                instead of loading the 8B checkpoint, for fast CPU shape tests.
        """
        super().__init__()
        if lora_config is None:
            lora_config = default_lora_config()

        if _tiny_config is not None:
            llama = AutoModelForCausalLM.from_config(_tiny_config, torch_dtype=dtype)
        else:
            llama = AutoModelForCausalLM.from_pretrained(
                llama_path, torch_dtype=dtype
            )

        self.llama = get_peft_model(llama, lora_config)
        # only LoRA params trainable; base frozen.
        for name, param in self.llama.named_parameters():
            param.requires_grad = "lora" in name

        self.tokenizer = AutoTokenizer.from_pretrained(llama_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        cfg = self.llama.config
        self.hidden_size = cfg.hidden_size
        self.num_layers = cfg.num_hidden_layers
        self.num_kv_heads = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)

        self.num_mem = num_mem
        self.chunk_size = chunk_size
        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # learnable memory tokens (read hidden_size from config, not hardcoded 4096)
        self.memory_embeddings = nn.Parameter(
            torch.randn(1, num_mem, self.hidden_size, dtype=dtype)
        )

        self.device = device if device is not None else torch.device("cpu")
        self.dtype = dtype
        self.eos_token_id = self.tokenizer.eos_token_id or EOS_FALLBACK
        self.bos_token_id = self.tokenizer.bos_token_id

    # --------------------------------------------------------------------- #
    #  Per-chunk compression
    # --------------------------------------------------------------------- #
    def _embed(self, ids):
        return self.llama.get_input_embeddings()(ids)

    def compress(self, text_ids, attention_mask=None):
        """Compress `text_ids` per `chunk_size`-token window into concatenated KV.

        Args:
            text_ids (LongTensor): [B, T] tokens to compress.
            attention_mask (LongTensor): [B, T] optional; chunks are sliced from it.

        Returns:
            past_key_values: legacy tuple of (key, value) per layer, each
                [B, num_kv_heads, num_chunks * num_mem, head_dim].
        """
        B, T = text_ids.shape
        dev = text_ids.device
        mem = self.memory_embeddings.repeat(B, 1, 1).to(dev)

        # per-layer accumulators of compressed K and V
        keys = [[] for _ in range(self.num_layers)]
        vals = [[] for _ in range(self.num_layers)]

        for start in range(0, T, self.chunk_size):
            end = min(start + self.chunk_size, T)
            chunk_ids = text_ids[:, start:end]
            chunk_emb = self._embed(chunk_ids)
            enc_in = torch.cat((chunk_emb, mem), dim=1)

            enc_mask = None
            if attention_mask is not None:
                chunk_mask = attention_mask[:, start:end]
                mem_mask = torch.ones(B, self.num_mem, dtype=chunk_mask.dtype, device=dev)
                enc_mask = torch.cat((chunk_mask, mem_mask), dim=1)

            enc_out = self.llama(inputs_embeds=enc_in, attention_mask=enc_mask)
            pkv = enc_out.past_key_values
            # legacy tuple OR DynamicCache -> normalise to per-layer (k, v)
            pkv = _to_legacy(pkv)
            for li, (lk, lv) in enumerate(pkv):
                keys[li].append(lk[:, :, -self.num_mem:, :])
                vals[li].append(lv[:, :, -self.num_mem:, :])

        trimmed = tuple(
            (torch.cat(keys[li], dim=2), torch.cat(vals[li], dim=2))
            for li in range(self.num_layers)
        )
        return trimmed

    # --------------------------------------------------------------------- #
    #  Training forwards
    # --------------------------------------------------------------------- #
    def forward(self, input_ids, labels, attention_mask=None, mode="regen",
                qa_input_ids=None):
        """Unified forward.

        mode="regen": Stage-1.  Compress `input_ids` per-chunk, decode
            [BOS ; input_text] with LoRA disabled, CE vs `labels` (= text+EOS,
            same length as the decoder input; no internal shift, per reference).

        mode="qa":  Stage-2.  Compress `input_ids` (the CONTEXT block, e.g.
            "### Context:\\n{ctx}") per-chunk into the KV cache.  The decoder then
            processes `qa_input_ids` = [question_block ; answer (+EOS)] UNCOMPRESSED
            against that compressed-context past_key_values (the question block is
            fed as real decoder tokens, then the teacher-forced answer + EOS).  CE
            vs `labels` (answer tokens + EOS only; question positions masked to
            -100).  Consistent with the eval regime: the model SEES the question.
        """
        if mode == "regen":
            return self._forward_regen(input_ids, labels, attention_mask)
        elif mode == "qa":
            return self._forward_qa(input_ids, qa_input_ids, labels, attention_mask)
        raise ValueError(f"unknown mode {mode}")

    def _forward_regen(self, input_ids, labels, attention_mask):
        B = input_ids.shape[0]
        dev = input_ids.device
        pkv = self.compress(input_ids, attention_mask)

        # Decode side: the pretokenized passages start with BOS ([bos]+passage,
        # kept on the COMPRESS side so chunk-0 matches the Stage2/eval context
        # convention). Strip it here before re-adding the decoder BOS, otherwise
        # the decoder input is [BOS, BOS, passage..] and the model is trained to
        # predict a BOS label (2026-06-13 review fix, F64; collator labels are
        # built from the stripped passage accordingly).
        text_ids = input_ids
        if self.bos_token_id is not None and \
                bool((input_ids[:, 0] == self.bos_token_id).all()):
            text_ids = input_ids[:, 1:]
        bos = torch.full((B, 1), self.bos_token_id, device=dev, dtype=torch.long)
        bos_emb = self._embed(bos)
        text_emb = self._embed(text_ids)
        dec_in = torch.cat((bos_emb, text_emb), dim=1)  # [B, 1+T', H]

        with self.llama.disable_adapter():
            dec_out = self.llama(inputs_embeds=dec_in, past_key_values=_to_cache(pkv))
        logits = dec_out.logits  # [B, 1+T, V]

        loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return {"loss": loss, "logits": logits}

    def _forward_qa(self, context_ids, qa_input_ids, labels, attention_mask):
        """Compress the CONTEXT per-chunk, then decode the (uncompressed) question
        block followed by the teacher-forced answer against that compressed KV.

        context_ids   : [B, Tc]    context to compress (e.g. "### Context:\\n..").
        attention_mask: [B, Tc]    mask over the context (chunked in compress()).
        qa_input_ids  : [B, S]     decoder real tokens = [question_block ; answer
                        ; EOS], right-padded.  The question block is uncompressed
                        ("\\n### Question:..\\n### Answer:\\n").
        labels        : [B, S]     -100 over the question positions (and trailing
                        pad), answer tokens (+EOS) over the answer positions.
                        One-to-one with `qa_input_ids` (no internal shift -- the
                        last compressed-KV / question position predicts the first
                        answer token, matching 500xC's no-shift convention; the
                        collator places answer labels at the right per-row offset).
        """
        pkv = self.compress(context_ids, attention_mask)
        qa_emb = self._embed(qa_input_ids)                   # [B, S, H] uncompressed
        with self.llama.disable_adapter():
            dec_out = self.llama(inputs_embeds=qa_emb, past_key_values=_to_cache(pkv))
        logits = dec_out.logits  # [B, S, V]
        loss = self.criterion(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return {"loss": loss, "logits": logits}

    # --------------------------------------------------------------------- #
    #  Inference
    # --------------------------------------------------------------------- #
    @torch.no_grad()
    def lm_inference(self, input_ids, lm_targets, max_new_tokens=128,
                     attention_mask=None):
        """Compress the CONTEXT per-chunk, append the UNCOMPRESSED question block,
        then greedy-decode the answer from the compressed KV.

        Mirrors PACT / the sibling ICAE bridge:
            * input_ids  : [B, Tc]  context to compress per-chunk.
            * lm_targets : [B, Q]   uncompressed question block ("\\n### Question:..
                           \\n### Answer:\\n"); fed as real tokens against the
                           compressed-context past_key_values.
        The question-block prefill advances the cache; we then argmax-decode token
        by token (do_sample=False), stop at EOS, max_new_tokens cap.
        Returns (generated_ids list, num_chunks, num_mem_total).
        """
        B = input_ids.shape[0]
        assert B == 1, "lm_inference supports batch size 1 (eval is bs=1)"
        dev = input_ids.device
        pkv = _to_cache(self.compress(input_ids, attention_mask))

        # prefill the uncompressed question block against the compressed-context KV
        q_emb = self._embed(lm_targets)                       # [B, Q, H]
        with self.llama.disable_adapter():
            out = self.llama(inputs_embeds=q_emb, past_key_values=pkv)
        pkv = out.past_key_values
        nxt = torch.argmax(out.logits[:, -1, :], dim=-1)      # [B] first answer tok

        generated = []
        for _ in range(max_new_tokens):
            tid = nxt.item()
            if tid == self.eos_token_id or tid == EOS_FALLBACK:
                break
            generated.append(tid)
            cur_emb = self._embed(nxt.unsqueeze(0))
            with self.llama.disable_adapter():
                out = self.llama(inputs_embeds=cur_emb, past_key_values=pkv)
            pkv = out.past_key_values
            nxt = torch.argmax(out.logits[:, -1, :], dim=-1)

        T = input_ids.shape[1]
        num_chunks = (T + self.chunk_size - 1) // self.chunk_size
        return generated, num_chunks, num_chunks * self.num_mem

    def trainable_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save_trainable(self, path):
        """Save only LoRA + memory_embeddings (the ~0.3% trainable state)."""
        state = {n: p.detach().cpu() for n, p in self.named_parameters()
                 if p.requires_grad}
        torch.save(state, path)

    def load_trainable(self, path, strict_mem=True):
        state = torch.load(path, map_location="cpu")
        own = dict(self.named_parameters())
        missing = [n for n in state if n not in own]
        mismatched = [n for n in state
                      if n in own and own[n].shape != state[n].shape]
        if missing or mismatched:
            raise RuntimeError(
                f"load_trainable({path}): checkpoint does not match model "
                f"(missing={missing[:5]}{'...' if len(missing) > 5 else ''}, "
                f"shape-mismatch={mismatched[:5]}). Check --lora_r/--lora_alpha "
                f"match the values the checkpoint was trained with.")
        with torch.no_grad():
            for n, p in state.items():
                own[n].copy_(p.to(own[n].dtype))


# ------------------------------------------------------------------------- #
#  Cache helpers (transformers 4.43.1: decoder accepts a Cache or legacy tuple;
#  encoder may return either depending on use_cache path -> normalise both ways)
# ------------------------------------------------------------------------- #
def _to_legacy(pkv):
    """Return a per-layer ((k, v), ...) tuple regardless of input cache type."""
    if pkv is None:
        return None
    if isinstance(pkv, tuple):
        return pkv
    # DynamicCache or similar
    if hasattr(pkv, "to_legacy_cache"):
        return pkv.to_legacy_cache()
    # iterable of layers
    return tuple((k, v) for k, v in pkv)


def _to_cache(pkv):
    """Decoder past_key_values: hand transformers a DynamicCache built from the
    legacy tuple so incremental decode advances positions correctly."""
    if pkv is None:
        return None
    from transformers.cache_utils import DynamicCache
    return DynamicCache.from_legacy_cache(pkv)
