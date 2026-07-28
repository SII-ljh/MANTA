"""Per-chunk In-Context Autoencoder (ICAE, Ge et al. ICLR 2024) for PACT baselines.

Controlled reproduction matched to the PACT / SAC / Activation-Beacon baselines:
  * Backbone  : Llama-3.1-8B base (frozen).  hidden/layers read from config.
  * Encoder   : Llama + LoRA(q,v).  Compresses EACH consecutive `chunk_size`(=512)
                window of the input into `num_mem` memory tokens by appending
                `num_mem` learnable memory embeddings after the chunk's token
                embeddings, running the encoder, and taking the LAST `num_mem`
                hidden states as that chunk's compressed representation.
                Per-chunk reps are concatenated along the sequence dim, so a
                T-token input -> ceil(T/chunk_size) * num_mem compressed vectors.
  * Decoder   : the SAME Llama with LoRA DISABLED (`disable_adapter()`),
                conditioned on `[compressed_reps ; ae_token ; uncompressed_prompt]`.
  * ratio     : chunk_size / num_mem  (16x -> num_mem = 512/16 = 32).

Three modes (single nn.Module):
  * forward(mode="ae")  : pretrain autoencoding. compress passage per-chunk ->
                          regenerate the passage tokens (+eos). CE over passage.
  * forward(mode="qa")  : QA SFT. compress the CONTEXT per-chunk, append the
                          (uncompressed) question block, CE on ANSWER tokens only.
  * lm_inference(...)   : compress context, append question block, greedy decode.

PACT-comparability note (important): PACT's eval feeds `input_ids`=context (to be
compressed) and `lm_targets`=the uncompressed question block ("### Question..\n###
Answer:\n"), then generates the answer. We mirror that EXACTLY: the question is
appended UNCOMPRESSED to the decoder (as in native ICAE-QA where the decoder sees
`[mem ; qa_tokens]`). This keeps compression query-agnostic (context compressed
without seeing the question) and directly comparable to SAC/AB/PACT.

Only LoRA + memory_embeddings + ae_embedding are trainable; base is frozen.
"""
import os
import torch
import torch.nn as nn
from peft import get_peft_model
from transformers import AutoTokenizer, AutoModelForCausalLM


class ICAEPerChunk(nn.Module):
    def __init__(
        self,
        llama_path,
        lora_config,
        num_mem,
        chunk_size=512,
        torch_dtype=torch.bfloat16,
        device=None,
        attn_impl="sdpa",
    ):
        super().__init__()
        self.chunk_size = int(chunk_size)
        self.num_mem = int(num_mem)

        llama = AutoModelForCausalLM.from_pretrained(
            llama_path,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
        self.llama = get_peft_model(llama, lora_config)
        # Freeze everything except LoRA adapters.
        for name, param in self.llama.named_parameters():
            param.requires_grad = "lora" in name

        cfg = self.llama.config
        self.hidden_size = cfg.hidden_size  # read from config (not hard-coded 4096)
        self.vocab_size = cfg.vocab_size

        self.tokenizer = AutoTokenizer.from_pretrained(llama_path)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.eos_token_id = self.tokenizer.eos_token_id

        self.criterion = nn.CrossEntropyLoss(ignore_index=-100)

        # Learnable compression prompts.  Init like native ICAE (randn).
        self.memory_embeddings = nn.Parameter(
            torch.randn(1, self.num_mem, self.hidden_size, dtype=torch_dtype) * 0.02
        )
        self.ae_embedding = nn.Parameter(
            torch.randn(1, 1, self.hidden_size, dtype=torch_dtype) * 0.02
        )
        # HF Trainer pokes model.config; expose the backbone's.
        self.config = cfg
        if device is not None:
            self.to(device)

    # ------------------------------------------------------------------ utils
    @property
    def device(self):
        return self.memory_embeddings.device

    def _embed(self, token_ids):
        return self.llama.get_input_embeddings()(token_ids)

    def trainable_parameters(self):
        return [p for p in self.parameters() if p.requires_grad]

    def num_trainable(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    # ----------------------------------------------------------- compression
    def compress(self, input_ids):
        """Per-chunk compress `input_ids` -> [B, n_chunks*num_mem, H] hidden states.

        input_ids: [B, T] (right-padded with eos handled by caller via labels=-100).
        We chunk along T into consecutive `chunk_size` windows; the final window
        may be < chunk_size (still >=1 token -> memory always engages).
        Returns (compressed_reps[B, M, H], n_chunks).
        """
        B, T = input_ids.shape
        dev = self.device
        input_ids = input_ids.to(dev)
        mem = self.memory_embeddings.to(self.memory_embeddings.dtype)

        chunk_outs = []
        n_chunks = (T + self.chunk_size - 1) // self.chunk_size
        for c in range(n_chunks):
            s = c * self.chunk_size
            e = min(s + self.chunk_size, T)
            chunk_ids = input_ids[:, s:e]                       # [B, L<=chunk]
            tok_emb = self._embed(chunk_ids)                    # [B, L, H]
            mem_emb = mem.repeat(B, 1, 1).to(tok_emb.dtype)     # [B, num_mem, H]
            enc_in = torch.cat([tok_emb, mem_emb], dim=1)       # [B, L+num_mem, H]
            enc_out = self.llama(
                inputs_embeds=enc_in, output_hidden_states=True
            ).hidden_states[-1]
            chunk_outs.append(enc_out[:, -self.num_mem:, :])    # [B, num_mem, H]
        compressed = torch.cat(chunk_outs, dim=1)               # [B, M, H]
        return compressed, n_chunks

    # ------------------------------------------------------------- decoding
    def _decode_logits(self, compressed, prompt_embeds):
        """Decoder = base Llama (LoRA disabled) over [compressed ; ae ; prompt]."""
        B = compressed.shape[0]
        ae = self.ae_embedding.repeat(B, 1, 1).to(compressed.dtype)  # [B,1,H]
        dec_in = torch.cat([compressed, ae, prompt_embeds], dim=1)
        with self.llama.disable_adapter():
            out = self.llama(inputs_embeds=dec_in)
        return out.logits

    # --------------------------------------------------------------- forward
    def forward(self, input_ids=None, labels=None, mode="ae", **kw):
        """Unified training forward.

        mode == "ae":  autoencoding pretrain.
            input_ids : [B, T] passage tokens.
            labels    : [B, M + 1 + T] (-100 except over the T regenerated tokens,
                        i.e. teacher-forced passage + trailing eos), built by the
                        collator so positions line up with [compressed ; ae ; text].
        mode == "qa":  QA SFT.
            input_ids : [B, ctx_len + qa_len] = context tokens || question||answer.
            labels    : [B, M + 1 + qa_len] (-100 except over answer (+eos) span).
            The split point is passed via `ctx_len` so we compress only context and
            feed the (q||a) span uncompressed to the decoder.
        """
        if mode == "ae":
            compressed, _ = self.compress(input_ids)
            text_emb = self._embed(input_ids.to(self.device))     # teacher forcing
            logits = self._decode_logits(compressed, text_emb)
        elif mode == "qa":
            ctx_len = kw["ctx_len"]
            ctx_ids = input_ids[:, :ctx_len]
            qa_ids = input_ids[:, ctx_len:]
            compressed, _ = self.compress(ctx_ids)
            qa_emb = self._embed(qa_ids.to(self.device))
            logits = self._decode_logits(compressed, qa_emb)
        else:
            raise ValueError(mode)

        labels = labels.to(logits.device)
        loss = self.criterion(
            logits.reshape(-1, logits.size(-1)).float(), labels.reshape(-1)
        )
        return {"loss": loss, "logits": logits}

    # ------------------------------------------------------------- inference
    @torch.no_grad()
    def lm_inference(self, inputs, generate_num=128, **kw):
        """PACT-schema greedy generation.

        inputs: dict with
            input_ids  : [B, T]  context to compress (per-chunk).
            lm_targets : [B, Q]  uncompressed question block (e.g. "\\n### Question:
                         ...\\n### Answer:\\n"); appended after the ae token.
        Returns (generated_token_ids: list[int], n_chunks: int).
        Greedy, do_sample=False, stop at eos, max_new_tokens=generate_num.
        """
        input_ids = inputs["input_ids"].to(self.device)
        lm_targets = inputs["lm_targets"].to(self.device)
        B = input_ids.shape[0]
        assert B == 1, "lm_inference is single-sample (B=1)."

        compressed, n_chunks = self.compress(input_ids)
        ae = self.ae_embedding.repeat(B, 1, 1).to(compressed.dtype)
        q_emb = self._embed(lm_targets)                        # uncompressed prompt
        prefix = torch.cat([compressed, ae, q_emb], dim=1)     # [B, M+1+Q, H]

        generated = []
        with self.llama.disable_adapter():
            out = self.llama(inputs_embeds=prefix, use_cache=True)
            past = out.past_key_values
            next_id = out.logits[:, -1].argmax(dim=-1)         # [B]
            generated.append(next_id.item())
            if next_id.item() == self.eos_token_id:
                return generated, n_chunks
            for _ in range(generate_num - 1):
                step_emb = self._embed(next_id).unsqueeze(1).to(compressed.dtype)
                out = self.llama(
                    inputs_embeds=step_emb, past_key_values=past, use_cache=True
                )
                past = out.past_key_values
                next_id = out.logits[:, -1].argmax(dim=-1)
                tid = next_id.item()
                if tid == self.eos_token_id:
                    break
                generated.append(tid)
        return generated, n_chunks

    # ------------------------------------------------- checkpoint (trainable only)
    def trainable_state_dict(self):
        sd = {}
        for name, p in self.named_parameters():
            if p.requires_grad or name.endswith("memory_embeddings") or name.endswith("ae_embedding"):
                sd[name] = p.detach().cpu()
        return sd

    def save_trainable(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.trainable_state_dict(), path)

    def load_trainable(self, path, map_location="cpu"):
        sd = torch.load(path, map_location=map_location)
        own = dict(self.named_parameters())
        missing = []
        with torch.no_grad():
            for name, val in sd.items():
                if name in own:
                    own[name].copy_(val.to(own[name].dtype).to(own[name].device))
                else:
                    missing.append(name)
        if missing:
            print(f"[load_trainable] {len(missing)} keys not found in model (skipped).")
        return self
