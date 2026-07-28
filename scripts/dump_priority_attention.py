#!/usr/bin/env python
"""PACT paper-figure dump (exp19 gate/priority + exp20 attention + exp21 case).

Loads the PACT-v5 (T1 v3) checkpoint, runs the KPredictor on selected MRQA
samples, and dumps for each sample:
  - per_chunk_mem_sizes
  - per-anchor monotone keep-prob p_i  (the continuous "priority/gate score";
    KPredictor is the priority mechanism in this ckpt, use_gate=False)
  - per-anchor hazard logit
  - selected k_c per chunk (predict_k @ given eps)  + realized retain rate
  - per-chunk text snippet (for annotation)
  - DECODER ATTENTION (exp20): one teacher-forced forward over the prompt with
    output_attentions=True, for compressed-KV (KPredictor @ eps) vs full-KV
    (all anchors). We record, per generated/LM query position, the attention
    mass landing on the memory (anchor) key positions, aggregated to per-chunk.

Reusable / parameterized by --ids, --eps, --work_dir, --out.

Usage (cards 2,3 only):
  CUDA_VISIBLE_DEVICES=2,3 python scripts/dump_priority_attention.py \
      --work_dir experiment/pact_v5_mrqa_dyneps_k32_1B_v3 \
      --eval_data_path data/mrqa_eval_full/qa_eval.pt \
      --eval_meta_path data/mrqa_eval_full/qa_eval_meta.json \
      --ids 52189,53813,33234,38854,6625 --eps 0.001 \
      --out analysis/_dump/priority_attn_eps0.001.json
"""
import argparse, json, os, sys
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modeling import (get_model, load_adapter, extract_full_kv,
                            monotone_keep_logprob, _build_keep_indices)
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    p.add_argument("--eval_data_path", required=True)
    p.add_argument("--eval_meta_path", required=True)
    p.add_argument("--ids", required=True, help="comma-separated sample ids (=index)")
    p.add_argument("--eps", type=float, default=0.001)
    p.add_argument("--adapter_name", default="instruction_adapter.pt")
    p.add_argument("--out", required=True)
    p.add_argument("--attn", action="store_true", default=True,
                   help="also dump decoder attention (exp20)")
    p.add_argument("--no-attn", dest="attn", action="store_false")
    return p.parse_args()


@torch.no_grad()
def decoder_attn_to_chunks(model, pkv, end_idx, lm_targets, per_chunk_mem_sizes,
                           kept_offsets_per_chunk, n_mem_kept):
    """One teacher-forced forward over [LM] + prompt with output_attentions.
    Returns per-(layer)-averaged attention mass from each query position to each
    memory key position, then aggregated per chunk. Memory keys occupy the first
    `n_mem_kept` key positions of past_key_values (the compressed/kept anchors).
    """
    from transformers import DynamicCache
    lm_target_emb = model.decoder.model.embed_tokens(lm_targets)
    bsz, seq_len, emb_size = lm_target_emb.size()
    expand_lm = model.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)
    lm_emb = torch.cat([expand_lm, lm_target_emb], dim=1)         # [1, S+1, E]
    pos_ids = torch.arange(end_idx, end_idx + seq_len + 1,
                           device=lm_target_emb.device).unsqueeze(0)
    cache = DynamicCache.from_legacy_cache(pkv)
    kw = dict(inputs_embeds=lm_emb, past_key_values=cache, use_cache=True,
              output_attentions=True)
    if model.task_config["use_pe"]:
        kw["position_ids"] = pos_ids
    out = model.decoder(**kw)
    # out.attentions: tuple[L] of [B, H, q_len, kv_len]; kv_len = n_mem_kept + q_len
    # we want q over the LM/prompt positions attending to the first n_mem_kept keys.
    attns = out.attentions
    L = len(attns)
    # mean over layers & heads -> [q_len, kv_len]
    A = torch.stack([a[0].mean(0) for a in attns], 0).mean(0).float().cpu()  # [q,kv]
    q_len = A.shape[0]
    mem_attn = A[:, :n_mem_kept]                                  # [q, n_mem_kept]
    # mean attention over all query positions onto each kept memory slot
    per_slot = mem_attn.mean(0)                                   # [n_mem_kept]
    # aggregate kept slots -> per chunk (sum of mass per chunk)
    per_chunk = []
    off = 0
    for nk in kept_offsets_per_chunk:
        per_chunk.append(float(per_slot[off:off + nk].sum().item()))
        off += nk
    return dict(per_chunk_attn_mass=per_chunk,
                per_slot_attn=per_slot.tolist(),
                total_mem_mass=float(per_slot.sum().item()),
                n_layers=L, q_len=int(q_len))


def main():
    args = parse_args()
    rank = 0
    with open(os.path.join(args.work_dir, "output", "config.json")) as f:
        config = json.load(f)
    tcfg = config["sft_training_config"]
    task_config = config["sft_task_config"]
    chunk_size = int(task_config.get("chunk_size", 512))

    model = get_model(tcfg["model_id"], task_config, rank)
    model = load_adapter(model, save_path_and_name=os.path.join(
        args.work_dir, "output", args.adapter_name))
    model.eval()
    if args.attn:
        # eager attention is required for output_attentions (sdpa/flash return None)
        model.decoder.config._attn_implementation = "eager"
        for mod in model.decoder.modules():
            if hasattr(mod, "config"):
                mod.config._attn_implementation = "eager"
    tok = AutoTokenizer.from_pretrained(tcfg["model_id"])

    eval_ex = torch.load(args.eval_data_path)
    with open(args.eval_meta_path) as f:
        meta = json.load(f)

    ids = [int(x) for x in args.ids.split(",")]
    dump = {"eps": args.eps, "samples": []}

    for sid in ids:
        ex = eval_ex[sid]
        m = meta[sid]
        inputs = {"input_ids": ex["input_ids"].unsqueeze(0).to("cuda:0"),
                  "lm_targets": ex["lm_targets"].unsqueeze(0).to("cuda:0")}

        (compress_token_ids, _ctok, end_idx, _mlh,
         enc_pkv, _ems, per_chunk_mem_sizes) = model.compress(inputs, truncate_k=None)
        device = enc_pkv[0][0].device
        anchor_kv = extract_full_kv(enc_pkv)

        # KPredictor: per-anchor hazard logit -> monotone keep-prob (priority score)
        eps_t = torch.tensor([float(args.eps)], device=device)
        logit_list = model.k_predictor.forward(anchor_kv, per_chunk_mem_sizes, eps_t)
        per_chunk_keepprob, per_chunk_hazard = [], []
        for lg in logit_list:
            log_p, _ = monotone_keep_logprob(lg)                 # [1, m_c]
            per_chunk_keepprob.append(log_p.exp()[0].float().cpu().tolist())
            per_chunk_hazard.append(lg[0].float().cpu().tolist())
        _, k_pred = model.k_predictor.predict_k(anchor_kv, per_chunk_mem_sizes, eps_t)
        k_pred = list(k_pred)

        total_anchors = sum(per_chunk_mem_sizes)
        total_kept = sum(min(k, mm) for k, mm in zip(k_pred, per_chunk_mem_sizes))

        # per-chunk text snippet
        chunk_texts = []
        n_ctx = inputs["input_ids"].size(1)
        for ci in range(len(per_chunk_mem_sizes)):
            s = ci * chunk_size
            e = min((ci + 1) * chunk_size, n_ctx)
            txt = tok.decode(inputs["input_ids"][0, s:e], skip_special_tokens=True)
            chunk_texts.append(txt.strip())

        rec = dict(
            id=sid, dataset=m["dataset"], answers=m.get("answers", []),
            n_chunks=len(per_chunk_mem_sizes),
            per_chunk_mem_sizes=list(per_chunk_mem_sizes),
            k_per_chunk=k_pred,
            total_anchors=total_anchors, total_kept=total_kept,
            retain_rate=total_kept / max(1, total_anchors),
            mean_k=total_kept / max(1, len(per_chunk_mem_sizes)),
            per_chunk_keepprob=per_chunk_keepprob,
            per_chunk_hazard=per_chunk_hazard,
            chunk_text_snippet=[t[:400] for t in chunk_texts],
            chunk_text_full=chunk_texts,
        )

        # ---- exp20 attention: compressed (KPredictor) vs full (all anchors) ----
        if args.attn:
            try:
                # compressed
                keep_idx = _build_keep_indices(per_chunk_mem_sizes, k_pred, device)
                pkv_c = tuple((K.index_select(2, keep_idx), V.index_select(2, keep_idx))
                              for (K, V) in enc_pkv)
                kept_per_chunk = [min(k, mm) for k, mm in zip(k_pred, per_chunk_mem_sizes)]
                attn_c = decoder_attn_to_chunks(
                    model, pkv_c, end_idx, inputs["lm_targets"],
                    per_chunk_mem_sizes, kept_per_chunk, int(keep_idx.numel()))
                # full
                attn_f = decoder_attn_to_chunks(
                    model, enc_pkv, end_idx, inputs["lm_targets"],
                    per_chunk_mem_sizes, list(per_chunk_mem_sizes), total_anchors)
                rec["attn_compressed"] = attn_c
                rec["attn_full"] = attn_f
            except Exception as e:
                rec["attn_error"] = repr(e)
                print(f"[id {sid}] attn dump failed: {e!r}", flush=True)

        dump["samples"].append(rec)
        print(f"[id {sid}] {m['dataset']} nchunk={rec['n_chunks']} "
              f"k_per_chunk={k_pred} retain={rec['retain_rate']:.3f}"
              + (f" attn_ok" if 'attn_compressed' in rec else ""), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(dump, f, ensure_ascii=False, indent=2)
    print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
