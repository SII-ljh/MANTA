#!/usr/bin/env python
"""Causal grounding of PACT priority-aware budget allocation (Experiment A / F99).

Question (closes the deferred §3 motivation TODO):
  Does the KPredictor allocate MORE per-chunk compression budget (k_c) to chunks
  that are causally MORE important for producing the correct answer?

Why non-circular (unlike F92): chunk causal importance I_c is measured on the
FROZEN BASE decoder (self.decoder; the adapter ckpt has no decoder.* keys, so it
is the pristine Llama-3.2-1B with adapter OFF) via leave-one-chunk-out erasure on
the answer NLL — independent of PACT's own nested encoding / oracle losses.

Length-control: chunks are equal-length 512-token blocks (the trailing partial
chunk is dropped from both I_c and k_c), so I_c is not a length artifact.

Data reality: multi-chunk MRQA eval samples (>=3 full 512-tok chunks) are all
IID (MRQA in-domain; OOD sets are short). So this is IID motivation evidence.

Single-GPU analysis. Reuses get_model/load_adapter/extract_full_kv/predict_k and
the IID/OOD + spearman helpers from analysis_predictor_fidelity.py.
"""
import argparse
import json
import math
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modeling import get_model, load_adapter, extract_full_kv

# ---- reused definitions (verbatim from analysis_predictor_fidelity.py) -------
IID_SETS = {"SQuAD", "TriviaQA-web", "NewsQA", "SearchQA", "HotpotQA",
            "NaturalQuestionsShort"}
OOD_SETS = {"BioASQ", "DROP", "DuoRC.ParaphraseRC", "RACE",
            "RelationExtraction", "TextbookQA"}


def bucket_of(dataset):
    if dataset in IID_SETS:
        return "IID"
    if dataset in OOD_SETS:
        return "OOD"
    return None


def rankdata(a):
    a = np.asarray(a, dtype=float)
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    sa = a[order]
    i = 0
    while i < len(a):
        j = i
        while j + 1 < len(a) and sa[j + 1] == sa[i]:
            j += 1
        ranks[order[i:j + 1]] = 0.5 * (i + j) + 1.0
        i = j + 1
    return ranks


def spearman(x, y):
    if len(x) < 2:
        return None
    rx, ry = rankdata(x), rankdata(y)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return None
    return float(np.corrcoef(rx, ry)[0, 1])


def partial_spearman(x, y, z):
    """Rank-based partial correlation of x,y controlling for z."""
    if len(x) < 3:
        return None
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)
    if np.std(rx) == 0 or np.std(ry) == 0 or np.std(rz) == 0:
        return None

    def c(a, b):
        return float(np.corrcoef(a, b)[0, 1])
    r_xy, r_xz, r_yz = c(rx, ry), c(rx, rz), c(ry, rz)
    denom = math.sqrt(max(1e-12, (1 - r_xz ** 2) * (1 - r_yz ** 2)))
    if denom == 0:
        return None
    return (r_xy - r_xz * r_yz) / denom


# -----------------------------------------------------------------------------
@torch.no_grad()
def answer_nll(model, ctx_ids, prompt_ids, ans_ids, device):
    """Vanilla base-decoder answer NLL: feed [ctx, prompt, answer], score answer
    tokens only. NO compression / PACT special token / end_idx offset."""
    seq = torch.cat([ctx_ids, prompt_ids, ans_ids]).unsqueeze(0).to(device)
    emb = model.decoder.model.embed_tokens(seq)
    pos = torch.arange(seq.size(1), device=device).unsqueeze(0)
    out = model.decoder(inputs_embeds=emb, position_ids=pos)
    logits = out.logits[0].float()                      # [T, V]
    n_ans = ans_ids.size(0)
    tgt = seq[0, -n_ans:]                               # answer tokens
    pred_logits = logits[-n_ans - 1:-1]                # positions predicting them
    ce = F.cross_entropy(pred_logits, tgt, reduction="mean")
    return float(ce.item())


@torch.no_grad()
def predict_budget(model, ctx_ids, device, eps):
    """KPredictor per-chunk budget k_c at fixed epsilon (deployment threshold)."""
    inputs = {"input_ids": ctx_ids.unsqueeze(0).to(device)}
    (_, _, _, _, enc_pkv, _, per_chunk_mem_sizes) = model.compress(
        inputs, truncate_k=None)
    anchor_kv = extract_full_kv(enc_pkv)
    eps_t = torch.tensor([eps], device=anchor_kv.device, dtype=torch.float32)
    _, ks_b0 = model.k_predictor.predict_k(
        anchor_kv, per_chunk_mem_sizes, eps_t, threshold=None)
    return ks_b0, per_chunk_mem_sizes               # list[int], list[int]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--work_dir", required=True)
    ap.add_argument("--eval_data_path", required=True)
    ap.add_argument("--eval_meta_path", required=True)
    ap.add_argument("--adapter_name", default="instruction_adapter.pt")
    ap.add_argument("--eps", type=float, default=0.05)
    ap.add_argument("--min_full_chunks", type=int, default=3)
    ap.add_argument("--max_samples", type=int, default=0, help="0 = all matching")
    ap.add_argument("--n_boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rng = np.random.RandomState(args.seed)
    device = "cuda:0"
    chunk_size = 512  # confirmed from v3 config

    with open(os.path.join(args.work_dir, "output", "config.json")) as f:
        config = json.load(f)
    training_config = config["sft_training_config"]
    task_config = config["sft_task_config"]
    cs = int(task_config.get("chunk_size", chunk_size))
    assert cs == chunk_size, f"chunk_size mismatch {cs}"

    model = get_model(training_config["model_id"], task_config, rank=0)
    model = load_adapter(
        model, save_path_and_name=os.path.join(
            args.work_dir, "output", args.adapter_name))
    model.eval()
    assert getattr(model, "k_predictor", None) is not None
    tokenizer = AutoTokenizer.from_pretrained(training_config["model_id"])
    eos = tokenizer.eos_token_id

    ex = torch.load(args.eval_data_path, map_location="cpu")
    meta = json.load(open(args.eval_meta_path))

    # filter to >= min_full_chunks full chunks
    cand = [i for i in range(len(ex))
            if ex[i]["input_ids"].shape[0] // chunk_size >= args.min_full_chunks]
    if args.max_samples and len(cand) > args.max_samples:
        cand = list(rng.choice(cand, size=args.max_samples, replace=False))
    print(f"[info] {len(cand)} samples with >={args.min_full_chunks} full chunks",
          flush=True)

    per_sample = []
    skipped = 0
    for n, i in enumerate(cand):
        ctx = ex[i]["input_ids"].cpu()
        prompt = ex[i]["lm_targets"].cpu()
        ans_text = meta[i]["answers"][0]
        ans_ids = tokenizer(ans_text, add_special_tokens=False)["input_ids"]
        if eos is not None:
            ans_ids = ans_ids + [eos]
        ans_ids = torch.tensor(ans_ids, dtype=ctx.dtype)
        if ans_ids.numel() == 0:
            skipped += 1
            continue

        total_len = ctx.shape[0]
        n_full = total_len // chunk_size

        # base full-context answer NLL
        nll_full = answer_nll(model, ctx, prompt, ans_ids, device)

        # leave-one-chunk-out on each FULL chunk
        I_c = []
        for c in range(n_full):
            s, e = c * chunk_size, (c + 1) * chunk_size
            if c == 0:
                # keep BOS (ctx[0]) so chunk-0 erasure isn't a BOS-removal
                # artifact; remove the remaining chunk-0 content only.
                ctx_erased = torch.cat([ctx[:1], ctx[e:]])
            else:
                ctx_erased = torch.cat([ctx[:s], ctx[e:]])
            nll_c = answer_nll(model, ctx_erased, prompt, ans_ids, device)
            I_c.append(nll_c - nll_full)

        # KPredictor budget per chunk (first n_full entries)
        k_b0, per_chunk_mem = predict_budget(model, ctx, device, args.eps)
        assert len(per_chunk_mem) >= n_full
        k_c = [int(k_b0[c]) for c in range(n_full)]

        per_sample.append({
            "idx": int(i),
            "dataset": meta[i]["dataset"],
            "bucket": bucket_of(meta[i]["dataset"]),
            "total_len": int(total_len),
            "n_full_chunks": int(n_full),
            "nll_full": nll_full,
            "I_c": [float(v) for v in I_c],       # causal importance per chunk
            "k_c": k_c,                            # predicted budget per chunk
            "positions": list(range(n_full)),
        })
        if (n + 1) % 50 == 0:
            print(f"[prog] {n + 1}/{len(cand)} done", flush=True)

    print(f"[info] usable samples={len(per_sample)} skipped={skipped}", flush=True)

    # ---- aggregate stats ----------------------------------------------------
    def pool(exclude_chunk0=False):
        I, K, P, SID = [], [], [], []
        for si, s in enumerate(per_sample):
            rng_c = range(1, s["n_full_chunks"]) if exclude_chunk0 \
                else range(s["n_full_chunks"])
            for c in rng_c:
                I.append(s["I_c"][c]); K.append(s["k_c"][c])
                P.append(c); SID.append(si)
        return np.array(I), np.array(K), np.array(P), np.array(SID)

    def within_sample_mean(exclude_chunk0=False):
        vals = []
        n_const = 0
        for s in per_sample:
            rng_c = range(1, s["n_full_chunks"]) if exclude_chunk0 \
                else range(s["n_full_chunks"])
            Ic = [s["I_c"][c] for c in rng_c]
            Kc = [s["k_c"][c] for c in rng_c]
            if len(Ic) < 3:
                continue
            r = spearman(Ic, Kc)
            if r is None:
                n_const += 1
                continue
            vals.append(r)
        return (float(np.mean(vals)) if vals else None, len(vals), n_const)

    def boot_partial(I, K, P, SID, n_boot):
        """Bootstrap CI on pooled partial-spearman(I,K|P) by resampling SAMPLES."""
        uniq = np.unique(SID)
        base = partial_spearman(I, K, P)
        boots = []
        for _ in range(n_boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            mask_idx = np.concatenate([np.where(SID == u)[0] for u in pick])
            r = partial_spearman(I[mask_idx], K[mask_idx], P[mask_idx])
            if r is not None:
                boots.append(r)
        boots = np.array(boots)
        ci = (float(np.percentile(boots, 2.5)),
              float(np.percentile(boots, 97.5))) if len(boots) else (None, None)
        return base, ci, (float((boots > 0).mean()) if len(boots) else None)

    summary = {
        "config": {"eps": args.eps, "min_full_chunks": args.min_full_chunks,
                   "chunk_size": chunk_size, "n_samples": len(per_sample),
                   "work_dir": args.work_dir, "seed": args.seed},
    }
    for tag, excl in [("all_chunks", False), ("exclude_chunk0", True)]:
        I, K, P, SID = pool(excl)
        ws_mean, ws_n, ws_const = within_sample_mean(excl)
        base_pc, ci_pc, frac_pos = boot_partial(I, K, P, SID, args.n_boot)
        # sanity: shuffling k breaks the I-k pairing -> partial corr ~ 0
        Kperm = K.copy(); rng.shuffle(Kperm)
        perm_null = partial_spearman(I, Kperm, P)
        summary[tag] = {
            "n_chunks": int(len(I)),
            "pooled_spearman_I_k": spearman(I, K),
            "pooled_spearman_pos_k": spearman(P, K),
            "pooled_spearman_pos_I": spearman(P, I),
            "pooled_partial_spearman_I_k_given_pos": base_pc,
            "partial_boot_ci95": ci_pc,
            "partial_boot_frac_positive": frac_pos,
            "sanity_permuted_k_partial": perm_null,
            "within_sample_mean_spearman": ws_mean,
            "within_sample_n": ws_n,
            "within_sample_n_constant_k": ws_const,
            "mean_k": float(np.mean(K)) if len(K) else None,
            "std_k": float(np.std(K)) if len(K) else None,
            "mean_I": float(np.mean(I)) if len(I) else None,
        }

    out = {"summary": summary, "per_sample": per_sample}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("[done] wrote", args.out, flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
