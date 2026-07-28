"""PACT "ordered residual code" analysis — verify + functional core.

Tests whether per-chunk anchor slots, IN THEIR LEARNED ORDER, form a residual code
(slot k+1 encodes the prefix's residual) rather than merely a decorrelated set.

Targets (functional):
  * answer-NLL: teacher-forced negative log-likelihood of the gold answer given the
    selected slots  -> Exp B (order-ablation) & Exp C (marginal-gain decay).
  * end-of-prompt next-token logit vector -> Exp A-functional residual alignment
    (r_k = logit_full - logit_k ;  D_k = logit_{k+1} - logit_k ;  cos(D_k, r_k)).

Policies (slot-selection orders, all via scripts/slot_select.py):
  canonical (PACT order) / reverse / shuffled-null / random-band / greedy-oracle.

Modes:
  --mode verify      : numerically check canonical-set selection == model.compress(truncate_k)
  --mode functional  : full B/C/A-functional sweep, dump per-example arrays to .npz/.json

Single-GPU (one rank); shard with --shard i/N across GPUs if needed. Reuses the
model's own compress()/_decoder_lm_loss_for_pkv() — no edits to modeling.py.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.modeling import get_model, load_adapter
from transformers import AutoTokenizer
from slot_select import (
    build_keep_indices_from_sets, select_kv,
    policy_canonical, policy_reverse, policy_shuffled, policy_random, make_perms,
)

K_LADDER = [1, 2, 4, 8, 16, 32]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    p.add_argument("--adapter_name", default="instruction_adapter.pt")
    p.add_argument("--eval_data_path", required=True)
    p.add_argument("--eval_meta_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mode", choices=["verify", "functional"], default="functional")
    p.add_argument("--num_samples", type=int, default=300)
    p.add_argument("--min_ctx_len", type=int, default=512,
                   help="only use examples whose context >= this (full 32-slot chunk)")
    p.add_argument("--n_random", type=int, default=8, help="random-subset draws for the band")
    p.add_argument("--do_greedy", type=int, default=0, help="1=compute greedy-oracle (O(m^2), slow)")
    p.add_argument("--null_seed", type=int, default=1234)
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--shard", default="0/1", help="i/N shard of the selected examples")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


@torch.no_grad()
def score_pkv(model, lm_emb_prompt, answer_emb, answer_ids, pkv, end_idx, device,
              capture_attn=False, attn_store=None):
    """Teacher-forced answer scoring given selected past_key_values (the slots).

    lm_emb_prompt: [1, 1+Lp, D]  = [LM] token + prompt(lm_targets) embeds  (precomputed)
    answer_emb:    [1, T, D]      = gold answer embeds
    answer_ids:    [1, T]         = gold answer token ids (targets)
    Returns: (answer_nll: float, first_logits: [V] tensor = next-token dist at last
             prompt position = prediction of answer token 1).
    Feeds [LM,prompt, answer[:-1]] so logit at the last-prompt position predicts a_1.
    """
    Lp1 = lm_emb_prompt.size(1)               # 1 + Lp
    T = answer_emb.size(1)
    dec_in = torch.cat([lm_emb_prompt, answer_emb[:, :-1, :]], dim=1) if T > 1 else lm_emb_prompt
    seq = dec_in.size(1)
    pos = torch.arange(end_idx, end_idx + seq, device=device).unsqueeze(0)
    use_pe = model.task_config["use_pe"]
    if use_pe:
        out = model.decoder(inputs_embeds=dec_in, position_ids=pos, past_key_values=pkv)
    else:
        out = model.decoder(inputs_embeds=dec_in, past_key_values=pkv)
    logits = out.logits                                    # [1, seq, V]
    # logits[Lp1-1 : Lp1-1+T] predict answer tokens a_1..a_T
    start = Lp1 - 1
    ans_logits = logits[:, start:start + T, :]             # [1, T, V]
    ce = torch.nn.functional.cross_entropy(
        ans_logits.reshape(-1, model.vocab_size).float(),
        answer_ids.reshape(-1).to(device), reduction="mean")
    first_logits = logits[0, start, :].float().detach().cpu()   # next-tok dist (predict a_1)
    return float(ce.item()), first_logits


def main():
    args = parse_args()
    device = torch.device(args.device)
    with open(os.path.join(args.work_dir, "output", "config.json")) as f:
        config = json.load(f)
    tcfg, traincfg = config["sft_task_config"], config["sft_training_config"]
    rank = int(args.device.split(":")[-1]) if "cuda" in args.device else 0
    model = get_model(traincfg["model_id"], tcfg, rank)
    model = load_adapter(model, save_path_and_name=os.path.join(args.work_dir, "output", args.adapter_name))
    model.eval()
    tok = AutoTokenizer.from_pretrained(traincfg["model_id"])

    examples = torch.load(args.eval_data_path)
    meta = json.load(open(args.eval_meta_path))
    # select full-chunk examples
    sel = [i for i in range(len(examples)) if examples[i]["input_ids"].shape[-1] >= args.min_ctx_len]
    rng = np.random.RandomState(args.sample_seed)
    rng.shuffle(sel)
    sel = sel[:args.num_samples]
    si, sn = (int(x) for x in args.shard.split("/"))
    sel = sel[si::sn]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] {len(sel)} examples (ctx>={args.min_ctx_len}), mode={args.mode}, shard={args.shard}")

    g_null = torch.Generator(device=device).manual_seed(args.null_seed)
    g_rand = torch.Generator(device=device).manual_seed(args.null_seed + 777)

    records = []
    verify_max_abs = 0.0
    for n, idx in enumerate(sel):
        ex, mt = examples[idx], meta[idx]
        inputs = {"input_ids": ex["input_ids"].unsqueeze(0).to(device),
                  "lm_targets": ex["lm_targets"].unsqueeze(0).to(device)}
        # gold answer (first reference), tokenized w/o special tokens
        ans_txt = mt["answers"][0]
        ans_ids = torch.tensor(tok(ans_txt, add_special_tokens=False)["input_ids"],
                               dtype=torch.long).unsqueeze(0)
        if ans_ids.numel() == 0:
            continue
        ans_emb = model.decoder.model.embed_tokens(ans_ids.to(device))

        # precompute prompt decoder embeds: [LM] + lm_targets
        lm_target_emb = model.decoder.model.embed_tokens(inputs["lm_targets"])
        bsz, Lp, D = lm_target_emb.size()
        lm_tok = model.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, D)
        lm_emb_prompt = torch.cat([lm_tok, lm_target_emb], dim=1)

        # compress full (no truncation)
        (_, _, end_idx, _, pkv_full, _, per_chunk) = model.compress(inputs, truncate_k=None)
        m_total = sum(per_chunk)
        ks = [k for k in K_LADDER if k <= max(per_chunk)] or [max(per_chunk)]

        if args.mode == "verify":
            for k in ks:
                # path A: full compress + my canonical selection
                sets = policy_canonical(per_chunk, k, device)
                keep = build_keep_indices_from_sets(per_chunk, sets, device)
                pkv_sel = select_kv(pkv_full, keep)
                nll_a, lg_a = score_pkv(model, lm_emb_prompt, ans_emb, ans_ids, pkv_sel, end_idx, device)
                # path B: model.compress(truncate_k=k)
                (_, _, e2, _, pkv_trunc, _, pc2) = model.compress(inputs, truncate_k=k)
                nll_b, lg_b = score_pkv(model, lm_emb_prompt, ans_emb, ans_ids, pkv_trunc, e2, device)
                d = max(abs(nll_a - nll_b), float((lg_a - lg_b).abs().max()))
                verify_max_abs = max(verify_max_abs, d)
            if n < 3:
                print(f"  [verify] ex{idx} per_chunk={per_chunk} max|Δ|so far={verify_max_abs:.2e}")
            continue

        # ---- functional mode ----
        rec = {"idx": int(idx), "dataset": mt.get("dataset"), "per_chunk": list(map(int, per_chunk)),
               "ks": ks, "m_total": int(m_total)}
        # full target (k = all): answer NLL + logit
        nll_full, lg_full = score_pkv(model, lm_emb_prompt, ans_emb, ans_ids, pkv_full, end_idx, device)
        rec["nll_full"] = nll_full

        def sweep(policy_fn):
            nlls, logits_by_k = [], []
            for k in ks:
                sets = policy_fn(k)
                keep = build_keep_indices_from_sets(per_chunk, sets, device)
                pkv_sel = select_kv(pkv_full, keep)
                nll, lg = score_pkv(model, lm_emb_prompt, ans_emb, ans_ids, pkv_sel, end_idx, device)
                nlls.append(nll); logits_by_k.append(lg)
            return nlls, logits_by_k

        # canonical (+ fine ladder 1..m for marginal-decay)
        fine_ks = list(range(1, max(per_chunk) + 1))
        nll_canon_fine = []
        logits_canon_fine = []
        for k in fine_ks:
            sets = policy_canonical(per_chunk, k, device)
            keep = build_keep_indices_from_sets(per_chunk, sets, device)
            nll, lg = score_pkv(model, lm_emb_prompt, ans_emb, ans_ids, select_kv(pkv_full, keep), end_idx, device)
            nll_canon_fine.append(nll); logits_canon_fine.append(lg)
        rec["fine_ks"] = fine_ks
        rec["nll_canonical_fine"] = nll_canon_fine

        rec["nll_canonical"], logits_canon = sweep(lambda k: policy_canonical(per_chunk, k, device))
        rec["nll_reverse"], _ = sweep(lambda k: policy_reverse(per_chunk, k, device))
        null_perms = make_perms(per_chunk, g_null, device)
        rec["nll_shuffled"], logits_shuf = sweep(lambda k: policy_shuffled(per_chunk, k, device, null_perms))
        # shuffled fine ladder for null residual-alignment
        nll_shuf_fine, logits_shuf_fine = [], []
        for k in fine_ks:
            sets = policy_shuffled(per_chunk, k, device, null_perms)
            keep = build_keep_indices_from_sets(per_chunk, sets, device)
            nll, lg = score_pkv(model, lm_emb_prompt, ans_emb, ans_ids, select_kv(pkv_full, keep), end_idx, device)
            nll_shuf_fine.append(nll); logits_shuf_fine.append(lg)
        rec["nll_shuffled_fine"] = nll_shuf_fine

        # random band
        rand_runs = []
        for _ in range(args.n_random):
            nlls, _ = sweep(lambda k: policy_random(per_chunk, k, device, g_rand))
            rand_runs.append(nlls)
        rec["nll_random_runs"] = rand_runs

        # greedy-oracle (optional)
        if args.do_greedy:
            rec["nll_greedy"] = greedy_oracle(model, per_chunk, pkv_full, lm_emb_prompt,
                                              ans_emb, ans_ids, end_idx, device, ks)

        # ---- functional residual alignment (canonical vs shuffled-null), fine ladder ----
        rec["align_canonical"] = residual_alignment(logits_canon_fine, lg_full)
        rec["align_shuffled"] = residual_alignment(logits_shuf_fine, lg_full)
        records.append(rec)
        if (n + 1) % 25 == 0:
            print(f"  [functional] {n+1}/{len(sel)} done")

    if args.mode == "verify":
        ok = verify_max_abs < 1e-3
        print(f"[VERIFY] max|Δ(nll,logit)| canonical-vs-truncate = {verify_max_abs:.3e}  -> {'PASS' if ok else 'FAIL'}")
        sys.exit(0 if ok else 1)

    out = os.path.join(args.out_dir, f"functional_shard{args.shard.replace('/','of')}.json")
    with open(out, "w") as f:
        json.dump(records, f)
    print(f"[INFO] wrote {len(records)} records -> {out}")


def residual_alignment(logits_by_k, logit_full):
    """cos(D_k, r_k) and ||r_k|| over the fine k ladder. logits_by_k[i] = logit at k=i+1."""
    lf = logit_full
    cos, rnorm = [], []
    for i in range(len(logits_by_k) - 1):
        r_k = (lf - logits_by_k[i])
        d_k = (logits_by_k[i + 1] - logits_by_k[i])
        nr, nd = r_k.norm().item(), d_k.norm().item()
        cos.append(float((r_k @ d_k).item() / (nr * nd + 1e-12)))
        rnorm.append(nr)
    rnorm.append((lf - logits_by_k[-1]).norm().item())
    return {"cos": cos, "rnorm": rnorm}


@torch.no_grad()
def greedy_oracle(model, per_chunk, pkv_full, lm_emb_prompt, ans_emb, ans_ids, end_idx, device, ks):
    """Greedily grow a per-chunk slot set minimizing answer NLL; report NLL at each k in ks.
    Single-chunk assumption for tractability (per_chunk has 1 entry for full-chunk examples)."""
    assert len(per_chunk) == 1, "greedy oracle implemented for single-chunk examples only"
    m = per_chunk[0]
    chosen, remaining = [], list(range(m))
    nll_at = {}
    while remaining:
        best, best_nll = None, 1e9
        for c in remaining:
            cand = chosen + [c]
            keep = build_keep_indices_from_sets(per_chunk, [torch.tensor(cand, device=device)], device)
            nll, _ = score_pkv(model, lm_emb_prompt, ans_emb, ans_ids, select_kv(pkv_full, keep), end_idx, device)
            if nll < best_nll:
                best_nll, best = nll, c
        chosen.append(best); remaining.remove(best)
        if len(chosen) in ks:
            nll_at[len(chosen)] = best_nll
    return [nll_at.get(k) for k in ks]


if __name__ == "__main__":
    main()
