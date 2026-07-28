"""Per-token NLL attribution for PACT "ordered residual code" — token-resolution test.

Question (sharper than F86's mean-NLL): when slot k is added on top of the canonical
prefix of (k-1) slots, is the loss reduction CONCENTRATED on exactly the answer tokens
that were already HARD (high NLL) under prefix k-1? If slots form a residual code, the
canonical next slot should specifically repair the residual errors of the prefix.

Controlled design (the only thing that varies is WHICH slot is added):
  base prefix  = canonical first (s-1) anchors per chunk           [FIXED across arms]
  per-token base NLL = NLL of gold answer tokens under that base   [computed ONCE]
  +1 slot      = one more anchor per chunk, chosen by policy:
     canonical : the learned next anchor  (min remaining local idx)   -> reproduces compress(truncate_k=s)
     reverse   : the learned LAST anchor  (max remaining local idx)
     shuffled  : a fixed-permutation pick from remaining (deterministic null ordering)
     random    : uniform pick from remaining, averaged over n draws (band null)
  delta_token  = NLL_base_token - NLL_(base + slot)_token           [+ = improvement]

We then correlate (base_nll_token, delta_token) across all (sample, s, token):
canonical should show the strongest positive correlation + largest mean delta +
highest concentration of total reduction on the hardest tokens. Controls are the null.

No edits to modeling.py: reuse model.compress(truncate_k=None) + scripts/slot_select.py.
Single GPU per rank; --shard i/N to split examples across GPUs.
"""
import argparse, json, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from model.modeling import get_model, load_adapter
from transformers import AutoTokenizer
from slot_select import build_keep_indices_from_sets, select_kv


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    p.add_argument("--adapter_name", default="instruction_adapter.pt")
    p.add_argument("--eval_data_path", required=True)
    p.add_argument("--eval_meta_path", required=True)
    p.add_argument("--out_dir", required=True)
    p.add_argument("--mode", choices=["verify", "functional"], default="functional")
    p.add_argument("--num_samples", type=int, default=150)
    p.add_argument("--min_ctx_len", type=int, default=512)
    p.add_argument("--n_random", type=int, default=6)
    p.add_argument("--max_steps", type=int, default=32,
                   help="cap on add-steps s (1..min(max_steps, max m_c)); slots are front-loaded")
    p.add_argument("--null_seed", type=int, default=1234)
    p.add_argument("--sample_seed", type=int, default=42)
    p.add_argument("--shard", default="0/1")
    p.add_argument("--device", default="cuda:0")
    return p.parse_args()


@torch.no_grad()
def per_token_nll(model, lm_emb_prompt, answer_emb, answer_ids, pkv, end_idx, device):
    """Teacher-forced PER-TOKEN NLL of the gold answer given selected slots `pkv`.

    Returns np.ndarray [T] of token-level cross-entropy (nats). Mirrors the scoring in
    slot_residual_analysis.score_pkv but with reduction='none'.
    Feeds [LM, prompt, answer[:-1]] so the logit at the last prompt position predicts a_1.
    """
    Lp1 = lm_emb_prompt.size(1)
    T = answer_emb.size(1)
    dec_in = torch.cat([lm_emb_prompt, answer_emb[:, :-1, :]], dim=1) if T > 1 else lm_emb_prompt
    seq = dec_in.size(1)
    pos = torch.arange(end_idx, end_idx + seq, device=device).unsqueeze(0)
    if model.task_config["use_pe"]:
        out = model.decoder(inputs_embeds=dec_in, position_ids=pos, past_key_values=pkv)
    else:
        out = model.decoder(inputs_embeds=dec_in, past_key_values=pkv)
    start = Lp1 - 1
    ans_logits = out.logits[:, start:start + T, :]            # [1, T, V]
    ce = torch.nn.functional.cross_entropy(
        ans_logits.reshape(-1, model.vocab_size).float(),
        answer_ids.reshape(-1).to(device), reduction="none")  # [T]
    return ce.detach().cpu().numpy()


def canonical_base_sets(per_chunk, base_k, device):
    """Canonical first base_k anchors per chunk (clamped to m_c). base_k=0 -> all empty."""
    return [torch.arange(min(int(base_k), int(m)), device=device) for m in per_chunk]


def add_one_slot(per_chunk, base_k, policy, device, rng=None, perms=None):
    """Return per-chunk LOCAL index sets = canonical base (base_k) PLUS one extra anchor.

    The extra anchor for chunk c is drawn from remaining = {min(base_k,m_c) .. m_c-1}:
      canonical -> min(remaining)   (== canonical prefix of length base_k+1)
      reverse   -> max(remaining)   (the learned-last anchor)
      shuffled  -> the remaining anchor ranked first by a fixed permutation perms[c]
      random    -> a uniform pick from remaining (rng)
    Chunks already saturated (base_k>=m_c) contribute no extra (set == base).
    """
    out = []
    for c, m in enumerate(per_chunk):
        m = int(m)
        b = min(int(base_k), m)
        base = torch.arange(b, device=device)
        remaining = list(range(b, m))
        if not remaining:
            out.append(base)
            continue
        if policy == "canonical":
            extra = remaining[0]
        elif policy == "reverse":
            extra = remaining[-1]
        elif policy == "shuffled":
            perm = perms[c]                          # 1D long perm of range(m)
            rank = {int(v): i for i, v in enumerate(perm.tolist())}
            extra = min(remaining, key=lambda i: rank[i])
        elif policy == "random":
            extra = remaining[int(torch.randint(len(remaining), (1,), generator=rng, device=device))]
        else:
            raise ValueError(policy)
        out.append(torch.cat([base, torch.tensor([extra], device=device, dtype=torch.long)]))
    return out


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
    sel = [i for i in range(len(examples)) if examples[i]["input_ids"].shape[-1] >= args.min_ctx_len]
    rng_np = np.random.RandomState(args.sample_seed)
    rng_np.shuffle(sel)
    sel = sel[:args.num_samples]
    si, sn = (int(x) for x in args.shard.split("/"))
    sel = sel[si::sn]
    os.makedirs(args.out_dir, exist_ok=True)
    print(f"[INFO] {len(sel)} examples (ctx>={args.min_ctx_len}) mode={args.mode} shard={args.shard}", flush=True)

    g_null = torch.Generator(device=device).manual_seed(args.null_seed)
    g_rand = torch.Generator(device=device).manual_seed(args.null_seed + 777)

    POLICIES = ["canonical", "reverse", "shuffled", "random"]
    # flat accumulators (pooled over sample,s,token)
    acc = {pol: {"base_nll": [], "delta": [], "s": [], "sample": [], "tok": []} for pol in POLICIES}
    verify_max = 0.0
    n_used = 0

    for n, idx in enumerate(sel):
        ex, mt = examples[idx], meta[idx]
        inputs = {"input_ids": ex["input_ids"].unsqueeze(0).to(device),
                  "lm_targets": ex["lm_targets"].unsqueeze(0).to(device)}
        ans_txt = mt["answers"][0]
        ans_ids = torch.tensor(tok(ans_txt, add_special_tokens=False)["input_ids"],
                               dtype=torch.long).unsqueeze(0)
        if ans_ids.numel() == 0:
            continue
        ans_emb = model.decoder.model.embed_tokens(ans_ids.to(device))
        lm_target_emb = model.decoder.model.embed_tokens(inputs["lm_targets"])
        bsz, Lp, D = lm_target_emb.size()
        lm_tok = model.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, D)
        lm_emb_prompt = torch.cat([lm_tok, lm_target_emb], dim=1)

        (_, _, end_idx, _, pkv_full, _, per_chunk) = model.compress(inputs, truncate_k=None)
        max_m = int(max(per_chunk))
        S = min(args.max_steps, max_m)
        # fixed shuffled-null perms (one per chunk), reused across all s
        null_perms = [torch.randperm(int(m), generator=g_null, device=device) for m in per_chunk]
        T = ans_ids.size(1)

        # canonical per-token NLL cache over base_k = 0..S  (base for step s == cache[s-1])
        canon_cache = {}

        def nll_for_sets(sets):
            keep = build_keep_indices_from_sets(per_chunk, sets, device)
            pkv = select_kv(pkv_full, keep)
            return per_token_nll(model, lm_emb_prompt, ans_emb, ans_ids, pkv, end_idx, device)

        for bk in range(0, S + 1):
            canon_cache[bk] = nll_for_sets(canonical_base_sets(per_chunk, bk, device))

        if args.mode == "verify":
            for s in range(1, S + 1):
                a = canon_cache[s]
                (_, _, e2, _, pkv_t, _, _) = model.compress(inputs, truncate_k=s)
                b = per_token_nll(model, lm_emb_prompt, ans_emb, ans_ids, pkv_t, e2, device)
                verify_max = max(verify_max, float(np.max(np.abs(a - b))))
            # empty-pkv (base_k=0) just needs to not crash; recorded above
            if n < 3:
                print(f"  [verify] ex{idx} per_chunk={list(map(int,per_chunk))} max|Δ|={verify_max:.2e}", flush=True)
            continue

        for s in range(1, S + 1):
            base_nll = canon_cache[s - 1]                       # [T]  (canonical prefix s-1)
            for pol in POLICIES:
                if pol == "canonical":
                    cand_nll = canon_cache[s]
                elif pol == "random":
                    # average per-token NLL over n_random draws -> smooth band null
                    accs = np.zeros(T, dtype=np.float64)
                    for _ in range(args.n_random):
                        sets = add_one_slot(per_chunk, s - 1, "random", device, rng=g_rand)
                        accs += nll_for_sets(sets)
                    cand_nll = accs / args.n_random
                else:
                    sets = add_one_slot(per_chunk, s - 1, pol, device, perms=null_perms)
                    cand_nll = nll_for_sets(sets)
                delta = base_nll - cand_nll                      # [T] + = improvement
                acc[pol]["base_nll"].append(base_nll.astype(np.float32))
                acc[pol]["delta"].append(delta.astype(np.float32))
                acc[pol]["s"].append(np.full(T, s, dtype=np.int16))
                acc[pol]["sample"].append(np.full(T, idx, dtype=np.int32))
                acc[pol]["tok"].append(np.arange(T, dtype=np.int16))
        n_used += 1
        if (n + 1) % 10 == 0:
            print(f"  [functional] {n+1}/{len(sel)} (used={n_used})", flush=True)

    if args.mode == "verify":
        ok = verify_max < 1e-3
        print(f"[VERIFY] max|Δ per-token nll| canonical-vs-truncate = {verify_max:.3e} -> {'PASS' if ok else 'FAIL'}", flush=True)
        sys.exit(0 if ok else 1)

    out = {}
    for pol in POLICIES:
        for key in ["base_nll", "delta", "s", "sample", "tok"]:
            arrs = acc[pol][key]
            out[f"{pol}__{key}"] = np.concatenate(arrs) if arrs else np.array([])
    out["meta__n_used"] = np.array([n_used])
    out["meta__n_random"] = np.array([args.n_random])
    fn = os.path.join(args.out_dir, f"tokattr_shard{args.shard.replace('/', 'of')}.npz")
    np.savez_compressed(fn, **out)
    print(f"[INFO] {n_used} samples -> {fn}", flush=True)


if __name__ == "__main__":
    main()
