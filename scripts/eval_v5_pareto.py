"""MANTA v5 — Pareto evaluator (KPredictor + ε), with optional oracle upper bound.

For each eval sample, run compress() ONCE then loop ε ∈ EPS_LIST:
  • KPredictor branch: call predict_k(epsilon) → truncate KV → generate
  • Oracle branch (--include_oracle): pre-cache per-chunk per-k LM losses,
    derive oracle k_per_chunk for each ε from cached losses → truncate KV → generate.

Produces per-ε (F1, EM, mean_k, comp_rate) and per-dataset breakdowns; with
oracle on, also yields the upper-bound Pareto curve for paper Figure X.
Designed for 8×H100 data-parallel via mp.spawn.
"""
import argparse
import json
import os
import random
import re
import string
import sys
import time
from collections import Counter, defaultdict

import numpy as np
import torch
import torch.multiprocessing as mp
try:
    mp.set_sharing_strategy("file_descriptor")
except (RuntimeError, AssertionError):
    mp.set_sharing_strategy("file_system")
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modeling import (
    get_model, load_adapter,
    extract_full_kv, _build_keep_indices,
    truncate_single_chunk_kv, compute_oracle_k_per_chunk,
)


# ----------------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------------
def normalize_answer(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def token_f1(pred: str, gold: str) -> float:
    pt = normalize_answer(pred).split()
    gt = normalize_answer(gold).split()
    if not gt:
        return float(not pt)
    if not pt:
        return 0.0
    common = Counter(pt) & Counter(gt)
    nc = sum(common.values())
    if nc == 0:
        return 0.0
    p = nc / len(pt); r = nc / len(gt)
    return 2 * p * r / (p + r)


def exact_match(pred: str, gold: str) -> float:
    return float(normalize_answer(pred) == normalize_answer(gold))


# ----------------------------------------------------------------------------
# Args
# ----------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    p.add_argument("--eval_data_path", required=True)
    p.add_argument("--eval_meta_path", required=True)
    p.add_argument("--num_samples", type=int, default=1000)
    p.add_argument("--idx_offset", type=int, default=0, help="shard offset for cross-host eval")
    p.add_argument("--idx_stride", type=int, default=1, help="shard stride for cross-host eval")
    p.add_argument("--idx_start", type=int, default=-1, help="shard start index (inclusive); -1=disabled")
    p.add_argument("--idx_end", type=int, default=-1, help="shard end index (exclusive); -1=disabled")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_gen", type=int, default=128)
    p.add_argument("--adapter_name", default="instruction_adapter.pt")
    p.add_argument("--output_name", default="pareto_v5")
    p.add_argument("--eps_list", default="0.01,0.05,0.1,0.2",
                   help="comma-separated ε values for KPredictor")
    # default=None → predict_k uses the pos_weight-calibrated threshold w/(1+w).
    # Pass an explicit float only to override calibration.
    p.add_argument("--predictor_threshold", type=float, default=None)
    p.add_argument("--include_oracle", action="store_true",
                   help="also compute per-chunk oracle k for each ε (slower)")
    p.add_argument("--fidelity_only", action="store_true",
                   help="skip answer generation; dump only k_pred/k_oracle per chunk "
                        "(predictor-fidelity analysis — ~20x cheaper, enables full-58K oracle)")
    p.add_argument("--k_choices", default="1,2,4,8,16,34",
                   help="K choices for oracle search (must match training)")
    return p.parse_args()


# ----------------------------------------------------------------------------
# Generation
# ----------------------------------------------------------------------------
@torch.no_grad()
def generate_answer(model, encoder_past_key_values, compress_token_ids, end_idx,
                    lm_targets, max_gen, eos_id):
    lm_target_emb = model.decoder.model.embed_tokens(lm_targets)
    bsz, seq_len, emb_size = lm_target_emb.size()
    expand_lm_token = model.special_tokens[1:2].unsqueeze(0).expand(bsz, 1, emb_size)
    lm_emb = torch.cat([expand_lm_token, lm_target_emb], dim=1)
    pos_ids = torch.arange(end_idx, end_idx + seq_len + 1,
                           device=lm_target_emb.device).unsqueeze(0)
    pkv = encoder_past_key_values
    next_emb = lm_emb.clone()
    next_pos = pos_ids.clone()
    out_ids = []
    use_pe = model.task_config["use_pe"]
    for _ in range(max_gen):
        if use_pe:
            out = model.decoder(position_ids=next_pos, inputs_embeds=next_emb,
                                past_key_values=pkv, use_cache=True)
        else:
            out = model.decoder(inputs_embeds=next_emb,
                                past_key_values=pkv, use_cache=True)
        pkv = out.past_key_values
        tok = torch.argmax(out.logits[:, -1], dim=-1)
        tid = int(tok.item())
        out_ids.append(tid)
        if tid == eos_id:
            break
        next_emb = model.decoder.model.embed_tokens(tok).unsqueeze(1).to(lm_target_emb.device)
        next_pos = next_pos[:, -1:] + 1
    return out_ids


def _apply_per_chunk_truncation(encoder_past_key_values, compress_token_ids,
                                per_chunk_mem_sizes, k_per_chunk, device):
    keep_idx = _build_keep_indices(per_chunk_mem_sizes, k_per_chunk, device)
    pkv = tuple(
        (K.index_select(2, keep_idx), V.index_select(2, keep_idx))
        for (K, V) in encoder_past_key_values
    )
    ctok = compress_token_ids.index_select(1, keep_idx)
    return pkv, ctok, keep_idx


# ----------------------------------------------------------------------------
# Per-sample, multi-ε evaluation
# ----------------------------------------------------------------------------
@torch.no_grad()
def eval_sample_multi_eps(model, inputs, eps_list, max_gen, predictor_threshold,
                          include_oracle, k_choices, fidelity_only=False):
    """Returns dict[label] = dict(gen_ids, k_per_chunk, total_kept, total_anchors).
    label ∈ {eps=X (KPredictor), oracle_eps=X (oracle)} for each ε in eps_list.
    """
    # 1. Compress once (no truncation)
    (compress_token_ids, _ctok, end_idx,
     _mlh, encoder_past_key_values, _ems,
     per_chunk_mem_sizes) = model.compress(inputs, truncate_k=None)

    anchor_kv = extract_full_kv(encoder_past_key_values)
    device = encoder_past_key_values[0][0].device
    total_anchors = sum(per_chunk_mem_sizes)
    eos_id = model.tokenizer.eos_token_id

    # 2. Pre-cache per-chunk per-k LM losses for oracle (one pass for all ε)
    losses_per_chunk_per_k = None
    L_full = None
    if include_oracle:
        N_chunks = len(per_chunk_mem_sizes)
        K_max = max(k_choices)
        # SFT-mode oracle needs answer-token supervision (instruction_target).
        # Eval sets carry prompt-only lm_targets, so the worker builds
        # `_oracle_loss_inputs` (prompt+gold-answer extended targets) — used
        # ONLY for loss caching here, never for generation (no leakage).
        loss_inputs = inputs.get("_oracle_loss_inputs", inputs)
        L_full = model._decoder_lm_loss_for_pkv(
            loss_inputs, encoder_past_key_values, end_idx, reduction='none'
        )                                                          # [B]
        losses_per_chunk_per_k = []
        for c in range(N_chunks):
            L_ck = {K_max: L_full}
            for k in k_choices:
                if k == K_max:
                    continue
                pkv = truncate_single_chunk_kv(
                    encoder_past_key_values, per_chunk_mem_sizes, c, k
                )
                L_ck[k] = model._decoder_lm_loss_for_pkv(
                    loss_inputs, pkv, end_idx, reduction='none'
                )
            losses_per_chunk_per_k.append(L_ck)

    results = {}
    for eps in eps_list:
        # --- KPredictor branch ----------------------------------------------
        eps_t = torch.tensor([float(eps)], device=device)
        _, k_pred = model.k_predictor.predict_k(
            anchor_kv, per_chunk_mem_sizes, eps_t, threshold=predictor_threshold
        )
        pkv_p, ctok_p, keep_p = _apply_per_chunk_truncation(
            encoder_past_key_values, compress_token_ids,
            per_chunk_mem_sizes, k_pred, device
        )
        gen_ids = None if fidelity_only else generate_answer(
            model, pkv_p, ctok_p, end_idx,
            inputs["lm_targets"], max_gen, eos_id,
        )
        results[f"eps={eps}"] = dict(
            gen_ids=gen_ids,
            k_per_chunk=list(k_pred),
            total_kept=int(keep_p.numel()),
            total_anchors=total_anchors,
        )

        # --- Oracle branch --------------------------------------------------
        if include_oracle:
            oracle_list = compute_oracle_k_per_chunk(
                losses_per_chunk_per_k, L_full, eps_t, k_choices
            )                                                       # list of [B] long
            k_oracle = [int(t[0].item()) for t in oracle_list]
            pkv_o, ctok_o, keep_o = _apply_per_chunk_truncation(
                encoder_past_key_values, compress_token_ids,
                per_chunk_mem_sizes, k_oracle, device
            )
            gen_ids_o = None if fidelity_only else generate_answer(
                model, pkv_o, ctok_o, end_idx,
                inputs["lm_targets"], max_gen, eos_id,
            )
            results[f"oracle_eps={eps}"] = dict(
                gen_ids=gen_ids_o,
                k_per_chunk=k_oracle,
                total_kept=int(keep_o.numel()),
                total_anchors=total_anchors,
            )
    return results


# ----------------------------------------------------------------------------
# Worker (one per GPU)
# ----------------------------------------------------------------------------
def evaluate_worker(rank, args, world_size, idx_list,
                    eps_list, k_choices):
    with open(os.path.join(args.work_dir, "output", "config.json")) as f:
        config = json.load(f)
    training_config = config["sft_training_config"]
    task_config = config["sft_task_config"]

    # Each worker loads the eval dataset itself — avoids pickling 67K+ tensors
    # through mp.spawn args (every shared tensor consumes one mmap region;
    # vm.max_map_count=65530 caps that out on full-eval scale).
    eval_examples = torch.load(args.eval_data_path)
    with open(args.eval_meta_path) as f:
        eval_meta = json.load(f)

    total = len(idx_list)
    per_gpu = total // world_size
    start = rank * per_gpu
    end = start + per_gpu if rank < world_size - 1 else total
    my_idx = idx_list[start:end]
    my_ex = [eval_examples[i] for i in my_idx]
    my_meta = [eval_meta[i] for i in my_idx]
    print(f"[GPU {rank}] {len(my_ex)} samples")

    model = get_model(training_config["model_id"], task_config, rank)
    adapter_path = os.path.join(args.work_dir, "output", args.adapter_name)
    model = load_adapter(model, save_path_and_name=adapter_path)
    model.eval()
    assert getattr(model, "k_predictor", None) is not None, \
        "Model has no k_predictor — was it trained with use_k_predictor=True?"

    tokenizer = AutoTokenizer.from_pretrained(training_config["model_id"])

    labels = []
    for e in eps_list:
        labels.append(f"eps={e}")
        if args.include_oracle:
            labels.append(f"oracle_eps={e}")
    results = {lab: [] for lab in labels}

    for ex, meta in tqdm(zip(my_ex, my_meta), total=len(my_ex),
                         desc=f"GPU{rank}", disable=(rank != 0)):
        inputs = {
            "input_ids": ex["input_ids"].unsqueeze(0).to(f"cuda:{rank}"),
            "lm_targets": ex["lm_targets"].unsqueeze(0).to(f"cuda:{rank}"),
        }
        if args.include_oracle and task_config.get("is_sft", False):
            # Build answer-supervised targets for the oracle loss cache.
            # ext = prompt + gold answer (+eos); instruction_target = ext[1:]
            # with prompt positions masked to -100 (answer-only loss), matching
            # the training-time oracle semantics.
            ans_ids = tokenizer(meta["answers"][0],
                                add_special_tokens=False)["input_ids"]
            if tokenizer.eos_token_id is not None:
                ans_ids = ans_ids + [tokenizer.eos_token_id]
            prompt = ex["lm_targets"]
            ans_t = torch.tensor(ans_ids, dtype=prompt.dtype)
            ext = torch.cat([prompt, ans_t], dim=0)
            tgt = ext[1:].clone()
            tgt[: prompt.numel() - 1] = -100
            inputs["_oracle_loss_inputs"] = {
                "input_ids": inputs["input_ids"],
                "lm_targets": ext.unsqueeze(0).to(f"cuda:{rank}"),
                "instruction_target": tgt.unsqueeze(0).to(f"cuda:{rank}"),
            }
        try:
            per_eps = eval_sample_multi_eps(
                model, inputs, eps_list, args.max_gen,
                args.predictor_threshold, args.include_oracle, k_choices,
                fidelity_only=args.fidelity_only,
            )
        except Exception as e:
            print(f"[GPU {rank}] sample {meta.get('id')} failed: {e}")
            continue

        for lab in labels:
            if lab not in per_eps:
                continue
            r = per_eps[lab]
            if r["gen_ids"] is None:        # fidelity_only: no generation
                gen_text, best_f1, best_em = "", None, None
            else:
                gen_text = tokenizer.decode(r["gen_ids"], skip_special_tokens=True).strip()
                best_f1 = max(token_f1(gen_text, ans) for ans in meta["answers"])
                best_em = max(exact_match(gen_text, ans) for ans in meta["answers"])
            results[lab].append({
                "id": meta["id"],
                "dataset": meta["dataset"],
                "f1": best_f1,
                "em": best_em,
                "generated": gen_text,
                "k_per_chunk": r["k_per_chunk"],
                "total_kept": r["total_kept"],
                "total_anchors": r["total_anchors"],
                "compression_rate": r["total_kept"] / max(1, r["total_anchors"]),
                "mean_k_per_chunk": r["total_kept"] / max(1, len(r["k_per_chunk"])),
            })

    out_path = os.path.join(args.work_dir, "output",
                            f"{args.output_name}_rank{rank}.json")
    with open(out_path, "w") as f:
        json.dump(results, f, ensure_ascii=False)
    print(f"[GPU {rank}] saved {out_path}")


# ----------------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------------
def _summarize(rs):
    f1s = np.array([r["f1"] for r in rs if r["f1"] is not None])  # fidelity_only -> None
    ems = np.array([r["em"] for r in rs if r["em"] is not None])
    kpc = np.array([r["mean_k_per_chunk"] for r in rs])
    cps = np.array([r["compression_rate"] for r in rs])
    within = [np.std(r["k_per_chunk"]) for r in rs if len(r["k_per_chunk"]) > 1]
    return dict(
        n=len(rs),
        f1=(float(f1s.mean()) if len(f1s) else None),
        em=(float(ems.mean()) if len(ems) else None),
        mean_k_per_chunk=float(kpc.mean()),
        compression_rate=float(cps.mean()),
        within_sample_k_std=float(np.mean(within)) if within else 0.0,
    )


def main():
    args = parse_args()
    world_size = torch.cuda.device_count()
    assert world_size > 0, "no GPU visible"
    print(f"[INFO] world_size={world_size}")

    eps_list = [float(e) for e in args.eps_list.split(",")]
    k_choices = [int(k) for k in args.k_choices.split(",")]
    print(f"[INFO] ε sweep: {eps_list}")
    print(f"[INFO] include_oracle = {args.include_oracle}"
          f"  (K_choices={k_choices})" if args.include_oracle else f"[INFO] include_oracle = False")

    # Main process loads ONLY the meta (small JSON). Each worker independently
    # loads the .pt of tensors and slices by idx_list[start:end]. This keeps
    # mp.spawn args tiny (a list of ints) and avoids the vm.max_map_count
    # blow-up that hits when 67K+ tensors are pickled through spawn.
    with open(args.eval_meta_path) as f:
        eval_meta = json.load(f)
    n_total = len(eval_meta)
    print(f"[INFO] {n_total} total samples")

    if 0 < args.num_samples < n_total:
        rng = random.Random(args.seed)
        idx_list = sorted(rng.sample(range(n_total), args.num_samples))
    else:
        idx_list = list(range(n_total))
    if args.idx_start >= 0 and args.idx_end >= 0:
        idx_list = idx_list[args.idx_start:args.idx_end]
        print(f"[INFO] sharded range: [{args.idx_start}, {args.idx_end}) -> {len(idx_list)} samples")
    elif args.idx_stride > 1:
        idx_list = idx_list[args.idx_offset::args.idx_stride]
        print(f"[INFO] sharded: off={args.idx_offset} stride={args.idx_stride} -> {len(idx_list)} samples")
    print(f"[INFO] evaluating {len(idx_list)} samples")

    t0 = time.time()
    if world_size == 1:
        evaluate_worker(0, args, 1, idx_list, eps_list, k_choices)
    else:
        mp.spawn(evaluate_worker,
                 args=(args, world_size, idx_list, eps_list, k_choices),
                 nprocs=world_size, join=True)
    elapsed_min = (time.time() - t0) / 60
    print(f"[INFO] eval done in {elapsed_min:.1f} min")

    labels = []
    for e in eps_list:
        labels.append(f"eps={e}")
        if args.include_oracle:
            labels.append(f"oracle_eps={e}")
    merged = {lab: [] for lab in labels}
    for rank in range(world_size):
        p = os.path.join(args.work_dir, "output",
                         f"{args.output_name}_rank{rank}.json")
        with open(p) as f:
            d = json.load(f)
        for lab in labels:
            merged[lab].extend(d.get(lab, []))

    summary = {"args": vars(args), "elapsed_min": elapsed_min,
               "per_label": {}, "per_label_per_dataset": {}}
    print("\n" + "=" * 90)
    print(f"{'label':>16} {'N':>6} {'F1':>8} {'EM':>8} {'mean_k':>8} {'comp_rate':>10} {'k_std':>8}")
    print("-" * 90)
    for lab in labels:
        rs = merged[lab]
        if not rs:
            continue
        row = _summarize(rs)
        summary["per_label"][lab] = row
        _f1 = "   n/a " if row["f1"] is None else f"{row['f1']:>8.4f}"
        _em = "   n/a " if row["em"] is None else f"{row['em']:>8.4f}"
        print(f"{lab:>16} {row['n']:>6} {_f1} {_em} "
              f"{row['mean_k_per_chunk']:>8.2f} {row['compression_rate']:>10.4f} "
              f"{row['within_sample_k_std']:>8.2f}")
        per_ds = defaultdict(list)
        for r in rs:
            per_ds[r["dataset"]].append(r)
        summary["per_label_per_dataset"][lab] = {
            ds: _summarize(rs_ds) for ds, rs_ds in per_ds.items()
        }
    print("=" * 90)

    out_summary = os.path.join(args.work_dir, "output",
                               f"{args.output_name}_summary.json")
    with open(out_summary, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[INFO] saved {out_summary}")

    # Pareto points: KPredictor + Oracle (if computed) on same axes
    pts = {"k_predictor": [], "oracle": []}
    for e in eps_list:
        kp = summary["per_label"].get(f"eps={e}")
        if kp:
            pts["k_predictor"].append(dict(
                eps=e, mean_k=kp["mean_k_per_chunk"], f1=kp["f1"],
                em=kp["em"], compression_rate=kp["compression_rate"],
            ))
        oc = summary["per_label"].get(f"oracle_eps={e}")
        if oc:
            pts["oracle"].append(dict(
                eps=e, mean_k=oc["mean_k_per_chunk"], f1=oc["f1"],
                em=oc["em"], compression_rate=oc["compression_rate"],
            ))
    pareto_path = os.path.join(args.work_dir, "output",
                               f"{args.output_name}_pareto_points.json")
    with open(pareto_path, "w") as f:
        json.dump(pts, f, indent=2)
    print(f"[INFO] saved {pareto_path}")

    if args.include_oracle:
        # Quick gap report
        print("\n[INFO] KPredictor vs Oracle gap (F1):")
        for e in eps_list:
            kp = summary["per_label"].get(f"eps={e}")
            oc = summary["per_label"].get(f"oracle_eps={e}")
            if kp and oc and kp["f1"] is not None and oc["f1"] is not None:
                print(f"  ε={e:>5}  oracle F1={oc['f1']:.4f}  kp F1={kp['f1']:.4f}  "
                      f"gap={oc['f1']-kp['f1']:+.4f}  "
                      f"(oracle mean_k={oc['mean_k_per_chunk']:.2f}, "
                      f"kp mean_k={kp['mean_k_per_chunk']:.2f})")
            elif kp and oc:
                print(f"  ε={e:>5}  (fidelity_only: no F1)  "
                      f"oracle mean_k={oc['mean_k_per_chunk']:.2f}, "
                      f"kp mean_k={kp['mean_k_per_chunk']:.2f}")


if __name__ == "__main__":
    main()
