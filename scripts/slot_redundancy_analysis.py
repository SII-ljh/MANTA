#!/usr/bin/env python3
"""Slot redundancy analysis: SAC baseline vs PACT.

Measures the *intrinsic redundancy* of the retained compressed KV slots that a
model actually stores in its cache, at budget-matched compression. For each
chunk we take the retained slot matrix S in R^{m x d} (kv-heads * head_dim
flattened) per layer, on K and V separately, and compute:

  1. Effective-rank ratio  erank(S)/m  in (0,1]   -- HIGHER = less redundant.
     erank = exp(H(sigma_bar)), sigma_bar = singular values normalized to sum 1
     (Roy & Vetterli entropy of the spectrum).
  2. Mean off-diagonal |cosine| of the m slot rows -- LOWER = less redundant.

Per chunk: average the metric over all layers. Macro-average over chunks over
samples. We also report a *global* (cross-chunk) erank-ratio by pooling all
total_M slots of a sample, to capture inter-chunk redundancy.

Budget matching: SAC-1B r64/r85/r128 emit 8/6/4 slots/chunk NATIVELY
(mem_size=compress_ratio geometry); PACT-1B v3 (single ckpt, mem_size=32)
truncates to first-k=8/6/4. Identical slots/chunk => any redundancy gap is
representational, not a budget artifact.

Forward-pass only (no generation). Single GPU. Shape-agnostic to layer count /
d_kv (works for Llama-3.2-1B: 16 layers, d_kv=512).
"""
import argparse
import json
import os
import random
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.modeling import get_model, load_adapter, extract_full_kv, _build_keep_indices


# --------------------------- metrics ---------------------------
def effrank_ratio(S: torch.Tensor) -> float:
    """erank(S)/m  in (0,1].  S: [m, d]."""
    m = S.shape[0]
    if m <= 1:
        return 1.0
    s = torch.linalg.svdvals(S.float())
    s = s[s > 1e-12]
    if s.numel() <= 1:
        return 1.0 / m
    p = s / s.sum()
    H = -(p * (p + 1e-12).log()).sum()
    erank = torch.exp(H).item()
    return erank / m


def mean_abs_cos(S: torch.Tensor) -> float:
    """Mean off-diagonal |cosine| of the m rows.  S: [m, d]."""
    m = S.shape[0]
    if m < 2:
        return float("nan")
    Sn = torch.nn.functional.normalize(S.float(), dim=1)
    G = Sn @ Sn.t()
    eye = torch.eye(m, dtype=torch.bool, device=G.device)
    return G[~eye].abs().mean().item()


def chunk_records(kv: torch.Tensor, per_chunk_mem_sizes):
    """kv: [1, L, 2, total_M, d] from extract_full_kv (already truncated).
    Returns list of per-chunk dicts (layer-averaged), one per chunk with m>=2."""
    L = kv.shape[1]
    recs = []
    off = 0
    for m in per_chunk_mem_sizes:
        m = int(m)
        if m >= 2:
            eK = eV = cK = cV = 0.0
            for l in range(L):
                SK = kv[0, l, 0, off:off + m, :]
                SV = kv[0, l, 1, off:off + m, :]
                eK += effrank_ratio(SK)
                eV += effrank_ratio(SV)
                cK += mean_abs_cos(SK)
                cV += mean_abs_cos(SV)
            recs.append({"m": m, "erankK": eK / L, "erankV": eV / L,
                         "cosK": cK / L, "cosV": cV / L})
        off += m
    return recs


def global_erank(kv: torch.Tensor) -> dict:
    """Pool ALL total_M slots (cross-chunk), layer-averaged erank-ratio."""
    L, M = kv.shape[1], kv.shape[3]
    if M < 2:
        return {"erankK_global": 1.0, "erankV_global": 1.0, "M": M}
    eK = eV = 0.0
    for l in range(L):
        eK += effrank_ratio(kv[0, l, 0])
        eV += effrank_ratio(kv[0, l, 1])
    return {"erankK_global": eK / L, "erankV_global": eV / L, "M": M}


# --------------------------- self-test ---------------------------
def selftest():
    g = torch.Generator().manual_seed(0)
    rnd = torch.randn(8, 512, generator=g)
    dup = rnd[:1].repeat(8, 1)
    print("[selftest] random[8,512]  erank/m=%.3f (expect ~1)  cos=%.3f (expect ~0)"
          % (effrank_ratio(rnd), mean_abs_cos(rnd)))
    print("[selftest] dup[8,512]     erank/m=%.3f (expect %.3f=1/8)  cos=%.3f (expect ~1)"
          % (effrank_ratio(dup), 1.0 / 8, mean_abs_cos(dup)))


# --------------------------- model cache ---------------------------
_MODEL_CACHE = {}


def get_cached_model(work_dir, adapter_name, device_rank):
    key = work_dir
    if key in _MODEL_CACHE:
        return _MODEL_CACHE[key]
    with open(os.path.join(work_dir, "output", "config.json")) as f:
        config = json.load(f)
    tc = config["sft_task_config"]
    trc = config["sft_training_config"]
    model = get_model(trc["model_id"], tc, device_rank)
    model = load_adapter(model, save_path_and_name=os.path.join(work_dir, "output", adapter_name))
    model.eval()
    _MODEL_CACHE[key] = (model, tc)
    return model, tc


# --------------------------- per-sample slot extraction ---------------------------
@torch.no_grad()
def slots_for_sample(model, ex, device, mode, k=None, epsilon=None):
    """Return (kv[1,L,2,M,d], per_chunk_mem_sizes) of the RETAINED slots."""
    inputs = {"input_ids": ex["input_ids"].unsqueeze(0).to(device)}
    if mode == "fixed":
        (_, _, _, _, enc_pkv, _, pcms) = model.compress(inputs, truncate_k=k)
        kv = extract_full_kv(enc_pkv)
        return kv, pcms
    elif mode == "adaptive":
        (_, _, _, _, enc_pkv, _, pcms) = model.compress(inputs, truncate_k=None)
        anchor_kv = extract_full_kv(enc_pkv)
        B = anchor_kv.shape[0]
        eps_t = torch.tensor([float(epsilon)] * B, device=device)
        _, k_b0 = model.k_predictor.predict_k(anchor_kv, pcms, eps_t)
        keep_idx = _build_keep_indices(pcms, k_b0, device)
        enc_pkv = tuple((K.index_select(2, keep_idx), V.index_select(2, keep_idx))
                        for (K, V) in enc_pkv)
        new_pcms = [min(int(kk), int(mm)) for kk, mm in zip(k_b0, pcms)]
        kv = extract_full_kv(enc_pkv)
        return kv, new_pcms
    else:
        raise ValueError(mode)


def calibrate_epsilon(model, examples, device, target_k, eps_grid, n_cal):
    """Pick eps whose realized mean_k over n_cal samples is closest to target_k."""
    best = None
    for eps in eps_grid:
        ks = []
        for ex in examples[:n_cal]:
            _, pcms = slots_for_sample(model, ex, device, "adaptive", epsilon=eps)
            ks.extend(pcms)
        mk = sum(ks) / max(1, len(ks))
        print(f"[calib] eps={eps}  mean_k={mk:.3f}  (target {target_k})")
        if best is None or abs(mk - target_k) < abs(best[1] - target_k):
            best = (eps, mk)
    print(f"[calib] chosen eps={best[0]}  realized mean_k={best[1]:.3f}")
    return best[0]


# --------------------------- aggregation ---------------------------
def aggregate(records, global_recs):
    import statistics as st
    out = {}
    for field in ["erankK", "erankV", "cosK", "cosV"]:
        vals = [r[field] for r in records if r[field] == r[field]]  # drop nan
        out[field + "_mean"] = sum(vals) / len(vals) if vals else float("nan")
        out[field + "_std"] = st.pstdev(vals) if len(vals) > 1 else 0.0
    out["n_chunks"] = len(records)
    out["mean_m"] = sum(r["m"] for r in records) / len(records) if records else 0.0
    for field in ["erankK_global", "erankV_global"]:
        vals = [g[field] for g in global_recs]
        out[field + "_mean"] = sum(vals) / len(vals) if vals else float("nan")
    out["mean_total_M"] = sum(g["M"] for g in global_recs) / len(global_recs) if global_recs else 0.0
    out["n_samples"] = len(global_recs)
    return out


# --------------------------- main ---------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--eval_data_path", default="data/mrqa_eval_full/qa_eval.pt")
    ap.add_argument("--eval_meta_path", default="data/mrqa_eval_full/qa_eval_meta.json")
    ap.add_argument("--pact_dir", default="experiment/pact_v5_mrqa_dyneps_k32_1B_v3")
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--adaptive_target_k", type=float, default=8.0)
    ap.add_argument("--adaptive_eps", type=float, default=None, help="skip calibration if set")
    ap.add_argument("--eps_grid", default="0.001,0.01,0.05,0.1")
    ap.add_argument("--n_cal", type=int, default=100)
    ap.add_argument("--out_dir", default="experiment/slot_redundancy")
    ap.add_argument("--adapter_name", default="instruction_adapter.pt")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    os.chdir(args.root)
    selftest()
    if args.selftest:
        return

    device = "cuda:0"
    os.makedirs(args.out_dir, exist_ok=True)

    # deterministic subsample (same convention as eval_k_sweep main)
    with open(args.eval_meta_path) as f:
        meta = json.load(f)
    n_total = len(meta)
    rng = random.Random(args.seed)
    idx_list = sorted(rng.sample(range(n_total), min(args.n, n_total)))
    print(f"[INFO] {n_total} total, using {len(idx_list)} samples (seed={args.seed})")
    eval_examples = torch.load(args.eval_data_path)
    examples = [eval_examples[i] for i in idx_list]

    # settings: (label, ratio, work_dir, mode, k)
    settings = [
        ("SAC_r64",  "64x",  "experiment/sac_1b_mrqa_r64",  "fixed", None),  # native 8
        ("SAC_r85",  "85x",  "experiment/sac_1b_mrqa_r85",  "fixed", None),  # native 6
        ("SAC_r128", "128x", "experiment/sac_1b_mrqa_r128", "fixed", None),  # native 4
        ("PACT_k8",  "64x",  args.pact_dir, "fixed", 8),
        ("PACT_k6",  "85x",  args.pact_dir, "fixed", 6),
        ("PACT_k4",  "128x", args.pact_dir, "fixed", 4),
        ("PACT_adaptive", "~64x", args.pact_dir, "adaptive", None),
    ]

    # calibrate adaptive eps once (uses PACT model)
    pact_model, _ = get_cached_model(args.pact_dir, args.adapter_name, 0)
    if args.adaptive_eps is not None:
        chosen_eps = args.adaptive_eps
        print(f"[calib] using provided eps={chosen_eps}")
    else:
        eps_grid = [float(x) for x in args.eps_grid.split(",")]
        chosen_eps = calibrate_epsilon(pact_model, examples, device,
                                       args.adaptive_target_k, eps_grid, args.n_cal)

    summary = {}
    for label, ratio, work_dir, mode, k in settings:
        t0 = time.time()
        model, _ = get_cached_model(work_dir, args.adapter_name, 0)
        eps = chosen_eps if mode == "adaptive" else None
        records, global_recs = [], []
        for ex in examples:
            kv, pcms = slots_for_sample(model, ex, device, mode, k=k, epsilon=eps)
            records.extend(chunk_records(kv, pcms))
            global_recs.append(global_erank(kv))
        agg = aggregate(records, global_recs)
        agg.update({"label": label, "ratio": ratio, "mode": mode, "k": k,
                    "epsilon": eps, "work_dir": work_dir,
                    "minutes": (time.time() - t0) / 60})
        summary[label] = agg
        with open(os.path.join(args.out_dir, f"redundancy_{label}.json"), "w") as f:
            json.dump(agg, f, indent=2)
        print(f"[{label}] erankK={agg['erankK_mean']:.3f} erankV={agg['erankV_mean']:.3f} "
              f"cosK={agg['cosK_mean']:.3f} cosV={agg['cosV_mean']:.3f} "
              f"mean_m={agg['mean_m']:.2f} n_chunks={agg['n_chunks']} ({agg['minutes']:.1f}m)")

    summary["_meta"] = {"n_samples": len(examples), "seed": args.seed,
                        "chosen_eps": chosen_eps, "eval_data": args.eval_data_path}
    with open(os.path.join(args.out_dir, "redundancy_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # markdown table
    lines = ["| setting | ratio | mean_m | erank/m K↑ | erank/m V↑ | cos K↓ | cos V↓ | erank/m K(global) |",
             "|---|---|---|---|---|---|---|---|"]
    for label, _, _, _, _ in settings:
        a = summary[label]
        lines.append(f"| {label} | {a['ratio']} | {a['mean_m']:.2f} | "
                     f"{a['erankK_mean']:.3f}±{a['erankK_std']:.3f} | "
                     f"{a['erankV_mean']:.3f}±{a['erankV_std']:.3f} | "
                     f"{a['cosK_mean']:.3f} | {a['cosV_mean']:.3f} | "
                     f"{a['erankK_global_mean']:.3f} |")
    table = "\n".join(lines)
    with open(os.path.join(args.out_dir, "redundancy_table.md"), "w") as f:
        f.write(table + "\n")
    print("\n" + table)

    open(os.path.join(args.out_dir, ".DONE"), "w").close()
    print("\n[DONE]")


if __name__ == "__main__":
    main()
