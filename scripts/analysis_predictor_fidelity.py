#!/usr/bin/env python
"""KPredictor cross-domain fidelity analysis (IID vs OOD predictor extrapolation).

Consumes eval_v5_pareto.py --include_oracle outputs (pareto_*_rank*.json), which
for each ε contain two labels:
  - "eps=<E>"        : KPredictor's per-chunk k  (k_pred), per sample id + dataset
  - "oracle_eps=<E>" : loss-domain oracle per-chunk k  (k_oracle), same ids
Each record: {id, dataset, k_per_chunk:[...], ...}.

Directly measures whether the predictor reproduces the oracle per-chunk budget
rule, split into IID (MRQA in-domain 6) vs OOD (MRQA out-of-domain 6) buckets.

Key identity: per-chunk LM loss is monotone-decreasing in k and
  oracle_k = min{k in K_choices : L_k - L_full < eps}.
Therefore the predictor's allocation SATISFIES the eps tolerance on a chunk iff
  k_pred >= k_oracle   (no loss matrix needed).
  under-spend (lossy, violates tolerance) <=> k_pred <  k_oracle
  over-spend  (wasteful, safe)            <=> k_pred >  k_oracle

Metrics per (bucket, eps):
  - k_MAE         = mean |k_pred - k_oracle|              (chunk level; main)
  - signed_bias   = mean (k_pred - k_oracle)             (<0 = systematic under-alloc)
  - constraint_sat= P(k_pred >= k_oracle)
  - under_rate / over_rate
  - spearman_in   = within-sample Spearman rho(k_pred, k_oracle), averaged over
                    samples with >=2 chunks and oracle variation (+ n)
  - mean_k_pred / mean_k_oracle (domain-level calibration)
  - uniform_null_MAE = per-sample flat allocation at matched mean(k_pred): every
                    chunk gets round(sample mean k_pred). Isolates the value of
                    per-chunk differentiation (vs just matching the budget level).
Bootstrap CI on (OOD k_MAE - IID k_MAE) by resampling samples.

CPU-only. Reads JSON, writes JSON + markdown table.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np

IID_SETS = {"SQuAD", "TriviaQA-web", "NewsQA", "SearchQA", "HotpotQA",
            "NaturalQuestionsShort"}
OOD_SETS = {"BioASQ", "DROP", "DuoRC.ParaphraseRC", "RACE",
            "RelationExtraction", "TextbookQA"}


def bucket_of(dataset):
    if dataset in IID_SETS:
        return "IID"
    if dataset in OOD_SETS:
        return "OOD"
    return None  # unknown / long-context — skipped


def rankdata(a):
    """Average-rank (ties) without scipy."""
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
        return None  # no variation -> rho undefined
    return float(np.corrcoef(rx, ry)[0, 1])


def load_records(glob_pat):
    """Return {eps(float): {id: {'pred':[..], 'oracle':[..], 'dataset':str}}}."""
    files = sorted(glob.glob(glob_pat))
    assert files, f"no files match {glob_pat}"
    out = defaultdict(lambda: defaultdict(dict))
    for f in files:
        d = json.load(open(f))
        for label, recs in d.items():
            if label.startswith("oracle_eps="):
                eps = float(label.split("=")[1]); kind = "oracle"
            elif label.startswith("eps="):
                eps = float(label.split("=")[1]); kind = "pred"
            else:
                continue
            for r in recs:
                slot = out[eps][r["id"]]
                slot[kind] = r["k_per_chunk"]
                slot["dataset"] = r["dataset"]
    return out, files


def per_sample_pairs(eps_map):
    """Yield (bucket, dataset, k_pred_arr, k_oracle_arr) for aligned samples."""
    for sid, slot in eps_map.items():
        if "pred" not in slot or "oracle" not in slot:
            continue
        kp, ko = slot["pred"], slot["oracle"]
        if len(kp) != len(ko) or len(kp) == 0:
            continue
        b = bucket_of(slot["dataset"])
        if b is None:
            continue
        yield b, slot["dataset"], np.array(kp, float), np.array(ko, float)


def analyze(eps_map):
    """Return per-bucket metrics + per-sample arrays for bootstrap."""
    # accumulate chunk-level and sample-level stats per bucket
    chunk = defaultdict(lambda: {"abs": [], "signed": [], "ge": [], "gt": [], "lt": []})
    rho = defaultdict(list)
    ds_meank = defaultdict(lambda: defaultdict(lambda: {"pred": [], "oracle": []}))
    sample_mae = defaultdict(list)  # for bootstrap: per-sample mean |Δ|
    # stash per-chunk pred/oracle per bucket for the global-uniform null
    pool = defaultdict(lambda: {"pred": [], "oracle": []})

    for b, ds, kp, ko in per_sample_pairs(eps_map):
        diff = kp - ko
        chunk[b]["abs"].extend(np.abs(diff).tolist())
        chunk[b]["signed"].extend(diff.tolist())
        chunk[b]["ge"].extend((kp >= ko).tolist())
        chunk[b]["gt"].extend((kp > ko).tolist())
        chunk[b]["lt"].extend((kp < ko).tolist())
        sample_mae[b].append(float(np.mean(np.abs(diff))))
        r = spearman(kp, ko)
        if r is not None:
            rho[b].append(r)
        ds_meank[b][ds]["pred"].append(float(kp.mean()))
        ds_meank[b][ds]["oracle"].append(float(ko.mean()))
        pool[b]["pred"].extend(kp.tolist())
        pool[b]["oracle"].extend(ko.tolist())

    res = {}
    for b in ("IID", "OOD"):
        c = chunk[b]
        if not c["abs"]:
            continue
        ora = np.array(pool[b]["oracle"])
        prd = np.array(pool[b]["pred"])
        # chunk-level (pooled over all chunks) agreement between k_pred and k_oracle
        chunk_pearson = float(np.corrcoef(prd, ora)[0, 1])
        chunk_spearman = spearman(prd, ora)
        # mean absolute relative error |k_pred - k_oracle| / k_oracle (oracle>=1)
        rel_err = float(np.mean(np.abs(prd - ora) / np.maximum(ora, 1)))
        # global-uniform null: ONE constant budget for every chunk in the bucket,
        # matched to the predictor's overall mean. Tests whether the predictor's
        # sample/dataset-level allocation beats one-size-fits-all. (Per-sample null
        # is degenerate here: most MRQA samples are single-chunk.)
        g_uni = round(float(np.mean(pool[b]["pred"])))
        # per-dataset mean-k arrays for cross-dataset calibration corr
        dm = {ds: {"pred": float(np.mean(v["pred"])),
                   "oracle": float(np.mean(v["oracle"])),
                   "n": len(v["pred"])}
              for ds, v in sorted(ds_meank[b].items())}
        # per-dataset aggregate error: avg over datasets of |k̄_pred - k̄_oracle|
        _da = [abs(v["pred"] - v["oracle"]) for v in dm.values()]
        _dr = [abs(v["pred"] - v["oracle"]) / max(v["oracle"], 1e-9) for v in dm.values()]
        agg_abs_err = float(np.mean(_da)) if _da else None
        agg_rel_err = float(np.mean(_dr)) if _dr else None
        res[b] = {
            "n_chunks": len(c["abs"]),
            "n_samples": len(sample_mae[b]),
            "k_MAE": float(np.mean(c["abs"])),
            "rel_err": rel_err,
            "chunk_pearson": chunk_pearson,
            "chunk_spearman": chunk_spearman,
            "agg_abs_err": agg_abs_err,
            "agg_rel_err": agg_rel_err,
            "signed_bias": float(np.mean(c["signed"])),
            "constraint_sat": float(np.mean(c["ge"])),
            "over_rate": float(np.mean(c["gt"])),
            "under_rate": float(np.mean(c["lt"])),
            "spearman_in_mean": (float(np.mean(rho[b])) if rho[b] else None),
            "spearman_in_n": len(rho[b]),
            "global_uniform_MAE": float(np.mean(np.abs(g_uni - ora))),
            "global_uniform_k": g_uni,
            "per_dataset_meank": dm,
        }
    # cross-dataset calibration: corr of per-dataset mean_k_pred vs mean_k_oracle
    # over ALL 12 datasets (IID+OOD pooled) — OOD points flagged for the figure.
    all_dm = {}
    for b in ("IID", "OOD"):
        for ds, v in res.get(b, {}).get("per_dataset_meank", {}).items():
            all_dm[ds] = {**v, "bucket": b}
    if len(all_dm) >= 3:
        xp = [all_dm[d]["pred"] for d in all_dm]
        xo = [all_dm[d]["oracle"] for d in all_dm]
        res["_calib"] = {
            "pearson": float(np.corrcoef(xp, xo)[0, 1]),
            "spearman": spearman(xp, xo),
            "n_datasets": len(all_dm),
            "points": all_dm,
        }
    return res, sample_mae


def bootstrap_ood_minus_iid(sample_mae, n_boot=2000, seed=0):
    """CI on mean(OOD per-sample MAE) - mean(IID per-sample MAE)."""
    iid = np.array(sample_mae.get("IID", []))
    ood = np.array(sample_mae.get("OOD", []))
    if len(iid) == 0 or len(ood) == 0:
        return None
    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        bi = rng.choice(iid, len(iid), replace=True)
        bo = rng.choice(ood, len(ood), replace=True)
        diffs[i] = bo.mean() - bi.mean()
    return {
        "point": float(ood.mean() - iid.mean()),
        "ci95": [float(np.percentile(diffs, 2.5)),
                 float(np.percentile(diffs, 97.5))],
    }


def md_table(all_res):
    eps_sorted = sorted(all_res.keys())
    lines = []
    lines.append("### KPredictor fidelity to loss-domain oracle (IID vs OOD)\n")
    hdr = ("| ε | bucket | n_chunk | k-MAE | global-uniform MAE | signed bias | "
           "constraint-sat | under-rate | Spearman ρ (n) | mean k_pred / k_oracle |")
    sep = "|---|---|---|---|---|---|---|---|---|---|"
    lines += [hdr, sep]
    for eps in eps_sorted:
        for b in ("IID", "OOD"):
            r = all_res[eps].get("res", {}).get(b)
            if not r:
                continue
            rho = (f"{r['spearman_in_mean']:.3f} ({r['spearman_in_n']})"
                   if r["spearman_in_mean"] is not None else "n/a")
            lines.append(
                f"| {eps} | {b} | {r['n_chunks']} | {r['k_MAE']:.2f} | "
                f"{r['global_uniform_MAE']:.2f} (k={r['global_uniform_k']}) | "
                f"{r['signed_bias']:+.2f} | "
                f"{r['constraint_sat']:.2%} | {r['under_rate']:.2%} | {rho} | "
                f"{np.mean([d['pred'] for d in r['per_dataset_meank'].values()]):.2f} / "
                f"{np.mean([d['oracle'] for d in r['per_dataset_meank'].values()]):.2f} |")
        bs = all_res[eps].get("bootstrap")
        if bs:
            lines.append(f"|   | ΔMAE(OOD−IID) | | {bs['point']:+.2f} "
                         f"(95% CI [{bs['ci95'][0]:+.2f}, {bs['ci95'][1]:+.2f}]) "
                         "| | | | | | |")
        calib = all_res[eps].get("res", {}).get("_calib")
        if calib:
            lines.append(f"|   | cross-dataset calib (12 ds) | | "
                         f"Pearson {calib['pearson']:.3f} / Spearman {calib['spearman']:.3f} "
                         "| | | | | | |")
    return "\n".join(lines)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--glob", required=True,
                   help="glob for pareto_*_rank*.json (oracle-mode dumps)")
    p.add_argument("--out_json", required=True)
    p.add_argument("--out_md", required=True)
    p.add_argument("--n_boot", type=int, default=2000)
    args = p.parse_args()

    by_eps, files = load_records(args.glob)
    all_res = {}
    for eps, eps_map in by_eps.items():
        res, sample_mae = analyze(eps_map)
        bs = bootstrap_ood_minus_iid(sample_mae, n_boot=args.n_boot)
        all_res[eps] = {"res": res, "bootstrap": bs}

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    json.dump({"source_files": files, "by_eps": all_res},
              open(args.out_json, "w"), indent=2)
    md = md_table(all_res)
    open(args.out_md, "w").write(md + "\n")
    print(md)
    print(f"\n[written] {args.out_json}\n[written] {args.out_md}")


if __name__ == "__main__":
    main()
