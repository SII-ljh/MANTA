#!/usr/bin/env python3
"""Paired bootstrap 95% CIs for MANTA-vs-baseline margins on full-MRQA (n=67854).

Pairings (compression-matched, v4@20k main ckpt):
  MANTA eps=0.05 vs SAC k8 (r64), 500xC r64, ICAE r64(+fs2024)
  MANTA eps=0.1  vs SAC k6 (r85), 500xC r85, ICAE r85
  MANTA eps=0.2  vs SAC k4 (r128), 500xC r128, ICAE r128
  MANTA eps=0.3  vs SAC k3 (r170), 500xC r170, ICAE r170

Metric: SQuAD-normalized token-F1 and EM (max-over-refs), identical to
scripts/eval_k_sweep.py / rescore_tokenf1_from_generated.py.
Aggregation: per-dataset mean -> macro over 6 IID / 6 OOD datasets (paper protocol).
Bootstrap: stratified per-dataset resampling (multinomial counts), B=10000,
shared resample draws across all methods => paired.
Also: per-sample mean-k percentiles for eps=0.05 / 0.1.
CPU only. Read-only on project files; writes result JSON to /tmp.
"""
import os
import json, glob, re, string, sys
from collections import Counter, defaultdict
import numpy as np

ROOT = os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
B = 10000
BLOCK = 1000
SEED = 20260706

IID_PREFIX = ("SQuAD", "NewsQA", "TriviaQA", "SearchQA", "HotpotQA", "NaturalQuestions")
def is_iid(ds): return any(ds.startswith(p) for p in IID_PREFIX)

# ---- identical scorer to eval_k_sweep.py ----
def normalize_answer(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

def token_f1(pred, gold):
    p = normalize_answer(pred).split(); g = normalize_answer(gold).split()
    if not g: return float(not p)
    if not p: return 0.0
    c = Counter(p) & Counter(g); nc = sum(c.values())
    if nc == 0: return 0.0
    pr = nc/len(p); rc = nc/len(g)
    return 2*pr*rc/(pr+rc)

def exact_match(pred, gold):
    return float(normalize_answer(pred) == normalize_answer(gold))

# ---- gold answers ----
meta = json.load(open(f"{ROOT}/data/mrqa_eval_full/qa_eval_meta.json"))
N = len(meta)
answers = [None]*N
dataset = [None]*N
for m in meta:
    i = m["id"]
    answers[i] = m.get("answers") or [m.get("answer","")]
    dataset[i] = m["dataset"]
assert all(d is not None for d in dataset), "meta ids not 0..N-1"
print("meta n =", N)

datasets = sorted(set(dataset))
ds_idx = {d: np.array([i for i in range(N) if dataset[i]==d]) for d in datasets}
iid_ds = [d for d in datasets if is_iid(d)]
ood_ds = [d for d in datasets if not is_iid(d)]
print("IID:", iid_ds); print("OOD:", ood_ds)

cols = {}   # name -> (f1 array, em array), values in [0,1]

def load_native(pattern, key, name):
    f1 = np.full(N, np.nan); em = np.full(N, np.nan)
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        for e in d[key]:
            f1[e["id"]] = e["f1"]; em[e["id"]] = e["em"]
    assert not np.isnan(f1).any(), f"{name}: missing samples"
    cols[name] = (f1, em)
    print(f"loaded {name} from {pattern} [{key}]")
    return

def load_bridge(pattern, key, name):
    f1 = np.full(N, np.nan); em = np.full(N, np.nan)
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        for e in d[key]:
            i = e["id"]; gen = e["generated"]
            f1[i] = max(token_f1(gen,a) for a in answers[i])
            em[i] = max(exact_match(gen,a) for a in answers[i])
    assert not np.isnan(f1).any(), f"{name}: missing samples"
    cols[name] = (f1, em)
    print(f"scored {name} from {pattern} [{key}]")

V4 = f"{ROOT}/experiment/v4eval_step19999/output"
meank = {}
for eps in ["0.05","0.1","0.2","0.3"]:
    f1 = np.full(N, np.nan); em = np.full(N, np.nan); mk = np.full(N, np.nan)
    for f in sorted(glob.glob(f"{V4}/pareto_full58k_dyneps_rank*.json")):
        d = json.load(open(f))
        for e in d[f"eps={eps}"]:
            f1[e["id"]] = e["f1"]; em[e["id"]] = e["em"]
            mk[e["id"]] = e["mean_k_per_chunk"]
    assert not np.isnan(f1).any()
    cols[f"MANTA_eps{eps}"] = (f1, em); meank[eps] = mk
    print(f"loaded MANTA eps={eps}")

load_native(f"{ROOT}/experiment/sac_1b_mrqa_r64/output/sac1b_full58k_r64_rank*.json", "8", "SAC_k8")
load_native(f"{ROOT}/experiment/sac_1b_mrqa_r85/output/sac1b_full58k_r85_rank*.json", "6", "SAC_k6")
load_native(f"{ROOT}/experiment/sac_1b_mrqa_r128/output/sac1b_full58k_r128_rank*.json", "4", "SAC_k4")
load_native(f"{ROOT}/experiment/sac_1b_mrqa_r170/output/sac1b_full58k_r170_rank*.json", "3", "SAC_k3")

for r in ["64","85","128","170"]:
    load_bridge(f"{ROOT}/experiment/c500x_1b_mrqa_r{r}/eval_full58k/c500x_mrqa_gen_rank*.json", r, f"C500X_r{r}")
for r,dirn in [("64","icae_1b_mrqa_r64"),("64","icae_1b_mrqa_r64_fs2024"),("85","icae_1b_mrqa_r85"),
               ("128","icae_1b_mrqa_r128"),("170","icae_1b_mrqa_r170")]:
    name = f"ICAE_r{r}_fs2024" if "fs2024" in dirn else f"ICAE_r{r}"
    load_bridge(f"{ROOT}/experiment/{dirn}/eval_full58k/icae_mrqa_gen_rank*.json", r, name)

# ---- mean-k percentiles ----
mk_out = {}
for eps, mk in meank.items():
    q = np.percentile(mk, [50,90,95,99])
    mk_out[eps] = dict(mean=float(mk.mean()), p50=float(q[0]), p90=float(q[1]),
                       p95=float(q[2]), p99=float(q[3]), max=float(mk.max()), n=int(len(mk)))
    print(f"mean_k eps={eps}: mean {mk.mean():.3f} p50 {q[0]:.2f} p90 {q[1]:.2f} p95 {q[2]:.2f} p99 {q[3]:.2f} max {mk.max():.1f}")

names = list(cols.keys())
V_f1 = np.stack([cols[n][0] for n in names], axis=1)  # N x C
V_em = np.stack([cols[n][1] for n in names], axis=1)
C = len(names)

# point estimates (macro)
def macro(vec_mat, dslist):
    return np.stack([vec_mat[ds_idx[d]].mean(axis=0) for d in dslist]).mean(axis=0)
pt = {}
for tag, dl in [("IID", iid_ds), ("OOD", ood_ds)]:
    pt[tag] = {"f1": dict(zip(names, (macro(V_f1, dl)*100).round(4))),
               "em": dict(zip(names, (macro(V_em, dl)*100).round(4)))}

# ---- bootstrap replicate per-dataset means, shared draws ----
rng = np.random.default_rng(SEED)
rep_ds_f1 = {}; rep_ds_em = {}
for d in datasets:
    idx = ds_idx[d]; n = len(idx)
    Vf = V_f1[idx]; Ve = V_em[idx]
    rf = np.empty((B, C)); re_ = np.empty((B, C))
    p = np.full(n, 1.0/n)
    for s in range(0, B, BLOCK):
        cnt = rng.multinomial(n, p, size=min(BLOCK, B-s)).astype(np.float64)
        rf[s:s+cnt.shape[0]] = cnt @ Vf / n
        re_[s:s+cnt.shape[0]] = cnt @ Ve / n
    rep_ds_f1[d] = rf; rep_ds_em[d] = re_
    print("bootstrapped", d, n, flush=True)

rep = {}
for tag, dl in [("IID", iid_ds), ("OOD", ood_ds)]:
    rep[tag] = {"f1": np.mean([rep_ds_f1[d] for d in dl], axis=0)*100,
                "em": np.mean([rep_ds_em[d] for d in dl], axis=0)*100}

pairings = [
    ("MANTA_eps0.05", ["SAC_k8", "C500X_r64", "ICAE_r64", "ICAE_r64_fs2024"]),
    ("MANTA_eps0.1",  ["SAC_k6", "C500X_r85", "ICAE_r85"]),
    ("MANTA_eps0.2",  ["SAC_k4", "C500X_r128", "ICAE_r128"]),
    ("MANTA_eps0.3",  ["SAC_k3", "C500X_r170", "ICAE_r170"]),
]
ni = {n:i for i,n in enumerate(names)}
out = {"n": N, "B": B, "seed": SEED, "point_macro": pt, "mean_k_percentiles": mk_out, "margins": {}}
for manta, bases in pairings:
    for b in bases:
        for tag in ["IID","OOD"]:
            for met in ["f1","em"]:
                marg = rep[tag][met][:, ni[manta]] - rep[tag][met][:, ni[b]]
                lo, hi = np.percentile(marg, [2.5, 97.5])
                point = pt[tag][met][manta] - pt[tag][met][b]
                key = f"{manta} - {b} | {tag} {met}"
                out["margins"][key] = dict(point=round(float(point),3),
                                           ci95=[round(float(lo),3), round(float(hi),3)],
                                           boot_mean=round(float(marg.mean()),3))
                print(f"{key}: {point:+.2f} [{lo:+.2f}, {hi:+.2f}]")

json.dump(out, open("/tmp/manta_bootstrap_results.json","w"), indent=1)
print("wrote /tmp/manta_bootstrap_results.json")
