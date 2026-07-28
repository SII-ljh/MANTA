#!/usr/bin/env python
"""Main-table per-dataset token-F1+EM @64x: PACT(eps=0.001) vs SAC-r64 vs c500x-r64 vs ICAE-r64.
Second metric (unified standard v2): SQuAD-normalized token-F1 (max-over-references) + normalized EM.
token_f1 / normalize_answer / exact_match identical to scripts/eval_k_sweep.py & rescore_tokenf1_from_generated.py.
Mirrors scripts/maintable_perdataset_64x.py (same sources/keys) so the two tables are apples-to-apples.
"""
import glob, json, re, string
from collections import defaultdict, Counter
import numpy as np

IID = ["SQuAD","NewsQA","TriviaQA-web","SearchQA","HotpotQA","NaturalQuestionsShort"]
OOD = ["BioASQ","DROP","DuoRC.ParaphraseRC","RACE","RelationExtraction","TextbookQA"]
PRETTY = {"TriviaQA-web":"TriviaQA","NaturalQuestionsShort":"NQ","DuoRC.ParaphraseRC":"DuoRC","RelationExtraction":"RE"}

def normalize_answer(s):
    s = (s or "").lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())

def _tf1(pred, gold):
    pt = normalize_answer(pred).split(); gt = normalize_answer(gold).split()
    if not gt: return float(not pt)
    if not pt: return 0.0
    nc = sum((Counter(pt) & Counter(gt)).values())
    if nc == 0: return 0.0
    p = nc/len(pt); r = nc/len(gt)
    return 2*p*r/(p+r)

def token_f1(pred, golds):
    return max(_tf1(pred, g) for g in golds)

def exact_match(pred, golds):
    return float(any(normalize_answer(pred) == normalize_answer(g) for g in golds))

# meta: all answers (max-over-refs) + dataset
meta = {}
for m in json.load(open("data/mrqa_eval_full/qa_eval_meta.json")):
    golds = m.get("answers") or [m.get("answer","")]
    meta[str(m["id"])] = (golds, m["dataset"])

def merge(pat, want_key):
    out=[]
    for fp in sorted(glob.glob(pat)):
        d=json.load(open(fp))
        if want_key in d: out.extend(d[want_key])
    return out

def score(samples):
    F=defaultdict(list); E=defaultdict(list)
    for s in samples:
        golds, ds = meta[str(s["id"])]
        gen=s.get("generated","")
        F[ds].append(token_f1(gen,golds)); E[ds].append(exact_match(gen,golds))
    per={}
    for ds in IID+OOD:
        if F[ds]:
            per[PRETTY.get(ds,ds)]={"f1":np.mean(F[ds])*100,"em":np.mean(E[ds])*100,"n":len(F[ds])}
    def macro(sp):
        vf=[np.mean(F[d]) for d in sp if F[d]]; ve=[np.mean(E[d]) for d in sp if E[d]]
        return float(np.mean(vf))*100, float(np.mean(ve))*100
    iif,iie=macro(IID); oof,ooe=macro(OOD)
    return {"per_dataset":per,"iid_f1":iif,"iid_em":iie,"ood_f1":oof,"ood_em":ooe,"n":len(samples)}

SRC = {
 "PACT (eps=0.05, 65x, v4@20k)": ("experiment/v4eval_step19999/output/pareto_full58k_dyneps_rank*.json","eps=0.05"),
 "SAC-1B (k8, 64x)":       ("experiment/sac_1b_mrqa_r64/output/sac1b_full58k_r64_rank*.json","8"),
 "500xC (r64, 37.8x)":     ("experiment/c500x_1b_mrqa_r64/eval_full58k/c500x_mrqa_gen_rank*.json","64"),
 "ICAE (r64, 37.8x)":      ("experiment/icae_1b_mrqa_r64_fs2024/eval_full58k/icae_mrqa_gen_rank*.json","64"),
}
rep={}
for lab,(pat,key) in SRC.items():
    s=merge(pat,key); rep[lab]=score(s)
    print("[%s] n=%d IID %.2f/%.2f OOD %.2f/%.2f"%(lab,rep[lab]["n"],rep[lab]["iid_f1"],rep[lab]["iid_em"],rep[lab]["ood_f1"],rep[lab]["ood_em"]))
json.dump(rep,open("reports/maintable_perdataset_v4_20k_65x_tokenf1_icaefix.json","w"),indent=2,ensure_ascii=False)
print("[saved] reports/maintable_perdataset_v4_20k_65x_tokenf1_icaefix.json")
