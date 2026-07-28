#!/usr/bin/env python
"""Consistent MEASURED realized compression (real-tokens / kept-slots) for all 64x-bucket methods.
- anchor count per chunk = len(arange(start, end, ratio)), start=x1+(ratio-1)//2, x1=chunk_start+1, end=chunk_end+1 (model rule, get_uniform_position_ids).
- SAC k8: ratio=64, keep all anchors. PACT-M k8: ratio=16, keep min(8,anchors). PACT adaptive: kept from log. bridge: T/(num_chunks*num_mem) from log.
Reports micro (sumT/sumSlots) and per-sample mean.
"""
import json, glob, math
import torch, numpy as np

CHUNK=512
def anchors(L, ratio):
    # one chunk of real length L tokens, offsets [0,L); x1=1, xn=L+1
    x1=1; xn=L+1
    start=x1+(ratio-1)//2
    if start>=xn: start=x1
    return max(0, (xn-start+ratio-1)//ratio) if xn>start else 0
def slots_full(T, ratio):
    nch=math.ceil(T/CHUNK); s=0
    for c in range(nch):
        L=min((c+1)*CHUNK,T)-c*CHUNK
        s+=anchors(L,ratio)
    return s, nch

print("loading eval lengths..."); 
ex=torch.load("data/mrqa_eval_full/qa_eval.pt")
meta=json.load(open("data/mrqa_eval_full/qa_eval_meta.json"))
T={}  # id -> context token length
for i,m in enumerate(meta):
    T[str(m["id"])]=int(ex[i]["input_ids"].shape[-1])
print("n=",len(T))

def agg(name, idlist, slotlist):
    T_=np.array([T[i] for i in idlist],float); S=np.array(slotlist,float)
    micro=T_.sum()/S.sum(); per=(T_/np.maximum(S,1)).mean()
    print("%-28s micro=%.1fx  per-sample-mean=%.1fx  (sumT=%d sumSlots=%d)"%(name,micro,per,T_.sum(),S.sum()))

# SAC k8 (ratio 64, keep all)
ids=[]; sl=[]
for i in T: ids.append(i); sl.append(slots_full(T[i],64)[0])
agg("SAC-1B k8 (ratio64)", ids, sl)

# PACT-M fixed k8 (ratio 16, keep min(8,anchors))  == mechanism comparison
def slots_trunck(Tt, ratio, k):
    nch=math.ceil(Tt/CHUNK); s=0
    for c in range(nch):
        L=min((c+1)*CHUNK,Tt)-c*CHUNK
        s+=min(k, anchors(L,ratio))
    return s
ids=[]; sl=[]
for i in T: ids.append(i); sl.append(slots_trunck(T[i],16,8))
agg("PACT-M k8 (ratio16,k8)", ids, sl)

# PACT adaptive eps=0.001: kept from log
kept={}
for f in sorted(glob.glob("experiment/pact_v5_mrqa_dyneps_k32_1B_v3/output/pareto_full58k_dyneps_rank*.json")):
    d=json.load(open(f))
    for s in d["eps=0.001"]: kept[str(s["id"])]=s["total_kept"]
ids=[i for i in T if i in kept]; sl=[kept[i] for i in ids]
agg("PACT adaptive eps=0.001", ids, sl)

# bridge: slots = T/comp from log
for nm,patt in [("500xC r64","experiment/c500x_1b_mrqa_r64/eval_full58k/c500x_mrqa_gen_rank*.json"),("ICAE r64","experiment/icae_1b_mrqa_r64/eval_full58k/icae_mrqa_gen_rank*.json")]:
    comp={}
    for f in glob.glob(patt):
        d=json.load(open(f))
        for s in d["64"]: comp[str(s["id"])]=s["compression_rate"]
    ids=[i for i in T if i in comp]; sl=[T[i]/comp[i] for i in ids]
    agg(nm, ids, sl)
