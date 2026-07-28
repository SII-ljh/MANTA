#!/usr/bin/env python3
"""F99 addendum: re-express KPredictor-budget vs frozen-base causal-importance
association with rank-agreement metrics (pairwise concordance + argmax agreement
+ per-sample Pearson), read from the existing leave-one-chunk-out erasure dumps.
Rank-agreement is robust to the heavily skewed I_c (~47% chunks have I_c<=0,
distractors) and the discrete k, which deflate a raw pooled Spearman."""
import json, math, os
from statistics import mean
BASE="experiment/analysis_causal_grounding"
FILES={"0.01":"full427_eps0.01.json","0.05":"full427_eps05.json","0.1":"full427_eps0.1.json"}
def pearson(x,y):
    n=len(x)
    if n<2: return None
    mx=mean(x); my=mean(y)
    sx=sum((a-mx)**2 for a in x); sy=sum((b-my)**2 for b in y)
    if sx==0 or sy==0: return None
    return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(sx*sy)
out={}
for eps,f in FILES.items():
    d=json.load(open(os.path.join(BASE,f)))
    ps=d["per_sample"]; persamp=[]; ah=0; at=0; conc=0; disc=0; chance=[]
    for s in ps:
        I=s["I_c"]; K=[float(k) for k in s["k_c"]]
        if len(I)<2: continue
        r=pearson(I,K)
        if r is not None: persamp.append(r)
        mi=max(range(len(I)),key=lambda j:I[j]); mk=max(range(len(K)),key=lambda j:K[j])
        at+=1; ah+=(mi==mk); chance.append(1.0/len(I))
        for i in range(len(I)):
            for j in range(i+1,len(I)):
                if I[i]==I[j] or K[i]==K[j]: continue
                if (I[i]-I[j])*(K[i]-K[j])>0: conc+=1
                else: disc+=1
    out[eps]={"pairwise_concordance":round(conc/(conc+disc),4),"n_pairs":conc+disc,
              "argmax_agreement":round(ah/at,4),"argmax_chance":round(mean(chance),4),
              "argmax_hit":ah,"argmax_tot":at,
              "persample_mean_pearson":round(mean(persamp),4),"n_samples":len(persamp)}
print(json.dumps(out,indent=2))
json.dump(out,open(os.path.join(BASE,"corr_recompute.json"),"w"),indent=2)
