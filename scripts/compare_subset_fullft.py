import os
import json, sys
ROOT=os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Paper IID/OOD split (macro = unweighted mean over the 6 datasets in each group).
IID=["SQuAD","NewsQA","TriviaQA-web","SearchQA","HotpotQA","NaturalQuestionsShort"]
OOD=["BioASQ","DROP","DuoRC.ParaphraseRC","RACE","RelationExtraction","TextbookQA"]
EXPS={"fullFT":"experiment/pact_v5_mrqa_fullft_1B/output",
      "v3LoRA":"experiment/pact_v5_mrqa_dyneps_k32_1B_v3/output"}

def macro(per_ds, split, key):
    vals=[per_ds[d][key] for d in split if d in per_ds]
    miss=[d for d in split if d not in per_ds]
    return (sum(vals)/len(vals) if vals else float('nan')), miss

def load(name,fn):
    return json.load(open(f"{ROOT}/{EXPS[name]}/{fn}"))

print("#"*78)
print("# FIXED-K SWEEP (2000 subset) — macro token-F1 / EM,  IID(6) | OOD(6)")
print("#"*78)
ks={n:load(n,"ksweep_2000_summary.json")["per_dataset_per_k"] for n in EXPS}
kvals=sorted({int(k) for k in ks["fullFT"]})
# sanity: n counts identical between the two (same subset)?
_missmsg=set()
print(f"{'k':>3} | {'fullFT F1(ID/OOD)':>18} | {'v3 F1(ID/OOD)':>18} | {'ΔF1':>13} || {'fullFT EM':>13} | {'v3 EM':>13}")
print("-"*100)
for k in kvals:
    kk=str(k)
    fID,mi=macro(ks['fullFT'][kk],IID,'f1'); fOD,_=macro(ks['fullFT'][kk],OOD,'f1')
    vID,_=macro(ks['v3LoRA'][kk],IID,'f1'); vOD,_=macro(ks['v3LoRA'][kk],OOD,'f1')
    feID,_=macro(ks['fullFT'][kk],IID,'em'); feOD,_=macro(ks['fullFT'][kk],OOD,'em')
    veID,_=macro(ks['v3LoRA'][kk],IID,'em'); veOD,_=macro(ks['v3LoRA'][kk],OOD,'em')
    _missmsg|=set(mi)
    print(f"{k:>3} | {fID*100:6.2f}/{fOD*100:6.2f}     | {vID*100:6.2f}/{vOD*100:6.2f}     | {(fID-vID)*100:+5.2f}/{(fOD-vOD)*100:+5.2f} || {feID*100:5.2f}/{feOD*100:5.2f}  | {veID*100:5.2f}/{veOD*100:5.2f}")
if _missmsg: print("  [warn] datasets missing from a summary:",_missmsg)

print()
print("#"*78)
print("# ADAPTIVE ε SWEEP (2000 subset) — macro token-F1,  IID | OOD,  + mean_k")
print("#"*78)
def pareto(name):
    d=load(name,"pareto_2000_dyneps_summary.json")
    return d
for tag in ["fullFT","v3LoRA"]:
    d=pareto(tag)
    pl=d.get("per_label",{}); pld=d.get("per_label_per_dataset",{})
    print(f"\n[{tag}] per_label keys sample:", list(pl.keys())[:8])
    # show the schema of one label
    if pl:
        k0=list(pl.keys())[0]; print(f"  per_label['{k0}'] fields:", {k:(round(v,4) if isinstance(v,(int,float)) else type(v).__name__) for k,v in (pl[k0].items() if isinstance(pl[k0],dict) else [])})
    for lab in pl:
        pds=pld.get(lab,{})
        fID,_=macro(pds,IID,'f1'); fOD,_=macro(pds,OOD,'f1')
        eID,_=macro(pds,IID,'em'); eOD,_=macro(pds,OOD,'em')
        mk=pl[lab].get('mean_k_per_chunk','?'); cr=pl[lab].get('compression_rate','?')
        print(f"  {tag} {lab:>10}  mean_k={mk:5.2f} comp={cr:.3f}  F1 ID/OOD={fID*100:6.2f}/{fOD*100:6.2f}  EM={eID*100:5.2f}/{eOD*100:5.2f}")
