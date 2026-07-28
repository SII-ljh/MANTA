import os
import json
ROOT=os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
IID=["SQuAD","NewsQA","TriviaQA-web","SearchQA","HotpotQA","NaturalQuestionsShort"]
OOD=["BioASQ","DROP","DuoRC.ParaphraseRC","RACE","RelationExtraction","TextbookQA"]

def macro(per_ds, split, key):
    vals=[]
    for dd in split:
        if dd in per_ds and isinstance(per_ds[dd],dict) and key in per_ds[dd]:
            x=per_ds[dd][key]
            vals.append(x*100 if x<=1.5 else x)  # normalize: fraction->percent, leave percent as-is
    return (sum(vals)/len(vals)) if vals else float('nan')

def find_perdataset(obj):
    """Return the dict {dataset:{f1,em,..}} wherever it is nested."""
    if isinstance(obj,dict):
        # is this already a dataset-map? (has one of our known datasets as key)
        if any(k in obj for k in IID+OOD):
            return obj
        for v in obj.values():
            r=find_perdataset(v)
            if r: return r
    return None

# --- topline (dense full-FT, uncompressed) ---
top=json.load(open(f"{ROOT}/reports/tokenf1_perdataset_fullft_1b_full58k.json"))
pd_top=find_perdataset(top)
print("topline per-dataset keys:", sorted(pd_top.keys()))
print("sample entry:", {k:pd_top[list(pd_top)[0]][k] for k in pd_top[list(pd_top)[0]]})
# detect f1/em key names
sample=pd_top[list(pd_top)[0]]
f1k = 'f1' if 'f1' in sample else ('token_f1' if 'token_f1' in sample else list(sample)[0])
emk = 'em' if 'em' in sample else ('exact_match' if 'exact_match' in sample else None)
tI=macro(pd_top,IID,f1k); tO=macro(pd_top,OOD,f1k)
print(f"\n### UNCOMPRESSED UPPER BOUND (dense full-FT, no compression) ###")
print(f"  token-F1  IID={tI:.2f}  OOD={tO:.2f}   (f1 key='{f1k}')")

# --- compressors: ksweep full58k ---
def ksweep(exp):
    return json.load(open(f"{ROOT}/experiment/{exp}/output/ksweep_full58k_summary.json"))["per_dataset_per_k"]
ff=ksweep("pact_v5_mrqa_fullft_1B"); v3=ksweep("pact_v5_mrqa_dyneps_k32_1B_v3")
kv=sorted({int(k) for k in ff})
print(f"\n### GAP TO UPPER BOUND  (token-F1; gap = topline - model; recovery = model/topline) ###")
print(f"{'k':>3} | {'full-FT ID/OOD':>14} | {'gap ID/OOD':>12} {'rec% ID/OOD':>12} || {'v3 ID/OOD':>14} | {'gap ID/OOD':>12} {'rec% ID/OOD':>12}")
print("-"*104)
for k in kv:
    kk=str(k)
    fI=macro(ff[kk],IID,'f1'); fO=macro(ff[kk],OOD,'f1')
    vI=macro(v3[kk],IID,'f1'); vO=macro(v3[kk],OOD,'f1')
    print(f"{k:>3} | {fI:6.2f}/{fO:6.2f}  | {tI-fI:5.2f}/{tO-fO:5.2f} {100*fI/tI:5.1f}/{100*fO/tO:5.1f}  || {vI:6.2f}/{vO:6.2f}  | {tI-vI:5.2f}/{tO-vO:5.2f} {100*vI/tI:5.1f}/{100*vO/tO:5.1f}")
