# Reconstruct pretrain/output/long_text.json from the existing Llama-3.1-8B
# pretrain cache by decoding token ids back to raw text. Lets the Qwen2.5
# prep re-tokenize the SAME source corpus (max comparability to Llama line).
# Idempotent: skips if long_text.json already exists.
import os, json, torch
from transformers import AutoTokenizer
WD=os.path.dirname(os.path.abspath(__file__))
OUT=os.path.join(WD,"output","long_text.json")
if os.path.exists(OUT):
    print("long_text.json exists, skip"); raise SystemExit(0)
LLAMA=os.environ.get("MANTA_MODEL_ID", "meta-llama/Llama-3.2-1B")
tok=AutoTokenizer.from_pretrained(LLAMA)
src=os.path.join(WD,"output","train_Llama-3.1-8B_320000samples_510-2040len.pt")
d=torch.load(src)
print("loaded",len(d),"train examples")
# also include the 1000 eval examples so retokenized eval split is non-empty
ev=os.path.join(WD,"output","eval_Llama-3.1-8B_320000samples_510-2040len.pt")
de=torch.load(ev)
print("loaded",len(de),"eval examples")
texts=[]
# eval first (prep takes examples[:1000] as eval), then train, to roughly
# preserve split membership. Order within prep: first 1000 -> eval, rest -> train.
for e in de+d:
    inp=e["inputs"].tolist(); lm=e["lm_target"].tolist()
    ids=inp[1:]+lm[:-1]   # strip leading bos, trailing eos
    texts.append(tok.decode(ids, skip_special_tokens=True))
print("decoded",len(texts),"texts; writing json")
with open(OUT,"w",encoding="utf-8") as f:
    json.dump(texts,f,ensure_ascii=False)
print("wrote",OUT, os.path.getsize(OUT),"bytes")
