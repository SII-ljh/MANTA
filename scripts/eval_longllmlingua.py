"""LongLLMLingua baseline — token-level compression then Llama-3.1-8B inference.

Backbone-matched comparison with PACT:
  - Compressor: NousResearch/Llama-2-7b-hf (LongLLMLingua default)
  - Decoder:    meta-llama/Llama-3.1-8B (same as PACT)

For each comi_eval sample:
  1. Detokenize input_ids → raw context text
  2. Detokenize lm_targets → question (instruction prompt without answer)
  3. LongLLMLingua compresses context at target rate (16x or 32x)
  4. Build new prompt: compressed_context + question
  5. Generate answer with Llama-3.1-8B
  6. Compute F1/EM vs gold answers
"""
import argparse
import json
import os
import re
import string
import sys
import time
from collections import Counter
from pathlib import Path

import torch
from llmlingua import PromptCompressor
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(
    os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = s.translate(str.maketrans("", "", string.punctuation))
    return " ".join(s.split())


def token_f1(p, g):
    pt = normalize(p).split()
    gt = normalize(g).split()
    if not gt:
        return float(not pt)
    if not pt:
        return 0.0
    c = sum((Counter(pt) & Counter(gt)).values())
    if c == 0:
        return 0.0
    pr = c / len(pt)
    rc = c / len(gt)
    return 2 * pr * rc / (pr + rc)


def em(p, g):
    return float(normalize(p) == normalize(g))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target_token", type=int, required=True,
                    help="target compressed tokens (e.g., 250 for 16x compression of 4000-token context)")
    ap.add_argument("--num_samples", type=int, default=1000)
    ap.add_argument("--output_name", required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument(
        "--compressor_model",
        default="NousResearch/Llama-2-7b-hf",
    )
    ap.add_argument(
        "--decoder_model",
        default=os.environ.get("MANTA_MODEL_ID", "meta-llama/Llama-3.1-8B"),
    )
    args = ap.parse_args()

    device = f"cuda:{args.gpu}"
    print(f"[init] compressor={args.compressor_model}")
    compressor = PromptCompressor(
        model_name=args.compressor_model,
        device_map=device,
    )

    print(f"[init] decoder={args.decoder_model}")
    dec_tok = AutoTokenizer.from_pretrained(args.decoder_model)
    decoder = AutoModelForCausalLM.from_pretrained(
        args.decoder_model, torch_dtype=torch.bfloat16, device_map=device
    )
    decoder.eval()

    print("[load] eval data")
    examples = torch.load(ROOT / "sft/output/comi_eval_Llama-3.1-8B_100000samples_instruction.pt")
    with open(ROOT / "data/unified_eval_subset/comi_eval_meta.json") as f:
        meta = json.load(f)
    assert len(examples) == len(meta)

    if args.num_samples > 0 and args.num_samples < len(examples):
        import random
        rng = random.Random(42)
        idx = rng.sample(range(len(examples)), args.num_samples)
        idx.sort()
        examples = [examples[i] for i in idx]
        meta = [meta[i] for i in idx]

    out_records = []
    t0 = time.time()
    for i, (ex, m) in enumerate(zip(examples, meta)):
        # 1. decode context: strip the "### Context:\n" prefix and BOS
        ctx_text = dec_tok.decode(ex["input_ids"], skip_special_tokens=True)
        ctx_text = ctx_text.replace("### Context:\n", "", 1).strip()

        # 2. decode question prompt
        q_text = dec_tok.decode(ex["lm_targets"], skip_special_tokens=True).strip()
        # q_text looks like "### Question:\n... ### Answer:"
        question_match = re.search(r"### Question:\s*(.*?)\s*### Answer:?\s*$",
                                   q_text, re.DOTALL)
        question_only = question_match.group(1).strip() if question_match else q_text

        # 3. compress
        try:
            comp = compressor.compress_prompt(
                context=[ctx_text],
                question=question_only,
                target_token=args.target_token,
                condition_in_question="after_condition",
                rank_method="longllmlingua",
                use_sentence_level_filter=False,
                concate_question=False,
            )
            compressed_text = comp["compressed_prompt"]
        except Exception as e:
            print(f"[err] sample {i}: {e}")
            compressed_text = ctx_text[: args.target_token * 4]  # crude fallback

        # 4. build full prompt
        full_prompt = (
            f"<|begin_of_text|>### Context:\n{compressed_text}\n\n"
            f"### Question:\n{question_only}\n### Answer:\n"
        )
        prompt_ids = dec_tok(full_prompt, return_tensors="pt", add_special_tokens=False)
        prompt_ids = {k: v.to(device) for k, v in prompt_ids.items()}

        # 5. generate
        with torch.no_grad():
            out = decoder.generate(
                **prompt_ids,
                max_new_tokens=64,
                do_sample=False,
                eos_token_id=dec_tok.eos_token_id,
            )
        gen_ids = out[0][prompt_ids["input_ids"].shape[1]:]
        gen_text = dec_tok.decode(gen_ids, skip_special_tokens=True).strip()
        # strip everything after first newline (answer usually one line)
        gen_text = gen_text.split("\n")[0].strip()

        # 6. score
        f1 = max(token_f1(gen_text, a) for a in m["answers"])
        e = max(em(gen_text, a) for a in m["answers"])

        out_records.append({
            "id": m["id"],
            "dataset": m["dataset"],
            "f1": f1,
            "em": e,
            "generated": gen_text,
            "compressed_len": len(compressed_text),
            "compressed_tokens_est": len(dec_tok.encode(compressed_text)),
        })

        if (i + 1) % 20 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (len(examples) - i - 1)
            print(f"  [{i + 1}/{len(examples)}] elapsed={elapsed:.0f}s eta={eta:.0f}s "
                  f"F1_so_far={sum(r['f1'] for r in out_records) / len(out_records):.3f}")

    # Aggregate
    overall_f1 = sum(r["f1"] for r in out_records) / len(out_records)
    overall_em = sum(r["em"] for r in out_records) / len(out_records)
    avg_comp_tokens = sum(r["compressed_tokens_est"] for r in out_records) / len(out_records)

    summary = {
        "method": "LongLLMLingua",
        "compressor": args.compressor_model,
        "decoder": args.decoder_model,
        "target_token": args.target_token,
        "n": len(out_records),
        "f1": overall_f1,
        "em": overall_em,
        "avg_compressed_tokens": avg_comp_tokens,
    }

    by_dataset = {}
    for r in out_records:
        by_dataset.setdefault(r["dataset"], []).append(r)
    per_ds = {}
    for ds, rs in by_dataset.items():
        per_ds[ds] = {
            "n": len(rs),
            "f1": sum(x["f1"] for x in rs) / len(rs),
            "em": sum(x["em"] for x in rs) / len(rs),
        }
    summary["per_dataset"] = per_ds

    out_dir = ROOT / "experiment" / "longllmlingua_baseline" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"{args.output_name}_records.json", "w") as f:
        json.dump(out_records, f)
    with open(out_dir / f"{args.output_name}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n" + "=" * 60)
    print(f"DONE n={len(out_records)} F1={overall_f1:.4f} EM={overall_em:.4f}")
    print(f"avg compressed tokens: {avg_comp_tokens:.0f}")
    print("Per-dataset:")
    for ds, s in per_ds.items():
        print(f"  {ds:15s} n={s['n']:4d} F1={s['f1']:.3f} EM={s['em']:.3f}")
    print(f"output: {out_dir / args.output_name}_summary.json")


if __name__ == "__main__":
    main()
