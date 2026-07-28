"""Zero-shot baseline: raw Llama-3.1-8B (no SAC adapter) on unified test.

Prompt format: {context_truncated}\n\nQuestion: {question}\nAnswer:
Context truncated to first --max_ctx_tokens (default 4096), Llama native length.

Single-GPU per process — caller spawns 8 workers via shell with CUDA_VISIBLE_DEVICES,
to avoid any interference with concurrent DDP training on shared cards.

Usage:
  CUDA_VISIBLE_DEVICES=0 python scripts/eval_zero_shot.py --rank 0 --world_size 8 --out_dir <dir>
"""
import argparse, json, os, re, string, time, sys
from collections import Counter
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_ID = os.environ.get("MANTA_MODEL_ID", "meta-llama/Llama-3.1-8B")
DATA_PATH = os.path.join(os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "sft/output/unified_test_instruction_dataset.json")


def normalize(s):
    s = s.lower().strip()
    s = re.sub(r'\b(a|an|the)\b', ' ', s)
    s = s.translate(str.maketrans('', '', string.punctuation))
    return ' '.join(s.split())


def f1_score(pred, gold):
    p, g = normalize(pred).split(), normalize(gold).split()
    if not g: return float(not p)
    if not p: return 0.0
    common = sum((Counter(p) & Counter(g)).values())
    if common == 0: return 0.0
    pr, rc = common/len(p), common/len(g)
    return 2*pr*rc/(pr+rc)


def em_score(pred, gold):
    return float(normalize(pred) == normalize(gold))


def best_score(pred, golds, fn):
    return max(fn(pred, g) for g in golds) if golds else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rank', type=int, required=True)
    ap.add_argument('--world_size', type=int, default=8)
    ap.add_argument('--out_dir', type=str, required=True)
    ap.add_argument('--max_ctx_tokens', type=int, default=4096)
    ap.add_argument('--max_gen', type=int, default=64)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--limit', type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(DATA_PATH) as f:
        data = json.load(f)
    total = len(data)
    per = total // args.world_size
    s = args.rank * per
    e = s + per if args.rank < args.world_size - 1 else total
    my = data[s:e]
    if args.limit > 0:
        my = my[:args.limit]
    print(f'[rank{args.rank}] {len(my)} samples [{s}:{e}]', flush=True)

    tok = AutoTokenizer.from_pretrained(MODEL_ID)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = 'left'

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, torch_dtype=torch.bfloat16, attn_implementation='sdpa'
    ).cuda().eval()

    results = []
    t0 = time.time()
    Q_PROMPT = '\n\nQuestion: '
    A_PROMPT = '\nAnswer:'

    def truncate_ctx(ctx_text, q_text, max_ctx):
        # Tokenize context, keep first max_ctx tokens
        ids = tok(ctx_text, add_special_tokens=False, truncation=True,
                  max_length=max_ctx)['input_ids']
        return tok.decode(ids, skip_special_tokens=True)

    B = args.batch_size
    for batch_start in range(0, len(my), B):
        batch = my[batch_start:batch_start+B]
        prompts = []
        for s_ in batch:
            ctx = truncate_ctx(s_['context'], s_['question'], args.max_ctx_tokens)
            prompt = ctx + Q_PROMPT + s_['question'] + A_PROMPT
            prompts.append(prompt)
        enc = tok(prompts, return_tensors='pt', padding=True, truncation=True,
                  max_length=args.max_ctx_tokens + 256).to('cuda')
        with torch.no_grad():
            out = model.generate(
                **enc, max_new_tokens=args.max_gen,
                do_sample=False, temperature=1.0,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
            )
        gen_ids = out[:, enc['input_ids'].shape[1]:]
        gens = tok.batch_decode(gen_ids, skip_special_tokens=True)
        for s_, g in zip(batch, gens):
            # take first non-empty line as answer
            ans = g.strip().split('\n')[0].strip()
            f1 = best_score(ans, s_['answers'], f1_score)
            em = best_score(ans, s_['answers'], em_score)
            results.append({
                'subset': s_['subset'],
                'question': s_['question'],
                'gold': s_['answers'],
                'pred': ans,
                'f1': f1, 'em': em,
                'token_length': s_['token_length'],
            })
        if (batch_start // B) % 50 == 0:
            elapsed = time.time() - t0
            rate = (batch_start + len(batch)) / max(elapsed, 1)
            eta = (len(my) - batch_start - len(batch)) / max(rate, 1e-9)
            print(f'[rank{args.rank}] {batch_start+len(batch)}/{len(my)} '
                  f'({rate:.2f}/s, ETA {eta/60:.1f}min)', flush=True)

    out_path = os.path.join(args.out_dir, f'zero_shot_rank{args.rank}.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, ensure_ascii=False)
    print(f'[rank{args.rank}] saved {len(results)} -> {out_path}', flush=True)


if __name__ == '__main__':
    main()
