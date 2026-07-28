"""mean_k-only ε sweep for PACT KPredictor.

For each eval sample: compress() ONCE, then for each ε call k_predictor.predict_k
and record per-sample mean_k (= total_kept / n_chunks). NO generation — this is
~100x cheaper than eval_v5_pareto and gives the exact ε→mean_k curve needed to
pick ε so that 512/mean_k beats integer-k baselines.

Single-GPU. Subset via --num_samples (mean_k is a stable population mean).
"""
import argparse, json, os, sys, random
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from model.modeling import get_model, load_adapter, extract_full_kv


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--work_dir", required=True)
    p.add_argument("--eval_data_path", required=True)
    p.add_argument("--eval_meta_path", required=True)
    p.add_argument("--num_samples", type=int, default=4000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--adapter_name", default="instruction_adapter.pt")
    p.add_argument("--eps_list", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    eps_list = [float(e) for e in args.eps_list.split(",")]
    rank = 0
    with open(os.path.join(args.work_dir, "output", "config.json")) as f:
        config = json.load(f)
    training_config = config["sft_training_config"]
    task_config = config["sft_task_config"]

    eval_examples = torch.load(args.eval_data_path)
    n_total = len(eval_examples)
    idx = list(range(n_total))
    random.seed(args.seed)
    random.shuffle(idx)
    idx = idx[: min(args.num_samples, n_total)]
    print(f"[INFO] {len(idx)}/{n_total} samples, ε={eps_list}", flush=True)

    model = get_model(training_config["model_id"], task_config, rank)
    model = load_adapter(model, save_path_and_name=os.path.join(args.work_dir, "output", args.adapter_name))
    model.eval()
    assert getattr(model, "k_predictor", None) is not None

    # per-sample mean_k accumulators
    acc = {e: [] for e in eps_list}

    for i in tqdm(idx):
        ex = eval_examples[i]
        inputs = {"input_ids": ex["input_ids"].unsqueeze(0).to(f"cuda:{rank}"),
                  "lm_targets": ex["lm_targets"].unsqueeze(0).to(f"cuda:{rank}")}
        try:
            with torch.no_grad():
                (_ctids, _ctok, _end, _mlh, enc_pkv, _ems,
                 per_chunk_mem_sizes) = model.compress(inputs, truncate_k=None)
                anchor_kv = extract_full_kv(enc_pkv)
                device = enc_pkv[0][0].device
                n_chunks = len(per_chunk_mem_sizes)
                for e in eps_list:
                    eps_t = torch.tensor([float(e)], device=device)
                    _, k_pred = model.k_predictor.predict_k(
                        anchor_kv, per_chunk_mem_sizes, eps_t)
                    total_kept = int(sum(int(x) for x in k_pred))
                    acc[e].append(total_kept / max(1, n_chunks))
        except Exception as ex_e:
            print(f"sample {i} failed: {ex_e}", flush=True)
            continue

    summary = {}
    for e in eps_list:
        a = np.array(acc[e])
        mk = float(a.mean())
        summary[f"eps={e}"] = dict(
            eps=e, n=len(a), mean_k=mk,
            comp_ratio=512.0 / mk if mk > 0 else None,
            mean_k_se=float(a.std() / np.sqrt(max(1, len(a)))),
        )
    with open(args.out, "w") as f:
        json.dump(summary, f, indent=1)

    print("\n  eps   mean_k   512/mean_k(×)   SE", flush=True)
    for e in eps_list:
        s = summary[f"eps={e}"]
        print(f"  {e:<5} {s['mean_k']:>7.3f}   {s['comp_ratio']:>9.1f}     ±{s['mean_k_se']:.3f}", flush=True)
    print(f"\nsaved {args.out}", flush=True)


if __name__ == "__main__":
    main()
