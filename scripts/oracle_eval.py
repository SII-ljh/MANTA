#!/usr/bin/env python
"""§3.3 + §5.7 + Appendix B — oracle per-chunk allocation analysis.

Reads per-chunk loss matrix produced by motivation_probe.py
(analysis/motivation_probe/per_chunk_loss.json) and computes:

  1. Joint-budget oracle (§3.3 / Figure 1 right panel):
       min_{k_1..k_N ∈ K_set} Σ L_i(k_i)  s.t.  Σ k_i ≤ B
     Solved per sample by 1D DP over (chunk_index, budget).
     Compare oracle LM loss vs best-fixed-k LM loss across average budgets.

  2. Per-chunk min-k threshold (§3.2 / Figure 2b):
       For each chunk, find min k s.t. L_k − L_K_max ≤ tolerance.

  3. Aggregate stats (§3.4 / Figure 6):
       Compare oracle (DP) mean LM loss to fixed-k strategies, plus an
       ε-conditioned per-chunk oracle (the same one used in v5 training).

  4. Per-dataset oracle k distribution (§5.7 / Table 6):
       For each dataset, average oracle k_i at a target budget.

CPU-only; reads JSON, writes JSON. No GPU needed.
"""
import argparse
import json
import os
from collections import defaultdict

import numpy as np


# K_SET override via env (e.g. PACT_K_SET=1,2,4,8,16,32 for k≤32 ckpts like dyneps_k32)
K_SET = [int(x) for x in os.environ.get("PACT_K_SET", "1,2,4,8,16,34").split(",")]
K_MAX = max(K_SET)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--per_chunk_loss",
                   default="analysis/motivation_probe/per_chunk_loss.json")
    p.add_argument("--out_dir", default="analysis/oracle_eval")
    p.add_argument("--budgets", default=None,
                   help="comma list of average k targets (e.g. 1,2,4,8,16,34). "
                        "Default = K_SET")
    p.add_argument("--tolerance", type=float, default=0.05,
                   help="per-chunk Δloss threshold for §3.2 Figure 2(b) hist")
    return p.parse_args()


def group_by_sample(records):
    """records is flat list across all chunks. Group into per-sample lists,
    preserving chunk order (chunk_idx)."""
    grouped = defaultdict(list)
    for r in records:
        grouped[r["sample_id"]].append(r)
    samples = []
    for sid, rs in grouped.items():
        rs = sorted(rs, key=lambda x: x["chunk_idx"])
        # Each chunk has losses dict {str(k): float}
        L = np.array([[float(r["losses"][str(k)]) for k in K_SET] for r in rs])
        samples.append({
            "sample_id": sid,
            "source": rs[0]["source"],
            "dataset": rs[0]["dataset"],
            "n_chunks": len(rs),
            "L": L,  # [N_chunks, |K_set|]
        })
    return samples


def joint_budget_oracle_dp(L, ks, budget):
    """Solve min Σ L_i[k_i] s.t. Σ ks[k_i] ≤ budget via DP.
    L: [N, |K|] loss matrix. ks: [|K|] integer budgets. budget: int total cap.
    Returns (best_loss, best_assignment_idx).
    """
    N, K = L.shape
    INF = 1e18
    # dp[b] = min sum loss using prefix budget b; trace[b][i] = chosen ki idx at chunk i
    dp = np.full(budget + 1, INF, dtype=np.float64)
    dp[0] = 0.0
    trace = [[-1] * (budget + 1) for _ in range(N)]
    for i in range(N):
        new_dp = np.full(budget + 1, INF, dtype=np.float64)
        for ki_idx, ki in enumerate(ks):
            cost = L[i, ki_idx]
            for b in range(budget + 1 - ki):
                if dp[b] + cost < new_dp[b + ki]:
                    new_dp[b + ki] = dp[b] + cost
                    trace[i][b + ki] = (b, ki_idx)
        dp = new_dp
    best_b = int(np.argmin(dp))
    best_loss = dp[best_b]
    # backtrack
    assignment = []
    b = best_b
    for i in reversed(range(N)):
        if trace[i][b] is None or trace[i][b] == -1:
            assignment.append(0)
            continue
        prev_b, ki_idx = trace[i][b]
        assignment.append(ki_idx)
        b = prev_b
    assignment.reverse()
    return best_loss, assignment, best_b


def fixed_k_loss(L, ki_idx):
    """Sum LM loss when every chunk uses k_set[ki_idx]."""
    return float(L[:, ki_idx].sum())


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    print(f"[oracle] loading {args.per_chunk_loss}")
    records = json.load(open(args.per_chunk_loss))
    print(f"[oracle] {len(records)} chunk records")

    samples = group_by_sample(records)
    print(f"[oracle] {len(samples)} samples after grouping")

    # ============================================================
    # 1. Per-chunk min-k threshold histogram (§3.2 / Figure 2b)
    # ============================================================
    tol = args.tolerance
    min_k_counts = defaultdict(int)
    for r in records:
        L_full = float(r["losses"][str(K_MAX)])
        for k in K_SET:
            if float(r["losses"][str(k)]) - L_full <= tol:
                min_k_counts[k] += 1
                break

    figure2b = {
        "tolerance": tol,
        "total_chunks": len(records),
        "min_k_hist": {str(k): min_k_counts[k] for k in K_SET},
        "min_k_frac": {str(k): min_k_counts[k] / max(1, len(records))
                       for k in K_SET},
    }
    print(f"\n[Figure 2b] min-k histogram (tol Δ={tol}):")
    for k in K_SET:
        print(f"  k={k:3d}: {min_k_counts[k]:5d} chunks "
              f"({100 * min_k_counts[k] / len(records):.1f}%)")

    # ============================================================
    # 2. Joint-budget oracle vs fixed-k (§3.3 / Figure 1 right panel)
    # ============================================================
    budgets = (
        [int(x) for x in args.budgets.split(",")] if args.budgets
        else K_SET
    )
    figure1_curves = {"oracle": {}, "fixed_best": {}, "fixed_per_k": {}}
    # Fixed-k baseline at every k — chunk-weighted mean (same denominator as oracle)
    fixed_per_k = {k: 0.0 for k in K_SET}
    total_chunks_global = 0
    for s in samples:
        for ki_idx, k in enumerate(K_SET):
            fixed_per_k[k] += float(s["L"][:, ki_idx].sum())
        total_chunks_global += s["L"].shape[0]
    for k in K_SET:
        fixed_per_k[k] /= max(1, total_chunks_global)
    figure1_curves["fixed_per_k"] = {str(k): fixed_per_k[k] for k in K_SET}
    print(f"\n[Figure 1] fixed-k average chunk LM loss:")
    for k in K_SET:
        print(f"  k={k:3d}: loss={fixed_per_k[k]:.4f}")

    # Oracle at each target average budget
    print(f"\n[Figure 1] joint-budget oracle:")
    oracle_curve = {}
    oracle_assignments = defaultdict(list)
    for tgt in budgets:
        total_loss = 0.0
        total_chunks = 0
        total_kept = 0
        for s in samples:
            N = s["n_chunks"]
            cap = tgt * N
            best_loss, asgn, used_b = joint_budget_oracle_dp(s["L"], K_SET, cap)
            total_loss += best_loss
            total_chunks += N
            total_kept += used_b
            oracle_assignments[tgt].append({
                "sample_id": s["sample_id"],
                "dataset": s["dataset"],
                "n_chunks": N,
                "assignment_k": [K_SET[i] for i in asgn],
                "used_budget": used_b,
                "loss": float(best_loss / max(1, N)),
            })
        avg_loss = total_loss / max(1, total_chunks)
        avg_kept = total_kept / max(1, total_chunks)
        oracle_curve[str(tgt)] = {
            "target_avg_k": tgt,
            "achieved_mean_k": avg_kept,
            "mean_chunk_loss": avg_loss,
        }
        # nearest fixed-k for comparison
        nearest_fixed = min(K_SET, key=lambda k: abs(k - tgt))
        delta = fixed_per_k[nearest_fixed] - avg_loss
        print(f"  target_k={tgt:3d}: oracle_loss={avg_loss:.4f}  "
              f"vs fixed_k={nearest_fixed} ({fixed_per_k[nearest_fixed]:.4f})  "
              f"Δ=+{delta:.4f}")
    figure1_curves["oracle"] = oracle_curve

    # ============================================================
    # 3. Per-dataset oracle k distribution at a chosen budget (Table 6)
    # ============================================================
    PRETTY = {"TriviaQA-web": "TriviaQA", "NaturalQuestionsShort": "NQ",
              "DuoRC.ParaphraseRC": "DuoRC", "RelationExtraction": "RE"}
    # Use the mid-range budget tgt=8 for the per-dataset breakdown
    tgt = 8 if 8 in budgets else budgets[len(budgets) // 2]
    per_ds_k_lists = defaultdict(list)
    for entry in oracle_assignments[tgt]:
        ds = PRETTY.get(entry["dataset"], entry["dataset"])
        per_ds_k_lists[ds].extend(entry["assignment_k"])
    per_ds_stats = {}
    for ds, ks in per_ds_k_lists.items():
        arr = np.array(ks)
        per_ds_stats[ds] = {
            "n_chunks": int(arr.size),
            "mean_k": float(arr.mean()),
            "median_k": float(np.median(arr)),
            "std_k": float(arr.std()),
            "hist": {str(k): int((arr == k).sum()) for k in K_SET},
        }
    print(f"\n[Table 6] per-dataset oracle k distribution @ target_k={tgt}:")
    for ds, st in sorted(per_ds_stats.items(),
                         key=lambda kv: -kv[1]["mean_k"]):
        print(f"  {ds:<14} n={st['n_chunks']:5d} "
              f"mean_k={st['mean_k']:.2f}  std={st['std_k']:.2f}")

    # ============================================================
    # 4. Save everything
    # ============================================================
    out_summary = {
        "n_chunks_total": len(records),
        "n_samples": len(samples),
        "K_set": K_SET,
        "figure2b_min_k_hist": figure2b,
        "figure1_curves": figure1_curves,
        "table6_per_dataset": per_ds_stats,
        "target_for_table6": tgt,
    }
    out_path = os.path.join(args.out_dir, "summary.json")
    json.dump(out_summary, open(out_path, "w"), indent=2, ensure_ascii=False)
    print(f"\n[saved] {out_path}")

    # Save per-sample oracle assignments for further analysis
    asn_path = os.path.join(args.out_dir, "oracle_assignments.json")
    json.dump({str(k): v for k, v in oracle_assignments.items()},
              open(asn_path, "w"), indent=2, ensure_ascii=False)
    print(f"[saved] {asn_path}")


if __name__ == "__main__":
    main()
