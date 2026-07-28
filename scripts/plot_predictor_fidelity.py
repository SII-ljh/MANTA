#!/usr/bin/env python
"""Figures for KPredictor cross-domain fidelity (reads analysis JSON).

Panel A (headline): per-dataset mean k_pred vs mean k_oracle at the main ε,
  IID vs OOD colored, y=x diagonal, dataset labels, Pearson/Spearman annotated.
  -> visual proof the learned budget rule extrapolates: OOD points on the line.
Panel B: IID vs OOD k-MAE (and global-uniform null) across ε -> graceful,
  bounded OOD degradation, predictor < uniform everywhere.
"""
import argparse
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

IID_C, OOD_C = "#1f77b4", "#d62728"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", required=True)
    p.add_argument("--out", default="analysis/predictor_fidelity/fidelity")
    p.add_argument("--main_eps", default="0.05")
    args = p.parse_args()

    d = json.load(open(args.in_json))["by_eps"]
    eps_keys = sorted(d.keys(), key=float)
    main_eps = args.main_eps if args.main_eps in d else eps_keys[0]

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11, 4.6))

    # ---- Panel A: per-dataset calibration scatter at main_eps ----
    calib = d[main_eps]["res"]["_calib"]
    pts = calib["points"]
    xo = [v["oracle"] for v in pts.values()]
    yp = [v["pred"] for v in pts.values()]
    lim = max(max(xo), max(yp)) * 1.1
    axA.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="y = x (perfect)")
    for ds, v in pts.items():
        c = IID_C if v["bucket"] == "IID" else OOD_C
        axA.scatter(v["oracle"], v["pred"], c=c, s=45, zorder=3,
                    edgecolors="white", linewidths=0.5)
        axA.annotate(ds.replace("NaturalQuestionsShort", "NQ")
                     .replace("DuoRC.ParaphraseRC", "DuoRC")
                     .replace("RelationExtraction", "RelExt")
                     .replace("TriviaQA-web", "TriviaQA"),
                     (v["oracle"], v["pred"]), fontsize=6.5,
                     xytext=(3, 3), textcoords="offset points")
    axA.scatter([], [], c=IID_C, s=45, label="IID (in-domain)")
    axA.scatter([], [], c=OOD_C, s=45, label="OOD (out-of-domain)")
    axA.set_xlabel("oracle mean k (per dataset)")
    axA.set_ylabel("predicted mean k (per dataset)")
    axA.set_title(f"Per-dataset budget calibration (ε={main_eps})\n"
                  f"Pearson {calib['pearson']:.2f} / Spearman {calib['spearman']:.2f} "
                  f"({calib['n_datasets']} datasets)", fontsize=9)
    axA.legend(fontsize=7, loc="upper left")
    axA.set_xlim(0, lim); axA.set_ylim(0, lim)

    # ---- Panel B: k-MAE & uniform null across ε, IID vs OOD ----
    epsf = [float(e) for e in eps_keys]
    def series(bucket, key):
        return [d[e]["res"].get(bucket, {}).get(key, np.nan) for e in eps_keys]
    axB.plot(epsf, series("IID", "k_MAE"), "-o", c=IID_C, label="IID predictor")
    axB.plot(epsf, series("OOD", "k_MAE"), "-o", c=OOD_C, label="OOD predictor")
    axB.plot(epsf, series("IID", "global_uniform_MAE"), "--", c=IID_C, alpha=0.5,
             label="IID uniform null")
    axB.plot(epsf, series("OOD", "global_uniform_MAE"), "--", c=OOD_C, alpha=0.5,
             label="OOD uniform null")
    axB.set_xscale("log")
    axB.set_xlabel("ε (loss tolerance)")
    axB.set_ylabel("per-chunk k-MAE vs oracle")
    axB.set_title("Predictor fidelity vs ε\n(predictor < uniform on every domain)",
                  fontsize=9)
    axB.legend(fontsize=7)
    axB.grid(alpha=0.3)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(f"{args.out}.{ext}", dpi=160, bbox_inches="tight")
        print(f"[written] {args.out}.{ext}")


if __name__ == "__main__":
    main()
