"""Generate config.json for three P0-B ablations.

All ablations follow the v4a_comi recipe:
  - COMI 100K (5 datasets x 20K), max_context_tokens=6000
  - 12500 steps, lr=3e-5, bs=16, mem_size=34
  - matryoshka_ks=[2,4,8,16,32,34], gate_mode=cross_chunk_softmax

The pieces that differ per ablation:
  - B.1 clean_baseline_comi    : init from SAC vanilla pretrain.
                                  use_matryoshka=False, matryoshka_ks=[34],
                                  use_gate=False, gate_mode="none".
  - B.2 pact_matryoshka_only_comi : init from Matryoshka pretrain (v3 stage-1).
                                     use_matryoshka=True, matryoshka_ks=[2,4,8,16,32,34],
                                     use_gate=False, gate_mode="none".
  - B.3 pact_gate_only_comi    : init from Matryoshka pretrain (v3 stage-1, same as v4a_comi).
                                  use_matryoshka=False, matryoshka_ks=[34],
                                  use_gate=True, gate_mode="cross_chunk_softmax", target_avg=12.
"""
import json
import os
from copy import deepcopy

ROOT = os.environ.get("MANTA_ROOT", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_CONFIG = os.path.join(ROOT, "experiment/pact_v4a_comi/config.json")

with open(SRC_CONFIG) as f:
    BASE = json.load(f)

# Drop the runtime-only fields so the new configs start fresh.
for k in ("Total_parameters", "Trainable_parameters",
          "Embedding_parameters", "non-Embedding_parameters"):
    BASE.pop(k, None)


SAC_PRETRAIN_CKPT = os.path.join(ROOT, "weights/sac_8b_paper/output/adapter.pt")
MAT_PRETRAIN_CKPT = os.path.join(ROOT, "experiment/pact_v3/output/adapter.pt")
TOTAL_STEPS = 12500


def make_config(exp_name: str,
                use_matryoshka: bool,
                use_gate: bool,
                init_adapter_path: str,
                matryoshka_ks=None,
                gate_mode="cross_chunk_softmax"):
    cfg = deepcopy(BASE)

    sft_task = cfg["sft_task_config"]
    sft_task["use_matryoshka"] = use_matryoshka
    sft_task["use_gate"] = use_gate
    if use_gate:
        sft_task["gate_mode"] = gate_mode
        sft_task["target_avg"] = 12.0
    else:
        sft_task["gate_mode"] = "none"
        sft_task["target_avg"] = 12.0
    sft_task["matryoshka_ks"] = matryoshka_ks or (
        [2, 4, 8, 16, 32, 34] if use_matryoshka else [34]
    )
    sft_task["matryoshka_weights"] = [1.0] * len(sft_task["matryoshka_ks"])

    mat = cfg["matryoshka_training_config"]
    mat["use_matryoshka"] = use_matryoshka
    mat["use_gate"] = use_gate
    mat["gate_mode"] = gate_mode if use_gate else "none"
    mat["matryoshka_ks"] = sft_task["matryoshka_ks"]
    mat["matryoshka_weights"] = sft_task["matryoshka_weights"]
    mat["init_adapter_path"] = init_adapter_path
    mat["total_steps"] = TOTAL_STEPS
    mat["target_avg"] = 12.0
    mat["gate_init_bias"] = 0.0

    exp_dir = os.path.join(ROOT, "experiment", exp_name)
    os.makedirs(os.path.join(exp_dir, "output"), exist_ok=True)
    with open(os.path.join(exp_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)
    print(f"[wrote] experiment/{exp_name}/config.json")
    print(f"        use_mat={use_matryoshka} use_gate={use_gate} "
          f"ks={sft_task['matryoshka_ks']} init={os.path.basename(init_adapter_path)}")


# B.1 — clean baseline (no Matryoshka, no gate, vanilla SAC pretrain init)
make_config(
    "clean_baseline_comi",
    use_matryoshka=False,
    use_gate=False,
    init_adapter_path=SAC_PRETRAIN_CKPT,
)

# B.2 — Matryoshka only (init from Matryoshka pretrain, no gate)
make_config(
    "pact_matryoshka_only_comi",
    use_matryoshka=True,
    use_gate=False,
    init_adapter_path=MAT_PRETRAIN_CKPT,
)

# B.3 — Gate only (Matryoshka pretrain init, single k=34 head, CCS gate)
make_config(
    "pact_gate_only_comi",
    use_matryoshka=False,
    use_gate=True,
    init_adapter_path=MAT_PRETRAIN_CKPT,
    matryoshka_ks=[34],
)

print("\nDone — 3 ablation configs ready.")
