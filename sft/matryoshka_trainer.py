"""Matryoshka SFT trainer.

Three-stage training for SAC + Matryoshka anchor truncation + monotonic gate:
  stage 1: Matryoshka only      (use_matryoshka=True, use_gate=False)
  stage 2: Matryoshka + gate    (use_gate=True, lambda_budget=0, entropy_weight=0)
  stage 3: + budget + entropy   (lambda_budget ramps 0→λ_max, entropy 0→ε_max)

Stage and ramp schedule are read from config["matryoshka_training_config"]:
  {
    "stage": 1 | 2 | 3,
    "total_steps": int,
    "init_adapter_path": str,
    "lambda_budget_max": float,
    "entropy_weight_max": float,
    "ramp_steps": int                 # over how many steps to ramp (stage 3)
  }

WandB metrics added on top of base trainer:
  - matryoshka/k_sampled        (per-step)
  - per-k running lm_loss       (rolling, by sampled k)
  - gate/mean_h, gate/E_sum_g, gate/N_kept   (when use_gate)
  - loss/budget_loss, loss/entropy           (when active)
"""
import argparse
import json
import logging
import os
import random
import sys
import time
from collections import defaultdict, deque

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.parallel as parallel_module  # noqa
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from path_config import BASE_PATH
sys.path.append(BASE_PATH)

from model.modeling import get_model, save_adapter, load_adapter
from util.utils import setup, count_parameters, get_wsd_scheduler, training_step

import wandb


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[logging.FileHandler('log.txt', mode='w'), logging.StreamHandler()],
)
_BASE_SEED = int(os.environ.get("PACT_BASE_SEED", "12345"))
torch.manual_seed(_BASE_SEED)
np.random.seed(_BASE_SEED)
random.seed(_BASE_SEED)


def _reseed_for_rank(rank: int) -> None:
    """Per-rank reseed so DDP workers sample different matryoshka k / ε."""
    s = _BASE_SEED + int(rank)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    np.random.seed(s)
    random.seed(s)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--work_dir', required=True)
    p.add_argument('--port', default='14527')
    p.add_argument('--wandb_mode', default='disabled', choices=['online', 'offline', 'disabled'])
    p.add_argument('--wandb_project', default='sac-matryoshka')
    p.add_argument('--wandb_run_name', default=None)
    p.add_argument('--data_mode', required=True, choices=['sft', 'pretrain'],
                   help='sft: instruction QA (Stage 3). pretrain: raw text PG-19 (Stage 2 in v5 plan B). '
                        'Required to avoid silent fall-through to SFT.')
    return p.parse_args()


_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def import_data_modules(data_mode: str):
    """Return (get_examples, get_dataset) for the chosen data path.

    sft      -> sft/instruction_*    (variable-length QA samples, bs_per_device=1)
    pretrain -> pretrain/pre_*       (raw text prefix→LM samples, bs_per_device=1)
    """
    if data_mode == 'pretrain':
        pretrain_dir = os.path.join(_PROJECT_ROOT, 'pretrain')
        if pretrain_dir not in sys.path:
            sys.path.insert(0, pretrain_dir)
        from pre_prepare_data import get_examples as _get_examples
        from pre_dataloader import get_dataset as _get_dataset
        return _get_examples, _get_dataset
    sft_dir = os.path.join(_PROJECT_ROOT, 'sft')
    if sft_dir not in sys.path:
        sys.path.insert(0, sft_dir)
    from instruction_prepare_data import get_examples as _get_examples
    from instruction_dataloader import get_dataset as _get_dataset
    return _get_examples, _get_dataset


def _apply_ramp(step: int, ramp_steps: int, max_value: float) -> float:
    if ramp_steps <= 0 or max_value == 0.0:
        return max_value
    return max_value * min(1.0, step / float(ramp_steps))


def train(rank, args, world_size):
    with open(os.path.join(args.work_dir, "config.json")) as f:
        config = json.load(f)

    setup(rank, world_size, args.port)
    torch.cuda.set_device(rank)
    _reseed_for_rank(rank)

    matryoshka_config = config["matryoshka_training_config"]
    # data_mode=pretrain → read pretrain_* config sections (raw text PG-19);
    # data_mode=sft → read sft_* sections (variable-length QA).
    if args.data_mode == 'pretrain':
        training_config = config["pretrain_training_config"]
        task_config = config["pretrain_task_config"]
    else:
        training_config = config["sft_training_config"]
        task_config = config["sft_task_config"]
    config["data_config"]["model_id"] = training_config["model_id"]
    # data_config kwargs differ between pretrain/sft get_examples signatures.
    if args.data_mode == 'pretrain':
        # pretrain get_examples needs output_dir (where the precomputed .pt sits)
        config["data_config"].setdefault(
            "output_dir", os.path.join(_PROJECT_ROOT, "pretrain", "output")
        )
        config["data_config"].pop("max_context_tokens", None)
    else:
        config["data_config"].pop("output_dir", None)
    get_examples, get_dataset = import_data_modules(args.data_mode)

    # Inject Matryoshka task flags (overridable per stage).
    task_config["use_matryoshka"] = matryoshka_config.get("use_matryoshka", True)
    task_config["matryoshka_ks"] = matryoshka_config.get(
        "matryoshka_ks", [2, 4, 8, 16, 32, task_config["mem_size"]]
    )
    task_config["matryoshka_weights"] = matryoshka_config.get(
        "matryoshka_weights", [4.0, 3.0, 2.0, 1.5, 1.0, 1.0]
    )
    task_config["use_gate"] = matryoshka_config.get("use_gate", False)
    task_config["gate_init_bias"] = matryoshka_config.get("gate_init_bias", 5.0)
    # v4 fields (default values preserve v3 cumprod behavior).
    task_config["gate_mode"] = matryoshka_config.get("gate_mode", "cumprod")
    task_config["sparsity_weight"] = matryoshka_config.get("sparsity_weight", 0.0)
    task_config["target_avg"] = matryoshka_config.get("target_avg", 12.0)
    task_config["elastic_tau"] = matryoshka_config.get("elastic_tau", 1.0)
    task_config["gumbel_tau"] = matryoshka_config.get("gumbel_tau", 1.0)
    # Budget / entropy are set per-step below via the ramp schedule.
    task_config["lambda_budget"] = 0.0
    task_config["entropy_weight"] = 0.0
    task_config["n_target"] = matryoshka_config.get("n_target", 12.0)

    # PACT v5 K-Predictor knobs (forwarded into task_config so model picks them up).
    task_config["use_k_predictor"] = matryoshka_config.get("use_k_predictor", False)
    task_config["k_choices"] = matryoshka_config.get("k_choices", [1, 2, 4, 8, 16, 34])
    task_config["eps_grid"] = matryoshka_config.get("eps_grid", [0.01, 0.05, 0.1, 0.2])
    task_config["predictor_warmup_skip"] = matryoshka_config.get("predictor_warmup_skip", 5000)
    task_config["alpha_max"] = matryoshka_config.get("alpha_max", 1.0)
    # Note: matryoshka_config["ramp_steps"] is the legacy gate/budget ramp; the
    # KPredictor uses its own ramp_steps key (forwarded if present, else falls
    # back to matryoshka_config["ramp_steps"]).
    task_config["ramp_steps"] = matryoshka_config.get(
        "predictor_ramp_steps", matryoshka_config.get("ramp_steps", 5000)
    )
    task_config["predictor_hidden"] = matryoshka_config.get("predictor_hidden", 256)
    task_config["predictor_attn_heads"] = matryoshka_config.get("predictor_attn_heads", 4)
    task_config["predictor_d_layer_low"] = matryoshka_config.get("predictor_d_layer_low", 4)
    task_config["predictor_eps_embed_dim"] = matryoshka_config.get("predictor_eps_embed_dim", 64)
    task_config["predictor_pos_weight"] = matryoshka_config.get("predictor_pos_weight", 4.0)
    task_config["predictor_lr"] = matryoshka_config.get("predictor_lr", None)
    task_config["oracle_context_mode"] = matryoshka_config.get("oracle_context_mode", "selfref")
    # KPredictor keep-prob parameterisation (A3 ablation): "monotone" (cumprod
    # survival, default == main) or "independent" (per-slot sigmoid).
    task_config["predictor_head_mode"] = matryoshka_config.get("predictor_head_mode", "monotone")
    # All-k matryoshka loss + fixed-ε ablation knobs (default off -> behaviour
    # identical to the sampled main run).
    task_config["matryoshka_all_k"] = matryoshka_config.get("matryoshka_all_k", False)
    task_config["predictor_fixed_eps"] = matryoshka_config.get("predictor_fixed_eps", None)
    task_config["oracle_relative_eps"] = matryoshka_config.get("oracle_relative_eps", False)
    task_config["oracle_eps_floor"] = matryoshka_config.get("oracle_eps_floor", 1e-3)
    # PACT naive-Gate ablation (T8): "epsilon" default vs "naive" (global
    # fixed-budget top-k, no eps tolerance) + the matched global budget.
    task_config["predictor_gate_mode"] = matryoshka_config.get("predictor_gate_mode", "epsilon")
    task_config["naive_global_budget"] = matryoshka_config.get("naive_global_budget", None)
    # KV-aware predictor redesign (2026-06-29). predictor_arch is a preset that
    # expands to the composable flags below ("default" | "kv_marginal"); explicit
    # flags override. All default to the original v4 behaviour.
    task_config["predictor_arch"] = matryoshka_config.get("predictor_arch", "default")
    task_config["predictor_feat_mode"] = matryoshka_config.get("predictor_feat_mode", None)
    task_config["predictor_use_residual_energy"] = matryoshka_config.get("predictor_use_residual_energy", None)
    task_config["predictor_slot_mixer"] = matryoshka_config.get("predictor_slot_mixer", "mha")
    task_config["predictor_eps_mode"] = matryoshka_config.get("predictor_eps_mode", "add")
    task_config["predictor_eps_monotone"] = matryoshka_config.get("predictor_eps_monotone", False)
    # Predictor-only training (frozen v4 encoder): freeze every non-predictor
    # param via LR=0 (NOT requires_grad=False, so save_adapter still writes a
    # self-contained adapter = v4 encoder + freshly-trained predictor). The
    # predictor is loaded fresh (its keys are excluded from the init adapter).
    task_config["train_predictor_only"] = matryoshka_config.get("train_predictor_only", False)
    # Within-sample (across-chunk) ranking loss on the predictor (2026-06-30).
    # default 0.0 == original pointwise BCE only. >0 adds a mean/scale-invariant
    # ranking term that directly optimises per-chunk RELATIVE allocation (the
    # within-sample Spearman that collapses under pointwise BCE — F101/F102).
    task_config["predictor_rank_loss_weight"] = matryoshka_config.get("predictor_rank_loss_weight", 0.0)
    task_config["predictor_rank_loss_mode"] = matryoshka_config.get("predictor_rank_loss_mode", "corr")
    # Full fine-tuning variant (no LoRA): train the whole encoder base directly.
    # Default False -> LoRA behaviour unchanged. The model stays bf16 for the
    # forward/backward (activations bf16, fits like the LoRA run), but the
    # optimizer keeps fp32 MASTER copies of the trainable params (see below) so
    # the small full-FT updates (lr~1e-5) don't underflow bf16 — the standard
    # bf16-compute + fp32-master mixed-precision recipe (matches HF Trainer).
    task_config["full_finetune"] = matryoshka_config.get("full_finetune", False)

    stage = int(matryoshka_config["stage"])
    init_adapter_path = matryoshka_config["init_adapter_path"]
    total_steps_override = matryoshka_config.get("total_steps", None)
    ramp_steps = int(matryoshka_config.get("ramp_steps", 0))
    lambda_budget_max = float(matryoshka_config.get("lambda_budget_max", 0.0))
    entropy_weight_max = float(matryoshka_config.get("entropy_weight_max", 0.0))

    if rank == 0:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name or f"matryoshka_stage{stage}",
            mode=args.wandb_mode,
            config={**config, "stage": stage},
        )
        logging.info(f"[Matryoshka] stage={stage} init_from={init_adapter_path}")
        logging.info(f"[Matryoshka] use_gate={task_config['use_gate']} "
                     f"λ_max={lambda_budget_max} ε_max={entropy_weight_max} ramp={ramp_steps}")

    train_examples, _ = get_examples(**config["data_config"])

    base_steps = len(train_examples) // training_config["total_batch_size"]
    training_steps = base_steps if total_steps_override is None else min(base_steps, total_steps_override)
    train_examples = train_examples[:training_steps * training_config["total_batch_size"]]
    if rank == 0:
        logging.info(f"[INFO] total_examples:{len(train_examples)} | training_steps:{training_steps}")

    example_num_per_gpu = len(train_examples) // training_config["device_count"]
    start_index = rank * example_num_per_gpu
    end_index = start_index + example_num_per_gpu
    train_examples = train_examples[start_index:end_index]
    logging.info(f"[INFO] rank{rank} examples[{start_index}:{end_index}]")

    model = get_model(training_config["model_id"], task_config, rank)
    if os.path.exists(init_adapter_path):
        # Predictor-only retrain: keep the freshly-initialised (possibly new-arch)
        # predictor instead of loading the v4 predictor weights (which may have a
        # different shape and would otherwise crash load_state_dict).
        _excl = ("k_predictor.",) if task_config.get("train_predictor_only", False) else ()
        model = load_adapter(model, save_path_and_name=init_adapter_path,
                             log=(rank == 0), exclude_prefixes=_excl)
        if rank == 0:
            logging.info(f"[INFO] loaded init adapter from {init_adapter_path}"
                         f"{' (predictor excluded, fresh init)' if _excl else ''}")
    else:
        if rank == 0:
            logging.warning(f"[WARN] init adapter not found: {init_adapter_path}; training from scratch")

    if rank == 0:
        count_parameters(model, config)

    ddp_model = DDP(model, device_ids=[rank], find_unused_parameters=True)
    dataset = get_dataset(task_config["task_type"], train_examples, training_config['batch_size_per_device'])
    loader = DataLoader(dataset, batch_size=None)

    # Optional separate LR for the KPredictor params (diagnostic lever
    # 2026-06-13). When matryoshka_config["predictor_lr"] is set, the predictor
    # gets its own param group at that LR while the rest of the model keeps
    # training_config["learning_rate"]. Default (None) -> single group, no-op.
    #
    # train_predictor_only (2026-06-29): freeze every non-predictor param by
    # pinning its LR to 0 (keeps requires_grad=True so save_adapter still writes
    # the v4 encoder into a self-contained adapter). The predictor trains at
    # predictor_lr (or the base LR if unset). This is the screen's frozen-v4 path.
    _pred_lr = matryoshka_config.get("predictor_lr", None)
    _predictor_only = bool(task_config.get("train_predictor_only", False))
    _full_ft = bool(task_config.get("full_finetune", False))
    # full_finetune takes precedence: it needs the fp32-master optimizer (below),
    # so it must not fall into the predictor_lr/predictor_only multi-group branch
    # (which would leave _ft_master_params undefined and crash the step loop).
    if (not _full_ft) and (_pred_lr is not None or _predictor_only):
        _rest_lr = 0.0 if _predictor_only else training_config["learning_rate"]
        _pred_lr = float(_pred_lr) if _pred_lr is not None else float(training_config["learning_rate"])
        _pred_params, _rest_params = [], []
        for _n, _p in ddp_model.named_parameters():
            if not _p.requires_grad:
                continue
            (_pred_params if "k_predictor." in _n else _rest_params).append(_p)
        logging.info(f"[INFO] predictor_only={_predictor_only} rest_lr={_rest_lr} "
                     f"predictor LR={_pred_lr} "
                     f"(predictor params={len(_pred_params)}, rest={len(_rest_params)})")
        optimizer = torch.optim.AdamW(
            [
                {"params": _rest_params, "lr": _rest_lr},
                {"params": _pred_params, "lr": _pred_lr},
            ],
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
    elif _full_ft:
        # Full fine-tuning: bf16 params for forward/backward (memory), but the
        # optimizer holds fp32 MASTER copies of the trainable params so lr~1e-5
        # updates don't underflow bf16. Grads are copied bf16->fp32 before each
        # step and the updated masters copied fp32->bf16 after (see loop below).
        _ft_bf16_params = [p for p in ddp_model.parameters() if p.requires_grad]
        # ZeRO-1 optimizer-state sharding (flag-gated, default OFF = replicated
        # masters exactly as before). Each rank keeps fp32 masters + AdamW state
        # ONLY for params it owns; updated bf16 params are broadcast from their
        # owner after each step. Math identical to the replicated path (same
        # grads, same clip coef, same AdamW update) — only WHERE each update is
        # computed changes. Safe with weights-only resume (masters rebuilt from
        # loaded bf16 weights; optimizer state is never checkpointed).
        _zero1 = bool(matryoshka_config.get("zero1_optimizer", False)) \
            and torch.distributed.is_available() and torch.distributed.is_initialized() \
            and torch.distributed.get_world_size() > 1
        if _zero1:
            _z_ws = torch.distributed.get_world_size()
            _z_rk = torch.distributed.get_rank()
            # greedy numel-balanced ownership (deterministic across ranks)
            _z_order = sorted(range(len(_ft_bf16_params)),
                              key=lambda i: (-_ft_bf16_params[i].numel(), i))
            _z_load = [0] * _z_ws
            _ft_owner = [0] * len(_ft_bf16_params)
            for _i in _z_order:
                _r = _z_load.index(min(_z_load))
                _ft_owner[_i] = _r
                _z_load[_r] += _ft_bf16_params[_i].numel()
            _ft_own_idx = [_i for _i, _o in enumerate(_ft_owner) if _o == _z_rk]
            assert _ft_own_idx, f"zero1: rank {_z_rk} owns no params (n={len(_ft_bf16_params)})"
            _ft_master_params = [
                _ft_bf16_params[_i].detach().to(torch.float32).requires_grad_(True)
                for _i in _ft_own_idx
            ]
            logging.info(f"[INFO] full_finetune+zero1: rank {_z_rk} owns "
                         f"{len(_ft_own_idx)}/{len(_ft_bf16_params)} tensors "
                         f"({_z_load[_z_rk]/1e6:.1f}M/{sum(_z_load)/1e6:.1f}M params)")
        else:
            _ft_master_params = [
                p.detach().to(torch.float32).requires_grad_(True) for p in _ft_bf16_params
            ]
            logging.info(f"[INFO] full_finetune: fp32 master over {len(_ft_master_params)} "
                         f"trainable tensors (bf16 compute, fp32 optimizer)")
        optimizer = torch.optim.AdamW(
            _ft_master_params,
            lr=training_config["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
    else:
        optimizer = torch.optim.AdamW(
            ddp_model.parameters(),
            lr=training_config["learning_rate"],
            betas=(0.9, 0.95),
            weight_decay=0.1,
        )
    _lr_warmup = int(training_config.get("lr_warmup_steps", 300))
    # Full-FT weights-only resume: the LR warmup already completed in the original
    # run (first ckpt lands at save_step > warmup), so don't restart it on reclaim
    # (would materially change the schedule). Also keeps S>0 if remaining<warmup.
    if task_config.get("full_finetune", False) and int(matryoshka_config.get("ckpt_step_offset", 0)) > 0:
        _lr_warmup = 0
    scheduler = get_wsd_scheduler(optimizer, training_steps, warmup_steps=_lr_warmup)

    accumulation_steps = training_config["gradient_accumulation_steps"]
    # Resume support: cumulative step offset when continuing from a step-tagged
    # ckpt (set by the launcher). Keeps predictor warmup/alpha-ramp and ckpt
    # naming on the cumulative timeline so a reclaim costs <= one save interval.
    step_offset = int(matryoshka_config.get("ckpt_step_offset", 0))
    step_num = 0
    optimizer.zero_grad()
    ddp_model.train()

    info_list = []
    start_time = time.time()

    # Per-k running loss (deque last 50 samples per k).
    per_k_loss = defaultdict(lambda: deque(maxlen=50))

    def save():
        if rank != 0:
            return
        with open(os.path.join(args.work_dir, "output/instruction_info.json"), 'w') as f:
            json.dump(info_list, f, indent=4)
        with open(os.path.join(args.work_dir, "output/config.json"), 'w') as f:
            json.dump(config, f, indent=4)
        save_adapter(ddp_model.module, save_path_and_name=os.path.join(args.work_dir, "output/instruction_adapter.pt"))
        # Tag with the number of optimizer steps actually completed, cumulative
        # across resumes: save() runs after the update for microbatch step_num,
        # so step_num // accumulation_steps windows have closed. The launcher
        # reads this number back as "steps already done" (launch/train_manta.sh).
        _cum = step_offset + step_num // accumulation_steps
        save_adapter(ddp_model.module, save_path_and_name=os.path.join(args.work_dir, "output/instruction_adapter_step%d.pt" % _cum))

    if rank == 0:
        progress_bar = tqdm(total=training_steps * accumulation_steps)

    for inputs in loader:
        step_num += 1
        optimizer_step = (step_num % accumulation_steps == 0)
        # Stable across all microbatches in one accumulation window: each window
        # of size `accumulation_steps` shares one global_step. Without -1 the
        # boundary microbatch flips warmup_skip / alpha mid-window.
        global_step = (step_num - 1) // accumulation_steps

        # Update ramps on the live module (DDP wraps it).
        live_task_cfg = ddp_model.module.task_config
        live_task_cfg["lambda_budget"] = _apply_ramp(global_step, ramp_steps, lambda_budget_max)
        live_task_cfg["entropy_weight"] = _apply_ramp(global_step, ramp_steps, entropy_weight_max)
        # PACT v5: push step counter to the model so it can gate predictor warmup.
        ddp_model.module._global_step = step_offset + global_step

        if optimizer_step:
            loss_info = training_step(ddp_model, inputs, rank, accumulation_steps)
        else:
            with ddp_model.no_sync():
                loss_info = training_step(ddp_model, inputs, rank, accumulation_steps)

        # Track per-k loss. For PACT v5 KPredictor branch the per-sample matryoshka
        # loss is "m_loss"; legacy LM path uses "lm_loss". Either may be present.
        k_sampled = loss_info.get("matryoshka_k")
        m_loss_val = loss_info.get("m_loss", loss_info.get("lm_loss"))
        if k_sampled is not None and m_loss_val is not None:
            per_k_loss[k_sampled].append(m_loss_val)

        info_list.append({
            "run_time(hours)": (time.time() - start_time) / 3600,
            "total_steps": training_steps,
            "steps": step_num / accumulation_steps,
            "training_loss": loss_info,
            "learning_rate": optimizer.param_groups[0]['lr'],
        })

        if rank == 0 and optimizer_step:
            log_dict = {
                "run_time(hours)": (time.time() - start_time) / 3600,
                "steps": step_num / accumulation_steps,
                "lr": optimizer.param_groups[0]['lr'],
                "lm_loss": loss_info.get("lm_loss"),
                "matryoshka_k": loss_info.get("matryoshka_k"),
                "lambda_budget": live_task_cfg["lambda_budget"],
                "entropy_weight": live_task_cfg["entropy_weight"],
            }
            if "budget_loss" in loss_info:
                log_dict["budget_loss"] = loss_info["budget_loss"]
            if "entropy" in loss_info:
                log_dict["entropy"] = loss_info["entropy"]
            for key in ("gate_mean_h", "gate_E_sum_g", "gate_N_kept"):
                if key in loss_info:
                    log_dict[key] = loss_info[key]
            # PACT v5 KPredictor metrics
            for key in (
                "m_loss", "p_loss", "alpha", "warmup_phase",
                "eps_mean", "L_full_mean",
                "oracle_mean_k", "oracle_per_chunk_var",
                "pred_mean_k", "oracle_relative", "oracle_floor",
            ):
                if key in loss_info:
                    log_dict[key] = loss_info[key]
            # All-k mode writes m_loss_k1 / m_loss_k2 / ... — forward them all.
            for key, val in loss_info.items():
                if key.startswith("m_loss_k"):
                    log_dict[key] = val
            # Periodic per-k aggregates
            if global_step % 50 == 0:
                for k, dq in per_k_loss.items():
                    if dq:
                        log_dict[f"lm_loss_k{k}"] = float(np.mean(dq))
            wandb.log(log_dict, step=global_step)

        if optimizer_step:
            if _full_ft and _zero1:
                # ZeRO-1 path. Global clip coef from ALL bf16 grads in fp32 —
                # the same values clip_grad_norm_ sees on the replicated path
                # (master grads there are exact fp32 casts of these bf16 grads).
                _sq = None
                for _p in _ft_bf16_params:
                    if _p.grad is not None:
                        _s = _p.grad.detach().to(torch.float32).pow(2).sum()
                        _sq = _s if _sq is None else _sq + _s
                _coef = 1.0
                if _sq is not None:
                    _total_norm = float(_sq.sqrt())
                    _coef = float(training_config["max_grad_norm"]) / (_total_norm + 1e-6)
                    if _coef > 1.0:
                        _coef = 1.0
                # owned bf16 grads -> fp32 master grads (pre-scaled by clip coef)
                for _mp, _i in zip(_ft_master_params, _ft_own_idx):
                    _g = _ft_bf16_params[_i].grad
                    _mp.grad = _g.to(torch.float32).mul_(_coef) if _g is not None else None
                optimizer.step()
                scheduler.step()
                with torch.no_grad():
                    # owned fp32 masters -> owned bf16 params, then sync all
                    # bf16 params from their owner rank.
                    for _mp, _i in zip(_ft_master_params, _ft_own_idx):
                        _ft_bf16_params[_i].copy_(_mp)
                    for _i, _p in enumerate(_ft_bf16_params):
                        torch.distributed.broadcast(_p.data, src=_ft_owner[_i])
                optimizer.zero_grad(set_to_none=True)
                for _p in _ft_bf16_params:
                    _p.grad = None
            elif _full_ft:
                # bf16 grads (DDP-synced during backward) -> fp32 master grads.
                for _mp, _p in zip(_ft_master_params, _ft_bf16_params):
                    _mp.grad = _p.grad.to(torch.float32) if _p.grad is not None else None
                torch.nn.utils.clip_grad_norm_(_ft_master_params, training_config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                # fp32 master -> bf16 params (what the forward/DDP and save use).
                with torch.no_grad():
                    for _mp, _p in zip(_ft_master_params, _ft_bf16_params):
                        _p.copy_(_mp)
                optimizer.zero_grad(set_to_none=True)
                for _p in _ft_bf16_params:
                    _p.grad = None
            else:
                torch.nn.utils.clip_grad_norm_(ddp_model.parameters(), training_config["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

        if step_num % (training_config["log_step"] * accumulation_steps) == 0 and rank == 0:
            logging.info(info_list[-1])

        save_every = training_config["save_step"] * accumulation_steps
        if save_every > 0 and step_num % save_every == 0:
            save()

        if rank == 0:
            progress_bar.update(1)

    if rank == 0:
        progress_bar.close()
    save()


if __name__ == "__main__":
    args = parse_args()
    world_size = torch.cuda.device_count()
    mp.spawn(train, args=(args, world_size), nprocs=world_size, join=True)
