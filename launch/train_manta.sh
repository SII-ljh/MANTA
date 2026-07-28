#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Two-stage MANTA training: Stage 1 (raw long text) -> Stage 2 (MRQA SFT).
#
# This is the main-line recipe of the paper: the encoder is fully fine-tuned,
# the decoder LLM stays frozen, the Matryoshka objective covers all k per step,
# and the tolerance is sampled from a grid so one model spans the whole
# compression range.
#
# Usage
#   bash launch/train_manta.sh                        # full run, all visible GPUs
#   DRYRUN=1 bash launch/train_manta.sh               # 50+50 step smoke test
#   MODEL_ID=meta-llama/Llama-3.2-3B EXP_TAG=manta_3b bash launch/train_manta.sh
#   CUDA_VISIBLE_DEVICES=0,1 bash launch/train_manta.sh
#
# Environment knobs
#   MODEL_ID      backbone (HF id or local path)      [meta-llama/Llama-3.2-1B]
#   EXP_TAG       experiment name prefix              [manta_1b]
#   EXP_ROOT      where experiment dirs are created   [<repo>/experiment]
#   STEPS         steps per stage                     [20000]
#   SAVE_STEP     step-tagged checkpoint interval     [2000]
#   S1_CONFIG/S2_CONFIG   config templates            [configs/manta_1b_stage{1,2}.json]
#   S1_EXTRA/S2_EXTRA     extra render_config.py args (e.g. --set ... for
#                         ablations); quoting is honoured, so a JSON value with
#                         spaces must be quoted: --set ...eps_grid='[0.05, 0.1]'
#   PORT_BASE     torch.distributed rendezvous port   [15200]
#   WANDB_MODE    online|offline|disabled             [offline]
#
# Each stage is idempotent: it is skipped once done, and a stage interrupted
# mid-run resumes from its newest step-tagged checkpoint (weights only; see
# docs/REPRODUCE.md for what resuming does and does not restore).
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_ID="${MODEL_ID:-meta-llama/Llama-3.2-1B}"
EXP_ROOT="${EXP_ROOT:-$REPO/experiment}"
LOG_DIR="${LOG_DIR:-$REPO/logs}"
S1_CONFIG="${S1_CONFIG:-$REPO/configs/manta_1b_stage1.json}"
S2_CONFIG="${S2_CONFIG:-$REPO/configs/manta_1b_stage2.json}"
S1_EXTRA="${S1_EXTRA:-}"
S2_EXTRA="${S2_EXTRA:-}"
PORT_BASE="${PORT_BASE:-15200}"
WANDB_MODE="${WANDB_MODE:-offline}"
DRYRUN="${DRYRUN:-0}"

if [ "$DRYRUN" = "1" ]; then
  EXP_TAG="${EXP_TAG:-manta_1b}_dryrun"
  STEPS="${STEPS:-50}"; SAVE_STEP="${SAVE_STEP:-50}"; WANDB_MODE=disabled
else
  EXP_TAG="${EXP_TAG:-manta_1b}"
  STEPS="${STEPS:-20000}"; SAVE_STEP="${SAVE_STEP:-2000}"
fi

# S1_DIR/S2_DIR can be set explicitly to reproduce the directory names the
# paper's aggregation scripts expect (docs/REPRODUCE.md lists the mapping).
S1_DIR="${S1_DIR:-$EXP_ROOT/${EXP_TAG}_stage1}"
S2_DIR="${S2_DIR:-$EXP_ROOT/${EXP_TAG}_stage2}"
mkdir -p "$LOG_DIR" "$S1_DIR/output" "$S2_DIR/output"
ulimit -n 65536 2>/dev/null || true
export PYTHONUNBUFFERED=1
export WANDB_MODE="$WANDB_MODE"

if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NGPU=$(echo "$CUDA_VISIBLE_DEVICES" | awk -F, '{print NF}')
else
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  [ "$NGPU" -ge 1 ] || { echo "[train] no GPU visible"; exit 1; }
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU - 1)))
fi

echo "[train] repo=$REPO"
echo "[train] backbone=$MODEL_ID tag=$EXP_TAG gpus=$CUDA_VISIBLE_DEVICES (n=$NGPU)"
echo "[train] steps/stage=$STEPS save_step=$SAVE_STEP dryrun=$DRYRUN"

newest_step() {  # highest step-tagged checkpoint in a work_dir, or empty
  ls "$1"/output/instruction_adapter_step*.pt 2>/dev/null \
    | sed -E 's/.*_step([0-9]+)\.pt/\1/' | sort -n | tail -1
}

run_stage() {  # name dir template init_default data_mode port log [extra args...]
  local name=$1 dir=$2 template=$3 init_default=$4 mode=$5 port=$6 log=$7
  shift 7
  local extra=("$@")

  if [ -f "$dir/.STAGE_DONE" ]; then
    echo "[train] skip $name (already done)"; return 0
  fi

  local done_steps init offset remaining
  done_steps=$(newest_step "$dir"); done_steps=${done_steps:-0}
  # Target already reached but never marked (e.g. the script was killed just
  # after the last save): adopt it instead of retraining from scratch.
  if [ "$done_steps" -ge "$STEPS" ] && [ -f "$dir/output/instruction_adapter.pt" ]; then
    touch "$dir/.STAGE_DONE"
    echo "[train] skip $name (already at step $done_steps)"; return 0
  fi
  if [ "$done_steps" -gt 0 ] && [ "$done_steps" -lt "$STEPS" ]; then
    init="$dir/output/instruction_adapter_step${done_steps}.pt"
    offset=$done_steps
    remaining=$((STEPS - done_steps))
    echo "[train] $name resuming at step $done_steps ($remaining to go)"
  else
    init="$init_default"; offset=0; remaining=$STEPS
    echo "[train] $name starting fresh (init=$init)"
  fi

  python "$REPO/launch/render_config.py" \
    --template "$template" --work_dir "$dir" \
    --model_id "$MODEL_ID" --device_count "$NGPU" \
    --total_steps "$remaining" --init "$init" \
    --save_step "$SAVE_STEP" --ckpt_step_offset "$offset" \
    ${extra[@]+"${extra[@]}"} \
    || { echo "[train] $name: config rendering failed"; exit 1; }

  ( cd "$REPO/sft" && python matryoshka_trainer.py \
      --work_dir "$dir" --data_mode "$mode" --port "$port" \
      --wandb_mode "$WANDB_MODE" --wandb_project manta \
      --wandb_run_name "${EXP_TAG}_${name}_${NGPU}gpu" 2>&1 | tee -a "$log" )
  local rc=${PIPESTATUS[0]}
  [ "$rc" -eq 0 ] || {
    echo "[train] $name failed (rc=$rc); rerun this script to resume"; exit 1; }

  # Completion is decided by the step count reached, not by the mere existence of
  # a final checkpoint: the trainer stops at min(len(data), total_steps), so a
  # dataset smaller than STEPS ends "successfully" after too few updates. Marking
  # that as done would make every rerun skip an under-trained stage.
  local final; final=$(newest_step "$dir"); final=${final:-0}
  if [ "$final" -ge "$STEPS" ]; then
    touch "$dir/.STAGE_DONE"
    echo "[train] $name done (step $final)"
  elif [ -f "$dir/output/instruction_adapter.pt" ]; then
    echo "[train] $name stopped at step $final of $STEPS — the prepared data ran"
    echo "[train]   out before the step target (trainer caps at min(len(data),"
    echo "[train]   total_steps)). Prepare more data, or lower STEPS. Not marking"
    echo "[train]   this stage done; the checkpoint is at $dir/output."
    exit 1
  else
    echo "[train] $name ended without a checkpoint"; exit 1
  fi
}

# S1_EXTRA / S2_EXTRA arrive as one string; split them the way the shell would
# split a command line, so quoted values survive intact:
#   S2_EXTRA="--set matryoshka_training_config.eps_grid='[0.05, 0.1]'"
# stays one argument instead of becoming three.
declare -a S1_ARGS S2_ARGS
eval "S1_ARGS=($S1_EXTRA)" || { echo "[train] cannot parse S1_EXTRA"; exit 1; }
eval "S2_ARGS=($S2_EXTRA)" || { echo "[train] cannot parse S2_EXTRA"; exit 1; }

run_stage stage1 "$S1_DIR" "$S1_CONFIG" SCRATCH pretrain \
  "$PORT_BASE" "$LOG_DIR/${EXP_TAG}_stage1.log" ${S1_ARGS[@]+"${S1_ARGS[@]}"}

S1_CKPT="$S1_DIR/output/instruction_adapter.pt"
[ -f "$S1_CKPT" ] || { echo "[train] missing Stage-1 checkpoint $S1_CKPT"; exit 1; }

run_stage stage2 "$S2_DIR" "$S2_CONFIG" "$S1_CKPT" sft \
  "$((PORT_BASE + 1))" "$LOG_DIR/${EXP_TAG}_stage2.log" ${S2_ARGS[@]+"${S2_ARGS[@]}"}

echo "[train] all done"
echo "[train]   stage 1: $S1_CKPT"
echo "[train]   stage 2: $S2_DIR/output/instruction_adapter.pt   <- evaluate this"
