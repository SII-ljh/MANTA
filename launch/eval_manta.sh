#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Evaluate a trained MANTA checkpoint on MRQA, in both operating modes:
#
#   1. fixed-k sweep   — the Matryoshka ladder, k in {1,2,4,8,16,32}
#   2. adaptive sweep  — the elastic allocator at each tolerance in the grid
#   3. completeness check on the number of scored samples
#   4. token-F1 / EM aggregation (the metric reported in the paper)
#
# Every step is skipped if its output already exists, so the script can be
# rerun after an interruption. Generations are written per rank and rescored
# on CPU, so the metric can be recomputed without re-running the model.
#
# Usage
#   EXP_DIR=experiment/manta_1b_stage2 \
#   EVAL_PT=sft/output/mrqa_eval_Llama-3.2-1B_320000samples_instruction.pt \
#   EVAL_META=data/mrqa_eval_full/qa_eval_meta.json \
#   bash launch/eval_manta.sh
#
# Environment knobs
#   EXP_DIR      work_dir holding output/instruction_adapter.pt   (required)
#   EVAL_PT      tokenized MRQA eval set                          (required)
#   EVAL_META    qa_eval_meta.json with the reference answers     (required)
#   TAG          label used in output file names                  [eval]
#   KS           fixed-k grid                                     [1,2,4,8,16,32]
#   EPS_LIST     tolerance grid for the adaptive mode             [0.001,0.01,0.05,0.1,0.2]
#   NUM_SAMPLES  0 = the whole eval set                           [0]
#   EXPECT_N     expected scored-sample count, 0 disables the check [0]
#   ADAPTER      checkpoint file name inside EXP_DIR/output       [instruction_adapter.pt]
#   REPORT_DIR   where the metric JSONs are written               [<repo>/reports]
# ---------------------------------------------------------------------------
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
: "${EXP_DIR:?set EXP_DIR to the work_dir of the checkpoint}"
: "${EVAL_PT:?set EVAL_PT to the tokenized eval set}"
: "${EVAL_META:?set EVAL_META to qa_eval_meta.json}"

# The tag names every output file. It defaults to the experiment directory so two
# checkpoints never write to the same report path (which the "skip if it exists"
# logic below would otherwise mistake for finished work).
TAG="${TAG:-$(basename "${EXP_DIR%/}")}"
KS="${KS:-1,2,4,8,16,32}"
EPS_LIST="${EPS_LIST:-0.001,0.01,0.05,0.1,0.2}"
NUM_SAMPLES="${NUM_SAMPLES:-0}"
EXPECT_N="${EXPECT_N:-0}"
ADAPTER="${ADAPTER:-instruction_adapter.pt}"
REPORT_DIR="${REPORT_DIR:-$REPO/reports}"
LOG_DIR="${LOG_DIR:-$REPO/logs}"

# The evaluators read the model/task settings from output/config.json, which the
# trainer writes next to the final checkpoint.
for p in "$EXP_DIR/output/$ADAPTER" "$EXP_DIR/output/config.json" \
         "$EVAL_PT" "$EVAL_META"; do
  [ -f "$p" ] || { echo "[eval] missing: $p"; exit 1; }
done
mkdir -p "$REPORT_DIR" "$LOG_DIR"

export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}
export RANK_RESUME=${RANK_RESUME:-1}
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ]; then
  NGPU=$(nvidia-smi -L 2>/dev/null | wc -l | tr -d ' ')
  [ "$NGPU" -ge 1 ] || { echo "[eval] no GPU visible"; exit 1; }
  export CUDA_VISIBLE_DEVICES=$(seq -s, 0 $((NGPU - 1)))
fi

KSWEEP="ksweep_${TAG}"
PARETO="pareto_${TAG}"
cd "$REPO"

# Outputs are reused only when they were produced by the same settings. Without
# this, changing KS / EPS_LIST / NUM_SAMPLES / ADAPTER under an existing tag would
# silently report the previous run's numbers.
MANIFEST="$EXP_DIR/output/${TAG}_eval_manifest.txt"
SETTINGS="adapter=$ADAPTER
eval_pt=$EVAL_PT
eval_meta=$EVAL_META
num_samples=$NUM_SAMPLES
ks=$KS
eps_list=$EPS_LIST"
if [ -f "$MANIFEST" ] && [ "$(cat "$MANIFEST")" != "$SETTINGS" ]; then
  echo "[eval] tag '$TAG' already has outputs from different settings:"
  diff <(cat "$MANIFEST") <(printf '%s\n' "$SETTINGS") | sed 's/^/[eval]   /'
  echo "[eval] use a different TAG, or delete $EXP_DIR/output/*_${TAG}_* first."
  exit 1
fi
printf '%s\n' "$SETTINGS" > "$MANIFEST"

echo "[eval] checkpoint : $EXP_DIR/output/$ADAPTER"
echo "[eval] eval set   : $EVAL_PT (num_samples=$NUM_SAMPLES)"
echo "[eval] gpus       : $CUDA_VISIBLE_DEVICES"

# ---- 1. fixed-k sweep (the Matryoshka ladder) ----
if [ ! -f "$EXP_DIR/output/${KSWEEP}_summary.json" ]; then
  echo "[eval] 1/4 fixed-k sweep k in {$KS}"
  python scripts/eval_k_sweep.py --work_dir "$EXP_DIR" \
    --eval_data_path "$EVAL_PT" --eval_meta_path "$EVAL_META" \
    --num_samples "$NUM_SAMPLES" --ks "$KS" --adapter_name "$ADAPTER" \
    --output_name "$KSWEEP" 2>&1 | tee "$LOG_DIR/${TAG}_ksweep.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[eval] fixed-k sweep failed"; exit 1; }
else
  echo "[eval] 1/4 fixed-k sweep: skip (summary exists)"
fi

# ---- 2. adaptive sweep (the elastic allocator across tolerances) ----
if [ ! -f "$EXP_DIR/output/${PARETO}_summary.json" ]; then
  echo "[eval] 2/4 adaptive sweep eps in {$EPS_LIST}"
  python scripts/eval_v5_pareto.py --work_dir "$EXP_DIR" \
    --eval_data_path "$EVAL_PT" --eval_meta_path "$EVAL_META" \
    --num_samples "$NUM_SAMPLES" --eps_list "$EPS_LIST" \
    --adapter_name "$ADAPTER" --output_name "$PARETO" \
    2>&1 | tee "$LOG_DIR/${TAG}_adaptive.log"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[eval] adaptive sweep failed"; exit 1; }
else
  echo "[eval] 2/4 adaptive sweep: skip (summary exists)"
fi

# ---- 3. completeness check ----
if [ "$EXPECT_N" != "0" ]; then
  echo "[eval] 3/4 checking scored-sample counts against n=$EXPECT_N"
  python scripts/check_eval_n.py "$EXP_DIR/output/${KSWEEP}_summary.json" \
    ksweep --n "$EXPECT_N" || { echo "[eval] fixed-k sweep is incomplete"; exit 1; }
  python scripts/check_eval_n.py "$EXP_DIR/output/${PARETO}_summary.json" \
    pareto --n "$EXPECT_N" || { echo "[eval] adaptive sweep is incomplete"; exit 1; }
else
  echo "[eval] 3/4 completeness check: skipped (EXPECT_N=0)"
fi

# ---- 4. token-F1 / EM aggregation ----
echo "[eval] 4/4 token-F1 / EM"
for kind in ksweep adaptive; do
  case $kind in
    ksweep)   pattern="$EXP_DIR/output/${KSWEEP}_rank*.json" ;;
    adaptive) pattern="$EXP_DIR/output/${PARETO}_rank*.json" ;;
  esac
  out="$REPORT_DIR/tokenf1_${TAG}_${kind}.json"
  if [ ! -f "$out" ]; then
    python scripts/aggregate_tokenf1.py \
      --sources "${TAG}_${kind}=${pattern}" --out "$out" \
      2>&1 | tee -a "$LOG_DIR/${TAG}_tokenf1.log"
    [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "[eval] $kind scoring failed"; exit 1; }
  else
    echo "[eval]   $kind: skip ($out exists)"
  fi
  # An empty report is a real failure mode: with no rank files matching the glob
  # the aggregator warns and still writes "{}", which reads as a clean run.
  if [ ! -s "$out" ] || ! grep -q '"n"' "$out"; then
    echo "[eval] $kind report is empty — no rank files matched $pattern"; exit 1
  fi
  # The aggregator globs rank files while the sweep summaries only read
  # range(world_size), so a leftover high-index rank file from an earlier run on
  # more GPUs inflates the metric here and nowhere else. Checking n catches it.
  if [ "$EXPECT_N" != "0" ]; then
    python scripts/check_eval_n.py "$out" report --n "$EXPECT_N" \
      || { echo "[eval] $kind report has the wrong sample count (stale rank"
           echo "[eval]   files from an earlier run with more GPUs?). Delete"
           echo "[eval]   $pattern and $out, then rerun."; exit 1; }
  fi
done

echo "[eval] done"
echo "[eval]   fixed-k : $REPORT_DIR/tokenf1_${TAG}_ksweep.json"
echo "[eval]   adaptive: $REPORT_DIR/tokenf1_${TAG}_adaptive.json"
echo "[eval] macro IID/OOD splits: python scripts/compare_subset_fullft.py"
