#!/usr/bin/env bash
set -euo pipefail

cd ~/research/cropscience

PEST="sheath_blight"
BASE_WARM="rice/outputs_stage2_batch_2024_bestgate/sheath_blight/lead_v3_final"

mkdir -p rice/outputs_stage2_batch_2024_mixedloss_lam1/_summary
mkdir -p rice/outputs_stage2_batch_2024_mixedloss_lam5/_summary

run_one () {
  local LAM="$1"
  local OUT_ROOT="$2"
  local LOG="$OUT_ROOT/_summary/sheath_blight_mixed_lam${LAM}.log"

  echo "============================================================"
  echo "[RUN] sheath_blight mixed loss lambda=${LAM}"
  echo "[OUT] ${OUT_ROOT}"
  echo "[LOG] ${LOG}"
  echo "============================================================"

  GAUSSIAN_LOSS_MODE=mixed \
  GAUSSIAN_INTERVAL_LAMBDA="${LAM}" \
  GAUSSIAN_INTERVAL_CC=0 \
  SKIP_PILOT_WARM_FROM="${BASE_WARM}" \
  MAX_EPOCHS=5 \
  bash scripts/run_stage2_split3_2024_pest_best_gate_batch.sh \
    --pests "${PEST}" \
    --out_root "${OUT_ROOT}" \
    2>&1 | tee "${LOG}"

  echo "[DONE] lambda=${LAM}"
}

run_one "1.0" "rice/outputs_stage2_batch_2024_mixedloss_lam1"
run_one "5.0" "rice/outputs_stage2_batch_2024_mixedloss_lam5"

echo
echo "All mixed-loss runs finished."
