#!/usr/bin/env bash
set -euo pipefail

cd /home/gpu4080/research/cropscience

LOG_DIR="rice/outputs/logs/wbph_dn_rolling_full"
mkdir -p "$LOG_DIR"

echo "=== [1/3] Train direct_neighbor rolling ckpts: WBPH 2022, 2023 ==="
YEARS="2022 2023" PESTS="WBPH" bash rice/scripts/run_s2n_direct_rolling.sh \
  2>&1 | tee "$LOG_DIR/train_dn_2022_2023.log"

echo "=== [2/3] Build multiyear grid including DN 2022/2023/2024 ==="
PYTHONPATH=$PWD .venv/bin/python -m rice.scripts.phase_t_wbph_multiyear_grid \
  --out-dir rice/outputs/diag/stage2_ckptnorm_selector_wbph/multiyear_full \
  --force \
  2>&1 | tee "$LOG_DIR/build_multiyear_grid.log"

echo "=== [3/3] Evaluate offset ablation multiyear full ==="
PYTHONPATH=$PWD .venv/bin/python -m rice.scripts.phase_t_wbph_offset_ablation_multiyear \
  --grid rice/outputs/diag/stage2_ckptnorm_selector_wbph/multiyear_full/wbph_grid_1to75_multiyear.csv \
  --out-dir rice/outputs/diag/stage2_ckptnorm_selector_wbph/offset_ablation_multiyear_full \
  --force \
  2>&1 | tee "$LOG_DIR/eval_offset_ablation.log"

echo "=== DONE ==="
echo "Results:"
echo "  rice/outputs/diag/stage2_ckptnorm_selector_wbph/offset_ablation_multiyear_full/"
echo "Logs:"
echo "  $LOG_DIR"
