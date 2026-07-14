#!/usr/bin/env bash
# Phase R2 + Phase S2 chain.
#
#   Step 1: phase_r refresh sample_grid with 4 models — baseline / 2-sided /
#           phenobias / phenobias_nohead (same phenobias ckpt, phen_head bypassed
#           at inference). Per-model operational σ injected.
#
#   Step 2: phase_s2 selector — three feature ablations (stage1_only,
#           stage1_plus_mu, mu_only) × {logreg, xgb} on each model.
#
# Usage:  bash scripts/run_phase_s2.sh
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOG=logs_phase_r2_s2_chain.log
SAMPLE_GRID=outputs_phase_r_sample_grid.csv
ORACLE_CSV=outputs_phase_r_oracle.csv

STAGE1_CKPT="rice/outputs_stage1/sheath_blight_yearsplit2023-24/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_yearsplit_ymin2002.pt"

# 4 models: baseline, 2-sided, phenobias (head ON), phenobias_nohead (head bypassed).
# phenobias_nohead reuses the phenobias ckpt; phase_r enforces bypass via --nohead_labels.
MODELS="D=15 baseline (asym=15)|rice/outputs_stage2_sheath_blight_yearsplit2023-24_final_aw15/ckpt/checkpoint_run4.pt;D=15 2-sided (asym=25)|rice/outputs_stage2_sheath_blight_d15_asym25_2sided_final_aw25/ckpt/checkpoint_run4.pt;D=15 phenobias|rice/outputs_stage2_sheath_blight_d15_phenobias_final/ckpt/checkpoint_run4.pt;D=15 phenobias_nohead|rice/outputs_stage2_sheath_blight_d15_phenobias_final/ckpt/checkpoint_run4.pt"

PER_MODEL_SIGMA="D=15 baseline (asym=15)=4.0,D=15 2-sided (asym=25)=5.0,D=15 phenobias=4.5,D=15 phenobias_nohead=4.5"
NOHEAD_LABELS="D=15 phenobias_nohead"

mkdir -p outputs/phase_s2

echo "===================================================" | tee "$LOG"
echo "=== Phase R2 + Phase S2 chain  $(date -Iseconds) ===" | tee -a "$LOG"
echo "=== 4 models: baseline / 2-sided / phenobias / phenobias_nohead ===" | tee -a "$LOG"
echo "=== σ map: baseline=4.0, 2-sided=5.0, phenobias=4.5, phenobias_nohead=4.5 ===" | tee -a "$LOG"
echo "===================================================" | tee -a "$LOG"

echo "=== Step 1/2 START : phase_r (sample_grid 4-model refresh) ===" | tee -a "$LOG"
.venv/bin/python -m rice.scripts.phase_r_oracle_iou \
    --pest sheath_blight --run 4 \
    --val_year 2022 --test_year_min 2023 --test_year_max 2024 \
    --stage1_ckpt "$STAGE1_CKPT" \
    --models "$MODELS" \
    --offsets 60,90,105,120 \
    --oracle_sigma 5.0 \
    --sigma_sweep 2.5,3.0,3.5,4.0,4.5,5.0,6.0 \
    --sigma_sweep_offsets 105,120 \
    --per_model_sigma "$PER_MODEL_SIGMA" \
    --nohead_labels "$NOHEAD_LABELS" \
    --sample_grid_csv "$SAMPLE_GRID" \
    --out_csv "$ORACLE_CSV" \
    2>&1 | tee -a "$LOG"
test -f "$SAMPLE_GRID"
echo "=== Step 1/2 DONE  : sample_grid=$SAMPLE_GRID  ($(wc -l <"$SAMPLE_GRID") lines) ===" | tee -a "$LOG"

echo "=== Step 2/2 START : phase_s2 (3 feature_set × 2 clf × 4 model) ===" | tee -a "$LOG"
.venv/bin/python -m rice.scripts.phase_s2_selector \
    --pest sheath_blight --run 4 \
    --val_year 2022 --test_year_min 2023 --test_year_max 2024 \
    --stage1_ckpt "$STAGE1_CKPT" \
    --sample_grid "$SAMPLE_GRID" \
    --out_dir outputs/phase_s2/ \
    --model all \
    --seed 42 \
    --shift 30 \
    2>&1 | tee -a "$LOG"
test -f outputs/phase_s2/selector_summary.csv
echo "=== Step 2/2 DONE  : outputs/phase_s2/selector_summary.csv ===" | tee -a "$LOG"

echo "===================================================" | tee -a "$LOG"
echo "=== ALL DONE: $(date -Iseconds)  log=$LOG ===" | tee -a "$LOG"
echo "==================================================="
