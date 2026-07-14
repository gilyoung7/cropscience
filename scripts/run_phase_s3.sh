#!/usr/bin/env bash
# Phase S3 chain: offset action-space expansion (A) + lead-aware selector (B).
#
#   Step 1: phase_r refresh sample_grid with 4 models, but apply
#           --per_model_extra_offsets {30,150,180} to the 2-sided model only.
#           Baseline / phenobias / phenobias_nohead remain on the original
#           {60,90,105,120} set.  Per-model operational σ injected.
#
#   Step 2: phase_s3 selector — 3 cells on the 2-sided model:
#               C_old    : v3_mu_only,   offsets={60,90,105,120}   (4-way)
#               C_new_v3 : v3_mu_only,   offsets=expanded          (7-way)
#               C_new_v4 : v4_lead_aware,offsets=expanded          (7-way)
#           OOF (5-fold StratifiedKFold) + lead-bin decomposition
#           (anchor = alert+60, bins 15-30/31-45/46-60/61-90/91-120).
#
# Usage:  bash scripts/run_phase_s3.sh
set -euo pipefail

export PYTORCH_CUDA_ALLOC_CONF=${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}

LOG=logs_phase_s3_chain.log
SAMPLE_GRID=outputs_phase_r_sample_grid.csv
ORACLE_CSV=outputs_phase_r_oracle.csv

STAGE1_CKPT="rice/outputs_stage1/sheath_blight_yearsplit2023-24/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_yearsplit_ymin2002.pt"

# 4 models (identical to phase R2/S2 layout).  2-sided ckpt is the one we
# extend with extra offsets at inference time.
MODELS="D=15 baseline (asym=15)|rice/outputs_stage2_sheath_blight_yearsplit2023-24_final_aw15/ckpt/checkpoint_run4.pt;D=15 2-sided (asym=25)|rice/outputs_stage2_sheath_blight_d15_asym25_2sided_final_aw25/ckpt/checkpoint_run4.pt;D=15 phenobias|rice/outputs_stage2_sheath_blight_d15_phenobias_final/ckpt/checkpoint_run4.pt;D=15 phenobias_nohead|rice/outputs_stage2_sheath_blight_d15_phenobias_final/ckpt/checkpoint_run4.pt"

PER_MODEL_SIGMA="D=15 baseline (asym=15)=4.0,D=15 2-sided (asym=25)=5.0,D=15 phenobias=4.5,D=15 phenobias_nohead=4.5"
NOHEAD_LABELS="D=15 phenobias_nohead"

# Action-space expansion: only the 2-sided model gets {30,150,180} added on
# top of the base offsets {60,90,105,120}.  Format = "LABEL=o1,o2,...;LABEL2=..."
PER_MODEL_EXTRA_OFFSETS="D=15 2-sided (asym=25)=30,150,180"

mkdir -p outputs/phase_s3

echo "===================================================" | tee "$LOG"
echo "=== Phase S3 chain  $(date -Iseconds) ===" | tee -a "$LOG"
echo "=== 2-sided offset action-space: {60,90,105,120} ∪ {30,150,180} = 7 ===" | tee -a "$LOG"
echo "=== Other models: {60,90,105,120} (unchanged) ===" | tee -a "$LOG"
echo "===================================================" | tee -a "$LOG"

# ckpt existence checks
for entry in "$MODELS"; do
    :   # MODELS is a single string; spot-check is done by phase_r itself.
done
test -f "$STAGE1_CKPT" || { echo "[abort] stage1 ckpt not found: $STAGE1_CKPT" | tee -a "$LOG"; exit 1; }

echo "=== Step 1/2 START : phase_r (4-model sample_grid, 2-sided extra offsets) ===" | tee -a "$LOG"
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
    --per_model_extra_offsets "$PER_MODEL_EXTRA_OFFSETS" \
    --nohead_labels "$NOHEAD_LABELS" \
    --sample_grid_csv "$SAMPLE_GRID" \
    --out_csv "$ORACLE_CSV" \
    2>&1 | tee -a "$LOG"
test -f "$SAMPLE_GRID" || { echo "[abort] sample_grid not produced: $SAMPLE_GRID" | tee -a "$LOG"; exit 1; }
echo "=== Step 1/2 DONE  : sample_grid=$SAMPLE_GRID  ($(wc -l <"$SAMPLE_GRID") lines) ===" | tee -a "$LOG"

# Spot-check that 2-sided has the expanded offset set in the grid
EXTRA_HIT=$(awk -F, 'NR>1 && $0 ~ /2-sided/ {print $0}' "$SAMPLE_GRID" \
            | awk -F, '{print $NF}' >/dev/null 2>&1; echo "$?")
echo "=== sanity: 2-sided rows in grid = $(grep -c '2-sided' "$SAMPLE_GRID" || true) ===" | tee -a "$LOG"

echo "=== Step 2/2 START : phase_s3 (3 cells × logreg OOF) ===" | tee -a "$LOG"
.venv/bin/python -m rice.scripts.phase_s3_selector \
    --pest sheath_blight --run 4 \
    --val_year 2022 --test_year_min 2023 --test_year_max 2024 \
    --stage1_ckpt "$STAGE1_CKPT" \
    --sample_grid "$SAMPLE_GRID" \
    --out_dir outputs/phase_s3/ \
    --target_label "2-sided" \
    --shift 30 \
    --seed 42 \
    2>&1 | tee -a "$LOG"
test -f outputs/phase_s3/selector_summary.csv \
    || { echo "[abort] phase_s3 summary not produced" | tee -a "$LOG"; exit 1; }
test -f outputs/phase_s3/lead_bin_decomp.csv \
    || { echo "[abort] phase_s3 lead-bin decomp not produced" | tee -a "$LOG"; exit 1; }
echo "=== Step 2/2 DONE  : outputs/phase_s3/{selector_summary,lead_bin_decomp}.csv ===" | tee -a "$LOG"

echo "===================================================" | tee -a "$LOG"
echo "=== ALL DONE: $(date -Iseconds)  log=$LOG ===" | tee -a "$LOG"
echo "==================================================="
