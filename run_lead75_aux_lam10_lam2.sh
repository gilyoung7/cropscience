#!/usr/bin/env bash
set -euo pipefail

cd ~/research/cropscience
mkdir -p logs

# 1) Train + eval + mu_diag for aux lead75 lambdas
for LAM in 10.0 2.0; do
  if [ "$LAM" = "10.0" ]; then
    TAG="lead75_aux_lam10"
  else
    TAG="lead75_aux_lam2"
  fi

  echo "============================================================"
  echo "[TRAIN] $TAG"
  echo "============================================================"

  .venv/bin/python -u -m rice.scripts.phase_s5_train \
    --pest sheath_blight --run 4 --seed 0 \
    --val_year 2022 --test_year_min 2023 --test_year_max 2024 \
    --uncond_ckpt rice/outputs_stage2_sheath_blight_d15_asym25_2sided_uncond/ckpt/checkpoint_run4.pt \
    --out_pilot rice/outputs_stage2_sheath_blight_dispatch_R088_cohortOnly_${TAG}_pilot \
    --out_final rice/outputs_stage2_sheath_blight_dispatch_R088_cohortOnly_${TAG}_final \
    --dispatch_feature_csv outputs_dispatch_R088_features_per_sy.csv \
    --dispatch_feature_mode causal \
    --cohort_dispatch_only \
    --mu_mode lead_from_alert \
    --lead_min 7 --lead_max 75 \
    --aux_lead_lambda "$LAM" \
    --aux_lead_huber_delta 10.0 \
    2>&1 | tee logs/phase_B_${TAG}_train.log

  echo "============================================================"
  echo "[EVAL] $TAG"
  echo "============================================================"

  STAGE2_CKPT=rice/outputs_stage2_sheath_blight_dispatch_R088_cohortOnly_${TAG}_final/ckpt/checkpoint_run4.pt \
  STAGE2_LABEL="dispatch_R088_${TAG}" \
  OFFSETS=7,14,21,30,45,60 \
  OUT_GRID_CSV=outputs_phase_B_${TAG}_newoff_sample_grid.csv \
  OUT_ORACLE_CSV=outputs_phase_B_${TAG}_newoff_oracle.csv \
  bash scripts/run_stage2_dispatch_sample_grid.sh \
    2>&1 | tee logs/phase_B_${TAG}_eval.log

  echo "============================================================"
  echo "[MU DIAG] $TAG"
  echo "============================================================"

  .venv/bin/python -m rice.scripts.phase2_dispatch_mu_diag \
    --sample_grid_csv outputs_phase_B_${TAG}_newoff_sample_grid.csv \
    --ref_offset 21 \
    --out_csv outputs_phase_B_${TAG}_newoff_mu_diag.csv \
    2>&1 | tee logs/phase_B_${TAG}_mu_diag.log
done

# 2) Probe comparison
echo "============================================================"
echo "[PROBE] lead_v3 + lead75 aux"
echo "============================================================"

.venv/bin/python -m rice.scripts.phase_b_lead_probe \
  --pest sheath_blight --run 4 \
  --val_year 2022 --test_year_min 2023 --test_year_max 2024 \
  --ckpts "lead_v3=rice/outputs_stage2_sheath_blight_dispatch_R088_cohortOnly_lead_v3_final/ckpt/checkpoint_run4.pt,lead75_aux_lam10=rice/outputs_stage2_sheath_blight_dispatch_R088_cohortOnly_lead75_aux_lam10_final/ckpt/checkpoint_run4.pt,lead75_aux_lam2=rice/outputs_stage2_sheath_blight_dispatch_R088_cohortOnly_lead75_aux_lam2_final/ckpt/checkpoint_run4.pt" \
  --dispatch_feature_csv outputs_dispatch_R088_features_per_sy.csv \
  --ridge_alpha 1.0 \
  --subgroups \
  --out_table outputs_phase_b_lead_probe_lead75_aux_table.csv \
  --out_subgroup_csv outputs_phase_b_lead_probe_lead75_aux_subgroup.csv \
  2>&1 | tee logs/phase_B_lead75_aux_probe.log

echo "============================================================"
echo "DONE"
echo "============================================================"
