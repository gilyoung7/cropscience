#!/usr/bin/env bash
set -euo pipefail

cd ~/research/cropscience
mkdir -p logs rice/outputs_viz

WANDB_PROJECT="agro-rice"
WANDB_RUN_NAME="viz_lead_v3_sheath_blight_test_off14"

.venv/bin/python -u -m rice.scripts.run_viz_interval \
  --pest sheath_blight \
  --run 4 \
  --split test \
  --stage1_ckpt rice/outputs_stage1/sheath_blight_yearsplit2023-24_lead14-45_ignore/ckpt/event_run4_xgb_w28_lead14-45_ignore.pt \
  --stage2_ckpt rice/outputs_stage2_sheath_blight_dispatch_R088_cohortOnly_lead_v3_final/ckpt/checkpoint_run4.pt \
  --dispatch_summary rice/outputs_stage1/sheath_blight_yearsplit2023-24_lead14-45_history_rolling/group_tau_hybrid/group_tau_hybrid_summary.json \
  --dispatch_feature_csv outputs_dispatch_R088_features_per_sy.csv \
  --dispatch_feature_mode causal \
  --cohort_dispatch_only \
  --stage2_offset 14 \
  --use_wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --wandb_group "phaseB_lead_v3_visual" \
  --wandb_tags "lead_v3,offset14,sheath_blight,test,viz" \
  2>&1 | tee logs/run_viz_lead_v3_off14.log
