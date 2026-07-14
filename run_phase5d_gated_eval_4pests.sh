#!/usr/bin/env bash
set -euo pipefail

declare -A SEED_BY_PEST=(
  [blast]=59
  [bacterial_blight]=58
  [BPH2]=129
  [brown_spot]=159
)

declare -A OFFSET_BY_PEST=(
  [blast]=50
  [bacterial_blight]=60
  [BPH2]=0
  [brown_spot]=60
)

ORDER=(blast bacterial_blight BPH2 brown_spot)

for pest in "${ORDER[@]}"; do
  seed=${SEED_BY_PEST[$pest]}
  offset=${OFFSET_BY_PEST[$pest]}

  stage1_ckpt="rice/outputs_stage1/${pest}_siteyear${seed}/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split${seed}_siteyear_ymin2002.pt"
  stage1_eval="rice/outputs_stage1/${pest}_siteyear${seed}/eval/event_eval_${pest}_run4_stage1_xgb_nowcast_w28_s1_tpos_split${seed}_siteyear_ymin2002_tauAlertR85_minFAR_k3_ma1_nogate_abs.csv"
  stage2_ckpt="rice/outputs_stage2_${pest}_gauss5_optionA_aw15/ckpt/checkpoint_run4.pt"
  out_root="rice/outputs_stage2_${pest}_gauss5_optionA_aw15/gated_eval"

  echo "=========================================="
  echo "[$pest] seed=$seed offset=$offset start: $(date -Iseconds)"
  echo "=========================================="

  .venv/bin/python -m rice.scripts.run_viz_interval \
    --pest "$pest" --run 4 \
    --stage1_ckpt "$stage1_ckpt" \
    --stage1_eval_csv "$stage1_eval" \
    --stage2_ckpt "$stage2_ckpt" \
    --split_mode site_year --split_seed "$seed" \
    --seeds 0 \
    --split test \
    --stage2_tstar_offset "$offset" \
    --pi_mass_level 0.95 \
    --tau_mode f1 \
    --use_wandb \
    --wandb_project agro-rice \
    --wandb_run_name "${pest}_phase5d_gated_off${offset}_pi95" \
    --wandb_tags "phase5d,gated,off${offset},pest_${pest}" \
    --wandb_job_type gated_eval \
    --out_root "$out_root" \
    --final_tag "phase5d_gated_off${offset}_pi95" 2>&1 | tee "/tmp/phase5d_gated_${pest}.log"

  echo "[$pest] DONE: $(date -Iseconds)"
  echo ""
done

echo "ALL DONE: $(date -Iseconds)"
