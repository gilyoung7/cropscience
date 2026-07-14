#!/usr/bin/env bash
set -euo pipefail

declare -A SEED=( [bacterial_blight]=58 [BPH2]=129 )
declare -A OFF=( [bacterial_blight]=60 [BPH2]=0 )

for pest in bacterial_blight BPH2; do
  seed=${SEED[$pest]}; offset=${OFF[$pest]}
  s1_ckpt="rice/outputs_stage1/${pest}_siteyear${seed}/ckpt/event_run4_xgb_nowcast_w56_s1_tpos_split${seed}_siteyear_ymin2002.pt"
  s1_eval="rice/outputs_stage1/${pest}_siteyear${seed}/eval/event_eval_${pest}_run4_stage1_xgb_nowcast_w56_s1_tpos_split${seed}_siteyear_ymin2002_tauAlertR85_minFAR_k3_ma1_nogate_abs.csv"
  s2_ckpt="rice/outputs_stage2_${pest}_gauss5_optionA_aw15/ckpt/checkpoint_run4.pt"
  out_root="rice/outputs_stage2_${pest}_gauss5_optionA_aw15/gated_eval"

  echo "[$pest] seed=$seed off=$offset (w=56 adopted op)"
  .venv/bin/python -m rice.scripts.run_viz_interval \
    --pest "$pest" --run 4 \
    --stage1_ckpt "$s1_ckpt" --stage1_eval_csv "$s1_eval" \
    --stage2_ckpt "$s2_ckpt" \
    --split_mode site_year --split_seed "$seed" \
    --seeds 0 --split test \
    --stage2_tstar_offset "$offset" --pi_mass_level 0.95 --tau_mode f1 \
    --use_wandb --wandb_project agro-rice \
    --wandb_run_name "${pest}_phase5d_gated_w56adopted_off${offset}_pi95" \
    --wandb_tags "phase5d,gated,w56adopted,off${offset},pest_${pest}" \
    --wandb_job_type gated_eval \
    --out_root "$out_root" \
    --final_tag "phase5d_gated_w56adopted_off${offset}_pi95" 2>&1 | tee "/tmp/phase5d_gated_${pest}_w56.log"
  echo "[$pest] DONE"
done
echo "ALL DONE"
