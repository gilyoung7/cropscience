#!/usr/bin/env bash
set -euo pipefail

declare -A SEED_BY_PEST=(
  [blast]=59
  [bacterial_blight]=58
  [BPH2]=129
  [brown_spot]=159
  [rice_stem_borer_1]=91
  [rice_stem_borer_2]=79
  [WBPH]=111
)

ORDER=(blast bacterial_blight BPH2 brown_spot rice_stem_borer_1 rice_stem_borer_2 WBPH)

for pest in "${ORDER[@]}"; do
  seed=${SEED_BY_PEST[$pest]}
  out_root="rice/outputs_stage1/${pest}_siteyear${seed}"
  ckpt_path="${out_root}/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split${seed}_siteyear_ymin2002.pt"
  eval_csv="${out_root}/eval/event_eval_${pest}_run4_stage1_xgb_nowcast_w28_s1_tpos_split${seed}_siteyear_ymin2002_tauAlertF1_cap05_k3_ma1_nogate_abs.csv"
  log_file="${out_root}/logs/train_eval.log"

  echo "=========================================="
  echo "[$pest] seed=$seed start: $(date -Iseconds)"
  echo "=========================================="

  mkdir -p "${out_root}/ckpt" "${out_root}/eval" "${out_root}/logs"

  # ---- TRAIN ----
  .venv/bin/python -m rice.scripts.run_event_train \
    --pest "$pest" --run 4 \
    --model xgb \
    --task_mode nowcast \
    --nowcast_window 28 --nowcast_stride 1 \
    --nowcast_only_pre_event 1 \
    --nowcast_event_time_proxy mid \
    --nowcast_label_mode eventually \
    --add_tstar_position_feature \
    --split_mode site_year \
    --out_root "$out_root" \
    --out "$ckpt_path" 2>&1 | tee "$log_file"

  # ---- EVAL ----
  .venv/bin/python -m rice.scripts.run_event_eval \
    --pest "$pest" --run 4 \
    --ckpt "$ckpt_path" \
    --out_root "$out_root" \
    --out_csv "$eval_csv" \
    --split_mode site_year \
    --gate_consecutive_k 3 \
    --gate_smooth_window 1 \
    --gate_use_t_alert_start 0 \
    --tau_select_level alert_site_year \
    --tau_mode f1 \
    --tau_target_precision 0.6 \
    --tau_target_recall 0.6 \
    --tau_alert_max_false_alert_rate 0.5 \
    --tau_alert_policy f1_then_false_alert_rate 2>&1 | tee -a "$log_file"

  echo "[$pest] done: $(date -Iseconds)"
  echo ""
done

echo "ALL DONE: $(date -Iseconds)"
