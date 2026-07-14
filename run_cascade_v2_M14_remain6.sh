#!/usr/bin/env bash
set -euo pipefail

declare -A SEED_BY_PEST=(
  [bacterial_blight]=58
  [BPH2]=129
  [brown_spot]=159
  [rice_stem_borer_1]=91
  [rice_stem_borer_2]=79
  [WBPH]=111
)

ORDER=(bacterial_blight BPH2 brown_spot rice_stem_borer_1 rice_stem_borer_2 WBPH)

for pest in "${ORDER[@]}"; do
  seed=${SEED_BY_PEST[$pest]}
  out_root="rice/outputs_stage1/${pest}_siteyear${seed}"
  ckpt="${out_root}/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split${seed}_siteyear_ymin2002.pt"
  oof_csv="${out_root}/ckpt/oof_train_seed${seed}_5fold.csv"
  probs_csv="${out_root}/eval/event_eval_${pest}_run4_stage1_xgb_nowcast_w28_s1_tpos_split${seed}_siteyear_ymin2002_tauAlertR85_minFAR_k3_ma1_nogate_abs_probs.csv"
  cascade_dir="${out_root}/cascade_v2_M14_full"
  log_file="${out_root}/logs/cascade_v2_M14.log"

  echo "=========================================="
  echo "[$pest] seed=$seed cascade v2 M=14 start: $(date -Iseconds)"
  echo "=========================================="

  mkdir -p "${cascade_dir}" "${out_root}/logs"

  .venv/bin/python -m rice.scripts.run_stage1_oof \
    --pest "$pest" --run 4 \
    --split_seed "$seed" --split_mode site_year \
    --ckpt "$ckpt" \
    --out_csv "$oof_csv" \
    --n_folds 5 --cv_seed 42 --xgb_seed 0 2>&1 | tee "$log_file"

  .venv/bin/python -m rice.scripts.run_stage1b_cascade_v2 \
    --pest "$pest" --run 4 \
    --split_seed "$seed" --split_mode site_year \
    --ckpt_stage1 "$ckpt" \
    --oof_csv "$oof_csv" \
    --probs_csv "$probs_csv" \
    --tau_a 0.51 --tau_baseline 0.65 \
    --monitor_M 14 \
    --out_dir "$cascade_dir" \
    --seed 0 --xgb_max_depth 4 --xgb_n_estimators 200 2>&1 | tee -a "$log_file"

  echo "[$pest] done: $(date -Iseconds)"
  echo ""
done

echo "ALL DONE: $(date -Iseconds)"
