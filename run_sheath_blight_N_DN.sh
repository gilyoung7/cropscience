#!/usr/bin/env bash
set -euo pipefail

P=sheath_blight
T=rice/outputs_stage1/sheath_blight/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split113_ymin2002.pt
B=rice/outputs_stage1/batch_rolling/sheath_blight/run0/split3_v2023_t2024

COMMON=(
  --pest sheath_blight
  --run 4
  --template_ckpt "$T"
  --val_year 2023
  --test_year_min 2024
  --test_year_max 2024
  --nowcast_window 28
  --lead_min 14
  --lead_max 45
  --outside_policy ignore
)

mkdir -p "$B/N/ckpt" "$B/DN/ckpt" "$B/logs"

echo "===== Train N: baseline + neighbor history ====="
python rice/scripts/phase_t_lead_aware_train.py "${COMMON[@]}" \
  --add_neighbor_history \
  --out_ckpt "$B/N/ckpt/event_xgb_w28_lead14-45_N.pt" \
  2>&1 | tee "$B/logs/train_N.log"

echo "===== Train DN: baseline + same-site history + neighbor history ====="
python rice/scripts/phase_t_lead_aware_train.py "${COMMON[@]}" \
  --add_site_history --site_history_policy rolling \
  --add_neighbor_history \
  --out_ckpt "$B/DN/ckpt/event_xgb_w28_lead14-45_DN.pt" \
  2>&1 | tee "$B/logs/train_DN.log"

echo "===== DONE ====="
echo "N ckpt : $B/N/ckpt/event_xgb_w28_lead14-45_N.pt"
echo "DN ckpt: $B/DN/ckpt/event_xgb_w28_lead14-45_DN.pt"
