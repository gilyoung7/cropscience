#!/usr/bin/env bash
set -euo pipefail

echo "===== Stage-2 Direct Neighbor: sheath_blight ====="

DISP=rice/outputs_stage2_batch_2024_bestgate/sheath_blight/gate_dispatch_group_tau_R088_features_per_sy.csv
OUT=rice/outputs_stage2_direct_neighbor/sheath_blight

mkdir -p "$OUT/logs"

COMMON=(
  --pest sheath_blight
  --run 4
  --seeds 0
  --split_seed 42
  --split_mode year
  --val_year 2023
  --test_year_min 2024
  --test_year_max 2024

  --dropout 0.2
  --weight_decay 0.0001
  --lr 0.0001
  --w_interval 1.0
  --w_left 0.5
  --w_right 0.5

  --stage2_nowcast
  --stage2_nowcast_window 28
  --stage2_nowcast_stride 1
  --stage2_nowcast_only_pre_event 1
  --stage2_nowcast_event_time_proxy r
  --stage2_nowcast_require_tstar_before_L 0
  --stage2_causal_tstar
  --stage2_tstar_layers 1
  --stage2_use_tstar_scalar_pos 0
  --stage2_early_tstar_weight_min 0.2
  --stage2_site_year_mean_loss 0
  --stage2_time_chunk_size 64
  --stage2_conditional_survival 0

  --stage2_pmf_mode gaussian
  --stage2_pmf_sigma 5.0
  --stage2_pmf_right_weight 0.3
  --stage2_pmf_target_mode l_offset
  --stage2_best_metric val_iou80
  --amp 1
  --amp_dtype bf16
  --d_model_override 48
  --stage2_pmf_long_lead_threshold 60.0
  --stage2_pmf_long_lead_weight 3.0

  --stage2_dispatch_feature_csv "$DISP"
  --stage2_dispatch_feature_mode causal
  --stage2_dispatch_feature_missing_value 0.0
  --stage2_cohort_dispatch_only
  --stage2_pmf_mu_mode lead_from_alert
  --stage2_pmf_lead_min 7.0
  --stage2_pmf_lead_max 75.0

  --stage2_add_neighbor_history
  --stage2_neighbor_decay_km 20.0
)

echo "===== Check dispatch CSV ====="
ls -lh "$DISP"

echo "===== 1. Dry-run shape check ====="
python -u -m rice.scripts.run_train "${COMMON[@]}" \
  --out_root "$OUT/_dryrun" \
  --max_epochs_override 1 \
  --stage2_sanity_only 1 \
  --stage2_sanity_batches 1 \
  --stage2_pmf_asym_weight 15.0 \
  --stage2_pmf_target_early_offset 30.0 \
  2>&1 | tee "$OUT/logs/dryrun.log"

echo "===== 2. Pilot training from scratch ====="
python -u -m rice.scripts.run_train "${COMMON[@]}" \
  --out_root "$OUT/lead_v3_pilot" \
  --stage2_pmf_asym_weight 15.0 \
  --stage2_pmf_asym_weight_early 0.0 \
  --stage2_pmf_target_offset 0.0 \
  --stage2_pmf_target_early_offset 30.0 \
  2>&1 | tee "$OUT/logs/pilot.log"

echo "===== Check pilot checkpoint ====="
ls -lh "$OUT/lead_v3_pilot/ckpt/checkpoint_run4.pt"

echo "===== 3. Final training warm-start from direct-neighbor pilot ====="
python -u -m rice.scripts.run_train "${COMMON[@]}" \
  --out_root "$OUT/lead_v3_final" \
  --stage2_warm_start_ckpt "$OUT/lead_v3_pilot/ckpt/checkpoint_run4.pt" \
  --stage2_warm_start_seed 0 \
  --stage2_pmf_asym_weight 25.0 \
  --stage2_pmf_asym_weight_early 5.0 \
  --stage2_pmf_target_offset 0.0 \
  --stage2_pmf_target_early_offset 30.0 \
  2>&1 | tee "$OUT/logs/final.log"

echo "===== DONE ====="
echo "Pilot output: $OUT/lead_v3_pilot"
echo "Final output: $OUT/lead_v3_final"
echo "Final eval   : $OUT/lead_v3_final/eval"
