#!/usr/bin/env bash
set -euo pipefail

# pest -> split_seed
declare -A SEED_BY_PEST=(
  [blast]=59
  [bacterial_blight]=58
  [BPH2]=129
  [brown_spot]=159
  [rice_stem_borer_1]=91
  [rice_stem_borer_2]=79
  [WBPH]=111
)

# pest -> stage2_tstar offset (= round(stage1 lead_time - 45))
declare -A OFFSET_BY_PEST=(
  [blast]=50
  [bacterial_blight]=60
  [BPH2]=0
  [brown_spot]=60
  [rice_stem_borer_1]=20
  [rice_stem_borer_2]=85
  [WBPH]=60
)

ORDER=(blast bacterial_blight BPH2 brown_spot rice_stem_borer_1 rice_stem_borer_2 WBPH)

COMMON_ARGS=(
  --run 4
  --split_mode site_year
  --seeds 0
  --doy_start_override 1 --doy_end_override 300 --d_model_override 48
  --w_interval 1.0 --w_left 0.5 --w_right 0.5
  --stage2_nowcast --nowcast_window 28 --nowcast_stride 1
  --nowcast_only_pre_event 1 --nowcast_event_time_proxy r
  --stage2_nowcast_require_tstar_before_L 0
  --stage2_causal_tstar --stage2_tstar_layers 1
  --stage2_early_tstar_weight_min 0.2
  --stage2_time_chunk_size 64 --stage2_conditional_survival 0
  --amp 1 --amp_dtype bf16
  --save_epoch_checkpoints 1
)

for pest in "${ORDER[@]}"; do
  seed=${SEED_BY_PEST[$pest]}
  offset=${OFFSET_BY_PEST[$pest]}

  uncond_out="rice/outputs_stage2_${pest}_uncond"
  pilot_out="rice/outputs_stage2_${pest}_gauss5_pilot"
  final_out="rice/outputs_stage2_${pest}_gauss5_optionA_aw15"

  echo "=========================================="
  echo "[$pest] seed=$seed offset=$offset start: $(date -Iseconds)"
  echo "=========================================="

  # ---- 1) uncond (hazard mode, no warm start) ----
  echo "[$pest] >>> Stage 1/3: uncond (hazard) | $(date -Iseconds)"
  .venv/bin/python -m rice.scripts.run_train \
    --pest "$pest" \
    --out_root "$uncond_out" \
    --split_seed "$seed" \
    "${COMMON_ARGS[@]}" \
    --gated_val_stage2_tstar_offset "$offset" \
    --max_epochs_override 20 --patience_override 5 2>&1 \
    | tee -a "/tmp/stage2_phase5d_${pest}.log"

  # ---- 2) pilot Gaussian PMF aw=10 (warm from uncond best) ----
  echo "[$pest] >>> Stage 2/3: pilot aw=10 (gaussian) | $(date -Iseconds)"
  .venv/bin/python -m rice.scripts.run_train \
    --pest "$pest" \
    --out_root "$pilot_out" \
    --split_seed "$seed" \
    "${COMMON_ARGS[@]}" \
    --stage2_pmf_mode gaussian \
    --stage2_pmf_sigma 5.0 \
    --stage2_pmf_target_offset 5.0 \
    --stage2_pmf_asym_weight 10.0 \
    --stage2_pmf_right_weight 0.3 \
    --stage2_warm_start_ckpt "${uncond_out}/ckpt/checkpoint_run4.pt" \
    --stage2_warm_start_seed 0 \
    --gated_val_stage2_tstar_offset "$offset" \
    --max_epochs_override 10 --patience_override 5 2>&1 \
    | tee -a "/tmp/stage2_phase5d_${pest}.log"

  # ---- 3) final Phase 5d aw=15 (warm from pilot best) ----
  echo "[$pest] >>> Stage 3/3: final aw=15 (gaussian) | $(date -Iseconds)"
  .venv/bin/python -m rice.scripts.run_train \
    --pest "$pest" \
    --out_root "$final_out" \
    --split_seed "$seed" \
    "${COMMON_ARGS[@]}" \
    --stage2_pmf_mode gaussian \
    --stage2_pmf_sigma 5.0 \
    --stage2_pmf_target_offset 5.0 \
    --stage2_pmf_asym_weight 15.0 \
    --stage2_pmf_right_weight 0.3 \
    --stage2_warm_start_ckpt "${pilot_out}/ckpt/checkpoint_run4.pt" \
    --stage2_warm_start_seed 0 \
    --gated_val_stage2_tstar_offset "$offset" \
    --max_epochs_override 15 --patience_override 5 2>&1 \
    | tee -a "/tmp/stage2_phase5d_${pest}.log"

  echo "[$pest] DONE: $(date -Iseconds)"
  echo ""
done

echo "ALL DONE: $(date -Iseconds)"
