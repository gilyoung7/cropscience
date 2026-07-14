#!/usr/bin/env bash
# S2N-direct ROLLING launcher — train WBPH direct_neighbor ckpts for HELD-OUT years
# 2022 and 2023 (the 2024 DN ckpt already exists and is NOT retrained here).
#
# Mirrors rice/scripts/run_s2n_direct.sh EXACTLY, but sources each year's PER-YEAR
# BASELINE production log so the extracted run_train command already carries that
# year's split (val=Y-1, test=Y) and that year's dispatch CSV:
#   2022 -> batch_2022_baseline  (val 2021 / test 2022, gate_dispatch_group_tau)
#   2023 -> batch_2023_baseline  (val 2022 / test 2023, gate_D_history)
# Transforms applied to the extracted pilot/final commands (identical to run_s2n_direct.sh):
#   * --out_root -> rice/outputs/stage2/direct_neighbor_rolling/<year>/<pest>/lead_v3_{pilot,final}
#   * pilot: warm-start STRIPPED (d_in +6 -> from scratch)
#   * final: warm-start -> this run's rolling pilot ckpt
#   * append --stage2_add_neighbor_history --stage2_neighbor_decay_km 20.0
#
# Each test year is HELD OUT (not in train/val). The 2024 DN ckpt is NEVER reused for
# 2022/2023. Existing outputs (batch_*_baseline, direct_neighbor/) are NEVER written.
#
# Usage (from the cropscience/ package root):
#   cd /home/gpu4080/research/cropscience
#   bash rice/scripts/run_s2n_direct_rolling.sh                 # WBPH, years 2022 2023
#   YEARS="2022" bash rice/scripts/run_s2n_direct_rolling.sh    # one year
#   STAGE=dryrun bash rice/scripts/run_s2n_direct_rolling.sh    # inspect d_in only
#   PESTS="WBPH" DECAY_KM=20.0 PY=.venv/bin/python bash rice/scripts/run_s2n_direct_rolling.sh
set -u

PY="${PY:-.venv/bin/python}"
DECAY_KM="${DECAY_KM:-20.0}"
STAGE="${STAGE:-all}"                 # all | dryrun | pilot | final
YEARS="${YEARS:-2022 2023}"
PESTS="${PESTS:-WBPH}"
OUT_BASE="rice/outputs/stage2/direct_neighbor_rolling"

export PYTHONPATH="${PYTHONPATH:-.}"

batch_for_year () {  # $1=year -> baseline batch dir name (underscore form used in logs)
  case "$1" in
    2024) echo "rice/outputs_stage2_batch_2024_bestgate" ;;
    *)    echo "rice/outputs_stage2_batch_$1_baseline" ;;
  esac
}

extract_args () {  # $1=log  $2=lead_v3_pilot|lead_v3_final -> args after run_train
  grep -E "rice\.scripts\.run_train" "$1" \
    | grep -E "out_root rice/[^ ]*/$2( |\$)" | head -1 \
    | sed -E 's|^.*-m rice\.scripts\.run_train ||'
}

run_cmd () {  # $1=argstring $2=logfile $3=label
  echo ">>> [$3] $PY -u -m rice.scripts.run_train $1"
  # shellcheck disable=SC2086
  $PY -u -m rice.scripts.run_train $1 2>&1 | tee "$2"
}

for year in $YEARS; do
  PROD_BASE="$(batch_for_year "$year")"
  for pest in $PESTS; do
    echo "==================== $pest / test=$year ===================="
    LOG="$PROD_BASE/$pest/logs/stage2_lead_v3_train.log"
    OUT="$OUT_BASE/$year/$pest"
    if [ ! -f "$LOG" ]; then echo "[skip] no baseline log: $LOG"; continue; fi
    mkdir -p "$OUT/logs"

    PILOT_ARGS="$(extract_args "$LOG" lead_v3_pilot)"
    FINAL_ARGS="$(extract_args "$LOG" lead_v3_final)"
    if [ -z "$PILOT_ARGS" ] || [ -z "$FINAL_ARGS" ]; then
      echo "[skip] could not extract pilot/final command from $LOG"; continue
    fi

    # --- PILOT: out_root, strip warm-start, add neighbor ---
    PILOT_NEW="$(printf '%s' "$PILOT_ARGS" \
      | sed -E "s| --out_root [^ ]+| --out_root $OUT/lead_v3_pilot|" \
      | sed -E "s| --stage2_warm_start_ckpt [^ ]+ --stage2_warm_start_seed [0-9]+||")"
    PILOT_NEW="$PILOT_NEW --stage2_add_neighbor_history --stage2_neighbor_decay_km $DECAY_KM"

    # --- FINAL: out_root, warm-start <- this run's pilot, add neighbor ---
    FINAL_NEW="$(printf '%s' "$FINAL_ARGS" \
      | sed -E "s| --out_root [^ ]+| --out_root $OUT/lead_v3_final|" \
      | sed -E "s| --stage2_warm_start_ckpt [^ ]+| --stage2_warm_start_ckpt $OUT/lead_v3_pilot/ckpt/checkpoint_run4.pt|")"
    FINAL_NEW="$FINAL_NEW --stage2_add_neighbor_history --stage2_neighbor_decay_km $DECAY_KM"

    # --- DRY-RUN: pilot args, tiny, out_root -> _dryrun ---
    DRY_NEW="$(printf '%s' "$PILOT_NEW" | sed -E "s| --out_root [^ ]+| --out_root $OUT/_dryrun|")"
    DRY_NEW="$DRY_NEW --max_epochs_override 1 --stage2_sanity_only 1 --stage2_sanity_batches 1"

    if [ "$STAGE" = "all" ] || [ "$STAGE" = "dryrun" ]; then
      run_cmd "$DRY_NEW" "$OUT/logs/dryrun.log" "$pest/$year dryrun"
      echo "--- [$pest/$year] d_in check ---"
      grep -E "stage2_neighbor|computed_from_dataset|new_feature_count|year_ranges" "$OUT/logs/dryrun.log" || true
    fi
    [ "$STAGE" = "dryrun" ] && { echo "[dryrun-only] stop for $pest/$year"; continue; }

    if [ "$STAGE" = "all" ] || [ "$STAGE" = "pilot" ]; then
      run_cmd "$PILOT_NEW" "$OUT/logs/pilot.log" "$pest/$year pilot"
    fi
    if [ ! -f "$OUT/lead_v3_pilot/ckpt/checkpoint_run4.pt" ]; then
      echo "[abort:$pest/$year] pilot ckpt missing -> skipping final"; continue
    fi
    if [ "$STAGE" = "all" ] || [ "$STAGE" = "final" ]; then
      run_cmd "$FINAL_NEW" "$OUT/logs/final.log" "$pest/$year final"
    fi
    echo "[done] $pest/$year -> $OUT/lead_v3_final"
  done
done

echo "ALL DONE. Rolling DN ckpts under $OUT_BASE/<year>/<pest>/lead_v3_final/ckpt/checkpoint_run4.pt"
echo "Next: regenerate the multiyear grid (now picks up DN 2022/2023) then re-run the ablation:"
echo "  $PY -m rice.scripts.phase_t_wbph_multiyear_grid --out-dir rice/outputs/diag/stage2_ckptnorm_selector_wbph/multiyear_full --force"
echo "  $PY -m rice.scripts.phase_t_wbph_offset_ablation_multiyear \\"
echo "     --grid rice/outputs/diag/stage2_ckptnorm_selector_wbph/multiyear_full/wbph_grid_1to75_multiyear.csv \\"
echo "     --out-dir rice/outputs/diag/stage2_ckptnorm_selector_wbph/offset_ablation_multiyear_full --force"
