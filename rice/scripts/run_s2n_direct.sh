#!/usr/bin/env bash
# S2N-direct launcher: production Stage-1 dispatch (fixed) + Stage-2 DIRECT neighbor (6ch).
#
# For each pest it EXTRACTS the production pilot+final run_train commands from
#   rice/outputs_stage2_batch_2024_bestgate/<pest>/logs/stage2_lead_v3_train.log
# (guaranteeing production-identical hyperparameters + the correct production
# dispatch CSV), then re-runs them with only these changes:
#   * --out_root          -> rice/outputs_stage2_direct_neighbor/<pest>/lead_v3_{pilot,final}
#   * pilot: warm-start STRIPPED (d_in +6 -> shape mismatch with prod ckpt) => from scratch
#   * final: warm-start  -> this run's direct-neighbor pilot ckpt
#   * append --stage2_add_neighbor_history --stage2_neighbor_decay_km 20.0
#
# Order per pest: dry-run (confirm d_in +6) -> pilot -> final.  Production outputs
# under outputs_stage2_batch_2024_bestgate/ are NEVER written.
#
# Usage (run from the cropscience/ package root):
#   cd /home/gpu4080/research/cropscience
#   bash rice/scripts/run_s2n_direct.sh                  # default 5 first-batch pests
#   bash rice/scripts/run_s2n_direct.sh WBPH BPH         # specific pests
#   STAGE=dryrun bash rice/scripts/run_s2n_direct.sh     # dry-run only (inspect d_in, no training)
#   DECAY_KM=20.0 PY=.venv/bin/python bash rice/scripts/run_s2n_direct.sh
set -u

PY="${PY:-.venv/bin/python}"
DECAY_KM="${DECAY_KM:-20.0}"
STAGE="${STAGE:-all}"          # all | dryrun | pilot | final
PROD_BASE="rice/outputs_stage2_batch_2024_bestgate"
OUT_BASE="rice/outputs_stage2_direct_neighbor"

PESTS=("$@")
if [ "${#PESTS[@]}" -eq 0 ]; then
  PESTS=(WBPH BPH blast bacterial_blight brown_spot)   # sheath_blight already done
fi

export PYTHONPATH="${PYTHONPATH:-.}"

extract_args () {  # $1=log  $2=lead_v3_pilot|lead_v3_final  -> args after run_train
  grep -E "rice\.scripts\.run_train" "$1" \
    | grep -E "out_root rice/[^ ]*/$2( |\$)" | head -1 \
    | sed -E 's|^.*-m rice\.scripts\.run_train ||'
}

run_cmd () {  # $1=argstring  $2=logfile  $3=label
  echo ">>> [$3] $PY -u -m rice.scripts.run_train $1"
  # shellcheck disable=SC2086
  $PY -u -m rice.scripts.run_train $1 2>&1 | tee "$2"
}

for pest in "${PESTS[@]}"; do
  echo "==================== $pest ===================="
  LOG="$PROD_BASE/$pest/logs/stage2_lead_v3_train.log"
  OUT="$OUT_BASE/$pest"
  if [ ! -f "$LOG" ]; then echo "[skip] no production log: $LOG"; continue; fi
  mkdir -p "$OUT/logs"

  PILOT_ARGS="$(extract_args "$LOG" lead_v3_pilot)"
  FINAL_ARGS="$(extract_args "$LOG" lead_v3_final)"
  if [ -z "$PILOT_ARGS" ] || [ -z "$FINAL_ARGS" ]; then
    echo "[skip] could not extract pilot/final command from $LOG"; continue
  fi

  # --- transform PILOT: out_root, strip warm-start, add neighbor ---
  PILOT_NEW="$(printf '%s' "$PILOT_ARGS" \
    | sed -E "s| --out_root [^ ]+| --out_root $OUT/lead_v3_pilot|" \
    | sed -E "s| --stage2_warm_start_ckpt [^ ]+ --stage2_warm_start_seed [0-9]+||")"
  PILOT_NEW="$PILOT_NEW --stage2_add_neighbor_history --stage2_neighbor_decay_km $DECAY_KM"

  # --- transform FINAL: out_root, warm-start <- our pilot, add neighbor ---
  FINAL_NEW="$(printf '%s' "$FINAL_ARGS" \
    | sed -E "s| --out_root [^ ]+| --out_root $OUT/lead_v3_final|" \
    | sed -E "s| --stage2_warm_start_ckpt [^ ]+| --stage2_warm_start_ckpt $OUT/lead_v3_pilot/ckpt/checkpoint_run4.pt|")"
  FINAL_NEW="$FINAL_NEW --stage2_add_neighbor_history --stage2_neighbor_decay_km $DECAY_KM"

  # --- DRY-RUN: pilot args, tiny, out_root -> _dryrun ---
  DRY_NEW="$(printf '%s' "$PILOT_NEW" | sed -E "s| --out_root [^ ]+| --out_root $OUT/_dryrun|")"
  DRY_NEW="$DRY_NEW --max_epochs_override 1 --stage2_sanity_only 1 --stage2_sanity_batches 1"

  if [ "$STAGE" = "all" ] || [ "$STAGE" = "dryrun" ]; then
    run_cmd "$DRY_NEW" "$OUT/logs/dryrun.log" "$pest dryrun"
    echo "--- [$pest] d_in check ---"
    grep -E "stage2_neighbor|computed_from_dataset|new_feature_count" "$OUT/logs/dryrun.log" || true
  fi
  [ "$STAGE" = "dryrun" ] && { echo "[dryrun-only] stop for $pest"; continue; }

  if [ "$STAGE" = "all" ] || [ "$STAGE" = "pilot" ]; then
    run_cmd "$PILOT_NEW" "$OUT/logs/pilot.log" "$pest pilot"
  fi
  if [ ! -f "$OUT/lead_v3_pilot/ckpt/checkpoint_run4.pt" ]; then
    echo "[abort:$pest] pilot ckpt missing -> skipping final"; continue
  fi
  if [ "$STAGE" = "all" ] || [ "$STAGE" = "final" ]; then
    run_cmd "$FINAL_NEW" "$OUT/logs/final.log" "$pest final"
  fi
  echo "[done] $pest -> $OUT/lead_v3_final"
done

echo "ALL DONE. Next: evaluate with"
echo "  $PY rice/scripts/eval_s2n_direct_compare.py --pests ${PESTS[*]} sheath_blight"
