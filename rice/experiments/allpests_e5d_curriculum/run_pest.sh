#!/usr/bin/env bash
# Full 3-stage curriculum chain for ONE pest. Idempotent via .done_<step> markers.
#
# Per (phase, year) EXACTLY three run_train calls fire, in order, each warm-starting from the
# checkpoint the previous stage just wrote in the SAME cell:
#     uncond (scratch, hazard) -> pilot (warm from uncond) -> final (warm from pilot)
# The final stage's arguments are byte-identical to the scratch arm's; the ONLY addition is
# --stage2_warm_start_ckpt.
#
#   bash run_pest.sh WBPH
#   PHASES=clean YEARS=2024 bash run_pest.sh WBPH     # single smoke cell
set -euo pipefail
AP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$AP/common.sh"

PEST="${1:?usage: run_pest.sh <pest>}"
YEARS="${YEARS:-$YEARS_DEFAULT}"
PHASES="${PHASES:-dev clean}"
OFFS="${OFFS:-$OFFS_DEFAULT}"
export_e5d_env

pests_all | grep -qx "$PEST" || { echo "[run_pest] unknown pest '$PEST'"; exit 2; }
PROOT="$OUT_ROOT/$PEST"
mkdir -p "$PROOT/logs"
echo "===== CURRICULUM $PEST | phases=$PHASES years=$YEARS | out=$PROOT ====="

stage_dir()  { echo "$PROOT/$1/$2/$3"; }
stage_ckpt() { echo "$(stage_dir "$1" "$2" "$3")/ckpt/checkpoint_run4.pt"; }

train_cell() {                       # train_cell <phase> <year>
  local phase="$1" year="$2" prev="" out args ck stage
  for stage in $STAGES; do
    out="$(stage_dir "$phase" "$year" "$stage")"
    ck="$(stage_ckpt "$phase" "$year" "$stage")"
    if [ -e "$ck" ] && [ "${FORCE:-0}" != "1" ]; then
      echo "[skip] $PEST/$phase/$year/$stage ckpt exists"; prev="$ck"; continue
    fi
    mkdir -p "$out"
    args="$(stage_args "$PEST" "$year" "$stage" "$prev")"

    # ---- audit trail: every stage records its full provenance ----
    echo "============================================================"
    echo "[stage]        $stage"
    echo "[pest/year]    $PEST / $year   phase=$phase"
    echo "[warm_start]   ${prev:-<none: scratch>}"
    if [ -n "$prev" ] && [ "${DRYRUN:-0}" != "1" ]; then
      if [ -e "$prev" ]; then
        echo "[warm_exists]  YES  $(stat -c %s "$prev") bytes"
      else
        echo "[warm_exists]  NO -- ABORT (previous stage produced no checkpoint)"; exit 4
      fi
    fi
    echo "[loss]         $(printf '%s' "$args" | grep -oE -- '--stage2_pmf_mode [^ ]+|--stage2_pmf_mu_mode [^ ]+|--stage2_pmf_asym_weight [^ ]+|--stage2_pmf_asym_weight_early [^ ]+|--stage2_conditional_survival [^ ]+' | tr '\n' ' ')"
    echo "[lr/seed]      $(printf '%s' "$args" | grep -oE -- '--lr [^ ]+|--seeds [^ ]+|--split_seed [^ ]+|--split_mode [^ ]+' | tr '\n' ' ')"
    echo "[split/year]   $(printf '%s' "$args" | grep -oE -- '--val_year [^ ]+|--test_year_min [^ ]+|--test_year_max [^ ]+' | tr '\n' ' ')"
    echo "[out_root]     $out"
    echo "============================================================"
    if [ "${DRYRUN:-0}" = "1" ]; then echo "[DRYRUN] skip training"; prev="$ck"; continue; fi

    cd "$CS"
    if [ "$phase" = "dev" ]; then
      PYTHONPATH="$VENDOR:$CS" $PY -u -m src.vendor.run_train $args --out_root "$out" \
        2>&1 | tee "$out/train.log"
    else
      assign="$PROOT/clean/split_assignment.json"
      [ -f "$assign" ] || { echo "[abort] missing $assign (splits step must run first)"; exit 4; }
      PYTHONPATH="$VENDOR:$CS" $PY -u "$VENDOR/_patched_train.py" \
        --eval_year "$year" --assign "$assign" -- $args --out_root "$out" \
        2>&1 | tee "$out/train.log"
    fi
    [ -e "$ck" ] || { echo "[abort] $stage produced no checkpoint at $ck"; exit 5; }
    prev="$ck"
  done
}

train_phase() { local phase="$1" y; for y in $YEARS; do train_cell "$phase" "$y"; done; }

run_step() {
  local name="$1"; shift
  if step_done "$PEST" "$name"; then echo "[skip] $PEST/$name (marker present)"; return 0; fi
  echo "--- step $name"
  "$@"
  if [ "${DRYRUN:-0}" = "1" ]; then echo "    [DRYRUN] marker NOT written for $name"; else
    mark_done "$PEST" "$name"
  fi
}

grid()     { cd "$CS"; PYTHONPATH="$VENDOR:$CS" $PY "$AP/pest_grid.py"   --pest "$PEST" --mode "$1" --years $YEARS --force 2>&1 | tee "$PROOT/logs/grid_$1.log"; }
splits()   { cd "$CS"; PYTHONPATH="$VENDOR:$CS" $PY "$AP/pest_splits.py" --pest "$PEST" --years $YEARS 2>&1 | tee "$PROOT/logs/splits.log"; }
evaluate() { cd "$CS"; PYTHONPATH="$VENDOR:$CS" $PY "$AP/pest_eval.py"   --pest "$PEST" --years $YEARS 2>&1 | tee "$PROOT/logs/eval.log"; }

case "$PHASES" in *dev*)   run_step train_dev   train_phase dev ;; esac
case "$PHASES" in *dev*)   run_step grid_dev    grid dev ;; esac
case "$PHASES" in *clean*) run_step splits      splits ;; esac
case "$PHASES" in *clean*) run_step train_clean train_phase clean ;; esac
case "$PHASES" in *clean*) run_step grid_clean  grid clean ;; esac
run_step eval evaluate

echo "===== CURRICULUM $PEST DONE -> $PROOT ====="
