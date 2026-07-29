#!/usr/bin/env bash
# Full E5d chain for ONE pest. Idempotent: every step writes a .done_<step> marker and is
# skipped on re-run, so killing this mid-way and re-launching resumes where it stopped.
# FORCE=1 ignores all markers.
#
#   bash rice/experiments/allpests_e5d/run_pest.sh blast
#   FORCE=1 bash rice/experiments/allpests_e5d/run_pest.sh blast          # redo everything
#   YEARS="2024" bash rice/experiments/allpests_e5d/run_pest.sh blast     # single year
#
# Chain (dev must finish before clean: the clean splits enumerate sample ids from the dev grid)
#   1 train_dev    -> dev/ckpt/<year>/         E5d, val = whole y-1 season   (Selection-OK=FAIL)
#   2 grid_dev     -> dev/grid/<pest>_...csv   RAW predictions, offsets 1..75
#   3 splits       -> clean/split_assignment.json  40/30/30 by md5(sample_id)
#   4 train_clean  -> clean/ckpt/<year>/       identical recipe, val restricted to val_ckpt
#   5 grid_clean   -> clean/grid/<pest>_...csv RAW predictions
#   6 eval         -> eval/                    dev calibration + clean fold-isolated
set -euo pipefail
AP="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$AP/common.sh"

PEST="${1:?usage: run_pest.sh <pest>}"
YEARS="${YEARS:-$YEARS_DEFAULT}"
OFFS="${OFFS:-$OFFS_DEFAULT}"
export_e5d_env

grep -v '^#' "$AP/pests.tsv" | awk 'NF{print $1}' | grep -qx "$PEST" \
  || { echo "[run_pest] unknown pest '$PEST' (not in pests.tsv)"; exit 2; }

PROOT="$OUT_ROOT/$PEST"
mkdir -p "$PROOT/logs"
echo "===== $PEST | years=$YEARS | out=$PROOT ====="

# ---------------------------------------------------------------- 1 & 4: training
# $1 = dev|clean. The two differ ONLY in out_root and, for clean, the val_ckpt restriction --
# same extracted production args, same E5d loss knobs, same env, same seed.
train_phase() {
  local phase="$1" year out args
  for year in $YEARS; do
    out="$PROOT/$phase/ckpt/$year"
    if [ -e "$out/ckpt/checkpoint_run4.pt" ] && [ "${FORCE:-0}" != "1" ]; then
      echo "[skip] $PEST/$phase/$year ckpt exists"; continue
    fi
    mkdir -p "$out"
    args="$(extract_args "$PEST" "$year")"
    echo "--- train $PEST/$phase/$year"
    echo "    loss: $(printf '%s' "$args" | grep -oE -- '--stage2_pmf_(target_mode|asym_weight|asym_weight_early) [^ ]+' | tr '\n' ' ')"
    if [ "${DRYRUN:-0}" = "1" ]; then echo "    [DRYRUN] skip"; continue; fi
    cd "$CS"
    if [ "$phase" = "dev" ]; then
      PYTHONPATH="$WS:$CS" $PY -u -m src.vendor.run_train $args --out_root "$out" \
        2>&1 | tee "$out/train.log"
    else
      local assign="$PROOT/clean/split_assignment.json"
      [ -f "$assign" ] || { echo "[abort] missing $assign (step 3 must run first)"; exit 4; }
      PYTHONPATH="$WS:$CS" $PY -u \
        "$WS/outputs/feature_experiments/e5d_clean_selection_3fold_20260716/_code/_patched_train.py" \
        --eval_year "$year" --assign "$assign" -- $args --out_root "$out" \
        2>&1 | tee "$out/train.log"
    fi
  done
}

run_step() {                       # run_step <name> <command...>
  local name="$1"; shift
  if step_done "$PEST" "$name"; then echo "[skip] $PEST/$name (marker present)"; return 0; fi
  echo "--- step $name"
  "$@"
  # DRYRUN must never write a marker: doing so would make a later real run skip the step
  # it never actually performed.
  if [ "${DRYRUN:-0}" = "1" ]; then echo "    [DRYRUN] marker NOT written for $name"; else
    mark_done "$PEST" "$name"
  fi
}

grid()   { cd "$CS"; PYTHONPATH="$WS:$CS" $PY "$AP/pest_grid.py"   --pest "$PEST" --mode "$1" --years $YEARS --force \
             2>&1 | tee "$PROOT/logs/grid_$1.log"; }
splits() { cd "$CS"; PYTHONPATH="$WS:$CS" $PY "$AP/pest_splits.py" --pest "$PEST" --years $YEARS \
             2>&1 | tee "$PROOT/logs/splits.log"; }
evaluate() { cd "$CS"; PYTHONPATH="$WS:$CS" $PY "$AP/pest_eval.py" --pest "$PEST" --years $YEARS \
             2>&1 | tee "$PROOT/logs/eval.log"; }

run_step train_dev   train_phase dev
run_step grid_dev    grid dev
run_step splits      splits
run_step train_clean train_phase clean
run_step grid_clean  grid clean
run_step eval        evaluate

echo "===== $PEST DONE -> $PROOT ====="
