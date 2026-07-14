#!/usr/bin/env bash
#
# Generate auxiliary sample_grids for the multi-year offset selector ablation
# (Path A — cheap). For sheath_blight only.
#
# For each aux year YEAR in {2021, 2022}:
#   1. Build a per-(site, year) dispatch CSV with --val_year=YEAR using the
#      Stage-1 ckpts/summary from the split where YEAR was held-out
#      (split1 for YEAR=2021, split2 for YEAR=2022). YEAR is tagged 'val'.
#   2. Forward the split3 baseline lead_v3 ckpt over the val cohort of the
#      new CSV via run_stage2_dispatch_sample_grid.sh with
#      DISPATCH_FEATURE_CSV_OVERRIDE.
#
# CAVEAT: Stage-1 alerts for YEAR are held-out for the *Stage-1* model used
# (split{1,2}'s Stage-1 had YEAR as val/test). The *Stage-2* lead_v3 model
# is split3's, which DID see YEAR=2021/2022 during training. Predictions on
# the aux years are therefore *training-fit* for Stage-2 — this is the
# "cheap" Path-A trade-off documented in the proposal.

set -euo pipefail

PEST=sheath_blight
RUN=4
S3_LEAD_V3_CKPT="rice/outputs_stage2_batch_2024_bestgate/sheath_blight/lead_v3_final/ckpt/checkpoint_run${RUN}.pt"
AUX_ROOT="rice/outputs_stage2_batch_2024_bestgate/sheath_blight/aux_grids"
mkdir -p "$AUX_ROOT"

PY=".venv/bin/python"
[[ -x "$PY" ]] || PY=python

# Sanity
[[ -s "$S3_LEAD_V3_CKPT" ]] || { echo "[abort] missing split3 lead_v3 ckpt: $S3_LEAD_V3_CKPT" >&2; exit 1; }

# (year, split_dir) mapping where YEAR was held-out for Stage-1
declare -A SPLIT_DIR=(
  ["2021"]="split1_v2021_t2022"
  ["2022"]="split2_v2022_t2023"
)
# In each split, val_year < test_year_min. For YEAR=2021, val=2021 test=2022.
declare -A TEST_YEAR=(
  ["2021"]="2022"
  ["2022"]="2023"
)

for YEAR in 2021 2022; do
  SDIR="${SPLIT_DIR[$YEAR]}"
  TYEAR="${TEST_YEAR[$YEAR]}"
  S1_ROOT="rice/outputs_stage1/batch_rolling/${PEST}/run0/${SDIR}"
  A_CKPT="${S1_ROOT}/A/ckpt/event_xgb_w28_lead14-45_A.pt"
  D_CKPT="${S1_ROOT}/D/ckpt/event_xgb_w28_lead14-45_D.pt"
  SUMMARY="${S1_ROOT}/group_tau/group_tau_hybrid_summary.json"
  for p in "$A_CKPT" "$D_CKPT" "$SUMMARY"; do
    [[ -s "$p" ]] || { echo "[abort] missing $p" >&2; exit 1; }
  done

  DISPATCH_CSV="${AUX_ROOT}/dispatch_val${YEAR}.csv"
  GRID_CSV="${AUX_ROOT}/lead_v3_val${YEAR}_sample_grid.csv"
  ORACLE_CSV="${AUX_ROOT}/lead_v3_val${YEAR}_oracle.csv"

  echo "================================================================"
  echo "[aux YEAR=$YEAR]  split_dir=$SDIR  test_year=$TYEAR"
  echo "  A_ckpt   = $A_CKPT"
  echo "  D_ckpt   = $D_CKPT"
  echo "  summary  = $SUMMARY"
  echo "  out dispatch = $DISPATCH_CSV"
  echo "  out grid     = $GRID_CSV"
  echo "================================================================"

  # 1) Build per-year dispatch CSV. YEAR -> 'val'.
  if [[ ! -s "$DISPATCH_CSV" ]]; then
    $PY -u -m rice.scripts.build_dispatch_feature_table \
      --pest "$PEST" --run "$RUN" \
      --val_year "$YEAR" \
      --test_year_min "$TYEAR" --test_year_max "$TYEAR" \
      --dispatch_summary_json "$SUMMARY" \
      --dispatch_target_label "R>=0.88" \
      --dispatch_a_ckpt "$A_CKPT" \
      --dispatch_d_ckpt "$D_CKPT" \
      --include_splits train,val,test \
      --gate_method dispatch_group_tau \
      --out_csv "$DISPATCH_CSV"
  else
    echo "[skip] dispatch CSV already exists: $DISPATCH_CSV"
  fi

  # 2) Forward split3 lead_v3 over val cohort of new CSV.
  if [[ ! -s "$GRID_CSV" ]]; then
    EVAL_SPLIT=val \
    VAL_YEAR="$YEAR" \
    TEST_YEAR_MIN="$TYEAR" TEST_YEAR_MAX="$TYEAR" \
    STAGE2_CKPT="$S3_LEAD_V3_CKPT" \
    STAGE2_LABEL="${PEST}_lead_v3_aux${YEAR}" \
    OFFSETS="7,14,21,30,45,60" \
    OUT_GRID_CSV="$GRID_CSV" \
    OUT_ORACLE_CSV="$ORACLE_CSV" \
    PEST="$PEST" RUN="$RUN" \
    DISPATCH_SUMMARY="$SUMMARY" \
    DISPATCH_TARGET="R>=0.88" \
    A_CKPT="$A_CKPT" D_CKPT="$D_CKPT" \
    LEGACY_STAGE1_CKPT="$A_CKPT" \
    DISPATCH_GATE_METHOD="dispatch_group_tau" \
    DISPATCH_FEATURE_CSV_OVERRIDE="$DISPATCH_CSV" \
      bash scripts/run_stage2_dispatch_sample_grid.sh
  else
    echo "[skip] sample_grid already exists: $GRID_CSV"
  fi
done

echo
echo "================================================================"
echo "[done] aux grids written under: $AUX_ROOT"
ls -la "$AUX_ROOT"
