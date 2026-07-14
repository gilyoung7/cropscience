#!/usr/bin/env bash
#
# Phase 1 — generate Stage-2 sample_grid using the Stage-1 dispatch_group_tau
# alert map (R>=0.88 selection from the year-split summary JSON).
#
# Does NOT retrain Stage 2 — uses the supplied Stage-2 ckpt for forward-only
# mu lookup (same row_map / interval / mu / sigma logic as the canonical
# phase_r_oracle_iou path). Only the alert source and 14 dispatch feature
# columns + alert_tstar update differ.
#
# Required env (override on the cmdline if needed):
#   STAGE2_CKPT     Stage-2 ckpt (.pt)
#   STAGE2_LABEL    Label string used in the sample_grid 'model' column
#   OUT_GRID_CSV    Output sample_grid CSV path
#   OUT_ORACLE_CSV  Oracle summary CSV path (phase_r oracle table)
#
# Pre-set Stage-1 dispatch context (from Phase 0):
#   - year-split val=2022, test=2023-2024
#   - R>=0.88 selection in the year-split group_tau_hybrid_summary.json
#   - A baseline ckpt: lead14-45_ignore (no history)
#   - D history ckpt:  lead14-45_history_rolling

set -euo pipefail

PEST=${PEST:-sheath_blight}
RUN=${RUN:-4}
VAL_YEAR=${VAL_YEAR:-2022}
TEST_YEAR_MIN=${TEST_YEAR_MIN:-2023}
TEST_YEAR_MAX=${TEST_YEAR_MAX:-2024}

DISPATCH_SUMMARY=${DISPATCH_SUMMARY:-rice/outputs_stage1/sheath_blight_yearsplit2023-24_lead14-45_history_rolling/group_tau_hybrid/group_tau_hybrid_summary.json}
DISPATCH_TARGET=${DISPATCH_TARGET:-R>=0.88}
A_CKPT=${A_CKPT:-rice/outputs_stage1/sheath_blight_yearsplit2023-24_lead14-45_ignore/ckpt/event_run4_xgb_w28_lead14-45_ignore.pt}
D_CKPT=${D_CKPT:-rice/outputs_stage1/sheath_blight_yearsplit2023-24_lead14-45_history_rolling/ckpt/event_run4_xgb_w28_lead14-45_history_rolling.pt}

# Legacy stage1 ckpt is used only for the lead-shift diagnostic
# (k=1 first-crossing baseline alert_map). Default to the same A ckpt.
LEGACY_STAGE1_CKPT=${LEGACY_STAGE1_CKPT:-$A_CKPT}

STAGE2_CKPT=${STAGE2_CKPT:-}
STAGE2_LABEL=${STAGE2_LABEL:-stage2_default}
OUT_GRID_CSV=${OUT_GRID_CSV:-outputs_phase_dispatch_sample_grid.csv}
OUT_ORACLE_CSV=${OUT_ORACLE_CSV:-outputs_phase_dispatch_oracle.csv}
OFFSETS=${OFFSETS:-60,90,105,120}
ORACLE_SIGMA=${ORACLE_SIGMA:-5.0}
# Phase A.3 ablation knob (default empty = production fill from ckpt meta).
# Accepted: "" | "ablate_missing" | "ablate_train_mean"
DISPATCH_ABLATION_MODE=${DISPATCH_ABLATION_MODE:-}
DISPATCH_ABLATION_CSV=${DISPATCH_ABLATION_CSV:-}
# Stage 1 gate policy passed to phase_r's build_dispatch_alert_map.
# Accepted: "dispatch_group_tau" (default) | "A_baseline" | "D_history"
DISPATCH_GATE_METHOD=${DISPATCH_GATE_METHOD:-dispatch_group_tau}
# Which split to evaluate. 'test' (default) reproduces existing behavior;
# 'val' produces the validation-side sample_grid for offset selection.
EVAL_SPLIT=${EVAL_SPLIT:-test}
# Optional: override the ckpt-stored stage2_dispatch_feature_csv path with
# an aux CSV (e.g. one built for a different val/test-year partition). Used
# by multi-year offset-selector aux grid generation. Empty = use ckpt path.
DISPATCH_FEATURE_CSV_OVERRIDE=${DISPATCH_FEATURE_CSV_OVERRIDE:-}

if [[ -z "$STAGE2_CKPT" ]]; then
  echo "[abort] STAGE2_CKPT env required (path to existing Stage-2 ckpt)" >&2
  exit 1
fi
for path in "$A_CKPT" "$D_CKPT" "$DISPATCH_SUMMARY" "$STAGE2_CKPT" "$LEGACY_STAGE1_CKPT"; do
  if [[ ! -s "$path" ]]; then
    echo "[abort] missing or empty: $path" >&2
    exit 1
  fi
done

echo "================================================================"
echo "Phase 1 dispatch sample_grid"
echo "  pest=$PEST  run=$RUN  val=$VAL_YEAR  test=$TEST_YEAR_MIN..$TEST_YEAR_MAX"
echo "  dispatch_summary  = $DISPATCH_SUMMARY"
echo "  dispatch_target   = $DISPATCH_TARGET"
echo "  A_ckpt            = $A_CKPT"
echo "  D_ckpt            = $D_CKPT"
echo "  legacy_stage1     = $LEGACY_STAGE1_CKPT  (lead-shift diagnostic only)"
echo "  stage2_ckpt       = $STAGE2_CKPT"
echo "  out_grid          = $OUT_GRID_CSV"
echo "  out_oracle        = $OUT_ORACLE_CSV"
echo "================================================================"

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then PY=python; fi

EXTRA_ARGS=()
if [[ -n "$DISPATCH_ABLATION_MODE" ]]; then
  EXTRA_ARGS+=(--dispatch_ablation_mode "$DISPATCH_ABLATION_MODE")
fi
if [[ -n "$DISPATCH_ABLATION_CSV" ]]; then
  EXTRA_ARGS+=(--dispatch_ablation_csv "$DISPATCH_ABLATION_CSV")
fi
if [[ -n "$DISPATCH_FEATURE_CSV_OVERRIDE" ]]; then
  EXTRA_ARGS+=(--stage2_dispatch_feature_csv_override "$DISPATCH_FEATURE_CSV_OVERRIDE")
fi

$PY -u -m rice.scripts.phase_r_oracle_iou \
  --pest "$PEST" --run "$RUN" \
  --val_year "$VAL_YEAR" \
  --test_year_min "$TEST_YEAR_MIN" --test_year_max "$TEST_YEAR_MAX" \
  --stage1_ckpt "$LEGACY_STAGE1_CKPT" \
  --models "${STAGE2_LABEL}|${STAGE2_CKPT}" \
  --offsets "$OFFSETS" \
  --oracle_sigma "$ORACLE_SIGMA" \
  --dispatch_summary_json "$DISPATCH_SUMMARY" \
  --dispatch_target_label "$DISPATCH_TARGET" \
  --dispatch_a_ckpt "$A_CKPT" \
  --dispatch_d_ckpt "$D_CKPT" \
  --sample_grid_csv "$OUT_GRID_CSV" \
  --out_csv "$OUT_ORACLE_CSV" \
  --eval_split "$EVAL_SPLIT" \
  --dispatch_gate_method "$DISPATCH_GATE_METHOD" \
  "${EXTRA_ARGS[@]}"

echo ""
echo "================================================================"
echo "Phase 1 validator:"
echo "================================================================"
$PY -m rice.scripts.phase1_dispatch_sample_grid_report \
  --sample_grid_csv "$OUT_GRID_CSV"
