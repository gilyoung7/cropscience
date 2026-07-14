#!/usr/bin/env bash
#
# Stage 2 v0 visualization: lead_v3 + val-selected offset=14 on TEST cohort.
#
# Why selector mode (no Stage 2 forward inside run_viz_interval):
#   run_viz_interval.py does NOT auto-append dispatch confidence channels
#   (no stage2_dispatch_features_added handling, no raw-norm forcing, no
#   mu_mode propagation). Feeding the lead_v3 ckpt directly into the script's
#   stage2_ckpt path would crash on input-dim mismatch (d_in=45 vs raw 30) and
#   even if it didn't, would interpret the head as absolute-DOY mu.
#
#   The clean workaround is the selector_per_sample_csv path
#   (line 1102+/1620+ of run_viz_interval.py): it skips Stage 2 forward and
#   reads μ/PI/alert directly from a per-sample CSV. We build that CSV by
#   filtering the existing lead_v3 sample_grid (offset=14 rows only). This
#   uses zero new training/inference — pure file repackaging.
#
# Inputs (must already exist):
#   outputs_phase_B_cohortOnly_lead_v3_off7-60_sample_grid.csv
#     = phase_r_oracle_iou TEST sample_grid for lead_v3, offsets 7..60.
#       offset=14 rows = the val-selected operating point.
#
# Outputs:
#   outputs_phase_B_lead_v3_selector_off14.csv          (intermediate)
#   rice/outputs_phase_B_viz_lead_v3_off14/selector_eval/
#       metrics_summary.csv, metrics_by_lead_bin.csv, per_sample.csv,
#       top/worst/random plot grids (PNG)
#   W&B run: viz_lead_v3_sheath_blight_test_off14
#       project=agro-rice, group=phaseB_lead_v3_visual

set -euo pipefail

# ===== Config =====
GRID_CSV="outputs_phase_B_cohortOnly_lead_v3_off7-60_sample_grid.csv"
OFFSET=14            # val-selected offset
PEST=sheath_blight
RUN=4
VAL_YEAR=2022
TEST_YEAR_MIN=2023
TEST_YEAR_MAX=2024
SPLIT=test           # test cohort

SELECTOR_CSV="outputs_phase_B_lead_v3_selector_off${OFFSET}.csv"
OUT_ROOT="rice/outputs_phase_B_viz_lead_v3_off${OFFSET}"
COHORT_LABEL="lead_v3_test_off${OFFSET}"

WANDB_PROJECT="agro-rice"
WANDB_RUN_NAME="viz_lead_v3_sheath_blight_test_off${OFFSET}"
WANDB_GROUP="phaseB_lead_v3_visual"
WANDB_TAGS="lead_v3,offset${OFFSET},sheath_blight,test,viz"

PY=".venv/bin/python"
[[ -x "$PY" ]] || PY=python

# ===== Sanity =====
if [[ ! -s "$GRID_CSV" ]]; then
  echo "[abort] missing input grid: $GRID_CSV" >&2
  exit 2
fi

mkdir -p "$OUT_ROOT"

# ===== Step 1: build selector_per_sample CSV (offset=14 rows from grid) =====
# Columns expected by rows_from_selector_csv():
#   sample_id, L, R, t_star_doy, sigma, mu_at_pred_off, pred_off
# In our dispatch grid t_star_doy already equals alert_tstar (absolute DOY),
# so the red dashed line (alert) and the orange dotted line
# (alert + pred_off = alert + 14) come out correctly.
$PY - <<EOF
import pandas as pd, sys
src = "$GRID_CSV"; offset = $OFFSET
df = pd.read_csv(src)
n_sy = df["sample_id"].nunique()
sub = df[(df["offset"] == offset) & (df["matched"] == True)].copy()
print(f"[selector_build] source={src}  n_total_sy={n_sy}  "
      f"offset={offset}  matched_rows={len(sub)}")
if len(sub) == 0:
    sys.exit(f"[abort] offset={offset} has no matched rows in {src}")
out = pd.DataFrame({
    "sample_id":      sub["sample_id"].astype(str),
    "L":              sub["L"].astype(int),
    "R":              sub["R"].astype(int),
    "t_star_doy":     sub["t_star_doy"].astype(int),
    "sigma":          sub["sigma"].astype(float),
    "mu_at_pred_off": sub["mu"].astype(float),
    "pred_off":       sub["offset"].astype(int),
})
out.to_csv("$SELECTOR_CSV", index=False)
print(f"[selector_build] wrote $SELECTOR_CSV  rows={len(out)}")
EOF

# n_total denominator = unique sample_id in the original grid
N_TOTAL=$($PY - <<EOF
import pandas as pd
print(pd.read_csv("$GRID_CSV")["sample_id"].nunique())
EOF
)
echo "[viz] n_total_test=${N_TOTAL}  cohort_label=${COHORT_LABEL}"

# ===== Step 2: run_viz_interval in selector mode + W&B =====
$PY -u -m rice.scripts.run_viz_interval \
  --pest "$PEST" --run "$RUN" \
  --val_year "$VAL_YEAR" \
  --test_year_min "$TEST_YEAR_MIN" --test_year_max "$TEST_YEAR_MAX" \
  --split "$SPLIT" \
  --selector_per_sample_csv "$SELECTOR_CSV" \
  --cohort_label "$COHORT_LABEL" \
  --n_total_test "$N_TOTAL" \
  --sigma_eval 5.0 \
  --operational_shift 46.0 \
  --ideal_lead_low 14.0 --ideal_lead_high 30.0 \
  --out_root "$OUT_ROOT" \
  --selector_out_dir "${OUT_ROOT}/selector_eval" \
  --topk 5 --worstk 5 --randomk 5 \
  --selector_random_grid_n 50 --selector_random_grid_cols 2 \
  --selector_Tend 300 \
  --use_wandb \
  --wandb_project "$WANDB_PROJECT" \
  --wandb_run_name "$WANDB_RUN_NAME" \
  --wandb_group "$WANDB_GROUP" \
  --wandb_tags "$WANDB_TAGS" \
  --wandb_job_type selector_eval

echo ""
echo "================================================================"
echo "Done."
echo "  selector CSV    : $SELECTOR_CSV"
echo "  output dir      : ${OUT_ROOT}/selector_eval"
echo "  W&B run name    : $WANDB_RUN_NAME"
echo "  W&B project     : $WANDB_PROJECT"
echo "  W&B group       : $WANDB_GROUP"
echo "================================================================"
