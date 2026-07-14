#!/usr/bin/env bash
#
# Drive run_viz_interval_selector.py across all 8 pests × 3 years.
#
# Fallback logic for the SELECTOR_NAME passed to the viz script:
#   - read best_selector_name from selector_by_pest_year.csv
#   - if that selector is NOT present in the per-sample selections CSV (e.g.
#     v1_bin_rule[...] which V2 doesn't store per-sample), fall back to the
#     best-test-IoU V2 selector for that (pest, year) cell instead. This
#     keeps the viz script honest (it can only render selector names whose
#     per-sample picks are stored in the V2 CSV).
#
# Resume / skip-completed:
#   - skip marker = <out_dir>/fig_top_interval.png (legacy-format figure).
#     Earlier runs that only produced fig_small_multiples.png will RE-RUN
#     and overwrite with the new format (top / worst / random / grid +
#     calibration scatter × 2 + lead-bin bars).
#   - Set FORCE_REDO=1 to force re-run even when the marker exists.
#
# Override entry points via env: PROJECT (W&B project), PETS, YEARS.

set -uo pipefail
trap 'echo "[driver] interrupted at $(date -Iseconds)"; exit 130' INT TERM

PROJECT="${WANDB_PROJECT:-rice-pest-stage2}"
SUMMARY_CSV="rice/outputs_stage2_selector_cross_split/selector_by_pest_year.csv"

PETS="${PETS:-BPH WBPH bacterial_blight blast brown_spot rice_stem_borer_1 rice_stem_borer_2 sheath_blight}"
YEARS="${YEARS:-2022 2023 2024}"

FAIL_LOG="rice/outputs_viz_selector/_driver_failures.log"
mkdir -p "rice/outputs_viz_selector"
: > "$FAIL_LOG"
record_fail() {
  echo "$(date -Iseconds) FAIL pest=$1 year=$2 step=$3 rc=$4 detail=$5" | tee -a "$FAIL_LOG" >&2
}

# bash 3-safe quoting for special chars (no associative arrays needed)
safe_selector() {
  echo "$1" | tr '[]()/, ' '______'
}

# Resolve effective selector (v1 → best-v2 fallback) using a single Python
# call per (pest, year). Returns the selector name on stdout, or "" on error.
resolve_selector() {
  local pest=$1 year=$2 offsets_csv=$3
  .venv/bin/python - <<PY 2>/dev/null
import pandas as pd, sys, os
SUM = "$SUMMARY_CSV"
PEST = "$pest"
YEAR = int("$year")
PER_SAMPLE = "$offsets_csv"

df = pd.read_csv(SUM)
row = df[(df["year"] == YEAR) & (df["pest"] == PEST)]
if row.empty:
    sys.exit(1)
best_name = str(row.iloc[0]["best_selector_name"])

# Available selectors in the per-sample CSV
per = pd.read_csv(PER_SAMPLE)
available = set(per["selector"].unique())

if best_name in available:
    print(best_name)
    sys.exit(0)

# Fallback: pick best-test-IoU V2 selector for this cell.
# v2_test_results_summary.csv lives next to per-sample selections.
results_csv = os.path.join(os.path.dirname(PER_SAMPLE), "v2_test_results_summary.csv")
if not os.path.isfile(results_csv):
    sys.exit(2)
res = pd.read_csv(results_csv)
v2_rows = res[res["selector"].astype(str).str.startswith("v2_")]
v2_rows = v2_rows[v2_rows["selector"].isin(available)]
if v2_rows.empty:
    sys.exit(3)
best_v2 = v2_rows.loc[v2_rows["test_iou"].idxmax(), "selector"]
print(str(best_v2))
PY
}

echo "================================================================"
echo "Selector-aware Stage 2 interval viz — all pests × all years"
echo "  W&B project: $PROJECT"
echo "  pests: $PETS"
echo "  years: $YEARS"
echo "================================================================"

n_done=0; n_skip=0; n_fail=0
for YEAR in $YEARS; do
  case "$YEAR" in
    2022) ROOT="rice/outputs_stage2_batch_2022_baseline" ;;
    2023) ROOT="rice/outputs_stage2_batch_2023_baseline" ;;
    2024) ROOT="rice/outputs_stage2_batch_2024_bestgate" ;;
    *)    echo "[abort] unknown year $YEAR" >&2; exit 2 ;;
  esac

  for PEST in $PETS; do
    SAMPLE_GRID="${ROOT}/${PEST}/lead_v3_test_sample_grid.csv"
    SELECTOR_OFFSETS="rice/outputs_stage2_selector_cross_split/${YEAR}_${PEST}/v2_per_sample_test_selections.csv"

    if [[ ! -s "$SAMPLE_GRID" ]]; then
      echo "[SKIP] $PEST $YEAR — missing sample_grid $SAMPLE_GRID"
      n_skip=$((n_skip+1)); continue
    fi
    if [[ ! -s "$SELECTOR_OFFSETS" ]]; then
      echo "[SKIP] $PEST $YEAR — missing selector_offsets $SELECTOR_OFFSETS"
      n_skip=$((n_skip+1)); continue
    fi

    SELECTOR_NAME=$(resolve_selector "$PEST" "$YEAR" "$SELECTOR_OFFSETS")
    rc=$?
    if [[ -z "$SELECTOR_NAME" || $rc -ne 0 ]]; then
      record_fail "$PEST" "$YEAR" "resolve_selector" "$rc" "$SELECTOR_OFFSETS"
      n_fail=$((n_fail+1)); continue
    fi
    SAFE=$(safe_selector "$SELECTOR_NAME")
    OUT_DIR="rice/outputs_viz_selector/${PEST}_${YEAR}_${SAFE}"
    DONE_MARKER="${OUT_DIR}/fig_top_interval.png"
    RUN_NAME="${PEST}_${YEAR}_${SAFE}_selector_viz"

    if [[ -s "$DONE_MARKER" && "${FORCE_REDO:-0}" != "1" ]]; then
      echo "[SKIP-DONE] $PEST $YEAR selector=$SELECTOR_NAME (fig_top_interval.png exists)"
      n_skip=$((n_skip+1)); continue
    fi

    echo "============================================================"
    echo "[RUN] $PEST $YEAR  selector=$SELECTOR_NAME"
    echo "      grid=$SAMPLE_GRID"
    echo "      offsets=$SELECTOR_OFFSETS"
    echo "      run_name=$RUN_NAME"
    echo "============================================================"

    .venv/bin/python rice/scripts/run_viz_interval_selector.py \
      --pest "$PEST" --year "$YEAR" \
      --sample_grid "$SAMPLE_GRID" \
      --selector_offsets "$SELECTOR_OFFSETS" \
      --selector_name "$SELECTOR_NAME" \
      --mode selector \
      --wandb_project "$PROJECT" \
      --wandb_run_name "$RUN_NAME"
    rc=$?
    if [[ $rc -ne 0 ]]; then
      record_fail "$PEST" "$YEAR" "viz" "$rc" "selector=$SELECTOR_NAME"
      n_fail=$((n_fail+1))
    else
      n_done=$((n_done+1))
    fi
  done
done

echo
echo "================================================================"
echo "Driver done: $(date -Iseconds)"
echo "  ok=$n_done  skip=$n_skip  fail=$n_fail"
echo "  failures log: $FAIL_LOG"
echo "================================================================"
