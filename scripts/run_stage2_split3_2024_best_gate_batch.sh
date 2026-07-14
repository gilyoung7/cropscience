#!/usr/bin/env bash
#
# Stage 2 lead_v3 + climatology batch for split3 ONLY (val=2023 / test=2024).
#
# Skips Stage 1 entirely: uses Stage 1 artifacts already in
# rice/outputs_stage1/batch_rolling/<pest>/run<S>/split3_v2023_t2024/,
# selecting the best-run via select_stage1_gate_split3_2024 (val-only).
#
# Per-pest pipeline (each step skip-if-exists, graceful failure):
#   0. Stage 1 gate selection (val-only)  — once, before the pest loop
#   1. dispatch feature CSV (train+val+test) from selected A/D ckpt + summary
#   2. Stage 2 uncond  (hazard head, only if missing)
#   3. Stage 2 pilot + final via phase_s5_train
#       (lead_from_alert, lead_min=7, lead_max=75, cohort_dispatch_only)
#   4. val + test sample_grid (offsets=7,14,21,30,45,60)
#   5. climatology baselines (mean_L / mean_mid / mean_R) val + test grids
#   6. per-pest canonical summary
# Final: all-pest aggregate + winner table (lead_v3 vs best climatology).
#
# Default 8 pest list excludes BPH2.

set -uo pipefail
trap 'echo "[batch] interrupted at $(date -Iseconds)"; exit 130' INT TERM

DEFAULT_PESTS="WBPH bacterial_blight brown_spot BPH rice_stem_borer_1 rice_stem_borer_2 blast sheath_blight"

VAL_YEAR=2023
TEST_YEAR_MIN=2024
TEST_YEAR_MAX=2024
RUN=4
DISPATCH_TARGET="R>=0.88"
LEAD_MIN=7
LEAD_MAX=75
FEATURE_MODE=causal
OFFSETS_GRID="7,14,21,30,45,60"

PESTS=""
FORCE=0
OUT_ROOT="rice/outputs_stage2_batch_2024"
SUMMARY_ROOT="${OUT_ROOT}/_summary"
GATE_CSV="${SUMMARY_ROOT}/stage1_gate_selection_split3_2024.csv"
FAIL_LOG=""

# uncond candidate paths (per pest). If absent we train one from scratch.
declare -A UNCOND_PATH=(
  ["sheath_blight"]="rice/outputs_stage2_sheath_blight_d15_asym25_2sided_uncond/ckpt/checkpoint_run4.pt"
  ["blast"]="rice/outputs_stage2_blast_uncond/ckpt/checkpoint_run4.pt"
  ["bacterial_blight"]="rice/outputs_stage2_bacterial_blight_uncond/ckpt/checkpoint_run4.pt"
  ["brown_spot"]="rice/outputs_stage2_brown_spot_uncond/ckpt/checkpoint_run4.pt"
  ["WBPH"]="rice/outputs_stage2_WBPH_uncond/ckpt/checkpoint_run4.pt"
  ["BPH"]="rice/outputs_stage2_BPH_uncond/ckpt/checkpoint_run4.pt"
  ["rice_stem_borer_1"]="rice/outputs_stage2_rice_stem_borer_1_uncond/ckpt/checkpoint_run4.pt"
  ["rice_stem_borer_2"]="rice/outputs_stage2_rice_stem_borer_2_uncond/ckpt/checkpoint_run4.pt"
)

usage() {
  cat <<EOF
Usage: $0 [--pests "p1 p2 ..."] [--force]
  --pests "p1 p2 ..."  default: $DEFAULT_PESTS
  --force              re-run every step even if outputs exist
  --out_root PATH      default: $OUT_ROOT
  -h, --help
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pests) PESTS="$2"; shift 2;;
    --force) FORCE=1; shift;;
    --out_root) OUT_ROOT="$2"; SUMMARY_ROOT="${OUT_ROOT}/_summary"; GATE_CSV="${SUMMARY_ROOT}/stage1_gate_selection_split3_2024.csv"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[abort] unknown arg: $1" >&2; usage; exit 2;;
  esac
done
PESTS="${PESTS:-$DEFAULT_PESTS}"
FAIL_LOG="${SUMMARY_ROOT}/batch_failures.log"
mkdir -p "$OUT_ROOT" "$SUMMARY_ROOT"
touch "$FAIL_LOG"

PY=".venv/bin/python"
[[ -x "$PY" ]] || PY=python

record_fail() {
  echo "$(date -Iseconds) FAIL pest=$1 step=$2 rc=$3 log=$4" | tee -a "$FAIL_LOG" >&2
}
run_step() {
  local tag=$1 pest=$2 logfile=$3; shift 3
  [[ "$1" == "--" ]] && shift
  echo "  [run] $tag  -> $logfile"
  "$@" >>"$logfile" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then record_fail "$pest" "$tag" "$rc" "$logfile"; return $rc; fi
  return 0
}

echo "================================================================"
echo "Stage 2 split3-only batch (val=$VAL_YEAR test=$TEST_YEAR_MIN..$TEST_YEAR_MAX)"
echo "  pests: $PESTS"
echo "  out_root: $OUT_ROOT"
echo "================================================================"

# ===== Step 0: gate selection (val-only) =====
if [[ ! -s "$GATE_CSV" || "$FORCE" == "1" ]]; then
  $PY -u -m rice.scripts.select_stage1_gate_split3_2024 \
    --exclude_pests BPH2 \
    --out_csv "$GATE_CSV" \
    2>&1 | tee "${SUMMARY_ROOT}/stage1_gate_selection.log"
fi

# Helper: read selected fields from GATE_CSV for a given pest
read_gate_field() {
  local pest=$1 col=$2
  $PY - <<PYEOF 2>/dev/null
import pandas as pd, sys
df = pd.read_csv("$GATE_CSV")
r = df[df['pest']=='$pest']
if len(r)==0: sys.exit(1)
print(r.iloc[0]['$col'])
PYEOF
}

# ===== per-pest loop =====
for PEST in $PESTS; do
  echo
  echo "================================================================"
  echo "[pest=$PEST]"
  echo "================================================================"

  SEL_RUN=$(read_gate_field "$PEST" "selected_run") || { record_fail "$PEST" "gate_lookup" 1 "-"; continue; }
  SEL_METHOD=$(read_gate_field "$PEST" "selected_method")
  A_CKPT=$(read_gate_field "$PEST" "a_ckpt")
  D_CKPT=$(read_gate_field "$PEST" "d_ckpt")
  DISPATCH_SUMMARY=$(read_gate_field "$PEST" "dispatch_summary")
  echo "  gate: method=$SEL_METHOD  run=$SEL_RUN"
  echo "  A_ckpt   = $A_CKPT"
  echo "  D_ckpt   = $D_CKPT"
  echo "  dispatch = $DISPATCH_SUMMARY"

  for p in "$A_CKPT" "$D_CKPT" "$DISPATCH_SUMMARY"; do
    if [[ ! -s "$p" ]]; then
      record_fail "$PEST" "stage1_artifact_missing" 1 "$p"; continue 2
    fi
  done

  PEST_OUT="${OUT_ROOT}/${PEST}"
  LOG_DIR="${PEST_OUT}/logs"
  mkdir -p "${PEST_OUT}" "${LOG_DIR}"

  DISPATCH_CSV="${PEST_OUT}/dispatch_R088_features_per_sy.csv"
  UNCOND_CKPT="${UNCOND_PATH[$PEST]:-rice/outputs_stage2_${PEST}_uncond/ckpt/checkpoint_run${RUN}.pt}"
  UNCOND_ROOT="$(dirname "$(dirname "$UNCOND_CKPT")")"

  PILOT_OUT="${PEST_OUT}/lead_v3_pilot"
  FINAL_OUT="${PEST_OUT}/lead_v3_final"
  PILOT_CKPT="${PILOT_OUT}/ckpt/checkpoint_run${RUN}.pt"
  FINAL_CKPT="${FINAL_OUT}/ckpt/checkpoint_run${RUN}.pt"
  GRID_VAL="${PEST_OUT}/lead_v3_val_sample_grid.csv"
  GRID_TEST="${PEST_OUT}/lead_v3_test_sample_grid.csv"
  ORACLE_VAL="${PEST_OUT}/lead_v3_val_oracle.csv"
  ORACLE_TEST="${PEST_OUT}/lead_v3_test_oracle.csv"
  CLIM_PREFIX="${PEST_OUT}/climatology"
  CLIM_STATS="${CLIM_PREFIX}_train_stats.csv"

  [[ "$FORCE" == "1" ]] && rm -f "$DISPATCH_CSV" "$PILOT_CKPT" "$FINAL_CKPT" "$GRID_VAL" "$GRID_TEST" "$CLIM_STATS"

  # 1) dispatch feature CSV
  if [[ ! -s "$DISPATCH_CSV" ]]; then
    run_step "dispatch_feature_table" "$PEST" "${LOG_DIR}/dispatch_feature_table.log" -- \
      $PY -u -m rice.scripts.build_dispatch_feature_table \
        --pest "$PEST" --run "$RUN" \
        --val_year "$VAL_YEAR" \
        --test_year_min "$TEST_YEAR_MIN" --test_year_max "$TEST_YEAR_MAX" \
        --dispatch_summary_json "$DISPATCH_SUMMARY" \
        --dispatch_target_label "$DISPATCH_TARGET" \
        --dispatch_a_ckpt "$A_CKPT" \
        --dispatch_d_ckpt "$D_CKPT" \
        --include_splits train,val,test \
        --out_csv "$DISPATCH_CSV" || continue
  else echo "  [skip] dispatch CSV exists"; fi

  # 2) uncond Stage 2
  if [[ ! -s "$UNCOND_CKPT" ]]; then
    mkdir -p "$(dirname "$UNCOND_CKPT")"
    run_step "stage2_uncond_train" "$PEST" "${LOG_DIR}/stage2_uncond_train.log" -- \
      $PY -u -m rice.scripts.run_train \
        --pest "$PEST" --run "$RUN" --seeds 0 \
        --split_seed 42 --split_mode year \
        --val_year "$VAL_YEAR" \
        --test_year_min "$TEST_YEAR_MIN" --test_year_max "$TEST_YEAR_MAX" \
        --dropout 0.2 --weight_decay 0.0001 --lr 0.0001 \
        --w_interval 1.0 --w_left 0.5 --w_right 0.5 \
        --stage2_nowcast \
        --stage2_nowcast_window 28 --stage2_nowcast_stride 1 \
        --stage2_nowcast_only_pre_event 1 \
        --stage2_nowcast_event_time_proxy r \
        --stage2_nowcast_require_tstar_before_L 0 \
        --stage2_causal_tstar --stage2_tstar_layers 1 \
        --stage2_early_tstar_weight_min 0.2 \
        --stage2_pmf_mode hazard --stage2_pmf_sigma 5.0 \
        --stage2_pmf_target_offset 0.0 --stage2_pmf_right_weight 0.3 \
        --stage2_best_metric val_iou80 \
        --amp 1 --amp_dtype bf16 --d_model_override 48 \
        --out_root "$UNCOND_ROOT" --out "$UNCOND_CKPT" || continue
  else echo "  [skip] uncond ckpt exists ($UNCOND_CKPT)"; fi

  # 3) pilot + final lead_v3
  if [[ ! -s "$FINAL_CKPT" ]]; then
    run_step "stage2_lead_v3_train" "$PEST" "${LOG_DIR}/stage2_lead_v3_train.log" -- \
      $PY -u -m rice.scripts.phase_s5_train \
        --pest "$PEST" --run "$RUN" --seed 0 \
        --val_year "$VAL_YEAR" \
        --test_year_min "$TEST_YEAR_MIN" --test_year_max "$TEST_YEAR_MAX" \
        --uncond_ckpt "$UNCOND_CKPT" \
        --out_pilot "$PILOT_OUT" --out_final "$FINAL_OUT" \
        --dispatch_feature_csv "$DISPATCH_CSV" \
        --dispatch_feature_mode "$FEATURE_MODE" \
        --cohort_dispatch_only \
        --mu_mode lead_from_alert \
        --lead_min "$LEAD_MIN" --lead_max "$LEAD_MAX" || continue
  else echo "  [skip] lead_v3 final ckpt exists"; fi

  # 4) val + test sample_grid
  for SPLIT_TAG in test val; do
    case "$SPLIT_TAG" in
      test) G="$GRID_TEST"; O="$ORACLE_TEST";;
      val)  G="$GRID_VAL";  O="$ORACLE_VAL";;
    esac
    if [[ ! -s "$G" ]]; then
      run_step "stage2_grid_${SPLIT_TAG}" "$PEST" "${LOG_DIR}/stage2_grid_${SPLIT_TAG}.log" -- \
        env STAGE2_CKPT="$FINAL_CKPT" \
            STAGE2_LABEL="${PEST}_lead_v3" \
            OFFSETS="$OFFSETS_GRID" \
            EVAL_SPLIT="$SPLIT_TAG" \
            OUT_GRID_CSV="$G" OUT_ORACLE_CSV="$O" \
            PEST="$PEST" RUN="$RUN" \
            VAL_YEAR="$VAL_YEAR" TEST_YEAR_MIN="$TEST_YEAR_MIN" TEST_YEAR_MAX="$TEST_YEAR_MAX" \
            DISPATCH_SUMMARY="$DISPATCH_SUMMARY" \
            DISPATCH_TARGET="$DISPATCH_TARGET" \
            A_CKPT="$A_CKPT" D_CKPT="$D_CKPT" \
            LEGACY_STAGE1_CKPT="$A_CKPT" \
            bash scripts/run_stage2_dispatch_sample_grid.sh || continue
    else echo "  [skip] ${SPLIT_TAG} grid exists"; fi
  done

  # 5) climatology baselines
  if [[ ! -s "$CLIM_STATS" ]]; then
    run_step "climatology" "$PEST" "${LOG_DIR}/climatology.log" -- \
      $PY -u -m rice.scripts.phase_b_climatology_baseline \
        --pest "$PEST" --run "$RUN" \
        --val_year "$VAL_YEAR" \
        --test_year_min "$TEST_YEAR_MIN" --test_year_max "$TEST_YEAR_MAX" \
        --doy_start 60 \
        --dispatch_feature_csv "$DISPATCH_CSV" \
        --base_val_grid "$GRID_VAL" --base_test_grid "$GRID_TEST" \
        --sigma 5.0 --out_prefix "$CLIM_PREFIX" || continue
  else echo "  [skip] climatology stats exist"; fi

  # 6) per-pest canonical summary
  PER_OFFSET_CSV="${SUMMARY_ROOT}/${PEST}_per_offset.csv"
  SELECTION_CSV="${SUMMARY_ROOT}/${PEST}_selection.csv"
  run_step "canonical_summary" "$PEST" "${LOG_DIR}/canonical_summary.log" -- \
    $PY -u -m rice.scripts.phase_b_canonical_summary \
      --entry "${PEST}_lead_v3|val=${GRID_VAL}|test=${GRID_TEST}" \
      --entry "${PEST}_clim_mean_L|val=${CLIM_PREFIX}_mean_L_val_sample_grid.csv|test=${CLIM_PREFIX}_mean_L_test_sample_grid.csv" \
      --entry "${PEST}_clim_mean_mid|val=${CLIM_PREFIX}_mean_mid_val_sample_grid.csv|test=${CLIM_PREFIX}_mean_mid_test_sample_grid.csv" \
      --entry "${PEST}_clim_mean_R|val=${CLIM_PREFIX}_mean_R_val_sample_grid.csv|test=${CLIM_PREFIX}_mean_R_test_sample_grid.csv" \
      --out_per_offset "$PER_OFFSET_CSV" \
      --out_selection  "$SELECTION_CSV"
done

# ===== Final aggregate + winner =====
echo
echo "================================================================"
echo "Aggregating all-pest selection + winner table"
echo "================================================================"
ALL_SEL="${SUMMARY_ROOT}/all_pests_selection.csv"
WINNER_CSV="${SUMMARY_ROOT}/stage2_2024_winner_by_pest.csv"

$PY - <<PYEOF
import glob, os, re, pandas as pd, sys
sel_files = sorted(glob.glob("${SUMMARY_ROOT}/*_selection.csv"))
sel_files = [f for f in sel_files
             if not f.endswith("all_pests_selection.csv")
             and not f.endswith("stage1_gate_selection_split3_2024.csv")]
if not sel_files:
    sys.exit("[merge] no per-pest selection CSVs found")
rows = []
for f in sel_files:
    base = os.path.basename(f).replace("_selection.csv", "")
    df = pd.read_csv(f)
    # 'model' looks like '<pest>_lead_v3' or '<pest>_clim_mean_*'
    df.insert(0, "pest", base)
    rows.append(df)
out = pd.concat(rows, ignore_index=True)

# Strip the pest prefix from model -> 'model_kind'
def _kind(row):
    m = str(row["model"])
    pest = row["pest"]
    return m[len(pest)+1:] if m.startswith(pest + "_") else m
out["model_kind"] = out.apply(_kind, axis=1)

# all_pests_selection.csv: keep canonical columns
keep_cols = ["pest", "model", "model_kind",
             "val_best_offset", "val_IoU_overall_n_total_at_best",
             "test_IoU_overall_n_total_at_val_offset",
             "test_oracle_IoU_overall_n_total",
             "test_best_offset_LEAKY",
             "test_IoU_overall_n_total_at_test_best_LEAKY"]
keep_cols = [c for c in keep_cols if c in out.columns]
out_save = out[keep_cols].copy()
out_save.to_csv("${ALL_SEL}", index=False)
print(f"[merge] wrote ${ALL_SEL}  rows={len(out_save)}  pests={out_save['pest'].nunique()}")

# Winner per pest: max(test_IoU_overall_n_total_at_val_offset)
SCORE = "test_IoU_overall_n_total_at_val_offset"
winners = []
for pest, sub in out.groupby("pest"):
    sub2 = sub.dropna(subset=[SCORE])
    if sub2.empty:
        continue
    sub2 = sub2.sort_values(SCORE, ascending=False)
    win = sub2.iloc[0]
    lead = sub[sub["model_kind"].str.startswith("lead_v3", na=False)]
    lead_score = float(lead.iloc[0][SCORE]) if len(lead) and lead.iloc[0][SCORE]==lead.iloc[0][SCORE] else float("nan")
    clim = sub[sub["model_kind"].str.startswith("clim_", na=False)].dropna(subset=[SCORE])
    if len(clim):
        clim = clim.sort_values(SCORE, ascending=False)
        best_clim_kind = clim.iloc[0]["model_kind"]
        best_clim_score = float(clim.iloc[0][SCORE])
    else:
        best_clim_kind, best_clim_score = "none", float("nan")
    beats = (lead_score == lead_score and lead_score > best_clim_score)
    delta = (lead_score - best_clim_score) if (lead_score==lead_score and best_clim_score==best_clim_score) else float("nan")
    if delta != delta:
        interp = "missing scores"
    elif delta > 0.02:
        interp = f"lead_v3 clearly beats climatology (+{delta:.3f})"
    elif delta > 0:
        interp = f"lead_v3 marginally above climatology (+{delta:.3f})"
    elif delta > -0.02:
        interp = f"climatology essentially ties lead_v3 ({delta:+.3f})"
    else:
        interp = f"climatology beats lead_v3 ({delta:+.3f})"
    winners.append({
        "pest": pest,
        "winner_model": win["model_kind"],
        "winner_score": float(win[SCORE]),
        "winner_val_best_offset": win.get("val_best_offset"),
        "lead_v3_score": lead_score,
        "best_climatology_model": best_clim_kind,
        "best_climatology_score": best_clim_score,
        "lead_v3_minus_climatology": delta,
        "lead_v3_beats_climatology": bool(beats),
        "interpretation": interp,
    })
win_df = pd.DataFrame(winners).sort_values("pest")
win_df.to_csv("${WINNER_CSV}", index=False)
print(f"[merge] wrote ${WINNER_CSV}  rows={len(win_df)}")
print()
with pd.option_context("display.width", 220, "display.max_columns", 30,
                        "display.float_format", "{:.4f}".format):
    show = ["pest", "winner_model", "winner_score", "lead_v3_score",
            "best_climatology_model", "best_climatology_score",
            "lead_v3_beats_climatology", "interpretation"]
    print(win_df[show].to_string(index=False))
PYEOF

n_fail=$(grep -c "^[0-9].*FAIL" "$FAIL_LOG" 2>/dev/null || echo 0)
echo
echo "================================================================"
echo "Done: $(date -Iseconds)"
echo "  out_root     : $OUT_ROOT"
echo "  gate_csv     : $GATE_CSV"
echo "  all_pests    : $ALL_SEL"
echo "  winners      : $WINNER_CSV"
echo "  failures     : $n_fail step(s) -> $FAIL_LOG"
echo "================================================================"
