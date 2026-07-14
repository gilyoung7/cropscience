#!/usr/bin/env bash
#
# Full Stage 2 lead_v3 + climatology pipeline for non-sheath_blight pests.
#
# Per-pest pipeline (each step skip-if-exists, graceful failure to next pest):
#   1. Stage 1 yearsplit train A_baseline (lead14-45 ignore, no history)
#   2. Stage 1 yearsplit train D_history (lead14-45 history_rolling)
#   3. Stage 1 dispatch group_tau_hybrid -> summary JSON
#   4. Per-(site, year) dispatch confidence feature CSV (train+val+test)
#   5. Stage 2 uncond  (hazard head, only if not already present)
#   6. Stage 2 pilot + final via phase_s5_train (lead_from_alert, lead_min=7,
#      lead_max=75, cohort_dispatch_only, dispatch_feature_mode=causal)
#   7. val + test sample_grid (phase_r dispatch mode, offsets=7,14,21,30,45,60)
#   8. climatology baselines (mean_L / mean_mid / mean_R) val + test grids
#
# After all pests finish: aggregate canonical comparison merge per pest.
#
# Year split: val=2022, test=2023-2024 (matches Phase B sheath_blight setup).
#
# Default pest list excludes sheath_blight (already done end-to-end).
# Use --pests "...sheath_blight..." to re-run.

set -uo pipefail
trap 'echo "[batch] interrupted at $(date -Iseconds)"; exit 130' INT TERM

# ===== Defaults =====
DEFAULT_PESTS="WBPH bacterial_blight brown_spot BPH rice_stem_borer_1 rice_stem_borer_2"

VAL_YEAR=2022
TEST_YEAR_MIN=2023
TEST_YEAR_MAX=2024
RUN=4
NOWCAST_WINDOW=28
LEAD_MIN_LABEL=14
LEAD_MAX_LABEL=45
OUTSIDE_POLICY=ignore
DISPATCH_TARGET="R>=0.88"
RECALL_TARGETS="0.85,0.88,0.90"

# Stage 2 lead_v3 recipe
LEAD_MIN=7
LEAD_MAX=75
FEATURE_MODE=causal
OFFSETS_GRID="7,14,21,30,45,60"

PESTS=""
SEED=0
GPU_INDEX=0
FORCE=0
SKIP_STAGE1=0   # if external Stage 1 yearsplit was prepared
SKIP_STAGE2=0   # to just refresh climatology/grids
OUT_ROOT_STAGE2="rice/outputs_stage2_batch"
SUMMARY_ROOT="rice/outputs_stage2_batch/_summary"
FAIL_LOG=""

# Per-pest seed (drives template ckpt name for Stage 1 hparams copy)
declare -A PEST_SITESEED=(
  ["WBPH"]=111
  ["bacterial_blight"]=58
  ["brown_spot"]=159
  ["BPH"]=42
  ["BPH2"]=129
  ["rice_stem_borer_1"]=91
  ["rice_stem_borer_2"]=79
  ["sheath_blight"]=54
)

# Templates the per-pest Stage 1 trainer needs (site_year baseline ckpts; already exist)
TEMPLATE_OF() {
  local pest=$1 seed=${PEST_SITESEED[$1]:-}
  echo "rice/outputs_stage1/${pest}_siteyear${seed}/ckpt/event_run4_xgb_nowcast_w${NOWCAST_WINDOW}_s1_tpos_split${seed}_siteyear_ymin2002.pt"
}

# ===== CLI =====
usage() {
  cat <<EOF
Usage: $0 [options]
  --pests "p1 p2 ..."  default: $DEFAULT_PESTS
  --seed N             XGB seed for Stage 1 trains (default $SEED)
  --force              re-run every step even if outputs exist
  --skip_stage1        assume Stage 1 yearsplit + dispatch summary already exist
  --skip_stage2        assume Stage 2 ckpt + sample_grids already exist
  --out_root PATH      default: $OUT_ROOT_STAGE2
  --gpu_index N        nvidia-smi index for diagnostic prints (default 0)
  -h, --help
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pests) PESTS="$2"; shift 2;;
    --seed) SEED="$2"; shift 2;;
    --force) FORCE=1; shift;;
    --skip_stage1) SKIP_STAGE1=1; shift;;
    --skip_stage2) SKIP_STAGE2=1; shift;;
    --out_root) OUT_ROOT_STAGE2="$2"; SUMMARY_ROOT="${OUT_ROOT_STAGE2}/_summary"; shift 2;;
    --gpu_index) GPU_INDEX="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[abort] unknown arg: $1" >&2; usage; exit 2;;
  esac
done
PESTS="${PESTS:-$DEFAULT_PESTS}"
[[ -z "$FAIL_LOG" ]] && FAIL_LOG="${SUMMARY_ROOT}/batch_failures.log"
mkdir -p "$OUT_ROOT_STAGE2" "$SUMMARY_ROOT" "$(dirname "$FAIL_LOG")"
touch "$FAIL_LOG"

PY=".venv/bin/python"
[[ -x "$PY" ]] || PY=python

record_fail() {
  local pest=$1 step=$2 rc=$3 log=$4
  echo "$(date -Iseconds) FAIL pest=$pest step=$step rc=$rc log=$log" \
    | tee -a "$FAIL_LOG" >&2
}

run_step() {
  # run_step <step-tag> <pest> <logfile> -- <cmd...>
  local tag=$1 pest=$2 logfile=$3; shift 3
  [[ "$1" == "--" ]] && shift
  echo "  [run] $tag  -> $logfile"
  "$@" >>"$logfile" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    record_fail "$pest" "$tag" "$rc" "$logfile"
    return $rc
  fi
  return 0
}

echo "================================================================"
echo "Stage 2 lead_v3 + climatology batch start: $(date -Iseconds)"
echo "  pests: $PESTS"
echo "  year split: val=$VAL_YEAR test=$TEST_YEAR_MIN..$TEST_YEAR_MAX"
echo "  Stage2: lead_from_alert lead_min=$LEAD_MIN lead_max=$LEAD_MAX cohort_dispatch_only"
echo "  offsets grid: $OFFSETS_GRID"
echo "  out root: $OUT_ROOT_STAGE2"
echo "================================================================"

for PEST in $PESTS; do
  SITESEED="${PEST_SITESEED[$PEST]:-}"
  if [[ -z "$SITESEED" ]]; then
    record_fail "$PEST" "config_missing" 1 "-"; continue
  fi
  TEMPLATE=$(TEMPLATE_OF "$PEST")
  if [[ ! -s "$TEMPLATE" ]]; then
    record_fail "$PEST" "template_ckpt_missing" 1 "$TEMPLATE"; continue
  fi

  # ===== Layout =====
  STAGE1_ROOT="rice/outputs_stage1/${PEST}_yearsplit${TEST_YEAR_MIN}-${TEST_YEAR_MAX}_lead${LEAD_MIN_LABEL}-${LEAD_MAX_LABEL}_${OUTSIDE_POLICY}"
  STAGE1_D_ROOT="rice/outputs_stage1/${PEST}_yearsplit${TEST_YEAR_MIN}-${TEST_YEAR_MAX}_lead${LEAD_MIN_LABEL}-${LEAD_MAX_LABEL}_history_rolling"
  A_CKPT="${STAGE1_ROOT}/ckpt/event_run4_xgb_w${NOWCAST_WINDOW}_lead${LEAD_MIN_LABEL}-${LEAD_MAX_LABEL}_ignore.pt"
  D_CKPT="${STAGE1_D_ROOT}/ckpt/event_run4_xgb_w${NOWCAST_WINDOW}_lead${LEAD_MIN_LABEL}-${LEAD_MAX_LABEL}_history_rolling.pt"
  DISPATCH_SUMMARY="${STAGE1_D_ROOT}/group_tau_hybrid/group_tau_hybrid_summary.json"
  DISPATCH_CSV="${OUT_ROOT_STAGE2}/${PEST}/dispatch_R088_features_per_sy.csv"

  PEST_OUT="${OUT_ROOT_STAGE2}/${PEST}"
  UNCOND_ROOT="rice/outputs_stage2_${PEST}_uncond"
  UNCOND_CKPT="${UNCOND_ROOT}/ckpt/checkpoint_run${RUN}.pt"
  PILOT_OUT="${PEST_OUT}/lead_v3_pilot"
  FINAL_OUT="${PEST_OUT}/lead_v3_final"
  PILOT_CKPT="${PILOT_OUT}/ckpt/checkpoint_run${RUN}.pt"
  FINAL_CKPT="${FINAL_OUT}/ckpt/checkpoint_run${RUN}.pt"

  LOG_DIR="${PEST_OUT}/logs"
  mkdir -p "${STAGE1_ROOT}/ckpt" "${STAGE1_D_ROOT}/ckpt" \
           "${STAGE1_D_ROOT}/group_tau_hybrid" \
           "${PEST_OUT}" "${UNCOND_ROOT}/ckpt" "${LOG_DIR}"

  echo "================================================================"
  echo "[pest=$PEST] template=$TEMPLATE  siteseed=$SITESEED"
  echo "  out_pest=$PEST_OUT"
  echo "================================================================"
  [[ "$FORCE" == "1" ]] && rm -f "$A_CKPT" "$D_CKPT" "$DISPATCH_SUMMARY" \
                                   "$DISPATCH_CSV" "$PILOT_CKPT" "$FINAL_CKPT"

  # ===== STAGE 1 (skip if --skip_stage1) =====
  if [[ "$SKIP_STAGE1" != "1" ]]; then
    # 1a. A_baseline yearsplit
    if [[ ! -s "$A_CKPT" ]]; then
      run_step "stage1_A_train" "$PEST" "${LOG_DIR}/stage1_A_train.log" -- \
        "$PY" -u -m rice.scripts.phase_t_lead_aware_train \
          --pest "$PEST" --run "$RUN" \
          --template_ckpt "$TEMPLATE" \
          --val_year "$VAL_YEAR" \
          --test_year_min "$TEST_YEAR_MIN" --test_year_max "$TEST_YEAR_MAX" \
          --nowcast_window "$NOWCAST_WINDOW" \
          --lead_min "$LEAD_MIN_LABEL" --lead_max "$LEAD_MAX_LABEL" \
          --outside_policy "$OUTSIDE_POLICY" \
          --xgb_seed "$SEED" \
          --out_ckpt "$A_CKPT" || continue
    else echo "  [skip] A ckpt exists"; fi

    # 1b. D_history yearsplit (history_train_year_max = val_year - 1 = 2021)
    HTYMAX=$((VAL_YEAR - 1))
    if [[ ! -s "$D_CKPT" ]]; then
      run_step "stage1_D_train" "$PEST" "${LOG_DIR}/stage1_D_train.log" -- \
        "$PY" -u -m rice.scripts.phase_t_lead_aware_train \
          --pest "$PEST" --run "$RUN" \
          --template_ckpt "$TEMPLATE" \
          --val_year "$VAL_YEAR" \
          --test_year_min "$TEST_YEAR_MIN" --test_year_max "$TEST_YEAR_MAX" \
          --nowcast_window "$NOWCAST_WINDOW" \
          --lead_min "$LEAD_MIN_LABEL" --lead_max "$LEAD_MAX_LABEL" \
          --outside_policy "$OUTSIDE_POLICY" \
          --add_site_history --site_history_policy rolling \
          --history_train_year_max "$HTYMAX" \
          --xgb_seed "$SEED" \
          --out_ckpt "$D_CKPT" || continue
    else echo "  [skip] D ckpt exists"; fi

    # 1c. dispatch group_tau_hybrid
    if [[ ! -s "$DISPATCH_SUMMARY" ]]; then
      run_step "stage1_dispatch" "$PEST" "${LOG_DIR}/stage1_dispatch.log" -- \
        "$PY" -u -m rice.scripts.phase_t_group_tau_hybrid \
          --pest "$PEST" --run "$RUN" \
          --baseline_ckpt "$A_CKPT" --d_ckpt "$D_CKPT" \
          --val_year "$VAL_YEAR" \
          --test_year_min "$TEST_YEAR_MIN" --test_year_max "$TEST_YEAR_MAX" \
          --tau_step 0.025 --ks 3 \
          --recall_targets "$RECALL_TARGETS" \
          --out_dir "${STAGE1_D_ROOT}/group_tau_hybrid" || continue
    else echo "  [skip] dispatch summary exists"; fi
  else
    [[ ! -s "$A_CKPT" || ! -s "$D_CKPT" || ! -s "$DISPATCH_SUMMARY" ]] && {
      record_fail "$PEST" "stage1_missing_after_skip" 1 "-"; continue
    }
  fi

  # ===== STAGE 2 PREP =====
  # 2a. dispatch feature table (train+val+test, alert sy with all 14 features)
  if [[ ! -s "$DISPATCH_CSV" ]]; then
    mkdir -p "$(dirname "$DISPATCH_CSV")"
    run_step "dispatch_feature_table" "$PEST" "${LOG_DIR}/dispatch_feature_table.log" -- \
      "$PY" -u -m rice.scripts.build_dispatch_feature_table \
        --pest "$PEST" --run "$RUN" \
        --val_year "$VAL_YEAR" \
        --test_year_min "$TEST_YEAR_MIN" --test_year_max "$TEST_YEAR_MAX" \
        --dispatch_summary_json "$DISPATCH_SUMMARY" \
        --dispatch_target_label "$DISPATCH_TARGET" \
        --dispatch_a_ckpt "$A_CKPT" \
        --dispatch_d_ckpt "$D_CKPT" \
        --include_splits train,val,test \
        --out_csv "$DISPATCH_CSV" || continue
  else echo "  [skip] dispatch feature CSV exists"; fi

  # ===== STAGE 2 TRAIN (skip if --skip_stage2) =====
  if [[ "$SKIP_STAGE2" != "1" ]]; then
    # 2b. uncond Stage 2 (hazard head). If absent, train from scratch using
    # the same recipe as sheath_blight uncond (pmf_mode=hazard, sigma=5,
    # causal_tstar, nowcast 28). Skip if already there.
    if [[ ! -s "$UNCOND_CKPT" ]]; then
      run_step "stage2_uncond_train" "$PEST" "${LOG_DIR}/stage2_uncond_train.log" -- \
        "$PY" -u -m rice.scripts.run_train \
          --pest "$PEST" --run "$RUN" \
          --seeds "$SEED" \
          --split_seed 42 \
          --split_mode year \
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
          --stage2_pmf_mode hazard \
          --stage2_pmf_sigma 5.0 \
          --stage2_pmf_target_offset 0.0 \
          --stage2_pmf_right_weight 0.3 \
          --stage2_best_metric val_iou80 \
          --amp 1 --amp_dtype bf16 \
          --d_model_override 48 \
          --out_root "$UNCOND_ROOT" \
          --out "$UNCOND_CKPT" || continue
    else echo "  [skip] uncond ckpt exists"; fi

    # 2c. pilot + final via phase_s5_train (lead_from_alert)
    if [[ ! -s "$FINAL_CKPT" ]]; then
      run_step "stage2_lead_v3_train" "$PEST" "${LOG_DIR}/stage2_lead_v3_train.log" -- \
        "$PY" -u -m rice.scripts.phase_s5_train \
          --pest "$PEST" --run "$RUN" --seed "$SEED" \
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
  else
    [[ ! -s "$UNCOND_CKPT" || ! -s "$FINAL_CKPT" ]] && {
      record_fail "$PEST" "stage2_missing_after_skip" 1 "-"; continue
    }
  fi

  # ===== STAGE 2 EVAL (val + test sample_grid) =====
  GRID_VAL="${PEST_OUT}/lead_v3_val_sample_grid.csv"
  GRID_TEST="${PEST_OUT}/lead_v3_test_sample_grid.csv"
  ORACLE_VAL="${PEST_OUT}/lead_v3_val_oracle.csv"
  ORACLE_TEST="${PEST_OUT}/lead_v3_test_oracle.csv"

  for SPLIT_TAG in test val; do
    case "$SPLIT_TAG" in
      test) GRID_OUT="$GRID_TEST"; ORC_OUT="$ORACLE_TEST";;
      val)  GRID_OUT="$GRID_VAL";  ORC_OUT="$ORACLE_VAL";;
    esac
    if [[ ! -s "$GRID_OUT" ]]; then
      run_step "stage2_grid_${SPLIT_TAG}" "$PEST" "${LOG_DIR}/stage2_grid_${SPLIT_TAG}.log" -- \
        env STAGE2_CKPT="$FINAL_CKPT" \
            STAGE2_LABEL="${PEST}_lead_v3" \
            OFFSETS="$OFFSETS_GRID" \
            EVAL_SPLIT="$SPLIT_TAG" \
            OUT_GRID_CSV="$GRID_OUT" \
            OUT_ORACLE_CSV="$ORC_OUT" \
            PEST="$PEST" RUN="$RUN" \
            VAL_YEAR="$VAL_YEAR" TEST_YEAR_MIN="$TEST_YEAR_MIN" TEST_YEAR_MAX="$TEST_YEAR_MAX" \
            DISPATCH_SUMMARY="$DISPATCH_SUMMARY" \
            DISPATCH_TARGET="$DISPATCH_TARGET" \
            A_CKPT="$A_CKPT" D_CKPT="$D_CKPT" \
            LEGACY_STAGE1_CKPT="$A_CKPT" \
            bash scripts/run_stage2_dispatch_sample_grid.sh || continue
    else echo "  [skip] ${SPLIT_TAG} grid exists"; fi
  done

  # ===== CLIMATOLOGY BASELINE =====
  CLIM_PREFIX="${PEST_OUT}/climatology"
  CLIM_TRAIN_STATS="${CLIM_PREFIX}_train_stats.csv"
  if [[ ! -s "$CLIM_TRAIN_STATS" ]]; then
    run_step "climatology" "$PEST" "${LOG_DIR}/climatology.log" -- \
      "$PY" -u -m rice.scripts.phase_b_climatology_baseline \
        --pest "$PEST" --run "$RUN" \
        --val_year "$VAL_YEAR" \
        --test_year_min "$TEST_YEAR_MIN" --test_year_max "$TEST_YEAR_MAX" \
        --doy_start 60 \
        --dispatch_feature_csv "$DISPATCH_CSV" \
        --base_val_grid "$GRID_VAL" \
        --base_test_grid "$GRID_TEST" \
        --sigma 5.0 \
        --out_prefix "$CLIM_PREFIX" || continue
  else echo "  [skip] climatology stats exist"; fi

  # ===== PER-PEST CANONICAL COMPARISON =====
  PER_OFFSET_CSV="${SUMMARY_ROOT}/${PEST}_per_offset.csv"
  SELECTION_CSV="${SUMMARY_ROOT}/${PEST}_selection.csv"
  run_step "canonical_summary" "$PEST" "${LOG_DIR}/canonical_summary.log" -- \
    "$PY" -u -m rice.scripts.phase_b_canonical_summary \
      --entry "${PEST}_lead_v3|val=${GRID_VAL}|test=${GRID_TEST}" \
      --entry "${PEST}_clim_mean_L|val=${CLIM_PREFIX}_mean_L_val_sample_grid.csv|test=${CLIM_PREFIX}_mean_L_test_sample_grid.csv" \
      --entry "${PEST}_clim_mean_mid|val=${CLIM_PREFIX}_mean_mid_val_sample_grid.csv|test=${CLIM_PREFIX}_mean_mid_test_sample_grid.csv" \
      --entry "${PEST}_clim_mean_R|val=${CLIM_PREFIX}_mean_R_val_sample_grid.csv|test=${CLIM_PREFIX}_mean_R_test_sample_grid.csv" \
      --out_per_offset "$PER_OFFSET_CSV" \
      --out_selection  "$SELECTION_CSV"
done

# ===== AGGREGATE ALL PESTS =====
echo
echo "================================================================"
echo "Aggregating per-pest selection CSVs..."
echo "================================================================"
ALL_SEL="${SUMMARY_ROOT}/all_pests_selection.csv"
$PY - <<PYEOF
import glob, os, pandas as pd, sys
sel_files = sorted(glob.glob("${SUMMARY_ROOT}/*_selection.csv"))
sel_files = [f for f in sel_files if not f.endswith("all_pests_selection.csv")]
if not sel_files:
    sys.exit("[merge] no per-pest selection CSVs found")
rows = []
for f in sel_files:
    pest = os.path.basename(f).replace("_selection.csv", "")
    df = pd.read_csv(f)
    df.insert(0, "pest", pest)
    rows.append(df)
out = pd.concat(rows, ignore_index=True)
out.to_csv("${ALL_SEL}", index=False)
print(f"[merge] wrote ${ALL_SEL}  rows={len(out)}  pests={out['pest'].nunique()}")
show_cols = ["pest", "model", "val_best_offset",
             "val_IoU_overall_n_total_at_best",
             "test_IoU_overall_n_total_at_val_offset",
             "test_oracle_IoU_overall_n_total"]
show_cols = [c for c in show_cols if c in out.columns]
print()
with pd.option_context("display.width", 200,
                        "display.max_columns", 30,
                        "display.float_format", "{:.4f}".format):
    print(out[show_cols].to_string(index=False))
PYEOF

n_fail=$(grep -c "^[0-9].*FAIL" "$FAIL_LOG" 2>/dev/null || echo 0)
echo
echo "================================================================"
echo "Batch done: $(date -Iseconds)"
echo "  out_root  : $OUT_ROOT_STAGE2"
echo "  summary   : $SUMMARY_ROOT"
echo "  all-pest  : $ALL_SEL"
echo "  failures  : $n_fail step(s) -> $FAIL_LOG"
echo "================================================================"
