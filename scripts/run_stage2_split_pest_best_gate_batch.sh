#!/usr/bin/env bash
#
# Stage 2 split-parameterizable batch with PER-PEST best Stage 1 gate.
# Generalization of run_stage2_split3_2024_pest_best_gate_batch.sh.
#
# Selectable via env vars (defaults match the existing split3/2024 baseline):
#   SPLIT=split{1,2,3}                 (default split3)
#   VAL_YEAR=2023                      (auto-derived from SPLIT if unset)
#   TEST_YEAR_MIN=2024                 (auto-derived from SPLIT if unset)
#   TEST_YEAR_MAX=2024                 (auto-derived from SPLIT if unset)
#   OUT_ROOT=rice/outputs_stage2_batch_2024_bestgate   (override per split)
#
# Stage 2 UNCOND ckpts are REUSED from the existing per-pest paths
# (originally trained with split3 settings). The lead_v3 head is trained
# from scratch using the current split's data — so head weights are honestly
# split-specific. The backbone (uncond) DID see future-relative data during
# its original training; this is a known minor information leak documented
# in the experiment notes. For maximum rigor, retrain uncond per split
# (see UNCOND_PATH override below).
#
# All other behaviour matches the existing split3 wrapper.

set -uo pipefail
trap 'echo "[batch] interrupted at $(date -Iseconds)"; exit 130' INT TERM

DEFAULT_PESTS="WBPH bacterial_blight brown_spot BPH rice_stem_borer_1 rice_stem_borer_2 blast sheath_blight"

# ---- split / years --------------------------------------------------------
SPLIT=${SPLIT:-split3}
case "$SPLIT" in
  split1) DEFAULT_VAL=2021; DEFAULT_TMIN=2022; DEFAULT_TMAX=2022;;
  split2) DEFAULT_VAL=2022; DEFAULT_TMIN=2023; DEFAULT_TMAX=2023;;
  split3) DEFAULT_VAL=2023; DEFAULT_TMIN=2024; DEFAULT_TMAX=2024;;
  *) echo "[abort] unknown SPLIT=$SPLIT (use split1|split2|split3)" >&2; exit 2;;
esac
VAL_YEAR=${VAL_YEAR:-$DEFAULT_VAL}
TEST_YEAR_MIN=${TEST_YEAR_MIN:-$DEFAULT_TMIN}
TEST_YEAR_MAX=${TEST_YEAR_MAX:-$DEFAULT_TMAX}

RUN=4
DISPATCH_TARGET="R>=0.88"
LEAD_MIN=7
LEAD_MAX=75
FEATURE_MODE=causal
OFFSETS_GRID="7,14,21,30,45,60"

PESTS=""
FORCE=0
LOW_MEM=0
BATCH_TRAIN_OVERRIDE=""
BATCH_EVAL_OVERRIDE=""
GAUSSIAN_LOSS_MODE="${GAUSSIAN_LOSS_MODE:-asym_mse}"
GAUSSIAN_INTERVAL_CC="${GAUSSIAN_INTERVAL_CC:-0}"
GAUSSIAN_INTERVAL_LAMBDA="${GAUSSIAN_INTERVAL_LAMBDA:-}"
SKIP_PILOT_WARM_FROM="${SKIP_PILOT_WARM_FROM:-}"
RIGHT_WEIGHT="${RIGHT_WEIGHT:-}"
MAX_EPOCHS="${MAX_EPOCHS:-}"
MU_MODE="${MU_MODE:-}"
DELTA_MAX="${DELTA_MAX:-60}"
RESET_HEAD_MU_FINAL="${RESET_HEAD_MU_FINAL:-0}"
TARGET_MODE="${TARGET_MODE:-}"
ASYM_WEIGHT_EARLY="${ASYM_WEIGHT_EARLY:-}"
RESET_ONLY="${RESET_ONLY:-0}"
# Strict per-split uncond ckpts. Default 0 = reuse legacy (split3-trained)
# per-pest uncond ckpts as warm-start (faster; minor backbone info leak from
# future-relative data). Set 1 to redirect UNCOND_CKPT to a SPLIT-tagged path
# (rice/outputs_stage2_<pest>_uncond_<split>/ckpt/checkpoint_runN.pt). The
# wrapper trains the uncond from scratch with the current split's val/test
# year args when the strict path is missing — eliminating the leak at the
# cost of an extra ~30-60 min per pest.
STRICT_UNCOND="${STRICT_UNCOND:-0}"

# Default out_root per split — kept apart so existing 2024 results stay intact
case "$SPLIT" in
  split1) DEFAULT_OUT="rice/outputs_stage2_batch_2022_baseline";;
  split2) DEFAULT_OUT="rice/outputs_stage2_batch_2023_baseline";;
  split3) DEFAULT_OUT="rice/outputs_stage2_batch_2024_bestgate";;
esac
OUT_ROOT="${OUT_ROOT:-$DEFAULT_OUT}"
SUMMARY_ROOT="${OUT_ROOT}/_summary"
GATE_CSV="${SUMMARY_ROOT}/stage1_gate_selection_${SPLIT}_${TEST_YEAR_MIN}.csv"
FAIL_LOG=""

# uncond candidates (per pest). Reuse split3-trained ckpts by default;
# wrapper trains a fresh uncond only if the path is missing.
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
Usage: SPLIT=split{1,2,3} $0 [--pests "p1 p2 ..."] [--force] [--low-mem]
  --pests "p1 p2 ..."  default: $DEFAULT_PESTS
  --force              re-run every step even if outputs exist
  --low-mem            shortcut for --batch_train 16 --batch_eval 32 (OOM safe)
  --batch_train N      explicit override
  --batch_eval N       explicit override
  --out_root PATH      default per split: ${DEFAULT_OUT}
Env defaults: SPLIT=$SPLIT  VAL=$VAL_YEAR  TEST=$TEST_YEAR_MIN..$TEST_YEAR_MAX
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pests) PESTS="$2"; shift 2;;
    --force) FORCE=1; shift;;
    --low-mem) LOW_MEM=1; shift;;
    --batch_train) BATCH_TRAIN_OVERRIDE="$2"; shift 2;;
    --batch_eval) BATCH_EVAL_OVERRIDE="$2"; shift 2;;
    --out_root) OUT_ROOT="$2"; SUMMARY_ROOT="${OUT_ROOT}/_summary"; GATE_CSV="${SUMMARY_ROOT}/stage1_gate_selection_${SPLIT}_${TEST_YEAR_MIN}.csv"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[abort] unknown arg: $1" >&2; usage; exit 2;;
  esac
done
if [[ "$LOW_MEM" == "1" ]]; then
  BATCH_TRAIN_OVERRIDE="${BATCH_TRAIN_OVERRIDE:-16}"
  BATCH_EVAL_OVERRIDE="${BATCH_EVAL_OVERRIDE:-32}"
fi
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
echo "Stage 2 PEST-BEST-GATE batch  SPLIT=$SPLIT  (val=$VAL_YEAR test=$TEST_YEAR_MIN..$TEST_YEAR_MAX)"
echo "  pests: $PESTS"
echo "  out_root: $OUT_ROOT"
echo "  gate_csv: $GATE_CSV"
echo "================================================================"

# Step 0: gate selection (val-only) for the chosen split
if [[ ! -s "$GATE_CSV" || "$FORCE" == "1" ]]; then
  $PY -u -m rice.scripts.select_stage1_gate \
    --exclude_pests BPH2 \
    --split "$SPLIT" \
    --out_csv "$GATE_CSV" \
    2>&1 | tee "${SUMMARY_ROOT}/stage1_gate_selection.log"
fi

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

for PEST in $PESTS; do
  echo
  echo "================================================================"
  echo "[pest=$PEST  split=$SPLIT]"
  echo "================================================================"

  SEL_METHOD=$(read_gate_field "$PEST" "selected_method") || { record_fail "$PEST" "gate_lookup" 1 "-"; continue; }
  SEL_RUN=$(read_gate_field "$PEST" "selected_run")
  A_CKPT=$(read_gate_field "$PEST" "a_ckpt")
  D_CKPT=$(read_gate_field "$PEST" "d_ckpt")
  DISPATCH_SUMMARY=$(read_gate_field "$PEST" "dispatch_summary")
  echo "  gate: method=$SEL_METHOD  run=$SEL_RUN"
  echo "  A_ckpt   = $A_CKPT"
  echo "  D_ckpt   = $D_CKPT"
  echo "  summary  = $DISPATCH_SUMMARY"

  for p in "$A_CKPT" "$D_CKPT" "$DISPATCH_SUMMARY"; do
    if [[ ! -s "$p" ]]; then
      record_fail "$PEST" "stage1_artifact_missing" 1 "$p"; continue 2
    fi
  done

  PEST_OUT="${OUT_ROOT}/${PEST}"
  LOG_DIR="${PEST_OUT}/logs"
  mkdir -p "${PEST_OUT}" "${LOG_DIR}"

  DISPATCH_CSV="${PEST_OUT}/gate_${SEL_METHOD}_R088_features_per_sy.csv"
  if [[ "$STRICT_UNCOND" == "1" ]]; then
    # Per-split uncond — wrapper will train if missing using the current split's
    # val/test years. Path includes SPLIT tag to avoid stomping legacy ckpts.
    UNCOND_CKPT="rice/outputs_stage2_${PEST}_uncond_${SPLIT}/ckpt/checkpoint_run${RUN}.pt"
    echo "  [strict_uncond] target = $UNCOND_CKPT"
  else
    UNCOND_CKPT="${UNCOND_PATH[$PEST]:-rice/outputs_stage2_${PEST}_uncond/ckpt/checkpoint_run${RUN}.pt}"
  fi
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

  # 1) dispatch feature CSV using selected gate_method
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
        --gate_method "$SEL_METHOD" \
        --out_csv "$DISPATCH_CSV" || continue
  else echo "  [skip] dispatch CSV exists ($SEL_METHOD)"; fi

  # 2) uncond Stage 2 — reuse if exists; otherwise train with current split's
  # val/test years.
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
        --stage2_nowcast_require_tstar_before_L 1 \
        --stage2_causal_tstar --stage2_tstar_layers 1 \
        --stage2_early_tstar_weight_min 0.2 \
        --stage2_pmf_mode hazard --stage2_pmf_sigma 5.0 \
        --stage2_pmf_target_offset 0.0 --stage2_pmf_right_weight 0.3 \
        --stage2_best_metric val_iou80 \
        --amp 1 --amp_dtype bf16 --d_model_override 48 \
        ${BATCH_TRAIN_OVERRIDE:+--batch_train_override $BATCH_TRAIN_OVERRIDE} \
        ${BATCH_EVAL_OVERRIDE:+--batch_eval_override $BATCH_EVAL_OVERRIDE} \
        --out_root "$UNCOND_ROOT" --out "$UNCOND_CKPT" || continue
  else echo "  [skip] uncond ckpt exists  (reused from $UNCOND_CKPT)"; fi

  # 3) pilot + final lead_v3
  EFFECTIVE_PILOT_OUT="$PILOT_OUT"
  PHASE_S5_EXTRA=()
  if [[ -n "$SKIP_PILOT_WARM_FROM" ]]; then
    EFFECTIVE_PILOT_OUT="$SKIP_PILOT_WARM_FROM"
    PHASE_S5_EXTRA+=(--skip_pilot)
    echo "  [event-only mode] skipping pilot; warm-start from: $EFFECTIVE_PILOT_OUT"
  fi
  if [[ -n "$RIGHT_WEIGHT" ]]; then
    PHASE_S5_EXTRA+=(--right_weight "$RIGHT_WEIGHT")
    echo "  [event-only mode] right_weight override = $RIGHT_WEIGHT"
  fi
  if [[ -n "$MAX_EPOCHS" ]]; then
    PHASE_S5_EXTRA+=(--max_epochs_override "$MAX_EPOCHS")
    echo "  [event-only mode] max_epochs cap = $MAX_EPOCHS"
  fi
  if [[ -n "$GAUSSIAN_INTERVAL_LAMBDA" ]]; then
    PHASE_S5_EXTRA+=(--gaussian_interval_lambda "$GAUSSIAN_INTERVAL_LAMBDA")
    echo "  [mixed loss] gaussian_interval_lambda = $GAUSSIAN_INTERVAL_LAMBDA"
  fi
  if [[ "$RESET_HEAD_MU_FINAL" == "1" ]]; then
    PHASE_S5_EXTRA+=(--reset_head_mu_final)
    echo "  [reset_head] dropping head_mu from warm-start ckpt (final stage)"
  fi
  if [[ -n "$TARGET_MODE" ]]; then
    PHASE_S5_EXTRA+=(--target_mode "$TARGET_MODE")
    echo "  [loss override] target_mode = $TARGET_MODE"
  fi
  if [[ -n "$ASYM_WEIGHT_EARLY" ]]; then
    PHASE_S5_EXTRA+=(--asym_weight_early "$ASYM_WEIGHT_EARLY")
    echo "  [loss override] asym_weight_early = $ASYM_WEIGHT_EARLY"
  fi
  if [[ "$MU_MODE" == "residual_clim" ]]; then
    CLIM_CSV=""
    if [[ -n "$SKIP_PILOT_WARM_FROM" ]]; then
      CLIM_CSV="$(dirname "$SKIP_PILOT_WARM_FROM")/climatology_train_stats.csv"
    fi
    if [[ -z "$CLIM_CSV" || ! -s "$CLIM_CSV" ]]; then
      CLIM_CSV="${PEST_OUT}/climatology_train_stats.csv"
    fi
    if [[ ! -s "$CLIM_CSV" ]]; then
      record_fail "$PEST" "residual_clim_missing_clim_csv" 1 "$CLIM_CSV"
      continue
    fi
    CLIM_MID=$($PY -c "import pandas as pd; print(pd.read_csv('$CLIM_CSV').iloc[0]['mean_mid'])")
    echo "  [residual_clim] clim_mid=${CLIM_MID} (from $CLIM_CSV)  delta_max=$DELTA_MAX"
    PHASE_S5_EXTRA+=(--mu_mode residual_clim
                     --clim_mid "$CLIM_MID"
                     --delta_max "$DELTA_MAX")
  fi
  if [[ "$RESET_ONLY" == "1" ]]; then
    if [[ -z "$SKIP_PILOT_WARM_FROM" || "$MU_MODE" != "residual_clim" ]]; then
      record_fail "$PEST" "reset_only_requires_warm_from_and_residual_clim" 1 "-"
      continue
    fi
    if [[ ! -s "$FINAL_CKPT" || "$FORCE" == "1" ]]; then
      WARM_CKPT="${SKIP_PILOT_WARM_FROM}/ckpt/checkpoint_run${RUN}.pt"
      [[ -s "$WARM_CKPT" ]] || { record_fail "$PEST" "reset_only_warm_ckpt_missing" 1 "$WARM_CKPT"; continue; }
      mkdir -p "$(dirname "$FINAL_CKPT")"
      echo "  [RESET_ONLY] zeroing head_mu in $WARM_CKPT -> $FINAL_CKPT"
      $PY - <<PYEOF || { record_fail "$PEST" "reset_only_save" 1 "$FINAL_CKPT"; continue; }
import torch
ck = torch.load("$WARM_CKPT", map_location="cpu", weights_only=False)
ts = ck["trained_states"][0]
sd = ts["state_dict"]
hm_keys = [k for k in sd if k.startswith("head_mu.")]
print(f"[RESET_ONLY] zeroing {len(hm_keys)} head_mu tensors: {hm_keys}")
for k in hm_keys:
    sd[k] = torch.zeros_like(sd[k])
ck["stage2_pmf_mu_mode"] = "residual_clim"
ck["stage2_pmf_clim_mid"] = float($CLIM_MID)
ck["stage2_pmf_delta_max"] = float($DELTA_MAX)
ck["stage2_reset_head_mu"] = True
torch.save(ck, "$FINAL_CKPT")
print(f"[RESET_ONLY] saved zeroed ckpt to $FINAL_CKPT")
PYEOF
    else echo "  [skip] final ckpt exists (reset_only)"; fi
  elif [[ ! -s "$FINAL_CKPT" ]]; then
    run_step "stage2_lead_v3_train" "$PEST" "${LOG_DIR}/stage2_lead_v3_train.log" -- \
      $PY -u -m rice.scripts.phase_s5_train \
        --pest "$PEST" --run "$RUN" --seed 0 \
        --val_year "$VAL_YEAR" \
        --test_year_min "$TEST_YEAR_MIN" --test_year_max "$TEST_YEAR_MAX" \
        --uncond_ckpt "$UNCOND_CKPT" \
        --out_pilot "$EFFECTIVE_PILOT_OUT" --out_final "$FINAL_OUT" \
        --dispatch_feature_csv "$DISPATCH_CSV" \
        --dispatch_feature_mode "$FEATURE_MODE" \
        --cohort_dispatch_only \
        --mu_mode lead_from_alert \
        --lead_min "$LEAD_MIN" --lead_max "$LEAD_MAX" \
        --require_tstar_before_L 1 \
        --gaussian_loss_mode "$GAUSSIAN_LOSS_MODE" \
        --gaussian_interval_continuity_correction "$GAUSSIAN_INTERVAL_CC" \
        "${PHASE_S5_EXTRA[@]}" \
        ${BATCH_TRAIN_OVERRIDE:+--batch_train_override $BATCH_TRAIN_OVERRIDE} \
        ${BATCH_EVAL_OVERRIDE:+--batch_eval_override $BATCH_EVAL_OVERRIDE} \
        || continue
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
            DISPATCH_GATE_METHOD="$SEL_METHOD" \
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

# ===== Final aggregate + winner with gate metadata =====
echo
echo "================================================================"
echo "Aggregating ${SPLIT} all-pest selection + winner (with gate metadata)"
echo "================================================================"
ALL_SEL="${SUMMARY_ROOT}/all_pests_selection.csv"
WINNER_CSV="${SUMMARY_ROOT}/stage2_${TEST_YEAR_MIN}_winner_by_pest.csv"

$PY - <<PYEOF
import glob, os, re, sys
import pandas as pd

SUMMARY_ROOT = "${SUMMARY_ROOT}"
GATE_CSV = "${GATE_CSV}"
ALL_SEL = "${ALL_SEL}"
WINNER_CSV = "${WINNER_CSV}"
OUT_ROOT = "${OUT_ROOT}"

gates = pd.read_csv(GATE_CSV)
gate_by_pest = {r["pest"]: r for _, r in gates.iterrows()}

sel_files = sorted(glob.glob(f"{SUMMARY_ROOT}/*_selection.csv"))
sel_files = [f for f in sel_files
             if not f.endswith("all_pests_selection.csv")
             and "stage1_gate_selection" not in os.path.basename(f)]
if not sel_files:
    sys.exit("[merge] no per-pest selection CSVs")

rows = []
for f in sel_files:
    base = os.path.basename(f).replace("_selection.csv", "")
    df = pd.read_csv(f)
    df.insert(0, "pest", base)
    rows.append(df)
out = pd.concat(rows, ignore_index=True)

def _kind(row):
    m = str(row["model"]); pest = row["pest"]
    return m[len(pest)+1:] if m.startswith(pest + "_") else m
out["model_kind"] = out.apply(_kind, axis=1)

keep_cols = ["pest", "model", "model_kind",
             "val_best_offset", "val_IoU_overall_n_total_at_best",
             "test_IoU_overall_n_total_at_val_offset",
             "test_oracle_IoU_overall_n_total",
             "test_best_offset_LEAKY",
             "test_IoU_overall_n_total_at_test_best_LEAKY"]
keep_cols = [c for c in keep_cols if c in out.columns]
out[keep_cols].to_csv(ALL_SEL, index=False)
print(f"[merge] wrote {ALL_SEL}  rows={len(out)}")

SCORE = "test_IoU_overall_n_total_at_val_offset"
winners = []
for pest, sub in out.groupby("pest"):
    sub2 = sub.dropna(subset=[SCORE])
    if sub2.empty: continue
    g = gate_by_pest.get(pest)
    sel_method = str(g["selected_method"]) if g is not None else "?"
    sel_run = int(g["selected_run"]) if g is not None else -1
    lead = sub[sub["model_kind"].str.startswith("lead_v3", na=False)].dropna(subset=[SCORE])
    lead_score = float(lead.iloc[0][SCORE]) if len(lead) else float("nan")
    clim = sub[sub["model_kind"].str.startswith("clim_", na=False)].dropna(subset=[SCORE])
    if len(clim):
        clim = clim.sort_values(SCORE, ascending=False)
        best_clim_kind = clim.iloc[0]["model_kind"]
        best_clim_score = float(clim.iloc[0][SCORE])
    else:
        best_clim_kind, best_clim_score = "none", float("nan")
    beats = (lead_score == lead_score and lead_score > best_clim_score)
    delta = lead_score - best_clim_score if (lead_score == lead_score and best_clim_score == best_clim_score) else float("nan")
    if delta != delta: interp = "missing scores"
    elif delta > 0.02: interp = f"lead_v3 beats climatology (+{delta:.3f})"
    elif delta > 0:    interp = f"lead_v3 marginally above (+{delta:.3f})"
    elif delta > -0.02: interp = f"essentially tied ({delta:+.3f})"
    else:               interp = f"climatology beats lead_v3 ({delta:+.3f})"
    winners.append({
        "pest": pest,
        "split": "${SPLIT}",
        "test_year": ${TEST_YEAR_MIN},
        "selected_method": sel_method,
        "selected_run": sel_run,
        "val_recall": float(g["val_recall"]) if g is not None else float("nan"),
        "val_FAR":    float(g["val_FAR"]) if g is not None else float("nan"),
        "test_recall": float(g["test_recall"]) if g is not None else float("nan"),
        "test_FAR":    float(g["test_FAR"]) if g is not None else float("nan"),
        "stage2_lead_v3_score":  lead_score,
        "best_climatology_model": best_clim_kind,
        "best_climatology_score": best_clim_score,
        "lead_v3_beats_climatology": bool(beats),
        "interpretation": interp,
    })
win = pd.DataFrame(winners).sort_values("pest")
win.to_csv(WINNER_CSV, index=False)
print(f"[merge] wrote {WINNER_CSV}  rows={len(win)}")
print()
with pd.option_context("display.width", 240, "display.max_columns", 30,
                        "display.float_format", "{:.4f}".format):
    print(win.to_string(index=False))
PYEOF

n_fail=$(grep -c "^[0-9].*FAIL" "$FAIL_LOG" 2>/dev/null || echo 0)
echo
echo "================================================================"
echo "Done: $(date -Iseconds)   SPLIT=$SPLIT"
echo "  out_root  : $OUT_ROOT"
echo "  gate_csv  : $GATE_CSV"
echo "  winners   : $WINNER_CSV"
echo "  failures  : $n_fail step(s) -> $FAIL_LOG"
echo "================================================================"
