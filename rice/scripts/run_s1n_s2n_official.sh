#!/usr/bin/env bash
# Official S1-neighbor -> S2-neighbor pipeline (per pest):
#   1. Stage-1 N  (baseline+neighbor)        [reuse if ckpt exists]
#   2. Stage-1 DN (baseline+history+neighbor) [reuse if ckpt exists]
#   3. neighbor-aware group_tau summary  (phase_t_group_tau_hybrid: baseline=N, d=DN)
#   4. Stage-1-neighbor dispatch CSV     (build_dispatch_feature_table, gate=prod method, a=N,d=DN)
#   5. dry-run Stage-2 (+neighbor, dispatch=S1N CSV): assert d_in == prod+6, report rows/cohort
#   6. Stage-2 pilot  (scratch)
#   7. Stage-2 final  (warm-start from this run's pilot)
#
# Stage-2 hyperparameters are EXTRACTED from the production log (production-identical);
# only out_root, dispatch CSV, warm-start, and the neighbor flag are changed.
#
# HALT-ON-ERROR: any failure logs "[FAIL] <pest> <stage>" and exits (does NOT continue
# to the next pest). Production outputs are never written. No deletes / no git.
#
# Usage (run from cropscience/ root, inside tmux):
#   cd /home/gpu4080/research/cropscience
#   bash rice/scripts/run_s1n_s2n_official.sh                 # default 6 pests
#   bash rice/scripts/run_s1n_s2n_official.sh WBPH sheath_blight
set -u

PY="${PY:-.venv/bin/python}"
DECAY_KM="${DECAY_KM:-20.0}"
PROD_BASE="rice/outputs_stage2_batch_2024_bestgate"
S1_BASE="rice/outputs_stage1/batch_rolling"
OUT_BASE="rice/outputs_stage2_s1n_s2n"
FAILLOG="$OUT_BASE/_FAILURES.log"
export PYTHONPATH="${PYTHONPATH:-.}"

# per-pest production config (selected_run seed, gate method, dispatch k)
declare -A SEED=(   [WBPH]=2 [sheath_blight]=0 [BPH]=2 [bacterial_blight]=0 [brown_spot]=2 [blast]=0 )
declare -A METHOD=( [WBPH]=dispatch_group_tau [sheath_blight]=dispatch_group_tau \
                    [BPH]=D_history [bacterial_blight]=D_history [brown_spot]=D_history [blast]=D_history )
declare -A KVAL=(   [WBPH]=3 [sheath_blight]=3 [BPH]=3 [bacterial_blight]=1 [brown_spot]=3 [blast]=3 )

PESTS=("$@"); [ "${#PESTS[@]}" -eq 0 ] && PESTS=(WBPH sheath_blight BPH bacterial_blight brown_spot blast)

mkdir -p "$OUT_BASE"
fail () {  # $1=pest $2=stage $3=msg
  echo "[FAIL] pest=$1 stage=$2 : $3" | tee -a "$FAILLOG" >&2
  echo "HALTED at pest=$1 stage=$2. See logs under $OUT_BASE/$1/logs/." >&2
  exit 1
}

extract_args () {  # $1=prodlog $2=lead_v3_pilot|lead_v3_final
  grep -E "rice\.scripts\.run_train" "$1" \
    | grep -E "out_root rice/[^ ]*/$2( |\$)" | head -1 \
    | sed -E 's|^.*-m rice\.scripts\.run_train ||'
}

for pest in "${PESTS[@]}"; do
  echo "==================== $pest ===================="
  seed="${SEED[$pest]:-}"; method="${METHOD[$pest]:-}"; k="${KVAL[$pest]:-3}"
  [ -z "$seed" ] && fail "$pest" config "no SEED/METHOD entry (unknown pest)"
  S1DIR="$S1_BASE/$pest/run$seed/split3_v2023_t2024"
  A_CKPT="$S1DIR/A/ckpt/event_xgb_w28_lead14-45_A.pt"
  N_CKPT="$S1DIR/N/ckpt/event_xgb_w28_lead14-45_N.pt"
  DN_CKPT="$S1DIR/DN/ckpt/event_xgb_w28_lead14-45_DN.pt"
  OUT="$OUT_BASE/$pest"; mkdir -p "$OUT/logs"
  DISP="$OUT/gate_dispatch_stage1_neighbor_features_per_sy.csv"
  PRODLOG="$PROD_BASE/$pest/logs/stage2_lead_v3_train.log"
  S1LOG="$OUT/logs/stage1.log"; DLOG="$OUT/logs/dispatch.log"
  PLOG="$OUT/logs/stage2_pilot.log"; FLOG="$OUT/logs/stage2_final.log"

  [ -f "$A_CKPT" ]  || fail "$pest" stage1 "production A template missing: $A_CKPT"
  [ -f "$PRODLOG" ] || fail "$pest" stage2 "production log missing: $PRODLOG"

  S1COMMON="--pest $pest --run 4 --template_ckpt $A_CKPT --xgb_seed $seed \
    --val_year 2023 --test_year_min 2024 --test_year_max 2024 \
    --nowcast_window 28 --lead_min 14 --lead_max 45 --outside_policy ignore \
    --neighbor_decay_km $DECAY_KM"

  # 1) Stage-1 N (reuse if present)
  if [ -f "$N_CKPT" ]; then echo "[reuse] N: $N_CKPT" | tee -a "$S1LOG"
  else
    echo ">>> Stage-1 N train" | tee -a "$S1LOG"
    # shellcheck disable=SC2086
    $PY -u -m rice.scripts.phase_t_lead_aware_train $S1COMMON \
      --add_neighbor_history --out_ckpt "$N_CKPT" 2>&1 | tee -a "$S1LOG"
    [ -f "$N_CKPT" ] || fail "$pest" stage1_N "N ckpt not produced"
  fi
  # 2) Stage-1 DN (reuse if present)
  if [ -f "$DN_CKPT" ]; then echo "[reuse] DN: $DN_CKPT" | tee -a "$S1LOG"
  else
    echo ">>> Stage-1 DN train" | tee -a "$S1LOG"
    # shellcheck disable=SC2086
    $PY -u -m rice.scripts.phase_t_lead_aware_train $S1COMMON \
      --add_site_history --site_history_policy rolling --add_neighbor_history \
      --out_ckpt "$DN_CKPT" 2>&1 | tee -a "$S1LOG"
    [ -f "$DN_CKPT" ] || fail "$pest" stage1_DN "DN ckpt not produced"
  fi

  # 3) neighbor-aware group_tau summary (baseline=N, d=DN)
  GT_DIR="$S1DIR/group_tau_neighbor"; GT_JSON="$GT_DIR/group_tau_hybrid_summary.json"
  echo ">>> group_tau (neighbor)" | tee -a "$S1LOG"
  # shellcheck disable=SC2086
  $PY -u -m rice.scripts.phase_t_group_tau_hybrid --pest $pest --run 4 \
    --baseline_ckpt "$N_CKPT" --d_ckpt "$DN_CKPT" \
    --val_year 2023 --test_year_min 2024 --test_year_max 2024 \
    --ks $k --recall_targets 0.85,0.88,0.90,0.92 --out_dir "$GT_DIR" 2>&1 | tee -a "$S1LOG"
  [ -f "$GT_JSON" ] || fail "$pest" group_tau "summary json not produced: $GT_JSON"

  # 4) Stage-1-neighbor dispatch CSV (gate = production method, a=N, d=DN)
  echo ">>> dispatch table (method=$method)" | tee "$DLOG"
  # shellcheck disable=SC2086
  $PY -u -m rice.scripts.build_dispatch_feature_table --pest $pest --run 4 \
    --val_year 2023 --test_year_min 2024 --test_year_max 2024 \
    --dispatch_summary_json "$GT_JSON" --dispatch_target_label "R>=0.88" \
    --gate_method $method --dispatch_a_ckpt "$N_CKPT" --dispatch_d_ckpt "$DN_CKPT" \
    --include_splits train,val,test --out_csv "$DISP" 2>&1 | tee -a "$DLOG"
  [ -f "$DISP" ] || fail "$pest" dispatch "dispatch CSV not produced: $DISP"
  DISP_ROWS=$(($(wc -l < "$DISP") - 1))
  echo "[check] dispatch rows=$DISP_ROWS  csv=$DISP" | tee -a "$DLOG"

  # 5) transform production Stage-2 cmds: out_root, dispatch=$DISP, warm-start, +neighbor
  PA="$(extract_args "$PRODLOG" lead_v3_pilot)"; FA="$(extract_args "$PRODLOG" lead_v3_final)"
  [ -n "$PA" ] && [ -n "$FA" ] || fail "$pest" stage2 "could not extract prod pilot/final cmd"
  PILOT="$(printf '%s' "$PA" \
    | sed -E "s| --out_root [^ ]+| --out_root $OUT/lead_v3_pilot|" \
    | sed -E "s| --stage2_dispatch_feature_csv [^ ]+| --stage2_dispatch_feature_csv $DISP|" \
    | sed -E "s| --stage2_warm_start_ckpt [^ ]+ --stage2_warm_start_seed [0-9]+||") \
    --stage2_add_neighbor_history --stage2_neighbor_decay_km $DECAY_KM"
  FINAL="$(printf '%s' "$FA" \
    | sed -E "s| --out_root [^ ]+| --out_root $OUT/lead_v3_final|" \
    | sed -E "s| --stage2_dispatch_feature_csv [^ ]+| --stage2_dispatch_feature_csv $DISP|" \
    | sed -E "s| --stage2_warm_start_ckpt [^ ]+| --stage2_warm_start_ckpt $OUT/lead_v3_pilot/ckpt/checkpoint_run4.pt|") \
    --stage2_add_neighbor_history --stage2_neighbor_decay_km $DECAY_KM"
  DRY="$(printf '%s' "$PILOT" | sed -E "s| --out_root [^ ]+| --out_root $OUT/_dryrun|") \
    --max_epochs_override 1 --stage2_sanity_only 1 --stage2_sanity_batches 1"

  # dry-run + checks (d_in == prod+6)
  echo ">>> Stage-2 DRY-RUN" | tee "$PLOG"
  # shellcheck disable=SC2086
  $PY -u -m rice.scripts.run_train $DRY 2>&1 | tee -a "$PLOG"
  PROD_DIN=$(grep -oE "computed_from_dataset=[0-9]+" "$PRODLOG" | head -1 | grep -oE "[0-9]+")
  DRY_DIN=$(grep -oE "computed_from_dataset=[0-9]+" "$PLOG" | tail -1 | grep -oE "[0-9]+")
  TEST_COHORT=$(grep -oE "samples train=[0-9]+ val=[0-9]+ test=[0-9]+" "$PLOG" | tail -1)
  echo "[check] prod_d_in=$PROD_DIN  s1n_d_in=$DRY_DIN  (expect +6)  | dispatch_rows=$DISP_ROWS | $TEST_COHORT" | tee -a "$PLOG"
  [ -n "$PROD_DIN" ] && [ -n "$DRY_DIN" ] || fail "$pest" dryrun "could not read d_in"
  [ "$DRY_DIN" -eq $((PROD_DIN + 6)) ] || fail "$pest" dryrun "d_in $DRY_DIN != prod $PROD_DIN +6"

  # 6) pilot (scratch)
  echo ">>> Stage-2 PILOT (scratch)" | tee -a "$PLOG"
  # shellcheck disable=SC2086
  $PY -u -m rice.scripts.run_train $PILOT 2>&1 | tee -a "$PLOG"
  [ -f "$OUT/lead_v3_pilot/ckpt/checkpoint_run4.pt" ] || fail "$pest" pilot "pilot ckpt not produced"

  # 7) final (warm-start from this run's pilot)
  echo ">>> Stage-2 FINAL" | tee "$FLOG"
  # shellcheck disable=SC2086
  $PY -u -m rice.scripts.run_train $FINAL 2>&1 | tee -a "$FLOG"
  [ -f "$OUT/lead_v3_final/ckpt/checkpoint_run4.pt" ] || fail "$pest" final "final ckpt not produced"
  echo "[done] $pest -> $OUT/lead_v3_final"
done

echo "ALL PESTS DONE. Run matched 3-way eval:"
echo "  $PY rice/scripts/eval_s1n_s2n_matched.py --pests ${PESTS[*]}"
