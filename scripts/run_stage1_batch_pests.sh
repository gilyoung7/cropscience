#!/usr/bin/env bash
#
# Stage 1 batch runner for 6 non-sheath_blight rice pests.
#
# Mirrors the sheath_blight seed-stability pipeline for each (pest, split, seed):
#   1) train A_baseline (no history) ckpt
#   2) train D_history ckpt (site-history rolling, year-conditioned)
#   3) useful_pareto sweep for A and D
#   4) dispatch_group_tau_hybrid eval (R>=0.85/0.88/0.90 selections)
#
# Output layout:
#   <out-root>/<pest>/run<seed>/<split>_v<val>_t<test>/
#       A/{ckpt,useful_pareto}/...
#       D/{ckpt,useful_pareto}/...
#       group_tau/group_tau_hybrid_summary.json
#       logs/{train_A,train_D,pareto_A,pareto_D,group_tau}.log
#
# Defaults:
#   - 6 pests (sheath_blight excluded; already finalized)
#   - seeds: 0 1 2
#   - splits: split1 (val=2021/test=2022), split2 (val=2022/test=2023),
#             split3 (val=2023/test=2024)
#   - nowcast_window=28, lead14-45, outside_policy=ignore  (matches sheath_blight)
#
# Per-step granularity skip: if an artifact already exists and is non-empty,
# the step is skipped. Use --force to re-run.
#
# Failures DO NOT abort the batch. Each failed step is appended to <fail-log>.

set -uo pipefail
trap 'echo "[batch] interrupted at $(date -Iseconds)"; exit 130' INT TERM

# ===== Defaults =====
DEFAULT_PESTS="WBPH bacterial_blight brown_spot BPH rice_stem_borer_1 rice_stem_borer_2 blast"
DEFAULT_SEEDS="0 1 2"
DEFAULT_SPLITS="split1 split2 split3"

NOWCAST_WINDOW=28
LEAD_MIN=14
LEAD_MAX=45
OUTSIDE_POLICY=ignore
RECALL_TARGETS="0.85,0.88,0.90"
KS="1,2,3"
TAU_STEP_DISPATCH=0.025
KS_DISPATCH="3"
RUN=4

PESTS=""
SEEDS=""
SPLITS=""
FORCE=0
GPU_MEM_WAIT_MB=0
GPU_INDEX=0
OUT_ROOT="rice/outputs_stage1/batch_rolling"
FAIL_LOG=""

# Per-pest baseline ckpt (template for hyperparam copy in phase_t_lead_aware_train)
declare -A PEST_TEMPLATE=(
  ["WBPH"]="rice/outputs_stage1/WBPH_siteyear111/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split111_siteyear_ymin2002.pt"
  ["bacterial_blight"]="rice/outputs_stage1/bacterial_blight_siteyear58/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split58_siteyear_ymin2002.pt"
  ["brown_spot"]="rice/outputs_stage1/brown_spot_siteyear159/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split159_siteyear_ymin2002.pt"
  ["BPH"]="rice/outputs_stage1/BPH_siteyear42/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split42_siteyear_ymin2002.pt"
  ["BPH2"]="rice/outputs_stage1/BPH2_siteyear129/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split129_siteyear_ymin2002.pt"
  ["rice_stem_borer_1"]="rice/outputs_stage1/rice_stem_borer_1_siteyear91/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split91_siteyear_ymin2002.pt"
  ["rice_stem_borer_2"]="rice/outputs_stage1/rice_stem_borer_2_siteyear79/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split79_siteyear_ymin2002.pt"
  ["blast"]="rice/outputs_stage1/blast_siteyear59/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split59_siteyear_ymin2002.pt"
)

# Per-split (val_year, test_year, history_train_year_max)
declare -A SPLIT_VAL=(   ["split1"]=2021 ["split2"]=2022 ["split3"]=2023 )
declare -A SPLIT_TEST=(  ["split1"]=2022 ["split2"]=2023 ["split3"]=2024 )
declare -A SPLIT_HTYR=(  ["split1"]=2020 ["split2"]=2021 ["split3"]=2022 )

# ===== CLI =====
usage() {
  cat <<EOF
Usage: $0 [options]
  --pests "p1 p2 ..."     default: $DEFAULT_PESTS
                          (sheath_blight is intentionally excluded; pass it
                          explicitly if you want to re-run anyway)
  --seeds "0 1 2"         default: $DEFAULT_SEEDS
  --splits "split1 ..."   default: $DEFAULT_SPLITS
  --force                 re-run even if outputs already exist (per-step)
  --gpu-mem-wait MB       wait until free GPU mem >= MB before each step
                          (0 = no wait, default)
  --gpu-index N           CUDA device index for free-mem check (default 0)
  --run N                 --run for phase_t_* (default $RUN)
  --out-root PATH         default: $OUT_ROOT
  --fail-log PATH         default: <out-root>/batch_failures.log
  -h, --help
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --pests) PESTS="$2"; shift 2;;
    --seeds) SEEDS="$2"; shift 2;;
    --splits) SPLITS="$2"; shift 2;;
    --force) FORCE=1; shift;;
    --gpu-mem-wait) GPU_MEM_WAIT_MB="$2"; shift 2;;
    --gpu-index) GPU_INDEX="$2"; shift 2;;
    --run) RUN="$2"; shift 2;;
    --out-root) OUT_ROOT="$2"; shift 2;;
    --fail-log) FAIL_LOG="$2"; shift 2;;
    -h|--help) usage; exit 0;;
    *) echo "[abort] unknown arg: $1" >&2; usage; exit 2;;
  esac
done

PESTS="${PESTS:-$DEFAULT_PESTS}"
SEEDS="${SEEDS:-$DEFAULT_SEEDS}"
SPLITS="${SPLITS:-$DEFAULT_SPLITS}"
[[ -z "$FAIL_LOG" ]] && FAIL_LOG="${OUT_ROOT}/batch_failures.log"
SUMMARY_DIR="${OUT_ROOT}/_summary"

mkdir -p "$OUT_ROOT" "$SUMMARY_DIR" "$(dirname "$FAIL_LOG")"
touch "$FAIL_LOG"

PY=".venv/bin/python"
[[ -x "$PY" ]] || PY=python

# ===== Helpers =====
wait_for_gpu() {
  local need_mb=$1
  [[ "$need_mb" -le 0 ]] && return 0
  while :; do
    local free
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits -i "$GPU_INDEX" 2>/dev/null | head -1)
    if [[ -z "$free" || "$free" == *"Failed"* ]]; then
      echo "[gpu-wait] nvidia-smi unavailable; skipping wait"
      return 0
    fi
    if (( free >= need_mb )); then return 0; fi
    echo "[gpu-wait] free=${free}MB < ${need_mb}MB; sleeping 30s"
    sleep 30
  done
}

record_fail() {
  local pest=$1 seed=$2 split=$3 step=$4 rc=$5 log=$6
  echo "$(date -Iseconds) FAIL pest=$pest seed=$seed split=$split step=$step rc=$rc log=$log" \
    | tee -a "$FAIL_LOG" >&2
}

run_step() {
  # run_step <step-tag> <pest> <seed> <split> <logfile> -- <cmd...>
  local tag=$1 pest=$2 seed=$3 split=$4 logfile=$5
  shift 5
  [[ "$1" == "--" ]] && shift
  echo "  [run] $tag  ->  $logfile"
  "$@" >>"$logfile" 2>&1
  local rc=$?
  if [[ $rc -ne 0 ]]; then
    record_fail "$pest" "$seed" "$split" "$tag" "$rc" "$logfile"
    return $rc
  fi
  return 0
}

echo "================================================================"
echo "Stage 1 batch start: $(date -Iseconds)"
echo "  pests : $PESTS"
echo "  seeds : $SEEDS"
echo "  splits: $SPLITS"
echo "  out   : $OUT_ROOT"
echo "  fail  : $FAIL_LOG"
echo "  force : $FORCE   gpu-wait: ${GPU_MEM_WAIT_MB}MB"
echo "================================================================"

# ===== Main loop =====
for pest in $PESTS; do
  template="${PEST_TEMPLATE[$pest]:-}"
  if [[ -z "$template" || ! -s "$template" ]]; then
    record_fail "$pest" "-" "-" "template_ckpt_missing" "1" "${template:-<unset>}"
    continue
  fi
  for seed in $SEEDS; do
    for split in $SPLITS; do
      val_year="${SPLIT_VAL[$split]:-}"
      test_year="${SPLIT_TEST[$split]:-}"
      htymax="${SPLIT_HTYR[$split]:-}"
      if [[ -z "$val_year" || -z "$test_year" || -z "$htymax" ]]; then
        record_fail "$pest" "$seed" "$split" "split_config_missing" "1" "-"
        continue
      fi
      run_dir="${OUT_ROOT}/${pest}/run${seed}/${split}_v${val_year}_t${test_year}"
      a_dir="${run_dir}/A"
      d_dir="${run_dir}/D"
      gt_dir="${run_dir}/group_tau"
      logdir="${run_dir}/logs"
      mkdir -p "${a_dir}/ckpt" "${a_dir}/useful_pareto" \
               "${d_dir}/ckpt" "${d_dir}/useful_pareto" \
               "${gt_dir}" "${logdir}"
      a_ckpt="${a_dir}/ckpt/event_xgb_w${NOWCAST_WINDOW}_lead${LEAD_MIN}-${LEAD_MAX}_A.pt"
      d_ckpt="${d_dir}/ckpt/event_xgb_w${NOWCAST_WINDOW}_lead${LEAD_MIN}-${LEAD_MAX}_D.pt"
      a_sweep="${a_dir}/useful_pareto/useful_sweep_A.csv"
      d_sweep="${d_dir}/useful_pareto/useful_sweep_D.csv"
      gt_summary="${gt_dir}/group_tau_hybrid_summary.json"

      echo "----------------------------------------------------------------"
      echo "[$pest seed=$seed $split] val=$val_year test=$test_year htymax=$htymax"
      echo "  run_dir=$run_dir"
      echo "----------------------------------------------------------------"

      if [[ "$FORCE" == "1" ]]; then
        rm -f "$a_ckpt" "$d_ckpt" "$a_sweep" "$d_sweep" "$gt_summary"
      fi

      # --- TRAIN A (baseline, no history)
      if [[ ! -s "$a_ckpt" ]]; then
        wait_for_gpu "$GPU_MEM_WAIT_MB"
        run_step "train_A" "$pest" "$seed" "$split" "${logdir}/train_A.log" -- \
          "$PY" -u -m rice.scripts.phase_t_lead_aware_train \
            --pest "$pest" --run "$RUN" \
            --template_ckpt "$template" \
            --val_year "$val_year" \
            --test_year_min "$test_year" --test_year_max "$test_year" \
            --nowcast_window "$NOWCAST_WINDOW" \
            --lead_min "$LEAD_MIN" --lead_max "$LEAD_MAX" \
            --outside_policy "$OUTSIDE_POLICY" \
            --xgb_seed "$seed" \
            --out_ckpt "$a_ckpt" \
            || continue
      else
        echo "  [skip] A ckpt exists ($a_ckpt)"
      fi

      # --- TRAIN D (with site-history features, rolling policy)
      if [[ ! -s "$d_ckpt" ]]; then
        wait_for_gpu "$GPU_MEM_WAIT_MB"
        run_step "train_D" "$pest" "$seed" "$split" "${logdir}/train_D.log" -- \
          "$PY" -u -m rice.scripts.phase_t_lead_aware_train \
            --pest "$pest" --run "$RUN" \
            --template_ckpt "$template" \
            --val_year "$val_year" \
            --test_year_min "$test_year" --test_year_max "$test_year" \
            --nowcast_window "$NOWCAST_WINDOW" \
            --lead_min "$LEAD_MIN" --lead_max "$LEAD_MAX" \
            --outside_policy "$OUTSIDE_POLICY" \
            --add_site_history --site_history_policy rolling \
            --history_train_year_max "$htymax" \
            --xgb_seed "$seed" \
            --out_ckpt "$d_ckpt" \
            || continue
      else
        echo "  [skip] D ckpt exists ($d_ckpt)"
      fi

      # --- PARETO A
      if [[ ! -s "$a_sweep" ]]; then
        wait_for_gpu "$GPU_MEM_WAIT_MB"
        run_step "pareto_A" "$pest" "$seed" "$split" "${logdir}/pareto_A.log" -- \
          "$PY" -u -m rice.scripts.phase_t_useful_pareto \
            --pest "$pest" --run "$RUN" \
            --stage1_ckpt "$a_ckpt" --label "A" \
            --val_year "$val_year" \
            --test_year_min "$test_year" --test_year_max "$test_year" \
            --ks "$KS" --recall_targets "$RECALL_TARGETS" \
            --out_dir "${a_dir}/useful_pareto" \
            || continue
      else
        echo "  [skip] A sweep exists"
      fi

      # --- PARETO D
      if [[ ! -s "$d_sweep" ]]; then
        wait_for_gpu "$GPU_MEM_WAIT_MB"
        run_step "pareto_D" "$pest" "$seed" "$split" "${logdir}/pareto_D.log" -- \
          "$PY" -u -m rice.scripts.phase_t_useful_pareto \
            --pest "$pest" --run "$RUN" \
            --stage1_ckpt "$d_ckpt" --label "D" \
            --val_year "$val_year" \
            --test_year_min "$test_year" --test_year_max "$test_year" \
            --ks "$KS" --recall_targets "$RECALL_TARGETS" \
            --out_dir "${d_dir}/useful_pareto" \
            || continue
      else
        echo "  [skip] D sweep exists"
      fi

      # --- DISPATCH group_tau_hybrid
      if [[ ! -s "$gt_summary" ]]; then
        wait_for_gpu "$GPU_MEM_WAIT_MB"
        run_step "group_tau" "$pest" "$seed" "$split" "${logdir}/group_tau.log" -- \
          "$PY" -u -m rice.scripts.phase_t_group_tau_hybrid \
            --pest "$pest" --run "$RUN" \
            --baseline_ckpt "$a_ckpt" --d_ckpt "$d_ckpt" \
            --val_year "$val_year" \
            --test_year_min "$test_year" --test_year_max "$test_year" \
            --tau_step "$TAU_STEP_DISPATCH" --ks "$KS_DISPATCH" \
            --recall_targets "$RECALL_TARGETS" \
            --out_dir "$gt_dir" \
            || continue
      else
        echo "  [skip] group_tau summary exists"
      fi
    done
  done
done

n_fail=$(grep -c "^[0-9].*FAIL" "$FAIL_LOG" 2>/dev/null || echo 0)
echo
echo "================================================================"
echo "Batch done: $(date -Iseconds)"
echo "  outputs   : $OUT_ROOT"
echo "  failures  : $n_fail step(s) in $FAIL_LOG"
echo "  aggregate :"
echo "    $PY -u -m rice.scripts.merge_pest_batch_farmin \\"
echo "      --base $OUT_ROOT \\"
echo "      --out_csv ${SUMMARY_DIR}/pest_batch_farmin_all.csv \\"
echo "      --out_summary_csv ${SUMMARY_DIR}/pest_batch_farmin_R088.csv \\"
echo "      --target_for_summary 0.88"
echo "================================================================"
