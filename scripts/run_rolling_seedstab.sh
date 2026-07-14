#!/usr/bin/env bash
#
# Rolling-split XGB seed stability: trains A_baseline + D_history at xgb_seed
# 0/1/2 on each of split1/2/3, then runs dispatch_group_tau evaluation.
#
# Outputs land under rice/outputs_stage1/seed_stability/ so the existing
# single-seed dirs (sheath_blight_rolling_split{1,2,3}_v..._t..._...) are
# untouched.
#
# Aggregation:
#   python -m rice.scripts.rolling_seed_stability_farmin
#
# Total wall time on a recent GPU is a few minutes (XGB fits in ~2s each).

set -euo pipefail

PEST=sheath_blight
RUN=4
TEMPLATE_CKPT="rice/outputs_stage1/sheath_blight_siteyear54/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split54_siteyear_ymin2002.pt"

if [[ ! -f "$TEMPLATE_CKPT" ]]; then
  echo "missing template ckpt: $TEMPLATE_CKPT" >&2
  exit 1
fi

PY=".venv/bin/python"
if [[ ! -x "$PY" ]]; then
  PY="python"
fi

# split_tag  val_year  test_year  history_train_year_max
SPLITS=(
  "split1 2021 2022 2020"
  "split2 2022 2023 2021"
  "split3 2023 2024 2022"
)
SEEDS=(0 1 2)

ROOT_OUT="rice/outputs_stage1/seed_stability"
mkdir -p "$ROOT_OUT"

for spec in "${SPLITS[@]}"; do
  read -r split val_year test_year htymax <<<"$spec"

  for seed in "${SEEDS[@]}"; do
    SPLIT_DIR="${ROOT_OUT}/${PEST}_${split}_v${val_year}_t${test_year}_seed${seed}"
    mkdir -p "${SPLIT_DIR}/A/ckpt" "${SPLIT_DIR}/D/ckpt" \
             "${SPLIT_DIR}/A/useful_pareto" "${SPLIT_DIR}/D/useful_pareto" \
             "${SPLIT_DIR}/group_tau" "${SPLIT_DIR}/logs"

    A_CKPT="${SPLIT_DIR}/A/ckpt/event_xgb_w28_lead14-45_A.pt"
    D_CKPT="${SPLIT_DIR}/D/ckpt/event_xgb_w28_lead14-45_D.pt"
    A_SWEEP="${SPLIT_DIR}/A/useful_pareto/useful_sweep_A.csv"
    D_SWEEP="${SPLIT_DIR}/D/useful_pareto/useful_sweep_D.csv"
    GT_DIR="${SPLIT_DIR}/group_tau"
    LOG_DIR="${SPLIT_DIR}/logs"

    echo "================================================================"
    echo "[${split} seed=${seed}] val=${val_year} test=${test_year} htymax=${htymax}"
    echo "  out=${SPLIT_DIR}"
    echo "================================================================"

    # ---- TRAIN A (baseline, no history) ----
    if [[ ! -s "$A_CKPT" ]]; then
      $PY -m rice.scripts.phase_t_lead_aware_train \
        --pest "$PEST" --run "$RUN" \
        --template_ckpt "$TEMPLATE_CKPT" \
        --val_year "$val_year" \
        --test_year_min "$test_year" --test_year_max "$test_year" \
        --lead_min 14 --lead_max 45 --outside_policy ignore \
        --xgb_seed "$seed" \
        --out_ckpt "$A_CKPT" 2>&1 | tee "${LOG_DIR}/train_A.log"
    else
      echo "  [skip] A ckpt exists"
    fi

    # ---- TRAIN D (with site-history features, rolling policy) ----
    if [[ ! -s "$D_CKPT" ]]; then
      $PY -m rice.scripts.phase_t_lead_aware_train \
        --pest "$PEST" --run "$RUN" \
        --template_ckpt "$TEMPLATE_CKPT" \
        --val_year "$val_year" \
        --test_year_min "$test_year" --test_year_max "$test_year" \
        --lead_min 14 --lead_max 45 --outside_policy ignore \
        --add_site_history --site_history_policy rolling \
        --history_train_year_max "$htymax" \
        --xgb_seed "$seed" \
        --out_ckpt "$D_CKPT" 2>&1 | tee "${LOG_DIR}/train_D.log"
    else
      echo "  [skip] D ckpt exists"
    fi

    # ---- USEFUL-PARETO sweeps (val + test, with USEFUL/lead metrics) for A and D ----
    if [[ ! -s "$A_SWEEP" ]]; then
      $PY -m rice.scripts.phase_t_useful_pareto \
        --pest "$PEST" --run "$RUN" \
        --stage1_ckpt "$A_CKPT" \
        --label "A" \
        --val_year "$val_year" \
        --test_year_min "$test_year" --test_year_max "$test_year" \
        --ks 1,2,3 \
        --recall_targets 0.85,0.88,0.90 \
        --out_dir "${SPLIT_DIR}/A/useful_pareto" 2>&1 | tee "${LOG_DIR}/pareto_A.log"
    else
      echo "  [skip] A sweep exists"
    fi
    if [[ ! -s "$D_SWEEP" ]]; then
      $PY -m rice.scripts.phase_t_useful_pareto \
        --pest "$PEST" --run "$RUN" \
        --stage1_ckpt "$D_CKPT" \
        --label "D" \
        --val_year "$val_year" \
        --test_year_min "$test_year" --test_year_max "$test_year" \
        --ks 1,2,3 \
        --recall_targets 0.85,0.88,0.90 \
        --out_dir "${SPLIT_DIR}/D/useful_pareto" 2>&1 | tee "${LOG_DIR}/pareto_D.log"
    else
      echo "  [skip] D sweep exists"
    fi

    # ---- DISPATCH group_tau_hybrid (val-selected, with test eval) ----
    if [[ ! -s "${GT_DIR}/group_tau_hybrid_summary.json" ]]; then
      $PY -m rice.scripts.phase_t_group_tau_hybrid \
        --pest "$PEST" --run "$RUN" \
        --baseline_ckpt "$A_CKPT" --d_ckpt "$D_CKPT" \
        --val_year "$val_year" \
        --test_year_min "$test_year" --test_year_max "$test_year" \
        --tau_step 0.025 --ks 3 \
        --recall_targets 0.85,0.88,0.90 \
        --out_dir "$GT_DIR" 2>&1 | tee "${LOG_DIR}/group_tau.log"
    else
      echo "  [skip] group_tau summary exists"
    fi
  done
done

echo "================================================================"
echo "ALL DONE: $(date -Iseconds)"
echo "Aggregate with:  python -m rice.scripts.rolling_seed_stability_farmin"
echo "================================================================"
