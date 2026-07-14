#!/usr/bin/env bash
#
# Overnight driver: run Stage 2 baseline_asym_mse lead_v3 for 8 pests × 2 splits.
#
# Splits / years:
#   split1: val=2021, test=2022   -> rice/outputs_stage2_batch_2022_baseline/
#   split2: val=2022, test=2023   -> rice/outputs_stage2_batch_2023_baseline/
# split3 (val=2023, test=2024) is NOT touched — its results live at
# rice/outputs_stage2_batch_2024_bestgate/ and are preserved.
#
# Pests (BPH2 excluded, ricestemborer 1/2 kept separate):
#   WBPH bacterial_blight brown_spot BPH rice_stem_borer_1 rice_stem_borer_2
#   blast sheath_blight
#
# Per-pest selection follows the existing rule: per-split val-only best-gate
# (D_history / dispatch_group_tau / A_baseline × run0/1/2), then Stage 2
# lead_v3 with mu_mode=lead_from_alert + asym_mse loss + sigma=5 (matching
# the existing split3 baseline). NO interval_nll / mixed / residual_clim.
#
# Existing per-pest Stage 2 uncond ckpts (split3-trained) are REUSED as
# warm-start — the lead_v3 head is trained from scratch for each split's
# data. This is the "Path B with shared uncond" trade-off in the design
# proposal (faster than per-split uncond; minor info leak through backbone).
#
# Run with:
#   bash scripts/run_stage2_all_pests_baseline_2022_2023.sh
# Optional:
#   LOW_MEM=1       -> 16/32 batch override (for OOM-prone GPUs)
#   STRICT_UNCOND=1 -> retrain per-split uncond (eliminates backbone info leak
#                     from reusing split3-trained uncond as warm-start; adds
#                     ~30-60 min per pest, so 16 pests * ~45 min = ~12 extra
#                     hours on top of lead_v3 training — make sure overnight
#                     budget covers it. Strict mode writes ckpts to
#                     rice/outputs_stage2_<pest>_uncond_<split>/ ).
#   PESTS="sheath_blight WBPH"  -> restrict pest list (debug)

set -uo pipefail
trap 'echo "[overnight] interrupted at $(date -Iseconds)"; exit 130' INT TERM

LOG_ROOT="logs/overnight_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_ROOT"
SUMMARY_LOG="${LOG_ROOT}/SUMMARY.log"
STRICT_UNCOND="${STRICT_UNCOND:-0}"
echo "[overnight] start $(date -Iseconds)  log_root=$LOG_ROOT  STRICT_UNCOND=$STRICT_UNCOND" | tee -a "$SUMMARY_LOG"

EXTRA_ARGS=()
if [[ "${LOW_MEM:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--low-mem)
fi
if [[ -n "${PESTS:-}" ]]; then
  EXTRA_ARGS+=(--pests "$PESTS")
fi

run_one_split() {
  local split=$1
  local logfile="${LOG_ROOT}/${split}.log"
  echo
  echo "================================================================" | tee -a "$SUMMARY_LOG"
  echo "[overnight] START $split  $(date -Iseconds)" | tee -a "$SUMMARY_LOG"
  echo "  log -> $logfile" | tee -a "$SUMMARY_LOG"
  echo "================================================================" | tee -a "$SUMMARY_LOG"
  SPLIT="$split" STRICT_UNCOND="$STRICT_UNCOND" \
    bash scripts/run_stage2_split_pest_best_gate_batch.sh \
    "${EXTRA_ARGS[@]}" 2>&1 | tee -a "$logfile"
  local rc=${PIPESTATUS[0]}
  echo "[overnight] END   $split  rc=$rc  $(date -Iseconds)" | tee -a "$SUMMARY_LOG"
  return $rc
}

# Run split1 then split2 sequentially.
run_one_split split1 || echo "[overnight] split1 returned non-zero (continuing)" | tee -a "$SUMMARY_LOG"
run_one_split split2 || echo "[overnight] split2 returned non-zero (continuing)" | tee -a "$SUMMARY_LOG"

echo
echo "================================================================" | tee -a "$SUMMARY_LOG"
echo "[overnight] BUILDING CROSS-SPLIT COMPARISON" | tee -a "$SUMMARY_LOG"
echo "================================================================" | tee -a "$SUMMARY_LOG"
.venv/bin/python -u -m rice.scripts.phase_b_stage2_baseline_cross_split_summary \
  --out_dir "${LOG_ROOT}/cross_split_summary" 2>&1 | tee -a "$SUMMARY_LOG"

echo
echo "================================================================" | tee -a "$SUMMARY_LOG"
echo "[overnight] DONE  $(date -Iseconds)" | tee -a "$SUMMARY_LOG"
echo "  per-split logs:        $LOG_ROOT/{split1,split2}.log" | tee -a "$SUMMARY_LOG"
echo "  cross-split comparison: ${LOG_ROOT}/cross_split_summary/" | tee -a "$SUMMARY_LOG"
echo "================================================================" | tee -a "$SUMMARY_LOG"
