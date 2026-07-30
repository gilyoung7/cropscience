#!/usr/bin/env bash
# Shared helpers for the 2-stage E5d Gaussian CURRICULUM arm. Source this; do not execute it.
#
# WHY THIS EXISTS
# The scratch arm (rice/experiments/allpests_e5d) extracts ONLY the production lead_v3_final
# command and then deletes --stage2_warm_start_ckpt, so every cell trains from random init.
# This arm restores a warm-start chain and changes NOTHING else, so scratch-vs-curriculum is
# attributable to the final stage's initialisation alone.
#
# WHY 2 STAGES AND NOT 3
# Production runs uncond(hazard) -> pilot -> final, but uncond is IMPOSSIBLE on the E5d
# shared-multi-offset architecture: the shared path leaves the time-axis hidden state unset
# (model.py:499 `h = None`) because it only gathers per-offset z, while the hazard head needs
# `h_fused = h + z_flat` (model.py:772) -> TypeError. Only the Gaussian mu path tolerates
# h=None (model.py:766 falls back to z). Verified by a real smoke run, whose artifacts are
# quarantined under rice/outputs_e5d_ablation/_failed_uncond_smoke/. model.py is NOT patched.
#
# STAGE ARGUMENTS -- both derive from the SAME production commands, so data, split, seed,
# optimizer, lr, batch sizes, architecture env and patience are identical across stages and
# identical to the scratch arm:
#
#   pilot   production 'lead_v3_pilot' line, out_root + warm-start stripped. gaussian /
#           lead_from_alert / asym 15 / early 0. Trains from scratch.
#   final   production 'lead_v3_final' line, out_root + warm-start stripped, then E5D_SED
#           (asym_weight 25 / asym_weight_early 0) -- byte-identical to what the scratch arm
#           passes (asserted by verify_final_parity.sh), plus exactly one added flag:
#           --stage2_warm_start_ckpt <this cell's pilot>.
#           production final's asym_weight_early=5 is deliberately NOT inherited.

set -euo pipefail

CS=/home/gpu4080/research/cropscience
VENDOR="$CS/rice/experiments/allpests_e5d/vendor"          # pinned deps, shared, NOT modified
AP="$CS/rice/experiments/allpests_e5d_curriculum"
PY="$CS/.venv/bin/python"
OUT_ROOT="${OUT_ROOT:-${CURRICULUM_OUT_ROOT:-$CS/rice/outputs_allpests_e5d_curriculum}}"

YEARS_DEFAULT="2022 2023 2024"
OFFS_DEFAULT="3,7,14,21,28,30,35,42,45,49,56,60"
DECAY_KM_DEFAULT="20.0"
STAGES="pilot final"   # uncond REMOVED: shared-multi-offset has no hazard path (model.py:499 h=None vs :772 h+z_flat)

E5D_SED='s/--stage2_pmf_asym_weight [^ ]+/--stage2_pmf_asym_weight 25.0/;
         s/--stage2_pmf_asym_weight_early [^ ]+/--stage2_pmf_asym_weight_early 0.0/'

pests_all()    { grep -v '^#' "$AP/pests.tsv" | awk 'NF{print $1}'; }
pests_server() { grep -v '^#' "$AP/pests.tsv" | awk -v s="$1" 'NF && $2==s {print $1}'; }

batch_dir() { [ "$1" = "2024" ] && echo "batch_2024_bestgate" || echo "batch_$1_baseline"; }
baseline_log() { echo "$CS/rice/outputs/stage2/$(batch_dir "$2")/$1/logs/stage2_lead_v3_train.log"; }

_raw_line() {   # _raw_line <pest> <year> <pilot|final>
  local log; log="$(baseline_log "$1" "$2")"
  [ -s "$log" ] || { echo "[stage_args] FATAL no log for $1/$2" >&2; return 3; }
  grep -E 'rice\.scripts\.run_train' "$log" \
    | grep -E "out_root rice/[^ ]*/lead_v3_$3( |$)" | head -1 \
    | sed -E 's|^.*-m rice\.scripts\.run_train ||'
}

_strip() {
  sed -E 's| --out_root [^ ]+||; s| --stage2_warm_start_ckpt [^ ]+||; s| --stage2_warm_start_seed [0-9]+||'
}

# stage_args <pest> <year> <stage> [warm_ckpt]
stage_args() {
  local pest="$1" year="$2" stage="$3" warm="${4:-}" args
  case "$stage" in
    pilot)  args="$(_raw_line "$pest" "$year" pilot | _strip)" ;;
    final)  args="$(_raw_line "$pest" "$year" final | _strip | sed -E "$E5D_SED")" ;;
    *) echo "[stage_args] FATAL unknown stage '$stage'" >&2; return 2 ;;
  esac
  [ -z "$args" ] && { echo "[stage_args] FATAL empty extract $pest/$year/$stage" >&2; return 3; }
  args="$args --stage2_add_neighbor_history --stage2_neighbor_decay_km ${DECAY_KM:-$DECAY_KM_DEFAULT}"
  [ -n "$warm" ] && args="$args --stage2_warm_start_ckpt $warm --stage2_warm_start_seed 0"
  printf '%s' "$args"
}

export_e5d_env() {
  export WBPH_SHARED_MULTI_OFFSET=1
  export WBPH_MU_HEAD_MODE=offset_specific
  export WBPH_CANDIDATE_OFFSETS="${OFFS:-$OFFS_DEFAULT}"
}

marker()    { echo "$OUT_ROOT/$1/.done_$2"; }
step_done() { [ -f "$(marker "$1" "$2")" ] && [ "${FORCE:-0}" != "1" ]; }
mark_done() { mkdir -p "$OUT_ROOT/$1"; date -Is > "$(marker "$1" "$2")"; }
