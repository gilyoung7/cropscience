#!/usr/bin/env bash
# Shared helpers for the all-pest E5d run. Source this; do not execute it.
#
# The ONE non-obvious thing here is arg extraction. scripts/93 (the WBPH E5d recipe) does not
# hard-code hyperparameters -- it greps the *production* training command out of that cell's own
# log and sed-patches only the mu-loss knobs. We keep that mechanism verbatim so every pest
# inherits its own production hyperparameters, because they genuinely differ per (pest, year):
#   - the Stage-1 gate CSV is one of gate_{dispatch_group_tau,A_baseline,D_history}_R088_*
#     chosen per cell,
#   - WBPH/BPH/rsb1/rsb2 carry --batch_train_override 16, the four diseases do not,
#   - 2024 dropped --stage2_gaussian_loss_mode / _interval_lambda / _pmf_delta_max.
# Substituting another pest's or another year's args would silently change the model.

set -euo pipefail

WS=/home/gpu4080/research/wbph_interval_perf_202607
CS=/home/gpu4080/research/cropscience
PY="$CS/.venv/bin/python"
AP="$CS/rice/experiments/allpests_e5d"
OUT_ROOT="${OUT_ROOT:-$WS/outputs/allpests_e5d}"

YEARS_DEFAULT="2022 2023 2024"
OFFS_DEFAULT="3,7,14,21,28,30,35,42,45,49,56,60"
DECAY_KM_DEFAULT="20.0"

# E5d cell = onset target (l_offset, inherited) + 25x late penalty + NO very-early term.
# Identical to scripts/93 EXP=E5d. These two knobs are the whole difference vs the D1 reference.
E5D_SED='s/--stage2_pmf_asym_weight [^ ]+/--stage2_pmf_asym_weight 25.0/;
         s/--stage2_pmf_asym_weight_early [^ ]+/--stage2_pmf_asym_weight_early 0.0/'

pests_all()    { grep -v '^#' "$AP/pests.tsv" | awk 'NF{print $1}'; }
pests_server() { grep -v '^#' "$AP/pests.tsv" | awk -v s="$1" 'NF && $2==s {print $1}'; }
pest_field()   { grep -v '^#' "$AP/pests.tsv" | awk -v p="$1" -v c="$2" 'NF && $1==p {print $c}'; }

batch_dir() { [ "$1" = "2024" ] && echo "batch_2024_bestgate" || echo "batch_$1_baseline"; }

baseline_log() { echo "$CS/rice/outputs/stage2/$(batch_dir "$2")/$1/logs/stage2_lead_v3_train.log"; }

# extract_args <pest> <year>
#
# Emits the production run_train arg string for that cell, already patched for E5d:
#   out_root stripped (caller sets it), warm-start stripped, neighbor history added,
#   mu-loss knobs overridden.
#
# blast/2023 is the one cell whose log is 0 bytes (training ran -- the ckpt exists -- but the
# log was never captured). We fall back to blast/2022's command and re-point the three year
# fields plus the dispatch CSV. This is safe ONLY because blast's 2023 hparams.json is
# identical to its 2022 hparams.json on every non-year key (verified 2026-07-29); dry_run.py
# re-asserts that equality before any training starts, so the fallback cannot go stale silently.
extract_args() {
  local pest="$1" year="$2" log args src_year="$2"
  log="$(baseline_log "$pest" "$year")"

  if [ ! -s "$log" ] || ! grep -qE 'out_root rice/[^ ]*/lead_v3_final( |$)' "$log" 2>/dev/null; then
    if [ "$pest" = "blast" ] && [ "$year" = "2023" ]; then
      src_year=2022
      log="$(baseline_log blast 2022)"
      echo "[extract_args] blast/2023 log empty -> falling back to blast/2022 cmd (hparams-verified)" >&2
    else
      echo "[extract_args] FATAL no usable log for $pest/$year: $log" >&2
      return 3
    fi
  fi

  args="$(grep -E 'rice\.scripts\.run_train' "$log" \
        | grep -E 'out_root rice/[^ ]*/lead_v3_final( |$)' | head -1 \
        | sed -E 's|^.*-m rice\.scripts\.run_train ||')"
  [ -z "$args" ] && { echo "[extract_args] FATAL empty extract $pest/$year" >&2; return 3; }

  # re-point the year fields + dispatch CSV when we borrowed a sibling year's command
  if [ "$src_year" != "$year" ]; then
    local disp
    disp="$(ls "$CS/rice/outputs/stage2/$(batch_dir "$year")/$pest"/gate_*_R088_features_per_sy.csv 2>/dev/null | head -1)"
    [ -z "$disp" ] && { echo "[extract_args] FATAL no dispatch CSV for $pest/$year" >&2; return 3; }
    args="$(printf '%s' "$args" | sed -E \
      "s/--val_year [0-9]+/--val_year $((year-1))/; \
       s/--test_year_min [0-9]+/--test_year_min $year/; \
       s/--test_year_max [0-9]+/--test_year_max $year/; \
       s|--stage2_dispatch_feature_csv [^ ]+|--stage2_dispatch_feature_csv $disp|")"
  fi

  args="$args --stage2_add_neighbor_history --stage2_neighbor_decay_km ${DECAY_KM:-$DECAY_KM_DEFAULT}"
  args="$(printf '%s' "$args" \
        | sed -E 's| --out_root [^ ]+||; s| --stage2_warm_start_ckpt [^ ]+||; s| --stage2_warm_start_seed [0-9]+||')"
  printf '%s' "$args" | sed -E "$E5D_SED"
}

# shared-encoder + offset-specific-head geometry (identical to scripts/89/93 and the clean fold)
export_e5d_env() {
  export WBPH_SHARED_MULTI_OFFSET=1
  export WBPH_MU_HEAD_MODE=offset_specific
  export WBPH_CANDIDATE_OFFSETS="${OFFS:-$OFFS_DEFAULT}"
}

# done-markers make the whole pipeline resumable: a step that wrote its marker is skipped.
marker()      { echo "$OUT_ROOT/$1/.done_$2"; }
step_done()   { [ -f "$(marker "$1" "$2")" ] && [ "${FORCE:-0}" != "1" ]; }
mark_done()   { mkdir -p "$OUT_ROOT/$1"; date -Is > "$(marker "$1" "$2")"; }
