#!/usr/bin/env bash
# Assert the curriculum arm's FINAL command is identical to the scratch arm's, except for
# warm-start. That equality is the entire validity of the A-vs-B comparison: if any other
# argument drifts, the measured difference is no longer attributable to initialisation.
#
# Exits non-zero on any drift. Reads only; trains nothing.
set -uo pipefail
CS=/home/gpu4080/research/cropscience
SCRATCH="$CS/rice/experiments/allpests_e5d/common.sh"
CURR="$CS/rice/experiments/allpests_e5d_curriculum/common.sh"
PESTS="${PESTS:-WBPH brown_spot rice_stem_borer_2 rice_stem_borer_1}"
YEARS="${YEARS:-2022 2023 2024}"

echo "=== final-argument parity: scratch vs curriculum ==="
fail=0
for p in $PESTS; do
  for y in $YEARS; do
    a="$(bash -c "source $SCRATCH; extract_args $p $y" 2>/dev/null)"
    b="$(bash -c "source $CURR;    stage_args   $p $y final" 2>/dev/null)"
    # the curriculum adds warm-start only when a pilot ckpt is passed; here it is absent,
    # so the two strings must match exactly.
    if [ "$a" = "$b" ]; then
      echo "  ok    $p/$y  identical ($(printf '%s' "$a" | wc -w) tokens)"
    else
      echo "  FAIL  $p/$y"
      diff <(printf '%s' "$a" | tr ' ' '\n') <(printf '%s' "$b" | tr ' ' '\n') | head -8
      fail=$((fail+1))
    fi
  done
done

echo
echo "=== controlled knobs (curriculum final, WBPH/2024) ==="
A="$(bash -c "source $CURR; stage_args WBPH 2024 final" 2>/dev/null)"
for k in stage2_pmf_mode stage2_pmf_mu_mode stage2_pmf_asym_weight stage2_pmf_asym_weight_early \
         lr dropout weight_decay split_mode split_seed seeds val_year test_year_min test_year_max \
         stage2_best_metric stage2_dispatch_feature_csv d_model_override batch_train_override; do
  v="$(printf '%s' "$A" | grep -oE -- "--$k [^ ]+" | head -1 | cut -d' ' -f2-)"
  printf "  %-34s %s\n" "$k" "${v:-<absent: trainer default>}"
done
echo "  (patience / max_epochs are absent from the command -> pest config: PATIENCE=6 MAX_EPOCHS=60,"
echo "   identical for both arms since neither passes an override)"

echo
if [ "$fail" -eq 0 ]; then echo "PARITY OK — only initialisation differs"; else echo "PARITY FAILED on $fail cell(s)"; fi
exit "$fail"
