#!/usr/bin/env bash
# PREPARE (do not run the pipeline) BPH's Stage-1 -> dispatch regeneration at DOY 60-300.
#
# WHY. BPH is the only pest whose model grid is 140-270 (T=131); the other seven are 60-300
# (T=241). Stage-1 and Stage-2 currently AGREE on 140-270 for BPH, so today's numbers are
# internally consistent -- they are just on a different axis, which is why BPH cannot be ranked
# against the rest. Moving Stage-2 alone would break `offset = issue - alert`, because
# alert_tstar was produced by a Stage-1 model fit in the 140-270 frame. So Stage-1 must be
# rebuilt first, then the dispatch table, then Stage-2.
#
# HOW. rice/scripts/phase_t_lead_aware_train.py reads DOY from the TEMPLATE ckpt
# (L131-132: C.DOY_START = ckpt["doy_start"]), not from a CLI flag, and the batch driver
# exposes no --doy override. The minimal intervention is therefore a PATCHED COPY of BPH's
# template with the two geometry keys rewritten. Nothing else in the template changes:
# feature_cols, feature_names and the hyperparameters (read off trained_states[0].sk_model)
# are carried over untouched, so the only difference from production BPH is the season window.
#
# NOTHING IS EXECUTED HERE beyond writing the patched template and printing the commands.
# Run with --emit to also write them to a runnable script.
#
#   bash bph_regen_prepare.sh            # patch template + print plan
#   bash bph_regen_prepare.sh --emit     # + write bph_regen_run.sh (still not executed)
set -euo pipefail
CS=/home/gpu4080/research/cropscience
AP="$CS/rice/experiments/allpests_e5d"
PY="$CS/.venv/bin/python"

SRC="$CS/rice/outputs_stage1/BPH_siteyear42/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split42_siteyear_ymin2002.pt"
DST_DIR="$CS/rice/outputs_stage1/BPH_doy60_300_template/ckpt"
DST="$DST_DIR/event_run4_xgb_nowcast_w28_s1_tpos_split42_siteyear_ymin2002_doy60_300.pt"
OUT_S1="$CS/rice/outputs_stage1/batch_rolling_bph_doy60_300"
NEW_START=60; NEW_END=300

[ -f "$SRC" ] || { echo "[abort] source template missing: $SRC"; exit 2; }

echo "=== 1. patched template ==="
mkdir -p "$DST_DIR"
"$PY" - "$SRC" "$DST" "$NEW_START" "$NEW_END" <<'PYEOF'
import sys, torch
src, dst, s, e = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
c = torch.load(src, map_location="cpu", weights_only=False)
old = (c.get("doy_start"), c.get("doy_end"))
c["doy_start"], c["doy_end"] = s, e
c["T"] = e - s + 1
c["_regen_note"] = (f"geometry patched {old} -> ({s},{e}) for the all-pest E5d comparison; "
                    "feature_cols / feature_names / hyperparameters unchanged from the source template")
torch.save(c, dst)
print(f"  {old} -> ({s},{e})  T={c['T']}  d_in={c.get('d_in')}  feats={len(c.get('feature_cols', []))}")
print(f"  wrote {dst}")
PYEOF

echo
echo "=== 2. Stage-1 rebuild (NOT executed) ==="
echo "  Uses the production driver so every other knob (lead 14-45 / ignore, window 28,"
echo "  KS_DISPATCH=3, RUN=4, seeds) stays identical to the frozen seven."
cat <<EOF

  cd $CS
  bash scripts/run_stage1_batch_pests.sh \\
      --pests BPH \\
      --splits "split1 split2 split3" \\
      --out_root $OUT_S1

  # NOTE: run_stage1_batch_pests.sh reads its template from the hard-coded PEST_TEMPLATE map.
  # Point BPH at the patched copy for this run WITHOUT editing the committed driver, e.g.
  #   PEST_TEMPLATE_BPH=$DST bash scripts/run_stage1_batch_pests.sh ...
  # if the driver is extended to honour that env var, or copy the driver to
  #   $AP/run_stage1_batch_pests_bph60300.sh
  # and change only that one map entry. Do NOT edit the production driver in place.
EOF

echo
echo "=== 3. dispatch table rebuild (NOT executed) ==="
cat <<EOF
  For each eval year 2022/2023/2024, rebuild the per-site-year dispatch features from the
  NEW Stage-1 output, writing to a NEW directory (never over the frozen production CSVs):

  cd $CS
  \$PY -m rice.scripts.build_dispatch_feature_table \\
      --pest BPH --stage1_root $OUT_S1 \\
      --val_year <Y-1> --test_year_min <Y> --test_year_max <Y> \\
      --out_csv $CS/rice/outputs/stage2/bph_doy60_300/<Y>/gate_<kind>_R088_features_per_sy.csv

  Confirm the exact flag names against build_dispatch_feature_table.py before running --
  they were NOT verified here (no execution was performed).
EOF

echo
echo "=== 4. Stage-2 E5d ==="
cat <<EOF
  Once (2) and (3) exist, flip BPH's row in $AP/pests.tsv to 60/300 and point
  pest_paths.dispatch_csv() at the new directory, then the normal chain runs unchanged:
      bash $AP/run_pest.sh BPH
  Until then dry_run.py check [10] keeps BPH marked REGENERATE so it is not silently
  compared against the frozen seven.
EOF

echo
echo "=== INVARIANTS this rebuild must preserve (verify before trusting BPH numbers) ==="
cat <<'EOF'
  - lead window [14,45], outside_policy=ignore, nowcast_window=28, RUN=4, KS_DISPATCH=3
  - forward-chained splits: train < val_year, val = Y-1, test = Y
  - the 7 other pests' Stage-1 artifacts are byte-identical (dry_run check [10] hashes them)
  - Stage-1 and Stage-2 must BOTH be 60-300 for BPH afterwards; a half-migration is worse
    than the current state, because offset = issue - alert would silently change meaning.
EOF

if [ "${1:-}" = "--emit" ]; then
  echo "[emit] writing $AP/bph_regen_run.sh (still not executed)"
  { echo '#!/usr/bin/env bash'; echo '# GENERATED PLAN -- review every line before running.';
    echo 'set -euo pipefail'; echo "cd $CS"; } > "$AP/bph_regen_run.sh"
  chmod +x "$AP/bph_regen_run.sh"
fi
echo
echo "[prepare] template ready. Stage-1 / dispatch / Stage-2 were NOT run."
