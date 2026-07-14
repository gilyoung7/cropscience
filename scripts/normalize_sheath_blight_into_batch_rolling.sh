#!/usr/bin/env bash
#
# Normalize sheath_blight Stage 1 rolling results into the batch_rolling
# layout so merge_pest_batch_farmin + phase_b_stage1_operational_review
# pick it up automatically as one of the 8 final pests.
#
# Source (existing): per-(split, seed) results from Phase B seed_stability,
# which already use the same {A,D}/{ckpt,useful_pareto} + group_tau + logs
# tree as batch_rolling — only the parent path needs remapping.
#
#   rice/outputs_stage1/seed_stability/sheath_blight_split{N}_v{val}_t{test}_seed{S}/
#       A/ckpt/event_xgb_w28_lead14-45_A.pt
#       A/useful_pareto/{useful_sweep_A.csv, useful_selections_A.json}
#       D/ckpt/event_xgb_w28_lead14-45_D.pt
#       D/useful_pareto/{useful_sweep_D.csv, useful_selections_D.json}
#       group_tau/group_tau_hybrid_summary.json (+ aux CSVs)
#       logs/
#
# Target (created by this script):
#   rice/outputs_stage1/batch_rolling/sheath_blight/run{S}/split{N}_v{val}_t{test}/
#       (same subtree, content shared via symlinks by default)
#
# Idempotent: re-running is safe (skip-if-link-exists; --force to recreate).
# Originals are NEVER touched.

set -uo pipefail

MODE=symlink            # symlink | copy
FORCE=0
SRC_ROOT="rice/outputs_stage1/seed_stability"
DST_ROOT="rice/outputs_stage1/batch_rolling/sheath_blight"

usage() {
  cat <<EOF
Usage: $0 [--mode symlink|copy] [--force]
  --mode      symlink (default; tiny + zero-copy) or copy (independent files)
  --force     remove existing target dir/links and recreate
EOF
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mode) MODE="$2"; shift 2;;
    --force) FORCE=1; shift;;
    -h|--help) usage; exit 0;;
    *) echo "[abort] unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ "$MODE" != "symlink" && "$MODE" != "copy" ]]; then
  echo "[abort] --mode must be symlink|copy" >&2; exit 2
fi

declare -a MAP=(
  "split1_v2021_t2022_seed0:run0:split1_v2021_t2022"
  "split1_v2021_t2022_seed1:run1:split1_v2021_t2022"
  "split1_v2021_t2022_seed2:run2:split1_v2021_t2022"
  "split2_v2022_t2023_seed0:run0:split2_v2022_t2023"
  "split2_v2022_t2023_seed1:run1:split2_v2022_t2023"
  "split2_v2022_t2023_seed2:run2:split2_v2022_t2023"
  "split3_v2023_t2024_seed0:run0:split3_v2023_t2024"
  "split3_v2023_t2024_seed1:run1:split3_v2023_t2024"
  "split3_v2023_t2024_seed2:run2:split3_v2023_t2024"
)

echo "[normalize] mode=$MODE  force=$FORCE"
echo "[normalize] src_root=$SRC_ROOT"
echo "[normalize] dst_root=$DST_ROOT"
mkdir -p "$DST_ROOT"

n_ok=0; n_skip=0; n_fail=0; n_missing_src=0
for entry in "${MAP[@]}"; do
  IFS=":" read -r src_tag run_tag split_tag <<<"$entry"
  src="${SRC_ROOT}/sheath_blight_${src_tag}"
  dst="${DST_ROOT}/${run_tag}/${split_tag}"

  if [[ ! -d "$src" ]]; then
    echo "  [missing src] $src  (skipping)"
    n_missing_src=$((n_missing_src+1))
    continue
  fi

  if [[ -e "$dst" || -L "$dst" ]]; then
    if [[ "$FORCE" == "1" ]]; then
      rm -rf "$dst"
    else
      echo "  [skip exists] $dst"
      n_skip=$((n_skip+1))
      continue
    fi
  fi

  mkdir -p "$(dirname "$dst")"
  if [[ "$MODE" == "symlink" ]]; then
    # Resolve source to absolute for robust links from any CWD.
    src_abs="$(cd "$src" && pwd)"
    ln -s "$src_abs" "$dst"
  else
    cp -r "$src" "$dst"
  fi
  if [[ $? -eq 0 ]]; then
    echo "  [linked] $src  ->  $dst"
    n_ok=$((n_ok+1))
  else
    echo "  [FAIL]  $src  ->  $dst" >&2
    n_fail=$((n_fail+1))
  fi
done

echo
echo "[normalize] done: ok=$n_ok  skip_existing=$n_skip  missing_src=$n_missing_src  fail=$n_fail"
echo
echo "[verify] structure check:"
find "$DST_ROOT" -maxdepth 2 -mindepth 1 -type d -o -type l | sort | sed "s|^|  |"
echo
echo "[verify] sample artifact resolution:"
for f in "$DST_ROOT/run0/split1_v2021_t2022/A/ckpt/event_xgb_w28_lead14-45_A.pt" \
         "$DST_ROOT/run0/split1_v2021_t2022/D/useful_pareto/useful_sweep_D.csv" \
         "$DST_ROOT/run2/split3_v2023_t2024/group_tau/group_tau_hybrid_summary.json"; do
  if [[ -s "$f" ]]; then
    echo "  [ok] $f  ($(stat -c %s "$f") bytes)"
  else
    echo "  [missing] $f"
  fi
done

echo
echo "[next] re-run merge + operational review to include sheath_blight in batch_rolling:"
echo "  .venv/bin/python -m rice.scripts.merge_pest_batch_farmin \\"
echo "    --base rice/outputs_stage1/batch_rolling \\"
echo "    --out_csv rice/outputs_stage1/batch_rolling/_summary/pest_batch_farmin_all.csv \\"
echo "    --out_summary_csv rice/outputs_stage1/batch_rolling/_summary/pest_batch_farmin_R088.csv \\"
echo "    --target_for_summary 0.88 --exclude_pests BPH2"
echo "  .venv/bin/python -m rice.scripts.phase_b_stage1_operational_review"
