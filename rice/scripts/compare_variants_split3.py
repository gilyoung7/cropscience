"""Compare Stage-1 Track B variants A / D / N / DN on one pest (default sheath_blight).

For each variant it reuses the EXISTING production engine end-to-end:
  build_probs (phase_t_lead_aware_eval) -> sweep_useful (phase_t_useful_pareto)
  -> gate selection (same rule as select_stage1_gate_split3_2024: val_recall >=
     0.875 then FAR-min, F1 tie-break; max-recall fallback)
and writes one combined TSV row per variant.

Variants are distinguished only by the ckpt's augmentation flags (the eval
re-appends the same channels the model was trained on):
  A  : baseline                         (site_history_added=F, neighbor=F)
  D  : + same-site history              (site_history_added=T)
  N  : + neighbor history               (neighbor_history_added=T)
  DN : + both
This script ONLY reads ckpts and writes a TSV — it does not modify existing
scripts and does not train.

Existing A/D production sweeps (useful_sweep_{A,D}.csv) are reused if present;
N/DN sweeps are generated with the same grid and saved alongside, matching the
production useful_pareto/ layout.

Usage:
    cd /home/gpu4080/research/cropscience
    python rice/scripts/compare_variants_split3.py \
        --pest sheath_blight --run 4 \
        --base rice/outputs_stage1/batch_rolling/sheath_blight/run0/split3_v2023_t2024
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from rice.scripts.phase_t_lead_aware_eval import build_probs
from rice.scripts.phase_t_useful_pareto import sweep_useful, merge_wide

# Same val-recall floor as select_stage1_gate_split3_2024.py (0.88 - 0.005).
DEFAULT_RECALL_THRESHOLD = 0.875

OUTPUT_COLUMNS = [
    "variant",
    "selected_k",
    "selected_tau",
    "val_recall",
    "val_FAR",
    "test_recall",
    "test_FAR",
    "test_precision",
    "test_F1",
    "test_lead_median",
    "test_lead_mean",
    "test_n_event",
    "fallback",
    "sweep_source",
]


def select_operating_point(merged: pd.DataFrame, threshold: float) -> tuple[pd.Series, bool]:
    """Mirror select_stage1_gate_split3_2024.pick() within a single-variant sweep.

    Feasible (recall_val >= threshold): min FAR_val, tie-break F1_val desc.
    Else: max-recall fallback, tie-break FAR_val asc.
    """
    feas = merged[merged["recall_val"] >= float(threshold)]
    if len(feas):
        ranked = feas.sort_values(["FAR_val", "F1_val"], ascending=[True, False])
        return ranked.iloc[0], False
    ranked = merged.sort_values(["recall_val", "FAR_val"], ascending=[False, True])
    return ranked.iloc[0], True


def get_or_build_sweep(
    variant: str,
    ckpt_path: Path,
    sweep_csv: Path,
    args,
    ks: list[int],
    tau_grid: np.ndarray,
    reuse_existing: bool,
) -> tuple[pd.DataFrame, str]:
    """Return (merged_sweep_df, source_label). Reuse existing CSV if allowed."""
    if reuse_existing and sweep_csv.exists():
        merged = pd.read_csv(sweep_csv)
        return merged, f"reused:{sweep_csv}"

    ns = SimpleNamespace(
        pest=args.pest,
        run=int(args.run),
        stage1_ckpt=str(ckpt_path),
        split_seed=int(args.split_seed),
        val_year=int(args.val_year),
        test_year_min=int(args.test_year_min),
        test_year_max=int(args.test_year_max),
    )
    cache = build_probs(ns)
    from rice.configs import config as C
    doy_start = int(C.DOY_START)
    val_sweep = sweep_useful(cache["val_df"], tau_grid, ks, doy_start)
    test_sweep = sweep_useful(cache["test_df"], tau_grid, ks, doy_start)
    merged = merge_wide(val_sweep, test_sweep)
    sweep_csv.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(sweep_csv, index=False)
    return merged, f"generated:{sweep_csv}"


def main() -> None:
    ap = argparse.ArgumentParser(description="Compare Track B variants A/D/N/DN.")
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument(
        "--base",
        default="rice/outputs_stage1/batch_rolling/sheath_blight/run0/split3_v2023_t2024",
        help="dir containing <variant>/ckpt/event_xgb_w28_lead14-45_<variant>.pt",
    )
    ap.add_argument("--variants", default="A,D,N,DN")
    ap.add_argument("--ckpt_template", default="event_xgb_w28_lead14-45_{variant}.pt",
                    help="ckpt filename under <base>/<variant>/ckpt/")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2023)
    ap.add_argument("--test_year_min", type=int, default=2024)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--tau_min", type=float, default=0.05)
    ap.add_argument("--tau_max", type=float, default=0.95)
    ap.add_argument("--tau_step", type=float, default=0.025)
    ap.add_argument("--recall_threshold", type=float, default=DEFAULT_RECALL_THRESHOLD)
    ap.add_argument("--force_regenerate", action="store_true",
                    help="regenerate every sweep (overwrites existing useful_sweep_*.csv); "
                         "default reuses existing A/D production sweeps and only builds N/DN")
    ap.add_argument("--out", default=None,
                    help="output TSV (default: <base>/compare_A_D_N_DN.tsv)")
    args = ap.parse_args()

    base = Path(args.base)
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    ks = [int(x) for x in str(args.ks).split(",") if x.strip()]
    tau_grid = np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step)
    out_path = Path(args.out) if args.out else base / "compare_A_D_N_DN.tsv"

    print(f"[cfg] pest={args.pest} run={args.run} base={base}")
    print(f"[cfg] val_year={args.val_year} test_year=[{args.test_year_min},{args.test_year_max}] "
          f"ks={ks} taus={len(tau_grid)} recall_thr={args.recall_threshold}")

    rows = []
    for variant in variants:
        ckpt_path = base / variant / "ckpt" / args.ckpt_template.format(variant=variant)
        sweep_csv = base / variant / "useful_pareto" / f"useful_sweep_{variant}.csv"
        if not ckpt_path.exists():
            print(f"[WARN] variant={variant}: ckpt not found, skipping: {ckpt_path}")
            continue
        print(f"\n===== variant {variant} :: {ckpt_path} =====")
        merged, source = get_or_build_sweep(
            variant, ckpt_path, sweep_csv, args, ks, tau_grid,
            reuse_existing=not args.force_regenerate)
        sel, fallback = select_operating_point(merged, args.recall_threshold)
        row = {
            "variant": variant,
            "selected_k": int(sel["k"]),
            "selected_tau": round(float(sel["tau"]), 4),
            "val_recall": round(float(sel["recall_val"]), 4),
            "val_FAR": round(float(sel["FAR_val"]), 4),
            "test_recall": round(float(sel["recall_test"]), 4),
            "test_FAR": round(float(sel["FAR_test"]), 4),
            "test_precision": round(float(sel["precision_test"]), 4),
            "test_F1": round(float(sel["F1_test"]), 4),
            "test_lead_median": round(float(sel["lead_median_test"]), 2),
            "test_lead_mean": round(float(sel["lead_mean_test"]), 2),
            "test_n_event": int(sel.get("n_event_test", sel.get("n_event_val", 0))),
            "fallback": "yes" if fallback else "no",
            "sweep_source": source,
        }
        rows.append(row)
        print(f"  selected k={row['selected_k']} tau={row['selected_tau']} "
              f"| val R={row['val_recall']} FAR={row['val_FAR']} "
              f"| test R={row['test_recall']} FAR={row['test_FAR']} "
              f"P={row['test_precision']} F1={row['test_F1']} "
              f"lead_med={row['test_lead_median']} lead_mean={row['test_lead_mean']}"
              + (" [FALLBACK]" if fallback else ""))

    if not rows:
        raise SystemExit("[abort] no variant produced a result")

    df = pd.DataFrame(rows)[OUTPUT_COLUMNS]
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, sep="\t", index=False)
    print(f"\n[ok] wrote {len(df)} variant rows -> {out_path}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
