"""Build climatology baseline sample_grids for canonical comparison vs lead_v3.

For each of three climatology baselines (mean_L, mean_mid, mean_R), this
script produces VAL and TEST sample_grid CSVs in the same schema as
phase_r_oracle_iou output. The mu value is a sample-INDEPENDENT constant
computed on the TRAIN cohort (dispatch-alerted interval-censored
site-years only — same restriction as Stage 2 evaluation).

For each output row:
  - mu       = train_cohort climatology constant (mean_L | mean_mid | mean_R)
  - sigma    = --sigma (default 5.0; matches lead_v3)
  - PI       = [mu - 1.96*sigma, mu + 1.96*sigma]
  - iou_matched = overlap_metrics(pL, pR, L, R)  recomputed per row
  - matched/n_match/coverage/alert_tstar carry over from the base sample_grid
    (climatology does not change the alert+offset matching set)

Output CSVs drop straight into phase_b_canonical_summary --entry. Climatology
mu does not depend on offset, so val-best offset selection naturally picks
the offset with the largest matched coverage.

NO new training / no Stage 2 forward — pure file repackaging from existing
lead_v3 (or any reference) sample_grid + train cohort label statistics.

NOTE on label semantics:
  L = last unobserved-as-occurred date, R = first observed-as-occurred date.
  True event time T ∈ (L, R]. mean_L is necessarily early-biased; mean_R
  late-biased; mean_mid the most neutral. Reporting all three lets the
  reader judge how much of Stage 2 IoU is "sample-specific timing" vs
  "climatology + L-target bias".
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples
from rice.src.train_eval import overlap_metrics
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.stage1_confidence_utils import load_dispatch_feature_table


def compute_train_climatology(pest: str, run: int, split_seed: int,
                                val_year: int, test_year_min: int,
                                test_year_max: int,
                                dispatch_feature_csv: str,
                                doy_start: int) -> dict:
    """Mean L_abs, R_abs, mid_abs across TRAIN dispatch-alerted INTERVAL sy.
    Returns a stats dict plus the three constants."""
    _, get_feature_cols = resolve_pest(pest)
    _, _, _T, samples = build_samples_for_run(run, get_feature_cols)
    train_s, _, _ = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=split_seed,
        split_mode="year", val_year=val_year,
        test_year_min=test_year_min, test_year_max=test_year_max,
    )
    conf_map = load_dispatch_feature_table(dispatch_feature_csv)
    LR_pairs: list[tuple[int, int]] = []
    n_total = n_alerted = n_interval = 0
    for s in train_s:
        n_total += 1
        sy = (str(s["site_id"]), int(s["year"]))
        if sy not in conf_map:
            continue
        n_alerted += 1
        if str(s.get("censor_type", "")) != "interval":
            continue
        n_interval += 1
        # samples store L/R as 1-based season indices (matches phase_r_oracle_iou
        # convention: L_abs = info["true_L"] + doy_start - 1).
        L_abs = int(s["L"]) + int(doy_start) - 1
        R_abs = int(s["R"]) + int(doy_start) - 1
        LR_pairs.append((L_abs, R_abs))
    if not LR_pairs:
        raise SystemExit(
            "[abort] no train cohort interval-censored alerted sy "
            f"(pest={pest!r}, dispatch_feature_csv={dispatch_feature_csv}, "
            f"doy_start={doy_start}). Check the CSV covers the train years."
        )
    L_arr = np.asarray([p[0] for p in LR_pairs], dtype=float)
    R_arr = np.asarray([p[1] for p in LR_pairs], dtype=float)
    mid_arr = (L_arr + R_arr) * 0.5
    return {
        "pest": pest,
        "n_train_total_sy": n_total,
        "n_train_alerted_sy": n_alerted,
        "n_train_alerted_interval_sy": n_interval,
        "mean_L": float(L_arr.mean()),
        "mean_R": float(R_arr.mean()),
        "mean_mid": float(mid_arr.mean()),
        "median_L": float(np.median(L_arr)),
        "median_R": float(np.median(R_arr)),
        "median_mid": float(np.median(mid_arr)),
        "std_L": float(L_arr.std(ddof=0)),
        "std_R": float(R_arr.std(ddof=0)),
        "std_mid": float(mid_arr.std(ddof=0)),
    }


def build_clim_grid(base_csv: str, mu_const: float, sigma: float,
                     label: str, out_csv: str) -> dict:
    """Copy base sample_grid; replace mu/sigma/iou_matched with climatology
    values. matched / alert_tstar / offset preserved verbatim.
    """
    df = pd.read_csv(base_csv)
    df = df.copy()
    df["model"] = label
    HW = 1.96 * float(sigma)
    pL = int(round(float(mu_const) - HW))
    pR = int(round(float(mu_const) + HW))
    new_iou = np.zeros(len(df), dtype=float)
    # Vectorize where possible; overlap_metrics is per-row.
    L_arr = df["L"].astype(int).values
    R_arr = df["R"].astype(int).values
    matched_arr = df["matched"].astype(bool).values
    for i in range(len(df)):
        if not matched_arr[i]:
            continue
        iou, _, _ = overlap_metrics(pL, pR, int(L_arr[i]), int(R_arr[i]))
        new_iou[i] = float(iou)
    df["mu"] = float(mu_const)
    df["sigma"] = float(sigma)
    df["iou_matched"] = new_iou
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    n_match = int(matched_arr.sum())
    n_sy = int(df["sample_id"].nunique())
    print(f"[clim_grid] {label}  mu={mu_const:.2f}  PI=[{pL},{pR}]  "
          f"-> {out_csv}  rows={len(df)}  n_match={n_match}  n_sy={n_sy}")
    return {
        "out_csv": out_csv, "mu": float(mu_const),
        "pL": pL, "pR": pR, "rows": len(df), "n_match": n_match,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--doy_start", type=int, default=60,
                    help="Must match the doy_start used to produce the base "
                         "sample_grids (lead_v3 ckpt has doy_start=60).")
    ap.add_argument("--dispatch_feature_csv", required=True,
                    help="Per-(site,year) dispatch confidence CSV (cohort + "
                         "alert_tstar). Same file used for Stage 2 dispatch eval.")
    ap.add_argument("--base_val_grid", required=True,
                    help="Reference val sample_grid CSV (any Stage 2 grid; "
                         "only the cohort/schema is reused).")
    ap.add_argument("--base_test_grid", required=True,
                    help="Reference test sample_grid CSV.")
    ap.add_argument("--sigma", type=float, default=5.0,
                    help="PI half-width = 1.96*sigma. Match Stage 2 (default 5.0).")
    ap.add_argument("--out_prefix", default="rice/outputs/diag/phase_B_climatology",
                    help="Output prefix; produces six CSVs and one stats CSV.")
    ap.add_argument("--stats_csv", default=None,
                    help="Optional path to dump the per-train-cohort climatology "
                         "stats. Default: <out_prefix>_train_stats.csv")
    args = ap.parse_args()

    print(f"[climatology] computing train cohort stats for pest={args.pest}")
    clim = compute_train_climatology(
        args.pest, args.run, args.split_seed,
        args.val_year, args.test_year_min, args.test_year_max,
        args.dispatch_feature_csv, args.doy_start,
    )
    print(f"[train_cohort] {clim}")

    stats_csv = args.stats_csv or f"{args.out_prefix}_train_stats.csv"
    pd.DataFrame([clim]).to_csv(stats_csv, index=False)
    print(f"[stats] wrote {stats_csv}")

    entries: list[tuple[str, str, str]] = []  # (label, val_csv, test_csv)
    for stat in ("mean_L", "mean_mid", "mean_R"):
        mu_const = clim[stat]
        label = f"climatology_{stat}"
        out_v = f"{args.out_prefix}_{stat}_val_sample_grid.csv"
        out_t = f"{args.out_prefix}_{stat}_test_sample_grid.csv"
        build_clim_grid(args.base_val_grid, mu_const, args.sigma, label, out_v)
        build_clim_grid(args.base_test_grid, mu_const, args.sigma, label, out_t)
        entries.append((label, out_v, out_t))

    print("\n" + "=" * 78)
    print("Next: canonical comparison with lead_v3.")
    print("=" * 78)
    print(".venv/bin/python -m rice.scripts.phase_b_canonical_summary \\")
    print(f"  --entry 'lead_v3|val={args.base_val_grid}|test={args.base_test_grid}' \\")
    for label, v, t in entries:
        print(f"  --entry '{label}|val={v}|test={t}' \\")
    print(f"  --out_per_offset {args.out_prefix}_per_offset.csv \\")
    print(f"  --out_selection  {args.out_prefix}_selection.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
