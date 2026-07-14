"""Phase 1 dispatch sample_grid validator and reporter.

Reads a sample_grid CSV produced by ``phase_r_oracle_iou`` running in dispatch
mode (``--dispatch_summary_json`` set) and reports the items required by the
Phase 1 success criteria:

  - dispatch alert site-year count (per model = unique sample_id)
  - sample_grid row count (rows = sample-years x offsets)
  - one row per (sample_id, offset) check
  - presence of the 14 dispatch feature columns + alert_tstar
  - NaN ratio and min/mean/max for each dispatch feature
  - with_history / no_history split count
  - dispatch_branch A/D split count
  - alert_tstar DOY mean/median/min/max
  - D_score_at_alert / score_margin NaN row counts

The dispatch alert count printed here can be cross-checked against the
``[dispatch] alerts: total=...`` line from the phase_r_oracle_iou stdout.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd


DISPATCH_FEATURE_COLS = [
    "alert_tstar",
    "dispatch_branch",
    "with_history",
    "A_score_at_alert",
    "D_score_at_alert",
    "score_margin",
    "dispatch_score_at_alert",
    "dispatch_tau_used",
    "score_over_tau_margin",
    "recent_14d_mean_score",
    "recent_28d_mean_score",
    "score_above_tau_streak",
    "score_rolling_slope_14d",
    "p_mean_so_far_at_alert",
]


def _fmt(v: float, ndigits: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "nan"
    return f"{v:.{ndigits}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample_grid_csv", required=True,
                    help="Path to the dispatch sample_grid CSV.")
    ap.add_argument("--per_model", action="store_true",
                    help="Report per-model in addition to overall.")
    args = ap.parse_args()

    df = pd.read_csv(args.sample_grid_csv)
    print(f"# sample_grid: {args.sample_grid_csv}")
    print(f"# rows={len(df)}  columns={len(df.columns)}")

    missing = [c for c in DISPATCH_FEATURE_COLS if c not in df.columns]
    if missing:
        print(f"[FAIL] missing dispatch columns: {missing}")
        return 2
    print(f"[ok] all 14 dispatch feature columns + alert_tstar present")

    models = sorted(df["model"].dropna().unique().tolist())
    print(f"# models: {models}")

    scopes = [("OVERALL", df)]
    if args.per_model and len(models) > 1:
        for m in models:
            scopes.append((f"model={m}", df[df["model"] == m]))

    for tag, sub in scopes:
        print("\n" + "=" * 72)
        print(f"## {tag}  rows={len(sub)}")
        print("=" * 72)

        # Per-(sample_id, offset) uniqueness
        sample_ids = sub["sample_id"].unique()
        n_sy = len(sample_ids)
        offsets = sorted(sub["offset"].unique().tolist())
        n_off = len(offsets)
        dup = sub.groupby(["sample_id", "offset"]).size()
        n_dup = int((dup > 1).sum())
        expected_rows = n_sy * n_off
        print(f"  unique sample_id (= dispatch-alerted site-years) = {n_sy}")
        print(f"  offsets in grid = {offsets}  (n={n_off})")
        print(f"  (sample_id, offset) duplicates = {n_dup}")
        print(f"  rows = {len(sub)}  expected = sy * offsets = {expected_rows}  "
              f"{'OK' if len(sub) == expected_rows and n_dup == 0 else 'MISMATCH'}")

        # Per-sample dispatch features (constant across offsets) — take one row per sample_id
        first = sub.drop_duplicates(subset=["sample_id"], keep="first").copy()

        # branch / with_history counts
        br = first["dispatch_branch"].value_counts(dropna=False).to_dict()
        wh = first["with_history"].value_counts(dropna=False).to_dict()
        print(f"  dispatch_branch counts = {br}")
        print(f"  with_history    counts = {wh}  (1 = with_history, 0 = no_history)")
        # Consistency check: branch=D <-> with_history=1, branch=A <-> with_history=0
        bad = first[
            ((first["dispatch_branch"] == "D") & (first["with_history"] != 1)) |
            ((first["dispatch_branch"] == "A") & (first["with_history"] != 0))
        ]
        if len(bad):
            print(f"  [WARN] {len(bad)} rows with dispatch_branch/with_history mismatch")
        else:
            print(f"  [ok] dispatch_branch and with_history are consistent")

        # alert_tstar DOY distribution
        adoy = first["alert_tstar"].dropna().astype(int)
        print(f"  alert_tstar (DOY): n={len(adoy)}  "
              f"mean={adoy.mean():.2f}  median={int(adoy.median())}  "
              f"min={int(adoy.min())}  max={int(adoy.max())}")

        # NaN and min/mean/max per dispatch feature
        print(f"  feature stats (one row per sample_id; NaN_ratio in [0,1]):")
        header = f"    {'feature':<28s}  {'NaN':>5s}  {'NaN_ratio':>9s}  {'min':>10s}  {'mean':>10s}  {'max':>10s}"
        print(header)
        for col in DISPATCH_FEATURE_COLS:
            if col == "dispatch_branch":
                v = first[col]
                n_nan = int(v.isna().sum())
                ratio = n_nan / max(len(v), 1)
                print(f"    {col:<28s}  {n_nan:>5d}  {ratio:>9.4f}  "
                      f"{'-':>10s}  {'-':>10s}  {'-':>10s}")
                continue
            v = pd.to_numeric(first[col], errors="coerce")
            n_nan = int(v.isna().sum())
            ratio = n_nan / max(len(v), 1)
            vmin = v.min() if v.notna().any() else float("nan")
            vmean = v.mean() if v.notna().any() else float("nan")
            vmax = v.max() if v.notna().any() else float("nan")
            print(f"    {col:<28s}  {n_nan:>5d}  {ratio:>9.4f}  "
                  f"{_fmt(vmin):>10s}  {_fmt(vmean):>10s}  {_fmt(vmax):>10s}")

        # Explicit NaN counts for D_score / score_margin (Phase 0 follow-up)
        n_dnan = int(pd.to_numeric(first["D_score_at_alert"], errors="coerce").isna().sum())
        n_mnan = int(pd.to_numeric(first["score_margin"], errors="coerce").isna().sum())
        print(f"  D_score_at_alert NaN rows = {n_dnan}  ({n_dnan/max(n_sy,1):.4f})")
        print(f"  score_margin     NaN rows = {n_mnan}  ({n_mnan/max(n_sy,1):.4f})")

    return 0


if __name__ == "__main__":
    sys.exit(main())
