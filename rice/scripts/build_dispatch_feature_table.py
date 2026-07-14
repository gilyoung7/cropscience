"""Build the per-(site, year) Stage-1 dispatch confidence feature table.

This is the *training-time* feature source for Stage 2. It is independent
of the offset-specific sample_grid used by phase_r_oracle_iou for mu-IoU
diagnostics; that grid only contains the subset of alerts that are also
interval-censored AND match a Stage-2 nowcast frame at >=1 offset (351 of
the 987 dispatch-alerted site-years for sheath_blight R>=0.88).

By default this script covers train + val + test splits so Stage-2 training
sees dispatch features on the train cohort too (otherwise train rows would
be all-missing and the model could not learn the confidence feature
relationship). Train-split A/D probabilities are *in-sample* (Stage-1 was
fit on those years); a warning is printed but the feature is still emitted.
A future OOF/rolling refit can replace train rows without changing the CSV
schema.

For each dispatch-alerted site-year (R>=target selection in the year-split
summary JSON), this script writes one row with:

    site, year, split,           ← 'train' / 'val' / 'test'
    alert_tstar (DOY),
    with_history, dispatch_branch,
    A_score_at_alert, D_score_at_alert, score_margin,
    dispatch_score_at_alert, dispatch_tau_used, score_over_tau_margin,
    recent_14d_mean_score, recent_28d_mean_score,
    score_above_tau_streak, score_rolling_slope_14d,
    p_mean_so_far_at_alert

The feature column ordering matches stage1_confidence_utils.DISPATCH_FEATURE_NAMES.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from rice.scripts.phase_r_oracle_iou import (
    build_dispatch_alert_map,
    _dispatch_features_for_sy,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--dispatch_summary_json", required=True,
                    help="Year-split group_tau_hybrid_summary.json")
    ap.add_argument("--dispatch_target_label", default="R>=0.88")
    ap.add_argument("--dispatch_a_ckpt", required=True,
                    help="Stage-1 A (baseline) ckpt path")
    ap.add_argument("--dispatch_d_ckpt", required=True,
                    help="Stage-1 D (history) ckpt path")
    ap.add_argument("--out_csv", required=True,
                    help="Output CSV path for the per-(site, year) feature table")
    ap.add_argument("--include_splits", default="train,val,test",
                    help="Comma-separated subset of {train,val,test} to emit "
                         "dispatch features for. Default 'train,val,test' is "
                         "required for Stage-2 training so the train cohort "
                         "carries non-missing dispatch features. Use 'test' "
                         "only to reproduce the previous diagnostic-grade "
                         "behavior.")
    ap.add_argument("--gate_method", default="dispatch_group_tau",
                    choices=["A_baseline", "D_history", "dispatch_group_tau"],
                    help="Stage-1 gate policy used to derive alert_tstar. "
                         "'dispatch_group_tau' uses with_history split (D for "
                         "with_h, A for no_h). 'A_baseline' uses A_raw_global "
                         "selection's (k, tau) for all sy. 'D_history' uses "
                         "D_raw_global selection's (k, tau) for all sy. The "
                         "summary JSON must contain the matching selection key "
                         "at --dispatch_target_label.")
    args = ap.parse_args()

    include_splits = tuple(
        s.strip() for s in str(args.include_splits).split(",") if s.strip()
    )
    valid = {"train", "val", "test"}
    bad = [s for s in include_splits if s not in valid]
    if bad:
        raise SystemExit(f"[abort] invalid splits {bad!r}; allowed: {sorted(valid)}")
    if not include_splits:
        raise SystemExit("[abort] --include_splits must list at least one split")

    alert_map, n_total, ctx = build_dispatch_alert_map(
        Path(args.dispatch_a_ckpt),
        Path(args.dispatch_d_ckpt),
        Path(args.dispatch_summary_json),
        args.dispatch_target_label,
        args.run,
        args,
        include_splits=include_splits,
        gate_method=args.gate_method,
    )

    split_by_sy = ctx.get("split_by_sy", {})

    rows = []
    for sy, alert_t in alert_map.items():
        feat = _dispatch_features_for_sy(sy, alert_t, ctx)
        rows.append({
            "site": str(sy[0]),
            "year": int(sy[1]),
            "split": split_by_sy.get(sy, "unknown"),
            **feat,
        })

    df = pd.DataFrame(rows)
    # Stable column order: site, year, split, then DISPATCH_FEATURE_NAMES order
    from rice.scripts.stage1_confidence_utils import DISPATCH_FEATURE_NAMES
    front = ["site", "year", "split"]
    ordered_cols = front + [c for c in DISPATCH_FEATURE_NAMES if c in df.columns]
    extra = [c for c in df.columns if c not in ordered_cols]
    df = df[ordered_cols + extra]

    out_path = Path(args.out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    counts = df["split"].value_counts(dropna=False).to_dict() if len(df) else {}
    print(f"[saved] {out_path}  ({len(df)} site-year rows; "
          f"target_label={args.dispatch_target_label}; "
          f"n_interval_test={n_total}; split_counts={counts})")
    if counts.get("train", 0) > 0:
        print(f"[warn] train rows ({counts['train']}) are IN-SAMPLE Stage-1 "
              f"probability outputs. Feature values may be optimistically "
              f"sharper than what an OOF refit would produce. Acceptable for a "
              f"first sanity training; replace with OOF before treating as final.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
