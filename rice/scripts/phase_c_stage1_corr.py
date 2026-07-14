"""
Phase C — Stage 1 score ↔ true_L correlation on nowcast samples (year-split).

Loads a trained Stage 1 XGBoost nowcast checkpoint, rebuilds season samples for
the configured run, splits by year, expands each split into per-t* nowcast
samples (matching the training-time pipeline), and runs inference per t*.

For every (site_id, year) test group the per-t* scores form a time series; we
extract four summary features used as Stage-2 conditioning candidates:

    score_at_tstar  : score at the latest t* in the group
    score_peak_30   : max score over the last 30 t* steps
    score_slope_14  : OLS slope (per t* step) over the last 14 t* steps
    score_auc_30    : trapezoidal AUC of the last 30 t* steps

Each feature is correlated against true_L (L_doy) using both Pearson r and
Spearman ρ, separately for all groups, event-only groups, and against the
remaining-time target (true_L − tstar_marker).

Baseline reference (year-split D=15, run=4, 2023-24 test): |r| ≈ 0.065.
Console output only; no CSVs are written.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr, spearmanr

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import (
    build_nowcast_samples,
    build_tabular_from_samples,
)
from rice.scripts.run_calibrate_event import apply_temperature, fit_temperature_grid


def _extract_group_features(scores: np.ndarray, tstars: np.ndarray):
    """
    scores, tstars: 1D arrays sorted by tstar ascending.
    Returns (score_at_tstar, score_peak_30, score_slope_14, score_auc_30, marker_tstar)
    """
    n = len(scores)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan

    marker_tstar = float(tstars[-1])
    score_at = float(scores[-1])

    last30 = scores[-min(30, n):]
    score_peak = float(np.max(last30))
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if _trapz is None:
        score_auc = float(np.sum((last30[:-1] + last30[1:]) * 0.5)) if len(last30) >= 2 else float("nan")
    else:
        score_auc = float(_trapz(last30))

    last14 = scores[-min(14, n):]
    if len(last14) >= 2:
        x14 = np.arange(len(last14), dtype=float)
        slope = float(np.polyfit(x14, last14, 1)[0])
    else:
        slope = np.nan
    return score_at, score_peak, slope, score_auc, marker_tstar


def _corr_pair(a: np.ndarray, b: np.ndarray):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    n = int(mask.sum())
    if n < 3:
        return float("nan"), float("nan"), n
    a2 = a[mask]
    b2 = b[mask]
    if np.std(a2) < 1e-12 or np.std(b2) < 1e-12:
        return float("nan"), float("nan"), n
    r, _ = pearsonr(a2, b2)
    rho, _ = spearmanr(a2, b2)
    return float(r), float(rho), n


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, required=True)
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--ckpt", type=str, required=True)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    p.add_argument("--doy_start_override", type=int, default=None)
    p.add_argument("--doy_end_override", type=int, default=None)
    args = p.parse_args()

    C_pest, get_feature_cols = resolve_pest(args.pest)

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        print(f"ckpt not found: {ckpt_path}", file=sys.stderr)
        sys.exit(2)
    ckpt = torch.load(ckpt_path, map_location="cpu")
    trained_states = ckpt["trained_states"]

    C.DOY_START = int(ckpt.get("doy_start", C.DOY_START))
    C.DOY_END = int(ckpt.get("doy_end", C.DOY_END))
    if args.doy_start_override is not None:
        C.DOY_START = int(args.doy_start_override)
    if args.doy_end_override is not None:
        C.DOY_END = int(args.doy_end_override)
    add_tstar_position_feature = bool(ckpt.get("add_tstar_position_feature", False))

    nc_window = int(ckpt.get("nowcast_window", 28))
    nc_stride = int(ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("nowcast_event_time_proxy", "r"))
    nc_tstar_start = ckpt.get("nowcast_tstar_start", None)
    nc_label_mode = str(ckpt.get("nowcast_label_mode", "eventually"))
    nc_label_horizon = ckpt.get("nowcast_label_horizon", None)

    print(
        f"[nowcast] window={nc_window} stride={nc_stride} only_pre_event={nc_only_pre} "
        f"proxy={nc_proxy} tstar_start={nc_tstar_start} label_mode={nc_label_mode} "
        f"label_horizon={nc_label_horizon} add_tstar_position_feature={add_tstar_position_feature}"
    )

    _, _, T, season_samples = build_samples_for_run(args.run, get_feature_cols)
    train_seas, val_seas, test_seas = split_samples(
        season_samples,
        val_frac=0.1, test_frac=0.1,
        seed=args.split_seed,
        split_mode="year",
        val_year=args.val_year,
        test_year_min=args.test_year_min,
        test_year_max=args.test_year_max,
    )
    print(
        f"[season split] train={len(train_seas)} val={len(val_seas)} test={len(test_seas)}"
    )

    def _expand(seas):
        return build_nowcast_samples(
            seas,
            window=nc_window,
            stride=nc_stride,
            tstar_start=nc_tstar_start,
            only_pre_event=nc_only_pre,
            event_time_proxy=nc_proxy,
            label_mode=nc_label_mode,
            label_horizon=int(nc_label_horizon) if nc_label_horizon is not None else None,
        )

    val_s = _expand(val_seas)
    test_s = _expand(test_seas)
    print(f"[nowcast split] val={len(val_s)} test={len(test_s)}")
    if len(test_s) == 0:
        print("[abort] test nowcast sample count = 0 — check split / window settings", file=sys.stderr)
        sys.exit(3)

    y_val = np.asarray([int(s["y_event"]) for s in val_s], dtype=float)
    y_test = np.asarray([int(s["y_event"]) for s in test_s], dtype=float)

    X_val_tab = build_tabular_from_samples(val_s, add_tstar_position_feature=add_tstar_position_feature)
    X_test_tab = build_tabular_from_samples(test_s, add_tstar_position_feature=add_tstar_position_feature)
    print(f"[ckpt] {ckpt_path}  | seeds: {[d['seed'] for d in trained_states]}")
    print(f"[features] D={X_test_tab.shape[1]}  | val_pos={int(y_val.sum())}/{len(y_val)}  test_pos={int(y_test.sum())}/{len(y_test)}")

    # Group test samples by (site_id, year)
    groups: dict[tuple[str, int], list[int]] = defaultdict(list)
    for i, s in enumerate(test_s):
        groups[(str(s["site_id"]), int(s["year"]))].append(i)

    print(f"[groups] test groups = {len(groups)}")

    for d in trained_states:
        seed = int(d["seed"])
        clf = d.get("sk_model")
        if clf is None:
            print(f"[seed {seed}] no sk_model in ckpt — skip")
            continue
        if hasattr(clf, "predict_proba"):
            p_val_raw = clf.predict_proba(X_val_tab)[:, 1]
            p_test_raw = clf.predict_proba(X_test_tab)[:, 1]
        else:
            p_val_raw = np.asarray(clf.predict(X_val_tab), dtype=float)
            p_test_raw = np.asarray(clf.predict(X_test_tab), dtype=float)

        t_best, _ = fit_temperature_grid(y_val, p_val_raw)
        p_test_cal = apply_temperature(p_test_raw, t_best)

        # Build per-group time series & 4 features
        feat_at, feat_peak, feat_slope, feat_auc = [], [], [], []
        true_L_arr, marker_arr, y_group = [], [], []

        for key, idx_list in groups.items():
            idx_arr = np.array(idx_list, dtype=int)
            tstars = np.asarray([int(test_s[i]["tstar"]) for i in idx_arr], dtype=float)
            order = np.argsort(tstars)
            idx_sorted = idx_arr[order]
            tstars_sorted = tstars[order]
            scores_sorted = p_test_cal[idx_sorted]

            score_at, score_peak, slope, auc, marker = _extract_group_features(scores_sorted, tstars_sorted)
            feat_at.append(score_at)
            feat_peak.append(score_peak)
            feat_slope.append(slope)
            feat_auc.append(auc)
            marker_arr.append(marker)
            # group y_event = any sample in group is positive (pre-event window)
            y_group.append(int(np.any([int(test_s[i]["y_event"]) for i in idx_sorted])))
            # true_L: from any sample with L not None; else NaN
            L_vals = [test_s[i]["L"] for i in idx_sorted if test_s[i]["L"] is not None]
            true_L_arr.append(float(L_vals[0]) if L_vals else float("nan"))

        feat_at = np.asarray(feat_at)
        feat_peak = np.asarray(feat_peak)
        feat_slope = np.asarray(feat_slope)
        feat_auc = np.asarray(feat_auc)
        true_L_arr = np.asarray(true_L_arr)
        marker_arr = np.asarray(marker_arr)
        y_group = np.asarray(y_group)

        Lrem = true_L_arr - marker_arr
        pos_mask = y_group > 0.5
        n_groups = len(y_group)
        n_pos = int(pos_mask.sum())
        n_L_finite = int(np.isfinite(true_L_arr).sum())

        print(
            f"\n[seed {seed}] T*={t_best:.3f}  groups={n_groups}  pos_groups={n_pos}  L_finite={n_L_finite}"
        )

        features = [
            ("score_at_tstar", feat_at),
            ("score_peak_30",  feat_peak),
            ("score_slope_14", feat_slope),
            ("score_auc_30",   feat_auc),
        ]

        header = "  feature           subset   target          | n     Pearson r    Spearman ρ   |r|"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for name, f in features:
            for subset_name, sel in [("all", slice(None)), ("pos", pos_mask)]:
                f_sel = f[sel]
                L_sel = true_L_arr[sel]
                Lrem_sel = Lrem[sel]
                r1, rho1, n1 = _corr_pair(f_sel, L_sel)
                r2, rho2, n2 = _corr_pair(f_sel, Lrem_sel)
                print(
                    f"  {name:<16}  {subset_name:<6}  true_L          | "
                    f"{n1:<5} {r1:+.4f}      {rho1:+.4f}      {abs(r1):.4f}"
                )
                print(
                    f"  {name:<16}  {subset_name:<6}  true_L - tstar  | "
                    f"{n2:<5} {r2:+.4f}      {rho2:+.4f}      {abs(r2):.4f}"
                )

        # baseline reference comparison
        baseline_abs_r = 0.065
        max_abs_r_all_true_L = max(
            abs(_corr_pair(f, true_L_arr)[0]) if np.isfinite(_corr_pair(f, true_L_arr)[0]) else 0.0
            for _, f in features
        )
        delta = max_abs_r_all_true_L - baseline_abs_r
        print(
            f"\n  [vs baseline] best |r| (all, true_L) = {max_abs_r_all_true_L:.4f}  "
            f"baseline = {baseline_abs_r:.3f}  Δ = {delta:+.4f}"
        )


if __name__ == "__main__":
    main()
