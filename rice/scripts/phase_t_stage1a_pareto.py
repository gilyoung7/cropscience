"""
Phase T — Stage 1a Pareto frontier diagnostic (yearsplit cohort).

For a Stage 1a XGB nowcast ckpt:
  - Compute val/test nowcast scores
  - Temperature-calibrate on val
  - For each (tau, k_consecutive) cell on val + test:
      apply k-consecutive alert rule -> (recall, precision, FAR, F1)
  - Pareto recommendation per k:
      * tau_F1max  : tau maximizing F1 on val
      * tau_R>=0.85: lowest FAR_val s.t. recall_val >= 0.85
      * tau_R>=0.90: same for 0.90

Outputs:
  - sweep CSV: every (k, tau) row with val + test metrics
  - recommendation summary printed to stdout
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import build_nowcast_samples, build_tabular_from_samples
from rice.scripts.run_calibrate_event import apply_temperature, fit_temperature_grid
from rice.scripts.run_stage1b_cascade_v2 import derive_alerts, metrics_from_alerts


def build_probs_df(stage1_ckpt_path: Path, run: int, args) -> tuple[pd.DataFrame, pd.DataFrame, float, float]:
    ckpt = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=False)
    C.DOY_START = int(ckpt.get("doy_start", 60))
    C.DOY_END = int(ckpt.get("doy_end", 300))
    add_tpos = bool(ckpt.get("add_tstar_position_feature", False))
    nc_window = int(ckpt.get("nowcast_window", 28))
    nc_stride = int(ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("nowcast_event_time_proxy", "r"))

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples = build_samples_for_run(run, get_feature_cols)

    if bool(ckpt.get("site_history_added", False)):
        from rice.scripts.site_history_utils import (
            compute_site_history, append_history_to_samples, HISTORY_FEATURE_DIM,
        )
        history = compute_site_history(
            samples, doy_start=int(C.DOY_START),
            policy=str(ckpt.get("site_history_policy", "rolling")),
            train_year_max=int(ckpt.get("history_train_year_max", 2021)),
        )
        append_history_to_samples(samples, history, doy_start=int(C.DOY_START))
        print(f"[history] appended {HISTORY_FEATURE_DIM} channels (eval; matches ckpt meta)")
    if bool(ckpt.get("phenology_added", False)):
        from rice.scripts.phenology_utils import (
            load_pheno_map, append_pheno_to_samples, PHENO_FEATURE_DIM,
        )
        pheno_map = load_pheno_map()
        append_pheno_to_samples(samples, pheno_map, doy_start=int(C.DOY_START))
        print(f"[phenology] appended {PHENO_FEATURE_DIM} channels (eval; matches ckpt meta)")
    if bool(ckpt.get("derived_weather_added", False)):
        from rice.scripts.derived_weather_utils import (
            append_derived_weather_to_samples, DERIVED_WEATHER_DIM,
        )
        append_derived_weather_to_samples(samples)
        print(f"[derived_weather] appended {DERIVED_WEATHER_DIM} channels (eval; matches ckpt meta)")

    _, val_seas, test_seas = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )
    val_s = build_nowcast_samples(val_seas, window=nc_window, stride=nc_stride,
                                  only_pre_event=nc_only_pre, event_time_proxy=nc_proxy)
    test_s = build_nowcast_samples(test_seas, window=nc_window, stride=nc_stride,
                                   only_pre_event=nc_only_pre, event_time_proxy=nc_proxy)
    y_val = np.asarray([int(s["y_event"]) for s in val_s])
    X_val = build_tabular_from_samples(val_s, add_tstar_position_feature=add_tpos)
    X_test = build_tabular_from_samples(test_s, add_tstar_position_feature=add_tpos)

    clf = ckpt["trained_states"][0]["sk_model"]
    p_val_raw = clf.predict_proba(X_val)[:, 1]
    p_test_raw = clf.predict_proba(X_test)[:, 1]
    t_best, _ = fit_temperature_grid(y_val, p_val_raw)
    p_val_cal = apply_temperature(p_val_raw, t_best)
    p_test_cal = apply_temperature(p_test_raw, t_best)

    def _df(seas, nowcast_s, p_cal):
        sy_meta = {(str(s["site_id"]), int(s["year"])): s for s in seas}
        rows = []
        for s, p in zip(nowcast_s, p_cal):
            key = (str(s["site_id"]), int(s["year"]))
            meta = sy_meta[key]
            ctype = str(meta["censor_type"])
            rows.append({
                "site": key[0], "year": key[1], "tstar": int(s["tstar"]),
                "p_cal": float(p), "y_event": int(s["y_event"]),
                "true_L": int(meta["L"]) if ctype != "right" else None,
                "true_R": int(meta["R"]) if ctype != "right" else None,
            })
        return pd.DataFrame(rows)

    n_interval_val = sum(1 for s in val_seas if str(s["censor_type"]) != "right")
    n_interval_test = sum(1 for s in test_seas if str(s["censor_type"]) != "right")
    print(f"[stage1] ckpt={stage1_ckpt_path.name}  T*={t_best:.3f}")
    print(f"[val]  groups={len(val_seas)}  interval={n_interval_val}  nowcast_rows={len(val_s)}")
    print(f"[test] groups={len(test_seas)}  interval={n_interval_test}  nowcast_rows={len(test_s)}")
    return _df(val_seas, val_s, p_val_cal), _df(test_seas, test_s, p_test_cal), float(t_best), float(n_interval_test)


def sweep_pareto(val_df: pd.DataFrame, test_df: pd.DataFrame, tau_grid: np.ndarray, ks: list[int]) -> pd.DataFrame:
    rows = []
    for k in ks:
        for tau in tau_grid:
            a_val = derive_alerts(val_df, float(tau), k_consecutive=k)
            a_test = derive_alerts(test_df, float(tau), k_consecutive=k)
            m_val = metrics_from_alerts(a_val, label=f"val k={k} tau={tau:.3f}")
            m_test = metrics_from_alerts(a_test, label=f"test k={k} tau={tau:.3f}")
            rows.append({
                "k": int(k), "tau": float(tau),
                "val_recall": m_val["recall"], "val_precision": m_val["precision"],
                "val_FAR": m_val["FAR"], "val_F1": m_val["F1"],
                "val_n_alert": m_val["n_alert"],
                "test_recall": m_test["recall"], "test_precision": m_test["precision"],
                "test_FAR": m_test["FAR"], "test_F1": m_test["F1"],
                "test_n_alert": m_test["n_alert"],
            })
    return pd.DataFrame(rows)


def recommend(sweep_df: pd.DataFrame, recall_targets=(0.85, 0.90)) -> pd.DataFrame:
    recs = []
    for k, g in sweep_df.groupby("k"):
        g = g.copy()
        f1_row = g.loc[g["val_F1"].idxmax()]
        recs.append({"k": int(k), "criterion": "F1max", **_pick_cols(f1_row)})
        for r in recall_targets:
            cands = g[g["val_recall"] >= r]
            if cands.empty:
                recs.append({"k": int(k), "criterion": f"recall>={r:.2f}",
                             "tau": None, "val_recall": None, "val_FAR": None,
                             "val_precision": None, "val_F1": None,
                             "test_recall": None, "test_FAR": None,
                             "test_precision": None, "test_F1": None,
                             "test_n_alert": None})
                continue
            best = cands.loc[cands["val_FAR"].idxmin()]
            recs.append({"k": int(k), "criterion": f"recall>={r:.2f}", **_pick_cols(best)})
    return pd.DataFrame(recs)


def _pick_cols(row):
    return {
        "tau": float(row["tau"]),
        "val_recall": float(row["val_recall"]),
        "val_FAR": float(row["val_FAR"]),
        "val_precision": float(row["val_precision"]),
        "val_F1": float(row["val_F1"]),
        "test_recall": float(row["test_recall"]),
        "test_FAR": float(row["test_FAR"]),
        "test_precision": float(row["test_precision"]),
        "test_F1": float(row["test_F1"]),
        "test_n_alert": int(row["test_n_alert"]),
    }


def print_full_sweep(df: pd.DataFrame) -> None:
    for k, g in df.groupby("k"):
        print(f"\n=== k_consecutive = {k} ===")
        print(f"  {'tau':>6}  {'val_R':>6} {'val_F':>6} {'val_P':>6} {'val_F1':>7} | "
              f"{'tst_R':>6} {'tst_F':>6} {'tst_P':>6} {'tst_F1':>7} {'tst_nA':>6}")
        for _, r in g.sort_values("tau").iterrows():
            print(f"  {r['tau']:>6.3f}  "
                  f"{r['val_recall']:>6.3f} {r['val_FAR']:>6.3f} {r['val_precision']:>6.3f} {r['val_F1']:>7.3f} | "
                  f"{r['test_recall']:>6.3f} {r['test_FAR']:>6.3f} {r['test_precision']:>6.3f} {r['test_F1']:>7.3f} {int(r['test_n_alert']):>6d}")


def print_recommendation(rec_df: pd.DataFrame) -> None:
    print("\n========== Recommendation (val-selected, test-reported) ==========")
    print(f"  {'k':>2} {'criterion':>12} {'tau':>6} | "
          f"{'val_R':>6} {'val_F':>6} {'val_P':>6} {'val_F1':>6} | "
          f"{'tst_R':>6} {'tst_F':>6} {'tst_P':>6} {'tst_F1':>6} {'tst_nA':>6}")
    for _, r in rec_df.iterrows():
        if pd.isna(r["tau"]):
            print(f"  {int(r['k']):>2} {r['criterion']:>12} {'-':>6} | (no operating point on val)")
            continue
        print(f"  {int(r['k']):>2} {r['criterion']:>12} {r['tau']:>6.3f} | "
              f"{r['val_recall']:>6.3f} {r['val_FAR']:>6.3f} {r['val_precision']:>6.3f} {r['val_F1']:>6.3f} | "
              f"{r['test_recall']:>6.3f} {r['test_FAR']:>6.3f} {r['test_precision']:>6.3f} {r['test_F1']:>6.3f} {int(r['test_n_alert']):>6d}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", type=str, default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--stage1_ckpt", type=str, required=True)
    ap.add_argument("--label", type=str, default="")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--tau_min", type=float, default=0.05)
    ap.add_argument("--tau_max", type=float, default=0.95)
    ap.add_argument("--tau_step", type=float, default=0.025)
    ap.add_argument("--ks", type=str, default="1,2,3", help="comma-separated k_consecutive values")
    ap.add_argument("--out_csv", type=str, default="")
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    stage1_ckpt = Path(args.stage1_ckpt)
    label = args.label or stage1_ckpt.parent.parent.name

    val_df, test_df, t_best, n_int_test = build_probs_df(stage1_ckpt, args.run, args)

    tau_grid = np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    print(f"\n[sweep] taus={len(tau_grid)}  ks={ks}  label='{label}'")

    sweep_df = sweep_pareto(val_df, test_df, tau_grid, ks)
    rec_df = recommend(sweep_df)

    print_full_sweep(sweep_df)
    print_recommendation(rec_df)

    if args.out_csv:
        out_path = Path(args.out_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sweep_df.to_csv(out_path, index=False)
        rec_path = out_path.with_name(out_path.stem + "_recommendation.csv")
        rec_df.to_csv(rec_path, index=False)
        print(f"\n[saved] sweep -> {out_path}")
        print(f"[saved] recommendation -> {rec_path}")


if __name__ == "__main__":
    main()
