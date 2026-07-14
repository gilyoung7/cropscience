"""
Phase T5 — Stage 1 occurrence diagnostic + label-timing + feature-group shortcut.

[A] Site-year occurrence classification (val + test, no retraining):
    Aggregate p_cal per site-year via {p_max, p_top5, p_top10, p_mean_season, p_area}.
    Report ROC-AUC, PR-AUC, F1max, precision@recall=0.85/0.90.
    Overlay current alert rule (--tau_a, --k_consecutive) operating point on the
    (FAR, TPR) plane for reference.

[B] Label-timing contamination (val + test, no retraining):
    For event site-years, bin tstar rows by days_to_event = true_L - tstar.
    Bins: [0,30), [30,60), [60,90), [90, +inf).
    Compare each bin's p_cal distribution vs non-event rows (KS, mean, p50, frac>=tau).
    -> if [90+) event rows look like non-event, model has no early signal.

[C] Shortcut diagnostic — retrain 2 XGB variants with restricted feature sets:
    - calendar_only: tstar_rel + lat + lon + days_since_growing_start (+ their __miss)
    - weather_only : weather rolling/stats (+ __miss), NO tstar_rel
    - full         : existing ckpt scores (no retraining)
    For each: occurrence ROC-AUC (p_max) + alert metrics at val-recall match
    Hyper-params copied from existing ckpt for fairness.

Reads probs cache from phase_t_stage1b_cascade if available (preferred — has train_now).
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from xgboost import XGBClassifier

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import (
    build_nowcast_samples, build_tabular_from_samples, make_event_labels,
)
from rice.scripts.run_calibrate_event import apply_temperature, fit_temperature_grid
from rice.scripts.run_stage1b_cascade_v2 import derive_alerts, metrics_from_alerts


SCORE_AGGS = ["p_max", "p_top5", "p_top10", "p_mean_season", "p_area"]

# Channel layout (30 channels = 15 base + 15 missing flags) from ckpt feature_names:
#   0-9:   weather rolling (rain_7d_sum/days, tmean/tmax/tmin, rh, sun, rad, trange, trange_7d_mean)
#   10-11: lat, lon
#   12-14: days_since_growing_start, days_until_growing_end, is_growing
#   15-29: missing flags for channels 0-14
WEATHER_CH = list(range(0, 10))
WEATHER_MISS = list(range(15, 25))
LATLON_CH = [10, 11]
LATLON_MISS = [25, 26]
DSGS_CH = [12]
DSGS_MISS = [27]

GROUPS = {
    "full":          {"channels": list(range(30)),                        "add_tpos": True},
    "calendar_only": {"channels": LATLON_CH + DSGS_CH + LATLON_MISS + DSGS_MISS,  "add_tpos": True},
    "weather_only":  {"channels": WEATHER_CH + WEATHER_MISS,              "add_tpos": False},
}


def load_data(args) -> dict:
    """Use cache if present (preferred). Else rebuild train_now/val_now/test_now + probs."""
    if args.probs_cache and Path(args.probs_cache).exists():
        with open(args.probs_cache, "rb") as f:
            cache = pickle.load(f)
        print(f"[cache] loaded {args.probs_cache}  has_train_now={'train_now' in cache}")
        if "train_now" in cache:
            return cache
        print("[cache] has dfs but no train_now -> rebuilding nowcast samples")
    return _rebuild(args, existing_cache=cache if args.probs_cache and Path(args.probs_cache).exists() else None)


def _rebuild(args, existing_cache=None) -> dict:
    print("[build] reading ckpt and building samples (this may take a minute)")
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    C.DOY_START = int(ckpt.get("doy_start", 60))
    C.DOY_END = int(ckpt.get("doy_end", 300))
    add_tpos = bool(ckpt.get("add_tstar_position_feature", False))
    nc_window = int(ckpt.get("nowcast_window", 28))
    nc_stride = int(ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("nowcast_event_time_proxy", "r"))
    nc_label_mode = str(ckpt.get("nowcast_label_mode", "eventually"))
    nc_label_horizon = ckpt.get("nowcast_label_horizon", None)
    nc_tstart = ckpt.get("nowcast_tstar_start", None)

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples = build_samples_for_run(args.run, get_feature_cols)
    train_seas, val_seas, test_seas = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )
    nc_kw = dict(window=nc_window, stride=nc_stride, tstar_start=nc_tstart,
                 only_pre_event=nc_only_pre, event_time_proxy=nc_proxy,
                 label_mode=nc_label_mode, label_horizon=nc_label_horizon)
    train_now = build_nowcast_samples(train_seas, **nc_kw)
    val_now = build_nowcast_samples(val_seas, **nc_kw)
    test_now = build_nowcast_samples(test_seas, **nc_kw)
    print(f"[build] train={len(train_now)} val={len(val_now)} test={len(test_now)}")

    # production probs from existing ckpt (full features)
    X_val = build_tabular_from_samples(val_now, add_tstar_position_feature=add_tpos)
    X_test = build_tabular_from_samples(test_now, add_tstar_position_feature=add_tpos)
    y_val = make_event_labels(val_now)
    clf = ckpt["trained_states"][0]["sk_model"]
    p_val_raw = clf.predict_proba(X_val)[:, 1]
    p_test_raw = clf.predict_proba(X_test)[:, 1]
    t_best, _ = fit_temperature_grid(y_val, p_val_raw)
    p_val_cal = apply_temperature(p_val_raw, t_best)
    p_test_cal = apply_temperature(p_test_raw, t_best)

    def _df(seas, nc, p, split_name):
        sy_meta = {(str(s["site_id"]), int(s["year"])): s for s in seas}
        rows = []
        for s, prob in zip(nc, p):
            key = (str(s["site_id"]), int(s["year"]))
            meta = sy_meta[key]
            ctype = str(meta["censor_type"])
            rows.append({
                "split": split_name, "site": key[0], "year": key[1],
                "tstar": int(s["tstar"]), "p_cal": float(prob),
                "y_event": int(s["y_event"]),
                "true_L": int(meta["L"]) if ctype != "right" else None,
                "true_R": int(meta["R"]) if ctype != "right" else None,
            })
        return pd.DataFrame(rows)

    cache = {
        "t_best": float(t_best),
        "val_df": _df(val_seas, val_now, p_val_cal, "val"),
        "test_df": _df(test_seas, test_now, p_test_cal, "test"),
        "train_now": train_now, "val_now": val_now, "test_now": test_now,
        "train_seas": train_seas, "val_seas": val_seas, "test_seas": test_seas,
    }
    return cache


def aggregate_scores(probs_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (site, year), g in probs_df.groupby(["site", "year"], sort=False):
        ps = g["p_cal"].values.astype(float)
        ps_sorted = np.sort(ps)[::-1]
        top5 = ps_sorted[:5].mean() if len(ps_sorted) else 0.0
        top10 = ps_sorted[:10].mean() if len(ps_sorted) else 0.0
        rows.append({
            "site": str(site), "year": int(year),
            "y_event": int(g["y_event"].iloc[0]),
            "p_max": float(ps.max()) if len(ps) else 0.0,
            "p_top5": float(top5),
            "p_top10": float(top10),
            "p_mean_season": float(ps.mean()) if len(ps) else 0.0,
            "p_area": float(ps.sum()),
        })
    return pd.DataFrame(rows)


def metrics_at_recall_targets(y: np.ndarray, scores: np.ndarray, targets=(0.85, 0.90)) -> dict:
    prec, rec, _ = precision_recall_curve(y, scores)
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    out = {
        "ROC_AUC": float(roc_auc_score(y, scores)),
        "PR_AUC": float(average_precision_score(y, scores)),
        "F1max": float(f1s.max()),
    }
    for t in targets:
        mask = rec >= t
        out[f"P@R{int(t*100)}"] = float(prec[mask].max()) if mask.any() else None
    return out


def site_year_alert_label(probs_df: pd.DataFrame, tau: float, k: int) -> pd.DataFrame:
    """Apply k-consecutive alert rule, return (site, year, y_event, alerted)."""
    a = derive_alerts(probs_df, tau, k_consecutive=k)
    a = a.rename(columns={"is_event": "y_event"})
    return a[["site", "year", "y_event", "alerted"]].copy()


def section_A(val_df: pd.DataFrame, test_df: pd.DataFrame, tau_a: float, k: int) -> dict:
    print("\n========== [A] Site-year occurrence classification ==========")
    out = {}
    for split_name, df in [("val", val_df), ("test", test_df)]:
        agg = aggregate_scores(df)
        y = agg["y_event"].values
        n_pos = int(y.sum()); n_neg = int(len(y) - n_pos)
        print(f"\n  [{split_name}] n={len(agg)}  n_event={n_pos}  n_nonevent={n_neg}")
        print(f"  {'aggregation':>14}  {'ROC':>6} {'PR':>6} {'F1max':>6} {'P@R85':>6} {'P@R90':>6}")
        agg_out = {}
        for col in SCORE_AGGS:
            m = metrics_at_recall_targets(y, agg[col].values)
            agg_out[col] = m
            p85 = f"{m['P@R85']:.3f}" if m['P@R85'] is not None else "  -  "
            p90 = f"{m['P@R90']:.3f}" if m['P@R90'] is not None else "  -  "
            print(f"  {col:>14}  {m['ROC_AUC']:>6.3f} {m['PR_AUC']:>6.3f} {m['F1max']:>6.3f}  {p85:>6}  {p90:>6}")
        # alert rule point on ROC plane (uses k-consecutive rule, site-year level)
        sy_alert = site_year_alert_label(df, tau_a, k)
        tp = int(((sy_alert.y_event == 1) & (sy_alert.alerted == 1)).sum())
        fp = int(((sy_alert.y_event == 0) & (sy_alert.alerted == 1)).sum())
        fn = int(((sy_alert.y_event == 1) & (sy_alert.alerted == 0)).sum())
        tn = int(((sy_alert.y_event == 0) & (sy_alert.alerted == 0)).sum())
        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        print(f"  alert (k={k}, tau={tau_a}) point on ROC: (FAR={fpr:.3f}, TPR={tpr:.3f})")
        out[split_name] = {"agg_metrics": agg_out, "alert_point": {"FAR": fpr, "TPR": tpr, "TP": tp, "FP": fp, "FN": fn, "TN": tn}}
    return out


def section_B(val_df: pd.DataFrame, test_df: pd.DataFrame, tau_a: float) -> dict:
    print("\n========== [B] Label-timing contamination (days_to_event bins) ==========")
    out = {}
    for split_name, df in [("val", val_df), ("test", test_df)]:
        print(f"\n  [{split_name}]")
        ev = df[(df.y_event == 1) & df.true_L.notna()].copy()
        ev["dte"] = ev["true_L"].astype(float) - ev["tstar"].astype(float)
        ne = df[df.y_event == 0].copy()
        ne_scores = ne["p_cal"].values
        def _bin(d):
            if d < 0: return "after_L"
            if d < 30: return "0_30"
            if d < 60: return "30_60"
            if d < 90: return "60_90"
            return "90+"
        ev["bin"] = ev["dte"].apply(_bin)

        bins_out = {}
        # Non-event reference
        bins_out["non_event"] = {
            "n": int(len(ne)),
            "mean_p": float(ne_scores.mean()), "p50": float(np.median(ne_scores)),
            "p90": float(np.quantile(ne_scores, 0.9)),
            "frac_above_tau": float((ne_scores >= tau_a).mean()),
        }
        print(f"  {'bin':>10} {'n':>5} {'mean':>6} {'p50':>6} {'p90':>6} {'frac>=tau':>10} {'KS':>6} {'p_KS':>9}")
        b = bins_out["non_event"]
        print(f"  {'non_event':>10} {b['n']:>5d} {b['mean_p']:>6.3f} {b['p50']:>6.3f} {b['p90']:>6.3f} "
              f"{b['frac_above_tau']:>10.3f} {'-':>6} {'-':>9}")
        for bn in ["0_30", "30_60", "60_90", "90+"]:
            sub = ev[ev.bin == bn]
            if sub.empty:
                continue
            scores = sub["p_cal"].values.astype(float)
            ks_stat, p_val = ks_2samp(scores, ne_scores)
            bins_out[bn] = {
                "n": int(len(sub)),
                "mean_p": float(scores.mean()), "p50": float(np.median(scores)),
                "p90": float(np.quantile(scores, 0.9)),
                "frac_above_tau": float((scores >= tau_a).mean()),
                "KS": float(ks_stat), "p_KS": float(p_val),
            }
            b = bins_out[bn]
            print(f"  {bn:>10} {b['n']:>5d} {b['mean_p']:>6.3f} {b['p50']:>6.3f} {b['p90']:>6.3f} "
                  f"{b['frac_above_tau']:>10.3f} {b['KS']:>6.3f} {b['p_KS']:>9.2e}")
        out[split_name] = bins_out
    return out


def build_subset_tabular(samples: list[dict], channels: list[int], add_tpos: bool) -> np.ndarray:
    sub = [{**s, "X": s["X"][:, channels]} for s in samples]
    return build_tabular_from_samples(sub, add_tstar_position_feature=add_tpos)


def make_per_tstar_df(seas, nc_samples, probs, split_name):
    sy_meta = {(str(s["site_id"]), int(s["year"])): s for s in seas}
    rows = []
    for s, p in zip(nc_samples, probs):
        key = (str(s["site_id"]), int(s["year"]))
        meta = sy_meta[key]
        ctype = str(meta["censor_type"])
        rows.append({
            "split": split_name, "site": key[0], "year": key[1],
            "tstar": int(s["tstar"]), "p_cal": float(p),
            "y_event": int(s["y_event"]),
            "true_L": int(meta["L"]) if ctype != "right" else None,
            "true_R": int(meta["R"]) if ctype != "right" else None,
        })
    return pd.DataFrame(rows)


def matched_recall_alert(val_df: pd.DataFrame, test_df: pd.DataFrame, k: int,
                         target_recall: float) -> dict:
    tau_grid = np.arange(0.05, 0.96, 0.025)
    best = None
    best_FAR = float("inf")
    for tau in tau_grid:
        a_v = derive_alerts(val_df, float(tau), k_consecutive=k)
        m_v = metrics_from_alerts(a_v, "")
        if m_v["recall"] >= target_recall and m_v["FAR"] < best_FAR:
            best_FAR = m_v["FAR"]
            best = {"tau": float(tau), "val": m_v}
    if best is None:
        return {"matched": False}
    a_t = derive_alerts(test_df, best["tau"], k_consecutive=k)
    m_t = metrics_from_alerts(a_t, "")
    return {"matched": True, "tau": best["tau"], "val": best["val"], "test": m_t}


def section_C(cache: dict, hyper: dict, tau_a: float, k: int, target_recall: float) -> dict:
    print("\n========== [C] Shortcut diagnostic — feature-group retraining ==========")
    print(f"[hyper] {hyper}")
    train_now = cache["train_now"]; val_now = cache["val_now"]; test_now = cache["test_now"]
    val_seas = cache["val_seas"]; test_seas = cache["test_seas"]
    val_df_full = cache["val_df"]; test_df_full = cache["test_df"]
    y_tr = make_event_labels(train_now)

    out = {}
    for group_name, spec in GROUPS.items():
        print(f"\n  ----- {group_name} -----")
        print(f"  channels={spec['channels']}  add_tpos={spec['add_tpos']}  n_ch={len(spec['channels'])}")
        if group_name == "full":
            val_df = val_df_full; test_df = test_df_full
            # occurrence AUC on val + test using p_max from existing scores
            for split_name, df in [("val", val_df), ("test", test_df)]:
                agg = aggregate_scores(df)
                roc = roc_auc_score(agg["y_event"].values, agg["p_max"].values)
                print(f"  {split_name} occurrence ROC-AUC (p_max): {roc:.3f}")
                out.setdefault(group_name, {})[f"{split_name}_ROC_AUC"] = float(roc)
        else:
            X_tr = build_subset_tabular(train_now, spec["channels"], spec["add_tpos"])
            X_va = build_subset_tabular(val_now, spec["channels"], spec["add_tpos"])
            X_te = build_subset_tabular(test_now, spec["channels"], spec["add_tpos"])
            print(f"  X_tr={X_tr.shape} X_va={X_va.shape} X_te={X_te.shape}")
            clf = XGBClassifier(**hyper)
            clf.fit(X_tr, y_tr)
            p_va = clf.predict_proba(X_va)[:, 1]
            p_te = clf.predict_proba(X_te)[:, 1]
            val_df = make_per_tstar_df(val_seas, val_now, p_va, "val")
            test_df = make_per_tstar_df(test_seas, test_now, p_te, "test")
            for split_name, df in [("val", val_df), ("test", test_df)]:
                agg = aggregate_scores(df)
                roc = roc_auc_score(agg["y_event"].values, agg["p_max"].values)
                print(f"  {split_name} occurrence ROC-AUC (p_max): {roc:.3f}")
                out.setdefault(group_name, {})[f"{split_name}_ROC_AUC"] = float(roc)
        # alert metric at matched val_recall
        mr = matched_recall_alert(val_df, test_df, k=k, target_recall=target_recall)
        if not mr["matched"]:
            print(f"  alert@val_recall>={target_recall}: no qualifying tau on val")
            out.setdefault(group_name, {})["alert_match"] = None
        else:
            print(f"  alert@val_recall>={target_recall}  k={k}  tau*={mr['tau']:.3f}")
            print(f"    val:  recall={mr['val']['recall']:.3f}  FAR={mr['val']['FAR']:.3f}  "
                  f"precision={mr['val']['precision']:.3f}  F1={mr['val']['F1']:.3f}")
            print(f"    test: recall={mr['test']['recall']:.3f}  FAR={mr['test']['FAR']:.3f}  "
                  f"precision={mr['test']['precision']:.3f}  F1={mr['test']['F1']:.3f}")
            out.setdefault(group_name, {})["alert_match"] = {
                "tau": mr["tau"],
                "val": {k_: v_ for k_, v_ in mr["val"].items() if k_ != "label"},
                "test": {k_: v_ for k_, v_ in mr["test"].items() if k_ != "label"},
            }
    return out


def verdict_C(out_C: dict) -> str:
    lines = ["\n  ----- [C] Verdict -----"]
    rocs = {g: (out_C[g].get("test_ROC_AUC"), out_C[g].get("val_ROC_AUC")) for g in GROUPS}
    full_test = rocs["full"][0]
    cal_test = rocs["calendar_only"][0]
    weat_test = rocs["weather_only"][0]
    if full_test is None or cal_test is None or weat_test is None:
        return "  (insufficient data)"
    d_cal = (full_test or 0) - (cal_test or 0)
    d_weat = (full_test or 0) - (weat_test or 0)
    lines.append(f"  test occurrence ROC-AUC: full={full_test:.3f}  "
                 f"calendar_only={cal_test:.3f}  weather_only={weat_test:.3f}")
    lines.append(f"  full - calendar_only = {d_cal:+.3f}    full - weather_only = {d_weat:+.3f}")
    if d_cal <= 0.02:
        lines.append("  > shortcut dominates: calendar features (lat/lon/growing-stage/tstar_rel) explain almost all of full")
    elif d_cal <= 0.05:
        lines.append("  > calendar features are a strong baseline; weather adds modest signal")
    else:
        lines.append("  > weather signal contributes substantially (full > calendar_only by >0.05 AUC)")
    if weat_test >= cal_test + 0.03:
        lines.append("  > weather_only alone outperforms calendar_only -> there IS real weather signal")
    elif weat_test < cal_test - 0.03:
        lines.append("  > weather_only alone is weaker than calendar_only -> weather without calendar context is poor")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--tau_a", type=float, required=True)
    ap.add_argument("--k_consecutive", type=int, required=True)
    ap.add_argument("--target_recall", type=float, default=0.90,
                    help="recall target for [C] matched-recall alert comparison")
    ap.add_argument("--probs_cache", default="")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Load probs + samples
    cache = load_data(args)
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    C.DOY_START = int(ckpt.get("doy_start", 60))
    C.DOY_END = int(ckpt.get("doy_end", 300))
    val_df = cache["val_df"]; test_df = cache["test_df"]
    print(f"\n[cfg] tau_a={args.tau_a}  k={args.k_consecutive}  target_recall={args.target_recall}")
    print(f"[cohort] val_rows={len(val_df)}  test_rows={len(test_df)}")

    out = {"args": {"tau_a": args.tau_a, "k": args.k_consecutive, "target_recall": args.target_recall}}

    out["A"] = section_A(val_df, test_df, args.tau_a, args.k_consecutive)
    out["B"] = section_B(val_df, test_df, args.tau_a)

    # Reuse hyper-params from the existing full ckpt
    full_clf = ckpt["trained_states"][0]["sk_model"]
    hyper = {k: v for k, v in full_clf.get_params().items()
             if v is not None and k in {
                 "n_estimators", "max_depth", "learning_rate", "subsample",
                 "colsample_bytree", "reg_lambda", "min_child_weight", "gamma",
                 "random_state", "eval_metric", "scale_pos_weight",
                 "tree_method", "device", "objective",
             }}
    out["C"] = section_C(cache, hyper, args.tau_a, args.k_consecutive, args.target_recall)
    print(verdict_C(out["C"]))

    (out_dir / "occurrence_diag_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {out_dir / 'occurrence_diag_summary.json'}")


if __name__ == "__main__":
    main()
