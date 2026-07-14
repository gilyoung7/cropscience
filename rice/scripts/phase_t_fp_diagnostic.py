"""
Phase T3 — FP/FN diagnostic for Stage 1a no-cascade operating point.

Fixed operating point (no-cascade): k_consecutive, tau_a.
Cohort: val + test (yearsplit). Reuses probs cache from phase_t_stage1b_cascade.

Per split, classifies each site-year as TP / FP / FN / TN and reports:
  1. Confusion counts + rates
  2. FP distribution: alert_tstar DOY hist, year split, score-shape stats
  3. FN distribution: peak score, days_above_tau, year/DOY of L
  4. TP vs FP separability:
       - per-feature AUC and KS (val + test)
       - multivariate AUC via logistic-regression trained on val, applied to test
  5. DOY-binned alert rate + per-bin score mean (dynamic threshold scoping)
  6. Verdict line

Rule for tau_b/operating-point selection:
  - All decisions on val. Test is diagnostic only — reported alongside but not used to choose.

No training. Single-run inference on cached probs.
"""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import build_nowcast_samples, build_tabular_from_samples
from rice.scripts.run_calibrate_event import apply_temperature, fit_temperature_grid


SHAPE_FEATURE_COLS = [
    "score_at_alert", "score_peak", "score_peak_pre",
    "score_slope_14d", "area_above_tau", "days_above_tau",
    "doy_alert", "first_above_tau_doy", "longest_run_above_tau",
]


def load_or_build_probs(args) -> dict:
    """Reuse phase_t_stage1b_cascade pickle cache if present; else rebuild val/test only."""
    if args.probs_cache and Path(args.probs_cache).exists():
        with open(args.probs_cache, "rb") as f:
            cache = pickle.load(f)
        print(f"[cache] loaded {args.probs_cache}  keys={list(cache.keys())[:8]}")
        return cache

    print(f"[cache miss] rebuilding val/test probs (no train OOF)")
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
    _, val_seas, test_seas = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )
    nc_kw = dict(window=nc_window, stride=nc_stride, tstar_start=nc_tstart,
                 only_pre_event=nc_only_pre, event_time_proxy=nc_proxy,
                 label_mode=nc_label_mode, label_horizon=nc_label_horizon)
    val_now = build_nowcast_samples(val_seas, **nc_kw)
    test_now = build_nowcast_samples(test_seas, **nc_kw)
    X_val = build_tabular_from_samples(val_now, add_tstar_position_feature=add_tpos)
    X_test = build_tabular_from_samples(test_now, add_tstar_position_feature=add_tpos)
    y_val = np.asarray([int(s["y_event"]) for s in val_now])

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

    return {
        "t_best": float(t_best),
        "val_df": _df(val_seas, val_now, p_val_cal, "val"),
        "test_df": _df(test_seas, test_now, p_test_cal, "test"),
        "val_seas": val_seas, "test_seas": test_seas,
    }


def classify_sites(probs_df: pd.DataFrame, tau: float, k: int) -> pd.DataFrame:
    """Per-(site,year) classification + score-shape features. is_event from y_event."""
    rows = []
    for (site, year), g in probs_df.groupby(["site", "year"], sort=False):
        g = g.sort_values("tstar")
        is_event = int(g.y_event.iloc[0])
        ts = g.tstar.values.astype(int)
        ps = g.p_cal.values.astype(float)
        true_L = g.true_L.iloc[0]
        true_R = g.true_R.iloc[0]

        # k-consecutive alert
        alert_tstar = None
        streak = 0
        for tstar, p in zip(ts, ps):
            if p >= tau:
                streak += 1
                if streak >= k:
                    alert_tstar = int(tstar)
                    break
            else:
                streak = 0
        alerted = int(alert_tstar is not None)

        # Score-shape features (computed even for non-alerted, useful for FN analysis)
        above = ps >= tau
        days_above_tau = int(above.sum())
        area_above_tau = float(np.clip(ps - tau, 0, None).sum())
        score_peak = float(ps.max()) if len(ps) else float("nan")

        if alert_tstar is not None:
            pre_mask = ts <= alert_tstar
            score_peak_pre = float(ps[pre_mask].max()) if pre_mask.any() else float("nan")
            score_at_alert = float(ps[ts == alert_tstar][0])
            # last 14 tstars up to and including alert
            window_mask = (ts <= alert_tstar) & (ts > alert_tstar - 14)
            if window_mask.sum() >= 2:
                tt = ts[window_mask].astype(float)
                tt_c = tt - tt.mean()
                pp = ps[window_mask]
                pp_c = pp - pp.mean()
                var_t = float((tt_c ** 2).sum()) + 1e-8
                score_slope_14d = float((pp_c * tt_c).sum() / var_t)
            else:
                score_slope_14d = float("nan")
            first_above = ts[above]
            first_above_tstar = int(first_above.min()) if first_above.size else -1
            doy_alert = int(C.DOY_START + alert_tstar)
            first_above_doy = int(C.DOY_START + first_above_tstar) if first_above_tstar >= 0 else -1
        else:
            score_peak_pre = float("nan")
            score_at_alert = float("nan")
            score_slope_14d = float("nan")
            doy_alert = -1
            first_above_doy = -1

        # Longest run of consecutive above-tau
        longest_run = 0
        cur = 0
        for v in above:
            if v:
                cur += 1
                longest_run = max(longest_run, cur)
            else:
                cur = 0

        # lead_days = L - alert_tstar (for interval events that were alerted)
        if is_event == 1 and pd.notna(true_L) and alert_tstar is not None:
            lead_days = int(true_L) - int(alert_tstar)
        else:
            lead_days = None

        # Classification
        if is_event == 1 and alerted == 1:
            cls = "TP"
        elif is_event == 0 and alerted == 1:
            cls = "FP"
        elif is_event == 1 and alerted == 0:
            cls = "FN"
        else:
            cls = "TN"

        rows.append({
            "site": str(site), "year": int(year), "is_event": is_event,
            "alerted": alerted, "cls": cls,
            "alert_tstar": alert_tstar, "doy_alert": doy_alert,
            "true_L": (int(true_L) if pd.notna(true_L) else None),
            "true_R": (int(true_R) if pd.notna(true_R) else None),
            "lead_days": lead_days,
            "score_at_alert": score_at_alert,
            "score_peak": score_peak, "score_peak_pre": score_peak_pre,
            "score_slope_14d": score_slope_14d,
            "area_above_tau": area_above_tau,
            "days_above_tau": days_above_tau,
            "first_above_tau_doy": first_above_doy,
            "longest_run_above_tau": longest_run,
        })
    return pd.DataFrame(rows)


def print_confusion(df: pd.DataFrame, label: str) -> dict:
    n = len(df)
    counts = Counter(df["cls"])
    tp, fp, fn, tn = counts.get("TP", 0), counts.get("FP", 0), counts.get("FN", 0), counts.get("TN", 0)
    n_event = tp + fn
    n_nonevent = fp + tn
    recall = tp / n_event if n_event else float("nan")
    far = fp / n_nonevent if n_nonevent else float("nan")
    precision = tp / (tp + fp) if (tp + fp) else float("nan")
    f1 = (2 * tp) / (2 * tp + fp + fn) if (2 * tp + fp + fn) else float("nan")
    print(f"\n[{label}] n={n}  n_event={n_event}  n_nonevent={n_nonevent}")
    print(f"   TP={tp}  FP={fp}  FN={fn}  TN={tn}")
    print(f"   recall={recall:.3f}  FAR={far:.3f}  precision={precision:.3f}  F1={f1:.3f}")
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall": recall, "FAR": far, "precision": precision, "F1": f1}


def doy_hist(values: list[int], lo: int = 60, hi: int = 300, bw: int = 20) -> str:
    if not values:
        return "(empty)"
    bins = list(range(lo, hi + 1, bw))
    counts, edges = np.histogram(values, bins=bins)
    lines = []
    for i, c in enumerate(counts):
        bar = "#" * int(c)
        lines.append(f"    DOY {edges[i]:>3d}-{edges[i+1]-1:>3d}: {c:>3d}  {bar}")
    return "\n".join(lines)


def print_fp_distribution(df: pd.DataFrame, label: str) -> None:
    fp = df[df.cls == "FP"]
    print(f"\n  ----- FP distribution ({label}) -----")
    print(f"  n_FP = {len(fp)}")
    if len(fp) == 0:
        return
    print(f"  by year:")
    for y, c in fp.year.value_counts().sort_index().items():
        print(f"    {int(y)}: {int(c)}")
    print(f"  alert_tstar DOY hist:")
    print(doy_hist(fp.doy_alert.tolist()))
    cols = ["score_at_alert", "score_peak", "score_peak_pre",
            "score_slope_14d", "area_above_tau", "days_above_tau", "longest_run_above_tau"]
    print(f"  FP score-shape (mean | std | median):")
    for c in cols:
        v = fp[c].dropna().values
        if len(v) == 0:
            print(f"    {c:>22}: (no data)")
            continue
        print(f"    {c:>22}: {v.mean():>6.3f} | {v.std():>6.3f} | {np.median(v):>6.3f}")


def print_fn_distribution(df: pd.DataFrame, label: str) -> None:
    fn = df[df.cls == "FN"]
    print(f"\n  ----- FN distribution ({label}) -----")
    print(f"  n_FN = {len(fn)}")
    if len(fn) == 0:
        return
    print(f"  by year:")
    for y, c in fn.year.value_counts().sort_index().items():
        print(f"    {int(y)}: {int(c)}")
    if fn.true_L.notna().any():
        L_doy = (fn.true_L.dropna().astype(int) + int(C.DOY_START)).tolist()
        print(f"  true L DOY (event start) hist:")
        print(doy_hist(L_doy))
    print(f"  near-miss vs deep-miss (score_peak):")
    near_miss = int((fn.score_peak >= 0.40).sum())
    very_near = int((fn.score_peak >= 0.55).sum())
    deep_miss = int((fn.score_peak < 0.25).sum())
    print(f"    score_peak >= 0.55 (nearly alerted): {very_near} / {len(fn)}")
    print(f"    score_peak >= 0.40              : {near_miss} / {len(fn)}")
    print(f"    score_peak <  0.25 (deep miss)  : {deep_miss} / {len(fn)}")
    print(f"  days_above_tau (count of t* >= tau but k=3 not satisfied):")
    print(f"    mean={fn.days_above_tau.mean():.2f}  max={int(fn.days_above_tau.max())}  >0: {int((fn.days_above_tau>0).sum())}/{len(fn)}")


def separability_table(df: pd.DataFrame, label: str) -> dict:
    """Per-feature AUC + KS for TP vs FP (alerted site-years only)."""
    alerted = df[df.alerted == 1].copy()
    tp = alerted[alerted.cls == "TP"]
    fp = alerted[alerted.cls == "FP"]
    print(f"\n  ----- TP vs FP separability ({label}) -----")
    print(f"  n_TP={len(tp)}  n_FP={len(fp)}")
    if len(tp) < 5 or len(fp) < 5:
        print("  (skip: too few of one class)")
        return {}
    y = (alerted.cls == "TP").astype(int).values
    out = {}
    print(f"  {'feature':>22}  {'AUC':>6}  {'KS':>6}  {'p_KS':>8}")
    for col in SHAPE_FEATURE_COLS:
        if col not in alerted.columns:
            continue
        x = alerted[col].astype(float).fillna(alerted[col].mean()).values
        if np.unique(x).size < 2:
            continue
        try:
            auc = roc_auc_score(y, x)
            auc = max(auc, 1 - auc)  # direction-agnostic
        except ValueError:
            auc = float("nan")
        ks_res = ks_2samp(tp[col].dropna(), fp[col].dropna())
        out[col] = {"AUC": float(auc), "KS": float(ks_res.statistic), "p_KS": float(ks_res.pvalue)}
        print(f"  {col:>22}  {auc:>6.3f}  {ks_res.statistic:>6.3f}  {ks_res.pvalue:>8.1e}")
    return out


def multivar_separability(val_df: pd.DataFrame, test_df: pd.DataFrame) -> dict:
    """Train logistic regression on val (TP=1, FP=0), report val + test AUC."""
    val_a = val_df[val_df.alerted == 1].copy()
    tst_a = test_df[test_df.alerted == 1].copy()
    if val_a.cls.nunique() < 2 or len(val_a) < 10:
        print("\n  (skip multivariate: val has <2 classes among alerted)")
        return {}
    feats = [c for c in SHAPE_FEATURE_COLS if c in val_a.columns]
    Xv = val_a[feats].astype(float).fillna(val_a[feats].mean()).values
    yv = (val_a.cls == "TP").astype(int).values
    Xt = tst_a[feats].astype(float).fillna(val_a[feats].mean()).values
    yt = (tst_a.cls == "TP").astype(int).values if tst_a.cls.nunique() >= 2 else None
    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(Xv, yv)
    pv = model.predict_proba(Xv)[:, 1]
    pt = model.predict_proba(Xt)[:, 1] if len(Xt) else None
    val_auc = roc_auc_score(yv, pv)
    test_auc = roc_auc_score(yt, pt) if yt is not None and len(set(yt.tolist())) > 1 else float("nan")
    print(f"\n  ----- Multivariate TP/FP discrimination (LogReg on val features) -----")
    print(f"  features used: {feats}")
    print(f"  val_AUC = {val_auc:.3f}    test_AUC = {test_auc:.3f}")
    coefs = sorted(zip(feats, model.coef_[0].tolist()), key=lambda x: -abs(x[1]))
    print(f"  top |coef|:")
    for f, c in coefs[:6]:
        print(f"    {f:>22}: {c:+.3f}")
    return {"val_AUC": float(val_auc), "test_AUC": float(test_auc),
            "coefs": dict(coefs)}


def doy_bin_analysis(val_df: pd.DataFrame, test_df: pd.DataFrame, probs_val: pd.DataFrame, probs_test: pd.DataFrame,
                     tau: float) -> dict:
    """DOY-binned alert rate (val + test) and mean score per bin (dynamic threshold scoping)."""
    BIN_EDGES = [60, 100, 140, 180, 220, 260, 301]
    def _bin(doy):
        for i in range(len(BIN_EDGES) - 1):
            if BIN_EDGES[i] <= doy < BIN_EDGES[i+1]:
                return f"DOY[{BIN_EDGES[i]:>3d},{BIN_EDGES[i+1]:>3d})"
        return "other"

    def per_split(name, df, probs):
        alerted = df[df.alerted == 1].copy()
        alerted["bin"] = alerted["doy_alert"].apply(_bin)
        events_by_bin = df[df.is_event == 1].copy()
        events_by_bin["doy_L"] = events_by_bin["true_L"].apply(lambda v: int(v) + int(C.DOY_START) if pd.notna(v) else None)
        events_by_bin["bin"] = events_by_bin["doy_L"].apply(lambda v: _bin(v) if v is not None else "other")
        probs = probs.copy()
        probs["doy"] = probs["tstar"].astype(int) + int(C.DOY_START)
        probs["bin"] = probs["doy"].apply(_bin)
        score_mean = probs.groupby("bin")["p_cal"].mean()
        score_q90 = probs.groupby("bin")["p_cal"].quantile(0.9)
        print(f"\n  [{name}] DOY-binned breakdown (tau={tau:.3f}):")
        print(f"  {'bin':>20}  {'mean_p':>6} {'q90_p':>6} | "
              f"{'TP':>3} {'FP':>3} {'FN':>3} | {'alert_rate':>10} {'precision':>9}")
        out = {}
        for b in [_bin(d) for d in [80, 120, 160, 200, 240, 280]]:
            n_alert = int((alerted.bin == b).sum())
            n_TP = int(((alerted.bin == b) & (alerted.cls == "TP")).sum())
            n_FP = int(((alerted.bin == b) & (alerted.cls == "FP")).sum())
            n_FN = int(((events_by_bin.bin == b) & (events_by_bin.alerted == 0)).sum())
            mp = float(score_mean.get(b, float("nan")))
            qp = float(score_q90.get(b, float("nan")))
            precision = n_TP / n_alert if n_alert > 0 else float("nan")
            print(f"  {b:>20}  {mp:>6.3f} {qp:>6.3f} | "
                  f"{n_TP:>3d} {n_FP:>3d} {n_FN:>3d} | {n_alert:>10d} {precision:>9.3f}")
            out[b] = {"n_TP": n_TP, "n_FP": n_FP, "n_FN": n_FN, "n_alert": n_alert,
                      "mean_p": mp, "q90_p": qp, "precision": precision}
        return out

    val_out = per_split("val", val_df, probs_val)
    test_out = per_split("test", test_df, probs_test)
    return {"val": val_out, "test": test_out}


def verdict(val_sep: dict, mvar: dict, doy_bins: dict) -> str:
    """One-line judgment based on separability + DOY concentration."""
    # Multivariate AUC tells us "feature richness given alerted set"
    val_auc = mvar.get("val_AUC", float("nan"))
    test_auc = mvar.get("test_AUC", float("nan"))
    if not np.isnan(test_auc) and test_auc >= 0.70:
        sep_msg = f"Separable on test (LogReg AUC={test_auc:.3f}>=0.70). Stage 1b cascade has room."
    elif not np.isnan(test_auc) and test_auc >= 0.60:
        sep_msg = f"Marginal separability (test AUC={test_auc:.3f}). Stage 1b will help modestly."
    elif not np.isnan(test_auc):
        sep_msg = f"Hard to separate FP from TP (test AUC={test_auc:.3f}). Data ceiling likely."
    else:
        sep_msg = "Insufficient data for multivariate AUC."
    # DOY concentration
    fp_by_bin_val = {b: v["n_FP"] for b, v in doy_bins["val"].items()}
    total_fp_val = sum(fp_by_bin_val.values())
    max_share_val = (max(fp_by_bin_val.values()) / total_fp_val) if total_fp_val else 0.0
    fp_by_bin_test = {b: v["n_FP"] for b, v in doy_bins["test"].items()}
    total_fp_test = sum(fp_by_bin_test.values())
    max_share_test = (max(fp_by_bin_test.values()) / total_fp_test) if total_fp_test else 0.0
    if max_share_val >= 0.50:
        doy_msg = (f"FP concentrates in one DOY bin on val ({max_share_val:.0%}) "
                   f"(test {max_share_test:.0%}) -> dynamic-threshold candidate.")
    elif max_share_val >= 0.35:
        doy_msg = (f"FP partially concentrated on val ({max_share_val:.0%}) "
                   f"(test {max_share_test:.0%}) -> mild dynamic-threshold benefit.")
    else:
        doy_msg = (f"FP spread across DOY (val max bin {max_share_val:.0%}) "
                   f"-> dynamic threshold unlikely to help much.")
    return f"  > Separability: {sep_msg}\n  > DOY pattern:  {doy_msg}"


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
    ap.add_argument("--probs_cache", default="")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    cache = load_or_build_probs(args)
    # Make sure DOY range is set (for tstar->DOY mapping)
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    C.DOY_START = int(ckpt.get("doy_start", 60))
    C.DOY_END = int(ckpt.get("doy_end", 300))

    val_df = cache["val_df"]; test_df = cache["test_df"]
    print(f"\n========== Phase T3 FP diagnostic ==========")
    print(f"[cfg] tau_a={args.tau_a} k_consecutive={args.k_consecutive}")
    print(f"[cohort] val_rows={len(val_df)} test_rows={len(test_df)}  "
          f"DOY_START={C.DOY_START} DOY_END={C.DOY_END}")

    val_cls = classify_sites(val_df, args.tau_a, args.k_consecutive)
    test_cls = classify_sites(test_df, args.tau_a, args.k_consecutive)

    # 1. Confusion
    print(f"\n----- 1. Operating point + Confusion -----")
    val_conf = print_confusion(val_cls, "val")
    test_conf = print_confusion(test_cls, "test")

    # 2. FP distribution
    print(f"\n----- 2. FP distribution -----")
    print_fp_distribution(val_cls, "val")
    print_fp_distribution(test_cls, "test")

    # 4. FN distribution
    print(f"\n----- 4. FN distribution -----")
    print_fn_distribution(val_cls, "val")
    print_fn_distribution(test_cls, "test")

    # 3. TP vs FP separability
    print(f"\n----- 3. TP vs FP separability -----")
    val_sep = separability_table(val_cls, "val")
    test_sep = separability_table(test_cls, "test")
    mvar = multivar_separability(val_cls, test_cls)

    # 5. DOY-binned alert rate
    print(f"\n----- 5. DOY-binned alert rate (dynamic threshold scoping) -----")
    doy_bins = doy_bin_analysis(val_cls, test_cls, val_df, test_df, args.tau_a)

    # 6. Verdict
    print(f"\n----- 6. Verdict -----")
    print(verdict(val_sep, mvar, doy_bins))

    # Save per-site CSV + summary JSON
    val_cls["split"] = "val"
    test_cls["split"] = "test"
    pd.concat([val_cls, test_cls], ignore_index=True).to_csv(
        out_dir / f"fp_diag_classified_tau{args.tau_a}_k{args.k_consecutive}.csv", index=False)
    summary = {
        "tau_a": args.tau_a, "k_consecutive": args.k_consecutive,
        "val_confusion": val_conf, "test_confusion": test_conf,
        "val_per_feature_sep": val_sep, "test_per_feature_sep": test_sep,
        "multivar": mvar,
        "doy_bins": doy_bins,
    }
    (out_dir / f"fp_diag_summary_tau{args.tau_a}_k{args.k_consecutive}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"\n[saved] {out_dir / f'fp_diag_classified_tau{args.tau_a}_k{args.k_consecutive}.csv'}")
    print(f"[saved] {out_dir / f'fp_diag_summary_tau{args.tau_a}_k{args.k_consecutive}.json'}")


if __name__ == "__main__":
    main()
