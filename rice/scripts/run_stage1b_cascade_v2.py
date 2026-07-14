"""
Stage 1b cascade v2 — adds 14-day post-alert monitoring window features.

Workflow:
  - Reuse existing Stage 1 OOF train probs and val/test probs
  - Apply Stage 1a tau (default 0.51) to derive alerts
  - Build Stage 1b feature for each alerted site-year:
      * 211-dim base tabular feats from the nowcast sample at alert_tstar
      * Stage 1a OOF/test score at that row (1)
      * Monitoring window features from BASE sample X[tstar+1 : tstar+1+M]:
          - per-channel mean (D)
          - per-channel std (D)
          - per-channel slope (D)
          - per-channel max (D)
          - per-channel min (D)
          - per-channel change vs pre-window mean over [tstar-M+1 : tstar+1] (D)
        → 6 * D = 180 extra (D=30) plus 'monitor_used_days' scalar (1)
  - Drop site-years where monitoring window cannot be built (alert too late in season)
  - Train Stage 1b XGB (max_depth 4, n_est 200)
  - Sweep tau_b on val, evaluate on test
"""

from __future__ import annotations
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from rice.configs import config as C
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import build_tabular_from_samples, build_nowcast_samples
from rice.src.dataset import split_samples
from rice.src.pest_resolver import resolve_pest


def derive_alerts(probs_df, tau, k_consecutive=3):
    rows = []
    for (site, year), g in probs_df.groupby(["site", "year"], sort=False):
        g = g.sort_values("tstar")
        is_event = int(g.y_event.iloc[0])
        true_L = g.true_L.iloc[0] if pd.notna(g.true_L.iloc[0]) else None
        true_R = g.true_R.iloc[0] if pd.notna(g.true_R.iloc[0]) else None
        ps = g.p_cal.values
        ts = g.tstar.values.astype(int)
        alert_tstar = None; streak = 0
        for tstar, p in zip(ts, ps):
            if p >= tau:
                streak += 1
                if streak >= k_consecutive:
                    alert_tstar = int(tstar); break
            else:
                streak = 0
        rows.append({"site": str(site), "year": int(year), "is_event": is_event,
                     "alerted": int(alert_tstar is not None), "alert_tstar": alert_tstar,
                     "true_L": true_L, "true_R": true_R})
    return pd.DataFrame(rows)


def metrics_from_alerts(df, label="cascade"):
    n_event = int(df.is_event.sum())
    n_nonevent = len(df) - n_event
    n_alert = int(df.alerted.sum())
    tp = int(((df.is_event == 1) & (df.alerted == 1)).sum())
    fp = int(((df.is_event == 0) & (df.alerted == 1)).sum())
    fn = n_event - tp
    rec = tp / n_event if n_event else float("nan")
    prec = tp / n_alert if n_alert else float("nan")
    far = fp / n_nonevent if n_nonevent else float("nan")
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else float("nan")
    return {"label": label, "n": len(df), "n_event": n_event, "n_alert": n_alert,
            "tp": tp, "fp": fp, "fn": fn,
            "recall": rec, "precision": prec, "FAR": far, "F1": f1}


def lead_bin(lead):
    if lead is None or pd.isna(lead): return "no_event"
    lead = int(lead)
    if lead <= 0: return "lead_le0"
    if lead <= 14: return "lead_1_14"
    if lead <= 29: return "lead_15_29"
    if lead <= 45: return "lead_30_45"
    if lead <= 60: return "lead_46_60"
    if lead <= 75: return "lead_61_75"
    return "lead_gt75"


def build_monitoring_features(base_X, tstar, M=14):
    """
    base_X: (T, D) full-season feature array
    tstar: alert tstar (1-indexed in season units == python index since 0-based pyhton matches 1-based DOY-T?)
    Actually: tstar in nowcast samples is the python index (T-axis) that "ends" the lookback.
    Monitoring window = base_X[tstar : tstar+M]  (M days after the alert moment)
    Pre-window         = base_X[max(0, tstar-M) : tstar]
    Returns: feature vector of length 6*D + 1, plus 'days_used'.
    """
    T, D = base_X.shape
    end = min(T, tstar + M)
    monit = base_X[tstar:end]  # may be shorter than M near season end
    days_used = monit.shape[0]
    if days_used == 0:
        return None, 0
    # nan-safe stats
    mu = np.nanmean(monit, axis=0)
    sd = np.nanstd(monit, axis=0)
    mx = np.nanmax(monit, axis=0)
    mn = np.nanmin(monit, axis=0)
    # slope using least-squares vs day index in window
    t = np.arange(monit.shape[0], dtype=np.float32)
    t_c = t - t.mean()
    var_t = float((t_c ** 2).sum()) + 1e-8
    slope = ((monit - mu) * t_c[:, None]).sum(axis=0) / var_t
    # change vs pre-window mean
    pre_lo = max(0, tstar - M)
    pre = base_X[pre_lo:tstar]
    pre_mu = np.nanmean(pre, axis=0) if pre.shape[0] > 0 else mu
    change = mu - pre_mu
    f = np.concatenate([mu, sd, mx, mn, slope, change], axis=0).astype(np.float32)
    f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0)
    f = np.append(f, np.float32(days_used))
    return f, days_used


def build_features_for_alerts(
    nowcast_samples, base_samples_by_sy, sy_to_alert_tstar, score_map, monitor_M=14
):
    """
    Returns: X (n, D'+1), y, keys, dropped
    Uses nowcast row at alert_tstar for base 211 feats; base sample X for monitoring.
    Drops rows where monitoring window has 0 days or base sample missing.
    """
    matched_now = []
    matched_base = []
    matched_score = []
    keys = []
    dropped_no_base = 0
    dropped_no_monitor = 0

    for s in nowcast_samples:
        sy = (str(s["site_id"]), int(s["year"]))
        if sy not in sy_to_alert_tstar:
            continue
        if int(s["tstar"]) != int(sy_to_alert_tstar[sy]):
            continue
        if sy not in base_samples_by_sy:
            dropped_no_base += 1
            continue
        # Build monitoring features
        base_X = base_samples_by_sy[sy]["X"]
        mon, days = build_monitoring_features(base_X, int(s["tstar"]), M=monitor_M)
        if mon is None:
            dropped_no_monitor += 1
            continue
        matched_now.append(s)
        matched_base.append(mon)
        matched_score.append(float(score_map.get((sy[0], sy[1], int(s["tstar"])), float("nan"))))
        keys.append(sy)

    if not matched_now:
        return np.zeros((0, 0)), np.zeros((0,), dtype=int), [], {"dropped_no_base": dropped_no_base, "dropped_no_monitor": dropped_no_monitor}

    X_base = build_tabular_from_samples(matched_now, add_tstar_position_feature=True)
    X_score = np.asarray(matched_score, dtype=np.float32).reshape(-1, 1)
    X_mon = np.stack(matched_base, axis=0)
    X = np.concatenate([X_base, X_score, X_mon], axis=1)
    y = np.asarray([int(s.get("y_event", 0)) for s in matched_now], dtype=np.int64)
    info = {"dropped_no_base": dropped_no_base, "dropped_no_monitor": dropped_no_monitor,
            "monitor_dim": int(X_mon.shape[1]), "base_dim": int(X_base.shape[1])}
    return X, y, keys, info


def cascade_metrics_for_tau_b(alerts_a, keys_b, p_b, tau_b, label):
    keep = {keys_b[i] for i in range(len(p_b)) if p_b[i] >= tau_b}
    df = alerts_a.copy()
    df["alerted"] = df.apply(
        lambda r: int(r.alerted == 1 and (str(r.site), int(r.year)) in keep), axis=1
    )
    return metrics_from_alerts(df[["site","year","is_event","true_L","true_R","alert_tstar","alerted"]], label)


def lead_bin_breakdown(df):
    out = {}
    events = df[df.is_event == 1].copy()
    events["lead_calc"] = events.apply(
        lambda r: (int(r.true_L) - int(r.alert_tstar)) if (r.alerted == 1 and r.true_L is not None) else None,
        axis=1
    )
    for _, r in events.iterrows():
        b = "no_alert" if r.alerted == 0 else lead_bin(r.lead_calc)
        out.setdefault(b, {"n_event": 0, "n_alert": 0})
        out[b]["n_event"] += 1
        if r.alerted == 1:
            out[b]["n_alert"] += 1
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--split_seed", type=int, default=54)
    ap.add_argument("--split_mode", default="site_year")
    ap.add_argument("--ckpt_stage1", default="rice/outputs_stage1/sheath_blight_siteyear54/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split54_siteyear_ymin2002.pt")
    ap.add_argument("--oof_csv", default="rice/outputs_stage1/sheath_blight_siteyear54/ckpt/oof_train_seed54_5fold.csv")
    ap.add_argument("--probs_csv", default="rice/outputs_stage1/sheath_blight_siteyear54/eval/event_eval_sheath_blight_run4_stage1_xgb_nowcast_w28_s1_tpos_split54_siteyear_ymin2002_tauAlertF1_cap05_k3_ma1_nogate_abs_probs.csv")
    ap.add_argument("--tau_a", type=float, default=0.51)
    ap.add_argument("--tau_baseline", type=float, default=0.65)
    ap.add_argument("--monitor_M", type=int, default=14)
    ap.add_argument("--out_dir", default="rice/outputs_stage1/sheath_blight_siteyear54/cascade_v2")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--xgb_max_depth", type=int, default=4)
    ap.add_argument("--xgb_n_estimators", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ck = torch.load(args.ckpt_stage1, map_location="cpu", weights_only=False)
    C.DOY_START = int(ck["doy_start"])
    C.DOY_END = int(ck["doy_end"])
    nowcast_window = int(ck["nowcast_window"])
    nowcast_stride = int(ck["nowcast_stride"])
    nowcast_only_pre_event = bool(int(ck.get("nowcast_only_pre_event", 1)))
    nowcast_event_time_proxy = str(ck.get("nowcast_event_time_proxy", "mid"))
    nowcast_label_mode = str(ck.get("nowcast_label_mode", "eventually"))
    nowcast_label_horizon = ck.get("nowcast_label_horizon", None)
    nowcast_tstar_start = ck.get("nowcast_tstar_start", None)
    print(f"[cfg] tau_a={args.tau_a} monitor_M={args.monitor_M} seed={args.seed}")

    _, get_feature_cols = resolve_pest(args.pest)
    feature_cols, feature_names, T, samples = build_samples_for_run(int(args.run), get_feature_cols)
    train_s, val_s, test_s = split_samples(samples, val_frac=0.1, test_frac=0.1,
                                            seed=int(args.split_seed), split_mode=args.split_mode)
    print(f"[split] train={len(train_s)} val={len(val_s)} test={len(test_s)}")

    # Build base-sample lookups (full-season X for monitoring slicing)
    base_train = {(str(s["site_id"]), int(s["year"])): s for s in train_s}
    base_val   = {(str(s["site_id"]), int(s["year"])): s for s in val_s}
    base_test  = {(str(s["site_id"]), int(s["year"])): s for s in test_s}

    nowcast_kwargs = dict(
        window=nowcast_window, stride=nowcast_stride,
        tstar_start=nowcast_tstar_start, only_pre_event=nowcast_only_pre_event,
        event_time_proxy=nowcast_event_time_proxy,
        label_mode=nowcast_label_mode, label_horizon=nowcast_label_horizon,
    )
    train_now = build_nowcast_samples(train_s, **nowcast_kwargs)
    val_now = build_nowcast_samples(val_s, **nowcast_kwargs)
    test_now = build_nowcast_samples(test_s, **nowcast_kwargs)
    print(f"[nowcast] train={len(train_now)} val={len(val_now)} test={len(test_now)}")

    # OOF Stage 1a alerts
    oof = pd.read_csv(args.oof_csv)
    oof_alerts = derive_alerts(oof, args.tau_a)
    print(f"[oof] Stage 1a metrics: {metrics_from_alerts(oof_alerts, 'oof_train Stage1a')}")
    oof_score_map = {(str(r.site), int(r.year), int(r.tstar)): float(r.p_cal) for r in oof.itertuples(index=False)}
    sy_alert_train = {(r.site, r.year): int(r.alert_tstar) for _, r in oof_alerts.iterrows() if r.alerted == 1}
    print(f"[train Stage1b] alerted site-years: {len(sy_alert_train)}")

    X_tr, y_tr, keys_tr, info_tr = build_features_for_alerts(
        train_now, base_train, sy_alert_train, oof_score_map, monitor_M=args.monitor_M
    )
    print(f"[train Stage1b] X={X_tr.shape} pos rate={y_tr.mean():.3f}  base_dim={info_tr['base_dim']} monitor_dim={info_tr['monitor_dim']}")
    print(f"  dropped_no_base={info_tr['dropped_no_base']} dropped_no_monitor={info_tr['dropped_no_monitor']}")

    # Production probs for val/test
    probs = pd.read_csv(args.probs_csv)
    val_p = probs[(probs.split == "val") & (probs.seed == int(args.seed))]
    test_p = probs[(probs.split == "test") & (probs.seed == int(args.seed))]
    val_alerts_a = derive_alerts(val_p, args.tau_a)
    test_alerts_a = derive_alerts(test_p, args.tau_a)
    print(f"[val   Stage1a] {metrics_from_alerts(val_alerts_a, 'val Stage1a')}")
    print(f"[test  Stage1a] {metrics_from_alerts(test_alerts_a, 'test Stage1a')}")

    val_score_map = {(str(r.site), int(r.year), int(r.tstar)): float(r.p_cal) for r in val_p.itertuples(index=False)}
    test_score_map = {(str(r.site), int(r.year), int(r.tstar)): float(r.p_cal) for r in test_p.itertuples(index=False)}
    sy_alert_val = {(r.site, r.year): int(r.alert_tstar) for _, r in val_alerts_a.iterrows() if r.alerted == 1}
    sy_alert_test = {(r.site, r.year): int(r.alert_tstar) for _, r in test_alerts_a.iterrows() if r.alerted == 1}

    X_va, y_va, keys_va, info_va = build_features_for_alerts(
        val_now, base_val, sy_alert_val, val_score_map, monitor_M=args.monitor_M
    )
    X_te, y_te, keys_te, info_te = build_features_for_alerts(
        test_now, base_test, sy_alert_test, test_score_map, monitor_M=args.monitor_M
    )
    print(f"[val   Stage1b] X={X_va.shape} pos rate={y_va.mean():.3f}  dropped={info_va}")
    print(f"[test  Stage1b] X={X_te.shape} pos rate={y_te.mean():.3f}  dropped={info_te}")

    # Train Stage 1b
    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"\n[Stage 1b] pos_weight={pos_w:.3f} max_depth={args.xgb_max_depth} n_est={args.xgb_n_estimators}")
    clf = XGBClassifier(
        n_estimators=int(args.xgb_n_estimators), max_depth=int(args.xgb_max_depth),
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=0, eval_metric="logloss", scale_pos_weight=float(pos_w),
        tree_method="hist", device="cuda",
    )
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    print(f"[Stage 1b] trained in {time.perf_counter()-t0:.1f}s")
    p_va = clf.predict_proba(X_va)[:, 1]
    p_te = clf.predict_proba(X_te)[:, 1]
    print(f"[Stage 1b] val AUC={roc_auc_score(y_va, p_va):.3f} | test AUC={roc_auc_score(y_te, p_te):.3f}")

    # Note on dropped alerts: they are excluded from cascade evaluation, treated as "Stage 1a alert that
    # cascade cannot evaluate". For fair comparison, dropped alerts pass through unchanged.
    dropped_val = set(sy_alert_val.keys()) - set(keys_va)
    dropped_test = set(sy_alert_test.keys()) - set(keys_te)
    print(f"[fairness] alerts not evaluated by Stage 1b (kept as alerts in cascade): val={len(dropped_val)} test={len(dropped_test)}")

    def cascade_metrics_with_passthrough(alerts_a, keys_b, p_b, tau_b, dropped_set, label):
        keep = {keys_b[i] for i in range(len(p_b)) if p_b[i] >= tau_b}
        df = alerts_a.copy()
        def newalert(r):
            if r.alerted != 1: return 0
            sy = (str(r.site), int(r.year))
            if sy in dropped_set: return 1  # passthrough: keep alert
            return int(sy in keep)
        df["alerted"] = df.apply(newalert, axis=1)
        return metrics_from_alerts(df[["site","year","is_event","true_L","true_R","alert_tstar","alerted"]], label)

    # Val sweep for tau_b (max F1)
    print("\n" + "="*90)
    print("Stage 1b val tau_b sweep")
    print("="*90)
    print(f"  {'tau_b':>6} {'recall':>7} {'precision':>10} {'FAR':>7} {'F1':>6} {'n_alert':>8}")
    best_tau_b = None; best_f1 = -1; best_metrics = None
    for tau_b in [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]:
        m = cascade_metrics_with_passthrough(val_alerts_a, keys_va, p_va, tau_b, dropped_val, f"val tau_b={tau_b}")
        print(f"  {tau_b:>6.2f} {m['recall']:>7.3f} {m['precision']:>10.3f} {m['FAR']:>7.3f} {m['F1']:>6.3f} {m['n_alert']:>8}")
        if m["F1"] > best_f1:
            best_f1 = m["F1"]; best_tau_b = tau_b; best_metrics = m
    print(f"\n  best tau_b (max F1): {best_tau_b}")

    # Test sweep + comparison
    print("\n" + "="*90)
    print("Step 4: TEST cascade-v2 (with monitoring) vs prior runs")
    print("="*90)
    test_baseline = metrics_from_alerts(derive_alerts(test_p, args.tau_baseline), "baseline tau=0.65")
    test_a_only = metrics_from_alerts(test_alerts_a, "Stage 1a only tau=0.51")
    print(f"  baseline (tau=0.65): {test_baseline}")
    print(f"  Stage 1a only (tau=0.51): {test_a_only}")
    print(f"\n  Cascade v2 (with monitoring) — TEST sweep:")
    print(f"  {'tau_b':>6} {'recall':>7} {'precision':>10} {'FAR':>7} {'F1':>6} {'n_alert':>8}")
    cascade_results = []
    for tau_b in [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]:
        m = cascade_metrics_with_passthrough(test_alerts_a, keys_te, p_te, tau_b, dropped_test, f"test tau_b={tau_b}")
        cascade_results.append((tau_b, m))
        marker = " ← best_val_tau_b" if abs(tau_b - best_tau_b) < 1e-6 else ""
        print(f"  {tau_b:>6.2f} {m['recall']:>7.3f} {m['precision']:>10.3f} {m['FAR']:>7.3f} {m['F1']:>6.3f} {m['n_alert']:>8}{marker}")

    summary = {
        "monitor_M": args.monitor_M, "tau_a": args.tau_a, "tau_baseline": args.tau_baseline,
        "best_tau_b_val_f1": best_tau_b,
        "stage1b_val_auc": float(roc_auc_score(y_va, p_va)),
        "stage1b_test_auc": float(roc_auc_score(y_te, p_te)),
        "test_baseline": test_baseline, "test_stage1a_only": test_a_only,
        "test_cascade_sweep": [{"tau_b": t, **m} for t, m in cascade_results],
        "feature_dims": {"base": int(info_tr["base_dim"]), "monitor": int(info_tr["monitor_dim"])},
        "dropped": {"train": info_tr, "val": info_va, "test": info_te},
        "passthrough_alerts": {"val": len(dropped_val), "test": len(dropped_test)},
    }
    (out_dir / "cascade_v2_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[done] saved {out_dir / 'cascade_v2_summary.json'}")

    # Lead-bin breakdown for best
    print("\n" + "="*90)
    print("LEAD BIN BREAKDOWN (test) — alerted/total events per bin")
    print("="*90)
    keep = {keys_te[i] for i in range(len(p_te)) if p_te[i] >= best_tau_b}
    cascade_alerts = test_alerts_a.copy()
    def newalert(r):
        if r.alerted != 1: return 0
        sy = (str(r.site), int(r.year))
        if sy in dropped_test: return 1
        return int(sy in keep)
    cascade_alerts["alerted"] = cascade_alerts.apply(newalert, axis=1)
    cascade_alerts_path = out_dir / "cascade_alerts_test.csv"
    cascade_alerts.to_csv(cascade_alerts_path, index=False)
    print(f"[saved] {cascade_alerts_path}")
    bins_b = lead_bin_breakdown(derive_alerts(test_p, args.tau_baseline))
    bins_a = lead_bin_breakdown(test_alerts_a)
    bins_c = lead_bin_breakdown(cascade_alerts)
    print(f"\n  {'lead_bin':>12}  {'baseline':>12}  {'Stage1a':>12}  {'CascadeV2':>12}")
    all_bins = sorted(set(bins_b) | set(bins_a) | set(bins_c))
    for b in ["lead_1_14","lead_15_29","lead_30_45","lead_46_60","lead_61_75","lead_gt75","no_alert"]:
        if b not in all_bins: continue
        bb = bins_b.get(b, {"n_event":0,"n_alert":0})
        ba = bins_a.get(b, {"n_event":0,"n_alert":0})
        bc = bins_c.get(b, {"n_event":0,"n_alert":0})
        print(f"  {b:>12}  {bb['n_alert']:>3}/{bb['n_event']:<6}  {ba['n_alert']:>3}/{ba['n_event']:<6}  {bc['n_alert']:>3}/{bc['n_event']:<6}")


if __name__ == "__main__":
    main()
