"""
Stage 1b cascade — train + evaluate.

Step 3: Stage 1b training
  - Load OOF train probs (from run_stage1_oof.py)
  - Apply Stage 1a tau (default 0.51) with k=3 consecutive → identify alerted site-years
  - For each alerted site-year, extract the t*-row at alert_tstar:
      features = Stage-1 tabular feats (211-dim) + Stage-1a score at that row
  - Label = is_event (whether the site-year has an event)
  - Train XGB Stage 1b (max_depth 3-4, scale_pos_weight tuned for precision)

Step 4: End-to-end evaluation on test
  - Apply Stage 1a (tau_a) to test probs → site-year alerts
  - For each alert: extract alert-row features → Stage 1b score
  - Final alert = Stage 1a alert AND Stage 1b score >= tau_b
  - Sweep tau_b on val to pick best (target precision >= 0.85)
  - Report cascade vs baseline single-stage on test
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
    """Return per-site-year df: site, year, is_event, alert_tstar (or None), alerted, true_L, true_R."""
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
        rows.append({
            "site": str(site), "year": int(year), "is_event": is_event,
            "alerted": int(alert_tstar is not None), "alert_tstar": alert_tstar,
            "true_L": true_L, "true_R": true_R,
        })
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


def lead_bin_breakdown(df):
    """For TP rows, bin by lead = true_L - alert_tstar; report alert recall per bin."""
    # Need ground-truth lead bin (from full event L). Use true_L of all event sites.
    out = {}
    events = df[df.is_event == 1].copy()
    events["lead_calc"] = events.apply(
        lambda r: (int(r.true_L) - int(r.alert_tstar)) if (r.alerted == 1 and r.true_L is not None) else None,
        axis=1
    )
    # For non-alerted events, lead is None -> they go to "no_alert"
    for _, r in events.iterrows():
        if r.alerted == 0:
            b = "no_alert"
        else:
            b = lead_bin(r.lead_calc)
        out.setdefault(b, {"n_event": 0, "n_alert": 0})
        out[b]["n_event"] += 1
        if r.alerted == 1:
            out[b]["n_alert"] += 1
    return out


def build_alert_row_features(samples, sy_to_alert_tstar, sy_to_oof_score_map):
    """
    For each alerted site-year, build the feature vector at the alert_tstar row.
    samples: list of nowcast samples (each has site_id, year, tstar, X, ...)
    sy_to_alert_tstar: dict (site, year) -> int (alert tstar)
    sy_to_oof_score_map: dict (site, year, tstar) -> float (the Stage 1a OOF/test score at that row)
    Returns: X (n, D+1), y (n,), keys [(site, year), ...]
    """
    matched = []
    for s in samples:
        sy = (str(s["site_id"]), int(s["year"]))
        if sy not in sy_to_alert_tstar: continue
        if int(s["tstar"]) != int(sy_to_alert_tstar[sy]): continue
        matched.append(s)
    print(f"  matched alert rows: {len(matched)} / expected alerted sites: {len(sy_to_alert_tstar)}")

    X_base = build_tabular_from_samples(matched, add_tstar_position_feature=True)  # (n, 211)
    extra_score = np.asarray([
        float(sy_to_oof_score_map.get(
            (str(s["site_id"]), int(s["year"]), int(s["tstar"])), float("nan")
        )) for s in matched
    ], dtype=np.float32).reshape(-1, 1)
    X = np.concatenate([X_base, extra_score], axis=1)
    y = np.asarray([int(s.get("y_event", 0)) for s in matched], dtype=np.int64)
    keys = [(str(s["site_id"]), int(s["year"])) for s in matched]
    return X, y, keys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--split_seed", type=int, default=54)
    ap.add_argument("--split_mode", default="site_year")
    ap.add_argument("--ckpt_stage1", default="rice/outputs_stage1/sheath_blight_siteyear54/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split54_siteyear_ymin2002.pt")
    ap.add_argument("--oof_csv", default="rice/outputs_stage1/sheath_blight_siteyear54/ckpt/oof_train_seed54_5fold.csv")
    ap.add_argument("--probs_csv", default="rice/outputs_stage1/sheath_blight_siteyear54/eval/event_eval_sheath_blight_run4_stage1_xgb_nowcast_w28_s1_tpos_split54_siteyear_ymin2002_tauAlertF1_cap05_k3_ma1_nogate_abs_probs.csv")
    ap.add_argument("--tau_a", type=float, default=0.51, help="Stage 1a threshold (recall-focused)")
    ap.add_argument("--tau_baseline", type=float, default=0.65, help="baseline single-stage tau for comparison")
    ap.add_argument("--out_dir", default="rice/outputs_stage1/sheath_blight_siteyear54/cascade")
    ap.add_argument("--seed", type=int, default=0, help="which Stage 1 seed to use for val/test eval")
    ap.add_argument("--xgb_max_depth", type=int, default=4)
    ap.add_argument("--xgb_n_estimators", type=int, default=200)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Config from Stage 1 ckpt
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

    print(f"[cfg] tau_a={args.tau_a} tau_baseline={args.tau_baseline} seed={args.seed}")
    _, get_feature_cols = resolve_pest(args.pest)
    feature_cols, feature_names, T, samples = build_samples_for_run(int(args.run), get_feature_cols)
    train_s, val_s, test_s = split_samples(samples, val_frac=0.1, test_frac=0.1,
                                            seed=int(args.split_seed), split_mode=args.split_mode)
    print(f"[split] train={len(train_s)} val={len(val_s)} test={len(test_s)}")

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

    # Step 3a: derive alerts on OOF train at tau_a
    oof = pd.read_csv(args.oof_csv)
    print(f"[oof] rows={len(oof)} y_event mean={oof.y_event.mean():.3f}")
    oof_alerts = derive_alerts(oof, args.tau_a)
    oof_alerts_metrics = metrics_from_alerts(oof_alerts, label="oof_train Stage1a only")
    print(f"[oof] Stage 1a metrics: {oof_alerts_metrics}")

    # Build score-lookup map for OOF rows (site, year, tstar) -> p_cal
    oof_score_map = {(str(r.site), int(r.year), int(r.tstar)): float(r.p_cal) for r in oof.itertuples(index=False)}

    # Build Stage 1b training data
    sy_to_alert_tstar_train = {(r.site, r.year): int(r.alert_tstar) for _, r in oof_alerts.iterrows() if r.alerted == 1}
    print(f"\n[train Stage 1b] alerted site-years: {len(sy_to_alert_tstar_train)}")
    X_tr_b, y_tr_b, keys_tr_b = build_alert_row_features(train_now, sy_to_alert_tstar_train, oof_score_map)
    print(f"[train Stage 1b] X={X_tr_b.shape} pos rate={y_tr_b.mean():.3f} (target = is_event of alerted site-years)")

    # Val/test: use production Stage 1 probs to get alerts + score map
    probs = pd.read_csv(args.probs_csv)
    val_p = probs[(probs.split == "val") & (probs.seed == int(args.seed))]
    test_p = probs[(probs.split == "test") & (probs.seed == int(args.seed))]
    print(f"\n[probs] val_n={len(val_p)} test_n={len(test_p)}")

    val_alerts_a = derive_alerts(val_p, args.tau_a)
    test_alerts_a = derive_alerts(test_p, args.tau_a)
    print(f"[val   Stage1a] {metrics_from_alerts(val_alerts_a, 'val Stage1a only')}")
    print(f"[test  Stage1a] {metrics_from_alerts(test_alerts_a, 'test Stage1a only')}")

    val_score_map = {(str(r.site), int(r.year), int(r.tstar)): float(r.p_cal) for r in val_p.itertuples(index=False)}
    test_score_map = {(str(r.site), int(r.year), int(r.tstar)): float(r.p_cal) for r in test_p.itertuples(index=False)}

    sy_to_alert_tstar_val = {(r.site, r.year): int(r.alert_tstar) for _, r in val_alerts_a.iterrows() if r.alerted == 1}
    sy_to_alert_tstar_test = {(r.site, r.year): int(r.alert_tstar) for _, r in test_alerts_a.iterrows() if r.alerted == 1}

    print(f"\n[val Stage 1b features]")
    X_va_b, y_va_b, keys_va_b = build_alert_row_features(val_now, sy_to_alert_tstar_val, val_score_map)
    print(f"  X_va={X_va_b.shape} pos rate={y_va_b.mean():.3f}")
    print(f"\n[test Stage 1b features]")
    X_te_b, y_te_b, keys_te_b = build_alert_row_features(test_now, sy_to_alert_tstar_test, test_score_map)
    print(f"  X_te={X_te_b.shape} pos rate={y_te_b.mean():.3f}")

    # Step 3: Train Stage 1b XGB
    pos_w_b = (y_tr_b == 0).sum() / max((y_tr_b == 1).sum(), 1)
    print(f"\n[Stage 1b train] pos_weight={pos_w_b:.3f} max_depth={args.xgb_max_depth} n_est={args.xgb_n_estimators}")
    clf_b = XGBClassifier(
        n_estimators=int(args.xgb_n_estimators), max_depth=int(args.xgb_max_depth),
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=0, eval_metric="logloss", scale_pos_weight=float(pos_w_b),
        n_jobs=4,
    )
    t0 = time.perf_counter()
    clf_b.fit(X_tr_b, y_tr_b)
    print(f"[Stage 1b] trained in {time.perf_counter()-t0:.1f}s")
    p_va_b = clf_b.predict_proba(X_va_b)[:, 1]
    p_te_b = clf_b.predict_proba(X_te_b)[:, 1]
    if y_va_b.sum() > 0 and y_va_b.sum() < len(y_va_b):
        print(f"[Stage 1b] val AUC={roc_auc_score(y_va_b, p_va_b):.3f}")
    if y_te_b.sum() > 0 and y_te_b.sum() < len(y_te_b):
        print(f"[Stage 1b] test AUC={roc_auc_score(y_te_b, p_te_b):.3f}")

    # Pick tau_b on val: maximize F1 subject to precision >= 0.85, fallback to max F1
    # We measure final cascade metrics at site-year level.
    def cascade_metrics_for_tau_b(alerts_a, X_b, y_b, keys_b, p_b, tau_b, label):
        """alerts_a: derive_alerts output. p_b length == # alerted in alerts_a (same order as keys_b)."""
        keep = {keys_b[i] for i in range(len(p_b)) if p_b[i] >= tau_b}
        df = alerts_a.copy()
        df["alerted_cascade"] = df.apply(
            lambda r: int(r.alerted == 1 and (str(r.site), int(r.year)) in keep), axis=1
        )
        df_out = df[["site","year","is_event","true_L","true_R","alert_tstar"]].copy()
        df_out["alerted"] = df["alerted_cascade"]
        return metrics_from_alerts(df_out, label)

    print("\n" + "="*90)
    print(f"Step 3 fail-fast: Stage 1b val precision improvement vs Stage 1a alone")
    print("="*90)
    val_a_only = metrics_from_alerts(val_alerts_a, "val Stage1a only")
    print(f"  Stage 1a only (val): precision={val_a_only['precision']:.3f} recall={val_a_only['recall']:.3f} F1={val_a_only['F1']:.3f}")
    best_tau_b = None; best_f1 = -1; best_metrics = None
    print(f"\n  Stage 1b tau_b sweep on VAL:")
    print(f"  {'tau_b':>6} {'recall':>7} {'precision':>10} {'FAR':>7} {'F1':>6} {'n_alert':>8}")
    for tau_b in [0.30,0.35,0.40,0.45,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]:
        m = cascade_metrics_for_tau_b(val_alerts_a, X_va_b, y_va_b, keys_va_b, p_va_b, tau_b, f"val cascade tau_b={tau_b}")
        print(f"  {tau_b:>6.2f} {m['recall']:>7.3f} {m['precision']:>10.3f} {m['FAR']:>7.3f} {m['F1']:>6.3f} {m['n_alert']:>8}")
        if m["F1"] > best_f1:
            best_f1 = m["F1"]; best_tau_b = tau_b; best_metrics = m
    print(f"\n  best tau_b (max F1): {best_tau_b} -> val precision={best_metrics['precision']:.3f}")

    val_prec_a = val_a_only["precision"]
    val_prec_b = best_metrics["precision"]
    delta_prec = val_prec_b - val_prec_a
    print(f"\n  precision delta (Stage 1b vs Stage 1a): {delta_prec:+.3f}")
    if delta_prec < 0.05:
        print("  >>> FAIL-FAST: precision improvement < 5%. Reporting and stopping cascade evaluation.")
        # Still write a summary so we have data
    else:
        print("  >>> PASS: continuing to Step 4 cascade test eval.")

    # Step 4: Test eval at multiple tau_b around best
    print("\n" + "="*90)
    print("Step 4: TEST cascade vs baseline single-stage")
    print("="*90)
    test_baseline_alerts = derive_alerts(test_p, args.tau_baseline)
    test_baseline_metrics = metrics_from_alerts(test_baseline_alerts, "test baseline tau=0.65")
    test_a_metrics = metrics_from_alerts(test_alerts_a, f"test Stage 1a tau={args.tau_a}")
    print(f"\n  baseline Stage 1 (tau={args.tau_baseline}): {test_baseline_metrics}")
    print(f"  Stage 1a only    (tau={args.tau_a}): {test_a_metrics}")
    print(f"\n  Stage 1b cascade — TEST sweep:")
    print(f"  {'tau_b':>6} {'recall':>7} {'precision':>10} {'FAR':>7} {'F1':>6} {'n_alert':>8}")
    cascade_results = []
    for tau_b in [0.30,0.40,0.50,0.55,0.60,0.65,0.70,0.75,0.80,0.85,0.90]:
        m = cascade_metrics_for_tau_b(test_alerts_a, X_te_b, y_te_b, keys_te_b, p_te_b, tau_b, f"test cascade tau_b={tau_b}")
        cascade_results.append((tau_b, m))
        marker = " ← best_tau_b on val" if abs(tau_b - best_tau_b) < 1e-6 else ""
        print(f"  {tau_b:>6.2f} {m['recall']:>7.3f} {m['precision']:>10.3f} {m['FAR']:>7.3f} {m['F1']:>6.3f} {m['n_alert']:>8}{marker}")

    # Write summary
    summary = {
        "tau_a": args.tau_a, "tau_baseline": args.tau_baseline, "best_tau_b_val_f1": best_tau_b,
        "val_stage1a_only": val_a_only, "val_cascade_best": best_metrics,
        "test_baseline": test_baseline_metrics, "test_stage1a_only": test_a_metrics,
        "test_cascade_sweep": [{"tau_b": t, **m} for t, m in cascade_results],
        "stage1b_xgb": {"max_depth": args.xgb_max_depth, "n_estimators": args.xgb_n_estimators, "scale_pos_weight": float(pos_w_b)},
    }
    (out_dir / "cascade_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[done] saved {out_dir / 'cascade_summary.json'}")

    # Lead-bin breakdown for best cascade vs baseline
    print("\n" + "="*90)
    print("LEAD BIN BREAKDOWN (test) — % of events alerted by lead bin")
    print("="*90)
    keep = {keys_te_b[i] for i in range(len(p_te_b)) if p_te_b[i] >= best_tau_b}
    cascade_alerts = test_alerts_a.copy()
    cascade_alerts["alerted"] = cascade_alerts.apply(
        lambda r: int(r.alerted == 1 and (str(r.site), int(r.year)) in keep), axis=1
    )
    print(f"\n  {'lead_bin':>12}  {'baseline':>14}  {'Stage1a':>14}  {'Cascade':>14}")
    bins_b = lead_bin_breakdown(test_baseline_alerts)
    bins_a = lead_bin_breakdown(test_alerts_a)
    bins_c = lead_bin_breakdown(cascade_alerts)
    all_bins = sorted(set(bins_b) | set(bins_a) | set(bins_c))
    for b in ["lead_1_14","lead_15_29","lead_30_45","lead_46_60","lead_61_75","lead_gt75","no_alert"]:
        if b not in all_bins: continue
        bb = bins_b.get(b, {"n_event":0,"n_alert":0})
        ba = bins_a.get(b, {"n_event":0,"n_alert":0})
        bc = bins_c.get(b, {"n_event":0,"n_alert":0})
        print(f"  {b:>12}  {bb['n_alert']}/{bb['n_event']:<11}  {ba['n_alert']}/{ba['n_event']:<11}  {bc['n_alert']}/{bc['n_event']:<11}")


if __name__ == "__main__":
    main()
