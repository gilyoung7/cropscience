"""
Phase T2 — Stage 1b cascade evaluation for yearsplit cohort.

Stage 1a operating point is fixed via (--tau_a, --k_consecutive).
Stage 1b (XGB on monitoring-window features) is trained on alerted site-years
from 5-fold OOF Stage 1a predictions on the train portion. Cascade is then
evaluated on val/test by sweeping tau_b.

tau_b selection criterion (val):
  - keep candidate tau_b values where final_recall_val >= --target_recall
  - among kept, pick FAR_val minimum (tie-break: F1 max)
  - if none qualify, fall back to F1 max overall

All probabilities are temperature-calibrated with T* fit on production-trained
Stage 1a (train -> val). The same T* is applied to the 5-fold OOF predictions
so tau_a means the same thing across train/val/test.

Outputs:
  - tau_b sweep CSV (val + test final cascade metrics)
  - JSON summary at out_dir/cascade_summary.json

Compared rows in final report:
  - no-cascade (Stage 1a only at tau_a, k)
  - cascade at recommended tau_b
"""

from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import build_nowcast_samples, build_tabular_from_samples, make_event_labels
from rice.scripts.run_calibrate_event import apply_temperature, fit_temperature_grid
from rice.scripts.run_stage1b_cascade_v2 import (
    derive_alerts,
    metrics_from_alerts,
    build_features_for_alerts,
    lead_bin_breakdown,
)


def _probs_df(seas, nowcast_s, p_cal, split_name: str) -> pd.DataFrame:
    sy_meta = {(str(s["site_id"]), int(s["year"])): s for s in seas}
    rows = []
    for s, p in zip(nowcast_s, p_cal):
        key = (str(s["site_id"]), int(s["year"]))
        meta = sy_meta[key]
        ctype = str(meta["censor_type"])
        rows.append({
            "split": split_name, "site": key[0], "year": key[1], "tstar": int(s["tstar"]),
            "p_cal": float(p), "y_event": int(s["y_event"]),
            "true_L": int(meta["L"]) if ctype != "right" else None,
            "true_R": int(meta["R"]) if ctype != "right" else None,
        })
    return pd.DataFrame(rows)


def build_stage1_probs(stage1_ckpt: Path, run: int, args, cache_path: Path | None) -> dict:
    """
    Build calibrated stage1a probabilities for train/val/test:
      - val/test: from production ckpt (single XGB), temperature-calibrated on val
      - train: 5-fold OOF, calibrated with the SAME production T*

    Returns dict with: t_best, probs_df (train+val+test), train_now, val_now, test_now,
                       train_seas, val_seas, test_seas, oof_score_map, val_score_map, test_score_map
    """
    if cache_path and cache_path.exists():
        with open(cache_path, "rb") as f:
            cache = pickle.load(f)
        print(f"[cache] loaded probs from {cache_path}")
        return cache

    ckpt = torch.load(stage1_ckpt, map_location="cpu", weights_only=False)
    C.DOY_START = int(ckpt.get("doy_start", C.DOY_START))
    C.DOY_END = int(ckpt.get("doy_end", C.DOY_END))
    add_tpos = bool(ckpt.get("add_tstar_position_feature", False))
    nc_window = int(ckpt.get("nowcast_window", 28))
    nc_stride = int(ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("nowcast_event_time_proxy", "r"))
    nc_label_mode = str(ckpt.get("nowcast_label_mode", "eventually"))
    nc_label_horizon = ckpt.get("nowcast_label_horizon", None)
    nc_tstart = ckpt.get("nowcast_tstar_start", None)

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples = build_samples_for_run(run, get_feature_cols)
    train_seas, val_seas, test_seas = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )
    print(f"[split:year] train_seas={len(train_seas)} val_seas={len(val_seas)} test_seas={len(test_seas)}")

    nc_kwargs = dict(window=nc_window, stride=nc_stride, tstar_start=nc_tstart,
                     only_pre_event=nc_only_pre, event_time_proxy=nc_proxy,
                     label_mode=nc_label_mode, label_horizon=nc_label_horizon)
    train_now = build_nowcast_samples(train_seas, **nc_kwargs)
    val_now = build_nowcast_samples(val_seas, **nc_kwargs)
    test_now = build_nowcast_samples(test_seas, **nc_kwargs)
    print(f"[nowcast] train={len(train_now)} val={len(val_now)} test={len(test_now)}")

    X_val = build_tabular_from_samples(val_now, add_tstar_position_feature=add_tpos)
    X_test = build_tabular_from_samples(test_now, add_tstar_position_feature=add_tpos)
    y_val = make_event_labels(val_now)

    clf = ckpt["trained_states"][0]["sk_model"]
    p_val_raw = clf.predict_proba(X_val)[:, 1]
    p_test_raw = clf.predict_proba(X_test)[:, 1]
    t_best, _ = fit_temperature_grid(y_val, p_val_raw)
    p_val_cal = apply_temperature(p_val_raw, t_best)
    p_test_cal = apply_temperature(p_test_raw, t_best)
    print(f"[stage1] T*={t_best:.3f}")

    # 5-fold OOF on training site-years
    sy_list = sorted({(s["site_id"], int(s["year"])) for s in train_seas})
    rng = np.random.default_rng(int(args.cv_seed))
    perm = rng.permutation(len(sy_list))
    folds = np.array_split(perm, int(args.n_folds))
    sy_to_fold = {}
    for fi, idx in enumerate(folds):
        for j in idx:
            sy_to_fold[sy_list[j]] = fi
    print(f"[oof] {args.n_folds}-fold sizes: {[len(f) for f in folds]}")

    event_pos_weight = float(clf.get_params().get("scale_pos_weight", 1.0))
    train_rows = []
    for fold in range(int(args.n_folds)):
        t0 = time.perf_counter()
        keep = [s for s in train_seas if sy_to_fold[(s["site_id"], int(s["year"]))] != fold]
        held = [s for s in train_seas if sy_to_fold[(s["site_id"], int(s["year"]))] == fold]
        keep_now = build_nowcast_samples(keep, **nc_kwargs)
        held_now = build_nowcast_samples(held, **nc_kwargs)
        X_keep = build_tabular_from_samples(keep_now, add_tstar_position_feature=add_tpos)
        y_keep = make_event_labels(keep_now)
        X_held = build_tabular_from_samples(held_now, add_tstar_position_feature=add_tpos)
        fold_clf = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            min_child_weight=1.0, gamma=0.0,
            random_state=int(args.xgb_seed), eval_metric="logloss",
            scale_pos_weight=float(event_pos_weight),
            tree_method="hist", device="cuda",
        )
        fold_clf.fit(X_keep, y_keep)
        p_held_raw = fold_clf.predict_proba(X_held)[:, 1]
        eps = 1e-8
        p_held_raw = np.clip(p_held_raw, eps, 1.0 - eps)
        p_held_cal = apply_temperature(p_held_raw, t_best)
        sy_meta = {(str(s["site_id"]), int(s["year"])): s for s in held}
        for s, p in zip(held_now, p_held_cal):
            key = (str(s["site_id"]), int(s["year"]))
            meta = sy_meta[key]
            ctype = str(meta["censor_type"])
            train_rows.append({
                "split": "oof_train", "fold": fold,
                "site": key[0], "year": key[1], "tstar": int(s["tstar"]),
                "p_cal": float(p), "y_event": int(s["y_event"]),
                "true_L": int(meta["L"]) if ctype != "right" else None,
                "true_R": int(meta["R"]) if ctype != "right" else None,
            })
        print(f"[oof fold {fold}] keep_sy={len(keep)} held_sy={len(held)} held_rows={len(held_now)} "
              f"p_mean={p_held_cal.mean():.3f}  ({time.perf_counter()-t0:.1f}s)")

    train_df = pd.DataFrame(train_rows)
    val_df = _probs_df(val_seas, val_now, p_val_cal, "val")
    test_df = _probs_df(test_seas, test_now, p_test_cal, "test")

    cache = {
        "t_best": float(t_best),
        "train_df": train_df, "val_df": val_df, "test_df": test_df,
        "train_now": train_now, "val_now": val_now, "test_now": test_now,
        "train_seas": train_seas, "val_seas": val_seas, "test_seas": test_seas,
    }
    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache, f)
        print(f"[cache] saved probs -> {cache_path}")
    return cache


def cascade_metrics_passthrough(alerts_a: pd.DataFrame, keys_b, p_b: np.ndarray, tau_b: float,
                                dropped_set: set, label: str) -> dict:
    keep = {keys_b[i] for i in range(len(p_b)) if p_b[i] >= tau_b}
    df = alerts_a.copy()
    def newalert(r):
        if r.alerted != 1: return 0
        sy = (str(r.site), int(r.year))
        if sy in dropped_set: return 1
        return int(sy in keep)
    df["alerted"] = df.apply(newalert, axis=1)
    return metrics_from_alerts(df[["site","year","is_event","true_L","true_R","alert_tstar","alerted"]], label)


def select_tau_b(val_sweep: list[dict], target_recall: float) -> tuple[float | None, str]:
    qualifying = [r for r in val_sweep if r["recall"] >= target_recall]
    if qualifying:
        # FAR min, tie-break by F1 max
        best = min(qualifying, key=lambda r: (r["FAR"], -r["F1"]))
        return float(best["tau_b"]), f"recall>={target_recall:.2f} (val) FAR-min"
    # Fallback: F1 max
    best = max(val_sweep, key=lambda r: r["F1"])
    return float(best["tau_b"]), "F1max (fallback: no tau_b meets target recall on val)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--tau_a", type=float, required=True)
    ap.add_argument("--k_consecutive", type=int, required=True)
    ap.add_argument("--monitor_M", type=int, default=14)
    ap.add_argument("--target_recall", type=float, default=0.85,
                    help="minimum required final_recall_val for tau_b selection")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--cv_seed", type=int, default=42)
    ap.add_argument("--xgb_seed", type=int, default=0)
    ap.add_argument("--xgb_max_depth", type=int, default=4)
    ap.add_argument("--xgb_n_estimators", type=int, default=200)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--probs_cache", default="",
                    help="path to pickle cache of stage1a probs (reused across (tau_a,k) runs)")
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    stage1_ckpt = Path(args.stage1_ckpt)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_path = Path(args.probs_cache) if args.probs_cache else None

    label = args.label or f"tau_a={args.tau_a}_k={args.k_consecutive}"
    print(f"\n========== Stage 1b cascade (yearsplit) :: {label} ==========")
    print(f"[cfg] tau_a={args.tau_a} k={args.k_consecutive} monitor_M={args.monitor_M} "
          f"target_recall={args.target_recall}")

    cache = build_stage1_probs(stage1_ckpt, args.run, args, cache_path)
    train_df = cache["train_df"]; val_df = cache["val_df"]; test_df = cache["test_df"]
    train_now = cache["train_now"]; val_now = cache["val_now"]; test_now = cache["test_now"]
    train_seas = cache["train_seas"]; val_seas = cache["val_seas"]; test_seas = cache["test_seas"]

    base_train = {(str(s["site_id"]), int(s["year"])): s for s in train_seas}
    base_val   = {(str(s["site_id"]), int(s["year"])): s for s in val_seas}
    base_test  = {(str(s["site_id"]), int(s["year"])): s for s in test_seas}

    # Apply Stage 1a alert rule at (tau_a, k)
    oof_alerts = derive_alerts(train_df, args.tau_a, k_consecutive=args.k_consecutive)
    val_alerts_a = derive_alerts(val_df, args.tau_a, k_consecutive=args.k_consecutive)
    test_alerts_a = derive_alerts(test_df, args.tau_a, k_consecutive=args.k_consecutive)
    m_oof = metrics_from_alerts(oof_alerts, "oof_train Stage1a")
    m_val_a = metrics_from_alerts(val_alerts_a, "val Stage1a")
    m_test_a = metrics_from_alerts(test_alerts_a, "test Stage1a")
    print(f"[stage1a oof  ] {m_oof}")
    print(f"[stage1a val  ] {m_val_a}")
    print(f"[stage1a test ] {m_test_a}")

    # Score maps for Stage 1b feature (one Stage 1a score concatenated to feats)
    oof_score = {(str(r.site), int(r.year), int(r.tstar)): float(r.p_cal)
                 for r in train_df.itertuples(index=False)}
    val_score = {(str(r.site), int(r.year), int(r.tstar)): float(r.p_cal)
                 for r in val_df.itertuples(index=False)}
    test_score = {(str(r.site), int(r.year), int(r.tstar)): float(r.p_cal)
                  for r in test_df.itertuples(index=False)}
    sy_alert_train = {(r.site, r.year): int(r.alert_tstar) for _, r in oof_alerts.iterrows() if r.alerted == 1}
    sy_alert_val = {(r.site, r.year): int(r.alert_tstar) for _, r in val_alerts_a.iterrows() if r.alerted == 1}
    sy_alert_test = {(r.site, r.year): int(r.alert_tstar) for _, r in test_alerts_a.iterrows() if r.alerted == 1}

    X_tr, y_tr, keys_tr, info_tr = build_features_for_alerts(
        train_now, base_train, sy_alert_train, oof_score, monitor_M=args.monitor_M)
    X_va, y_va, keys_va, info_va = build_features_for_alerts(
        val_now, base_val, sy_alert_val, val_score, monitor_M=args.monitor_M)
    X_te, y_te, keys_te, info_te = build_features_for_alerts(
        test_now, base_test, sy_alert_test, test_score, monitor_M=args.monitor_M)
    print(f"[1b train] X={X_tr.shape} pos={y_tr.mean():.3f}  drop_no_base={info_tr['dropped_no_base']} drop_no_monitor={info_tr['dropped_no_monitor']}")
    print(f"[1b val  ] X={X_va.shape} pos={y_va.mean():.3f}  {info_va}")
    print(f"[1b test ] X={X_te.shape} pos={y_te.mean():.3f}  {info_te}")

    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"[1b] pos_weight={pos_w:.3f} max_depth={args.xgb_max_depth} n_est={args.xgb_n_estimators}")
    clf = XGBClassifier(
        n_estimators=int(args.xgb_n_estimators), max_depth=int(args.xgb_max_depth),
        learning_rate=0.05, subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
        random_state=int(args.xgb_seed), eval_metric="logloss",
        scale_pos_weight=float(pos_w),
        tree_method="hist", device="cuda",
    )
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    print(f"[1b] trained in {time.perf_counter()-t0:.1f}s")
    p_va = clf.predict_proba(X_va)[:, 1] if X_va.shape[0] > 0 else np.zeros((0,))
    p_te = clf.predict_proba(X_te)[:, 1] if X_te.shape[0] > 0 else np.zeros((0,))
    auc_va = float(roc_auc_score(y_va, p_va)) if len(set(y_va.tolist())) > 1 else float("nan")
    auc_te = float(roc_auc_score(y_te, p_te)) if len(set(y_te.tolist())) > 1 else float("nan")
    print(f"[1b] val AUC={auc_va:.3f}  test AUC={auc_te:.3f}")

    dropped_val = set(sy_alert_val.keys()) - set(keys_va)
    dropped_test = set(sy_alert_test.keys()) - set(keys_te)
    print(f"[passthrough] val={len(dropped_val)} test={len(dropped_test)}")

    # tau_b sweep on val
    tau_b_grid = np.round(np.arange(0.10, 0.96, 0.025), 4)
    val_sweep = []
    test_sweep = []
    for tau_b in tau_b_grid:
        m_v = cascade_metrics_passthrough(val_alerts_a, keys_va, p_va, float(tau_b), dropped_val, f"val tau_b={tau_b}")
        m_t = cascade_metrics_passthrough(test_alerts_a, keys_te, p_te, float(tau_b), dropped_test, f"test tau_b={tau_b}")
        val_sweep.append({"tau_b": float(tau_b), **{k: v for k, v in m_v.items() if k != "label"}})
        test_sweep.append({"tau_b": float(tau_b), **{k: v for k, v in m_t.items() if k != "label"}})

    rec_tau_b, rec_reason = select_tau_b(val_sweep, args.target_recall)

    # Print sweep
    print("\n  tau_b sweep (val | test):")
    print(f"  {'tau_b':>6} | {'v_R':>6} {'v_F':>6} {'v_P':>6} {'v_F1':>6} {'v_nA':>5} | "
          f"{'t_R':>6} {'t_F':>6} {'t_P':>6} {'t_F1':>6} {'t_nA':>5}")
    for v, t in zip(val_sweep, test_sweep):
        mark = "  ← REC" if rec_tau_b is not None and abs(v["tau_b"] - rec_tau_b) < 1e-6 else ""
        print(f"  {v['tau_b']:>6.3f} | "
              f"{v['recall']:>6.3f} {v['FAR']:>6.3f} {v['precision']:>6.3f} {v['F1']:>6.3f} {int(v['n_alert']):>5d} | "
              f"{t['recall']:>6.3f} {t['FAR']:>6.3f} {t['precision']:>6.3f} {t['F1']:>6.3f} {int(t['n_alert']):>5d}{mark}")

    # Final comparison
    no_cascade = m_test_a
    if rec_tau_b is not None:
        rec_test = cascade_metrics_passthrough(test_alerts_a, keys_te, p_te, rec_tau_b, dropped_test, "cascade rec")
    else:
        rec_test = None

    print("\n========== FINAL COMPARISON (test) ==========")
    print(f"  criterion: {rec_reason}")
    print(f"  no-cascade (Stage 1a only @ tau_a={args.tau_a}, k={args.k_consecutive}):")
    print(f"    n_alert={no_cascade['n_alert']}  recall={no_cascade['recall']:.3f}  "
          f"FAR={no_cascade['FAR']:.3f}  precision={no_cascade['precision']:.3f}  F1={no_cascade['F1']:.3f}")
    if rec_test is not None:
        print(f"  cascade (Stage 1a then Stage 1b @ tau_b={rec_tau_b:.3f}):")
        print(f"    n_alert={rec_test['n_alert']}  recall={rec_test['recall']:.3f}  "
              f"FAR={rec_test['FAR']:.3f}  precision={rec_test['precision']:.3f}  F1={rec_test['F1']:.3f}")
        d_recall = rec_test['recall'] - no_cascade['recall']
        d_far = rec_test['FAR'] - no_cascade['FAR']
        d_prec = rec_test['precision'] - no_cascade['precision']
        d_f1 = rec_test['F1'] - no_cascade['F1']
        print(f"  delta cascade - no_cascade: "
              f"d_recall={d_recall:+.3f}  d_FAR={d_far:+.3f}  d_precision={d_prec:+.3f}  d_F1={d_f1:+.3f}")

    # Lead-bin breakdown
    if rec_tau_b is not None:
        keep_te = {keys_te[i] for i in range(len(p_te)) if p_te[i] >= rec_tau_b}
        cas_alerts = test_alerts_a.copy()
        def _newa(r):
            if r.alerted != 1: return 0
            sy = (str(r.site), int(r.year))
            if sy in dropped_test: return 1
            return int(sy in keep_te)
        cas_alerts["alerted"] = cas_alerts.apply(_newa, axis=1)
        bins_a = lead_bin_breakdown(test_alerts_a)
        bins_c = lead_bin_breakdown(cas_alerts)
        print("\n  lead-bin breakdown (interval events only, alerted/total):")
        print(f"  {'lead_bin':>12}  {'Stage1a':>10}  {'Cascade':>10}")
        for b in ["lead_1_14","lead_15_29","lead_30_45","lead_46_60","lead_61_75","lead_gt75","no_alert"]:
            if b not in bins_a and b not in bins_c: continue
            ba = bins_a.get(b, {"n_event":0,"n_alert":0})
            bc = bins_c.get(b, {"n_event":0,"n_alert":0})
            print(f"  {b:>12}  {ba['n_alert']:>3}/{ba['n_event']:<6d}  {bc['n_alert']:>3}/{bc['n_event']:<6d}")

    # Save outputs
    sweep_df = pd.DataFrame([
        {"tau_b": v["tau_b"],
         "val_recall": v["recall"], "val_FAR": v["FAR"], "val_precision": v["precision"], "val_F1": v["F1"], "val_n_alert": v["n_alert"],
         "test_recall": t["recall"], "test_FAR": t["FAR"], "test_precision": t["precision"], "test_F1": t["F1"], "test_n_alert": t["n_alert"]}
        for v, t in zip(val_sweep, test_sweep)
    ])
    sweep_csv = out_dir / f"cascade_sweep_{label.replace('=','_').replace(' ','_')}.csv"
    sweep_df.to_csv(sweep_csv, index=False)

    summary = {
        "label": label,
        "stage1_ckpt": str(stage1_ckpt),
        "split": {"split_seed": args.split_seed, "val_year": args.val_year,
                  "test_year_min": args.test_year_min, "test_year_max": args.test_year_max},
        "tau_a": args.tau_a, "k_consecutive": args.k_consecutive,
        "monitor_M": args.monitor_M, "target_recall": args.target_recall,
        "stage1_T_star": float(cache["t_best"]),
        "stage1a_oof": m_oof, "stage1a_val": m_val_a, "stage1a_test": m_test_a,
        "stage1b_val_AUC": auc_va, "stage1b_test_AUC": auc_te,
        "stage1b_feature_dims": {"base": info_tr.get("base_dim"), "monitor": info_tr.get("monitor_dim")},
        "passthrough_alerts": {"val": len(dropped_val), "test": len(dropped_test)},
        "recommended_tau_b": rec_tau_b,
        "recommendation_reason": rec_reason,
        "no_cascade_test": no_cascade,
        "cascade_test_at_rec": rec_test,
    }
    (out_dir / f"cascade_summary_{label.replace('=','_').replace(' ','_')}.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"\n[saved] {sweep_csv}")
    print(f"[saved] {out_dir / f'cascade_summary_{label.replace(chr(61),chr(95)).replace(chr(32),chr(95))}.json'}")


if __name__ == "__main__":
    main()
