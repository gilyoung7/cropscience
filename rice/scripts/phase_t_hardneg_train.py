"""
Phase T15 — Hard-negative reweighting train.

Pipeline:
  1. Load base ckpt (A baseline lead14-45 ignore) and run inference on TRAIN.
  2. Identify hard-negative rows: train non-event nowcast rows with p_cal in
     the top --hard_neg_pct%.
  3. Build sample weights:
       positive (lead in [lead_min, lead_max])  -> pos_weight_scale
       hard_negative                              -> hard_neg_weight
       normal negative                            -> 1.0
       event row outside lead window              -> dropped (ignore policy)
  4. Train new XGB with same hypers + sample_weight.
  5. Save ckpt with hard-neg meta. Eval compatible with lead_aware_eval.

Constraint: hard-negative mining uses TRAIN ONLY (year <= train_year_max).
val and test never enter the hard-negative definition.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from xgboost import XGBClassifier

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import (
    build_nowcast_samples, build_tabular_from_samples, make_event_labels,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--base_ckpt", required=True,
                    help="A baseline ckpt used for hard-negative mining")
    ap.add_argument("--lead_min", type=int, required=True)
    ap.add_argument("--lead_max", type=int, required=True)
    ap.add_argument("--hard_neg_pct", type=float, required=True,
                    help="top X%% of train non-event rows by base ckpt p_cal")
    ap.add_argument("--hard_neg_weight", type=float, required=True,
                    help="sample weight applied to hard-negative rows (normal=1)")
    ap.add_argument("--pos_weight_scale", type=float, default=1.0,
                    help="multiplicative scale on positive row weight (base 1.0)")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--out_ckpt", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    base_ckpt = torch.load(args.base_ckpt, map_location="cpu", weights_only=False)
    C.DOY_START = int(base_ckpt.get("doy_start", 60))
    C.DOY_END = int(base_ckpt.get("doy_end", 300))
    add_tpos = bool(base_ckpt.get("add_tstar_position_feature", True))
    nc_window = int(base_ckpt.get("nowcast_window", 28))
    nc_stride = int(base_ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(base_ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(base_ckpt.get("nowcast_event_time_proxy", "mid"))
    nc_label_mode = str(base_ckpt.get("nowcast_label_mode", "eventually"))
    nc_label_horizon = base_ckpt.get("nowcast_label_horizon", None)
    nc_tstart = base_ckpt.get("nowcast_tstar_start", None)
    feature_cols = base_ckpt["feature_cols"]
    feature_names = base_ckpt["feature_names"]
    base_clf = base_ckpt["trained_states"][0]["sk_model"]
    hyper = {k: v for k, v in base_clf.get_params().items()
             if v is not None and k in {
                 "n_estimators", "max_depth", "learning_rate", "subsample",
                 "colsample_bytree", "reg_lambda", "min_child_weight", "gamma",
                 "random_state", "eval_metric", "tree_method", "device", "objective",
             }}
    print(f"[base hyper] {hyper}")
    print(f"[cfg] lead_window=[{args.lead_min},{args.lead_max}]  "
          f"hard_neg_pct={args.hard_neg_pct}  hard_neg_weight={args.hard_neg_weight}  "
          f"pos_weight_scale={args.pos_weight_scale}")

    _, gfc = resolve_pest(args.pest)
    _, _, _, samples = build_samples_for_run(args.run, gfc)
    train_seas, _, _ = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max)

    nc_kw = dict(window=nc_window, stride=nc_stride, tstar_start=nc_tstart,
                 only_pre_event=nc_only_pre, event_time_proxy=nc_proxy,
                 label_mode=nc_label_mode, label_horizon=nc_label_horizon)
    train_now = build_nowcast_samples(train_seas, **nc_kw)
    print(f"[nowcast] train rows = {len(train_now)}")

    # 1. base ckpt inference on train rows
    X_tr_all = build_tabular_from_samples(train_now, add_tstar_position_feature=add_tpos)
    p_tr_all = base_clf.predict_proba(X_tr_all)[:, 1]

    # 2. hard-negative threshold from train non-event rows
    is_event_row = np.array([str(s.get("base_censor_type", "right")) != "right"
                              for s in train_now], dtype=bool)
    ne_scores = p_tr_all[~is_event_row]
    if len(ne_scores) == 0:
        raise SystemExit("no non-event rows in train")
    pct = float(args.hard_neg_pct) / 100.0
    ne_thresh = float(np.quantile(ne_scores, 1.0 - pct))
    n_hardneg_est = int((ne_scores >= ne_thresh).sum())
    print(f"[hard-neg]  non-event rows = {len(ne_scores)}  threshold = {ne_thresh:.4f}  "
          f"~{n_hardneg_est} ({n_hardneg_est/len(ne_scores)*100:.2f}%) qualify")

    # 3. Apply lead filter + sample weights
    out_rows = []
    weights = []
    stats = {"n_pos_kept": 0, "n_hard_neg": 0, "n_normal_neg": 0,
             "n_event_dropped": 0}
    for s, p in zip(train_now, p_tr_all):
        ctype = str(s.get("base_censor_type", "right"))
        if ctype == "right":
            s2 = dict(s); s2["y_event"] = 0
            if p >= ne_thresh:
                w = float(args.hard_neg_weight)
                stats["n_hard_neg"] += 1
            else:
                w = 1.0
                stats["n_normal_neg"] += 1
            out_rows.append(s2); weights.append(w)
            continue
        L = s.get("L")
        if L is None:
            continue
        tstar = int(s["tstar"])
        lead = int(L) - tstar
        if args.lead_min <= lead <= args.lead_max:
            s2 = dict(s); s2["y_event"] = 1
            out_rows.append(s2); weights.append(float(args.pos_weight_scale))
            stats["n_pos_kept"] += 1
        else:
            stats["n_event_dropped"] += 1
    print(f"[filter stats] {stats}")
    if stats["n_pos_kept"] == 0:
        raise SystemExit("no positive rows after lead filter")

    # 4. Build X, y, w + train
    X = build_tabular_from_samples(out_rows, add_tstar_position_feature=add_tpos)
    y = make_event_labels(out_rows)
    w = np.asarray(weights, dtype=np.float32)
    pos_w_actual = float((y == 0).sum() / max((y == 1).sum(), 1))
    print(f"[train] X={X.shape}  pos_rate={y.mean():.3f}  "
          f"actual pos/neg={pos_w_actual:.3f}  w_mean={w.mean():.3f}  "
          f"w_pos_mean={w[y==1].mean():.3f}  w_neg_mean={w[y==0].mean():.3f}")

    hyper_local = dict(hyper)
    hyper_local["scale_pos_weight"] = pos_w_actual
    clf = XGBClassifier(**hyper_local)
    t0 = time.perf_counter()
    clf.fit(X, y, sample_weight=w)
    print(f"[xgb] trained in {time.perf_counter()-t0:.1f}s")

    out_ckpt = Path(args.out_ckpt)
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "run": int(args.run), "pest": str(args.pest),
        "d_in": int(base_ckpt.get("d_in", 30)),
        "feature_cols": feature_cols, "feature_names": feature_names,
        "year_max": base_ckpt.get("year_max"),
        "model_type": "event_tabular", "event_model": "xgb",
        "doy_start": int(C.DOY_START), "doy_end": int(C.DOY_END),
        "T": int(nc_window), "task_mode": "nowcast",
        "nowcast_window": int(nc_window), "nowcast_stride": int(nc_stride),
        "nowcast_tstar_start": (None if nc_tstart is None else int(nc_tstart)),
        "nowcast_only_pre_event": int(nc_only_pre),
        "nowcast_event_time_proxy": nc_proxy,
        "nowcast_label_mode": nc_label_mode,
        "nowcast_label_horizon": (None if nc_label_horizon is None else int(nc_label_horizon)),
        "add_tstar_position_feature": bool(add_tpos),
        "split_seed": int(args.split_seed), "split_mode": "year",
        "trained_states": [{"seed": 0, "best_epoch": None, "best_val_bce": None,
                              "sk_model": clf}],
        "lead_aware_label": {"lead_min": int(args.lead_min),
                              "lead_max": int(args.lead_max),
                              "outside_policy": "ignore"},
        "hard_negative": {
            "base_ckpt": str(args.base_ckpt),
            "hard_neg_pct": float(args.hard_neg_pct),
            "hard_neg_threshold": float(ne_thresh),
            "hard_neg_weight": float(args.hard_neg_weight),
            "pos_weight_scale": float(args.pos_weight_scale),
            "stats": stats,
        },
        "site_history_added": False,
        "phenology_added": False,
        "derived_weather_added": False,
    }
    torch.save(bundle, out_ckpt)
    print(f"[saved] {out_ckpt}")


if __name__ == "__main__":
    main()
