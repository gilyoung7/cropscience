"""
Stage-1 5-fold OOF predictions on train split (split_seed=54, site_year mode).
Uses same XGB hyperparameters as the production Stage 1 ckpt.
Folds split by (site, year) groups so a site-year never spans folds.
Outputs: rice/outputs_stage1/sheath_blight_siteyear54/ckpt/oof_train_seed54_5fold.csv

Schema: split=oof_train, fold, sample_id, site, year, tstar, y_event, true_L, true_R, p_raw, p_cal
(p_cal = p_raw here; no per-fold temperature scaling — Stage 1b training is robust to that)
"""

from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

from rice.configs import config as C
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import build_tabular_from_samples, build_nowcast_samples, make_event_labels
from rice.src.dataset import split_samples
from rice.src.pest_resolver import resolve_pest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--split_seed", type=int, default=54)
    ap.add_argument("--split_mode", default="site_year")
    ap.add_argument("--ckpt", default="rice/outputs_stage1/sheath_blight_siteyear54/ckpt/event_run4_xgb_nowcast_w28_s1_tpos_split54_siteyear_ymin2002.pt")
    ap.add_argument("--out_csv", default="rice/outputs_stage1/sheath_blight_siteyear54/ckpt/oof_train_seed54_5fold.csv")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--cv_seed", type=int, default=42)
    ap.add_argument("--xgb_seed", type=int, default=0)
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    C.DOY_START = int(ck["doy_start"])
    C.DOY_END = int(ck["doy_end"])
    nowcast_window = int(ck["nowcast_window"])
    nowcast_stride = int(ck["nowcast_stride"])
    nowcast_only_pre_event = bool(int(ck.get("nowcast_only_pre_event", 1)))
    nowcast_event_time_proxy = str(ck.get("nowcast_event_time_proxy", "mid"))
    nowcast_label_mode = str(ck.get("nowcast_label_mode", "eventually"))
    nowcast_label_horizon = ck.get("nowcast_label_horizon", None)
    nowcast_tstar_start = ck.get("nowcast_tstar_start", None)
    add_tstar_position_feature = bool(ck.get("add_tstar_position_feature", True))
    event_pos_weight = float(ck["trained_states"][0]["sk_model"].get_params().get("scale_pos_weight", 1.0))
    print(f"[cfg] DOY {C.DOY_START}-{C.DOY_END} window={nowcast_window} stride={nowcast_stride} "
          f"label_mode={nowcast_label_mode} pos_weight={event_pos_weight:.4f}")

    _, get_feature_cols = resolve_pest(args.pest)
    feature_cols, feature_names, T, samples = build_samples_for_run(int(args.run), get_feature_cols)
    print(f"[base] samples={len(samples)} D_in={len(feature_names)}")

    train_s, _val_s, _test_s = split_samples(samples, val_frac=0.1, test_frac=0.1,
                                              seed=int(args.split_seed), split_mode=args.split_mode)
    print(f"[split] train base site-years: {len(train_s)}")

    # 5-fold by (site, year)
    site_years = sorted({(s["site_id"], int(s["year"])) for s in train_s})
    rng = np.random.default_rng(int(args.cv_seed))
    perm = rng.permutation(len(site_years))
    folds = np.array_split(perm, int(args.n_folds))
    sy_to_fold = {}
    for fi, idx in enumerate(folds):
        for j in idx:
            sy_to_fold[site_years[j]] = fi
    print(f"[cv] folds: {[len(f) for f in folds]}")

    out_rows = []
    for fold in range(int(args.n_folds)):
        t0 = time.perf_counter()
        train_keep = [s for s in train_s if sy_to_fold[(s["site_id"], int(s["year"]))] != fold]
        held = [s for s in train_s if sy_to_fold[(s["site_id"], int(s["year"]))] == fold]
        print(f"\n[fold {fold}] base: train={len(train_keep)} held={len(held)}")

        train_now = build_nowcast_samples(
            train_keep,
            window=nowcast_window, stride=nowcast_stride,
            tstar_start=nowcast_tstar_start, only_pre_event=nowcast_only_pre_event,
            event_time_proxy=nowcast_event_time_proxy,
            label_mode=nowcast_label_mode, label_horizon=nowcast_label_horizon,
        )
        held_now = build_nowcast_samples(
            held,
            window=nowcast_window, stride=nowcast_stride,
            tstar_start=nowcast_tstar_start, only_pre_event=nowcast_only_pre_event,
            event_time_proxy=nowcast_event_time_proxy,
            label_mode=nowcast_label_mode, label_horizon=nowcast_label_horizon,
        )
        print(f"[fold {fold}] nowcast: train={len(train_now)} held={len(held_now)}")

        X_tr = build_tabular_from_samples(train_now, add_tstar_position_feature=add_tstar_position_feature)
        y_tr = make_event_labels(train_now)
        X_he = build_tabular_from_samples(held_now, add_tstar_position_feature=add_tstar_position_feature)
        y_he = make_event_labels(held_now)
        print(f"[fold {fold}] X_tr={X_tr.shape} y_tr.mean={y_tr.mean():.4f}")

        clf = XGBClassifier(
            n_estimators=400, max_depth=5, learning_rate=0.05,
            subsample=0.9, colsample_bytree=0.9, reg_lambda=1.0,
            min_child_weight=1.0, gamma=0.0,
            random_state=int(args.xgb_seed), eval_metric="logloss",
            scale_pos_weight=float(event_pos_weight),
            tree_method="hist", device="cuda",
        )
        clf.fit(X_tr, y_tr)
        p_he = clf.predict_proba(X_he)[:, 1]
        eps = 1e-8
        p_he = np.clip(p_he, eps, 1.0 - eps)

        for s, p in zip(held_now, p_he):
            sample_id = f"{s.get('site_id','')}-{int(s.get('year',-1))}-t{int(s.get('tstar',-1))}"
            out_rows.append({
                "split": "oof_train", "fold": fold, "sample_id": sample_id,
                "site": str(s.get("site_id", "")), "year": int(s.get("year", -1)),
                "tstar": int(s.get("tstar", -1)),
                "y_event": int(s.get("y_event", 0)),
                "true_L": s.get("L"), "true_R": s.get("R"),
                "p_raw": float(p), "p_cal": float(p),
            })
        print(f"[fold {fold}] done in {time.perf_counter()-t0:.1f}s, held p mean={p_he.mean():.3f}")

    out = pd.DataFrame(out_rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"\n[done] saved {args.out_csv} rows={len(out)}")


if __name__ == "__main__":
    main()
