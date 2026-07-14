"""
Phase T6 — MIL proxy 1: cutoff-based weather-only site-year classifier.

For each cutoff DOY in --cutoffs:
  - extract weather-only stats per site-year using ONLY data up to cutoff_doy
  - target: y_event = (censor_type != "right"), 1 row per site-year
  - train XGB on train (<= val_year-1), evaluate val + test
  - report ROC-AUC, PR-AUC, F1max, precision@recall=0.85/0.90, FAR@recall

Features (no calendar / no lat-lon / no growing-stage / no tstar_rel):
  10 weather channels x 7 stats (mean/std/min/max/sum/slope/last)
  + 10 weather missing-flag means
  = 80 features

Goal: discriminate "weather signal genuinely weak" vs "label contamination".
  cutoff 180 AUC >= 0.70  -> contamination dominant, MIL retraining worthwhile
  cutoff 180 AUC 0.55-0.6 -> weather feature truly weak, season-risk gate only

cutoff_doy slice uses ONLY past info: base_X[:cutoff_idx] where
  cutoff_idx = cutoff_doy - DOY_START.

split: train_year <= val_year-1, val=val_year, test in [test_year_min, test_year_max].
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score, roc_curve,
)
from xgboost import XGBClassifier

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples
from rice.scripts.run_eval import build_samples_for_run


WEATHER_CH = list(range(0, 10))          # rain_7d_sum/days, tmean/tmax/tmin, rh, sun, rad, trange, trange_7d_mean
WEATHER_MISS_CH = list(range(15, 25))    # corresponding __miss flags

WEATHER_NAMES = [
    "rain_7d_sum", "rain_7d_days", "tmean_7d_mean", "tmax_7d_max", "tmin_7d_min",
    "rh_7d_mean", "sun_7d_sum", "rad_7d_sum", "trange", "trange_7d_mean",
]


def y_from_seas(s) -> int:
    return 1 if str(s.get("censor_type", "right")) != "right" else 0


def build_cutoff_features(base_seas, cutoff_doy: int, doy_start: int):
    """Site-year features from weather channels up to cutoff_doy (exclusive).
    Returns: X (n, 80), y (n,), keys [(site, year), ...], skipped count.
    """
    cutoff_idx = max(0, cutoff_doy - doy_start)
    Xs, ys, keys = [], [], []
    skipped = 0
    for s in base_seas:
        X = np.asarray(s["X"], dtype=np.float32)
        if X.shape[0] == 0 or cutoff_idx == 0:
            skipped += 1
            continue
        end = min(cutoff_idx, X.shape[0])
        Xw = X[:end, WEATHER_CH]            # (end, 10)
        Xm = X[:end, WEATHER_MISS_CH]       # (end, 10)
        if Xw.shape[0] < 7:
            skipped += 1
            continue
        mu = np.nanmean(Xw, axis=0)
        sd = np.nanstd(Xw, axis=0)
        mn = np.nanmin(Xw, axis=0)
        mx = np.nanmax(Xw, axis=0)
        sm = np.nansum(Xw, axis=0)
        t = np.arange(Xw.shape[0], dtype=np.float32)
        t_c = t - t.mean()
        var_t = float((t_c ** 2).sum()) + 1e-8
        slope = ((Xw - mu) * t_c[:, None]).sum(axis=0) / var_t
        last = Xw[-1]
        miss_mean = np.nanmean(Xm, axis=0)
        f = np.concatenate([mu, sd, mn, mx, sm, slope, last, miss_mean], axis=0)
        f = np.nan_to_num(f, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
        Xs.append(f)
        ys.append(y_from_seas(s))
        keys.append((str(s["site_id"]), int(s["year"])))
    if not Xs:
        return np.zeros((0, 80), dtype=np.float32), np.zeros((0,), dtype=np.int64), [], skipped
    return np.stack(Xs, axis=0), np.asarray(ys, dtype=np.int64), keys, skipped


def feature_names() -> list[str]:
    stats = ["mean", "std", "min", "max", "sum", "slope", "last"]
    out = []
    for st in stats:
        for nm in WEATHER_NAMES:
            out.append(f"{nm}_{st}")
    for nm in WEATHER_NAMES:
        out.append(f"{nm}_miss_mean")
    return out


def metrics_at_recall_targets(y: np.ndarray, scores: np.ndarray, targets=(0.85, 0.90)) -> dict:
    if len(set(y.tolist())) < 2 or len(y) < 5:
        return {k: None for k in ["ROC_AUC", "PR_AUC", "F1max",
                                  "P@R85", "P@R90", "FAR@R85", "FAR@R90"]}
    prec, rec, thr = precision_recall_curve(y, scores)
    f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-9)
    out = {
        "ROC_AUC": float(roc_auc_score(y, scores)),
        "PR_AUC": float(average_precision_score(y, scores)),
        "F1max": float(f1s.max()),
    }
    fpr, tpr, roc_thr = roc_curve(y, scores)
    for t in targets:
        # P@R: precision among rec>=t (smallest tau achieving the recall is best precision in this monotone curve)
        mask = rec >= t
        out[f"P@R{int(t*100)}"] = float(prec[mask].max()) if mask.any() else None
        # FAR @ R: lowest fpr among points with tpr >= t
        mask2 = tpr >= t
        out[f"FAR@R{int(t*100)}"] = float(fpr[mask2].min()) if mask2.any() else None
    return out


def fmt_metric(v):
    return f"{v:.3f}" if isinstance(v, float) else "  -  "


def run_cutoff(train_seas, val_seas, test_seas, cutoff_doy: int, doy_start: int,
               hyper: dict) -> dict:
    Xtr, ytr, keys_tr, sk_tr = build_cutoff_features(train_seas, cutoff_doy, doy_start)
    Xva, yva, keys_va, sk_va = build_cutoff_features(val_seas,   cutoff_doy, doy_start)
    Xte, yte, keys_te, sk_te = build_cutoff_features(test_seas,  cutoff_doy, doy_start)
    print(f"\n----- cutoff_doy={cutoff_doy}  (cutoff_idx={cutoff_doy - doy_start}) -----")
    print(f"  train: n={len(ytr)}  pos={int(ytr.sum())}  skipped={sk_tr}")
    print(f"  val  : n={len(yva)}  pos={int(yva.sum())}  skipped={sk_va}")
    print(f"  test : n={len(yte)}  pos={int(yte.sum())}  skipped={sk_te}")
    pos_w = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
    print(f"  pos_weight={pos_w:.3f}  X_tr.shape={Xtr.shape}")
    hyper_local = dict(hyper)
    hyper_local["scale_pos_weight"] = float(pos_w)
    clf = XGBClassifier(**hyper_local)
    clf.fit(Xtr, ytr)
    p_va = clf.predict_proba(Xva)[:, 1]
    p_te = clf.predict_proba(Xte)[:, 1]
    m_va = metrics_at_recall_targets(yva, p_va)
    m_te = metrics_at_recall_targets(yte, p_te)
    print(f"  {'split':>5}  {'ROC':>6} {'PR':>6} {'F1max':>6} "
          f"{'P@R85':>6} {'P@R90':>6} {'FAR@R85':>8} {'FAR@R90':>8}")
    for split_name, m in [("val", m_va), ("test", m_te)]:
        print(f"  {split_name:>5}  {fmt_metric(m['ROC_AUC']):>6} {fmt_metric(m['PR_AUC']):>6} "
              f"{fmt_metric(m['F1max']):>6} {fmt_metric(m['P@R85']):>6} {fmt_metric(m['P@R90']):>6} "
              f"{fmt_metric(m['FAR@R85']):>8} {fmt_metric(m['FAR@R90']):>8}")
    # Top-10 feature importances
    importances = clf.feature_importances_
    fnames = feature_names()
    top = sorted(zip(fnames, importances.tolist()), key=lambda x: -x[1])[:10]
    print(f"  top-10 feature importances:")
    for nm, imp in top:
        print(f"    {nm:>30}  {imp:.4f}")
    return {"cutoff_doy": cutoff_doy, "n_train": int(len(ytr)),
            "n_val": int(len(yva)), "n_test": int(len(yte)),
            "val_metrics": m_va, "test_metrics": m_te,
            "top_feats": [{"name": nm, "importance": float(imp)} for nm, imp in top]}


def verdict(results: list[dict]) -> str:
    lines = ["\n----- [proxy 1] Verdict -----"]
    by_cutoff = {r["cutoff_doy"]: r for r in results}
    last_cutoff = max(by_cutoff.keys())
    last_test_auc = by_cutoff[last_cutoff]["test_metrics"]["ROC_AUC"]
    if last_test_auc is None:
        return "\n".join(lines + ["  insufficient data"])
    lines.append(f"  cutoff_doy={last_cutoff}  test ROC-AUC = {last_test_auc:.3f}")
    if last_test_auc >= 0.70:
        lines.append("  > strong site-year-level weather signal exists.")
        lines.append("  > row-level XGB (weather_only AUC ~0.56) was bottlenecked by label noise,")
        lines.append("    not weather features. MIL / lead-aware Stage 1 retraining worthwhile.")
    elif last_test_auc >= 0.60:
        lines.append("  > moderate weather signal at site-year level (AUC 0.60-0.70).")
        lines.append("  > MIL would help some, but ceiling is lower than 0.70.")
    else:
        lines.append("  > weather features alone do not carry enough site-year signal.")
        lines.append("  > Stage 1 row-level vs site-year-level both ~0.55-0.60 -> data ceiling.")
        lines.append("  > Reposition Stage 1 as season-risk gate; timing precision is not recoverable.")
    # AUC trend across cutoffs
    aucs = [(r["cutoff_doy"], r["test_metrics"]["ROC_AUC"]) for r in results
            if r["test_metrics"]["ROC_AUC"] is not None]
    if len(aucs) >= 2:
        lines.append(f"  AUC trend across cutoffs: " +
                     "  ".join(f"DOY{c}={a:.3f}" for c, a in aucs))
        if aucs[-1][1] - aucs[0][1] >= 0.05:
            lines.append("  > AUC rises with later cutoff -> more weather data helps -> signal is real")
        else:
            lines.append("  > AUC flat across cutoffs -> additional weather data does not help -> noise floor")
    return "\n".join(lines)


def load_seas(args) -> tuple[list, list, list]:
    """Load val/test/train base season samples. Reuse probs cache if present."""
    if args.probs_cache and Path(args.probs_cache).exists():
        with open(args.probs_cache, "rb") as f:
            cache = pickle.load(f)
        if all(k in cache for k in ("train_seas", "val_seas", "test_seas")):
            print(f"[cache] loaded seasonal samples from {args.probs_cache}")
            return cache["train_seas"], cache["val_seas"], cache["test_seas"]
        print("[cache] missing one of train_seas/val_seas/test_seas -> rebuilding")
    print("[build] reading samples + splitting")
    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples = build_samples_for_run(args.run, get_feature_cols)
    train_seas, val_seas, test_seas = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )
    return train_seas, val_seas, test_seas


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--stage1_ckpt", required=True,
                    help="used only to copy XGB hyper-params for fair comparison")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--cutoffs", default="120,150,180")
    ap.add_argument("--probs_cache", default="")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    C.DOY_START = int(ckpt.get("doy_start", 60))
    C.DOY_END = int(ckpt.get("doy_end", 300))
    full_clf = ckpt["trained_states"][0]["sk_model"]
    hyper = {k: v for k, v in full_clf.get_params().items()
             if v is not None and k in {
                 "n_estimators", "max_depth", "learning_rate", "subsample",
                 "colsample_bytree", "reg_lambda", "min_child_weight", "gamma",
                 "random_state", "eval_metric",
                 "tree_method", "device", "objective",
             }}
    print(f"[cfg] DOY_START={C.DOY_START}  DOY_END={C.DOY_END}")
    print(f"[hyper] {hyper}")

    train_seas, val_seas, test_seas = load_seas(args)
    print(f"[split] train_seas={len(train_seas)}  val_seas={len(val_seas)}  test_seas={len(test_seas)}")
    n_pos_tr = sum(y_from_seas(s) for s in train_seas)
    n_pos_va = sum(y_from_seas(s) for s in val_seas)
    n_pos_te = sum(y_from_seas(s) for s in test_seas)
    print(f"[label] interval (event=1): train={n_pos_tr}/{len(train_seas)}  "
          f"val={n_pos_va}/{len(val_seas)}  test={n_pos_te}/{len(test_seas)}")

    cutoffs = [int(x) for x in args.cutoffs.split(",") if x.strip()]
    print(f"\n========== Phase T6 MIL proxy 1 — weather-only site-year ==========")
    print(f"  cutoffs (DOY): {cutoffs}")

    results = []
    for cutoff in cutoffs:
        r = run_cutoff(train_seas, val_seas, test_seas, cutoff, C.DOY_START, hyper)
        results.append(r)

    v = verdict(results)
    print(v)
    summary = {
        "args": {"pest": args.pest, "run": args.run, "cutoffs": cutoffs,
                 "split": {"val_year": args.val_year,
                           "test_year_min": args.test_year_min,
                           "test_year_max": args.test_year_max}},
        "hyper": hyper, "results": results, "verdict_text": v,
    }
    (out_dir / "mil_proxy1_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[saved] {out_dir / 'mil_proxy1_summary.json'}")


if __name__ == "__main__":
    main()
