"""
Phase S — Per-sample best-offset selector for IoU_overall maximization.

Differentiator vs the inherited site-level GBM selector: this one is fully
sample-level (each (site, year) gets its own predicted offset).

Pipeline:
    1. Load outputs_phase_r_sample_grid.csv produced by phase_r --sample_grid_csv.
       Required columns: model, sample_id, site, year, t_star_doy,
                         true_event_doy, L, R, mu, sigma, offset, iou_matched,
                         matched.
       Each (model, sample_id) has one row per offset ∈ {60, 90, 105, 120}.
    2. For each model:
       a. Pivot to per-sample wide table:
              iou_off60, iou_off90, iou_off105, iou_off120
              mu_off60,  mu_off90,  mu_off105,  mu_off120
       b. Target = argmax_off iou_off (ties → shorter offset).
          Drop samples where every offset has matched=False (report count).
       c. Re-derive per-sample Stage 1 features (score_at_tstar, score_peak_to_tstar,
          score_mean_28d_before, score_slope_14d, score_auc_to_tstar) from
          calibrated Stage 1 scores. The Stage 1 ckpt and run id mirror the
          settings used by phase_r.
       d. Add static features: tstar_doy, mu (default offset), lead_predicted,
          year, site (target-encoded with per-fold leakage-free mean target).
       e. 5-fold stratified CV by target class. Two classifiers:
              - LogisticRegression  (standard-scaled, multinomial, balanced)
              - XGBClassifier       (depth=3, n_estimators=200)
       f. For each test fold, get predicted offset → look up
          actual iou_matched[sample, predicted_offset] from sample_grid.
          IoU_overall_fold = sum(iou) / N_TOTAL_TEST   (N_TOTAL_TEST = 575).
          P_ideal_overall_fold = #{lead_predicted_at_pred_off ∈ [14, 30)} / N_TOTAL_TEST.
       g. Aggregate fold means/stds; compare against baselines:
              fixed_60, fixed_120, oracle, mode-class (predict 60).
    3. Dump four CSVs to --out_dir, print decision line at end:
              Δ = selector_IoU_overall − max(fixed_60, fixed_120)
              Δ > +0.02  → GO
              0 < Δ ≤ +0.02 → MARGINAL
              Δ ≤ 0      → NO-GO

Sample grid σ is taken from the CSV (per-model operational σ); selector does
not change σ. Reported IoU_overall therefore uses each model's best σ
(baseline 4.0, phenobias 4.5, 2-sided 5.0).
"""

from __future__ import annotations

import argparse
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except Exception:
    HAS_XGB = False

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import build_nowcast_samples, build_tabular_from_samples
from rice.scripts.run_calibrate_event import apply_temperature, fit_temperature_grid


N_TOTAL_TEST = 575
OFFSETS = [60, 90, 105, 120]
P_IDEAL_LOW = 14
P_IDEAL_HIGH = 30


def best_tau_by_f1(y, p):
    """Best F1 threshold (val) for stage1 calibrated alert tau."""
    taus = np.linspace(0.05, 0.95, 19)
    best_tau, best_f1 = 0.5, -1.0
    for t in taus:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        if (2 * tp + fp + fn) == 0:
            continue
        f1 = (2 * tp) / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(t)
    return best_tau


def stage1_test_scores(stage1_ckpt_path: Path, run: int, args):
    """
    Returns:
        test_seas:  list of season-level test samples (site, year, X, L, R, censor_type)
        test_s:     list of nowcast samples (each has site_id/year/tstar/X)
        p_test_cal: calibrated stage 1 score per nowcast sample (same length)
        tau:        F1-best threshold on val
    """
    ckpt = torch.load(stage1_ckpt_path, map_location="cpu")
    add_tpos = bool(ckpt.get("add_tstar_position_feature", False))
    nc_window = int(ckpt.get("nowcast_window", 28))
    nc_stride = int(ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("nowcast_event_time_proxy", "r"))

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples = build_samples_for_run(run, get_feature_cols)
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
    X_val_tab = build_tabular_from_samples(val_s, add_tstar_position_feature=add_tpos)
    X_test_tab = build_tabular_from_samples(test_s, add_tstar_position_feature=add_tpos)
    clf = ckpt["trained_states"][0]["sk_model"]
    p_val_raw = clf.predict_proba(X_val_tab)[:, 1]
    p_test_raw = clf.predict_proba(X_test_tab)[:, 1]
    t_best, _ = fit_temperature_grid(y_val, p_val_raw)
    p_val_cal = apply_temperature(p_val_raw, t_best)
    p_test_cal = apply_temperature(p_test_raw, t_best)
    tau = best_tau_by_f1(y_val, p_val_cal)
    return test_seas, test_s, p_test_cal, tau


def compute_stage1_features_per_sample(test_s, p_test_cal, tstar_abs_map, doy_start):
    """
    For each sample_id, compute t*-anchored stage 1 features using score values
    at nowcast t* <= alert_tstar.

    Inputs:
        test_s:         flat list of stage1 nowcast rows (site_id/year/tstar/score)
        p_test_cal:     calibrated probability per row
        tstar_abs_map:  {sample_id_str: alert_tstar_abs_doy}  (absolute DOY)
                        sample_id_str = "{site_id}-{year}"
        doy_start:      season DOY start (frame 1 = doy_start)

    Returns DataFrame indexed by sample_id "{site}-{year}" with columns:
        score_at_tstar, score_peak_to_tstar, score_mean_28d_before,
        score_slope_14d, score_auc_to_tstar

    Robustness fixes (vs original):
      - groups key uses the same string sample_id as the merge target → no
        tuple-vs-csv-dtype mismatch.
      - When alert_tstar lands at the first nowcast t* (window=28), there is
        only one available score point. Slope/AUC default to 0.0 instead of
        NaN so the row is still emitted and survives the downstream dropna().
    """
    # Group scores by sample_id (string) → sorted (tstar_abs, score).
    groups = defaultdict(list)
    for s, p_cal in zip(test_s, p_test_cal):
        sid = f"{s['site_id']}-{int(s['year'])}"
        tstar_abs = int(s["tstar"]) + doy_start - 1
        groups[sid].append((tstar_abs, float(p_cal)))

    print(f"  [compute_stage1] groups={len(groups)}  tstar_abs_map={len(tstar_abs_map)}  "
          f"overlap={len(set(groups.keys()) & set(tstar_abs_map.keys()))}")
    # Pre-emptive sanity print: any non-overlap, show a few diagnostics.
    only_in_groups = set(groups.keys()) - set(tstar_abs_map.keys())
    only_in_map = set(tstar_abs_map.keys()) - set(groups.keys())
    if only_in_map:
        sample_missing = list(only_in_map)[:3]
        print(f"  [compute_stage1] {len(only_in_map)} sample_ids in tstar_abs_map missing from "
              f"stage1 groups (e.g. {sample_missing}) — these will get no features")

    rows = []
    _trapz = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)

    for sid, alert_abs in tstar_abs_map.items():
        vals = groups.get(sid)
        if not vals:
            continue
        alert_abs = int(alert_abs)
        vals = sorted(vals, key=lambda x: x[0])
        ts_arr = np.asarray([v[0] for v in vals], dtype=int)
        sc_arr = np.asarray([v[1] for v in vals], dtype=float)

        mask_to = ts_arr <= alert_abs
        if not mask_to.any():
            # Fall back: use the earliest available point (no scores before alert).
            sc_to = sc_arr[:1]
            ts_to = ts_arr[:1]
        else:
            sc_to = sc_arr[mask_to]
            ts_to = ts_arr[mask_to]

        # score_at_tstar = score at alert_abs (or closest preceding)
        idx_at = np.where(ts_to == alert_abs)[0]
        s_at = float(sc_to[idx_at[0]]) if len(idx_at) > 0 else float(sc_to[-1])
        s_peak_to = float(np.max(sc_to))

        # mean over 28 days before alert_abs (inclusive). Fall back to s_at.
        mask_28 = (ts_arr >= alert_abs - 27) & (ts_arr <= alert_abs)
        s_mean_28 = float(np.mean(sc_arr[mask_28])) if mask_28.any() else s_at

        # slope of last 14 days. 1-point case → 0.0 (no trend signal).
        mask_14 = (ts_arr >= alert_abs - 13) & (ts_arr <= alert_abs)
        if int(mask_14.sum()) >= 2:
            x14 = ts_arr[mask_14].astype(float) - float(alert_abs)
            y14 = sc_arr[mask_14]
            slope = float(np.polyfit(x14, y14, 1)[0])
        else:
            slope = 0.0

        # AUC up to alert_abs (trapezoidal). 1-point case → s_at (no width).
        if _trapz is not None and len(sc_to) >= 2:
            s_auc = float(_trapz(sc_to, x=ts_to.astype(float)))
        elif len(sc_to) >= 2:
            s_auc = float(np.sum((sc_to[:-1] + sc_to[1:]) * 0.5))
        else:
            s_auc = s_at

        rows.append({
            "sample_id": sid,
            "score_at_tstar": s_at,
            "score_peak_to_tstar": s_peak_to,
            "score_mean_28d_before": s_mean_28,
            "score_slope_14d": slope,
            "score_auc_to_tstar": s_auc,
        })
    return pd.DataFrame(rows).set_index("sample_id")


def build_per_sample_table(grid_model: pd.DataFrame, default_offset_for_mu: int = 90) -> pd.DataFrame:
    """
    Pivot the long sample_grid for one model into a per-sample wide table.

    Output columns:
        sample_id, site, year, t_star_doy, true_event_doy, L, R, sigma,
        iou_off60, iou_off90, iou_off105, iou_off120,
        mu_off60,  mu_off90,  mu_off105,  mu_off120,
        matched_off60, matched_off90, matched_off105, matched_off120,
        mu_default, lead_predicted, best_offset (target), n_offsets_matched
    """
    rows = []
    static_cols = ["sample_id", "site", "year", "t_star_doy", "true_event_doy",
                   "L", "R", "sigma"]
    for sample_id, sub in grid_model.groupby("sample_id"):
        d = {c: sub[c].iloc[0] for c in static_cols}
        offset_to_iou = {}
        offset_to_mu = {}
        offset_to_match = {}
        for _, r in sub.iterrows():
            o = int(r["offset"])
            offset_to_iou[o] = float(r["iou_matched"])
            offset_to_mu[o] = float(r["mu"]) if pd.notna(r["mu"]) else float("nan")
            offset_to_match[o] = bool(r["matched"])
        for o in OFFSETS:
            d[f"iou_off{o}"] = offset_to_iou.get(o, 0.0)
            d[f"mu_off{o}"] = offset_to_mu.get(o, float("nan"))
            d[f"matched_off{o}"] = bool(offset_to_match.get(o, False))
        # mu_default: default offset if matched, else first available
        mu_def = offset_to_mu.get(default_offset_for_mu)
        if mu_def is None or not np.isfinite(mu_def):
            for cand in OFFSETS:
                v = offset_to_mu.get(cand)
                if v is not None and np.isfinite(v):
                    mu_def = v
                    break
        d["mu_default"] = float(mu_def) if mu_def is not None else float("nan")
        d["lead_predicted"] = (d["mu_default"] - float(d["t_star_doy"])
                                if np.isfinite(d["mu_default"]) else float("nan"))
        d["n_offsets_matched"] = int(sum(offset_to_match.values()))
        # Target: argmax iou with tie-break to shorter offset.
        best_iou = -1.0
        best_off = None
        for o in OFFSETS:
            iou = offset_to_iou.get(o, 0.0)
            if iou > best_iou + 1e-12:
                best_iou = iou
                best_off = o
        d["best_offset"] = int(best_off) if best_off is not None else int(OFFSETS[0])
        d["best_iou"] = float(best_iou) if best_iou >= 0 else 0.0
        rows.append(d)
    return pd.DataFrame(rows)


def target_encode_site(train_df: pd.DataFrame, test_df: pd.DataFrame,
                      target_col: str = "best_offset") -> tuple[pd.Series, pd.Series, float]:
    """
    Per-fold leak-free target encoding for `site`:
        encoded(site) = mean(target) on train rows with that site,
                        falling back to global train mean for unseen sites.
    Returns (train_enc, test_enc, global_mean).
    """
    global_mean = float(train_df[target_col].mean())
    by_site = train_df.groupby("site")[target_col].mean().to_dict()
    train_enc = train_df["site"].map(by_site).fillna(global_mean)
    test_enc = test_df["site"].map(by_site).fillna(global_mean)
    return train_enc.astype(float), test_enc.astype(float), global_mean


def evaluate_predictions(per_sample: pd.DataFrame, pred_offsets: np.ndarray,
                          shift: float = 30.0, debug_label: str | None = None) -> dict:
    """
    Compute IoU_overall and P_ideal_overall for an array of predicted offsets
    aligned with `per_sample`.

    Per-sample mapping:
        o          = int(pred_off)
        iou        = per_sample.row['iou_off{o}']
        mu_at      = per_sample.row['mu_off{o}']
        sigma      = per_sample.row['sigma']        (per-model operational σ from sample_grid)
        L          = per_sample.row['L']            (true L, absolute DOY)
        PI_op_end  = mu_at + 1.96 σ − shift
        lead       = L − PI_op_end
        IDEAL when  14 ≤ lead < 30

    Aggregations (denominator N_TOTAL_TEST = 575):
        IoU_overall     = sum(iou)         / N_TOTAL_TEST
        P_ideal_overall = #{IDEAL samples} / N_TOTAL_TEST

    If `debug_label` is set, print the first 5 sample mappings (predicted offset,
    mu, σ, L, t_star, computed lead, looked-up iou) for sanity checking.
    """
    iou_sum = 0.0
    n_ideal = 0
    diag_rows = []
    n_matched_lookup = 0
    n_unmatched_lookup = 0
    for (_, row), pred_off in zip(per_sample.iterrows(), pred_offsets):
        o = int(pred_off)
        iou_col = f"iou_off{o}"
        mu_col = f"mu_off{o}"
        iou = float(row[iou_col]) if iou_col in row.index else float("nan")
        if not np.isfinite(iou):
            iou = 0.0
        iou_sum += iou
        mu_at = row[mu_col] if mu_col in row.index else float("nan")
        sigma = float(row["sigma"]) if "sigma" in row.index and pd.notna(row["sigma"]) else 5.0
        L = float(row["L"]) if "L" in row.index and pd.notna(row["L"]) else float("nan")
        lead = float("nan")
        if (mu_at is not None) and np.isfinite(mu_at) and np.isfinite(L):
            PI_op_end = float(mu_at) + 1.96 * sigma - float(shift)
            lead = L - PI_op_end
            if P_IDEAL_LOW <= lead < P_IDEAL_HIGH:
                n_ideal += 1
        if mu_at is not None and np.isfinite(mu_at):
            n_matched_lookup += 1
        else:
            n_unmatched_lookup += 1
        if debug_label is not None and len(diag_rows) < 5:
            diag_rows.append({
                "sample_id": row.get("sample_id", "?"),
                "pred_off": o,
                "mu_at_off": (None if not np.isfinite(mu_at) else float(mu_at)),
                "sigma": sigma,
                "L": L,
                "t_star_doy": float(row.get("t_star_doy", float("nan"))),
                "PI_op_end": (float(mu_at) + 1.96 * sigma - shift) if np.isfinite(mu_at) else None,
                "lead": (None if not np.isfinite(lead) else float(lead)),
                "iou_looked_up": iou,
                "in_IDEAL": (P_IDEAL_LOW <= lead < P_IDEAL_HIGH) if np.isfinite(lead) else False,
            })
    if debug_label is not None:
        print(f"    [eval:{debug_label}] mu_matched={n_matched_lookup} mu_unmatched={n_unmatched_lookup}")
        for d in diag_rows:
            print(f"      sid={d['sample_id']} pred_off={d['pred_off']} "
                  f"mu={d['mu_at_off']} σ={d['sigma']:.2f} L={d['L']} "
                  f"t*={d['t_star_doy']:.0f} PI_end={d['PI_op_end']} "
                  f"lead={d['lead']} iou={d['iou_looked_up']:.4f} IDEAL={d['in_IDEAL']}")
    # Six-bucket lead histogram (denom = N_TOTAL_TEST).
    leads_arr = []
    bucket_counts = {"MISSED": 0, "TOO_LATE": 0, "URGENT": 0,
                     "IDEAL": 0, "ADVANCE": 0, "TOO_EARLY": 0}
    for (_, row), pred_off in zip(per_sample.iterrows(), pred_offsets):
        o = int(pred_off)
        mu_col = f"mu_off{o}"
        mu_at = row[mu_col] if mu_col in row.index else float("nan")
        sigma = float(row["sigma"]) if "sigma" in row.index and pd.notna(row["sigma"]) else 5.0
        L = float(row["L"]) if "L" in row.index and pd.notna(row["L"]) else float("nan")
        if (mu_at is None) or not np.isfinite(mu_at) or not np.isfinite(L):
            leads_arr.append(float("nan"))
            continue
        lead = float(L) - (float(mu_at) + 1.96 * sigma - float(shift))
        leads_arr.append(lead)
        if lead < 0:                    bucket_counts["MISSED"] += 1
        elif lead < 7:                  bucket_counts["TOO_LATE"] += 1
        elif lead < 14:                 bucket_counts["URGENT"] += 1
        elif lead < 30:                 bucket_counts["IDEAL"] += 1
        elif lead < 45:                 bucket_counts["ADVANCE"] += 1
        else:                           bucket_counts["TOO_EARLY"] += 1
    return {
        "IoU_overall": iou_sum / N_TOTAL_TEST,
        "P_ideal_overall": n_ideal / N_TOTAL_TEST,
        "n_used": int(len(per_sample)),
        "n_matched_lookup": n_matched_lookup,
        "n_unmatched_lookup": n_unmatched_lookup,
        "bucket_counts": bucket_counts,
        "leads": leads_arr,
    }


def baseline_metrics(per_sample: pd.DataFrame, shift: float = 30.0) -> dict:
    """Compute fixed_60, fixed_120, oracle, mode-60 IoU/P_ideal_overall."""
    n = len(per_sample)
    fixed60 = evaluate_predictions(per_sample, np.full(n, 60), shift=shift,
                                    debug_label="fixed60")
    fixed120 = evaluate_predictions(per_sample, np.full(n, 120), shift=shift,
                                     debug_label="fixed120")
    mode60 = fixed60  # mode class assumed = shortest offset (cohort-friendly)
    # oracle = best per-sample iou (with tie-break to shorter offset)
    oracle_offs = per_sample["best_offset"].to_numpy(dtype=int)
    oracle = evaluate_predictions(per_sample, oracle_offs, shift=shift,
                                   debug_label="oracle")
    return {
        "iou_fixed60": fixed60["IoU_overall"],
        "iou_fixed120": fixed120["IoU_overall"],
        "iou_oracle": oracle["IoU_overall"],
        "iou_mode60": mode60["IoU_overall"],
        "p_ideal_fixed60": fixed60["P_ideal_overall"],
        "p_ideal_fixed120": fixed120["P_ideal_overall"],
        "p_ideal_oracle": oracle["P_ideal_overall"],
    }


def run_cv_for_classifier(per_sample: pd.DataFrame, feature_cols: list[str],
                          target_col: str, classifier: str, args,
                          shift: float = 30.0) -> dict:
    """5-fold stratified CV. Returns aggregated metrics + per-fold per-class info."""
    X = per_sample[feature_cols].to_numpy(dtype=float)
    y = per_sample[target_col].to_numpy(dtype=int)

    classes = sorted(np.unique(y).tolist())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    fold_iou = []
    fold_p_ideal = []
    fold_acc = []
    confusions = []
    per_class_acc = []
    feat_imp_runs = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X, y)):
        train_df = per_sample.iloc[tr_idx].copy()
        test_df = per_sample.iloc[te_idx].copy()

        # per-fold leak-free target encoding of site
        tr_site_enc, te_site_enc, gmean = target_encode_site(
            train_df, test_df, target_col=target_col)

        X_tr = train_df[feature_cols].to_numpy(dtype=float)
        X_te = test_df[feature_cols].to_numpy(dtype=float)
        # append site target-encoded as last feature
        X_tr = np.column_stack([X_tr, tr_site_enc.to_numpy()])
        X_te = np.column_stack([X_te, te_site_enc.to_numpy()])

        y_tr = train_df[target_col].to_numpy(dtype=int)
        y_te = test_df[target_col].to_numpy(dtype=int)

        if classifier == "logreg":
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    multi_class="multinomial", solver="lbfgs",
                    class_weight="balanced", max_iter=2000,
                    random_state=args.seed)),
            ])
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            # Logistic coefficients aren't reported as feature importances
        elif classifier == "xgb":
            if not HAS_XGB:
                raise SystemExit("xgboost not available; install or rerun with --classifier logreg")
            # class_weight via sample_weight balanced
            uniq, cnts = np.unique(y_tr, return_counts=True)
            inv = {int(c): float(len(y_tr)) / (len(uniq) * cnt) for c, cnt in zip(uniq, cnts)}
            sw = np.asarray([inv[int(c)] for c in y_tr], dtype=float)
            clf = XGBClassifier(
                max_depth=3, n_estimators=200, learning_rate=0.1,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=args.seed, verbosity=0,
                num_class=len(classes),
            )
            # label remap to 0..K-1 for xgb
            label_map = {c: i for i, c in enumerate(classes)}
            inv_label = {i: c for c, i in label_map.items()}
            y_tr_idx = np.asarray([label_map[int(c)] for c in y_tr], dtype=int)
            clf.fit(X_tr, y_tr_idx, sample_weight=sw)
            y_pred_idx = clf.predict(X_te)
            y_pred = np.asarray([inv_label[int(p)] for p in y_pred_idx], dtype=int)
            feat_imp_runs.append(clf.feature_importances_.tolist())
        else:
            raise SystemExit(f"unknown classifier: {classifier}")

        # First fold: print mapping diagnostics so we can verify
        # (predicted_offset, mu, lead, iou) per sample.
        dbg = f"{classifier}-fold{fold}" if fold == 0 else None
        m = evaluate_predictions(test_df, y_pred, shift=shift, debug_label=dbg)
        if fold == 0:
            uniq_pred, cnt_pred = np.unique(y_pred, return_counts=True)
            print(f"    [fold0 pred dist] " +
                  " ".join(f"{int(u)}:{int(c)}" for u, c in zip(uniq_pred, cnt_pred)))
        fold_iou.append(m["IoU_overall"])
        fold_p_ideal.append(m["P_ideal_overall"])
        fold_acc.append(accuracy_score(y_te, y_pred))
        confusions.append(confusion_matrix(y_te, y_pred, labels=classes))
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_te, y_pred, labels=classes, zero_division=0)
        per_class_acc.append(list(zip(classes, prec, rec, f1, sup)))

    confusion_mean = np.mean(np.stack(confusions, axis=0), axis=0)

    # per-class report (mean over folds)
    perclass_rows = []
    n_folds = len(per_class_acc)
    for i, c in enumerate(classes):
        precs = [pca[i][1] for pca in per_class_acc]
        recs = [pca[i][2] for pca in per_class_acc]
        f1s = [pca[i][3] for pca in per_class_acc]
        sups = [pca[i][4] for pca in per_class_acc]
        perclass_rows.append({
            "offset": int(c),
            "precision_mean": float(np.mean(precs)),
            "recall_mean": float(np.mean(recs)),
            "f1_mean": float(np.mean(f1s)),
            "support_total": int(np.sum(sups)),
        })

    feat_imp_mean = (np.mean(np.stack(feat_imp_runs, axis=0), axis=0).tolist()
                     if feat_imp_runs else None)

    return {
        "fold_iou": fold_iou,
        "fold_p_ideal": fold_p_ideal,
        "fold_acc": fold_acc,
        "iou_overall_mean": float(np.mean(fold_iou)),
        "iou_overall_std": float(np.std(fold_iou, ddof=0)),
        "p_ideal_overall_mean": float(np.mean(fold_p_ideal)),
        "p_ideal_overall_std": float(np.std(fold_p_ideal, ddof=0)),
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc, ddof=0)),
        "confusion_mean": confusion_mean,
        "classes": classes,
        "perclass": perclass_rows,
        "feat_imp_mean": feat_imp_mean,
    }


def process_model(model_label: str, grid_model: pd.DataFrame, stage1_features: pd.DataFrame,
                  args) -> dict:
    """Full per-model pipeline. Returns summary dict for the model."""
    print(f"\n========== Model: {model_label} ==========")
    per_sample = build_per_sample_table(grid_model)
    # Drop samples with no matched offset
    n_total_samples = len(per_sample)
    no_match_mask = (per_sample["n_offsets_matched"] == 0)
    n_no_match = int(no_match_mask.sum())
    per_sample = per_sample[~no_match_mask].copy()
    print(f"  samples_total={n_total_samples}  dropped_no_match={n_no_match}  "
          f"usable={len(per_sample)}")

    # Merge stage 1 features
    per_sample = per_sample.merge(stage1_features, left_on="sample_id",
                                   right_index=True, how="left")
    feat_cols = [
        "score_at_tstar", "score_peak_to_tstar", "score_mean_28d_before",
        "score_slope_14d", "score_auc_to_tstar",
        "tstar_doy_feat", "mu_default", "lead_predicted", "year_feat",
    ]
    per_sample["tstar_doy_feat"] = per_sample["t_star_doy"].astype(float)
    per_sample["year_feat"] = per_sample["year"].astype(float)
    # site feature is target-encoded inside CV; not in feat_cols list directly.

    # Drop rows missing any feature (cannot fit classifier)
    bad = per_sample[feat_cols].isna().any(axis=1)
    n_bad = int(bad.sum())
    if n_bad > 0:
        print(f"  dropped feature-NaN rows: {n_bad}")
    per_sample = per_sample[~bad].copy()

    shift = float(getattr(args, "shift", 30.0))
    print(f"  [config] operational shift = {shift}  (lead = L − (mu + 1.96σ − shift); "
          f"IDEAL band = [{P_IDEAL_LOW}, {P_IDEAL_HIGH}))")
    baselines = baseline_metrics(per_sample, shift=shift)
    print(f"  baselines: fixed60={baselines['iou_fixed60']:.4f}  "
          f"fixed120={baselines['iou_fixed120']:.4f}  oracle={baselines['iou_oracle']:.4f}")
    print(f"             p_ideal: fixed60={baselines['p_ideal_fixed60']:.4f}  "
          f"fixed120={baselines['p_ideal_fixed120']:.4f}  oracle={baselines['p_ideal_oracle']:.4f}")
    cls_counts = per_sample["best_offset"].value_counts().sort_index()
    print(f"  target class distribution: {dict(cls_counts)}")

    results = {"model": model_label, "n_used": len(per_sample),
               "n_dropped_no_match": n_no_match,
               "baselines": baselines,
               "feature_cols": feat_cols + ["site_target_enc"]}

    for classifier in ("logreg", "xgb") if HAS_XGB else ("logreg",):
        print(f"  --- classifier: {classifier} ---")
        cv = run_cv_for_classifier(per_sample, feat_cols, "best_offset",
                                    classifier=classifier, args=args, shift=shift)
        print(f"    IoU_overall = {cv['iou_overall_mean']:.4f} ± {cv['iou_overall_std']:.4f}")
        print(f"    P_ideal_overall = {cv['p_ideal_overall_mean']:.4f} ± "
              f"{cv['p_ideal_overall_std']:.4f}")
        print(f"    accuracy = {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}")
        print(f"    per-class:")
        for r in cv["perclass"]:
            print(f"      off={r['offset']}  P={r['precision_mean']:.3f}  "
                  f"R={r['recall_mean']:.3f}  F1={r['f1_mean']:.3f}  N={r['support_total']}")
        results[f"cv_{classifier}"] = cv
    return results


def write_outputs(model_results: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows = []
    for r in model_results:
        b = r["baselines"]
        for clf in ("logreg", "xgb"):
            key = f"cv_{clf}"
            if key not in r:
                continue
            cv = r[key]
            summary_rows.append({
                "model": r["model"],
                "classifier": clf,
                "iou_overall_mean": cv["iou_overall_mean"],
                "iou_overall_std": cv["iou_overall_std"],
                "p_ideal_overall_mean": cv["p_ideal_overall_mean"],
                "p_ideal_overall_std": cv["p_ideal_overall_std"],
                "accuracy_mean": cv["accuracy_mean"],
                "accuracy_std": cv["accuracy_std"],
                "iou_fixed60": b["iou_fixed60"],
                "iou_fixed120": b["iou_fixed120"],
                "iou_oracle": b["iou_oracle"],
                "iou_mode60": b["iou_mode60"],
                "n_used": r["n_used"],
                "n_dropped_no_match": r["n_dropped_no_match"],
            })
    summary_df = pd.DataFrame(summary_rows)
    summary_path = out_dir / "rice/outputs/diag/phase_s_selector_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[csv] summary → {summary_path}")

    for r in model_results:
        slug = r["model"].replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
        for clf in ("logreg", "xgb"):
            key = f"cv_{clf}"
            if key not in r:
                continue
            cv = r[key]
            pc_df = pd.DataFrame(cv["perclass"])
            pc_path = out_dir / f"rice/outputs/diag/phase_s_selector_perclass_{slug}_{clf}.csv"
            pc_df.to_csv(pc_path, index=False)
            cm_df = pd.DataFrame(cv["confusion_mean"],
                                  index=[f"true_{c}" for c in cv["classes"]],
                                  columns=[f"pred_{c}" for c in cv["classes"]])
            cm_path = out_dir / f"rice/outputs/diag/phase_s_selector_confusion_{slug}_{clf}.csv"
            cm_df.to_csv(cm_path)
            if clf == "xgb" and cv["feat_imp_mean"] is not None:
                fi_df = pd.DataFrame({
                    "feature": r["feature_cols"],
                    "importance_mean": cv["feat_imp_mean"],
                })
                fi_path = out_dir / f"rice/outputs/diag/phase_s_selector_featimp_{slug}_xgb.csv"
                fi_df.sort_values("importance_mean", ascending=False).to_csv(fi_path, index=False)
    return summary_df


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, default="sheath_blight")
    p.add_argument("--run", type=int, default=4)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    p.add_argument("--stage1_ckpt", type=str, required=True,
                   help="Stage 1 ckpt used to recompute calibrated scores for features.")
    p.add_argument("--sample_grid", type=str,
                   default="rice/outputs/diag/phase_r_sample_grid.csv",
                   help="CSV produced by phase_r --sample_grid_csv.")
    p.add_argument("--out_dir", type=str, default="rice/outputs/phase_s/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shift", type=float, default=30.0,
                   help="Operational shift used to compute lead = L − (mu + 1.96σ − shift). "
                        "Same definition as phase_n; IDEAL band is [14, 30).")
    p.add_argument("--model", type=str, default="all",
                   choices=["all", "phenobias", "2-sided", "baseline"],
                   help="Filter models to process. Maps to grid model labels by substring.")
    p.add_argument("--model_substring_map", type=str,
                   default="phenobias=phenobias;2-sided=2-sided;baseline=baseline",
                   help="LABEL_KEY=substring pairs; substring matched into grid 'model' column.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.sample_grid):
        raise SystemExit(f"[abort] sample_grid not found: {args.sample_grid}\n"
                         f"Run phase_r with --sample_grid_csv first.")
    grid = pd.read_csv(args.sample_grid)
    grid_models = list(grid["model"].unique())
    print(f"[input] sample_grid rows={len(grid)}  models={grid_models}")

    substring_map = {}
    for pair in str(args.model_substring_map).split(";"):
        if "=" not in pair: continue
        k, v = pair.split("=", 1)
        substring_map[k.strip()] = v.strip()

    if args.model == "all":
        labels_to_process = grid_models
    else:
        substr = substring_map.get(args.model, args.model)
        labels_to_process = [m for m in grid_models if substr in m]
    print(f"[plan] process {len(labels_to_process)} model(s): {labels_to_process}")
    if not labels_to_process:
        raise SystemExit("[abort] no matching model in sample_grid")

    # Compute Stage 1 calibrated scores + features once (shared across models).
    _ = resolve_pest(args.pest)
    print("\n[stage1] computing calibrated scores + per-sample features…")
    test_seas, test_s, p_test_cal, tau = stage1_test_scores(
        Path(args.stage1_ckpt), args.run, args)
    # alert_tstar_abs map from grid (Stage 1 is shared across models, so any model row works).
    # Key is the sample_id string ("{site}-{year}") for a leak-free string match with
    # compute_stage1_features_per_sample(). Using a (site,year) tuple risks dtype mismatch
    # because pandas may coerce 'site' / 'year' on csv reload.
    tstar_abs_map = {}
    for sid, sub in grid.groupby("sample_id"):
        r = sub.iloc[0]
        tstar_abs_map[str(sid)] = int(r["t_star_doy"])
    print(f"[stage1] tstar_abs_map built from sample_grid: {len(tstar_abs_map)} unique sample_ids")
    # doy_start: infer from earliest tstar - alert_frame_offset (or use 1 if doy_start_override=1).
    # Stage 1 ckpt usually has doy_start; we read it again for safety.
    s1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    doy_start = int(s1_ckpt.get("doy_start", 1))
    stage1_features = compute_stage1_features_per_sample(
        test_s, p_test_cal, tstar_abs_map, doy_start)
    print(f"[stage1 features] rows={len(stage1_features)}  cols={list(stage1_features.columns)}")

    # Process each model
    model_results = []
    for label in labels_to_process:
        grid_model = grid[grid["model"] == label].copy()
        r = process_model(label, grid_model, stage1_features, args)
        model_results.append(r)

    summary_df = write_outputs(model_results, out_dir)
    print("\n=================== Phase S summary ===================")
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(summary_df.to_string(index=False))

    # Decision line: best classifier-IoU vs max(fixed_60, fixed_120) per model.
    print("\n=================== Decision ===================")
    for r in model_results:
        b = r["baselines"]
        best_clf_iou = -1.0
        best_clf_name = "n/a"
        for clf in ("logreg", "xgb"):
            key = f"cv_{clf}"
            if key not in r: continue
            v = r[key]["iou_overall_mean"]
            if v > best_clf_iou:
                best_clf_iou = v
                best_clf_name = clf
        anchor = max(b["iou_fixed60"], b["iou_fixed120"])
        delta = best_clf_iou - anchor
        if delta > 0.02:
            verdict = "GO: operational change to per-sample selector"
        elif delta > 0.0:
            verdict = "MARGINAL: add features / try NN selector"
        else:
            verdict = "NO-GO: per-sample signal insufficient → 본질 트랙 검토"
        print(f"[{r['model']}]  best_clf={best_clf_name}  IoU_overall={best_clf_iou:.4f}  "
              f"anchor=max(fixed60, fixed120)={anchor:.4f}  Δ={delta:+.4f}  → {verdict}")

    # Operational σ visibility (sample_grid σ per model)
    print("\n--- operational σ used in sample_grid (per model) ---")
    sigma_table = grid.groupby("model")["sigma"].agg(["mean", "min", "max"])
    print(sigma_table.to_string())


if __name__ == "__main__":
    main()
