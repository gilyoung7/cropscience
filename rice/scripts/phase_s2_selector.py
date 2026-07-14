"""
Phase S2 — Per-sample best-offset selector with mu features (ablation).

Extends phase_s_selector by:
    1. Accepting up to 4 models (baseline, 2-sided, phenobias, phenobias_nohead).
    2. Adding per-sample mu features from sample_grid:
            mu_off60, mu_off90, mu_off105, mu_off120
            mu_delta_60_120 = mu_off60 - mu_off120
            mu_range_4off   = max - min  over the 4 offsets
            mu_mean_4off    = mean        over the 4 offsets
       Missing-cell mu is NaN; per-fold train-median imputation + binary
       indicator column 'mu_off{X}_missing' is added.
    3. Three feature ablations:
            v1: stage1_only        (5 stage 1 score features + static)
            v2: stage1_plus_mu     (v1 + mu features)   ← main
            v3: mu_only            (mu features only, no stage 1)
    4. Decision per (model × feature_set × classifier):
            Δ = best_clf IoU − max(fixed_60, fixed_120)
            Δ > +0.02 → GO ; 0 < Δ ≤ +0.02 → MARGINAL ; Δ ≤ 0 → NO-GO

Reuses lead and IoU_overall conventions from phase_s_selector
(lead = L − (mu + 1.96σ − shift), IDEAL band [14, 30); denom 575).
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

# Reuse common pieces from phase_s_selector (stage 1 score helpers, evaluators,
# and the per-sample table builder).
from rice.scripts.phase_s_selector import (
    N_TOTAL_TEST, OFFSETS, P_IDEAL_LOW, P_IDEAL_HIGH,
    stage1_test_scores, compute_stage1_features_per_sample,
    build_per_sample_table, target_encode_site,
    evaluate_predictions, baseline_metrics,
)
from rice.src.pest_resolver import resolve_pest


STAGE1_FEATURES = [
    "score_at_tstar", "score_peak_to_tstar", "score_mean_28d_before",
    "score_slope_14d", "score_auc_to_tstar",
]
STATIC_FEATURES = ["tstar_doy_feat", "year_feat"]  # site is target-encoded inline
MU_FEATURES = [
    "mu_off60", "mu_off90", "mu_off105", "mu_off120",
    "mu_delta_60_120", "mu_range_4off", "mu_mean_4off",
]
MU_MISSING_INDICATORS = [
    "mu_off60_missing", "mu_off90_missing", "mu_off105_missing", "mu_off120_missing",
]


def attach_mu_features(per_sample: pd.DataFrame) -> pd.DataFrame:
    """Build mu_delta_60_120 / mu_range_4off / mu_mean_4off + missing indicators."""
    df = per_sample.copy()
    mu_cols = ["mu_off60", "mu_off90", "mu_off105", "mu_off120"]
    for c in mu_cols:
        df[f"{c}_missing"] = (~np.isfinite(df[c].to_numpy(dtype=float))).astype(int)
    mu_arr = df[mu_cols].to_numpy(dtype=float)
    df["mu_delta_60_120"] = mu_arr[:, 0] - mu_arr[:, 3]
    df["mu_range_4off"] = np.nanmax(mu_arr, axis=1) - np.nanmin(mu_arr, axis=1)
    df["mu_mean_4off"] = np.nanmean(mu_arr, axis=1)
    return df


def impute_train_median(train_X: np.ndarray, test_X: np.ndarray,
                        impute_cols_idx: list[int]) -> tuple[np.ndarray, np.ndarray, dict]:
    """In-place median imputation using train medians only (leak-free)."""
    medians = {}
    for j in impute_cols_idx:
        col_train = train_X[:, j]
        finite = np.isfinite(col_train)
        med = float(np.nanmedian(col_train[finite])) if finite.any() else 0.0
        medians[j] = med
        mask_tr = ~np.isfinite(train_X[:, j])
        train_X[mask_tr, j] = med
        mask_te = ~np.isfinite(test_X[:, j])
        test_X[mask_te, j] = med
    return train_X, test_X, medians


def feature_cols_for_set(feature_set: str) -> list[str]:
    if feature_set == "v1_stage1_only":
        return STAGE1_FEATURES + STATIC_FEATURES
    if feature_set == "v2_stage1_plus_mu":
        return STAGE1_FEATURES + STATIC_FEATURES + MU_FEATURES + MU_MISSING_INDICATORS
    if feature_set == "v3_mu_only":
        return STATIC_FEATURES + MU_FEATURES + MU_MISSING_INDICATORS
    raise ValueError(f"unknown feature_set: {feature_set}")


def run_cv(per_sample: pd.DataFrame, feat_cols: list[str], target_col: str,
           classifier: str, args, shift: float = 30.0,
           impute_cols: list[str] | None = None) -> dict:
    """
    5-fold stratified CV with per-fold target encoding + median imputation.

    Evaluation strategy (fixed in this version):
        - Each fold predicts on its held-out chunk → store (rows, y_pred).
        - After all 5 folds, concatenate into a single OOF (out-of-fold)
          prediction covering every usable sample.
        - IoU_overall / P_ideal_overall are computed *once* on the OOF set so
          the denominator (N_TOTAL_TEST = 575) is comparable to fixed60/120/oracle.
        - Per-fold conditional IoU (= fold sum / fold size) is reported as a
          fold-level stability diagnostic, separate from the headline OOF metric.
    """
    if impute_cols is None:
        impute_cols = []
    impute_idx_global = [feat_cols.index(c) for c in impute_cols if c in feat_cols]

    classes = sorted(per_sample[target_col].astype(int).unique().tolist())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)

    fold_cond_iou = []        # fold sum / fold size  (per-fold conditional)
    fold_cond_p_ideal = []
    fold_acc = []
    confusions = []
    per_class_runs = []
    feat_imp_runs = []
    oof_rows = []             # accumulated test-fold rows (DataFrame chunks)
    oof_preds = []            # accumulated predictions (np arrays)

    y_all = per_sample[target_col].to_numpy(dtype=int)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(per_sample)), y_all)):
        train_df = per_sample.iloc[tr_idx].copy()
        test_df = per_sample.iloc[te_idx].copy()
        tr_site_enc, te_site_enc, _ = target_encode_site(train_df, test_df, target_col=target_col)
        X_tr = train_df[feat_cols].to_numpy(dtype=float)
        X_te = test_df[feat_cols].to_numpy(dtype=float)
        # leak-free median imputation for mu features
        X_tr, X_te, _ = impute_train_median(X_tr, X_te, impute_idx_global)
        # append site target-encoded as final column
        X_tr = np.column_stack([X_tr, tr_site_enc.to_numpy()])
        X_te = np.column_stack([X_te, te_site_enc.to_numpy()])
        y_tr = train_df[target_col].to_numpy(dtype=int)
        y_te = test_df[target_col].to_numpy(dtype=int)

        if classifier == "logreg":
            clf = Pipeline([
                ("scaler", StandardScaler()),
                ("lr", LogisticRegression(
                    solver="lbfgs", class_weight="balanced",
                    max_iter=2000, random_state=args.seed)),
            ])
            clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
        elif classifier == "xgb":
            if not HAS_XGB:
                raise SystemExit("xgboost not available")
            uniq, cnts = np.unique(y_tr, return_counts=True)
            inv = {int(c): float(len(y_tr)) / (len(uniq) * cnt) for c, cnt in zip(uniq, cnts)}
            sw = np.asarray([inv[int(c)] for c in y_tr], dtype=float)
            label_map = {c: i for i, c in enumerate(classes)}
            inv_label = {i: c for c, i in label_map.items()}
            y_tr_idx = np.asarray([label_map[int(c)] for c in y_tr], dtype=int)
            clf = XGBClassifier(
                max_depth=3, n_estimators=200, learning_rate=0.1,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=args.seed, verbosity=0,
                num_class=len(classes),
            )
            clf.fit(X_tr, y_tr_idx, sample_weight=sw)
            y_pred_idx = clf.predict(X_te)
            y_pred = np.asarray([inv_label[int(p)] for p in y_pred_idx], dtype=int)
            feat_imp_runs.append(clf.feature_importances_.tolist())
        else:
            raise SystemExit(f"unknown classifier: {classifier}")

        # Fold-level conditional metrics: divide by fold size, NOT by N_TOTAL_TEST.
        # This is a stability diagnostic only; the headline result uses OOF below.
        fold_size = len(test_df)
        cond_iou_sum = 0.0
        cond_ideal_n = 0
        for (_, row), pred_off in zip(test_df.iterrows(), y_pred):
            o = int(pred_off)
            iou_col = f"iou_off{o}"
            iou_v = float(row[iou_col]) if iou_col in row.index else 0.0
            if not np.isfinite(iou_v):
                iou_v = 0.0
            cond_iou_sum += iou_v
            mu_col = f"mu_off{o}"
            mu_at = row[mu_col] if mu_col in row.index else float("nan")
            sigma = float(row["sigma"]) if "sigma" in row.index and pd.notna(row["sigma"]) else 5.0
            L_v = float(row["L"]) if "L" in row.index and pd.notna(row["L"]) else float("nan")
            if (mu_at is not None) and np.isfinite(mu_at) and np.isfinite(L_v):
                lead_v = L_v - (float(mu_at) + 1.96 * sigma - float(shift))
                if P_IDEAL_LOW <= lead_v < P_IDEAL_HIGH:
                    cond_ideal_n += 1
        fold_cond_iou.append(cond_iou_sum / max(fold_size, 1))
        fold_cond_p_ideal.append(cond_ideal_n / max(fold_size, 1))
        fold_acc.append(accuracy_score(y_te, y_pred))
        confusions.append(confusion_matrix(y_te, y_pred, labels=classes))
        prec, rec, f1, sup = precision_recall_fscore_support(
            y_te, y_pred, labels=classes, zero_division=0)
        per_class_runs.append(list(zip(classes, prec, rec, f1, sup)))

        # Accumulate for OOF eval.
        oof_rows.append(test_df.reset_index(drop=True))
        oof_preds.append(np.asarray(y_pred, dtype=int))

        if fold == 0:
            uniq_pred, cnt_pred = np.unique(y_pred, return_counts=True)
            print(f"    [fold0 pred dist] " +
                  " ".join(f"{int(u)}:{int(c)}" for u, c in zip(uniq_pred, cnt_pred)))

    # OOF concat + single overall evaluation against N_TOTAL_TEST denom.
    oof_df = pd.concat(oof_rows, axis=0, ignore_index=True)
    oof_pred_arr = np.concatenate(oof_preds, axis=0)
    print(f"    [OOF] n={len(oof_df)} (sum of folds), "
          "predicting on the same per_sample population the baselines evaluate on")
    uniq_oof, cnt_oof = np.unique(oof_pred_arr, return_counts=True)
    oof_dist = {int(u): int(c) for u, c in zip(uniq_oof, cnt_oof)}
    print(f"    [OOF pred dist] " + " ".join(f"{o}:{n}" for o, n in sorted(oof_dist.items())))
    oof_metrics = evaluate_predictions(oof_df, oof_pred_arr, shift=shift,
                                        debug_label=f"{classifier}-OOF")
    print(f"    [OOF] IoU_overall = {oof_metrics['IoU_overall']:.4f}  "
          f"P_ideal_overall = {oof_metrics['P_ideal_overall']:.4f}  "
          f"n_used = {oof_metrics['n_used']}")
    print(f"    [per-fold conditional IoU] " +
          "  ".join(f"f{i}:{v:.4f}" for i, v in enumerate(fold_cond_iou)) +
          f"  mean={np.mean(fold_cond_iou):.4f}  std={np.std(fold_cond_iou, ddof=0):.4f}")

    cm_mean = np.mean(np.stack(confusions, axis=0), axis=0)
    perclass_rows = []
    for i, c in enumerate(classes):
        precs = [pca[i][1] for pca in per_class_runs]
        recs = [pca[i][2] for pca in per_class_runs]
        f1s = [pca[i][3] for pca in per_class_runs]
        sups = [pca[i][4] for pca in per_class_runs]
        perclass_rows.append({
            "offset": int(c),
            "precision_mean": float(np.mean(precs)),
            "recall_mean": float(np.mean(recs)),
            "f1_mean": float(np.mean(f1s)),
            "support_total": int(np.sum(sups)),
        })
    feat_imp_mean = (np.mean(np.stack(feat_imp_runs, axis=0), axis=0).tolist()
                     if feat_imp_runs else None)

    # Attach OOF predictions + lead + bucket back onto oof_df for downstream analysis.
    oof_df = oof_df.copy()
    oof_df["pred_offset"] = oof_pred_arr
    oof_df["lead"] = oof_metrics["leads"]
    def _bucket(v):
        if not np.isfinite(v):                  return "NA"
        if v < 0:                               return "MISSED"
        if v < 7:                               return "TOO_LATE"
        if v < 14:                              return "URGENT"
        if v < 30:                              return "IDEAL"
        if v < 45:                              return "ADVANCE"
        return "TOO_EARLY"
    oof_df["bucket"] = [_bucket(v) for v in oof_metrics["leads"]]

    return {
        # Headline = OOF metrics, comparable to baselines (denom N_TOTAL_TEST=575).
        "iou_overall_mean": float(oof_metrics["IoU_overall"]),
        "iou_overall_std": float(np.std(fold_cond_iou, ddof=0)),  # fold-level variability
        "p_ideal_overall_mean": float(oof_metrics["P_ideal_overall"]),
        "p_ideal_overall_std": float(np.std(fold_cond_p_ideal, ddof=0)),
        # Per-fold conditional metrics (fold sum / fold size) for stability checks.
        "fold_cond_iou_mean": float(np.mean(fold_cond_iou)),
        "fold_cond_iou_std": float(np.std(fold_cond_iou, ddof=0)),
        "fold_cond_p_ideal_mean": float(np.mean(fold_cond_p_ideal)),
        "fold_cond_p_ideal_std": float(np.std(fold_cond_p_ideal, ddof=0)),
        "accuracy_mean": float(np.mean(fold_acc)),
        "accuracy_std": float(np.std(fold_acc, ddof=0)),
        "confusion_mean": cm_mean,
        "classes": classes,
        "perclass": perclass_rows,
        "feat_imp_mean": feat_imp_mean,
        "oof_pred_dist": oof_dist,
        "bucket_counts": oof_metrics["bucket_counts"],
        "oof_df": oof_df,
    }


def process_model(model_label: str, grid_model: pd.DataFrame,
                  stage1_features: pd.DataFrame, args) -> dict:
    print(f"\n========== Model: {model_label} ==========")
    per_sample = build_per_sample_table(grid_model)
    no_match_mask = (per_sample["n_offsets_matched"] == 0)
    n_no_match = int(no_match_mask.sum())
    per_sample = per_sample[~no_match_mask].copy()
    print(f"  samples_total_post_grid={len(per_sample) + n_no_match}  "
          f"dropped_no_match={n_no_match}  usable={len(per_sample)}")

    per_sample = per_sample.merge(stage1_features, left_on="sample_id",
                                   right_index=True, how="left")
    per_sample["tstar_doy_feat"] = per_sample["t_star_doy"].astype(float)
    per_sample["year_feat"] = per_sample["year"].astype(float)
    per_sample = attach_mu_features(per_sample)

    # Drop rows missing any stage1 feature value or sigma (cannot evaluate).
    # mu features are NOT dropped — they are imputed per fold.
    must_have = STAGE1_FEATURES + ["sigma", "L", "t_star_doy"]
    bad = per_sample[must_have].isna().any(axis=1)
    n_bad = int(bad.sum())
    if n_bad > 0:
        print(f"  dropped stage1/static-NaN rows: {n_bad}")
    per_sample = per_sample[~bad].copy()

    shift = float(getattr(args, "shift", 30.0))
    print(f"  [config] operational shift = {shift}")

    baselines = baseline_metrics(per_sample, shift=shift)
    print(f"  baselines:  fixed60={baselines['iou_fixed60']:.4f}  "
          f"fixed120={baselines['iou_fixed120']:.4f}  oracle={baselines['iou_oracle']:.4f}")
    print(f"              p_ideal: fixed60={baselines['p_ideal_fixed60']:.4f}  "
          f"fixed120={baselines['p_ideal_fixed120']:.4f}  "
          f"oracle={baselines['p_ideal_oracle']:.4f}")
    cls_counts = per_sample["best_offset"].value_counts().sort_index()
    print(f"  target class distribution: {dict(cls_counts)}")

    results = {"model": model_label, "n_used": len(per_sample),
               "n_dropped_no_match": n_no_match,
               "baselines": baselines, "by_set": {}}

    for feature_set in ("v1_stage1_only", "v2_stage1_plus_mu", "v3_mu_only"):
        feat_cols = feature_cols_for_set(feature_set)
        # Confirm columns exist
        missing = [c for c in feat_cols if c not in per_sample.columns]
        if missing:
            print(f"  [skip {feature_set}] missing cols: {missing}")
            continue
        impute_cols = [c for c in feat_cols if c in (MU_FEATURES + MU_MISSING_INDICATORS)]
        print(f"\n  --- feature_set: {feature_set}  ({len(feat_cols)} cols incl. site_target_enc later) ---")
        per_clf = {}
        for classifier in ("logreg", "xgb") if HAS_XGB else ("logreg",):
            print(f"    [clf: {classifier}]")
            cv = run_cv(per_sample, feat_cols, "best_offset",
                        classifier=classifier, args=args, shift=shift,
                        impute_cols=impute_cols)
            print(f"      IoU_overall   = {cv['iou_overall_mean']:.4f} ± {cv['iou_overall_std']:.4f}")
            print(f"      P_ideal_over  = {cv['p_ideal_overall_mean']:.4f} ± "
                  f"{cv['p_ideal_overall_std']:.4f}")
            print(f"      accuracy      = {cv['accuracy_mean']:.4f} ± {cv['accuracy_std']:.4f}")
            per_clf[classifier] = cv
        results["by_set"][feature_set] = {
            "feature_cols_used": feat_cols + ["site_target_enc"],
            "cv": per_clf,
        }
    return results


def write_outputs(model_results: list[dict], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    sum_rows = []
    for r in model_results:
        b = r["baselines"]
        for feature_set, payload in r["by_set"].items():
            for clf, cv in payload["cv"].items():
                sum_rows.append({
                    "model": r["model"],
                    "feature_set": feature_set,
                    "classifier": clf,
                    "iou_overall_oof": cv["iou_overall_mean"],
                    "iou_overall_fold_std": cv["iou_overall_std"],
                    "fold_cond_iou_mean": cv["fold_cond_iou_mean"],
                    "fold_cond_iou_std": cv["fold_cond_iou_std"],
                    "p_ideal_overall_oof": cv["p_ideal_overall_mean"],
                    "fold_cond_p_ideal_mean": cv["fold_cond_p_ideal_mean"],
                    "accuracy_mean": cv["accuracy_mean"],
                    "accuracy_std": cv["accuracy_std"],
                    "iou_fixed60": b["iou_fixed60"],
                    "iou_fixed120": b["iou_fixed120"],
                    "iou_oracle": b["iou_oracle"],
                    "iou_mode60": b["iou_mode60"],
                    "oof_pred_dist": str(cv["oof_pred_dist"]),
                    "n_used": r["n_used"],
                    "n_dropped_no_match": r["n_dropped_no_match"],
                })
    summary_df = pd.DataFrame(sum_rows)
    summary_path = out_dir / "selector_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n[csv] summary → {summary_path}  ({len(summary_df)} rows)")

    for r in model_results:
        slug = (r["model"].replace(" ", "_").replace("/", "_")
                              .replace("(", "").replace(")", ""))
        for feature_set, payload in r["by_set"].items():
            for clf, cv in payload["cv"].items():
                fs_slug = feature_set
                pc_df = pd.DataFrame(cv["perclass"])
                pc_df.to_csv(out_dir / f"perclass_{slug}_{fs_slug}_{clf}.csv", index=False)
                cm_df = pd.DataFrame(cv["confusion_mean"],
                                      index=[f"true_{c}" for c in cv["classes"]],
                                      columns=[f"pred_{c}" for c in cv["classes"]])
                cm_df.to_csv(out_dir / f"confusion_{slug}_{fs_slug}_{clf}.csv")
                if clf == "xgb" and cv["feat_imp_mean"] is not None:
                    fi_df = pd.DataFrame({
                        "feature": payload["feature_cols_used"],
                        "importance_mean": cv["feat_imp_mean"],
                    }).sort_values("importance_mean", ascending=False)
                    fi_df.to_csv(out_dir / f"featimp_{slug}_{fs_slug}_xgb.csv", index=False)
    return summary_df


def lead_bucket_comparison(model_results: list[dict], grid: pd.DataFrame,
                            args, out_dir: Path) -> None:
    """
    Print + CSV: lead-bucket distribution for the operational comparison.

    Compares:
      A) Current operational anchor: phenobias @ fixed offset=120, σ=4.5, shift.
         Computed directly from sample_grid rows (all phenobias samples with
         offset=120 and matched=True). P_ideal_overall uses n_total = 575.
      B) Candidate: D=15 2-sided + v3_mu_only + logreg selector (OOF predictions).
         Pulled from this run's model_results.
    """
    N_TOTAL = N_TOTAL_TEST
    shift = float(getattr(args, "shift", 30.0))
    BUCKETS = ["MISSED", "TOO_LATE", "URGENT", "IDEAL", "ADVANCE", "TOO_EARLY"]

    def _bucket_of(lead):
        if not np.isfinite(lead):       return "NA"
        if lead < 0:                    return "MISSED"
        if lead < 7:                    return "TOO_LATE"
        if lead < 14:                   return "URGENT"
        if lead < 30:                   return "IDEAL"
        if lead < 45:                   return "ADVANCE"
        return "TOO_EARLY"

    # --- (A) Current operational : phenobias @ fixed off=120 ---
    ph_label_candidates = [m for m in grid["model"].unique() if "phenobias" in m and "nohead" not in m]
    if not ph_label_candidates:
        print("\n[lead_buckets] no 'phenobias' rows found in sample_grid; skipping comparison.")
        return
    ph_label = ph_label_candidates[0]
    ph120 = grid[(grid["model"] == ph_label) & (grid["offset"] == 120) & (grid["matched"] == True)].copy()
    ph120["lead"] = ph120["L"].astype(float) - (ph120["mu"].astype(float) + 1.96 * ph120["sigma"].astype(float) - shift)
    ph120["bucket"] = ph120["lead"].apply(_bucket_of)
    ph_counts = {b: int((ph120["bucket"] == b).sum()) for b in BUCKETS}
    ph_n_match = int(ph120["bucket"].isin(BUCKETS).sum())

    # --- (B) Candidate selector: 2-sided + v3_mu_only + logreg OOF ---
    cand_label_substring = "2-sided"
    cand_fset = "v3_mu_only"
    cand_clf = "logreg"
    cand_counts = None
    cand_n_match = 0
    cand_model_label = None
    cand_n_used = 0
    for r in model_results:
        if cand_label_substring not in r["model"]:
            continue
        cand_model_label = r["model"]
        payload = r["by_set"].get(cand_fset)
        if payload is None:
            continue
        cv = payload["cv"].get(cand_clf)
        if cv is None:
            continue
        cand_counts = dict(cv["bucket_counts"])
        cand_n_match = sum(cand_counts.values())
        cand_n_used = r["n_used"]
        break
    if cand_counts is None:
        print("\n[lead_buckets] candidate combination not found in results; skipping.")
        return

    # Print comparison table.
    print("\n=================== Lead-bucket comparison (denom = 575) ===================")
    print(f"  shift = {shift}    σ source: per-model sample_grid (op σ)")
    print(f"  current : phenobias @ fixed off=120  (rows in sample_grid: {len(ph120)})")
    print(f"  candidate: {cand_model_label} + {cand_fset} + {cand_clf} (OOF, n_used={cand_n_used})")
    header = f"  {'bucket':<10} {'phenobias-fixed-120':>22} {'2-sided-selector(OOF)':>24} {'Δ pp':>10}"
    print(header)
    rows_csv = []
    for b in BUCKETS:
        a_n = ph_counts.get(b, 0)
        c_n = cand_counts.get(b, 0)
        a_pct = 100.0 * a_n / N_TOTAL
        c_pct = 100.0 * c_n / N_TOTAL
        d_pp = c_pct - a_pct
        print(f"  {b:<10} {a_n:>5}  ({a_pct:>6.2f}%) {c_n:>9}  ({c_pct:>6.2f}%) {d_pp:>+9.2f}")
        rows_csv.append({"bucket": b,
                          "phenobias_fixed120_n": a_n, "phenobias_fixed120_pct": a_pct,
                          "selector_oof_n": c_n,      "selector_oof_pct": c_pct,
                          "delta_pp": d_pp})
    # totals row
    a_total_pct = 100.0 * ph_n_match / N_TOTAL
    c_total_pct = 100.0 * cand_n_match / N_TOTAL
    print(f"  {'(matched)':<10} {ph_n_match:>5}  ({a_total_pct:>6.2f}%) "
          f"{cand_n_match:>9}  ({c_total_pct:>6.2f}%)")
    a_failed = ph_counts.get("MISSED", 0) + ph_counts.get("TOO_LATE", 0) + (N_TOTAL - ph_n_match)
    c_failed = cand_counts.get("MISSED", 0) + cand_counts.get("TOO_LATE", 0) + (N_TOTAL - cand_n_match)
    print(f"  P_ideal_overall :  phenobias-fixed-120 = {ph_counts.get('IDEAL', 0) / N_TOTAL * 100:.2f}%   "
          f"selector(OOF) = {cand_counts.get('IDEAL', 0) / N_TOTAL * 100:.2f}%   "
          f"Δ = {(cand_counts.get('IDEAL', 0) - ph_counts.get('IDEAL', 0)) / N_TOTAL * 100:+.2f}pp")
    print(f"  P_failed_overall:  phenobias-fixed-120 = {a_failed / N_TOTAL * 100:.2f}%   "
          f"selector(OOF) = {c_failed / N_TOTAL * 100:.2f}%   "
          f"Δ = {(c_failed - a_failed) / N_TOTAL * 100:+.2f}pp")

    csv_path = out_dir / "lead_bucket_comparison.csv"
    pd.DataFrame(rows_csv).to_csv(csv_path, index=False)
    print(f"  [csv] {csv_path}")

    # Operational change justification:
    p_ideal_a = ph_counts.get("IDEAL", 0) / N_TOTAL
    p_ideal_c = cand_counts.get("IDEAL", 0) / N_TOTAL
    if p_ideal_c > p_ideal_a:
        print(f"  → Selector P_ideal_overall ({p_ideal_c:.4f}) > current op ({p_ideal_a:.4f}) "
              f"by {(p_ideal_c - p_ideal_a) * 100:+.2f}pp : operational change CAN be justified on lead position.")
    else:
        print(f"  → Selector P_ideal_overall ({p_ideal_c:.4f}) <= current op ({p_ideal_a:.4f}) "
              f"by {(p_ideal_c - p_ideal_a) * 100:+.2f}pp : selector wins on IoU but loses on lead position. "
              f"Operational change NOT justified by P_ideal alone — diagnose lead shift / per-class lead distribution.")


def decision_block(model_results: list[dict]) -> None:
    """Decision uses OOF IoU_overall (denom N_TOTAL_TEST), comparable to baselines.

    Anchor: max(fixed60, fixed120).  Δ thresholds: >+0.02 GO, >0 MARGINAL, ≤0 NO-GO.
    Per-fold conditional IoU mean is reported alongside as a stability check.
    """
    print("\n=================== Decision (per model × feature_set, OOF) ===================")
    for r in model_results:
        b = r["baselines"]
        anchor = max(b["iou_fixed60"], b["iou_fixed120"])
        print(f"\n[{r['model']}]  anchor=max(fixed60,fixed120)={anchor:.4f}  "
              f"(fixed60={b['iou_fixed60']:.4f}, fixed120={b['iou_fixed120']:.4f}, "
              f"oracle={b['iou_oracle']:.4f})")
        for feature_set, payload in r["by_set"].items():
            best_iou, best_clf, best_cv = -1.0, "n/a", None
            for clf, cv in payload["cv"].items():
                if cv["iou_overall_mean"] > best_iou:
                    best_iou = cv["iou_overall_mean"]
                    best_clf = clf
                    best_cv = cv
            delta = best_iou - anchor
            if delta > 0.02:
                verdict = "GO"
            elif delta > 0.0:
                verdict = "MARGINAL"
            else:
                verdict = "NO-GO"
            cond_mean = best_cv["fold_cond_iou_mean"] if best_cv is not None else float("nan")
            cond_std  = best_cv["fold_cond_iou_std"]  if best_cv is not None else float("nan")
            acc = best_cv["accuracy_mean"] if best_cv is not None else float("nan")
            print(f"  feature_set={feature_set:<22} best_clf={best_clf:<6} "
                  f"IoU_oof={best_iou:.4f}  Δ={delta:+.4f}  → {verdict}  "
                  f"(fold_cond_IoU={cond_mean:.4f}±{cond_std:.4f}  acc={acc:.3f})")
        # OOF predicted offset distribution per best feature_set (informational only)
        for feature_set, payload in r["by_set"].items():
            for clf, cv in payload["cv"].items():
                print(f"    {feature_set}/{clf} OOF pred dist: " +
                      " ".join(f"{o}:{n}" for o, n in sorted(cv["oof_pred_dist"].items())))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, default="sheath_blight")
    p.add_argument("--run", type=int, default=4)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--sample_grid", type=str,
                   default="rice/outputs/diag/phase_r_sample_grid.csv")
    p.add_argument("--out_dir", type=str, default="rice/outputs/phase_s2/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shift", type=float, default=30.0)
    p.add_argument("--model", type=str, default="all",
                   help="'all' or comma-separated model substrings to match grid 'model' column.")
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if not os.path.exists(args.sample_grid):
        raise SystemExit(f"[abort] sample_grid not found: {args.sample_grid}")
    grid = pd.read_csv(args.sample_grid)
    grid_models = list(grid["model"].unique())
    print(f"[input] sample_grid rows={len(grid)}  models={grid_models}")

    if args.model.strip() == "all":
        labels_to_process = grid_models
    else:
        wants = [s.strip() for s in args.model.split(",") if s.strip()]
        labels_to_process = [m for m in grid_models if any(w in m for w in wants)]
    print(f"[plan] process {len(labels_to_process)} model(s): {labels_to_process}")
    if not labels_to_process:
        raise SystemExit("[abort] no models match --model filter")

    _ = resolve_pest(args.pest)
    print("\n[stage1] computing calibrated scores + per-sample features…")
    test_seas, test_s, p_test_cal, tau = stage1_test_scores(
        Path(args.stage1_ckpt), args.run, args)
    tstar_abs_map = {}
    for sid, sub in grid.groupby("sample_id"):
        r = sub.iloc[0]
        tstar_abs_map[str(sid)] = int(r["t_star_doy"])
    s1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    doy_start = int(s1_ckpt.get("doy_start", 1))
    stage1_features = compute_stage1_features_per_sample(
        test_s, p_test_cal, tstar_abs_map, doy_start)
    print(f"[stage1 features] rows={len(stage1_features)}")

    model_results = []
    for label in labels_to_process:
        gm = grid[grid["model"] == label].copy()
        if gm.empty:
            print(f"[warn] no sample_grid rows for label '{label}', skip")
            continue
        r = process_model(label, gm, stage1_features, args)
        model_results.append(r)

    summary_df = write_outputs(model_results, out_dir)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print("\n=================== Phase S2 summary ===================")
    print(summary_df.to_string(index=False))
    decision_block(model_results)
    lead_bucket_comparison(model_results, grid, args, out_dir)


if __name__ == "__main__":
    main()
