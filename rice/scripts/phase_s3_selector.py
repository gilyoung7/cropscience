"""
Phase S3 — Offset action-space expansion + lead-aware features.

Builds on phase_s2 by:
    (A) Allowing the 2-sided selector to choose from an expanded offset set
        {30, 60, 90, 105, 120, 150, 180} instead of the original 4-class set.
    (B) Adding a 'lead-aware' feature set v4:
            v3 (mu_only) + [alert_tstar_doy, year, mu_off60_minus_120,
                            mu_off120_div_off60]

Three cells are compared (all on the 2-sided model):
    C_old    : v3_mu_only,   offsets={60,90,105,120}   (phase_s2 original; 4-way)
    C_new_v3 : v3_mu_only,   offsets=expanded          (7-way)
    C_new_v4 : v4_lead_aware,offsets=expanded          (7-way)

Each cell evaluated with 5-fold stratified CV → OOF predictions →
IoU_overall (denom 575) + lead-bin decomposition (lead = L − (alert+60),
bins 15-30/31-45/46-60/61-90/91-120).

Inputs:
    outputs_phase_r_sample_grid.csv  — must contain the 2-sided rows for the
    expanded offset set (phase_r --per_model_extra_offsets).

Outputs:
    outputs/phase_s3/selector_summary.csv
    outputs/phase_s3/lead_bin_decomp.csv
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

from rice.src.pest_resolver import resolve_pest
from rice.src.train_eval import overlap_metrics
from rice.scripts.phase_s_selector import (
    N_TOTAL_TEST, P_IDEAL_LOW, P_IDEAL_HIGH,
    stage1_test_scores, compute_stage1_features_per_sample,
    target_encode_site,
)


OFFSETS_ORIG = [60, 90, 105, 120]
OFFSETS_EXT = [30, 60, 90, 105, 120, 150, 180]

STAGE1_FEATURES = [
    "score_at_tstar", "score_peak_to_tstar", "score_mean_28d_before",
    "score_slope_14d", "score_auc_to_tstar",
]
LEAD_BINS = [(15, 30), (31, 45), (46, 60), (61, 90), (91, 120)]
LEAD_BIN_NAMES = ["15-30", "31-45", "46-60", "61-90", "91-120"]


def lead_bin_of(lead):
    if not np.isfinite(lead):
        return "NA"
    for name, (lo, hi) in zip(LEAD_BIN_NAMES, LEAD_BINS):
        if lo <= lead <= hi:
            return name
    if lead < LEAD_BINS[0][0]:
        return "<15"
    return ">120"


def build_per_sample_table_ext(grid_model: pd.DataFrame, offsets: list[int]) -> pd.DataFrame:
    """Wide per-sample table covering an arbitrary offset set.

    Output columns (per sample_id):
        sample_id, site, year, t_star_doy, true_event_doy, L, R, sigma,
        iou_off{o},  mu_off{o},  matched_off{o}    for each o in `offsets`
        mu_default, lead_predicted, best_offset (target), n_offsets_matched
    """
    rows = []
    static_cols = ["sample_id", "site", "year", "t_star_doy", "true_event_doy",
                   "L", "R", "sigma"]
    for sample_id, sub in grid_model.groupby("sample_id"):
        d = {c: sub[c].iloc[0] for c in static_cols}
        off_iou, off_mu, off_match = {}, {}, {}
        for _, r in sub.iterrows():
            o = int(r["offset"])
            off_iou[o] = float(r["iou_matched"])
            off_mu[o] = float(r["mu"]) if pd.notna(r["mu"]) else float("nan")
            off_match[o] = bool(r["matched"])
        for o in offsets:
            d[f"iou_off{o}"] = off_iou.get(o, 0.0)
            d[f"mu_off{o}"]  = off_mu.get(o, float("nan"))
            d[f"matched_off{o}"] = bool(off_match.get(o, False))
        # Default mu = first matched offset, scanning a sensible cascade
        mu_def = None
        for cand in (90, 105, 120, 60, 75, 135, 150, 30, 180):
            if cand in off_mu and np.isfinite(off_mu[cand]):
                mu_def = off_mu[cand]; break
        d["mu_default"] = float(mu_def) if mu_def is not None else float("nan")
        d["lead_predicted"] = (d["mu_default"] - float(d["t_star_doy"])
                                if np.isfinite(d["mu_default"]) else float("nan"))
        d["n_offsets_matched"] = int(sum(off_match.get(o, False) for o in offsets))
        # Target: argmax IoU over `offsets`. Tie-break: shorter offset preferred.
        best_iou = -1.0; best_off = None
        for o in offsets:
            iou = off_iou.get(o, 0.0)
            if iou > best_iou + 1e-12:
                best_iou = iou; best_off = o
        d["best_offset"] = int(best_off) if best_off is not None else int(offsets[0])
        d["best_iou"] = float(best_iou) if best_iou >= 0 else 0.0
        rows.append(d)
    return pd.DataFrame(rows)


def attach_features_ext(per_sample: pd.DataFrame, offsets: list[int]) -> pd.DataFrame:
    """Adds mu_off* missing indicators, spread/ratio, and lead-aware static cols."""
    df = per_sample.copy()
    for o in offsets:
        df[f"mu_off{o}_missing"] = (~np.isfinite(df[f"mu_off{o}"].to_numpy(dtype=float))).astype(int)
    mu_arr = df[[f"mu_off{o}" for o in offsets]].to_numpy(dtype=float)
    df["mu_range_4off"] = np.nanmax(mu_arr, axis=1) - np.nanmin(mu_arr, axis=1)
    df["mu_mean_4off"] = np.nanmean(mu_arr, axis=1)
    if 60 in offsets and 120 in offsets:
        df["mu_off60_minus_120"] = df["mu_off60"] - df["mu_off120"]
        with np.errstate(divide="ignore", invalid="ignore"):
            df["mu_off120_div_off60"] = df["mu_off120"] / df["mu_off60"]
    else:
        df["mu_off60_minus_120"] = float("nan")
        df["mu_off120_div_off60"] = float("nan")
    if 60 in offsets:
        df["mu_off60_minus_default"] = df["mu_off60"] - df["mu_default"]
    return df


def feature_cols_for_set(feature_set: str, offsets: list[int]) -> tuple[list[str], list[str]]:
    """Returns (feat_cols, impute_cols). impute_cols are NaN-tolerant via train median."""
    mu_offset_cols = [f"mu_off{o}" for o in offsets]
    mu_miss_cols = [f"mu_off{o}_missing" for o in offsets]
    spread_cols = ["mu_range_4off", "mu_mean_4off"]
    static_cols = ["tstar_doy_feat", "year_feat"]
    lead_aware_extra = ["alert_tstar_doy", "mu_off60_minus_120", "mu_off120_div_off60"]
    if feature_set == "v3_mu_only":
        cols = mu_offset_cols + spread_cols + mu_miss_cols + static_cols
        impute = mu_offset_cols + spread_cols + mu_miss_cols
    elif feature_set == "v4_lead_aware":
        cols = (mu_offset_cols + spread_cols + mu_miss_cols + static_cols
                + lead_aware_extra)
        impute = mu_offset_cols + spread_cols + mu_miss_cols + lead_aware_extra
    else:
        raise ValueError(f"unknown feature_set: {feature_set}")
    return cols, impute


def impute_train_median(train_X: np.ndarray, test_X: np.ndarray,
                         impute_cols_idx: list[int]) -> None:
    for j in impute_cols_idx:
        col_train = train_X[:, j]
        finite = np.isfinite(col_train)
        med = float(np.nanmedian(col_train[finite])) if finite.any() else 0.0
        train_X[~np.isfinite(train_X[:, j]), j] = med
        test_X[~np.isfinite(test_X[:, j]), j] = med


def evaluate_oof(per_sample: pd.DataFrame, pred_offsets: np.ndarray,
                 shift: float, offsets: list[int]) -> dict:
    """Compute IoU_overall (denom N_TOTAL_TEST) + lead-bin decomposition."""
    iou_sum = 0.0
    n_ideal = 0
    bucket_counts = {n: 0 for n in (
        "MISSED", "TOO_LATE", "URGENT", "IDEAL", "ADVANCE", "TOO_EARLY")}
    per_sample_rows = []
    for (_, row), pred_off in zip(per_sample.iterrows(), pred_offsets):
        o = int(pred_off)
        iou_v = float(row.get(f"iou_off{o}", 0.0))
        if not np.isfinite(iou_v):
            iou_v = 0.0
        iou_sum += iou_v
        mu_at = row.get(f"mu_off{o}")
        sigma = float(row["sigma"]) if pd.notna(row["sigma"]) else 5.0
        L = float(row["L"]) if pd.notna(row["L"]) else float("nan")
        R = float(row["R"]) if pd.notna(row["R"]) else float("nan")
        t_star = float(row["t_star_doy"])
        # bucket lead = L − (mu + 1.96σ − shift)
        if (mu_at is None) or not np.isfinite(mu_at) or not np.isfinite(L):
            bucket_lead = float("nan")
        else:
            bucket_lead = L - (float(mu_at) + 1.96 * sigma - float(shift))
        # IoU at shift=0 (PI vs [L,R])
        if (mu_at is None) or not np.isfinite(mu_at) or not np.isfinite(L):
            iou_pi_lr = 0.0
        else:
            HW = 1.96 * sigma
            pL = int(round(float(mu_at) - HW))
            pR = int(round(float(mu_at) + HW))
            iou_pi_lr, _, _ = overlap_metrics(pL, pR, int(L), int(R))
        # lead bin grouping: L − (alert_tstar + 60), sample-intrinsic
        lead_bin_value = L - (t_star + 60.0) if np.isfinite(L) else float("nan")
        lead_bin = lead_bin_of(lead_bin_value)
        # bucket count (operational shift-based)
        if np.isfinite(bucket_lead):
            if bucket_lead < 0:        bucket_counts["MISSED"] += 1
            elif bucket_lead < 7:      bucket_counts["TOO_LATE"] += 1
            elif bucket_lead < 14:     bucket_counts["URGENT"] += 1
            elif bucket_lead < 30:     bucket_counts["IDEAL"] += 1
            elif bucket_lead < 45:     bucket_counts["ADVANCE"] += 1
            else:                      bucket_counts["TOO_EARLY"] += 1
            if P_IDEAL_LOW <= bucket_lead < P_IDEAL_HIGH:
                n_ideal += 1
        per_sample_rows.append({
            "sample_id": row.get("sample_id", "?"),
            "pred_off": o,
            "lead_bin_anchor60": lead_bin,
            "iou_pi_lr_shift0": float(iou_pi_lr),
            "bucket_lead_shift": float(bucket_lead),
            "L": L, "R": R, "t_star_doy": t_star, "sigma": sigma,
            "mu_at_pred_off": float(mu_at) if np.isfinite(mu_at) else float("nan"),
        })
    return {
        "IoU_overall": iou_sum / N_TOTAL_TEST,
        "P_ideal_overall": n_ideal / N_TOTAL_TEST,
        "bucket_counts": bucket_counts,
        "per_sample": pd.DataFrame(per_sample_rows),
    }


def run_cv_cell(per_sample: pd.DataFrame, feat_cols: list[str], impute_cols: list[str],
                target_col: str, classifier: str, args, shift: float, offsets: list[int]) -> dict:
    """5-fold stratified CV. Returns OOF metrics + lead-bin breakdown."""
    impute_idx = [feat_cols.index(c) for c in impute_cols if c in feat_cols]
    classes = sorted(per_sample[target_col].astype(int).unique().tolist())
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    y_all = per_sample[target_col].to_numpy(dtype=int)

    fold_acc, oof_rows, oof_preds, feat_imp_runs = [], [], [], []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(per_sample)), y_all)):
        train_df = per_sample.iloc[tr_idx].copy()
        test_df = per_sample.iloc[te_idx].copy()
        tr_site_enc, te_site_enc, _ = target_encode_site(train_df, test_df, target_col=target_col)
        X_tr = train_df[feat_cols].to_numpy(dtype=float)
        X_te = test_df[feat_cols].to_numpy(dtype=float)
        impute_train_median(X_tr, X_te, impute_idx)
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
            uniq, cnts = np.unique(y_tr, return_counts=True)
            inv = {int(c): float(len(y_tr)) / (len(uniq) * cnt) for c, cnt in zip(uniq, cnts)}
            sw = np.asarray([inv[int(c)] for c in y_tr], dtype=float)
            label_map = {c: i for i, c in enumerate(classes)}
            inv_label = {i: c for c, i in label_map.items()}
            y_tr_idx = np.asarray([label_map[int(c)] for c in y_tr], dtype=int)
            clf = XGBClassifier(
                max_depth=3, n_estimators=200, learning_rate=0.1,
                objective="multi:softprob", eval_metric="mlogloss",
                random_state=args.seed, verbosity=0, num_class=len(classes),
            )
            clf.fit(X_tr, y_tr_idx, sample_weight=sw)
            y_pred_idx = clf.predict(X_te)
            y_pred = np.asarray([inv_label[int(p)] for p in y_pred_idx], dtype=int)
            feat_imp_runs.append(clf.feature_importances_.tolist())
        else:
            raise SystemExit(f"unknown classifier: {classifier}")
        fold_acc.append(accuracy_score(y_te, y_pred))
        oof_rows.append(test_df.reset_index(drop=True))
        oof_preds.append(np.asarray(y_pred, dtype=int))
        if fold == 0:
            uniq_pred, cnt_pred = np.unique(y_pred, return_counts=True)
            print(f"      [fold0 pred dist] " +
                  " ".join(f"{int(u)}:{int(c)}" for u, c in zip(uniq_pred, cnt_pred)))

    oof_df = pd.concat(oof_rows, axis=0, ignore_index=True)
    oof_pred_arr = np.concatenate(oof_preds, axis=0)
    eval_m = evaluate_oof(oof_df, oof_pred_arr, shift=shift, offsets=offsets)
    print(f"      [OOF] IoU_overall={eval_m['IoU_overall']:.4f}  "
          f"P_ideal_overall={eval_m['P_ideal_overall']:.4f}  "
          f"acc={float(np.mean(fold_acc)):.4f}")
    return {
        "iou_overall": eval_m["IoU_overall"],
        "p_ideal_overall": eval_m["P_ideal_overall"],
        "accuracy_mean": float(np.mean(fold_acc)),
        "bucket_counts": eval_m["bucket_counts"],
        "classes": classes,
        "feat_imp_mean": (np.mean(np.stack(feat_imp_runs, axis=0), axis=0).tolist()
                          if feat_imp_runs else None),
        "feat_cols": feat_cols + ["site_target_enc"],
        "oof_pred_dist": dict(zip(*np.unique(oof_pred_arr, return_counts=True))),
        "per_sample_eval": eval_m["per_sample"],   # incl. lead_bin_anchor60, iou_pi_lr_shift0
    }


def lead_bin_decomp(per_sample_eval: pd.DataFrame) -> pd.DataFrame:
    """Per lead-bin (anchor=alert+60): n, IoU mean."""
    bin_order = LEAD_BIN_NAMES + ["<15", ">120"]
    rows = []
    for b in bin_order:
        sub = per_sample_eval[per_sample_eval["lead_bin_anchor60"] == b]
        n = len(sub)
        if n == 0:
            rows.append({"lead_bin": b, "n": 0, "IoU_mean": float("nan"),
                         "iou_sum": 0.0, "contrib_to_overall": 0.0})
            continue
        iou_mean = float(sub["iou_pi_lr_shift0"].mean())
        iou_sum = float(sub["iou_pi_lr_shift0"].sum())
        rows.append({"lead_bin": b, "n": n,
                     "IoU_mean": iou_mean, "iou_sum": iou_sum,
                     "contrib_to_overall": iou_sum / N_TOTAL_TEST})
    return pd.DataFrame(rows)


def per_sample_build(grid_2sided: pd.DataFrame, stage1_features: pd.DataFrame,
                      offsets: list[int]) -> pd.DataFrame:
    """Build the per-sample table for the 2-sided model with the requested offset set."""
    per_sample = build_per_sample_table_ext(grid_2sided.copy(), offsets)
    per_sample = per_sample[per_sample["n_offsets_matched"] > 0].copy()
    per_sample = per_sample.merge(stage1_features, left_on="sample_id",
                                   right_index=True, how="left")
    per_sample["tstar_doy_feat"] = per_sample["t_star_doy"].astype(float)
    per_sample["year_feat"] = per_sample["year"].astype(float)
    per_sample["alert_tstar_doy"] = per_sample["t_star_doy"].astype(float)   # alias
    per_sample = attach_features_ext(per_sample, offsets)
    bad = per_sample[STAGE1_FEATURES + ["sigma", "L", "t_star_doy"]].isna().any(axis=1)
    if bad.any():
        print(f"  dropped stage1/static-NaN rows: {int(bad.sum())}")
        per_sample = per_sample[~bad].copy()
    return per_sample


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, default="sheath_blight")
    p.add_argument("--run", type=int, default=4)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--sample_grid", type=str, default="rice/outputs/diag/phase_r_sample_grid.csv")
    p.add_argument("--out_dir", type=str, default="rice/outputs/phase_s3/")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--shift", type=float, default=30.0)
    p.add_argument("--target_label", type=str, default="2-sided",
                   help="Substring matched into sample_grid 'model' column to pick the 2-sided rows.")
    args = p.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.sample_grid):
        raise SystemExit(f"[abort] sample_grid not found: {args.sample_grid}")
    grid = pd.read_csv(args.sample_grid)
    grid_models = list(grid["model"].unique())
    print(f"[input] sample_grid rows={len(grid)}  models={grid_models}")

    label = [m for m in grid_models if args.target_label in m]
    if not label:
        raise SystemExit(f"no model matches '{args.target_label}'")
    label = label[0]
    grid_2sided = grid[grid["model"] == label].copy()
    avail_offs = sorted(int(x) for x in grid_2sided["offset"].unique())
    print(f"[2-sided '{label}'] rows={len(grid_2sided)}  offsets in grid={avail_offs}")

    # Tables: original 4-offset, expanded 7-offset (using whatever offsets are present)
    expanded_offs = sorted(set(avail_offs) | set(OFFSETS_EXT) & set(avail_offs))
    if not all(o in avail_offs for o in OFFSETS_EXT):
        missing = [o for o in OFFSETS_EXT if o not in avail_offs]
        print(f"  [warn] expanded set has missing offsets in grid: {missing}; "
              f"using intersection = {expanded_offs}")
        expanded_offs = [o for o in OFFSETS_EXT if o in avail_offs]

    _ = resolve_pest(args.pest)
    print("\n[stage1] computing calibrated scores + per-sample features…")
    test_seas, test_s, p_test_cal, tau = stage1_test_scores(
        Path(args.stage1_ckpt), args.run, args)
    tstar_abs_map = {str(sid): int(sub.iloc[0]["t_star_doy"])
                     for sid, sub in grid.groupby("sample_id")}
    s1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    doy_start = int(s1_ckpt.get("doy_start", 1))
    stage1_features = compute_stage1_features_per_sample(
        test_s, p_test_cal, tstar_abs_map, doy_start)
    print(f"[stage1 features] rows={len(stage1_features)}")

    print("\n----- build per_sample (original 4-offset) -----")
    ps_orig = per_sample_build(grid_2sided, stage1_features, OFFSETS_ORIG)
    print(f"  ps_orig n={len(ps_orig)}  target_classes={sorted(ps_orig['best_offset'].unique())}")
    print(f"\n----- build per_sample (expanded {expanded_offs}) -----")
    ps_ext = per_sample_build(grid_2sided, stage1_features, expanded_offs)
    print(f"  ps_ext  n={len(ps_ext)}  target_classes={sorted(ps_ext['best_offset'].unique())}")
    print(f"  target distribution (ext): {dict(ps_ext['best_offset'].value_counts().sort_index())}")

    print("\n=================== Cell C_old (v3_mu_only, 4-way) ===================")
    feat_v3_orig, imp_v3_orig = feature_cols_for_set("v3_mu_only", OFFSETS_ORIG)
    cv_c_old = run_cv_cell(ps_orig, feat_v3_orig, imp_v3_orig,
                            "best_offset", "logreg", args,
                            shift=args.shift, offsets=OFFSETS_ORIG)

    print("\n=================== Cell C_new_v3 (v3_mu_only, expanded) ===================")
    feat_v3_ext, imp_v3_ext = feature_cols_for_set("v3_mu_only", expanded_offs)
    cv_c_new_v3 = run_cv_cell(ps_ext, feat_v3_ext, imp_v3_ext,
                               "best_offset", "logreg", args,
                               shift=args.shift, offsets=expanded_offs)

    print("\n=================== Cell C_new_v4 (v4_lead_aware, expanded) ===================")
    feat_v4_ext, imp_v4_ext = feature_cols_for_set("v4_lead_aware", expanded_offs)
    cv_c_new_v4 = run_cv_cell(ps_ext, feat_v4_ext, imp_v4_ext,
                               "best_offset", "logreg", args,
                               shift=args.shift, offsets=expanded_offs)

    # Summary table
    rows = []
    for cell_name, cell, ps_used, fset in [
        ("C_old",       cv_c_old,    ps_orig, "v3_mu_only"),
        ("C_new_v3",    cv_c_new_v3, ps_ext,  "v3_mu_only"),
        ("C_new_v4",    cv_c_new_v4, ps_ext,  "v4_lead_aware"),
    ]:
        rows.append({
            "cell": cell_name,
            "feature_set": fset,
            "offset_set_size": len(OFFSETS_ORIG if cell_name == "C_old" else expanded_offs),
            "n_used": len(ps_used),
            "n_classes": len(cell["classes"]),
            "IoU_overall": cell["iou_overall"],
            "P_ideal_overall": cell["p_ideal_overall"],
            "accuracy_mean": cell["accuracy_mean"],
            "oof_pred_dist": str(cell["oof_pred_dist"]),
        })
    summary = pd.DataFrame(rows)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print("\n=================== Phase S3 summary ===================")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "selector_summary.csv", index=False)
    print(f"\n[csv] {out_dir}/selector_summary.csv")

    # Lead-bin decomposition (all cells side-by-side)
    print("\n=================== Lead-bin decomposition (anchor = alert+60, IoU shift=0, PI 95%) ===================")
    decomp_C_old    = lead_bin_decomp(cv_c_old["per_sample_eval"]).rename(
        columns={"n": "n_C_old", "IoU_mean": "IoU_C_old",
                  "iou_sum": "iou_sum_C_old", "contrib_to_overall": "contrib_C_old"})
    decomp_C_new_v3 = lead_bin_decomp(cv_c_new_v3["per_sample_eval"]).rename(
        columns={"n": "n_C_new_v3", "IoU_mean": "IoU_C_new_v3",
                  "iou_sum": "iou_sum_C_new_v3", "contrib_to_overall": "contrib_C_new_v3"})
    decomp_C_new_v4 = lead_bin_decomp(cv_c_new_v4["per_sample_eval"]).rename(
        columns={"n": "n_C_new_v4", "IoU_mean": "IoU_C_new_v4",
                  "iou_sum": "iou_sum_C_new_v4", "contrib_to_overall": "contrib_C_new_v4"})
    decomp = decomp_C_old.merge(decomp_C_new_v3, on="lead_bin", how="outer") \
                          .merge(decomp_C_new_v4, on="lead_bin", how="outer")
    decomp["delta_v3_minus_old"] = decomp["IoU_C_new_v3"] - decomp["IoU_C_old"]
    decomp["delta_v4_minus_old"] = decomp["IoU_C_new_v4"] - decomp["IoU_C_old"]
    decomp["delta_v4_minus_v3"]  = decomp["IoU_C_new_v4"] - decomp["IoU_C_new_v3"]
    print(decomp.to_string(index=False))
    decomp.to_csv(out_dir / "lead_bin_decomp.csv", index=False)
    print(f"\n[csv] {out_dir}/lead_bin_decomp.csv")

    # Per-sample dumps (for follow-up analysis)
    for cell_name, cell in [("C_old", cv_c_old),
                             ("C_new_v3", cv_c_new_v3),
                             ("C_new_v4", cv_c_new_v4)]:
        cell["per_sample_eval"].to_csv(out_dir / f"per_sample_{cell_name}.csv", index=False)
    print(f"[csv] {out_dir}/per_sample_*.csv  (per-cell oof predictions + lead-bin)")

    # Headline lines for stdout
    print("\n=================== Decision (anchor = max baseline IoU) ===================")
    # We need anchors from baselines; pull them from the grid (matching sample_grid columns).
    # For simplicity, pick the maximum IoU_overall across (baseline, phenobias) @ off=120
    # using the existing sample_grid σ; this mirrors phase_s2 anchor logic.
    anchor_iou = None
    for src_label in grid_models:
        if src_label == label:
            continue
        sub_src = grid[grid["model"] == src_label]
        # Take rows at offset=120 with matched=True
        sub120 = sub_src[(sub_src["offset"] == 120) & (sub_src["matched"] == True)]
        if sub120.empty:
            continue
        ious = []
        for _, r in sub120.iterrows():
            mu = float(r["mu"]); sigma = float(r["sigma"])
            L = int(r["L"]); R = int(r["R"])
            HW = 1.96 * sigma
            iou, _, _ = overlap_metrics(int(round(mu - HW)), int(round(mu + HW)), L, R)
            ious.append(iou)
        if not ious:
            continue
        iou_ov = sum(ious) / N_TOTAL_TEST
        print(f"  baseline anchor : {src_label!r} fixed off=120 → IoU_overall = {iou_ov:.4f}")
        if anchor_iou is None or iou_ov > anchor_iou:
            anchor_iou = iou_ov
    if anchor_iou is not None:
        print(f"  anchor (max) = {anchor_iou:.4f}")
        for r in rows:
            delta = r["IoU_overall"] - anchor_iou
            verdict = "GO" if delta > 0.02 else ("MARGINAL" if delta > 0.0 else "NO-GO")
            print(f"  [{r['cell']:<10}] IoU={r['IoU_overall']:.4f}  Δ={delta:+.4f}  → {verdict}")


if __name__ == "__main__":
    main()
