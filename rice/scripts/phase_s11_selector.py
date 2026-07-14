"""Phase S11 Step 3 — 8-way (model × offset) selector.

Builds a per-sample wide table that combines both models' mu predictions
over 4 offsets (8 mu_off* columns), then trains a logreg classifier OOF to
pick the per-sample best (model, offset) class.

Classes (8 total):
    0..3 : baseline 2-sided × {60, 90, 105, 120}
    4..7 : center_ra (alt)  × {60, 90, 105, 120}

Features (v3-style, extended for two models):
    8 × mu_off{60,90,105,120}_{base,alt}
    8 × mu_off{60,90,105,120}_{base,alt}_missing (NaN indicators)
    mu_range_4off_base, mu_mean_4off_base
    mu_range_4off_alt,  mu_mean_4off_alt
    tstar_doy_feat, year_feat
    site_target_enc (per-fold target encoding, leak-free)

OOF metrics:
    IoU_overall (denom = N_TOTAL_TEST = 575)
    bucket distribution
    Lead-bin decomposition (anchor = alert+60)

Outputs (out_dir, default outputs/phase_s11/):
    selector_summary.csv      — single line summary (IoU, accuracy, bucket counts)
    lead_bin_decomp.csv       — per-bin contrib to overall IoU
    oof_predictions.csv       — per-sample OOF predicted class + ground-truth best
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from rice.src.pest_resolver import resolve_pest
from rice.src.train_eval import overlap_metrics
from rice.scripts.phase_s_selector import (
    N_TOTAL_TEST, P_IDEAL_LOW, P_IDEAL_HIGH,
    stage1_test_scores, compute_stage1_features_per_sample,
    target_encode_site,
)
from rice.scripts.phase_s3_selector import (
    STAGE1_FEATURES, LEAD_BIN_NAMES, LEAD_BINS, lead_bin_of,
)


OFFSETS = [60, 90, 105, 120]


def _wide_per_sample(grid_one_model: pd.DataFrame, suffix: str) -> pd.DataFrame:
    """Build a wide per-sample table for one model with suffix '_base' or '_alt'."""
    rows = []
    static_cols = ["sample_id", "site", "year", "t_star_doy", "true_event_doy",
                   "L", "R", "sigma"]
    for sid, sub in grid_one_model.groupby("sample_id"):
        d = {c: sub[c].iloc[0] for c in static_cols}
        for o in OFFSETS:
            r = sub[sub["offset"] == int(o)]
            if r.empty:
                d[f"iou_off{o}_{suffix}"] = 0.0
                d[f"mu_off{o}_{suffix}"] = float("nan")
                d[f"matched_off{o}_{suffix}"] = False
            else:
                d[f"iou_off{o}_{suffix}"] = float(r["iou_matched"].iloc[0])
                d[f"mu_off{o}_{suffix}"] = (float(r["mu"].iloc[0])
                                            if pd.notna(r["mu"].iloc[0]) else float("nan"))
                d[f"matched_off{o}_{suffix}"] = bool(r["matched"].iloc[0])
        rows.append(d)
    return pd.DataFrame(rows)


def build_combined_per_sample(grid: pd.DataFrame, base_label: str, alt_label: str,
                               stage1_features: pd.DataFrame) -> Tuple[pd.DataFrame, list[str], list[str]]:
    """Wide per-sample table with both models side-by-side + target (best class 0..7)."""
    g_base = grid[grid["model"] == base_label].copy()
    g_alt = grid[grid["model"] == alt_label].copy()
    if g_base.empty:
        raise SystemExit(f"no rows for baseline label {base_label!r}")
    if g_alt.empty:
        raise SystemExit(f"no rows for alt label {alt_label!r}")
    ps_base = _wide_per_sample(g_base, "base")
    ps_alt = _wide_per_sample(g_alt, "alt")

    # Outer join on sample_id to capture the union cohort. Prefer baseline's
    # static columns; fall back to alt where baseline missing.
    ps = ps_base.merge(ps_alt, on="sample_id", how="outer", suffixes=("", "_altdup"))
    for c in ("site", "year", "t_star_doy", "true_event_doy", "L", "R", "sigma"):
        ps[c] = ps[c].combine_first(ps.get(f"{c}_altdup"))
    drop_dups = [c for c in ps.columns if c.endswith("_altdup")]
    ps = ps.drop(columns=drop_dups)

    # Attach stage1 features.
    ps = ps.merge(stage1_features, left_on="sample_id", right_index=True, how="left")
    ps["tstar_doy_feat"] = ps["t_star_doy"].astype(float)
    ps["year_feat"] = ps["year"].astype(float)

    # Mu spread features per model.
    for suffix in ("base", "alt"):
        mu_cols = [f"mu_off{o}_{suffix}" for o in OFFSETS]
        ps[f"mu_range_4off_{suffix}"] = (ps[mu_cols].max(axis=1, skipna=True)
                                          - ps[mu_cols].min(axis=1, skipna=True))
        ps[f"mu_mean_4off_{suffix}"] = ps[mu_cols].mean(axis=1, skipna=True)
        for o in OFFSETS:
            ps[f"mu_off{o}_{suffix}_missing"] = (~np.isfinite(
                ps[f"mu_off{o}_{suffix}"].to_numpy(dtype=float))).astype(int)

    # Drop samples with no matched (model, offset) at all (no learning signal).
    iou_cols = [f"iou_off{o}_{s}" for s in ("base", "alt") for o in OFFSETS]
    matched_cols = [f"matched_off{o}_{s}" for s in ("base", "alt") for o in OFFSETS]
    any_matched = ps[matched_cols].any(axis=1)
    ps = ps[any_matched].copy()
    # Drop samples missing the bare stage1 features.
    bad = ps[STAGE1_FEATURES + ["sigma", "L", "t_star_doy"]].isna().any(axis=1)
    if bad.any():
        print(f"  dropped {int(bad.sum())} samples missing stage1/static cols", flush=True)
        ps = ps[~bad].copy()

    # Target: argmax IoU over the 8 (model, offset) combos. Tie-break: prefer
    # baseline (lower class index) then shorter offset (lower offset value).
    class_names = []
    class_iou_col = []
    for s in ("base", "alt"):
        for o in OFFSETS:
            class_names.append(f"{s}_off{o}")
            class_iou_col.append(f"iou_off{o}_{s}")
    iou_mat = ps[class_iou_col].to_numpy(dtype=float)
    # NaN → 0 so argmax stable
    iou_mat = np.where(np.isfinite(iou_mat), iou_mat, 0.0)
    best_class = np.argmax(iou_mat, axis=1)
    ps["best_class"] = best_class
    ps["best_iou"] = iou_mat[np.arange(len(ps)), best_class]

    # Feature column lists for the classifier.
    mu_feat_cols = [f"mu_off{o}_{s}" for s in ("base", "alt") for o in OFFSETS]
    mu_miss_cols = [f"mu_off{o}_{s}_missing" for s in ("base", "alt") for o in OFFSETS]
    spread_cols = [f"mu_range_4off_{s}" for s in ("base", "alt")] + \
                  [f"mu_mean_4off_{s}" for s in ("base", "alt")]
    feat_cols = mu_feat_cols + spread_cols + mu_miss_cols + ["tstar_doy_feat", "year_feat"]
    impute_cols = mu_feat_cols + spread_cols + mu_miss_cols
    return ps, feat_cols, impute_cols, class_names, class_iou_col


def impute_train_median(train_X: np.ndarray, test_X: np.ndarray,
                         impute_idx: list[int]) -> None:
    for j in impute_idx:
        col_train = train_X[:, j]
        finite = np.isfinite(col_train)
        med = float(np.nanmedian(col_train[finite])) if finite.any() else 0.0
        train_X[~np.isfinite(train_X[:, j]), j] = med
        test_X[~np.isfinite(test_X[:, j]), j] = med


def lead_bin_decomp(per_sample: pd.DataFrame, iou_col: str) -> pd.DataFrame:
    """Per lead-bin (anchor = alert+60): n, IoU mean."""
    bin_order = LEAD_BIN_NAMES + ["<15", ">120"]
    rows = []
    for b in bin_order:
        sub = per_sample[per_sample["lead_bin_anchor60"] == b]
        n = len(sub)
        if n == 0:
            rows.append({"lead_bin": b, "n": 0, "IoU_mean": float("nan"),
                         "iou_sum": 0.0, "contrib_to_overall": 0.0})
            continue
        iou_mean = float(sub[iou_col].mean())
        iou_sum = float(sub[iou_col].sum())
        rows.append({"lead_bin": b, "n": n, "IoU_mean": iou_mean,
                     "iou_sum": iou_sum,
                     "contrib_to_overall": iou_sum / N_TOTAL_TEST})
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", type=str, default="sheath_blight")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--stage1_ckpt", type=str, required=True)
    ap.add_argument("--sample_grid", type=str,
                    default="rice/outputs/diag/phase_s10_sample_grid.csv")
    ap.add_argument("--out_dir", type=str, default="rice/outputs/phase_s11/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--base_label_substr", type=str, default="(asym=25)")
    ap.add_argument("--alt_label_substr", type=str, default="center_ra")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.sample_grid):
        raise SystemExit(f"[abort] sample_grid not found: {args.sample_grid}")
    grid = pd.read_csv(args.sample_grid)
    models_in_grid = list(grid["model"].unique())
    base_match = [m for m in models_in_grid if args.base_label_substr in m
                  and args.alt_label_substr not in m]
    alt_match = [m for m in models_in_grid if args.alt_label_substr in m]
    if not base_match or not alt_match:
        raise SystemExit(f"[abort] base or alt missing in grid; got base={base_match} alt={alt_match}")
    base_label, alt_label = base_match[0], alt_match[0]
    print(f"[input] grid rows={len(grid)}", flush=True)
    print(f"  baseline label : {base_label}", flush=True)
    print(f"  alt label      : {alt_label}", flush=True)

    _ = resolve_pest(args.pest)
    print("\n[stage1] computing calibrated scores + per-sample features…", flush=True)
    test_seas, test_s, p_test_cal, tau = stage1_test_scores(
        Path(args.stage1_ckpt), args.run, args)
    tstar_abs_map = {str(sid): int(sub.iloc[0]["t_star_doy"])
                     for sid, sub in grid.groupby("sample_id")}
    s1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    doy_start = int(s1_ckpt.get("doy_start", 1))
    stage1_features = compute_stage1_features_per_sample(
        test_s, p_test_cal, tstar_abs_map, doy_start)
    print(f"[stage1 features] rows={len(stage1_features)}", flush=True)

    ps, feat_cols, impute_cols, class_names, class_iou_col = build_combined_per_sample(
        grid, base_label, alt_label, stage1_features)
    print(f"\n[combined per_sample] n={len(ps)}  classes={class_names}", flush=True)
    cls_dist = dict(ps["best_class"].value_counts().sort_index())
    print(f"  best_class dist = {cls_dist}", flush=True)
    print(f"  oracle IoU (within cohort) = {ps['best_iou'].sum() / N_TOTAL_TEST:.4f}  "
          f"(denom={N_TOTAL_TEST})", flush=True)

    # ----- 5-fold OOF logreg ------------------------------------------------
    impute_idx = [feat_cols.index(c) for c in impute_cols if c in feat_cols]
    y_all = ps["best_class"].to_numpy(dtype=int)
    classes_present = sorted(set(int(c) for c in y_all))
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.seed)
    fold_acc = []
    oof_rows = []
    oof_preds = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(np.zeros(len(ps)), y_all)):
        train_df = ps.iloc[tr_idx].copy()
        test_df = ps.iloc[te_idx].copy()
        tr_site_enc, te_site_enc, _ = target_encode_site(
            train_df, test_df, target_col="best_class")
        X_tr = train_df[feat_cols].to_numpy(dtype=float)
        X_te = test_df[feat_cols].to_numpy(dtype=float)
        impute_train_median(X_tr, X_te, impute_idx)
        X_tr = np.column_stack([X_tr, tr_site_enc.to_numpy()])
        X_te = np.column_stack([X_te, te_site_enc.to_numpy()])
        y_tr = train_df["best_class"].to_numpy(dtype=int)
        y_te = test_df["best_class"].to_numpy(dtype=int)
        clf = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(solver="lbfgs", class_weight="balanced",
                                       max_iter=2000, random_state=args.seed)),
        ])
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        fold_acc.append(accuracy_score(y_te, y_pred))
        oof_rows.append(test_df.reset_index(drop=True))
        oof_preds.append(np.asarray(y_pred, dtype=int))
        if fold == 0:
            uniq, cnt = np.unique(y_pred, return_counts=True)
            print(f"  [fold0 pred dist] " + " ".join(f"{int(u)}:{int(c)}" for u, c in zip(uniq, cnt)), flush=True)

    oof_df = pd.concat(oof_rows, axis=0, ignore_index=True)
    oof_pred = np.concatenate(oof_preds, axis=0)
    oof_df["pred_class"] = oof_pred
    # IoU at the predicted class = iou_off{offset}_{suffix} of that class
    iou_at_pred = []
    for _, row in oof_df.iterrows():
        cls = int(row["pred_class"])
        col = class_iou_col[cls]
        iou_at_pred.append(float(row.get(col, 0.0)))
    oof_df["iou_at_pred"] = iou_at_pred

    iou_overall = float(oof_df["iou_at_pred"].sum()) / float(N_TOTAL_TEST)
    accuracy_mean = float(np.mean(fold_acc))
    print("\n=================== Phase S11 selector OOF ===================", flush=True)
    print(f"  n_used (cohort) = {len(oof_df)}", flush=True)
    print(f"  accuracy_mean   = {accuracy_mean:.4f}", flush=True)
    print(f"  IoU_overall     = {iou_overall:.4f}   (denom = {N_TOTAL_TEST})", flush=True)
    print(f"  oracle (within cohort) = "
          f"{oof_df['best_iou'].sum() / N_TOTAL_TEST:.4f}", flush=True)

    # OOF prediction distribution
    uniq, cnt = np.unique(oof_pred, return_counts=True)
    pred_dist_named = {class_names[int(u)]: int(c) for u, c in zip(uniq, cnt)}
    print(f"  pred class dist = {pred_dist_named}", flush=True)

    # ----- Lead-bin decomp --------------------------------------------------
    L = oof_df["L"].astype(float)
    t_star = oof_df["t_star_doy"].astype(float)
    lead_bin_value = L - (t_star + 60.0)
    oof_df["lead_bin_anchor60"] = lead_bin_value.apply(lead_bin_of)
    decomp = lead_bin_decomp(oof_df, "iou_at_pred")
    print("\n=================== Lead-bin decomposition (anchor=alert+60, shift=0) ===================", flush=True)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 200); pd.set_option("display.max_columns", 20)
    print(decomp.to_string(index=False), flush=True)

    # ----- Save outputs -----------------------------------------------------
    summary_df = pd.DataFrame([{
        "n_used": len(oof_df),
        "n_total": N_TOTAL_TEST,
        "IoU_overall": iou_overall,
        "oracle_IoU_overall": oof_df["best_iou"].sum() / N_TOTAL_TEST,
        "accuracy_mean": accuracy_mean,
        "n_classes": len(class_names),
        "class_names": "|".join(class_names),
        "pred_dist": str(pred_dist_named),
    }])
    summary_df.to_csv(out_dir / "selector_summary.csv", index=False)
    decomp.to_csv(out_dir / "lead_bin_decomp.csv", index=False)
    keep_cols = ["sample_id", "L", "R", "t_star_doy", "lead_bin_anchor60",
                 "best_class", "best_iou", "pred_class", "iou_at_pred"]
    oof_df[keep_cols].to_csv(out_dir / "oof_predictions.csv", index=False)
    print(f"\n[csv] {out_dir}/selector_summary.csv", flush=True)
    print(f"[csv] {out_dir}/lead_bin_decomp.csv", flush=True)
    print(f"[csv] {out_dir}/oof_predictions.csv", flush=True)

    # ----- Compare to single-best (re-derive from grid) --------------------
    single_best = 0.0
    single_best_combo = None
    for (m, off), sub in grid.groupby(["model", "offset"]):
        iou_sum = float(sub[sub["matched"] == True]["iou_matched"].sum())
        iou_ov = iou_sum / float(N_TOTAL_TEST)
        if iou_ov > single_best:
            single_best = iou_ov
            single_best_combo = (m, int(off))
    print("\n=================== Phase S11 verdict ===================", flush=True)
    print(f"  single-best (model, offset) = {single_best_combo} → IoU = {single_best:.4f}", flush=True)
    print(f"  selector OOF IoU            = {iou_overall:.4f}  "
          f"Δ vs single-best = {iou_overall - single_best:+.4f}", flush=True)
    if iou_overall > single_best + 0.02:
        print("  ===== GO ===== selector clearly beats single-best (>+0.02). Adopt.", flush=True)
    elif iou_overall > single_best + 0.005:
        print("  ===== MARGINAL ===== selector edges out single-best. Sanity-check fold stability.", flush=True)
    else:
        print("  ===== NO-GO ===== selector ≤ single-best. The 8-way pool did not unlock gain "
              "the v3 features can exploit.", flush=True)


if __name__ == "__main__":
    main()
