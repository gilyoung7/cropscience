"""Phase S6 evaluation — selector (C_old) + lead-bin + shift sweep + mu sanity.

For each 2-sided variant present in the sample_grid (typically the original
2-sided and the ε-target 2-sided), this script:

    0. Prints a post-train mu_mean sanity table per model
       (mean L, R, mid, mu_off60, mu_off120, mu - L bias).
    1. Builds the per-sample table (4-offset action space).
    2. Trains a v3_mu_only logreg selector OOF (Phase S3 C_old setup).
    3. Reports IoU_overall (denom 575) + bucket counts at shift=0.
    4. Decomposes IoU per lead bin (anchor = alert+60).
    5. Sweeps the operational shift over {0,15,30,46,60} to report
       P_ideal_overall at each shift, using the same OOF predictions.

Outputs (out_dir, default outputs/phase_s6/):
    selector_summary.csv
    lead_bin_decomp.csv
    shift_sweep.csv
    mu_sanity.csv
    per_sample_<model>.csv (one per evaluated variant)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rice.src.pest_resolver import resolve_pest
from rice.src.train_eval import overlap_metrics
from rice.scripts.phase_s_selector import (
    N_TOTAL_TEST, P_IDEAL_LOW, P_IDEAL_HIGH,
    stage1_test_scores, compute_stage1_features_per_sample,
)
from rice.scripts.phase_s3_selector import (
    OFFSETS_ORIG, STAGE1_FEATURES,
    build_per_sample_table_ext, attach_features_ext,
    feature_cols_for_set, run_cv_cell, lead_bin_decomp,
)


BASE_LABEL_DEFAULT = "D=15 2-sided (asym=25)"
SHIFT_GRID_DEFAULT = (0, 15, 30, 46, 60)


def per_sample_build_for_model(grid_model: pd.DataFrame, stage1_features: pd.DataFrame,
                                offsets: list[int]) -> pd.DataFrame:
    ps = build_per_sample_table_ext(grid_model.copy(), offsets)
    ps = ps[ps["n_offsets_matched"] > 0].copy()
    ps = ps.merge(stage1_features, left_on="sample_id", right_index=True, how="left")
    ps["tstar_doy_feat"] = ps["t_star_doy"].astype(float)
    ps["year_feat"] = ps["year"].astype(float)
    ps["alert_tstar_doy"] = ps["t_star_doy"].astype(float)
    ps = attach_features_ext(ps, offsets)
    bad = ps[STAGE1_FEATURES + ["sigma", "L", "t_star_doy"]].isna().any(axis=1)
    if bad.any():
        print(f"  dropped stage1/static-NaN rows: {int(bad.sum())}")
        ps = ps[~bad].copy()
    return ps


def mu_sanity(label: str, ps: pd.DataFrame, target_offset_expected: float | None) -> dict:
    """Post-train mu sanity: mean(L), mean(mu_off*), mean(mu - L)."""
    L = ps["L"].astype(float)
    R = ps["R"].astype(float)
    mid = (L + R) / 2.0
    row = {
        "model": label, "n": int(len(ps)),
        "L_mean": float(L.mean()),
        "R_mean": float(R.mean()),
        "mid_mean": float(mid.mean()),
    }
    for o in OFFSETS_ORIG:
        col = f"mu_off{o}"
        if col not in ps.columns:
            continue
        mu = ps[col].astype(float)
        bias = (mu - L)
        row[f"mu_off{o}_mean"] = float(mu.mean(skipna=True))
        row[f"mu_off{o}_minus_L_mean"] = float(bias.mean(skipna=True))
        row[f"mu_off{o}_minus_L_p50"] = float(bias.median(skipna=True))
        row[f"mu_off{o}_n_finite"] = int(np.isfinite(mu).sum())
    if target_offset_expected is not None:
        row["target_offset_expected"] = float(target_offset_expected)
        # The asymmetric loss pushes mu just below L+ε; report the gap.
        if "mu_off60_minus_L_mean" in row:
            row["mu_off60_minus_L_minus_expected"] = (
                row["mu_off60_minus_L_mean"] - float(target_offset_expected))
    return row


def shift_sweep_for_oof(per_sample_eval: pd.DataFrame, shifts: list[float]) -> pd.DataFrame:
    """Recompute bucket distribution + IoU at each operational shift."""
    rows = []
    for shift in shifts:
        bc = {n: 0 for n in ("MISSED", "TOO_LATE", "URGENT", "IDEAL",
                              "ADVANCE", "TOO_EARLY")}
        iou_sum = 0.0
        n_total = N_TOTAL_TEST
        for _, r in per_sample_eval.iterrows():
            mu_at = r["mu_at_pred_off"]
            sigma = float(r["sigma"])
            L = r["L"]; R = r["R"]
            if not np.isfinite(mu_at) or not np.isfinite(L):
                continue
            HW = 1.96 * sigma
            pL = int(round(float(mu_at) - float(shift) - HW))
            pR = int(round(float(mu_at) - float(shift) + HW))
            iou, _, _ = overlap_metrics(pL, pR, int(L), int(R))
            iou_sum += float(iou)
            bucket_lead = float(L) - (float(mu_at) + 1.96 * sigma - float(shift))
            if bucket_lead < 0:        bc["MISSED"] += 1
            elif bucket_lead < 7:      bc["TOO_LATE"] += 1
            elif bucket_lead < 14:     bc["URGENT"] += 1
            elif bucket_lead < 30:     bc["IDEAL"] += 1
            elif bucket_lead < 45:     bc["ADVANCE"] += 1
            else:                      bc["TOO_EARLY"] += 1
        rows.append({
            "shift": float(shift),
            "IoU_overall": iou_sum / n_total,
            "P_ideal_overall": bc["IDEAL"] / n_total,
            "P_useful_overall": (bc["TOO_LATE"] + bc["URGENT"] + bc["IDEAL"]) / n_total,
            "n_MISSED": bc["MISSED"], "n_TOO_LATE": bc["TOO_LATE"],
            "n_URGENT": bc["URGENT"], "n_IDEAL": bc["IDEAL"],
            "n_ADVANCE": bc["ADVANCE"], "n_TOO_EARLY": bc["TOO_EARLY"],
        })
    return pd.DataFrame(rows)


def evaluate_one_model(label: str, grid_model: pd.DataFrame,
                        stage1_features: pd.DataFrame, args,
                        shifts: list[float],
                        target_offset_expected: float | None) -> dict:
    print(f"\n========== [model] {label} ==========")
    ps = per_sample_build_for_model(grid_model, stage1_features, OFFSETS_ORIG)
    print(f"  per_sample n={len(ps)}  best_offset dist="
          f"{dict(ps['best_offset'].value_counts().sort_index())}")
    mu_row = mu_sanity(label, ps, target_offset_expected)
    print(f"  [mu sanity] L_mean={mu_row['L_mean']:.2f}  "
          f"mid_mean={mu_row['mid_mean']:.2f}  "
          f"mu_off60_mean={mu_row.get('mu_off60_mean', float('nan')):.2f}  "
          f"(mu - L)_mean={mu_row.get('mu_off60_minus_L_mean', float('nan')):+.2f}  "
          f"vs expected ε={target_offset_expected}")
    feat_cols, impute_cols = feature_cols_for_set("v3_mu_only", OFFSETS_ORIG)
    cv = run_cv_cell(ps, feat_cols, impute_cols, "best_offset",
                      classifier="logreg", args=args,
                      shift=args.eval_shift, offsets=OFFSETS_ORIG)
    decomp = lead_bin_decomp(cv["per_sample_eval"])
    sweep = shift_sweep_for_oof(cv["per_sample_eval"], shifts)
    return {
        "label": label, "n_used": len(ps),
        "iou_overall": cv["iou_overall"],
        "p_ideal_overall": cv["p_ideal_overall"],
        "accuracy_mean": cv["accuracy_mean"],
        "bucket_counts": cv["bucket_counts"],
        "oof_pred_dist": cv["oof_pred_dist"],
        "per_sample_eval": cv["per_sample_eval"],
        "lead_bin_decomp": decomp,
        "shift_sweep": sweep,
        "mu_sanity": mu_row,
    }


def _safe_label(label: str) -> str:
    return label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")


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
                    default="rice/outputs/diag/phase_s6_sample_grid.csv")
    ap.add_argument("--out_dir", type=str, default="rice/outputs/phase_s6/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_shift", type=float, default=0.0,
                    help="Shift used inside run_cv_cell for headline P_ideal "
                         "(0 = sample-intrinsic IoU). Shift sweep below is independent.")
    ap.add_argument("--shifts", type=str, default=",".join(str(s) for s in SHIFT_GRID_DEFAULT),
                    help="Comma-separated shift grid.")
    ap.add_argument("--base_label", type=str, default=BASE_LABEL_DEFAULT)
    ap.add_argument("--tc_label_substr", type=str, default="tc",
                    help="Substring identifying the ε-target 2-sided variant.")
    ap.add_argument("--target_offset_baseline", type=float, default=0.0,
                    help="ε of the baseline 2-sided row (used in mu sanity).")
    ap.add_argument("--target_offset_tc", type=float, default=7.5,
                    help="ε of the new ε-target row (used in mu sanity).")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shifts = [float(x) for x in str(args.shifts).split(",") if x.strip()]

    if not os.path.exists(args.sample_grid):
        raise SystemExit(f"[abort] sample_grid not found: {args.sample_grid}")
    grid = pd.read_csv(args.sample_grid)
    print(f"[input] sample_grid rows={len(grid)}  models={list(grid['model'].unique())}")

    models_in_grid = list(grid["model"].unique())
    base_match = [m for m in models_in_grid
                  if args.base_label in m and args.tc_label_substr not in m]
    tc_match = [m for m in models_in_grid if args.tc_label_substr in m]
    eval_labels: list[tuple[str, float | None]] = []
    if base_match:
        eval_labels.append((base_match[0], args.target_offset_baseline))
    for m in tc_match:
        eval_labels.append((m, args.target_offset_tc))
    if not eval_labels:
        raise SystemExit(
            f"no matching models for base='{args.base_label}' or tc='{args.tc_label_substr}'")
    print(f"[eval] labels = {[lab for lab, _ in eval_labels]}")

    # Stage 1 features (shared).
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

    results = []
    for label, eps in eval_labels:
        gm = grid[grid["model"] == label].copy()
        results.append(evaluate_one_model(label, gm, stage1_features, args, shifts, eps))

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 280); pd.set_option("display.max_columns", 40)

    # --- mu sanity table -----------------------------------------------------
    sanity_df = pd.DataFrame([r["mu_sanity"] for r in results])
    print("\n=================== mu sanity (post-train) ===================")
    print(sanity_df.to_string(index=False))
    sanity_df.to_csv(out_dir / "mu_sanity.csv", index=False)
    print(f"[csv] {out_dir}/mu_sanity.csv")

    # --- selector summary ----------------------------------------------------
    rows = []
    for r in results:
        rows.append({
            "model": r["label"], "n_used": r["n_used"],
            "IoU_overall": r["iou_overall"],
            "P_ideal_overall": r["p_ideal_overall"],
            "accuracy_mean": r["accuracy_mean"],
            "oof_pred_dist": str(r["oof_pred_dist"]),
            **{f"n_{k}": v for k, v in r["bucket_counts"].items()},
        })
    summary = pd.DataFrame(rows)
    print("\n=================== Phase S6 selector summary (C_old setup) ===================")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "selector_summary.csv", index=False)
    print(f"[csv] {out_dir}/selector_summary.csv")

    # --- lead-bin decomp -----------------------------------------------------
    print("\n=================== Lead-bin IoU decomposition (anchor=alert+60, shift=0) ===================")
    merged = None
    for r in results:
        d = r["lead_bin_decomp"].copy()
        sfx = "_" + _safe_label(r["label"])
        d = d.rename(columns={
            "n": f"n{sfx}", "IoU_mean": f"IoU{sfx}",
            "iou_sum": f"iou_sum{sfx}", "contrib_to_overall": f"contrib{sfx}",
        })
        merged = d if merged is None else merged.merge(d, on="lead_bin", how="outer")
    if merged is not None and len(results) == 2:
        sfx_base = "_" + _safe_label(results[0]["label"])
        sfx_tc = "_" + _safe_label(results[1]["label"])
        merged["delta_IoU_tc_minus_base"] = merged[f"IoU{sfx_tc}"] - merged[f"IoU{sfx_base}"]
        merged["delta_contrib_tc_minus_base"] = (
            merged[f"contrib{sfx_tc}"] - merged[f"contrib{sfx_base}"])
    print(merged.to_string(index=False))
    merged.to_csv(out_dir / "lead_bin_decomp.csv", index=False)
    print(f"[csv] {out_dir}/lead_bin_decomp.csv")

    # --- shift sweep ---------------------------------------------------------
    print("\n=================== Shift sweep (per model, OOF) ===================")
    sweep_rows = []
    for r in results:
        s = r["shift_sweep"].copy()
        s.insert(0, "model", r["label"])
        sweep_rows.append(s)
    sweep_df = pd.concat(sweep_rows, axis=0, ignore_index=True)
    print(sweep_df.to_string(index=False))
    sweep_df.to_csv(out_dir / "shift_sweep.csv", index=False)
    print(f"[csv] {out_dir}/shift_sweep.csv")

    # --- per-sample dumps ----------------------------------------------------
    for r in results:
        r["per_sample_eval"].to_csv(
            out_dir / f"per_sample_{_safe_label(r['label'])}.csv", index=False)
    print(f"[csv] {out_dir}/per_sample_*.csv")

    # --- Success criteria check (Phase S6 spec) ------------------------------
    if len(results) == 2:
        print("\n=================== Success criteria (Phase S6 spec) ===================")
        base_r = results[0]; tc_r = results[1]
        def _bin_iou(decomp_df: pd.DataFrame, b: str) -> float:
            sub = decomp_df[decomp_df["lead_bin"] == b]
            if sub.empty or pd.isna(sub["IoU_mean"].iloc[0]):
                return 0.0
            return float(sub["IoU_mean"].iloc[0])
        ok_iou = "OK " if tc_r["iou_overall"] > 0.35 else "FAIL"
        print(f"  [{ok_iou}] overall IoU base={base_r['iou_overall']:.4f} → "
              f"tc={tc_r['iou_overall']:.4f}  (thr > 0.35)")
        sweet_base = _bin_iou(base_r["lead_bin_decomp"], "46-60")
        sweet_tc = _bin_iou(tc_r["lead_bin_decomp"], "46-60")
        ok_sweet = "OK " if sweet_tc >= sweet_base else "FAIL"
        print(f"  [{ok_sweet}] bin 46-60   IoU base={sweet_base:.4f} → tc={sweet_tc:.4f}  "
              f"(Δ = {sweet_tc - sweet_base:+.4f}; expect ≥0)")
        for bname in ("31-45", "61-90"):
            b_base = _bin_iou(base_r["lead_bin_decomp"], bname)
            b_tc = _bin_iou(tc_r["lead_bin_decomp"], bname)
            ok_b = "OK " if b_tc > b_base else "FAIL"
            print(f"  [{ok_b}] bin {bname:<8} IoU base={b_base:.4f} → tc={b_tc:.4f}  "
                  f"(Δ = {b_tc - b_base:+.4f}; expect > 0)")
        # mu sanity verdict
        s_tc = tc_r["mu_sanity"]
        if "mu_off60_minus_L_mean" in s_tc and "target_offset_expected" in s_tc:
            gap = float(s_tc["mu_off60_minus_L_mean"]) - float(s_tc["target_offset_expected"])
            ok_mu = "OK " if -3.0 <= gap <= 1.0 else "WARN"
            print(f"  [{ok_mu}] mu_off60 − L mean = {s_tc['mu_off60_minus_L_mean']:+.2f}  "
                  f"(expected ≈ ε = {s_tc['target_offset_expected']}, gap = {gap:+.2f})")


if __name__ == "__main__":
    main()
