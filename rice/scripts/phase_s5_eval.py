"""Phase S5 evaluation — selector (C_old setup) + lead-bin decomp + shift sweep.

For each 2-sided variant present in the sample_grid (typically the original
2-sided and the lw-weighted 2-sided), this script:

    1. Builds the per-sample table (4-offset action space).
    2. Trains a v3_mu_only logreg selector OOF (Phase S3 C_old setup).
    3. Reports IoU_overall (denom 575) and per-bucket counts at shift=0.
    4. Decomposes IoU per lead bin (anchor = alert+60),
       bins = 15-30 / 31-45 / 46-60 / 61-90 / 91-120 (+ <15, >120).
    5. Sweeps the operational shift over {0, 15, 30, 46, 60} to report
       P_ideal_overall at each shift, using the same OOF predictions.

Outputs (out_dir, default outputs/phase_s5/):
    selector_summary.csv  — per-model IoU/P_ideal/accuracy + bucket counts
    lead_bin_decomp.csv   — per-model per-bin n and IoU mean + delta lw vs base
    shift_sweep.csv       — per-model per-shift IoU + bucket distribution
    per_sample_<model>.csv (one file per evaluated variant)
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
    OFFSETS_ORIG, STAGE1_FEATURES, LEAD_BIN_NAMES,
    build_per_sample_table_ext, attach_features_ext,
    feature_cols_for_set, run_cv_cell, lead_bin_decomp, lead_bin_of,
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


def shift_sweep_for_oof(per_sample_eval: pd.DataFrame, shifts: list[float]) -> pd.DataFrame:
    """Recompute bucket distribution + IoU at each shift using OOF predictions."""
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
                        shifts: list[float]) -> dict:
    print(f"\n========== [model] {label} ==========")
    ps = per_sample_build_for_model(grid_model, stage1_features, OFFSETS_ORIG)
    print(f"  per_sample n={len(ps)}  best_offset dist="
          f"{dict(ps['best_offset'].value_counts().sort_index())}")
    feat_cols, impute_cols = feature_cols_for_set("v3_mu_only", OFFSETS_ORIG)
    cv = run_cv_cell(ps, feat_cols, impute_cols, "best_offset",
                      classifier="logreg", args=args,
                      shift=args.eval_shift, offsets=OFFSETS_ORIG)
    decomp = lead_bin_decomp(cv["per_sample_eval"])
    sweep = shift_sweep_for_oof(cv["per_sample_eval"], shifts)
    return {
        "label": label,
        "n_used": len(ps),
        "iou_overall": cv["iou_overall"],
        "p_ideal_overall": cv["p_ideal_overall"],
        "accuracy_mean": cv["accuracy_mean"],
        "bucket_counts": cv["bucket_counts"],
        "oof_pred_dist": cv["oof_pred_dist"],
        "per_sample_eval": cv["per_sample_eval"],
        "lead_bin_decomp": decomp,
        "shift_sweep": sweep,
    }


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
                    default="rice/outputs/diag/phase_s5_sample_grid.csv")
    ap.add_argument("--out_dir", type=str, default="rice/outputs/phase_s5/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_shift", type=float, default=0.0,
                    help="Shift used inside run_cv_cell when computing the headline P_ideal "
                         "(0 = sample-intrinsic IoU). Shift sweep below is independent.")
    ap.add_argument("--shifts", type=str, default=",".join(str(s) for s in SHIFT_GRID_DEFAULT),
                    help="Comma-separated shift grid (e.g. '0,15,30,46,60').")
    ap.add_argument("--base_label", type=str, default=BASE_LABEL_DEFAULT,
                    help="Substring of the baseline 2-sided 'model' label in the grid.")
    ap.add_argument("--lw_label_substr", type=str, default="lw",
                    help="Substring identifying the lw-weighted 2-sided variant.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shifts = [float(x) for x in str(args.shifts).split(",") if x.strip()]

    if not os.path.exists(args.sample_grid):
        raise SystemExit(f"[abort] sample_grid not found: {args.sample_grid}")
    grid = pd.read_csv(args.sample_grid)
    print(f"[input] sample_grid rows={len(grid)}  models={list(grid['model'].unique())}")

    # Resolve model labels in the grid.
    models_in_grid = list(grid["model"].unique())
    base_match = [m for m in models_in_grid
                  if args.base_label in m and args.lw_label_substr not in m]
    lw_match = [m for m in models_in_grid if args.lw_label_substr in m]
    eval_labels = []
    if base_match:
        eval_labels.append(base_match[0])
    if lw_match:
        eval_labels.extend(lw_match)
    if not eval_labels:
        raise SystemExit(
            f"no matching models for base='{args.base_label}' or lw='{args.lw_label_substr}'")
    print(f"[eval] labels = {eval_labels}")

    # Stage 1 features (shared across models — depends only on stage1 ckpt + grid t*).
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

    # Per-model evaluation.
    results = []
    for label in eval_labels:
        gm = grid[grid["model"] == label].copy()
        results.append(evaluate_one_model(label, gm, stage1_features, args, shifts))

    # --- Summary table -------------------------------------------------------
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
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
    print("\n=================== Phase S5 selector summary (C_old setup) ===================")
    print(summary.to_string(index=False))
    summary.to_csv(out_dir / "selector_summary.csv", index=False)
    print(f"[csv] {out_dir}/selector_summary.csv")

    # --- Lead-bin decomp (side-by-side) --------------------------------------
    print("\n=================== Lead-bin IoU decomposition (anchor=alert+60, shift=0) ===================")
    merged = None
    for r in results:
        d = r["lead_bin_decomp"].copy()
        suffix = "_" + r["label"].replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        d = d.rename(columns={
            "n": f"n{suffix}", "IoU_mean": f"IoU{suffix}",
            "iou_sum": f"iou_sum{suffix}", "contrib_to_overall": f"contrib{suffix}",
        })
        merged = d if merged is None else merged.merge(d, on="lead_bin", how="outer")
    if merged is not None and len(results) == 2:
        # Add explicit delta lw vs baseline columns.
        sfx_base = "_" + results[0]["label"].replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        sfx_lw   = "_" + results[1]["label"].replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        merged["delta_IoU_lw_minus_base"] = merged[f"IoU{sfx_lw}"] - merged[f"IoU{sfx_base}"]
        merged["delta_contrib_lw_minus_base"] = merged[f"contrib{sfx_lw}"] - merged[f"contrib{sfx_base}"]
    print(merged.to_string(index=False))
    merged.to_csv(out_dir / "lead_bin_decomp.csv", index=False)
    print(f"[csv] {out_dir}/lead_bin_decomp.csv")

    # --- Shift sweep ---------------------------------------------------------
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

    # --- Per-sample dumps ----------------------------------------------------
    for r in results:
        safe = r["label"].replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")
        r["per_sample_eval"].to_csv(out_dir / f"per_sample_{safe}.csv", index=False)
    print(f"[csv] {out_dir}/per_sample_*.csv")

    # --- Success criteria check (from spec) ----------------------------------
    if len(results) == 2:
        print("\n=================== Success criteria (Phase S5 spec) ===================")
        base_r = results[0]; lw_r = results[1]
        # Lead bin IoUs by name.
        def _bin_iou(decomp_df: pd.DataFrame, b: str) -> float:
            sub = decomp_df[decomp_df["lead_bin"] == b]
            if sub.empty or pd.isna(sub["IoU_mean"].iloc[0]):
                return 0.0
            return float(sub["IoU_mean"].iloc[0])
        bins_to_check = [("61-90", 0.15), ("91-120", 0.02)]
        for bname, thr in bins_to_check:
            base_v = _bin_iou(base_r["lead_bin_decomp"], bname)
            lw_v = _bin_iou(lw_r["lead_bin_decomp"], bname)
            ok = "OK " if lw_v > thr else "FAIL"
            print(f"  [{ok}] bin {bname:<8} IoU base={base_v:.4f} → lw={lw_v:.4f}  (thr > {thr})")
        ok_iou = "OK " if lw_r["iou_overall"] > 0.32 else "FAIL"
        print(f"  [{ok_iou}] overall IoU base={base_r['iou_overall']:.4f} → "
              f"lw={lw_r['iou_overall']:.4f}  (thr > 0.32)")
        # Sweet-spot trade-off.
        sweet_base = _bin_iou(base_r["lead_bin_decomp"], "46-60")
        sweet_lw = _bin_iou(lw_r["lead_bin_decomp"], "46-60")
        print(f"  [TRADE] bin 46-60   IoU base={sweet_base:.4f} → lw={sweet_lw:.4f}  "
              f"(Δ = {sweet_lw - sweet_base:+.4f})")


if __name__ == "__main__":
    main()
