"""Phase S12 evaluation — mu_sanity (with std), selector C_old, lead-bin, shift sweep.

The S12 sample_grid contains only one model (the GDD-incorporating S12
final).  Stage 1 also uses run=8, so the cohort may differ from baseline.
This script:

    1. Computes mu_sanity with explicit std and percentiles for mu, mu-L,
       mu-mid (S12 spec calls out mu_std as a key sanity).
    2. Reports the cohort size + overlap with the baseline sample_grid
       (if provided via --baseline_sample_grid).
    3. Runs the v3_mu_only logreg selector OOF (Phase S3 C_old setup).
    4. Lead-bin decomp (anchor = alert+60).
    5. Shift sweep over {0,15,30,46,60}.
    6. Optionally side-by-side baseline comparison if --baseline_sample_grid
       is provided (notes the cohort overlap explicitly).

Outputs (out_dir, default outputs/phase_s12/):
    mu_sanity.csv
    selector_summary.csv
    lead_bin_decomp.csv
    shift_sweep.csv
    cohort_overlap.csv         (only if baseline sample_grid given)
    per_sample_<model>.csv
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
        print(f"  dropped stage1/static-NaN rows: {int(bad.sum())}", flush=True)
        ps = ps[~bad].copy()
    return ps


def mu_sanity_extended(label: str, ps: pd.DataFrame) -> dict:
    """Adds mean/std/p10/p50/p90 for mu, mu-L, mu-mid per offset (S12 spec)."""
    L = ps["L"].astype(float)
    R = ps["R"].astype(float)
    mid = (L + R) / 2.0
    row = {
        "model": label, "n": int(len(ps)),
        "L_mean": float(L.mean()),
        "L_std": float(L.std()),
        "R_mean": float(R.mean()),
        "mid_mean": float(mid.mean()),
        "mid_std": float(mid.std()),
    }
    for o in OFFSETS_ORIG:
        col = f"mu_off{o}"
        if col not in ps.columns:
            continue
        mu = ps[col].astype(float).dropna()
        if len(mu) == 0:
            continue
        bias_L = (ps[col].astype(float) - L).dropna()
        bias_mid = (ps[col].astype(float) - mid).dropna()
        row[f"mu_off{o}_mean"] = float(mu.mean())
        row[f"mu_off{o}_std"] = float(mu.std())
        row[f"mu_off{o}_p10"] = float(mu.quantile(0.10))
        row[f"mu_off{o}_p50"] = float(mu.quantile(0.50))
        row[f"mu_off{o}_p90"] = float(mu.quantile(0.90))
        row[f"mu_off{o}_minus_L_mean"] = float(bias_L.mean())
        row[f"mu_off{o}_minus_L_std"] = float(bias_L.std())
        row[f"mu_off{o}_minus_L_p50"] = float(bias_L.median())
        row[f"mu_off{o}_minus_mid_mean"] = float(bias_mid.mean())
        row[f"mu_off{o}_minus_mid_std"] = float(bias_mid.std())
        row[f"mu_off{o}_n_finite"] = int(len(mu))
    return row


def mu_by_lead_bin(ps: pd.DataFrame) -> pd.DataFrame:
    """Per lead-bin (anchor = alert+60) summary of mu_off60 distribution."""
    from rice.scripts.phase_s3_selector import lead_bin_of, LEAD_BIN_NAMES
    L = ps["L"].astype(float)
    t_star = ps["t_star_doy"].astype(float)
    ps = ps.assign(lead_bin=(L - (t_star + 60.0)).apply(lead_bin_of))
    rows = []
    bin_order = LEAD_BIN_NAMES + ["<15", ">120"]
    for b in bin_order:
        sub = ps[ps["lead_bin"] == b]
        if sub.empty:
            rows.append({"lead_bin": b, "n": 0,
                          "mu_off60_mean": float("nan"), "mu_off60_std": float("nan"),
                          "mu_off60_minus_L_mean": float("nan"),
                          "mu_off60_minus_mid_mean": float("nan")})
            continue
        L_b = sub["L"].astype(float)
        R_b = sub["R"].astype(float)
        mid_b = (L_b + R_b) / 2.0
        mu = sub["mu_off60"].astype(float)
        rows.append({
            "lead_bin": b, "n": int(len(sub)),
            "mu_off60_mean": float(mu.mean(skipna=True)),
            "mu_off60_std": float(mu.std(skipna=True)),
            "mu_off60_minus_L_mean": float((mu - L_b).mean(skipna=True)),
            "mu_off60_minus_mid_mean": float((mu - mid_b).mean(skipna=True)),
        })
    return pd.DataFrame(rows)


def shift_sweep_for_oof(per_sample_eval: pd.DataFrame, shifts: list[float]) -> pd.DataFrame:
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


def _safe_label(label: str) -> str:
    return label.replace(" ", "_").replace("(", "").replace(")", "").replace("=", "")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", type=str, default="sheath_blight")
    ap.add_argument("--run", type=int, default=8)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--stage1_ckpt", type=str, required=True,
                    help="Stage 1 ckpt used to build the s12 sample_grid (run=8).")
    ap.add_argument("--sample_grid", type=str,
                    default="rice/outputs/diag/phase_s12_sample_grid.csv")
    ap.add_argument("--baseline_sample_grid", type=str, default=None,
                    help="Optional: a baseline sample_grid (e.g. "
                         "rice/outputs/diag/phase_s10_sample_grid.csv) for cohort-overlap "
                         "and side-by-side mu_sanity. Cohort overlap is the "
                         "intersection of sample_ids.")
    ap.add_argument("--baseline_label_substr", type=str, default="(asym=25)")
    ap.add_argument("--out_dir", type=str, default="rice/outputs/phase_s12/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_shift", type=float, default=0.0)
    ap.add_argument("--shifts", type=str, default=",".join(str(s) for s in SHIFT_GRID_DEFAULT))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shifts = [float(x) for x in str(args.shifts).split(",") if x.strip()]

    if not os.path.exists(args.sample_grid):
        raise SystemExit(f"[abort] sample_grid not found: {args.sample_grid}")
    grid = pd.read_csv(args.sample_grid)
    print(f"[input] sample_grid rows={len(grid)}  models={list(grid['model'].unique())}", flush=True)

    s12_label = list(grid["model"].unique())[0]
    print(f"[eval] s12 label = {s12_label!r}", flush=True)

    _ = resolve_pest(args.pest)
    print("\n[stage1] computing calibrated scores + per-sample features (run=8)…", flush=True)
    test_seas, test_s, p_test_cal, tau = stage1_test_scores(
        Path(args.stage1_ckpt), args.run, args)
    tstar_abs_map = {str(sid): int(sub.iloc[0]["t_star_doy"])
                     for sid, sub in grid.groupby("sample_id")}
    s1_ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    doy_start = int(s1_ckpt.get("doy_start", 1))
    stage1_features = compute_stage1_features_per_sample(
        test_s, p_test_cal, tstar_abs_map, doy_start)
    print(f"[stage1 features] rows={len(stage1_features)}  "
          f"stage1 ckpt run={int(s1_ckpt.get('run', -1))}  d_in={int(s1_ckpt.get('d_in', -1))}", flush=True)

    # --- Cohort overlap with baseline grid (if provided) -------------------
    cohort_row = None
    if args.baseline_sample_grid and os.path.exists(args.baseline_sample_grid):
        bg = pd.read_csv(args.baseline_sample_grid)
        base_match = [m for m in bg["model"].unique() if args.baseline_label_substr in m]
        if base_match:
            base_label = base_match[0]
            base_sids = set(bg[bg["model"] == base_label]["sample_id"].astype(str).unique())
            s12_sids = set(grid["sample_id"].astype(str).unique())
            inter = base_sids & s12_sids
            cohort_row = {
                "baseline_grid": args.baseline_sample_grid,
                "baseline_label": base_label,
                "n_baseline": len(base_sids),
                "n_s12": len(s12_sids),
                "n_intersection": len(inter),
                "n_baseline_only": len(base_sids - s12_sids),
                "n_s12_only": len(s12_sids - base_sids),
            }
            print(f"\n=================== Cohort overlap with baseline ===================", flush=True)
            for k, v in cohort_row.items():
                print(f"  {k}: {v}", flush=True)
            pd.DataFrame([cohort_row]).to_csv(out_dir / "cohort_overlap.csv", index=False)
            print(f"[csv] {out_dir}/cohort_overlap.csv", flush=True)

    # --- Build per-sample table ---------------------------------------------
    ps = per_sample_build_for_model(grid, stage1_features, OFFSETS_ORIG)
    print(f"\n[per_sample] n={len(ps)}  best_offset dist="
          f"{dict(ps['best_offset'].value_counts().sort_index())}", flush=True)

    # --- mu sanity (extended: mean/std/p10/p50/p90) -------------------------
    mu_row = mu_sanity_extended(s12_label, ps)
    sanity_df = pd.DataFrame([mu_row])
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 280); pd.set_option("display.max_columns", 50)
    print("\n=================== mu sanity (extended, post-train) ===================", flush=True)
    print(sanity_df.to_string(index=False), flush=True)
    sanity_df.to_csv(out_dir / "mu_sanity.csv", index=False)
    print(f"[csv] {out_dir}/mu_sanity.csv", flush=True)

    # --- mu by lead bin (S12 spec: lead bin 별 mu 분포) ----------------------
    mu_bin_df = mu_by_lead_bin(ps)
    print("\n=================== mu_off60 by lead bin (anchor = alert+60) ===================", flush=True)
    print(mu_bin_df.to_string(index=False), flush=True)
    mu_bin_df.to_csv(out_dir / "mu_by_lead_bin.csv", index=False)
    print(f"[csv] {out_dir}/mu_by_lead_bin.csv", flush=True)

    # --- Selector (Phase S3 C_old) ------------------------------------------
    feat_cols, impute_cols = feature_cols_for_set("v3_mu_only", OFFSETS_ORIG)
    cv = run_cv_cell(ps, feat_cols, impute_cols, "best_offset",
                      classifier="logreg", args=args,
                      shift=args.eval_shift, offsets=OFFSETS_ORIG)
    iou_overall = cv["iou_overall"]
    print("\n=================== Phase S12 selector summary (C_old setup) ===================", flush=True)
    summary_row = {
        "model": s12_label, "n_used": len(ps),
        "IoU_overall": iou_overall,
        "P_ideal_overall": cv["p_ideal_overall"],
        "accuracy_mean": cv["accuracy_mean"],
        "oof_pred_dist": str(cv["oof_pred_dist"]),
        **{f"n_{k}": v for k, v in cv["bucket_counts"].items()},
    }
    summary_df = pd.DataFrame([summary_row])
    print(summary_df.to_string(index=False), flush=True)
    summary_df.to_csv(out_dir / "selector_summary.csv", index=False)
    print(f"[csv] {out_dir}/selector_summary.csv", flush=True)

    # --- Lead-bin decomposition ---------------------------------------------
    decomp = lead_bin_decomp(cv["per_sample_eval"])
    print("\n=================== Lead-bin IoU decomposition (anchor=alert+60, shift=0) ===================", flush=True)
    print(decomp.to_string(index=False), flush=True)
    decomp.to_csv(out_dir / "lead_bin_decomp.csv", index=False)
    print(f"[csv] {out_dir}/lead_bin_decomp.csv", flush=True)

    # --- Shift sweep --------------------------------------------------------
    sweep = shift_sweep_for_oof(cv["per_sample_eval"], shifts)
    sweep.insert(0, "model", s12_label)
    print("\n=================== Shift sweep (per-shift IoU / P_ideal) ===================", flush=True)
    print(sweep.to_string(index=False), flush=True)
    sweep.to_csv(out_dir / "shift_sweep.csv", index=False)
    print(f"[csv] {out_dir}/shift_sweep.csv", flush=True)

    cv["per_sample_eval"].to_csv(
        out_dir / f"per_sample_{_safe_label(s12_label)}.csv", index=False)
    print(f"[csv] {out_dir}/per_sample_*.csv", flush=True)

    # --- Phase S12 verdict --------------------------------------------------
    print("\n=================== Phase S12 verdict ===================", flush=True)
    mu60_L = mu_row.get("mu_off60_minus_L_mean", float("nan"))
    mu60_std = mu_row.get("mu_off60_std", float("nan"))
    mu60_minus_L_std = mu_row.get("mu_off60_minus_L_std", float("nan"))
    bin_6190 = decomp[decomp["lead_bin"] == "61-90"]
    bin_91120 = decomp[decomp["lead_bin"] == "91-120"]
    iou_6190 = float(bin_6190["IoU_mean"].iloc[0]) if len(bin_6190) else float("nan")
    iou_91120 = float(bin_91120["IoU_mean"].iloc[0]) if len(bin_91120) else float("nan")
    ok_std = "OK " if (np.isfinite(mu60_std) and mu60_std > 10.0) else "FAIL"
    ok_muL = "OK " if -3.0 <= mu60_L <= 13.0 else "FAIL"   # spec: mu-L ≈ +5, ±8 tol
    ok_iou = "OK " if iou_overall > 0.35 else "FAIL"
    ok_long = "OK " if iou_6190 > 0.10 or iou_91120 > 0.02 else "FAIL"
    print(f"  [{ok_std}] mu_off60 std = {mu60_std:.2f}    (target > 10)", flush=True)
    print(f"  [{ok_muL}] mu_off60 − L mean = {mu60_L:+.2f}   (target ≈ +5; tol [-3, +13])", flush=True)
    print(f"  [{ok_iou}] overall IoU = {iou_overall:.4f}  (target > 0.35)", flush=True)
    print(f"  [{ok_long}] bin 61-90 IoU = {iou_6190:.4f}   bin 91-120 IoU = {iou_91120:.4f}  "
          f"(target either > 0.10 / > 0.02)", flush=True)


if __name__ == "__main__":
    main()
