"""Phase S8 evaluation — selector (C_old) + lead-bin + shift sweep + mu/mid sanity.

For each 2-sided variant present in the sample_grid (original baseline + the
new center-mode variant), this script:

    0. Prints a post-train mu sanity table with mu − L AND mu − mid bias
       (mid = (L+R)/2).  The mu − mid columns are the headline for S8.
    1. Builds the per-sample table (4-offset action space).
    2. Trains a v3_mu_only logreg selector OOF (Phase S3 C_old setup).
    3. Reports IoU_overall (denom 575) + bucket counts at shift=0.
    4. Decomposes IoU per lead bin (anchor = alert+60).
    5. Sweeps the operational shift over {0,15,30,46,60}.
    6. Right-cens diagnostic: compare mu_off60 vs mu_off120 deltas to L
       (uniform right-cens pull would make them similar; t*-dependence
       would make them differ).

Outputs (out_dir, default outputs/phase_s8/):
    selector_summary.csv
    lead_bin_decomp.csv
    shift_sweep.csv
    mu_sanity.csv
    right_cens_diag.csv
    per_sample_<model>.csv  (one per evaluated variant)

Success criteria (printed at end, per Phase S8 spec):
    - mu_off60 − mid ∈ [-5, +5]  (mu reaches interval midpoint)
    - overall IoU > 0.40
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
        print(f"  dropped stage1/static-NaN rows: {int(bad.sum())}", flush=True)
        ps = ps[~bad].copy()
    return ps


def mu_sanity(label: str, ps: pd.DataFrame, target_mode: str | None) -> dict:
    L = ps["L"].astype(float)
    R = ps["R"].astype(float)
    mid = (L + R) / 2.0
    row = {
        "model": label, "n": int(len(ps)),
        "L_mean": float(L.mean()),
        "R_mean": float(R.mean()),
        "mid_mean": float(mid.mean()),
        "target_mode": target_mode,
    }
    for o in OFFSETS_ORIG:
        col = f"mu_off{o}"
        if col not in ps.columns:
            continue
        mu = ps[col].astype(float)
        bias_L = (mu - L)
        bias_mid = (mu - mid)
        row[f"mu_off{o}_mean"] = float(mu.mean(skipna=True))
        row[f"mu_off{o}_minus_L_mean"] = float(bias_L.mean(skipna=True))
        row[f"mu_off{o}_minus_L_p50"] = float(bias_L.median(skipna=True))
        row[f"mu_off{o}_minus_mid_mean"] = float(bias_mid.mean(skipna=True))
        row[f"mu_off{o}_minus_mid_p50"] = float(bias_mid.median(skipna=True))
        row[f"mu_off{o}_n_finite"] = int(np.isfinite(mu).sum())
    return row


def right_cens_diag(label: str, ps: pd.DataFrame) -> dict:
    """Compare mu predictions across offsets to detect right-cens pull-toward-Tend.

    If right-cens MSE (mu_right toward Tend) is the dominant force pulling mu
    away from its target, then mu should drift more (i.e., look less like L)
    as t* moves farther from L.  offset=60 (t* closer to L) vs offset=120
    (t* farther from L) gives a within-model contrast.

    Reports:
      mu_off60_minus_off120_mean : positive → mu_off60 > mu_off120 (later prediction
          when t* is closer to L; consistent with t*-anchored attention drift)
      pct_off60_gt_off120        : fraction of samples where mu_off60 > mu_off120
    """
    out = {"model": label, "n": int(len(ps))}
    if "mu_off60" not in ps.columns or "mu_off120" not in ps.columns:
        return out
    a = ps["mu_off60"].astype(float)
    b = ps["mu_off120"].astype(float)
    diff = a - b
    mask = np.isfinite(diff)
    n_valid = int(mask.sum())
    out["n_paired"] = n_valid
    if n_valid == 0:
        return out
    out["mu_off60_minus_off120_mean"] = float(diff[mask].mean())
    out["mu_off60_minus_off120_p50"] = float(diff[mask].median())
    out["pct_off60_gt_off120"] = float((diff[mask] > 0).mean())
    return out


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


def evaluate_one_model(label: str, grid_model: pd.DataFrame,
                        stage1_features: pd.DataFrame, args,
                        shifts: list[float],
                        target_mode: str | None) -> dict:
    print(f"\n========== [model] {label} ==========", flush=True)
    ps = per_sample_build_for_model(grid_model, stage1_features, OFFSETS_ORIG)
    print(f"  per_sample n={len(ps)}  best_offset dist="
          f"{dict(ps['best_offset'].value_counts().sort_index())}", flush=True)
    mu_row = mu_sanity(label, ps, target_mode)
    rc_row = right_cens_diag(label, ps)
    print(f"  [mu sanity] L_mean={mu_row['L_mean']:.2f}  mid_mean={mu_row['mid_mean']:.2f}  "
          f"mu_off60_mean={mu_row.get('mu_off60_mean', float('nan')):.2f}", flush=True)
    print(f"              mu_off60−L  mean={mu_row.get('mu_off60_minus_L_mean', float('nan')):+.2f}  "
          f"mu_off60−mid mean={mu_row.get('mu_off60_minus_mid_mean', float('nan')):+.2f}", flush=True)
    print(f"              mu_off120−L mean={mu_row.get('mu_off120_minus_L_mean', float('nan')):+.2f}  "
          f"mu_off120−mid mean={mu_row.get('mu_off120_minus_mid_mean', float('nan')):+.2f}", flush=True)
    print(f"  [right-cens diag] mu_off60−mu_off120 mean="
          f"{rc_row.get('mu_off60_minus_off120_mean', float('nan')):+.2f}  "
          f"pct_off60>off120={rc_row.get('pct_off60_gt_off120', float('nan')):.3f}", flush=True)
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
        "right_cens_diag": rc_row,
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
                    default="rice/outputs/diag/phase_s8_sample_grid.csv")
    ap.add_argument("--out_dir", type=str, default="rice/outputs/phase_s8/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_shift", type=float, default=0.0,
                    help="Shift used inside run_cv_cell for headline P_ideal "
                         "(0 = sample-intrinsic IoU). Shift sweep below is independent.")
    ap.add_argument("--shifts", type=str, default=",".join(str(s) for s in SHIFT_GRID_DEFAULT))
    ap.add_argument("--base_label", type=str, default=BASE_LABEL_DEFAULT)
    ap.add_argument("--center_label_substr", type=str, default="center",
                    help="Substring identifying the center-mode row in the grid.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shifts = [float(x) for x in str(args.shifts).split(",") if x.strip()]

    if not os.path.exists(args.sample_grid):
        raise SystemExit(f"[abort] sample_grid not found: {args.sample_grid}")
    grid = pd.read_csv(args.sample_grid)
    print(f"[input] sample_grid rows={len(grid)}  models={list(grid['model'].unique())}", flush=True)

    models_in_grid = list(grid["model"].unique())
    base_match = [m for m in models_in_grid
                  if args.base_label in m and args.center_label_substr not in m]
    center_match = [m for m in models_in_grid if args.center_label_substr in m]
    eval_labels: list[tuple[str, str | None]] = []
    if base_match:
        eval_labels.append((base_match[0], "l_offset"))
    for m in center_match:
        eval_labels.append((m, "center"))
    if not eval_labels:
        raise SystemExit(
            f"no matching models for base='{args.base_label}' or "
            f"center='{args.center_label_substr}'")
    print(f"[eval] labels = {[lab for lab, _ in eval_labels]}", flush=True)

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

    results = []
    for label, tm in eval_labels:
        gm = grid[grid["model"] == label].copy()
        results.append(evaluate_one_model(label, gm, stage1_features, args, shifts, tm))

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 280); pd.set_option("display.max_columns", 40)

    # --- mu sanity table -----------------------------------------------------
    sanity_df = pd.DataFrame([r["mu_sanity"] for r in results])
    print("\n=================== mu sanity (post-train; mid = (L+R)/2) ===================", flush=True)
    print(sanity_df.to_string(index=False), flush=True)
    sanity_df.to_csv(out_dir / "mu_sanity.csv", index=False)
    print(f"[csv] {out_dir}/mu_sanity.csv", flush=True)

    # --- right-cens diagnostic -----------------------------------------------
    rc_df = pd.DataFrame([r["right_cens_diag"] for r in results])
    print("\n=================== Right-cens diagnostic (mu_off60 vs mu_off120) ===================", flush=True)
    print(rc_df.to_string(index=False), flush=True)
    rc_df.to_csv(out_dir / "right_cens_diag.csv", index=False)
    print(f"[csv] {out_dir}/right_cens_diag.csv", flush=True)

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
    print("\n=================== Phase S8 selector summary (C_old setup) ===================", flush=True)
    print(summary.to_string(index=False), flush=True)
    summary.to_csv(out_dir / "selector_summary.csv", index=False)
    print(f"[csv] {out_dir}/selector_summary.csv", flush=True)

    # --- lead-bin decomp -----------------------------------------------------
    print("\n=================== Lead-bin IoU decomposition (anchor=alert+60, shift=0) ===================", flush=True)
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
        sfx_c = "_" + _safe_label(results[1]["label"])
        merged["delta_IoU_center_minus_base"] = merged[f"IoU{sfx_c}"] - merged[f"IoU{sfx_base}"]
        merged["delta_contrib_center_minus_base"] = (
            merged[f"contrib{sfx_c}"] - merged[f"contrib{sfx_base}"])
    print(merged.to_string(index=False), flush=True)
    merged.to_csv(out_dir / "lead_bin_decomp.csv", index=False)
    print(f"[csv] {out_dir}/lead_bin_decomp.csv", flush=True)

    # --- shift sweep ---------------------------------------------------------
    print("\n=================== Shift sweep (per model, OOF) ===================", flush=True)
    sweep_rows = []
    for r in results:
        s = r["shift_sweep"].copy()
        s.insert(0, "model", r["label"])
        sweep_rows.append(s)
    sweep_df = pd.concat(sweep_rows, axis=0, ignore_index=True)
    print(sweep_df.to_string(index=False), flush=True)
    sweep_df.to_csv(out_dir / "shift_sweep.csv", index=False)
    print(f"[csv] {out_dir}/shift_sweep.csv", flush=True)

    # --- per-sample dumps ----------------------------------------------------
    for r in results:
        r["per_sample_eval"].to_csv(
            out_dir / f"per_sample_{_safe_label(r['label'])}.csv", index=False)
    print(f"[csv] {out_dir}/per_sample_*.csv", flush=True)

    # --- Success criteria check (Phase S8 spec) + outcome branching --------
    if len(results) == 2:
        print("\n=================== Phase S8 verdict ===================", flush=True)
        base_r = results[0]
        c_r = results[1]
        s_c = c_r["mu_sanity"]

        # 1) mu reaches the interval midpoint
        mu_off60_mid = float(s_c.get("mu_off60_minus_mid_mean", float("nan")))
        mu_off120_mid = float(s_c.get("mu_off120_minus_mid_mean", float("nan")))
        ok_mu60 = "OK " if -5.0 <= mu_off60_mid <= 5.0 else "FAIL"
        ok_mu120 = "OK " if -5.0 <= mu_off120_mid <= 5.0 else "FAIL"
        print(f"  [{ok_mu60}] mu_off60 − mid mean = {mu_off60_mid:+.2f}  (target |.| ≤ 5)", flush=True)
        print(f"  [{ok_mu120}] mu_off120 − mid mean = {mu_off120_mid:+.2f}  (target |.| ≤ 5)", flush=True)

        # 2) overall IoU > 0.40
        ok_iou = "OK " if c_r["iou_overall"] > 0.40 else "FAIL"
        print(f"  [{ok_iou}] overall IoU base={base_r['iou_overall']:.4f} → "
              f"center={c_r['iou_overall']:.4f}  (thr > 0.40)", flush=True)

        # 3) Outcome branching (per Phase S8 spec)
        print("\n  ====== Outcome branch ======", flush=True)
        # mu vs mid → next action
        if abs(mu_off60_mid) <= 5.0 and c_r["iou_overall"] > 0.40:
            print("  → mu ≈ mid AND IoU↑ : training-dynamics hypothesis CONFIRMED; "
                  "adopt center mode for operations.", flush=True)
        elif mu_off60_mid < -10.0:
            print("  → mu ≪ mid : right-cens pull or other dynamics; "
                  "diagnose stage2_pmf_right_weight next.", flush=True)
        elif abs(mu_off60_mid) <= 5.0 and c_r["iou_overall"] <= 0.40:
            print("  → mu reached mid but IoU did not improve: selector/lead-bin "
                  "trade-off issue, not a mu placement issue.", flush=True)
        else:
            print("  → mu landed in an intermediate region; inspect mu_sanity.csv "
                  "per-bin breakdown.", flush=True)

        # Right-cens diagnostic verdict
        rc = c_r["right_cens_diag"]
        diff60_120 = float(rc.get("mu_off60_minus_off120_mean", float("nan")))
        if np.isfinite(diff60_120):
            tag = ("[INFO]" if abs(diff60_120) <= 5
                   else ("[WARN] strong t*-dependence" if diff60_120 > 5
                         else "[WARN] inverted t*-dependence"))
            print(f"  {tag} mu_off60 − mu_off120 mean = {diff60_120:+.2f}  "
                  f"(uniform right-cens pull would give ≈ 0; large +ve indicates "
                  f"t* near L predicts later than t* far from L)", flush=True)


if __name__ == "__main__":
    main()
