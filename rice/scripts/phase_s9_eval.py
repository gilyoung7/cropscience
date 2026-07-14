"""Phase S9 evaluation — same diagnostic suite as Phase S8.

Re-uses the S8 building blocks (mu_sanity with mu−mid columns, right-cens
diagnostic comparing mu_off60 vs mu_off120, lead-bin decomp, shift sweep).
The only differences from phase_s8_eval.py are the default sample_grid and
out_dir paths, plus a Phase-S9-specific verdict block (compares against S8's
center mode to attribute mu placement to right_weight).

Outputs (out_dir, default outputs/phase_s9/):
    selector_summary.csv
    lead_bin_decomp.csv
    shift_sweep.csv
    mu_sanity.csv
    right_cens_diag.csv
    per_sample_<model>.csv  (one per evaluated variant)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rice.src.pest_resolver import resolve_pest
from rice.scripts.phase_s_selector import (
    stage1_test_scores, compute_stage1_features_per_sample,
)
from rice.scripts.phase_s8_eval import (
    BASE_LABEL_DEFAULT, SHIFT_GRID_DEFAULT,
    evaluate_one_model, _safe_label,
)


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
                    default="rice/outputs/diag/phase_s9_sample_grid.csv")
    ap.add_argument("--out_dir", type=str, default="rice/outputs/phase_s9/")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--eval_shift", type=float, default=0.0)
    ap.add_argument("--shifts", type=str, default=",".join(str(s) for s in SHIFT_GRID_DEFAULT))
    ap.add_argument("--base_label", type=str, default=BASE_LABEL_DEFAULT)
    ap.add_argument("--center_label_substr", type=str, default="center_rw",
                    help="Substring identifying the center_rw row in the grid "
                         "(use 'center_rw' to distinguish from S8's 'center').")
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
    new_match = [m for m in models_in_grid if args.center_label_substr in m]
    eval_labels: list[tuple[str, str | None]] = []
    if base_match:
        eval_labels.append((base_match[0], "l_offset"))
    for m in new_match:
        eval_labels.append((m, "center_rw0"))
    if not eval_labels:
        raise SystemExit(
            f"no matching models for base='{args.base_label}' or "
            f"experiment='{args.center_label_substr}'")
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
    print("\n=================== Phase S9 selector summary (C_old setup) ===================", flush=True)
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
        sfx_n = "_" + _safe_label(results[1]["label"])
        merged["delta_IoU_new_minus_base"] = merged[f"IoU{sfx_n}"] - merged[f"IoU{sfx_base}"]
        merged["delta_contrib_new_minus_base"] = (
            merged[f"contrib{sfx_n}"] - merged[f"contrib{sfx_base}"])
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

    for r in results:
        r["per_sample_eval"].to_csv(
            out_dir / f"per_sample_{_safe_label(r['label'])}.csv", index=False)
    print(f"[csv] {out_dir}/per_sample_*.csv", flush=True)

    # --- Phase S9 verdict ---------------------------------------------------
    if len(results) == 2:
        print("\n=================== Phase S9 verdict ===================", flush=True)
        base_r = results[0]
        n_r = results[1]
        s_n = n_r["mu_sanity"]
        mu60_mid = float(s_n.get("mu_off60_minus_mid_mean", float("nan")))
        mu120_mid = float(s_n.get("mu_off120_minus_mid_mean", float("nan")))
        ok_mu60 = "OK " if -5.0 <= mu60_mid <= 5.0 else "FAIL"
        ok_mu120 = "OK " if -5.0 <= mu120_mid <= 5.0 else "FAIL"
        print(f"  [{ok_mu60}] mu_off60 − mid mean = {mu60_mid:+.2f}  (target |.| ≤ 5)", flush=True)
        print(f"  [{ok_mu120}] mu_off120 − mid mean = {mu120_mid:+.2f}  (target |.| ≤ 5)", flush=True)
        ok_iou = "OK " if n_r["iou_overall"] > 0.40 else "FAIL"
        print(f"  [{ok_iou}] overall IoU base={base_r['iou_overall']:.4f} → "
              f"center_rw0={n_r['iou_overall']:.4f}  (thr > 0.40)", flush=True)

        # Attribution: did dropping right_weight move mu closer to mid?
        # (Caller needs S8 mu sanity to compare directly; we report the S9 value
        # so the user can diff manually.)
        print("\n  ====== Right-cens attribution check ======", flush=True)
        if -5.0 <= mu60_mid <= 5.0:
            print("  → right-cens force CONFIRMED as the culprit: "
                  "removing right_weight let mu reach mid. "
                  "Adopt center + right_weight=0 (or small).", flush=True)
        elif mu60_mid > 30.0:
            print("  → mu still very late despite right_weight=0; "
                  "right-cens is NOT the main cause. "
                  "Suspect another structural pull (e.g., data lead distribution, "
                  "head_mu init at sigmoid(0)=0.5 → mu_init ≈ 150 → drift toward "
                  "data majority).", flush=True)
        else:
            print(f"  → mu partially recovered (mu − mid = {mu60_mid:+.2f}); "
                  "right_weight is one component but not sole. "
                  "Try further reducing or also try gaussian_mu_max bound, etc.", flush=True)


if __name__ == "__main__":
    main()
