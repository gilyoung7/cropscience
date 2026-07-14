"""
Phase S2.5 — Operational shift sweep on three concrete setups.

Setups (fixed σ per model, drawn from sample_grid):
    A) phenobias + fixed offset=120 + σ=4.5   (current operational anchor)
    B) 2-sided  + selector(v3_mu_only/logreg) + σ=5.0   (selector candidate)
    C) baseline + fixed offset=120 + σ=4.0    (reference)

Shift values:
    {25, 30, 35, 40, 42, 44, 46, 48, 50}

Per (setup, shift) cell:
    For each in-cohort sample s:
        lead_s = L_s − (mu_s + 1.96 σ_s − shift)
    Buckets: MISSED / TOO_LATE / URGENT / IDEAL / ADVANCE / TOO_EARLY
    Aggregations (denom N_TOTAL_TEST = 575):
        IoU_overall      = Σ iou_s / 575     (iou from sample_grid at the matched offset)
        P_ideal_overall  = #{14 ≤ lead < 30} / 575
        P_useful_B       = #{7  ≤ lead < 45} / 575
        P_failed         = (MISSED + TOO_LATE + (575 − n_match)) / 575

The 2-sided selector OOF is trained once with seed=42; only the operational shift
changes the bucket boundaries during evaluation. CSV: outputs/phase_s2/shift_sweep.csv.
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
    N_TOTAL_TEST, OFFSETS, P_IDEAL_LOW, P_IDEAL_HIGH,
    stage1_test_scores, compute_stage1_features_per_sample,
    build_per_sample_table, evaluate_predictions,
)
from rice.scripts.phase_s2_selector import (
    STAGE1_FEATURES, MU_FEATURES, MU_MISSING_INDICATORS, STATIC_FEATURES,
    attach_mu_features, feature_cols_for_set, run_cv,
)


BUCKET_ORDER = ["MISSED", "TOO_LATE", "URGENT", "IDEAL", "ADVANCE", "TOO_EARLY"]


def per_sample_table_for(grid: pd.DataFrame, label_substr: str,
                          stage1_features: pd.DataFrame) -> pd.DataFrame:
    """Build the per-sample wide table for the model whose 'model' contains label_substr."""
    sub = grid[grid["model"].str.contains(label_substr, regex=False)]
    if sub.empty:
        raise SystemExit(f"[abort] no sample_grid rows for '{label_substr}'")
    label = sub["model"].iloc[0]
    print(f"  [{label}] sample_grid rows={len(sub)}")
    per_sample = build_per_sample_table(sub.copy())
    per_sample = per_sample[per_sample["n_offsets_matched"] > 0].copy()
    per_sample = per_sample.merge(stage1_features, left_on="sample_id",
                                   right_index=True, how="left")
    per_sample["tstar_doy_feat"] = per_sample["t_star_doy"].astype(float)
    per_sample["year_feat"] = per_sample["year"].astype(float)
    per_sample = attach_mu_features(per_sample)
    must_have = STAGE1_FEATURES + ["sigma", "L", "t_star_doy"]
    bad = per_sample[must_have].isna().any(axis=1)
    if bad.any():
        print(f"    dropped {int(bad.sum())} rows with stage1/static NaN")
        per_sample = per_sample[~bad].copy()
    return per_sample


def evaluate_setup(per_sample: pd.DataFrame, pred_offsets: np.ndarray,
                    shift: float, n_total: int) -> dict:
    """Return a row dict with metrics + bucket counts at the given shift."""
    m = evaluate_predictions(per_sample, pred_offsets, shift=shift, debug_label=None)
    b = m["bucket_counts"]
    n_match = int(sum(b.values()))
    n_M = int(b.get("MISSED", 0))
    n_TL = int(b.get("TOO_LATE", 0))
    n_U = int(b.get("URGENT", 0))
    n_I = int(b.get("IDEAL", 0))
    n_A = int(b.get("ADVANCE", 0))
    n_TE = int(b.get("TOO_EARLY", 0))
    n_used = len(per_sample)
    n_out_cohort = n_total - n_used   # we evaluate over per_sample population only
    row = {
        "shift": float(shift),
        "n_used": n_used,
        "n_match_buckets": n_match,
        "MISSED_n": n_M, "TOO_LATE_n": n_TL, "URGENT_n": n_U,
        "IDEAL_n": n_I, "ADVANCE_n": n_A, "TOO_EARLY_n": n_TE,
        "MISSED_pct": 100.0 * n_M / n_total,
        "TOO_LATE_pct": 100.0 * n_TL / n_total,
        "URGENT_pct": 100.0 * n_U / n_total,
        "IDEAL_pct": 100.0 * n_I / n_total,
        "ADVANCE_pct": 100.0 * n_A / n_total,
        "TOO_EARLY_pct": 100.0 * n_TE / n_total,
        "IoU_overall": float(m["IoU_overall"]),
        "P_ideal_overall": float(m["P_ideal_overall"]),
        "P_useful_A_overall": 100.0 * (n_TL + n_U + n_I + n_A) / n_total / 100.0,
        "P_useful_B_overall": 100.0 * (n_U + n_I + n_A) / n_total / 100.0,
        "P_failed_overall": 100.0 * (n_M + n_TL + (n_total - n_used)) / n_total / 100.0,
    }
    return row


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
    p.add_argument("--out_dir", type=str, default="rice/outputs/phase_s2/")
    p.add_argument("--shifts", type=str, default="25,30,35,40,42,44,46,48,50")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shifts = [float(x) for x in str(args.shifts).split(",") if x.strip()]
    print(f"[config] shifts={shifts}  N_TOTAL_TEST={N_TOTAL_TEST}")

    if not os.path.exists(args.sample_grid):
        raise SystemExit(f"[abort] sample_grid not found: {args.sample_grid}")
    grid = pd.read_csv(args.sample_grid)
    print(f"[input] sample_grid rows={len(grid)}  models={list(grid['model'].unique())}")

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

    print("\n----- Setup A: phenobias + fixed off=120 (σ=4.5 from grid) -----")
    ps_A = per_sample_table_for(grid, "phenobias", stage1_features)
    # Filter out rows where σ != 4.5 to be safe (sample_grid stores σ per model row)
    pred_A = np.full(len(ps_A), 120, dtype=int)
    sigma_A_mean = float(ps_A["sigma"].mean())
    print(f"  [A] n={len(ps_A)}  σ_mean={sigma_A_mean:.2f}")

    print("\n----- Setup B: 2-sided + selector(v3_mu_only/logreg) (σ=5.0 from grid) -----")
    ps_B = per_sample_table_for(grid, "2-sided", stage1_features)
    feat_cols = feature_cols_for_set("v3_mu_only")
    impute_cols = [c for c in feat_cols if c in (MU_FEATURES + MU_MISSING_INDICATORS)]
    cv = run_cv(ps_B, feat_cols, "best_offset",
                classifier="logreg", args=args, shift=30.0,
                impute_cols=impute_cols)
    # OOF predictions live on oof_df; align to ps_B order via sample_id.
    oof_df = cv["oof_df"]
    pred_B_lookup = dict(zip(oof_df["sample_id"], oof_df["pred_offset"]))
    pred_B = np.asarray([int(pred_B_lookup[sid]) for sid in ps_B["sample_id"]], dtype=int)
    sigma_B_mean = float(ps_B["sigma"].mean())
    print(f"  [B] n={len(ps_B)}  σ_mean={sigma_B_mean:.2f}  "
          f"selector pred dist: {dict(zip(*np.unique(pred_B, return_counts=True)))}")

    print("\n----- Setup C: baseline + fixed off=120 (σ=4.0 from grid) -----")
    ps_C = per_sample_table_for(grid, "baseline", stage1_features)
    pred_C = np.full(len(ps_C), 120, dtype=int)
    sigma_C_mean = float(ps_C["sigma"].mean())
    print(f"  [C] n={len(ps_C)}  σ_mean={sigma_C_mean:.2f}")

    print("\n----- Shift sweep -----")
    rows = []
    for shift in shifts:
        for label, ps, preds in [
            ("phenobias_fixed120",   ps_A, pred_A),
            ("2sided_selector_v3lr", ps_B, pred_B),
            ("baseline_fixed120",    ps_C, pred_C),
        ]:
            m = evaluate_setup(ps, preds, shift=shift, n_total=N_TOTAL_TEST)
            m["setup"] = label
            rows.append(m)
        # Compact stdout view for this shift
        slc = [r for r in rows if r["shift"] == shift]
        line = f"  shift={shift:>5.1f} | "
        for sl in slc:
            line += (f"{sl['setup'][:9]:<9} P_ideal={sl['P_ideal_overall']*100:>5.2f}% "
                     f"P_failed={sl['P_failed_overall']*100:>5.2f}% IoU={sl['IoU_overall']:.4f} | ")
        print(line)

    df = pd.DataFrame(rows)
    cols = ["setup", "shift", "n_used", "n_match_buckets",
            "MISSED_n", "TOO_LATE_n", "URGENT_n", "IDEAL_n", "ADVANCE_n", "TOO_EARLY_n",
            "MISSED_pct", "TOO_LATE_pct", "URGENT_pct", "IDEAL_pct",
            "ADVANCE_pct", "TOO_EARLY_pct",
            "IoU_overall", "P_ideal_overall",
            "P_useful_A_overall", "P_useful_B_overall", "P_failed_overall"]
    df = df[cols]
    csv_path = out_dir / "shift_sweep.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n[csv] {csv_path}  ({len(df)} rows)")

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print("\n=================== Shift sweep ===================")
    show = ["setup", "shift", "n_used",
            "MISSED_pct", "TOO_LATE_pct", "URGENT_pct", "IDEAL_pct", "ADVANCE_pct", "TOO_EARLY_pct",
            "IoU_overall", "P_ideal_overall", "P_useful_B_overall", "P_failed_overall"]
    print(df[show].to_string(index=False))

    # Best-shift per setup
    print("\n=================== Best shift per setup (by P_ideal_overall) ===================")
    for setup, sub in df.groupby("setup"):
        i = sub["P_ideal_overall"].idxmax()
        r = sub.loc[i]
        print(f"  [{setup:<22}] best shift={r['shift']:>5.1f}  "
              f"P_ideal={r['P_ideal_overall']*100:>5.2f}%  "
              f"IoU={r['IoU_overall']:.4f}  "
              f"P_failed={r['P_failed_overall']*100:>5.2f}%  "
              f"P_useful_B={r['P_useful_B_overall']*100:>5.2f}%")

    print("\n=================== Best shift per setup (by IoU_overall) ===================")
    for setup, sub in df.groupby("setup"):
        i = sub["IoU_overall"].idxmax()
        r = sub.loc[i]
        print(f"  [{setup:<22}] best shift={r['shift']:>5.1f}  "
              f"IoU={r['IoU_overall']:.4f}  "
              f"P_ideal={r['P_ideal_overall']*100:>5.2f}%")


if __name__ == "__main__":
    main()
