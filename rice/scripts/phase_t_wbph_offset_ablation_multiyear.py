"""Offset-grid ablation + year stability (WBPH 2022/2023/2024, ckpt-norm, sigma=8).

Compares offset grids under the FIXED practical setting (coverage-aware selector, sigma=8):
  dense     : 7,14,21,28,30,35,42,45,49,56,60        (selector)
  uniform5  : 5,10,15,20,25,30,35,40,45,50,55,60      (selector)
  full_1day : 1..75                                   (oracle_feasible CEILING only, no selector)
for baseline AND direct_neighbor, per year, with a same-seed DN-baseline paired diff
where both variants exist.

Reads the multiyear grid from phase_t_wbph_multiyear_grid.py (offsets 1..75). Per-year
climatology and dispatch features are loaded per the matching batch (they differ by year:
2023 uses gate_D_history; 2022/2024 use gate_dispatch_group_tau).

IMPORTANT — read me:
  * Uses whatever per-year HELD-OUT ckpts are present in the multiyear grid. Baseline has
    2022/2023/2024; direct_neighbor originally had 2024 only, plus 2022/2023 ROLLING ckpts
    once run_s2n_direct_rolling.sh has produced them (each trained val=Y-1/test=Y, never
    reusing the 2024 ckpt). (variant, year) pairs absent from the grid are skipped, and the
    printed [note] lists exactly which combinations were evaluated + where the DN-baseline
    paired diff is available.

ckpt-norm only. No mu calibration. No wrong-norm. Does not overwrite without --force.
Output: rice/outputs/diag/stage2_ckptnorm_selector_wbph/offset_ablation_multiyear/

Example
-------
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_offset_ablation_multiyear --force
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from rice.configs.base import RICE_ROOT
import rice.scripts.phase_t_wbph_iou_common as C
from rice.scripts.phase_b_stage2_offset_selector_v2_ranking import SAMPLE_FEATURES

SIGMA = 8.0
GRID = RICE_ROOT / "outputs/diag/stage2_ckptnorm_selector_wbph/multiyear/wbph_grid_1to75_multiyear.csv"
OUT_DIR = RICE_ROOT / "outputs/diag/stage2_ckptnorm_selector_wbph/offset_ablation_multiyear"
DENSE = [7, 14, 21, 28, 30, 35, 42, 45, 49, 56, 60]
UNIFORM5 = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
SELECTOR_SETS = {"dense": DENSE, "uniform5": UNIFORM5}

YEAR_BATCH = {2022: "batch_2022_baseline", 2023: "batch_2023_baseline", 2024: "batch_2024_bestgate"}
YEAR_DISPATCH = {
    2022: "batch_2022_baseline/WBPH/gate_dispatch_group_tau_R088_features_per_sy.csv",
    2023: "batch_2023_baseline/WBPH/gate_D_history_R088_features_per_sy.csv",
    2024: "batch_2024_bestgate/WBPH/gate_dispatch_group_tau_R088_features_per_sy.csv",
}


def load_year_clim_disp(year: int):
    clim = float(pd.read_csv(RICE_ROOT / "outputs/stage2" / YEAR_BATCH[year] / "WBPH" /
                             "climatology_train_stats.csv").iloc[0]["mean_mid"])
    d = pd.read_csv(RICE_ROOT / "outputs/stage2" / YEAR_DISPATCH[year])
    d["sample_id"] = d["site"].astype(str) + "-" + d["year"].astype(int).astype(str)
    disp = d[["sample_id"] + SAMPLE_FEATURES + ["dispatch_branch"]].drop_duplicates("sample_id")
    return clim, disp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--variants", nargs="+", default=["baseline", "direct_neighbor"])
    ap.add_argument("--grid", default=str(GRID))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    for f in ("ablation_summary.csv", "paired_summary.csv"):
        if (out_dir / f).exists() and not args.force:
            raise SystemExit(f"Refuse to overwrite {out_dir / f} (use --force)")
    if not Path(args.grid).exists():
        raise SystemExit(f"Multiyear grid not found: {args.grid}. Run phase_t_wbph_multiyear_grid.py first.")

    grid = pd.read_csv(args.grid)
    print(f"[setting] sigma={SIGMA} FIXED, no mu-calibration, seeds={args.seeds}")
    print(f"[validate 2024] {C.validate_reconstruction(grid[grid.year == 2024], 5.0)}  "
          "(other years have no independent reference)\n")

    rows = []
    # iou_map[(year, oset, variant, seed)][sample_id] = realized iou80  (for paired diff)
    iou_map = {}

    for year in args.years:
        gy = grid[grid.year == year]
        if gy.empty:
            continue
        clim, disp = load_year_clim_disp(year)
        for variant in args.variants:
            gv = gy[gy.variant == variant]
            if gv.empty:
                print(f"[skip] {variant}/{year}: not in grid (no held-out ckpt)")
                continue
            # selector offset sets
            for oname, oset in SELECTOR_SETS.items():
                offs = sorted(set(oset) & set(int(o) for o in gv.offset.unique()))
                val_c = C.build_cand(gv[gv.split == "val"], disp, clim, offs, SIGMA)
                test_c = C.build_cand(gv[gv.split == "test"], disp, clim, offs, SIGMA)
                n_total = test_c.sample_id.nunique()
                seed_m = []
                for sd in args.seeds:
                    reg = C.train_selector(val_c, "target_cov", sd)
                    pm = C.pick_argmax(test_c, reg)
                    seed_m.append(C.realized(test_c, pm, n_total))
                    pr = pd.DataFrame([test_c[(test_c.sample_id == s) & (test_c.offset == o)].iloc[0]
                                       for s, o in pm.items()])
                    iou_map[(year, oname, variant, sd)] = dict(zip(pr.sample_id, pr.iou80_real))
                sm = pd.DataFrame(seed_m)
                orac = C.realized(test_c, C.pick_oracle(test_c), n_total)
                rows.append({
                    "year": year, "variant": variant, "offset_set": oname, "sigma": SIGMA,
                    "n_offsets": len(offs), "n_total": n_total, "n_seeds": len(args.seeds),
                    "IoU80_overall_mean": round(float(sm["IoU80_overall"].mean()), 4),
                    "IoU80_overall_std": round(float(sm["IoU80_overall"].std()), 4),
                    "coverage_mean": round(float(sm["coverage"].mean()), 4),
                    "late_count_mean": round(float(sm["late_count"].mean()), 2),
                    "mean_offset_mean": round(float(sm["mean_offset"].mean()), 2),
                    "MAE_center_mean": round(float(sm["MAE_center"].mean()), 4),
                    "PI_hit_mean": round(float(sm["PI_hit"].mean()), 4),
                    "width80_mean": round(float(sm["width80"].mean()), 2),
                    "oracle_IoU80_overall": orac["IoU80_overall"],
                })
            # full 1-day oracle ceiling (no selector)
            offs_full = sorted(int(o) for o in gv.offset.unique())
            test_full = C.build_cand(gv[gv.split == "test"], disp, clim, offs_full, SIGMA)
            n_total = test_full.sample_id.nunique()
            of = C.realized(test_full, C.pick_oracle(test_full), n_total)
            rows.append({
                "year": year, "variant": variant, "offset_set": "full_1day_oracle", "sigma": SIGMA,
                "n_offsets": len(offs_full), "n_total": n_total, "n_seeds": 0,
                "IoU80_overall_mean": np.nan, "IoU80_overall_std": np.nan,
                "coverage_mean": of["coverage"], "late_count_mean": of["late_count"],
                "mean_offset_mean": of["mean_offset"], "MAE_center_mean": of["MAE_center"],
                "PI_hit_mean": of["PI_hit"], "width80_mean": of["width80"],
                "oracle_IoU80_overall": of["IoU80_overall"],
            })

    summ = pd.DataFrame(rows)
    summ.to_csv(out_dir / "ablation_summary.csv", index=False)

    # ---- paired DN - baseline (same seed) where both variants exist ----
    paired = []
    for year in args.years:
        for oname in SELECTOR_SETS:
            for sd in args.seeds:
                bm = iou_map.get((year, oname, "baseline", sd))
                dm = iou_map.get((year, oname, "direct_neighbor", sd))
                if bm is None or dm is None:
                    continue
                sids = sorted(set(bm) & set(dm))
                diffs = np.array([dm[s] - bm[s] for s in sids])
                paired.append({
                    "year": year, "offset_set": oname, "seed": sd, "n": len(sids),
                    "mean_diff_dn_minus_base": round(float(diffs.mean()), 4),
                    "dn_win": int((diffs > 1e-9).sum()), "base_win": int((diffs < -1e-9).sum()),
                    "tie": int((np.abs(diffs) <= 1e-9).sum()),
                })
    paired_df = pd.DataFrame(paired)
    if not paired_df.empty:
        ov = (paired_df.groupby(["year", "offset_set"])
              .agg(mean_diff=("mean_diff_dn_minus_base", "mean"),
                   dn_win=("dn_win", "sum"), base_win=("base_win", "sum"), tie=("tie", "sum"),
                   n_seeds=("seed", "nunique")).round(4).reset_index())
    else:
        ov = pd.DataFrame()
    paired_df.to_csv(out_dir / "paired_per_seed.csv", index=False)
    ov.to_csv(out_dir / "paired_summary.csv", index=False)
    print(f"[done] wrote ablation_summary.csv, paired_per_seed.csv, paired_summary.csv to {out_dir}\n")

    print("=== offset ablation × year (sigma=8, cov-aware selector) ===")
    cols = ["year", "variant", "offset_set", "n_offsets", "IoU80_overall_mean", "IoU80_overall_std",
            "coverage_mean", "mean_offset_mean", "MAE_center_mean", "PI_hit_mean",
            "width80_mean", "oracle_IoU80_overall"]
    print(summ[cols].to_string(index=False))
    if not ov.empty:
        print("\n=== DN - baseline paired (same seed; only where both ckpts exist) ===")
        print(ov.to_string(index=False))

    # data-driven coverage note (reflects which (variant, year) actually entered the grid)
    present = summ[["variant", "year"]].drop_duplicates()
    by_variant = {v: sorted(present[present.variant == v].year.unique().tolist())
                  for v in sorted(present.variant.unique())}
    dn_years = set(by_variant.get("direct_neighbor", []))
    base_years = set(by_variant.get("baseline", []))
    both = sorted(dn_years & base_years)
    dn_only = sorted(dn_years - base_years)
    base_only = sorted(base_years - dn_years)
    print("\n[note] (variant, year) evaluated in this grid:")
    for v, ys in by_variant.items():
        print(f"       {v:16s}: {ys}")
    print(f"       DN-baseline paired available: {both}"
          + (f" | baseline-only years: {base_only}" if base_only else "")
          + (f" | DN-only years: {dn_only}" if dn_only else ""))


if __name__ == "__main__":
    main()
