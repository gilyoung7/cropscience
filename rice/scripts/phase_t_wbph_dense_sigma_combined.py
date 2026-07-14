"""Combined experiment — dense offset grid x wider sigma (WBPH 2024, ckpt-norm).

Question: how high does the coverage-aware selector's realized IoU80 go when we
combine the denser offset grid (Exp2) with a wider interval sigma (Exp3)? mu
calibration is intentionally EXCLUDED (it hurt — see Exp1).

Setup
  offsets = 7,14,21,28,30,35,42,45,49,56,60   (dense; from wbph_dense_offset_grid.csv)
  sigma   = 5,6,7,8,9,10,12                    (extended sweep)
  per (variant, sigma): cov-aware selector (seed-mean) + oracle_feasible, with metrics
  realized IoU80_overall, coverage, mean_offset, MAE_center, PI_hit, interval width,
  and the selected-offset distribution.

sigma selection is by VAL cov-aware selector IoU80 (no test labels); the val-best sigma's
TEST metrics are reported separately. ckpt-norm only; never overwrites without --force.
Requires the dense grid first:  python -m rice.scripts.phase_t_wbph_dense_offset_grid --force

Example
-------
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_dense_sigma_combined --force
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_dense_sigma_combined \
      --sigmas 5 6 7 8 9 10 12 --seeds 0 1 2 3 4 --force
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import rice.scripts.phase_t_wbph_iou_common as C

DENSE_GRID = C.OUT_DIR / "wbph_dense_offset_grid.csv"
OUT_DIR = C.RICE_ROOT / "outputs/diag/stage2_ckptnorm_selector_wbph/dense_sigma_combined"
DENSE_OFFSETS = [7, 14, 21, 28, 30, 35, 42, 45, 49, 56, 60]
DEFAULT_SIGMAS = [5, 6, 7, 8, 9, 10, 12]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+", default=DEFAULT_SIGMAS)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variants", nargs="+", default=["baseline", "direct_neighbor"])
    ap.add_argument("--dense-grid", default=str(DENSE_GRID))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    summ_csv = out_dir / "combined_summary.csv"
    dist_csv = out_dir / "offset_distribution.csv"
    best_csv = out_dir / "best_sigma_selected.csv"
    for p in (summ_csv, dist_csv, best_csv):
        if p.exists() and not args.force:
            raise SystemExit(f"Refuse to overwrite {p} (use --force)")
    if not Path(args.dense_grid).exists():
        raise SystemExit(f"Dense grid not found: {args.dense_grid}. "
                         "Run phase_t_wbph_dense_offset_grid.py first.")

    grid = pd.read_csv(args.dense_grid)
    _g, clim_mid, disp = C.load_inputs()  # clim + dispatch; grid replaced by dense
    print(f"[validate] dense grid recompute vs iou80 max|Δ| (σ=5): {C.validate_reconstruction(grid, 5.0)}")

    offsets = sorted(int(o) for o in DENSE_OFFSETS)
    none_bias = lambda o, a: 0.0
    rows, dist_rows = [], []

    for variant in args.variants:
        gv = grid[grid.variant == variant]
        for sigma in args.sigmas:
            val_c = C.build_cand(gv[gv.split == "val"], disp, clim_mid, offsets, sigma, none_bias)
            test_c = C.build_cand(gv[gv.split == "test"], disp, clim_mid, offsets, sigma, none_bias)
            n_val = val_c.sample_id.nunique()
            n_test = test_c.sample_id.nunique()

            val_scores, test_metrics, pick_counts = [], [], {o: 0 for o in offsets}
            for sd in args.seeds:
                reg = C.train_selector(val_c, "target_cov", sd)
                # VAL selection signal (val labels only)
                val_scores.append(C.realized(val_c, C.pick_argmax(val_c, reg), n_val)["IoU80_overall"])
                # TEST eval
                pm = C.pick_argmax(test_c, reg)
                test_metrics.append(C.realized(test_c, pm, n_test))
                for o in pm.values():
                    pick_counts[int(o)] += 1
            tm = pd.DataFrame(test_metrics)
            orac = C.realized(test_c, C.pick_oracle(test_c), n_test)
            row = {
                "variant": variant, "sigma": sigma, "n_seeds": len(args.seeds),
                "val_cov_IoU80": round(float(np.mean(val_scores)), 4),
                "cov_IoU80_overall_mean": round(float(tm["IoU80_overall"].mean()), 4),
                "cov_IoU80_std": round(float(tm["IoU80_overall"].std()), 4),
                "cov_coverage_mean": round(float(tm["coverage"].mean()), 4),
                "cov_mean_offset_mean": round(float(tm["mean_offset"].mean()), 2),
                "cov_MAE_center_mean": round(float(tm["MAE_center"].mean()), 4),
                "cov_PI_hit_mean": round(float(tm["PI_hit"].mean()), 4),
                "cov_width80_mean": round(float(tm["width80"].mean()), 2),
                "oracle_IoU80_overall": orac["IoU80_overall"],
                "oracle_MAE_center": orac["MAE_center"],
                "oracle_PI_hit": orac["PI_hit"],
            }
            rows.append(row)
            # selected-offset distribution = mean count per offset across seeds
            for o in offsets:
                dist_rows.append({"variant": variant, "sigma": sigma, "offset": o,
                                  "mean_count": round(pick_counts[o] / len(args.seeds), 2)})

    summ = pd.DataFrame(rows)
    dist = pd.DataFrame(dist_rows)
    summ.to_csv(summ_csv, index=False)
    dist.to_csv(dist_csv, index=False)

    # val-selected best sigma per variant -> test
    picks = []
    for variant in args.variants:
        sub = summ[summ.variant == variant]
        best = sub.loc[sub["val_cov_IoU80"].idxmax()]
        picks.append({"variant": variant, "best_sigma_by_val": best["sigma"],
                      "val_cov_IoU80": best["val_cov_IoU80"],
                      "test_cov_IoU80": best["cov_IoU80_overall_mean"],
                      "test_cov_coverage": best["cov_coverage_mean"],
                      "test_cov_MAE_center": best["cov_MAE_center_mean"],
                      "test_cov_PI_hit": best["cov_PI_hit_mean"],
                      "test_cov_width80": best["cov_width80_mean"],
                      "test_oracle_IoU80": best["oracle_IoU80_overall"]})
    pk = pd.DataFrame(picks)
    pk.to_csv(best_csv, index=False)
    print(f"[done] wrote {summ_csv.name}, {dist_csv.name}, {best_csv.name} to {out_dir}\n")

    print("=== dense offset x sigma sweep — coverage-aware selector (WBPH 2024) ===")
    cols = ["variant", "sigma", "val_cov_IoU80", "cov_IoU80_overall_mean", "cov_IoU80_std",
            "cov_coverage_mean", "cov_mean_offset_mean", "cov_MAE_center_mean",
            "cov_PI_hit_mean", "cov_width80_mean", "oracle_IoU80_overall"]
    print(summ[cols].to_string(index=False))
    print("\n=== VAL-selected best sigma -> TEST ===")
    print(pk.to_string(index=False))
    print("\n  주의: best sigma가 sweep 경계(12)면 최적이 더 넓을 수 있음 -> sweep 확장 검토.")


if __name__ == "__main__":
    main()
