"""Exp 1 — mu calibration (WBPH 2024, ckpt-norm). IoU improvement, NOT coverage.

Learn an additive mu bias b on VAL only (residual = true_mid - mu), apply to TEST
(mu_corrected = mu + b), recompute the fixed-sigma Gaussian 80% interval, and evaluate
combined with the coverage-aware selector. Compares calibration granularities:
  none | global | offset | alert_bin
against deploy_q20 and oracle_feasible. Metrics: IoU80_overall, MAE_center, PI_hit.

NO test labels used to learn the bias or the selector. ckpt-norm grid only.
Reuses rice/scripts/phase_t_wbph_iou_common.py.

Example
-------
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_mu_calibration --force
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_mu_calibration \
      --seeds 0 1 2 3 4 --sigma 5.0 --force
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

import rice.scripts.phase_t_wbph_iou_common as C

KINDS = ["none", "global", "offset", "alert_bin"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--variants", nargs="+", default=["baseline", "direct_neighbor"])
    ap.add_argument("--out-dir", default=str(C.OUT_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "mu_calibration_summary.csv"
    if out_csv.exists() and not args.force:
        raise SystemExit(f"Refuse to overwrite {out_csv} (use --force)")

    grid, clim_mid, disp = C.load_inputs()
    vchk = C.validate_reconstruction(grid, sigma=5.0)
    print(f"[validate] recompute vs grid iou80 max|Δ| (sigma=5): {vchk}  (≈0 => faithful)")

    rows = []
    for variant in args.variants:
        gv = grid[grid.variant == variant]
        val_feas = gv[gv.split == "val"]   # all val grid rows are feasible (real)
        for kind in KINDS:
            bias_fn = C.learn_bias(val_feas, kind)
            res = C.cov_aware_eval(grid, variant, disp, clim_mid,
                                   C.COARSE_OFFSETS, args.sigma, bias_fn, args.seeds)
            rows.append({"variant": variant, "calibration": kind,
                         "sigma": args.sigma, "n_seeds": len(args.seeds), **res})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv}\n")

    cols = ["variant", "calibration", "cov_IoU80_overall_mean", "cov_IoU80_std",
            "cov_MAE_center_mean", "cov_PI_hit_mean", "cov_mean_offset_mean",
            "deploy_q20_IoU80_overall", "oracle_IoU80_overall"]
    print("=== Exp1 mu calibration × coverage-aware selector (WBPH 2024) ===")
    print(df[cols].to_string(index=False))


if __name__ == "__main__":
    main()
