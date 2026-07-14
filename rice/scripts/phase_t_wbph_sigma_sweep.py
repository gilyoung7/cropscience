"""Exp 3 — interval width (sigma) sweep (WBPH 2024, ckpt-norm). IoU improvement.

sigma is a POST-HOC interval-width parameter on the fixed-Gaussian head: the 80%
shortest-mass interval (and its IoU vs the true interval) can be recomputed for any
sigma WITHOUT re-running the model. Sweeps sigma in {4,5,6,7,8}, combined with the
coverage-aware selector. Reports realized IoU80_overall / PI_hit / MAE_center per
sigma, and applies the VAL-selected best sigma to TEST (no test labels).

ckpt-norm only. Reuses phase_t_wbph_iou_common.py.

Example
-------
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_sigma_sweep --force
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_sigma_sweep --sigmas 4 5 6 7 8 --force
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import rice.scripts.phase_t_wbph_iou_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigmas", type=float, nargs="+", default=[4, 5, 6, 7, 8])
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--variants", nargs="+", default=["baseline", "direct_neighbor"])
    ap.add_argument("--out-dir", default=str(C.OUT_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    sweep_csv = out_dir / "sigma_sweep_summary.csv"
    pick_csv = out_dir / "sigma_val_selected.csv"
    for p in (sweep_csv, pick_csv):
        if p.exists() and not args.force:
            raise SystemExit(f"Refuse to overwrite {p} (use --force)")

    grid, clim_mid, disp = C.load_inputs()
    print(f"[validate] recompute vs grid iou80 max|Δ| (sigma=5): {C.validate_reconstruction(grid, 5.0)}")

    none_bias = lambda o, a: 0.0
    rows = []
    for variant in args.variants:
        for sigma in args.sigmas:
            # VAL score for sigma selection: cov-aware selector mean IoU80 on... we need a
            # val-only criterion. Use val oracle_feasible IoU80 at this sigma as the val
            # selection signal (label-free wrt TEST; uses val labels only, which is allowed).
            gv = grid[grid.variant == variant]
            val_c = C.build_cand(gv[gv.split == "val"], disp, clim_mid, C.COARSE_OFFSETS, sigma, none_bias)
            n_val = val_c.sample_id.nunique()
            val_cov = []
            for sd in args.seeds:
                reg = C.train_selector(val_c, "target_cov", sd)
                # evaluate selector ON VAL (val labels only) as the sigma-selection score
                val_cov.append(C.realized(val_c, C.pick_argmax(val_c, reg), n_val)["IoU80_overall"])
            val_score = float(np.mean(val_cov))

            res = C.cov_aware_eval(grid, variant, disp, clim_mid, C.COARSE_OFFSETS,
                                   sigma, none_bias, args.seeds)
            rows.append({"variant": variant, "sigma": sigma, "val_cov_IoU80": round(val_score, 4),
                         "n_seeds": len(args.seeds), **res})
    df = pd.DataFrame(rows)
    df.to_csv(sweep_csv, index=False)

    # val-selected best sigma per variant -> its TEST metrics
    picks = []
    for variant in args.variants:
        sub = df[df.variant == variant]
        best = sub.loc[sub["val_cov_IoU80"].idxmax()]
        picks.append({"variant": variant, "best_sigma_by_val": best["sigma"],
                      "val_cov_IoU80": best["val_cov_IoU80"],
                      "test_cov_IoU80": best["cov_IoU80_overall_mean"],
                      "test_cov_MAE_center": best["cov_MAE_center_mean"],
                      "test_cov_PI_hit": best["cov_PI_hit_mean"],
                      "test_deploy_q20_IoU80": best["deploy_q20_IoU80_overall"],
                      "test_oracle_IoU80": best["oracle_IoU80_overall"]})
    pk = pd.DataFrame(picks)
    pk.to_csv(pick_csv, index=False)
    print(f"[done] wrote {sweep_csv}, {pick_csv}\n")

    print("=== Exp3 sigma sweep × coverage-aware selector (WBPH 2024) ===")
    cols = ["variant", "sigma", "val_cov_IoU80", "cov_IoU80_overall_mean", "cov_IoU80_std",
            "cov_MAE_center_mean", "cov_PI_hit_mean", "oracle_IoU80_overall"]
    print(df[cols].to_string(index=False))
    print("\n=== VAL-selected best sigma -> TEST ===")
    print(pk.to_string(index=False))


if __name__ == "__main__":
    main()
