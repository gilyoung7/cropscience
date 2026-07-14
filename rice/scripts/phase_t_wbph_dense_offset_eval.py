"""Exp 2b — does a denser offset grid raise the ceiling / help the selector? (WBPH 2024, ckpt-norm)

Compares COARSE {7,14,21,30,45,60} (original wbph_offset_grid.csv) vs DENSE
(wbph_dense_offset_grid.csv from phase_t_wbph_dense_offset_grid.py) on:
  - oracle_feasible IoU80_overall  (ceiling — does it rise with more offsets?)
  - coverage-aware selector IoU80_overall / MAE_center / PI_hit
for baseline AND direct_neighbor. Run phase_t_wbph_dense_offset_grid.py FIRST.

NO test labels in learning. ckpt-norm only. Does not overwrite without --force.
Reuses phase_t_wbph_iou_common.py.

Example
-------
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_dense_offset_eval --force
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd

import rice.scripts.phase_t_wbph_iou_common as C

DENSE_GRID = C.OUT_DIR / "wbph_dense_offset_grid.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--sigma", type=float, default=5.0)
    ap.add_argument("--variants", nargs="+", default=["baseline", "direct_neighbor"])
    ap.add_argument("--dense-grid", default=str(DENSE_GRID))
    ap.add_argument("--out-dir", default=str(C.OUT_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "dense_offset_summary.csv"
    if out_csv.exists() and not args.force:
        raise SystemExit(f"Refuse to overwrite {out_csv} (use --force)")
    if not Path(args.dense_grid).exists():
        raise SystemExit(f"Dense grid not found: {args.dense_grid}. Run phase_t_wbph_dense_offset_grid.py first.")

    coarse_grid, clim_mid, disp = C.load_inputs()
    dense_grid = pd.read_csv(args.dense_grid)
    print(f"[validate] coarse recompute max|Δ|: {C.validate_reconstruction(coarse_grid)}")
    print(f"[validate] dense  recompute max|Δ|: {C.validate_reconstruction(dense_grid)}")

    none_bias = lambda o, a: 0.0
    rows = []
    for variant in args.variants:
        for tag, grid in [("coarse", coarse_grid), ("dense", dense_grid)]:
            offs = sorted(int(o) for o in grid[grid.variant == variant].offset.unique())
            res = C.cov_aware_eval(grid, variant, disp, clim_mid, offs,
                                   args.sigma, none_bias, args.seeds)
            rows.append({"variant": variant, "grid": tag, "n_offsets": len(offs),
                         "offsets": str(offs), **res})
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[done] wrote {out_csv}\n")

    cols = ["variant", "grid", "n_offsets", "oracle_IoU80_overall",
            "cov_IoU80_overall_mean", "cov_IoU80_std", "cov_MAE_center_mean",
            "cov_PI_hit_mean", "deploy_q20_IoU80_overall"]
    print("=== Exp2 coarse vs dense offset grid (WBPH 2024) ===")
    print(df[cols].to_string(index=False))
    print("\n  oracle_IoU80_overall 상승 => 더 촘촘한 offset이 ceiling을 올림; "
          "cov_*_mean 상승 => selector가 그걸 활용함.")


if __name__ == "__main__":
    main()
