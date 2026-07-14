"""Phase S5 inference — refresh sample_grid with the lw-weighted 2-sided ckpt.

Calls `phase_r_oracle_iou` with two models:
    [A] the original 2-sided final  (baseline for comparison)
    [B] the new 2-sided + lw{W}     (this experiment)

Both use σ=5.0.  Offsets remain the original {60, 90, 105, 120} (4-way), which
matches phase_s3 C_old.  The lw model can optionally also receive
--per_model_extra_offsets if needed downstream.

Output:
    outputs_phase_s5_sample_grid.csv
    outputs_phase_s5_oracle.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PY = str(REPO_ROOT / ".venv" / "bin" / "python")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", type=str, default="sheath_blight")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--stage1_ckpt", type=str,
                    default="rice/outputs_stage1/sheath_blight_yearsplit2023-24/ckpt/"
                            "event_run4_xgb_nowcast_w28_s1_tpos_yearsplit_ymin2002.pt")
    ap.add_argument("--baseline_2sided_ckpt", type=str,
                    default="rice/outputs_stage2_sheath_blight_d15_asym25_2sided_final_aw25/"
                            "ckpt/checkpoint_run4.pt",
                    help="Original 2-sided ckpt (for comparison row in sample_grid).")
    ap.add_argument("--lw_ckpt", type=str, required=True,
                    help="Path to the new lw-weighted 2-sided ckpt (Phase S5 final).")
    ap.add_argument("--lw_label", type=str, default="D=15 2-sided lw3",
                    help="Label used in the sample_grid 'model' column for the lw model.")
    ap.add_argument("--sigma_2sided", type=float, default=5.0)
    ap.add_argument("--sigma_lw", type=float, default=5.0)
    ap.add_argument("--offsets", type=str, default="60,90,105,120")
    ap.add_argument("--sample_grid_csv", type=str,
                    default="rice/outputs/diag/phase_s5_sample_grid.csv")
    ap.add_argument("--out_csv", type=str, default="rice/outputs/diag/phase_s5_oracle.csv")
    args = ap.parse_args()

    for label, p in [("stage1", args.stage1_ckpt),
                      ("baseline_2sided", args.baseline_2sided_ckpt),
                      ("lw", args.lw_ckpt)]:
        if not os.path.exists(p):
            raise SystemExit(f"[abort] {label} ckpt not found: {p}")

    base_label = "D=15 2-sided (asym=25)"
    models = (f"{base_label}|{args.baseline_2sided_ckpt};"
              f"{args.lw_label}|{args.lw_ckpt}")
    per_model_sigma = (f"{base_label}={args.sigma_2sided},"
                       f"{args.lw_label}={args.sigma_lw}")

    cmd = [
        PY, "-m", "rice.scripts.phase_r_oracle_iou",
        "--pest", args.pest, "--run", str(args.run),
        "--val_year", str(args.val_year),
        "--test_year_min", str(args.test_year_min),
        "--test_year_max", str(args.test_year_max),
        "--stage1_ckpt", args.stage1_ckpt,
        "--models", models,
        "--offsets", args.offsets,
        "--oracle_sigma", str(args.sigma_lw),
        "--sigma_sweep", "2.5,3.0,3.5,4.0,4.5,5.0,6.0",
        "--sigma_sweep_offsets", "105,120",
        "--per_model_sigma", per_model_sigma,
        "--sample_grid_csv", args.sample_grid_csv,
        "--out_csv", args.out_csv,
    ]
    print("=" * 70)
    print("Phase S5 inference (phase_r_oracle_iou)")
    print(" ".join(cmd))
    print("=" * 70)
    env = {**os.environ,
           "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
               "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")}
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        raise SystemExit(f"[abort] phase_r_oracle_iou failed (rc={rc})")
    if not os.path.exists(args.sample_grid_csv):
        raise SystemExit(f"[abort] sample_grid not produced: {args.sample_grid_csv}")
    print(f"\n[done] sample_grid = {args.sample_grid_csv}")
    print(f"[done] oracle      = {args.out_csv}")


if __name__ == "__main__":
    main()
