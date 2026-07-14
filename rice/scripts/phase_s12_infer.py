"""Phase S12 inference — sample_grid with the GDD-incorporating ckpt.

Calls `phase_r_oracle_iou` with one model (the S12 final) so the grid is
internally consistent (Stage 1 with run=8 features + Stage 2 with run=8
features).  The cohort may differ from the baseline (which uses run=4
Stage 1); this is expected — comparison is done at the per-sample level
in phase_s12_eval, with explicit cohort overlap reporting.

Outputs:
    outputs_phase_s12_sample_grid.csv
    outputs_phase_s12_oracle.csv
"""
from __future__ import annotations

import argparse
import os
import subprocess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", type=str, default="sheath_blight")
    ap.add_argument("--run", type=int, default=8)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--stage1_ckpt", type=str, required=True,
                    help="Stage 1 ckpt trained with --run 8 (incl. GDD).")
    ap.add_argument("--ckpt", type=str, required=True,
                    help="Stage 2 GDD ckpt (Phase S12 final).")
    ap.add_argument("--label", type=str, default="D=15 2-sided gdd",
                    help="Label in the sample_grid 'model' column.")
    ap.add_argument("--sigma_new", type=float, default=5.0)
    ap.add_argument("--offsets", type=str, default="60,90,105,120")
    ap.add_argument("--sample_grid_csv", type=str,
                    default="rice/outputs/diag/phase_s12_sample_grid.csv")
    ap.add_argument("--out_csv", type=str, default="rice/outputs/diag/phase_s12_oracle.csv")
    args = ap.parse_args()

    for label, p in [("stage1", args.stage1_ckpt), ("gdd", args.ckpt)]:
        if not os.path.exists(p):
            raise SystemExit(f"[abort] {label} ckpt not found: {p}")

    models = f"{args.label}|{args.ckpt}"
    per_model_sigma = f"{args.label}={args.sigma_new}"

    cmd = [
        ".venv/bin/python", "-u", "-m", "rice.scripts.phase_r_oracle_iou",
        "--pest", args.pest, "--run", str(args.run),
        "--val_year", str(args.val_year),
        "--test_year_min", str(args.test_year_min),
        "--test_year_max", str(args.test_year_max),
        "--stage1_ckpt", args.stage1_ckpt,
        "--models", models,
        "--offsets", args.offsets,
        "--oracle_sigma", str(args.sigma_new),
        "--sigma_sweep", "2.5,3.0,3.5,4.0,4.5,5.0,6.0",
        "--sigma_sweep_offsets", "105,120",
        "--per_model_sigma", per_model_sigma,
        "--sample_grid_csv", args.sample_grid_csv,
        "--out_csv", args.out_csv,
    ]
    print("=" * 70, flush=True)
    print(f"Phase S12 inference (phase_r_oracle_iou, run={args.run})", flush=True)
    print(" ".join(cmd), flush=True)
    print("=" * 70, flush=True)
    env = {**os.environ,
           "PYTHONUNBUFFERED": "1",
           "PYTORCH_CUDA_ALLOC_CONF": os.environ.get(
               "PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")}
    rc = subprocess.call(cmd, env=env)
    if rc != 0:
        raise SystemExit(f"[abort] phase_r_oracle_iou failed (rc={rc})")
    if not os.path.exists(args.sample_grid_csv):
        raise SystemExit(f"[abort] sample_grid not produced: {args.sample_grid_csv}")
    print(f"\n[done] sample_grid = {args.sample_grid_csv}", flush=True)
    print(f"[done] oracle      = {args.out_csv}", flush=True)


if __name__ == "__main__":
    main()
