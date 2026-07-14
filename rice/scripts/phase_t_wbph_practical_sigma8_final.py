"""FINAL practical validation — WBPH 2024 ckpt-norm:
dense offset grid + coverage-aware selector + sigma=8 (FIXED), seed sweep.

Locks in the practical setting (sigma=8) and reports baseline vs direct_neighbor with
a same-seed paired comparison. mu calibration is NOT used. ckpt-norm only. dense offsets.
Requires wbph_dense_offset_grid.csv (from phase_t_wbph_dense_offset_grid.py).

Outputs (never overwrite without --force) ->
  rice/outputs/diag/stage2_ckptnorm_selector_wbph/practical_sigma8_final/
    final_summary.csv        variant-level, seed-aggregated (+ oracle_feasible, deploy_q20 refs)
    per_sample_results.csv    one row per (variant, seed, sample_id): picked offset + realized metrics
    offset_distribution.csv   (variant, seed, offset) selected-offset counts
    paired_per_sample.csv     (seed, sample_id): DN_iou80 - baseline_iou80 (same seed, same pick policy)
    paired_summary.csv        per-seed and overall paired DN-baseline IoU80 diff + win/loss/tie counts

Metrics: realized IoU80_overall, coverage, MAE_center, PI_hit, width80_mean, mean_offset,
late_count, and per-sample IoU difference.

Example
-------
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_practical_sigma8_final --force
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_practical_sigma8_final \
      --seeds 0 1 2 3 4 5 10 42 100 --force
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

import rice.scripts.phase_t_wbph_iou_common as C

SIGMA = 8.0
DENSE_GRID = C.OUT_DIR / "wbph_dense_offset_grid.csv"
OUT_DIR = C.RICE_ROOT / "outputs/diag/stage2_ckptnorm_selector_wbph/practical_sigma8_final"
DENSE_OFFSETS = [7, 14, 21, 28, 30, 35, 42, 45, 49, 56, 60]
NONE_BIAS = lambda o, a: 0.0


def picked_rows(test_c: pd.DataFrame, pm: dict) -> pd.DataFrame:
    rows = [test_c[(test_c.sample_id == s) & (test_c.offset == o)].iloc[0] for s, o in pm.items()]
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4, 5, 10, 42, 100])
    ap.add_argument("--variants", nargs="+", default=["baseline", "direct_neighbor"])
    ap.add_argument("--dense-grid", default=str(DENSE_GRID))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    files = ["final_summary.csv", "per_sample_results.csv", "offset_distribution.csv",
             "paired_per_sample.csv", "paired_summary.csv"]
    for f in files:
        if (out_dir / f).exists() and not args.force:
            raise SystemExit(f"Refuse to overwrite {out_dir / f} (use --force)")
    if not Path(args.dense_grid).exists():
        raise SystemExit(f"Dense grid not found: {args.dense_grid}. "
                         "Run phase_t_wbph_dense_offset_grid.py first.")

    grid = pd.read_csv(args.dense_grid)
    _g, clim_mid, disp = C.load_inputs()
    print(f"[validate] dense grid recompute vs iou80 max|Δ| (σ=5): {C.validate_reconstruction(grid, 5.0)}")
    print(f"[setting] sigma={SIGMA} FIXED, dense offsets={DENSE_OFFSETS}, no mu-calibration\n")

    offsets = sorted(set(int(o) for o in DENSE_OFFSETS) &
                     set(int(o) for o in grid.offset.unique()))

    per_sample_rows, dist_rows, summary_rows = [], [], []
    # realized per-sample iou keyed for paired comparison: iou_map[variant][seed][sample] = iou80
    iou_map = {v: {} for v in args.variants}

    for variant in args.variants:
        gv = grid[grid.variant == variant]
        val_c = C.build_cand(gv[gv.split == "val"], disp, clim_mid, offsets, SIGMA, NONE_BIAS)
        test_c = C.build_cand(gv[gv.split == "test"], disp, clim_mid, offsets, SIGMA, NONE_BIAS)
        n_total = test_c.sample_id.nunique()

        seed_metrics = []
        for sd in args.seeds:
            reg = C.train_selector(val_c, "target_cov", sd)
            pm = C.pick_argmax(test_c, reg)
            seed_metrics.append(C.realized(test_c, pm, n_total))
            pr = picked_rows(test_c, pm)
            iou_map[variant][sd] = dict(zip(pr["sample_id"], pr["iou80_real"]))
            for r in pr.itertuples():
                per_sample_rows.append({
                    "variant": variant, "seed": sd, "sample_id": r.sample_id,
                    "picked_offset": int(r.offset), "feasible": int(r.feasible),
                    "iou80": round(float(r.iou80_real), 4),
                    "mae_center": (round(float(r.mae_center_real), 3) if r.feasible else np.nan),
                    "pi_hit80": int(r.pi_hit80_real), "width80": (float(r.width80_real) if r.feasible else np.nan),
                    "alert_tstar": float(r.alert_tstar), "L": float(r.L), "R": float(r.R),
                })
            for o in offsets:
                dist_rows.append({"variant": variant, "seed": sd, "offset": o,
                                  "count": int((pr["offset"] == o).sum())})

        sm = pd.DataFrame(seed_metrics)
        orac = C.realized(test_c, C.pick_oracle(test_c), n_total)
        vfix = C.val_fixed_offset(val_c)
        dq = C.realized(test_c, C.pick_fixed(test_c, vfix, C.deploy_q20_cap_map(test_c)), n_total)
        summary_rows.append({
            "variant": variant, "sigma": SIGMA, "n_seeds": len(args.seeds), "n_total": n_total,
            "IoU80_overall_mean": round(float(sm["IoU80_overall"].mean()), 4),
            "IoU80_overall_std": round(float(sm["IoU80_overall"].std()), 4),
            "IoU80_overall_min": round(float(sm["IoU80_overall"].min()), 4),
            "IoU80_overall_max": round(float(sm["IoU80_overall"].max()), 4),
            "coverage_mean": round(float(sm["coverage"].mean()), 4),
            "late_count_mean": round(float(sm["late_count"].mean()), 2),
            "mean_offset_mean": round(float(sm["mean_offset"].mean()), 2),
            "MAE_center_mean": round(float(sm["MAE_center"].mean()), 4),
            "PI_hit_mean": round(float(sm["PI_hit"].mean()), 4),
            "width80_mean": round(float(sm["width80"].mean()), 2),
            "oracle_IoU80_overall": orac["IoU80_overall"],
            "deploy_q20_IoU80_overall": dq["IoU80_overall"],
        })

    # ---- paired DN - baseline (same seed, per sample) ----
    paired_rows, paired_summary = [], []
    if set(["baseline", "direct_neighbor"]).issubset(set(args.variants)):
        for sd in args.seeds:
            bm = iou_map["baseline"].get(sd, {})
            dm = iou_map["direct_neighbor"].get(sd, {})
            sids = sorted(set(bm) & set(dm))
            diffs = []
            for s in sids:
                d = float(dm[s]) - float(bm[s])
                diffs.append(d)
                paired_rows.append({"seed": sd, "sample_id": s,
                                    "baseline_iou80": round(bm[s], 4),
                                    "dn_iou80": round(dm[s], 4), "diff_dn_minus_base": round(d, 4)})
            arr = np.array(diffs)
            paired_summary.append({
                "seed": sd, "n": len(arr),
                "mean_diff": round(float(arr.mean()), 4),
                "median_diff": round(float(np.median(arr)), 4),
                "dn_win": int((arr > 1e-9).sum()), "base_win": int((arr < -1e-9).sum()),
                "tie": int((np.abs(arr) <= 1e-9).sum()),
            })
        ps = pd.DataFrame(paired_summary)
        overall = {"seed": "ALL", "n": int(ps["n"].sum()),
                   "mean_diff": round(float(pd.DataFrame(paired_rows)["diff_dn_minus_base"].mean()), 4),
                   "median_diff": round(float(pd.DataFrame(paired_rows)["diff_dn_minus_base"].median()), 4),
                   "dn_win": int(ps["dn_win"].sum()), "base_win": int(ps["base_win"].sum()),
                   "tie": int(ps["tie"].sum())}
        paired_summary_df = pd.concat([ps, pd.DataFrame([overall])], ignore_index=True)
    else:
        paired_summary_df = pd.DataFrame()

    summ = pd.DataFrame(summary_rows)
    summ.to_csv(out_dir / "final_summary.csv", index=False)
    pd.DataFrame(per_sample_rows).to_csv(out_dir / "per_sample_results.csv", index=False)
    pd.DataFrame(dist_rows).to_csv(out_dir / "offset_distribution.csv", index=False)
    pd.DataFrame(paired_rows).to_csv(out_dir / "paired_per_sample.csv", index=False)
    paired_summary_df.to_csv(out_dir / "paired_summary.csv", index=False)
    print(f"[done] wrote {len(files)} files to {out_dir}\n")

    print("=== FINAL (dense + cov-aware selector + sigma=8) WBPH 2024 ===")
    print(summ.to_string(index=False))
    if not paired_summary_df.empty:
        print("\n=== paired DN - baseline (same seed) ===")
        print(paired_summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
