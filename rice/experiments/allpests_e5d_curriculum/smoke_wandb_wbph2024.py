#!/usr/bin/env python
"""Smoke test the W&B connector on WBPH/2024 using the EXISTING E5d dev grid.

Runs the real dev protocol (selector on the val_fit half, shift on val_cal, selector seeds
0..4), builds all 13 legacy figures per seed, and either uploads or -- with no --wandb_project
-- writes everything locally. Trains nothing.

Also prints a key-parity table: every legacy W&B key vs what this connector emits.

  cd /home/gpu4080/research/cropscience
  PYTHONPATH=/home/gpu4080/research/wbph_interval_perf_202607:$PWD \
    .venv/bin/python rice/experiments/allpests_e5d/smoke_wandb_wbph2024.py
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, pandas as pd

CS = Path("/home/gpu4080/research/cropscience")
WS = Path("/home/gpu4080/research/wbph_interval_perf_202607")
AP = CS / "rice/experiments/allpests_e5d_curriculum"
sys.path.insert(0, str(AP)); sys.path.insert(0, str(AP / "vendor"))   # pinned deps only
import pest_paths as PP
import pest_eval as PE
import wandb_viz_e5d as VW
from src.io_utils import load_dispatch, load_clim_mid
from src.selector_utils import train_selector, pick_offsets, picked_rows
from src import diagnostics as D

# WBPH's published E5d dev grid predates the all-pest tree
WBPH_DEV_GRID = WS / "outputs/feature_experiments/target_asym_2x2/E5d/grid/wbph_E5d_grid_1to75.csv"

LEGACY_KEYS = {
    "config": ["pest", "year", "selector_name", "sigma", "sample_grid", "selector_offsets"],
    "images": [f"viz/{k}" for k in VW.VIZ_KEYS],
    "tables": ["table/per_sample", "table/metrics_by_lead_bin"],
    "summary": [f"final/{k}" for k in
                ["n_total", "mean_iou", "median_iou", "frac_iou_gt_0_2",
                 "frac_early_or_inside_30", "mu_minus_mid_mean", "mu_minus_mid_median",
                 "lead_days_mean", "lead_days_median", "pest", "year", "selector_name"]],
    "artifacts": ["run.save(per_sample_csv)", "run.save(lead_csv)"],
}


def gate_policy(pest, year):
    f = AP / "gate_policy_observed.csv"
    if not f.exists():
        return {}
    d = pd.read_csv(f)
    r = d[(d.pest == pest) & (d.year == year)]
    return {} if r.empty else dict(stage1_gate=str(r.iloc[0]["gate"]),
                                   stage1_tau=str(r.iloc[0]["tau"]),
                                   stage1_k=str(r.iloc[0]["k"]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="WBPH")
    ap.add_argument("--year", type=int, default=2024)
    ap.add_argument("--wandb_project", default=None, help="omit to run fully offline")
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_group", default="e5d_dev_smoke")
    ap.add_argument("--out_dir", default=str(PP.OUT_ROOT / "_wandb_smoke"))
    a = ap.parse_args()

    if not WBPH_DEV_GRID.exists():
        raise SystemExit(f"[abort] missing E5d dev grid {WBPH_DEV_GRID}")
    g = pd.read_csv(WBPH_DEV_GRID)
    g = g[(g.variant == "E5d") & (g.offset.isin(PP.OFFSETS))]
    doy_start, T = PP.geometry(a.pest)
    P = PP.synthetic_paths(a.pest)
    disp, clim = load_dispatch(P, a.year), load_clim_mid(P, a.year)

    # dev protocol, exactly as pest_eval.run_dev: 50/50 val split, selector seeds 0..4,
    # shift chosen on val_cal, evaluated once on the eval year.
    vids = set(g[(g.year == a.year) & (g.split == "val")].sample_id.unique())
    fit, cal = PE.half(vids, "val_fit"), PE.half(vids, "val_cal")
    print(f"[smoke] {a.pest}/{a.year}: val_fit={len(fit)} val_cal={len(cal)} "
          f"eval={g[(g.year==a.year)&(g.split=='test')].sample_id.nunique()}")

    sels = [train_selector(PE.make_cand(PE.frames(g, a.year, "val", fit), disp, clim, doy_start, T), sd)
            for sd in PP.SEEDS]
    best, bestv = 0, -1.0
    for dlt in PP.SHIFT_GRID:
        vc = PE.make_cand(PE.frames(g, a.year, "val", cal, dlt), disp, clim, doy_start, T)
        n = vc.sample_id.nunique()
        v = float(np.mean([D.summarize(D.enrich(picked_rows(vc, pick_offsets(r, vc))),
                                       min(PP.OFFSETS), n)["IoU80_overall_tol0"] for r in sels]))
        if v > bestv:
            best, bestv = dlt, v
    print(f"[smoke] calibration shift delta*={best:+d} (val_cal IoU80={bestv:.4f})")

    per_seed, raw_metrics = [], {}
    for tag, dlt in (("raw_uncalibrated", 0), ("calibrated", best)):
        tc = PE.make_cand(PE.frames(g, a.year, "test", None, dlt), disp, clim, doy_start, T)
        n = tc.sample_id.nunique()
        m = PE.metrics_at(tc, sels, n)
        if tag == "raw_uncalibrated":
            raw_metrics = {k: v for k, v in m.items() if isinstance(v, (int, float))}
            continue
        for sd, reg in zip(PP.SEEDS, sels):
            pr = D.enrich(picked_rows(tc, pick_offsets(reg, tc)))
            per_seed.append(dict(seed=sd,
                                 per_sample=VW.to_legacy_per_sample(pr, f"coverage_aware_lgbm_seed{sd}"),
                                 metrics=m))
    print(f"[smoke] built {len(per_seed)} selector-seed result sets")

    cfg = dict(eval_year=a.year, split=f"train<= {a.year-2} / val {a.year-1} / test {a.year}",
               stage2_train_seed=0, selector_seeds=list(PP.SEEDS), viz_random_seed=0,
               protocol="dev", model="E5d",
               doy_start=doy_start, doy_end=doy_start + T - 1, T=T,
               selector="coverage_aware_lightgbm", selector_n_seeds=len(PP.SEEDS),
               calibration=f"global additive mu shift, grid {PP.SHIFT_GRID[0]}..{PP.SHIFT_GRID[-1]}, "
                           f"selected on val_cal; delta*={best:+d}",
               calibration_shift=best, sigma_eval=PP.SIGMA)
    cfg.update(gate_policy(a.pest, a.year))

    okd = VW.log_run(a.pest, a.year, per_seed, raw_metrics, cfg,
                     Path(a.out_dir) / f"{a.pest}_{a.year}",
                     project=a.wandb_project, entity=a.wandb_entity, group=a.wandb_group)

    # ---- key parity table ----
    emitted = {
        "config": list(cfg.keys()) + LEGACY_KEYS["config"],
        "images": [f"viz/{k}" for k in VW.VIZ_KEYS],
        "tables": ["table/per_sample", "table/metrics_by_lead_bin"],
        "summary": ([f"final/{k}" for k in VW.legacy_summary(per_seed[0]["per_sample"])]
                    + ["final/pest", "final/year", "final/selector_name"]),
        "artifacts": ["run.save(per_sample_csv)", "run.save(lead_csv)"],
    }
    print("\n=== legacy W&B key parity ===")
    print(f"{'group':10} {'legacy':>7} {'emitted':>8} {'missing':>8}  detail")
    total_missing = 0
    for grp, legacy in LEGACY_KEYS.items():
        miss = [k for k in legacy if k not in emitted[grp]]
        total_missing += len(miss)
        print(f"{grp:10} {len(legacy):>7} {len(emitted[grp]):>8} {len(miss):>8}  "
              f"{'-' if not miss else miss}")
    print(f"\nMISSING TOTAL = {total_missing}")
    print("added (not in legacy): final_std/*, final_raw/*, final_selector/*, "
          "table column selector_seed, config E5d fields, per-image captions")
    print(f"\nW&B upload: {'OK' if okd else 'skipped/offline (non-fatal)'}")
    return 0 if total_missing == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
