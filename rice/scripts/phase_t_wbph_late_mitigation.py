"""WBPH 2022/2023/2024 — late-eval mitigation comparison (ckpt-norm, sigma=8, cov-aware selector).

Compares three configs, all SUBSET from the existing 1..75 multiyear grid (NO re-forward):
  dense            : offsets 7,14,21,28,30,35,42,45,49,56,60          (current final setting)
  dense+off3       : dense + offset 3                                 (reduce STRUCTURAL late)
  dense+off3+guard : dense+off3 + val-tuned deploy-lead cap guard on the selector's
                     candidate offsets (reduce SELECTOR-INDUCED late). The guard restricts
                     candidates to offset <= cap_q where cap_q = val lead_to_start q-quantile
                     (per alert-bin). q in {none,0.30,0.20,0.10} is chosen to MAXIMIZE VAL
                     cov-aware IoU80 (no test labels). Mild: val may pick 'none'.

Metrics per (year, variant, config), seed-mean over --seeds:
  IoU80_overall (tol0 = STRICT, main) and IoU80_overall (tol1 = 1-day operational tolerance),
  late total / structural / selector-induced, coverage, MAE_center, PI_hit, selected-offset dist.

tol0 is the main metric; tol1 (a late pick within 1 day of onset keeps its geometric IoU) is
reported only as operational sensitivity. ckpt-norm only; no mu calibration; no wrong-norm.
Does NOT overwrite existing results (new out-dir; --force to overwrite its own outputs).

Example
-------
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_late_mitigation --force
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

from rice.configs.base import RICE_ROOT
import rice.scripts.phase_t_wbph_iou_common as C
from rice.scripts.phase_t_wbph_offset_ablation_multiyear import load_year_clim_disp
from rice.scripts.phase_t_stage2_offset_constraint import build_val_caps, lookup_deploy_cap

SIGMA = 8.0
BIN_W = 20
PEST = "WBPH"
GRID = RICE_ROOT / "outputs/diag/stage2_ckptnorm_selector_wbph/multiyear_full/wbph_grid_1to75_multiyear.csv"
OUT_DIR = RICE_ROOT / "outputs/diag/stage2_ckptnorm_selector_wbph/late_mitigation"
DENSE = [7, 14, 21, 28, 30, 35, 42, 45, 49, 56, 60]
CONFIGS = {
    "dense": {"offsets": DENSE, "guard": False},
    "dense+off3": {"offsets": sorted(set(DENSE) | {3}), "guard": False},
    "dense+off3+guard": {"offsets": sorted(set(DENSE) | {3}), "guard": True},
}
GUARD_QS = [None, 0.30, 0.20, 0.10]


def cap_map_for_q(test_c: pd.DataFrame, q):
    if q is None:
        return None
    caps, _v, _c = build_val_caps([q], min_bin=20, min_pest=30, bin_w=BIN_W)
    cm = {}
    for sid in test_c.sample_id.unique():
        alert = int(test_c[test_c.sample_id == sid]["alert_tstar"].iloc[0])
        cm[sid], _ = lookup_deploy_cap(caps, q, PEST, alert, BIN_W)
    return cm


def eval_picks(test_c: pd.DataFrame, pm: dict, min_cfg_off: int) -> dict:
    """Metrics for one pick map. tol-k IoU counts geometric IoU when lateness d<=k else 0."""
    n = len(pm)
    tol0 = tol1 = 0.0
    late = struct = sel = 0
    feas_mae, feas_pi, feas_n = [], 0, 0
    dist = {}
    for sid, off in pm.items():
        r = test_c[(test_c.sample_id == sid) & (test_c.offset == off)].iloc[0]
        alert = float(r.alert_tstar); L = float(r.L); R = float(r.R)
        eval_doy = alert + off; true_start = L + 1; d = eval_doy - true_start
        geom = C.recompute(float(r.pred_mu), SIGMA, eval_doy, L, R)["iou80"]
        tol0 += geom if d <= 0 else 0.0
        tol1 += geom if d <= 1 else 0.0
        if d > 0:
            late += 1
            if (alert + min_cfg_off) > true_start:
                struct += 1
            else:
                sel += 1
        else:
            feas_n += 1
            feas_mae.append(float(r.mae_center_real))
            feas_pi += int(r.pi_hit80_real)
        dist[int(off)] = dist.get(int(off), 0) + 1
    return {
        "IoU80_tol0": tol0 / n, "IoU80_tol1": tol1 / n,
        "late": late, "structural": struct, "selector_induced": sel,
        "coverage": feas_n / n, "MAE_center": (np.mean(feas_mae) if feas_mae else np.nan),
        "PI_hit": feas_pi / n, "dist": dist,
    }


def tune_guard_q(val_c, seed):
    """Pick q maximizing VAL cov-aware IoU80 (tol0). No test labels."""
    n_val = val_c.sample_id.nunique()
    best_q, best = None, -1.0
    reg = C.train_selector(val_c, "target_cov", seed)
    for q in GUARD_QS:
        cm = cap_map_for_q(val_c, q)
        pm = C.pick_argmax(val_c, reg, cap_map=cm)
        score = eval_picks(val_c, pm, min_cfg_off=min(int(o) for o in val_c.offset.unique()))["IoU80_tol0"]
        if score > best:
            best, best_q = score, q
    return best_q, reg


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
    for f in ("late_mitigation_summary.csv", "offset_distribution.csv"):
        if (out_dir / f).exists() and not args.force:
            raise SystemExit(f"Refuse to overwrite {out_dir / f} (use --force)")
    if not Path(args.grid).exists():
        raise SystemExit(f"Grid not found: {args.grid}. Run phase_t_wbph_multiyear_grid (multiyear_full) first.")

    grid = pd.read_csv(args.grid)
    rows, dist_rows = [], []
    for year in args.years:
        clim, disp = load_year_clim_disp(year)
        gy = grid[grid.year == year]
        for variant in args.variants:
            gv = gy[gy.variant == variant]
            if gv.empty:
                continue
            avail = set(int(o) for o in gv.offset.unique())
            for cname, cfg in CONFIGS.items():
                offs = sorted(set(cfg["offsets"]) & avail)
                min_off = min(offs)
                val_c = C.build_cand(gv[gv.split == "val"], disp, clim, offs, SIGMA)
                test_c = C.build_cand(gv[gv.split == "test"], disp, clim, offs, SIGMA)
                n_total = test_c.sample_id.nunique()
                per_seed, chosen_qs = [], []
                agg_dist = {}
                for sd in args.seeds:
                    if cfg["guard"]:
                        q, reg = tune_guard_q(val_c, sd)      # val-tuned q (no test labels)
                        cm = cap_map_for_q(test_c, q)
                        chosen_qs.append(q)
                    else:
                        reg = C.train_selector(val_c, "target_cov", sd); cm = None
                    pm = C.pick_argmax(test_c, reg, cap_map=cm)
                    m = eval_picks(test_c, pm, min_off)
                    per_seed.append(m)
                    for o, c in m["dist"].items():
                        agg_dist[o] = agg_dist.get(o, 0) + c
                pm_df = pd.DataFrame([{k: v for k, v in m.items() if k != "dist"} for m in per_seed])
                rows.append({
                    "year": year, "variant": variant, "config": cname,
                    "n_offsets": len(offs), "n_total": n_total, "n_seeds": len(args.seeds),
                    "IoU80_tol0": round(pm_df["IoU80_tol0"].mean(), 4),
                    "IoU80_tol1": round(pm_df["IoU80_tol1"].mean(), 4),
                    "late": round(pm_df["late"].mean(), 2),
                    "structural": round(pm_df["structural"].mean(), 2),
                    "selector_induced": round(pm_df["selector_induced"].mean(), 2),
                    "coverage": round(pm_df["coverage"].mean(), 4),
                    "MAE_center": round(pm_df["MAE_center"].mean(), 3),
                    "PI_hit": round(pm_df["PI_hit"].mean(), 4),
                    "guard_q": (str(sorted(set(str(q) for q in chosen_qs))) if cfg["guard"] else ""),
                })
                for o, c in sorted(agg_dist.items()):
                    dist_rows.append({"year": year, "variant": variant, "config": cname,
                                      "offset": o, "mean_count": round(c / len(args.seeds), 2)})

    summ = pd.DataFrame(rows)
    dist = pd.DataFrame(dist_rows)
    summ.to_csv(out_dir / "late_mitigation_summary.csv", index=False)
    dist.to_csv(out_dir / "offset_distribution.csv", index=False)
    print(f"[done] wrote late_mitigation_summary.csv, offset_distribution.csv to {out_dir}\n")

    print("=== late-eval mitigation (WBPH, sigma=8, cov-aware; seed-mean) ===")
    print("  IoU80_tol0 = STRICT main metric; IoU80_tol1 = 1-day operational tolerance")
    cols = ["year", "variant", "config", "IoU80_tol0", "IoU80_tol1", "late", "structural",
            "selector_induced", "coverage", "MAE_center", "PI_hit", "guard_q"]
    print(summ[cols].to_string(index=False))
    print("\n=== selected-offset distribution (mean count/seed) ===")
    print(dist.pivot_table(index=["year", "variant", "config"], columns="offset",
                           values="mean_count", fill_value=0).to_string())


if __name__ == "__main__":
    main()
