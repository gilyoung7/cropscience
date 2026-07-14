"""WBPH 2024 (ckpt-norm) — coverage-aware offset selector ROBUSTNESS via seed sweep.

Evaluates how stable the coverage-aware selector is across LightGBM random seeds,
for baseline AND direct_neighbor, against fixed references (deploy_q20,
selector_blind, oracle_feasible). WBPH only, test=2024, ckpt-norm grid only —
NO wrong-norm / phase_r artifacts are read.

Methods compared
----------------
  selector_blind        per-sample argmax of a selector trained on mu-band IoU over
                        interpolated candidates (feasibility-blind; reproduces collapse). [per seed]
  selector_cov_aware    same selector but target = REALIZED IoU (0 when offset is
                        infeasible/late). Coverage-aware objective.                        [per seed]
  deploy_q20            val-fixed offset capped by val lead q=0.20 (deployable).            [seed-independent]
  oracle_feasible       per-sample best FEASIBLE offset (upper bound).                      [seed-independent]

Metrics (realized; coverage-weighted over all n_total alerted events, late pick -> 0)
  realized IoU80 overall, coverage, late_count, mean_offset,
  selected-offset distribution, MAE_center (over feasible picks), PI_hit (realized / n_total)

Inputs (ckpt-norm, validated bit-exact vs compare_eval/WBPH/matched_eval.json):
  rice/outputs/diag/stage2_direct_neighbor_wbph_2024/wbph_offset_grid.csv
  rice/outputs/stage2/batch_2024_bestgate/WBPH/gate_dispatch_group_tau_R088_features_per_sy.csv
  rice/outputs/stage2/batch_2024_bestgate/WBPH/climatology_train_stats.csv

Outputs (new dir, never overwrites without --force)
  rice/outputs/diag/stage2_ckptnorm_selector_wbph/selector_robustness/
    per_seed_results.csv          one row per (variant, method, seed)
    offset_distribution.csv       long: (variant, method, seed, offset, count)
    summary.csv                   selector methods aggregated across seeds (mean/std/min/max) + refs

Example
-------
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_selector_robustness --force
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_selector_robustness \
      --seeds 0 1 2 3 4 5 10 42 100 --variants baseline direct_neighbor --force
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rice.configs.base import RICE_ROOT
from rice.scripts.phase_b_stage2_offset_selector_v2_ranking import (
    iou_from_mu, interp_mu_curve, train_lgb_regressor, _prep_X,
    COARSE_OFFSETS, SAMPLE_FEATURES)
from rice.scripts.phase_t_stage2_offset_constraint import (
    build_val_caps, lookup_deploy_cap, choose_offset)

PEST = "WBPH"
GRID = RICE_ROOT / "outputs/diag/stage2_direct_neighbor_wbph_2024/wbph_offset_grid.csv"
DISPATCH_CSV = RICE_ROOT / "outputs/stage2/batch_2024_bestgate/WBPH/gate_dispatch_group_tau_R088_features_per_sy.csv"
CLIM_CSV = RICE_ROOT / "outputs/stage2/batch_2024_bestgate/WBPH/climatology_train_stats.csv"
OUT_DIR = RICE_ROOT / "outputs/diag/stage2_ckptnorm_selector_wbph/selector_robustness"
DEFAULT_SEEDS = [0, 1, 2, 3, 4, 5, 10, 42, 100]
DEFAULT_VARIANTS = ["baseline", "direct_neighbor"]
BIN_W = 20


def build_cand(gvs: pd.DataFrame, disp: pd.DataFrame, clim_mid: float) -> pd.DataFrame:
    """Coarse candidate grid for one (variant, split). Real grid rows -> feasible
    (carry realized iou80 / MAE_center / PI_hit80); missing offsets -> infeasible
    (interpolated mu as feature; realized reward 0)."""
    dmap = disp.set_index("sample_id")
    rows = []
    for sid, sub in gvs.groupby("sample_id"):
        feas = {int(r.offset): r for r in sub.itertuples()}
        coarse_mu = {o: float(feas[o].mu) for o in feas}
        L = float(sub["L"].iloc[0]); R = float(sub["R"].iloc[0])
        alert = float(sub["alert_tstar"].iloc[0])
        mu_curve = interp_mu_curve(coarse_mu, COARSE_OFFSETS)
        if sid not in dmap.index:
            continue
        df = dmap.loc[sid]
        for o in COARSE_OFFSETS:
            feasible = o in feas
            mu = coarse_mu[o] if feasible else float(mu_curve[o])
            band = iou_from_mu(mu, L, R)
            rec = {
                "sample_id": sid, "offset": o, "L": L, "R": R, "alert_tstar": alert,
                "pred_mu": mu, "pred_lead": mu - alert,
                "mu_minus_clim_mid": mu - clim_mid, "tstar_minus_clim_mid": alert - clim_mid,
                "dispatch_branch_is_D": 1 if str(df.get("dispatch_branch")) == "D" else 0,
                "feasible": int(feasible),
                "iou80_real": float(feas[o].iou80) if feasible else 0.0,
                "mae_center_real": float(feas[o].MAE_center) if feasible else np.nan,
                "pi_hit80_real": int(bool(feas[o].PI_hit80)) if feasible else 0,
                "target_blind": band,
                "target_cov": (band if feasible else 0.0),
            }
            for f in SAMPLE_FEATURES:
                if f != "alert_tstar":
                    rec[f] = df.get(f)
            rows.append(rec)
    return pd.DataFrame(rows)


def pick(cand: pd.DataFrame, score: np.ndarray | None, cap_map: dict | None = None,
         fixed_offset: int | None = None, oracle_col: str | None = None) -> dict:
    """Per-sample offset choice.
      - score given   : argmax score (optionally restricted to offset <= cap)
      - fixed_offset   : same offset for all (optionally capped)
      - oracle_col     : argmax of that realized column per sample
    """
    out = {}
    if score is not None:
        c = cand.copy(); c["__s"] = score
        for sid, g in c.groupby("sample_id"):
            gg = g if cap_map is None else g[g.offset <= cap_map.get(sid, 1e9)]
            if gg.empty:
                gg = g[g.offset == g.offset.min()]
            out[sid] = int(gg.loc[gg["__s"].idxmax(), "offset"])
    elif oracle_col is not None:
        for sid, g in cand.groupby("sample_id"):
            out[sid] = int(g.loc[g[oracle_col].idxmax(), "offset"])
    else:  # fixed offset, optionally capped
        for sid, g in cand.groupby("sample_id"):
            o = fixed_offset if cap_map is None else int(choose_offset(fixed_offset, cap_map.get(sid, 1e9)))
            out[sid] = int(o)
    return out


def realized(cand: pd.DataFrame, pm: dict, n_total: int) -> tuple[dict, dict]:
    rows = [cand[(cand.sample_id == s) & (cand.offset == o)].iloc[0] for s, o in pm.items()]
    d = pd.DataFrame(rows)
    feas = d[d.feasible == 1]
    metrics = {
        "n_total": n_total,
        "IoU80_overall": round(float(d["iou80_real"].sum()) / n_total, 4),
        "coverage": round(float(d["feasible"].mean()), 4),
        "late_count": int((d["feasible"] == 0).sum()),
        "mean_offset": round(float(d["offset"].mean()), 2),
        # MAE over feasible picks (late picks have no early prediction to score)
        "MAE_center_feasible": round(float(feas["mae_center_real"].mean()), 4) if len(feas) else float("nan"),
        # realized PI_hit over ALL alerted events (late pick -> not hit)
        "PI_hit_overall": round(float(feas["pi_hit80_real"].sum()) / n_total, 4),
    }
    dist = {int(o): int((d["offset"] == o).sum()) for o in COARSE_OFFSETS}
    return metrics, dist


def main():
    ap = argparse.ArgumentParser(description="WBPH coverage-aware selector robustness (seed sweep)")
    ap.add_argument("--seeds", type=int, nargs="+", default=DEFAULT_SEEDS)
    ap.add_argument("--variants", nargs="+", default=DEFAULT_VARIANTS)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir / f for f in ("per_seed_results.csv", "offset_distribution.csv", "summary.csv")]
    existing = [str(p) for p in targets if p.exists()]
    if existing and not args.force:
        raise SystemExit(f"Refuse to overwrite (use --force): {existing}")

    grid = pd.read_csv(GRID)
    clim_mid = float(pd.read_csv(CLIM_CSV).iloc[0]["mean_mid"])
    disp_raw = pd.read_csv(DISPATCH_CSV)
    disp_raw["sample_id"] = disp_raw["site"].astype(str) + "-" + disp_raw["year"].astype(int).astype(str)
    disp = disp_raw[["sample_id"] + SAMPLE_FEATURES + ["dispatch_branch"]].drop_duplicates("sample_id")
    caps, _vp, _cd = build_val_caps([0.20], min_bin=20, min_pest=30, bin_w=BIN_W)

    per_seed_rows = []
    dist_rows = []

    def record(variant, method, seed, metrics, dist):
        per_seed_rows.append({"variant": variant, "method": method, "seed": seed, **metrics})
        for o, c in dist.items():
            dist_rows.append({"variant": variant, "method": method, "seed": seed, "offset": o, "count": c})

    for variant in args.variants:
        gv = grid[grid.variant == variant]
        val_c = build_cand(gv[gv.split == "val"], disp, clim_mid)
        test_c = build_cand(gv[gv.split == "test"], disp, clim_mid)
        n_total = test_c.sample_id.nunique()
        X_test = _prep_X(test_c).values

        # ---- seed-independent references (seed = -1 sentinel) ----
        # deploy_q20: val-fixed (overall realized iou80 on val) base, capped by val q20 lead
        vfix = int((val_c.groupby("offset")["iou80_real"].sum() / val_c.sample_id.nunique()).idxmax())
        cap_map = {}
        for sid in test_c.sample_id.unique():
            alert = int(test_c[test_c.sample_id == sid]["alert_tstar"].iloc[0])
            cap_map[sid], _ = lookup_deploy_cap(caps, 0.20, PEST, alert, BIN_W)
        record(variant, "deploy_q20", -1,
               *realized(test_c, pick(test_c, None, cap_map=cap_map, fixed_offset=vfix), n_total))
        record(variant, "oracle_feasible", -1,
               *realized(test_c, pick(test_c, None, oracle_col="iou80_real"), n_total))

        # ---- per-seed selector methods ----
        for seed in args.seeds:
            vb = val_c.copy(); vb["iou"] = vb["target_blind"]
            vc = val_c.copy(); vc["iou"] = vc["target_cov"]
            reg_blind = train_lgb_regressor(vb, seed=seed)
            reg_cov = train_lgb_regressor(vc, seed=seed)
            s_blind = reg_blind.predict(X_test)
            s_cov = reg_cov.predict(X_test)
            record(variant, "selector_blind", seed,
                   *realized(test_c, pick(test_c, s_blind), n_total))
            record(variant, "selector_cov_aware", seed,
                   *realized(test_c, pick(test_c, s_cov), n_total))

    per_seed = pd.DataFrame(per_seed_rows)
    dist_df = pd.DataFrame(dist_rows)
    per_seed.to_csv(out_dir / "per_seed_results.csv", index=False)
    dist_df.to_csv(out_dir / "offset_distribution.csv", index=False)

    # ---- summary: aggregate selector methods across seeds; refs passthrough ----
    agg_cols = ["IoU80_overall", "coverage", "late_count", "mean_offset",
                "MAE_center_feasible", "PI_hit_overall"]
    sel = per_seed[per_seed.seed >= 0]
    g = sel.groupby(["variant", "method"])[agg_cols]
    summ = g.agg(["mean", "std", "min", "max"]).round(4)
    summ.columns = ["_".join(c) for c in summ.columns]
    summ = summ.reset_index()
    summ["n_seeds"] = sel.groupby(["variant", "method"]).size().values
    refs = per_seed[per_seed.seed < 0][["variant", "method"] + agg_cols].copy()
    refs.columns = ["variant", "method"] + [f"{c}_mean" for c in agg_cols]
    refs["n_seeds"] = 0
    summary = pd.concat([summ, refs], ignore_index=True, sort=False)
    summary.to_csv(out_dir / "summary.csv", index=False)

    print(f"[done] wrote per_seed_results.csv, offset_distribution.csv, summary.csv to {out_dir}")


if __name__ == "__main__":
    main()
