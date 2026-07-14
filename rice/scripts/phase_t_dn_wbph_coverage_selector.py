"""WBPH 2024: coverage-aware offset selector (ckpt-norm). Keeps the selector's
original purpose (pick per-sample best offset) but fixes the off60-collapse seen
under ckpt-norm by making the training objective feasibility/coverage-aware.

ROOT CAUSE of collapse: the v2 candidate grid INTERPOLATES mu to fabricate a
candidate at every offset (incl. offsets whose eval time lands AFTER event onset
= "late" = no real early prediction). The selector maximizes per-candidate IoU
and so prefers off60 (highest interpolated IoU) even when off60 is infeasible for
most samples -> coverage collapses to ~33%.

FIX (objective): same candidate grid + features; only the TARGET changes.
  - target_blind : band IoU from mu, computed for ALL offsets (feasible-blind = current behavior)
  - target_cov   : REALIZED band IoU = 0 when the offset is infeasible (late). Teaches the
                   selector that a late offset yields no early prediction (coverage-aware).
Feasible = a real Stage-2 nowcast row exists at eval_tstar=alert+offset (require_tstar_before_L
=> eval before onset). Infeasible offsets keep interpolated mu as a FEATURE but realized reward 0.

Optional hard guard: deploy_q20 cap (val lead-quantile) applied at pick time.

Final metric = REALIZED coverage-weighted IoU80 over all alerted events
(sum of real iou80 at picked offset, 0 if picked offset infeasible, / n_total). Compared to
deploy_q20, val_fixed, and the per-sample feasible oracle. For baseline AND direct_neighbor.

Run:
    PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_dn_wbph_coverage_selector --force
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
OUT_DIR = RICE_ROOT / "outputs/diag/stage2_ckptnorm_selector_wbph"
VARIANTS = ["baseline", "direct_neighbor"]
BIN_W = 20


def build_cand(gvs: pd.DataFrame, disp: pd.DataFrame, clim_mid: float) -> pd.DataFrame:
    """Coarse candidate grid for one (variant, split). Real rows -> feasible;
    missing offsets -> infeasible (interpolated mu feature, realized reward 0)."""
    rows = []
    dmap = disp.set_index("sample_id")
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
            iou80_real = float(feas[o].iou80) if feasible else 0.0
            band = iou_from_mu(mu, L, R)
            rec = {
                "sample_id": sid, "offset": o, "L": L, "R": R, "alert_tstar": alert,
                "pred_mu": mu, "pred_lead": mu - alert,
                "mu_minus_clim_mid": mu - clim_mid, "tstar_minus_clim_mid": alert - clim_mid,
                "dispatch_branch_is_D": 1 if str(df.get("dispatch_branch")) == "D" else 0,
                "feasible": int(feasible), "iou80_real": iou80_real,
                "target_blind": band, "target_cov": (band if feasible else 0.0),
            }
            for f in SAMPLE_FEATURES:
                if f != "alert_tstar":
                    rec[f] = df.get(f)
            rows.append(rec)
    return pd.DataFrame(rows)


def scores_of(reg, cand: pd.DataFrame) -> np.ndarray:
    return reg.predict(_prep_X(cand).values)


def pick(cand: pd.DataFrame, score: np.ndarray, cap_map: dict | None = None) -> dict:
    """Per-sample argmax score; if cap_map given, restrict offsets <= cap."""
    c = cand.copy(); c["__s"] = score
    out = {}
    for sid, g in c.groupby("sample_id"):
        gg = g
        if cap_map is not None:
            gg = g[g.offset <= cap_map.get(sid, 1e9)]
            if gg.empty:
                gg = g[g.offset == g.offset.min()]
        out[sid] = int(gg.loc[gg["__s"].idxmax(), "offset"])
    return out


def realized(cand: pd.DataFrame, pickmap: dict, n_total: int) -> dict:
    rows = [cand[(cand.sample_id == s) & (cand.offset == o)].iloc[0] for s, o in pickmap.items()]
    d = pd.DataFrame(rows)
    return {
        "n_total": n_total,
        "IoU80_overall": round(d["iou80_real"].sum() / n_total, 4),
        "coverage": round(d["feasible"].mean(), 4),
        "mean_offset": round(d["offset"].mean(), 2),
        "late_picks": int((d["feasible"] == 0).sum()),
    }


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    grid = pd.read_csv(GRID)
    clim_mid = float(pd.read_csv(CLIM_CSV).iloc[0]["mean_mid"])
    disp_raw = pd.read_csv(DISPATCH_CSV)
    disp_raw["sample_id"] = disp_raw["site"].astype(str) + "-" + disp_raw["year"].astype(int).astype(str)
    disp = disp_raw[["sample_id"] + SAMPLE_FEATURES + ["dispatch_branch"]].drop_duplicates("sample_id")
    caps, _vp, _cd = build_val_caps([0.20], min_bin=20, min_pest=30, bin_w=BIN_W)

    summary = []
    for variant in VARIANTS:
        gv = grid[grid.variant == variant]
        val_c = build_cand(gv[gv.split == "val"], disp, clim_mid)
        test_c = build_cand(gv[gv.split == "test"], disp, clim_mid)
        n_total = test_c.sample_id.nunique()

        # train two selectors: blind vs coverage-aware (target only differs)
        vb = val_c.copy(); vb["iou"] = vb["target_blind"]; reg_blind = train_lgb_regressor(vb)
        vc = val_c.copy(); vc["iou"] = vc["target_cov"]; reg_cov = train_lgb_regressor(vc)
        s_blind = scores_of(reg_blind, test_c)
        s_cov = scores_of(reg_cov, test_c)

        # deploy_q20 cap per sample
        cap_map = {}
        for sid in test_c.sample_id.unique():
            alert = int(test_c[test_c.sample_id == sid]["alert_tstar"].iloc[0])
            cap, _ = lookup_deploy_cap(caps, 0.20, PEST, alert, BIN_W)
            cap_map[sid] = cap

        methods = {
            "selector_blind(현재)": pick(test_c, s_blind),
            "selector_cov_aware": pick(test_c, s_cov),
            "selector_cov+q20cap": pick(test_c, s_cov, cap_map),
        }
        # references
        # val-fixed (overall realized iou80 on val)
        vfix = int((val_c.groupby("offset")["iou80_real"].sum() / val_c.sample_id.nunique()).idxmax())
        methods[f"val_fixed(off{vfix})"] = {s: vfix for s in test_c.sample_id.unique()}
        # deploy_q20 on val_fixed base
        methods["deploy_q20(base val_fixed)"] = {s: int(choose_offset(vfix, cap_map[s]))
                                                 for s in test_c.sample_id.unique()}
        # per-sample feasible oracle
        orac = {}
        for sid, g in test_c.groupby("sample_id"):
            orac[sid] = int(g.loc[g["iou80_real"].idxmax(), "offset"])
        methods["oracle_feasible(상한)"] = orac

        for name, pm in methods.items():
            summary.append({"variant": variant, "method": name, **realized(test_c, pm, n_total)})

    summ = pd.DataFrame(summary)
    p = OUT_DIR / "coverage_selector_summary.csv"
    if p.exists() and not args.force:
        raise SystemExit(f"Refuse to overwrite {p}")
    summ.to_csv(p, index=False)
    print(f"  wrote {p}\n")
    print("=== COVERAGE-AWARE SELECTOR (WBPH 2024, ckpt-norm) — realized IoU80 (coverage-weighted, /58) ===")
    piv = summ.pivot_table(index="method", columns="variant",
                           values=["IoU80_overall", "coverage", "mean_offset"], aggfunc="first")
    order = ["selector_blind(현재)", "selector_cov_aware", "selector_cov+q20cap",
             None, "deploy_q20(base val_fixed)", "oracle_feasible(상한)"]
    print(summ[["variant", "method", "IoU80_overall", "coverage", "mean_offset", "late_picks"]]
          .sort_values(["variant", "method"]).to_string(index=False))
    print("\n  핵심: selector_blind=현재(붕괴 재현), selector_cov_aware=objective 수정본, deploy_q20=넘어야 할 기준")


if __name__ == "__main__":
    main()
