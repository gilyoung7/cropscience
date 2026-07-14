"""Shared helpers for WBPH 2024 ckpt-norm IoU-improvement experiments
(mu calibration / denser offset grid / sigma sweep).

NOT a runnable script — imported by:
  phase_t_wbph_mu_calibration.py, phase_t_wbph_sigma_sweep.py,
  phase_t_wbph_dense_offset_eval.py

Key idea: the Stage-2 head is a FIXED-sigma Gaussian on the lead-predicted center mu.
So the conditional PMF (and hence the 80% shortest-mass interval / IoU80) can be
reconstructed ANALYTICALLY from (mu, sigma, eval_tstar) — no model re-run needed for
mu-shift (calibration) or sigma changes. Only NEW offsets need a real model forward
(see phase_t_wbph_dense_offset_grid.py). Reconstruction is self-checked against the
validated grid's iou80 (validate_reconstruction()).

ckpt-norm ONLY. No wrong-norm / phase_r artifacts. No test labels in any learning step.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

from rice.configs.base import RICE_ROOT
from rice.src.train_eval import shortest_mass_interval_1d, overlap_metrics
from rice.scripts.phase_b_stage2_offset_selector_v2_ranking import (
    interp_mu_curve, train_lgb_regressor, _prep_X, SAMPLE_FEATURES)
from rice.scripts.phase_t_stage2_offset_constraint import (
    build_val_caps, lookup_deploy_cap, choose_offset)

# WBPH lead_v3 ckpt geometry (doy_start=60, doy_end=300)
DOY_START = 60
T = 300 - 60 + 1                      # 241
COARSE_OFFSETS = [7, 14, 21, 30, 45, 60]
BIN_W = 20
PEST = "WBPH"

GRID = RICE_ROOT / "outputs/diag/stage2_direct_neighbor_wbph_2024/wbph_offset_grid.csv"
DISPATCH_CSV = RICE_ROOT / "outputs/stage2/batch_2024_bestgate/WBPH/gate_dispatch_group_tau_R088_features_per_sy.csv"
CLIM_CSV = RICE_ROOT / "outputs/stage2/batch_2024_bestgate/WBPH/climatology_train_stats.csv"
OUT_DIR = RICE_ROOT / "outputs/diag/stage2_ckptnorm_selector_wbph/iou_improvement"


# ----------------------------- inputs --------------------------------------
def load_inputs(grid_path: Path | str = GRID):
    grid = pd.read_csv(grid_path)
    clim_mid = float(pd.read_csv(CLIM_CSV).iloc[0]["mean_mid"])
    d = pd.read_csv(DISPATCH_CSV)
    d["sample_id"] = d["site"].astype(str) + "-" + d["year"].astype(int).astype(str)
    disp = d[["sample_id"] + SAMPLE_FEATURES + ["dispatch_branch"]].drop_duplicates("sample_id")
    return grid, clim_mid, disp


# ------------------- analytic Gaussian PMF -> IoU80 ------------------------
def recompute(mu_DOY: float, sigma: float, eval_tstar_DOY: float,
              L_DOY: float, R_DOY: float) -> dict:
    """Conditional fixed-sigma Gaussian PMF (zero mass at/before eval_tstar),
    its 80% shortest-mass interval, and IoU vs true [L+1, R]. All DOY in, DOY out."""
    d = DOY_START - 1
    mu_rel = mu_DOY - d
    tstar_rel = int(round(eval_tstar_DOY - d))
    tL = int(round(L_DOY - d)); tR = int(round(R_DOY - d))
    t = np.arange(1, T + 1, dtype=float)
    logp = -0.5 * ((t - mu_rel) / float(sigma)) ** 2
    logp -= logp.max()
    p = np.exp(logp)
    if tstar_rel > 0:
        p[:tstar_rel] = 0.0            # condition on event strictly after eval_tstar
    s = float(p.sum())
    if s <= 0.0:
        pL, pR, pp = 1, T, T
    else:
        p = p / s
        pL, pR, _ = shortest_mass_interval_1d(p, target_mass=0.8, Tend=T, normalize=False)
        cdf = np.cumsum(p)
        pp = int(np.searchsorted(cdf, 0.5) + 1) if cdf[-1] >= 0.5 else T
    pL = max(1, min(int(pL), T)); pR = max(1, min(int(pR), T))
    if pL > pR:
        pL, pR = pR, pL
    iou, _, _ = overlap_metrics(pL, pR, tL, tR)
    true_mid = 0.5 * (L_DOY + R_DOY)
    return {
        "iou80": float(iou), "pred_L80": pL + d, "pred_R80": pR + d,
        "pred_point": pp + d, "width80": pR - pL + 1,
        "pi_hit80": int((pL + d) <= true_mid <= (pR + d)),
        "mae_center": abs(mu_DOY - true_mid),
    }


def validate_reconstruction(grid: pd.DataFrame, sigma: float = 5.0) -> dict:
    """Recompute iou80 from (mu, sigma=5) for every real grid row; compare to stored
    grid iou80. max|Δ| ~ 0 confirms the analytic PMF reconstruction is faithful."""
    out = {}
    for variant in grid.variant.unique():
        sub = grid[(grid.variant == variant) & (grid.split == "test")]
        diffs = []
        for r in sub.itertuples():
            m = recompute(float(r.mu), sigma, float(r.eval_tstar), float(r.L), float(r.R))
            diffs.append(abs(m["iou80"] - float(r.iou80)))
        out[variant] = float(np.max(diffs)) if diffs else float("nan")
    return out


# ------------------------ mu calibration (val only) ------------------------
def learn_bias(grid_val_feasible: pd.DataFrame, kind: str):
    """Learn additive mu bias b (mu_corrected = mu + b) from VAL feasible rows.
    residual = true_mid - mu. kind in {none, global, offset, alert_bin}.
    Returns bias_fn(offset, alert_tstar) -> b."""
    g = grid_val_feasible.copy()
    g["resid"] = g["true_mid"] - g["mu"]
    if kind == "none":
        return lambda o, a: 0.0
    gb = float(g["resid"].mean())
    if kind == "global":
        return lambda o, a: gb
    if kind == "offset":
        per = g.groupby("offset")["resid"].mean().to_dict()
        return lambda o, a: float(per.get(int(o), gb))
    if kind == "alert_bin":
        g["__bin"] = (g["alert_tstar"] // BIN_W).astype(int) * BIN_W
        per = g.groupby("__bin")["resid"].mean().to_dict()
        return lambda o, a: float(per.get(int(a) // BIN_W * BIN_W, gb))
    raise ValueError(kind)


# ----------------------------- candidate grid ------------------------------
def build_cand(gvs: pd.DataFrame, disp: pd.DataFrame, clim_mid: float,
               offsets: list[int], sigma: float = 5.0, bias_fn=None) -> pd.DataFrame:
    """Candidate grid for one (variant, split) over `offsets`.
      feasible (real grid row) -> realized iou80 recomputed at (mu+bias, sigma)
      infeasible (no row)       -> interpolated mu (feature only), realized reward 0
    Selector target_cov = realized iou80 (coverage-aware); target_blind = iou80 from
    the (possibly fake) mu regardless of feasibility."""
    dmap = disp.set_index("sample_id")
    rows = []
    bias_fn = bias_fn or (lambda o, a: 0.0)
    for sid, sub in gvs.groupby("sample_id"):
        feas = {int(r.offset): r for r in sub.itertuples()}
        coarse_mu = {o: float(feas[o].mu) for o in feas}
        L = float(sub["L"].iloc[0]); R = float(sub["R"].iloc[0])
        alert = float(sub["alert_tstar"].iloc[0])
        mu_curve = interp_mu_curve(coarse_mu, offsets)
        if sid not in dmap.index:
            continue
        df = dmap.loc[sid]
        for o in offsets:
            feasible = o in feas
            mu0 = coarse_mu[o] if feasible else float(mu_curve[o])
            mu = mu0 + bias_fn(o, alert)
            m = recompute(mu, sigma, alert + o, L, R)
            rec = {
                "sample_id": sid, "offset": int(o), "L": L, "R": R, "alert_tstar": alert,
                "pred_mu": mu, "pred_lead": mu - alert,
                "mu_minus_clim_mid": mu - clim_mid, "tstar_minus_clim_mid": alert - clim_mid,
                "dispatch_branch_is_D": 1 if str(df.get("dispatch_branch")) == "D" else 0,
                "feasible": int(feasible),
                "iou80_real": m["iou80"] if feasible else 0.0,
                "mae_center_real": m["mae_center"] if feasible else np.nan,
                "pi_hit80_real": m["pi_hit80"] if feasible else 0,
                "width80_real": m["width80"] if feasible else np.nan,
                "target_blind": m["iou80"],
                "target_cov": (m["iou80"] if feasible else 0.0),
            }
            for f in SAMPLE_FEATURES:
                if f != "alert_tstar":
                    rec[f] = df.get(f)
            rows.append(rec)
    return pd.DataFrame(rows)


# ----------------------------- selector ------------------------------------
def train_selector(val_cand: pd.DataFrame, target: str, seed: int):
    v = val_cand.copy(); v["iou"] = v[target]
    return train_lgb_regressor(v, seed=seed)


def pick_argmax(cand: pd.DataFrame, reg, cap_map: dict | None = None) -> dict:
    score = reg.predict(_prep_X(cand).values)
    c = cand.copy(); c["__s"] = score
    out = {}
    for sid, g in c.groupby("sample_id"):
        gg = g if cap_map is None else g[g.offset <= cap_map.get(sid, 1e9)]
        if gg.empty:
            gg = g[g.offset == g.offset.min()]
        out[sid] = int(gg.loc[gg["__s"].idxmax(), "offset"])
    return out


def pick_fixed(cand: pd.DataFrame, offset: int, cap_map: dict | None = None) -> dict:
    out = {}
    for sid, g in cand.groupby("sample_id"):
        o = offset if cap_map is None else int(choose_offset(offset, cap_map.get(sid, 1e9)))
        out[sid] = int(o)
    return out


def pick_oracle(cand: pd.DataFrame) -> dict:
    out = {}
    for sid, g in cand.groupby("sample_id"):
        out[sid] = int(g.loc[g["iou80_real"].idxmax(), "offset"])
    return out


def realized(cand: pd.DataFrame, pm: dict, n_total: int) -> dict:
    rows = [cand[(cand.sample_id == s) & (cand.offset == o)].iloc[0] for s, o in pm.items()]
    d = pd.DataFrame(rows)
    feas = d[d.feasible == 1]
    return {
        "n_total": n_total,
        "IoU80_overall": round(float(d["iou80_real"].sum()) / n_total, 4),
        "coverage": round(float(d["feasible"].mean()), 4),
        "late_count": int((d["feasible"] == 0).sum()),
        "mean_offset": round(float(d["offset"].mean()), 2),
        "MAE_center": round(float(feas["mae_center_real"].mean()), 4) if len(feas) else float("nan"),
        "PI_hit": round(float(feas["pi_hit80_real"].sum()) / n_total, 4),
        "width80": round(float(feas["width80_real"].mean()), 2) if len(feas) else float("nan"),
    }


def deploy_q20_cap_map(test_cand: pd.DataFrame) -> dict:
    caps, _vp, _cd = build_val_caps([0.20], min_bin=20, min_pest=30, bin_w=BIN_W)
    cm = {}
    for sid in test_cand.sample_id.unique():
        alert = int(test_cand[test_cand.sample_id == sid]["alert_tstar"].iloc[0])
        cm[sid], _ = lookup_deploy_cap(caps, 0.20, PEST, alert, BIN_W)
    return cm


def val_fixed_offset(val_cand: pd.DataFrame) -> int:
    n_val = val_cand.sample_id.nunique()
    return int((val_cand.groupby("offset")["iou80_real"].sum() / n_val).idxmax())


def cov_aware_eval(grid: pd.DataFrame, variant: str, disp, clim_mid, offsets,
                   sigma: float, bias_fn, seeds: list[int]) -> dict:
    """Train cov-aware selector per seed on val, eval on test; return seed-mean metrics
    plus oracle_feasible and deploy_q20 references. Used by all three experiments."""
    gv = grid[grid.variant == variant]
    val_c = build_cand(gv[gv.split == "val"], disp, clim_mid, offsets, sigma, bias_fn)
    test_c = build_cand(gv[gv.split == "test"], disp, clim_mid, offsets, sigma, bias_fn)
    n_total = test_c.sample_id.nunique()
    per_seed = []
    for sd in seeds:
        reg = train_selector(val_c, "target_cov", sd)
        per_seed.append(realized(test_c, pick_argmax(test_c, reg), n_total))
    sel = pd.DataFrame(per_seed)
    res = {f"cov_{k}_mean": round(float(sel[k].mean()), 4) for k in
           ["IoU80_overall", "coverage", "mean_offset", "MAE_center", "PI_hit"]}
    res["cov_IoU80_std"] = round(float(sel["IoU80_overall"].std()), 4)
    orac = realized(test_c, pick_oracle(test_c), n_total)
    res.update({f"oracle_{k}": orac[k] for k in ["IoU80_overall", "MAE_center", "PI_hit", "mean_offset"]})
    vfix = val_fixed_offset(val_c)
    dq = realized(test_c, pick_fixed(test_c, vfix, deploy_q20_cap_map(test_c)), n_total)
    res.update({f"deploy_q20_{k}": dq[k] for k in ["IoU80_overall", "MAE_center", "PI_hit"]})
    res["val_fixed_offset"] = vfix
    return res
