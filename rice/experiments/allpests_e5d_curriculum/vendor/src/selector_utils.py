"""Coverage-aware offset selector (self-contained) for the WBPH workspace.

Reproduces the validated design: candidate table per (sample, offset) from the ckpt-norm
grid (mu per feasible offset; interpolated mu for infeasible), realized IoU80 at sigma
(0 if the offset is infeasible/late) as the LightGBM regression target, pick argmax.
No guard, no mu calibration. Selector is trained on VAL only (no test labels).
"""
from __future__ import annotations
import numpy as np
import pandas as pd

from .eval_metrics import recompute
from .io_utils import SAMPLE_FEATURES

FEATURE_COLS = [
    "offset", "pred_mu", "pred_lead", "mu_minus_clim_mid", "tstar_minus_clim_mid",
    "alert_tstar", "A_score_at_alert", "D_score_at_alert", "score_margin",
    "dispatch_score_at_alert", "dispatch_tau_used", "score_over_tau_margin",
    "recent_14d_mean_score", "recent_28d_mean_score",
    "score_above_tau_streak", "score_rolling_slope_14d",
    "p_mean_so_far_at_alert", "with_history", "dispatch_branch_is_D",
]


def _interp_mu(coarse_mu: dict, offsets):
    valid = sorted((o, m) for o, m in coarse_mu.items() if not pd.isna(m))
    if not valid:
        return {o: float("nan") for o in offsets}
    xs = np.array([v[0] for v in valid], float); ys = np.array([v[1] for v in valid], float)
    out = {}
    for o in offsets:
        out[o] = float(ys[0]) if o <= xs[0] else float(ys[-1]) if o >= xs[-1] else float(np.interp(o, xs, ys))
    return out


def build_candidates(grid_vs: pd.DataFrame, disp: pd.DataFrame, clim_mid: float,
                     offsets, sigma: float, doy_start: int, T: int) -> pd.DataFrame:
    """grid_vs = grid rows for one (variant, year, split). Returns candidate table with
    realized iou80 (feasible) / 0 (infeasible), features, and diagnostics per (sample, offset)."""
    dmap = disp.set_index("sample_id")
    rows = []
    for sid, sub in grid_vs.groupby("sample_id"):
        feas = {int(r.offset): r for r in sub.itertuples()}
        coarse_mu = {o: float(feas[o].mu) for o in feas}
        L = float(sub["L"].iloc[0]); R = float(sub["R"].iloc[0]); alert = float(sub["alert_tstar"].iloc[0])
        mu_curve = _interp_mu(coarse_mu, offsets)
        if sid not in dmap.index:
            continue
        df = dmap.loc[sid]
        for o in offsets:
            feasible = o in feas
            mu = coarse_mu[o] if feasible else float(mu_curve[o])
            eval_doy = alert + o
            m = recompute(mu, sigma, eval_doy, L, R, doy_start, T)
            rec = {
                "sample_id": sid, "offset": int(o), "alert_tstar": alert, "L": L, "R": R,
                "eval_doy": eval_doy, "true_start": L + 1, "true_mid": 0.5 * (L + R),
                "pred_mu": mu, "pred_lead": mu - alert,
                "mu_minus_clim_mid": mu - clim_mid, "tstar_minus_clim_mid": alert - clim_mid,
                "dispatch_branch_is_D": 1 if str(df.get("dispatch_branch")) == "D" else 0,
                "feasible": int(feasible),
                "pred_L80": m["pred_L80"], "pred_R80": m["pred_R80"], "pred_point": m["pred_point"],
                "width80": m["width80"], "pi_hit80": m["pi_hit80"], "mae_center": m["mae_center"],
                "iou80_geom": m["iou80"],
                "iou80_real": m["iou80"] if feasible else 0.0,   # realized (late/infeasible -> 0)
                "target_cov": m["iou80"] if feasible else 0.0,
            }
            for f in SAMPLE_FEATURES:
                if f != "alert_tstar":
                    rec[f] = df.get(f)
            rows.append(rec)
    return pd.DataFrame(rows)


def _prep_X(df: pd.DataFrame) -> np.ndarray:
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.median(numeric_only=True)).fillna(0.0).values


def train_selector(val_cand: pd.DataFrame, seed: int = 0):
    import lightgbm as lgb
    m = lgb.LGBMRegressor(n_estimators=120, learning_rate=0.05, num_leaves=15,
                          min_child_samples=20, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.9, reg_lambda=1.0, random_state=seed, verbosity=-1)
    m.fit(_prep_X(val_cand), val_cand["target_cov"].values)
    return m


def pick_offsets(model, test_cand: pd.DataFrame) -> dict:
    s = model.predict(_prep_X(test_cand))
    c = test_cand.copy(); c["__s"] = s
    return {sid: int(g.loc[g["__s"].idxmax(), "offset"]) for sid, g in c.groupby("sample_id")}


def picked_rows(test_cand: pd.DataFrame, pick: dict) -> pd.DataFrame:
    return pd.DataFrame([test_cand[(test_cand.sample_id == s) & (test_cand.offset == o)].iloc[0]
                         for s, o in pick.items()])
