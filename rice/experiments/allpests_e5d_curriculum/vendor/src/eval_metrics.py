"""Self-contained analytic Stage-2 interval metrics for the WBPH workspace.

The Stage-2 head is a FIXED-sigma Gaussian on the lead-predicted center mu, so the
conditional PMF, its 80% shortest-mass interval, and IoU vs the true interval can be
recomputed analytically from (mu, sigma, eval_tstar) — no model re-run. This mirrors the
legacy rice.src.train_eval math (validated bit-exact vs compare_eval/matched_eval.json);
reimplemented here so this workspace has NO dependency on legacy code (legacy = data only).

Coordinate convention: DOY in, DOY out. Internally relative index rel = DOY - (doy_start-1),
t = 1..T. True event interval is [L+1, R] (discrete label convention).
"""
from __future__ import annotations
import numpy as np


def shortest_mass_interval(pmf, target_mass: float = 0.8):
    """Shortest contiguous [L,R] (1-indexed inclusive) with mass >= target. Tie -> earlier L."""
    p = np.clip(np.nan_to_num(np.asarray(pmf, float), nan=0.0, posinf=0.0, neginf=0.0), 0.0, None)
    T = len(p)
    tot = float(p.sum())
    if tot <= 0.0:
        return 1, T, True
    a = 0; cum = 0.0; best_a = 0; best_b = T - 1; best_len = 10**18; found = False
    for b in range(T):
        cum += p[b]
        while a <= b and (cum - p[a]) >= target_mass:
            cum -= p[a]; a += 1
        if cum >= target_mass:
            cur = b - a
            if (not found) or (cur < best_len) or (cur == best_len and a < best_a):
                found = True; best_len = cur; best_a = a; best_b = b
    if not found:
        return 1, T, True
    return best_a + 1, best_b + 1, False


def overlap_iou(pL, pR, tL, tR) -> float:
    """IoU of predicted [pL,pR] vs true [tL+1, tR] (matches legacy overlap_metrics)."""
    tL2 = int(tL) + 1
    inter = max(0, min(int(pR), int(tR)) - max(int(pL), tL2) + 1)
    union = max(1, max(int(pR), int(tR)) - min(int(pL), tL2) + 1)
    return float(inter) / float(union) if union > 0 else 0.0


def cond_gaussian_pmf(mu_rel: float, sigma: float, tstar_rel: int, T: int) -> np.ndarray:
    t = np.arange(1, T + 1, dtype=float)
    lp = -0.5 * ((t - float(mu_rel)) / float(sigma)) ** 2
    lp -= lp.max()
    p = np.exp(lp)
    if tstar_rel > 0:
        p[:tstar_rel] = 0.0     # condition on event strictly after eval_tstar
    s = float(p.sum())
    return p / s if s > 0 else p


def recompute(mu_DOY, sigma, eval_tstar_DOY, L_DOY, R_DOY, doy_start, T) -> dict:
    """Analytic 80% interval + IoU at (mu, sigma) conditioned on eval_tstar. DOY units."""
    d = int(doy_start) - 1
    mu_rel = float(mu_DOY) - d
    tstar_rel = int(round(float(eval_tstar_DOY) - d))
    tL = int(round(float(L_DOY) - d)); tR = int(round(float(R_DOY) - d))
    p = cond_gaussian_pmf(mu_rel, sigma, tstar_rel, int(T))
    if float(p.sum()) <= 0.0:
        pL, pR, pp = 1, int(T), int(T)
    else:
        pL, pR, _ = shortest_mass_interval(p, 0.8)
        cdf = np.cumsum(p)
        pp = int(np.searchsorted(cdf, 0.5) + 1) if cdf[-1] >= 0.5 else int(T)
    pL = max(1, min(int(pL), int(T))); pR = max(1, min(int(pR), int(T)))
    if pL > pR:
        pL, pR = pR, pL
    iou = overlap_iou(pL, pR, tL, tR)
    true_mid = 0.5 * (float(L_DOY) + float(R_DOY))
    return {
        "iou80": float(iou), "pred_L80": int(pL + d), "pred_R80": int(pR + d),
        "pred_point": int(pp + d), "width80": int(pR - pL + 1),
        "pi_hit80": int((pL + d) <= true_mid <= (pR + d)),
        "mae_center": float(abs(float(mu_DOY) - true_mid)),
    }
