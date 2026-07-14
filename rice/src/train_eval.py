from __future__ import annotations

import numpy as np
import torch
from rice.configs import config as C
from contextlib import nullcontext

CTYPE_INTERVAL = 0  # interval=0, right=1, left=2


# -------------------------
# Loss (interval/right/left censored)
# -------------------------
def interval_nll_per_sample(hazard, L, R, ctype, Tend, tstar=None):
    """
    hazard: (B,T) in (0,1)
    L,R,ctype: (B,)
    returns: (B,) nll
    """
    B, Tcur = hazard.shape
    assert Tcur == Tend

    log_surv_terms = torch.log1p(-hazard)          # (B,T)
    logS_full = torch.cumsum(log_surv_terms, dim=1)  # log S_t
    if tstar is None:
        logS = logS_full
    else:
        idx_tstar = (tstar.long().clamp(min=1, max=Tend) - 1).view(-1, 1)
        logS_at_tstar = logS_full.gather(1, idx_tstar).squeeze(1)
        logS = logS_full - logS_at_tstar.view(-1, 1)

    L = torch.clamp(L, 1, Tend)
    R = torch.clamp(R, 1, Tend)
    idxL = (L - 1).long()
    idxR = (R - 1).long()

    logS_L = logS.gather(1, idxL.view(-1, 1)).squeeze(1)
    logS_R = logS.gather(1, idxR.view(-1, 1)).squeeze(1)

    nll = torch.zeros(B, device=hazard.device)

    mi = (ctype == 0)   # interval
    mr = (ctype == 1)   # right
    ml = (ctype == 2)   # left

    eps = 1e-12

    # interval: -log(S_L - S_R)
    if mi.any():
        a = logS_L[mi]
        b = logS_R[mi]
        a = torch.maximum(a, b + 1e-8)
        log_interval = a + torch.log1p(-torch.exp(b - a))
        nll[mi] = -torch.clamp(log_interval, min=np.log(eps))

    # right: -log(S_T)
    if mr.any():
        nll[mr] = torch.clamp(-logS[:, -1][mr], max=1e6)

    # left: -log(1 - S_R)
    if ml.any():
        a = logS_R[ml]
        cutoff = -0.6931471805599453  # log(0.5)
        out = torch.empty_like(a)
        m = a < cutoff
        out[m] = torch.log1p(-torch.exp(a[m]))
        out[~m] = torch.log(-torch.expm1(a[~m]))
        nll[ml] = torch.clamp(-out, max=1e6)

    nll = torch.nan_to_num(nll, nan=1e6, posinf=1e6, neginf=0.0)

    return nll


def weighted_loss_from_ctype(nll_vec, ctype):
    w = torch.ones_like(nll_vec)
    w = torch.where(ctype == 0, w.new_tensor(C.W_INTERVAL), w)
    w = torch.where(ctype == 1, w.new_tensor(C.W_RIGHT), w)
    w = torch.where(ctype == 2, w.new_tensor(C.W_LEFT), w)
    return (w * nll_vec).mean()


def _class_weights_from_ctype(ctype):
    w = torch.ones_like(ctype, dtype=torch.float32)
    w = torch.where(ctype == 0, w.new_tensor(C.W_INTERVAL), w)
    w = torch.where(ctype == 1, w.new_tensor(C.W_RIGHT), w)
    w = torch.where(ctype == 2, w.new_tensor(C.W_LEFT), w)
    return w


def _early_tstar_weights(tstar_f, Tend, min_weight: float):
    if min_weight >= 1.0:
        return torch.ones_like(tstar_f, dtype=torch.float32)
    denom = max(int(Tend) - 1, 1)
    rel = (tstar_f.float() - 1.0) / float(denom)
    w = float(min_weight) + (1.0 - float(min_weight)) * rel
    return torch.clamp(w, min=float(min_weight), max=1.0)


def _lead_window_weights(
    L_f,
    ctype_f,
    tstar_f,
    *,
    enabled: bool,
    target_lead_min: int,
    target_lead_max: int,
    support_lead_min: int,
    support_lead_max: int,
    min_weight: float,
):
    if not enabled:
        return torch.ones_like(tstar_f, dtype=torch.float32)
    if not (int(support_lead_min) <= int(target_lead_min) <= int(target_lead_max) <= int(support_lead_max)):
        raise ValueError("lead weighting requires support_min <= target_min <= target_max <= support_max")

    w = torch.ones_like(tstar_f, dtype=torch.float32)
    mi = ctype_f == CTYPE_INTERVAL
    if not mi.any():
        return w

    lead = (L_f.float() + 1.0) - tstar_f.float()
    ww = torch.full_like(lead, float(min_weight), dtype=torch.float32)
    target = (lead >= float(target_lead_min)) & (lead <= float(target_lead_max))
    ww = torch.where(target, ww.new_tensor(1.0), ww)

    left = (lead >= float(support_lead_min)) & (lead < float(target_lead_min))
    if int(target_lead_min) > int(support_lead_min):
        rel = (lead - float(support_lead_min)) / float(int(target_lead_min) - int(support_lead_min))
        ww = torch.where(left, float(min_weight) + (1.0 - float(min_weight)) * rel, ww)

    right = (lead > float(target_lead_max)) & (lead <= float(support_lead_max))
    if int(support_lead_max) > int(target_lead_max):
        rel = (float(support_lead_max) - lead) / float(int(support_lead_max) - int(target_lead_max))
        ww = torch.where(right, float(min_weight) + (1.0 - float(min_weight)) * rel, ww)

    w = torch.where(mi, torch.clamp(ww, min=float(min_weight), max=1.0), w)
    return w


def _lead_loss_mode_weights(
    L_f,
    ctype_f,
    tstar_f,
    *,
    mode: str,
    lead_min: int,
    lead_max: int,
    mid_lead_min: int,
    mid_lead_max: int,
    late_exclude_days: int,
    weight_1_14: float = 0.0,
    weight_15_29: float = 0.7,
    weight_30_60: float = 1.5,
    weight_61_75: float = 1.0,
    weight_gt75: float = 0.25,
):
    """
    Additional Stage-2 event-row loss weighting by lead.

    This does not remove t* rows from the grouped sequence. It only changes the
    flat per-row loss contribution after causal context has been computed.
    Right-censored rows have no event day, so they keep weight 1.
    """
    mode = str(mode).lower()
    if mode in ("none", "", "off", "false", "0"):
        return torch.ones_like(tstar_f, dtype=torch.float32)
    if mode not in {"mask", "weighted"}:
        raise ValueError(f"Unknown stage2_lead_loss_mode: {mode}")
    if not (int(lead_min) <= int(mid_lead_min) <= int(mid_lead_max) <= int(lead_max)):
        raise ValueError("lead loss mode requires lead_min <= mid_lead_min <= mid_lead_max <= lead_max")

    w = torch.ones_like(tstar_f, dtype=torch.float32)
    mi = ctype_f == CTYPE_INTERVAL
    if not mi.any():
        return w

    # Interval-censored event starts at L+1 in the current label convention.
    lead = (L_f.float() + 1.0) - tstar_f.float()
    ew = torch.ones_like(lead, dtype=torch.float32)
    if mode == "mask":
        ew = ((lead >= float(lead_min)) & (lead <= float(lead_max))).to(torch.float32)
    else:
        early = lead > float(lead_max)
        support_late = (lead >= float(lead_min)) & (lead < float(mid_lead_min))
        target = (lead >= float(mid_lead_min)) & (lead <= float(mid_lead_max))
        support_early = (lead > float(mid_lead_max)) & (lead <= float(lead_max))
        late = lead <= float(late_exclude_days)
        ew = torch.full_like(lead, float(weight_gt75), dtype=torch.float32)
        ew = torch.where(early, ew.new_tensor(float(weight_gt75)), ew)
        ew = torch.where(support_late, ew.new_tensor(float(weight_15_29)), ew)
        ew = torch.where(target, ew.new_tensor(float(weight_30_60)), ew)
        ew = torch.where(support_early, ew.new_tensor(float(weight_61_75)), ew)
        ew = torch.where(late, ew.new_tensor(float(weight_1_14)), ew)
        ew = torch.where(lead < 1.0, ew.new_tensor(0.0), ew)

    return torch.where(mi, ew, w)


def _lead_bucket_counts(L_f, ctype_f, tstar_f, *, late_exclude_days: int, lead_min: int, lead_max: int, mid_lead_min: int, mid_lead_max: int):
    mi = ctype_f == CTYPE_INTERVAL
    out = {"lead_1_14": 0, "lead_15_29": 0, "lead_30_60": 0, "lead_61_75": 0, "lead_gt75": 0}
    if not mi.any():
        return out
    lead = ((L_f.float() + 1.0) - tstar_f.float())[mi]
    out["lead_1_14"] = int(((lead >= 1.0) & (lead <= float(late_exclude_days))).sum().item())
    out["lead_15_29"] = int(((lead >= float(lead_min)) & (lead < float(mid_lead_min))).sum().item())
    out["lead_30_60"] = int(((lead >= float(mid_lead_min)) & (lead <= float(mid_lead_max))).sum().item())
    out["lead_61_75"] = int(((lead > float(mid_lead_max)) & (lead <= float(lead_max))).sum().item())
    out["lead_gt75"] = int((lead > float(lead_max)).sum().item())
    return out


def conditional_pmf_entropy(hazard_f, tstar_f, Tend: int, eps: float = 1e-12):
    pmf, _, _ = hazard_to_pmf_cdf_logS(hazard_f, tstar=tstar_f)
    idx = torch.arange(int(Tend), device=pmf.device).view(1, -1)
    post_mask = idx >= tstar_f.long().clamp(min=1, max=int(Tend)).view(-1, 1)
    pmf_post = torch.where(post_mask, pmf, torch.zeros_like(pmf))
    entropy = -(pmf_post * torch.log(pmf_post.clamp_min(float(eps)))).sum(dim=1)
    return entropy


def pmf_entropy(hazard_f, tstar_f, Tend: int, *, conditional: bool = True, eps: float = 1e-12):
    if bool(conditional):
        return conditional_pmf_entropy(hazard_f, tstar_f, Tend=Tend, eps=eps)
    pmf, _, _ = hazard_to_pmf_cdf_logS(hazard_f, tstar=None)
    entropy = -(pmf * torch.log(pmf.clamp_min(float(eps)))).sum(dim=1)
    return entropy


def asymmetric_mu_loss(
    mu,
    L,
    R,
    ctype,
    *,
    Tend: int,
    asym_weight: float = 10.0,
    right_weight: float = 0.3,
    target_offset: float = 0.0,
    asym_weight_early: float = 0.0,
    target_early_offset: float = 30.0,
    target_mode: str = "l_offset",
    zone_late_weight: float = 0.0,
    zone_too_late_weight: float = 0.0,
    zone_missed_weight: float = 0.0,
    zone_too_early_weight: float = 0.0,
    zone_too_late_threshold: float = 15.0,
    zone_missed_threshold: float = 22.0,
    zone_too_early_threshold: float = 23.0,
    sample_weight=None,
    right_anchor: float | None = None,
    lead_loss_mask=None,
    aux_lead_lambda: float = 0.0,
    aux_lead_huber_delta: float = 10.0,
):
    """
    Gaussian PMF mu-loss for Stage 2.

    target_mode='l_offset' (default, backward-compatible):
        target = L + target_offset
        loss_event = mean(weight * (mu - target)^2)
            weight = asym_weight if (mu - target) > 0 else 1.0
        optional early one-sided MSE:
            loss_early = asym_weight_early * mean(max(0, (L - target_early_offset) - mu)^2)

    target_mode='center' (zone-aware, mid-target):
        target = mid = (L + R) / 2
        base           = (mu - mid)^2                              -- symmetric MSE
        soft_late      = w_late      * max(0, mu - mid)^2          -- mid 우측
        soft_too_late  = w_too_late  * max(0, mu - (L + thr_tl))^2 -- USEFUL 우측 초과
        soft_missed    = w_missed    * max(0, mu - (L + thr_m))^2  -- MISSED 진입
        soft_too_early = w_too_early * max(0, (L - thr_te) - mu)^2 -- TOO_EARLY 진입
        loss_event = mean(base) + mean(soft_late) + mean(soft_too_late)
                     + mean(soft_missed) + mean(soft_too_early)
        In this mode asym_weight / asym_weight_early / target_offset /
        target_early_offset are ignored; zone_* hparams take over.

    Right-censored (ctype==1): loss_right = right_weight * mean((mu - Tend)^2)  (same in both modes)
    Left-censored (ctype==2): ignored.
    """
    mi = ctype == CTYPE_INTERVAL
    # Phase B: lead_from_alert mode restricts mu/L loss to cells where alert
    # actually occurred. Pre-alert cells stay in attention/context (valid_mask
    # unchanged) but are excluded from this asymmetric loss. lead_loss_mask is
    # None for absolute mode (no-op).
    if lead_loss_mask is not None:
        mi = mi & lead_loss_mask.to(mi.dtype if mi.dtype == torch.bool else torch.bool)
    mr = ctype == 1
    parts: dict = {
        "mu_event": 0.0,
        "mu_event_n": 0,
        "mu_right": 0.0,
        "mu_right_n": 0,
        "mu_early": 0.0,
        "mu_early_n": 0,
        "mu_late": 0.0,
        "mu_late_n": 0,
        "mu_too_late": 0.0,
        "mu_too_late_n": 0,
        "mu_missed": 0.0,
        "mu_missed_n": 0,
        "mu_too_early": 0.0,
        "mu_too_early_n": 0,
        "mu_mean_event": float("nan"),
        "mu_minus_L_mean": float("nan"),
        "mu_minus_L_abs_mean": float("nan"),
        "mu_minus_mid_mean": float("nan"),
        "mu_minus_mid_abs_mean": float("nan"),
        "mu_pos_frac": float("nan"),
        "mu_minus_target_mean": float("nan"),
        "aux_lead_raw": 0.0,
        "aux_lead_weighted": 0.0,
        "aux_lead_n": 0,
        "aux_lead_mean_abs_delta": float("nan"),
        "aux_lead_lambda": float(aux_lead_lambda),
        "aux_lead_huber_delta": float(aux_lead_huber_delta),
        "target_offset": float(target_offset),
        "target_early_offset": float(target_early_offset),
        "target_mode": str(target_mode),
        "zone_too_late_threshold": float(zone_too_late_threshold),
        "zone_missed_threshold": float(zone_missed_threshold),
        "zone_too_early_threshold": float(zone_too_early_threshold),
    }
    loss = mu.new_tensor(0.0)

    if mi.any():
        L_e = L[mi].float()
        R_e = R[mi].float()
        mu_e = mu[mi]
        mid_e = (L_e + R_e) * 0.5
        # Per-sample multiplier (Phase S5 long-lead weighting).
        # When sample_weight is None, behaves as all-ones (legacy behaviour).
        if sample_weight is not None:
            sw_e = sample_weight[mi].to(mu_e.dtype)
        else:
            sw_e = mu_e.new_ones(mu_e.shape)

        if str(target_mode) == "center":
            target = mid_e
            delta = mu_e - target
            loss_event = ((delta * delta) * sw_e).mean()
        else:
            target = L_e + float(target_offset)
            delta = mu_e - target
            weight = torch.where(
                delta > 0,
                mu.new_tensor(float(asym_weight)),
                mu.new_tensor(1.0),
            )
            loss_event = (weight * delta * delta * sw_e).mean()

        loss = loss + loss_event
        parts["mu_event"] = float(loss_event.detach().item())
        parts["mu_event_n"] = int(mi.long().sum().item())
        parts["mu_mean_event"] = float(mu_e.detach().mean().item())

        # Phase B aux lead loss (Huber on delta = mu - L = predicted_lead -
        # target_lead). Applies to the same interval+lead-loss-mask cells as
        # the event loss; unweighted (no sample_weight, no asym). Only active
        # when --stage2_aux_lead_lambda > 0. In l_offset mode delta is
        # (mu - L - target_offset); for the standard target_offset=0 this is
        # exactly the lead residual.
        if float(aux_lead_lambda) > 0.0:
            huber_d = float(aux_lead_huber_delta)
            abs_d = delta.abs()
            quad = torch.where(
                abs_d <= huber_d,
                0.5 * delta * delta,
                huber_d * (abs_d - 0.5 * huber_d),
            )
            aux_raw = quad.mean()
            aux_weighted = float(aux_lead_lambda) * aux_raw
            loss = loss + aux_weighted
            parts["aux_lead_raw"] = float(aux_raw.detach().item())
            parts["aux_lead_weighted"] = float(aux_weighted.detach().item())
            parts["aux_lead_n"] = int(mi.long().sum().item())
            parts["aux_lead_mean_abs_delta"] = float(abs_d.detach().mean().item())

        delta_L = (mu_e - L_e).detach()
        parts["mu_minus_L_mean"] = float(delta_L.mean().item())
        parts["mu_minus_L_abs_mean"] = float(delta_L.abs().mean().item())
        parts["mu_pos_frac"] = float((delta_L > 0).float().mean().item())
        parts["mu_minus_target_mean"] = float(delta.detach().mean().item())
        delta_mid = (mu_e - mid_e).detach()
        parts["mu_minus_mid_mean"] = float(delta_mid.mean().item())
        parts["mu_minus_mid_abs_mean"] = float(delta_mid.abs().mean().item())

        # Legacy early one-sided MSE (l_offset mode only).
        if str(target_mode) != "center" and float(asym_weight_early) > 0.0:
            lower = L_e - float(target_early_offset)
            early_excess = (lower - mu_e).clamp(min=0.0)
            n_early = int((early_excess > 0).long().sum().item())
            if n_early > 0:
                loss_early = float(asym_weight_early) * (early_excess * early_excess * sw_e).mean()
                loss = loss + loss_early
                parts["mu_early"] = float(loss_early.detach().item())
                parts["mu_early_n"] = n_early

        # Zone-aware soft penalties (always available; default weights = 0).
        if float(zone_late_weight) > 0.0:
            late_excess = (mu_e - mid_e).clamp(min=0.0)
            n_late = int((late_excess > 0).long().sum().item())
            if n_late > 0:
                loss_late = float(zone_late_weight) * (late_excess * late_excess).mean()
                loss = loss + loss_late
                parts["mu_late"] = float(loss_late.detach().item())
                parts["mu_late_n"] = n_late
        if float(zone_too_late_weight) > 0.0:
            thr_tl = L_e + float(zone_too_late_threshold)
            too_late_excess = (mu_e - thr_tl).clamp(min=0.0)
            n_tl = int((too_late_excess > 0).long().sum().item())
            if n_tl > 0:
                loss_tl = float(zone_too_late_weight) * (too_late_excess * too_late_excess).mean()
                loss = loss + loss_tl
                parts["mu_too_late"] = float(loss_tl.detach().item())
                parts["mu_too_late_n"] = n_tl
        if float(zone_missed_weight) > 0.0:
            thr_m = L_e + float(zone_missed_threshold)
            missed_excess = (mu_e - thr_m).clamp(min=0.0)
            n_m = int((missed_excess > 0).long().sum().item())
            if n_m > 0:
                loss_m = float(zone_missed_weight) * (missed_excess * missed_excess).mean()
                loss = loss + loss_m
                parts["mu_missed"] = float(loss_m.detach().item())
                parts["mu_missed_n"] = n_m
        if float(zone_too_early_weight) > 0.0:
            thr_te = L_e - float(zone_too_early_threshold)
            too_early_excess = (thr_te - mu_e).clamp(min=0.0)
            n_te = int((too_early_excess > 0).long().sum().item())
            if n_te > 0:
                loss_te = float(zone_too_early_weight) * (too_early_excess * too_early_excess).mean()
                loss = loss + loss_te
                parts["mu_too_early"] = float(loss_te.detach().item())
                parts["mu_too_early_n"] = n_te

    if mr.any():
        # Phase S10: right_anchor lets the right-cens pull aim at a value
        # smaller than Tend (e.g., L_max ≈ 240 or mid_max ≈ 220) so the
        # right-cens MSE keeps producing a learning signal without dragging
        # mu of *interval* samples all the way to Tend through the shared
        # head_mu parameters. When right_anchor is None or <= 0, fall back
        # to the legacy Tend anchor (backward-compatible).
        if right_anchor is not None and float(right_anchor) > 0.0:
            anchor_val = float(right_anchor)
        else:
            anchor_val = float(Tend)
        target_r = mu.new_tensor(anchor_val)
        delta_r = mu[mr] - target_r
        loss_right = (delta_r * delta_r).mean() * float(right_weight)
        loss = loss + loss_right
        parts["mu_right"] = float(loss_right.detach().item())
        parts["mu_right_n"] = int(mr.long().sum().item())
        parts["right_anchor"] = anchor_val

    return loss, parts


def gaussian_interval_nll_loss(
    mu,
    L,
    R,
    ctype,
    *,
    sigma: float,
    Tend: int,
    right_weight: float = 0.3,
    right_anchor: float | None = None,
    sample_weight=None,
    lead_loss_mask=None,
    continuity_correction: bool = False,
    eps: float = 1e-8,
):
    """
    Gaussian interval NLL for Stage 2 mu head (sigma fixed).

    Label convention (matches asymmetric_mu_loss / interval_nll_per_sample):
        Discrete event interval is T ∈ (L, R]  i.e. day-inclusive [L+1, R].

    Default continuous mapping:
        P(L < T ≤ R) = Φ((R - μ)/σ) - Φ((L - μ)/σ)
    This uses raw L, R as the half-open boundary, matching the discrete
    label semantics directly. (No half-day shift.)

    Continuity-corrected variant (ablation, off by default):
        Set continuity_correction=True to use (L + 0.5, R + 0.5) as the
        continuous proxies for the inclusive day boundary [L+1, R].

    Right-censored (ctype==1):
        T > C survival likelihood: P(T > C) = 1 - Φ((C - μ)/σ) = Φ((μ - C)/σ).
        nll_r = -log Φ((μ - C)/σ).  C = right_anchor if >0 else Tend.

    Left-censored (ctype==2):
        T ≤ R: nll_l = -log Φ((R - μ)/σ).

    Numerical stability:
        log(Φ(zR) - Φ(zL)) = log_ndtr(zR) + log1p(-exp(log_ndtr(zL) - log_ndtr(zR)))
        when zR >= zL.  We clamp zR >= zL + eps to avoid log(0).
    """
    mi = ctype == CTYPE_INTERVAL
    if lead_loss_mask is not None:
        mi = mi & lead_loss_mask.to(mi.dtype if mi.dtype == torch.bool else torch.bool)
    mr = ctype == 1
    ml = ctype == 2

    parts: dict = {
        "nll_event": 0.0,
        "nll_event_n": 0,
        "nll_right": 0.0,
        "nll_right_n": 0,
        "nll_left": 0.0,
        "nll_left_n": 0,
        "mu_mean_event": float("nan"),
        "mu_minus_L_mean": float("nan"),
        "mu_minus_mid_mean": float("nan"),
        "sigma": float(sigma),
        "continuity_correction": bool(continuity_correction),
        "right_anchor": float("nan"),
    }

    sigma_t = mu.new_tensor(float(sigma)).clamp(min=1e-6)
    loss = mu.new_tensor(0.0)
    log_ndtr = torch.special.log_ndtr

    if mi.any():
        L_e = L[mi].float()
        R_e = R[mi].float()
        mu_e = mu[mi]
        if continuity_correction:
            L_cont = L_e + 0.5
            R_cont = R_e + 0.5
        else:
            L_cont = L_e
            R_cont = R_e
        zL = (L_cont - mu_e) / sigma_t
        zR = (R_cont - mu_e) / sigma_t
        # Enforce zR > zL (always true given R > L); add small floor for safety.
        zR_safe = torch.maximum(zR, zL + eps)
        log_pL = log_ndtr(zL)
        log_pR = log_ndtr(zR_safe)
        # log(Φ(zR) - Φ(zL)) = log_pR + log1p(-exp(log_pL - log_pR))
        diff = (log_pL - log_pR).clamp(max=-eps)
        log_interval = log_pR + torch.log1p(-torch.exp(diff))
        # Match interval_nll_per_sample stability: clamp NaN/inf and bound the
        # per-sample NLL so a far-out mu cannot poison the batch with +inf.
        nll_e = torch.nan_to_num(-log_interval, nan=1e6, posinf=1e6, neginf=0.0)
        nll_e = nll_e.clamp(max=1e6)
        if sample_weight is not None:
            sw_e = sample_weight[mi].to(nll_e.dtype)
            nll_e = nll_e * sw_e
        loss = loss + nll_e.mean()
        parts["nll_event"] = float(nll_e.detach().mean().item())
        parts["nll_event_n"] = int(mi.long().sum().item())
        parts["mu_mean_event"] = float(mu_e.detach().mean().item())
        mid_e = (L_e + R_e) * 0.5
        parts["mu_minus_L_mean"] = float((mu_e - L_e).detach().mean().item())
        parts["mu_minus_mid_mean"] = float((mu_e - mid_e).detach().mean().item())

    if mr.any():
        anchor_val = float(right_anchor) if (right_anchor is not None and float(right_anchor) > 0.0) else float(Tend)
        mu_r = mu[mr]
        z_r = (mu_r - anchor_val) / sigma_t
        nll_r = -log_ndtr(z_r)
        nll_r = torch.nan_to_num(nll_r, nan=1e6, posinf=1e6, neginf=0.0).clamp(max=1e6)
        loss = loss + float(right_weight) * nll_r.mean()
        parts["nll_right"] = float(nll_r.detach().mean().item())
        parts["nll_right_n"] = int(mr.long().sum().item())
        parts["right_anchor"] = anchor_val

    if ml.any():
        R_l = R[ml].float()
        if continuity_correction:
            R_l = R_l + 0.5
        mu_l = mu[ml]
        z_l = (R_l - mu_l) / sigma_t
        nll_l = -log_ndtr(z_l)
        nll_l = torch.nan_to_num(nll_l, nan=1e6, posinf=1e6, neginf=0.0).clamp(max=1e6)
        loss = loss + nll_l.mean()
        parts["nll_left"] = float(nll_l.detach().mean().item())
        parts["nll_left_n"] = int(ml.long().sum().item())

    return loss, parts


def expected_time_location_loss(hazard_f, L_f, R_f, ctype_f, Tend: int):
    mi = ctype_f == CTYPE_INTERVAL
    if not mi.any():
        return None, {}
    pmf, _, _ = hazard_to_pmf_cdf_logS(hazard_f[mi], tstar=None)
    t = torch.arange(1, int(Tend) + 1, device=hazard_f.device, dtype=pmf.dtype).view(1, -1)
    exp_t = (pmf * t).sum(dim=1)
    true_mid = (L_f[mi].float() + R_f[mi].float()) * 0.5
    err = exp_t - true_mid
    loss = (err ** 2).mean()
    stats = {
        "loc_loss": float(loss.detach().item()),
        "loc_abs_err_mean": float(err.detach().abs().mean().item()),
        "loc_err_mean": float(err.detach().mean().item()),
        "loc_err_std": float(err.detach().std(unbiased=False).item()) if err.numel() > 1 else 0.0,
        "loc_n": int(err.numel()),
    }
    return loss, stats


def expected_time_location_loss_with_leads(hazard_f, L_f, R_f, ctype_f, tstar_f, Tend: int):
    loss, stats = expected_time_location_loss(hazard_f, L_f, R_f, ctype_f, Tend=Tend)
    mi = ctype_f == CTYPE_INTERVAL
    if loss is None or not mi.any():
        return loss, stats

    with torch.no_grad():
        pmf, _, _ = hazard_to_pmf_cdf_logS(hazard_f[mi], tstar=None)
        t = torch.arange(1, int(Tend) + 1, device=hazard_f.device, dtype=pmf.dtype).view(1, -1)
        exp_t = (pmf * t).sum(dim=1)
        true_mid = (L_f[mi].float() + R_f[mi].float()) * 0.5
        abs_err = (exp_t - true_mid).abs()
        lead = (L_f[mi].float() + 1.0) - tstar_f[mi].float()
        bins = {
            "1_14": (lead >= 1.0) & (lead <= 14.0),
            "15_29": (lead >= 15.0) & (lead <= 29.0),
            "30_60": (lead >= 30.0) & (lead <= 60.0),
            "61_75": (lead >= 61.0) & (lead <= 75.0),
            "gt75": lead > 75.0,
        }
        for name, mask in bins.items():
            key = f"loc_abs_err_{name}"
            nkey = f"loc_n_{name}"
            if mask.any():
                stats[key] = float(abs_err[mask].mean().item())
                stats[nkey] = int(mask.long().sum().item())
            else:
                stats[key] = float("nan")
                stats[nkey] = 0
    return loss, stats


def _grouped_weighted_base_loss(
    nll_vec,
    L_f,
    ctype_f,
    tstar_f,
    group_idx_f,
    B,
    *,
    Tend,
    early_tstar_weight_min: float,
    site_year_mean_loss: bool,
    lead_weighting: bool = False,
    target_lead_min: int = 30,
    target_lead_max: int = 60,
    support_lead_min: int = 15,
    support_lead_max: int = 75,
    lead_weight_min: float = 0.2,
    lead_loss_mode: str = "none",
    lead_loss_min: int = 15,
    lead_loss_max: int = 75,
    lead_loss_mid_min: int = 30,
    lead_loss_mid_max: int = 60,
    lead_loss_late_exclude_days: int = 14,
    lead_loss_weight_1_14: float = 0.0,
    lead_loss_weight_15_29: float = 0.7,
    lead_loss_weight_30_60: float = 1.5,
    lead_loss_weight_61_75: float = 1.0,
    lead_loss_weight_gt75: float = 0.25,
):
    cw = _class_weights_from_ctype(ctype_f)
    tw = _early_tstar_weights(tstar_f, Tend=Tend, min_weight=float(early_tstar_weight_min))
    lw = _lead_window_weights(
        L_f,
        ctype_f,
        tstar_f,
        enabled=bool(lead_weighting),
        target_lead_min=int(target_lead_min),
        target_lead_max=int(target_lead_max),
        support_lead_min=int(support_lead_min),
        support_lead_max=int(support_lead_max),
        min_weight=float(lead_weight_min),
    )
    llw = _lead_loss_mode_weights(
        L_f,
        ctype_f,
        tstar_f,
        mode=str(lead_loss_mode),
        lead_min=int(lead_loss_min),
        lead_max=int(lead_loss_max),
        mid_lead_min=int(lead_loss_mid_min),
        mid_lead_max=int(lead_loss_mid_max),
        late_exclude_days=int(lead_loss_late_exclude_days),
        weight_1_14=float(lead_loss_weight_1_14),
        weight_15_29=float(lead_loss_weight_15_29),
        weight_30_60=float(lead_loss_weight_30_60),
        weight_61_75=float(lead_loss_weight_61_75),
        weight_gt75=float(lead_loss_weight_gt75),
    )
    w = cw * tw * lw * llw

    if site_year_mean_loss:
        numer = torch.zeros(B, device=nll_vec.device, dtype=nll_vec.dtype)
        denom = torch.zeros(B, device=nll_vec.device, dtype=nll_vec.dtype)
        numer.index_add_(0, group_idx_f, w * nll_vec)
        denom.index_add_(0, group_idx_f, w)
        mg = denom > 0
        if mg.any():
            return (numer[mg] / denom[mg]).mean()
        return torch.tensor(0.0, device=nll_vec.device, dtype=nll_vec.dtype)

    # keep baseline-compatible behavior: mean(w * nll), not normalized by mean(w)
    return (w * nll_vec).mean()


def _flatten_grouped_valid(hazard, L, R, ctype, valid_mask):
    """
    Convert grouped tensors to flat per-t* tensors using valid_mask.
    hazard: (B,K,T), L/R/ctype/valid_mask: (B,K)
    returns:
      hazard_f: (N,T), L_f/R_f/ctype_f: (N,), n_valid
    """
    B, K, T = hazard.shape
    assert L.shape == (B, K)
    assert R.shape == (B, K)
    assert ctype.shape == (B, K)
    assert valid_mask.shape == (B, K)
    m = valid_mask.reshape(-1)
    n_valid = int(m.long().sum().item())
    if n_valid <= 0:
        return None, None, None, None, 0
    hazard_f = hazard.reshape(B * K, T)[m]
    L_f = L.reshape(B * K)[m]
    R_f = R.reshape(B * K)[m]
    ctype_f = ctype.reshape(B * K)[m]
    return hazard_f, L_f, R_f, ctype_f, n_valid


def _resolve_autocast_dtype(amp_dtype: str):
    s = str(amp_dtype).lower()
    if s in ("bf16", "bfloat16"):
        return torch.bfloat16
    if s in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unknown amp_dtype: {amp_dtype}. expected one of: bf16, fp16")


def _autocast_ctx(device: torch.device, use_amp: bool, amp_dtype: str):
    if not use_amp or device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=_resolve_autocast_dtype(amp_dtype))


def run_epoch_weighted_grouped(
    model,
    opt,
    loader,
    Tend,
    device,
    train=True,
    lambda_mass: float = 0.0,
    lambda_right_late: float = 0.0,
    right_late_tau: float | None = None,
    early_tstar_weight_min: float = 1.0,
    site_year_mean_loss: bool = False,
    lead_weighting: bool = False,
    target_lead_min: int = 30,
    target_lead_max: int = 60,
    support_lead_min: int = 15,
    support_lead_max: int = 75,
    lead_weight_min: float = 0.2,
    mass_lead_weighting: bool = False,
    lead_loss_mode: str = "none",
    lead_loss_min: int = 15,
    lead_loss_max: int = 75,
    lead_loss_mid_min: int = 30,
    lead_loss_mid_max: int = 60,
    lead_loss_late_exclude_days: int = 14,
    lead_loss_weight_1_14: float = 0.0,
    lead_loss_weight_15_29: float = 0.7,
    lead_loss_weight_30_60: float = 1.5,
    lead_loss_weight_61_75: float = 1.0,
    lead_loss_weight_gt75: float = 0.25,
    log_mass: bool = False,
    entropy_lambda: float = 0.0,
    entropy_conditional: bool = True,
    location_lambda: float = 0.0,
    conditional_survival: bool = True,
    epoch_idx: int | None = None,
    return_parts: bool = False,
    use_amp: bool = False,
    amp_dtype: str = "bf16",
    scaler: torch.cuda.amp.GradScaler | None = None,
    max_batches: int | None = None,
):
    model.train(train)
    total, n = 0.0, 0
    base_total = 0.0
    mass_total = 0.0
    late_total = 0.0
    entropy_total = 0.0
    location_total = 0.0
    mr_total = 0.0
    logged = False
    bad_batches = 0
    lead_count_totals = {"lead_1_14": 0, "lead_15_29": 0, "lead_30_60": 0, "lead_61_75": 0, "lead_gt75": 0}
    lead_weight_sum = 0.0
    lead_weight_n = 0

    if early_tstar_weight_min <= 0.0 or early_tstar_weight_min > 1.0:
        raise ValueError("early_tstar_weight_min must be in (0, 1]")

    for batch_idx, _batch in enumerate(loader):
        if len(_batch) == 7:
            X, L, R, ctype, tstar, valid_mask, pheno = _batch
        else:
            X, L, R, ctype, tstar, valid_mask = _batch
            pheno = None
        if max_batches is not None and batch_idx >= int(max_batches):
            break
        X = X.to(device, non_blocking=True)
        L = L.to(device, non_blocking=True)
        R = R.to(device, non_blocking=True)
        ctype = ctype.to(device, non_blocking=True)
        tstar = tstar.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        if pheno is not None:
            pheno = pheno.to(device, non_blocking=True)

        with torch.set_grad_enabled(bool(train)), _autocast_ctx(device, use_amp=use_amp, amp_dtype=amp_dtype):
            hazard = model(X, tstar=tstar, valid_mask=valid_mask, pheno=pheno)

            hazard_f, L_f, R_f, ctype_f, n_valid = _flatten_grouped_valid(hazard, L, R, ctype, valid_mask)
            if n_valid <= 0:
                continue
            if not torch.isfinite(hazard_f).all():
                bad_batches += 1
                continue
            flat_idx = torch.arange(valid_mask.numel(), device=valid_mask.device)[valid_mask.reshape(-1)]
            group_idx_f = torch.div(flat_idx, valid_mask.shape[1], rounding_mode="floor")
            tstar_f = tstar.reshape(-1)[valid_mask.reshape(-1)]

            if log_mass:
                batch_lead_counts = _lead_bucket_counts(
                    L_f,
                    ctype_f,
                    tstar_f,
                    late_exclude_days=int(lead_loss_late_exclude_days),
                    lead_min=int(lead_loss_min),
                    lead_max=int(lead_loss_max),
                    mid_lead_min=int(lead_loss_mid_min),
                    mid_lead_max=int(lead_loss_mid_max),
                )
                for key, value in batch_lead_counts.items():
                    lead_count_totals[key] += int(value)
                lead_mode_w_dbg = _lead_loss_mode_weights(
                    L_f,
                    ctype_f,
                    tstar_f,
                    mode=str(lead_loss_mode),
                    lead_min=int(lead_loss_min),
                    lead_max=int(lead_loss_max),
                    mid_lead_min=int(lead_loss_mid_min),
                    mid_lead_max=int(lead_loss_mid_max),
                    late_exclude_days=int(lead_loss_late_exclude_days),
                    weight_1_14=float(lead_loss_weight_1_14),
                    weight_15_29=float(lead_loss_weight_15_29),
                    weight_30_60=float(lead_loss_weight_30_60),
                    weight_61_75=float(lead_loss_weight_61_75),
                    weight_gt75=float(lead_loss_weight_gt75),
                )
                mi_dbg = ctype_f == CTYPE_INTERVAL
                if mi_dbg.any():
                    lead_weight_sum += float(lead_mode_w_dbg[mi_dbg].sum().item())
                    lead_weight_n += int(mi_dbg.long().sum().item())

            event_mask = ctype_f == CTYPE_INTERVAL
            if bool(conditional_survival) and event_mask.any():
                bad = event_mask & (tstar_f >= L_f)
                assert not bad.any(), f"{int(bad.long().sum().item())} event rows have tstar >= L; dataset filtering broken"

            is_gaussian = str(getattr(model, "pmf_mode", "hazard")) == "gaussian"
            mu_parts: dict = {}
            nll_tstar = tstar_f if bool(conditional_survival) else None
            if is_gaussian:
                mu_BK = getattr(model, "_last_mu_BK", None)
                if mu_BK is None:
                    raise RuntimeError("model.pmf_mode='gaussian' but _last_mu_BK is None after forward")
                mu_f = mu_BK.reshape(-1)[valid_mask.reshape(-1)]
                # Phase S5: per-sample long-lead weighting.
                # lead = (L + 1) - tstar (days). When threshold > 0 and weight != 1,
                # samples with lead >= threshold receive an upweighted gradient in
                # the event/early mu-loss terms.
                ll_thr = float(getattr(model, "long_lead_threshold", 0.0))
                ll_w = float(getattr(model, "long_lead_weight", 1.0))
                sw_f = None
                if ll_thr > 0.0 and abs(ll_w - 1.0) > 1e-12:
                    lead_f = (L_f.float() + 1.0) - tstar_f.float()
                    sw_f = torch.where(lead_f >= ll_thr,
                                       mu_f.new_tensor(ll_w),
                                       mu_f.new_tensor(1.0))
                    if bool(train) and not getattr(model, "_phase_s5_sw_logged", False):
                        with torch.no_grad():
                            mi_dbg = ctype_f == CTYPE_INTERVAL
                            if int(mi_dbg.long().sum().item()) > 0:
                                lead_mi = lead_f[mi_dbg]
                                sw_mi = sw_f[mi_dbg]
                                print(
                                    f"[phase_s5_sw] thr={ll_thr:.0f} w_long={ll_w:.2f}  "
                                    f"n_event={int(mi_dbg.long().sum().item())}  "
                                    f"n_long={int((sw_mi > 1.0).long().sum().item())}  "
                                    f"frac_long={float((sw_mi > 1.0).float().mean().item()):.3f}  "
                                    f"lead_mean={float(lead_mi.mean().item()):.2f}  "
                                    f"lead_med={float(lead_mi.median().item()):.2f}  "
                                    f"lead_max={float(lead_mi.max().item()):.2f}"
                                )
                        model._phase_s5_sw_logged = True
                # Phase B: lead_from_alert excludes pre-alert cells from the
                # asymmetric mu/L loss. Flatten the (B,K) mask the same way as
                # mu_f / L_f (valid_mask-indexed) so dims match.
                _lead_BK = getattr(model, "_last_lead_loss_mask", None)
                if _lead_BK is not None:
                    lead_loss_mask_f = _lead_BK.reshape(-1)[valid_mask.reshape(-1)]
                else:
                    lead_loss_mask_f = None
                gaussian_loss_mode = str(getattr(model, "gaussian_loss_mode", "asym_mse"))
                if gaussian_loss_mode == "interval_nll":
                    base_loss, mu_parts = gaussian_interval_nll_loss(
                        mu_f,
                        L_f,
                        R_f,
                        ctype_f,
                        sigma=float(getattr(model, "gaussian_sigma", 5.0)),
                        Tend=Tend,
                        right_weight=float(getattr(model, "right_weight", 0.3)),
                        right_anchor=float(getattr(model, "right_anchor", 0.0)) or None,
                        sample_weight=sw_f,
                        lead_loss_mask=lead_loss_mask_f,
                        continuity_correction=bool(getattr(model, "gaussian_interval_continuity_correction", False)),
                    )
                elif gaussian_loss_mode == "mixed":
                    # asym_mse keeps mu anchored (prevents t*-collapse), interval_nll
                    # softly nudges mu's PI toward (L, R]. Single right_weight applies
                    # to both losses' right-cens terms.
                    base_loss, mu_parts = asymmetric_mu_loss(
                        mu_f,
                        L_f,
                        R_f,
                        ctype_f,
                        Tend=Tend,
                        asym_weight=float(getattr(model, "asym_weight", 10.0)),
                        right_weight=float(getattr(model, "right_weight", 0.3)),
                        target_offset=float(getattr(model, "target_offset", 0.0)),
                        asym_weight_early=float(getattr(model, "asym_weight_early", 0.0)),
                        target_early_offset=float(getattr(model, "target_early_offset", 30.0)),
                        target_mode=str(getattr(model, "target_mode", "l_offset")),
                        zone_late_weight=float(getattr(model, "zone_late_weight", 0.0)),
                        zone_too_late_weight=float(getattr(model, "zone_too_late_weight", 0.0)),
                        zone_missed_weight=float(getattr(model, "zone_missed_weight", 0.0)),
                        zone_too_early_weight=float(getattr(model, "zone_too_early_weight", 0.0)),
                        zone_too_late_threshold=float(getattr(model, "zone_too_late_threshold", 15.0)),
                        zone_missed_threshold=float(getattr(model, "zone_missed_threshold", 22.0)),
                        zone_too_early_threshold=float(getattr(model, "zone_too_early_threshold", 23.0)),
                        sample_weight=sw_f,
                        right_anchor=float(getattr(model, "right_anchor", 0.0)),
                        lead_loss_mask=lead_loss_mask_f,
                        aux_lead_lambda=float(getattr(model, "aux_lead_lambda", 0.0)),
                        aux_lead_huber_delta=float(getattr(model, "aux_lead_huber_delta", 10.0)),
                    )
                    _intnll_loss, _intnll_parts = gaussian_interval_nll_loss(
                        mu_f,
                        L_f,
                        R_f,
                        ctype_f,
                        sigma=float(getattr(model, "gaussian_sigma", 5.0)),
                        Tend=Tend,
                        right_weight=float(getattr(model, "right_weight", 0.3)),
                        right_anchor=float(getattr(model, "right_anchor", 0.0)) or None,
                        sample_weight=sw_f,
                        lead_loss_mask=lead_loss_mask_f,
                        continuity_correction=bool(getattr(model, "gaussian_interval_continuity_correction", False)),
                    )
                    _lam = float(getattr(model, "gaussian_interval_lambda", 0.1))
                    base_loss = base_loss + _lam * _intnll_loss
                    for _k, _v in _intnll_parts.items():
                        mu_parts[f"intnll_{_k}"] = _v
                    mu_parts["intnll_lambda"] = _lam
                    mu_parts["intnll_loss_raw"] = float(_intnll_loss.detach().item())
                else:
                    base_loss, mu_parts = asymmetric_mu_loss(
                        mu_f,
                        L_f,
                        R_f,
                        ctype_f,
                        Tend=Tend,
                        asym_weight=float(getattr(model, "asym_weight", 10.0)),
                        right_weight=float(getattr(model, "right_weight", 0.3)),
                        target_offset=float(getattr(model, "target_offset", 0.0)),
                        asym_weight_early=float(getattr(model, "asym_weight_early", 0.0)),
                        target_early_offset=float(getattr(model, "target_early_offset", 30.0)),
                        target_mode=str(getattr(model, "target_mode", "l_offset")),
                        zone_late_weight=float(getattr(model, "zone_late_weight", 0.0)),
                        zone_too_late_weight=float(getattr(model, "zone_too_late_weight", 0.0)),
                        zone_missed_weight=float(getattr(model, "zone_missed_weight", 0.0)),
                        zone_too_early_weight=float(getattr(model, "zone_too_early_weight", 0.0)),
                        zone_too_late_threshold=float(getattr(model, "zone_too_late_threshold", 15.0)),
                        zone_missed_threshold=float(getattr(model, "zone_missed_threshold", 22.0)),
                        zone_too_early_threshold=float(getattr(model, "zone_too_early_threshold", 23.0)),
                        sample_weight=sw_f,
                        right_anchor=float(getattr(model, "right_anchor", 0.0)),
                        lead_loss_mask=lead_loss_mask_f,
                        aux_lead_lambda=float(getattr(model, "aux_lead_lambda", 0.0)),
                        aux_lead_huber_delta=float(getattr(model, "aux_lead_huber_delta", 10.0)),
                    )
            else:
                nll_vec = interval_nll_per_sample(hazard_f, L_f, R_f, ctype_f, Tend=Tend, tstar=nll_tstar)
                base_loss = _grouped_weighted_base_loss(
                    nll_vec,
                    L_f,
                    ctype_f,
                    tstar_f,
                    group_idx_f,
                    B=valid_mask.shape[0],
                    Tend=Tend,
                    early_tstar_weight_min=float(early_tstar_weight_min),
                    site_year_mean_loss=bool(site_year_mean_loss),
                    lead_weighting=bool(lead_weighting),
                    target_lead_min=int(target_lead_min),
                    target_lead_max=int(target_lead_max),
                    support_lead_min=int(support_lead_min),
                    support_lead_max=int(support_lead_max),
                    lead_weight_min=float(lead_weight_min),
                    lead_loss_mode=str(lead_loss_mode),
                    lead_loss_min=int(lead_loss_min),
                    lead_loss_max=int(lead_loss_max),
                    lead_loss_mid_min=int(lead_loss_mid_min),
                    lead_loss_mid_max=int(lead_loss_mid_max),
                    lead_loss_late_exclude_days=int(lead_loss_late_exclude_days),
                    lead_loss_weight_1_14=float(lead_loss_weight_1_14),
                    lead_loss_weight_15_29=float(lead_loss_weight_15_29),
                    lead_loss_weight_30_60=float(lead_loss_weight_30_60),
                    lead_loss_weight_61_75=float(lead_loss_weight_61_75),
                    lead_loss_weight_gt75=float(lead_loss_weight_gt75),
                )
            loss = base_loss
            mass_loss_tensor = None
            late_loss_tensor = None
            entropy_loss_tensor = None
            location_loss_tensor = None
            location_stats = {}
            mr = (ctype_f == 1)
            mr_total += float(mr.float().sum().item())

        if log_mass and not logged:
            mi = (ctype_f == CTYPE_INTERVAL)
            mr = (ctype_f == 1)
            mi_frac = float(mi.float().mean().item())
            mr_frac = float(mr.float().mean().item())
            if mi.any():
                _, _, logS = hazard_to_pmf_cdf_logS(hazard_f, tstar=nll_tstar)
                idxL = (torch.clamp(L_f, 1, Tend) - 1).long()
                idxR = (torch.clamp(R_f, 1, Tend) - 1).long()
                logS_L = logS.gather(1, idxL.view(-1, 1)).squeeze(1)
                logS_R = logS.gather(1, idxR.view(-1, 1)).squeeze(1)
                mass = (torch.exp(logS_L) - torch.exp(logS_R)).clamp(min=0.0)
                mass_mi = mass[mi]
                mass_mean = float(mass_mi.mean().item())
                mass_min = float(mass_mi.min().item())
                mass_max = float(mass_mi.max().item())
                mass_loss_tensor = -mass_mi.mean()
                mass_loss_val = float(mass_loss_tensor.item())
            else:
                mass_mean = float("nan")
                mass_min = float("nan")
                mass_max = float("nan")
                mass_loss_val = 0.0

            late_mean = float("nan")
            late_min = float("nan")
            late_max = float("nan")
            late_loss_val = 0.0
            if mr.any() and (lambda_right_late > 0) and (right_late_tau is not None):
                pmf, _, logS_l = hazard_to_pmf_cdf_logS(hazard_f, tstar=nll_tstar)
                t = torch.arange(1, Tend + 1, device=hazard_f.device, dtype=pmf.dtype).view(1, -1)
                exp_doy = (pmf * t).sum(dim=1) + torch.exp(logS_l[:, -1]) * float(Tend)
                late_margin = torch.relu(exp_doy[mr] - float(right_late_tau))
                if late_margin.numel() > 0:
                    late_mean = float(late_margin.mean().item())
                    late_min = float(late_margin.min().item())
                    late_max = float(late_margin.max().item())
                    late_loss_val = float(late_mean)
            prefix = f"[mass] epoch {epoch_idx:02d} " if epoch_idx is not None else "[mass] "
            print(
                prefix
                + f"mi_frac={mi_frac:.4f} mr_frac={mr_frac:.4f} mass_mean={mass_mean:.6f} "
                + f"mass_min={mass_min:.6f} mass_max={mass_max:.6f} "
                + f"mass_loss={mass_loss_val:.6f} lambda_mass={lambda_mass} "
                + f"early_tstar_weight_min={float(early_tstar_weight_min):.3f} "
                + f"site_year_mean_loss={int(bool(site_year_mean_loss))}"
            )
            print(
                prefix
                + f"late_loss_mean={late_loss_val:.6f} late_min={late_min:.6f} late_max={late_max:.6f} "
                + f"lambda_right_late={lambda_right_late} right_late_tau={right_late_tau}"
            )
            lead_counts = _lead_bucket_counts(
                L_f,
                ctype_f,
                tstar_f,
                late_exclude_days=int(lead_loss_late_exclude_days),
                lead_min=int(lead_loss_min),
                lead_max=int(lead_loss_max),
                mid_lead_min=int(lead_loss_mid_min),
                mid_lead_max=int(lead_loss_mid_max),
            )
            lead_mode_w = _lead_loss_mode_weights(
                L_f,
                ctype_f,
                tstar_f,
                mode=str(lead_loss_mode),
                lead_min=int(lead_loss_min),
                lead_max=int(lead_loss_max),
                mid_lead_min=int(lead_loss_mid_min),
                mid_lead_max=int(lead_loss_mid_max),
                late_exclude_days=int(lead_loss_late_exclude_days),
                weight_1_14=float(lead_loss_weight_1_14),
                weight_15_29=float(lead_loss_weight_15_29),
                weight_30_60=float(lead_loss_weight_30_60),
                weight_61_75=float(lead_loss_weight_61_75),
                weight_gt75=float(lead_loss_weight_gt75),
            )
            mi_for_weight = ctype_f == CTYPE_INTERVAL
            lead_weight_mean = float(lead_mode_w[mi_for_weight].mean().item()) if mi_for_weight.any() else float("nan")
            print(
                prefix
                + f"lead_loss_mode={lead_loss_mode} "
                + f"lead_counts={lead_counts} lead_loss_weight_mean={lead_weight_mean:.6f}"
            )
            entropy_dbg = pmf_entropy(hazard_f, tstar_f, Tend=Tend, conditional=bool(entropy_conditional))
            entropy_mean_dbg = float(entropy_dbg.mean().item()) if entropy_dbg.numel() else float("nan")
            print(
                prefix
                + f"entropy_mean={entropy_mean_dbg:.6f} "
                + f"entropy_lambda={float(entropy_lambda):.6f} "
                + f"lambda_entropy={float(entropy_lambda) * entropy_mean_dbg:.6f} "
                + f"entropy_conditional={int(bool(entropy_conditional))}"
            )
            loc_dbg, loc_stats_dbg = expected_time_location_loss_with_leads(hazard_f, L_f, R_f, ctype_f, tstar_f, Tend=Tend)
            loc_mean_dbg = float(loc_stats_dbg.get("loc_loss", float("nan")))
            print(
                prefix
                + f"location_loss={loc_mean_dbg:.6f} "
                + f"lambda_location={float(location_lambda):.6f} "
                + f"lambda_location_term={float(location_lambda) * loc_mean_dbg:.6f} "
                + f"loc_abs_err_mean={float(loc_stats_dbg.get('loc_abs_err_mean', float('nan'))):.6f} "
                + f"loc_err_mean={float(loc_stats_dbg.get('loc_err_mean', float('nan'))):.6f} "
                + f"loc_err_std={float(loc_stats_dbg.get('loc_err_std', float('nan'))):.6f} "
                + f"loc_n={int(loc_stats_dbg.get('loc_n', 0))}"
            )
            print(
                prefix
                + "location_abs_err_by_lead="
                + "{"
                + f"'1_14': {float(loc_stats_dbg.get('loc_abs_err_1_14', float('nan'))):.3f} (n={int(loc_stats_dbg.get('loc_n_1_14', 0))}), "
                + f"'15_29': {float(loc_stats_dbg.get('loc_abs_err_15_29', float('nan'))):.3f} (n={int(loc_stats_dbg.get('loc_n_15_29', 0))}), "
                + f"'30_60': {float(loc_stats_dbg.get('loc_abs_err_30_60', float('nan'))):.3f} (n={int(loc_stats_dbg.get('loc_n_30_60', 0))}), "
                + f"'61_75': {float(loc_stats_dbg.get('loc_abs_err_61_75', float('nan'))):.3f} (n={int(loc_stats_dbg.get('loc_n_61_75', 0))}), "
                + f"'gt75': {float(loc_stats_dbg.get('loc_abs_err_gt75', float('nan'))):.3f} (n={int(loc_stats_dbg.get('loc_n_gt75', 0))})"
                + "}"
            )
            if is_gaussian and mu_parts:
                pb = getattr(model, "_last_phen_bias", None)
                mut = getattr(model, "_last_mu_temporal", None)
                pb_mean = float(pb.float().mean().item()) if (pb is not None) else float("nan")
                pb_std = float(pb.float().std(unbiased=False).item()) if (pb is not None and pb.numel() > 1) else float("nan")
                mut_mean = float(mut.float().mean().item()) if (mut is not None) else float("nan")
                print(
                    prefix
                    + f"phenobias_head={int(bool(getattr(model, 'phenology_bias_head', False)))} "
                    + f"phen_bias_mean={pb_mean:.3f} phen_bias_std={pb_std:.3f} "
                    + f"mu_temporal_mean={mut_mean:.3f}"
                )
                print(
                    prefix
                    + f"pmf_mode=gaussian sigma={float(getattr(model, 'gaussian_sigma', 5.0)):.3f} "
                    + f"target_mode={str(mu_parts.get('target_mode', 'l_offset'))} "
                    + f"asym_w={float(getattr(model, 'asym_weight', 10.0)):.2f} "
                    + f"asym_w_early={float(getattr(model, 'asym_weight_early', 0.0)):.2f} "
                    + f"right_w={float(getattr(model, 'right_weight', 0.3)):.2f} "
                    + f"target_offset={float(mu_parts.get('target_offset', 0.0)):.2f} "
                    + f"target_early_offset={float(mu_parts.get('target_early_offset', 30.0)):.2f} "
                    + f"zone_thr_tl={float(mu_parts.get('zone_too_late_threshold', 15.0)):.1f} "
                    + f"zone_thr_m={float(mu_parts.get('zone_missed_threshold', 22.0)):.1f} "
                    + f"zone_thr_te={float(mu_parts.get('zone_too_early_threshold', 23.0)):.1f} "
                    + f"mu_event_loss={float(mu_parts.get('mu_event', float('nan'))):.4f} "
                    + f"mu_event_n={int(mu_parts.get('mu_event_n', 0))} "
                    + f"mu_early_loss={float(mu_parts.get('mu_early', 0.0)):.4f} "
                    + f"mu_early_n={int(mu_parts.get('mu_early_n', 0))} "
                    + f"mu_late_loss={float(mu_parts.get('mu_late', 0.0)):.4f} "
                    + f"mu_late_n={int(mu_parts.get('mu_late_n', 0))} "
                    + f"mu_too_late_loss={float(mu_parts.get('mu_too_late', 0.0)):.4f} "
                    + f"mu_too_late_n={int(mu_parts.get('mu_too_late_n', 0))} "
                    + f"mu_missed_loss={float(mu_parts.get('mu_missed', 0.0)):.4f} "
                    + f"mu_missed_n={int(mu_parts.get('mu_missed_n', 0))} "
                    + f"mu_too_early_loss={float(mu_parts.get('mu_too_early', 0.0)):.4f} "
                    + f"mu_too_early_n={int(mu_parts.get('mu_too_early_n', 0))} "
                    + f"mu_right_loss={float(mu_parts.get('mu_right', float('nan'))):.4f} "
                    + f"mu_right_n={int(mu_parts.get('mu_right_n', 0))} "
                    + f"mu_mean_event={float(mu_parts.get('mu_mean_event', float('nan'))):.3f} "
                    + f"mu_minus_L_mean={float(mu_parts.get('mu_minus_L_mean', float('nan'))):.3f} "
                    + f"mu_minus_L_abs_mean={float(mu_parts.get('mu_minus_L_abs_mean', float('nan'))):.3f} "
                    + f"mu_minus_mid_mean={float(mu_parts.get('mu_minus_mid_mean', float('nan'))):.3f} "
                    + f"mu_minus_mid_abs_mean={float(mu_parts.get('mu_minus_mid_abs_mean', float('nan'))):.3f} "
                    + f"mu_minus_target_mean={float(mu_parts.get('mu_minus_target_mean', float('nan'))):.3f} "
                    + f"mu_pos_frac={float(mu_parts.get('mu_pos_frac', float('nan'))):.3f}"
                )
            logged = True

        if (not is_gaussian) and lambda_mass > 0:
            mi = (ctype_f == CTYPE_INTERVAL)
            if mi.any():
                _, _, logS = hazard_to_pmf_cdf_logS(hazard_f, tstar=nll_tstar)
                idxL = (torch.clamp(L_f, 1, Tend) - 1).long()
                idxR = (torch.clamp(R_f, 1, Tend) - 1).long()
                logS_L = logS.gather(1, idxL.view(-1, 1)).squeeze(1)
                logS_R = logS.gather(1, idxR.view(-1, 1)).squeeze(1)
                mass = (torch.exp(logS_L) - torch.exp(logS_R)).clamp(min=0.0)
                if bool(mass_lead_weighting):
                    mw = _lead_window_weights(
                        L_f,
                        ctype_f,
                        tstar_f,
                        enabled=True,
                        target_lead_min=int(target_lead_min),
                        target_lead_max=int(target_lead_max),
                        support_lead_min=int(support_lead_min),
                        support_lead_max=int(support_lead_max),
                        min_weight=float(lead_weight_min),
                    )
                    numer = (mass[mi] * mw[mi]).sum()
                    denom = mw[mi].sum().clamp_min(1e-8)
                    mass_loss_tensor = -(numer / denom)
                else:
                    mass_loss_tensor = -mass[mi].mean()
                loss = loss + lambda_mass * mass_loss_tensor

        if (not is_gaussian) and lambda_right_late > 0 and right_late_tau is not None:
            mr = (ctype_f == 1)
            if mr.any():
                pmf, _, logS = hazard_to_pmf_cdf_logS(hazard_f, tstar=nll_tstar)
                t = torch.arange(1, Tend + 1, device=hazard_f.device, dtype=pmf.dtype).view(1, -1)
                exp_doy = (pmf * t).sum(dim=1) + torch.exp(logS[:, -1]) * float(Tend)
                late_margin = torch.relu(exp_doy[mr] - float(right_late_tau))
                late_loss_tensor = late_margin.mean()
                loss = loss + lambda_right_late * late_loss_tensor

        if (not is_gaussian) and float(entropy_lambda) > 0.0:
            entropy_vec = pmf_entropy(hazard_f, tstar_f, Tend=Tend, conditional=bool(entropy_conditional))
            entropy_loss_tensor = entropy_vec.mean()
            loss = loss + float(entropy_lambda) * entropy_loss_tensor

        if (not is_gaussian) and float(location_lambda) > 0.0:
            location_loss_tensor, location_stats = expected_time_location_loss(hazard_f, L_f, R_f, ctype_f, Tend=Tend)
            if location_loss_tensor is not None:
                loss = loss + float(location_lambda) * location_loss_tensor

        if not torch.isfinite(loss):
            bad_batches += 1
            continue

        if train:
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            grad_finite = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grad_finite = False
                    break
            if not grad_finite:
                bad_batches += 1
                opt.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

        total += float(loss.item()) * n_valid
        base_total += float(base_loss.item()) * n_valid
        if mass_loss_tensor is not None:
            mass_total += float(mass_loss_tensor.item()) * n_valid
        if late_loss_tensor is not None:
            late_total += float(late_loss_tensor.item()) * n_valid
        if entropy_loss_tensor is not None:
            entropy_total += float(entropy_loss_tensor.item()) * n_valid
        if location_loss_tensor is not None:
            location_total += float(location_loss_tensor.item()) * n_valid
        n += n_valid

    if bad_batches > 0 and train:
        print(f"[warn] skipped non-finite batches: {bad_batches}")
    if log_mass:
        prefix = f"[lead_loss] epoch {epoch_idx:02d} " if epoch_idx is not None else "[lead_loss] "
        lead_weight_mean_epoch = lead_weight_sum / lead_weight_n if lead_weight_n > 0 else float("nan")
        print(
            prefix
            + f"mode={lead_loss_mode} lead_counts_epoch={lead_count_totals} "
            + f"lead_loss_weight_mean_epoch={lead_weight_mean_epoch:.6f}"
        )

    total_avg = total / n if n > 0 else float("nan")
    if return_parts:
        base_avg = base_total / max(n, 1)
        mass_avg = mass_total / max(n, 1)
        late_avg = late_total / max(n, 1)
        entropy_avg = entropy_total / max(n, 1)
        location_avg = location_total / max(n, 1)
        right_frac = mr_total / max(n, 1)
        return total_avg, base_avg, mass_avg, late_avg, entropy_avg, location_avg, right_frac
    return total_avg


def run_epoch_weighted(
    model,
    opt,
    loader,
    Tend,
    device,
    train=True,
    lambda_mass: float = 0.0,
    lambda_right_late: float = 0.0,
    right_late_tau: float | None = None,
    log_mass: bool = False,
    epoch_idx: int | None = None,
    return_parts: bool = False,
    use_amp: bool = False,
    amp_dtype: str = "bf16",
    scaler: torch.cuda.amp.GradScaler | None = None,
):
    model.train(train)
    total, n = 0.0, 0
    base_total = 0.0
    mass_total = 0.0
    late_total = 0.0
    mr_total = 0.0
    logged = False
    bad_batches = 0
    for X, L, R, ctype in loader:
        X = X.to(device, non_blocking=True)
        L = L.to(device, non_blocking=True)
        R = R.to(device, non_blocking=True)
        ctype = ctype.to(device, non_blocking=True)

        with _autocast_ctx(device, use_amp=use_amp, amp_dtype=amp_dtype):
            hazard = model(X)
            if not torch.isfinite(hazard).all():
                bad_batches += 1
                continue
            nll_vec = interval_nll_per_sample(hazard, L, R, ctype, Tend=Tend)
            base_loss = weighted_loss_from_ctype(nll_vec, ctype)
            loss = base_loss
            mass_loss_tensor = None
            late_loss_tensor = None
            mr = (ctype == 1)
            mr_total += float(mr.float().sum().item())

        if train and log_mass and not logged:
            mi = (ctype == CTYPE_INTERVAL)
            mr = (ctype == 1)
            mi_frac = float(mi.float().mean().item())
            mr_frac = float(mr.float().mean().item())
            if mi.any():
                _, _, logS = hazard_to_pmf_cdf_logS(hazard)
                idxL = (torch.clamp(L, 1, Tend) - 1).long()
                idxR = (torch.clamp(R, 1, Tend) - 1).long()
                logS_L = logS.gather(1, idxL.view(-1, 1)).squeeze(1)
                logS_R = logS.gather(1, idxR.view(-1, 1)).squeeze(1)
                mass = (torch.exp(logS_L) - torch.exp(logS_R)).clamp(min=0.0)
                mass_mi = mass[mi]
                mass_mean = float(mass_mi.mean().item())
                mass_min = float(mass_mi.min().item())
                mass_max = float(mass_mi.max().item())
                mass_loss_tensor = -mass_mi.mean()
                mass_loss_val = float(mass_loss_tensor.item())
                mass_loss_requires_grad = mass_loss_tensor.requires_grad
                mass_loss_grad_fn = str(mass_loss_tensor.grad_fn)
                logS_requires_grad = logS.requires_grad
            else:
                mass_mean = float("nan")
                mass_min = float("nan")
                mass_max = float("nan")
                mass_loss_val = 0.0
                mass_loss_requires_grad = False
                mass_loss_grad_fn = "None"
                logS_requires_grad = False
            # late-loss diagnostics (right-censored only)
            late_mean = float("nan")
            late_min = float("nan")
            late_max = float("nan")
            late_loss_val = 0.0
            if mr.any() and (lambda_right_late > 0) and (right_late_tau is not None):
                pmf, _, logS_l = hazard_to_pmf_cdf_logS(hazard)
                t = torch.arange(1, Tend + 1, device=hazard.device, dtype=pmf.dtype).view(1, -1)
                exp_doy = (pmf * t).sum(dim=1) + torch.exp(logS_l[:, -1]) * float(Tend)
                late_margin = torch.relu(exp_doy[mr] - float(right_late_tau))
                if late_margin.numel() > 0:
                    late_mean = float(late_margin.mean().item())
                    late_min = float(late_margin.min().item())
                    late_max = float(late_margin.max().item())
                    late_loss_val = float(late_mean)
            prefix = f"[mass] epoch {epoch_idx:02d} " if epoch_idx is not None else "[mass] "
            print(
                prefix
                + f"mi_frac={mi_frac:.4f} mr_frac={mr_frac:.4f} mass_mean={mass_mean:.6f} "
                + f"mass_min={mass_min:.6f} mass_max={mass_max:.6f} "
                + f"mass_loss={mass_loss_val:.6f} lambda_mass={lambda_mass} "
                + f"mass_loss_requires_grad={mass_loss_requires_grad} "
                + f"mass_loss_grad_fn={mass_loss_grad_fn}"
            )
            print(
                prefix
                + f"late_loss_mean={late_loss_val:.6f} late_min={late_min:.6f} late_max={late_max:.6f} "
                + f"lambda_right_late={lambda_right_late} right_late_tau={right_late_tau}"
            )
            if epoch_idx == 1:
                print(
                    prefix
                    + f"logS_requires_grad={logS_requires_grad} "
                    + f"mass_loss_requires_grad={mass_loss_requires_grad} "
                    + f"mass_loss_grad_fn={mass_loss_grad_fn}"
                )
            logged = True

        if lambda_mass > 0:
            mi = (ctype == CTYPE_INTERVAL)
            if mi.any():
                _, _, logS = hazard_to_pmf_cdf_logS(hazard)
                idxL = (torch.clamp(L, 1, Tend) - 1).long()
                idxR = (torch.clamp(R, 1, Tend) - 1).long()
                logS_L = logS.gather(1, idxL.view(-1, 1)).squeeze(1)
                logS_R = logS.gather(1, idxR.view(-1, 1)).squeeze(1)
                mass = (torch.exp(logS_L) - torch.exp(logS_R)).clamp(min=0.0)
                mass_loss_tensor = -mass[mi].mean()
                loss = loss + lambda_mass * mass_loss_tensor

        if lambda_right_late > 0 and right_late_tau is not None:
            mr = (ctype == 1)
            if mr.any():
                pmf, _, logS = hazard_to_pmf_cdf_logS(hazard)
                t = torch.arange(1, Tend + 1, device=hazard.device, dtype=pmf.dtype).view(1, -1)
                exp_doy = (pmf * t).sum(dim=1) + torch.exp(logS[:, -1]) * float(Tend)
                late_margin = torch.relu(exp_doy[mr] - float(right_late_tau))
                late_loss_tensor = late_margin.mean()
                loss = loss + lambda_right_late * late_loss_tensor

        if not torch.isfinite(loss):
            bad_batches += 1
            continue

        if train:
            opt.zero_grad(set_to_none=True)
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
            else:
                loss.backward()
            grad_finite = True
            for p in model.parameters():
                if p.grad is not None and not torch.isfinite(p.grad).all():
                    grad_finite = False
                    break
            if not grad_finite:
                bad_batches += 1
                opt.zero_grad(set_to_none=True)
                continue
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
            if scaler is not None:
                scaler.step(opt)
                scaler.update()
            else:
                opt.step()

        total += float(loss.item()) * X.size(0)
        base_total += float(base_loss.item()) * X.size(0)
        if mass_loss_tensor is not None:
            mass_total += float(mass_loss_tensor.item()) * X.size(0)
        if late_loss_tensor is not None:
            late_total += float(late_loss_tensor.item()) * X.size(0)
        n += X.size(0)

    if bad_batches > 0 and train:
        print(f"[warn] skipped non-finite batches: {bad_batches}")

    if n == 0:
        total_avg = float("nan")
    else:
        total_avg = total / n
    if return_parts:
        base_avg = base_total / max(n, 1)
        mass_avg = mass_total / max(n, 1)
        late_avg = late_total / max(n, 1)
        right_frac = mr_total / max(n, 1)
        return total_avg, base_avg, mass_avg, late_avg, right_frac
    return total_avg


@torch.no_grad()
def eval_nll_model(model, loader, Tend, device):
    model.eval()
    total, n = 0.0, 0
    for X, L, R, ctype in loader:
        X = X.to(device, non_blocking=True)
        L = L.to(device, non_blocking=True)
        R = R.to(device, non_blocking=True)
        ctype = ctype.to(device, non_blocking=True)

        with _autocast_ctx(device, use_amp=bool(getattr(model, "use_amp_eval", False)), amp_dtype=str(getattr(model, "amp_dtype_eval", "bf16"))):
            hazard = model(X)
            nll_vec = interval_nll_per_sample(hazard, L, R, ctype, Tend=Tend)
            loss = weighted_loss_from_ctype(nll_vec, ctype)

        total += float(loss.item()) * X.size(0)
        n += X.size(0)

    return total / max(n, 1)


@torch.no_grad()
def eval_nll_model_grouped(model, loader, Tend, device):
    model.eval()
    total, n = 0.0, 0
    early_tstar_weight_min = float(getattr(model, "early_tstar_weight_min", 1.0))
    site_year_mean_loss = bool(getattr(model, "site_year_mean_loss", False))
    conditional_survival = bool(getattr(model, "conditional_survival", True))
    is_gaussian = str(getattr(model, "pmf_mode", "hazard")) == "gaussian"
    for _batch in loader:
        if len(_batch) == 7:
            X, L, R, ctype, tstar, valid_mask, pheno = _batch
        else:
            X, L, R, ctype, tstar, valid_mask = _batch
            pheno = None
        X = X.to(device, non_blocking=True)
        L = L.to(device, non_blocking=True)
        R = R.to(device, non_blocking=True)
        ctype = ctype.to(device, non_blocking=True)
        tstar = tstar.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        if pheno is not None:
            pheno = pheno.to(device, non_blocking=True)

        with _autocast_ctx(device, use_amp=bool(getattr(model, "use_amp_eval", False)), amp_dtype=str(getattr(model, "amp_dtype_eval", "bf16"))):
            hazard = model(X, tstar=tstar, valid_mask=valid_mask, pheno=pheno)
            hazard_f, L_f, R_f, ctype_f, n_valid = _flatten_grouped_valid(hazard, L, R, ctype, valid_mask)
            if n_valid <= 0:
                continue
            flat_idx = torch.arange(valid_mask.numel(), device=valid_mask.device)[valid_mask.reshape(-1)]
            group_idx_f = torch.div(flat_idx, valid_mask.shape[1], rounding_mode="floor")
            tstar_f = tstar.reshape(-1)[valid_mask.reshape(-1)]
            event_mask = ctype_f == CTYPE_INTERVAL
            if conditional_survival and event_mask.any():
                bad = event_mask & (tstar_f >= L_f)
                assert not bad.any(), f"{int(bad.long().sum().item())} event rows have tstar >= L; dataset filtering broken"
            if is_gaussian:
                mu_BK = getattr(model, "_last_mu_BK", None)
                if mu_BK is None:
                    raise RuntimeError("model.pmf_mode='gaussian' but _last_mu_BK is None after forward")
                mu_f = mu_BK.reshape(-1)[valid_mask.reshape(-1)]
                _lead_BK = getattr(model, "_last_lead_loss_mask", None)
                if _lead_BK is not None:
                    lead_loss_mask_f = _lead_BK.reshape(-1)[valid_mask.reshape(-1)]
                else:
                    lead_loss_mask_f = None
                gaussian_loss_mode = str(getattr(model, "gaussian_loss_mode", "asym_mse"))
                if gaussian_loss_mode == "interval_nll":
                    loss, _ = gaussian_interval_nll_loss(
                        mu_f,
                        L_f,
                        R_f,
                        ctype_f,
                        sigma=float(getattr(model, "gaussian_sigma", 5.0)),
                        Tend=Tend,
                        right_weight=float(getattr(model, "right_weight", 0.3)),
                        right_anchor=float(getattr(model, "right_anchor", 0.0)) or None,
                        sample_weight=None,
                        lead_loss_mask=lead_loss_mask_f,
                        continuity_correction=bool(getattr(model, "gaussian_interval_continuity_correction", False)),
                    )
                elif gaussian_loss_mode == "mixed":
                    loss, _ = asymmetric_mu_loss(
                        mu_f,
                        L_f,
                        R_f,
                        ctype_f,
                        Tend=Tend,
                        asym_weight=float(getattr(model, "asym_weight", 10.0)),
                        right_weight=float(getattr(model, "right_weight", 0.3)),
                        target_offset=float(getattr(model, "target_offset", 0.0)),
                        asym_weight_early=float(getattr(model, "asym_weight_early", 0.0)),
                        target_early_offset=float(getattr(model, "target_early_offset", 30.0)),
                        target_mode=str(getattr(model, "target_mode", "l_offset")),
                        zone_late_weight=float(getattr(model, "zone_late_weight", 0.0)),
                        zone_too_late_weight=float(getattr(model, "zone_too_late_weight", 0.0)),
                        zone_missed_weight=float(getattr(model, "zone_missed_weight", 0.0)),
                        zone_too_early_weight=float(getattr(model, "zone_too_early_weight", 0.0)),
                        zone_too_late_threshold=float(getattr(model, "zone_too_late_threshold", 15.0)),
                        zone_missed_threshold=float(getattr(model, "zone_missed_threshold", 22.0)),
                        zone_too_early_threshold=float(getattr(model, "zone_too_early_threshold", 23.0)),
                        right_anchor=float(getattr(model, "right_anchor", 0.0)),
                        lead_loss_mask=lead_loss_mask_f,
                        aux_lead_lambda=float(getattr(model, "aux_lead_lambda", 0.0)),
                        aux_lead_huber_delta=float(getattr(model, "aux_lead_huber_delta", 10.0)),
                    )
                    _intnll_loss_eval, _ = gaussian_interval_nll_loss(
                        mu_f,
                        L_f,
                        R_f,
                        ctype_f,
                        sigma=float(getattr(model, "gaussian_sigma", 5.0)),
                        Tend=Tend,
                        right_weight=float(getattr(model, "right_weight", 0.3)),
                        right_anchor=float(getattr(model, "right_anchor", 0.0)) or None,
                        sample_weight=None,
                        lead_loss_mask=lead_loss_mask_f,
                        continuity_correction=bool(getattr(model, "gaussian_interval_continuity_correction", False)),
                    )
                    _lam_eval = float(getattr(model, "gaussian_interval_lambda", 0.1))
                    loss = loss + _lam_eval * _intnll_loss_eval
                else:
                    loss, _ = asymmetric_mu_loss(
                        mu_f,
                        L_f,
                        R_f,
                        ctype_f,
                        Tend=Tend,
                        asym_weight=float(getattr(model, "asym_weight", 10.0)),
                        right_weight=float(getattr(model, "right_weight", 0.3)),
                        target_offset=float(getattr(model, "target_offset", 0.0)),
                        asym_weight_early=float(getattr(model, "asym_weight_early", 0.0)),
                        target_early_offset=float(getattr(model, "target_early_offset", 30.0)),
                        target_mode=str(getattr(model, "target_mode", "l_offset")),
                        zone_late_weight=float(getattr(model, "zone_late_weight", 0.0)),
                        zone_too_late_weight=float(getattr(model, "zone_too_late_weight", 0.0)),
                        zone_missed_weight=float(getattr(model, "zone_missed_weight", 0.0)),
                        zone_too_early_weight=float(getattr(model, "zone_too_early_weight", 0.0)),
                        zone_too_late_threshold=float(getattr(model, "zone_too_late_threshold", 15.0)),
                        zone_missed_threshold=float(getattr(model, "zone_missed_threshold", 22.0)),
                        zone_too_early_threshold=float(getattr(model, "zone_too_early_threshold", 23.0)),
                        right_anchor=float(getattr(model, "right_anchor", 0.0)),
                        lead_loss_mask=lead_loss_mask_f,
                        aux_lead_lambda=float(getattr(model, "aux_lead_lambda", 0.0)),
                        aux_lead_huber_delta=float(getattr(model, "aux_lead_huber_delta", 10.0)),
                    )
            else:
                nll_vec = interval_nll_per_sample(
                    hazard_f,
                    L_f,
                    R_f,
                    ctype_f,
                    Tend=Tend,
                    tstar=tstar_f if conditional_survival else None,
                )
                loss = _grouped_weighted_base_loss(
                    nll_vec,
                    L_f,
                    ctype_f,
                    tstar_f,
                    group_idx_f,
                    B=valid_mask.shape[0],
                    Tend=Tend,
                    early_tstar_weight_min=early_tstar_weight_min,
                    site_year_mean_loss=site_year_mean_loss,
                )

        total += float(loss.item()) * n_valid
        n += n_valid
    return total / max(n, 1)


# -------------------------
# Metrics
# -------------------------
def hazard_to_pmf_cdf_logS(hazard, tstar=None):
    B, T = hazard.shape
    logS_full = torch.cumsum(torch.log1p(-hazard), dim=1)  # log S_t
    if tstar is None:
        logS = logS_full
        S_prev = torch.cat([torch.ones(B, 1, device=hazard.device), torch.exp(logS[:, :-1])], dim=1)
    else:
        idx_tstar = (tstar.long().clamp(min=1, max=T) - 1).view(-1, 1)
        logS_at_tstar = logS_full.gather(1, idx_tstar).squeeze(1)
        logS = logS_full - logS_at_tstar.view(-1, 1)
        S_prev_full = torch.cat([torch.ones(B, 1, device=hazard.device), torch.exp(logS_full[:, :-1])], dim=1)
        S_prev = S_prev_full / torch.exp(logS_at_tstar).view(-1, 1).clamp_min(1e-12)
    pmf = S_prev * hazard
    if tstar is not None:
        idx = (tstar.long().clamp(min=1, max=T) - 1).view(-1, 1)
        time_idx = torch.arange(T, device=hazard.device).view(1, -1)
        pmf = torch.where(time_idx <= idx, torch.zeros_like(pmf), pmf)
    cdf = torch.cumsum(pmf, dim=1).clamp(0, 1)
    return pmf, cdf, logS


def quantile_from_cdf_1d(cdf_1d, q, Tend):
    if cdf_1d[-1] < q:
        return Tend
    return int(np.searchsorted(cdf_1d, q) + 1)  # 1..T


def shortest_mass_interval_1d(pmf_1d, target_mass, Tend, normalize: bool = True):
    """
    Find shortest contiguous [L,R] (1-indexed, inclusive) with mass >= target_mass.
    Tie-breaker: earlier L (smaller start index).
    Fallback: [1, Tend] when total mass <= 0 or no valid window.
    If normalize=True, pmf is renormalized to sum to 1 (conditional on event in horizon).
    """
    p = np.asarray(pmf_1d, dtype=float)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    total_mass = float(p.sum())
    if total_mass <= 0.0:
        return 1, int(Tend), True
    if normalize:
        p = p / total_mass
        total_mass = 1.0

    a = 0
    cum = 0.0
    best_a = 0
    best_b = int(Tend) - 1
    best_len = int(1e18)
    found = False

    for b in range(int(Tend)):
        cum += float(p[b])
        while a <= b and (cum - float(p[a])) >= float(target_mass):
            cum -= float(p[a])
            a += 1
        if cum >= float(target_mass):
            cur_len = b - a
            if (not found) or (cur_len < best_len) or (cur_len == best_len and a < best_a):
                found = True
                best_len = cur_len
                best_a = a
                best_b = b

    if not found:
        return 1, int(Tend), True
    return int(best_a + 1), int(best_b + 1), False


def overlap_metrics(pred_L, pred_R, true_L, true_R):
    true_L2 = true_L + 1
    inter_L = max(pred_L, true_L2)
    inter_R = min(pred_R, true_R)
    inter = max(0, inter_R - inter_L + 1)

    pred_len = max(1, pred_R - pred_L + 1)
    true_len = max(1, true_R - true_L)
    union_L = min(pred_L, true_L2)
    union_R = max(pred_R, true_R)
    union = max(1, union_R - union_L + 1)

    iou = inter / union
    recall = inter / true_len
    precision = inter / pred_len
    return iou, recall, precision


def early_recall80_site_year(rows: list[dict], key_field: str = "sample_id") -> tuple[float, int, int]:
    """
    Site-year early-warning recall for Stage2 gated interval rows.

    A site-year succeeds if any gated t* row satisfies:
      tstar < true_start and pred_L <= true_mid,
      where true_start = true_L + 1 and true_mid is the interval midpoint.
    """
    by_site_year: dict[str, bool] = {}
    for i, row in enumerate(rows):
        key = row.get(key_field)
        if key is None:
            site = row.get("site_id")
            year = row.get("year")
            key = f"{site}-{int(year)}" if site is not None and year is not None else str(i)

        success = False
        try:
            true_start = int(row["true_L"]) + 1
            true_end = int(row["true_R"])
            true_mid = (float(true_start) + float(true_end)) / 2.0
            tstar = int(row["tstar"])
            pred_L = int(row["pred_L"])
            success = bool(tstar < true_start and pred_L <= true_mid)
        except (KeyError, TypeError, ValueError):
            success = False

        by_site_year[str(key)] = bool(by_site_year.get(str(key), False) or success)

    denom = len(by_site_year)
    n_success = int(sum(1 for v in by_site_year.values() if v))
    value = float(n_success / denom) if denom > 0 else float("nan")
    return value, n_success, denom


@torch.no_grad()
def eval_metrics_with_overlap(model, loader, Tend, device, alpha=0.2, pi_method: str = "shortest"):
    model.eval()
    q_lo = alpha / 2
    q_hi = 1 - alpha / 2
    target_mass = 1.0 - float(alpha)

    hits_all, maes_all, mass_all = [], [], []
    ious, recalls, precs = [], [], []
    hits_int, maes_int, mass_int = [], [], []
    n_int = 0
    shortest_fallback_count = 0

    for X, L, R, ctype in loader:
        X = X.to(device, non_blocking=True)
        L_t = L.to(device, non_blocking=True)
        R_t = R.to(device, non_blocking=True)

        ctype_np = ctype.cpu().numpy().astype(int)
        L_np = L.cpu().numpy().astype(int)
        R_np = R.cpu().numpy().astype(int)

        hazard = model(X)
        pmf, cdf, logS = hazard_to_pmf_cdf_logS(hazard)

        # median
        cdf_last = cdf[:, -1]
        arg = (cdf >= 0.5).float().argmax(dim=1) + 1
        median = torch.where(cdf_last >= 0.5, arg, torch.tensor(Tend, device=cdf.device))
        median_np = median.cpu().numpy().astype(int)

        hit = ((median_np > L_np) & (median_np <= R_np)).astype(float)
        mid = np.round((L_np + R_np) / 2.0).astype(int)
        mae = np.abs(median_np - mid).astype(float)

        hits_all.extend(hit.tolist())
        maes_all.extend(mae.tolist())

        # mass-in-interval: S_L - S_R
        idxL = (torch.clamp(L_t, 1, Tend) - 1).long()
        idxR = (torch.clamp(R_t, 1, Tend) - 1).long()
        logS_L = logS.gather(1, idxL.view(-1, 1)).squeeze(1)
        logS_R = logS.gather(1, idxR.view(-1, 1)).squeeze(1)
        mass = (torch.exp(logS_L) - torch.exp(logS_R)).clamp(min=0.0).cpu().numpy()
        mass_all.extend(mass.tolist())

        # overlap (interval-only)
        cdf_np = cdf.cpu().numpy()
        pmf_np = pmf.cpu().numpy()
        for b in range(len(L_np)):
            if ctype_np[b] != CTYPE_INTERVAL:
                continue
            n_int += 1
            hits_int.append(hit[b]); maes_int.append(mae[b]); mass_int.append(mass[b])

            if pi_method == "shortest":
                pL, pR, used_fallback = shortest_mass_interval_1d(pmf_np[b], target_mass=target_mass, Tend=Tend)
                if used_fallback:
                    shortest_fallback_count += 1
            elif pi_method == "quantile":
                pL = quantile_from_cdf_1d(cdf_np[b], q_lo, Tend)
                pR = quantile_from_cdf_1d(cdf_np[b], q_hi, Tend)
            else:
                raise ValueError(f"Unknown pi_method: {pi_method}. expected 'shortest' or 'quantile'")

            # [reference: old quantile PI logic]
            # pL = quantile_from_cdf_1d(cdf_np[b], q_lo, Tend)
            # pR = quantile_from_cdf_1d(cdf_np[b], q_hi, Tend)
            pL = max(1, min(pL, Tend))
            pR = max(1, min(pR, Tend))
            if pL > pR:
                pL, pR = pR, pL

            iou, rec, prec = overlap_metrics(pL, pR, int(L_np[b]), int(R_np[b]))
            ious.append(iou); recalls.append(rec); precs.append(prec)

    return {
        "point_cov_mean_all": float(np.mean(hits_all)),
        "mae_mid_mean_all": float(np.mean(maes_all)),
        "mass_in_interval_mean_all": float(np.mean(mass_all)),
        "mass_in_interval_median_all": float(np.median(mass_all)),

        "point_cov_mean_interval_only": float(np.mean(hits_int)) if n_int > 0 else np.nan,
        "mae_mid_mean_interval_only": float(np.mean(maes_int)) if n_int > 0 else np.nan,
        "mass_in_interval_mean_interval_only": float(np.mean(mass_int)) if n_int > 0 else np.nan,

        "IoU_mean_interval_only(80%)": float(np.mean(ious)) if n_int > 0 else np.nan,
        "Recall_mean_interval_only(80%)": float(np.mean(recalls)) if n_int > 0 else np.nan,
        "Precision_mean_interval_only(80%)": float(np.mean(precs)) if n_int > 0 else np.nan,
        "N_interval_samples": int(n_int),
        "PI_shortest_fallback_count_interval_only": int(shortest_fallback_count),
    }


@torch.no_grad()
def eval_metrics_with_overlap_grouped(model, loader, Tend, device, alpha=0.2, pi_method: str = "shortest"):
    model.eval()
    q_lo = alpha / 2
    q_hi = 1 - alpha / 2
    target_mass = 1.0 - float(alpha)

    hits_all, maes_all, mass_all = [], [], []
    ious, recalls, precs = [], [], []
    hits_int, maes_int, mass_int = [], [], []
    n_int = 0
    shortest_fallback_count = 0
    conditional_survival = bool(getattr(model, "conditional_survival", True))

    for _batch in loader:
        if len(_batch) == 7:
            X, L, R, ctype, tstar, valid_mask, pheno = _batch
        else:
            X, L, R, ctype, tstar, valid_mask = _batch
            pheno = None
        X = X.to(device, non_blocking=True)
        L_t = L.to(device, non_blocking=True)
        R_t = R.to(device, non_blocking=True)
        ctype_t = ctype.to(device, non_blocking=True)
        tstar = tstar.to(device, non_blocking=True)
        valid_mask = valid_mask.to(device, non_blocking=True)
        if pheno is not None:
            pheno = pheno.to(device, non_blocking=True)

        hazard = model(X, tstar=tstar, valid_mask=valid_mask, pheno=pheno)
        hazard_f, L_f, R_f, ctype_f, n_valid = _flatten_grouped_valid(hazard, L_t, R_t, ctype_t, valid_mask)
        if n_valid <= 0:
            continue

        ctype_np = ctype_f.cpu().numpy().astype(int)
        L_np = L_f.cpu().numpy().astype(int)
        R_np = R_f.cpu().numpy().astype(int)

        tstar_f = tstar.reshape(-1)[valid_mask.reshape(-1)]
        pmf, cdf, logS = hazard_to_pmf_cdf_logS(hazard_f, tstar=tstar_f if conditional_survival else None)

        cdf_last = cdf[:, -1]
        arg = (cdf >= 0.5).float().argmax(dim=1) + 1
        median = torch.where(cdf_last >= 0.5, arg, torch.tensor(Tend, device=cdf.device))
        median_np = median.cpu().numpy().astype(int)

        hit = ((median_np > L_np) & (median_np <= R_np)).astype(float)
        mid = np.round((L_np + R_np) / 2.0).astype(int)
        mae = np.abs(median_np - mid).astype(float)

        hits_all.extend(hit.tolist())
        maes_all.extend(mae.tolist())

        idxL = (torch.clamp(L_f, 1, Tend) - 1).long()
        idxR = (torch.clamp(R_f, 1, Tend) - 1).long()
        logS_L = logS.gather(1, idxL.view(-1, 1)).squeeze(1)
        logS_R = logS.gather(1, idxR.view(-1, 1)).squeeze(1)
        mass = (torch.exp(logS_L) - torch.exp(logS_R)).clamp(min=0.0).cpu().numpy()
        mass_all.extend(mass.tolist())

        cdf_np = cdf.cpu().numpy()
        pmf_np = pmf.cpu().numpy()
        for b in range(len(L_np)):
            if ctype_np[b] != CTYPE_INTERVAL:
                continue
            n_int += 1
            hits_int.append(hit[b])
            maes_int.append(mae[b])
            mass_int.append(mass[b])

            if pi_method == "shortest":
                pL, pR, used_fallback = shortest_mass_interval_1d(pmf_np[b], target_mass=target_mass, Tend=Tend)
                if used_fallback:
                    shortest_fallback_count += 1
            elif pi_method == "quantile":
                pL = quantile_from_cdf_1d(cdf_np[b], q_lo, Tend)
                pR = quantile_from_cdf_1d(cdf_np[b], q_hi, Tend)
            else:
                raise ValueError(f"Unknown pi_method: {pi_method}. expected 'shortest' or 'quantile'")

            pL = max(1, min(pL, Tend))
            pR = max(1, min(pR, Tend))
            if pL > pR:
                pL, pR = pR, pL

            iou, rec, prec = overlap_metrics(pL, pR, int(L_np[b]), int(R_np[b]))
            ious.append(iou)
            recalls.append(rec)
            precs.append(prec)

    return {
        "point_cov_mean_all": float(np.mean(hits_all)) if hits_all else np.nan,
        "mae_mid_mean_all": float(np.mean(maes_all)) if maes_all else np.nan,
        "mass_in_interval_mean_all": float(np.mean(mass_all)) if mass_all else np.nan,
        "mass_in_interval_median_all": float(np.median(mass_all)) if mass_all else np.nan,
        "point_cov_mean_interval_only": float(np.mean(hits_int)) if n_int > 0 else np.nan,
        "mae_mid_mean_interval_only": float(np.mean(maes_int)) if n_int > 0 else np.nan,
        "mass_in_interval_mean_interval_only": float(np.mean(mass_int)) if n_int > 0 else np.nan,
        "IoU_mean_interval_only(80%)": float(np.mean(ious)) if n_int > 0 else np.nan,
        "Recall_mean_interval_only(80%)": float(np.mean(recalls)) if n_int > 0 else np.nan,
        "Precision_mean_interval_only(80%)": float(np.mean(precs)) if n_int > 0 else np.nan,
        "N_interval_samples": int(n_int),
        "PI_shortest_fallback_count_interval_only": int(shortest_fallback_count),
    }
