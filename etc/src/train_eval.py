from __future__ import annotations

import numpy as np
import torch
from etc.configs import config as C

CTYPE_INTERVAL = 0  # interval=0, right=1, left=2


# -------------------------
# Loss (interval/right/left censored)
# -------------------------
def interval_nll_per_sample(hazard, L, R, ctype, Tend):
    """
    hazard: (B,T) in (0,1)
    L,R,ctype: (B,)
    returns: (B,) nll
    """
    B, Tcur = hazard.shape
    assert Tcur == Tend

    log_surv_terms = torch.log1p(-hazard)          # (B,T)
    logS = torch.cumsum(log_surv_terms, dim=1)     # log S_t

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
):
    model.train(train)
    total, n = 0.0, 0
    base_total = 0.0
    mass_total = 0.0
    late_total = 0.0
    logged = False
    bad_batches = 0
    for X, L, R, ctype in loader:
        X = X.to(device, non_blocking=True)
        L = L.to(device, non_blocking=True)
        R = R.to(device, non_blocking=True)
        ctype = ctype.to(device, non_blocking=True)

        hazard = model(X)
        if not torch.isfinite(hazard).all():
            bad_batches += 1
            continue
        nll_vec = interval_nll_per_sample(hazard, L, R, ctype, Tend=Tend)
        base_loss = weighted_loss_from_ctype(nll_vec, ctype)
        loss = base_loss
        mass_loss_tensor = None
        late_loss_tensor = None

        if train and log_mass and not logged:
            mi = (ctype == CTYPE_INTERVAL)
            mi_frac = float(mi.float().mean().item())
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
            prefix = f"[mass] epoch {epoch_idx:02d} " if epoch_idx is not None else "[mass] "
            print(
                prefix
                + f"mi_frac={mi_frac:.4f} mass_mean={mass_mean:.6f} "
                + f"mass_min={mass_min:.6f} mass_max={mass_max:.6f} "
                + f"mass_loss={mass_loss_val:.6f} lambda_mass={lambda_mass} "
                + f"mass_loss_requires_grad={mass_loss_requires_grad} "
                + f"mass_loss_grad_fn={mass_loss_grad_fn}"
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
        return total_avg, base_avg, mass_avg, late_avg
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

        hazard = model(X)
        nll_vec = interval_nll_per_sample(hazard, L, R, ctype, Tend=Tend)
        loss = weighted_loss_from_ctype(nll_vec, ctype)

        total += float(loss.item()) * X.size(0)
        n += X.size(0)

    return total / max(n, 1)


# -------------------------
# Metrics
# -------------------------
def hazard_to_pmf_cdf_logS(hazard):
    B, T = hazard.shape
    logS = torch.cumsum(torch.log1p(-hazard), dim=1)  # log S_t
    S_prev = torch.cat([torch.ones(B, 1, device=hazard.device), torch.exp(logS[:, :-1])], dim=1)
    pmf = S_prev * hazard
    cdf = torch.cumsum(pmf, dim=1).clamp(0, 1)
    return pmf, cdf, logS


def quantile_from_cdf_1d(cdf_1d, q, Tend):
    if cdf_1d[-1] < q:
        return Tend
    return int(np.searchsorted(cdf_1d, q) + 1)  # 1..T


def shortest_mass_interval_1d(pmf_1d, target_mass, Tend):
    """
    Find shortest contiguous [L,R] (1-indexed, inclusive) with mass >= target_mass.
    Tie-breaker: earlier L (smaller start index).
    Fallback: [1, Tend] when total mass < target_mass or no valid window.
    """
    p = np.asarray(pmf_1d, dtype=float)
    p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
    p = np.clip(p, 0.0, None)
    total_mass = float(p.sum())
    if total_mass < float(target_mass):
        return 1, int(Tend), True

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
