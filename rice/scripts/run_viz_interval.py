from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest, default_out_root, ensure_output_dirs
from rice.scripts.common import (
    collate_grouped_stage2,
    make_loader,
    parse_seed_candidates,
    parse_tags,
    init_wandb_run,
    finish_wandb_run,
)
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import (
    EventTransformer,
    build_nowcast_samples,
    build_tabular_from_samples,
    make_event_labels,
)
from rice.scripts.run_event_eval import (
    apply_temperature,
    best_tau_by_target,
    build_alert_rows,
    fit_temperature_grid,
)
from rice.src.dataset import (
    split_by_site,
    split_samples,
    compute_norm_stats,
    GroupedIntervalEventDataset,
    IntervalEventDataset,
    split_seed_search_topk,
    log_split_fingerprint,
    log_split_sanity,
    build_stage2_nowcast_samples,
    group_stage2_samples_by_site_year,
)
from rice.src.model import HazardTransformer, HierarchicalCausalHazardTransformer
from rice.src.train_eval import (
    CTYPE_INTERVAL,
    hazard_to_pmf_cdf_logS,
    shortest_mass_interval_1d,
    quantile_from_cdf_1d,
    overlap_metrics,
    early_recall80_site_year,
)

WANDB_ENTITY_DEFAULT = "gilyoung7-seoul-national-university"
WANDB_PROJECT_DEFAULT = "agro-rice"


def resolve_ckpt_path(run: int, out_root: str, ckpt_path: str | None, fallback_name: str) -> Path:
    if ckpt_path:
        return Path(ckpt_path)
    return Path(out_root) / "ckpt" / fallback_name.format(run=run)


@torch.no_grad()
def predict_event_prob_event_model(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            X, _, _, ctype = batch
            X = X.to(device, non_blocking=True)
            y = (ctype.to(device, non_blocking=True) != 1).float()
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            X, y = batch
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
        else:
            raise ValueError("unexpected batch format in predict_event_prob_event_model")
        logits = model(X)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    if probs:
        return np.concatenate(probs)
    return np.asarray([], dtype=float)


@torch.no_grad()
def collect_interval_preds(
    model,
    loader,
    source_samples: list[dict],
    Tend: int,
    device,
    pi_method: str,
    pi_mass_level: float,
    ablate_feature_indices: list[int] | None,
    max_samples: int,
):
    rows: list[dict] = []
    sample_idx = 0
    q_lo = 0.1
    q_hi = 0.9
    target_mass = float(pi_mass_level)

    model.eval()
    for X, L, R, ctype in loader:
        X = X.to(device, non_blocking=True)
        if ablate_feature_indices:
            X = X.clone()
            X[:, :, ablate_feature_indices] = 0.0
        batch_n = int(L.shape[0])
        tstar_batch = torch.tensor(
            [int((source_samples[sample_idx + i] if (sample_idx + i) < len(source_samples) else {}).get("tstar", 1) or 1) for i in range(batch_n)],
            dtype=torch.long,
            device=device,
        )
        hazard = model(X)
        pmf, cdf, logS = hazard_to_pmf_cdf_logS(hazard, tstar=tstar_batch)

        L_np = L.cpu().numpy().astype(int)
        R_np = R.cpu().numpy().astype(int)
        ctype_np = ctype.cpu().numpy().astype(int)
        cdf_np = cdf.cpu().numpy()
        pmf_np = pmf.cpu().numpy()
        logS_np = logS.cpu().numpy()
        hazard_np = hazard.detach().cpu().numpy()

        for b in range(len(L_np)):
            if int(ctype_np[b]) != int(CTYPE_INTERVAL):
                continue
            if len(rows) >= max_samples:
                return rows

            sidx = sample_idx + b
            sample_meta = source_samples[sidx] if sidx < len(source_samples) else {}

            pmf_raw = pmf_np[b]
            # Enforce no event before t* (shift mass to t*+1..T)
            tstar_val = int(sample_meta.get("tstar", 0) or 0)
            if tstar_val > 0:
                assert np.all(pmf_raw[:tstar_val] < 1e-10), "PMF not conditional"
                right_mass = float(np.exp(logS_np[b, -1]))
                assert abs(float(np.sum(pmf_raw)) + right_mass - 1.0) < 1e-4, "PMF not normalized"
                pmf_raw = pmf_raw.copy()
                pmf_raw[:tstar_val] = 0.0  # pmf index 0 -> time 1
            total_mass = float(np.sum(pmf_raw))
            if total_mass > 0.0:
                pmf_cond = pmf_raw / total_mass
            else:
                pmf_cond = pmf_raw
            cdf_cond = np.cumsum(pmf_cond)

            if pi_method == "shortest":
                # use conditional pmf; already normalized -> normalize=False
                pL, pR, _ = shortest_mass_interval_1d(pmf_cond, target_mass=target_mass, Tend=Tend, normalize=False)
            elif pi_method == "quantile":
                pL = quantile_from_cdf_1d(cdf_cond, q_lo, Tend)
                pR = quantile_from_cdf_1d(cdf_cond, q_hi, Tend)
            else:
                raise ValueError(f"Unknown pi_method: {pi_method}. expected 'shortest' or 'quantile'")

            pL = max(1, min(int(pL), int(Tend)))
            pR = max(1, min(int(pR), int(Tend)))
            if pL > pR:
                pL, pR = pR, pL

            if total_mass > 0.0 and cdf_cond[-1] >= 0.5:
                p_point = int(np.searchsorted(cdf_cond, 0.5) + 1)
            else:
                p_point = int(Tend)

            true_L = int(L_np[b])
            true_R = int(R_np[b])
            iou, rec, prec = overlap_metrics(pL, pR, true_L, true_R)
            site = sample_meta.get("site_id")
            year = sample_meta.get("year")
            if site is not None and year is not None:
                sample_id = f"{site}-{int(year)}"
            else:
                sample_id = f"{int(sidx)}"

            rows.append(
                {
                    "sample_id": sample_id,
                    "true_L": true_L,
                    "true_R": true_R,
                    "pred_L": int(pL),
                    "pred_R": int(pR),
                    "pred_point": int(p_point),
                    "tstar": sample_meta.get("tstar"),
                    "iou": float(iou),
                    "pmf": pmf_cond.astype(np.float32, copy=False),
                    "hazard": hazard_np[b].astype(np.float32, copy=False),
                }
            )
        sample_idx += len(L_np)
    return rows


@torch.no_grad()
def collect_interval_preds_grouped(
    model,
    loader,
    source_groups: list[dict],
    Tend: int,
    device,
    pi_method: str,
    pi_mass_level: float,
    ablate_feature_indices: list[int] | None,
    max_samples: int,
):
    rows: list[dict] = []
    group_idx = 0
    q_lo = 0.1
    q_hi = 0.9
    target_mass = float(pi_mass_level)

    model.eval()
    for X, L, R, ctype, tstar, valid_mask in loader:
        X = X.to(device, non_blocking=True)
        if ablate_feature_indices:
            X = X.clone()
            X[:, :, :, ablate_feature_indices] = 0.0
        tstar_t = tstar.to(device, non_blocking=True)
        valid_mask_t = valid_mask.to(device, non_blocking=True)
        hazard = model(X, tstar=tstar_t, valid_mask=valid_mask_t)
        B, K, T_h = hazard.shape
        pmf, cdf, logS = hazard_to_pmf_cdf_logS(hazard.reshape(B * K, T_h), tstar=tstar_t.reshape(B * K))
        hazard_np = hazard.detach().cpu().numpy()
        pmf_np = pmf.cpu().numpy().reshape(B, K, T_h)
        cdf_np = cdf.cpu().numpy().reshape(B, K, T_h)
        logS_np = logS.cpu().numpy().reshape(B, K, T_h)
        L_np = L.cpu().numpy().astype(int)
        R_np = R.cpu().numpy().astype(int)
        ctype_np = ctype.cpu().numpy().astype(int)
        tstar_np = tstar.cpu().numpy().astype(int)
        valid_np = valid_mask.cpu().numpy().astype(bool)

        for bi in range(B):
            group = source_groups[group_idx + bi] if (group_idx + bi) < len(source_groups) else {"samples": []}
            source_rows = group.get("samples", [])
            for ki in range(K):
                if len(rows) >= max_samples:
                    return rows
                if not valid_np[bi, ki] or int(ctype_np[bi, ki]) != int(CTYPE_INTERVAL):
                    continue
                sample_meta = source_rows[ki] if ki < len(source_rows) else {}

                pmf_raw = pmf_np[bi, ki]
                tstar_val = int(tstar_np[bi, ki])
                if tstar_val > 0:
                    assert np.all(pmf_raw[:tstar_val] < 1e-10), "PMF not conditional"
                    right_mass = float(np.exp(logS_np[bi, ki, -1]))
                    assert abs(float(np.sum(pmf_raw)) + right_mass - 1.0) < 1e-4, "PMF not normalized"
                    pmf_raw = pmf_raw.copy()
                    pmf_raw[:tstar_val] = 0.0
                total_mass = float(np.sum(pmf_raw))
                if total_mass > 0.0:
                    pmf_cond = pmf_raw / total_mass
                else:
                    pmf_cond = pmf_raw
                cdf_cond = np.cumsum(pmf_cond)

                if pi_method == "shortest":
                    pL, pR, _ = shortest_mass_interval_1d(pmf_cond, target_mass=target_mass, Tend=Tend, normalize=False)
                elif pi_method == "quantile":
                    pL = quantile_from_cdf_1d(cdf_cond, q_lo, Tend)
                    pR = quantile_from_cdf_1d(cdf_cond, q_hi, Tend)
                else:
                    raise ValueError(f"Unknown pi_method: {pi_method}. expected 'shortest' or 'quantile'")

                pL = max(1, min(int(pL), int(Tend)))
                pR = max(1, min(int(pR), int(Tend)))
                if pL > pR:
                    pL, pR = pR, pL
                if total_mass > 0.0 and cdf_cond[-1] >= 0.5:
                    p_point = int(np.searchsorted(cdf_cond, 0.5) + 1)
                else:
                    p_point = int(Tend)

                true_L = int(L_np[bi, ki])
                true_R = int(R_np[bi, ki])
                iou, _rec, _prec = overlap_metrics(pL, pR, true_L, true_R)
                site = sample_meta.get("site_id")
                year = sample_meta.get("year")
                sample_id = f"{site}-{int(year)}" if site is not None and year is not None else f"{group_idx + bi}-{ki}"
                rows.append(
                    {
                        "sample_id": sample_id,
                        "true_L": true_L,
                        "true_R": true_R,
                        "pred_L": int(pL),
                        "pred_R": int(pR),
                        "pred_point": int(p_point),
                        "tstar": int(tstar_val),
                        "iou": float(iou),
                        "pmf": pmf_cond.astype(np.float32, copy=False),
                        "hazard": hazard_np[bi, ki].astype(np.float32, copy=False),
                    }
                )
        group_idx += B
    return rows


def plot_interval_rows(rows: list[dict], Tend: int, title: str):
    import matplotlib.pyplot as plt

    if not rows:
        return None
    n = len(rows)
    fig_h = max(2.0, 1.1 * n)
    fig, axes = plt.subplots(n, 1, figsize=(10, fig_h), sharex=True)
    if n == 1:
        axes = [axes]
    for i, (ax, r) in enumerate(zip(axes, rows)):
        ax.hlines(0, r["true_L"], r["true_R"], color="black", lw=6, alpha=0.25, label="true interval")
        ax.hlines(0, r["pred_L"], r["pred_R"], color="tab:blue", lw=3, label="pred interval")
        ax.plot(r["pred_point"], 0, marker="o", color="tab:blue", ms=5, label="pred point")
        alert = r.get("alert_tstar")
        if alert is not None:
            ax.axvline(int(alert), color="tab:red", lw=1.2, ls="--", label="alert t* (Stage 1)")
        # Stage 2 evaluation time = alert_tstar + offset. Marks the window
        # position used by the model to produce mu (selector-aware variant).
        s2_t = r.get("stage2_tstar")
        if s2_t is not None and alert is not None and int(s2_t) != int(alert):
            ax.axvline(int(s2_t), color="tab:orange", lw=1.2, ls=":", label="Stage 2 eval t* (alert+offset)")
        ax.set_yticks([])
        ax.set_xlim(1, Tend)
        ax.set_title(f"{r['sample_id']} | IoU={r['iou']:.2f}")
        if i == 0:
            # Single compact legend on the top row only.
            ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_interval_grid(rows: list[dict], Tend: int, title: str, n_cols: int = 2):
    """Multi-column PI bar grid for visualizing many samples in one figure.

    Each row shows:
        black thick line  = true interval [L, R]
        blue line + dot   = predicted interval [μ−1.96σ, μ+1.96σ] + μ
        red dashed line   = alert t* (Stage 1)
        orange dotted     = Stage 2 evaluation t* (alert + selected offset)

    Compared to plot_interval_rows (single column, K=5-10), this lays out
    K=20-100 rows in n_cols columns so the entire random subsample fits in
    one wandb image.
    """
    import matplotlib.pyplot as plt

    if not rows:
        return None
    n = len(rows)
    n_cols = max(1, int(n_cols))
    n_rows = (n + n_cols - 1) // n_cols
    fig_h = max(6.0, 0.55 * n_rows)
    fig_w = 8.0 * n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_w, fig_h),
                              sharex=True, squeeze=False)
    for idx in range(n_rows * n_cols):
        ax = axes[idx // n_cols, idx % n_cols]
        if idx >= n:
            ax.axis("off")
            continue
        r = rows[idx]
        ax.hlines(0, r["true_L"], r["true_R"], color="black", lw=4, alpha=0.30)
        ax.hlines(0, r["pred_L"], r["pred_R"], color="tab:blue", lw=2)
        ax.plot(r["pred_point"], 0, marker="o", color="tab:blue", ms=3)
        alert = r.get("alert_tstar")
        if alert is not None:
            ax.axvline(int(alert), color="tab:red", lw=0.8, ls="--")
        s2_t = r.get("stage2_tstar")
        if s2_t is not None and alert is not None and int(s2_t) != int(alert):
            ax.axvline(int(s2_t), color="tab:orange", lw=0.8, ls=":")
        ax.set_yticks([])
        ax.set_xlim(1, Tend)
        ax.set_title(f"{r['sample_id']} | IoU={r['iou']:.2f}",
                      fontsize=7, pad=1)
    # Shared legend on the top-left axis only (keep clutter low).
    handles = [
        plt.Line2D([0], [0], color="black", lw=4, alpha=0.30, label="true [L, R]"),
        plt.Line2D([0], [0], color="tab:blue", lw=2, marker="o", ms=4,
                    label="pred [μ ± 1.96σ]"),
        plt.Line2D([0], [0], color="tab:red", lw=1.0, ls="--", label="alert t* (Stage 1)"),
        plt.Line2D([0], [0], color="tab:orange", lw=1.0, ls=":",
                    label="Stage 2 eval t* (alert+offset)"),
    ]
    axes[0, 0].legend(handles=handles, loc="upper right",
                       fontsize=7, framealpha=0.85)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_pmf_rows(rows: list[dict], Tend: int, title: str):
    import matplotlib.pyplot as plt

    rows = [r for r in rows if np.asarray(r.get("pmf", []), dtype=float).size > 0]
    if not rows:
        return None
    n = len(rows)
    fig_h = max(2.4, 1.35 * n)
    fig, axes = plt.subplots(n, 1, figsize=(10, fig_h), sharex=True)
    if n == 1:
        axes = [axes]
    x = np.arange(1, int(Tend) + 1)
    for ax, r in zip(axes, rows):
        pmf = np.asarray(r.get("pmf", []), dtype=float)
        if pmf.size == 0:
            continue
        x_plot = x[: pmf.size]
        ax.plot(x_plot, pmf, color="tab:blue", lw=1.2, label="pmf")
        ax.axvspan(int(r["true_L"]), int(r["true_R"]), color="black", alpha=0.12, label="true interval")
        ax.axvspan(int(r["pred_L"]), int(r["pred_R"]), color="tab:blue", alpha=0.12, label="pred interval")
        ax.axvline(int(r["pred_point"]), color="tab:blue", lw=1.0, ls=":", label="pred point")
        alert = r.get("alert_tstar")
        if alert is not None:
            ax.axvline(int(alert), color="tab:red", lw=1.0, ls="--", label="early warning t*")
        ax.set_xlim(1, Tend)
        ax.set_ylabel("pmf")
        ax.set_title(f"{r['sample_id']} | IoU={r['iou']:.2f}")
        if ax is axes[0]:
            ax.legend(loc="upper right", fontsize=8)
    axes[-1].set_xlabel("DOY")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_hazard_rows(rows: list[dict], Tend: int, title: str, include_logit: bool = True):
    import matplotlib.pyplot as plt

    rows = [r for r in rows if np.asarray(r.get("hazard", []), dtype=float).size > 0]
    if not rows:
        return None
    n = len(rows)
    ncols = 2 if include_logit else 1
    fig_h = max(2.6, 1.45 * n)
    fig_w = 14 if include_logit else 10
    fig, axes = plt.subplots(n, ncols, figsize=(fig_w, fig_h), sharex=True)
    if n == 1 and ncols == 1:
        axes = np.asarray([[axes]])
    elif n == 1:
        axes = np.asarray([axes])
    elif ncols == 1:
        axes = np.asarray(axes).reshape(n, 1)

    x = np.arange(1, int(Tend) + 1)
    for i, r in enumerate(rows):
        hazard = np.asarray(r.get("hazard", []), dtype=float)
        if hazard.size == 0:
            continue
        h = np.clip(hazard, 1e-6, 1.0 - 1e-6)
        x_plot = x[: h.size]

        ax = axes[i, 0]
        ax.plot(x_plot, h, color="tab:orange", lw=1.2, label="hazard")
        ax.axvspan(int(r["true_L"]), int(r["true_R"]), color="black", alpha=0.12, label="true interval")
        ax.axvspan(int(r["pred_L"]), int(r["pred_R"]), color="tab:blue", alpha=0.12, label="pred interval")
        alert = r.get("alert_tstar")
        if alert is not None:
            ax.axvline(int(alert), color="tab:red", lw=1.0, ls="--", label="early warning t*")
        ax.set_xlim(1, Tend)
        ax.set_ylim(bottom=0.0)
        ax.set_ylabel("hazard")
        ax.set_title(f"{r['sample_id']} | IoU={r['iou']:.2f}")
        if i == 0:
            ax.legend(loc="upper right", fontsize=8)

        if include_logit:
            ax_l = axes[i, 1]
            logit = np.log(h / (1.0 - h))
            ax_l.plot(x_plot, logit, color="tab:purple", lw=1.0, label="logit(h)")
            ax_l.axvspan(int(r["true_L"]), int(r["true_R"]), color="black", alpha=0.12)
            ax_l.axvspan(int(r["pred_L"]), int(r["pred_R"]), color="tab:blue", alpha=0.12)
            if alert is not None:
                ax_l.axvline(int(alert), color="tab:red", lw=1.0, ls="--")
            ax_l.set_xlim(1, Tend)
            ax_l.set_ylabel("logit")
            ax_l.set_title("derived logit(h)")
            if i == 0:
                ax_l.legend(loc="upper right", fontsize=8)

    for ax in axes[-1, :]:
        ax.set_xlabel("DOY")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def summarize_hazard_diagnostics(rows: list[dict]) -> dict:
    hs = []
    post_hs = []
    for r in rows:
        h = np.asarray(r.get("hazard", []), dtype=float)
        if h.size == 0:
            continue
        hs.append(np.clip(h, 0.0, 1.0))
        t = int(r.get("stage2_tstar", r.get("tstar", 1)) or 1)
        start = max(0, min(t, h.size - 1))
        post_hs.append(np.clip(h[start:], 0.0, 1.0))
    if not hs:
        return {}

    h_max = np.asarray([float(np.max(h)) for h in hs], dtype=float)
    h_std = np.asarray([float(np.std(h)) for h in hs], dtype=float)
    post_max = np.asarray([float(np.max(h)) for h in post_hs if h.size > 0], dtype=float)
    post_std = np.asarray([float(np.std(h)) for h in post_hs if h.size > 0], dtype=float)

    min_len = min((h.size for h in post_hs if h.size > 0), default=0)
    if min_len > 1:
        mat = np.stack([h[:min_len] for h in post_hs if h.size >= min_len], axis=0)
        cross_sample_std = np.std(mat, axis=0)
        mean_cross_std = float(np.mean(cross_sample_std))
        median_cross_std = float(np.median(cross_sample_std))
        if mat.shape[0] > 1:
            centered = mat - mat.mean(axis=1, keepdims=True)
            denom = np.linalg.norm(centered, axis=1, keepdims=True)
            valid = denom.squeeze(1) > 1e-12
            if int(valid.sum()) > 1:
                z = centered[valid] / denom[valid]
                sim = z @ z.T
                tri = sim[np.triu_indices(sim.shape[0], k=1)]
                pairwise_cos_mean = float(np.mean(tri))
                pairwise_cos_median = float(np.median(tri))
            else:
                pairwise_cos_mean = float("nan")
                pairwise_cos_median = float("nan")
        else:
            pairwise_cos_mean = float("nan")
            pairwise_cos_median = float("nan")
    else:
        mean_cross_std = float("nan")
        median_cross_std = float("nan")
        pairwise_cos_mean = float("nan")
        pairwise_cos_median = float("nan")

    def q(a: np.ndarray, p: float) -> float:
        return float(np.nanpercentile(a, p)) if a.size else float("nan")

    return {
        "hazard_n": int(len(hs)),
        "hazard_max_mean": float(np.mean(h_max)),
        "hazard_max_p50": q(h_max, 50),
        "hazard_max_p90": q(h_max, 90),
        "hazard_std_mean": float(np.mean(h_std)),
        "hazard_std_p50": q(h_std, 50),
        "hazard_std_p90": q(h_std, 90),
        "post_hazard_max_mean": float(np.mean(post_max)) if post_max.size else float("nan"),
        "post_hazard_max_p50": q(post_max, 50),
        "post_hazard_max_p90": q(post_max, 90),
        "post_hazard_std_mean": float(np.mean(post_std)) if post_std.size else float("nan"),
        "post_hazard_std_p50": q(post_std, 50),
        "post_hazard_std_p90": q(post_std, 90),
        "post_hazard_cross_sample_std_mean": mean_cross_std,
        "post_hazard_cross_sample_std_median": median_cross_std,
        "post_hazard_pairwise_cos_mean": pairwise_cos_mean,
        "post_hazard_pairwise_cos_median": pairwise_cos_median,
    }


def plot_hazard_overlay(rows: list[dict], Tend: int, title: str, max_rows: int = 20):
    import matplotlib.pyplot as plt

    rows = [r for r in rows if np.asarray(r.get("hazard", []), dtype=float).size > 0]
    if not rows:
        return None
    rows = rows[: int(max_rows)]
    fig, ax = plt.subplots(1, 1, figsize=(10, 4.5))
    x = np.arange(1, int(Tend) + 1)
    for r in rows:
        h = np.asarray(r.get("hazard", []), dtype=float)
        t = int(r.get("stage2_tstar", r.get("tstar", 1)) or 1)
        start = max(0, min(t, h.size - 1))
        x_post = x[start : start + h[start:].size]
        ax.plot(x_post, h[start:], lw=1.0, alpha=0.35)
    ax.set_xlim(1, Tend)
    ax.set_ylim(bottom=0.0)
    ax.set_xlabel("DOY")
    ax.set_ylabel("hazard after stage2_tstar")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def pr_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    recall = tp / max(n_pos, 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapezoid(precision, recall))


def fp_rate_at_tau(y_true: np.ndarray, p: np.ndarray, tau: float) -> float:
    y = np.asarray(y_true, dtype=int)
    pred = (p >= float(tau)).astype(int)
    neg = (y == 0)
    if int(neg.sum()) == 0:
        return float("nan")
    fp = int(((pred == 1) & neg).sum())
    return float(fp / int(neg.sum()))


def recall_at_tau(y_true: np.ndarray, p: np.ndarray, tau: float) -> float:
    y = np.asarray(y_true, dtype=int)
    pred = (p >= float(tau)).astype(int)
    pos = (y == 1)
    if int(pos.sum()) == 0:
        return float("nan")
    tp = int(((pred == 1) & pos).sum())
    return float(tp / int(pos.sum()))


def compute_t_alert_start(
    y_true: np.ndarray,
    p_cal: np.ndarray,
    tstar: np.ndarray,
    tau: float,
    pr_auc_min: float = 0.70,
    fp_rate_max: float = 0.03,
    recall_min: float = 0.55,
    consecutive: int = 3,
) -> int | None:
    uniq = np.unique(tstar)
    if uniq.size == 0:
        return None
    cond = []
    for t in uniq:
        m = (tstar == t)
        y_t = y_true[m]
        p_t = p_cal[m]
        if y_t.size == 0:
            cond.append(False)
            continue
        pr_auc = pr_auc_binary(y_t, p_t)
        fp_rate = fp_rate_at_tau(y_t, p_t, tau)
        rec = recall_at_tau(y_t, p_t, tau)
        ok = (pr_auc >= pr_auc_min) and (fp_rate <= fp_rate_max) and (rec >= recall_min)
        cond.append(ok)
    cond = np.array(cond, dtype=bool)
    if cond.size < consecutive:
        return None
    for i in range(cond.size - consecutive + 1):
        if cond[i:i + consecutive].all():
            return int(uniq[i])
    return None


def build_alert_map(samples: list[dict], probs: np.ndarray, tau: float, t_alert_start: int | None) -> dict[str, int | None]:
    alert = {}
    for s, p in zip(samples, probs):
        tstar = int(s.get("tstar", -1))
        if t_alert_start is not None and tstar < int(t_alert_start):
            continue
        site = str(s.get("site_id", ""))
        year = int(s.get("year", -1))
        key = f"{site}-{year}"
        if p < float(tau):
            continue
        prev = alert.get(key)
        if prev is None or tstar < int(prev):
            alert[key] = tstar
    return alert


def build_alert_map_consecutive(
    samples: list[dict],
    probs: np.ndarray,
    tau: float,
    *,
    split_name: str,
    seed: int,
    t_alert_start: int | None,
    consecutive_k: int,
    smooth_window: int,
) -> dict[str, int | None]:
    rows = build_alert_rows(
        samples,
        probs,
        tau,
        split_name=split_name,
        seed=seed,
        t_alert_start=t_alert_start,
        consecutive_k=consecutive_k,
        smooth_window=smooth_window,
    )
    alert = {}
    for row in rows:
        sample_id = str(row[2])
        alert_tstar = row[7]
        if alert_tstar is not None and not pd.isna(alert_tstar):
            alert[sample_id] = int(alert_tstar)
    return alert


def mass_in_true_interval(row: dict, doy_start: int) -> float:
    pmf = np.asarray(row.get("pmf", []), dtype=float)
    if pmf.size == 0:
        return float("nan")
    true_L_rel = int(row["true_L"]) - int(doy_start) + 1
    true_R_rel = int(row["true_R"]) - int(doy_start) + 1
    l = max(1, true_L_rel)
    r = min(int(true_R_rel), int(pmf.size))
    if r < l:
        return 0.0
    return float(np.sum(pmf[l - 1 : r]))


def summarize_matched_interval_rows(rows: list[dict], n_true: int, doy_start: int) -> dict:
    ious, recs, precs, maes, masses, widths = [], [], [], [], [], []
    hit_count = 0
    post_count = 0
    lead_vals = []
    for r in rows:
        iou, rec, prec = overlap_metrics(r["pred_L"], r["pred_R"], r["true_L"], r["true_R"])
        ious.append(float(iou))
        recs.append(float(rec))
        precs.append(float(prec))
        hit_count += int(float(iou) > 0.0)
        pred_mid = (int(r["pred_L"]) + int(r["pred_R"])) / 2.0
        true_mid = ((int(r["true_L"]) + 1) + int(r["true_R"])) / 2.0
        maes.append(float(abs(pred_mid - true_mid)))
        masses.append(mass_in_true_interval(r, doy_start=doy_start))
        widths.append(float(int(r["pred_R"]) - int(r["pred_L"]) + 1))
        true_start = int(r["true_L"]) + 1
        stage2_t = int(r.get("stage2_tstar", r.get("tstar")))
        lead_vals.append(float(true_start - stage2_t))
        post_count += int(stage2_t >= true_start)

    interval_hit_precision = hit_count / len(rows) if rows else 0.0
    interval_hit_recall = hit_count / n_true if n_true > 0 else 0.0
    interval_hit_f1 = (
        2 * interval_hit_precision * interval_hit_recall / (interval_hit_precision + interval_hit_recall)
        if (interval_hit_precision + interval_hit_recall) > 0
        else 0.0
    )
    return {
        "IoU80": float(np.mean(ious)) if ious else float("nan"),
        "Rec80": float(np.mean(recs)) if recs else float("nan"),
        "Prec80": float(np.mean(precs)) if precs else float("nan"),
        "MAE_int": float(np.mean(maes)) if maes else float("nan"),
        "Mass_int": float(np.nanmean(np.asarray(masses, dtype=float))) if masses else float("nan"),
        "pred_width": float(np.mean(widths)) if widths else float("nan"),
        "interval_hit_precision": float(interval_hit_precision),
        "interval_hit_recall": float(interval_hit_recall),
        "interval_hit_f1": float(interval_hit_f1),
        "post_true_start_rate": float(post_count / len(rows)) if rows else 0.0,
        "lead_time_stage2": float(np.mean(lead_vals)) if lead_vals else float("nan"),
    }


def lead_bin_name(lead: float) -> str:
    if lead < 1:
        return "post_or_in"
    if lead <= 14:
        return "1-14"
    if lead <= 30:
        return "15-30"
    if lead <= 45:
        return "31-45"
    if lead <= 60:
        return "46-60"
    if lead <= 90:
        return "61-90"
    if lead <= 120:
        return "91-120"
    if lead <= 150:
        return "121-150"
    if lead <= 180:
        return "151-180"
    return "181+"


def fixed_width_top_mass_interval(row: dict, width: int, doy_start: int) -> tuple[int, int, float]:
    """
    Select the fixed-width absolute-DOY interval with maximum conditional PMF mass.
    PMF indices are relative to the Stage2 season, while row coordinates are
    absolute DOY after gated matching.
    """
    pmf = np.asarray(row.get("pmf", []), dtype=float)
    if pmf.size == 0:
        return int(row["pred_L"]), int(row["pred_R"]), float("nan")
    w = max(1, int(width))
    n = int(pmf.size)
    stage2_t_abs = int(row.get("stage2_tstar", row.get("tstar", doy_start)))
    start_rel0 = max(0, int(stage2_t_abs) - int(doy_start))  # after t*: rel index t*+1 => 0-based tstar
    if start_rel0 >= n:
        return int(doy_start + n - 1), int(doy_start + n - 1), 0.0
    if w >= n:
        return int(doy_start), int(doy_start + n - 1), float(np.sum(pmf))

    best_i = start_rel0
    best_mass = -1.0
    last_i = max(start_rel0, n - w)
    csum = np.concatenate([[0.0], np.cumsum(np.nan_to_num(pmf, nan=0.0, posinf=0.0, neginf=0.0))])
    for i in range(start_rel0, last_i + 1):
        j = min(n, i + w)
        mass = float(csum[j] - csum[i])
        if mass > best_mass:
            best_mass = mass
            best_i = i
    pred_L = int(doy_start + best_i)
    pred_R = int(doy_start + min(n - 1, best_i + w - 1))
    return pred_L, pred_R, float(max(best_mass, 0.0))


def add_fixed_width_metrics(rows: list[dict], widths: tuple[int, ...], doy_start: int) -> dict:
    out: dict[str, float | int] = {}
    for w in widths:
        ious, recs, precs, hits, masses = [], [], [], [], []
        early_rows = []
        for r in rows:
            pL, pR, mass = fixed_width_top_mass_interval(r, width=int(w), doy_start=doy_start)
            r[f"fixed{w}_L"] = int(pL)
            r[f"fixed{w}_R"] = int(pR)
            r[f"fixed{w}_mass"] = float(mass)
            iou, rec, prec = overlap_metrics(pL, pR, int(r["true_L"]), int(r["true_R"]))
            ious.append(float(iou))
            recs.append(float(rec))
            precs.append(float(prec))
            hits.append(float(iou > 0.0))
            masses.append(float(mass))
            rr = dict(r)
            rr["pred_L"] = int(pL)
            rr["pred_R"] = int(pR)
            early_rows.append(rr)

        early, early_success, early_denom = early_recall80_site_year(early_rows)
        out[f"IoU{w}"] = float(np.mean(ious)) if ious else float("nan")
        out[f"Rec{w}"] = float(np.mean(recs)) if recs else float("nan")
        out[f"Prec{w}"] = float(np.mean(precs)) if precs else float("nan")
        out[f"Hit{w}"] = float(np.mean(hits)) if hits else float("nan")
        out[f"Mass{w}"] = float(np.nanmean(np.asarray(masses, dtype=float))) if masses else float("nan")
        out[f"EarlyRecall{w}"] = float(early)
        out[f"EarlyRecall{w}_success"] = int(early_success)
        out[f"EarlyRecall{w}_denominator"] = int(early_denom)
    return out


def summarize_lead_bins(rows: list[dict], fixed_widths: tuple[int, ...], doy_start: int) -> pd.DataFrame:
    records = []
    for r in rows:
        true_start = int(r["true_L"]) + 1
        stage2_t = int(r.get("stage2_tstar", r.get("tstar")))
        lead = int(true_start - stage2_t)
        iou, rec, prec = overlap_metrics(int(r["pred_L"]), int(r["pred_R"]), int(r["true_L"]), int(r["true_R"]))
        rec_row = {
            "lead_bin": lead_bin_name(float(lead)),
            "lead": float(lead),
            "IoU80": float(iou),
            "Rec80": float(rec),
            "Prec80": float(prec),
            "MAE_int": float(abs(((int(r["pred_L"]) + int(r["pred_R"])) / 2.0) - (((int(r["true_L"]) + 1) + int(r["true_R"])) / 2.0))),
            "Mass_int": mass_in_true_interval(r, doy_start=doy_start),
            "pred_width": float(int(r["pred_R"]) - int(r["pred_L"]) + 1),
        }
        for w in fixed_widths:
            pL = int(r[f"fixed{w}_L"])
            pR = int(r[f"fixed{w}_R"])
            fiou, frec, fprec = overlap_metrics(pL, pR, int(r["true_L"]), int(r["true_R"]))
            rec_row[f"IoU{w}"] = float(fiou)
            rec_row[f"Rec{w}"] = float(frec)
            rec_row[f"Prec{w}"] = float(fprec)
            rec_row[f"Hit{w}"] = float(fiou > 0.0)
            rec_row[f"Mass{w}"] = float(r[f"fixed{w}_mass"])
        records.append(rec_row)
    if not records:
        return pd.DataFrame()
    df = pd.DataFrame(records)
    agg = {
        "n": ("lead", "size"),
        "lead_mean": ("lead", "mean"),
        "IoU80": ("IoU80", "mean"),
        "Rec80": ("Rec80", "mean"),
        "Prec80": ("Prec80", "mean"),
        "MAE_int": ("MAE_int", "mean"),
        "Mass_int": ("Mass_int", "mean"),
        "pred_width": ("pred_width", "mean"),
    }
    for w in fixed_widths:
        for c in (f"IoU{w}", f"Rec{w}", f"Prec{w}", f"Hit{w}", f"Mass{w}"):
            agg[c] = (c, "mean")
    order = ["post_or_in", "1-14", "15-30", "31-45", "46-60", "61-90", "91-120", "121-150", "151-180", "181+"]
    out = df.groupby("lead_bin", as_index=False).agg(**agg)
    out["lead_bin"] = pd.Categorical(out["lead_bin"], categories=order, ordered=True)
    return out.sort_values("lead_bin").reset_index(drop=True)


def synthetic_width_ceiling_diag(rows: list[dict], doy_start: int, Tend: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    eps = 1e-12
    x_abs = np.arange(int(doy_start), int(doy_start) + int(Tend), dtype=float)

    def triangular_center(center: float, half_width: int, future_mask: np.ndarray) -> np.ndarray:
        if int(half_width) <= 0:
            out = (np.abs(x_abs - float(center)) < 0.5).astype(float)
        else:
            out = np.maximum(0.0, 1.0 - np.abs(x_abs - float(center)) / float(half_width))
        out = np.where(future_mask, out, 0.0)
        total = float(np.sum(out))
        return out / total if total > 0.0 else out

    for r in rows:
        pmf = np.asarray(r.get("pmf", []), dtype=float)
        if pmf.size == 0:
            continue
        true_L = int(r["true_L"])
        true_R = int(r["true_R"])
        true_start = true_L + 1
        stage2_t = int(r.get("stage2_tstar", r.get("tstar")))
        lead = int(true_start - stage2_t)
        mid = (float(true_start) + float(true_R)) / 2.0

        true_L_rel = true_L - int(doy_start) + 1
        true_R_rel = true_R - int(doy_start) + 1
        l = max(1, true_L_rel)
        rr = min(int(true_R_rel), int(pmf.size))
        if rr < l:
            continue

        model_mass = float(np.sum(pmf[l - 1 : rr]))
        model_nll = -float(np.log(max(model_mass, eps)))

        future = x_abs > float(stage2_t)
        tri = triangular_center(mid, 7, future)
        tri_mass = float(np.sum(tri[l - 1 : rr])) if float(np.sum(tri)) > 0.0 else float("nan")
        tri_nll = -float(np.log(max(tri_mass, eps))) if np.isfinite(tri_mass) else float("nan")
        tri15 = triangular_center(mid, 15, future)
        tri15_mass = float(np.sum(tri15[l - 1 : rr])) if float(np.sum(tri15)) > 0.0 else float("nan")
        tri15_nll = -float(np.log(max(tri15_mass, eps))) if np.isfinite(tri15_mass) else float("nan")

        gauss = np.exp(-0.5 * ((x_abs - mid) / 4.0) ** 2)
        gauss = np.where(future, gauss, 0.0)
        gauss_sum = float(np.sum(gauss))
        if gauss_sum > 0.0:
            gauss = gauss / gauss_sum
            gauss_mass = float(np.sum(gauss[l - 1 : rr]))
            gauss_nll = -float(np.log(max(gauss_mass, eps)))
        else:
            gauss_mass = float("nan")
            gauss_nll = float("nan")

        pmf_future = np.where(future, np.nan_to_num(pmf, nan=0.0, posinf=0.0, neginf=0.0), 0.0)
        if float(np.sum(pmf_future)) > 0.0:
            mode_idx = int(np.argmax(pmf_future))
            mode_abs = float(x_abs[mode_idx])
            cdf_future = np.cumsum(pmf_future) / float(np.sum(pmf_future))
            median_idx = int(np.searchsorted(cdf_future, 0.5))
            median_idx = max(0, min(int(Tend) - 1, median_idx))
            median_abs = float(x_abs[median_idx])
        else:
            mode_abs = float("nan")
            median_abs = float("nan")

        narrowed = {}
        for prefix, center in (("mode", mode_abs), ("median", median_abs)):
            for hw in (7, 15):
                key = f"{prefix}{hw}"
                if np.isfinite(center):
                    q = triangular_center(center, hw, future)
                    mass_q = float(np.sum(q[l - 1 : rr])) if float(np.sum(q)) > 0.0 else float("nan")
                    nll_q = -float(np.log(max(mass_q, eps))) if np.isfinite(mass_q) else float("nan")
                else:
                    mass_q = float("nan")
                    nll_q = float("nan")
                narrowed[f"{key}_mass"] = mass_q
                narrowed[f"{key}_NLL"] = nll_q
                narrowed[f"diff_model_minus_{key}"] = model_nll - nll_q if np.isfinite(nll_q) else float("nan")

        rec = {
            "sample_id": r.get("sample_id", ""),
            "true_L": true_L,
            "true_start": true_start,
            "true_R": true_R,
            "true_mid": mid,
            "stage2_tstar": stage2_t,
            "lead": lead,
            "lead_bin": lead_bin_name(float(lead)),
            "model_mode": mode_abs,
            "model_median": median_abs,
            "mode_abs_err": abs(mode_abs - mid) if np.isfinite(mode_abs) else float("nan"),
            "median_abs_err": abs(median_abs - mid) if np.isfinite(median_abs) else float("nan"),
            "model_mass": model_mass,
            "model_NLL": model_nll,
            "tri_mass": tri_mass,
            "tri_NLL": tri_nll,
            "diff_model_minus_tri": model_nll - tri_nll if np.isfinite(tri_nll) else float("nan"),
            "tri15_mass": tri15_mass,
            "tri15_NLL": tri15_nll,
            "diff_model_minus_tri15": model_nll - tri15_nll if np.isfinite(tri15_nll) else float("nan"),
            "gauss_mass": gauss_mass,
            "gauss_NLL": gauss_nll,
            "diff_model_minus_gauss": model_nll - gauss_nll if np.isfinite(gauss_nll) else float("nan"),
        }
        rec.update(narrowed)
        records.append(rec)

    row_df = pd.DataFrame(records)
    if row_df.empty:
        return row_df, pd.DataFrame()

    def summarize_one(df: pd.DataFrame, prefix: str) -> dict:
        diff = df[f"diff_model_minus_{prefix}"].astype(float)
        out = {
            "n": int(len(df)),
            "model_NLL_mean": float(df["model_NLL"].mean()),
            f"{prefix}_NLL_mean": float(df[f"{prefix}_NLL"].mean()),
            f"{prefix}_diff_mean": float(diff.mean()),
            f"{prefix}_diff_p10": float(diff.quantile(0.10)),
            f"{prefix}_diff_p25": float(diff.quantile(0.25)),
            f"{prefix}_diff_p50": float(diff.quantile(0.50)),
            f"{prefix}_diff_p75": float(diff.quantile(0.75)),
            f"{prefix}_diff_p90": float(diff.quantile(0.90)),
            f"{prefix}_diff_gt0_rate": float((diff > 0.0).mean()),
            f"{prefix}_diff_gt05_rate": float((diff > 0.5).mean()),
        }
        return out

    def distance_summary(df: pd.DataFrame) -> dict:
        out = {}
        for prefix in ("mode", "median"):
            d = df[f"{prefix}_abs_err"].astype(float)
            out[f"{prefix}_abs_err_mean"] = float(d.mean())
            out[f"{prefix}_abs_err_p50"] = float(d.quantile(0.50))
            out[f"{prefix}_abs_err_p90"] = float(d.quantile(0.90))
        return out

    summary_records = []
    groups = [("all", row_df)]
    for lb in ["15-30", "31-45", "46-60", "61-90"]:
        sub = row_df[row_df["lead_bin"] == lb]
        if not sub.empty:
            groups.append((lb, sub))
    for name, df in groups:
        rec = {"lead_bin": name}
        for prefix in ("tri", "tri15", "gauss", "mode7", "mode15", "median7", "median15"):
            rec.update({k: v for k, v in summarize_one(df, prefix).items() if k not in rec})
        rec.update(distance_summary(df))
        summary_records.append(rec)
    summary_df = pd.DataFrame(summary_records)
    return row_df, summary_df


def load_stage1_eval_policy(stage1_eval_csv: str | None) -> dict[int, dict]:
    if not stage1_eval_csv:
        return {}
    df = pd.read_csv(stage1_eval_csv)
    out = {}
    for row in df.to_dict("records"):
        seed = int(row["seed"])
        out[seed] = {
            "tau": float(row["tau_selected"]),
            "gate_consecutive_k": int(row.get("gate_consecutive_k", 1)),
            "gate_smooth_window": int(row.get("gate_smooth_window", 1)),
            "gate_use_t_alert_start": int(row.get("gate_use_t_alert_start", 0)),
            "gate_policy_name": str(row.get("gate_policy_name", "")),
        }
    return out


def resolve_stage2_ablate_feature_indices(
    feature_names: list[str],
    *,
    ablate_calendar: bool,
    ablate_features: str | None,
) -> list[int]:
    targets: set[str] = set()
    if ablate_calendar:
        targets.update(
            {
                "days_since_growing_start",
                "days_until_growing_end",
                "is_growing",
            }
        )
    if ablate_features:
        targets.update({s.strip() for s in str(ablate_features).split(",") if s.strip()})
    if not targets:
        return []
    out = []
    for i, name in enumerate(feature_names):
        base = str(name).removesuffix("__miss")
        if str(name) in targets or base in targets:
            out.append(i)
    return sorted(set(out))


# ============================================================================
# Selector-aware evaluation (Phase S11/S12-style; bypasses Stage 2 inference)
# ----------------------------------------------------------------------------
# Activated by passing --selector_per_sample_csv.  Reads a phase_s3-style
# per-sample CSV that already has the selector's chosen mu (mu_at_pred_off),
# builds Gaussian PI = [μ − 1.96σ, μ + 1.96σ], and computes the full metrics
# suite (gate / interval pass rates, IoU_overall, P_ideal/useful/failed at the
# operational shift, ME/MAE/RMSE, OLS calibration slope) + visualizations.
# ============================================================================

SELECTOR_LEAD_BINS = [
    ("<15", lambda x: x < 15),
    ("15-30", lambda x: 15 <= x <= 30),
    ("31-45", lambda x: 31 <= x <= 45),
    ("46-60", lambda x: 46 <= x <= 60),
    ("61-90", lambda x: 61 <= x <= 90),
    ("91-120", lambda x: 91 <= x <= 120),
    (">120", lambda x: x > 120),
]


def _selector_bin_of_lead(lead: float) -> str:
    if not np.isfinite(lead):
        return "NA"
    for name, fn in SELECTOR_LEAD_BINS:
        if fn(lead):
            return name
    return "NA"


def _selector_bucket_label(bucket_lead: float, ideal_lo: float, ideal_hi: float) -> str:
    if not np.isfinite(bucket_lead):
        return "NA"
    if bucket_lead < 0:           return "MISSED"
    if bucket_lead < 7:           return "TOO_LATE"
    if bucket_lead < ideal_lo:    return "URGENT"
    if bucket_lead < ideal_hi:    return "IDEAL"
    if bucket_lead < 45:          return "ADVANCE"
    return "TOO_EARLY"


def rows_from_selector_csv(df: pd.DataFrame, sigma_fallback: float) -> list[dict]:
    """Convert phase_s3 per_sample CSV → rows compatible with plot_interval_rows.

    Expected columns: sample_id, L, R, t_star_doy, sigma, mu_at_pred_off,
    pred_off (optional).  PI = [μ − 1.96σ, μ + 1.96σ]; iou uses overlap_metrics
    with shift=0 (sample-intrinsic, matching phase_r convention).
    """
    rows: list[dict] = []
    for _, r in df.iterrows():
        mu = r.get("mu_at_pred_off")
        if mu is None or not np.isfinite(mu):
            continue
        sigma_s = float(r.get("sigma", sigma_fallback))
        L = int(r["L"]); Rg = int(r["R"])
        HW = 1.96 * sigma_s
        pL = int(round(float(mu) - HW))
        pR = int(round(float(mu) + HW))
        iou, _, _ = overlap_metrics(pL, pR, L, Rg)
        rows.append({
            "sample_id": str(r.get("sample_id", "?")),
            "true_L": L, "true_R": Rg,
            "pred_L": pL, "pred_R": pR,
            "pred_point": int(round(float(mu))),
            "alert_tstar": int(r["t_star_doy"]),
            "stage2_tstar": int(r["t_star_doy"]) + int(r.get("pred_off", 0) or 0),
            "pred_off": int(r.get("pred_off", 0) or 0),
            "sigma": sigma_s,
            "mu": float(mu),
            "iou": float(iou),
        })
    return rows


def compute_selector_metrics_summary(
    rows: list[dict], n_total: int, shift: float = 46.0,
    ideal_lo: float = 14.0, ideal_hi: float = 30.0,
    sigma_eval: float = 5.0, cohort: str = "",
) -> dict:
    """Headline metrics: IoU_overall(shift=0), P_ideal/useful/failed(shift),
    ME/MAE/RMSE on mu vs (L+R)/2, OLS calibration slope/intercept."""
    n_alerted = len(rows)
    buckets = {n: 0 for n in ("MISSED", "TOO_LATE", "URGENT", "IDEAL",
                                "ADVANCE", "TOO_EARLY")}
    iou_sum = 0.0
    me_list, abs_err_list, sq_err_list = [], [], []
    mu_list, mid_list = [], []
    for r in rows:
        mu = float(r["mu"]); s = float(r["sigma"])
        L = float(r["true_L"]); Rg = float(r["true_R"])
        iou_sum += float(r["iou"])
        bucket_lead = L - (mu + 1.96 * s - float(shift))
        buckets[_selector_bucket_label(bucket_lead, ideal_lo, ideal_hi)] += 1
        mid = (L + Rg) / 2.0
        delta = mu - mid
        me_list.append(delta); abs_err_list.append(abs(delta)); sq_err_list.append(delta * delta)
        mu_list.append(mu); mid_list.append(mid)

    if len(rows) >= 2:
        beta, alpha = np.polyfit(np.asarray(mid_list), np.asarray(mu_list), 1)
    else:
        beta, alpha = float("nan"), float("nan")

    denom = float(n_total) if n_total > 0 else float("nan")
    out = {
        "cohort": cohort,
        "n_total": int(n_total),
        "n_alerted": int(n_alerted),
        "gate_pass_rate": n_alerted / denom,
        "interval_pass_rate": n_alerted / denom,   # = gate_pass when CSV is post-Stage2
        "IoU_overall_shift0": iou_sum / denom,
        "shift": float(shift),
        "ideal_lead_low": float(ideal_lo),
        "ideal_lead_high": float(ideal_hi),
        "P_ideal_shift": buckets["IDEAL"] / denom,
        "P_useful_shift": (buckets["TOO_LATE"] + buckets["URGENT"] + buckets["IDEAL"]) / denom,
        "P_failed_shift": buckets["MISSED"] / denom,
        "ME": float(np.mean(me_list)) if me_list else float("nan"),
        "MAE": float(np.mean(abs_err_list)) if abs_err_list else float("nan"),
        "RMSE": float(np.sqrt(np.mean(sq_err_list))) if sq_err_list else float("nan"),
        "calibration_slope": float(beta),
        "calibration_intercept": float(alpha),
        "sigma_eval": float(sigma_eval),
        "pi_method": "gaussian_1.96sigma",
    }
    out.update({f"n_{k}_shift{int(shift)}": v for k, v in buckets.items()})
    return out


def compute_selector_lead_bin_metrics(
    rows: list[dict], n_total: int, shift: float = 46.0,
    ideal_lo: float = 14.0, ideal_hi: float = 30.0,
) -> pd.DataFrame:
    """Per lead bin (lead = L − alert_tstar): n, IoU, ME/MAE/RMSE, P_ideal(shift)."""
    bin_order = [n for n, _ in SELECTOR_LEAD_BINS] + ["NA"]
    by_bin: dict[str, list[dict]] = {b: [] for b in bin_order}
    for r in rows:
        lead = float(r["true_L"]) - float(r["alert_tstar"])
        by_bin[_selector_bin_of_lead(lead)].append(r)
    out_rows = []
    denom = float(n_total) if n_total > 0 else float("nan")
    for b in bin_order:
        rs = by_bin[b]
        n = len(rs)
        if n == 0:
            out_rows.append({"lead_bin": b, "n": 0,
                              "IoU_mean": float("nan"),
                              "ME": float("nan"), "MAE": float("nan"),
                              "RMSE": float("nan"),
                              "P_ideal_within_bin": float("nan"),
                              "iou_sum": 0.0, "contrib_to_overall_IoU": 0.0})
            continue
        ious = [float(x["iou"]) for x in rs]
        me   = [float(x["mu"]) - (float(x["true_L"]) + float(x["true_R"])) / 2.0 for x in rs]
        n_id = 0
        for x in rs:
            bl = float(x["true_L"]) - (float(x["mu"]) + 1.96 * float(x["sigma"]) - float(shift))
            if ideal_lo <= bl < ideal_hi:
                n_id += 1
        out_rows.append({
            "lead_bin": b, "n": n,
            "IoU_mean": float(np.mean(ious)),
            "ME": float(np.mean(me)),
            "MAE": float(np.mean([abs(x) for x in me])),
            "RMSE": float(np.sqrt(np.mean([x*x for x in me]))),
            "P_ideal_within_bin": float(n_id) / n,
            "iou_sum": float(sum(ious)),
            "contrib_to_overall_IoU": float(sum(ious)) / denom,
        })
    return pd.DataFrame(out_rows)


def plot_calibration_scatter(
    rows: list[dict],
    title: str,
    color_mode: str = "iou",
):
    """μ vs (L+R)/2 scatter; OLS + identity reference; no marginal histograms.

    Two color modes:
        "iou"     : point color = per-sample IoU (viridis, vmin=0, vmax=1).
                    Highlights *where* the model gets samples right.
        "density" : point color = local point density (gaussian KDE).
                    Reveals stacked points hidden by alpha-blending —
                    surfaces the cohort distribution that IoU coloring can hide.

    Both variants are scatter-only (no marginal hists) so they tile cleanly
    side-by-side in wandb / paper figures.
    """
    import matplotlib.pyplot as plt
    if not rows:
        return None
    mid = np.asarray([(r["true_L"] + r["true_R"]) / 2.0 for r in rows], dtype=float)
    mu  = np.asarray([r["mu"] for r in rows], dtype=float)

    if str(color_mode).lower() == "density":
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([mid, mu])
            # Add a tiny isotropic jitter when the data are degenerate
            # (e.g., μ collapsed onto one DOY) so KDE doesn't blow up.
            if np.std(mid) < 1e-6 or np.std(mu) < 1e-6:
                rng = np.random.default_rng(0)
                xy = xy + rng.normal(0.0, 1e-3, size=xy.shape)
            density = gaussian_kde(xy)(xy)
        except Exception:
            # Fallback: 2D histogram-based density (no scipy required).
            H, x_e, y_e = np.histogram2d(mid, mu, bins=40)
            x_idx = np.clip(np.searchsorted(x_e, mid) - 1, 0, H.shape[0] - 1)
            y_idx = np.clip(np.searchsorted(y_e, mu)  - 1, 0, H.shape[1] - 1)
            density = H[x_idx, y_idx]
        c_values = density
        cmap = "plasma"
        cbar_label = "point density"
        vmin = None
        vmax = None
    else:
        c_values = np.asarray([r["iou"] for r in rows], dtype=float)
        cmap = "viridis"
        cbar_label = "IoU"
        vmin = 0.0
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(mid, mu, c=c_values, cmap=cmap, s=18, alpha=0.75,
                     edgecolor="none", vmin=vmin, vmax=vmax)
    lo = float(min(mid.min(), mu.min())) - 5.0
    hi = float(max(mid.max(), mu.max())) + 5.0
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls=":",
            label="identity (perfect calibration)")
    if len(rows) >= 2:
        beta, alpha_ = np.polyfit(mid, mu, 1)
        x_line = np.array([lo, hi])
        ax.plot(x_line, alpha_ + beta * x_line, color="tab:red", lw=1.5,
                label=f"OLS: μ = {beta:.2f}·mid + {alpha_:.1f}")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_aspect("equal")
    ax.set_xlabel("true mid (L+R)/2  [DOY]")
    ax.set_ylabel("predicted μ  [DOY]")
    cbar = fig.colorbar(sc, ax=ax, shrink=0.7); cbar.set_label(cbar_label)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.85)
    ax.set_title(title)
    fig.tight_layout()
    return fig


def plot_lead_bin_bars(lead_bin_df: pd.DataFrame, title: str):
    """Two-panel bar chart: IoU mean (top) + MAE mean (bottom) per lead bin."""
    import matplotlib.pyplot as plt
    df = lead_bin_df[lead_bin_df["n"] > 0].copy()
    if df.empty:
        return None
    fig, axes = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
    bins = df["lead_bin"].astype(str).tolist()
    ax_iou, ax_mae = axes
    ax_iou.bar(bins, df["IoU_mean"].to_numpy(), color="tab:blue", alpha=0.75)
    for i, n in enumerate(df["n"].to_numpy()):
        y = float(df["IoU_mean"].iloc[i])
        ax_iou.text(i, y + 0.005, f"n={int(n)}", ha="center", fontsize=8)
    ax_iou.set_ylabel("IoU mean")
    iou_max = float(df["IoU_mean"].max()) if df["IoU_mean"].notna().any() else 0.05
    ax_iou.set_ylim(0.0, max(0.05, iou_max * 1.2))
    ax_iou.set_title(title)
    ax_mae.bar(bins, df["MAE"].to_numpy(), color="tab:red", alpha=0.75)
    ax_mae.set_ylabel("MAE  [days]")
    ax_mae.set_xlabel("lead bin  (L − alert_tstar, days)")
    fig.tight_layout()
    return fig


def run_selector_metrics(
    *,
    selector_per_sample_csv: str,
    out_dir: str | Path,
    cohort_label: str,
    n_total: int = 575,
    sigma_eval: float = 5.0,
    operational_shift: float = 46.0,
    ideal_lead_low: float = 14.0,
    ideal_lead_high: float = 30.0,
    topk: int = 5,
    worstk: int = 5,
    randomk: int = 5,
    random_grid_n: int = 50,
    random_grid_n_cols: int = 2,
    seed: int = 42,
    Tend: int = 300,
    wandb_run=None,
):
    """Selector-aware evaluation entry point.

    Bypasses Stage 2 inference: reads a phase_s3 per-sample CSV that already
    has (μ, σ, L, R, t_star_doy, pred_off) per sample, builds Gaussian PI,
    computes metrics + visualizations, writes CSVs + wandb artifacts.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_in = pd.read_csv(selector_per_sample_csv)
    print(f"[selector_eval] {selector_per_sample_csv}  rows={len(df_in)}", flush=True)
    rows = rows_from_selector_csv(df_in, sigma_fallback=sigma_eval)
    print(f"[selector_eval] valid rows={len(rows)}  cohort='{cohort_label}'  "
          f"n_total={n_total}", flush=True)

    summary = compute_selector_metrics_summary(
        rows, n_total=n_total, shift=operational_shift,
        ideal_lo=ideal_lead_low, ideal_hi=ideal_lead_high,
        sigma_eval=sigma_eval, cohort=cohort_label,
    )
    summary_df = pd.DataFrame([summary])
    summary_df.to_csv(out_dir / "metrics_summary.csv", index=False)
    print("=" * 70, flush=True)
    print(f"[metrics] IoU_overall(shift=0) = {summary['IoU_overall_shift0']:.4f}", flush=True)
    print(f"          P_ideal @shift{int(operational_shift)}  = {summary['P_ideal_shift']:.4f}", flush=True)
    print(f"          P_useful@shift{int(operational_shift)}  = {summary['P_useful_shift']:.4f}", flush=True)
    print(f"          P_failed@shift{int(operational_shift)}  = {summary['P_failed_shift']:.4f}", flush=True)
    print(f"          ME = {summary['ME']:+.2f}   MAE = {summary['MAE']:.2f}   "
          f"RMSE = {summary['RMSE']:.2f}", flush=True)
    print(f"          calibration_slope = {summary['calibration_slope']:.3f}   "
          f"intercept = {summary['calibration_intercept']:+.2f}", flush=True)
    print(f"[csv] {out_dir}/metrics_summary.csv", flush=True)

    lead_df = compute_selector_lead_bin_metrics(
        rows, n_total=n_total, shift=operational_shift,
        ideal_lo=ideal_lead_low, ideal_hi=ideal_lead_high,
    )
    lead_df.to_csv(out_dir / "metrics_by_lead_bin.csv", index=False)
    print(f"[csv] {out_dir}/metrics_by_lead_bin.csv", flush=True)

    aug_rows = []
    for r in rows:
        mid = (r["true_L"] + r["true_R"]) / 2.0
        lead = float(r["true_L"]) - float(r["alert_tstar"])
        bucket_lead = float(r["true_L"]) - (
            float(r["mu"]) + 1.96 * float(r["sigma"]) - float(operational_shift))
        aug_rows.append({
            "sample_id": r["sample_id"],
            "true_L": r["true_L"], "true_R": r["true_R"],
            "pred_L": r["pred_L"], "pred_R": r["pred_R"], "mu": r["mu"],
            "alert_tstar": r["alert_tstar"],
            "pred_off": r.get("pred_off"),
            "sigma": r["sigma"],
            "iou_shift0": r["iou"],
            "lead_from_alert": lead,
            "lead_bin": _selector_bin_of_lead(lead),
            "mu_minus_mid": float(r["mu"]) - mid,
            "abs_mu_minus_mid": abs(float(r["mu"]) - mid),
            "bucket_lead_shift": bucket_lead,
            "bucket_label": _selector_bucket_label(bucket_lead, ideal_lead_low, ideal_lead_high),
        })
    per_sample_df = pd.DataFrame(aug_rows)
    per_sample_df.to_csv(out_dir / "per_sample.csv", index=False)
    print(f"[csv] {out_dir}/per_sample.csv", flush=True)

    # --- visualizations -----------------------------------------------------
    # Two calibration scatters (same x/y, no marginal hists, different colors):
    #   "iou"     — shows where the model is accurate (per-sample IoU).
    #   "density" — surfaces overlapping points / cohort distribution.
    fig_cal_iou = plot_calibration_scatter(
        rows, title=f"[{cohort_label}] μ vs mid — IoU  (n={len(rows)})",
        color_mode="iou",
    )
    fig_cal_density = plot_calibration_scatter(
        rows, title=f"[{cohort_label}] μ vs mid — density  (n={len(rows)})",
        color_mode="density",
    )
    fig_lb = plot_lead_bin_bars(
        lead_df, title=f"[{cohort_label}] Lead-bin IoU / MAE")

    rows_sorted = sorted(rows, key=lambda x: x["iou"], reverse=True)
    top_rows = rows_sorted[:int(topk)]
    worst_rows = list(reversed(rows_sorted[-int(worstk):])) if rows_sorted else []
    if rows_sorted:
        rng = np.random.default_rng(int(seed))
        idx_small = rng.choice(len(rows_sorted),
                                size=min(int(randomk), len(rows_sorted)),
                                replace=False)
        rand_rows_small = [rows_sorted[int(i)] for i in idx_small]
        rng_grid = np.random.default_rng(int(seed) + 1)
        idx_grid = rng_grid.choice(
            len(rows_sorted),
            size=min(int(random_grid_n), len(rows_sorted)),
            replace=False,
        )
        rand_rows_grid = [rows_sorted[int(i)] for i in idx_grid]
    else:
        rand_rows_small = []
        rand_rows_grid = []
    fig_top = plot_interval_rows(top_rows, Tend=Tend,
                                  title=f"[{cohort_label}] Top-{len(top_rows)} IoU")
    fig_worst = plot_interval_rows(worst_rows, Tend=Tend,
                                    title=f"[{cohort_label}] Worst-{len(worst_rows)} IoU")
    fig_rand = plot_interval_rows(rand_rows_small, Tend=Tend,
                                   title=f"[{cohort_label}] Random-{len(rand_rows_small)}")
    # Dense random grid (50 rows × 2 cols by default) — gives a population view
    # of how typical samples are placed by the selector.
    fig_rand_grid = plot_interval_grid(
        rand_rows_grid, Tend=Tend, n_cols=int(random_grid_n_cols),
        title=f"[{cohort_label}] Random-{len(rand_rows_grid)} grid "
              f"({int(random_grid_n_cols)}-col)",
    )

    figures = {
        "calibration_iou": fig_cal_iou,
        "calibration_density": fig_cal_density,
        "lead_bin_bars": fig_lb,
        "top_interval": fig_top,
        "worst_interval": fig_worst,
        "random_interval": fig_rand,
        "random_interval_grid": fig_rand_grid,
    }
    for name, fig in figures.items():
        if fig is None:
            continue
        fp = out_dir / f"viz_{name}.png"
        fig.savefig(fp, dpi=160, bbox_inches="tight")
        print(f"[png] {fp}", flush=True)

    # --- wandb upload -------------------------------------------------------
    if wandb_run is not None:
        import wandb
        import matplotlib.pyplot as plt
        log_payload: dict = {}
        for name, fig in figures.items():
            if fig is not None:
                log_payload[f"viz/{name}"] = wandb.Image(fig)
        for k in ("IoU_overall_shift0", "P_ideal_shift", "P_useful_shift",
                  "P_failed_shift", "ME", "MAE", "RMSE",
                  "calibration_slope", "calibration_intercept",
                  "gate_pass_rate", "interval_pass_rate", "n_alerted"):
            log_payload[f"metrics/{k}"] = float(summary[k])
        log_payload["table/metrics_by_lead_bin"] = wandb.Table(dataframe=lead_df)
        log_payload["table/per_sample"] = wandb.Table(dataframe=per_sample_df)
        wandb_run.log(log_payload)
        for k in ("IoU_overall_shift0", "P_ideal_shift", "P_useful_shift",
                  "P_failed_shift", "MAE", "calibration_slope", "n_alerted"):
            wandb_run.summary[f"final/{k}"] = float(summary[k])
        wandb_run.summary["final/cohort"] = cohort_label
        wandb_run.summary["final/n_total"] = int(n_total)
        try:
            wandb_run.save(str(out_dir / "metrics_summary.csv"), policy="now")
            wandb_run.save(str(out_dir / "metrics_by_lead_bin.csv"), policy="now")
            wandb_run.save(str(out_dir / "per_sample.csv"), policy="now")
        except Exception as e:
            print(f"[wandb] save() failed (non-fatal): {e}", flush=True)
        print("[wandb] logged metrics, figures, tables; uploaded CSVs", flush=True)
        # Close figures to free memory.
        for fig in figures.values():
            if fig is not None:
                plt.close(fig)
    return summary, lead_df, per_sample_df


def main(
    pest: str,
    run: int,
    stage1_ckpt: str,
    stage1_eval_csv: str | None,
    stage2_ckpt: str,
    out_root: str,
    split_seed: int,
    split_mode: str,
    val_year: int,
    test_year_min: int,
    test_year_max: int,
    seeds: list[int] | None,
    auto_split_seed: bool,
    seed_candidates_raw: str | None,
    target_test_interval: int | None,
    tol_test_interval: int | None,
    auto_split_topk: int,
    split_seed_from_topk_idx: int | None,
    split_seeds_json: str | None,
    split: str,
    topk: int,
    worstk: int,
    randomk: int,
    max_pool: int,
    stage2_tstar_offset: int,
    pi_mass_level: float,
    synthetic_width_diag: bool,
    stage2_ablate_calendar: bool,
    stage2_ablate_features: str | None,
    final_tag: str | None,
    reference_sample_ids: str | None,
    tau_mode: str,
    tau_target_precision: float,
    tau_target_recall: float,
    use_wandb: bool,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_group: str | None,
    wandb_run_name: str | None,
    wandb_tags: str | None,
    wandb_job_type: str | None,
    alert_map_csv: str | None = None,
    # --- Phase S11/S12-style selector-aware eval (added 2026-05) ---
    selector_per_sample_csv: str | None = None,
    cohort_label: str = "selector_eval",
    n_total_test: int = 575,
    sigma_eval: float = 5.0,
    operational_shift: float = 46.0,
    ideal_lead_low: float = 14.0,
    ideal_lead_high: float = 30.0,
    selector_out_dir: str | None = None,
    selector_Tend: int = 300,
    selector_random_grid_n: int = 50,
    selector_random_grid_cols: int = 2,
):
    _, get_feature_cols = resolve_pest(pest)
    if not out_root:
        out_root = default_out_root(pest)
    ensure_output_dirs(out_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # If a selector per-sample CSV is provided, bypass Stage 2 inference and
    # run the lightweight metrics + viz suite directly off the CSV.  This
    # supports the Phase S11/S12 best = "2-sided baseline + selector (C_old)"
    # cohort where μ per sample was already chosen by the v3_mu_only logreg
    # OOF in phase_s3.
    if selector_per_sample_csv:
        wandb_run_sel = init_wandb_run(
            use_wandb=use_wandb,
            project=wandb_project or WANDB_PROJECT_DEFAULT,
            entity=wandb_entity or WANDB_ENTITY_DEFAULT,
            run_name=wandb_run_name,
            group=wandb_group,
            job_type=wandb_job_type or "selector_eval",
            tags=parse_tags(wandb_tags) + [f"pest:{pest}", "script:run_viz_interval",
                                              "mode:selector_eval",
                                              f"cohort:{cohort_label}"],
            config={
                "pest": pest, "run": int(run),
                "selector_per_sample_csv": selector_per_sample_csv,
                "cohort_label": cohort_label,
                "n_total_test": int(n_total_test),
                "sigma_eval": float(sigma_eval),
                "operational_shift": float(operational_shift),
                "ideal_lead_low": float(ideal_lead_low),
                "ideal_lead_high": float(ideal_lead_high),
            },
        )
        sel_out = Path(selector_out_dir) if selector_out_dir else Path(out_root) / "selector_eval"
        run_selector_metrics(
            selector_per_sample_csv=selector_per_sample_csv,
            out_dir=sel_out,
            cohort_label=cohort_label,
            n_total=int(n_total_test),
            sigma_eval=float(sigma_eval),
            operational_shift=float(operational_shift),
            ideal_lead_low=float(ideal_lead_low),
            ideal_lead_high=float(ideal_lead_high),
            topk=int(topk), worstk=int(worstk), randomk=int(randomk),
            random_grid_n=int(selector_random_grid_n),
            random_grid_n_cols=int(selector_random_grid_cols),
            seed=int(split_seed),
            Tend=int(selector_Tend),
            wandb_run=wandb_run_sel,
        )
        finish_wandb_run(wandb_run_sel)
        return

    wandb_run = init_wandb_run(
        use_wandb=use_wandb,
        project=wandb_project or WANDB_PROJECT_DEFAULT,
        entity=wandb_entity or WANDB_ENTITY_DEFAULT,
        run_name=wandb_run_name,
        group=wandb_group,
        job_type=wandb_job_type,
        tags=parse_tags(wandb_tags) + [f"pest:{pest}", "script:run_viz_interval"],
        config={
            "pest": pest,
            "run": run,
            "split_seed": int(split_seed),
            "split_mode": split_mode,
            "split": split,
            "topk": int(topk),
            "worstk": int(worstk),
            "randomk": int(randomk),
            "max_pool": int(max_pool),
            "stage2_tstar_offset": int(stage2_tstar_offset),
            "final_tag": final_tag,
            "tau_mode": tau_mode,
            "tau_target_precision": float(tau_target_precision),
            "tau_target_recall": float(tau_target_recall),
        },
    )

    stage1_path = resolve_ckpt_path(run, out_root, stage1_ckpt, "event_classifier_run{run}.pt")
    stage2_path = resolve_ckpt_path(run, out_root, stage2_ckpt, "checkpoint_run{run}.pt")
    print(f"Using stage1 ckpt: {stage1_path}")
    print(f"Using stage2 ckpt: {stage2_path}")
    ckpt1 = torch.load(stage1_path, map_location="cpu")
    ckpt2 = torch.load(stage2_path, map_location="cpu")
    split_mode = str(ckpt2.get("split_mode", split_mode))
    print(f"Effective Viz Split config: split_mode={split_mode}, split_seed={split_seed}")
    stage1_eval_policy = load_stage1_eval_policy(stage1_eval_csv)
    if stage1_eval_policy:
        print(f"Using stage1 eval policy CSV: {stage1_eval_csv}")
    stage1_doy_start = int(ckpt1.get("doy_start", C.DOY_START))
    stage1_doy_end = int(ckpt1.get("doy_end", C.DOY_END))
    stage2_doy_start = int(ckpt2.get("doy_start", C.DOY_START))
    stage2_doy_end = int(ckpt2.get("doy_end", C.DOY_END))
    if "d_model" in ckpt2:
        C.D_MODEL = int(ckpt2["d_model"])
    if "n_head" in ckpt2:
        C.N_HEAD = int(ckpt2["n_head"])
    if "n_layers" in ckpt2:
        C.N_LAYERS = int(ckpt2["n_layers"])

    C.DOY_START = stage2_doy_start
    C.DOY_END = stage2_doy_end
    feature_cols2, feature_names_eval, T, samples2 = build_samples_for_run(run, get_feature_cols)
    T2 = int(T)
    print(
        f"[features:stage2] doy={stage2_doy_start}-{stage2_doy_end} "
        f"n={len(feature_names_eval)} head={feature_names_eval[:5]} tail={feature_names_eval[-5:]}"
    )
    # Stage-2 DIRECT neighbor: re-append the 6 channels the model was trained with
    # (driven by ckpt2 metadata), before ablation/split so feature indices stay
    # aligned. No-op for ckpts trained without --stage2_add_neighbor_history.
    if bool(ckpt2.get("stage2_neighbor_history_added", False)):
        from rice.scripts.neighbor_history_utils import (
            load_long_events, build_neighbor_index, append_neighbor_to_samples,
            NEIGHBOR_CHANNEL_NAMES, NEIGHBOR_FEATURE_DIM, DEFAULT_DECAY_KM,
        )
        _nb_decay = float(ckpt2.get("stage2_neighbor_decay_km", DEFAULT_DECAY_KM))
        _nb_ev, _nb_co, _nb_sy = load_long_events(
            C.PATH_OBS, label_col=getattr(C, "LABEL_COL", "label_event"),
            year_min=getattr(C, "YEAR_MIN", None), year_max=getattr(C, "YEAR_MAX", None),
        )
        _nb_index = build_neighbor_index(_nb_ev, _nb_co)
        _nb_before = int(samples2[0]["X"].shape[1])
        append_neighbor_to_samples(samples2, _nb_index, doy_start=int(C.DOY_START), decay_km=_nb_decay)
        feature_names_eval = list(feature_names_eval) + list(NEIGHBOR_CHANNEL_NAMES)
        print(f"[stage2_neighbor] viz re-append: before_d_in={_nb_before} added={NEIGHBOR_FEATURE_DIM} "
              f"after_d_in={int(samples2[0]['X'].shape[1])} decay_km={_nb_decay} (matches ckpt meta)")

    ablate_feature_indices = resolve_stage2_ablate_feature_indices(
        feature_names_eval,
        ablate_calendar=bool(stage2_ablate_calendar),
        ablate_features=stage2_ablate_features,
    )
    if ablate_feature_indices:
        ablated = [feature_names_eval[i] for i in ablate_feature_indices]
        print(f"[stage2_ablate] indices={ablate_feature_indices} names={ablated}")

    if split_seeds_json is not None:
        from rice.scripts.run_eval import resolve_split_seeds_json_path, load_split_seed_from_topk
        split_seeds_json_path = resolve_split_seeds_json_path(out_root, split_seeds_json)
        split_seed, chosen_idx, chosen, _ = load_split_seed_from_topk(split_seeds_json_path, split_seed_from_topk_idx)
        train_s2_base, val_s2_base, test_s2_base = split_samples(
            samples2, val_frac=0.1, test_frac=0.1, seed=split_seed, split_mode=split_mode,
            val_year=val_year, test_year_min=test_year_min, test_year_max=test_year_max,
        )
        print(f"[split_seed_json] selected seed={split_seed} idx={chosen_idx} file={split_seeds_json_path}")
    elif auto_split_seed:
        candidates = parse_seed_candidates(seed_candidates_raw) or list(range(0, 200))
        result = split_seed_search_topk(
            samples2,
            val_frac=0.1,
            test_frac=0.1,
            seed_candidates=candidates,
            target_test_interval=target_test_interval,
            tol_test_interval=tol_test_interval,
            topk=auto_split_topk,
            split_mode=split_mode,
        )
        topk_list = result["topk"]
        if not topk_list:
            raise ValueError("auto_split_seed produced no candidates")
        if split_seed_from_topk_idx is None:
            split_seed_from_topk_idx = 0
        chosen = topk_list[split_seed_from_topk_idx]
        split_seed = int(chosen["seed"])
        train_s2_base, val_s2_base, test_s2_base = split_samples(
            samples2, val_frac=0.1, test_frac=0.1, seed=split_seed, split_mode=split_mode,
            val_year=val_year, test_year_min=test_year_min, test_year_max=test_year_max,
        )
        print(f"[auto_split] selected seed={split_seed} score={chosen['score']:.6f} counts={chosen['counts']}")
    else:
        train_s2_base, val_s2_base, test_s2_base = split_samples(
            samples2, val_frac=0.1, test_frac=0.1, seed=split_seed, split_mode=split_mode,
            val_year=val_year, test_year_min=test_year_min, test_year_max=test_year_max,
        )

    log_split_sanity("viz_stage2_base", train_s2_base, val_s2_base, test_s2_base, split_mode=split_mode)

    C.DOY_START = stage1_doy_start
    C.DOY_END = stage1_doy_end
    _feature_cols1, feature_names1, _T1, samples1_base = build_samples_for_run(run, get_feature_cols)
    train_s1_base, val_s1_base, test_s1_base = split_samples(
        samples1_base, val_frac=0.1, test_frac=0.1, seed=split_seed, split_mode=split_mode,
        val_year=val_year, test_year_min=test_year_min, test_year_max=test_year_max,
    )
    log_split_sanity("viz_stage1_base", train_s1_base, val_s1_base, test_s1_base, split_mode=split_mode)
    print(
        f"[features:stage1] doy={stage1_doy_start}-{stage1_doy_end} "
        f"n={len(feature_names1)} head={feature_names1[:5]} tail={feature_names1[-5:]}"
    )

    # Stage1 nowcast samples (for alert t*)
    task_mode = str(ckpt1.get("task_mode", "season_complete"))
    nowcast_window = int(ckpt1.get("nowcast_window", 28))
    nowcast_stride = int(ckpt1.get("nowcast_stride", 7))
    nowcast_tstar_start = ckpt1.get("nowcast_tstar_start", None)
    nowcast_only_pre_event = int(ckpt1.get("nowcast_only_pre_event", 1))
    nowcast_event_time_proxy = str(ckpt1.get("nowcast_event_time_proxy", "r"))
    add_tstar_position_feature = bool(ckpt1.get("add_tstar_position_feature", False))
    if task_mode == "nowcast":
        val_s1 = build_nowcast_samples(
            val_s1_base,
            window=nowcast_window,
            stride=nowcast_stride,
            tstar_start=nowcast_tstar_start,
            only_pre_event=bool(nowcast_only_pre_event),
            event_time_proxy=nowcast_event_time_proxy,
        )
        test_s1 = build_nowcast_samples(
            test_s1_base,
            window=nowcast_window,
            stride=nowcast_stride,
            tstar_start=nowcast_tstar_start,
            only_pre_event=bool(nowcast_only_pre_event),
            event_time_proxy=nowcast_event_time_proxy,
        )
    else:
        raise ValueError("Stage1 visualization expects nowcast task_mode")

    # Stage2 nowcast samples (for interval viz)
    C.DOY_START = stage2_doy_start
    C.DOY_END = stage2_doy_end
    stage2_nowcast = bool(ckpt2.get("stage2_nowcast", False))
    if stage2_nowcast:
        stage2_nowcast_window = int(ckpt2.get("stage2_nowcast_window", 56))
        stage2_nowcast_stride = int(ckpt2.get("stage2_nowcast_stride", 7))
        stage2_nowcast_tstar_start = ckpt2.get("stage2_nowcast_tstar_start", None)
        stage2_nowcast_only_pre_event = int(ckpt2.get("stage2_nowcast_only_pre_event", 1))
        stage2_nowcast_event_time_proxy = str(ckpt2.get("stage2_nowcast_event_time_proxy", "mid"))
        stage2_nowcast_require_tstar_before_L = int(ckpt2.get("stage2_nowcast_require_tstar_before_L", 1))
        val_s2 = build_stage2_nowcast_samples(
            val_s2_base,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
            require_tstar_before_L=bool(stage2_nowcast_require_tstar_before_L),
        )
        test_s2 = build_stage2_nowcast_samples(
            test_s2_base,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
            require_tstar_before_L=bool(stage2_nowcast_require_tstar_before_L),
        )
    else:
        val_s2 = val_s2_base
        test_s2 = test_s2_base

    log_split_fingerprint("viz_final_stage2", train_s2_base, val_s2_base, test_s2_base)

    x_mean, x_std = compute_norm_stats(train_s2_base)
    grouped_mode = bool(
        ckpt2.get("stage2_causal_tstar", False)
        or str(ckpt2.get("stage2_model_kind", "flat")) == "hierarchical_causal_tstar"
    )
    stage2_tstar_layers = int(ckpt2.get("stage2_tstar_layers", 1))
    stage2_use_tstar_scalar_pos = int(ckpt2.get("stage2_use_tstar_scalar_pos", 0))
    stage2_time_chunk_size = int(ckpt2.get("stage2_time_chunk_size", 64))
    stage2_conditional_survival = bool(int(ckpt2.get("stage2_conditional_survival", 1)))
    stage2_pmf_mode = str(ckpt2.get("stage2_pmf_mode", "hazard"))
    stage2_pmf_sigma = float(ckpt2.get("stage2_pmf_sigma", 5.0))
    stage2_pmf_mu_max = float(ckpt2.get("stage2_pmf_mu_max", 0.0))
    stage2_pmf_asym_weight = float(ckpt2.get("stage2_pmf_asym_weight", 10.0))
    stage2_pmf_right_weight = float(ckpt2.get("stage2_pmf_right_weight", 0.3))
    stage2_pmf_target_offset = float(ckpt2.get("stage2_pmf_target_offset", 0.0))
    if grouped_mode:
        val_groups2 = group_stage2_samples_by_site_year(val_s2)
        test_groups2 = group_stage2_samples_by_site_year(test_s2)
        val_ds2 = GroupedIntervalEventDataset(val_groups2, x_mean, x_std)
        test_ds2 = GroupedIntervalEventDataset(test_groups2, x_mean, x_std)
        val_loader2 = make_loader(val_ds2, 1, shuffle=False, collate_fn=collate_grouped_stage2)
        test_loader2 = make_loader(test_ds2, 1, shuffle=False, collate_fn=collate_grouped_stage2)
        print(f"[stage2_causal_tstar] grouped viz set sizes: val={len(val_groups2)} test={len(test_groups2)}")
    else:
        val_groups2 = []
        test_groups2 = []
        val_ds2 = IntervalEventDataset(val_s2, x_mean, x_std)
        test_ds2 = IntervalEventDataset(test_s2, x_mean, x_std)
        val_loader2 = make_loader(val_ds2, C.BATCH_EVAL, shuffle=False)
        test_loader2 = make_loader(test_ds2, C.BATCH_EVAL, shuffle=False)

    if split not in {"val", "test"}:
        raise ValueError("--split must be 'val' or 'test'")
    source_samples2 = val_s2 if split == "val" else test_s2
    source_groups2 = val_groups2 if split == "val" else test_groups2
    loader2 = val_loader2 if split == "val" else test_loader2
    samples1 = val_s1 if split == "val" else test_s1

    # Stage1 labels for tau
    y_val = make_event_labels(val_s1)
    y_test = make_event_labels(test_s1)

    for d2 in ckpt2["trained_states"]:
        seed = int(d2["seed"])
        if seeds is not None and seed not in seeds:
            continue

        # Stage1: pick matching seed
        d1 = None
        for s in ckpt1["trained_states"]:
            if int(s["seed"]) == seed:
                d1 = s
                break
        if d1 is None:
            print(f"[seed {seed}] no stage1 state; skipping")
            continue

        # Stage1 model
        model_kind = str(ckpt1.get("event_model", "transformer"))
        if model_kind == "transformer":
            model1 = EventTransformer(
                d_in=int(val_ds2[0][0].shape[-1]),
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=2,
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
            ).to(device)
            model1.load_state_dict(d1["state_dict"])
            model1.eval()
            val_loader1 = make_loader(IntervalEventDataset(val_s1, x_mean, x_std), C.BATCH_EVAL, shuffle=False)
            test_loader1 = make_loader(IntervalEventDataset(test_s1, x_mean, x_std), C.BATCH_EVAL, shuffle=False)
            p_val_raw = predict_event_prob_event_model(model1, val_loader1, device=device)
            p_test_raw = predict_event_prob_event_model(model1, test_loader1, device=device)
        else:
            clf = d1.get("sk_model")
            if clf is None:
                raise ValueError("stage1 checkpoint missing sk_model")
            X_val_tab = build_tabular_from_samples(
                val_s1,
                add_tstar_position_feature=bool(add_tstar_position_feature),
            )
            X_test_tab = build_tabular_from_samples(
                test_s1,
                add_tstar_position_feature=bool(add_tstar_position_feature),
            )
            if hasattr(clf, "predict_proba"):
                p_val_raw = clf.predict_proba(X_val_tab)[:, 1]
                p_test_raw = clf.predict_proba(X_test_tab)[:, 1]
            else:
                p_val_raw = np.asarray(clf.predict(X_val_tab), dtype=float)
                p_test_raw = np.asarray(clf.predict(X_test_tab), dtype=float)

        t_best, _ = fit_temperature_grid(y_val, p_val_raw)
        p_val_cal = apply_temperature(p_val_raw, t_best)
        p_test_cal = apply_temperature(p_test_raw, t_best)

        policy_row = stage1_eval_policy.get(seed)
        if policy_row is not None:
            tau = float(policy_row["tau"])
            gate_consecutive_k = int(policy_row.get("gate_consecutive_k", 1))
            gate_smooth_window = int(policy_row.get("gate_smooth_window", 1))
            gate_use_t_alert_start = int(policy_row.get("gate_use_t_alert_start", 0))
            gate_policy_name = str(policy_row.get("gate_policy_name", "stage1_eval_csv"))
            print(
                f"[seed {seed}] stage1_eval_policy tau={tau:.6f} "
                f"policy={gate_policy_name} k={gate_consecutive_k} ma={gate_smooth_window} "
                f"use_t_alert_start={gate_use_t_alert_start}"
            )
        else:
            tau, _val_f1, _val_prec, _val_rec = best_tau_by_target(
                y_val,
                p_val_cal,
                mode=tau_mode,
                target_precision=tau_target_precision,
                target_recall=tau_target_recall,
            )
            gate_consecutive_k = 1
            gate_smooth_window = 1
            gate_use_t_alert_start = 0
            gate_policy_name = f"simple_{tau_mode}"

        # Stage1 alerts: no t* condition (use first t* with p>=tau)
        t_alert_start = None
        probs_for_split = p_val_cal if split == "val" else p_test_cal
        if alert_map_csv:
            # Pre-computed alerts (e.g. cascade Stage1a+Stage1b output).
            # Expected columns: site, year, alerted, alert_tstar (Stage1 frame index).
            df_amap = pd.read_csv(alert_map_csv)
            alert_map = {}
            for _, r in df_amap.iterrows():
                if int(r.get("alerted", 0)) != 1:
                    continue
                at = r.get("alert_tstar")
                if at is None or pd.isna(at):
                    continue
                key = f"{r['site']}-{int(r['year'])}"
                alert_map[key] = int(at)
            print(f"[seed {seed}] alert_map loaded from {alert_map_csv}: {len(alert_map)} alerts")
        elif gate_consecutive_k > 1 or gate_smooth_window > 1:
            alert_map = build_alert_map_consecutive(
                samples1,
                probs_for_split,
                tau,
                split_name=split,
                seed=seed,
                t_alert_start=t_alert_start,
                consecutive_k=gate_consecutive_k,
                smooth_window=gate_smooth_window,
            )
        else:
            alert_map = build_alert_map(
                samples1,
                probs_for_split,
                tau,
                t_alert_start,
            )

        # Stage2 model
        if grouped_mode:
            model2 = HierarchicalCausalHazardTransformer(
                d_in=int(val_ds2[0][0].shape[-1]),
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=C.N_LAYERS,
                num_tstar_layers=int(stage2_tstar_layers),
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
                max_tstar_len=512,
                use_tstar_scalar_pos=bool(stage2_use_tstar_scalar_pos),
            ).to(device)
            model2.time_chunk_size = int(stage2_time_chunk_size)
            model2.conditional_survival = bool(stage2_conditional_survival)
            model2.pmf_mode = str(stage2_pmf_mode)
            model2.gaussian_sigma = float(stage2_pmf_sigma)
            model2.gaussian_mu_max = float(stage2_pmf_mu_max)
            model2.asym_weight = float(stage2_pmf_asym_weight)
            model2.right_weight = float(stage2_pmf_right_weight)
            model2.target_offset = float(stage2_pmf_target_offset)
        else:
            model2 = HazardTransformer(
                d_in=int(val_ds2[0][0].shape[-1]),
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=C.N_LAYERS,
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
            ).to(device)
        missing2, unexpected2 = model2.load_state_dict(d2["state_dict"], strict=False)
        if missing2:
            print(f"[viz] model2 missing keys ({len(missing2)}, kept random init): {missing2[:8]}")
        if unexpected2:
            print(f"[viz] model2 unexpected keys ({len(unexpected2)}, ignored): {unexpected2[:8]}")
        model2.eval()

        if grouped_mode:
            # In grouped mode, each site-year group emits one row per t*
            # candidate (~K rows per group). max_pool=300 (single-row default)
            # truncates after the first group or two, breaking alert matching.
            # Auto-raise pool to cover every candidate.
            auto_pool = sum(len(g.get("samples", [])) for g in source_groups2)
            effective_max_pool = max(int(max_pool), int(auto_pool))
            rows = collect_interval_preds_grouped(
                model2,
                loader2,
                source_groups=source_groups2,
                Tend=T2,
                device=device,
                pi_method=getattr(C, "PI_METHOD", "shortest"),
                pi_mass_level=pi_mass_level,
                ablate_feature_indices=ablate_feature_indices,
                max_samples=effective_max_pool,
            )
        else:
            rows = collect_interval_preds(
                model2,
                loader2,
                source_samples=source_samples2,
                Tend=T2,
                device=device,
                pi_method=getattr(C, "PI_METHOD", "shortest"),
                pi_mass_level=pi_mass_level,
                ablate_feature_indices=ablate_feature_indices,
                max_samples=max_pool,
            )
        if not rows:
            print(f"[seed {seed}] no interval samples for visualization.")
            continue

        # Build lookup for stage2 rows by absolute DOY t*.
        row_map = {}
        for r in rows:
            tstar = r.get("tstar")
            if tstar is None:
                continue
            tstar_abs = int(tstar) + stage2_doy_start - 1
            row_map[(r["sample_id"], int(tstar_abs))] = r

        # Interval site-year denominator (unique)
        interval_site_year = {
            (s.get("site_id"), int(s.get("year")))
            for s in source_samples2
            if str(s.get("censor_type", "")) == "interval"
        }
        n_true = len(interval_site_year)

        # Attach alert t* and select stage2 prediction at alert absolute DOY + offset.
        matched_rows = []
        tp = 0
        pred_pos = 0
        lead_times = []
        stage2_lead_times = []
        offset_missed = 0
        offset_after_true_start = 0
        for sid, alert_t in alert_map.items():
            if alert_t is None:
                continue
            alert_abs = int(alert_t) + stage1_doy_start - 1
            stage2_tstar_abs = int(alert_abs) + int(stage2_tstar_offset)
            pred_row = row_map.get((sid, int(stage2_tstar_abs)))
            if pred_row is None:
                offset_missed += 1
                continue
            pred_row = dict(pred_row)
            pred_row["tstar"] = int(stage2_tstar_abs)
            pred_row["true_L"] = int(pred_row["true_L"]) + stage2_doy_start - 1
            pred_row["true_R"] = int(pred_row["true_R"]) + stage2_doy_start - 1
            pred_row["pred_L"] = int(pred_row["pred_L"]) + stage2_doy_start - 1
            pred_row["pred_R"] = int(pred_row["pred_R"]) + stage2_doy_start - 1
            pred_row["pred_point"] = int(pred_row["pred_point"]) + stage2_doy_start - 1
            pred_row["alert_tstar"] = int(alert_abs)
            pred_row["stage2_tstar"] = int(stage2_tstar_abs)
            pred_row["stage2_tstar_offset"] = int(stage2_tstar_offset)
            matched_rows.append(pred_row)
            pred_pos += 1
            hit = (min(pred_row["pred_R"], pred_row["true_R"]) - max(pred_row["pred_L"], pred_row["true_L"])) > 0
            if hit:
                tp += 1
            try:
                true_start = int(pred_row["true_L"]) + 1
                lt = int(true_start) - int(alert_abs)
                lead_times.append(float(lt))
                stage2_lt = int(true_start) - int(stage2_tstar_abs)
                stage2_lead_times.append(float(stage2_lt))
                if stage2_lt <= 0:
                    offset_after_true_start += 1
            except Exception:
                pass

        precision = tp / pred_pos if pred_pos > 0 else 0.0
        recall = tp / n_true if n_true > 0 else 0.0
        early_recall80, early_success, early_denom = early_recall80_site_year(matched_rows)
        interval_stats = summarize_matched_interval_rows(matched_rows, n_true=n_true, doy_start=stage2_doy_start)
        fixed_widths = (14, 21, 30, 45)
        fixed_stats = add_fixed_width_metrics(matched_rows, widths=fixed_widths, doy_start=stage2_doy_start)
        lead_bin_df = summarize_lead_bins(matched_rows, fixed_widths=fixed_widths, doy_start=stage2_doy_start)
        synthetic_row_df, synthetic_summary_df = (
            synthetic_width_ceiling_diag(matched_rows, doy_start=stage2_doy_start, Tend=T2)
            if synthetic_width_diag
            else (pd.DataFrame(), pd.DataFrame())
        )
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        lead_time_mean = float(np.mean(lead_times)) if lead_times else float("nan")
        lead_time_median = float(np.median(lead_times)) if lead_times else float("nan")
        stage2_lead_time_mean = float(np.mean(stage2_lead_times)) if stage2_lead_times else float("nan")
        stage2_lead_time_median = float(np.median(stage2_lead_times)) if stage2_lead_times else float("nan")

        rows_sorted = sorted(matched_rows, key=lambda r: r["iou"], reverse=True)
        top_rows = rows_sorted[:topk]
        worst_rows = list(reversed(rows_sorted[-worstk:])) if rows_sorted else []
        median_k = min(5, len(rows_sorted))
        if median_k > 0:
            mid = len(rows_sorted) // 2
            start = max(0, mid - median_k // 2)
            end = min(len(rows_sorted), start + median_k)
            start = max(0, end - median_k)
            median_rows = rows_sorted[start:end]
        else:
            median_rows = []
        if rows_sorted:
            rng = np.random.default_rng(int(seed))
            rand_rows = rng.choice(rows_sorted, size=min(randomk, len(rows_sorted)), replace=False).tolist()
        else:
            rand_rows = []

        out_dir = Path(out_root) / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        tag = "" if not final_tag else f"_{str(final_tag)}"

        fig_top = plot_interval_rows(top_rows, Tend=stage2_doy_end, title=f"{split.upper()} Top-{len(top_rows)} (seed {seed})")
        fig_worst = plot_interval_rows(worst_rows, Tend=stage2_doy_end, title=f"{split.upper()} Worst-{len(worst_rows)} (seed {seed})")
        fig_rand = plot_interval_rows(rand_rows, Tend=stage2_doy_end, title=f"{split.upper()} Random-{len(rand_rows)} (seed {seed})")
        fig_top_pmf = plot_pmf_rows(top_rows, Tend=stage2_doy_end, title=f"{split.upper()} Top PMF-{len(top_rows)} (seed {seed})")
        fig_worst_pmf = plot_pmf_rows(worst_rows, Tend=stage2_doy_end, title=f"{split.upper()} Worst PMF-{len(worst_rows)} (seed {seed})")
        fig_rand_pmf = plot_pmf_rows(rand_rows, Tend=stage2_doy_end, title=f"{split.upper()} Random PMF-{len(rand_rows)} (seed {seed})")
        fig_top_hazard = plot_hazard_rows(top_rows, Tend=stage2_doy_end, title=f"{split.upper()} Top Hazard-{len(top_rows)} (seed {seed})")
        fig_median_hazard = plot_hazard_rows(median_rows, Tend=stage2_doy_end, title=f"{split.upper()} Median Hazard-{len(median_rows)} (seed {seed})")
        fig_bottom_hazard = plot_hazard_rows(worst_rows, Tend=stage2_doy_end, title=f"{split.upper()} Bottom Hazard-{len(worst_rows)} (seed {seed})")
        reference_rows = []
        if reference_sample_ids:
            ref_order = [s.strip() for s in str(reference_sample_ids).split(",") if s.strip()]
            by_id = {str(r.get("sample_id")): r for r in rows_sorted}
            reference_rows = [by_id[sid] for sid in ref_order if sid in by_id]
        fig_reference_pmf = plot_pmf_rows(reference_rows, Tend=stage2_doy_end, title=f"{split.upper()} Reference PMF-{len(reference_rows)} (seed {seed})")
        fig_reference_hazard = plot_hazard_rows(reference_rows, Tend=stage2_doy_end, title=f"{split.upper()} Reference Hazard-{len(reference_rows)} (seed {seed})")
        fig_hazard_overlay = plot_hazard_overlay(rows_sorted, Tend=stage2_doy_end, title=f"{split.upper()} Hazard Overlay (seed {seed})")
        hazard_diag = summarize_hazard_diagnostics(matched_rows)

        hazard_figs = {
            f"{split}_top_hazard_seed{seed}": fig_top_hazard,
            f"{split}_median_hazard_seed{seed}": fig_median_hazard,
            f"{split}_bottom_hazard_seed{seed}": fig_bottom_hazard,
            f"{split}_reference_hazard_seed{seed}": fig_reference_hazard,
            f"{split}_hazard_overlay_seed{seed}": fig_hazard_overlay,
        }
        interval_figs = {
            f"{split}_top_seed{seed}": fig_top,
            f"{split}_worst_seed{seed}": fig_worst,
            f"{split}_random_seed{seed}": fig_rand,
            f"{split}_top_pmf_seed{seed}": fig_top_pmf,
            f"{split}_worst_pmf_seed{seed}": fig_worst_pmf,
            f"{split}_random_pmf_seed{seed}": fig_rand_pmf,
            f"{split}_reference_pmf_seed{seed}": fig_reference_pmf,
        }
        for name, fig in interval_figs.items():
            if fig is not None:
                fig.savefig(out_dir / f"viz_{name}{tag}.png", dpi=160, bbox_inches="tight")
        for name, fig in hazard_figs.items():
            if fig is not None:
                fig.savefig(out_dir / f"viz_{name}{tag}.png", dpi=160, bbox_inches="tight")

        if hazard_diag:
            hazard_diag_out = out_dir / f"final_viz_{pest}_run{run}_{split}_seed{seed}_off{int(stage2_tstar_offset)}{tag}_hazard_diag.csv"
            pd.DataFrame([hazard_diag]).to_csv(hazard_diag_out, index=False)
            print("saved:", hazard_diag_out)

        if synthetic_width_diag:
            synth_rows_out = out_dir / f"final_viz_{pest}_run{run}_{split}_seed{seed}_off{int(stage2_tstar_offset)}{tag}_synthetic_width_rows.csv"
            synth_summary_out = out_dir / f"final_viz_{pest}_run{run}_{split}_seed{seed}_off{int(stage2_tstar_offset)}{tag}_synthetic_width_summary.csv"
            synthetic_row_df.to_csv(synth_rows_out, index=False)
            synthetic_summary_df.to_csv(synth_summary_out, index=False)
            print("saved:", synth_rows_out)
            print("saved:", synth_summary_out)

        if wandb_run is not None:
            import wandb
            import matplotlib.pyplot as plt

            log_payload = {"seed": int(seed), "split": split}
            if fig_top is not None:
                log_payload[f"viz/{split}_top_seed{seed}"] = wandb.Image(fig_top)
            if fig_worst is not None:
                log_payload[f"viz/{split}_worst_seed{seed}"] = wandb.Image(fig_worst)
            if fig_rand is not None:
                log_payload[f"viz/{split}_random_seed{seed}"] = wandb.Image(fig_rand)
            if fig_top_pmf is not None:
                log_payload[f"viz/{split}_top_pmf_seed{seed}"] = wandb.Image(fig_top_pmf)
            if fig_worst_pmf is not None:
                log_payload[f"viz/{split}_worst_pmf_seed{seed}"] = wandb.Image(fig_worst_pmf)
            if fig_rand_pmf is not None:
                log_payload[f"viz/{split}_random_pmf_seed{seed}"] = wandb.Image(fig_rand_pmf)
            if fig_top_hazard is not None:
                log_payload[f"viz/{split}_top_hazard_seed{seed}"] = wandb.Image(fig_top_hazard)
            if fig_median_hazard is not None:
                log_payload[f"viz/{split}_median_hazard_seed{seed}"] = wandb.Image(fig_median_hazard)
            if fig_bottom_hazard is not None:
                log_payload[f"viz/{split}_bottom_hazard_seed{seed}"] = wandb.Image(fig_bottom_hazard)
            if fig_reference_hazard is not None:
                log_payload[f"viz/{split}_reference_hazard_seed{seed}"] = wandb.Image(fig_reference_hazard)
            if fig_reference_pmf is not None:
                log_payload[f"viz/{split}_reference_pmf_seed{seed}"] = wandb.Image(fig_reference_pmf)
            if fig_hazard_overlay is not None:
                log_payload[f"viz/{split}_hazard_overlay_seed{seed}"] = wandb.Image(fig_hazard_overlay)
            for k, v in hazard_diag.items():
                log_payload[f"final/{k}"] = v

            table_rows = []
            for r in (top_rows + worst_rows + rand_rows):
                table_rows.append(
                    [
                        r["sample_id"],
                        r["true_L"],
                        r["true_R"],
                        r["pred_L"],
                        r["pred_R"],
                        r["pred_point"],
                        r.get("alert_tstar"),
                        r["iou"],
                    ]
                )
            log_payload[f"viz/{split}_samples_seed{seed}"] = wandb.Table(
                columns=["sample_id", "true_L", "true_R", "pred_L", "pred_R", "pred_point", "alert_tstar", "iou"],
                data=table_rows,
            )
            log_payload["final/t_alert_start"] = None
            log_payload["final/tau"] = float(tau)
            log_payload["final/precision"] = float(precision)
            log_payload["final/recall"] = float(recall)
            log_payload["final/early_recall80"] = float(early_recall80)
            log_payload["final/early_recall80_success"] = int(early_success)
            log_payload["final/early_recall80_denominator"] = int(early_denom)
            log_payload["final/f1"] = float(f1)
            for k, v in interval_stats.items():
                log_payload[f"final/{k}"] = float(v)
            for k, v in fixed_stats.items():
                if not str(k).endswith(("_success", "_denominator")):
                    log_payload[f"final/{k}"] = float(v)
            log_payload["final/pred_pos_rate"] = float(pred_pos / max(n_true, 1))
            log_payload["final/lead_time_mean"] = float(lead_time_mean)
            log_payload["final/lead_time_median"] = float(lead_time_median)
            log_payload["final/stage2_lead_time_mean"] = float(stage2_lead_time_mean)
            log_payload["final/stage2_lead_time_median"] = float(stage2_lead_time_median)
            log_payload["final/stage2_tstar_offset"] = int(stage2_tstar_offset)
            log_payload["final/stage2_offset_missed"] = int(offset_missed)
            log_payload["final/stage2_offset_after_true_start"] = int(offset_after_true_start)
            wandb_run.log(log_payload)
            for fig in (
                fig_top,
                fig_worst,
                fig_rand,
                fig_top_pmf,
                fig_worst_pmf,
                fig_rand_pmf,
                fig_top_hazard,
                fig_median_hazard,
                fig_bottom_hazard,
                fig_reference_hazard,
                fig_reference_pmf,
            ):
                if fig is not None:
                    plt.close(fig)
        else:
            import matplotlib.pyplot as plt

            for fig in (
                fig_top,
                fig_worst,
                fig_rand,
                fig_top_pmf,
                fig_worst_pmf,
                fig_rand_pmf,
                fig_top_hazard,
                fig_median_hazard,
                fig_bottom_hazard,
                fig_reference_hazard,
                fig_reference_pmf,
            ):
                if fig is not None:
                    plt.close(fig)

        print(f"[seed {seed}] logged {len(top_rows)}/{len(worst_rows)}/{len(rand_rows)} rows")

        # Save final metrics locally
        out_path = out_dir / f"final_viz_{pest}_run{run}_{split}_seed{seed}_off{int(stage2_tstar_offset)}{tag}.csv"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(
                "seed,split,precision,recall,EarlyRecall80,early_recall80_success,"
                "early_recall80_denominator,f1,IoU80,Rec80,Prec80,MAE_int,Mass_int,"
                "pred_width,interval_hit_precision,interval_hit_recall,interval_hit_f1,"
                "post_true_start_rate,"
                "IoU14,Rec14,Prec14,Hit14,Mass14,EarlyRecall14,"
                "IoU21,Rec21,Prec21,Hit21,Mass21,EarlyRecall21,"
                "IoU30,Rec30,Prec30,Hit30,Mass30,EarlyRecall30,"
                "IoU45,Rec45,Prec45,Hit45,Mass45,EarlyRecall45,"
                "pred_pos_rate,lead_time_mean,"
                "lead_time_median,stage2_lead_time_mean,stage2_lead_time_median,"
                "stage2_tstar_offset,stage2_offset_missed,stage2_offset_after_true_start,"
                "t_alert_start,tau,gate_policy_name,gate_consecutive_k,gate_smooth_window\n"
            )
            t_alert_str = "" if t_alert_start is None else str(int(t_alert_start))
            f.write(
                f"{seed},{split},{precision:.6f},{recall:.6f},{early_recall80:.6f},"
                f"{early_success},{early_denom},{f1:.6f},"
                f"{interval_stats['IoU80']:.6f},{interval_stats['Rec80']:.6f},"
                f"{interval_stats['Prec80']:.6f},{interval_stats['MAE_int']:.6f},"
                f"{interval_stats['Mass_int']:.6f},{interval_stats['pred_width']:.6f},"
                f"{interval_stats['interval_hit_precision']:.6f},{interval_stats['interval_hit_recall']:.6f},"
                f"{interval_stats['interval_hit_f1']:.6f},{interval_stats['post_true_start_rate']:.6f},"
                f"{fixed_stats['IoU14']:.6f},{fixed_stats['Rec14']:.6f},{fixed_stats['Prec14']:.6f},"
                f"{fixed_stats['Hit14']:.6f},{fixed_stats['Mass14']:.6f},{fixed_stats['EarlyRecall14']:.6f},"
                f"{fixed_stats['IoU21']:.6f},{fixed_stats['Rec21']:.6f},{fixed_stats['Prec21']:.6f},"
                f"{fixed_stats['Hit21']:.6f},{fixed_stats['Mass21']:.6f},{fixed_stats['EarlyRecall21']:.6f},"
                f"{fixed_stats['IoU30']:.6f},{fixed_stats['Rec30']:.6f},{fixed_stats['Prec30']:.6f},"
                f"{fixed_stats['Hit30']:.6f},{fixed_stats['Mass30']:.6f},{fixed_stats['EarlyRecall30']:.6f},"
                f"{fixed_stats['IoU45']:.6f},{fixed_stats['Rec45']:.6f},{fixed_stats['Prec45']:.6f},"
                f"{fixed_stats['Hit45']:.6f},{fixed_stats['Mass45']:.6f},{fixed_stats['EarlyRecall45']:.6f},"
                f"{(pred_pos / max(n_true, 1)):.6f},{lead_time_mean:.6f},{lead_time_median:.6f},"
                f"{stage2_lead_time_mean:.6f},{stage2_lead_time_median:.6f},"
                f"{int(stage2_tstar_offset)},{int(offset_missed)},{int(offset_after_true_start)},"
                f"{t_alert_str},{tau:.6f},{gate_policy_name},{gate_consecutive_k},{gate_smooth_window}\n"
            )
        print("saved:", out_path)

        lead_bin_out = out_dir / f"final_viz_{pest}_run{run}_{split}_seed{seed}_off{int(stage2_tstar_offset)}{tag}_lead_bins.csv"
        lead_bin_df.to_csv(lead_bin_out, index=False)
        print("saved:", lead_bin_out)

        # Save sample-level table locally (same rows as W&B table)
        sample_out = out_dir / f"final_viz_{pest}_run{run}_{split}_seed{seed}_off{int(stage2_tstar_offset)}{tag}_samples.csv"
        with open(sample_out, "w", encoding="utf-8") as f:
            f.write(
                "sample_id,true_L,true_R,true_start,pred_L,pred_R,pred_point,"
                "fixed14_L,fixed14_R,fixed21_L,fixed21_R,fixed30_L,fixed30_R,fixed45_L,fixed45_R,"
                "alert_tstar,stage2_tstar,stage2_tstar_offset,iou\n"
            )
            for r in (top_rows + worst_rows + rand_rows):
                alert = "" if r.get("alert_tstar") is None else str(int(r.get("alert_tstar")))
                stage2_t = "" if r.get("stage2_tstar") is None else str(int(r.get("stage2_tstar")))
                f.write(
                    f"{r['sample_id']},{r['true_L']},{r['true_R']},{int(r['true_L']) + 1},"
                    f"{r['pred_L']},{r['pred_R']},{r['pred_point']},"
                    f"{r.get('fixed14_L','')},{r.get('fixed14_R','')},"
                    f"{r.get('fixed21_L','')},{r.get('fixed21_R','')},"
                    f"{r.get('fixed30_L','')},{r.get('fixed30_R','')},"
                    f"{r.get('fixed45_L','')},{r.get('fixed45_R','')},"
                    f"{alert},{stage2_t},{int(stage2_tstar_offset)},{r['iou']:.6f}\n"
                )
        print("saved:", sample_out)

    finish_wandb_run(wandb_run)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, required=True)
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--stage1_ckpt", type=str, default=None,
                   help="Required for the legacy Stage 2 inference path. "
                        "Optional when --selector_per_sample_csv is provided.")
    p.add_argument("--stage1_eval_csv", type=str, default=None)
    p.add_argument("--stage2_ckpt", type=str, default=None,
                   help="Required for the legacy Stage 2 inference path. "
                        "Optional when --selector_per_sample_csv is provided.")
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--split_seed", type=int, default=C.SPLIT_SEED)
    p.add_argument("--split_mode", type=str, default="site", choices=["site", "site_year", "temporal", "year"])
    p.add_argument("--val_year", type=int, default=2022,
                   help="split_mode=year: val = samples whose year == val_year")
    p.add_argument("--test_year_min", type=int, default=2023,
                   help="split_mode=year: test = samples whose year in [test_year_min, test_year_max]")
    p.add_argument("--test_year_max", type=int, default=2024,
                   help="split_mode=year: test = samples whose year in [test_year_min, test_year_max]")
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--auto_split_seed", action="store_true")
    p.add_argument("--auto_split_topk", type=int, default=1)
    p.add_argument("--split_seed_from_topk_idx", type=int, default=None)
    p.add_argument("--split_seeds_json", type=str, default=None)
    p.add_argument("--seed_candidates", type=str, default=None)
    p.add_argument("--target_test_interval", type=int, default=None)
    p.add_argument("--tol_test_interval", type=int, default=None)
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--worstk", type=int, default=5)
    p.add_argument("--randomk", type=int, default=5)
    p.add_argument("--max_pool", type=int, default=300)
    p.add_argument("--stage2_tstar_offset", type=int, default=0)
    p.add_argument("--pi_mass_level", type=float, default=0.8)
    p.add_argument("--synthetic_width_diag", action="store_true")
    p.add_argument("--stage2_ablate_calendar", action="store_true")
    p.add_argument("--stage2_ablate_features", type=str, default=None)
    p.add_argument("--final_tag", type=str, default=None)
    p.add_argument("--reference_sample_ids", type=str, default=None)
    p.add_argument("--tau_mode", type=str, default="f1", choices=["f1", "precision_target", "recall_target"])
    p.add_argument("--tau_target_precision", type=float, default=0.6)
    p.add_argument("--tau_target_recall", type=float, default=0.6)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default=None)
    p.add_argument("--wandb_job_type", type=str, default=None)
    p.add_argument("--alert_map_csv", type=str, default=None,
                   help="Pre-computed alerts CSV (site,year,alerted,alert_tstar). "
                        "If given, overrides Stage1 tau-based alert derivation. "
                        "Used for cascade Stage1a+Stage1b operating points.")
    # --- Phase S11/S12-style selector-aware eval flags ---
    p.add_argument("--selector_per_sample_csv", type=str, default=None,
                   help="phase_s3-style per-sample CSV with (sample_id, L, R, "
                        "t_star_doy, sigma, mu_at_pred_off, pred_off). When "
                        "provided, bypass Stage 2 inference and run the metrics "
                        "+ viz suite directly off this CSV (selector OOF eval).")
    p.add_argument("--cohort_label", type=str, default="selector_eval",
                   help="Cohort label used in metrics_summary.csv and wandb tags.")
    p.add_argument("--n_total_test", type=int, default=575,
                   help="Denominator for IoU_overall / P_ideal etc. (Phase S "
                        "test cohort size = 575).")
    p.add_argument("--sigma_eval", type=float, default=5.0,
                   help="Fallback σ for PI = [μ−1.96σ, μ+1.96σ] when 'sigma' "
                        "column is missing from the selector CSV.")
    p.add_argument("--operational_shift", type=float, default=46.0,
                   help="Operational shift applied to PI for bucket / P_ideal "
                        "computation (lead = L − (μ + 1.96σ − shift)).")
    p.add_argument("--ideal_lead_low", type=float, default=14.0)
    p.add_argument("--ideal_lead_high", type=float, default=30.0)
    p.add_argument("--selector_out_dir", type=str, default=None,
                   help="Output directory for selector eval CSVs + PNGs. "
                        "Default: <out_root>/selector_eval/")
    p.add_argument("--selector_Tend", type=int, default=300,
                   help="Tend used for plot_interval_rows x-axis when in "
                        "selector-eval mode.")
    p.add_argument("--selector_random_grid_n", type=int, default=50,
                   help="Number of random samples for the dense PI grid figure "
                        "in selector mode (default 50).")
    p.add_argument("--selector_random_grid_cols", type=int, default=2,
                   help="Number of columns in the dense PI grid (default 2).")
    args = p.parse_args()
    main(
        pest=args.pest,
        run=args.run,
        stage1_ckpt=args.stage1_ckpt,
        stage1_eval_csv=args.stage1_eval_csv,
        stage2_ckpt=args.stage2_ckpt,
        out_root=args.out_root,
        split_seed=args.split_seed,
        split_mode=args.split_mode,
        val_year=args.val_year,
        test_year_min=args.test_year_min,
        test_year_max=args.test_year_max,
        seeds=args.seeds,
        auto_split_seed=args.auto_split_seed,
        seed_candidates_raw=args.seed_candidates,
        target_test_interval=args.target_test_interval,
        tol_test_interval=args.tol_test_interval,
        auto_split_topk=args.auto_split_topk,
        split_seed_from_topk_idx=args.split_seed_from_topk_idx,
        split_seeds_json=args.split_seeds_json,
        split=args.split,
        topk=args.topk,
        worstk=args.worstk,
        randomk=args.randomk,
        max_pool=args.max_pool,
        stage2_tstar_offset=args.stage2_tstar_offset,
        pi_mass_level=args.pi_mass_level,
        synthetic_width_diag=args.synthetic_width_diag,
        stage2_ablate_calendar=args.stage2_ablate_calendar,
        stage2_ablate_features=args.stage2_ablate_features,
        final_tag=args.final_tag,
        reference_sample_ids=args.reference_sample_ids,
        tau_mode=args.tau_mode,
        tau_target_precision=args.tau_target_precision,
        tau_target_recall=args.tau_target_recall,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        wandb_job_type=args.wandb_job_type,
        alert_map_csv=args.alert_map_csv,
        # selector-aware eval (Phase S11/S12)
        selector_per_sample_csv=args.selector_per_sample_csv,
        cohort_label=args.cohort_label,
        n_total_test=args.n_total_test,
        sigma_eval=args.sigma_eval,
        operational_shift=args.operational_shift,
        ideal_lead_low=args.ideal_lead_low,
        ideal_lead_high=args.ideal_lead_high,
        selector_out_dir=args.selector_out_dir,
        selector_Tend=args.selector_Tend,
        selector_random_grid_n=args.selector_random_grid_n,
        selector_random_grid_cols=args.selector_random_grid_cols,
    )
