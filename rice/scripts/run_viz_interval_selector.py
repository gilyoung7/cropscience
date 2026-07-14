#!/usr/bin/env python3
"""Selector-aware Stage 2 interval visualization (lightweight, CSV-based).

Reads:
  - Stage 2 baseline sample_grid CSV   (mu per (sample, coarse_offset))
  - V2 selector per-sample selections CSV (selected offset per sample × selector)
Produces:
  - Per-sample table with selected offset, predicted mu, 95% PI, true interval, IoU
  - Aggregate figures (offset hist, mu vs true scatter, IoU sorted, lead/bias hist,
    small-multiples interval plot)
  - W&B upload of figures + table + summary stats

Does NOT load the model. Works entirely off the sample_grid + selector CSVs
already produced by phase_b_stage2_offset_selector_v2_ranking.py.

Dense-offset selections (offset not in {7,14,21,30,45,60}) are handled by
linear interpolation of mu(offset) across coarse anchors — same logic the V2
selector uses internally.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SIGMA_DEFAULT = 5.0
Z = 1.96
COARSE_OFFSETS = [7, 14, 21, 30, 45, 60]

# Lead-bin definitions (lead = L − alert_tstar). Match the legacy
# run_viz_interval.py / phase_b series so the bar plot is comparable.
LEAD_BINS = [
    ("<15",    lambda x: x < 15),
    ("15-30",  lambda x: 15 <= x <= 30),
    ("31-45",  lambda x: 31 <= x <= 45),
    ("46-60",  lambda x: 46 <= x <= 60),
    ("61-90",  lambda x: 61 <= x <= 90),
    ("91-120", lambda x: 91 <= x <= 120),
    (">120",   lambda x: x > 120),
]


def _bin_of_lead(lead: float) -> str:
    if not np.isfinite(lead):
        return "NA"
    for name, fn in LEAD_BINS:
        if fn(lead):
            return name
    return "NA"


# ---------------- IoU + PI helpers (mirror selector V2) -------------------
def iou_from_mu(mu: float, L: float, R: float, sigma: float = SIGMA_DEFAULT) -> float:
    if pd.isna(mu) or pd.isna(L) or pd.isna(R):
        return 0.0
    pL = int(round(mu - Z * sigma))
    pR = int(round(mu + Z * sigma))
    tL = int(L) + 1
    tR = int(R)
    lo_hi = min(pR, tR); hi_lo = max(pL, tL)
    ov = (lo_hi - hi_lo + 1) if lo_hi >= hi_lo else 0
    un = max(pR, tR) - min(pL, tL) + 1
    return float(max(0.0, ov / un)) if un > 0 else 0.0


def mu_at_offset(off_mu_coarse: Dict[int, float], offset: int) -> float:
    """Linear-interpolate mu(offset) using valid coarse anchors; clamp outside range."""
    valid = sorted([(o, m) for o, m in off_mu_coarse.items() if not pd.isna(m)])
    if not valid:
        return float("nan")
    xs = np.array([v[0] for v in valid], dtype=float)
    ys = np.array([v[1] for v in valid], dtype=float)
    if offset <= xs[0]: return float(ys[0])
    if offset >= xs[-1]: return float(ys[-1])
    return float(np.interp(offset, xs, ys))


def build_per_sample(sample_grid_csv: Path,
                     selector_offsets_csv: Path,
                     selector_name: str,
                     sigma: float) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """Merge sample_grid + selector per-sample picks; compute PI + IoU + bias."""
    grid = pd.read_csv(sample_grid_csv)
    sel = pd.read_csv(selector_offsets_csv)
    sel = sel[sel["selector"] == selector_name].copy()
    if sel.empty:
        raise SystemExit(f"[abort] selector '{selector_name}' not found in {selector_offsets_csv}. "
                         f"Available: {sorted(pd.read_csv(selector_offsets_csv).selector.unique())}")

    rows = []
    by_sid = dict(tuple(grid.groupby("sample_id")))
    for sid, sel_row in sel.set_index("sample_id").iterrows():
        if sid not in by_sid: continue
        g = by_sid[sid].sort_values("offset")
        head = g.iloc[0]
        off_mu = {int(o): float(m) if not pd.isna(m) else float("nan")
                  for o, m in zip(g["offset"], g["mu"])}
        selected_off = int(sel_row["offset"])
        mu_sel = mu_at_offset(off_mu, selected_off)
        L = float(head["L"]); R = float(head["R"])
        tL = int(L) + 1; tR = int(R)
        mid = 0.5 * (L + R)
        if pd.isna(mu_sel):
            pL = pR = float("nan"); iou = 0.0
        else:
            pL = int(round(mu_sel - Z * sigma))
            pR = int(round(mu_sel + Z * sigma))
            iou = iou_from_mu(mu_sel, L, R, sigma)
        alert_tstar = float(head.get("alert_tstar", float("nan")))
        # Early-or-Inside @ 30 days early tolerance: mu in [L+1 - 30, R]
        early_or_inside_30 = ((not pd.isna(mu_sel)) and
                              (mu_sel >= (tL - 30)) and (mu_sel <= tR))
        # lead = (L+1) - mu  (positive = early)
        lead = (tL - mu_sel) if not pd.isna(mu_sel) else float("nan")
        rows.append({
            "sample_id": sid,
            "site": str(head.get("site", "")),
            "year": int(head["year"]) if not pd.isna(head["year"]) else -1,
            "alert_tstar": alert_tstar,
            "selected_offset": selected_off,
            "selector": selector_name,
            "mu": mu_sel,
            "pred_L": pL, "pred_R": pR,
            "true_L_plus_1": tL, "true_R": tR, "true_mid": mid,
            "iou": iou,
            "mu_minus_mid": (mu_sel - mid) if not pd.isna(mu_sel) else float("nan"),
            "lead_days": lead,
            "early_or_inside_30": bool(early_or_inside_30),
        })
    df = pd.DataFrame(rows)
    n_total = len(df)
    summary = {
        "n_total": int(n_total),
        "mean_iou": float(df["iou"].mean()) if n_total else 0.0,
        "median_iou": float(df["iou"].median()) if n_total else 0.0,
        "frac_iou_gt_0_2": float((df["iou"] > 0.2).mean()) if n_total else 0.0,
        "frac_early_or_inside_30": float(df["early_or_inside_30"].mean()) if n_total else 0.0,
        "mu_minus_mid_mean": float(df["mu_minus_mid"].mean()) if n_total else float("nan"),
        "mu_minus_mid_median": float(df["mu_minus_mid"].median()) if n_total else float("nan"),
        "lead_days_mean": float(df["lead_days"].mean()) if n_total else float("nan"),
        "lead_days_median": float(df["lead_days"].median()) if n_total else float("nan"),
    }
    return df, summary


# ----------------------- Figures ------------------------------------------
def fig_offset_histogram(df: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(7, 3.5))
    bins = np.arange(0, 78, 2)
    ax.hist(df["selected_offset"], bins=bins, color="tab:blue", alpha=0.75, edgecolor="white")
    for o in COARSE_OFFSETS:
        ax.axvline(o, color="grey", lw=0.6, ls="--", alpha=0.5)
    ax.set_xlabel("selected offset (days after alert_tstar)")
    ax.set_ylabel("# samples")
    ax.set_title(title)
    fig.tight_layout()
    return fig


def fig_calibration_scatter(df: pd.DataFrame, title: str,
                             color_mode: str = "iou") -> plt.Figure:
    """μ vs (L+R)/2 scatter (legacy plot_calibration_scatter format).
    color_mode='iou' → viridis colored by per-sample IoU.
    color_mode='density' → plasma colored by local point density (KDE).
    """
    sub = df.dropna(subset=["mu", "true_mid"])
    if sub.empty:
        fig, ax = plt.subplots(figsize=(7, 7))
        ax.text(0.5, 0.5, "no samples", ha="center", va="center")
        return fig
    mid = sub["true_mid"].to_numpy(dtype=float)
    mu = sub["mu"].to_numpy(dtype=float)
    if str(color_mode).lower() == "density":
        try:
            from scipy.stats import gaussian_kde
            xy = np.vstack([mid, mu])
            if np.std(mid) < 1e-6 or np.std(mu) < 1e-6:
                rng = np.random.default_rng(0)
                xy = xy + rng.normal(0.0, 1e-3, size=xy.shape)
            c = gaussian_kde(xy)(xy)
        except Exception:
            H, x_e, y_e = np.histogram2d(mid, mu, bins=40)
            x_idx = np.clip(np.searchsorted(x_e, mid) - 1, 0, H.shape[0] - 1)
            y_idx = np.clip(np.searchsorted(y_e, mu)  - 1, 0, H.shape[1] - 1)
            c = H[x_idx, y_idx]
        cmap = "plasma"; cbar_label = "point density"
        vmin = vmax = None
    else:
        c = sub["iou"].to_numpy(dtype=float)
        cmap = "viridis"; cbar_label = "IoU"
        vmin, vmax = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(7, 7))
    sc = ax.scatter(mid, mu, c=c, cmap=cmap, s=18, alpha=0.75,
                    edgecolor="none", vmin=vmin, vmax=vmax)
    lo = float(min(mid.min(), mu.min())) - 5.0
    hi = float(max(mid.max(), mu.max())) + 5.0
    ax.plot([lo, hi], [lo, hi], color="black", lw=1.0, ls=":",
            label="identity (perfect calibration)")
    if len(sub) >= 2:
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


def fig_interval_rows(rows: list, title: str, Tend: int = 366) -> plt.Figure:
    """Legacy plot_interval_rows format — n samples, single column.
    black thick line   = true interval [L+1, R]
    blue line + dot    = predicted PI [μ-1.96σ, μ+1.96σ] + μ
    red dashed vline   = alert t* (Stage 1)
    orange dotted vline = Stage 2 eval t* (alert + selected offset)
    """
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no samples", ha="center", va="center")
        return fig
    n = len(rows)
    fig_h = max(2.0, 1.1 * n)
    fig, axes = plt.subplots(n, 1, figsize=(10, fig_h), sharex=True)
    if n == 1:
        axes = [axes]
    for i, (ax, r) in enumerate(zip(axes, rows)):
        ax.hlines(0, r["true_L_plus_1"], r["true_R"], color="black", lw=6,
                  alpha=0.25, label="true interval")
        ax.hlines(0, r["pred_L"], r["pred_R"], color="tab:blue", lw=3,
                  label="pred interval")
        ax.plot(r["mu"], 0, marker="o", color="tab:blue", ms=5, label="pred μ")
        alert = r.get("alert_tstar")
        if alert is not None and not pd.isna(alert):
            ax.axvline(int(alert), color="tab:red", lw=1.2, ls="--",
                       label="alert t* (Stage 1)")
        s2_t = (int(alert) + int(r["selected_offset"])
                if (alert is not None and not pd.isna(alert)) else None)
        if s2_t is not None and s2_t != int(alert):
            ax.axvline(s2_t, color="tab:orange", lw=1.2, ls=":",
                       label="Stage 2 eval t* (alert+offset)")
        ax.set_yticks([])
        ax.set_xlim(1, Tend)
        ax.set_title(f"{r['sample_id']} | off={int(r['selected_offset'])} | "
                     f"IoU={r['iou']:.2f}", fontsize=8)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.85)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def fig_interval_grid(rows: list, title: str, Tend: int = 366,
                      n_cols: int = 2) -> plt.Figure:
    """Legacy plot_interval_grid format — multi-column dense grid."""
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no samples", ha="center", va="center")
        return fig
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
            ax.axis("off"); continue
        r = rows[idx]
        ax.hlines(0, r["true_L_plus_1"], r["true_R"], color="black", lw=4, alpha=0.30)
        ax.hlines(0, r["pred_L"], r["pred_R"], color="tab:blue", lw=2)
        ax.plot(r["mu"], 0, marker="o", color="tab:blue", ms=3)
        alert = r.get("alert_tstar")
        if alert is not None and not pd.isna(alert):
            ax.axvline(int(alert), color="tab:red", lw=0.8, ls="--")
            s2_t = int(alert) + int(r["selected_offset"])
            if s2_t != int(alert):
                ax.axvline(s2_t, color="tab:orange", lw=0.8, ls=":")
        ax.set_yticks([])
        ax.set_xlim(1, Tend)
        ax.set_title(f"{r['sample_id']} | off={int(r['selected_offset'])} | "
                     f"IoU={r['iou']:.2f}", fontsize=7, pad=1)
    handles = [
        plt.Line2D([0], [0], color="black", lw=4, alpha=0.30, label="true [L+1, R]"),
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


def compute_lead_bin_table(df: pd.DataFrame) -> pd.DataFrame:
    """Per lead-bin (lead = L − alert_tstar) IoU/MAE/RMSE/n table."""
    sub = df.dropna(subset=["mu"]).copy()
    if sub.empty:
        return pd.DataFrame(columns=["lead_bin", "n", "IoU_mean", "MAE", "RMSE"])
    sub["lead_from_alert"] = (sub["true_L_plus_1"] - 1) - sub["alert_tstar"]
    sub["lead_bin"] = sub["lead_from_alert"].apply(_bin_of_lead)
    bin_order = [n for n, _ in LEAD_BINS] + ["NA"]
    rows = []
    for b in bin_order:
        rs = sub[sub["lead_bin"] == b]
        n = len(rs)
        if n == 0:
            rows.append({"lead_bin": b, "n": 0, "IoU_mean": float("nan"),
                         "ME": float("nan"), "MAE": float("nan"), "RMSE": float("nan")})
            continue
        me = (rs["mu"] - rs["true_mid"]).to_numpy()
        rows.append({
            "lead_bin": b, "n": int(n),
            "IoU_mean": float(rs["iou"].mean()),
            "ME":   float(np.mean(me)),
            "MAE":  float(np.mean(np.abs(me))),
            "RMSE": float(np.sqrt(np.mean(me * me))),
        })
    return pd.DataFrame(rows)


def fig_lead_bin_bars(lead_df: pd.DataFrame, title: str) -> plt.Figure:
    """IoU mean + MAE mean per lead bin (legacy two-panel bar)."""
    df = lead_df[lead_df["n"] > 0].copy()
    if df.empty:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        return fig
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
    ax_mae.set_ylabel("MAE [days]")
    ax_mae.set_xlabel("lead bin (L − alert_tstar, days)")
    fig.tight_layout()
    return fig


def fig_iou_sorted(df: pd.DataFrame, title: str) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 3.5))
    iou_sorted = df["iou"].sort_values(ascending=False).reset_index(drop=True)
    ax.bar(range(len(iou_sorted)), iou_sorted.values, width=1.0,
           color=["tab:green" if x > 0 else "tab:red" for x in iou_sorted.values])
    ax.axhline(float(iou_sorted.mean()), color="black", lw=0.8, ls="--",
               label=f"mean = {iou_sorted.mean():.3f}")
    ax.set_xlabel("sample (sorted by IoU desc)")
    ax.set_ylabel("IoU")
    ax.set_title(title)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_ylim(0, max(0.05, iou_sorted.max() * 1.05))
    fig.tight_layout()
    return fig


def fig_lead_bias(df: pd.DataFrame, title: str) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    # lead = (true_L+1) - μ (positive = early)
    axes[0].hist(df["lead_days"].dropna(), bins=30, color="tab:blue",
                 alpha=0.75, edgecolor="white")
    axes[0].axvline(0, color="red", lw=1.0, ls="--", label="μ = true_L+1")
    axes[0].axvline(30, color="orange", lw=1.0, ls="--", label="30-day early tolerance")
    axes[0].set_xlabel("lead = (true_L + 1) − μ  [days, positive = early]")
    axes[0].set_ylabel("# samples")
    axes[0].set_title("Lead distribution")
    axes[0].legend(fontsize=8)
    # mu - mid
    axes[1].hist(df["mu_minus_mid"].dropna(), bins=30, color="tab:purple",
                 alpha=0.75, edgecolor="white")
    axes[1].axvline(0, color="red", lw=1.0, ls="--", label="μ = mid")
    axes[1].set_xlabel("μ − mid  [days, negative = early bias]")
    axes[1].set_ylabel("# samples")
    axes[1].set_title("μ vs true mid (bias)")
    axes[1].legend(fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def gaussian_pmf(mu: float, sigma: float, Tend: int) -> np.ndarray:
    """Discretized Gaussian PMF over t = 1..Tend, sum-normalized.
    Closed-form reconstruction (no model forward needed; the gaussian mu-head
    used at training time produces exactly this PMF post-hoc via σ=5.0)."""
    t = np.arange(1, int(Tend) + 1, dtype=float)
    log_p = -0.5 * ((t - float(mu)) / float(sigma)) ** 2
    log_p -= log_p.max()  # for numerical stability before exp
    p = np.exp(log_p)
    s = p.sum()
    return p / s if s > 0 else p


def fig_pmf_rows(rows: list, title: str, sigma: float, Tend: int = 366) -> plt.Figure:
    """Legacy plot_pmf_rows format — n samples, single column, PMF lines.
    blue line              = Gaussian PMF (μ, σ=5 fixed)
    black axvspan          = true interval [L+1, R]
    blue axvspan           = pred PI [μ ± 1.96σ]
    blue dotted vline      = μ
    red dashed vline       = alert t* (Stage 1)
    orange dotted vline    = Stage 2 eval t* (alert + selected offset)
    """
    if not rows:
        fig, ax = plt.subplots(figsize=(6, 2))
        ax.text(0.5, 0.5, "no samples", ha="center", va="center")
        return fig
    n = len(rows)
    fig_h = max(2.4, 1.35 * n)
    fig, axes = plt.subplots(n, 1, figsize=(10, fig_h), sharex=True)
    if n == 1:
        axes = [axes]
    x = np.arange(1, int(Tend) + 1)
    for i, (ax, r) in enumerate(zip(axes, rows)):
        pmf = gaussian_pmf(float(r["mu"]), float(sigma), int(Tend))
        ax.plot(x, pmf, color="tab:blue", lw=1.2, label="pmf")
        ax.axvspan(int(r["true_L_plus_1"]), int(r["true_R"]),
                   color="black", alpha=0.12, label="true interval")
        ax.axvspan(int(r["pred_L"]), int(r["pred_R"]),
                   color="tab:blue", alpha=0.12, label="pred 95% PI")
        ax.axvline(float(r["mu"]), color="tab:blue", lw=1.0, ls=":", label="μ")
        alert = r.get("alert_tstar")
        if alert is not None and not pd.isna(alert):
            ax.axvline(int(alert), color="tab:red", lw=1.0, ls="--",
                       label="alert t* (Stage 1)")
            s2_t = int(alert) + int(r["selected_offset"])
            if s2_t != int(alert):
                ax.axvline(s2_t, color="tab:orange", lw=1.0, ls=":",
                           label="Stage 2 eval t* (alert+offset)")
        ax.set_xlim(1, Tend)
        ax.set_ylabel("pmf")
        ax.set_title(f"{r['sample_id']} | off={int(r['selected_offset'])} | "
                     f"IoU={r['iou']:.2f}", fontsize=8)
        if i == 0:
            ax.legend(loc="upper right", fontsize=7, framealpha=0.85, ncol=2)
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def select_top_worst_random(df: pd.DataFrame, topk: int, worstk: int,
                             randomk: int, random_grid_n: int,
                             seed: int = 0) -> Tuple[list, list, list, list]:
    """Build top-K / worst-K / random-K / random_grid lists of per-sample dicts
    (sorted) for use by fig_interval_rows / fig_interval_grid."""
    sub = df.dropna(subset=["mu"]).copy()
    if sub.empty:
        return [], [], [], []
    sub = sub.sort_values("iou", ascending=False).reset_index(drop=True)
    rows = sub.to_dict("records")
    top = rows[:int(topk)]
    worst = list(reversed(rows[-int(worstk):])) if len(rows) >= worstk else []
    rng_small = np.random.default_rng(int(seed))
    rng_grid = np.random.default_rng(int(seed) + 1)
    rand_small = []; rand_grid = []
    if rows:
        n = len(rows)
        idx_s = rng_small.choice(n, size=min(int(randomk), n), replace=False)
        rand_small = [rows[int(i)] for i in idx_s]
        idx_g = rng_grid.choice(n, size=min(int(random_grid_n), n), replace=False)
        rand_grid = [rows[int(i)] for i in idx_g]
    return top, worst, rand_small, rand_grid


# ----------------------- W&B + main ---------------------------------------
def maybe_init_wandb(args):
    if not args.wandb_project:
        return None
    try:
        import wandb
    except ImportError:
        print("[warn] wandb not installed; skipping upload")
        return None
    run = wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity or None,
        name=args.wandb_run_name or f"{args.pest}_{args.year}_{args.selector_name}",
        group=args.wandb_group or None,
        job_type="interval_viz_selector",
        config={
            "pest": args.pest, "year": args.year,
            "selector_name": args.selector_name,
            "sigma": args.sigma,
            "sample_grid": args.sample_grid,
            "selector_offsets": args.selector_offsets,
        },
    )
    return run


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", required=True)
    ap.add_argument("--year", type=int, required=True)
    ap.add_argument("--sample_grid", required=True,
                    help="Stage 2 lead_v3_test_sample_grid.csv (mu per (sample, coarse_offset))")
    ap.add_argument("--selector_offsets", required=True,
                    help="V2 v2_per_sample_test_selections.csv (selected offset per sample × selector)")
    ap.add_argument("--selector_name", default="v2_regressor_dense",
                    help="Which selector variant to visualize (column 'selector' value).")
    ap.add_argument("--sigma", type=float, default=SIGMA_DEFAULT,
                    help="Gaussian sigma for PI (default 5).")
    ap.add_argument("--out_dir", default=None,
                    help="Local output dir for figures + CSV. Default: "
                         "rice/outputs_viz_selector/<pest>_<year>_<selector>/")
    ap.add_argument("--topk", type=int, default=10,
                    help="K for top-K-by-IoU interval plot (legacy plot_interval_rows).")
    ap.add_argument("--worstk", type=int, default=10,
                    help="K for worst-K-by-IoU interval plot.")
    ap.add_argument("--randomk", type=int, default=10,
                    help="K for random sample interval plot (small set).")
    ap.add_argument("--random_grid_n", type=int, default=50,
                    help="Number of samples in dense random grid (legacy plot_interval_grid).")
    ap.add_argument("--random_grid_n_cols", type=int, default=2,
                    help="Column count for the random grid plot.")
    ap.add_argument("--seed", type=int, default=0,
                    help="RNG seed for random sample selection (reproducible).")
    ap.add_argument("--Tend", type=int, default=366,
                    help="DOY x-axis upper bound for interval plots.")
    ap.add_argument("--mode", default="selector", choices=["selector", "fixed"],
                    help="'selector' = use --selector_name picks. 'fixed' = override all "
                         "samples to a single --fixed_offset (for comparison).")
    ap.add_argument("--fixed_offset", type=int, default=None,
                    help="Used when --mode=fixed. The single offset to apply per sample.")
    ap.add_argument("--wandb_project", default=None)
    ap.add_argument("--wandb_entity", default=None)
    ap.add_argument("--wandb_run_name", default=None)
    ap.add_argument("--wandb_group", default=None)
    args = ap.parse_args()

    if args.out_dir is None:
        args.out_dir = f"rice/outputs_viz_selector/{args.pest}_{args.year}_{args.selector_name}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # --- mode handling: build a "selector_offsets" view either from the
    # supplied selector CSV (mode=selector) or by faking it as a single fixed
    # offset for all samples in the sample_grid (mode=fixed).
    if args.mode == "fixed":
        if args.fixed_offset is None:
            print("[abort] --mode=fixed requires --fixed_offset", file=sys.stderr)
            return 2
        grid = pd.read_csv(args.sample_grid)
        sids = grid["sample_id"].unique()
        fake_sel = pd.DataFrame({
            "sample_id": sids,
            "offset": int(args.fixed_offset),
            "iou": float("nan"), "score": float("nan"),
            "selector": f"fixed_offset_{int(args.fixed_offset)}",
        })
        fake_path = out_dir / "_fake_fixed_selector.csv"
        fake_sel.to_csv(fake_path, index=False)
        selector_offsets_csv = fake_path
        selector_name = f"fixed_offset_{int(args.fixed_offset)}"
    else:
        selector_offsets_csv = Path(args.selector_offsets)
        selector_name = args.selector_name

    print(f"[viz] pest={args.pest}  year={args.year}  selector={selector_name}  sigma={args.sigma}")
    df, summary = build_per_sample(
        Path(args.sample_grid), selector_offsets_csv, selector_name, args.sigma)
    print(f"[viz] n_samples={summary['n_total']}  "
          f"mean_iou={summary['mean_iou']:.3f}  "
          f"frac_iou>0.2={summary['frac_iou_gt_0_2']:.3f}  "
          f"frac_early_or_inside_30={summary['frac_early_or_inside_30']:.3f}")

    # Save the per-sample table locally.
    per_sample_csv = out_dir / "per_sample.csv"
    df.to_csv(per_sample_csv, index=False)
    print(f"[wrote] {per_sample_csv}")

    # Build figures.
    title = f"{args.pest} test={args.year} | selector={selector_name} | σ={args.sigma} (95% PI)"
    # Legacy-style top/worst/random/grid sample lists.
    top_rows, worst_rows, rand_small, rand_grid = select_top_worst_random(
        df, args.topk, args.worstk, args.randomk, args.random_grid_n,
        seed=args.seed,
    )
    lead_df = compute_lead_bin_table(df)
    lead_csv = out_dir / "metrics_by_lead_bin.csv"
    lead_df.to_csv(lead_csv, index=False)
    print(f"[wrote] {lead_csv}")

    figs = {
        # selector-specific overview
        "offset_histogram":     fig_offset_histogram(df, title),
        "iou_sorted":           fig_iou_sorted(df, title),
        "lead_bias":            fig_lead_bias(df, title),
        # legacy calibration + lead-bin + interval plots
        "calibration_iou":      fig_calibration_scatter(df, title, color_mode="iou"),
        "calibration_density":  fig_calibration_scatter(df, title, color_mode="density"),
        "lead_bin_bars":        fig_lead_bin_bars(lead_df, title + "  — Lead-bin IoU / MAE"),
        "top_interval":         fig_interval_rows(top_rows,
                                  title=f"{title}  — Top-{len(top_rows)} IoU",
                                  Tend=args.Tend),
        "worst_interval":       fig_interval_rows(worst_rows,
                                  title=f"{title}  — Worst-{len(worst_rows)} IoU",
                                  Tend=args.Tend),
        "random_interval":      fig_interval_rows(rand_small,
                                  title=f"{title}  — Random-{len(rand_small)}",
                                  Tend=args.Tend),
        "random_interval_grid": fig_interval_grid(rand_grid,
                                  title=f"{title}  — Random-{len(rand_grid)} grid",
                                  Tend=args.Tend, n_cols=args.random_grid_n_cols),
        # legacy PMF plots (Gaussian closed-form reconstruction; σ fixed)
        "top_pmf":              fig_pmf_rows(top_rows,
                                  title=f"{title}  — Top-{len(top_rows)} PMF",
                                  sigma=args.sigma, Tend=args.Tend),
        "worst_pmf":            fig_pmf_rows(worst_rows,
                                  title=f"{title}  — Worst-{len(worst_rows)} PMF",
                                  sigma=args.sigma, Tend=args.Tend),
        "random_pmf":           fig_pmf_rows(rand_small,
                                  title=f"{title}  — Random-{len(rand_small)} PMF",
                                  sigma=args.sigma, Tend=args.Tend),
    }
    for name, fig in figs.items():
        png = out_dir / f"fig_{name}.png"
        fig.savefig(png, dpi=130, bbox_inches="tight")
        plt.close(fig)
        print(f"[wrote] {png}")

    # W&B upload.
    run = maybe_init_wandb(args) if args.wandb_project else None
    if run is not None:
        import wandb
        # Re-open figures by reading the saved PNGs (easier than keeping handles)
        log_payload = {}
        for name in figs:
            png = out_dir / f"fig_{name}.png"
            log_payload[f"viz/{name}"] = wandb.Image(str(png))
        # Per-sample table
        # Convert bool to int so wandb table renders cleanly
        wandb_df = df.copy()
        wandb_df["early_or_inside_30"] = wandb_df["early_or_inside_30"].astype(int)
        log_payload["table/per_sample"] = wandb.Table(dataframe=wandb_df)
        log_payload["table/metrics_by_lead_bin"] = wandb.Table(dataframe=lead_df)
        run.log(log_payload)
        for k, v in summary.items():
            run.summary[f"final/{k}"] = v
        run.summary["final/pest"] = args.pest
        run.summary["final/year"] = args.year
        run.summary["final/selector_name"] = selector_name
        # Save CSV artifacts.
        try:
            run.save(str(per_sample_csv), policy="now")
            run.save(str(lead_csv), policy="now")
        except Exception as e:
            print(f"[wandb] save() failed (non-fatal): {e}")
        run.finish()
        print(f"[wandb] logged figures, table, summary -> {args.wandb_project}")
    else:
        print("[info] --wandb_project not set; figures + CSV saved locally only.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
