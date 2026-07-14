#!/usr/bin/env python3
"""
Stage 2 mu distribution diagnostic.

Compares predicted mu distributions across multiple Stage 2 sample_grid CSVs
(e.g. baseline asym_mse, full interval_nll, event-only interval_nll).

For each (label, csv) and for a fixed activation offset, reports:
  * n samples (matched & unmatched), mu mean / std / min / max / pct{5,50,95}
  * Pearson correlation of mu with true_L, true_R, true_mid, alert_tstar
  * fraction of samples whose mu lies inside [L, R] (point-coverage)
  * IoU @ that offset (n_total denominator) for quick cross-check vs the
    canonical summary

Then prints a side-by-side comparison table so collapse / shift / coverage
shifts are obvious at a glance.

Usage:
  python -m rice.scripts.phase_b_stage2_mu_distribution_diag \
      --grid "baseline_asym_mse:rice/outputs_stage2_batch_2024_bestgate/sheath_blight/lead_v3_test_sample_grid.csv" \
      --grid "full_intnll:rice/outputs_stage2_batch_2024_intervalnll/sheath_blight/lead_v3_test_sample_grid.csv" \
      --grid "event_only_intnll:rice/outputs_stage2_batch_2024_intervalnll_eventonly/sheath_blight/lead_v3_test_sample_grid.csv" \
      --offset 14 \
      --out rice/outputs_stage2_batch_2024_intervalnll_eventonly/_summary/mu_diag.txt
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import List, Tuple
import numpy as np
import pandas as pd

SIGMA = 5.0
Z = 1.96


def iou_from_mu(mu, L, R, sigma=SIGMA):
    if pd.isna(mu) or pd.isna(L) or pd.isna(R):
        return 0.0
    pL = int(round(mu - Z * sigma))
    pR = int(round(mu + Z * sigma))
    tL = int(L) + 1
    tR = int(R)
    lo_hi = min(pR, tR)
    hi_lo = max(pL, tL)
    ov = max(0, lo_hi - hi_lo + 1) if lo_hi >= hi_lo else 0
    un = max(pR, tR) - min(pL, tL) + 1
    return float(max(0.0, ov / un)) if un > 0 else 0.0


def _safe_corr(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    a_, b_ = a[m], b[m]
    if a_.std() < 1e-12 or b_.std() < 1e-12:
        return float("nan")
    return float(np.corrcoef(a_, b_)[0, 1])


def diag_one(label: str, csv_path: Path, offset: int) -> dict:
    if not csv_path.exists():
        return {"label": label, "path": str(csv_path), "error": "missing"}
    df = pd.read_csv(csv_path)
    if "offset" in df.columns:
        sub = df[df["offset"] == int(offset)].copy()
    else:
        sub = df.copy()
    n_total = len(sub)
    if n_total == 0:
        return {"label": label, "path": str(csv_path), "error": f"no rows at offset={offset}"}

    mu = pd.to_numeric(sub.get("mu"), errors="coerce").to_numpy()
    L = pd.to_numeric(sub.get("L"), errors="coerce").to_numpy()
    R = pd.to_numeric(sub.get("R"), errors="coerce").to_numpy()
    tstar = pd.to_numeric(sub.get("alert_tstar"), errors="coerce").to_numpy()
    iou_csv = pd.to_numeric(sub.get("iou_matched"), errors="coerce").to_numpy()
    mid = (L + R) / 2.0
    mu_finite_mask = np.isfinite(mu)
    n_matched = int(mu_finite_mask.sum())

    mu_f = mu[mu_finite_mask]
    L_f = L[mu_finite_mask]
    R_f = R[mu_finite_mask]
    mid_f = mid[mu_finite_mask]
    tstar_f = tstar[mu_finite_mask]

    # point-coverage: fraction of samples whose mu sits in [L, R]
    in_LR = ((mu_f >= L_f) & (mu_f <= R_f)).mean() if n_matched else float("nan")
    # IoU sum from csv (already computed) — divide by n_total to mirror
    # canonical "overall_n_total" definition.
    iou_overall_n_total = float(np.nansum(iou_csv) / n_total) if n_total else 0.0
    # also recompute from mu to cross-check formula
    iou_recalc = np.array([iou_from_mu(m, l, r) for m, l, r in zip(mu, L, R)])
    iou_recalc_overall_n_total = float(iou_recalc.sum() / n_total) if n_total else 0.0

    return {
        "label": label,
        "path": str(csv_path),
        "offset": int(offset),
        "n_total": n_total,
        "n_matched": n_matched,
        "mu_mean": float(np.nanmean(mu_f)) if n_matched else float("nan"),
        "mu_std": float(np.nanstd(mu_f)) if n_matched else float("nan"),
        "mu_min": float(np.nanmin(mu_f)) if n_matched else float("nan"),
        "mu_p05": float(np.nanpercentile(mu_f, 5)) if n_matched else float("nan"),
        "mu_p50": float(np.nanpercentile(mu_f, 50)) if n_matched else float("nan"),
        "mu_p95": float(np.nanpercentile(mu_f, 95)) if n_matched else float("nan"),
        "mu_max": float(np.nanmax(mu_f)) if n_matched else float("nan"),
        "mu_minus_mid_mean": float(np.nanmean(mu_f - mid_f)) if n_matched else float("nan"),
        "mu_minus_L_mean":   float(np.nanmean(mu_f - L_f))   if n_matched else float("nan"),
        "corr_mu_L":     _safe_corr(mu_f, L_f),
        "corr_mu_R":     _safe_corr(mu_f, R_f),
        "corr_mu_mid":   _safe_corr(mu_f, mid_f),
        "corr_mu_tstar": _safe_corr(mu_f, tstar_f),
        "frac_in_LR": float(in_LR),
        "iou_overall_n_total_from_csv":  iou_overall_n_total,
        "iou_overall_n_total_recalc":    iou_recalc_overall_n_total,
    }


def collapse_flag(d: dict) -> str:
    tags = []
    if d.get("mu_std", float("nan")) < 5.0:
        tags.append("MU_COLLAPSE")
    if d.get("corr_mu_mid", float("nan")) < 0.2:
        tags.append("LOW_TIMING_SIGNAL")
    if d.get("frac_in_LR", float("nan")) < 0.1:
        tags.append("LOW_POINT_COVERAGE")
    return "+".join(tags) if tags else "OK"


def format_text_report(rows: List[dict], offset: int) -> str:
    lines: List[str] = []
    lines.append(f"Stage 2 mu distribution diagnostic  (offset={offset}, sigma={SIGMA})")
    lines.append("=" * 100)
    lines.append("")
    lines.append("Per-model summary:")
    header = (
        f"  {'label':24s} {'n':>4s} {'n_match':>7s}  "
        f"{'mu_mean':>8s} {'mu_std':>7s} {'mu_min':>7s} {'mu_p50':>7s} {'mu_max':>7s}  "
        f"{'frac_in_LR':>10s}  "
        f"{'ρ(μ,L)':>7s} {'ρ(μ,R)':>7s} {'ρ(μ,mid)':>9s} {'ρ(μ,t*)':>9s}  "
        f"{'IoU_csv':>8s} {'flag':<22s}"
    )
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for d in rows:
        if "error" in d:
            lines.append(f"  {d['label']:24s}  ERROR: {d['error']}  ({d['path']})")
            continue
        flag = collapse_flag(d)
        lines.append(
            f"  {d['label']:24s} {d['n_total']:>4d} {d['n_matched']:>7d}  "
            f"{d['mu_mean']:>8.2f} {d['mu_std']:>7.2f} {d['mu_min']:>7.1f} {d['mu_p50']:>7.1f} {d['mu_max']:>7.1f}  "
            f"{d['frac_in_LR']:>10.3f}  "
            f"{d['corr_mu_L']:>7.3f} {d['corr_mu_R']:>7.3f} {d['corr_mu_mid']:>9.3f} {d['corr_mu_tstar']:>9.3f}  "
            f"{d['iou_overall_n_total_from_csv']:>8.3f} {flag:<22s}"
        )

    lines.append("")
    lines.append("Reading guide:")
    lines.append("  * mu_std < 5   → μ collapse (model output near-constant)")
    lines.append("  * ρ(μ, mid) < 0.2 → predictions don't track event timing")
    lines.append("  * frac_in_LR < 0.1 → μ rarely lands inside the true interval")
    lines.append("  * mu_minus_mid_mean ≫ 0 → systematic late bias; ≪ 0 → early bias")
    lines.append("")
    lines.append("μ - center summary (event timing offset / bias):")
    for d in rows:
        if "error" in d: continue
        lines.append(
            f"  {d['label']:24s}  "
            f"mu - mid mean = {d['mu_minus_mid_mean']:+7.2f}    "
            f"mu - L   mean = {d['mu_minus_L_mean']:+7.2f}"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", action="append", required=True,
                    help='Pairs "label:path/to/sample_grid.csv". Repeat for each model. '
                         'Order in the output table mirrors invocation order.')
    ap.add_argument("--offset", type=int, default=14,
                    help="Activation offset to slice each sample_grid on. Default 14 "
                         "(the val-best offset chosen by full interval_nll). Pick the "
                         "val-selected offset of the model you are most interested in.")
    ap.add_argument("--out", type=str, default=None,
                    help="Write the text report here. If omitted, prints to stdout only.")
    ap.add_argument("--out_csv", type=str, default=None,
                    help="Optional: also dump per-model rows as a CSV for programmatic use.")
    args = ap.parse_args()

    rows: List[dict] = []
    for spec in args.grid:
        if ":" not in spec:
            print(f"[abort] --grid expects 'label:path', got {spec!r}", file=sys.stderr)
            return 2
        label, path = spec.split(":", 1)
        rows.append(diag_one(label.strip(), Path(path.strip()), args.offset))

    report = format_text_report(rows, args.offset)
    print(report, end="")

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(report)
        print(f"\n[wrote] {out_path}")
    if args.out_csv:
        out_csv = Path(args.out_csv)
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df = pd.DataFrame([{k: v for k, v in d.items() if k != "path"} for d in rows])
        df.to_csv(out_csv, index=False)
        print(f"[wrote] {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
