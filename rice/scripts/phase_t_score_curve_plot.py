"""
Phase T14 — Score curve visualization.

For one or more Stage 1 ckpts, makes 3 plots per split:
  Plot 1 (absolute_doy): mean score curve vs DOY for event vs non-event
                          + q25/q75 band
  Plot 2 (event_aligned): event site-years, x = tstar_DOY - L_DOY, range [-120,+30]
                           early/normal/late group curves (L_DOY quantile)
  Plot 3 (nonevent_spike): non-event mean/p75/p90 vs DOY + event mean overlay

Outputs PNG + CSV per plot.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.scripts.phase_t_lead_aware_eval import build_probs


def per_doy_stats(probs_df: pd.DataFrame, group_filter, doy_start: int) -> pd.DataFrame:
    df = probs_df.copy()
    if group_filter is not None:
        df = df[group_filter(df)]
    df["doy"] = df["tstar"].astype(int) + int(doy_start)
    return df.groupby("doy")["p_cal"].agg(
        mean="mean", median="median",
        p25=lambda x: float(np.quantile(x, 0.25)),
        p75=lambda x: float(np.quantile(x, 0.75)),
        p90=lambda x: float(np.quantile(x, 0.90)),
        n="count",
    ).reset_index()


def event_aligned(probs_df: pd.DataFrame, doy_start: int,
                   rel_min: int = -120, rel_max: int = 30) -> pd.DataFrame:
    """Compute relative-to-L mean curve. probs_df should contain event site-years only."""
    df = probs_df[probs_df.y_event == 1].copy()
    df = df[df.true_L.notna()].copy()
    df["L_DOY"] = df["true_L"].astype(int) + int(doy_start)
    df["doy"] = df["tstar"].astype(int) + int(doy_start)
    df["rel_day"] = df["doy"] - df["L_DOY"]
    df = df[(df["rel_day"] >= rel_min) & (df["rel_day"] <= rel_max)]
    return df


def event_aligned_with_group(probs_df: pd.DataFrame, doy_start: int,
                              rel_min: int = -120, rel_max: int = 30) -> tuple[pd.DataFrame, dict]:
    df = event_aligned(probs_df, doy_start, rel_min, rel_max)
    if len(df) == 0:
        return df, {}
    sy_L = df.drop_duplicates(["site", "year"]).set_index(["site", "year"])["L_DOY"]
    q10 = float(sy_L.quantile(0.10)); q25 = float(sy_L.quantile(0.25))
    q75 = float(sy_L.quantile(0.75)); q90 = float(sy_L.quantile(0.90))
    def _g(L):
        if L <= q10: return "early_outlier"
        if L >= q90: return "late_outlier"
        if q25 <= L <= q75: return "normal_event"
        return "between"
    df = df.copy()
    df["L_group"] = df["L_DOY"].apply(_g)
    return df, {"q10": q10, "q25": q25, "q75": q75, "q90": q90}


def plot_absolute(curve_ev: pd.DataFrame, curve_ne: pd.DataFrame,
                  out_png: Path, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(11, 5))
    if len(curve_ev) > 0:
        ax.plot(curve_ev["doy"], curve_ev["mean"], color="C3", lw=1.8,
                label=f"event (n_sy mean rows={int(curve_ev['n'].mean())})")
        ax.fill_between(curve_ev["doy"], curve_ev["p25"], curve_ev["p75"],
                         color="C3", alpha=0.15)
    if len(curve_ne) > 0:
        ax.plot(curve_ne["doy"], curve_ne["mean"], color="C0", lw=1.8,
                label=f"non_event (n_sy mean rows={int(curve_ne['n'].mean())})")
        ax.fill_between(curve_ne["doy"], curve_ne["p25"], curve_ne["p75"],
                         color="C0", alpha=0.15)
    ax.set_xlabel("DOY"); ax.set_ylabel("p_cal")
    ax.set_title(title); ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    ax.set_xlim(60, 300); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return True


def plot_event_aligned(df: pd.DataFrame, out_png: Path, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(11, 5))
    colors = {"early_outlier": "C0", "normal_event": "C2",
              "late_outlier": "C3", "between": "gray"}
    for grp in ["early_outlier", "normal_event", "late_outlier"]:
        sub = df[df.L_group == grp]
        if len(sub) == 0: continue
        agg = sub.groupby("rel_day")["p_cal"].agg(
            mean="mean",
            p25=lambda x: float(np.quantile(x, 0.25)),
            p75=lambda x: float(np.quantile(x, 0.75)),
            n="count").reset_index()
        n_sy = sub.drop_duplicates(["site", "year"]).shape[0]
        ax.plot(agg["rel_day"], agg["mean"], color=colors[grp], lw=1.7,
                label=f"{grp} (n_sy={n_sy})")
        ax.fill_between(agg["rel_day"], agg["p25"], agg["p75"],
                         color=colors[grp], alpha=0.12)
    # also overall mean
    all_agg = df.groupby("rel_day")["p_cal"].mean().reset_index()
    ax.plot(all_agg["rel_day"], all_agg["p_cal"], color="black",
            lw=1.2, ls="--", label="all events (mean)")
    ax.axvline(0, color="k", ls=":", lw=1.0, alpha=0.5)
    ax.set_xlabel("relative day  (tstar_DOY - L_DOY)")
    ax.set_ylabel("p_cal")
    ax.set_title(title); ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return True


def plot_nonevent_spike(curve_ev: pd.DataFrame, curve_ne: pd.DataFrame,
                         out_png: Path, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    fig, ax = plt.subplots(figsize=(11, 5))
    if len(curve_ne) > 0:
        ax.plot(curve_ne["doy"], curve_ne["mean"], color="C0", lw=1.6, label="non_event mean")
        ax.plot(curve_ne["doy"], curve_ne["p75"], color="C0", lw=1.0, ls="--", label="non_event p75")
        ax.plot(curve_ne["doy"], curve_ne["p90"], color="C0", lw=1.0, ls=":", label="non_event p90")
    if len(curve_ev) > 0:
        ax.plot(curve_ev["doy"], curve_ev["mean"], color="C3", lw=2.0,
                label="event mean (reference)")
    ax.set_xlabel("DOY"); ax.set_ylabel("p_cal")
    ax.set_title(title); ax.legend(loc="best"); ax.grid(True, alpha=0.3)
    ax.set_xlim(60, 300); ax.set_ylim(0, 1)
    fig.tight_layout(); fig.savefig(out_png, dpi=120); plt.close(fig)
    return True


def run_one_ckpt(label: str, ckpt_path: str, args, out_dir: Path) -> dict:
    print(f"\n========== {label} ==========")
    class N: pass
    common = N()
    for f in ["pest", "run", "split_seed", "val_year", "test_year_min", "test_year_max"]:
        setattr(common, f, getattr(args, f))
    common.stage1_ckpt = ckpt_path
    cache = build_probs(common)
    doy_start = int(C.DOY_START)

    summary = {"label": label, "ckpt": ckpt_path, "splits": {}}
    for split in ["val", "test"]:
        df = cache[f"{split}_df"]
        print(f"  [{split}] rows={len(df)}  event_rows={(df.y_event==1).sum()}  nonev_rows={(df.y_event==0).sum()}")
        # Plot 1: absolute DOY
        ev_curve = per_doy_stats(df, lambda x: x.y_event == 1, doy_start)
        ne_curve = per_doy_stats(df, lambda x: x.y_event == 0, doy_start)
        ev_curve.to_csv(out_dir / f"curve_abs_event_{label}_{split}.csv", index=False)
        ne_curve.to_csv(out_dir / f"curve_abs_nonevent_{label}_{split}.csv", index=False)
        png1 = out_dir / f"plot1_absolute_{label}_{split}.png"
        plot_absolute(ev_curve, ne_curve, png1,
                       f"Plot 1 — Absolute DOY  ({label}, {split})  event vs non-event")
        print(f"    [saved] {png1.name}")
        # Plot 2: event-aligned
        ea_df, quants = event_aligned_with_group(df, doy_start, args.rel_min, args.rel_max)
        if len(ea_df) > 0:
            ea_df[["site","year","tstar","doy","L_DOY","rel_day","L_group","p_cal","y_event"]].to_csv(
                out_dir / f"event_aligned_{label}_{split}.csv", index=False)
            png2 = out_dir / f"plot2_event_aligned_{label}_{split}.png"
            plot_event_aligned(ea_df, png2,
                                f"Plot 2 — Event-aligned  ({label}, {split})  "
                                f"L_DOY q10={quants.get('q10', 0):.0f}/q90={quants.get('q90', 0):.0f}")
            print(f"    [saved] {png2.name}")
        # Plot 3: non-event spike vs event
        png3 = out_dir / f"plot3_nonevent_spike_{label}_{split}.png"
        plot_nonevent_spike(ev_curve, ne_curve, png3,
                              f"Plot 3 — Non-event spike vs event mean  ({label}, {split})")
        print(f"    [saved] {png3.name}")
        # Summary stats
        ev_overall_mean = float(df.loc[df.y_event==1, "p_cal"].mean())
        ne_overall_mean = float(df.loc[df.y_event==0, "p_cal"].mean())
        ne_overall_p90 = float(df.loc[df.y_event==0, "p_cal"].quantile(0.90))
        ev_doy_peak_mean = int(ev_curve.loc[ev_curve["mean"].idxmax(), "doy"]) if len(ev_curve) else None
        ne_doy_peak_mean = int(ne_curve.loc[ne_curve["mean"].idxmax(), "doy"]) if len(ne_curve) else None
        summary["splits"][split] = {
            "n_rows": int(len(df)),
            "n_event_rows": int((df.y_event == 1).sum()),
            "n_nonevent_rows": int((df.y_event == 0).sum()),
            "event_overall_mean_p": ev_overall_mean,
            "nonevent_overall_mean_p": ne_overall_mean,
            "nonevent_overall_p90": ne_overall_p90,
            "event_doy_peak_mean": ev_doy_peak_mean,
            "nonevent_doy_peak_mean": ne_doy_peak_mean,
            "L_DOY_quantiles": quants if len(ea_df) > 0 else None,
        }
        print(f"    event mean p_cal (all rows): {ev_overall_mean:.3f}  "
              f"non-event mean: {ne_overall_mean:.3f}  delta: {ev_overall_mean-ne_overall_mean:+.3f}")
        print(f"    non-event p90: {ne_overall_p90:.3f}  event mean at peak DOY={ev_doy_peak_mean}  "
              f"non-event mean at peak DOY={ne_doy_peak_mean}")
    return summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--d_ckpt", default="")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--rel_min", type=int, default=-120)
    ap.add_argument("--rel_max", type=int, default=30)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n========== Score curve plot ==========")

    out = {"args": vars(args), "summaries": {}}
    out["summaries"]["A_baseline"] = run_one_ckpt("A_baseline", args.baseline_ckpt, args, out_dir)
    if args.d_ckpt:
        out["summaries"]["D_history"] = run_one_ckpt("D_history", args.d_ckpt, args, out_dir)

    (out_dir / "score_curve_summary.json").write_text(json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {out_dir / 'score_curve_summary.json'}")


if __name__ == "__main__":
    main()
