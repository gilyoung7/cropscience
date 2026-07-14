"""
Phase T7f — Alert-timing diversity comparison between operating points.

Two input modes:
  --ckpts MODE (preferred): pass --ckpts c1,c2 --taus t1,t2 --ks k1,k2 --labels l1,l2
                            Runs first_crossing alert at the exact (tau, k) per ckpt
                            on the test split. Use this when ckpts use different
                            operating points.
  --csvs MODE: pass --csvs path1,path2 --labels l1,l2 (CSV must contain
               site, year, is_event, true_L, alert_tstar; uses 'split'==test if present)

For each (label, ckpt|csv):
  - filter event site-years with alert (is_event==1, alert_tstar notna)
  - alert_lead = true_L - alert_tstar
  - alert_doy  = alert_tstar + --doy_start

Reports:
  - n_alerted_events
  - lead: mean, std, quartiles (q10/q25/q50/q75/q90), 10-day histogram
  - alert_doy: same stats, 10-day histogram
  - corr(true_L, alert_tstar): Pearson + Spearman   <-- key indicator

corr ~ 0:  alerts move independently of event timing -> calendar prior dominates
corr high: alerts shift with L -> model reads per-site-year timing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest


def load_test_alerts_from_csv(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"] == "test"].copy()
    df = df[df["is_event"] == 1].copy()
    df = df[df["alert_tstar"].notna()].copy()
    df["alert_tstar"] = df["alert_tstar"].astype(int)
    df["true_L"] = df["true_L"].astype(int)
    if "lead_days" in df.columns and df["lead_days"].notna().any():
        df["lead_days"] = df["lead_days"].astype(int)
    else:
        df["lead_days"] = df["true_L"] - df["alert_tstar"]
    return df


def load_test_alerts_from_ckpt(ckpt_path: Path, tau: float, k: int,
                                pest: str, run: int, split_seed: int,
                                val_year: int, test_year_min: int, test_year_max: int) -> tuple[pd.DataFrame, int]:
    """Run XGB inference on test cohort and apply first_crossing(tau, k) per site-year.
    Returns (filtered event-alert df, doy_start)."""
    # Reuse build_probs from phase_t_lead_aware_eval to produce calibrated test probs
    from argparse import Namespace
    from rice.scripts.phase_t_lead_aware_eval import build_probs, apply_alert_rule
    args = Namespace(pest=pest, run=run, stage1_ckpt=str(ckpt_path),
                     split_seed=split_seed, val_year=val_year,
                     test_year_min=test_year_min, test_year_max=test_year_max)
    cache = build_probs(args)
    test_df = cache["test_df"]
    spec = {"name": "first_crossing", "tau": float(tau), "k": int(k)}
    cls = apply_alert_rule(test_df, spec, int(C.DOY_START))
    cls = cls[(cls["is_event"] == 1) & cls["alert_tstar"].notna()].copy()
    cls["alert_tstar"] = cls["alert_tstar"].astype(int)
    cls["true_L"] = cls["true_L"].astype(int)
    cls["lead_days"] = cls["true_L"] - cls["alert_tstar"]
    return cls, int(C.DOY_START)


def compute_stats(df: pd.DataFrame, doy_start: int) -> dict:
    lead = df["lead_days"].astype(int).values
    alert_doy = df["alert_tstar"].astype(int).values + int(doy_start)
    L_arr = df["true_L"].astype(int).values
    a_arr = df["alert_tstar"].astype(int).values

    def _arr_stats(x):
        return {
            "n": int(len(x)),
            "mean": float(np.mean(x)),
            "std": float(np.std(x, ddof=0)),
            "min": int(np.min(x)),
            "max": int(np.max(x)),
            "q10": float(np.quantile(x, 0.10)),
            "q25": float(np.quantile(x, 0.25)),
            "q50": float(np.median(x)),
            "q75": float(np.quantile(x, 0.75)),
            "q90": float(np.quantile(x, 0.90)),
        }

    out = {"n_alerted_events": int(len(df))}
    out["lead"] = _arr_stats(lead)
    out["alert_doy"] = _arr_stats(alert_doy)
    if len(L_arr) > 1 and np.std(L_arr) > 0 and np.std(a_arr) > 0:
        pearson = float(np.corrcoef(L_arr, a_arr)[0, 1])
        spearman = float(spearmanr(L_arr, a_arr).correlation)
    else:
        pearson = float("nan"); spearman = float("nan")
    out["corr_L_alert_tstar"] = {"pearson": pearson, "spearman": spearman}
    out["L"] = _arr_stats(L_arr)
    return out


def text_hist(values: np.ndarray, lo: int, hi: int, bw: int, max_bar: int = 50) -> str:
    bins = np.arange(lo, hi + bw, bw)
    counts, edges = np.histogram(values, bins=bins)
    if counts.max() > max_bar:
        scale = max_bar / counts.max()
    else:
        scale = 1.0
    lines = []
    for i, c in enumerate(counts):
        if c == 0:
            continue
        bar = "#" * max(1, int(round(c * scale)))
        lines.append(f"    [{int(edges[i]):>4d}, {int(edges[i+1]):>4d})  n={int(c):>4d}  {bar}")
    return "\n".join(lines)


def print_block(label: str, s: dict, lead_arr, doy_arr, doy_start: int) -> None:
    print(f"\n========== {label}  (n_alerted_events={s['n_alerted_events']}) ==========")
    print(f"  alert lead = L - alert_tstar")
    ld = s["lead"]
    print(f"    mean={ld['mean']:.1f}  std={ld['std']:.1f}  min={ld['min']}  max={ld['max']}")
    print(f"    quartiles: q10={ld['q10']:.1f}  q25={ld['q25']:.1f}  q50={ld['q50']:.1f}  "
          f"q75={ld['q75']:.1f}  q90={ld['q90']:.1f}")
    print(f"  histogram (10-day bins):")
    print(text_hist(lead_arr, lo=-50, hi=210, bw=10))
    print(f"\n  alert_doy = alert_tstar + DOY_START({doy_start})")
    dd = s["alert_doy"]
    print(f"    mean={dd['mean']:.1f}  std={dd['std']:.1f}  min={dd['min']}  max={dd['max']}")
    print(f"    quartiles: q10={dd['q10']:.1f}  q25={dd['q25']:.1f}  q50={dd['q50']:.1f}  "
          f"q75={dd['q75']:.1f}  q90={dd['q90']:.1f}")
    print(f"  histogram (10-day bins):")
    print(text_hist(doy_arr, lo=doy_start, hi=300, bw=10))
    cr = s["corr_L_alert_tstar"]
    print(f"\n  corr(true_L, alert_tstar):  pearson={cr['pearson']:.3f}  "
          f"spearman={cr['spearman']:.3f}")
    print(f"  (reference) L distribution: mean={s['L']['mean']:.1f}  std={s['L']['std']:.1f}  "
          f"q25={s['L']['q25']:.1f}  q50={s['L']['q50']:.1f}  q75={s['L']['q75']:.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpts", default="",
                    help="comma-separated ckpt paths (preferred mode); requires --taus --ks")
    ap.add_argument("--taus", default="", help="comma-separated tau per ckpt")
    ap.add_argument("--ks", default="", help="comma-separated k per ckpt")
    ap.add_argument("--csvs", default="",
                    help="comma-separated classification CSV paths (alt mode)")
    ap.add_argument("--labels", required=True,
                    help="comma-separated labels matching either --ckpts or --csvs")
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--doy_start", type=int, default=60)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    labels = [s.strip() for s in args.labels.split(",")]
    if args.ckpts:
        ckpts = [s.strip() for s in args.ckpts.split(",")]
        taus = [float(s) for s in args.taus.split(",")]
        ks = [int(s) for s in args.ks.split(",")]
        if not (len(ckpts) == len(labels) == len(taus) == len(ks)):
            raise SystemExit("--ckpts/--labels/--taus/--ks must have same length")
        _ = resolve_pest(args.pest)
        inputs = list(zip(labels, ckpts, taus, ks))
        mode = "ckpts"
    else:
        csvs = [s.strip() for s in args.csvs.split(",")]
        if len(csvs) != len(labels):
            raise SystemExit("--csvs and --labels must have same length")
        inputs = list(zip(labels, csvs))
        mode = "csvs"

    out = {}
    summary_rows = []
    doy_start = int(args.doy_start)
    for entry in inputs:
        if mode == "ckpts":
            label, ckpt, tau, k = entry
            df, doy_start = load_test_alerts_from_ckpt(
                Path(ckpt), tau, k, args.pest, args.run, args.split_seed,
                args.val_year, args.test_year_min, args.test_year_max)
            label = f"{label} (tau={tau:.3f} k={k})"
        else:
            label, path = entry
            df = load_test_alerts_from_csv(Path(path))
        s = compute_stats(df, doy_start)
        lead_arr = df["lead_days"].astype(int).values
        doy_arr = df["alert_tstar"].astype(int).values + doy_start
        print_block(label, s, lead_arr, doy_arr, doy_start)
        out[label] = s
        summary_rows.append({
            "label": label,
            "n_alerted": s["n_alerted_events"],
            "lead_mean": s["lead"]["mean"], "lead_std": s["lead"]["std"],
            "lead_q10": s["lead"]["q10"], "lead_q50": s["lead"]["q50"],
            "lead_q90": s["lead"]["q90"],
            "alert_doy_mean": s["alert_doy"]["mean"],
            "alert_doy_std": s["alert_doy"]["std"],
            "alert_doy_q10": s["alert_doy"]["q10"],
            "alert_doy_q90": s["alert_doy"]["q90"],
            "corr_L_at_pearson": s["corr_L_alert_tstar"]["pearson"],
            "corr_L_at_spearman": s["corr_L_alert_tstar"]["spearman"],
        })

    print(f"\n========== Side-by-side summary ==========")
    keys = ["label", "n_alerted", "lead_mean", "lead_std",
            "lead_q10", "lead_q50", "lead_q90",
            "alert_doy_mean", "alert_doy_std", "alert_doy_q10", "alert_doy_q90",
            "corr_L_at_pearson", "corr_L_at_spearman"]
    widths = {"label": 24, "n_alerted": 9}
    print("  " + "  ".join(f"{k:>{widths.get(k, 10)}}" for k in keys))
    for row in summary_rows:
        cells = []
        for k in keys:
            v = row[k]
            if isinstance(v, int):
                cells.append(f"{v:>{widths.get(k, 10)}d}")
            elif isinstance(v, float):
                cells.append(f"{v:>{widths.get(k, 10)}.3f}")
            else:
                cells.append(f"{str(v):>{widths.get(k, 10)}}")
        print("  " + "  ".join(cells))

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
