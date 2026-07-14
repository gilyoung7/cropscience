"""
Phase T9 — prev_year_L correlation diagnostic.

Asks: does site-history (prev year's L_doy, recent-3y avg L_doy) carry timing
information correlated with this year's L_doy? No retraining; data analysis only.

Two history-construction conventions:
  rolling      : history uses any year < target year (operational rolling forecast)
  strict_train : history only from years <= --train_year_max (no val/test leakage
                 even into history; most conservative)

Per (site, year) interval sample, computes:
  this_L_doy             = base sample L + DOY_START
  rolling_prev_year_L    = L at (site, year-1) if event occurred that year
  rolling_avg3y          = mean L at (site, year-3 .. year-1)  (drop missing)
  rolling_years_since    = year - max{y < year : event occurred at site}
  strict_prev_year_L     = same but capped at min(year-1, train_year_max)
  strict_avg3y           = mean L at site over years <= min(year-1, train_year_max)
                           in window 3-year-wide ending at that cap
  strict_years_since     = year - max{y <= train_year_max : event at site}

Reports:
  - missing rate per history feature
  - corr(prev_year_L, this_L_doy) and corr(avg3y, this_L_doy):
    Pearson + Spearman, both conventions
  - split-aware breakdown (corr within train / val / test cohorts, rolling)
  - per-site year-to-year L_doy std summary (sites with >= 2 events)
  - years_since_last_event histogram

Output:
  - history_features.csv (per-(site,year) record + history columns)
  - prev_year_L_diag_summary.json
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
from rice.scripts.run_eval import build_samples_for_run


def corr_summary(df: pd.DataFrame, hist_col: str, target_col: str = "this_L_doy") -> dict:
    valid = df[hist_col].notna() & df[target_col].notna()
    n = int(valid.sum()); total = int(len(df))
    miss = (1 - n / total) if total else float("nan")
    if n < 5:
        return {"n_valid": n, "n_total": total, "missing_rate": miss,
                "pearson": None, "spearman": None}
    x = df.loc[valid, hist_col].astype(float).values
    y = df.loc[valid, target_col].astype(float).values
    if np.std(x) <= 0 or np.std(y) <= 0:
        return {"n_valid": n, "n_total": total, "missing_rate": miss,
                "pearson": float("nan"), "spearman": float("nan")}
    pe = float(np.corrcoef(x, y)[0, 1])
    sp = float(spearmanr(x, y).correlation)
    return {"n_valid": n, "n_total": total, "missing_rate": miss,
            "pearson": pe, "spearman": sp}


def history_features(site: str, year: int, site_year_L: dict,
                     site_years_with_event: dict, train_year_max: int) -> dict:
    """site_year_L: (site, year) -> L_doy
       site_years_with_event: site -> sorted list of years with event"""
    yrs = site_years_with_event.get(site, [])
    # ----- rolling -----
    rolling_prev_year_L = site_year_L.get((site, year - 1))
    win = [site_year_L.get((site, y)) for y in (year - 1, year - 2, year - 3)]
    win = [v for v in win if v is not None]
    rolling_avg3y = float(np.mean(win)) if win else None
    prev_yrs_all = [y for y in yrs if y < year]
    rolling_years_since = (year - prev_yrs_all[-1]) if prev_yrs_all else None
    # ----- strict_train -----
    cap = min(year - 1, train_year_max)
    prev_yrs_strict = [y for y in yrs if y <= cap]
    if prev_yrs_strict:
        last_y = prev_yrs_strict[-1]
        strict_prev_year_L = site_year_L.get((site, last_y))
        strict_years_since = year - last_y
    else:
        strict_prev_year_L = None
        strict_years_since = None
    win_strict = [site_year_L.get((site, y))
                  for y in (cap, cap - 1, cap - 2) if y >= 0]
    win_strict = [v for v in win_strict if v is not None]
    strict_avg3y = float(np.mean(win_strict)) if win_strict else None
    return {
        "rolling_prev_year_L": rolling_prev_year_L,
        "rolling_avg3y": rolling_avg3y,
        "rolling_years_since": rolling_years_since,
        "strict_prev_year_L": strict_prev_year_L,
        "strict_avg3y": strict_avg3y,
        "strict_years_since": strict_years_since,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--train_year_max", type=int, default=2021,
                    help="strict_train history convention upper bound")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _, get_feature_cols = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n========== Phase T9 prev_year_L correlation diagnostic ==========")
    print(f"[cfg] pest={args.pest}  run={args.run}  train_year_max={args.train_year_max}")
    print(f"[cfg] DOY_START={int(C.DOY_START)}  DOY_END={int(C.DOY_END)}")

    feature_cols, feature_names, T, samples = build_samples_for_run(args.run, get_feature_cols)
    doy_start = int(C.DOY_START)
    print(f"[samples] total base seasonal samples = {len(samples)}")

    # Extract event site-years (interval cohort) and their L_doy
    rows = []
    for s in samples:
        site = str(s["site_id"])
        year = int(s["year"])
        ctype = str(s.get("censor_type", "right"))
        if ctype != "right" and s.get("L") is not None and pd.notna(s["L"]):
            rows.append({"site": site, "year": year,
                         "L": int(s["L"]),
                         "L_doy": int(s["L"]) + doy_start})
    ev = pd.DataFrame(rows).drop_duplicates(["site", "year"]).reset_index(drop=True)
    print(f"[interval] n_events={len(ev)}  sites={ev.site.nunique()}  "
          f"year_range=[{int(ev.year.min())}, {int(ev.year.max())}]")

    # Build lookup tables
    site_year_L = {(r.site, int(r.year)): int(r.L_doy) for r in ev.itertuples(index=False)}
    site_years_with_event = (
        ev.groupby("site")["year"].apply(lambda s: sorted(int(x) for x in s)).to_dict()
    )

    # Per (site, year): compute history features (rolling + strict)
    records = []
    for r in ev.itertuples(index=False):
        h = history_features(r.site, int(r.year), site_year_L,
                             site_years_with_event, args.train_year_max)
        records.append({"site": r.site, "year": int(r.year),
                        "this_L_doy": int(r.L_doy), **h})
    hist_df = pd.DataFrame(records)
    hist_df.to_csv(out_dir / "history_features.csv", index=False)
    print(f"[saved] {out_dir / 'history_features.csv'}")

    # Correlations (overall)
    print(f"\n----- Correlations with this_L_doy (overall cohort) -----")
    print(f"  {'feature':>22}  {'convention':>14}  {'n_valid':>7} {'n_total':>7} "
          f"{'miss%':>6}  {'pearson':>8} {'spearman':>9}")
    corr_table = {}
    for hist_base in ["prev_year_L", "avg3y"]:
        for conv in ["rolling", "strict"]:
            col = f"{conv}_{hist_base}"
            res = corr_summary(hist_df, col)
            corr_table[col] = res
            p_str = f"{res['pearson']:>8.3f}" if res["pearson"] is not None and not (isinstance(res["pearson"], float) and np.isnan(res["pearson"])) else f"{'-':>8}"
            s_str = f"{res['spearman']:>9.3f}" if res["spearman"] is not None and not (isinstance(res["spearman"], float) and np.isnan(res["spearman"])) else f"{'-':>9}"
            print(f"  {hist_base:>22}  {conv:>14}  "
                  f"{res['n_valid']:>7d} {res['n_total']:>7d} "
                  f"{res['missing_rate']*100:>5.1f}%  {p_str} {s_str}")

    # Split-aware corr (rolling convention; strict's val/test mostly empty anyway)
    print(f"\n----- Split-aware corr (rolling, prev_year_L vs this_L_doy) -----")
    split_corrs = {}
    for label, year_filter in [("train (year<=2021)", lambda y: y <= 2021),
                                ("val (2022)", lambda y: y == 2022),
                                ("test (2023-24)", lambda y: 2023 <= y <= 2024)]:
        sub = hist_df[hist_df.year.apply(year_filter)].copy()
        if len(sub) < 5:
            print(f"  [{label}] n={len(sub)}  too few")
            split_corrs[label] = {"n": len(sub)}
            continue
        for hist_base in ["prev_year_L", "avg3y"]:
            col = f"rolling_{hist_base}"
            res = corr_summary(sub, col)
            p_str = f"{res['pearson']:>7.3f}" if res["pearson"] is not None and not (isinstance(res["pearson"], float) and np.isnan(res["pearson"])) else f"{'-':>7}"
            s_str = f"{res['spearman']:>8.3f}" if res["spearman"] is not None and not (isinstance(res["spearman"], float) and np.isnan(res["spearman"])) else f"{'-':>8}"
            print(f"  [{label}]  rolling_{hist_base:>11}  "
                  f"n={res['n_valid']:>4d}/{res['n_total']:>4d}  "
                  f"miss={res['missing_rate']*100:>5.1f}%  "
                  f"pearson={p_str}  spearman={s_str}")
            split_corrs.setdefault(label, {})[f"rolling_{hist_base}"] = res

    # Per-site year-to-year std
    print(f"\n----- Per-site year-to-year L_doy std -----")
    site_grp = ev.groupby("site")["L_doy"]
    counts = site_grp.count()
    multi = counts[counts >= 2].index
    site_std_summary = {}
    if len(multi) > 0:
        stds = site_grp.std(ddof=0).loc[multi].dropna()
        site_std_summary = {
            "n_sites_total": int(ev.site.nunique()),
            "n_sites_with_2plus_events": int(len(multi)),
            "per_site_std_mean": float(stds.mean()),
            "per_site_std_median": float(stds.median()),
            "per_site_std_min": float(stds.min()),
            "per_site_std_max": float(stds.max()),
            "per_site_std_q25": float(stds.quantile(0.25)),
            "per_site_std_q75": float(stds.quantile(0.75)),
            "pooled_L_doy_std": float(ev["L_doy"].std(ddof=0)),
            "pooled_L_doy_mean": float(ev["L_doy"].mean()),
        }
        s = site_std_summary
        print(f"  n_sites_total = {s['n_sites_total']}  "
              f"with 2+ events = {s['n_sites_with_2plus_events']}")
        print(f"  per-site std (multi-event sites): mean={s['per_site_std_mean']:.1f}  "
              f"median={s['per_site_std_median']:.1f}  q25={s['per_site_std_q25']:.1f}  "
              f"q75={s['per_site_std_q75']:.1f}  min={s['per_site_std_min']:.1f}  "
              f"max={s['per_site_std_max']:.1f}")
        print(f"  pooled L_doy std (all events) = {s['pooled_L_doy_std']:.1f}  "
              f"mean = {s['pooled_L_doy_mean']:.1f}")
        print(f"  interpretation: per-site std << pooled std => L_doy is site-specific "
              f"(history would help)")
        print(f"                  per-site std ≈ pooled std => year-to-year noise dominates")
    else:
        print("  no site has 2+ events")

    # years_since_last_event distribution
    print(f"\n----- years_since_last_event_at_site distribution (rolling) -----")
    yr_since = hist_df["rolling_years_since"]
    n_miss = int(yr_since.isna().sum())
    valid = yr_since.dropna().astype(int)
    print(f"  n_with_history = {len(valid)}  n_first_event_at_site = {n_miss}  "
          f"({n_miss/len(hist_df)*100:.1f}%)")
    bins = {}
    if len(valid) > 0:
        for v in [1, 2, 3, 4, 5]:
            bins[f"={v}y"] = int((valid == v).sum())
        bins[">5y"] = int((valid > 5).sum())
        for k_, v_ in bins.items():
            print(f"    {k_:>5}: {v_:>4d}  ({v_/len(valid)*100:.1f}% of those with history)")

    summary = {
        "args": {"pest": args.pest, "run": args.run, "train_year_max": args.train_year_max},
        "cohort": {"n_events": int(len(ev)), "n_sites": int(ev.site.nunique()),
                   "year_range": [int(ev.year.min()), int(ev.year.max())]},
        "overall_corr": corr_table,
        "split_corr_rolling": split_corrs,
        "per_site_std": site_std_summary,
        "years_since_last_event_dist": {"n_miss_first_event": n_miss,
                                         **bins} if len(valid) > 0 else {"n_miss_first_event": n_miss},
    }
    (out_dir / "prev_year_L_diag_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\n[saved] {out_dir / 'prev_year_L_diag_summary.json'}")

    # Verdict hint
    rolling_pe = corr_table.get("rolling_prev_year_L", {}).get("pearson")
    if rolling_pe is None or (isinstance(rolling_pe, float) and np.isnan(rolling_pe)):
        verdict_hint = "(no signal)"
    elif rolling_pe >= 0.4:
        verdict_hint = "STRONG signal — site history is a clear timing cue; add to features"
    elif rolling_pe >= 0.2:
        verdict_hint = "MODERATE signal — worth trying as feature, but expect partial gains"
    elif rolling_pe >= 0.1:
        verdict_hint = "WEAK signal — marginal; B/C runs (phenology/weather) higher priority"
    else:
        verdict_hint = "NONE — site history does not encode this-year timing"
    print(f"\n----- Verdict hint -----")
    print(f"  rolling prev_year_L pearson r = "
          f"{rolling_pe if rolling_pe is not None else 'N/A'}")
    print(f"  -> {verdict_hint}")


if __name__ == "__main__":
    main()
