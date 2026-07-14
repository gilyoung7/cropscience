"""
Phase T10c — Site-wise score normalization (percentile + z-score).

For one Stage 1 ckpt, evaluate three score transforms before first_crossing:
  raw         : original calibrated p_cal (baseline)
  percentile  : within-site rank of the score among that site's val scores
                  -> score' = rank(score, site_quantiles(s)) / len(site_quantiles(s))
                  Range: [0, 1]
  zscore      : (score - site_mean) / site_std    using val-cohort stats
                  Range: roughly [-3, +3]

Site stats are computed on VAL only (no test leakage). Sites with too few val
scores fall back to the cohort-global mean/std/sorted-array.

For each transform: sweep first_crossing(tau, k); select best on val at
recall>=0.85/0.88/0.90 (FAR-min); report test metrics at the same (k, tau).

Compares per-target test recall/FAR/no_alert/USEFUL with raw and against
the dispatch_group_tau benchmark (R=0.854, FAR=0.678, no_alert=84).
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.scripts.phase_t_lead_aware_eval import build_probs


def first_crossing_k(ts: np.ndarray, ps: np.ndarray, tau: float, k: int) -> int | None:
    streak = 0
    for i in range(len(ts)):
        if ps[i] >= tau:
            streak += 1
            if streak >= k: return int(ts[i])
        else: streak = 0
    return None


def event_bucket(lead):
    if lead is None or pd.isna(lead): return "no_alert"
    d = int(lead)
    if d < 0: return "MISSED"
    if d < 14: return "TOO_LATE"
    if d <= 60: return "USEFUL"
    return "TOO_EARLY"


def build_site_stats(val_df: pd.DataFrame, min_n: int = 5) -> tuple[dict, dict]:
    stats = {}
    for site, g in val_df.groupby("site"):
        ps = g["p_cal"].astype(float).values
        if len(ps) < min_n:
            continue
        std = float(np.std(ps, ddof=0))
        if std < 1e-9: std = 1.0
        stats[str(site)] = {"mean": float(np.mean(ps)),
                            "std": std,
                            "sorted": np.sort(ps).astype(float)}
    all_p = val_df["p_cal"].astype(float).values
    glob = {"mean": float(np.mean(all_p)),
            "std": float(np.std(all_p, ddof=0)) if np.std(all_p) > 1e-9 else 1.0,
            "sorted": np.sort(all_p).astype(float)}
    return stats, glob


def transform_probs(df: pd.DataFrame, stats: dict, glob: dict, mode: str) -> pd.DataFrame:
    sites = df["site"].astype(str).values
    raw = df["p_cal"].astype(float).values
    out = np.empty_like(raw)
    if mode == "percentile":
        for i, (s, p) in enumerate(zip(sites, raw)):
            ss = stats.get(s, glob)
            arr = ss["sorted"]
            out[i] = float(np.searchsorted(arr, p, side="right")) / max(len(arr), 1)
    elif mode == "zscore":
        for i, (s, p) in enumerate(zip(sites, raw)):
            ss = stats.get(s, glob)
            out[i] = (p - ss["mean"]) / ss["std"]
    elif mode == "raw":
        out = raw
    else:
        raise ValueError(mode)
    df2 = df.copy()
    df2["p_cal"] = out
    return df2


def classify(probs_df: pd.DataFrame, tau: float, k: int, doy_start: int) -> pd.DataFrame:
    rows = []
    for (s, y), g in probs_df.groupby(["site", "year"], sort=False):
        gs = g.sort_values("tstar")
        ts = gs["tstar"].values.astype(int)
        ps = gs["p_cal"].astype(float).values
        is_event = int(gs["y_event"].iloc[0])
        true_L = gs["true_L"].iloc[0]
        at = first_crossing_k(ts, ps, tau, k)
        lead = (int(true_L) - int(at)) if (is_event == 1 and pd.notna(true_L) and at is not None) else None
        bucket = event_bucket(lead) if is_event == 1 else ("FP" if at is not None else "TN")
        rows.append({"site": str(s), "year": int(y), "is_event": is_event,
                     "alert_tstar": (int(at) if at is not None else None),
                     "lead_days": (int(lead) if lead is not None else None),
                     "bucket": bucket,
                     "true_L": (int(true_L) if pd.notna(true_L) else None)})
    return pd.DataFrame(rows)


def metrics(df: pd.DataFrame) -> dict:
    n_e = int((df.is_event == 1).sum())
    n_ne = int((df.is_event == 0).sum())
    tp = int(((df.is_event == 1) & df.alert_tstar.notna()).sum())
    fp = int(((df.is_event == 0) & df.alert_tstar.notna()).sum())
    bk = Counter(df[df.is_event == 1]["bucket"])
    leads = df.loc[(df.is_event == 1) & df.lead_days.notna(), "lead_days"].astype(int).values
    prec = tp / max(tp + fp, 1) if (tp + fp) else float("nan")
    rec = tp / max(n_e, 1)
    return {"recall": rec, "FAR": fp / max(n_ne, 1),
            "precision": prec,
            "F1": 2 * prec * rec / max(prec + rec, 1e-9) if (prec + rec) > 0 else float("nan"),
            "TP": tp, "FP": fp, "n_alert": tp + fp, "n_event": n_e,
            "no_alert": int(bk.get("no_alert", 0)),
            "TOO_LATE": int(bk.get("TOO_LATE", 0)),
            "MISSED": int(bk.get("MISSED", 0)),
            "USEFUL": int(bk.get("USEFUL", 0)),
            "lead_median": float(np.median(leads)) if len(leads) else None}


def sweep(probs_df: pd.DataFrame, tau_grid: np.ndarray, ks: list[int],
          doy_start: int) -> pd.DataFrame:
    rows = []
    for k in ks:
        for tau in tau_grid:
            df = classify(probs_df, float(tau), int(k), doy_start)
            m = metrics(df)
            rows.append({"k": int(k), "tau": float(tau), **m})
    return pd.DataFrame(rows)


def select(df: pd.DataFrame, target: float) -> dict | None:
    cands = df[df["recall"] >= target]
    if cands.empty:
        return None
    c = cands.copy()
    c["_lead"] = c["lead_median"].fillna(999.0)
    return c.sort_values(["FAR", "_lead", "tau", "k"]).iloc[0].to_dict()


def lookup_test(test_sw: pd.DataFrame, k: int, tau: float) -> dict:
    sub = test_sw[(test_sw.k == k) & (np.isclose(test_sw.tau, tau, atol=0.01))]
    if len(sub) == 0:
        s = test_sw[test_sw.k == k].copy()
        s["_d"] = (s.tau - tau).abs()
        return s.loc[s["_d"].idxmin()].to_dict()
    return sub.iloc[0].to_dict()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--ks", default="3")
    ap.add_argument("--recall_targets", default="0.85,0.88,0.90")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.stage1_ckpt).stem
    print(f"\n========== Site-wise normalization :: {label} ==========")

    cache = build_probs(args)
    doy_start = int(C.DOY_START)
    val_df = cache["val_df"]; test_df = cache["test_df"]
    print(f"[cohort] val_rows={len(val_df)}  test_rows={len(test_df)}")

    stats, glob = build_site_stats(val_df)
    sites_total = val_df["site"].nunique()
    print(f"[site stats] sites_with_stats={len(stats)} / sites_total={sites_total}  "
          f"global_mean={glob['mean']:.3f}  global_std={glob['std']:.3f}")

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    targets = [float(x) for x in args.recall_targets.split(",") if x.strip()]

    modes = {
        "raw":        {"tau_grid": np.arange(0.05, 0.95 + 1e-9, 0.025)},
        "percentile": {"tau_grid": np.arange(0.05, 1.00 + 1e-9, 0.025)},
        "zscore":     {"tau_grid": np.round(np.arange(-2.5, 3.0 + 1e-9, 0.1), 3)},
    }

    out = {"label": label, "site_stats_count": len(stats), "global": glob_stats_safe(glob), "selections": {}}
    for mode_name, m_cfg in modes.items():
        v_df = transform_probs(val_df, stats, glob, mode_name)
        t_df = transform_probs(test_df, stats, glob, mode_name)
        tau_grid = m_cfg["tau_grid"]
        print(f"\n========== mode={mode_name}  tau_grid={len(tau_grid)}  range=[{tau_grid[0]:.3f}, {tau_grid[-1]:.3f}] ==========")
        v_sw = sweep(v_df, tau_grid, ks, doy_start)
        t_sw = sweep(t_df, tau_grid, ks, doy_start)
        v_sw.to_csv(out_dir / f"sweep_val_{mode_name}.csv", index=False)
        t_sw.to_csv(out_dir / f"sweep_test_{mode_name}.csv", index=False)
        for tgt in targets:
            pick = select(v_sw, tgt)
            if pick is None:
                print(f"  R>={tgt:.2f}: no qualifying cell on val")
                out["selections"].setdefault(f"R>={tgt:.2f}", {})[mode_name] = None
                continue
            t_row = lookup_test(t_sw, int(pick["k"]), float(pick["tau"]))
            print(f"  R>={tgt:.2f}  mode={mode_name}  k={int(pick['k'])}  tau={float(pick['tau']):.3f}")
            print(f"    val:  R={pick['recall']:.3f}  FAR={pick['FAR']:.3f}  P={pick['precision']:.3f}  "
                  f"F1={pick['F1']:.3f}  noA={int(pick['no_alert'])}  TL={int(pick['TOO_LATE'])}  "
                  f"MS={int(pick['MISSED'])}  USE={int(pick['USEFUL'])}")
            print(f"    test: R={t_row['recall']:.3f}  FAR={t_row['FAR']:.3f}  P={t_row['precision']:.3f}  "
                  f"F1={t_row['F1']:.3f}  noA={int(t_row['no_alert'])}  TL={int(t_row['TOO_LATE'])}  "
                  f"MS={int(t_row['MISSED'])}  USE={int(t_row['USEFUL'])}  lead_med={t_row['lead_median']}")
            out["selections"].setdefault(f"R>={tgt:.2f}", {})[mode_name] = {
                "k": int(pick["k"]), "tau": float(pick["tau"]),
                "val": pick, "test": t_row}

    # Side-by-side test summary
    print(f"\n========== Side-by-side (test) ==========")
    print(f"  {'target':>8}  {'mode':>11}  {'k':>2} {'tau':>7}  {'R':>5} {'FAR':>5} {'P':>5} {'F1':>5} "
          f"{'noA':>4} {'TL':>3} {'MS':>3} {'USE':>4}")
    for tgt in targets:
        for mode_name in modes:
            sel = out["selections"].get(f"R>={tgt:.2f}", {}).get(mode_name)
            if sel is None:
                print(f"  R>={tgt:.2f}  {mode_name:>11}  (no qualifying cell)")
                continue
            t = sel["test"]
            print(f"  R>={tgt:.2f}  {mode_name:>11}  {sel['k']:>2d} {sel['tau']:>7.3f}  "
                  f"{t['recall']:>5.3f} {t['FAR']:>5.3f} {t['precision']:>5.3f} {t['F1']:>5.3f} "
                  f"{int(t['no_alert']):>4d} {int(t['TOO_LATE']):>3d} {int(t['MISSED']):>3d} {int(t['USEFUL']):>4d}")

    (out_dir / f"site_norm_summary_{label}.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {out_dir / f'site_norm_summary_{label}.json'}")


def glob_stats_safe(g: dict) -> dict:
    return {"mean": float(g["mean"]), "std": float(g["std"]),
            "n_sorted": int(len(g["sorted"]))}


if __name__ == "__main__":
    main()
