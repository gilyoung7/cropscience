"""
Phase T9b — Sub-cohort timing-corr diagnostic for site-history experiments.

Given a Stage 1 ckpt (with or without site-history channels), runs inference,
applies first_crossing alert at (tau, k), and reports timing correlations
   corr(L_DOY, alert_DOY) and corr(L_DOY, score_peak_DOY)
separately for:
  - ALL test events
  - SUB-COHORT: only events whose site has a non-missing prev_year_L

Sub-cohort split uses the SAME history policy as the ckpt (or the
--history_policy arg if the ckpt has no history info).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.scripts.phase_t_lead_aware_eval import build_probs


def first_crossing_k(ts: np.ndarray, ps: np.ndarray, tau: float, k: int) -> int | None:
    streak = 0
    for i in range(len(ts)):
        if ps[i] >= tau:
            streak += 1
            if streak >= k: return int(ts[i])
        else:
            streak = 0
    return None


def per_site_year(probs_df: pd.DataFrame, tau: float, k: int, doy_start: int) -> pd.DataFrame:
    rows = []
    for (site, year), g in probs_df.groupby(["site", "year"], sort=False):
        g = g.sort_values("tstar")
        ts = g.tstar.values.astype(int)
        ps = g.p_cal.values.astype(float)
        is_event = int(g.y_event.iloc[0])
        true_L = g.true_L.iloc[0]
        at = first_crossing_k(ts, ps, tau, k)
        peak_idx = int(np.argmax(ps)) if len(ps) else None
        peak_tstar = int(ts[peak_idx]) if peak_idx is not None else None
        rows.append({
            "site": str(site), "year": int(year), "is_event": is_event,
            "true_L": int(true_L) if pd.notna(true_L) else None,
            "L_DOY": (int(true_L) + int(doy_start)) if pd.notna(true_L) else None,
            "alert_tstar": int(at) if at is not None else None,
            "alert_DOY": (int(at) + int(doy_start)) if at is not None else None,
            "peak_tstar": peak_tstar,
            "peak_DOY": (int(peak_tstar) + int(doy_start)) if peak_tstar is not None else None,
        })
    return pd.DataFrame(rows)


def get_history_mask(probs_df: pd.DataFrame, doy_start: int,
                     policy: str, train_year_max: int,
                     pest: str, run: int) -> dict:
    """Build (site, year) -> prev_year_L_miss flag using the same logic as training."""
    from rice.scripts.site_history_utils import compute_site_history
    from rice.scripts.run_eval import build_samples_for_run
    _, get_feature_cols = resolve_pest(pest)
    _, _, _, samples = build_samples_for_run(run, get_feature_cols)
    history = compute_site_history(samples, doy_start=doy_start, policy=policy,
                                    train_year_max=train_year_max)
    return {k: int(v["prev_year_L_miss"]) for k, v in history.items()}


def corr_block(df: pd.DataFrame, label: str) -> dict:
    out = {"n": int(len(df))}
    if len(df) < 5:
        out["note"] = "too few"
        return out
    L = df["L_DOY"].astype(float).values
    for col in ["alert_DOY", "peak_DOY"]:
        sub = df[df[col].notna()]
        n = int(len(sub))
        if n < 5 or np.std(sub["L_DOY"]) <= 0 or np.std(sub[col]) <= 0:
            out[col] = {"n": n, "pearson": None, "spearman": None}
            continue
        x = sub["L_DOY"].astype(float).values
        y = sub[col].astype(float).values
        pe = float(np.corrcoef(x, y)[0, 1])
        sp = float(spearmanr(x, y).correlation)
        out[col] = {"n": n, "pearson": pe, "spearman": sp}
    return out


def print_block(label: str, c: dict) -> None:
    print(f"  [{label}]  n_event={c['n']}")
    for col in ["alert_DOY", "peak_DOY"]:
        if col not in c: continue
        m = c[col]
        if m.get("pearson") is None:
            print(f"    corr(L_DOY, {col}):  n={m['n']}  pearson=N/A  spearman=N/A")
        else:
            print(f"    corr(L_DOY, {col}):  n={m['n']}  "
                  f"pearson={m['pearson']:.3f}  spearman={m['spearman']:.3f}")


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
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--history_policy", default="rolling",
                    choices=["rolling", "strict_train"])
    ap.add_argument("--history_train_year_max", type=int, default=2021)
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    print(f"\n========== Sub-cohort timing corr :: {args.label or Path(args.stage1_ckpt).stem} ==========")
    print(f"[cfg] tau={args.tau}  k={args.k}")

    # ckpt meta -> use same policy
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    if bool(ckpt.get("site_history_added", False)):
        pol = str(ckpt.get("site_history_policy", "rolling"))
        tyrmax = int(ckpt.get("history_train_year_max", 2021))
        print(f"[history meta from ckpt] policy={pol}  train_year_max={tyrmax}")
    else:
        pol = args.history_policy
        tyrmax = args.history_train_year_max
        print(f"[history meta] ckpt has no history; using policy={pol}  train_year_max={tyrmax}")

    cache = build_probs(args)
    doy_start = int(C.DOY_START)
    val_df = cache["val_df"]; test_df = cache["test_df"]

    miss_map = get_history_mask(test_df, doy_start, pol, tyrmax, args.pest, args.run)

    out = {"label": args.label or Path(args.stage1_ckpt).stem,
           "cfg": {"tau": args.tau, "k": args.k},
           "history": {"policy": pol, "train_year_max": tyrmax},
           "splits": {}}
    for split_name, df in [("val", val_df), ("test", test_df)]:
        recs = per_site_year(df, args.tau, args.k, doy_start)
        recs["prev_year_L_miss"] = recs.apply(
            lambda r: int(miss_map.get((r["site"], int(r["year"])), 1)), axis=1)
        ev_all = recs[recs.is_event == 1].copy()
        ev_sub = ev_all[ev_all.prev_year_L_miss == 0].copy()
        print(f"\n----- {split_name} -----")
        print(f"  total events={len(ev_all)}  with-history sub={len(ev_sub)}  "
              f"no-history (prev_year_L missing)={len(ev_all) - len(ev_sub)}")
        c_all = corr_block(ev_all, "all"); c_sub = corr_block(ev_sub, "sub")
        print_block("all events", c_all)
        print_block("with-history sub-cohort", c_sub)
        out["splits"][split_name] = {"all_events": c_all, "with_history_subcohort": c_sub,
                                     "n_total": int(len(ev_all)),
                                     "n_with_history": int(len(ev_sub))}

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
