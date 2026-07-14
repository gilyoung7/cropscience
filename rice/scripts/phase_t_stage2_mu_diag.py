"""
Phase T11 — Stage 2 mu calibration diagnostic.

Given a Stage 1 ckpt (for alert_tstar) and a Stage 2 ckpt (for pred_mu),
asks whether Stage 2 mu actually tracks alert_tstar / true_L_DOY, or whether
mu is stuck on a calendar prior independent of either.

For each event site-year alerted by Stage 1 (at given tau, k):
  alert_tstar    = first_crossing(tau, k) on Stage 1 score
  pred_mu        = Stage 2 mu evaluated at alert_tstar  (offset=0)
                   plus offsets {7, 14, 21, 30}
  true_L_DOY     = L + DOY_START

Reports (val + test, full + with_history + no_history):
  - distribution: alert_tstar / pred_mu (DOY) / true_L_DOY  (mean, std, quartiles)
  - corr(alert_tstar, pred_mu_at_offset)            : at each offset
  - corr(true_L_DOY, pred_mu_at_offset)              : at each offset
  - corr(alert_tstar, true_L_DOY)                    : reference
  - per-subcohort breakdown (with_history vs no_history)

Interpretation:
  corr(alert_tstar, pred_mu) ≈ 1     -> mu = alert_tstar + fixed offset only
  corr(alert_tstar, pred_mu) ≈ 0     -> Stage 2 ignores alert; calendar prior
  corr(true_L_DOY, pred_mu) high     -> mu tracks event timing (good)
  corr(true_L_DOY, pred_mu) low      -> mu does NOT track L; calendar prior

No retraining; Stage 1 inference + Stage 2 forward only.
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
from rice.scripts.phase_t_history_subcohort_compare import make_history_mask


def first_crossing_k(ts: np.ndarray, ps: np.ndarray, tau: float, k: int) -> int | None:
    streak = 0
    for i in range(len(ts)):
        if ps[i] >= tau:
            streak += 1
            if streak >= k: return int(ts[i])
        else: streak = 0
    return None


def build_stage1_alerts(probs_df: pd.DataFrame, tau: float, k: int) -> dict:
    """Returns {(site, year): {alert_tstar, true_L, true_R, is_event}}."""
    out = {}
    for (s, y), g in probs_df.groupby(["site", "year"], sort=False):
        gs = g.sort_values("tstar")
        ts = gs["tstar"].values.astype(int)
        ps = gs["p_cal"].values.astype(float)
        at = first_crossing_k(ts, ps, tau, k)
        out[(str(s), int(y))] = {
            "alert_tstar": at,
            "true_L": gs["true_L"].iloc[0],
            "true_R": gs["true_R"].iloc[0],
            "is_event": int(gs["y_event"].iloc[0]),
        }
    return out


def get_stage2_row_map(stage2_ckpt: str, run: int, pest: str,
                       split_seed: int, val_year: int,
                       test_year_min: int, test_year_max: int,
                       device: str = "cuda") -> tuple[dict, int]:
    """Wraps phase_r_oracle_iou.build_stage2_row_map for the test cohort.
    Returns (row_map: {(site, year, tstar): {mu, true_L, true_R, ctype}}, doy_start)."""
    from rice.scripts.phase_r_oracle_iou import build_stage2_row_map
    class A: pass
    args = A()
    args.pest = pest
    args.split_seed = split_seed
    args.val_year = val_year
    args.test_year_min = test_year_min
    args.test_year_max = test_year_max
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    row_map, doy_start = build_stage2_row_map(stage2_ckpt, run, args, dev,
                                                bypass_phen_head=False)
    return row_map, int(doy_start)


def corr_pair(x: np.ndarray, y: np.ndarray) -> dict:
    if len(x) < 5 or np.std(x) <= 0 or np.std(y) <= 0:
        return {"n": int(len(x)), "pearson": None, "spearman": None}
    return {"n": int(len(x)),
            "pearson": float(np.corrcoef(x, y)[0, 1]),
            "spearman": float(spearmanr(x, y).correlation)}


def stats(arr: np.ndarray) -> dict:
    if len(arr) == 0:
        return {"n": 0}
    return {"n": int(len(arr)), "mean": float(np.mean(arr)),
            "std": float(np.std(arr, ddof=0)),
            "min": int(np.min(arr)), "max": int(np.max(arr)),
            "q25": float(np.quantile(arr, 0.25)),
            "q50": float(np.median(arr)),
            "q75": float(np.quantile(arr, 0.75))}


def analyze(alerts: dict, row_map: dict, doy_start: int,
             miss_map: dict, offsets: list[int], split_label: str) -> dict:
    """For event site-years alerted by Stage 1, build per-offset df:
       alert_tstar, alert_DOY, true_L_DOY, mu@(alert+offset). Then corrs/stats."""
    rows = []
    for sy, info in alerts.items():
        if info["is_event"] != 1 or info["alert_tstar"] is None:
            continue
        if pd.isna(info["true_L"]):
            continue
        at = int(info["alert_tstar"])
        L = int(info["true_L"])
        with_h = (miss_map.get(sy, {"prev_year_L_miss": 1})["prev_year_L_miss"] == 0)
        row = {"site": sy[0], "year": sy[1], "alert_tstar": at,
               "alert_DOY": at + doy_start, "true_L": L,
               "L_DOY": L + doy_start, "with_history": int(with_h)}
        for off in offsets:
            t = at + off
            mu = row_map.get((sy[0], sy[1], t))
            if mu is None:
                row[f"mu_off{off}"] = None
                row[f"mu_off{off}_DOY"] = None
            else:
                # mu is in same units as tstar (python index); convert to DOY
                row[f"mu_off{off}"] = float(mu["mu"])
                row[f"mu_off{off}_DOY"] = float(mu["mu"]) + doy_start
        rows.append(row)
    df = pd.DataFrame(rows)
    print(f"\n  [{split_label}]  n_alerted_events = {len(df)}")
    if len(df) == 0:
        return {"empty": True}

    # Distributions
    print(f"    alert_tstar (python idx):  mean={df['alert_tstar'].mean():.1f}  "
          f"std={df['alert_tstar'].std(ddof=0):.1f}  q25={df['alert_tstar'].quantile(0.25):.0f}  "
          f"q50={df['alert_tstar'].median():.0f}  q75={df['alert_tstar'].quantile(0.75):.0f}")
    print(f"    alert_DOY :  mean={df['alert_DOY'].mean():.1f}  "
          f"std={df['alert_DOY'].std(ddof=0):.1f}")
    print(f"    true_L_DOY:  mean={df['L_DOY'].mean():.1f}  "
          f"std={df['L_DOY'].std(ddof=0):.1f}")
    ref = corr_pair(df["alert_tstar"].astype(float).values,
                     df["true_L"].astype(float).values)
    rp = f"{ref['pearson']:.3f}" if ref["pearson"] is not None else "N/A"
    rs = f"{ref['spearman']:.3f}" if ref["spearman"] is not None else "N/A"
    print(f"    corr(alert_tstar, true_L_DOY) [reference]:  pearson={rp}  spearman={rs}  (n={ref['n']})")

    # Per-offset correlations + mu distribution
    per_off = {}
    print(f"\n    --- per-offset mu calibration ---")
    print(f"    {'offset':>6} {'n_matched':>10} {'mu_DOY_mean':>12} {'mu_DOY_std':>11}  "
          f"{'corr_a_pe':>10} {'corr_L_pe':>10} {'corr_a_sp':>10} {'corr_L_sp':>10}")
    for off in offsets:
        sub = df[df[f"mu_off{off}"].notna()].copy()
        n = int(len(sub))
        if n < 5:
            print(f"    {off:>6d} {n:>10d}  (too few matched)")
            per_off[off] = {"n_matched": n}
            continue
        mu_doy = sub[f"mu_off{off}_DOY"].astype(float).values
        a = sub["alert_tstar"].astype(float).values
        L = sub["true_L"].astype(float).values
        ca = corr_pair(a, mu_doy)
        cL = corr_pair(L, mu_doy)
        per_off[off] = {"n_matched": n,
                        "mu_DOY_mean": float(mu_doy.mean()),
                        "mu_DOY_std": float(mu_doy.std(ddof=0)),
                        "corr_alert_mu": ca,
                        "corr_L_mu": cL}
        cap = f"{ca['pearson']:.3f}" if ca["pearson"] is not None else "N/A"
        cLp = f"{cL['pearson']:.3f}" if cL["pearson"] is not None else "N/A"
        cas = f"{ca['spearman']:.3f}" if ca["spearman"] is not None else "N/A"
        cLs = f"{cL['spearman']:.3f}" if cL["spearman"] is not None else "N/A"
        print(f"    {off:>6d} {n:>10d} {mu_doy.mean():>12.1f} {mu_doy.std(ddof=0):>11.1f}  "
              f"{cap:>10} {cLp:>10} {cas:>10} {cLs:>10}")

    # Subcohort breakdown for offset=0
    subcohorts = {}
    print(f"\n    --- subcohort breakdown (offset=0) ---")
    for lab, mask in [("with_history", df.with_history == 1),
                       ("no_history", df.with_history == 0)]:
        sub = df[mask & df["mu_off0"].notna()]
        if len(sub) < 5:
            print(f"    [{lab}] n={len(sub)} (too few)")
            subcohorts[lab] = {"n": int(len(sub))}
            continue
        a = sub["alert_tstar"].astype(float).values
        L = sub["true_L"].astype(float).values
        mu_doy = sub["mu_off0_DOY"].astype(float).values
        ca = corr_pair(a, mu_doy)
        cL = corr_pair(L, mu_doy)
        cap = f"{ca['pearson']:.3f}" if ca["pearson"] is not None else "N/A"
        cLp = f"{cL['pearson']:.3f}" if cL["pearson"] is not None else "N/A"
        print(f"    [{lab}]  n={len(sub)}  mu_DOY_mean={mu_doy.mean():.1f}  "
              f"corr(alert, mu)={cap}  corr(L, mu)={cLp}")
        subcohorts[lab] = {"n": int(len(sub)),
                           "mu_DOY_mean": float(mu_doy.mean()),
                           "corr_alert_mu": ca, "corr_L_mu": cL}

    return {
        "n_alerted_events": int(len(df)),
        "alert_tstar_stats": stats(df["alert_tstar"].values),
        "L_DOY_stats": stats(df["L_DOY"].values),
        "corr_alert_L_DOY_ref": ref,
        "per_offset": per_off,
        "subcohorts_offset0": subcohorts,
        "df": df,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--stage2_ckpt", required=True)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--tau", type=float, required=True,
                    help="Stage 1 first_crossing tau")
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--offsets", default="0,7,14,21,30")
    ap.add_argument("--history_policy", default="rolling",
                    choices=["rolling", "strict_train"])
    ap.add_argument("--history_train_year_max", type=int, default=2021)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    offsets = [int(x) for x in args.offsets.split(",") if x.strip()]

    print(f"\n========== Stage 2 mu diagnostic ==========")
    print(f"[stage1 ckpt] {args.stage1_ckpt}")
    print(f"[stage2 ckpt] {args.stage2_ckpt}")
    print(f"[cfg] tau={args.tau}  k={args.k}  offsets={offsets}")

    # Stage 1 alerts on val + test
    class N: pass
    common = N()
    for f in ["pest", "run", "split_seed", "val_year", "test_year_min", "test_year_max"]:
        setattr(common, f, getattr(args, f))
    common.stage1_ckpt = args.stage1_ckpt
    cache = build_probs(common)
    doy_start = int(C.DOY_START)
    val_alerts = build_stage1_alerts(cache["val_df"], args.tau, args.k)
    test_alerts = build_stage1_alerts(cache["test_df"], args.tau, args.k)
    print(f"[stage1] val alerts={sum(1 for v in val_alerts.values() if v['alert_tstar'] is not None)}/{len(val_alerts)}  "
          f"test alerts={sum(1 for v in test_alerts.values() if v['alert_tstar'] is not None)}/{len(test_alerts)}")

    # Stage 2 forward (test cohort only — phase_r builds on test)
    print(f"\n========== Stage 2 forward (test cohort) ==========")
    test_row_map, _ = get_stage2_row_map(args.stage2_ckpt, args.run, args.pest,
                                          args.split_seed, args.val_year,
                                          args.test_year_min, args.test_year_max,
                                          device=args.device)
    print(f"[stage2] test row_map entries = {len(test_row_map)}")

    # history miss map
    miss_map = make_history_mask(args.pest, args.run, doy_start,
                                  args.history_policy, args.history_train_year_max)

    out = {"args": vars(args)}
    print(f"\n========== test analysis ==========")
    test_res = analyze(test_alerts, test_row_map, doy_start, miss_map, offsets, "test")
    if not test_res.get("empty"):
        test_res["df"].to_csv(out_dir / "stage2_mu_diag_test.csv", index=False)
        test_res.pop("df")
    out["test"] = test_res

    # Optional: val (Stage 2 row_map for val is not built by phase_r — skip unless we extend it)
    out["val"] = {"note": "phase_r_oracle_iou.build_stage2_row_map only builds the test cohort"}

    (out_dir / "stage2_mu_diag_summary.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {out_dir / 'stage2_mu_diag_summary.json'}")


if __name__ == "__main__":
    main()
