"""
Phase T7d — USEFUL-aware tau x k Pareto sweep (first_crossing rule fixed).

For each (tau, k) cell on val + test:
  - recall, FAR, precision, F1, n_alert
  - bucket counts: USEFUL / TOO_EARLY / TOO_LATE / MISSED / no_alert (event site-years)
  - bucket percentages (e2e = / n_event; alert = / n_alert_event)
  - alert lead mean / median
  - USEFUL_e2e = USEFUL / n_event_sites (operational end-to-end yield)
  - USEFUL_alert = USEFUL / n_alert_event (precision of alerted timing)

Selection (val-only, test reported for the same cell):
  (1) For each recall target in --recall_targets: max USEFUL_e2e on val
  (2) USEFUL_alert >= --useful_alert_min on val: FAR-min
  (3) Per-k top-5 by USEFUL_e2e on val

Output: wide val+test sweep CSV, selection JSON, console tables.
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
from rice.scripts.phase_t_lead_aware_eval import (
    build_probs, apply_alert_rule, alert_metrics,
)


def sweep_useful(probs_df: pd.DataFrame, tau_grid: np.ndarray, ks: list[int],
                 doy_start: int) -> pd.DataFrame:
    rows = []
    for k in ks:
        for tau in tau_grid:
            spec = {"name": "first_crossing", "tau": float(tau), "k": int(k)}
            cls = apply_alert_rule(probs_df, spec, doy_start)
            m = alert_metrics(cls)
            ev = cls[cls.is_event == 1]
            n_event = m["n_event"]
            buckets = Counter(ev["bucket"])
            useful = int(buckets.get("USEFUL", 0))
            too_early = int(buckets.get("TOO_EARLY", 0))
            too_late = int(buckets.get("TOO_LATE", 0))
            missed = int(buckets.get("MISSED", 0))
            no_alert = int(buckets.get("no_alert", 0))
            alert_event = useful + too_early + too_late + missed
            leads = ev.loc[ev.lead_days.notna(), "lead_days"].astype(int).values
            rows.append({
                "k": int(k), "tau": float(tau),
                "recall": float(m["recall"]), "FAR": float(m["FAR"]),
                "precision": float(m["precision"]), "F1": float(m["F1"]),
                "n_alert": int(m["n_alert"]), "n_event": int(n_event),
                "n_alert_event": int(alert_event),
                "USEFUL": useful, "TOO_EARLY": too_early, "TOO_LATE": too_late,
                "MISSED": missed, "no_alert": no_alert,
                "USEFUL_e2e": useful / max(n_event, 1),
                "USEFUL_alert": useful / max(alert_event, 1),
                "TOO_EARLY_e2e": too_early / max(n_event, 1),
                "TOO_LATE_e2e": too_late / max(n_event, 1),
                "no_alert_e2e": no_alert / max(n_event, 1),
                "lead_mean": float(leads.mean()) if len(leads) else float("nan"),
                "lead_median": float(np.median(leads)) if len(leads) else float("nan"),
            })
    return pd.DataFrame(rows)


def merge_wide(val_sweep: pd.DataFrame, test_sweep: pd.DataFrame) -> pd.DataFrame:
    return val_sweep.merge(test_sweep, on=["k", "tau"], suffixes=("_val", "_test"))


def _pick_row(df: pd.DataFrame, idx) -> dict:
    r = df.loc[idx]
    return {col: (float(r[col]) if isinstance(r[col], (np.floating, float)) else
                  (int(r[col]) if isinstance(r[col], (np.integer, int)) else r[col]))
            for col in df.columns}


def select_recall_then_useful(merged: pd.DataFrame, target: float) -> dict | None:
    cands = merged[merged["recall_val"] >= target]
    if cands.empty:
        return None
    return _pick_row(merged, cands["USEFUL_e2e_val"].idxmax())


def select_useful_alert(merged: pd.DataFrame, useful_alert_min: float) -> dict | None:
    cands = merged[merged["USEFUL_alert_val"] >= useful_alert_min]
    if cands.empty:
        return None
    return _pick_row(merged, cands["FAR_val"].idxmin())


def fmt_pct(v): return f"{v*100:5.1f}%"


def print_row(label: str, r: dict) -> None:
    print(f"  {label}")
    print(f"    op: k={r['k']}  tau={r['tau']:.3f}")
    print(f"    val:  R={r['recall_val']:.3f}  FAR={r['FAR_val']:.3f}  P={r['precision_val']:.3f}  "
          f"F1={r['F1_val']:.3f}  n_alert={int(r['n_alert_val'])}")
    print(f"          USEFUL={int(r['USEFUL_val'])} ({fmt_pct(r['USEFUL_e2e_val'])} e2e | "
          f"{fmt_pct(r['USEFUL_alert_val'])} of alerted_events)")
    print(f"          TOO_EARLY={int(r['TOO_EARLY_val'])} ({fmt_pct(r['TOO_EARLY_e2e_val'])})  "
          f"TOO_LATE={int(r['TOO_LATE_val'])}  no_alert={int(r['no_alert_val'])} ({fmt_pct(r['no_alert_e2e_val'])})")
    lm_v = r["lead_mean_val"]; lmd_v = r["lead_median_val"]
    if not (isinstance(lm_v, float) and np.isnan(lm_v)):
        print(f"          lead mean={lm_v:.1f}  median={lmd_v:.1f}")
    print(f"    test: R={r['recall_test']:.3f}  FAR={r['FAR_test']:.3f}  P={r['precision_test']:.3f}  "
          f"F1={r['F1_test']:.3f}  n_alert={int(r['n_alert_test'])}")
    print(f"          USEFUL={int(r['USEFUL_test'])} ({fmt_pct(r['USEFUL_e2e_test'])} e2e | "
          f"{fmt_pct(r['USEFUL_alert_test'])} of alerted_events)")
    print(f"          TOO_EARLY={int(r['TOO_EARLY_test'])} ({fmt_pct(r['TOO_EARLY_e2e_test'])})  "
          f"TOO_LATE={int(r['TOO_LATE_test'])}  no_alert={int(r['no_alert_test'])} ({fmt_pct(r['no_alert_e2e_test'])})")
    lm_t = r["lead_mean_test"]; lmd_t = r["lead_median_test"]
    if not (isinstance(lm_t, float) and np.isnan(lm_t)):
        print(f"          lead mean={lm_t:.1f}  median={lmd_t:.1f}")


def print_top_per_k(merged: pd.DataFrame, top: int = 5) -> None:
    print(f"\n  --- per-k top {top} by USEFUL_e2e on val ---")
    for k in sorted(merged["k"].unique()):
        sub = merged[merged["k"] == k].nlargest(top, "USEFUL_e2e_val")
        print(f"\n  [k={k}]  {'tau':>6} | "
              f"{'v_R':>5} {'v_FAR':>6} {'v_USE':>6} {'v_e2e':>7} {'v_alt':>7} | "
              f"{'t_R':>5} {'t_FAR':>6} {'t_USE':>6} {'t_e2e':>7} {'t_alt':>7}")
        for _, r in sub.iterrows():
            print(f"       {r['tau']:>6.3f} | "
                  f"{r['recall_val']:>5.3f} {r['FAR_val']:>6.3f} {int(r['USEFUL_val']):>6d} "
                  f"{r['USEFUL_e2e_val']*100:>6.1f}% {r['USEFUL_alert_val']*100:>6.1f}% | "
                  f"{r['recall_test']:>5.3f} {r['FAR_test']:>6.3f} {int(r['USEFUL_test']):>6d} "
                  f"{r['USEFUL_e2e_test']*100:>6.1f}% {r['USEFUL_alert_test']*100:>6.1f}%")


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
    ap.add_argument("--tau_min", type=float, default=0.05)
    ap.add_argument("--tau_max", type=float, default=0.95)
    ap.add_argument("--tau_step", type=float, default=0.025)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--recall_targets", default="0.83,0.85",
                    help="val recall thresholds; for each, pick max USEFUL_e2e")
    ap.add_argument("--useful_alert_min", type=float, default=0.40,
                    help="USEFUL_alert floor for the secondary selection (FAR-min)")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.stage1_ckpt).stem
    print(f"\n========== USEFUL-aware Pareto sweep :: {label} ==========")

    cache = build_probs(args)
    val_df = cache["val_df"]; test_df = cache["test_df"]
    doy_start = int(C.DOY_START)
    print(f"[ckpt_meta] {cache['ckpt_meta']}")
    print(f"[cohort] val_rows={len(val_df)} test_rows={len(test_df)}  DOY_START={doy_start}")

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    tau_grid = np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step)
    print(f"[sweep] ks={ks}  taus={len(tau_grid)}  rule=first_crossing")

    val_sweep = sweep_useful(val_df, tau_grid, ks, doy_start)
    test_sweep = sweep_useful(test_df, tau_grid, ks, doy_start)
    merged = merge_wide(val_sweep, test_sweep)
    merged.to_csv(out_dir / f"useful_sweep_{label}.csv", index=False)

    selections = {}
    print(f"\n----- Selection (1): val recall >= R_target -> max USEFUL_e2e -----")
    targets = [float(x) for x in args.recall_targets.split(",") if x.strip()]
    for tgt in targets:
        pick = select_recall_then_useful(merged, tgt)
        key = f"recall>={tgt:.2f}_then_max_USEFUL_e2e"
        selections[key] = pick
        if pick is None:
            print(f"\n  recall>={tgt:.2f}: (no qualifying cell on val)")
        else:
            print_row(f"recall>={tgt:.2f}  -> max USEFUL_e2e", pick)

    print(f"\n----- Selection (2): val USEFUL_alert >= {args.useful_alert_min:.2f} -> FAR-min -----")
    pick2 = select_useful_alert(merged, args.useful_alert_min)
    selections[f"USEFUL_alert>={args.useful_alert_min}_then_FAR_min"] = pick2
    if pick2 is None:
        print(f"\n  USEFUL_alert>={args.useful_alert_min:.2f}: (no qualifying cell on val)")
    else:
        print_row(f"USEFUL_alert>={args.useful_alert_min:.2f}  -> FAR-min", pick2)

    print_top_per_k(merged, top=5)

    (out_dir / f"useful_selections_{label}.json").write_text(
        json.dumps(selections, indent=2, default=str))
    print(f"\n[saved] {out_dir / f'useful_sweep_{label}.csv'}")
    print(f"[saved] {out_dir / f'useful_selections_{label}.json'}")


if __name__ == "__main__":
    main()
