"""
Phase T11 — 2-tier cascade diagnostic.

Combines two operating points on the same test/val cohort:
  A gate          : A ckpt (lead14-45 no_history) at (tau_A, k_A)        [high-recall]
  Balanced gate   : dispatch (with_history -> D@(tau_with,k_B); else A@(tau_no,k_B))

Per (site, year):
  no_alert      : A does not alert
  weak_alert    : A alerts AND Balanced does NOT
  strong_alert  : both alert

Sections (val + test):
  1. tier counts by event vs non-event
  2. per-tier TP/FP, precision, recall contribution, FAR contribution
  3. weak_alert composition: TP vs FP rate (key indicator)
  4. tier x bucket (USEFUL/TOO_EARLY/TOO_LATE/MISSED on events)
  5. tier x alert_lead distribution (A's alert)
  6. tier x subcohort (with_history vs no_history)
  + summary: A-gate aggregate vs Balanced-strong aggregate
  + verdict hint based on weak_alert TP_rate

No retraining; two ckpts inference only.
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
from rice.scripts.phase_t_history_subcohort_compare import make_history_mask


def first_crossing_k(ts, ps, tau, k):
    streak = 0
    for i in range(len(ts)):
        if ps[i] >= tau:
            streak += 1
            if streak >= k: return int(ts[i])
        else: streak = 0
    return None


def event_bucket(lead):
    if lead is None or pd.isna(lead): return None
    d = int(lead)
    if d < 0: return "MISSED"
    if d < 14: return "TOO_LATE"
    if d <= 60: return "USEFUL"
    return "TOO_EARLY"


def build_per_sy(probs_df):
    out = {}
    for (s, y), g in probs_df.groupby(["site", "year"], sort=False):
        gs = g.sort_values("tstar")
        sy = (str(s), int(y))
        out[sy] = {"ts": gs["tstar"].values.astype(int),
                   "ps": gs["p_cal"].astype(float).values,
                   "y_event": int(gs["y_event"].iloc[0]),
                   "true_L": gs["true_L"].iloc[0]}
    return out


def per_sy_alert(per_sy, tau, k):
    return {sy: first_crossing_k(d["ts"], d["ps"], tau, k) for sy, d in per_sy.items()}


def dispatch_alert(per_sy_A, per_sy_D, miss_map, tau_no, tau_with, k):
    out = {}
    keys = set(per_sy_A) | set(per_sy_D)
    for sy in keys:
        with_h = (miss_map.get(sy, {"prev_year_L_miss": 1})["prev_year_L_miss"] == 0)
        if with_h:
            d = per_sy_D.get(sy) or per_sy_A.get(sy)
            at = first_crossing_k(d["ts"], d["ps"], tau_with, k)
        else:
            d = per_sy_A.get(sy) or per_sy_D.get(sy)
            at = first_crossing_k(d["ts"], d["ps"], tau_no, k)
        out[sy] = at
    return out


def make_tier_df(per_sy_A, A_alerts, B_alerts, miss_map, doy_start):
    rows = []
    keys = set(per_sy_A)
    for sy in keys:
        site, year = sy
        d = per_sy_A.get(sy)
        if d is None: continue
        is_event = int(d["y_event"])
        true_L = d["true_L"]
        a_at = A_alerts.get(sy)
        b_at = B_alerts.get(sy)
        if a_at is None:
            tier = "no_alert"
        elif b_at is None:
            tier = "weak_alert"
        else:
            tier = "strong_alert"
        at = a_at  # primary alert from A (high recall gate)
        lead = (int(true_L) - int(at)) if (is_event == 1 and pd.notna(true_L) and at is not None) else None
        if is_event == 1:
            bucket = event_bucket(lead) if at is not None else "no_alert_event"
        else:
            bucket = "FP" if at is not None else "TN"
        with_h = (miss_map.get(sy, {"prev_year_L_miss": 1})["prev_year_L_miss"] == 0)
        rows.append({"site": site, "year": year, "is_event": is_event,
                     "with_history": int(with_h),
                     "tier": tier,
                     "A_alert_tstar": (int(a_at) if a_at is not None else None),
                     "B_alert_tstar": (int(b_at) if b_at is not None else None),
                     "lead_days_A": (int(lead) if lead is not None else None),
                     "bucket": bucket,
                     "true_L": (int(true_L) if pd.notna(true_L) else None)})
    return pd.DataFrame(rows)


def analyze_tiers(df: pd.DataFrame, label: str) -> dict:
    print(f"\n========== {label} ==========")
    n_ev = int((df.is_event == 1).sum())
    n_ne = int((df.is_event == 0).sum())
    print(f"  cohort: events={n_ev}  non_events={n_ne}  total={len(df)}")

    print(f"\n  --- 1. tier count by is_event ---")
    print(f"    {'tier':>14}  {'event':>6} {'non_ev':>7} {'total':>6}")
    for tier in ["no_alert", "weak_alert", "strong_alert"]:
        sub = df[df.tier == tier]
        e = int((sub.is_event == 1).sum()); ne = int((sub.is_event == 0).sum())
        print(f"    {tier:>14}  {e:>6d} {ne:>7d} {len(sub):>6d}")

    print(f"\n  --- 2. tier metrics ---")
    print(f"    {'tier':>14}  {'n':>5}  {'TP':>4} {'FP':>4}  {'prec':>5}  "
          f"{'recall_contr':>12} {'FAR_contr':>10}")
    tier_metrics = {}
    for tier in ["no_alert", "weak_alert", "strong_alert"]:
        sub = df[df.tier == tier]
        tp = int((sub.is_event == 1).sum()); fp = int((sub.is_event == 0).sum())
        prec = tp / max(tp + fp, 1); rc = tp / max(n_ev, 1); far_c = fp / max(n_ne, 1)
        print(f"    {tier:>14}  {len(sub):>5d}  {tp:>4d} {fp:>4d}  {prec:>5.3f}  "
              f"{rc:>12.3f} {far_c:>10.3f}")
        tier_metrics[tier] = {"n": int(len(sub)), "TP": tp, "FP": fp,
                                "precision": prec, "recall_contribution": rc,
                                "FAR_contribution": far_c}

    weak = df[df.tier == "weak_alert"]
    tp_w = int((weak.is_event == 1).sum()); fp_w = int((weak.is_event == 0).sum())
    weak_tp_rate = tp_w / max(len(weak), 1)
    print(f"\n  --- 3. weak_alert composition ---")
    print(f"    n_weak={len(weak)}  TP={tp_w}  FP={fp_w}  TP_rate={weak_tp_rate:.3f}")
    if len(weak) > 0:
        print(f"    -> if hard cascade (strong_only): would LOSE {tp_w} events and SAVE {fp_w} FPs")

    print(f"\n  --- 4. tier × bucket (events) ---")
    print(f"    {'tier':>14}  {'MISSED':>6} {'TOO_LATE':>8} {'USEFUL':>6} {'TOO_EARLY':>9} {'no_alert':>8}")
    tier_buckets = {}
    for tier in ["no_alert", "weak_alert", "strong_alert"]:
        sub = df[(df.tier == tier) & (df.is_event == 1)]
        bk = Counter(sub["bucket"])
        ms = int(bk.get("MISSED", 0)); tl = int(bk.get("TOO_LATE", 0))
        us = int(bk.get("USEFUL", 0)); te = int(bk.get("TOO_EARLY", 0))
        na = int(bk.get("no_alert_event", 0))
        print(f"    {tier:>14}  {ms:>6d} {tl:>8d} {us:>6d} {te:>9d} {na:>8d}")
        tier_buckets[tier] = {"MISSED": ms, "TOO_LATE": tl, "USEFUL": us,
                                "TOO_EARLY": te, "no_alert_event": na}

    print(f"\n  --- 5. tier × alert_lead (A's alert, event rows) ---")
    print(f"    {'tier':>14}  {'n':>4}  {'mean':>5} {'median':>6} {'q25':>5} {'q75':>5}")
    tier_leads = {}
    for tier in ["weak_alert", "strong_alert"]:
        sub = df[(df.tier == tier) & (df.is_event == 1) & df.lead_days_A.notna()]
        if len(sub) == 0:
            print(f"    {tier:>14}  {0:>4d}  (none)")
            tier_leads[tier] = {"n": 0}
            continue
        leads = sub["lead_days_A"].astype(int).values
        s = {"n": int(len(sub)), "mean": float(leads.mean()),
             "median": float(np.median(leads)),
             "q25": float(np.quantile(leads, 0.25)),
             "q75": float(np.quantile(leads, 0.75))}
        tier_leads[tier] = s
        print(f"    {tier:>14}  {s['n']:>4d}  {s['mean']:>5.1f} {s['median']:>6.1f} "
              f"{s['q25']:>5.1f} {s['q75']:>5.1f}")

    print(f"\n  --- 6. tier × subcohort ---")
    print(f"    {'tier':>14}  {'wh_ev':>6} {'wh_ne':>6} {'nh_ev':>6} {'nh_ne':>6}")
    tier_sub = {}
    for tier in ["no_alert", "weak_alert", "strong_alert"]:
        sub = df[df.tier == tier]
        wh_e = int(((sub.with_history == 1) & (sub.is_event == 1)).sum())
        wh_ne = int(((sub.with_history == 1) & (sub.is_event == 0)).sum())
        nh_e = int(((sub.with_history == 0) & (sub.is_event == 1)).sum())
        nh_ne = int(((sub.with_history == 0) & (sub.is_event == 0)).sum())
        print(f"    {tier:>14}  {wh_e:>6d} {wh_ne:>6d} {nh_e:>6d} {nh_ne:>6d}")
        tier_sub[tier] = {"wh_event": wh_e, "wh_non_event": wh_ne,
                            "nh_event": nh_e, "nh_non_event": nh_ne}

    a_alerted = df[df.tier != "no_alert"]
    a_tp = int((a_alerted.is_event == 1).sum()); a_fp = int((a_alerted.is_event == 0).sum())
    b_alerted = df[df.tier == "strong_alert"]
    b_tp = int((b_alerted.is_event == 1).sum()); b_fp = int((b_alerted.is_event == 0).sum())
    print(f"\n  --- summary ---")
    print(f"    A-gate (any-alert):       R={a_tp/max(n_ev,1):.3f}  FAR={a_fp/max(n_ne,1):.3f}  "
          f"P={a_tp/max(a_tp+a_fp,1):.3f}")
    print(f"    Balanced-strong only:     R={b_tp/max(n_ev,1):.3f}  FAR={b_fp/max(n_ne,1):.3f}  "
          f"P={b_tp/max(b_tp+b_fp,1):.3f}")
    print(f"    delta (strong vs A):  dR={b_tp/max(n_ev,1)-a_tp/max(n_ev,1):+.3f}  "
          f"dFAR={b_fp/max(n_ne,1)-a_fp/max(n_ne,1):+.3f}")

    if weak_tp_rate < 0.15:
        verdict = "HARD CASCADE OK — weak_alert mostly FP"
    elif weak_tp_rate < 0.30:
        verdict = "PARTIAL — weak_alert has some TP; hard cascade loses real events"
    else:
        verdict = "AVOID HARD CASCADE — weak_alert has many TP; use as Stage 2 confidence feature"
    print(f"  > verdict: weak_TP_rate={weak_tp_rate:.3f}  ->  {verdict}")

    return {"n_event": n_ev, "n_nonevent": n_ne,
            "tier_metrics": tier_metrics,
            "weak_TP_rate": weak_tp_rate,
            "weak_TP": tp_w, "weak_FP": fp_w,
            "tier_buckets_events": tier_buckets,
            "tier_leads": tier_leads,
            "tier_subcohort": tier_sub,
            "A_gate_total": {"TP": a_tp, "FP": a_fp,
                              "recall": a_tp / max(n_ev, 1),
                              "FAR": a_fp / max(n_ne, 1),
                              "precision": a_tp / max(a_tp + a_fp, 1)},
            "Balanced_strong": {"TP": b_tp, "FP": b_fp,
                                  "recall": b_tp / max(n_ev, 1),
                                  "FAR": b_fp / max(n_ne, 1),
                                  "precision": b_tp / max(b_tp + b_fp, 1)},
            "verdict": verdict}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--d_ckpt", required=True)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--tau_A", type=float, required=True)
    ap.add_argument("--k_A", type=int, default=3)
    ap.add_argument("--tau_no", type=float, required=True)
    ap.add_argument("--tau_with", type=float, required=True)
    ap.add_argument("--k_B", type=int, default=3)
    ap.add_argument("--history_train_year_max", type=int, default=2021)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    class N: pass
    common = N()
    for f in ["pest", "run", "split_seed", "val_year", "test_year_min", "test_year_max"]:
        setattr(common, f, getattr(args, f))

    print("\n========== building A probs ==========")
    common.stage1_ckpt = args.baseline_ckpt
    base_cache = build_probs(common)
    doy_start = int(C.DOY_START)
    print("\n========== building D probs ==========")
    common.stage1_ckpt = args.d_ckpt
    d_cache = build_probs(common)

    miss_map = make_history_mask(args.pest, args.run, doy_start,
                                  "rolling", args.history_train_year_max)
    print(f"\n[cfg] A:    tau={args.tau_A}  k={args.k_A}")
    print(f"[cfg] Disp: tau_no={args.tau_no}  tau_with={args.tau_with}  k={args.k_B}")

    out = {"cfg": vars(args), "splits": {}}
    for split_name in ["val", "test"]:
        base_df = base_cache[f"{split_name}_df"]
        d_df = d_cache[f"{split_name}_df"]
        per_sy_A = build_per_sy(base_df)
        per_sy_D = build_per_sy(d_df)
        A_alerts = per_sy_alert(per_sy_A, args.tau_A, args.k_A)
        B_alerts = dispatch_alert(per_sy_A, per_sy_D, miss_map,
                                    args.tau_no, args.tau_with, args.k_B)
        df = make_tier_df(per_sy_A, A_alerts, B_alerts, miss_map, doy_start)
        df.to_csv(out_dir / f"tier_{split_name}.csv", index=False)
        out["splits"][split_name] = analyze_tiers(df, split_name)

    (out_dir / "cascade_2tier_summary.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {out_dir / 'cascade_2tier_summary.json'}")


if __name__ == "__main__":
    main()
