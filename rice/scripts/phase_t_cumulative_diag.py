"""
Phase T13 — Causal cumulative score diagnostic + alert rule comparison.

Part 1 (diagnostic): DOY-checkpoint causal aggregation AUC.
  For each checkpoint DOY in --checkpoints:
    compute per-site-year aggregations using ONLY p_cal[t <= cp]:
      p_mean_so_far, p_sum_so_far, p_area_above_tau_so_far(tau=--ref_tau),
      recent_14d_mean, recent_28d_mean,
      recent_14d_count_above_tau, recent_28d_count_above_tau
    -> ROC-AUC, PR-AUC, P@R85, P@R90 per aggregation

Part 2 (candidate alert rules): Sweep on val + report on test.
  Rule fixed_checkpoint_DOY{X}:
    At day X, alert if p_mean_so_far(X) >= tau.
    alert_DOY = X (no per-sample timing).
  Rule cumulative_mean_crossing:
    alert_DOY = first t where p_mean_so_far(t) >= tau.
  Rule cumulative_area_crossing:
    alert_DOY = first t where p_area_above_tau_so_far(t) >= tau (tau = --ref_tau ref).
  Rule two_condition:
    alert_DOY = first t where p_cal(t) >= tau_instant AND p_mean_so_far(t) >= tau_cum.

Selection: val recall >= target (0.85/0.88/0.90), FAR min, tie-break lead_median.
Report test: recall, FAR, precision, F1, n_alert, no_alert, USEFUL, lead_median.

References:
  - oracle p_mean_season AUC printed for context (NOT used as operating point).
  - first_crossing baseline (--baseline_tau, --baseline_k) printed for comparison.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_recall_curve, roc_auc_score,
)

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.scripts.phase_t_lead_aware_eval import build_probs


def first_crossing_k(ts, ps, tau, k):
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


def build_per_sy(probs_df):
    out = {}
    for (s, y), g in probs_df.groupby(["site", "year"], sort=False):
        gs = g.sort_values("tstar")
        out[(str(s), int(y))] = {
            "ts": gs["tstar"].values.astype(int),
            "ps": gs["p_cal"].values.astype(float),
            "y_event": int(gs["y_event"].iloc[0]),
            "true_L": gs["true_L"].iloc[0],
            "true_R": gs["true_R"].iloc[0],
        }
    return out


def metrics_at_recall(y: np.ndarray, scores: np.ndarray, targets=(0.85, 0.90)) -> dict:
    if len(set(y.tolist())) < 2 or len(y) < 5:
        return {"ROC_AUC": None, "PR_AUC": None,
                **{f"P@R{int(t*100)}": None for t in targets}}
    out = {
        "ROC_AUC": float(roc_auc_score(y, scores)),
        "PR_AUC": float(average_precision_score(y, scores)),
    }
    prec, rec, _ = precision_recall_curve(y, scores)
    for t in targets:
        mask = rec >= t
        out[f"P@R{int(t*100)}"] = float(prec[mask].max()) if mask.any() else None
    return out


def aggregate_so_far(ts: np.ndarray, ps: np.ndarray, cp_tstar: int, ref_tau: float) -> dict:
    """Return aggregations using only ps[ts <= cp_tstar]."""
    mask = ts <= cp_tstar
    sub_ps = ps[mask]
    sub_ts = ts[mask]
    if len(sub_ps) == 0:
        return {"p_mean_so_far": 0.0, "p_sum_so_far": 0.0,
                "p_area_above_tau_so_far": 0.0,
                "recent_14d_mean": 0.0, "recent_28d_mean": 0.0,
                "recent_14d_count_above_tau": 0, "recent_28d_count_above_tau": 0}
    p_mean = float(sub_ps.mean())
    p_sum = float(sub_ps.sum())
    p_area = float(np.maximum(sub_ps - ref_tau, 0).sum())
    # recent windows (ending at cp_tstar)
    recent_14 = sub_ps[sub_ts > cp_tstar - 14]
    recent_28 = sub_ps[sub_ts > cp_tstar - 28]
    r14_mean = float(recent_14.mean()) if len(recent_14) else 0.0
    r28_mean = float(recent_28.mean()) if len(recent_28) else 0.0
    r14_cnt = int((recent_14 >= ref_tau).sum())
    r28_cnt = int((recent_28 >= ref_tau).sum())
    return {"p_mean_so_far": p_mean, "p_sum_so_far": p_sum,
            "p_area_above_tau_so_far": p_area,
            "recent_14d_mean": r14_mean, "recent_28d_mean": r28_mean,
            "recent_14d_count_above_tau": r14_cnt,
            "recent_28d_count_above_tau": r28_cnt}


def diagnose_checkpoints(per_sy: dict, checkpoints: list, ref_tau: float, doy_start: int) -> pd.DataFrame:
    """Return DataFrame: (checkpoint_DOY, aggregation, ROC_AUC, PR_AUC, P@R85, P@R90)."""
    rows = []
    aggs = ["p_mean_so_far", "p_sum_so_far", "p_area_above_tau_so_far",
            "recent_14d_mean", "recent_28d_mean",
            "recent_14d_count_above_tau", "recent_28d_count_above_tau"]
    for cp_doy in checkpoints:
        cp_tstar = int(cp_doy - doy_start)
        records = []
        for sy, d in per_sy.items():
            agg = aggregate_so_far(d["ts"], d["ps"], cp_tstar, ref_tau)
            agg["y_event"] = d["y_event"]
            records.append(agg)
        df = pd.DataFrame(records)
        y = df["y_event"].values
        for col in aggs:
            m = metrics_at_recall(y, df[col].astype(float).values)
            rows.append({"checkpoint_DOY": cp_doy, "aggregation": col, **m})
    return pd.DataFrame(rows)


def oracle_p_mean_season(per_sy: dict) -> dict:
    rows = []
    for sy, d in per_sy.items():
        rows.append({"y_event": d["y_event"], "p_mean_season": float(d["ps"].mean())})
    df = pd.DataFrame(rows)
    return metrics_at_recall(df["y_event"].values, df["p_mean_season"].values)


# ---------- Alert rules ----------

def rule_fixed_checkpoint(per_sy: dict, cp_doy: int, tau: float, doy_start: int,
                          ref_tau: float):
    """At cp_doy, alert if p_mean_so_far(cp) >= tau. alert_DOY = cp_doy."""
    alerts = {}
    cp_tstar = int(cp_doy - doy_start)
    for sy, d in per_sy.items():
        mask = d["ts"] <= cp_tstar
        if not mask.any():
            alerts[sy] = None
            continue
        m = float(d["ps"][mask].mean())
        alerts[sy] = cp_tstar if m >= tau else None
    return alerts


def rule_cumulative_mean_crossing(per_sy: dict, tau: float, min_tstar: int = 0):
    alerts = {}
    for sy, d in per_sy.items():
        ts = d["ts"]; ps = d["ps"]
        cs = np.cumsum(ps) / np.arange(1, len(ps) + 1)
        idx_cross = None
        for i in range(len(ts)):
            if int(ts[i]) < min_tstar: continue
            if cs[i] >= tau:
                idx_cross = i
                break
        alerts[sy] = int(ts[idx_cross]) if idx_cross is not None else None
    return alerts


def rule_cumulative_area_crossing(per_sy: dict, tau: float, ref_tau: float, min_tstar: int = 0):
    alerts = {}
    for sy, d in per_sy.items():
        ts = d["ts"]; ps = d["ps"]
        excess = np.maximum(ps - ref_tau, 0)
        area = np.cumsum(excess)
        idx_cross = None
        for i in range(len(ts)):
            if int(ts[i]) < min_tstar: continue
            if area[i] >= tau:
                idx_cross = i
                break
        alerts[sy] = int(ts[idx_cross]) if idx_cross is not None else None
    return alerts


def rule_two_condition(per_sy: dict, tau_instant: float, tau_cum: float, min_tstar: int = 0):
    alerts = {}
    for sy, d in per_sy.items():
        ts = d["ts"]; ps = d["ps"]
        cs = np.cumsum(ps) / np.arange(1, len(ps) + 1)
        idx_cross = None
        for i in range(len(ts)):
            if int(ts[i]) < min_tstar: continue
            if ps[i] >= tau_instant and cs[i] >= tau_cum:
                idx_cross = i; break
        alerts[sy] = int(ts[idx_cross]) if idx_cross is not None else None
    return alerts


def classify(per_sy, alerts, doy_start):
    rows = []
    for sy, at in alerts.items():
        d = per_sy[sy]
        is_event = int(d["y_event"])
        true_L = d["true_L"]
        lead = (int(true_L) - int(at)) if (is_event == 1 and pd.notna(true_L) and at is not None) else None
        bucket = event_bucket(lead) if is_event == 1 else ("FP" if at is not None else "TN")
        rows.append({"site": sy[0], "year": sy[1], "is_event": is_event,
                     "alert_tstar": (int(at) if at is not None else None),
                     "alert_DOY": ((int(at) + int(doy_start)) if at is not None else None),
                     "lead_days": (int(lead) if lead is not None else None),
                     "bucket": bucket,
                     "true_L": (int(true_L) if pd.notna(true_L) else None)})
    return pd.DataFrame(rows)


def metrics(df):
    n_e = int((df.is_event == 1).sum()); n_ne = int((df.is_event == 0).sum())
    tp = int(((df.is_event == 1) & df.alert_tstar.notna()).sum())
    fp = int(((df.is_event == 0) & df.alert_tstar.notna()).sum())
    bk = Counter(df[df.is_event == 1]["bucket"])
    leads = df.loc[(df.is_event == 1) & df.lead_days.notna(), "lead_days"].astype(int).values
    prec = tp / max(tp + fp, 1); rec = tp / max(n_e, 1)
    return {"recall": rec, "FAR": fp / max(n_ne, 1),
            "precision": prec,
            "F1": 2 * prec * rec / max(prec + rec, 1e-9) if (prec + rec) > 0 else float("nan"),
            "n_alert": tp + fp, "n_event": n_e,
            "no_alert": int(bk.get("no_alert", 0)),
            "TOO_LATE": int(bk.get("TOO_LATE", 0)),
            "MISSED": int(bk.get("MISSED", 0)),
            "USEFUL": int(bk.get("USEFUL", 0)),
            "lead_median": float(np.median(leads)) if len(leads) else None,
            "lead_mean": float(leads.mean()) if len(leads) else None}


def sweep_select(per_sy_val, per_sy_test, rule_fn, tau_grid, targets, doy_start,
                  rule_name="rule"):
    """rule_fn(per_sy, tau) -> alerts. Sweep tau, select best on val per target."""
    rows = []
    for tau in tau_grid:
        v_alerts = rule_fn(per_sy_val, float(tau))
        v_cls = classify(per_sy_val, v_alerts, doy_start)
        v_m = metrics(v_cls)
        rows.append({"tau": float(tau), **{f"val_{k_}": v for k_, v in v_m.items()}})
    sweep_df = pd.DataFrame(rows)
    picks = {}
    for tgt in targets:
        cands = sweep_df[sweep_df["val_recall"] >= tgt]
        if cands.empty:
            picks[tgt] = None
            continue
        c = cands.copy(); c["_lead"] = c["val_lead_median"].fillna(999.0)
        c = c.sort_values(["val_FAR", "_lead", "tau"]).iloc[0]
        tau_sel = float(c["tau"])
        # Test
        t_alerts = rule_fn(per_sy_test, tau_sel)
        t_cls = classify(per_sy_test, t_alerts, doy_start)
        t_m = metrics(t_cls)
        picks[tgt] = {"rule": rule_name, "tau": tau_sel,
                       "val": {k_.replace("val_", ""): v for k_, v in c.items() if k_.startswith("val_")},
                       "test": t_m}
    return sweep_df, picks


def sweep_select_2d(per_sy_val, per_sy_test, rule_fn, tau_inst_grid, tau_cum_grid,
                     targets, doy_start, rule_name="two_condition"):
    rows = []
    for ti in tau_inst_grid:
        for tc in tau_cum_grid:
            v_alerts = rule_fn(per_sy_val, float(ti), float(tc))
            v_cls = classify(per_sy_val, v_alerts, doy_start)
            v_m = metrics(v_cls)
            rows.append({"tau_inst": float(ti), "tau_cum": float(tc),
                         **{f"val_{k_}": v for k_, v in v_m.items()}})
    sweep_df = pd.DataFrame(rows)
    picks = {}
    for tgt in targets:
        cands = sweep_df[sweep_df["val_recall"] >= tgt]
        if cands.empty:
            picks[tgt] = None; continue
        c = cands.copy(); c["_lead"] = c["val_lead_median"].fillna(999.0)
        c = c.sort_values(["val_FAR", "_lead"]).iloc[0]
        ti = float(c["tau_inst"]); tc = float(c["tau_cum"])
        t_alerts = rule_fn(per_sy_test, ti, tc)
        t_cls = classify(per_sy_test, t_alerts, doy_start)
        t_m = metrics(t_cls)
        picks[tgt] = {"rule": rule_name, "tau_instant": ti, "tau_cum": tc,
                       "val": {k_.replace("val_", ""): v for k_, v in c.items() if k_.startswith("val_")},
                       "test": t_m}
    return sweep_df, picks


def print_pick(label, pick):
    if pick is None:
        print(f"  {label}: no qualifying val cell")
        return
    t = pick["test"]; v = pick["val"]
    tau_str = f"tau={pick.get('tau', 'N/A'):.3f}" if pick.get('tau') is not None else \
              f"tau_inst={pick.get('tau_instant'):.3f} tau_cum={pick.get('tau_cum'):.3f}"
    print(f"  {label}  {tau_str}")
    print(f"    val:  R={v['recall']:.3f} FAR={v['FAR']:.3f} noA={int(v['no_alert'])} USE={int(v['USEFUL'])}")
    print(f"    test: R={t['recall']:.3f} FAR={t['FAR']:.3f} P={t['precision']:.3f} "
          f"F1={t['F1']:.3f} noA={int(t['no_alert'])} TL={int(t['TOO_LATE'])} "
          f"MS={int(t['MISSED'])} USE={int(t['USEFUL'])} lead_med={t['lead_median']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--label", default="")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--checkpoints", default="100,120,140,160,180,200")
    ap.add_argument("--ref_tau", type=float, default=0.55,
                    help="reference tau for area_above_tau & count_above_tau")
    ap.add_argument("--baseline_tau", type=float, default=0.55)
    ap.add_argument("--baseline_k", type=int, default=3)
    ap.add_argument("--recall_targets", default="0.85,0.88,0.90")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.stage1_ckpt).stem
    print(f"\n========== Causal cumulative diagnostic :: {label} ==========")

    cache = build_probs(args)
    doy_start = int(C.DOY_START)
    val_df = cache["val_df"]; test_df = cache["test_df"]
    per_sy_val = build_per_sy(val_df)
    per_sy_test = build_per_sy(test_df)
    print(f"[cohort] val_sy={len(per_sy_val)} test_sy={len(per_sy_test)}  DOY_START={doy_start}")

    cps = [int(x) for x in args.checkpoints.split(",")]
    targets = [float(x) for x in args.recall_targets.split(",")]

    # ===== Part 1: checkpoint AUC =====
    print(f"\n========== Part 1: checkpoint AUC (ref_tau={args.ref_tau}) ==========")
    for split_name, per_sy in [("val", per_sy_val), ("test", per_sy_test)]:
        print(f"\n  [{split_name}]")
        diag = diagnose_checkpoints(per_sy, cps, args.ref_tau, doy_start)
        diag["split"] = split_name
        diag.to_csv(out_dir / f"checkpoint_auc_{split_name}.csv", index=False)
        oracle = oracle_p_mean_season(per_sy)
        oracle_auc = oracle["ROC_AUC"]; oracle_p85 = oracle.get("P@R85"); oracle_p90 = oracle.get("P@R90")
        print(f"    oracle p_mean_season:  ROC={oracle_auc:.3f}  "
              f"P@R85={oracle_p85}  P@R90={oracle_p90}  [reference, NOT operational]")
        print(f"    {'agg':>32}  {'cpDOY':>5}  {'ROC':>5} {'PR':>5} {'P@R85':>6} {'P@R90':>6}")
        for _, r in diag.iterrows():
            auc = r["ROC_AUC"]; pr = r["PR_AUC"]; p85 = r.get("P@R85"); p90 = r.get("P@R90")
            auc_s = f"{auc:.3f}" if auc is not None else "N/A"
            pr_s = f"{pr:.3f}" if pr is not None else "N/A"
            p85_s = f"{p85:.3f}" if p85 is not None else "N/A"
            p90_s = f"{p90:.3f}" if p90 is not None else "N/A"
            print(f"    {r['aggregation']:>32}  {int(r['checkpoint_DOY']):>5d}  "
                  f"{auc_s:>5} {pr_s:>5} {p85_s:>6} {p90_s:>6}")

    # ===== Baseline first_crossing =====
    print(f"\n========== Baseline first_crossing (tau={args.baseline_tau}, k={args.baseline_k}) ==========")
    for split_name, per_sy in [("val", per_sy_val), ("test", per_sy_test)]:
        alerts = {sy: first_crossing_k(d["ts"], d["ps"], args.baseline_tau, args.baseline_k)
                  for sy, d in per_sy.items()}
        cls = classify(per_sy, alerts, doy_start)
        m = metrics(cls)
        print(f"  [{split_name}] R={m['recall']:.3f} FAR={m['FAR']:.3f} P={m['precision']:.3f} "
              f"F1={m['F1']:.3f} noA={m['no_alert']} USE={m['USEFUL']} lead_med={m['lead_median']}")

    # ===== Part 2: alert rule candidates =====
    print(f"\n========== Part 2: alert rule candidates ==========")
    out = {"label": label, "args": vars(args), "checkpoint_AUCs": {}, "rules": {}}

    tau_mean_grid = np.round(np.arange(0.25, 0.65 + 1e-9, 0.025), 4)
    tau_area_grid = np.round(np.arange(0, 30, 0.5), 4)

    # Rule cumulative_mean_crossing
    print(f"\n--- Rule cumulative_mean_crossing ---")
    sweep_df, picks = sweep_select(
        per_sy_val, per_sy_test,
        lambda per_sy, tau: rule_cumulative_mean_crossing(per_sy, tau, min_tstar=0),
        tau_mean_grid, targets, doy_start, "cumulative_mean_crossing")
    sweep_df.to_csv(out_dir / "sweep_cum_mean.csv", index=False)
    for tgt in targets:
        print_pick(f"R>={tgt:.2f}  cumulative_mean", picks[tgt])
    out["rules"]["cumulative_mean_crossing"] = picks

    # Rule cumulative_area_crossing
    print(f"\n--- Rule cumulative_area_crossing (ref_tau={args.ref_tau}) ---")
    sweep_df, picks = sweep_select(
        per_sy_val, per_sy_test,
        lambda per_sy, tau: rule_cumulative_area_crossing(per_sy, tau, args.ref_tau, min_tstar=0),
        tau_area_grid, targets, doy_start, "cumulative_area_crossing")
    sweep_df.to_csv(out_dir / "sweep_cum_area.csv", index=False)
    for tgt in targets:
        print_pick(f"R>={tgt:.2f}  cumulative_area", picks[tgt])
    out["rules"]["cumulative_area_crossing"] = picks

    # Rule two_condition (instant + cumulative)
    print(f"\n--- Rule two_condition (p_cal >= tau_inst AND cum_mean >= tau_cum) ---")
    tau_inst_grid = np.round(np.arange(0.45, 0.70 + 1e-9, 0.025), 4)
    tau_cum_grid = np.round(np.arange(0.30, 0.55 + 1e-9, 0.025), 4)
    sweep_df, picks = sweep_select_2d(
        per_sy_val, per_sy_test,
        lambda per_sy, ti, tc: rule_two_condition(per_sy, ti, tc, min_tstar=0),
        tau_inst_grid, tau_cum_grid, targets, doy_start, "two_condition")
    sweep_df.to_csv(out_dir / "sweep_two_cond.csv", index=False)
    for tgt in targets:
        print_pick(f"R>={tgt:.2f}  two_condition", picks[tgt])
    out["rules"]["two_condition"] = picks

    # Rule fixed_checkpoint (only a few representative DOYs)
    print(f"\n--- Rule fixed_checkpoint (per DOY) ---")
    fixed_results = {}
    for cp_doy in cps:
        sweep_df, picks = sweep_select(
            per_sy_val, per_sy_test,
            lambda per_sy, tau, cp_doy=cp_doy: rule_fixed_checkpoint(per_sy, cp_doy, tau, doy_start, args.ref_tau),
            tau_mean_grid, targets, doy_start, f"fixed_DOY{cp_doy}")
        fixed_results[cp_doy] = picks
        for tgt in targets:
            print_pick(f"R>={tgt:.2f}  fixed_DOY{cp_doy}", picks[tgt])
    out["rules"]["fixed_checkpoint"] = {str(k): v for k, v in fixed_results.items()}

    # Side-by-side
    print(f"\n========== Side-by-side (test) ==========")
    print(f"  {'target':>8}  {'method':>30}  {'R':>5} {'FAR':>5} {'P':>5} {'F1':>5} {'noA':>4} {'USE':>4} {'lead_med':>8}")
    methods = [("cumulative_mean_crossing", out["rules"]["cumulative_mean_crossing"]),
               ("cumulative_area_crossing", out["rules"]["cumulative_area_crossing"]),
               ("two_condition", out["rules"]["two_condition"])]
    for cp in cps:
        methods.append((f"fixed_DOY{cp}", out["rules"]["fixed_checkpoint"][str(cp)]))
    for tgt in targets:
        for name, picks in methods:
            pick = picks.get(tgt) if isinstance(picks, dict) else None
            if pick is None:
                print(f"  R>={tgt:.2f}  {name:>30}  (no cell)")
                continue
            t = pick["test"]
            lm = t["lead_median"]; lm_s = f"{lm:.1f}" if lm is not None else "N/A"
            print(f"  R>={tgt:.2f}  {name:>30}  {t['recall']:>5.3f} {t['FAR']:>5.3f} "
                  f"{t['precision']:>5.3f} {t['F1']:>5.3f} {int(t['no_alert']):>4d} "
                  f"{int(t['USEFUL']):>4d} {lm_s:>8}")

    (out_dir / f"cumulative_diag_summary_{label}.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {out_dir / f'cumulative_diag_summary_{label}.json'}")


if __name__ == "__main__":
    main()
