"""
Phase T7-eval — Lead-aware Stage 1 single-ckpt integrated evaluation.

Sections (val + test):
  A. alert tau x k Pareto sweep                       (recall, FAR, precision, F1)
     + baseline operating-point row at (--baseline_tau, --baseline_k)
     + F1max operating point on val -> test metrics at the same (k, tau)
  B. days_to_event bin score shape                    (mountain check)
     bins: [0,30) [30,60) [60,90) [90+) vs non_event
     mean / median / frac>=tau / KS vs non_event
  C. Alert-rule comparison + transition matrix vs baseline rule
     7 rules: first_crossing, first_crossing_after_DOY{100,110,120},
              global_peak, causal_peak_confirmed, rolling_local_peak_w14
     Per rule: TP/FP/FN/TN, recall/FAR/precision/F1, lead bucket counts
              (MISSED/TOO_LATE/USEFUL/TOO_EARLY/no_alert), lead stats
     vs baseline rule: 5x5 event transition + 2x2 non-event transition
     Verdict helper table: USEFUL/TOO_EARLY/no_alert per rule + deltas
  D. TP / FP alert_tstar DOY distribution at F1max op
  E. Occurrence AUC (reference)                       p_max, p_top5, p_mean_season

Designed to evaluate phase_t_lead_aware_train.py ckpts but works on any Stage 1a
nowcast XGB ckpt with structurally compatible metadata.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import (
    build_nowcast_samples, build_tabular_from_samples, make_event_labels,
)
from rice.scripts.run_calibrate_event import apply_temperature, fit_temperature_grid
from rice.scripts.run_stage1b_cascade_v2 import derive_alerts, metrics_from_alerts

# Reuse helpers from sibling scripts
from rice.scripts.phase_t_stage1a_pareto import sweep_pareto
from rice.scripts.phase_t_occurrence_diag import (
    aggregate_scores, metrics_at_recall_targets, section_B as occ_section_B,
)
from rice.scripts.phase_t_fp_diagnostic import classify_sites, doy_hist


EVENT_BUCKETS = ["MISSED", "TOO_LATE", "USEFUL", "TOO_EARLY", "no_alert"]
NONEVENT_BUCKETS = ["FP", "TN"]


def event_bucket(lead_days):
    if lead_days is None or pd.isna(lead_days):
        return "no_alert"
    d = int(lead_days)
    if d < 0: return "MISSED"
    if d < 14: return "TOO_LATE"
    if d <= 60: return "USEFUL"
    return "TOO_EARLY"


# ---------- Alert rules ----------

def rule_first_crossing(ts: np.ndarray, ps: np.ndarray, tau: float, k: int,
                        tstar_min: int = 0) -> int | None:
    streak = 0
    for i in range(len(ts)):
        if int(ts[i]) < tstar_min:
            streak = 0
            continue
        if ps[i] >= tau:
            streak += 1
            if streak >= k:
                return int(ts[i])
        else:
            streak = 0
    return None


def rule_global_peak(ts: np.ndarray, ps: np.ndarray, tau: float) -> int | None:
    if len(ps) == 0:
        return None
    idx = int(np.argmax(ps))
    if ps[idx] >= tau:
        return int(ts[idx])
    return None


def rule_causal_peak_confirmed(ts: np.ndarray, ps: np.ndarray, tau: float,
                               confirm_days: int = 5) -> int | None:
    if len(ps) == 0:
        return None
    running_max = -np.inf
    peak_idx = -1
    for i in range(len(ps)):
        if ps[i] > running_max:
            running_max = float(ps[i])
            peak_idx = i
        if peak_idx >= 0 and (i - peak_idx) >= confirm_days and running_max >= tau:
            return int(ts[peak_idx])
    if peak_idx >= 0 and running_max >= tau:
        return int(ts[peak_idx])
    return None


def rule_rolling_local_peak(ts: np.ndarray, ps: np.ndarray, tau: float,
                            window: int = 14) -> int | None:
    for i in range(len(ps)):
        if ps[i] < tau:
            continue
        lo = max(0, i - window + 1)
        if ps[i] >= float(ps[lo:i + 1].max()) - 1e-9:
            return int(ts[i])
    return None


def apply_alert_rule(probs_df: pd.DataFrame, rule_spec: dict, doy_start: int) -> pd.DataFrame:
    """Apply one alert rule to each (site, year)."""
    name = rule_spec["name"]
    tau = float(rule_spec["tau"])
    k = int(rule_spec.get("k", 3))
    rows = []
    for (site, year), g in probs_df.groupby(["site", "year"], sort=False):
        g = g.sort_values("tstar")
        ts = g.tstar.values.astype(int)
        ps = g.p_cal.values.astype(float)
        is_event = int(g.y_event.iloc[0])
        true_L = g.true_L.iloc[0]
        true_R = g.true_R.iloc[0]
        if name == "first_crossing":
            at = rule_first_crossing(ts, ps, tau, k, tstar_min=0)
        elif name.startswith("first_crossing_after_DOY"):
            doy_cut = int(name.replace("first_crossing_after_DOY", ""))
            tstar_min = max(0, doy_cut - doy_start)
            at = rule_first_crossing(ts, ps, tau, k, tstar_min=tstar_min)
        elif name == "global_peak":
            at = rule_global_peak(ts, ps, tau)
        elif name == "causal_peak_confirmed":
            at = rule_causal_peak_confirmed(ts, ps, tau,
                                            confirm_days=int(rule_spec.get("confirm_days", 5)))
        elif name == "rolling_local_peak_w14":
            at = rule_rolling_local_peak(ts, ps, tau,
                                         window=int(rule_spec.get("window", 14)))
        else:
            raise ValueError(f"unknown rule: {name}")
        if is_event == 1 and pd.notna(true_L) and at is not None:
            lead = int(true_L) - int(at)
        else:
            lead = None
        if is_event == 1:
            bucket = event_bucket(lead)
        else:
            bucket = "FP" if at is not None else "TN"
        rows.append({
            "site": str(site), "year": int(year), "is_event": is_event,
            "true_L": (int(true_L) if pd.notna(true_L) else None),
            "true_R": (int(true_R) if pd.notna(true_R) else None),
            "alert_tstar": (int(at) if at is not None else None),
            "lead_days": (int(lead) if lead is not None else None),
            "bucket": bucket,
        })
    return pd.DataFrame(rows)


def alert_metrics(cls_df: pd.DataFrame) -> dict:
    n_event = int((cls_df.is_event == 1).sum())
    n_nonevent = int((cls_df.is_event == 0).sum())
    tp = int(((cls_df.is_event == 1) & (cls_df.alert_tstar.notna())).sum())
    fn = n_event - tp
    fp = int(((cls_df.is_event == 0) & (cls_df.alert_tstar.notna())).sum())
    tn = n_nonevent - fp
    rec = tp / n_event if n_event else float("nan")
    far = fp / n_nonevent if n_nonevent else float("nan")
    prec = tp / max(tp + fp, 1) if (tp + fp) else float("nan")
    f1 = 2 * prec * rec / max(prec + rec, 1e-9) if (prec + rec) > 0 else float("nan")
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn,
            "recall": rec, "FAR": far, "precision": prec, "F1": f1,
            "n_event": n_event, "n_nonevent": n_nonevent, "n_alert": tp + fp}


def event_transition_matrix(base_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    base_ev = base_df[base_df.is_event == 1].set_index(["site", "year"])
    new_ev = new_df[new_df.is_event == 1].set_index(["site", "year"])
    common = base_ev.index.intersection(new_ev.index)
    mat = {b: {c: 0 for c in EVENT_BUCKETS} for b in EVENT_BUCKETS}
    for key in common:
        b = base_ev.loc[key, "bucket"]
        c = new_ev.loc[key, "bucket"]
        mat[b][c] += 1
    return mat


def nonevent_transition_matrix(base_df: pd.DataFrame, new_df: pd.DataFrame) -> dict:
    base_ne = base_df[base_df.is_event == 0].set_index(["site", "year"])
    new_ne = new_df[new_df.is_event == 0].set_index(["site", "year"])
    common = base_ne.index.intersection(new_ne.index)
    mat = {b: {c: 0 for c in NONEVENT_BUCKETS} for b in NONEVENT_BUCKETS}
    for key in common:
        b = base_ne.loc[key, "bucket"]
        c = new_ne.loc[key, "bucket"]
        mat[b][c] += 1
    return mat


def print_rule_result(name: str, split: str, cls_df: pd.DataFrame) -> dict:
    m = alert_metrics(cls_df)
    buckets = Counter(cls_df[cls_df.is_event == 1]["bucket"])
    leads = cls_df.loc[(cls_df.is_event == 1) & cls_df.lead_days.notna(), "lead_days"].astype(int).values
    print(f"  [{split}] {name}")
    print(f"    metrics: TP={m['TP']} FP={m['FP']} FN={m['FN']} TN={m['TN']}  "
          f"R={m['recall']:.3f} FAR={m['FAR']:.3f} P={m['precision']:.3f} F1={m['F1']:.3f}")
    if len(leads):
        print(f"    lead among alerted events: mean={leads.mean():.1f} "
              f"median={float(np.median(leads)):.1f} min={int(leads.min())} max={int(leads.max())} (n={len(leads)})")
    n_event = m["n_event"]
    parts = [f"{b}={buckets.get(b, 0):d}({buckets.get(b, 0)/max(n_event,1)*100:.1f}%)"
             for b in EVENT_BUCKETS]
    print(f"    event buckets ({n_event} total): " + "  ".join(parts))
    return {"metrics": m, "buckets": dict(buckets), "lead_stats": {
        "n": int(len(leads)),
        "mean": float(leads.mean()) if len(leads) else None,
        "median": float(np.median(leads)) if len(leads) else None,
        "min": int(leads.min()) if len(leads) else None,
        "max": int(leads.max()) if len(leads) else None,
    }}


def print_transition(base_name: str, new_name: str, ev_mat: dict, ne_mat: dict) -> None:
    print(f"  -- {base_name}  ->  {new_name} --")
    print(f"    [event site-years]")
    print("    " + " " * 11 + "  ".join(f"{c:>9}" for c in EVENT_BUCKETS))
    for b in EVENT_BUCKETS:
        cells = "  ".join(f"{ev_mat[b][c]:>9d}" for c in EVENT_BUCKETS)
        print(f"    {b:>10}  {cells}")
    print(f"    [non-event site-years]")
    print("    " + " " * 9 + "  ".join(f"{c:>7}" for c in NONEVENT_BUCKETS))
    for b in NONEVENT_BUCKETS:
        cells = "  ".join(f"{ne_mat[b][c]:>7d}" for c in NONEVENT_BUCKETS)
        print(f"    {b:>8}  {cells}")


def build_probs(args) -> dict:
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    C.DOY_START = int(ckpt.get("doy_start", 60))
    C.DOY_END = int(ckpt.get("doy_end", 300))
    add_tpos = bool(ckpt.get("add_tstar_position_feature", False))
    nc_window = int(ckpt.get("nowcast_window", 28))
    nc_stride = int(ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("nowcast_event_time_proxy", "mid"))
    nc_label_mode = str(ckpt.get("nowcast_label_mode", "eventually"))
    nc_label_horizon = ckpt.get("nowcast_label_horizon", None)
    nc_tstart = ckpt.get("nowcast_tstar_start", None)

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples = build_samples_for_run(args.run, get_feature_cols)

    if bool(ckpt.get("site_history_added", False)):
        from rice.scripts.site_history_utils import (
            compute_site_history, append_history_to_samples, HISTORY_FEATURE_DIM,
        )
        history = compute_site_history(
            samples, doy_start=int(C.DOY_START),
            policy=str(ckpt.get("site_history_policy", "rolling")),
            train_year_max=int(ckpt.get("history_train_year_max", 2021)),
        )
        append_history_to_samples(samples, history, doy_start=int(C.DOY_START))
        print(f"[history] appended {HISTORY_FEATURE_DIM} channels (eval; matches ckpt meta)")
    if bool(ckpt.get("phenology_added", False)):
        from rice.scripts.phenology_utils import (
            load_pheno_map, append_pheno_to_samples, PHENO_FEATURE_DIM,
        )
        pheno_map = load_pheno_map()
        append_pheno_to_samples(samples, pheno_map, doy_start=int(C.DOY_START))
        print(f"[phenology] appended {PHENO_FEATURE_DIM} channels (eval; matches ckpt meta)")
    if bool(ckpt.get("derived_weather_added", False)):
        from rice.scripts.derived_weather_utils import (
            append_derived_weather_to_samples, DERIVED_WEATHER_DIM,
        )
        append_derived_weather_to_samples(samples)
        print(f"[derived_weather] appended {DERIVED_WEATHER_DIM} channels (eval; matches ckpt meta)")

    if bool(ckpt.get("neighbor_history_added", False)):
        from rice.scripts.neighbor_history_utils import (
            load_long_events, build_neighbor_index, append_neighbor_to_samples,
            NEIGHBOR_FEATURE_DIM, DEFAULT_DECAY_KM,
        )
        decay_km = float(ckpt.get("neighbor_decay_km", DEFAULT_DECAY_KM))
        ev_df, co_df, _site_years = load_long_events(
            C.PATH_OBS, label_col=getattr(C, "LABEL_COL", "label_event"),
            year_min=getattr(C, "YEAR_MIN", None), year_max=getattr(C, "YEAR_MAX", None),
        )
        nb_index = build_neighbor_index(ev_df, co_df)
        append_neighbor_to_samples(samples, nb_index, doy_start=int(C.DOY_START), decay_km=decay_km)
        print(f"[neighbor] appended {NEIGHBOR_FEATURE_DIM} channels decay_km={decay_km} "
              f"(eval; matches ckpt meta)")

    _, val_seas, test_seas = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )
    nc_kw = dict(window=nc_window, stride=nc_stride, tstar_start=nc_tstart,
                 only_pre_event=nc_only_pre, event_time_proxy=nc_proxy,
                 label_mode=nc_label_mode, label_horizon=nc_label_horizon)
    val_now = build_nowcast_samples(val_seas, **nc_kw)
    test_now = build_nowcast_samples(test_seas, **nc_kw)
    X_val = build_tabular_from_samples(val_now, add_tstar_position_feature=add_tpos)
    X_test = build_tabular_from_samples(test_now, add_tstar_position_feature=add_tpos)
    y_val = make_event_labels(val_now)

    clf = ckpt["trained_states"][0]["sk_model"]
    p_val_raw = clf.predict_proba(X_val)[:, 1]
    p_test_raw = clf.predict_proba(X_test)[:, 1]
    t_best, _ = fit_temperature_grid(y_val, p_val_raw)
    p_val_cal = apply_temperature(p_val_raw, t_best)
    p_test_cal = apply_temperature(p_test_raw, t_best)
    print(f"[stage1] T*={t_best:.3f}  lead_meta={ckpt.get('lead_aware_label', 'baseline-eventually')}")

    def _df(seas, nc, p, split_name):
        sy_meta = {(str(s["site_id"]), int(s["year"])): s for s in seas}
        rows = []
        for s, prob in zip(nc, p):
            key = (str(s["site_id"]), int(s["year"]))
            meta = sy_meta[key]
            ctype = str(meta["censor_type"])
            rows.append({
                "split": split_name, "site": key[0], "year": key[1],
                "tstar": int(s["tstar"]), "p_cal": float(prob),
                "y_event": int(s["y_event"]),
                "true_L": int(meta["L"]) if ctype != "right" else None,
                "true_R": int(meta["R"]) if ctype != "right" else None,
            })
        return pd.DataFrame(rows)

    return {
        "ckpt_meta": {k: ckpt.get(k) for k in
                      ["nowcast_window", "nowcast_event_time_proxy",
                       "lead_aware_label", "lead_filter_stats"]},
        "val_df": _df(val_seas, val_now, p_val_cal, "val"),
        "test_df": _df(test_seas, test_now, p_test_cal, "test"),
        "val_seas": val_seas, "test_seas": test_seas,
    }


def section_A(val_df, test_df, baseline_tau, baseline_k, tau_step, ks):
    print("\n========== A. alert tau x k Pareto sweep ==========")
    tau_grid = np.arange(0.05, 0.95 + 1e-9, tau_step)
    df = sweep_pareto(val_df, test_df, tau_grid, ks)
    # Baseline reference row
    base = df[(df["k"] == baseline_k) & (np.isclose(df["tau"], baseline_tau, atol=tau_step / 2))]
    if base.empty:
        sub = df[df["k"] == baseline_k].copy()
        sub["d"] = (sub["tau"] - baseline_tau).abs()
        base = sub.loc[[sub["d"].idxmin()]]
    b = base.iloc[0]
    print(f"\n  reference baseline (k={baseline_k}, tau={baseline_tau:.3f}):")
    print(f"    val:  R={b['val_recall']:.3f} FAR={b['val_FAR']:.3f} P={b['val_precision']:.3f} F1={b['val_F1']:.3f}")
    print(f"    test: R={b['test_recall']:.3f} FAR={b['test_FAR']:.3f} P={b['test_precision']:.3f} F1={b['test_F1']:.3f}")
    # F1max selection on val
    f1_pick = {}
    print(f"\n  F1max on val (per k):")
    for k in ks:
        sub = df[df["k"] == k]
        best = sub.loc[sub["val_F1"].idxmax()]
        f1_pick[k] = {"tau": float(best["tau"]),
                      "val": {"recall": float(best["val_recall"]),
                              "FAR": float(best["val_FAR"]),
                              "precision": float(best["val_precision"]),
                              "F1": float(best["val_F1"]),
                              "n_alert": int(best["val_n_alert"])},
                      "test": {"recall": float(best["test_recall"]),
                               "FAR": float(best["test_FAR"]),
                               "precision": float(best["test_precision"]),
                               "F1": float(best["test_F1"]),
                               "n_alert": int(best["test_n_alert"])}}
        print(f"    k={k}  tau*={best['tau']:.3f}  |  "
              f"val R={best['val_recall']:.3f} FAR={best['val_FAR']:.3f} F1={best['val_F1']:.3f}  |  "
              f"test R={best['test_recall']:.3f} FAR={best['test_FAR']:.3f} P={best['test_precision']:.3f} F1={best['test_F1']:.3f}")
    # Recall-target rows
    print(f"\n  recall>=0.85/0.90 FAR-min on val (per k, test reported):")
    for k in ks:
        sub = df[df["k"] == k]
        for target in [0.85, 0.90]:
            cands = sub[sub["val_recall"] >= target]
            if cands.empty:
                print(f"    k={k} R>={target:.2f}: (none on val)")
                continue
            best = cands.loc[cands["val_FAR"].idxmin()]
            print(f"    k={k} R>={target:.2f}: tau*={best['tau']:.3f}  |  "
                  f"val R={best['val_recall']:.3f} FAR={best['val_FAR']:.3f}  |  "
                  f"test R={best['test_recall']:.3f} FAR={best['test_FAR']:.3f} P={best['test_precision']:.3f} F1={best['test_F1']:.3f}")
    return {"sweep_df": df, "baseline_row": b.to_dict(), "f1max_picks": f1_pick}


def section_C_alert_rules(val_df: pd.DataFrame, test_df: pd.DataFrame,
                          rule_names: list[str], tau: float, k: int,
                          confirm_days: int, rolling_window: int,
                          doy_start: int) -> dict:
    """Apply each alert rule; print per-rule metrics + lead buckets + lead stats,
    then transition matrices vs baseline (= first rule), then a verdict helper."""
    print("\n========== C. Alert-rule comparison + transitions vs baseline ==========")
    rule_specs = []
    for nm in rule_names:
        rule_specs.append({"name": nm, "tau": float(tau), "k": int(k),
                           "confirm_days": int(confirm_days), "window": int(rolling_window)})
    baseline_name = rule_specs[0]["name"]
    print(f"  rules: baseline={baseline_name}  others={[s['name'] for s in rule_specs[1:]]}")

    cls_cache = {"val": {}, "test": {}}
    results = {"val": {}, "test": {}}
    for split_name, df in [("val", val_df), ("test", test_df)]:
        print(f"\n  --- per-rule metrics ({split_name}) ---")
        for spec in rule_specs:
            cls_df = apply_alert_rule(df, spec, doy_start)
            cls_cache[split_name][spec["name"]] = cls_df
            results[split_name][spec["name"]] = print_rule_result(spec["name"], split_name, cls_df)

    transitions = {"val": {}, "test": {}}
    for split_name in ["val", "test"]:
        print(f"\n  --- transitions vs baseline ({split_name}) ---")
        base_cls = cls_cache[split_name][baseline_name]
        for spec in rule_specs[1:]:
            new_cls = cls_cache[split_name][spec["name"]]
            ev_mat = event_transition_matrix(base_cls, new_cls)
            ne_mat = nonevent_transition_matrix(base_cls, new_cls)
            print_transition(baseline_name, spec["name"], ev_mat, ne_mat)
            transitions[split_name][spec["name"]] = {"event": ev_mat, "nonevent": ne_mat}

    # Verdict helper on test — separate oracle (global_peak) vs causal rules
    print(f"\n  --- verdict helper (test) ---")
    print(f"    [note] global_peak uses future info (oracle); causal rules are operational.")
    b_buckets = results["test"][baseline_name]["buckets"]
    b_useful = b_buckets.get("USEFUL", 0)
    b_te = b_buckets.get("TOO_EARLY", 0)
    b_na = b_buckets.get("no_alert", 0)
    print(f"  {'rule':>34}  {'kind':>8}  {'USEFUL':>7} {'TOO_EARLY':>9} {'no_alert':>8} "
          f"{'recall':>7} {'FAR':>7}  {'dUSEFUL':>8} {'dTOO_EARLY':>10} {'dno_alert':>9}")
    causal_best = None
    for spec in rule_specs:
        nm = spec["name"]
        kind = "oracle" if nm == "global_peak" else "causal"
        bk = results["test"][nm]["buckets"]
        mt = results["test"][nm]["metrics"]
        u = bk.get("USEFUL", 0); te = bk.get("TOO_EARLY", 0); na = bk.get("no_alert", 0)
        print(f"  {nm:>34}  {kind:>8}  {u:>7d} {te:>9d} {na:>8d} {mt['recall']:>7.3f} {mt['FAR']:>7.3f}  "
              f"{u-b_useful:>+8d} {te-b_te:>+10d} {na-b_na:>+9d}")
        if kind == "causal" and (causal_best is None or u > causal_best[1]):
            causal_best = (nm, u, te, na, mt["recall"], mt["FAR"])
    if causal_best is not None:
        nm, u, te, na, rec, far = causal_best
        print(f"\n    => best causal rule by USEFUL: {nm}  "
              f"USEFUL={u} (d={u-b_useful:+d})  TOO_EARLY={te}  no_alert={na}  "
              f"recall={rec:.3f}  FAR={far:.3f}")

    return {"results": results, "transitions": transitions, "cls_cache": cls_cache}


# Fine-grained days_to_event bins for B section
DTE_BINS = [(0, 15), (15, 30), (30, 45), (45, 60), (60, 90), (90, 10**6)]
DTE_BIN_LABELS = ["0_15", "15_30", "30_45", "45_60", "60_90", "90+"]


def section_B_fine(val_df: pd.DataFrame, test_df: pd.DataFrame, tau_a: float) -> dict:
    """days_to_event score shape with fine bins (0-15/15-30/30-45/45-60/60-90/90+).
    Lead-aware target window check: which bin has highest mean score? Should be
    in [lead_min, lead_max] for a successful lead-aware fit; flat/early-shifted
    if shortcut still dominates.
    """
    from scipy.stats import ks_2samp
    print("\n========== B. days_to_event score shape (fine bins) ==========")
    out = {}
    for split_name, df in [("val", val_df), ("test", test_df)]:
        ev = df[(df.y_event == 1) & df.true_L.notna()].copy()
        ev["dte"] = ev["true_L"].astype(float) - ev["tstar"].astype(float)
        ne = df[df.y_event == 0].copy()
        ne_scores = ne["p_cal"].values
        print(f"\n  [{split_name}]  n_event_rows={len(ev)}  n_nonevent_rows={len(ne)}")
        print(f"  {'bin':>10} {'n':>5} {'mean':>6} {'p50':>6} {'p90':>6} {'frac>=tau':>10} {'KS':>6} {'p_KS':>9}")
        b = {"n": int(len(ne)), "mean_p": float(ne_scores.mean()), "p50": float(np.median(ne_scores)),
             "p90": float(np.quantile(ne_scores, 0.9)),
             "frac_above_tau": float((ne_scores >= tau_a).mean())}
        print(f"  {'non_event':>10} {b['n']:>5d} {b['mean_p']:>6.3f} {b['p50']:>6.3f} {b['p90']:>6.3f} "
              f"{b['frac_above_tau']:>10.3f} {'-':>6} {'-':>9}")
        bins_out = {"non_event": b}
        for (lo, hi), label in zip(DTE_BINS, DTE_BIN_LABELS):
            mask = (ev["dte"] >= lo) & (ev["dte"] < hi)
            sub = ev[mask]
            if sub.empty:
                continue
            scores = sub["p_cal"].values.astype(float)
            ks_stat, p_val = ks_2samp(scores, ne_scores)
            cell = {
                "n": int(len(sub)),
                "mean_p": float(scores.mean()), "p50": float(np.median(scores)),
                "p90": float(np.quantile(scores, 0.9)),
                "frac_above_tau": float((scores >= tau_a).mean()),
                "KS": float(ks_stat), "p_KS": float(p_val),
            }
            bins_out[label] = cell
            print(f"  {label:>10} {cell['n']:>5d} {cell['mean_p']:>6.3f} {cell['p50']:>6.3f} {cell['p90']:>6.3f} "
                  f"{cell['frac_above_tau']:>10.3f} {cell['KS']:>6.3f} {cell['p_KS']:>9.2e}")
        # peak bin (mean_p) among event rows
        event_cells = {k: v for k, v in bins_out.items() if k != "non_event"}
        if event_cells:
            peak_bin = max(event_cells.keys(), key=lambda k: event_cells[k]["mean_p"])
            print(f"  -> peak event-bin by mean_p: {peak_bin}  (mean_p={event_cells[peak_bin]['mean_p']:.3f})")
        out[split_name] = bins_out
    return out


def section_D_doy(val_df, test_df, k, tau) -> dict:
    print("\n========== D. TP / FP alert_tstar DOY distribution at F1max op ==========")
    out = {}
    for split_name, df in [("val", val_df), ("test", test_df)]:
        cls = classify_sites(df, tau, k)
        tp = cls[cls.cls == "TP"]
        fp = cls[cls.cls == "FP"]
        print(f"\n  [{split_name}]  n_TP={len(tp)}  n_FP={len(fp)}")
        print(f"  TP alert_tstar DOY hist:")
        print(doy_hist(tp["doy_alert"].tolist()))
        print(f"  FP alert_tstar DOY hist:")
        print(doy_hist(fp["doy_alert"].tolist()))
        out[split_name] = {"n_TP": int(len(tp)), "n_FP": int(len(fp))}
    return out


def section_E_occurrence(val_df, test_df) -> dict:
    print("\n========== E. Occurrence AUC (reference) ==========")
    out = {}
    for split_name, df in [("val", val_df), ("test", test_df)]:
        agg = aggregate_scores(df)
        y = agg["y_event"].values
        print(f"  [{split_name}] n={len(agg)} n_event={int(y.sum())}")
        print(f"  {'aggregation':>14}  {'ROC':>6} {'PR':>6} {'F1max':>6} {'P@R85':>6} {'P@R90':>6}")
        sp = {}
        for col in ["p_max", "p_top5", "p_mean_season"]:
            m = metrics_at_recall_targets(y, agg[col].values)
            sp[col] = m
            p85 = f"{m['P@R85']:.3f}" if m['P@R85'] is not None else "  -  "
            p90 = f"{m['P@R90']:.3f}" if m['P@R90'] is not None else "  -  "
            print(f"  {col:>14}  {m['ROC_AUC']:>6.3f} {m['PR_AUC']:>6.3f} {m['F1max']:>6.3f}  {p85:>6}  {p90:>6}")
        out[split_name] = sp
    return out


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
    ap.add_argument("--baseline_tau", type=float, default=0.575)
    ap.add_argument("--baseline_k", type=int, default=3)
    ap.add_argument("--tau_step", type=float, default=0.025)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--alert_rules", default=("first_crossing,"
                                              "first_crossing_after_DOY100,"
                                              "first_crossing_after_DOY110,"
                                              "first_crossing_after_DOY120,"
                                              "global_peak,"
                                              "causal_peak_confirmed,"
                                              "rolling_local_peak_w14"),
                    help="comma-separated rule names; baseline = first")
    ap.add_argument("--confirm_days", type=int, default=5,
                    help="causal_peak_confirmed confirmation period")
    ap.add_argument("--rolling_window", type=int, default=14,
                    help="rolling_local_peak backward window")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.stage1_ckpt).stem
    print(f"\n========== Lead-aware Stage 1 evaluation :: {label} ==========")

    cache = build_probs(args)
    val_df = cache["val_df"]; test_df = cache["test_df"]
    print(f"[ckpt_meta] {cache['ckpt_meta']}")
    print(f"[cohort] val_rows={len(val_df)}  test_rows={len(test_df)}")

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    A = section_A(val_df, test_df, args.baseline_tau, args.baseline_k, args.tau_step, ks)
    pick_k = args.baseline_k if args.baseline_k in A["f1max_picks"] else ks[-1]
    pick_tau = A["f1max_picks"][pick_k]["tau"]
    B = section_B_fine(val_df, test_df, args.baseline_tau)
    rule_names = [r.strip() for r in args.alert_rules.split(",") if r.strip()]
    C_rules = section_C_alert_rules(val_df, test_df, rule_names,
                                    tau=args.baseline_tau, k=args.baseline_k,
                                    confirm_days=args.confirm_days,
                                    rolling_window=args.rolling_window,
                                    doy_start=int(C.DOY_START))
    D_doy = section_D_doy(val_df, test_df, pick_k, pick_tau)
    E_occ = section_E_occurrence(val_df, test_df)

    # Save sweep CSV + per-rule classification CSVs + JSON summary
    A["sweep_df"].to_csv(out_dir / f"sweep_{label}.csv", index=False)
    for split_name in ["val", "test"]:
        for nm, df in C_rules["cls_cache"][split_name].items():
            df.to_csv(out_dir / f"cls_{split_name}_{nm}_{label}.csv", index=False)
    summary = {
        "label": label,
        "ckpt_meta": cache["ckpt_meta"],
        "baseline_op": {"tau": args.baseline_tau, "k": args.baseline_k,
                        "row": A["baseline_row"]},
        "f1max_picks": A["f1max_picks"],
        "alert_rules": {
            "rule_names": rule_names,
            "tau": args.baseline_tau, "k": args.baseline_k,
            "confirm_days": args.confirm_days, "rolling_window": args.rolling_window,
            "per_rule": C_rules["results"],
            "transitions_vs_baseline": C_rules["transitions"],
        },
        "doy_counts": D_doy,
        "occurrence": E_occ,
        "days_to_event_shape": B,
    }
    (out_dir / f"lead_aware_eval_{label}.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[saved] {out_dir / f'sweep_{label}.csv'}")
    print(f"[saved] {out_dir / f'lead_aware_eval_{label}.json'}")
    print(f"[saved] cls_<split>_<rule>_{label}.csv per rule")


if __name__ == "__main__":
    main()
