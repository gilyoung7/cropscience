"""
Phase T9c — Sub-cohort decomposition + hybrid gate simulation.

Compares baseline lead14-45_ignore ckpt vs D lead14-45_history_rolling ckpt
on the same test (and val) cohort, decomposed by history availability.

Sections:
  1. coverage table — prev_year_L missing rate, avg3y missing rate, val/test split
  2. event-level alert overlap at given (tau, k):
       both alerted / baseline alerted ∩ D missed (lost by D) /
       D alerted ∩ baseline missed (gained by D) / both missed
       broken down by with-history vs no-history
  3. with-history sub-cohort Pareto:
       baseline vs D on the SAME sub-cohort (apples-to-apples)
  4. no-history sub-cohort Pareto:
       baseline vs D on the no-history sub-cohort
  5. hybrid gate:
       score(site, year, tstar) = D_score   if (site, year) has history
                                 baseline_score else
       sweep (tau, k) on hybrid; report best at recall>=0.90/0.92
       compare with baseline-only and D-only

No retraining. Two ckpts -> two build_probs calls.
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
from rice.scripts.phase_t_lead_aware_eval import (
    build_probs, apply_alert_rule, alert_metrics,
)
from rice.scripts.site_history_utils import compute_site_history
from rice.scripts.run_eval import build_samples_for_run


def make_history_mask(pest: str, run: int, doy_start: int,
                       policy: str, train_year_max: int) -> dict:
    """Returns {(site, year): {prev_year_L_miss, site_avg_L_recent3y_miss}}."""
    _, gfc = resolve_pest(pest)
    _, _, _, samples = build_samples_for_run(run, gfc)
    hist = compute_site_history(samples, doy_start=doy_start, policy=policy,
                                 train_year_max=train_year_max)
    return {k: {"prev_year_L_miss": int(v["prev_year_L_miss"]),
                "site_avg_L_recent3y_miss": int(v["site_avg_L_recent3y_miss"])}
            for k, v in hist.items()}


def filter_probs(probs_df: pd.DataFrame, keep_sy: set) -> pd.DataFrame:
    mask = probs_df.apply(lambda r: (r["site"], int(r["year"])) in keep_sy, axis=1)
    return probs_df[mask].copy()


def sweep_useful_simple(probs_df: pd.DataFrame, tau_grid: np.ndarray, ks: list,
                        doy_start: int) -> pd.DataFrame:
    rows = []
    for k in ks:
        for tau in tau_grid:
            spec = {"name": "first_crossing", "tau": float(tau), "k": int(k)}
            cls = apply_alert_rule(probs_df, spec, doy_start)
            m = alert_metrics(cls)
            ev = cls[cls.is_event == 1]
            buckets = Counter(ev["bucket"])
            useful = int(buckets.get("USEFUL", 0))
            too_early = int(buckets.get("TOO_EARLY", 0))
            no_alert = int(buckets.get("no_alert", 0))
            leads = ev.loc[ev.lead_days.notna(), "lead_days"].astype(int).values
            rows.append({
                "k": int(k), "tau": float(tau),
                "recall": m["recall"], "FAR": m["FAR"],
                "precision": m["precision"], "F1": m["F1"],
                "n_alert": m["n_alert"], "n_event": m["n_event"],
                "USEFUL": useful, "TOO_EARLY": too_early, "no_alert": no_alert,
                "USEFUL_e2e": (useful / max(m["n_event"], 1)),
                "lead_median": float(np.median(leads)) if len(leads) else None,
            })
    return pd.DataFrame(rows)


def select_recall_far_min(df: pd.DataFrame, target: float):
    cands = df[df["recall"] >= target]
    if cands.empty:
        return None
    # tie-break: lead_median shorter, then tau, then k
    s = cands.copy()
    s["_lead_med"] = s["lead_median"].fillna(999.0)
    s = s.sort_values(["FAR", "_lead_med", "tau", "k"],
                       ascending=[True, True, True, True])
    return s.iloc[0].to_dict()


def print_pick(label: str, pick: dict | None, target: float) -> None:
    if pick is None:
        print(f"  [{label}, R>={target:.2f}] no qualifying cell")
        return
    lm = pick["lead_median"]
    lm_str = f"{lm:.1f}" if lm is not None else "N/A"
    print(f"  [{label}, R>={target:.2f}] k={int(pick['k'])}  tau={pick['tau']:.3f}  "
          f"R={pick['recall']:.3f}  FAR={pick['FAR']:.3f}  "
          f"P={pick['precision']:.3f}  F1={pick['F1']:.3f}  "
          f"n_alert={int(pick['n_alert'])}  no_alert={int(pick['no_alert'])}  "
          f"USEFUL={int(pick['USEFUL'])} ({pick['USEFUL_e2e']*100:.1f}%)  "
          f"lead_med={lm_str}")


def section_alert_overlap(base_df: pd.DataFrame, d_df: pd.DataFrame,
                          tau: float, k: int, doy_start: int,
                          with_history: set) -> dict:
    spec = {"name": "first_crossing", "tau": float(tau), "k": int(k)}
    base_cls = apply_alert_rule(base_df, spec, doy_start)
    d_cls = apply_alert_rule(d_df, spec, doy_start)
    base_cls["_key"] = list(zip(base_cls["site"], base_cls["year"]))
    d_cls["_key"] = list(zip(d_cls["site"], d_cls["year"]))
    base_alerted = {k_ for k_, a in zip(base_cls["_key"], base_cls["alert_tstar"].notna()) if a}
    d_alerted = {k_ for k_, a in zip(d_cls["_key"], d_cls["alert_tstar"].notna()) if a}
    ev_keys = set(base_cls.loc[base_cls["is_event"] == 1, "_key"]) | \
              set(d_cls.loc[d_cls["is_event"] == 1, "_key"])

    both_a = base_alerted & d_alerted
    only_base = base_alerted - d_alerted
    only_d = d_alerted - base_alerted
    neither = ev_keys - base_alerted - d_alerted

    def _split(s):
        evs = [x for x in s if x in ev_keys]
        wh = [x for x in evs if x in with_history]
        nh = [x for x in evs if x not in with_history]
        return {"n": len(evs), "with_history": len(wh), "no_history": len(nh)}

    out = {
        "events_total": len(ev_keys),
        "events_with_history": len([k for k in ev_keys if k in with_history]),
        "events_no_history": len([k for k in ev_keys if k not in with_history]),
        "both_alerted": _split(both_a),
        "only_baseline (LOST by D)": _split(only_base),
        "only_D (GAINED by D)": _split(only_d),
        "both_missed (no_alert)": _split(neither),
    }
    return out


def hybrid_probs(base_df: pd.DataFrame, d_df: pd.DataFrame,
                 with_history: set) -> pd.DataFrame:
    """Merge per-tstar probs; use D score if (site, year) has history else baseline."""
    m = pd.merge(base_df, d_df, on=["site", "year", "tstar"], suffixes=("_b", "_d"))
    sy_has = m.apply(lambda r: (r["site"], int(r["year"])) in with_history, axis=1).values
    p = np.where(sy_has, m["p_cal_d"].values, m["p_cal_b"].values)
    out = pd.DataFrame({
        "split": m["split_b"].values,
        "site": m["site"].values,
        "year": m["year"].values,
        "tstar": m["tstar"].values,
        "p_cal": p,
        "y_event": m["y_event_b"].values,
        "true_L": m["true_L_b"].values,
        "true_R": m["true_R_b"].values,
    })
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--baseline_ckpt", required=True,
                    help="lead14-45_ignore (no history) ckpt")
    ap.add_argument("--d_ckpt", required=True,
                    help="lead14-45_history_rolling ckpt")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--tau_step", type=float, default=0.025)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--recall_targets", default="0.90,0.92,0.95")
    ap.add_argument("--overlap_tau", type=float, default=0.55,
                    help="(tau, k) for alert-overlap section")
    ap.add_argument("--overlap_k", type=int, default=3)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    # Common namespace for both build_probs calls
    class A:
        pass
    common = A()
    for f in ["pest", "run", "split_seed", "val_year",
              "test_year_min", "test_year_max"]:
        setattr(common, f, getattr(args, f))

    print("\n========== building baseline probs ==========")
    common.stage1_ckpt = args.baseline_ckpt
    base_cache = build_probs(common)
    doy_start = int(C.DOY_START)

    print("\n========== building D probs ==========")
    common.stage1_ckpt = args.d_ckpt
    d_cache = build_probs(common)

    # Sanity: ckpt 2 should have site_history_added=True
    d_ckpt = torch.load(args.d_ckpt, map_location="cpu", weights_only=False)
    if not bool(d_ckpt.get("site_history_added", False)):
        print("[warn] d_ckpt does not have site_history_added=True; results may be misleading")
    pol = str(d_ckpt.get("site_history_policy", "rolling"))
    tyrmax = int(d_ckpt.get("history_train_year_max", 2021))
    print(f"[history meta] policy={pol}  train_year_max={tyrmax}")

    miss_map = make_history_mask(args.pest, args.run, doy_start, pol, tyrmax)

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    tau_grid = np.arange(0.05, 0.95 + 1e-9, args.tau_step)
    targets = [float(x) for x in args.recall_targets.split(",") if x.strip()]

    summary = {"history_policy": pol, "train_year_max": tyrmax, "splits": {}}

    for split_name, base_df, d_df in [("val", base_cache["val_df"], d_cache["val_df"]),
                                       ("test", base_cache["test_df"], d_cache["test_df"])]:
        print(f"\n\n========== {split_name} ==========")
        # Build with-history set scoped to this split (only site-years present in this split)
        present = set(zip(base_df["site"], base_df["year"].astype(int)))
        with_history = {sy for sy in present
                        if miss_map.get(sy, {"prev_year_L_miss": 1})["prev_year_L_miss"] == 0}
        # Events restricted
        ev_keys = {(r["site"], int(r["year"]))
                   for _, r in base_df.drop_duplicates(["site", "year"]).iterrows()
                   if int(r["y_event"]) == 1}
        ev_with = ev_keys & with_history
        ev_no = ev_keys - with_history
        ne_keys = present - ev_keys
        ne_with = ne_keys & with_history
        ne_no = ne_keys - with_history

        cov = {
            "n_site_years_total": len(present),
            "n_events_total": len(ev_keys),
            "n_with_history": len(with_history),
            "n_no_history": len(present - with_history),
            "with_history_rate": len(with_history) / max(len(present), 1),
            "events_with_history": len(ev_with),
            "events_no_history": len(ev_no),
            "events_with_history_rate": len(ev_with) / max(len(ev_keys), 1),
        }
        print(f"\n  --- 1. coverage ---")
        for k_, v_ in cov.items():
            if isinstance(v_, float):
                print(f"    {k_:>30}: {v_:.3f}")
            else:
                print(f"    {k_:>30}: {v_}")

        # 2. alert overlap at (overlap_tau, overlap_k)
        print(f"\n  --- 2. alert overlap @ tau={args.overlap_tau} k={args.overlap_k} ---")
        ov = section_alert_overlap(base_df, d_df, args.overlap_tau, args.overlap_k,
                                    doy_start, with_history)
        for k_, v_ in ov.items():
            if isinstance(v_, dict):
                print(f"    {k_:>30}: n={v_['n']}  with_hist={v_['with_history']}  "
                      f"no_hist={v_['no_history']}")
            else:
                print(f"    {k_:>30}: {v_}")

        # 3. with-history sub-cohort Pareto
        if len(with_history) >= 5:
            print(f"\n  --- 3. with-history sub-cohort Pareto (same {len(with_history)} site-years) ---")
            base_wh = filter_probs(base_df, with_history)
            d_wh = filter_probs(d_df, with_history)
            base_wh_sw = sweep_useful_simple(base_wh, tau_grid, ks, doy_start)
            d_wh_sw = sweep_useful_simple(d_wh, tau_grid, ks, doy_start)
            base_wh_sw.to_csv(out_dir / f"sweep_wh_baseline_{split_name}.csv", index=False)
            d_wh_sw.to_csv(out_dir / f"sweep_wh_d_{split_name}.csv", index=False)
            sec3 = {"baseline": {}, "d": {}}
            for tgt in targets:
                pb = select_recall_far_min(base_wh_sw, tgt)
                pd_ = select_recall_far_min(d_wh_sw, tgt)
                print_pick(f"baseline / with-history", pb, tgt)
                print_pick(f"D        / with-history", pd_, tgt)
                sec3["baseline"][f"R>={tgt:.2f}"] = pb
                sec3["d"][f"R>={tgt:.2f}"] = pd_
        else:
            sec3 = None
            print("    (with-history cohort too small)")

        # 4. no-history sub-cohort Pareto
        no_history = present - with_history
        if len(no_history) >= 5:
            print(f"\n  --- 4. no-history sub-cohort Pareto (same {len(no_history)} site-years) ---")
            base_nh = filter_probs(base_df, no_history)
            d_nh = filter_probs(d_df, no_history)
            base_nh_sw = sweep_useful_simple(base_nh, tau_grid, ks, doy_start)
            d_nh_sw = sweep_useful_simple(d_nh, tau_grid, ks, doy_start)
            base_nh_sw.to_csv(out_dir / f"sweep_nh_baseline_{split_name}.csv", index=False)
            d_nh_sw.to_csv(out_dir / f"sweep_nh_d_{split_name}.csv", index=False)
            sec4 = {"baseline": {}, "d": {}}
            for tgt in targets:
                pb = select_recall_far_min(base_nh_sw, tgt)
                pd_ = select_recall_far_min(d_nh_sw, tgt)
                print_pick(f"baseline / no-history", pb, tgt)
                print_pick(f"D        / no-history", pd_, tgt)
                sec4["baseline"][f"R>={tgt:.2f}"] = pb
                sec4["d"][f"R>={tgt:.2f}"] = pd_
        else:
            sec4 = None
            print("    (no-history cohort too small)")

        # 5. hybrid gate sweep (full cohort)
        print(f"\n  --- 5. hybrid gate Pareto (use D if has-history else baseline) ---")
        hyb_df = hybrid_probs(base_df, d_df, with_history)
        hyb_sw = sweep_useful_simple(hyb_df, tau_grid, ks, doy_start)
        base_sw = sweep_useful_simple(base_df, tau_grid, ks, doy_start)
        d_sw = sweep_useful_simple(d_df, tau_grid, ks, doy_start)
        hyb_sw.to_csv(out_dir / f"sweep_hybrid_{split_name}.csv", index=False)
        base_sw.to_csv(out_dir / f"sweep_full_baseline_{split_name}.csv", index=False)
        d_sw.to_csv(out_dir / f"sweep_full_d_{split_name}.csv", index=False)
        sec5 = {"baseline_full": {}, "d_full": {}, "hybrid": {}}
        for tgt in targets:
            pb = select_recall_far_min(base_sw, tgt)
            pd_ = select_recall_far_min(d_sw, tgt)
            ph = select_recall_far_min(hyb_sw, tgt)
            print_pick(f"baseline / full cohort", pb, tgt)
            print_pick(f"D        / full cohort", pd_, tgt)
            print_pick(f"HYBRID   / full cohort", ph, tgt)
            sec5["baseline_full"][f"R>={tgt:.2f}"] = pb
            sec5["d_full"][f"R>={tgt:.2f}"] = pd_
            sec5["hybrid"][f"R>={tgt:.2f}"] = ph

        summary["splits"][split_name] = {
            "coverage": cov,
            "alert_overlap": ov,
            "section3_with_history": sec3,
            "section4_no_history": sec4,
            "section5_hybrid_vs_full": sec5,
        }

    (out_dir / "subcohort_compare_summary.json").write_text(
        json.dumps(summary, indent=2, default=str))
    print(f"\n[saved] sweeps CSVs + {out_dir / 'subcohort_compare_summary.json'}")


if __name__ == "__main__":
    main()
