"""
Phase T10d — Extended group-tau experiments.

Three new methods on top of dispatch_group_tau (R=0.854, FAR=0.678):
  phase         : 3 site groups by site-avg L_DOY tercile (early/mid/late)
                   site_avg over train events (year <= --train_year_max)
                   tau grid: tau_early × tau_mid × tau_late
  risk          : 2 site groups by event count median (high/low risk)
                   tau grid: tau_high × tau_low
  global_delta  : with-history vs no-history binary group;
                   tau_with = tau_global + delta_with
                   tau_no   = tau_global + delta_no
                   delta grid: -0.10/-0.05/0/+0.05/+0.10

Single Stage 1 ckpt input. Val sweep -> test report. k fixed (default 3).
Same target recalls (0.85/0.88/0.90). Tie-break FAR ascending, then lead_median.
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
from rice.scripts.run_eval import build_samples_for_run
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
    if lead is None or pd.isna(lead): return "no_alert"
    d = int(lead)
    if d < 0: return "MISSED"
    if d < 14: return "TOO_LATE"
    if d <= 60: return "USEFUL"
    return "TOO_EARLY"


def metrics(df):
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


def classify(per_sy, sy_group_map, tau_map, k, default_group):
    rows = []
    for sy, d in per_sy.items():
        grp = sy_group_map.get(sy, default_group)
        tau = tau_map.get(grp, tau_map[default_group])
        at = first_crossing_k(d["ts"], d["ps"], tau, k)
        is_event = int(d["y_event"])
        true_L = d["true_L"]
        lead = (int(true_L) - int(at)) if (is_event == 1 and pd.notna(true_L) and at is not None) else None
        bucket = event_bucket(lead) if is_event == 1 else ("FP" if at is not None else "TN")
        rows.append({"site": sy[0], "year": sy[1], "is_event": is_event,
                     "group": grp,
                     "alert_tstar": (int(at) if at is not None else None),
                     "lead_days": (int(lead) if lead is not None else None),
                     "bucket": bucket,
                     "true_L": (int(true_L) if pd.notna(true_L) else None)})
    return pd.DataFrame(rows)


def build_train_events(pest, run, train_year_max):
    _, gfc = resolve_pest(pest)
    _, _, _, samples = build_samples_for_run(run, gfc)
    doy_start = int(C.DOY_START)
    rows = []
    for s in samples:
        year = int(s["year"])
        if year > train_year_max: continue
        if str(s.get("censor_type", "right")) == "right": continue
        if s.get("L") is None or pd.isna(s["L"]): continue
        rows.append({"site": str(s["site_id"]), "year": year,
                     "L_doy": int(s["L"]) + doy_start})
    return pd.DataFrame(rows)


def build_site_maps(train_ev_df, all_sites):
    stats = train_ev_df.groupby("site").agg(
        avg_L_DOY=("L_doy", "mean"),
        n_events=("year", "count"),
    )
    if len(stats) == 0:
        return ({s: "mid" for s in all_sites},
                {s: "low" for s in all_sites},
                {"phase_q33": None, "phase_q67": None, "risk_median": None,
                 "n_sites_with_stats": 0})
    q33 = float(stats["avg_L_DOY"].quantile(0.33))
    q67 = float(stats["avg_L_DOY"].quantile(0.67))
    rmed = float(stats["n_events"].median())
    def _phase(s):
        if s not in stats.index: return "mid"
        v = stats.loc[s, "avg_L_DOY"]
        if v <= q33: return "early"
        if v <= q67: return "mid"
        return "late"
    def _risk(s):
        if s not in stats.index: return "low"
        return "high" if stats.loc[s, "n_events"] >= rmed else "low"
    return ({s: _phase(s) for s in all_sites},
            {s: _risk(s) for s in all_sites},
            {"phase_q33": q33, "phase_q67": q67, "risk_median": rmed,
             "n_sites_with_stats": int(len(stats))})


def sweep_phase(per_sy, phase_sy, tau_grid, k):
    rows = []
    for te in tau_grid:
        for tm in tau_grid:
            for tl in tau_grid:
                df = classify(per_sy, phase_sy,
                              {"early": float(te), "mid": float(tm), "late": float(tl)},
                              k, "mid")
                m = metrics(df)
                rows.append({"k": k, "tau_early": float(te), "tau_mid": float(tm),
                             "tau_late": float(tl), **m})
    return pd.DataFrame(rows)


def sweep_risk(per_sy, risk_sy, tau_grid, k):
    rows = []
    for th in tau_grid:
        for tl in tau_grid:
            df = classify(per_sy, risk_sy,
                          {"high": float(th), "low": float(tl)}, k, "low")
            m = metrics(df)
            rows.append({"k": k, "tau_high": float(th), "tau_low": float(tl), **m})
    return pd.DataFrame(rows)


def sweep_global_delta(per_sy, history_sy, tau_grid, delta_grid, k):
    rows = []
    for tg in tau_grid:
        for dw in delta_grid:
            for dn in delta_grid:
                tau_with = float(tg + dw)
                tau_no = float(tg + dn)
                df = classify(per_sy, history_sy,
                              {"with": tau_with, "no": tau_no}, k, "no")
                m = metrics(df)
                rows.append({"k": k, "tau_global": float(tg),
                             "delta_with": float(dw), "delta_no": float(dn), **m})
    return pd.DataFrame(rows)


def select(df, target, sort_extra_cols):
    cands = df[df["recall"] >= target]
    if cands.empty: return None
    c = cands.copy()
    c["_lead"] = c["lead_median"].fillna(999.0)
    return c.sort_values(["FAR", "_lead"] + sort_extra_cols).iloc[0].to_dict()


def lookup_test(test_sw, key_cols, vals):
    cond = np.ones(len(test_sw), dtype=bool)
    for c, v in zip(key_cols, vals):
        cond &= np.isclose(test_sw[c].astype(float).values, float(v), atol=0.005)
    sub = test_sw[cond]
    if len(sub) == 0: return None
    return sub.iloc[0].to_dict()


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
    ap.add_argument("--train_year_max", type=int, default=2021)
    ap.add_argument("--tau_min", type=float, default=0.40)
    ap.add_argument("--tau_max", type=float, default=0.75)
    ap.add_argument("--tau_step", type=float, default=0.025)
    ap.add_argument("--ks", default="3")
    ap.add_argument("--delta_grid", default="-0.10,-0.05,0,0.05,0.10")
    ap.add_argument("--recall_targets", default="0.85,0.88,0.90")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.stage1_ckpt).stem
    print(f"\n========== Extended group-tau :: {label} ==========")

    cache = build_probs(args)
    doy_start = int(C.DOY_START)
    val_df = cache["val_df"]; test_df = cache["test_df"]
    per_sy_val = build_per_sy(val_df)
    per_sy_test = build_per_sy(test_df)
    all_sites = set(val_df["site"].astype(str)) | set(test_df["site"].astype(str))

    train_ev = build_train_events(args.pest, args.run, args.train_year_max)
    print(f"[train events] n={len(train_ev)}  sites={train_ev['site'].nunique()}")
    phase_map, risk_map, cutoffs = build_site_maps(train_ev, all_sites)
    print(f"[cutoffs] phase_q33={cutoffs['phase_q33']:.1f}  phase_q67={cutoffs['phase_q67']:.1f}  "
          f"risk_median={cutoffs['risk_median']}  n_sites_with_stats={cutoffs['n_sites_with_stats']}")

    miss_map = make_history_mask(args.pest, args.run, doy_start,
                                  "rolling", args.train_year_max)
    sy_keys = set(per_sy_val) | set(per_sy_test)
    phase_sy = {sy: phase_map[sy[0]] for sy in sy_keys}
    risk_sy = {sy: risk_map[sy[0]] for sy in sy_keys}
    history_sy = {sy: ("with" if miss_map.get(sy, {"prev_year_L_miss": 1})["prev_year_L_miss"] == 0
                       else "no")
                  for sy in sy_keys}

    k = int(args.ks.split(",")[0])
    tau_grid = np.round(np.arange(args.tau_min, args.tau_max + 1e-9, args.tau_step), 4)
    delta_grid = [float(x) for x in args.delta_grid.split(",") if x.strip()]
    targets = [float(x) for x in args.recall_targets.split(",") if x.strip()]
    print(f"[grid] tau {len(tau_grid)} pts  delta {len(delta_grid)} pts  k={k}")
    print(f"  expect ~{len(tau_grid)**3} (phase) + {len(tau_grid)**2} (risk) + "
          f"{len(tau_grid)*len(delta_grid)**2} (global+delta) cells per split")

    out = {"label": label, "cutoffs": cutoffs, "k": k, "selections": {}}

    # ---- phase ----
    print(f"\n========== sweep group_tau_phase ==========")
    v_sw = sweep_phase(per_sy_val, phase_sy, tau_grid, k)
    t_sw = sweep_phase(per_sy_test, phase_sy, tau_grid, k)
    v_sw.to_csv(out_dir / "sweep_val_phase.csv", index=False)
    t_sw.to_csv(out_dir / "sweep_test_phase.csv", index=False)
    for tgt in targets:
        pick = select(v_sw, tgt, ["tau_early", "tau_mid", "tau_late"])
        if pick is None:
            print(f"  R>={tgt:.2f}  phase: (no qualifying val cell)")
            out["selections"].setdefault(f"R>={tgt:.2f}", {})["phase"] = None
            continue
        t_row = lookup_test(t_sw, ["k", "tau_early", "tau_mid", "tau_late"],
                            [k, pick["tau_early"], pick["tau_mid"], pick["tau_late"]])
        print(f"  R>={tgt:.2f}  phase   k={k}  e={pick['tau_early']:.3f} m={pick['tau_mid']:.3f} l={pick['tau_late']:.3f}")
        print(f"    val:  R={pick['recall']:.3f} FAR={pick['FAR']:.3f} noA={int(pick['no_alert'])} USE={int(pick['USEFUL'])}")
        if t_row:
            print(f"    test: R={t_row['recall']:.3f} FAR={t_row['FAR']:.3f} "
                  f"P={t_row['precision']:.3f} F1={t_row['F1']:.3f} "
                  f"noA={int(t_row['no_alert'])} TL={int(t_row['TOO_LATE'])} "
                  f"MS={int(t_row['MISSED'])} USE={int(t_row['USEFUL'])} "
                  f"lead_med={t_row['lead_median']}")
        out["selections"].setdefault(f"R>={tgt:.2f}", {})["phase"] = {
            "k": k, "tau_early": float(pick["tau_early"]),
            "tau_mid": float(pick["tau_mid"]), "tau_late": float(pick["tau_late"]),
            "val": pick, "test": t_row}

    # ---- risk ----
    print(f"\n========== sweep group_tau_risk ==========")
    v_sw = sweep_risk(per_sy_val, risk_sy, tau_grid, k)
    t_sw = sweep_risk(per_sy_test, risk_sy, tau_grid, k)
    v_sw.to_csv(out_dir / "sweep_val_risk.csv", index=False)
    t_sw.to_csv(out_dir / "sweep_test_risk.csv", index=False)
    for tgt in targets:
        pick = select(v_sw, tgt, ["tau_high", "tau_low"])
        if pick is None:
            print(f"  R>={tgt:.2f}  risk: (no qualifying val cell)")
            out["selections"].setdefault(f"R>={tgt:.2f}", {})["risk"] = None
            continue
        t_row = lookup_test(t_sw, ["k", "tau_high", "tau_low"],
                            [k, pick["tau_high"], pick["tau_low"]])
        print(f"  R>={tgt:.2f}  risk    k={k}  h={pick['tau_high']:.3f} l={pick['tau_low']:.3f}")
        print(f"    val:  R={pick['recall']:.3f} FAR={pick['FAR']:.3f} noA={int(pick['no_alert'])} USE={int(pick['USEFUL'])}")
        if t_row:
            print(f"    test: R={t_row['recall']:.3f} FAR={t_row['FAR']:.3f} "
                  f"P={t_row['precision']:.3f} F1={t_row['F1']:.3f} "
                  f"noA={int(t_row['no_alert'])} TL={int(t_row['TOO_LATE'])} "
                  f"MS={int(t_row['MISSED'])} USE={int(t_row['USEFUL'])} "
                  f"lead_med={t_row['lead_median']}")
        out["selections"].setdefault(f"R>={tgt:.2f}", {})["risk"] = {
            "k": k, "tau_high": float(pick["tau_high"]), "tau_low": float(pick["tau_low"]),
            "val": pick, "test": t_row}

    # ---- global+delta ----
    print(f"\n========== sweep global_plus_offset ==========")
    v_sw = sweep_global_delta(per_sy_val, history_sy, tau_grid, delta_grid, k)
    t_sw = sweep_global_delta(per_sy_test, history_sy, tau_grid, delta_grid, k)
    v_sw.to_csv(out_dir / "sweep_val_global_delta.csv", index=False)
    t_sw.to_csv(out_dir / "sweep_test_global_delta.csv", index=False)
    for tgt in targets:
        pick = select(v_sw, tgt, ["tau_global", "delta_with", "delta_no"])
        if pick is None:
            print(f"  R>={tgt:.2f}  global+delta: (no qualifying val cell)")
            out["selections"].setdefault(f"R>={tgt:.2f}", {})["global_delta"] = None
            continue
        t_row = lookup_test(t_sw, ["k", "tau_global", "delta_with", "delta_no"],
                            [k, pick["tau_global"], pick["delta_with"], pick["delta_no"]])
        print(f"  R>={tgt:.2f}  glb+dlt k={k}  tg={pick['tau_global']:.3f} "
              f"d_w={pick['delta_with']:+.3f} d_n={pick['delta_no']:+.3f}")
        print(f"    val:  R={pick['recall']:.3f} FAR={pick['FAR']:.3f} noA={int(pick['no_alert'])} USE={int(pick['USEFUL'])}")
        if t_row:
            print(f"    test: R={t_row['recall']:.3f} FAR={t_row['FAR']:.3f} "
                  f"P={t_row['precision']:.3f} F1={t_row['F1']:.3f} "
                  f"noA={int(t_row['no_alert'])} TL={int(t_row['TOO_LATE'])} "
                  f"MS={int(t_row['MISSED'])} USE={int(t_row['USEFUL'])} "
                  f"lead_med={t_row['lead_median']}")
        out["selections"].setdefault(f"R>={tgt:.2f}", {})["global_delta"] = {
            "k": k, "tau_global": float(pick["tau_global"]),
            "delta_with": float(pick["delta_with"]),
            "delta_no": float(pick["delta_no"]),
            "val": pick, "test": t_row}

    # Side-by-side
    print(f"\n========== Side-by-side (test, dispatch_group_tau ref: R=0.854 FAR=0.678) ==========")
    print(f"  {'target':>8}  {'method':>14}  {'R':>5} {'FAR':>5} {'P':>5} {'F1':>5}  "
          f"{'noA':>4} {'TL':>3} {'MS':>3} {'USE':>4}")
    for tgt in targets:
        for m in ["phase", "risk", "global_delta"]:
            sel = out["selections"].get(f"R>={tgt:.2f}", {}).get(m)
            if sel is None or sel.get("test") is None:
                print(f"  R>={tgt:.2f}  {m:>14}  (no cell)")
                continue
            t = sel["test"]
            print(f"  R>={tgt:.2f}  {m:>14}  {t['recall']:>5.3f} {t['FAR']:>5.3f} "
                  f"{t['precision']:>5.3f} {t['F1']:>5.3f}  "
                  f"{int(t['no_alert']):>4d} {int(t['TOO_LATE']):>3d} "
                  f"{int(t['MISSED']):>3d} {int(t['USEFUL']):>4d}")

    (out_dir / f"extended_summary_{label}.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {out_dir / f'extended_summary_{label}.json'}")


if __name__ == "__main__":
    main()
