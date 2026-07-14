"""
Phase T10 — Group-tau hybrid gate.

For two ckpts (A=baseline no_history, D=history_rolling) and a history-availability
mask, evaluate four alert strategies on val + test:

  1. dispatch_group_tau (primary):
       with_history site-year -> D first_crossing(tau_with, k)
       no_history  site-year -> A first_crossing(tau_no,  k)
       group-specific tau, single k

  2. global_tau_hybrid:
       same dispatch, single tau (= tau_with = tau_no)  (baseline of method 1)

  3. A_raw_global:
       A first_crossing(tau, k) for all site-years

  4. D_raw_global:
       D first_crossing(tau, k) for all site-years

  5. OR_hybrid (reference only):
       site-year alerted if A_alert OR D_alert (both with same tau)
       expected high recall but high FAR

Selection (val-only):
  per target_recall in {0.85, 0.88, 0.90, 0.92}:
    pick (k, tau_no, tau_with) min FAR_val subject to recall_val >= target

Output per selected operating point (val + test):
  - recall, FAR, precision, F1
  - n_alert, no_alert, TOO_LATE, MISSED, USEFUL (info)
  - alert lead mean/median
  - group-decomposed recall/FAR/no_alert (with_history vs no_history)
  - corr(true_L_DOY, alert_DOY), corr(true_L_DOY, score_peak_DOY) per cohort

No retraining; two ckpts + inference only.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import spearmanr

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.scripts.phase_t_lead_aware_eval import build_probs
from rice.scripts.phase_t_history_subcohort_compare import make_history_mask


EVENT_BUCKETS = ["MISSED", "TOO_LATE", "USEFUL", "TOO_EARLY", "no_alert"]


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


def build_per_sy(probs_df: pd.DataFrame) -> dict:
    out = {}
    for (s, y), g in probs_df.groupby(["site", "year"], sort=False):
        gs = g.sort_values("tstar")
        sy = (str(s), int(y))
        out[sy] = {
            "ts": gs["tstar"].values.astype(int),
            "ps": gs["p_cal"].values.astype(float),
            "y_event": int(gs["y_event"].iloc[0]),
            "true_L": gs["true_L"].iloc[0],
            "true_R": gs["true_R"].iloc[0],
        }
    return out


def build_classifications(per_sy_A: dict, per_sy_D: dict, miss_map: dict,
                           method: str, tau_no: float, tau_with: float, k: int,
                           doy_start: int) -> pd.DataFrame:
    """Returns per-(site, year) df: site, year, is_event, alerted, alert_tstar,
    lead_days, bucket, with_history, true_L."""
    keys = set(per_sy_A) | set(per_sy_D)
    rows = []
    for sy in keys:
        site, year = sy
        with_h = miss_map.get(sy, {"prev_year_L_miss": 1})["prev_year_L_miss"] == 0
        A = per_sy_A.get(sy)
        D = per_sy_D.get(sy)
        if A is None or D is None:
            continue
        is_event = int(A["y_event"])
        true_L = A["true_L"]
        true_R = A["true_R"]
        # method-specific alert decision
        if method == "dispatch_group_tau":
            if with_h:
                at = first_crossing_k(D["ts"], D["ps"], tau_with, k)
            else:
                at = first_crossing_k(A["ts"], A["ps"], tau_no, k)
        elif method == "global_tau_hybrid":
            # group dispatch by score only, single tau
            src = D if with_h else A
            at = first_crossing_k(src["ts"], src["ps"], tau_no, k)
        elif method == "A_raw_global":
            at = first_crossing_k(A["ts"], A["ps"], tau_no, k)
        elif method == "D_raw_global":
            at = first_crossing_k(D["ts"], D["ps"], tau_with, k)
        elif method == "OR_hybrid":
            at_a = first_crossing_k(A["ts"], A["ps"], tau_no, k)
            at_d = first_crossing_k(D["ts"], D["ps"], tau_with, k)
            if at_a is None and at_d is None: at = None
            elif at_a is None: at = at_d
            elif at_d is None: at = at_a
            else: at = min(at_a, at_d)
        else:
            raise ValueError(method)
        if is_event == 1 and pd.notna(true_L) and at is not None:
            lead = int(true_L) - int(at)
        else:
            lead = None
        if is_event == 1:
            bucket = event_bucket(lead)
        else:
            bucket = "FP" if at is not None else "TN"
        rows.append({
            "site": site, "year": year, "is_event": is_event,
            "alerted": int(at is not None),
            "alert_tstar": (int(at) if at is not None else None),
            "lead_days": (int(lead) if lead is not None else None),
            "bucket": bucket,
            "with_history": int(with_h),
            "true_L": (int(true_L) if pd.notna(true_L) else None),
            "L_DOY": (int(true_L) + int(doy_start) if pd.notna(true_L) else None),
            "alert_DOY": (int(at) + int(doy_start) if at is not None else None),
        })
    return pd.DataFrame(rows)


def metrics_from_cls(df: pd.DataFrame) -> dict:
    n_event = int((df.is_event == 1).sum())
    n_nonevent = int((df.is_event == 0).sum())
    tp = int(((df.is_event == 1) & df.alert_tstar.notna()).sum())
    fp = int(((df.is_event == 0) & df.alert_tstar.notna()).sum())
    fn = n_event - tp
    tn = n_nonevent - fp
    rec = tp / max(n_event, 1)
    far = fp / max(n_nonevent, 1)
    prec = tp / max(tp + fp, 1) if (tp + fp) else float("nan")
    f1 = 2 * prec * rec / max(prec + rec, 1e-9) if (prec + rec) > 0 else float("nan")
    bk = Counter(df[df.is_event == 1]["bucket"])
    leads = df.loc[(df.is_event == 1) & df.lead_days.notna(), "lead_days"].astype(int).values
    return {
        "TP": tp, "FP": fp, "FN": fn, "TN": tn,
        "recall": rec, "FAR": far, "precision": prec, "F1": f1,
        "n_alert": tp + fp, "n_event": n_event, "n_nonevent": n_nonevent,
        **{b: int(bk.get(b, 0)) for b in EVENT_BUCKETS},
        "lead_mean": float(leads.mean()) if len(leads) else float("nan"),
        "lead_median": float(np.median(leads)) if len(leads) else float("nan"),
    }


def group_decomposed(df: pd.DataFrame) -> dict:
    out = {}
    for g_label, g_filter in [("with_history", df.with_history == 1),
                                ("no_history", df.with_history == 0)]:
        sub = df[g_filter]
        ev = int((sub.is_event == 1).sum())
        ne = int((sub.is_event == 0).sum())
        tp = int(((sub.is_event == 1) & sub.alert_tstar.notna()).sum())
        fp = int(((sub.is_event == 0) & sub.alert_tstar.notna()).sum())
        bk = Counter(sub[sub.is_event == 1]["bucket"])
        out[g_label] = {
            "n_event": ev, "n_nonevent": ne,
            "TP": tp, "FP": fp,
            "recall": tp / max(ev, 1),
            "FAR": fp / max(ne, 1),
            "no_alert": int(bk.get("no_alert", 0)),
            "USEFUL": int(bk.get("USEFUL", 0)),
        }
    return out


def corr_blocks(df: pd.DataFrame) -> dict:
    out = {}
    for label, sub in [("all", df[df.is_event == 1]),
                        ("with_history", df[(df.is_event == 1) & (df.with_history == 1)]),
                        ("no_history", df[(df.is_event == 1) & (df.with_history == 0)])]:
        ev_alerted = sub[sub.alert_DOY.notna() & sub.L_DOY.notna()]
        if len(ev_alerted) < 5:
            out[label] = {"n": int(len(ev_alerted))}
            continue
        L = ev_alerted["L_DOY"].astype(float).values
        a = ev_alerted["alert_DOY"].astype(float).values
        if np.std(L) > 0 and np.std(a) > 0:
            pe = float(np.corrcoef(L, a)[0, 1])
            sp = float(spearmanr(L, a).correlation)
        else:
            pe = float("nan"); sp = float("nan")
        out[label] = {"n": int(len(ev_alerted)), "pearson": pe, "spearman": sp}
    return out


def sweep_dispatch(per_sy_A_val, per_sy_D_val, miss_map, tau_grid, ks, doy_start):
    """Returns list of rows: (k, tau_no, tau_with) -> val metrics."""
    rows = []
    for k in ks:
        for tau_no in tau_grid:
            for tau_with in tau_grid:
                df = build_classifications(per_sy_A_val, per_sy_D_val, miss_map,
                                            "dispatch_group_tau",
                                            float(tau_no), float(tau_with), int(k),
                                            doy_start)
                m = metrics_from_cls(df)
                rows.append({"k": int(k), "tau_no": float(tau_no),
                             "tau_with": float(tau_with),
                             "recall": m["recall"], "FAR": m["FAR"],
                             "precision": m["precision"], "F1": m["F1"],
                             "no_alert": m["no_alert"], "TOO_LATE": m["TOO_LATE"],
                             "MISSED": m["MISSED"], "USEFUL": m["USEFUL"],
                             "lead_median": m["lead_median"]})
    return pd.DataFrame(rows)


def select_dispatch(df: pd.DataFrame, target: float) -> dict | None:
    cands = df[df.recall >= target]
    if cands.empty:
        return None
    cands = cands.copy()
    cands["_lead"] = cands["lead_median"].fillna(999.0)
    cands = cands.sort_values(["FAR", "_lead", "tau_with", "tau_no", "k"],
                               ascending=[True, True, True, True, True])
    return cands.iloc[0].to_dict()


def sweep_single_tau(per_sy_A, per_sy_D, miss_map, method, tau_grid, ks, doy_start):
    rows = []
    for k in ks:
        for tau in tau_grid:
            df = build_classifications(per_sy_A, per_sy_D, miss_map, method,
                                        float(tau), float(tau), int(k), doy_start)
            m = metrics_from_cls(df)
            rows.append({"k": int(k), "tau": float(tau),
                         "recall": m["recall"], "FAR": m["FAR"],
                         "precision": m["precision"], "F1": m["F1"],
                         "no_alert": m["no_alert"], "TOO_LATE": m["TOO_LATE"],
                         "MISSED": m["MISSED"], "USEFUL": m["USEFUL"],
                         "lead_median": m["lead_median"]})
    return pd.DataFrame(rows)


def select_single(df, target):
    cands = df[df.recall >= target]
    if cands.empty: return None
    cands = cands.copy()
    cands["_lead"] = cands["lead_median"].fillna(999.0)
    cands = cands.sort_values(["FAR", "_lead", "tau", "k"], ascending=[True, True, True, True])
    return cands.iloc[0].to_dict()


def print_block(name: str, val_m: dict, test_m: dict, val_groups: dict,
                 test_groups: dict, val_corrs: dict, test_corrs: dict) -> None:
    print(f"\n  [{name}]")
    for split, m in [("val ", val_m), ("test", test_m)]:
        print(f"    {split}: R={m['recall']:.3f}  FAR={m['FAR']:.3f}  "
              f"P={m['precision']:.3f}  F1={m['F1']:.3f}  n_alert={m['n_alert']}  "
              f"no_alert={m['no_alert']}  TOO_LATE={m['TOO_LATE']}  MISSED={m['MISSED']}  "
              f"USEFUL={m['USEFUL']}  lead_med={m['lead_median']:.1f}")
    print(f"    val  groups: with_history R={val_groups['with_history']['recall']:.3f} "
          f"FAR={val_groups['with_history']['FAR']:.3f}  no_history R={val_groups['no_history']['recall']:.3f} "
          f"FAR={val_groups['no_history']['FAR']:.3f}")
    print(f"    test groups: with_history R={test_groups['with_history']['recall']:.3f} "
          f"FAR={test_groups['with_history']['FAR']:.3f}  no_history R={test_groups['no_history']['recall']:.3f} "
          f"FAR={test_groups['no_history']['FAR']:.3f}")
    for lab in ["all", "with_history", "no_history"]:
        vc = val_corrs.get(lab, {}); tc = test_corrs.get(lab, {})
        vp = vc.get("pearson"); tp = tc.get("pearson")
        vp_s = f"{vp:.3f}" if vp is not None and not (isinstance(vp, float) and np.isnan(vp)) else "N/A"
        tp_s = f"{tp:.3f}" if tp is not None and not (isinstance(tp, float) and np.isnan(tp)) else "N/A"
        print(f"    corr(L_DOY, alert_DOY) [{lab}]:  val_pe={vp_s} (n={vc.get('n',0)})  "
              f"test_pe={tp_s} (n={tc.get('n',0)})")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--baseline_ckpt", required=True, help="A: lead14-45 no_history")
    ap.add_argument("--d_ckpt", required=True, help="D: lead14-45 history_rolling")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--tau_step", type=float, default=0.025)
    ap.add_argument("--ks", default="3", help="for dispatch, single k value; use '1,2,3' for sweep")
    ap.add_argument("--recall_targets", default="0.85,0.88,0.90,0.92")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    class N: pass
    common = N()
    for f in ["pest", "run", "split_seed", "val_year", "test_year_min", "test_year_max"]:
        setattr(common, f, getattr(args, f))

    print("\n========== building A (baseline) probs ==========")
    common.stage1_ckpt = args.baseline_ckpt
    base_cache = build_probs(common)
    doy_start = int(C.DOY_START)

    print("\n========== building D (history) probs ==========")
    common.stage1_ckpt = args.d_ckpt
    d_cache = build_probs(common)

    d_ckpt = torch.load(args.d_ckpt, map_location="cpu", weights_only=False)
    pol = str(d_ckpt.get("site_history_policy", "rolling"))
    tyrmax = int(d_ckpt.get("history_train_year_max", 2021))
    print(f"\n[history meta] policy={pol}  train_year_max={tyrmax}")
    miss_map = make_history_mask(args.pest, args.run, doy_start, pol, tyrmax)

    per_sy_A_val = build_per_sy(base_cache["val_df"])
    per_sy_A_test = build_per_sy(base_cache["test_df"])
    per_sy_D_val = build_per_sy(d_cache["val_df"])
    per_sy_D_test = build_per_sy(d_cache["test_df"])

    tau_grid = np.round(np.arange(0.05, 0.95 + 1e-9, args.tau_step), 4)
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    targets = [float(x) for x in args.recall_targets.split(",") if x.strip()]

    print(f"\n[grid] taus={len(tau_grid)}  ks={ks}  targets={targets}")
    print("\n========== sweeping dispatch_group_tau (val) ==========")
    disp_val = sweep_dispatch(per_sy_A_val, per_sy_D_val, miss_map, tau_grid, ks, doy_start)
    disp_val.to_csv(out_dir / "dispatch_sweep_val.csv", index=False)
    print(f"  dispatch cells = {len(disp_val)}")

    # Single-tau sweeps for the other methods
    print("\n========== sweeping single-tau methods (val) ==========")
    methods = ["global_tau_hybrid", "A_raw_global", "D_raw_global", "OR_hybrid"]
    single_val = {m: sweep_single_tau(per_sy_A_val, per_sy_D_val, miss_map, m,
                                       tau_grid, ks, doy_start) for m in methods}
    for m, df in single_val.items():
        df.to_csv(out_dir / f"single_sweep_{m}_val.csv", index=False)

    out = {"selections": {}}
    for target in targets:
        print(f"\n\n========== target recall >= {target:.2f} ==========")
        # dispatch
        disp_pick = select_dispatch(disp_val, target)
        if disp_pick is None:
            print(f"  dispatch_group_tau: no qualifying cell on val")
        else:
            k_ = int(disp_pick["k"]); tn = float(disp_pick["tau_no"]); tw = float(disp_pick["tau_with"])
            print(f"  dispatch_group_tau best:  k={k_}  tau_no={tn:.3f}  tau_with={tw:.3f}")
            v_df = build_classifications(per_sy_A_val, per_sy_D_val, miss_map,
                                          "dispatch_group_tau", tn, tw, k_, doy_start)
            t_df = build_classifications(per_sy_A_test, per_sy_D_test, miss_map,
                                          "dispatch_group_tau", tn, tw, k_, doy_start)
            v_m = metrics_from_cls(v_df); t_m = metrics_from_cls(t_df)
            v_g = group_decomposed(v_df); t_g = group_decomposed(t_df)
            v_c = corr_blocks(v_df); t_c = corr_blocks(t_df)
            print_block(f"dispatch (k={k_}, tau_no={tn:.3f}, tau_with={tw:.3f})",
                        v_m, t_m, v_g, t_g, v_c, t_c)
            out["selections"].setdefault(f"R>={target:.2f}", {})["dispatch_group_tau"] = {
                "k": k_, "tau_no": tn, "tau_with": tw,
                "val": v_m, "test": t_m,
                "val_groups": v_g, "test_groups": t_g,
                "val_corrs": v_c, "test_corrs": t_c,
            }
            v_df.to_csv(out_dir / f"cls_val_dispatch_R{int(target*100)}.csv", index=False)
            t_df.to_csv(out_dir / f"cls_test_dispatch_R{int(target*100)}.csv", index=False)
        # other methods
        for m in methods:
            pick = select_single(single_val[m], target)
            if pick is None:
                print(f"  {m}: no qualifying cell on val")
                continue
            k_ = int(pick["k"]); tau = float(pick["tau"])
            print(f"  {m} best:  k={k_}  tau={tau:.3f}")
            v_df = build_classifications(per_sy_A_val, per_sy_D_val, miss_map,
                                          m, tau, tau, k_, doy_start)
            t_df = build_classifications(per_sy_A_test, per_sy_D_test, miss_map,
                                          m, tau, tau, k_, doy_start)
            v_m = metrics_from_cls(v_df); t_m = metrics_from_cls(t_df)
            v_g = group_decomposed(v_df); t_g = group_decomposed(t_df)
            v_c = corr_blocks(v_df); t_c = corr_blocks(t_df)
            print_block(f"{m} (k={k_}, tau={tau:.3f})",
                        v_m, t_m, v_g, t_g, v_c, t_c)
            out["selections"].setdefault(f"R>={target:.2f}", {})[m] = {
                "k": k_, "tau": tau, "val": v_m, "test": t_m,
                "val_groups": v_g, "test_groups": t_g,
                "val_corrs": v_c, "test_corrs": t_c,
            }

    (out_dir / "group_tau_hybrid_summary.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {out_dir / 'group_tau_hybrid_summary.json'}")


if __name__ == "__main__":
    main()
