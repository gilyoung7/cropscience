"""
Phase T12 — Early/late outlier event score-shape diagnostic.

Group event site-years by true_L_DOY quantile:
  early_outlier : L_DOY <= q10
  normal_event  : q25 <= L_DOY <= q75
  late_outlier  : L_DOY >= q90

Per group, compares:
  1. Score curve (mean p_cal by DOY) for A score and D score
  2. Per-site-year summary: peak_DOY, peak_score, first_crossing_DOY,
                            alert_DOY, score_at_alert, lead
  3. Tier/gate behavior: no/weak/strong rate, A pass rate, D pass rate,
                          dispatch pass rate
  4. Statistical separation:
      - early vs normal, late vs normal: KS + Mann-Whitney on peak_DOY,
        alert_DOY, score_at_alert
      - one-vs-rest AUC for early/late discrimination

Inputs:
  - A ckpt, D ckpt
  - cascade_2tier tier CSVs (with tier + with_history columns)

Reads tier csv if --tier_dir is given (re-uses tier classifications).
Else recomputes A_alert/B_alert/tier internally.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, mannwhitneyu
from sklearn.metrics import roc_auc_score

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


def per_sy_score_summary(probs_df: pd.DataFrame, ev_keys: set, doy_start: int,
                         tau: float, k: int) -> pd.DataFrame:
    df_idx = probs_df.set_index(["site", "year"])
    rows = []
    for sy in ev_keys:
        try:
            g = df_idx.loc[sy]
            if isinstance(g, pd.Series):
                g = g.to_frame().T
            g = g.sort_values("tstar")
        except KeyError:
            continue
        ts = g["tstar"].values.astype(int)
        ps = g["p_cal"].values.astype(float)
        if len(ps) == 0:
            continue
        peak_idx = int(np.argmax(ps))
        fc = first_crossing_k(ts, ps, tau, k)
        rows.append({
            "site": sy[0], "year": sy[1],
            "peak_tstar": int(ts[peak_idx]),
            "peak_DOY": int(ts[peak_idx]) + int(doy_start),
            "peak_score": float(ps[peak_idx]),
            "first_crossing_tstar": (int(fc) if fc is not None else None),
            "alert_DOY": (int(fc) + int(doy_start)) if fc is not None else None,
            "score_at_alert": float(ps[ts == fc][0]) if fc is not None else None,
        })
    return pd.DataFrame(rows)


def score_curve_per_group(probs_df: pd.DataFrame, sy_group_map: dict, doy_start: int) -> pd.DataFrame:
    df = probs_df.copy()
    df["doy"] = df["tstar"].astype(int) + int(doy_start)
    df["group"] = df.apply(lambda r: sy_group_map.get((r["site"], int(r["year"])), None), axis=1)
    df = df[df["group"].notna()]
    return df.groupby(["group", "doy"])["p_cal"].agg(
        mean="mean", median="median",
        q25=lambda x: float(np.quantile(x, 0.25)),
        q75=lambda x: float(np.quantile(x, 0.75)),
        n="count"
    ).reset_index()


def cmp_groups_test(a: np.ndarray, b: np.ndarray, label: str) -> dict:
    if len(a) < 5 or len(b) < 5:
        return {"n_a": len(a), "n_b": len(b), "note": "too few"}
    ks = ks_2samp(a, b)
    try:
        mwu = mannwhitneyu(a, b, alternative="two-sided")
    except ValueError:
        mwu = type("X", (), {"statistic": float("nan"), "pvalue": float("nan")})()
    return {"n_a": int(len(a)), "n_b": int(len(b)),
            "a_mean": float(np.mean(a)), "b_mean": float(np.mean(b)),
            "a_median": float(np.median(a)), "b_median": float(np.median(b)),
            "delta_mean": float(np.mean(a) - np.mean(b)),
            "KS": float(ks.statistic), "KS_p": float(ks.pvalue),
            "MWU": float(mwu.statistic), "MWU_p": float(mwu.pvalue)}


def one_vs_rest_auc(values_outlier: np.ndarray, values_rest: np.ndarray) -> dict:
    if len(values_outlier) < 5 or len(values_rest) < 5:
        return {"n_out": len(values_outlier), "n_rest": len(values_rest), "AUC": None}
    y = np.concatenate([np.ones(len(values_outlier)), np.zeros(len(values_rest))])
    x = np.concatenate([values_outlier, values_rest])
    valid = ~np.isnan(x)
    if valid.sum() < 5 or len(set(y[valid].tolist())) < 2:
        return {"n_out": len(values_outlier), "n_rest": len(values_rest), "AUC": None}
    try:
        auc = float(roc_auc_score(y[valid], x[valid]))
        return {"n_out": int(len(values_outlier)), "n_rest": int(len(values_rest)),
                "AUC_dir_agnostic": max(auc, 1 - auc), "AUC_raw": auc}
    except ValueError:
        return {"n_out": len(values_outlier), "n_rest": len(values_rest), "AUC": None}


def analyze_split(split: str, tier_csv: Path, A_df: pd.DataFrame, D_df: pd.DataFrame,
                   doy_start: int, tau_A: float, k_A: int, out_dir: Path) -> dict:
    print(f"\n{'='*68}\n  {split.upper()}\n{'='*68}")
    tier_df = pd.read_csv(tier_csv)
    ev = tier_df[tier_df.is_event == 1].copy()
    if len(ev) < 20:
        print(f"  too few events ({len(ev)}); skip")
        return {}
    ev["L_DOY"] = ev["true_L"].astype(float) + int(doy_start)
    q10 = float(ev["L_DOY"].quantile(0.10))
    q25 = float(ev["L_DOY"].quantile(0.25))
    q75 = float(ev["L_DOY"].quantile(0.75))
    q90 = float(ev["L_DOY"].quantile(0.90))

    def _g(L):
        if L <= q10: return "early_outlier"
        if L >= q90: return "late_outlier"
        if q25 <= L <= q75: return "normal_event"
        return "between"
    ev["L_group"] = ev["L_DOY"].apply(_g)
    n_groups = Counter(ev["L_group"])
    print(f"  n_events={len(ev)}  L_DOY q10/q25/q50/q75/q90 = "
          f"{q10:.0f}/{q25:.0f}/{ev['L_DOY'].median():.0f}/{q75:.0f}/{q90:.0f}")
    print(f"  groups: " + "  ".join(f"{g}={n_groups.get(g, 0)}"
                                       for g in ["early_outlier", "between", "normal_event", "late_outlier"]))

    sy_group = {(r.site, int(r.year)): r.L_group for r in ev.itertuples(index=False)}

    # === tier / gate behavior ===
    print(f"\n  --- tier rates by group ---")
    print(f"    {'group':>14}  {'n':>4}  {'no_alert':>14}  {'weak_alert':>14}  {'strong_alert':>14}")
    tier_rates = {}
    for grp in ["early_outlier", "normal_event", "late_outlier"]:
        sub = ev[ev.L_group == grp]
        n = len(sub)
        if n == 0:
            continue
        tc = Counter(sub["tier"])
        na = tc.get("no_alert", 0); we = tc.get("weak_alert", 0); st = tc.get("strong_alert", 0)
        a_pass = (sub.A_alert_tstar.notna()).sum()
        # dispatch B pass = (weak or strong)? Actually strong = both A and B alert
        # B alert pass for the dispatch op
        b_pass = (sub.B_alert_tstar.notna()).sum() if "B_alert_tstar" in sub.columns else st
        tier_rates[grp] = {"n": int(n), "no_alert": int(na), "weak_alert": int(we),
                            "strong_alert": int(st),
                            "A_pass": int(a_pass), "B_pass": int(b_pass)}
        print(f"    {grp:>14}  {n:>4d}  "
              f"{na:>3d} ({na/n*100:>5.1f}%)  "
              f"{we:>3d} ({we/n*100:>5.1f}%)  "
              f"{st:>3d} ({st/n*100:>5.1f}%)")
        print(f"    {'':>14}  {'':>4}  A_pass={a_pass}/{n}={a_pass/n*100:.1f}%  "
              f"B_pass={b_pass}/{n}={b_pass/n*100:.1f}%")

    # === per-site-year score summary per group, for A and D ===
    summaries = {}  # {(score, group): df}
    for score_name, src in [("A", A_df), ("D", D_df)]:
        ev_keys = {(r.site, int(r.year)) for r in ev.itertuples(index=False)}
        per_sy = per_sy_score_summary(src, ev_keys, doy_start, tau_A, k_A)
        per_sy["L_group"] = per_sy.apply(lambda r: sy_group.get((r["site"], int(r["year"])), None), axis=1)
        print(f"\n  --- {score_name} score summary per group ---")
        print(f"    {'group':>14}  {'n':>4}  {'peak_DOY':>15}  {'alert_DOY':>15}  {'peak_score':>12}  {'score@alert':>13}")
        for grp in ["early_outlier", "normal_event", "late_outlier"]:
            sub = per_sy[per_sy.L_group == grp]
            n = len(sub)
            if n == 0:
                continue
            pkD = sub["peak_DOY"].astype(float).values
            alD = sub["alert_DOY"].dropna().astype(float).values
            pkS = sub["peak_score"].astype(float).values
            atS = sub["score_at_alert"].dropna().astype(float).values
            print(f"    {grp:>14}  {n:>4d}  "
                  f"{pkD.mean():>5.1f}±{pkD.std(ddof=0):>4.1f}      "
                  f"{(alD.mean() if len(alD) else float('nan')):>5.1f}±{(alD.std(ddof=0) if len(alD) else float('nan')):>4.1f}      "
                  f"{pkS.mean():>4.3f}±{pkS.std(ddof=0):>5.3f}  "
                  f"{(atS.mean() if len(atS) else float('nan')):>5.3f}±{(atS.std(ddof=0) if len(atS) else float('nan')):>5.3f}")
            summaries[(score_name, grp)] = sub

    # === score curve per group (A and D) ===
    for score_name, src in [("A", A_df), ("D", D_df)]:
        curve = score_curve_per_group(src, sy_group, doy_start)
        curve.to_csv(out_dir / f"curve_{score_name}_{split}.csv", index=False)
        print(f"  [saved] curve_{score_name}_{split}.csv")

    # === statistical separation ===
    print(f"\n  --- statistical separation ---")
    sep_results = {}
    for score_name in ["A", "D"]:
        print(f"  [score={score_name}]")
        for metric in ["peak_DOY", "alert_DOY", "peak_score", "score_at_alert"]:
            normal = summaries.get((score_name, "normal_event"))
            if normal is None: continue
            n_vals = normal[metric].dropna().astype(float).values
            for outlier in ["early_outlier", "late_outlier"]:
                out_df = summaries.get((score_name, outlier))
                if out_df is None: continue
                o_vals = out_df[metric].dropna().astype(float).values
                cmp = cmp_groups_test(o_vals, n_vals, f"{outlier}_vs_normal_{metric}_{score_name}")
                if "note" in cmp:
                    print(f"    {metric:>15}  {outlier:>14} vs normal:  {cmp['note']}")
                    continue
                print(f"    {metric:>15}  {outlier:>14} vs normal:  "
                      f"d_mean={cmp['delta_mean']:+6.1f}  KS={cmp['KS']:.3f} (p={cmp['KS_p']:.2e})  "
                      f"MWU_p={cmp['MWU_p']:.2e}")
                sep_results[f"{score_name}_{metric}_{outlier}_vs_normal"] = cmp

    # === one-vs-rest AUC for early/late discrimination using peak_DOY ===
    print(f"\n  --- one-vs-rest AUC (early or late vs rest) ---")
    aucs = {}
    for score_name in ["A", "D"]:
        normal = summaries.get((score_name, "normal_event"))
        if normal is None: continue
        for outlier in ["early_outlier", "late_outlier"]:
            out_df = summaries.get((score_name, outlier))
            if out_df is None: continue
            for metric in ["peak_DOY", "alert_DOY", "peak_score"]:
                o_vals = out_df[metric].dropna().astype(float).values
                # rest = all NON-outlier events (incl between)
                rest_keys = ev[ev.L_group != outlier]
                # take per_sy rows where in rest
                rest_keys = [g_key for g_key in summaries
                             if g_key[0] == score_name and g_key[1] != outlier]
                per_sy_all = pd.concat([summaries[g_key] for g_key in rest_keys],
                                        ignore_index=True) if rest_keys else None
                if per_sy_all is None:
                    continue
                r_vals = per_sy_all[metric].dropna().astype(float).values
                res = one_vs_rest_auc(o_vals, r_vals)
                aucs[f"{score_name}_{metric}_{outlier}"] = res
                if res.get("AUC_dir_agnostic") is not None:
                    print(f"    {score_name}  {metric:>11}  {outlier:>14}_vs_rest:  "
                          f"AUC={res['AUC_dir_agnostic']:.3f}  raw={res['AUC_raw']:.3f}  "
                          f"(n_out={res['n_out']}, n_rest={res['n_rest']})")

    return {"split": split, "n_events": int(len(ev)),
            "quantiles": {"q10": q10, "q25": q25, "q75": q75, "q90": q90},
            "group_counts": {g: int(n_groups.get(g, 0))
                              for g in ["early_outlier", "between", "normal_event", "late_outlier"]},
            "tier_rates": tier_rates,
            "separation": sep_results,
            "aucs": aucs}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--d_ckpt", required=True)
    ap.add_argument("--tier_dir", required=True,
                    help="cascade_2tier output dir containing tier_val.csv tier_test.csv")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--tau_A", type=float, default=0.55)
    ap.add_argument("--k_A", type=int, default=3)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tier_dir = Path(args.tier_dir)

    class N: pass
    common = N()
    for f in ["pest", "run", "split_seed", "val_year", "test_year_min", "test_year_max"]:
        setattr(common, f, getattr(args, f))

    print("\n========== building A probs ==========")
    common.stage1_ckpt = args.baseline_ckpt
    A_cache = build_probs(common)
    doy_start = int(C.DOY_START)
    print("\n========== building D probs ==========")
    common.stage1_ckpt = args.d_ckpt
    D_cache = build_probs(common)

    out = {"args": vars(args), "splits": {}}
    for split in ["val", "test"]:
        tier_csv = tier_dir / f"tier_{split}.csv"
        if not tier_csv.exists():
            print(f"[warn] {tier_csv} not found; skip {split}")
            continue
        res = analyze_split(split, tier_csv,
                              A_cache[f"{split}_df"], D_cache[f"{split}_df"],
                              doy_start, args.tau_A, args.k_A, out_dir)
        out["splits"][split] = res

    (out_dir / "outlier_score_shape_summary.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {out_dir / 'outlier_score_shape_summary.json'}")


if __name__ == "__main__":
    main()
