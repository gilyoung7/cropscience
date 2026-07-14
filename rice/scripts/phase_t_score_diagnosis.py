"""
Phase T8 — Per-sample timing diagnostic on Stage 1 score curves.

Loads a Stage 1 nowcast XGB ckpt, runs inference on val + test, then asks:
  Does the score trajectory across tstars carry per-sample timing info (L_DOY)?

Event site-years are grouped by L_DOY quartile (per split):
  early_event:  L_DOY <= Q1
  normal_event: Q1 < L_DOY <= Q3
  late_event:   L_DOY > Q3
Plus non_event_FP reference (non-event site-years that alerted under tau, k).

Sections:
  1. Score curve shape — per group, per DOY: mean / median / q25 / q75 (CSV + plot)
  2. Peak position    — per site-year: peak_DOY, peak_score; group stats
  3. First crossing   — per site-year: first_crossing_DOY at (tau, k); group stats
  4. Correlations     — corr(L_DOY, score_peak_DOY / alert_DOY / peak_score)
                        Pearson + Spearman, val + test
  5. TP vs FP separability — score-shape features per alerted site-year
                              per-feature AUC + KS; LogReg multivar AUC

No retraining. Reads ckpt -> inference only.

Coordinates: everything reported in DOY (= tstar + DOY_START).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest


# ---------- Score-shape features ----------

SHAPE_FEATURES = [
    "score_at_alert",
    "score_mean_3d", "score_mean_7d", "score_mean_14d",
    "score_max_7d", "score_max_14d",
    "score_slope_7d", "score_slope_14d",
    "area_above_tau_7d", "area_above_tau_14d", "area_above_tau_season_so_far",
    "days_above_tau_before_alert",
    "consecutive_days_above_tau",
    "peak_score_so_far",
    "alert_DOY",
]


def first_crossing_tstar(ts: np.ndarray, ps: np.ndarray, tau: float, k: int) -> int | None:
    streak = 0
    for i in range(len(ts)):
        if ps[i] >= tau:
            streak += 1
            if streak >= k:
                return int(ts[i])
        else:
            streak = 0
    return None


def per_site_year_records(probs_df: pd.DataFrame, tau: float, k: int,
                          doy_start: int) -> pd.DataFrame:
    """Build per-(site, year) record with peak/alert info and shape features."""
    rows = []
    for (site, year), g in probs_df.groupby(["site", "year"], sort=False):
        g = g.sort_values("tstar")
        ts = g.tstar.values.astype(int)
        ps = g.p_cal.values.astype(float)
        if len(ps) == 0:
            continue
        is_event = int(g.y_event.iloc[0])
        true_L = g.true_L.iloc[0]
        true_R = g.true_R.iloc[0]
        # Peak (whole season, oracle)
        peak_idx = int(np.argmax(ps))
        peak_tstar = int(ts[peak_idx])
        peak_score = float(ps[peak_idx])
        # First crossing alert
        at = first_crossing_tstar(ts, ps, tau, k)
        alerted = int(at is not None)
        # Shape features (only meaningful when alerted)
        if at is not None:
            idx_at = int(np.where(ts == at)[0][0])
            score_at_alert = float(ps[idx_at])
            def _win(n):
                lo = max(0, idx_at - n)
                return ps[lo:idx_at + 1]  # includes the alert day
            w3 = _win(3); w7 = _win(7); w14 = _win(14)
            mean_3d = float(np.mean(w3)) if len(w3) > 0 else float("nan")
            mean_7d = float(np.mean(w7)) if len(w7) > 0 else float("nan")
            mean_14d = float(np.mean(w14)) if len(w14) > 0 else float("nan")
            max_7d = float(np.max(w7)) if len(w7) > 0 else float("nan")
            max_14d = float(np.max(w14)) if len(w14) > 0 else float("nan")
            def _slope(w):
                if len(w) < 2: return 0.0
                t = np.arange(len(w), dtype=float)
                tc = t - t.mean()
                v = float((tc ** 2).sum()) + 1e-8
                wc = w - w.mean()
                return float((wc * tc).sum() / v)
            slope_7d = _slope(w7)
            slope_14d = _slope(w14)
            def _area(w, tau_):
                if len(w) == 0: return 0.0
                return float(np.clip(w - tau_, 0, None).sum())
            area_7d = _area(w7, tau)
            area_14d = _area(w14, tau)
            area_season = _area(ps[:idx_at + 1], tau)
            # streak ending at alert
            streak = 0
            for v in ps[:idx_at + 1][::-1]:
                if v >= tau:
                    streak += 1
                else:
                    break
            consecutive_days_above_tau = streak
            days_above_tau_before_alert = int((ps[:idx_at + 1] >= tau).sum())
            peak_score_so_far = float(ps[:idx_at + 1].max())
            alert_DOY = int(at + doy_start)
        else:
            score_at_alert = mean_3d = mean_7d = mean_14d = float("nan")
            max_7d = max_14d = float("nan")
            slope_7d = slope_14d = float("nan")
            area_7d = area_14d = area_season = float("nan")
            consecutive_days_above_tau = 0
            days_above_tau_before_alert = 0
            peak_score_so_far = float("nan")
            alert_DOY = None

        rec = {
            "site": str(site), "year": int(year),
            "is_event": is_event,
            "true_L": (int(true_L) if pd.notna(true_L) else None),
            "true_R": (int(true_R) if pd.notna(true_R) else None),
            "L_DOY": (int(true_L) + int(doy_start) if pd.notna(true_L) else None),
            "alerted": alerted,
            "alert_tstar": (int(at) if at is not None else None),
            "alert_DOY": alert_DOY,
            "peak_tstar": peak_tstar,
            "peak_DOY": int(peak_tstar + doy_start),
            "peak_score": peak_score,
            "score_at_alert": score_at_alert,
            "score_mean_3d": mean_3d,
            "score_mean_7d": mean_7d,
            "score_mean_14d": mean_14d,
            "score_max_7d": max_7d,
            "score_max_14d": max_14d,
            "score_slope_7d": slope_7d,
            "score_slope_14d": slope_14d,
            "area_above_tau_7d": area_7d,
            "area_above_tau_14d": area_14d,
            "area_above_tau_season_so_far": area_season,
            "days_above_tau_before_alert": days_above_tau_before_alert,
            "consecutive_days_above_tau": consecutive_days_above_tau,
            "peak_score_so_far": peak_score_so_far,
        }
        # cls (TP/FP/FN/TN)
        if is_event == 1 and alerted == 1: rec["cls"] = "TP"
        elif is_event == 0 and alerted == 1: rec["cls"] = "FP"
        elif is_event == 1 and alerted == 0: rec["cls"] = "FN"
        else: rec["cls"] = "TN"
        rows.append(rec)
    return pd.DataFrame(rows)


def assign_event_group(records: pd.DataFrame) -> pd.DataFrame:
    """Add 'group' column: early_event / normal_event / late_event / non_event_FP / non_event_TN / no_alert_event"""
    df = records.copy()
    ev = df[df.is_event == 1].copy()
    if len(ev) >= 4:
        q1 = float(ev["L_DOY"].quantile(0.25))
        q3 = float(ev["L_DOY"].quantile(0.75))
    else:
        q1 = q3 = float("nan")

    def _g(r):
        if r.is_event == 1:
            if r.L_DOY is None or pd.isna(r.L_DOY):
                return "event_unknownL"
            if r.L_DOY <= q1: return "early_event"
            if r.L_DOY > q3: return "late_event"
            return "normal_event"
        else:
            return "non_event_FP" if r.alerted == 1 else "non_event_TN"
    df["group"] = df.apply(_g, axis=1)
    return df, {"L_DOY_q25": q1, "L_DOY_q75": q3,
                "n_early": int((df.group == "early_event").sum()),
                "n_normal": int((df.group == "normal_event").sum()),
                "n_late": int((df.group == "late_event").sum())}


# ---------- Section 1: score curve shape per group per DOY ----------

def score_curve_by_group(probs_df: pd.DataFrame, group_map: dict, doy_start: int) -> pd.DataFrame:
    """For each (group, DOY): mean / median / q25 / q75 of p_cal across the group."""
    df = probs_df.copy()
    df["doy"] = df["tstar"].astype(int) + int(doy_start)
    df["key"] = df["site"].astype(str) + "|" + df["year"].astype(str)
    df["group"] = df["key"].map(group_map)
    df = df[df["group"].notna()].copy()
    agg = df.groupby(["group", "doy"])["p_cal"].agg(
        mean="mean", median="median",
        q25=lambda x: float(np.quantile(x, 0.25)),
        q75=lambda x: float(np.quantile(x, 0.75)),
        n="count",
    ).reset_index()
    return agg


def maybe_plot(curve_df: pd.DataFrame, out_path: Path, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return False
    colors = {"early_event": "tab:blue", "normal_event": "tab:green",
              "late_event": "tab:red", "non_event_FP": "tab:orange",
              "non_event_TN": "tab:gray"}
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for grp, sub in curve_df.groupby("group"):
        sub = sub.sort_values("doy")
        c = colors.get(grp, "k")
        ax.plot(sub["doy"], sub["mean"], color=c, label=f"{grp} (n_doy={len(sub)})", linewidth=1.6)
        ax.fill_between(sub["doy"], sub["q25"], sub["q75"], color=c, alpha=0.15)
    ax.set_xlabel("DOY")
    ax.set_ylabel("p_cal")
    ax.set_title(title)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)
    return True


# ---------- Section 4: correlations ----------

def correlations(records: pd.DataFrame, target_groups=("early_event", "normal_event", "late_event")) -> dict:
    ev = records[records["is_event"] == 1].copy()
    ev = ev[ev["L_DOY"].notna()]
    out = {}
    for col in ["peak_DOY", "alert_DOY", "peak_score"]:
        sub = ev[ev[col].notna()] if col == "alert_DOY" else ev
        if len(sub) < 5:
            out[col] = {"pearson": float("nan"), "spearman": float("nan"), "n": int(len(sub))}
            continue
        L = sub["L_DOY"].astype(float).values
        x = sub[col].astype(float).values
        pe = float(np.corrcoef(L, x)[0, 1]) if np.std(L) > 0 and np.std(x) > 0 else float("nan")
        sp = float(spearmanr(L, x).correlation) if np.std(L) > 0 and np.std(x) > 0 else float("nan")
        out[col] = {"pearson": pe, "spearman": sp, "n": int(len(sub))}
    return out


# ---------- Section 5: TP vs FP separability ----------

def tp_fp_separability(records: pd.DataFrame) -> dict:
    alerted = records[records.alerted == 1].copy()
    tp = alerted[alerted.cls == "TP"]
    fp = alerted[alerted.cls == "FP"]
    n_tp = int(len(tp)); n_fp = int(len(fp))
    if n_tp < 5 or n_fp < 5:
        return {"n_tp": n_tp, "n_fp": n_fp, "per_feature": {}, "multivar": None}
    y = (alerted.cls == "TP").astype(int).values
    per_feat = {}
    for col in SHAPE_FEATURES:
        if col not in alerted.columns:
            continue
        x_raw = alerted[col].astype(float).values
        valid_mask = ~np.isnan(x_raw)
        if valid_mask.sum() < 5 or np.unique(x_raw[valid_mask]).size < 2:
            continue
        x = x_raw[valid_mask]
        yv = y[valid_mask]
        try:
            auc = float(roc_auc_score(yv, x))
            auc = max(auc, 1 - auc)
        except ValueError:
            auc = float("nan")
        ks_res = ks_2samp(tp[col].dropna(), fp[col].dropna())
        per_feat[col] = {"AUC_dir_agnostic": auc,
                         "KS": float(ks_res.statistic), "p_KS": float(ks_res.pvalue),
                         "n_used": int(len(yv))}
    # Multivariate
    cols = [c for c in SHAPE_FEATURES if c in alerted.columns]
    X = alerted[cols].astype(float).copy()
    X = X.fillna(X.mean()).values
    mvar = {}
    if len(np.unique(y)) >= 2:
        try:
            model = LogisticRegression(max_iter=2000, C=1.0)
            model.fit(X, y)
            p_in = model.predict_proba(X)[:, 1]
            in_auc = float(roc_auc_score(y, p_in))
            # 5-fold CV by site
            from sklearn.model_selection import GroupKFold
            from sklearn.metrics import roc_auc_score as _rocauc
            groups = alerted["site"].astype(str).values
            gkf = GroupKFold(n_splits=min(5, len(set(groups))))
            cv_aucs = []
            for tr_idx, te_idx in gkf.split(X, y, groups):
                if len(set(y[tr_idx])) < 2 or len(set(y[te_idx])) < 2:
                    continue
                m = LogisticRegression(max_iter=2000, C=1.0)
                m.fit(X[tr_idx], y[tr_idx])
                p_te = m.predict_proba(X[te_idx])[:, 1]
                cv_aucs.append(float(_rocauc(y[te_idx], p_te)))
            mvar = {"in_sample_AUC": in_auc,
                    "cv_AUC_mean": float(np.mean(cv_aucs)) if cv_aucs else float("nan"),
                    "cv_AUC_std": float(np.std(cv_aucs)) if cv_aucs else float("nan"),
                    "n_cv_folds": int(len(cv_aucs)),
                    "coefs": dict(zip(cols, model.coef_[0].tolist()))}
        except Exception as e:
            mvar = {"error": str(e)}
    return {"n_tp": n_tp, "n_fp": n_fp, "per_feature": per_feat, "multivar": mvar}


# ---------- Main ----------

def run_split(probs_df: pd.DataFrame, split_name: str, tau: float, k: int,
              doy_start: int, out_dir: Path, plot: bool) -> dict:
    print(f"\n========== {split_name} ==========")
    records = per_site_year_records(probs_df, tau, k, doy_start)
    records, gstats = assign_event_group(records)
    print(f"  records: {len(records)}  events: {int((records.is_event==1).sum())}  "
          f"alerted: {int((records.alerted==1).sum())}")
    print(f"  L_DOY quartiles (event-only): q25={gstats['L_DOY_q25']}  q75={gstats['L_DOY_q75']}  "
          f"groups: early={gstats['n_early']} normal={gstats['n_normal']} late={gstats['n_late']}")
    records.to_csv(out_dir / f"records_{split_name}.csv", index=False)

    # Section 1: score curve by group
    group_map = {f"{r.site}|{r.year}": r.group for r in records.itertuples(index=False)}
    curve = score_curve_by_group(probs_df, group_map, doy_start)
    curve.to_csv(out_dir / f"score_curve_by_group_{split_name}.csv", index=False)
    print(f"  [section 1] score_curve_by_group rows={len(curve)}  groups={sorted(curve['group'].unique())}")
    plot_path = out_dir / f"score_curve_{split_name}.png"
    if plot and maybe_plot(curve, plot_path, f"Score curve by group ({split_name})"):
        print(f"  [section 1] plot saved: {plot_path}")

    # Section 2/3: peak + first_crossing group stats
    ev = records[records.is_event == 1].copy()
    print(f"  [section 2/3] group stats:")
    print(f"    {'group':>14} {'n':>4}  {'peak_DOY_mean':>14} {'peak_DOY_std':>13} {'peak_DOY_q25':>13} {'peak_DOY_q75':>13}  "
          f"{'alert_DOY_mean':>15} {'alert_DOY_std':>13}  {'L_DOY_mean':>11}")
    sec23 = {}
    for grp in ["early_event", "normal_event", "late_event"]:
        sub = ev[ev.group == grp]
        if sub.empty:
            continue
        peaks = sub["peak_DOY"].astype(float).values
        alerts_doy = sub.loc[sub.alert_DOY.notna(), "alert_DOY"].astype(float).values
        L = sub["L_DOY"].astype(float).values
        cell = {
            "n": int(len(sub)),
            "peak_DOY_mean": float(peaks.mean()), "peak_DOY_std": float(peaks.std(ddof=0)),
            "peak_DOY_q25": float(np.quantile(peaks, 0.25)),
            "peak_DOY_q75": float(np.quantile(peaks, 0.75)),
            "alert_DOY_mean": float(alerts_doy.mean()) if len(alerts_doy) else float("nan"),
            "alert_DOY_std": float(alerts_doy.std(ddof=0)) if len(alerts_doy) else float("nan"),
            "n_alert": int(len(alerts_doy)),
            "L_DOY_mean": float(L.mean()),
        }
        sec23[grp] = cell
        print(f"    {grp:>14} {cell['n']:>4d}  {cell['peak_DOY_mean']:>14.1f} "
              f"{cell['peak_DOY_std']:>13.1f} {cell['peak_DOY_q25']:>13.1f} {cell['peak_DOY_q75']:>13.1f}  "
              f"{cell['alert_DOY_mean']:>15.1f} {cell['alert_DOY_std']:>13.1f}  {cell['L_DOY_mean']:>11.1f}")

    # Section 4: correlations
    corr = correlations(records)
    print(f"  [section 4] correlations (event site-years only):")
    for k_, v in corr.items():
        print(f"    corr(L_DOY, {k_:>11s}): pearson={v['pearson']:.3f}  spearman={v['spearman']:.3f}  (n={v['n']})")

    # Section 5: TP/FP separability
    sep = tp_fp_separability(records)
    print(f"  [section 5] TP/FP separability (alerted only, n_TP={sep.get('n_tp', 0)}, n_FP={sep.get('n_fp', 0)}):")
    pf = sep.get("per_feature", {})
    if pf:
        print(f"    {'feature':>30}  {'AUC':>6} {'KS':>6} {'p_KS':>9}")
        for feat, m in sorted(pf.items(), key=lambda x: -x[1].get("AUC_dir_agnostic", 0)):
            print(f"    {feat:>30}  {m['AUC_dir_agnostic']:>6.3f} {m['KS']:>6.3f} {m['p_KS']:>9.2e}")
    mv = sep.get("multivar")
    if mv and "in_sample_AUC" in mv:
        print(f"    multivar LogReg: in-sample AUC={mv['in_sample_AUC']:.3f}  "
              f"CV AUC mean={mv['cv_AUC_mean']:.3f} ± {mv['cv_AUC_std']:.3f}  "
              f"(folds={mv['n_cv_folds']})")
    return {
        "n_records": int(len(records)),
        "group_stats": gstats,
        "section2_3_group_stats": sec23,
        "section4_correlations": corr,
        "section5_separability": sep,
    }


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
    ap.add_argument("--no_plot", action="store_true",
                    help="skip matplotlib plot generation")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.stage1_ckpt).stem
    print(f"\n========== Phase T8 score diagnostic :: {label} ==========")
    print(f"[cfg] tau={args.tau}  k={args.k}")

    # Lazy import to share probs builder
    from rice.scripts.phase_t_lead_aware_eval import build_probs
    cache = build_probs(args)
    doy_start = int(C.DOY_START)
    print(f"[DOY] DOY_START={doy_start}  DOY_END={int(C.DOY_END)}")

    out = {"label": label, "args": {"tau": args.tau, "k": args.k},
           "ckpt_meta": cache["ckpt_meta"], "splits": {}}
    for split_name, probs_df in [("val", cache["val_df"]), ("test", cache["test_df"])]:
        res = run_split(probs_df, split_name, args.tau, args.k, doy_start,
                        out_dir, plot=not args.no_plot)
        out["splits"][split_name] = res

    (out_dir / f"score_diag_summary_{label}.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n[saved] all CSVs + plot + summary in {out_dir}")
    print(f"[saved] {out_dir / f'score_diag_summary_{label}.json'}")


if __name__ == "__main__":
    main()
