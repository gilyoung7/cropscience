"""
Phase T8b — Score-sequence motif diagnostic on Stage 1 ckpt.

Computes per-site-year motif features from p_cal sequences and asks whether
they help discriminate TP vs FP at (tau, k). Causal vs oracle features are
separated; recall-preserving FAR sweep uses causal-only secondary filter.

Causal features (info up to alert_tstar only):
  C1  first_crossing_DOY                            (raw, k=1 crossing)
  C2  rise_time_to_tau                              (= first_crossing_tstar)
  C3  max_slope_7d_before_alert
  C4  max_slope_14d_before_alert
  C5  total_variation_28d_before_alert
  C6  sign_change_count_slope_28d_before_alert
  C7  consecutive_days_above_tau_at_alert
  C8  peak_score_so_far
  C9  score_at_alert
  C10 score_percentile_by_DOY_at_alert              (in-split DOY-rank, weak leak)
  C11 alert_DOY

Oracle features (full season; diagnostic only):
  O1  crossing_count_season
  O2  last_crossing_DOY_season
  O3  total_days_above_tau_season
  O4  longest_consecutive_above_tau_season
  O5  num_local_peaks_season
  O6  num_peaks_above_tau_season
  O7  peak_width_at_global_peak_50pct
  O8  second_peak_score_season
  O9  peak_gap_days_season

28d sequence (alert-anchored, causal) -> PCA(3) + KMeans(4) clusters:
  cluster_id, pc1..pc3   per alerted site-year
  cluster breakdown: #TP / #FP / TP lead_mean

Outputs (val + test):
  - per-feature TP/FP AUC (direction-agnostic) + KS + p_KS
  - LogReg multivariate AUC: causal-only / oracle-only / combined
    (in-sample + 5-fold GroupKFold CV by site)
  - recall-preserving FAR sweep on alerted set, secondary causal LogReg score
    (using CV scores if all alerted samples are CV-assigned; else in-sample)
    target: recall >= --target_recall (default 0.90)
  - cluster breakdown table
  - motif_records_<split>.csv with all features
  - motif_diag_summary_<label>.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest


CAUSAL_FEATS = [
    "first_crossing_DOY", "rise_time_to_tau",
    "max_slope_7d_before_alert", "max_slope_14d_before_alert",
    "total_variation_28d_before_alert", "sign_change_count_slope_28d_before_alert",
    "consecutive_days_above_tau_at_alert",
    "peak_score_so_far", "score_at_alert",
    "score_percentile_by_DOY_at_alert", "alert_DOY",
]

ORACLE_FEATS = [
    "crossing_count_season", "last_crossing_DOY_season",
    "total_days_above_tau_season", "longest_consecutive_above_tau_season",
    "num_local_peaks_season", "num_peaks_above_tau_season",
    "peak_width_at_global_peak_50pct", "second_peak_score_season",
    "peak_gap_days_season",
]


def first_crossing_k(ts: np.ndarray, ps: np.ndarray, tau: float, k: int) -> int | None:
    streak = 0
    for i in range(len(ts)):
        if ps[i] >= tau:
            streak += 1
            if streak >= k:
                return int(ts[i])
        else:
            streak = 0
    return None


def build_doy_percentile_table(probs_df: pd.DataFrame, doy_start: int) -> dict:
    df = probs_df.copy()
    df["doy"] = df["tstar"].astype(int) + int(doy_start)
    table = {}
    for doy, g in df.groupby("doy"):
        arr = np.sort(g["p_cal"].astype(float).values)
        table[int(doy)] = arr
    return table


def doy_percentile_lookup(table: dict, doy: int, score: float) -> float:
    arr = table.get(int(doy))
    if arr is None or len(arr) == 0:
        return float("nan")
    rank = float(np.searchsorted(arr, score, side="right"))
    return rank / float(len(arr))


def _max_rolling_slope(ps: np.ndarray, w_len: int) -> float:
    n = len(ps)
    if n < 2:
        return float("nan")
    w_len = min(w_len, n)
    if w_len < 2:
        return float("nan")
    t = np.arange(w_len, dtype=float)
    tc = t - t.mean()
    var = float((tc ** 2).sum()) + 1e-8
    slopes = []
    for j in range(w_len - 1, n):
        win = ps[j - w_len + 1: j + 1]
        wc = win - win.mean()
        slopes.append(float((wc * tc).sum() / var))
    return max(slopes) if slopes else float("nan")


def compute_motifs(probs_df: pd.DataFrame, tau: float, k: int, doy_start: int,
                   pct_table: dict) -> pd.DataFrame:
    rows = []
    for (site, year), g in probs_df.groupby(["site", "year"], sort=False):
        g = g.sort_values("tstar")
        ts = g.tstar.values.astype(int)
        ps = g.p_cal.values.astype(float)
        n = len(ps)
        if n == 0:
            continue
        is_event = int(g.y_event.iloc[0])
        true_L = g.true_L.iloc[0]
        alert_tstar = first_crossing_k(ts, ps, tau, k)
        alerted = alert_tstar is not None
        fcr = first_crossing_k(ts, ps, tau, 1)
        first_crossing_DOY = (int(fcr) + int(doy_start)) if fcr is not None else None
        rise_time_to_tau = int(fcr) if fcr is not None else None

        # Oracle (full season)
        above = ps >= tau
        crossing_count_season = int(np.sum(np.diff(above.astype(int)) != 0))
        above_idx = np.where(above)[0]
        last_crossing_DOY_season = (int(ts[above_idx[-1]] + doy_start)
                                    if len(above_idx) > 0 else None)
        total_days_above_tau_season = int(above.sum())
        longest = 0; cur = 0
        for v in above:
            if v: cur += 1; longest = max(longest, cur)
            else: cur = 0
        longest_consecutive_above_tau_season = int(longest)
        peak_idxs = [i for i in range(1, n - 1)
                     if ps[i] > ps[i - 1] and ps[i] > ps[i + 1]]
        num_local_peaks_season = len(peak_idxs)
        peak_scored = sorted([(float(ps[i]), int(i)) for i in peak_idxs], reverse=True)
        num_peaks_above_tau_season = int(sum(1 for sc, _ in peak_scored if sc >= tau))
        global_idx = int(np.argmax(ps))
        global_peak_score = float(ps[global_idx])
        half = global_peak_score * 0.5
        left = global_idx
        while left > 0 and ps[left - 1] >= half:
            left -= 1
        right = global_idx
        while right < n - 1 and ps[right + 1] >= half:
            right += 1
        peak_width_at_global_peak_50pct = int(right - left + 1)
        if len(peak_scored) >= 2:
            second_peak_score_season = peak_scored[1][0]
            i1 = peak_scored[0][1]; i2 = peak_scored[1][1]
            peak_gap_days_season = int(abs(ts[i1] - ts[i2]))
        else:
            second_peak_score_season = float("nan")
            peak_gap_days_season = float("nan")

        # Causal (pre-alert window)
        if alerted:
            idx_at = int(np.where(ts == alert_tstar)[0][0])
            ps_pre = ps[: idx_at + 1]
            max_slope_7d_before_alert = _max_rolling_slope(ps_pre, 7)
            max_slope_14d_before_alert = _max_rolling_slope(ps_pre, 14)
            lo28 = max(0, idx_at - 27)
            win28 = ps[lo28: idx_at + 1]
            if len(win28) >= 2:
                diffs = np.diff(win28)
                total_variation_28d_before_alert = float(np.abs(diffs).sum())
                signs = np.sign(diffs)
                signs = signs[signs != 0]
                sign_change_count_slope_28d_before_alert = (
                    int((signs[1:] != signs[:-1]).sum()) if len(signs) >= 2 else 0
                )
            else:
                total_variation_28d_before_alert = float("nan")
                sign_change_count_slope_28d_before_alert = float("nan")
            streak = 0
            for v in ps_pre[::-1]:
                if v >= tau: streak += 1
                else: break
            consecutive_days_above_tau_at_alert = int(streak)
            peak_score_so_far = float(ps_pre.max())
            score_at_alert = float(ps_pre[-1])
            doy_alert = int(alert_tstar + doy_start)
            score_percentile_by_DOY_at_alert = doy_percentile_lookup(
                pct_table, doy_alert, score_at_alert)
            alert_DOY = doy_alert
            seq28 = np.zeros(28, dtype=float)
            for i in range(28):
                j = idx_at - 27 + i
                seq28[i] = ps[j] if j >= 0 else 0.0
        else:
            max_slope_7d_before_alert = float("nan")
            max_slope_14d_before_alert = float("nan")
            total_variation_28d_before_alert = float("nan")
            sign_change_count_slope_28d_before_alert = float("nan")
            consecutive_days_above_tau_at_alert = 0
            peak_score_so_far = float("nan")
            score_at_alert = float("nan")
            score_percentile_by_DOY_at_alert = float("nan")
            alert_DOY = None
            seq28 = np.full(28, np.nan)

        if is_event == 1 and alerted: cls = "TP"
        elif is_event == 0 and alerted: cls = "FP"
        elif is_event == 1 and not alerted: cls = "FN"
        else: cls = "TN"

        rows.append({
            "site": str(site), "year": int(year), "is_event": is_event,
            "true_L": (int(true_L) if pd.notna(true_L) else None),
            "L_DOY": (int(true_L) + int(doy_start) if pd.notna(true_L) else None),
            "alerted": int(alerted), "cls": cls,
            "alert_tstar": (int(alert_tstar) if alert_tstar is not None else None),
            "alert_DOY": alert_DOY,
            # causal
            "first_crossing_DOY": first_crossing_DOY,
            "rise_time_to_tau": rise_time_to_tau,
            "max_slope_7d_before_alert": max_slope_7d_before_alert,
            "max_slope_14d_before_alert": max_slope_14d_before_alert,
            "total_variation_28d_before_alert": total_variation_28d_before_alert,
            "sign_change_count_slope_28d_before_alert": sign_change_count_slope_28d_before_alert,
            "consecutive_days_above_tau_at_alert": consecutive_days_above_tau_at_alert,
            "peak_score_so_far": peak_score_so_far,
            "score_at_alert": score_at_alert,
            "score_percentile_by_DOY_at_alert": score_percentile_by_DOY_at_alert,
            # oracle
            "crossing_count_season": crossing_count_season,
            "last_crossing_DOY_season": last_crossing_DOY_season,
            "total_days_above_tau_season": total_days_above_tau_season,
            "longest_consecutive_above_tau_season": longest_consecutive_above_tau_season,
            "num_local_peaks_season": num_local_peaks_season,
            "num_peaks_above_tau_season": num_peaks_above_tau_season,
            "peak_width_at_global_peak_50pct": peak_width_at_global_peak_50pct,
            "second_peak_score_season": second_peak_score_season,
            "peak_gap_days_season": peak_gap_days_season,
            "_seq28": seq28.tolist(),
        })
    return pd.DataFrame(rows)


def per_feature_tpfp(records: pd.DataFrame, feature_cols: list[str]) -> dict:
    alerted = records[records.alerted == 1].copy()
    if alerted.cls.nunique() < 2:
        return {}
    tp_df = alerted[alerted.cls == "TP"]
    fp_df = alerted[alerted.cls == "FP"]
    y = (alerted.cls == "TP").astype(int).values
    out = {}
    for col in feature_cols:
        if col not in alerted.columns:
            continue
        x_raw = pd.to_numeric(alerted[col], errors="coerce").values
        valid = ~np.isnan(x_raw)
        if valid.sum() < 5 or np.unique(x_raw[valid]).size < 2:
            continue
        x = x_raw[valid]; yv = y[valid]
        if len(set(yv.tolist())) < 2:
            continue
        try:
            auc = float(roc_auc_score(yv, x))
            auc = max(auc, 1 - auc)
        except ValueError:
            auc = float("nan")
        ks = ks_2samp(pd.to_numeric(tp_df[col], errors="coerce").dropna(),
                      pd.to_numeric(fp_df[col], errors="coerce").dropna())
        out[col] = {"AUC_dir_agnostic": auc, "KS": float(ks.statistic),
                    "p_KS": float(ks.pvalue), "n_valid": int(valid.sum())}
    return out


def multivar_logreg(records: pd.DataFrame, feature_cols: list[str]) -> dict | None:
    alerted = records[records.alerted == 1].copy()
    if alerted.cls.nunique() < 2 or len(alerted) < 10:
        return None
    feats = [c for c in feature_cols if c in alerted.columns]
    if not feats:
        return None
    X = alerted[feats].apply(pd.to_numeric, errors="coerce").copy()
    X = X.fillna(X.mean()).fillna(0.0).values
    y = (alerted.cls == "TP").astype(int).values
    groups = alerted["site"].astype(str).values

    model = LogisticRegression(max_iter=2000, C=1.0)
    model.fit(X, y)
    p_in = model.predict_proba(X)[:, 1]
    in_auc = float(roc_auc_score(y, p_in))

    n_groups = len(set(groups))
    n_splits = min(5, n_groups)
    cv_aucs = []
    cv_scores = np.full(len(y), np.nan, dtype=float)
    cv_assigned = np.zeros(len(y), dtype=bool)
    if n_splits >= 2:
        gkf = GroupKFold(n_splits=n_splits)
        for tr, te in gkf.split(X, y, groups):
            if len(set(y[tr])) < 2 or len(set(y[te])) < 2:
                continue
            m = LogisticRegression(max_iter=2000, C=1.0)
            m.fit(X[tr], y[tr])
            p_te = m.predict_proba(X[te])[:, 1]
            cv_scores[te] = p_te
            cv_assigned[te] = True
            cv_aucs.append(float(roc_auc_score(y[te], p_te)))
    return {
        "feats": feats,
        "in_sample_AUC": in_auc,
        "cv_AUC_mean": float(np.mean(cv_aucs)) if cv_aucs else float("nan"),
        "cv_AUC_std": float(np.std(cv_aucs)) if cv_aucs else float("nan"),
        "n_cv_folds": int(len(cv_aucs)),
        "in_sample_scores": p_in,
        "cv_scores": cv_scores,
        "cv_assigned": cv_assigned,
        "coefs": dict(zip(feats, model.coef_[0].tolist())),
    }


def recall_preserving_sweep(records: pd.DataFrame, mvar: dict | None,
                            target_recall: float) -> dict | None:
    if mvar is None:
        return None
    n_event_total = int((records.is_event == 1).sum())
    n_nonevent_total = int((records.is_event == 0).sum())
    if n_event_total == 0:
        return None
    alerted = records[records.alerted == 1].copy().reset_index(drop=True)
    is_tp = (alerted["cls"] == "TP").values
    is_fp = (alerted["cls"] == "FP").values
    p_in = mvar["in_sample_scores"]
    cv_scores = mvar["cv_scores"]; cv_assigned = mvar["cv_assigned"]
    use_cv = bool(cv_assigned.all())
    scores = cv_scores if use_cv else p_in
    base_tp = int(is_tp.sum()); base_fp = int(is_fp.sum())
    base_recall = base_tp / max(n_event_total, 1)
    base_far = base_fp / max(n_nonevent_total, 1)
    sweep = []
    for s in np.linspace(0.0, 1.0, 101):
        keep = scores >= s
        tp_kept = int((is_tp & keep).sum())
        fp_kept = int((is_fp & keep).sum())
        recall = tp_kept / max(n_event_total, 1)
        far = fp_kept / max(n_nonevent_total, 1)
        sweep.append({"thr": float(s), "TP": tp_kept, "FP": fp_kept,
                      "recall": recall, "FAR": far})
    cands = [r for r in sweep if r["recall"] >= target_recall]
    best = min(cands, key=lambda r: r["FAR"]) if cands else None
    return {"score_source": ("cv" if use_cv else "in_sample"),
            "n_event_total": n_event_total, "n_nonevent_total": n_nonevent_total,
            "base": {"TP": base_tp, "FP": base_fp,
                     "recall": base_recall, "FAR": base_far},
            "target_recall": target_recall,
            "best_at_target": best, "sweep": sweep}


def pca_kmeans_clustering(records: pd.DataFrame, n_pca: int = 3, n_clusters: int = 4) -> dict | None:
    alerted = records[records.alerted == 1].copy().reset_index(drop=True)
    if len(alerted) < n_clusters:
        return None
    seqs = np.array(alerted["_seq28"].tolist(), dtype=float)
    if seqs.ndim != 2:
        return None
    # replace nan with column mean (preserve causal padding=0 above)
    col_mean = np.nanmean(seqs, axis=0)
    inds = np.where(np.isnan(seqs))
    seqs[inds] = np.take(col_mean, inds[1])
    # standardize column-wise
    mu = seqs.mean(axis=0); sd = seqs.std(axis=0, ddof=0)
    sd[sd < 1e-6] = 1.0
    seqs_z = (seqs - mu) / sd
    pca = PCA(n_components=min(n_pca, seqs_z.shape[1]))
    pc = pca.fit_transform(seqs_z)
    km = KMeans(n_clusters=n_clusters, random_state=0, n_init="auto")
    cl = km.fit_predict(pc)
    alerted = alerted.copy()
    for i in range(pc.shape[1]):
        alerted[f"pc{i+1}"] = pc[:, i]
    alerted["cluster_id"] = cl.astype(int)
    breakdown = []
    for c in range(n_clusters):
        sub = alerted[alerted.cluster_id == c]
        n = int(len(sub))
        tp = int((sub.cls == "TP").sum())
        fp = int((sub.cls == "FP").sum())
        tp_rows = sub[sub.cls == "TP"]
        leads = (pd.to_numeric(tp_rows["true_L"], errors="coerce")
                 - pd.to_numeric(tp_rows["alert_tstar"], errors="coerce")).dropna().values
        breakdown.append({
            "cluster_id": int(c), "n": n, "TP": tp, "FP": fp,
            "TP_share": (tp / max(n, 1)),
            "n_event_lead": int(len(leads)),
            "lead_mean": float(leads.mean()) if len(leads) else float("nan"),
            "lead_median": float(np.median(leads)) if len(leads) else float("nan"),
        })
    return {
        "pca_explained_ratio": [float(x) for x in pca.explained_variance_ratio_],
        "cluster_breakdown": breakdown,
        "cluster_df": alerted[["site", "year", "cls", "cluster_id", "pc1", "pc2", "pc3"]]
                          if "pc3" in alerted.columns else
                          alerted[["site", "year", "cls", "cluster_id", "pc1", "pc2"]],
    }


def _print_per_feat(label: str, per: dict) -> None:
    if not per:
        print(f"    ({label}: no separable features)")
        return
    print(f"    {label}:")
    print(f"      {'feature':>44}  {'AUC':>6} {'KS':>6} {'p_KS':>9}")
    for feat, m in sorted(per.items(), key=lambda x: -x[1]["AUC_dir_agnostic"]):
        print(f"      {feat:>44}  {m['AUC_dir_agnostic']:>6.3f} {m['KS']:>6.3f} {m['p_KS']:>9.2e}")


def _print_mvar(label: str, mvar: dict | None) -> None:
    if mvar is None:
        print(f"    {label}: (skipped — too few samples)")
        return
    print(f"    {label}: in-sample AUC={mvar['in_sample_AUC']:.3f}  "
          f"cv AUC={mvar['cv_AUC_mean']:.3f} ± {mvar['cv_AUC_std']:.3f}  "
          f"(folds={mvar['n_cv_folds']})")
    coefs = sorted(mvar["coefs"].items(), key=lambda x: -abs(x[1]))
    for f, c in coefs[:5]:
        print(f"      {f:>44}: {c:+.3f}")


def serialize_mvar(mvar: dict | None) -> dict | None:
    if mvar is None:
        return None
    return {
        "feats": mvar["feats"],
        "in_sample_AUC": mvar["in_sample_AUC"],
        "cv_AUC_mean": mvar["cv_AUC_mean"],
        "cv_AUC_std": mvar["cv_AUC_std"],
        "n_cv_folds": mvar["n_cv_folds"],
        "coefs": mvar["coefs"],
    }


def run_split(probs_df: pd.DataFrame, split_name: str, tau: float, k: int,
              doy_start: int, target_recall: float, out_dir: Path) -> dict:
    print(f"\n========== {split_name} ==========")
    pct_table = build_doy_percentile_table(probs_df, doy_start)
    records = compute_motifs(probs_df, tau, k, doy_start, pct_table)
    n_event = int((records.is_event == 1).sum())
    n_nonevent = int((records.is_event == 0).sum())
    n_alerted = int((records.alerted == 1).sum())
    print(f"  records: {len(records)}  n_event={n_event}  n_nonevent={n_nonevent}  n_alerted={n_alerted}")
    # save records (drop _seq28)
    records_csv = records.drop(columns=["_seq28"]).copy()
    records_csv.to_csv(out_dir / f"motif_records_{split_name}.csv", index=False)
    print(f"  [saved] {out_dir / f'motif_records_{split_name}.csv'}")

    print(f"\n  --- per-feature TP/FP AUC + KS ---")
    causal_per = per_feature_tpfp(records, CAUSAL_FEATS)
    oracle_per = per_feature_tpfp(records, ORACLE_FEATS)
    _print_per_feat("causal", causal_per)
    _print_per_feat("oracle", oracle_per)

    print(f"\n  --- multivariate LogReg AUC ---")
    causal_mvar = multivar_logreg(records, CAUSAL_FEATS)
    oracle_mvar = multivar_logreg(records, ORACLE_FEATS)
    combined_mvar = multivar_logreg(records, CAUSAL_FEATS + ORACLE_FEATS)
    _print_mvar("causal only", causal_mvar)
    _print_mvar("oracle only [diagnostic]", oracle_mvar)
    _print_mvar("combined [diagnostic]", combined_mvar)

    print(f"\n  --- recall-preserving FAR sweep (causal-only secondary filter) ---")
    recall_sweep = recall_preserving_sweep(records, causal_mvar, target_recall)
    if recall_sweep is None:
        print("    (skipped)")
    else:
        b = recall_sweep["base"]
        print(f"    score_source={recall_sweep['score_source']}  "
              f"target_recall={target_recall:.2f}")
        print(f"    base (no secondary filter):  TP={b['TP']}  FP={b['FP']}  "
              f"recall={b['recall']:.3f}  FAR={b['FAR']:.3f}")
        bp = recall_sweep["best_at_target"]
        if bp is None:
            print(f"    no threshold meets recall>={target_recall:.2f}")
        else:
            print(f"    best @ recall>={target_recall:.2f}: thr={bp['thr']:.3f}  "
                  f"TP={bp['TP']}  FP={bp['FP']}  recall={bp['recall']:.3f}  FAR={bp['FAR']:.3f}")
            d_recall = bp["recall"] - b["recall"]; d_far = bp["FAR"] - b["FAR"]
            print(f"    delta:  dRecall={d_recall:+.3f}  dFAR={d_far:+.3f}")

    print(f"\n  --- PCA(3) + KMeans(4) cluster breakdown (alerted, 28d seq) ---")
    cl_info = pca_kmeans_clustering(records, n_pca=3, n_clusters=4)
    if cl_info is None:
        print("    (skipped)")
    else:
        print(f"    pca explained var: " +
              "  ".join(f"PC{i+1}={r:.3f}" for i, r in enumerate(cl_info["pca_explained_ratio"])))
        print(f"    {'cluster':>7}  {'n':>4}  {'TP':>3} {'FP':>3} {'TP_share':>9}  "
              f"{'lead_mean':>9} {'lead_median':>11}")
        for cb in cl_info["cluster_breakdown"]:
            print(f"    {cb['cluster_id']:>7d}  {cb['n']:>4d}  {cb['TP']:>3d} {cb['FP']:>3d} "
                  f"{cb['TP_share']:>9.3f}  {cb['lead_mean']:>9.1f} {cb['lead_median']:>11.1f}")
        cl_info["cluster_df"].to_csv(out_dir / f"motif_clusters_{split_name}.csv", index=False)

    return {
        "n_records": int(len(records)), "n_event": n_event, "n_nonevent": n_nonevent,
        "n_alerted": n_alerted,
        "per_feature": {"causal": causal_per, "oracle": oracle_per},
        "multivar": {"causal": serialize_mvar(causal_mvar),
                     "oracle": serialize_mvar(oracle_mvar),
                     "combined": serialize_mvar(combined_mvar)},
        "recall_sweep": ({"base": recall_sweep["base"],
                           "best_at_target": recall_sweep["best_at_target"],
                           "target_recall": recall_sweep["target_recall"],
                           "score_source": recall_sweep["score_source"]}
                          if recall_sweep else None),
        "cluster": ({"pca_explained_ratio": cl_info["pca_explained_ratio"],
                     "cluster_breakdown": cl_info["cluster_breakdown"]}
                    if cl_info else None),
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
    ap.add_argument("--target_recall", type=float, default=0.90,
                    help="recall floor for the secondary-filter sweep")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or Path(args.stage1_ckpt).stem
    print(f"\n========== Phase T8b motif diagnostic :: {label} ==========")
    print(f"[cfg] tau={args.tau}  k={args.k}  target_recall={args.target_recall}")

    from rice.scripts.phase_t_lead_aware_eval import build_probs
    cache = build_probs(args)
    doy_start = int(C.DOY_START)
    print(f"[DOY] DOY_START={doy_start} DOY_END={int(C.DOY_END)}")

    out = {"label": label, "args": {"tau": args.tau, "k": args.k,
                                    "target_recall": args.target_recall},
           "ckpt_meta": cache["ckpt_meta"], "splits": {}}
    for split_name, probs_df in [("val", cache["val_df"]), ("test", cache["test_df"])]:
        out["splits"][split_name] = run_split(
            probs_df, split_name, args.tau, args.k, doy_start, args.target_recall, out_dir)

    (out_dir / f"motif_diag_summary_{label}.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {out_dir / f'motif_diag_summary_{label}.json'}")


if __name__ == "__main__":
    main()
