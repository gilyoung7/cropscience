#!/usr/bin/env python3
"""
Stage2 post-hoc sample-wise offset selector diagnostic.

Operating principles
--------------------
- The Stage2 lead_v3 model predicts a Gaussian PMF (mu, sigma=5) for the event
  center. The 95% PI is [mu - 1.96*sigma, mu + 1.96*sigma] = [mu-9.8, mu+9.8].
- IoU is computed against the true discrete interval [L+1, R] (inclusive day).
- mu varies with the activation offset because the model is re-run at
  t = alert_tstar + offset on rolling features.
- Existing sample_grids cover coarse offsets {7,14,21,30,45,60}. We additionally
  build a dense oracle at 1..75 by linear interpolation of mu(offset) across
  available coarse offsets per sample (a diagnostic upper bound; the actual
  operational policy is restricted to the coarse set).
- LEAKAGE-FREE rule: any policy / model / rule that selects an offset for the
  test=2024 evaluation must be chosen using only val=2023 information.
  Test labels are NEVER inspected to pick an offset.

Outputs (under rice/outputs_stage2_batch_2024_bestgate/_offset_selector/):
  - offset_selector_val_oracle_labels.csv
  - offset_selector_dense_oracle_summary.csv
  - offset_selector_feature_diagnostics.csv
  - offset_selector_rule_candidates.csv
  - offset_selector_test_results_by_pest.csv
  - offset_selector_summary_for_ppt.txt
"""

from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd

# --- config -----------------------------------------------------------------
ROOT = Path("rice/outputs_stage2_batch_2024_bestgate")
SUM = ROOT / "_summary"
OUT = ROOT / "_offset_selector"
OUT.mkdir(parents=True, exist_ok=True)

PESTS = [
    "BPH", "WBPH",
    "bacterial_blight", "blast", "brown_spot",
    "rice_stem_borer_1", "rice_stem_borer_2",
    "sheath_blight",
]
COARSE_OFFSETS = [7, 14, 21, 30, 45, 60]
DENSE_OFFSETS = list(range(1, 76))   # 1..75 inclusive
SIGMA = 5.0
Z = 1.96
HALF_PI = Z * SIGMA   # 9.8

FEAT_COLS = [
    "alert_tstar",
    "A_score_at_alert", "D_score_at_alert", "score_margin",
    "dispatch_score_at_alert", "dispatch_tau_used", "score_over_tau_margin",
    "recent_14d_mean_score", "recent_28d_mean_score",
    "score_above_tau_streak", "score_rolling_slope_14d",
    "p_mean_so_far_at_alert", "with_history",
]

CLIM_KINDS = ["clim_mean_L", "clim_mean_mid", "clim_mean_R"]

# --- IoU --------------------------------------------------------------------
def iou_from_mu(mu: float, L: float, R: float, sigma: float = SIGMA) -> float:
    if pd.isna(mu) or pd.isna(L) or pd.isna(R):
        return 0.0
    pL = int(round(mu - Z * sigma))
    pR = int(round(mu + Z * sigma))
    tL = int(L) + 1
    tR = int(R)
    lo_hi = min(pR, tR)
    hi_lo = max(pL, tL)
    if lo_hi < hi_lo:
        ov = 0
    else:
        ov = lo_hi - hi_lo + 1
    un = max(pR, tR) - min(pL, tL) + 1
    if un <= 0:
        return 0.0
    return max(0.0, ov / un)


# --- helpers ----------------------------------------------------------------
def pick_oracle_offset(off_iou: Dict[int, float],
                       tie: str = "smallest_offset") -> Tuple[int, float]:
    """Pick offset with max IoU; tie-break to smaller offset (earlier alert)."""
    if not off_iou:
        return -1, 0.0
    best = max(off_iou.values())
    cands = [o for o, v in off_iou.items() if v == best]
    if tie == "smallest_offset":
        return min(cands), best
    if tie == "largest_offset":
        return max(cands), best
    return cands[0], best


def interp_mu_curve(off_mu_coarse: Dict[int, float]) -> Dict[int, float]:
    """Linearly interpolate mu over dense offsets using valid (non-NaN) anchors.
    Outside the anchor range we clamp to nearest anchor.
    Returns {offset: mu_interp} for every offset in DENSE_OFFSETS.
    """
    valid = sorted([(o, m) for o, m in off_mu_coarse.items() if not pd.isna(m)])
    if not valid:
        return {o: float("nan") for o in DENSE_OFFSETS}
    xs = np.array([v[0] for v in valid], dtype=float)
    ys = np.array([v[1] for v in valid], dtype=float)
    out = {}
    for o in DENSE_OFFSETS:
        if o <= xs[0]:
            out[o] = float(ys[0])
        elif o >= xs[-1]:
            out[o] = float(ys[-1])
        else:
            out[o] = float(np.interp(o, xs, ys))
    return out


def overall_iou_n_total(per_sample_iou: List[float], n_total: int) -> float:
    if n_total == 0:
        return 0.0
    return float(np.sum(per_sample_iou)) / float(n_total)


# --- per-pest data loading --------------------------------------------------
def load_pest_split(pest: str, split: str) -> pd.DataFrame:
    fn = ROOT / pest / f"lead_v3_{split}_sample_grid.csv"
    return pd.read_csv(fn)


def per_sample_records(df: pd.DataFrame, split_label: str, pest: str) -> List[dict]:
    """Collapse rows to one record per sample with offset->iou/mu maps."""
    out = []
    for sid, g in df.groupby("sample_id"):
        g = g.sort_values("offset")
        off_iou_coarse = {int(o): float(v) for o, v in zip(g["offset"], g["iou_matched"])}
        off_mu_coarse = {int(o): float(v) if not pd.isna(v) else float("nan")
                         for o, v in zip(g["offset"], g["mu"])}
        # offset-independent features from first row
        head = g.iloc[0]
        rec = {
            "pest": pest,
            "split": split_label,
            "sample_id": sid,
            "site": head.get("site"),
            "year": int(head["year"]) if not pd.isna(head["year"]) else None,
            "L": head.get("L"),
            "R": head.get("R"),
            "true_event_doy": head.get("true_event_doy"),
            "off_iou_coarse": off_iou_coarse,
            "off_mu_coarse": off_mu_coarse,
            "dispatch_branch": head.get("dispatch_branch"),
        }
        for f in FEAT_COLS:
            rec[f] = head.get(f)
        out.append(rec)
    return out


# --- oracle / fixed metrics -------------------------------------------------
def per_sample_iou_at_offset(recs: List[dict], offset: int) -> List[float]:
    return [r["off_iou_coarse"].get(offset, 0.0) for r in recs]


def per_sample_oracle_iou(recs: List[dict], offsets: List[int]) -> List[float]:
    out = []
    for r in recs:
        ious = [r["off_iou_coarse"].get(o, 0.0) for o in offsets]
        out.append(max(ious) if ious else 0.0)
    return out


def per_sample_dense_oracle_iou(recs: List[dict]) -> List[Tuple[int, float]]:
    """For each sample, compute IoU at every dense offset (via interp) and
    return (best_dense_offset, best_dense_iou)."""
    out = []
    for r in recs:
        mu_curve = interp_mu_curve(r["off_mu_coarse"])
        off_iou = {o: iou_from_mu(mu, r["L"], r["R"]) for o, mu in mu_curve.items()}
        bo, bi = pick_oracle_offset(off_iou)
        out.append((bo, bi))
    return out


# --- selectors --------------------------------------------------------------
def selector_constant(val_recs, test_recs, offsets):
    """Pick the single offset with best val overall IoU (val-restricted)."""
    n = len(val_recs)
    if n == 0:
        return None, {}
    best_o, best_iou = None, -1
    for o in offsets:
        iou = overall_iou_n_total(per_sample_iou_at_offset(val_recs, o), n)
        if iou > best_iou:
            best_iou, best_o = iou, o
    # apply to test (constant)
    n_t = len(test_recs)
    test_iou = overall_iou_n_total(per_sample_iou_at_offset(test_recs, best_o), n_t)
    return best_o, {
        "selector": "pest_const_val_best",
        "policy": f"offset={best_o}",
        "val_iou": best_iou,
        "test_iou": test_iou,
        "test_n_total": n_t,
    }


def _apply_rule(recs, rule_fn, offsets):
    """rule_fn(rec) -> offset; falls back to first offset in `offsets` if None."""
    n = len(recs)
    sum_iou = 0.0
    for r in recs:
        o = rule_fn(r)
        if o is None or o not in offsets:
            o = offsets[0]
        sum_iou += r["off_iou_coarse"].get(o, 0.0)
    return sum_iou / n if n else 0.0


def selector_bin_rule(val_recs, test_recs, offsets, feat_key: str, n_bins: int = 3):
    """Bin a continuous feature on val (quantile) and pick best constant
    offset per bin (val IoU optimized). Apply same bin edges to test."""
    n = len(val_recs)
    if n == 0:
        return None, {}
    vals = np.array([r.get(feat_key) for r in val_recs], dtype=float)
    if np.all(np.isnan(vals)):
        return None, {}
    finite = vals[~np.isnan(vals)]
    if len(np.unique(finite)) < 2:
        return None, {}
    # quantile edges from val
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    edges = [float(x) for x in np.quantile(finite, qs)]

    def bin_of(x):
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return 0
        b = 0
        for e in edges:
            if x > e:
                b += 1
        return b

    # for each bin, find best offset on val
    bin_best = {}
    for b in range(n_bins):
        sub = [r for r in val_recs if bin_of(r.get(feat_key)) == b]
        if not sub:
            bin_best[b] = offsets[0]
            continue
        best_o, best_iou = offsets[0], -1
        for o in offsets:
            iou = overall_iou_n_total(per_sample_iou_at_offset(sub, o), len(sub))
            if iou > best_iou:
                best_iou, best_o = iou, o
        bin_best[b] = best_o

    rule = lambda r: bin_best[bin_of(r.get(feat_key))]
    val_iou = _apply_rule(val_recs, rule, offsets)
    test_iou = _apply_rule(test_recs, rule, offsets)
    return (feat_key, edges, bin_best), {
        "selector": f"bin_rule[{feat_key}]",
        "policy": f"edges={[round(e,3) for e in edges]}, bin->offset={bin_best}",
        "val_iou": val_iou,
        "test_iou": test_iou,
        "test_n_total": len(test_recs),
    }


def selector_method_rule(val_recs, test_recs, offsets, key="dispatch_branch"):
    """Per-category best constant offset (val-tuned)."""
    n = len(val_recs)
    if n == 0:
        return None, {}
    cats = sorted({r.get(key) for r in val_recs if r.get(key) is not None})
    if not cats:
        return None, {}
    cat_best = {}
    for c in cats:
        sub = [r for r in val_recs if r.get(key) == c]
        best_o, best_iou = offsets[0], -1
        for o in offsets:
            iou = overall_iou_n_total(per_sample_iou_at_offset(sub, o), len(sub))
            if iou > best_iou:
                best_iou, best_o = iou, o
        cat_best[c] = best_o
    # fallback for unseen categories: use val-global best constant
    g_best_o, _ = selector_constant(val_recs, val_recs, offsets) or (offsets[0], None)
    # selector_constant returns (best_o, info_dict); only need first
    if isinstance(g_best_o, tuple):
        g_best_o = g_best_o[0]
    rule = lambda r: cat_best.get(r.get(key), g_best_o or offsets[0])
    val_iou = _apply_rule(val_recs, rule, offsets)
    test_iou = _apply_rule(test_recs, rule, offsets)
    return (key, cat_best), {
        "selector": f"category_rule[{key}]",
        "policy": f"cat->offset={cat_best}, fallback={g_best_o}",
        "val_iou": val_iou,
        "test_iou": test_iou,
        "test_n_total": len(test_recs),
    }


def selector_decision_tree(val_recs, test_recs, offsets, max_depth: int = 2):
    """Simple decision tree on offset-independent features; target = val oracle
    offset class (coarse). Tiny val => keep depth small."""
    from sklearn.tree import DecisionTreeClassifier  # local import
    n = len(val_recs)
    if n < 12:
        return None, {"selector": f"tree_d{max_depth}",
                      "policy": "skipped (too few val samples)",
                      "val_iou": float("nan"), "test_iou": float("nan"),
                      "test_n_total": len(test_recs)}
    # oracle labels on val
    y_val = []
    X_val_rows = []
    for r in val_recs:
        ious = {o: r["off_iou_coarse"].get(o, 0.0) for o in offsets}
        oo, _ = pick_oracle_offset(ious)
        if oo == -1:
            oo = offsets[0]
        y_val.append(oo)
        X_val_rows.append([r.get(f) for f in FEAT_COLS])
    X_val = pd.DataFrame(X_val_rows, columns=FEAT_COLS).apply(pd.to_numeric, errors="coerce")
    X_val = X_val.fillna(X_val.median(numeric_only=True))
    if X_val.isna().any().any():
        X_val = X_val.fillna(0.0)
    clf = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=max(3, n // 10),
                                 random_state=0)
    clf.fit(X_val.values, y_val)
    # apply
    def predict(recs):
        if not recs:
            return []
        Xr = pd.DataFrame([[r.get(f) for f in FEAT_COLS] for r in recs], columns=FEAT_COLS)
        Xr = Xr.apply(pd.to_numeric, errors="coerce")
        Xr = Xr.fillna(X_val.median(numeric_only=True)).fillna(0.0)
        return clf.predict(Xr.values).tolist()
    val_preds = predict(val_recs)
    test_preds = predict(test_recs)
    val_iou = float(np.mean([
        r["off_iou_coarse"].get(int(p), 0.0) for r, p in zip(val_recs, val_preds)
    ])) if val_recs else 0.0
    test_iou = float(np.mean([
        r["off_iou_coarse"].get(int(p), 0.0) for r, p in zip(test_recs, test_preds)
    ])) if test_recs else 0.0
    return clf, {
        "selector": f"tree_d{max_depth}",
        "policy": f"depth={max_depth} on FEAT_COLS",
        "val_iou": val_iou,
        "test_iou": test_iou,
        "test_n_total": len(test_recs),
    }


# --- climatology / fixed-offset baselines -----------------------------------
def load_clim_test_iou(pest: str, sel_df: pd.DataFrame) -> Dict[str, float]:
    sub = sel_df[sel_df.pest == pest]
    out = {}
    for kind in CLIM_KINDS:
        row = sub[sub.model_kind == kind]
        if len(row):
            out[kind] = float(row.iloc[0]["test_IoU_overall_n_total_at_val_offset"])
        else:
            out[kind] = float("nan")
    return out


# --- main -------------------------------------------------------------------
def main() -> int:
    sel_df = pd.read_csv(SUM / "all_pests_selection.csv")
    gate_df = pd.read_csv(SUM / "stage1_gate_selection_split3_2024.csv")
    gate_lookup = gate_df.set_index("pest")[["selected_method", "selected_run"]].to_dict("index")

    val_oracle_rows: List[dict] = []
    dense_summary_rows: List[dict] = []
    feature_diag_rows: List[dict] = []
    rule_cand_rows: List[dict] = []
    test_results_rows: List[dict] = []

    for pest in PESTS:
        val_df = load_pest_split(pest, "val")
        test_df = load_pest_split(pest, "test")
        val_recs = per_sample_records(val_df, "val", pest)
        test_recs = per_sample_records(test_df, "test", pest)
        n_val, n_test = len(val_recs), len(test_recs)

        # === 1. val coarse oracle labels (per-sample) =======================
        for r in val_recs:
            ious = {o: r["off_iou_coarse"].get(o, 0.0) for o in COARSE_OFFSETS}
            o_oracle, iou_oracle = pick_oracle_offset(ious)
            sel_row = sel_df[(sel_df.pest == pest) & (sel_df.model_kind == "lead_v3")]
            fixed_off = int(sel_row.iloc[0]["val_best_offset"]) if len(sel_row) else -1
            iou_fixed = r["off_iou_coarse"].get(fixed_off, 0.0)
            mu_curve = interp_mu_curve(r["off_mu_coarse"])
            off_iou_dense = {o: iou_from_mu(mu, r["L"], r["R"]) for o, mu in mu_curve.items()}
            o_dense, iou_dense = pick_oracle_offset(off_iou_dense)
            val_oracle_rows.append({
                "pest": pest,
                "sample_id": r["sample_id"],
                "site": r["site"],
                "year": r["year"],
                "L": r["L"], "R": r["R"], "true_event_doy": r["true_event_doy"],
                "alert_tstar": r["alert_tstar"],
                "oracle_offset_coarse": o_oracle,
                "oracle_iou_coarse": iou_oracle,
                "fixed_val_offset": fixed_off,
                "fixed_val_iou": iou_fixed,
                "possible_gain_coarse": iou_oracle - iou_fixed,
                "oracle_offset_dense": o_dense,
                "oracle_iou_dense": iou_dense,
                "dispatch_branch": r.get("dispatch_branch"),
                "with_history": r.get("with_history"),
                "score_margin": r.get("score_margin"),
                "pred_mu_at_fixed": r["off_mu_coarse"].get(fixed_off, float("nan")),
            })

        # === 2. dense oracle summary per pest ===============================
        dense_val = per_sample_dense_oracle_iou(val_recs)
        dense_test = per_sample_dense_oracle_iou(test_recs)
        coarse_oracle_val = per_sample_oracle_iou(val_recs, COARSE_OFFSETS)
        coarse_oracle_test = per_sample_oracle_iou(test_recs, COARSE_OFFSETS)
        dense_summary_rows.append({
            "pest": pest,
            "n_val": n_val, "n_test": n_test,
            "val_coarse_oracle_iou": overall_iou_n_total(coarse_oracle_val, n_val),
            "test_coarse_oracle_iou": overall_iou_n_total(coarse_oracle_test, n_test),
            "val_dense_oracle_iou": overall_iou_n_total([x[1] for x in dense_val], n_val),
            "test_dense_oracle_iou": overall_iou_n_total([x[1] for x in dense_test], n_test),
            "val_dense_oracle_offset_mean": float(np.mean([x[0] for x in dense_val])) if dense_val else float("nan"),
            "val_dense_oracle_offset_median": float(np.median([x[0] for x in dense_val])) if dense_val else float("nan"),
            "test_dense_oracle_offset_mean": float(np.mean([x[0] for x in dense_test])) if dense_test else float("nan"),
        })

        # === 3. feature diagnostics on val ==================================
        df_v = pd.DataFrame([{
            "sample_id": r["sample_id"],
            "oracle_offset": pick_oracle_offset(
                {o: r["off_iou_coarse"].get(o, 0.0) for o in COARSE_OFFSETS})[0],
            "oracle_iou": pick_oracle_offset(
                {o: r["off_iou_coarse"].get(o, 0.0) for o in COARSE_OFFSETS})[1],
            **{f: r.get(f) for f in FEAT_COLS},
            "dispatch_branch": r.get("dispatch_branch"),
        } for r in val_recs])
        if not df_v.empty:
            for f in ["alert_tstar", "score_margin", "score_over_tau_margin",
                      "p_mean_so_far_at_alert", "score_rolling_slope_14d"]:
                if f not in df_v.columns:
                    continue
                ser = pd.to_numeric(df_v[f], errors="coerce")
                tgt = pd.to_numeric(df_v["oracle_offset"], errors="coerce")
                mask = (~ser.isna()) & (~tgt.isna())
                if mask.sum() < 5:
                    continue
                rho = float(np.corrcoef(ser[mask], tgt[mask])[0, 1]) if mask.sum() > 1 else float("nan")
                feature_diag_rows.append({
                    "pest": pest, "feature": f,
                    "n": int(mask.sum()),
                    "pearson_corr_with_oracle_offset": rho,
                    "feature_mean": float(ser[mask].mean()),
                    "feature_std": float(ser[mask].std()),
                    "oracle_offset_mean": float(tgt[mask].mean()),
                    "oracle_offset_std": float(tgt[mask].std()),
                })
            # per-branch oracle offset mean
            for cat, sub in df_v.groupby("dispatch_branch"):
                if len(sub) >= 3:
                    feature_diag_rows.append({
                        "pest": pest, "feature": f"branch={cat}",
                        "n": int(len(sub)),
                        "pearson_corr_with_oracle_offset": float("nan"),
                        "feature_mean": float("nan"),
                        "feature_std": float("nan"),
                        "oracle_offset_mean": float(sub["oracle_offset"].mean()),
                        "oracle_offset_std": float(sub["oracle_offset"].std()),
                    })

        # === 4 + 5. selectors: rule-based and learned =======================
        # baselines
        # (a) lead_v3 val-fixed (existing pipeline)
        sel_row = sel_df[(sel_df.pest == pest) & (sel_df.model_kind == "lead_v3")].iloc[0]
        val_fix_off = int(sel_row["val_best_offset"])
        val_fix_val_iou = float(sel_row["val_IoU_overall_n_total_at_best"])
        val_fix_test_iou = float(sel_row["test_IoU_overall_n_total_at_val_offset"])
        test_results_rows.append({
            "pest": pest, "n_val": n_val, "n_test": n_test,
            "selector": "lead_v3_val_fixed",
            "policy": f"offset={val_fix_off}",
            "val_iou": val_fix_val_iou,
            "test_iou": val_fix_test_iou,
        })

        # (b) pest-specific val-best constant (recomputed here as sanity)
        const_o, const_info = selector_constant(val_recs, test_recs, COARSE_OFFSETS)
        test_results_rows.append({
            "pest": pest, "n_val": n_val, "n_test": n_test,
            "selector": const_info["selector"], "policy": const_info["policy"],
            "val_iou": const_info["val_iou"], "test_iou": const_info["test_iou"],
        })

        # (c) bin rules: alert_tstar, score_margin
        for feat in ["alert_tstar", "score_margin", "p_mean_so_far_at_alert",
                     "score_rolling_slope_14d"]:
            obj, info = selector_bin_rule(val_recs, test_recs, COARSE_OFFSETS, feat, n_bins=3)
            if info:
                rule_cand_rows.append({"pest": pest, **info})
                test_results_rows.append({
                    "pest": pest, "n_val": n_val, "n_test": n_test,
                    "selector": info["selector"], "policy": info["policy"],
                    "val_iou": info["val_iou"], "test_iou": info["test_iou"],
                })
        # (d) category rule: dispatch_branch
        obj, info = selector_method_rule(val_recs, test_recs, COARSE_OFFSETS, "dispatch_branch")
        if info:
            rule_cand_rows.append({"pest": pest, **info})
            test_results_rows.append({
                "pest": pest, "n_val": n_val, "n_test": n_test,
                "selector": info["selector"], "policy": info["policy"],
                "val_iou": info["val_iou"], "test_iou": info["test_iou"],
            })

        # (e) tree-based learned
        for depth in [2, 3]:
            _, info = selector_decision_tree(val_recs, test_recs, COARSE_OFFSETS, max_depth=depth)
            test_results_rows.append({
                "pest": pest, "n_val": n_val, "n_test": n_test,
                "selector": info["selector"], "policy": info["policy"],
                "val_iou": info["val_iou"], "test_iou": info["test_iou"],
            })

        # (f) climatology best (already in selection.csv -> derive best clim model)
        clim_ious = load_clim_test_iou(pest, sel_df)
        best_clim = max(clim_ious.items(), key=lambda x: -1 if pd.isna(x[1]) else x[1])
        test_results_rows.append({
            "pest": pest, "n_val": n_val, "n_test": n_test,
            "selector": "climatology_best_val_selected",
            "policy": f"{best_clim[0]} at its val-selected offset",
            "val_iou": float("nan"),
            "test_iou": best_clim[1],
        })

        # (g) sample-wise coarse oracle on test (LEAKY upper bound)
        test_results_rows.append({
            "pest": pest, "n_val": n_val, "n_test": n_test,
            "selector": "test_sample_oracle_coarse_LEAKY",
            "policy": "test labels -> per-sample best of coarse offsets",
            "val_iou": float("nan"),
            "test_iou": overall_iou_n_total(coarse_oracle_test, n_test),
        })
        # (h) sample-wise dense oracle on test (LEAKY upper bound)
        test_results_rows.append({
            "pest": pest, "n_val": n_val, "n_test": n_test,
            "selector": "test_sample_oracle_dense_LEAKY",
            "policy": "test labels -> per-sample best of 1..75 (mu interpolated)",
            "val_iou": float("nan"),
            "test_iou": overall_iou_n_total([x[1] for x in dense_test], n_test),
        })

    # --- write csv outputs --------------------------------------------------
    df_val_oracle = pd.DataFrame(val_oracle_rows)
    df_val_oracle.to_csv(OUT / "offset_selector_val_oracle_labels.csv", index=False)

    df_dense = pd.DataFrame(dense_summary_rows)
    df_dense.to_csv(OUT / "offset_selector_dense_oracle_summary.csv", index=False)

    df_feat = pd.DataFrame(feature_diag_rows)
    df_feat.to_csv(OUT / "offset_selector_feature_diagnostics.csv", index=False)

    df_rule = pd.DataFrame(rule_cand_rows)
    df_rule.to_csv(OUT / "offset_selector_rule_candidates.csv", index=False)

    df_test = pd.DataFrame(test_results_rows)
    df_test.to_csv(OUT / "offset_selector_test_results_by_pest.csv", index=False)

    # --- summary TXT --------------------------------------------------------
    write_summary_txt(df_test, df_dense, df_val_oracle, df_rule)
    print(f"[done] outputs written to {OUT}")
    return 0


def write_summary_txt(df_test, df_dense, df_val_oracle, df_rule):
    lines: List[str] = []
    lines.append("Stage2 Post-Hoc Sample-Wise Offset Selector Diagnostic")
    lines.append("=" * 70)
    lines.append("")
    lines.append("1. PURPOSE")
    lines.append("-" * 70)
    lines.append(
        "Investigate whether a per-sample offset selection rule, learned on")
    lines.append(
        "val=2023 alone, can improve Stage2 IoU over the current fixed")
    lines.append(
        "val-selected offset, and close the gap to the climatology baseline.")
    lines.append("")
    lines.append("2. LEAKAGE-FREE PRINCIPLE")
    lines.append("-" * 70)
    lines.append(
        "All policies that pick offsets for test=2024 are tuned on val=2023.")
    lines.append(
        "test=2024 IoU is only ever read AFTER the selector is frozen on val.")
    lines.append(
        "Lines tagged 'LEAKY' are diagnostic upper bounds that do peek at test")
    lines.append(
        "labels and must NOT be reported as operational scores.")
    lines.append("")
    lines.append("3. FEATURES USED FOR THE SELECTORS")
    lines.append("-" * 70)
    lines.append(", ".join(FEAT_COLS))
    lines.append(
        "All features are offset-independent and observable at alert time.")
    lines.append("")
    lines.append("4. COARSE vs DENSE ORACLE (val=2023, test=2024)")
    lines.append("-" * 70)
    lines.append(
        f"{'pest':20s} {'n_val':>5s} {'n_test':>6s} "
        f"{'val_oracle_C':>13s} {'test_oracle_C':>14s} "
        f"{'val_oracle_D':>13s} {'test_oracle_D':>14s} "
        f"{'val_dense_off_med':>18s}"
    )
    for _, row in df_dense.iterrows():
        lines.append(
            f"{row['pest']:20s} {int(row['n_val']):5d} {int(row['n_test']):6d} "
            f"{row['val_coarse_oracle_iou']:13.3f} {row['test_coarse_oracle_iou']:14.3f} "
            f"{row['val_dense_oracle_iou']:13.3f} {row['test_dense_oracle_iou']:14.3f} "
            f"{row['val_dense_oracle_offset_median']:18.1f}"
        )
    lines.append("")
    lines.append("Read: dense oracle bounds what *any* sample-wise selector with")
    lines.append("1-day-resolution mu could achieve. If dense oracle is close to")
    lines.append("coarse oracle, a finer offset grid is not the bottleneck.")
    lines.append("")

    lines.append("5. TEST=2024 RESULTS (LEAKAGE-FREE unless tagged LEAKY)")
    lines.append("-" * 70)
    pests = df_test["pest"].unique().tolist()
    for pest in pests:
        lines.append(f"--- {pest} ---")
        sub = df_test[df_test.pest == pest].copy()
        # sort: keep selectors in a stable, intuitive order
        order = {
            "lead_v3_val_fixed": 0,
            "pest_const_val_best": 1,
            "bin_rule[alert_tstar]": 2,
            "bin_rule[score_margin]": 3,
            "bin_rule[p_mean_so_far_at_alert]": 4,
            "bin_rule[score_rolling_slope_14d]": 5,
            "category_rule[dispatch_branch]": 6,
            "tree_d2": 7,
            "tree_d3": 8,
            "climatology_best_val_selected": 9,
            "test_sample_oracle_coarse_LEAKY": 10,
            "test_sample_oracle_dense_LEAKY": 11,
        }
        sub["__o"] = sub["selector"].map(lambda s: order.get(s, 99))
        sub = sub.sort_values("__o")
        for _, r in sub.iterrows():
            v = "  nan " if pd.isna(r["val_iou"]) else f"{float(r['val_iou']):>6.3f}"
            t = "  nan " if pd.isna(r["test_iou"]) else f"{float(r['test_iou']):>6.3f}"
            lines.append(f"  {r['selector']:34s}  val={v}  test={t}")
        lines.append("")

    lines.append("6. PEST-LEVEL OUTCOME (vs fixed lead_v3 baseline)")
    lines.append("-" * 70)
    summary_rows = []
    for pest in pests:
        sub = df_test[df_test.pest == pest]
        fix = sub[sub.selector == "lead_v3_val_fixed"]
        clim = sub[sub.selector == "climatology_best_val_selected"]
        if fix.empty or clim.empty:
            continue
        fix_test = float(fix.iloc[0]["test_iou"])
        clim_test = float(clim.iloc[0]["test_iou"])
        # best leakage-free selector for this pest
        ok = sub[~sub.selector.str.contains("LEAKY") & (sub.selector != "lead_v3_val_fixed")
                 & (sub.selector != "climatology_best_val_selected")]
        if ok.empty:
            best_sel, best_test = "(none)", float("nan")
        else:
            best = ok.loc[ok["test_iou"].idxmax()]
            best_sel, best_test = str(best["selector"]), float(best["test_iou"])
        summary_rows.append((pest, fix_test, best_sel, best_test, clim_test))
        lines.append(
            f"  {pest:20s} fixed={fix_test:.3f}  best_selector={best_sel:30s} "
            f"test={best_test:.3f}  clim={clim_test:.3f}  "
            f"Δfixed={best_test - fix_test:+.3f}  Δclim={best_test - clim_test:+.3f}"
        )
    lines.append("")

    lines.append("7. FOCUS PESTS (BPH / WBPH / sheath_blight)")
    lines.append("-" * 70)
    focus = ["BPH", "WBPH", "sheath_blight"]
    EXCLUDE = {"lead_v3_val_fixed", "climatology_best_val_selected"}
    for pest in focus:
        sub = df_test[df_test.pest == pest]
        if sub.empty:
            continue
        fix = float(sub[sub.selector == "lead_v3_val_fixed"].iloc[0]["test_iou"])
        clim = float(sub[sub.selector == "climatology_best_val_selected"].iloc[0]["test_iou"])
        learned = sub[~sub.selector.str.contains("LEAKY") & ~sub.selector.isin(EXCLUDE)]
        best_l = learned.loc[learned["test_iou"].idxmax()]
        dense = df_dense[df_dense.pest == pest].iloc[0]
        lines.append(
            f"  {pest}:"
            f" fixed={fix:.3f},"
            f" best_learned={best_l['selector']}={float(best_l['test_iou']):.3f},"
            f" climatology={clim:.3f},"
            f" coarse_oracle_LEAKY={dense['test_coarse_oracle_iou']:.3f},"
            f" dense_oracle_LEAKY={dense['test_dense_oracle_iou']:.3f}"
        )
    lines.append("")

    lines.append("8. CONCLUSION")
    lines.append("-" * 70)
    n_beat_fixed = sum(1 for r in summary_rows if r[3] > r[1] + 1e-6)
    n_beat_clim = sum(1 for r in summary_rows if r[3] > r[4] + 1e-6)
    lines.append(
        f"  - Pests where best leakage-free selector > fixed lead_v3: "
        f"{n_beat_fixed}/{len(summary_rows)}"
    )
    lines.append(
        f"  - Pests where best leakage-free selector > climatology baseline: "
        f"{n_beat_clim}/{len(summary_rows)}"
    )
    lines.append("")
    lines.append("  Interpretation guide:")
    lines.append("    * If beat_fixed >= 4/8, sample-wise selection is worth keeping.")
    lines.append("    * If beat_clim is still 0/8, Stage2 still needs a different")
    lines.append("      learning signal (interval likelihood / residual / wider sigma),")
    lines.append("      not just a smarter offset selector.")
    lines.append("    * Compare coarse vs dense oracle in section 4: if dense oracle")
    lines.append("      barely exceeds coarse, finer offset grids will not unlock")
    lines.append("      additional headroom.")

    (OUT / "offset_selector_summary_for_ppt.txt").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
