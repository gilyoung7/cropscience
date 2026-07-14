#!/usr/bin/env python3
"""
Stage 2 offset selector V2 — ranking-style per-sample offset chooser.

Motivation: baseline_asym_mse for sheath_blight has test sample-wise oracle
IoU ≈ 0.433 vs best climatology ≈ 0.369, i.e. there is real headroom from
per-sample offset selection — if we can learn a selector that approaches
oracle. V1 (rule / tree on coarse {7,14,21,30,45,60}) reached ~0.27.
V2 widens the candidate space to dense offsets 1..75 (via mu interpolation)
and frames selection as a ranking problem: predict a per-candidate score and
pick argmax per sample.

Pipeline (LEAKAGE-FREE):
  1. baseline_asym_mse val sample_grid (val=2023). For each sample, linearly
     interpolate mu(offset) across valid coarse anchors {7,14,21,30,45,60}
     to build dense candidates at offsets 1..75.
  2. Compute per-candidate IoU against (L+1, R) using sigma=5. This is the
     regression / ranking target.
  3. Train 3 LightGBM models on val only:
       (a) regressor   : predict iou_at_offset (continuous)
       (b) ranker      : lambdarank, sample_id as group
       (c) classifier  : predict is_oracle (=1 if this offset = sample best)
     Hyperparameters fixed and conservative (single-year val = overfit risk).
  4. Apply each selector to the test sample_grid (test=2024). For each
     test sample, score every candidate offset via the trained model and pick
     argmax. Two policy spaces:
       * coarse {7,14,21,30,45,60}   — operationally clean (matches existing
         model emission). Primary reported metric.
       * dense 1..75                 — diagnostic upper bound (uses interp).
  5. Compare against fixed val-best offset, the V1 best learned selector
     (bin_rule[score_rolling_slope_14d] from
     phase_b_stage2_offset_selector_diagnostic.py), best climatology, and
     LEAKY sample-wise oracle on test.

NO TEST LABELS are ever used to train or select. All hyperparameters fixed.

Outputs (under <out_root>/_offset_selector_v2/):
  - v2_per_candidate_val.csv          (sample × offset rows + target on val)
  - v2_per_sample_test_selections.csv (offset/IoU chosen per sample × method)
  - v2_test_results_summary.csv       (per-method test IoU)
  - v2_summary_for_ppt.txt            (human-readable summary)
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd

SIGMA = 5.0
Z = 1.96
COARSE_OFFSETS = [7, 14, 21, 30, 45, 60]
DENSE_OFFSETS = list(range(1, 76))

# offset-INDEPENDENT features (one value per sample). Used as candidate features
# alongside the offset-dependent terms.
SAMPLE_FEATURES = [
    "alert_tstar",
    "A_score_at_alert", "D_score_at_alert", "score_margin",
    "dispatch_score_at_alert", "dispatch_tau_used", "score_over_tau_margin",
    "recent_14d_mean_score", "recent_28d_mean_score",
    "score_above_tau_streak", "score_rolling_slope_14d",
    "p_mean_so_far_at_alert", "with_history",
]


# ---------- IoU ------------------------------------------------------------
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


def interp_mu_curve(off_mu_coarse: Dict[int, float],
                    offsets: List[int]) -> Dict[int, float]:
    """Linear interpolation of mu over `offsets` using valid coarse anchors.
    Clamp to nearest anchor outside the coarse range."""
    valid = sorted([(o, m) for o, m in off_mu_coarse.items() if not pd.isna(m)])
    if not valid:
        return {o: float("nan") for o in offsets}
    xs = np.array([v[0] for v in valid], dtype=float)
    ys = np.array([v[1] for v in valid], dtype=float)
    out = {}
    for o in offsets:
        if o <= xs[0]:
            out[o] = float(ys[0])
        elif o >= xs[-1]:
            out[o] = float(ys[-1])
        else:
            out[o] = float(np.interp(o, xs, ys))
    return out


# ---------- candidate row builder ------------------------------------------
def build_candidate_rows(sample_grid_csv: Path,
                         offsets: List[int],
                         clim_mid_doy: float) -> pd.DataFrame:
    """Return long-format DataFrame (sample × offset rows) with features +
    target IoU + sample_id (group key) + is_oracle flag."""
    raw = pd.read_csv(sample_grid_csv)
    rows = []
    for sid, g in raw.groupby("sample_id"):
        g = g.sort_values("offset")
        off_mu_coarse = {int(o): float(m) if not pd.isna(m) else float("nan")
                         for o, m in zip(g["offset"], g["mu"])}
        L = float(g["L"].iloc[0])
        R = float(g["R"].iloc[0])
        head = g.iloc[0]
        # per-sample features (constant across offsets)
        sample_feats = {f: head.get(f) for f in SAMPLE_FEATURES}
        sample_feats["dispatch_branch_is_D"] = 1 if str(head.get("dispatch_branch")) == "D" else 0
        # dense mu curve via interpolation
        mu_curve = interp_mu_curve(off_mu_coarse, offsets)
        # per-candidate rows
        for o in offsets:
            mu_o = mu_curve[o]
            pred_lead = mu_o - float(sample_feats["alert_tstar"]) \
                if not pd.isna(mu_o) else float("nan")
            mu_minus_clim = mu_o - float(clim_mid_doy) \
                if not pd.isna(mu_o) else float("nan")
            tstar_minus_clim = float(sample_feats["alert_tstar"]) - float(clim_mid_doy)
            iou_o = iou_from_mu(mu_o, L, R)
            rows.append({
                "sample_id": sid,
                "L": L, "R": R,
                "offset": o,
                "pred_mu": mu_o,
                "pred_lead": pred_lead,
                "mu_minus_clim_mid": mu_minus_clim,
                "tstar_minus_clim_mid": tstar_minus_clim,
                **sample_feats,
                "iou": iou_o,
            })
    df = pd.DataFrame(rows)
    # per-sample best offset / IoU (oracle target)
    g = df.groupby("sample_id")["iou"].transform("max")
    df["is_oracle"] = (df["iou"] >= g - 1e-9).astype(int)
    df["sample_best_iou"] = g
    return df


# ---------- selector training ----------------------------------------------
FEATURE_COLS = [
    "offset", "pred_mu", "pred_lead", "mu_minus_clim_mid", "tstar_minus_clim_mid",
    "alert_tstar", "A_score_at_alert", "D_score_at_alert", "score_margin",
    "dispatch_score_at_alert", "dispatch_tau_used", "score_over_tau_margin",
    "recent_14d_mean_score", "recent_28d_mean_score",
    "score_above_tau_streak", "score_rolling_slope_14d",
    "p_mean_so_far_at_alert", "with_history", "dispatch_branch_is_D",
]


def _prep_X(df: pd.DataFrame) -> pd.DataFrame:
    X = df[FEATURE_COLS].apply(pd.to_numeric, errors="coerce")
    return X.fillna(X.median(numeric_only=True)).fillna(0.0)


def train_lgb_regressor(val_df: pd.DataFrame, seed: int = 0):
    """Predict iou per (sample, offset). Higher predicted iou → pick that
    offset. Robust simple baseline."""
    import lightgbm as lgb
    X = _prep_X(val_df).values
    y = val_df["iou"].values
    model = lgb.LGBMRegressor(
        n_estimators=120, learning_rate=0.05, num_leaves=15,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.9, reg_lambda=1.0, random_state=seed, verbosity=-1,
    )
    model.fit(X, y)
    return model


def train_lgb_ranker(val_df: pd.DataFrame, seed: int = 0):
    """LambdaRank: per-sample ranking. Group = sample_id."""
    import lightgbm as lgb
    df = val_df.sort_values("sample_id").reset_index(drop=True)
    X = _prep_X(df).values
    # discretize iou into rank labels 0..4 to feed lambdarank stably
    bins = np.array([0.0, 0.05, 0.15, 0.30, 0.50, 1.01])
    y = np.digitize(df["iou"].values, bins) - 1   # 0..5 → clamp
    y = np.clip(y, 0, 4).astype(int)
    group_sizes = df.groupby("sample_id", sort=False).size().values
    model = lgb.LGBMRanker(
        n_estimators=120, learning_rate=0.05, num_leaves=15,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.9, reg_lambda=1.0, random_state=seed, verbosity=-1,
        objective="lambdarank", label_gain=[0, 1, 3, 7, 15],
    )
    model.fit(X, y, group=group_sizes)
    return model, df.index   # the sorted index (we won't reuse but for reference


def train_lgb_classifier(val_df: pd.DataFrame, seed: int = 0):
    """Per-(sample, offset) classification: is this offset the sample oracle?
    Score = predict_proba(is_oracle=1). Pick argmax per sample."""
    import lightgbm as lgb
    X = _prep_X(val_df).values
    y = val_df["is_oracle"].values
    # heavily imbalanced (~1 of ~75 is positive in dense) → use class_weight
    pos_frac = max(float(y.mean()), 1e-6)
    scale_pos_weight = (1.0 - pos_frac) / pos_frac
    model = lgb.LGBMClassifier(
        n_estimators=120, learning_rate=0.05, num_leaves=15,
        min_child_samples=20, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.9, reg_lambda=1.0, random_state=seed, verbosity=-1,
        scale_pos_weight=scale_pos_weight,
    )
    model.fit(X, y)
    return model


def pick_offsets(model, df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """Score every (sample, offset) row, then for each sample return the
    argmax-score offset and its IoU."""
    X = _prep_X(df).values
    if kind == "classifier":
        scores = model.predict_proba(X)[:, 1]
    else:
        scores = model.predict(X)
    out = df[["sample_id", "offset", "iou"]].copy()
    out["score"] = scores
    # pick per-sample argmax
    idx = out.groupby("sample_id")["score"].idxmax()
    return out.loc[idx].reset_index(drop=True)


def overall_iou_n_total(per_sample_iou: List[float], n_total: int) -> float:
    if n_total == 0:
        return 0.0
    return float(np.sum(per_sample_iou)) / float(n_total)


def fixed_offset_iou(df_coarse: pd.DataFrame, offset: int) -> float:
    sub = df_coarse[df_coarse["offset"] == int(offset)]
    n = sub["sample_id"].nunique()
    return overall_iou_n_total(sub["iou"].tolist(), n)


# ---------- previous selector reproduction --------------------------------
def previous_bin_rule_selector(val_coarse: pd.DataFrame,
                               test_coarse: pd.DataFrame,
                               feature: str = "score_rolling_slope_14d",
                               n_bins: int = 3) -> Tuple[float, str]:
    """Replicates the V1 best learned selector (bin_rule[score_rolling_slope_14d])
    from phase_b_stage2_offset_selector_diagnostic.py on coarse offsets."""
    # build per-sample tables
    val_samples = val_coarse.groupby("sample_id").first().reset_index()
    test_samples = test_coarse.groupby("sample_id").first().reset_index()

    vals = pd.to_numeric(val_samples[feature], errors="coerce").to_numpy()
    finite = vals[~np.isnan(vals)]
    if len(np.unique(finite)) < 2:
        return float("nan"), f"bin_rule[{feature}]: degenerate"
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

    # for each bin, find best coarse offset on val
    val_samples["__bin"] = vals
    val_samples["__bin"] = val_samples["__bin"].apply(bin_of)
    bin_best = {}
    for b in range(n_bins):
        sub_ids = set(val_samples[val_samples["__bin"] == b]["sample_id"].tolist())
        sub = val_coarse[val_coarse["sample_id"].isin(sub_ids)]
        if sub.empty:
            bin_best[b] = COARSE_OFFSETS[0]
            continue
        best_o, best_iou = COARSE_OFFSETS[0], -1.0
        for o in COARSE_OFFSETS:
            iou = fixed_offset_iou(sub, o)
            if iou > best_iou:
                best_iou, best_o = iou, o
        bin_best[b] = best_o
    # apply to test
    test_samples["__bin"] = pd.to_numeric(test_samples[feature], errors="coerce").apply(bin_of)
    sample_iou = []
    for _, r in test_samples.iterrows():
        chosen = bin_best[int(r["__bin"])]
        row = test_coarse[(test_coarse["sample_id"] == r["sample_id"]) &
                          (test_coarse["offset"] == chosen)]
        sample_iou.append(float(row["iou"].iloc[0]) if len(row) else 0.0)
    n_test = test_samples["sample_id"].nunique()
    return overall_iou_n_total(sample_iou, n_test), f"bin_rule[{feature}]: bins={bin_best}"


# ---------- main -----------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline_val_csv",
                    default="rice/outputs_stage2_batch_2024_bestgate/sheath_blight/lead_v3_val_sample_grid.csv")
    ap.add_argument("--baseline_test_csv",
                    default="rice/outputs_stage2_batch_2024_bestgate/sheath_blight/lead_v3_test_sample_grid.csv")
    ap.add_argument("--clim_train_stats_csv",
                    default="rice/outputs_stage2_batch_2024_bestgate/sheath_blight/climatology_train_stats.csv")
    ap.add_argument("--clim_test_grid_csv",
                    default="rice/outputs_stage2_batch_2024_bestgate/sheath_blight/climatology_mean_mid_test_sample_grid.csv",
                    help="A climatology test sample_grid (any of mean_L/mean_mid/mean_R) "
                         "whose IoU @ val_offset will be reported as 'best climatology' "
                         "after selecting the best of the 3 in the bestgate selection CSV.")
    ap.add_argument("--bestgate_selection_csv",
                    default="rice/outputs_stage2_batch_2024_bestgate/_summary/sheath_blight_selection.csv")
    ap.add_argument("--out_root", default="rice/outputs_stage2_batch_2024_bestgate/_offset_selector_v2")
    ap.add_argument("--fixed_val_offset", type=int, default=45,
                    help="Existing fixed val-selected offset for baseline_asym_mse. "
                         "Used for sanity-checking the fixed baseline IoU.")
    ap.add_argument("--extra_train_grids", type=str, default="",
                    help="Optional extra sample_grid CSVs concatenated into the "
                         "selector TRAINING set (the main val_csv is always "
                         "included). Format: 'label1:path1[,label2:path2,...]'. "
                         "Each extra grid is loaded via the same dense candidate "
                         "builder. Useful for multi-year selector training "
                         "(Path A — sample_grids from earlier years using the "
                         "same Stage-2 ckpt). NOTE: extra grids from years the "
                         "Stage-2 model has SEEN during training are 'training-fit' "
                         "predictions; mark this fact in your summary.")
    args = ap.parse_args()

    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    # ---- climatology mean_mid (per-pest scalar) ---------------------------
    clim_stats = pd.read_csv(args.clim_train_stats_csv).iloc[0]
    clim_mid_doy = float(clim_stats["mean_mid"])
    print(f"[init] clim_mid_doy = {clim_mid_doy:.3f}")

    # ---- build candidate rows: val (dense for training) -------------------
    val_dense_main = build_candidate_rows(Path(args.baseline_val_csv),
                                          DENSE_OFFSETS, clim_mid_doy)
    val_coarse = build_candidate_rows(Path(args.baseline_val_csv),
                                      COARSE_OFFSETS, clim_mid_doy)
    test_dense = build_candidate_rows(Path(args.baseline_test_csv),
                                      DENSE_OFFSETS, clim_mid_doy)
    test_coarse = build_candidate_rows(Path(args.baseline_test_csv),
                                       COARSE_OFFSETS, clim_mid_doy)
    n_val_main = val_dense_main["sample_id"].nunique()
    n_test_samples = test_dense["sample_id"].nunique()
    print(f"[data] main val:  {n_val_main} samples × {len(DENSE_OFFSETS)} dense "
          f"offsets = {len(val_dense_main)} rows")

    # ---- optional extra training grids (multi-year cheap Path A) ----------
    extra_train_specs: List[Tuple[str, Path]] = []
    extra_dfs: List[pd.DataFrame] = []
    extra_summary_rows: List[dict] = []
    if args.extra_train_grids:
        for spec in args.extra_train_grids.split(","):
            spec = spec.strip()
            if not spec:
                continue
            if ":" not in spec:
                print(f"[abort] --extra_train_grids: expected 'label:path', got {spec!r}",
                      file=sys.stderr)
                return 2
            lab, pth = spec.split(":", 1)
            lab, pth = lab.strip(), pth.strip()
            df = build_candidate_rows(Path(pth), DENSE_OFFSETS, clim_mid_doy)
            # tag so we can introspect later
            df["_source_label"] = lab
            extra_train_specs.append((lab, Path(pth)))
            extra_dfs.append(df)
            n = df["sample_id"].nunique()
            extra_summary_rows.append({"label": lab, "path": pth, "n_samples": n,
                                       "n_rows": len(df)})
            print(f"[data] extra train ({lab}): {n} samples ×"
                  f" {len(DENSE_OFFSETS)} = {len(df)} rows  (from {pth})")
    val_dense_main["_source_label"] = "val_main"
    val_dense = pd.concat([val_dense_main] + extra_dfs, ignore_index=True) \
        if extra_dfs else val_dense_main
    n_val_samples = val_dense["sample_id"].nunique()   # union of sample IDs
    print(f"[data] combined training set: {n_val_samples} samples "
          f"({len(val_dense)} rows)  [main={n_val_main}, extras={n_val_samples - n_val_main}]")
    print(f"[data] test (held-out): {n_test_samples} samples × {len(DENSE_OFFSETS)} = "
          f"{len(test_dense)} rows")

    val_dense.to_csv(out_root / "v2_per_candidate_val.csv", index=False)
    if extra_summary_rows:
        pd.DataFrame(extra_summary_rows).to_csv(
            out_root / "v2_extra_train_grids_summary.csv", index=False)

    # ---- train 3 selectors on val (dense candidates) ----------------------
    print("[train] LightGBM regressor on iou ...")
    reg = train_lgb_regressor(val_dense)
    print("[train] LightGBM lambdarank ranker ...")
    rnk, _ = train_lgb_ranker(val_dense)
    print("[train] LightGBM classifier (is_oracle) ...")
    clf = train_lgb_classifier(val_dense)

    # ---- apply to test, in both candidate spaces --------------------------
    def evaluate(model, name: str, test_cands: pd.DataFrame, kind: str):
        picks = pick_offsets(model, test_cands, kind)
        iou_per_sample = picks["iou"].tolist()
        n = test_cands["sample_id"].nunique()
        return picks, overall_iou_n_total(iou_per_sample, n)

    results: List[dict] = []
    per_sample_records: Dict[str, pd.DataFrame] = {}

    # operational (coarse offsets only)
    for name, model, kind in [("v2_regressor_coarse",  reg, "regressor"),
                              ("v2_ranker_coarse",     rnk, "ranker"),
                              ("v2_classifier_coarse", clf, "classifier")]:
        picks, score = evaluate(model, name, test_coarse, kind)
        results.append({"selector": name, "test_iou": score, "policy_space": "coarse-6"})
        per_sample_records[name] = picks

    # diagnostic (dense — uses interpolated mu)
    for name, model, kind in [("v2_regressor_dense",  reg, "regressor"),
                              ("v2_ranker_dense",     rnk, "ranker"),
                              ("v2_classifier_dense", clf, "classifier")]:
        picks, score = evaluate(model, name, test_dense, kind)
        results.append({"selector": name, "test_iou": score, "policy_space": "dense-1..75 (interp)"})
        per_sample_records[name] = picks

    # ---- baselines --------------------------------------------------------
    # fixed val-selected
    fix_iou = fixed_offset_iou(test_coarse, args.fixed_val_offset)
    results.append({"selector": f"fixed_val_offset_{args.fixed_val_offset}",
                    "test_iou": fix_iou, "policy_space": "coarse-fixed"})

    # previous V1 selector (bin_rule[score_rolling_slope_14d])
    v1_iou, v1_desc = previous_bin_rule_selector(val_coarse, test_coarse,
                                                  feature="score_rolling_slope_14d",
                                                  n_bins=3)
    results.append({"selector": "v1_bin_rule[score_rolling_slope_14d]",
                    "test_iou": v1_iou, "policy_space": "coarse-rule"})

    # best climatology — pull from existing bestgate selection.csv
    best_clim = float("nan"); best_clim_kind = "?"
    try:
        sel = pd.read_csv(args.bestgate_selection_csv)
        clim_rows = sel[sel["model"].astype(str).str.contains("clim_")]
        if not clim_rows.empty:
            score_col = "test_IoU_overall_n_total_at_val_offset"
            best = clim_rows.loc[clim_rows[score_col].idxmax()]
            best_clim = float(best[score_col])
            best_clim_kind = str(best["model"])
    except Exception as e:
        print(f"[warn] could not load best climatology: {e}")
    results.append({"selector": f"best_climatology({best_clim_kind})",
                    "test_iou": best_clim, "policy_space": "climatology"})

    # LEAKY oracles (upper bounds)
    sample_oracle_coarse = test_coarse.groupby("sample_id")["iou"].max().tolist()
    n_t = test_coarse["sample_id"].nunique()
    results.append({"selector": "LEAKY_sample_oracle_coarse",
                    "test_iou": overall_iou_n_total(sample_oracle_coarse, n_t),
                    "policy_space": "leaky"})
    sample_oracle_dense = test_dense.groupby("sample_id")["iou"].max().tolist()
    results.append({"selector": "LEAKY_sample_oracle_dense",
                    "test_iou": overall_iou_n_total(sample_oracle_dense, n_t),
                    "policy_space": "leaky"})

    # ---- write results ----------------------------------------------------
    res_df = pd.DataFrame(results)
    res_df.to_csv(out_root / "v2_test_results_summary.csv", index=False)

    # one CSV with selector chosen offset per sample (just dense regressor &
    # ranker as primary reference; classifier too for completeness)
    big_pick = []
    for label, picks in per_sample_records.items():
        tmp = picks.copy()
        tmp["selector"] = label
        big_pick.append(tmp)
    if big_pick:
        pd.concat(big_pick, ignore_index=True).to_csv(
            out_root / "v2_per_sample_test_selections.csv", index=False)

    # ---- summary TXT ------------------------------------------------------
    write_summary_txt(out_root / "v2_summary_for_ppt.txt", res_df,
                      n_val_samples, n_test_samples, clim_mid_doy,
                      v1_desc, args.fixed_val_offset,
                      extra_summary_rows=extra_summary_rows,
                      n_val_main=n_val_main)
    print(f"[done] outputs under {out_root}")
    return 0


def write_summary_txt(out_path: Path, res: pd.DataFrame,
                      n_val: int, n_test: int, clim_mid_doy: float,
                      v1_desc: str, fixed_off: int,
                      extra_summary_rows: List[dict] | None = None,
                      n_val_main: int | None = None):
    lines: List[str] = []
    lines.append("Stage 2 offset selector V2 — ranking on dense candidates")
    lines.append("=" * 70)
    lines.append("")
    lines.append("Setup")
    lines.append("-" * 70)
    lines.append(f"  data           : baseline_asym_mse sample_grid (sheath_blight)")
    if extra_summary_rows:
        lines.append(f"  val (train)    : {n_val} samples total = main {n_val_main} + "
                     f"{sum(r['n_samples'] for r in extra_summary_rows)} extra "
                     f"(× 75 dense offsets)")
        for r in extra_summary_rows:
            lines.append(f"     + extra      : {r['label']:<20s} n={r['n_samples']:>5d}  ({r['path']})")
        lines.append(f"  NOTE           : extra grids from years the Stage-2 ckpt has SEEN")
        lines.append(f"                   during training are 'training-fit' predictions —")
        lines.append(f"                   selector may overfit to that distribution.")
    else:
        lines.append(f"  val (train)    : {n_val} samples × 75 dense offsets")
    lines.append(f"  test (eval)    : {n_test} samples × 75 dense offsets")
    lines.append(f"  clim_mid_doy   : {clim_mid_doy:.2f}")
    lines.append(f"  sigma          : {SIGMA}")
    lines.append(f"  candidate space: coarse {COARSE_OFFSETS} (operational); "
                 f"dense 1..75 (interp, diagnostic)")
    lines.append(f"  selectors      : LightGBM regressor on iou; ranker (lambdarank); "
                 f"classifier on is_oracle")
    lines.append(f"  features       : {', '.join(FEATURE_COLS)}")
    lines.append(f"  fixed baseline : val-selected offset = {fixed_off}")
    lines.append(f"  v1 selector    : {v1_desc}")
    lines.append("")
    lines.append("Test IoU per method")
    lines.append("-" * 70)
    lines.append(f"  {'selector':<48s}  {'test_iou':>9s}  {'policy_space':<22s}")
    # ordering: baselines first, then operational selectors, then leaky
    order_keys = {
        "fixed_val_offset": 0,
        "v1_bin_rule": 1,
        "v2_regressor_coarse": 2,
        "v2_ranker_coarse": 3,
        "v2_classifier_coarse": 4,
        "best_climatology": 5,
        "v2_regressor_dense": 6,
        "v2_ranker_dense": 7,
        "v2_classifier_dense": 8,
        "LEAKY": 9,
    }
    def _ord(name):
        for k, v in order_keys.items():
            if name.startswith(k):
                return v
        return 99
    res2 = res.copy()
    res2["_o"] = res2["selector"].map(_ord)
    res2 = res2.sort_values("_o")
    for _, r in res2.iterrows():
        s = "  nan " if pd.isna(r["test_iou"]) else f"{float(r['test_iou']):>9.3f}"
        lines.append(f"  {r['selector']:<48s}  {s}  {str(r['policy_space']):<22s}")
    lines.append("")
    lines.append("Reading guide / success criteria")
    lines.append("-" * 70)
    lines.append("  * Floor       : v2 selector ≥ fixed baseline (≈ 0.239). Otherwise")
    lines.append("                   the ranker is harmful.")
    lines.append("  * Meaningful  : v2 selector > v1 best (≈ 0.273).")
    lines.append("  * Strong win  : v2 selector > best climatology (≈ 0.369).")
    lines.append("  * Headroom    : LEAKY sample_oracle_coarse ≈ 0.433; how much of the")
    lines.append("                   gap (0.239 → 0.433) does v2 close?")
    lines.append("  * Dense vs coarse: if dense > coarse for the same selector, finer")
    lines.append("                   grid matters and we should consider running the")
    lines.append("                   Stage 2 model at more offsets.")
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
