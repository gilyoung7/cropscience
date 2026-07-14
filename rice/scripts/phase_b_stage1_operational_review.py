"""Stage 1 batch operational review — recall + false-positive perspective.

Re-analyzes Stage 1 dispatch batch outputs through the FP / operational-alert
lens rather than F1. For each (pest, method) at target R>=0.88, computes
mean/std of precision/recall/F1/FAR/lead_median/no_alert/USEFUL and the
fallback rate, then picks an 'operational best' method per pest using:

    1) If recall_mean >= 0.88: among candidates that meet recall, pick the
       one with the lowest FAR_mean; tie-break by highest precision_mean.
    2) Else: pick the method with the highest recall_mean; tie-break by
       lowest FAR_mean.

Each pest is also tagged with a coarse grade:
    'usable'     : recall >= 0.88 AND FAR <= 0.50
    'borderline' : (recall >= 0.88 AND FAR <= 0.70) OR (recall >= 0.85 AND FAR <= 0.50)
    'weak'       : otherwise

Stdout: per-pest recommendation + interpretation; CSV writes for the
aggregate table and the per-pest best.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


SHEATH_BLIGHT_LEGACY_CSV = Path("rice/outputs_stage1/seed_stability/farmin_rows.csv")


def _load_batch(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # canonical columns expected:
    needed = {"pest", "run", "split", "method", "target", "tau_repr",
              "precision_test", "recall_test", "F1_test", "FAR_test",
              "lead_median_test", "no_alert_test", "USEFUL_test",
              "n_event_test", "fallback"}
    miss = needed - set(df.columns)
    if miss:
        raise SystemExit(f"[abort] {csv_path} missing columns: {miss}")
    return df


def _load_sheath_legacy(csv_path: Path) -> pd.DataFrame | None:
    """Convert seed_stability/farmin_rows.csv schema -> batch schema for
    sheath_blight. Adds pest='sheath_blight', renames seed->run, derives
    tau_repr from tau / tau_no / tau_with."""
    if not csv_path.is_file():
        print(f"[legacy] not found: {csv_path}  (sheath_blight will be skipped)")
        return None
    df = pd.read_csv(csv_path)
    df = df.rename(columns={"seed": "run"})
    df["pest"] = "sheath_blight"

    def _tau_repr(r):
        if r["method"] == "dispatch_group_tau":
            return f"no={r.get('tau_no')}/with={r.get('tau_with')}"
        return str(r.get("tau", "-"))

    df["tau_repr"] = df.apply(_tau_repr, axis=1)
    # Keep the same target labeling as batch CSVs (R>=0.88).
    keep = ["pest", "run", "split", "method", "target", "k", "tau_repr",
            "precision_test", "recall_test", "F1_test", "FAR_test",
            "lead_median_test", "no_alert_test", "USEFUL_test",
            "n_event_test", "fallback"]
    keep = [c for c in keep if c in df.columns]
    return df[keep]


def _normalize_target(t: str) -> str:
    """R>=0.9 and R>=0.90 are the same. Normalize."""
    if t in ("R>=0.9", "R>=0.90"):
        return "R>=0.90"
    return t


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """pest × method mean/std + fallback_rate."""
    metric_cols = ["precision_test", "recall_test", "F1_test", "FAR_test",
                   "lead_median_test", "no_alert_test", "USEFUL_test"]
    grp = df.groupby(["pest", "method"], dropna=False)
    rows = []
    for (pest, method), sub in grp:
        rec = {"pest": pest, "method": method, "n_rows": int(len(sub))}
        for c in metric_cols:
            v = pd.to_numeric(sub[c], errors="coerce")
            rec[f"{c}_mean"] = float(v.mean()) if v.notna().any() else float("nan")
            rec[f"{c}_std"] = float(v.std(ddof=0)) if v.notna().any() else float("nan")
        # fallback_rate: 'yes' / total
        fb = sub["fallback"].astype(str).str.lower()
        rec["fallback_rate"] = float((fb == "yes").mean())
        rec["fallback_count"] = int((fb == "yes").sum())
        rows.append(rec)
    return pd.DataFrame(rows)


def grade_pest(best_row: dict, recall_target: float, recall_tolerance: float) -> str:
    r = best_row["recall_test_mean"]
    f = best_row["FAR_test_mean"]
    if not np.isfinite(r) or not np.isfinite(f):
        return "weak"
    eff = float(recall_target) - float(recall_tolerance)
    if r >= eff and f <= 0.50:
        return "usable"
    if (r >= eff and f <= 0.70) or (r >= eff - 0.03 and f <= 0.50):
        return "borderline"
    return "weak"


def pick_operational_best(agg_sub: pd.DataFrame, recall_target: float,
                            recall_tolerance: float) -> tuple[pd.Series, str]:
    """For one pest's aggregate (per-method rows), pick the operational best.

    recall_tolerance: near-miss buffer. A method whose recall_mean is within
    [target - tolerance, target) is treated as 'effectively meeting' the
    recall constraint. This prevents the hard cutoff from rejecting methods
    that are statistically indistinguishable from the target (e.g., recall
    0.879 vs target 0.88 with std ~0.03).
    """
    df = agg_sub.copy()
    df = df[df["recall_test_mean"].notna()]
    if df.empty:
        return None, "no data"

    eff_threshold = float(recall_target) - float(recall_tolerance)
    meets = df[df["recall_test_mean"] >= eff_threshold]
    if len(meets) > 0:
        # FAR min, tie-break by precision max, then F1 max.
        meets = meets.sort_values(
            ["FAR_test_mean", "precision_test_mean", "F1_test_mean"],
            ascending=[True, False, False],
        )
        tol_note = (f" (recall >= {eff_threshold:.4f} "
                    f"= target {recall_target} - tol {recall_tolerance})")
        return meets.iloc[0], f"recall constraint met{tol_note}; FAR-min selection"
    else:
        # Recall fallback: max recall, tie-break by FAR min, precision max.
        df2 = df.sort_values(
            ["recall_test_mean", "FAR_test_mean", "precision_test_mean"],
            ascending=[False, True, False],
        )
        return df2.iloc[0], (f"no method meets recall>={eff_threshold:.4f}; "
                              f"recall-max fallback")


def _fmt(v, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "nan"
    return f"{v:.{nd}f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--batch_csv", default="rice/outputs_stage1/batch_rolling/_summary/pest_batch_farmin_R088.csv")
    ap.add_argument("--sheath_legacy_csv", default=str(SHEATH_BLIGHT_LEGACY_CSV))
    ap.add_argument("--target", default="R>=0.88")
    ap.add_argument("--out_aggregate", default="rice/outputs_stage1/batch_rolling/_summary/pest_batch_R088_operational_summary.csv")
    ap.add_argument("--out_best", default="rice/outputs_stage1/batch_rolling/_summary/pest_batch_R088_operational_best_by_pest.csv")
    ap.add_argument("--lead_min_useful", type=float, default=14.0,
                    help="lead_median threshold for 'lead practical' interpretation.")
    ap.add_argument("--recall_target", type=float, default=0.88,
                    help="Operational recall constraint (default 0.88, matching "
                         "the dispatch selection target).")
    ap.add_argument("--recall_tolerance", type=float, default=0.005,
                    help="Near-miss buffer for the recall constraint. A method "
                         "with recall_mean in [target - tol, target) is treated "
                         "as effectively meeting recall (default 0.005, ~17%% of "
                         "typical inter-seed std). Set to 0 for strict hard cutoff.")
    ap.add_argument("--exclude_pests", default="BPH2",
                    help="Comma-separated pest names to drop from analysis. "
                         "Default 'BPH2' since the final 8-pest selection uses "
                         "BPH (full 1998-2024) instead of BPH2 (2020-2024 only).")
    args = ap.parse_args()

    batch = _load_batch(Path(args.batch_csv))
    print(f"[load] batch rows={len(batch)}  pests={sorted(batch.pest.unique())}")
    legacy = _load_sheath_legacy(Path(args.sheath_legacy_csv))
    if legacy is not None:
        print(f"[load] sheath_blight legacy rows={len(legacy)}")
        combined = pd.concat([batch, legacy], ignore_index=True)
    else:
        combined = batch.copy()

    combined["target"] = combined["target"].astype(str).map(_normalize_target)
    target_norm = _normalize_target(args.target)
    sub = combined[combined["target"] == target_norm].copy()
    if sub.empty:
        raise SystemExit(f"[abort] no rows at target={target_norm!r}")
    excluded_pests = {s.strip() for s in str(args.exclude_pests).split(",") if s.strip()}
    if excluded_pests:
        before = sub["pest"].nunique()
        sub = sub[~sub["pest"].isin(excluded_pests)]
        print(f"[filter] excluded pests: {sorted(excluded_pests)}  "
              f"(pests: {before} -> {sub['pest'].nunique()})")

    print(f"[filter] target={target_norm}  rows={len(sub)}  "
          f"pests={sorted(sub.pest.unique())}  methods={sorted(sub.method.unique())}")

    # Aggregate
    agg = aggregate(sub)
    agg = agg.sort_values(["pest", "method"])
    Path(args.out_aggregate).parent.mkdir(parents=True, exist_ok=True)
    agg.to_csv(args.out_aggregate, index=False)
    print(f"\n[wrote] {args.out_aggregate}  rows={len(agg)}")

    # Per-pest operational best + interpretation
    pests_ordered = sorted(sub.pest.unique())
    best_rows = []
    print("\n" + "=" * 90)
    print(" PER-PEST OPERATIONAL REVIEW  (target=R>=0.88, n_total=3 splits × 3 seeds = 9 rows/method)")
    print("=" * 90)
    for pest in pests_ordered:
        agg_p = agg[agg["pest"] == pest]
        if agg_p.empty:
            continue
        best, reason = pick_operational_best(agg_p, args.recall_target,
                                               args.recall_tolerance)
        f1_best_row = agg_p.sort_values("F1_test_mean", ascending=False).iloc[0]
        grade = grade_pest(best.to_dict(), args.recall_target, args.recall_tolerance)
        # interpretation flags
        fp_severe = (best["FAR_test_mean"] > 0.6)
        recall_ok = (best["recall_test_mean"] >= args.recall_target - args.recall_tolerance)
        lead_practical = (best["lead_median_test_mean"] >= args.lead_min_useful)
        best_record = {
            "pest": pest,
            "operational_best_method": best["method"],
            "selection_reason": reason,
            "grade": grade,
            "fp_severe": bool(fp_severe),
            "recall_ok": bool(recall_ok),
            "lead_practical": bool(lead_practical),
            "f1_best_method": f1_best_row["method"],
            "f1_best_differs": (best["method"] != f1_best_row["method"]),
            "n_rows": int(best["n_rows"]),
            "recall_mean": float(best["recall_test_mean"]),
            "recall_std":  float(best["recall_test_std"]),
            "FAR_mean":    float(best["FAR_test_mean"]),
            "FAR_std":     float(best["FAR_test_std"]),
            "precision_mean": float(best["precision_test_mean"]),
            "precision_std":  float(best["precision_test_std"]),
            "F1_mean":     float(best["F1_test_mean"]),
            "F1_std":      float(best["F1_test_std"]),
            "lead_median_mean": float(best["lead_median_test_mean"]),
            "lead_median_std":  float(best["lead_median_test_std"]),
            "no_alert_mean":    float(best["no_alert_test_mean"]),
            "USEFUL_mean":      float(best["USEFUL_test_mean"]),
            "fallback_rate":    float(best["fallback_rate"]),
            "f1_best_recall_mean": float(f1_best_row["recall_test_mean"]),
            "f1_best_FAR_mean":    float(f1_best_row["FAR_test_mean"]),
            "f1_best_F1_mean":     float(f1_best_row["F1_test_mean"]),
        }
        best_rows.append(best_record)

        # Stdout block per pest
        print(f"\n## {pest}   grade=[{grade}]")
        print(f"  rec: {best['method']}  ({reason})")
        print(f"     recall={_fmt(best['recall_test_mean'])} ± {_fmt(best['recall_test_std'])}  "
              f"FAR={_fmt(best['FAR_test_mean'])} ± {_fmt(best['FAR_test_std'])}  "
              f"precision={_fmt(best['precision_test_mean'])} ± {_fmt(best['precision_test_std'])}")
        print(f"     F1={_fmt(best['F1_test_mean'])} ± {_fmt(best['F1_test_std'])}  "
              f"lead_med={_fmt(best['lead_median_test_mean'], 1)} d  "
              f"no_alert={_fmt(best['no_alert_test_mean'], 1)}  "
              f"USEFUL={_fmt(best['USEFUL_test_mean'], 1)}  "
              f"fallback_rate={_fmt(best['fallback_rate'], 2)}")
        flags = []
        if not recall_ok:
            flags.append(f"recall<{0.88} → 못 잡는 event 있음")
        else:
            flags.append("recall 목표 충족")
        if fp_severe:
            flags.append(f"false positive 심함 (FAR>0.6)")
        elif best["FAR_test_mean"] > 0.45:
            flags.append("false positive 중간 수준")
        else:
            flags.append("false positive 양호")
        if lead_practical:
            flags.append(f"lead 실용적 (median {best['lead_median_test_mean']:.0f}d)")
        else:
            flags.append(f"lead 짧음 (median {best['lead_median_test_mean']:.0f}d)")
        print(f"  진단: {' · '.join(flags)}")

        # F1 best vs operational best
        if best["method"] != f1_best_row["method"]:
            print(f"  F1 best: {f1_best_row['method']}  "
                  f"(F1={_fmt(f1_best_row['F1_test_mean'])}  "
                  f"recall={_fmt(f1_best_row['recall_test_mean'])}  "
                  f"FAR={_fmt(f1_best_row['FAR_test_mean'])})")
            print(f"  → F1 best와 operational best가 다른 이유: F1는 recall과 precision의 조화평균이라 "
                  f"recall이 약간 낮아도 precision이 높으면 1등이 될 수 있음. operational 기준은 "
                  f"recall 충족을 hard constraint로 두므로 다른 선택이 나옴.")
        else:
            print(f"  F1 best ≡ operational best ({f1_best_row['method']})")

    best_df = pd.DataFrame(best_rows)
    Path(args.out_best).parent.mkdir(parents=True, exist_ok=True)
    best_df.to_csv(args.out_best, index=False)
    print(f"\n[wrote] {args.out_best}  rows={len(best_df)}")

    # Final ranking summary table
    if not best_df.empty:
        print("\n" + "=" * 90)
        print(" FINAL — operational best by pest (canonical recall + FAR view)")
        print("=" * 90)
        show = best_df[["pest", "operational_best_method", "grade",
                        "recall_mean", "FAR_mean", "precision_mean",
                        "F1_mean", "lead_median_mean", "no_alert_mean",
                        "fallback_rate", "f1_best_method", "f1_best_differs"]].copy()
        show = show.sort_values(["grade", "FAR_mean"],
                                 key=lambda c: c.map(
                                     {"usable": 0, "borderline": 1, "weak": 2}
                                 ) if c.name == "grade" else c,
                                 ascending=[True, True])
        with pd.option_context("display.width", 200,
                                "display.max_columns", 30,
                                "display.float_format", "{:.4f}".format):
            print(show.to_string(index=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
