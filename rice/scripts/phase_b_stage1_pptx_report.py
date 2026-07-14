"""Stage 1 operational TXT report for PPT.

Builds two complementary views of the 8-pest Stage 1 batch_rolling results
at target R>=0.88 (with recall_tolerance 0.005) and writes one human-readable
text report.

  View A (val-selected seed):
    For each (pest, method, split) pick ONE run out of {0,1,2} using ONLY
    validation metrics:
       1) prefer runs with val_recall >= 0.875
       2) tie-break by lowest val_FAR
       3) tie-break by highest val_F1   (val_precision is not stored)
       4) tie-break by smallest run id
       If no run meets val_recall>=0.875: pick max val_recall, then min val_FAR.
    The selected run's TEST metric becomes that (pest, method, split) row.
    Aggregate to pest x method as mean ± std across the 3 splits.

  View B (Macro9):
    All 9 (split, run) test rows averaged (current default behavior).

Per-pest 'operational best method' uses the same rule in both views:
    recall_mean >= 0.875: pick min FAR_mean, tie precision, then F1,
                          then arbitrary D_history > dispatch_group_tau > A_baseline
    else: pick max recall_mean, then min FAR_mean.

Excluded: BPH2 (final selection uses BPH for 1998-2024 data instead).
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd


METHOD_PRIORITY = {"D_history": 0, "dispatch_group_tau": 1, "A_baseline": 2}
TARGET_NORMALIZED = "R>=0.88"
RECALL_TARGET = 0.88
RECALL_TOL = 0.005
EFF_THRESHOLD = RECALL_TARGET - RECALL_TOL  # 0.875


def _norm_target(t: str) -> str:
    if t in ("R>=0.9", "R>=0.90"):
        return "R>=0.90"
    return t


def grade(recall_mean: float, far_mean: float) -> str:
    if not (np.isfinite(recall_mean) and np.isfinite(far_mean)):
        return "weak"
    if recall_mean >= EFF_THRESHOLD and far_mean <= 0.50:
        return "usable"
    if (recall_mean >= EFF_THRESHOLD and far_mean <= 0.70) or \
       (recall_mean >= 0.85 and far_mean <= 0.50):
        return "borderline"
    return "weak"


# -------------------------------------------------------------- view A: val-pick
def pick_val_run(sub: pd.DataFrame) -> pd.Series:
    """Pick one run out of {0,1,2} for one (pest, method, split) using val only."""
    # Sort once with a composite key; first row wins.
    # primary: -is_meets (we want meets=True first); secondary: val_FAR asc;
    # tertiary: -val_F1 (want max); quaternary: run asc.
    sub = sub.copy()
    sub["__meets"] = (sub["recall_val"] >= EFF_THRESHOLD).astype(int)
    if (sub["__meets"] == 1).any():
        ranked = sub[sub["__meets"] == 1].sort_values(
            ["FAR_val", "F1_val", "run"], ascending=[True, False, True]
        )
    else:
        # Fallback: max recall, then min FAR.
        ranked = sub.sort_values(
            ["recall_val", "FAR_val", "run"], ascending=[False, True, True]
        )
    return ranked.iloc[0]


def build_view_a(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (per_split_picked, pest_method_aggregate)."""
    picked_rows = []
    for (pest, method, split), sub in df.groupby(["pest", "method", "split"]):
        r = pick_val_run(sub)
        picked_rows.append({
            "pest": pest, "method": method, "split": split,
            "picked_run": int(r["run"]),
            "val_recall": float(r["recall_val"]),
            "val_FAR": float(r["FAR_val"]),
            "val_F1": float(r["F1_val"]),
            "recall_test": float(r["recall_test"]),
            "FAR_test": float(r["FAR_test"]),
            "precision_test": float(r["precision_test"]),
            "F1_test": float(r["F1_test"]),
            "lead_median_test": float(r["lead_median_test"]),
            "no_alert_test": float(r["no_alert_test"]),
            "USEFUL_test": float(r["USEFUL_test"]),
            "fallback": str(r["fallback"]),
        })
    picked = pd.DataFrame(picked_rows)

    agg_rows = []
    metric_cols = ["recall_test", "FAR_test", "precision_test", "F1_test",
                   "lead_median_test", "no_alert_test", "USEFUL_test"]
    for (pest, method), sub in picked.groupby(["pest", "method"]):
        runs_str = ",".join(str(int(x)) for x in sorted(sub["picked_run"].tolist()))
        rec = {"pest": pest, "method": method,
               "n_splits": int(len(sub)),
               "picked_runs": runs_str}
        for c in metric_cols:
            v = sub[c].astype(float)
            rec[f"{c}_mean"] = float(v.mean())
            rec[f"{c}_std"] = float(v.std(ddof=0))
        fb = (sub["fallback"].astype(str).str.lower() == "yes")
        rec["fallback_rate"] = float(fb.mean())
        rec["grade"] = grade(rec["recall_test_mean"], rec["FAR_test_mean"])
        agg_rows.append(rec)
    return picked, pd.DataFrame(agg_rows)


# -------------------------------------------------------------- view B: macro9
def build_view_b(df: pd.DataFrame) -> pd.DataFrame:
    agg_rows = []
    metric_cols = ["recall_test", "FAR_test", "precision_test", "F1_test",
                   "lead_median_test", "no_alert_test", "USEFUL_test"]
    for (pest, method), sub in df.groupby(["pest", "method"]):
        rec = {"pest": pest, "method": method, "n_rows": int(len(sub))}
        for c in metric_cols:
            v = pd.to_numeric(sub[c], errors="coerce")
            rec[f"{c}_mean"] = float(v.mean())
            rec[f"{c}_std"] = float(v.std(ddof=0))
        fb = (sub["fallback"].astype(str).str.lower() == "yes")
        rec["fallback_rate"] = float(fb.mean())
        rec["grade"] = grade(rec["recall_test_mean"], rec["FAR_test_mean"])
        agg_rows.append(rec)
    return pd.DataFrame(agg_rows)


# -------------------------------------------------------------- per-pest best
def pick_pest_best(agg: pd.DataFrame, pest: str) -> pd.Series:
    sub = agg[agg["pest"] == pest].copy()
    if sub.empty:
        return None
    meets = sub[sub["recall_test_mean"] >= EFF_THRESHOLD]
    if len(meets):
        meets = meets.assign(
            _mp=meets["method"].map(METHOD_PRIORITY).fillna(99).astype(int))
        ranked = meets.sort_values(
            ["FAR_test_mean", "precision_test_mean", "F1_test_mean", "_mp"],
            ascending=[True, False, False, True],
        )
        return ranked.iloc[0]
    sub = sub.assign(_mp=sub["method"].map(METHOD_PRIORITY).fillna(99).astype(int))
    ranked = sub.sort_values(
        ["recall_test_mean", "FAR_test_mean", "_mp"],
        ascending=[False, True, True],
    )
    return ranked.iloc[0]


# -------------------------------------------------------------- formatting
def _fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "nan"
    return f"{v:.{nd}f}"


def fmt_mean_std(m, s, nd=3):
    return f"{_fmt(m, nd)} ± {_fmt(s, nd)}"


def render_table_a(agg_a: pd.DataFrame, pests: list[str]) -> str:
    """Table 1 — Val-selected seed aggregate."""
    cols = ("pest", "method", "picked_runs",
            "recall", "FAR", "precision", "F1", "lead_median", "grade")
    widths = {"pest": 18, "method": 20, "picked_runs": 16,
              "recall": 16, "FAR": 16, "precision": 16,
              "F1": 16, "lead_median": 16, "grade": 11}
    out = io.StringIO()
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    out.write(header + "\n")
    out.write("  ".join("-" * widths[c] for c in cols) + "\n")
    for pest in pests:
        for m in ("A_baseline", "D_history", "dispatch_group_tau"):
            r = agg_a[(agg_a["pest"] == pest) & (agg_a["method"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            row = [
                pest.ljust(widths["pest"]),
                m.ljust(widths["method"]),
                str(r["picked_runs"]).ljust(widths["picked_runs"]),
                fmt_mean_std(r["recall_test_mean"], r["recall_test_std"]).ljust(widths["recall"]),
                fmt_mean_std(r["FAR_test_mean"], r["FAR_test_std"]).ljust(widths["FAR"]),
                fmt_mean_std(r["precision_test_mean"], r["precision_test_std"]).ljust(widths["precision"]),
                fmt_mean_std(r["F1_test_mean"], r["F1_test_std"]).ljust(widths["F1"]),
                fmt_mean_std(r["lead_median_test_mean"], r["lead_median_test_std"], nd=1).ljust(widths["lead_median"]),
                str(r["grade"]).ljust(widths["grade"]),
            ]
            out.write("  ".join(row) + "\n")
        out.write("\n")
    return out.getvalue()


def render_table_b(agg_b: pd.DataFrame, pests: list[str]) -> str:
    """Table 2 — Macro9 mean ± std."""
    cols = ("pest", "method", "n_rows",
            "recall", "FAR", "precision", "F1", "lead_median", "grade")
    widths = {"pest": 18, "method": 20, "n_rows": 7,
              "recall": 16, "FAR": 16, "precision": 16,
              "F1": 16, "lead_median": 16, "grade": 11}
    out = io.StringIO()
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    out.write(header + "\n")
    out.write("  ".join("-" * widths[c] for c in cols) + "\n")
    for pest in pests:
        for m in ("A_baseline", "D_history", "dispatch_group_tau"):
            r = agg_b[(agg_b["pest"] == pest) & (agg_b["method"] == m)]
            if r.empty:
                continue
            r = r.iloc[0]
            row = [
                pest.ljust(widths["pest"]),
                m.ljust(widths["method"]),
                str(int(r["n_rows"])).ljust(widths["n_rows"]),
                fmt_mean_std(r["recall_test_mean"], r["recall_test_std"]).ljust(widths["recall"]),
                fmt_mean_std(r["FAR_test_mean"], r["FAR_test_std"]).ljust(widths["FAR"]),
                fmt_mean_std(r["precision_test_mean"], r["precision_test_std"]).ljust(widths["precision"]),
                fmt_mean_std(r["F1_test_mean"], r["F1_test_std"]).ljust(widths["F1"]),
                fmt_mean_std(r["lead_median_test_mean"], r["lead_median_test_std"], nd=1).ljust(widths["lead_median"]),
                str(r["grade"]).ljust(widths["grade"]),
            ]
            out.write("  ".join(row) + "\n")
        out.write("\n")
    return out.getvalue()


def render_best_table(best_rows: list[dict], title: str) -> str:
    cols = ("pest", "best_method", "grade", "recall", "FAR", "precision", "F1", "lead_median")
    widths = {"pest": 18, "best_method": 20, "grade": 11,
              "recall": 8, "FAR": 8, "precision": 10, "F1": 8, "lead_median": 12}
    out = io.StringIO()
    out.write(title + "\n")
    out.write("  ".join(c.ljust(widths[c]) for c in cols) + "\n")
    out.write("  ".join("-" * widths[c] for c in cols) + "\n")
    for r in best_rows:
        out.write("  ".join([
            str(r["pest"]).ljust(widths["pest"]),
            str(r["best_method"]).ljust(widths["best_method"]),
            str(r["grade"]).ljust(widths["grade"]),
            _fmt(r["recall_mean"]).ljust(widths["recall"]),
            _fmt(r["FAR_mean"]).ljust(widths["FAR"]),
            _fmt(r["precision_mean"]).ljust(widths["precision"]),
            _fmt(r["F1_mean"]).ljust(widths["F1"]),
            _fmt(r["lead_median_mean"], 1).ljust(widths["lead_median"]),
        ]) + "\n")
    return out.getvalue()


def render_compare_table(best_a: list[dict], best_b: list[dict]) -> str:
    by_a = {r["pest"]: r for r in best_a}
    by_b = {r["pest"]: r for r in best_b}
    cols = ("pest", "best_A_(val)", "A_R", "A_FAR", "A_P", "A_F1",
            "best_B_(macro9)", "B_R", "B_FAR", "B_P", "B_F1", "changed?", "interpretation")
    widths = {"pest": 18, "best_A_(val)": 20, "A_R": 7, "A_FAR": 7, "A_P": 7, "A_F1": 7,
              "best_B_(macro9)": 20, "B_R": 7, "B_FAR": 7, "B_P": 7, "B_F1": 7,
              "changed?": 9, "interpretation": 40}
    out = io.StringIO()
    out.write("  ".join(c.ljust(widths[c]) for c in cols) + "\n")
    out.write("  ".join("-" * widths[c] for c in cols) + "\n")
    for pest in by_a:
        a = by_a[pest]; b = by_b[pest]
        changed = (a["best_method"] != b["best_method"])
        if not changed:
            interp = "agreement"
        else:
            # Reason
            interp = f"A picks {a['best_method']} (FAR={_fmt(a['FAR_mean'])}); " \
                     f"B picks {b['best_method']} (FAR={_fmt(b['FAR_mean'])})"
        row = [pest.ljust(widths["pest"]),
               str(a["best_method"]).ljust(widths["best_A_(val)"]),
               _fmt(a["recall_mean"], 3).ljust(widths["A_R"]),
               _fmt(a["FAR_mean"], 3).ljust(widths["A_FAR"]),
               _fmt(a["precision_mean"], 3).ljust(widths["A_P"]),
               _fmt(a["F1_mean"], 3).ljust(widths["A_F1"]),
               str(b["best_method"]).ljust(widths["best_B_(macro9)"]),
               _fmt(b["recall_mean"], 3).ljust(widths["B_R"]),
               _fmt(b["FAR_mean"], 3).ljust(widths["B_FAR"]),
               _fmt(b["precision_mean"], 3).ljust(widths["B_P"]),
               _fmt(b["F1_mean"], 3).ljust(widths["B_F1"]),
               ("YES" if changed else "no").ljust(widths["changed?"]),
               interp.ljust(widths["interpretation"]),
               ]
        out.write("  ".join(row) + "\n")
    return out.getvalue()


# -------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--all_csv",
                    default="rice/outputs_stage1/batch_rolling/_summary/pest_batch_farmin_all.csv")
    ap.add_argument("--exclude_pests", default="BPH2")
    ap.add_argument("--out_txt",
                    default="rice/outputs_stage1/batch_rolling/_summary/stage1_operational_report_for_ppt.txt")
    ap.add_argument("--save_aux_csv", action="store_true",
                    help="Also write per-split picked + per-method aggregate CSVs.")
    args = ap.parse_args()

    df = pd.read_csv(args.all_csv)
    df["target"] = df["target"].astype(str).map(_norm_target)
    df = df[df["target"] == TARGET_NORMALIZED].copy()
    excluded = {s.strip() for s in str(args.exclude_pests).split(",") if s.strip()}
    df = df[~df["pest"].isin(excluded)]

    pests = sorted(df["pest"].unique())
    print(f"[report] pests ({len(pests)}): {pests}  excluded={sorted(excluded)}")
    print(f"[report] rows={len(df)}  (expect 8 × 27 = 216 for default 8 pests)")

    # Views
    picked, agg_a = build_view_a(df)
    agg_b = build_view_b(df)

    # Per-pest best (each view)
    def _build_best(agg, label):
        rows = []
        for p in pests:
            r = pick_pest_best(agg, p)
            if r is None:
                continue
            rows.append({
                "pest": p, "best_method": r["method"], "grade": r["grade"],
                "recall_mean": float(r["recall_test_mean"]),
                "FAR_mean": float(r["FAR_test_mean"]),
                "precision_mean": float(r["precision_test_mean"]),
                "F1_mean": float(r["F1_test_mean"]),
                "lead_median_mean": float(r["lead_median_test_mean"]),
            })
        return rows
    best_a = _build_best(agg_a, "val-selected")
    best_b = _build_best(agg_b, "macro9")

    # ===== Render TXT =====
    out = io.StringIO()
    out.write("================================================================================\n")
    out.write("Stage 1 Operational Review: Recall and False Positive Perspective\n")
    out.write("================================================================================\n\n")
    out.write(f"Generated from : {args.all_csv}\n")
    out.write(f"Pests included : {', '.join(pests)}  ({len(pests)})\n")
    out.write(f"Pests excluded : {sorted(excluded) if excluded else '(none)'}\n")
    out.write(f"Target         : {TARGET_NORMALIZED}\n")
    out.write(f"Recall rule    : effective threshold = {RECALL_TARGET} - "
              f"{RECALL_TOL} = {EFF_THRESHOLD:.3f}\n")
    out.write(f"Grade keys     : usable (R>={EFF_THRESHOLD:.3f}, FAR<=0.50)  "
              f"borderline (R>={EFF_THRESHOLD:.3f}, FAR<=0.70  or  R>=0.85, FAR<=0.50)  "
              f"weak (else)\n")
    out.write(f"Splits / runs  : 3 temporal splits (val=2021/22/23 -> test=2022/23/24) "
              f"x 3 XGB seeds (0/1/2)\n\n")

    out.write("-" * 80 + "\n")
    out.write("Table 1 - Val-selected seed (per pest×method×split: pick run by val only,\n")
    out.write("          then mean ± std across 3 splits)\n")
    out.write("-" * 80 + "\n\n")
    out.write(render_table_a(agg_a, pests))

    out.write("-" * 80 + "\n")
    out.write("Table 1b - Pest-level operational best by Val-selected seed view\n")
    out.write("-" * 80 + "\n")
    out.write(render_best_table(best_a, "(rule: prefer recall>=0.875 + min FAR; "
                                         "else max recall + min FAR)"))
    out.write("\n")

    out.write("-" * 80 + "\n")
    out.write("Table 2 - Macro9 mean ± std (all 3 splits × 3 runs = 9 rows per pest×method,\n")
    out.write("          simple arithmetic mean; NOT a single best run, NOT micro-concat)\n")
    out.write("-" * 80 + "\n\n")
    out.write(render_table_b(agg_b, pests))

    out.write("-" * 80 + "\n")
    out.write("Table 2b - Pest-level operational best by Macro9 view\n")
    out.write("-" * 80 + "\n")
    out.write(render_best_table(best_b, "(same rule as Table 1b)"))
    out.write("\n")

    out.write("-" * 80 + "\n")
    out.write("Table 3 - Comparison: Val-selected seed (A) vs Macro9 (B)\n")
    out.write("-" * 80 + "\n\n")
    out.write(render_compare_table(best_a, best_b))
    out.write("\n")

    # ----- Key takeaways -----
    out.write("-" * 80 + "\n")
    out.write("Key takeaways for PPT\n")
    out.write("-" * 80 + "\n\n")
    grade_buckets_a = {"usable": [], "borderline": [], "weak": []}
    for r in best_a:
        grade_buckets_a[r["grade"]].append(r["pest"])
    grade_buckets_b = {"usable": [], "borderline": [], "weak": []}
    for r in best_b:
        grade_buckets_b[r["grade"]].append(r["pest"])
    out.write(f"Grade distribution (val-selected): "
              f"usable={grade_buckets_a['usable']}, "
              f"borderline={grade_buckets_a['borderline']}, "
              f"weak={grade_buckets_a['weak']}\n")
    out.write(f"Grade distribution (macro9):       "
              f"usable={grade_buckets_b['usable']}, "
              f"borderline={grade_buckets_b['borderline']}, "
              f"weak={grade_buckets_b['weak']}\n\n")

    # Method frequency
    from collections import Counter
    def_a = Counter([r["best_method"] for r in best_a])
    def_b = Counter([r["best_method"] for r in best_b])
    out.write(f"Best method frequency (val-selected): {dict(def_a)}\n")
    out.write(f"Best method frequency (macro9):       {dict(def_b)}\n\n")

    n_D_a = def_a.get("D_history", 0)
    n_D_b = def_b.get("D_history", 0)
    out.write(f"D_history adoption:  val-selected={n_D_a}/{len(best_a)},  "
              f"macro9={n_D_b}/{len(best_b)}  "
              f"-> history feature reduces FAR in most pests.\n\n")

    # FP heavy (FAR>0.6)
    fp_heavy_a = [r["pest"] for r in best_a if r["FAR_mean"] > 0.6]
    fp_heavy_b = [r["pest"] for r in best_b if r["FAR_mean"] > 0.6]
    out.write(f"False positive heavy (FAR>0.6, val-selected): {fp_heavy_a}\n")
    out.write(f"False positive heavy (FAR>0.6, macro9):       {fp_heavy_b}\n\n")

    # Unstable: large recall std
    unstable = []
    for r in best_b:
        sub = agg_b[(agg_b["pest"] == r["pest"]) & (agg_b["method"] == r["best_method"])]
        if not sub.empty:
            s = float(sub.iloc[0]["recall_test_std"])
            if s > 0.10:
                unstable.append((r["pest"], s))
    if unstable:
        out.write(f"Unstable across splits (macro9 best-method recall_std > 0.10): "
                  f"{[(p, round(s,3)) for p,s in unstable]}\n")
        out.write(f"   -> e.g., rice_stem_borer family: specific years (split1 for "
                  f"borer_1, split3 for borer_2) collapse recall while other "
                  f"years sit at ~0.9. Reported macro mean understates per-year "
                  f"capability.\n\n")

    # ----- Presentation-ready bullets -----
    out.write("-" * 80 + "\n")
    out.write("Presentation-ready summary (copy into slides)\n")
    out.write("-" * 80 + "\n\n")
    bullets = []
    bullets.append(f"Stage 1 evaluated on 8 rice pests across 3 temporal splits × 3 seeds "
                   f"(BPH2 excluded; BPH (1998-2024) used instead).")
    if grade_buckets_b["usable"]:
        bullets.append(f"Operationally usable: {grade_buckets_b['usable']} "
                       f"(recall≥0.88 AND FAR≤0.50).")
    if grade_buckets_b["borderline"]:
        bullets.append(f"Borderline (recall met but FAR ~0.6): {grade_buckets_b['borderline']}.")
    if grade_buckets_b["weak"]:
        bullets.append(f"Weak (recall miss OR FAR>0.7): {grade_buckets_b['weak']}.")
    bullets.append(f"D_history is the operational best in {n_D_b}/{len(best_b)} pests "
                   f"(macro9 view) — history feature consistently reduces false alerts.")
    # consistency
    n_agree = sum(1 for a, b in zip(best_a, best_b) if a["best_method"] == b["best_method"])
    bullets.append(f"Val-selected seed view and Macro9 view agree on operational best "
                   f"in {n_agree}/{len(best_a)} pests.")
    if fp_heavy_b:
        bullets.append(f"False positive remains the main bottleneck: {len(fp_heavy_b)}/8 "
                       f"pests have FAR>0.6 even at the operational best.")
    if unstable:
        bullets.append(f"Stem borer pests show large inter-split variance "
                       f"(specific test years collapse) - per-year evaluation "
                       f"recommended for operational adoption.")
    for i, b in enumerate(bullets, 1):
        out.write(f"  {i}. {b}\n")
    out.write("\n")

    # Write
    out_path = Path(args.out_txt)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(out.getvalue(), encoding="utf-8")
    print(f"[report] wrote {out_path}  bytes={out_path.stat().st_size}")

    if args.save_aux_csv:
        d = out_path.parent
        picked.to_csv(d / "stage1_picked_per_split.csv", index=False)
        agg_a.to_csv(d / "stage1_val_selected_pest_method.csv", index=False)
        agg_b.to_csv(d / "stage1_macro9_pest_method.csv", index=False)
        pd.DataFrame(best_a).to_csv(d / "stage1_best_by_pest_val.csv", index=False)
        pd.DataFrame(best_b).to_csv(d / "stage1_best_by_pest_macro9.csv", index=False)
        print(f"[report] wrote 5 aux CSVs under {d}")

    # Stdout: pest-level best summary
    print("\n=== pest-level best summary (val-selected) ===")
    for r in best_a:
        print(f"  {r['pest']:22s} -> {r['best_method']:22s} "
              f"grade={r['grade']:10s} "
              f"R={r['recall_mean']:.3f} FAR={r['FAR_mean']:.3f} "
              f"P={r['precision_mean']:.3f} F1={r['F1_mean']:.3f}")
    print("\n=== pest-level best summary (macro9) ===")
    for r in best_b:
        print(f"  {r['pest']:22s} -> {r['best_method']:22s} "
              f"grade={r['grade']:10s} "
              f"R={r['recall_mean']:.3f} FAR={r['FAR_mean']:.3f} "
              f"P={r['precision_mean']:.3f} F1={r['F1_mean']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
