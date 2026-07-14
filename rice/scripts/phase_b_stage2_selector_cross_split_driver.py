#!/usr/bin/env python3
"""Run V2 offset selector across 8 pests × 3 split-years and aggregate.

Reuses phase_b_stage2_offset_selector_v2_ranking.py for the per-(pest, year)
selector training. This driver only loops over (year, pest), passes the
correct paths + fixed_val_offset, and stitches per-call outputs into:

  selector_by_pest_year.csv     long format, 1 row per (year, pest)
  selector_summary_wide.csv     wide format, all selectors as columns
  selector_summary_for_ppt.txt  human-readable rollup

LEAKAGE-FREE: each per-year selector is trained on THAT year's val sample_grid
(val_year from the rolling split), test labels from test_year are used only to
compute final IoU. No test labels enter the selector.
"""

from __future__ import annotations
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd


YEAR_TO_ROOT = {
    2022: Path("rice/outputs_stage2_batch_2022_baseline"),
    2023: Path("rice/outputs_stage2_batch_2023_baseline"),
    2024: Path("rice/outputs_stage2_batch_2024_bestgate"),
}
PESTS = [
    "BPH", "WBPH",
    "bacterial_blight", "blast", "brown_spot",
    "rice_stem_borer_1", "rice_stem_borer_2",
    "sheath_blight",
]
# canonical "learned selector" set considered for best_selector_name
LEARNED_SELECTOR_PREFIXES = (
    "v1_bin_rule",
    "v2_regressor_coarse", "v2_regressor_dense",
    "v2_ranker_coarse", "v2_ranker_dense",
    "v2_classifier_coarse", "v2_classifier_dense",
)
PY = ".venv/bin/python"


def get_fixed_val_offset(selection_csv: Path) -> int:
    """Return lead_v3's val_best_offset for this (root, pest)."""
    df = pd.read_csv(selection_csv)
    lead = df[df["model"].astype(str).str.endswith("_lead_v3")]
    if lead.empty or "val_best_offset" not in lead.columns:
        return 45  # safe fallback
    return int(lead.iloc[0]["val_best_offset"])


def run_v2(year: int, pest: str, force: bool, out_root: Path) -> Path:
    """Invoke phase_b_stage2_offset_selector_v2_ranking.py for one (pest,year).
    Returns the per-call v2_test_results_summary.csv path."""
    root = YEAR_TO_ROOT[year]
    pest_dir = root / pest
    sel_csv = root / "_summary" / f"{pest}_selection.csv"
    fixed_off = get_fixed_val_offset(sel_csv)
    sub_out = out_root / f"{year}_{pest}"
    sub_out.mkdir(parents=True, exist_ok=True)
    results_csv = sub_out / "v2_test_results_summary.csv"
    if results_csv.exists() and not force:
        return results_csv
    cmd = [
        PY, "rice/scripts/phase_b_stage2_offset_selector_v2_ranking.py",
        "--baseline_val_csv", str(pest_dir / "lead_v3_val_sample_grid.csv"),
        "--baseline_test_csv", str(pest_dir / "lead_v3_test_sample_grid.csv"),
        "--clim_train_stats_csv", str(pest_dir / "climatology_train_stats.csv"),
        "--clim_test_grid_csv",   str(pest_dir / "climatology_mean_mid_test_sample_grid.csv"),
        "--bestgate_selection_csv", str(sel_csv),
        "--out_root", str(sub_out),
        "--fixed_val_offset", str(fixed_off),
    ]
    print(f"\n[run] year={year}  pest={pest}  fixed_offset={fixed_off}")
    print(f"      out={sub_out}")
    log_path = sub_out / "run.log"
    with log_path.open("w") as logf:
        rc = subprocess.call(cmd, stdout=logf, stderr=subprocess.STDOUT)
    if rc != 0:
        print(f"  [FAIL] rc={rc}; see {log_path}")
        return None
    return results_csv


def parse_per_call(results_csv: Path) -> dict:
    """Extract canonical metrics from one v2 run's results CSV."""
    df = pd.read_csv(results_csv)
    out = {}
    # fixed baseline (row name format: fixed_val_offset_<N>)
    fix = df[df["selector"].str.startswith("fixed_val_offset_")]
    out["fixed_iou"] = float(fix.iloc[0]["test_iou"]) if len(fix) else float("nan")
    out["fixed_selector"] = str(fix.iloc[0]["selector"]) if len(fix) else "?"
    # climatology
    clim = df[df["selector"].str.startswith("best_climatology")]
    out["best_climatology_iou"] = float(clim.iloc[0]["test_iou"]) if len(clim) else float("nan")
    out["best_climatology_name"] = (str(clim.iloc[0]["selector"]) if len(clim)
                                     else "?")
    # oracle (coarse)
    orc = df[df["selector"] == "LEAKY_sample_oracle_coarse"]
    out["oracle_iou_coarse"] = float(orc.iloc[0]["test_iou"]) if len(orc) else float("nan")
    orc_d = df[df["selector"] == "LEAKY_sample_oracle_dense"]
    out["oracle_iou_dense"] = float(orc_d.iloc[0]["test_iou"]) if len(orc_d) else float("nan")
    # learned selectors → store each individually + pick best
    learned_rows = df[df["selector"].apply(
        lambda s: any(s.startswith(p) for p in LEARNED_SELECTOR_PREFIXES))]
    for _, r in learned_rows.iterrows():
        # column-safe key: replace [ ] with _
        key = str(r["selector"]).replace("[", "_").replace("]", "").replace(",", "_")
        out[f"sel__{key}"] = float(r["test_iou"])
    if not learned_rows.empty:
        best_row = learned_rows.loc[learned_rows["test_iou"].idxmax()]
        out["best_selector_name"] = str(best_row["selector"])
        out["best_selector_iou"] = float(best_row["test_iou"])
    else:
        out["best_selector_name"] = "(none)"
        out["best_selector_iou"] = float("nan")
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="rice/outputs_stage2_selector_cross_split")
    ap.add_argument("--force", action="store_true",
                    help="Re-run V2 selector even when per-(pest,year) output exists.")
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--pests", type=str, nargs="+", default=PESTS)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long_rows: List[dict] = []
    failures: List[Tuple[int, str]] = []

    for year in args.years:
        if year not in YEAR_TO_ROOT:
            print(f"[skip] year {year} not in YEAR_TO_ROOT")
            continue
        root = YEAR_TO_ROOT[year]
        if not root.exists():
            print(f"[skip] year {year} root missing: {root}")
            continue
        for pest in args.pests:
            sel = root / "_summary" / f"{pest}_selection.csv"
            if not sel.exists():
                print(f"[skip] {year}/{pest}: missing {sel}")
                failures.append((year, pest))
                continue
            res = run_v2(year, pest, force=args.force, out_root=out_dir)
            if res is None or not res.exists():
                failures.append((year, pest))
                continue
            try:
                rec = parse_per_call(res)
            except Exception as e:
                print(f"[fail] parse {year}/{pest}: {e}")
                failures.append((year, pest))
                continue
            rec_full = {
                "year": year, "pest": pest,
                "fixed_iou": rec["fixed_iou"],
                "fixed_selector": rec["fixed_selector"],
                "best_selector_name": rec["best_selector_name"],
                "best_selector_iou": rec["best_selector_iou"],
                "best_climatology_name": rec["best_climatology_name"],
                "best_climatology_iou": rec["best_climatology_iou"],
                "oracle_iou_coarse": rec["oracle_iou_coarse"],
                "oracle_iou_dense":  rec["oracle_iou_dense"],
                "selector_minus_fixed": rec["best_selector_iou"] - rec["fixed_iou"],
                "selector_minus_clim":  rec["best_selector_iou"] - rec["best_climatology_iou"],
                "oracle_minus_clim":    rec["oracle_iou_coarse"] - rec["best_climatology_iou"],
                "beats_fixed": (rec["best_selector_iou"] - rec["fixed_iou"]) > 1e-9,
                "beats_clim":  (rec["best_selector_iou"] - rec["best_climatology_iou"]) > 1e-9,
            }
            # also keep each learned selector's IoU as a column
            for k, v in rec.items():
                if k.startswith("sel__"):
                    rec_full[k] = v
            long_rows.append(rec_full)
            print(f"  {year}/{pest}: fixed={rec['fixed_iou']:.3f}  "
                  f"best_learned={rec['best_selector_iou']:.3f} ({rec['best_selector_name']})  "
                  f"clim={rec['best_climatology_iou']:.3f}  "
                  f"oracle={rec['oracle_iou_coarse']:.3f}")

    if not long_rows:
        print("[abort] no successful runs", file=sys.stderr)
        return 1

    long_df = pd.DataFrame(long_rows)
    keep = ["year", "pest", "fixed_iou", "best_selector_name", "best_selector_iou",
            "best_climatology_iou", "oracle_iou_coarse",
            "selector_minus_fixed", "selector_minus_clim", "oracle_minus_clim",
            "beats_fixed", "beats_clim",
            "fixed_selector", "best_climatology_name", "oracle_iou_dense"]
    sel_cols = [c for c in long_df.columns if c.startswith("sel__")]
    long_df_out = long_df[keep + sel_cols]
    long_df_out.to_csv(out_dir / "selector_by_pest_year.csv", index=False)
    print(f"\n[wrote] {out_dir / 'selector_by_pest_year.csv'}")

    # wide: per (pest, metric) cross-year
    wide_metrics = ["fixed_iou", "best_selector_iou", "best_climatology_iou",
                    "oracle_iou_coarse"]
    wide_rows = []
    for pest in args.pests:
        sub = long_df[long_df["pest"] == pest]
        row = {"pest": pest}
        for m in wide_metrics:
            for y in args.years:
                v = sub[sub["year"] == y]
                row[f"{m}_{y}"] = float(v.iloc[0][m]) if len(v) else float("nan")
        # avg over present years
        for m in wide_metrics:
            vals = [row[f"{m}_{y}"] for y in args.years if row[f"{m}_{y}"] == row[f"{m}_{y}"]]
            row[f"{m}_mean"] = float(np.mean(vals)) if vals else float("nan")
        wide_rows.append(row)
    wide = pd.DataFrame(wide_rows)
    wide.to_csv(out_dir / "selector_summary_wide.csv", index=False)
    print(f"[wrote] {out_dir / 'selector_summary_wide.csv'}")

    write_ppt_summary(out_dir / "selector_summary_for_ppt.txt",
                       long_df, failures, args.years, args.pests)
    print(f"[wrote] {out_dir / 'selector_summary_for_ppt.txt'}")
    return 0


def write_ppt_summary(out_path: Path, df: pd.DataFrame,
                       failures: List, years: List[int], pests: List[str]):
    lines: List[str] = []
    lines.append("Stage 2 offset selector — cross-split (8 pests × 3 years)")
    lines.append("=" * 110)
    lines.append("")
    lines.append("Setup")
    lines.append("-" * 110)
    lines.append("  per-(pest, year) V2 ranking selector trained on that year's val sample_grid only")
    lines.append("  test labels never seen by the selector during training or offset selection")
    lines.append("  candidate space coarse {7,14,21,30,45,60} + dense 1..75 (mu interpolated)")
    lines.append("  best_selector = argmax over all learned variants (v1 bin / v2 reg+rnk+clf × coarse+dense)")
    lines.append("")
    if failures:
        lines.append(f"FAILED runs: {failures}")
        lines.append("")

    lines.append("Per (pest, year)")
    lines.append("-" * 110)
    hdr = (f"  {'pest':20s} {'year':>4s}  {'fixed':>6s}  {'best_learn':>10s} "
           f"({'who':<34s})  {'clim':>6s}  {'oracle':>6s}  "
           f"{'d_fix':>6s}  {'d_clim':>6s}  {'b_fx':>4s} {'b_cl':>4s}")
    lines.append(hdr)
    lines.append("  " + "-" * (len(hdr) - 2))
    for pest in pests:
        for year in years:
            sub = df[(df["pest"] == pest) & (df["year"] == year)]
            if sub.empty: continue
            r = sub.iloc[0]
            bfx = "Y" if r["beats_fixed"] else "."
            bcl = "Y" if r["beats_clim"] else "."
            lines.append(
                f"  {pest:20s} {int(year):>4d}  "
                f"{r['fixed_iou']:>6.3f}  {r['best_selector_iou']:>10.3f} "
                f"({str(r['best_selector_name']):<34s})  "
                f"{r['best_climatology_iou']:>6.3f}  {r['oracle_iou_coarse']:>6.3f}  "
                f"{r['selector_minus_fixed']:>+6.3f}  {r['selector_minus_clim']:>+6.3f}  "
                f"{bfx:>4s} {bcl:>4s}"
            )

    lines.append("")
    lines.append("Per-year roll-up (mean over pests)")
    lines.append("-" * 110)
    lines.append(f"  {'year':>4s}  {'n':>3s}  "
                 f"{'fixed':>6s}  {'best_lrn':>8s}  {'clim':>6s}  {'oracle':>6s}  "
                 f"{'beats_fx':>9s}  {'beats_cl':>9s}")
    for year in years:
        sub = df[df["year"] == year]
        if sub.empty: continue
        n = len(sub)
        n_fx = int(sub["beats_fixed"].sum())
        n_cl = int(sub["beats_clim"].sum())
        lines.append(
            f"  {int(year):>4d}  {n:>3d}  "
            f"{sub['fixed_iou'].mean():>6.3f}  {sub['best_selector_iou'].mean():>8.3f}  "
            f"{sub['best_climatology_iou'].mean():>6.3f}  {sub['oracle_iou_coarse'].mean():>6.3f}  "
            f"{n_fx}/{n:<7d} {n_cl}/{n:<7d}"
        )

    lines.append("")
    lines.append("Per-pest stability (years where present)")
    lines.append("-" * 110)
    lines.append(f"  {'pest':20s}  "
                 f"{'fixed_mean':>10s}  {'sel_mean':>9s}  {'clim_mean':>9s}  "
                 f"{'sel_beats_fix':>13s}  {'sel_beats_clim':>14s}")
    for pest in pests:
        sub = df[df["pest"] == pest]
        if sub.empty: continue
        n = len(sub)
        n_fx = int(sub["beats_fixed"].sum())
        n_cl = int(sub["beats_clim"].sum())
        lines.append(
            f"  {pest:20s}  "
            f"{sub['fixed_iou'].mean():>10.3f}  {sub['best_selector_iou'].mean():>9.3f}  "
            f"{sub['best_climatology_iou'].mean():>9.3f}  "
            f"{n_fx}/{n:<12d} {n_cl}/{n:<13d}"
        )

    lines.append("")
    lines.append("Reading guide")
    lines.append("-" * 110)
    lines.append("  d_fix  = best_learned − fixed         (positive = selector improves over fixed offset)")
    lines.append("  d_clim = best_learned − climatology   (positive = selector beats climatology)")
    lines.append("  b_fx, b_cl = per-row binary win flags")
    lines.append("  oracle is LEAKY (uses test labels) — upper bound only, never an operational policy")
    lines.append("")
    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
