#!/usr/bin/env python3
"""Cross-split summary for Stage 2 baseline_asym_mse lead_v3 (split1/2/3).

Reads the per-split selection.csv files produced by
run_stage2_split_pest_best_gate_batch.sh (one out_root per split) and merges
them into a unified comparison table. For each pest × test_year row:

  - selected gate (method / run) — picked on val-only
  - lead_v3 test IoU @ val-selected offset (canonical operational metric)
  - lead_v3 test oracle IoU (coarse {7,14,21,30,45,60} sample-wise max)
  - best climatology test IoU @ val-selected offset
  - delta lead_v3 vs climatology, win/loss

Outputs (under --out_dir):
  - cross_split_per_pest.csv       (one row per pest × test_year)
  - cross_split_summary_for_ppt.txt (human-readable)

Defaults: looks for the conventional out_roots produced by the wrapper.
Override individual paths with --out_root_2022, --out_root_2023,
--out_root_2024 if you placed the runs somewhere else.
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path
from typing import Dict, List
import pandas as pd


DEFAULT_OUT_ROOTS = {
    2022: "rice/outputs_stage2_batch_2022_baseline",
    2023: "rice/outputs_stage2_batch_2023_baseline",
    2024: "rice/outputs_stage2_batch_2024_bestgate",
}
SCORE_COL = "test_IoU_overall_n_total_at_val_offset"
ORACLE_COL = "test_oracle_IoU_overall_n_total"
PESTS_ORDER = [
    "BPH", "WBPH",
    "bacterial_blight", "blast", "brown_spot",
    "rice_stem_borer_1", "rice_stem_borer_2",
    "sheath_blight",
]


def load_split(out_root: Path, test_year: int) -> pd.DataFrame:
    sel_path = out_root / "_summary" / "all_pests_selection.csv"
    gate_path_glob = list((out_root / "_summary").glob("stage1_gate_selection_*.csv"))
    if not sel_path.exists():
        print(f"[warn] missing {sel_path}; split test_year={test_year} skipped")
        return pd.DataFrame()
    sel = pd.read_csv(sel_path)
    gate = pd.read_csv(gate_path_glob[0]) if gate_path_glob else pd.DataFrame()

    rows = []
    for pest, sub in sel.groupby("pest"):
        lead = sub[sub["model_kind"].astype(str).str.startswith("lead_v3", na=False)]
        clim = sub[sub["model_kind"].astype(str).str.startswith("clim_", na=False)]
        lead = lead.dropna(subset=[SCORE_COL]) if not lead.empty else lead
        clim = clim.dropna(subset=[SCORE_COL]) if not clim.empty else clim
        lead_score = float(lead.iloc[0][SCORE_COL]) if len(lead) else float("nan")
        lead_oracle = float(lead.iloc[0][ORACLE_COL]) if len(lead) and ORACLE_COL in lead.columns else float("nan")
        lead_val_offset = (int(lead.iloc[0]["val_best_offset"])
                           if len(lead) and "val_best_offset" in lead.columns
                           else -1)
        if len(clim):
            clim = clim.sort_values(SCORE_COL, ascending=False)
            best_clim_kind = str(clim.iloc[0]["model_kind"])
            best_clim_score = float(clim.iloc[0][SCORE_COL])
        else:
            best_clim_kind, best_clim_score = "(none)", float("nan")
        if not gate.empty and pest in gate["pest"].values:
            g = gate[gate["pest"] == pest].iloc[0]
            sel_method = str(g["selected_method"])
            sel_run = int(g["selected_run"])
            val_recall = float(g["val_recall"])
            val_FAR = float(g["val_FAR"])
            test_recall = float(g["test_recall"])
            test_FAR = float(g["test_FAR"])
        else:
            sel_method, sel_run = "?", -1
            val_recall = val_FAR = test_recall = test_FAR = float("nan")
        delta = (lead_score - best_clim_score
                 if (lead_score == lead_score and best_clim_score == best_clim_score)
                 else float("nan"))
        rows.append({
            "pest": pest,
            "test_year": test_year,
            "selected_method": sel_method,
            "selected_run": sel_run,
            "val_recall": val_recall, "val_FAR": val_FAR,
            "test_recall": test_recall, "test_FAR": test_FAR,
            "lead_v3_val_offset": lead_val_offset,
            "lead_v3_test_iou": lead_score,
            "lead_v3_test_oracle": lead_oracle,
            "best_clim_kind": best_clim_kind,
            "best_clim_test_iou": best_clim_score,
            "delta_lead_minus_clim": delta,
            "lead_beats_clim": bool(delta == delta and delta > 0),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_root_2022", default=DEFAULT_OUT_ROOTS[2022])
    ap.add_argument("--out_root_2023", default=DEFAULT_OUT_ROOTS[2023])
    ap.add_argument("--out_root_2024", default=DEFAULT_OUT_ROOTS[2024])
    ap.add_argument("--out_dir", default="rice/outputs_stage2_cross_split_summary",
                    help="Directory to write the merged CSV + summary TXT.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dfs = []
    for ty, root in [(2022, args.out_root_2022),
                     (2023, args.out_root_2023),
                     (2024, args.out_root_2024)]:
        df = load_split(Path(root), ty)
        if not df.empty:
            print(f"[load] test_year={ty} rows={len(df)} from {root}")
            dfs.append(df)
    if not dfs:
        print("[abort] no per-split data found", file=sys.stderr)
        return 1

    merged = pd.concat(dfs, ignore_index=True)
    merged["__o"] = merged["pest"].map(lambda p: PESTS_ORDER.index(p) if p in PESTS_ORDER else 99)
    merged = merged.sort_values(["__o", "test_year"]).drop(columns="__o")
    out_csv = out_dir / "cross_split_per_pest.csv"
    merged.to_csv(out_csv, index=False)
    print(f"[wrote] {out_csv}  rows={len(merged)}")

    write_summary_txt(out_dir / "cross_split_summary_for_ppt.txt", merged)
    print(f"[wrote] {out_dir / 'cross_split_summary_for_ppt.txt'}")
    return 0


def write_summary_txt(out_path: Path, df: pd.DataFrame):
    lines: List[str] = []
    lines.append("Stage 2 baseline_asym_mse lead_v3 — cross-split comparison")
    lines.append("=" * 96)
    lines.append("")
    lines.append("All splits use the SAME pipeline:")
    lines.append("  * per-pest val-only best-gate selection (D_history / dispatch_group_tau / A_baseline × 3 runs)")
    lines.append("  * Stage 2 mu_mode=lead_from_alert, asym_mse loss, sigma=5 fixed")
    lines.append("  * sample_grid offsets {7,14,21,30,45,60}; offset = val-selected best")
    lines.append("")
    lines.append("Per-pest per-split table")
    lines.append("-" * 96)
    header = (f"  {'pest':20s} {'year':>4s}  {'gate(method,run)':<24s}  "
              f"{'val_R':>5s} {'val_FAR':>7s} {'test_R':>6s} {'test_FAR':>8s}  "
              f"{'lead@off':>9s} {'oracle':>7s}  {'best_clim':>9s} {'delta':>7s}  win")
    lines.append(header)
    lines.append("  " + "-" * (len(header) - 2))
    for _, r in df.iterrows():
        gate = f"{r['selected_method']}/{r['selected_run']}"
        win = "✓" if r["lead_beats_clim"] else " "
        lines.append(
            f"  {r['pest']:20s} {int(r['test_year']):>4d}  {gate:<24s}  "
            f"{r['val_recall']:>5.2f} {r['val_FAR']:>7.3f} "
            f"{r['test_recall']:>6.2f} {r['test_FAR']:>8.3f}  "
            f"{r['lead_v3_test_iou']:>9.3f} {r['lead_v3_test_oracle']:>7.3f}  "
            f"{r['best_clim_test_iou']:>9.3f} {r['delta_lead_minus_clim']:>+7.3f}  {win}"
        )

    lines.append("")
    lines.append("Per-year roll-up (mean over pests)")
    lines.append("-" * 96)
    lines.append(f"  {'year':>4s}  {'pests':>5s}  "
                 f"{'lead@off_mean':>13s}  {'oracle_mean':>11s}  "
                 f"{'clim_mean':>9s}  {'lead>clim n/N':>14s}")
    for ty, sub in df.groupby("test_year"):
        n = len(sub)
        n_win = int(sub["lead_beats_clim"].sum())
        lines.append(
            f"  {int(ty):>4d}  {n:>5d}  "
            f"{sub['lead_v3_test_iou'].mean():>13.3f}  "
            f"{sub['lead_v3_test_oracle'].mean():>11.3f}  "
            f"{sub['best_clim_test_iou'].mean():>9.3f}  "
            f"{n_win}/{n:>10d}"
        )

    lines.append("")
    lines.append("Per-pest year-over-year stability (where all 3 years present)")
    lines.append("-" * 96)
    lines.append(f"  {'pest':20s}  {'2022':>7s}  {'2023':>7s}  {'2024':>7s}  "
                 f"{'mean':>6s}  {'std':>6s}  {'clim22':>7s}  {'clim23':>7s}  {'clim24':>7s}")
    for pest in PESTS_ORDER:
        sub = df[df["pest"] == pest]
        if len(sub) < 1: continue
        by_year = {int(r["test_year"]): r for _, r in sub.iterrows()}
        v22 = by_year.get(2022, {}).get("lead_v3_test_iou", float("nan"))
        v23 = by_year.get(2023, {}).get("lead_v3_test_iou", float("nan"))
        v24 = by_year.get(2024, {}).get("lead_v3_test_iou", float("nan"))
        c22 = by_year.get(2022, {}).get("best_clim_test_iou", float("nan"))
        c23 = by_year.get(2023, {}).get("best_clim_test_iou", float("nan"))
        c24 = by_year.get(2024, {}).get("best_clim_test_iou", float("nan"))
        vals = [x for x in (v22, v23, v24) if x == x]
        m = sum(vals) / len(vals) if vals else float("nan")
        s = (sum((x - m) ** 2 for x in vals) / len(vals)) ** 0.5 if vals else float("nan")
        def f(x): return f"{x:>7.3f}" if x == x else f"{'-':>7s}"
        lines.append(f"  {pest:20s}  {f(v22)}  {f(v23)}  {f(v24)}  "
                     f"{m:>6.3f}  {s:>6.3f}  {f(c22)}  {f(c23)}  {f(c24)}")

    out_path.write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    sys.exit(main())
