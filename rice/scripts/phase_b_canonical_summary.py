"""Canonical Stage 2 comparison using n_total denominator and val-selected offset.

Goal: eliminate two sources of confusion that contaminated earlier reports.

  (a) Mixed denominators. mu_diag's per-offset "IoU_overall" used a per-offset
      matched count as the denominator, while phase_r's oracle summary used
      n_total (the full alerted cohort). The two are not comparable; the former
      makes coverage-poor offsets look stronger than they are. This script
      always uses n_total as the canonical denominator.

  (b) Test-side offset cherry-picking. Picking the offset that maximizes IoU
      on test is leakage. This script picks the best offset on VAL
      (validation sample_grid CSV) and applies that same offset to TEST.

Inputs:
  --entry "label|val=PATH|test=PATH[|n_total_val=N|n_total_test=N]"
    repeatable. PATH points to a sample_grid CSV produced by
    phase_r_oracle_iou (val grid requires running phase_r with
    --eval_split val first). n_total defaults to unique sample_id in the
    grid.

Outputs:
  --out_per_offset PATH        # one row per (model, split, offset)
  --out_selection  PATH        # one row per model: val-best offset + test
                               # IoU at that offset + test oracle reference
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def per_offset_metrics(df: pd.DataFrame, n_total: int) -> pd.DataFrame:
    """For each offset: n_match, coverage, IoU_matched (matched mean),
    IoU_overall_n_total (= IoU_matched * coverage = sum(iou)/n_total)."""
    rows = []
    for o in sorted(df["offset"].unique()):
        g = df[(df["offset"] == o) & (df["matched"] == True)]  # noqa: E712
        n_m = int(len(g))
        if n_m == 0:
            rows.append({
                "offset": int(o), "n_match": 0,
                "coverage": 0.0,
                "IoU_matched": float("nan"),
                "IoU_overall_n_total": 0.0,
            })
            continue
        iou_m = float(g["iou_matched"].mean())
        coverage = n_m / max(int(n_total), 1)
        rows.append({
            "offset": int(o), "n_match": n_m,
            "coverage": coverage,
            "IoU_matched": iou_m,
            "IoU_overall_n_total": iou_m * coverage,
        })
    return pd.DataFrame(rows)


def oracle_metrics(df: pd.DataFrame, n_total: int) -> dict:
    """Per-sample best offset (oracle); both denominators reported.
    n_total denominator is the canonical one.
    """
    m = df[df["matched"] == True]  # noqa: E712
    if m.empty:
        return {"oracle_IoU_matched": float("nan"),
                "oracle_IoU_overall_n_total": 0.0,
                "oracle_n_match_any": 0}
    best = (m.sort_values("iou_matched", ascending=False)
              .drop_duplicates("sample_id", keep="first"))
    n_best = int(len(best))
    iou_m = float(best["iou_matched"].mean())
    return {
        "oracle_IoU_matched": iou_m,
        "oracle_IoU_overall_n_total": iou_m * n_best / max(int(n_total), 1),
        "oracle_n_match_any": n_best,
    }


def parse_entry(s: str) -> tuple[str, dict]:
    parts = s.split("|")
    if len(parts) < 2:
        raise ValueError(f"bad --entry {s!r}; need 'label|key=val|...' form")
    label = parts[0].strip()
    kv: dict = {}
    for p in parts[1:]:
        if "=" not in p:
            raise ValueError(f"bad key=val token in entry: {p!r}")
        k, v = p.split("=", 1)
        kv[k.strip()] = v.strip()
    return label, kv


def _load_grid(path: str, expected_label: str | None) -> pd.DataFrame:
    df = pd.read_csv(path)
    # If file mixes multiple models, restrict to the label if present.
    if expected_label is not None and "model" in df.columns:
        unique = set(str(x) for x in df["model"].dropna().unique())
        if expected_label in unique:
            df = df[df["model"].astype(str) == expected_label]
    return df


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--entry", action="append", required=True,
                    help="Repeatable: 'label|val=PATH|test=PATH[|n_total_val=N|n_total_test=N]'")
    ap.add_argument("--out_per_offset", required=True,
                    help="CSV: per (model, split, offset) row.")
    ap.add_argument("--out_selection", required=True,
                    help="CSV: one row per model summarizing val-selected offset "
                         "applied to test plus test oracle reference.")
    args = ap.parse_args()

    per_off_rows: list[dict] = []
    sel_rows: list[dict] = []

    for raw in args.entry:
        label, kv = parse_entry(raw)
        print(f"\n========== model: {label!r}")

        val_csv = kv.get("val")
        test_csv = kv.get("test")
        val_off_df = None
        n_total_val = None
        val_oracle = {}
        if val_csv:
            if not os.path.isfile(val_csv):
                print(f"  [skip val] missing: {val_csv}")
            else:
                df_val = _load_grid(val_csv, label)
                n_total_val = int(kv.get("n_total_val", df_val["sample_id"].nunique()))
                val_off_df = per_offset_metrics(df_val, n_total_val)
                val_oracle = oracle_metrics(df_val, n_total_val)
                print(f"  val  : csv={val_csv}  n_total={n_total_val}  "
                      f"oracle IoU_matched={val_oracle['oracle_IoU_matched']:.4f}  "
                      f"oracle IoU_overall_n_total={val_oracle['oracle_IoU_overall_n_total']:.4f}")
                for _, r in val_off_df.iterrows():
                    per_off_rows.append({
                        "model": label, "split": "val", "n_total": n_total_val,
                        **r.to_dict(),
                        "oracle_IoU_matched": val_oracle["oracle_IoU_matched"],
                        "oracle_IoU_overall_n_total": val_oracle["oracle_IoU_overall_n_total"],
                    })

        test_off_df = None
        n_total_test = None
        test_oracle = {}
        if test_csv:
            if not os.path.isfile(test_csv):
                print(f"  [skip test] missing: {test_csv}")
            else:
                df_test = _load_grid(test_csv, label)
                n_total_test = int(kv.get("n_total_test", df_test["sample_id"].nunique()))
                test_off_df = per_offset_metrics(df_test, n_total_test)
                test_oracle = oracle_metrics(df_test, n_total_test)
                print(f"  test : csv={test_csv}  n_total={n_total_test}  "
                      f"oracle IoU_matched={test_oracle['oracle_IoU_matched']:.4f}  "
                      f"oracle IoU_overall_n_total={test_oracle['oracle_IoU_overall_n_total']:.4f}")
                for _, r in test_off_df.iterrows():
                    per_off_rows.append({
                        "model": label, "split": "test", "n_total": n_total_test,
                        **r.to_dict(),
                        "oracle_IoU_matched": test_oracle["oracle_IoU_matched"],
                        "oracle_IoU_overall_n_total": test_oracle["oracle_IoU_overall_n_total"],
                    })

        # Selection: val-best offset → applied to test
        sel: dict = {"model": label}
        best_val_off = None
        if val_off_df is not None:
            valid_val = val_off_df[val_off_df["n_match"] > 0]
            if len(valid_val):
                ix = valid_val["IoU_overall_n_total"].idxmax()
                best_val_off = int(valid_val.loc[ix, "offset"])
                sel["val_best_offset"] = best_val_off
                sel["val_n_total"] = int(n_total_val)
                sel["val_n_match_at_best"] = int(valid_val.loc[ix, "n_match"])
                sel["val_coverage_at_best"] = float(valid_val.loc[ix, "coverage"])
                sel["val_IoU_matched_at_best"] = float(valid_val.loc[ix, "IoU_matched"])
                sel["val_IoU_overall_n_total_at_best"] = float(valid_val.loc[ix, "IoU_overall_n_total"])
                sel["val_oracle_IoU_matched"] = val_oracle.get("oracle_IoU_matched", float("nan"))
                sel["val_oracle_IoU_overall_n_total"] = val_oracle.get("oracle_IoU_overall_n_total", float("nan"))
        if test_off_df is not None and best_val_off is not None:
            row_at = test_off_df[test_off_df["offset"] == best_val_off]
            if len(row_at):
                r0 = row_at.iloc[0]
                sel["test_n_total"] = int(n_total_test)
                sel["test_n_match_at_val_offset"] = int(r0["n_match"])
                sel["test_coverage_at_val_offset"] = float(r0["coverage"])
                sel["test_IoU_matched_at_val_offset"] = float(r0["IoU_matched"])
                sel["test_IoU_overall_n_total_at_val_offset"] = float(r0["IoU_overall_n_total"])
            else:
                sel["note"] = f"val_best_offset={best_val_off} not present in test grid"
            sel["test_oracle_IoU_matched"] = test_oracle.get("oracle_IoU_matched", float("nan"))
            sel["test_oracle_IoU_overall_n_total"] = test_oracle.get("oracle_IoU_overall_n_total", float("nan"))
        # Also record test-side best (for transparency; do NOT use for ranking)
        if test_off_df is not None:
            valid_te = test_off_df[test_off_df["n_match"] > 0]
            if len(valid_te):
                ix = valid_te["IoU_overall_n_total"].idxmax()
                sel["test_best_offset_LEAKY"] = int(valid_te.loc[ix, "offset"])
                sel["test_IoU_overall_n_total_at_test_best_LEAKY"] = float(valid_te.loc[ix, "IoU_overall_n_total"])
        sel_rows.append(sel)

    # Save
    Path(args.out_per_offset).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_selection).parent.mkdir(parents=True, exist_ok=True)
    per_off = pd.DataFrame(per_off_rows)
    if not per_off.empty:
        front = ["model", "split", "offset", "n_match", "n_total",
                 "coverage", "IoU_matched", "IoU_overall_n_total",
                 "oracle_IoU_matched", "oracle_IoU_overall_n_total"]
        front = [c for c in front if c in per_off.columns]
        per_off = per_off[front + [c for c in per_off.columns if c not in front]]
        per_off.sort_values(["model", "split", "offset"], inplace=True)
    per_off.to_csv(args.out_per_offset, index=False)
    print(f"\n# wrote {args.out_per_offset}  rows={len(per_off)}")

    sel_df = pd.DataFrame(sel_rows)
    sel_df.to_csv(args.out_selection, index=False)
    print(f"# wrote {args.out_selection}  rows={len(sel_df)}")

    if not sel_df.empty:
        print("\n" + "=" * 78)
        print("FINAL — val-selected offset applied to test (canonical IoU_overall_n_total)")
        print("=" * 78)
        show_cols = ["model", "val_best_offset",
                     "val_IoU_overall_n_total_at_best",
                     "test_IoU_overall_n_total_at_val_offset",
                     "test_oracle_IoU_overall_n_total",
                     "test_best_offset_LEAKY",
                     "test_IoU_overall_n_total_at_test_best_LEAKY"]
        show_cols = [c for c in show_cols if c in sel_df.columns]
        sel_show = sel_df[show_cols].copy()
        # rank by canonical
        if "test_IoU_overall_n_total_at_val_offset" in sel_show.columns:
            sel_show = sel_show.sort_values(
                "test_IoU_overall_n_total_at_val_offset", ascending=False,
            )
        with pd.option_context("display.width", 200,
                                "display.max_columns", 30,
                                "display.float_format", "{:.4f}".format):
            print(sel_show.to_string(index=False))

    return 0


if __name__ == "__main__":
    sys.exit(main())
