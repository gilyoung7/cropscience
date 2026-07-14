"""Stage-2 late-prediction diagnostic (diagnosis only; does not touch models/selectors).

Reads per-sample selector evaluation CSVs (one per pest/year/selector variant) and adds
per-sample "late prediction" flags, then writes pest x offset summaries and late-case
example tables.

Source per_sample.csv schema (from run_viz_interval_selector):
    alert_tstar, selected_offset, mu, pred_L, pred_R, true_L_plus_1, true_R, true_mid, iou, ...
Pest / year / selector variant are parsed from the parent directory name
    <pest>_<year>_<selector_variant>/per_sample.csv

Run:
    python -m rice.scripts.phase_t_stage2_late_diagnostic
"""
from __future__ import annotations

import argparse
import glob
import os
import re
from pathlib import Path

import pandas as pd

from rice.configs.base import RICE_ROOT

DEFAULT_GLOB = str(RICE_ROOT / "outputs" / "viz" / "viz_selector" / "**" / "per_sample.csv")
DEFAULT_OUT = str(RICE_ROOT / "outputs" / "diag" / "stage2_late_prediction")

# <pest>_<4-digit-year>_<selector variant>  (pest slug may contain underscores)
DIR_RE = re.compile(r"^(?P<pest>.+)_(?P<year>\d{4})_(?P<variant>v\d.*)$")

# columns the diagnostic depends on
NEEDED = ["alert_tstar", "selected_offset", "mu", "pred_L", "pred_R",
          "true_L_plus_1", "true_R", "true_mid", "iou"]


def parse_source(per_sample_path: str) -> dict:
    name = Path(per_sample_path).parent.name
    m = DIR_RE.match(name)
    if not m:
        return {"pest": name, "src_year": None, "selector_variant": None, "source_dir": name}
    return {"pest": m.group("pest"), "src_year": int(m.group("year")),
            "selector_variant": m.group("variant"), "source_dir": name}


def load_all(glob_pat: str) -> pd.DataFrame:
    files = sorted(glob.glob(glob_pat, recursive=True))
    if not files:
        raise SystemExit(f"No per_sample.csv matched: {glob_pat}")
    frames = []
    for f in files:
        df = pd.read_csv(f)
        missing = [c for c in NEEDED if c not in df.columns]
        if missing:
            print(f"  WARN skip (missing {missing}): {f}")
            continue
        meta = parse_source(f)
        for k, v in meta.items():
            df[k] = v
        frames.append(df)
    if not frames:
        raise SystemExit("No usable per_sample.csv after schema check.")
    print(f"Loaded {len(frames)} files / {sum(len(x) for x in frames)} rows")
    return pd.concat(frames, ignore_index=True)


def add_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # true_start_doy = labelled interval start (true_L + 1, stored as true_L_plus_1)
    true_start = df["true_L_plus_1"]
    df["eval_doy"] = df["alert_tstar"] + df["selected_offset"]
    df["late_eval"] = df["eval_doy"] > true_start
    df["late_mu"] = df["mu"] > df["true_mid"]
    df["late_PI_start"] = df["pred_L"] > true_start
    df["no_overlap"] = df["iou"] == 0
    # extra helpers used by summaries
    df["MAE_center"] = (df["mu"] - df["true_mid"]).abs()
    df["PI_hit"] = (df["pred_L"] <= df["true_mid"]) & (df["true_mid"] <= df["pred_R"])
    return df


def summarize(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    g = df.groupby(group_cols, dropna=False)
    out = g.agg(
        n=("late_eval", "size"),
        late_eval_rate=("late_eval", "mean"),
        late_mu_rate=("late_mu", "mean"),
        late_PI_start_rate=("late_PI_start", "mean"),
        no_overlap_rate=("no_overlap", "mean"),
        mean_MAE_center=("MAE_center", "mean"),
        mean_IoU=("iou", "mean"),
        PI_hit_rate=("PI_hit", "mean"),
    ).reset_index()
    rate_cols = ["late_eval_rate", "late_mu_rate", "late_PI_start_rate",
                 "no_overlap_rate", "mean_MAE_center", "mean_IoU", "PI_hit_rate"]
    out[rate_cols] = out[rate_cols].round(4)
    return out


def write_no_overwrite(df: pd.DataFrame, path: Path, force: bool):
    if path.exists() and not force:
        raise SystemExit(f"Refuse to overwrite existing file: {path}\n"
                         f"Use --force or remove it first.")
    df.to_csv(path, index=False)
    print(f"  wrote {path}  ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default=DEFAULT_GLOB,
                    help="glob for per_sample.csv files")
    ap.add_argument("--out-dir", default=DEFAULT_OUT)
    ap.add_argument("--focus-offsets", default="45,60",
                    help="comma offsets for the focused late_eval example table")
    ap.add_argument("--force", action="store_true",
                    help="allow overwriting existing output files")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = load_all(args.glob)
    df = add_diagnostics(raw)

    # 1) per-sample with diagnostics
    persample_cols = (["pest", "src_year", "selector_variant", "source_dir",
                       "sample_id", "site", "year", "selector",
                       "alert_tstar", "selected_offset", "eval_doy",
                       "mu", "pred_L", "pred_R",
                       "true_L_plus_1", "true_mid", "true_R", "iou",
                       "MAE_center", "PI_hit",
                       "late_eval", "late_mu", "late_PI_start", "no_overlap"])
    persample_cols = [c for c in persample_cols if c in df.columns]
    write_no_overwrite(df[persample_cols], out_dir / "per_sample_late_diagnostic.csv", args.force)

    # 2) summaries
    write_no_overwrite(summarize(df, ["pest", "selected_offset"]),
                       out_dir / "summary_by_pest_offset.csv", args.force)
    write_no_overwrite(summarize(df, ["pest"]),
                       out_dir / "summary_by_pest.csv", args.force)
    write_no_overwrite(summarize(df, ["selected_offset"]),
                       out_dir / "summary_by_offset.csv", args.force)

    # 3) late_eval example tables
    late = df[df["late_eval"]].copy().sort_values(["pest", "selected_offset", "sample_id"])
    write_no_overwrite(late[persample_cols], out_dir / "late_eval_examples.csv", args.force)

    focus = [int(x) for x in args.focus_offsets.split(",") if x.strip()]
    late_focus = late[late["selected_offset"].isin(focus)]
    write_no_overwrite(late_focus[persample_cols],
                       out_dir / f"late_eval_examples_off{'_'.join(map(str, focus))}.csv",
                       args.force)

    # console digest
    print("\n=== late_eval rate by pest x offset (focus offsets) ===")
    sub = summarize(df[df["selected_offset"].isin(focus)], ["pest", "selected_offset"])
    with pd.option_context("display.width", 200, "display.max_rows", None):
        print(sub.to_string(index=False))
    print(f"\ntotal samples={len(df)}  late_eval={int(df['late_eval'].sum())} "
          f"({df['late_eval'].mean():.1%})  no_overlap={df['no_overlap'].mean():.1%}")


if __name__ == "__main__":
    main()
