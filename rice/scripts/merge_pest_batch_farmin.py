"""Merge Stage 1 batch_rolling outputs across pests into a single CSV.

Walks ``<base>/<pest>/run<seed>/<split>_v<val>_t<test>/`` produced by
``scripts/run_stage1_batch_pests.sh`` and pulls per-row selections:

  - A_baseline / D_history: FAR-min selection on the useful_sweep CSV
    (val recall >= target, then min FAR_val, tie-break F1_val / USEFUL_val /
    lead_mean_val). Same selection used in rolling_seed_stability_farmin.

  - dispatch_group_tau: selection stored in
    group_tau/group_tau_hybrid_summary.json (FAR_val min subject to recall
    target; tie-break (_lead desc, tau_with, tau_no, k)).

Writes:
  1. ``--out_csv``: every (pest, run, split, method, target) row
  2. ``--out_summary_csv`` (optional): one row per (pest, run, split, method)
     at ``--target_for_summary`` (default R>=0.88), restricted to the columns
     pest / run / split / method / target / k / tau_repr /
     precision_test / recall_test / F1_test / FAR_test / lead_median_test /
     no_alert_test / USEFUL_test / n_event_test / fallback.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

import pandas as pd

from rice.scripts.rolling_seed_stability_farmin import (
    USEFUL_SWEEP_REL,
    JSON_METHOD_KEY,
    find_target_entry,
    row_from_ad_csv,
    row_from_dispatch_json,
)


def _target_label(t: float) -> str:
    if t == int(t):
        return f"R>={int(t)}"
    return f"R>={t:.2f}".rstrip("0").rstrip(".")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base", default="rice/outputs_stage1/batch_rolling")
    ap.add_argument("--pest_pattern", default="*",
                    help="Glob of pest names under --base (default: all).")
    ap.add_argument("--exclude_pests", default="",
                    help="Comma-separated pest names to exclude after the "
                         "glob match (e.g. 'BPH2'). Default: none.")
    ap.add_argument("--targets", type=float, nargs="+",
                    default=[0.85, 0.88, 0.90])
    ap.add_argument("--target_for_summary", type=float, default=0.88)
    ap.add_argument("--out_csv", required=True,
                    help="Full per-(pest, run, split, method, target) CSV.")
    ap.add_argument("--out_summary_csv", default=None,
                    help="Optional simplified summary CSV restricted to "
                         "--target_for_summary.")
    args = ap.parse_args()

    excluded = {s.strip() for s in str(args.exclude_pests).split(",") if s.strip()}
    pest_roots = sorted(
        p for p in glob.glob(os.path.join(args.base, args.pest_pattern))
        if os.path.isdir(p) and os.path.basename(p) not in {"_summary", "logs"}
        and os.path.basename(p) not in excluded
    )
    if excluded:
        print(f"[merge] excluded pests: {sorted(excluded)}")
    if not pest_roots:
        print(f"[merge] no pest dirs under {args.base}/{args.pest_pattern}",
              file=sys.stderr)
        return 1

    rows: list[dict] = []
    missing: list[str] = []
    for pest_dir in pest_roots:
        pest = os.path.basename(pest_dir)
        run_dirs = sorted(glob.glob(os.path.join(pest_dir, "run*", "split*_v*_t*")))
        if not run_dirs:
            missing.append(f"{pest}: no run*/split* subdirs")
            continue
        for rd in run_dirs:
            mr = re.search(r"/run(\d+)/", rd + "/")
            ms = re.search(r"(split\d+)", os.path.basename(rd))
            run = int(mr.group(1)) if mr else -1
            split = ms.group(1) if ms else "-"

            a_csv = os.path.join(rd, USEFUL_SWEEP_REL["A_baseline"])
            d_csv = os.path.join(rd, USEFUL_SWEEP_REL["D_history"])
            gt_json = os.path.join(rd, "group_tau", "group_tau_hybrid_summary.json")

            # A / D from useful_sweep CSVs
            for method, csv_path in (("A_baseline", a_csv), ("D_history", d_csv)):
                if not os.path.isfile(csv_path):
                    missing.append(f"{pest}/run{run}/{split}/{method}: missing {csv_path}")
                    continue
                for t in args.targets:
                    lbl = _target_label(t)
                    row = row_from_ad_csv(split, run, method, t, lbl, csv_path)
                    if row is None:
                        missing.append(f"{pest}/run{run}/{split}/{method}/{lbl}: empty sweep CSV")
                        continue
                    row["pest"] = pest
                    rows.append(row)

            # dispatch_group_tau from JSON
            if not os.path.isfile(gt_json):
                missing.append(f"{pest}/run{run}/{split}/dispatch_group_tau: missing {gt_json}")
                continue
            try:
                data = json.load(open(gt_json))
            except Exception as e:
                missing.append(f"{pest}/run{run}/{split}/dispatch: json load fail {e}")
                continue
            sel = data.get("selections", {}) or {}
            for t in args.targets:
                found = find_target_entry(sel, t)
                if not found:
                    missing.append(f"{pest}/run{run}/{split}/dispatch: no key for {t}")
                    continue
                tgt_label, methods_at = found
                entry = methods_at.get(JSON_METHOD_KEY["dispatch_group_tau"])
                if entry is None:
                    missing.append(f"{pest}/run{run}/{split}/{tgt_label}: dispatch entry missing")
                    continue
                row = row_from_dispatch_json(split, run, t, tgt_label, entry)
                row["pest"] = pest
                rows.append(row)

    if not rows:
        print(f"[merge] no rows produced. {len(missing)} missing entries.",
              file=sys.stderr)
        for m in missing[:30]:
            print(f"  - {m}", file=sys.stderr)
        return 1

    df = pd.DataFrame(rows)
    # Normalize column naming: 'seed' (from helpers) -> 'run' (batch terminology)
    if "seed" in df.columns and "run" not in df.columns:
        df = df.rename(columns={"seed": "run"})
    front = ["pest", "run", "split", "method", "target",
             "k", "tau", "tau_no", "tau_with"]
    others = [c for c in df.columns if c not in front]
    df = df[front + others]
    df.sort_values(["pest", "target", "split", "run", "method"], inplace=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"[merge] wrote {args.out_csv}  rows={len(df)}  "
          f"pests={df['pest'].nunique()}")

    if args.out_summary_csv:
        lbl = _target_label(args.target_for_summary)
        sub = df[df["target"] == lbl].copy()
        if sub.empty:
            print(f"[merge] no rows at target={lbl!r} for summary", file=sys.stderr)
        else:
            def _tau_repr(r):
                if r["method"] == "dispatch_group_tau":
                    return f"no={r.get('tau_no')}/with={r.get('tau_with')}"
                return str(r.get("tau", "-"))
            sub["tau_repr"] = sub.apply(_tau_repr, axis=1)
            keep = ["pest", "run", "split", "method", "target", "k", "tau_repr",
                    "precision_test", "recall_test", "F1_test", "FAR_test",
                    "lead_median_test", "no_alert_test", "USEFUL_test",
                    "n_event_test", "fallback"]
            keep = [c for c in keep if c in sub.columns]
            sub = sub[keep]
            Path(args.out_summary_csv).parent.mkdir(parents=True, exist_ok=True)
            sub.to_csv(args.out_summary_csv, index=False)
            print(f"[merge] wrote summary {args.out_summary_csv}  rows={len(sub)}")

    if missing:
        print(f"\n[merge] missing entries ({len(missing)}):", file=sys.stderr)
        for m in missing[:50]:
            print(f"  - {m}", file=sys.stderr)
        if len(missing) > 50:
            print(f"  ... +{len(missing)-50} more", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
