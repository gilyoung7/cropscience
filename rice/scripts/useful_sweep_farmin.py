"""Pick FAR-min operating points from useful_sweep CSVs.

Policy: among rows with recall_val >= --min-recall, choose the one with the
smallest FAR_val. Tie-break by F1_val desc, then lead_mean_val desc, then
recall_val desc. If no row satisfies the recall constraint, fall back to the
max-recall row (matches the stage-1 baseline strict fallback).

Default pattern targets the rolling-split layout:
    <base>/sheath_blight_rolling_split*/[AD]/useful_pareto/useful_sweep_*.csv
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from typing import Optional


REPORT_COLS = (
    "split",
    "branch",
    "k",
    "tau",
    "recall_val",
    "FAR_val",
    "F1_val",
    "lead_mean_val",
    "recall_test",
    "FAR_test",
    "F1_test",
    "lead_mean_test",
    "n_event_test",
    "fallback",
    "source",
)


def parse_meta_from_path(path: str) -> tuple[str, str]:
    """Extract (split_tag, branch) from a rolling-split CSV path.

    split_tag: e.g. 'split1', 'split2', 'split3' from a parent dir match.
    branch:    'A' or 'D' from the immediate parent-of-parent dir, when present.
    """
    parts = path.replace("\\", "/").split("/")
    split_tag = "-"
    branch = "-"
    for p in parts:
        m = re.search(r"(split\d+)", p)
        if m:
            split_tag = m.group(1)
            break
    # branch directory is two levels up from the CSV: <branch>/useful_pareto/<csv>
    if len(parts) >= 3 and parts[-2] == "useful_pareto":
        cand = parts[-3]
        if cand in {"A", "D"}:
            branch = cand
    return split_tag, branch


def _to_float(x: str) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def pick_farmin_row(
    rows: list[dict], min_recall: float
) -> tuple[dict, bool]:
    """Return (chosen_row, fallback_used)."""
    feasible = [
        r for r in rows
        if (_to_float(r.get("recall_val", "")) or -1.0) >= min_recall
    ]
    fallback = False
    if not feasible:
        fallback = True
        feasible = rows
        # max-recall fallback, tie-break by F1_val
        key = lambda r: (
            -(_to_float(r.get("recall_val", "")) or -1.0),
            -(_to_float(r.get("F1_val", "")) or -1.0),
            (_to_float(r.get("FAR_val", "")) or float("inf")),
        )
    else:
        key = lambda r: (
            (_to_float(r.get("FAR_val", "")) or float("inf")),
            -(_to_float(r.get("F1_val", "")) or -1.0),
            -(_to_float(r.get("lead_mean_val", "")) or -1.0),
            -(_to_float(r.get("recall_val", "")) or -1.0),
        )
    chosen = min(feasible, key=key)
    return chosen, fallback


def process_csv(path: str, min_recall: float) -> Optional[dict]:
    with open(path, "r", newline="") as fh:
        reader = csv.DictReader(fh)
        rows = list(reader)
    if not rows:
        return None
    chosen, fallback = pick_farmin_row(rows, min_recall)
    split_tag, branch = parse_meta_from_path(path)
    out = {
        "split": split_tag,
        "branch": branch,
        "k": chosen.get("k", ""),
        "tau": chosen.get("tau", ""),
        "recall_val": chosen.get("recall_val", ""),
        "FAR_val": chosen.get("FAR_val", ""),
        "F1_val": chosen.get("F1_val", ""),
        "lead_mean_val": chosen.get("lead_mean_val", ""),
        "recall_test": chosen.get("recall_test", ""),
        "FAR_test": chosen.get("FAR_test", ""),
        "F1_test": chosen.get("F1_test", ""),
        "lead_mean_test": chosen.get("lead_mean_test", ""),
        "n_event_test": chosen.get("n_event_test", ""),
        "fallback": "yes" if fallback else "no",
        "source": os.path.relpath(path),
    }
    return out


def format_table(rows: list[dict]) -> str:
    if not rows:
        return "(no rows)"
    # Compute column widths
    widths = {c: len(c) for c in REPORT_COLS}
    fmt_rows: list[dict] = []
    for r in rows:
        f = {}
        for c in REPORT_COLS:
            v = r.get(c, "")
            if c in {
                "tau", "recall_val", "FAR_val", "F1_val", "lead_mean_val",
                "recall_test", "FAR_test", "F1_test", "lead_mean_test",
            }:
                fv = _to_float(v)
                f[c] = f"{fv:.4f}" if fv is not None else str(v)
            else:
                f[c] = str(v)
            widths[c] = max(widths[c], len(f[c]))
        fmt_rows.append(f)
    header = "  ".join(c.ljust(widths[c]) for c in REPORT_COLS)
    sep = "  ".join("-" * widths[c] for c in REPORT_COLS)
    lines = [header, sep]
    for f in fmt_rows:
        lines.append("  ".join(f[c].ljust(widths[c]) for c in REPORT_COLS))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        default="rice/outputs_stage1",
        help="Base directory containing rolling-split run dirs.",
    )
    ap.add_argument(
        "--pattern",
        default="sheath_blight_rolling_split*/[AD]/useful_pareto/useful_sweep_*.csv",
        help="Glob pattern (relative to --base) for useful_sweep CSVs.",
    )
    ap.add_argument(
        "--files",
        nargs="*",
        default=None,
        help="Explicit CSV paths; overrides --base/--pattern when given.",
    )
    ap.add_argument(
        "--min-recall",
        type=float,
        default=0.85,
        help="Minimum required recall_val (default 0.85).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional path to write the selected rows as CSV.",
    )
    args = ap.parse_args()

    if args.files:
        paths = list(args.files)
    else:
        paths = sorted(glob.glob(os.path.join(args.base, args.pattern)))

    if not paths:
        print(
            f"[useful_sweep_farmin] no CSVs matched "
            f"(base={args.base!r}, pattern={args.pattern!r})",
            file=sys.stderr,
        )
        return 1

    selected: list[dict] = []
    for p in paths:
        row = process_csv(p, args.min_recall)
        if row is None:
            print(f"[useful_sweep_farmin] empty CSV skipped: {p}", file=sys.stderr)
            continue
        selected.append(row)

    selected.sort(key=lambda r: (r["split"], r["branch"]))

    print(f"# FAR-min selection  min_recall={args.min_recall}  n={len(selected)}")
    print(format_table(selected))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=REPORT_COLS)
            w.writeheader()
            for r in selected:
                w.writerow({c: r.get(c, "") for c in REPORT_COLS})
        print(f"# wrote {args.out}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
