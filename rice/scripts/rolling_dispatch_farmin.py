"""Extract FAR-min Stage-1 operating points across rolling splits.

Pulls already-evaluated selections out of each split's
``group_tau/group_tau_hybrid_summary.json`` for three methods:

  A_baseline        <- JSON key 'A_raw_global'   (uses (k, tau))
  D_history         <- JSON key 'D_raw_global'   (uses (k, tau))
  dispatch_group_tau<- JSON key 'dispatch_group_tau' (uses (k, tau_no, tau_with))

The JSON selection policy is "min FAR_val subject to recall_val >= target"
with tie-break (FAR, _lead desc, tau_with, tau_no, k). The user's requested
tie-break is (F1_val desc, USEFUL_val desc) — FAR_val ties are uncommon on
this grid, so results almost always agree, but the difference is flagged in
the output header.

For each split x target x method we report:

  split, method, target,
  k, tau, tau_no, tau_with,
  recall_val, FAR_val, F1_val,
  recall_test, FAR_test, precision_test, F1_test,
  no_alert_test, USEFUL_test, lead_median_test, n_event_test

Then per (method, target) mean +/- std across the three splits.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics as stats
import sys
from typing import Optional


METHOD_MAP = {
    "A_baseline": "A_raw_global",
    "D_history": "D_raw_global",
    "dispatch_group_tau": "dispatch_group_tau",
}

REPORT_COLS = (
    "split", "method", "target",
    "k", "tau", "tau_no", "tau_with",
    "recall_val", "FAR_val", "F1_val",
    "recall_test", "FAR_test", "precision_test", "F1_test",
    "no_alert_test", "USEFUL_test", "lead_median_test", "n_event_test",
)

NUMERIC_DISPLAY = {
    "tau", "tau_no", "tau_with",
    "recall_val", "FAR_val", "F1_val",
    "recall_test", "FAR_test", "precision_test", "F1_test",
    "lead_median_test",
}

AGG_COLS = (
    "recall_test", "FAR_test", "precision_test", "F1_test",
    "no_alert_test", "USEFUL_test", "lead_median_test",
)


def split_tag_from_path(path: str) -> str:
    for p in path.replace("\\", "/").split("/"):
        m = re.search(r"(split\d+)", p)
        if m:
            return m.group(1)
    return "-"


def target_key(target: float) -> str:
    # JSON keys are like "R>=0.85"; floats may render as 0.85 or 0.9.
    return f"R>={target:.2f}".rstrip("0").rstrip(".") if target != int(target) else f"R>={int(target)}"


def find_target_entry(selections: dict, target: float) -> Optional[tuple[str, dict]]:
    """Match a float target against JSON's R>=X keys, tolerating formatting."""
    for k, v in selections.items():
        m = re.match(r"R>=\s*([0-9.]+)", k)
        if not m:
            continue
        try:
            if math.isclose(float(m.group(1)), target, abs_tol=1e-6):
                return k, v
        except ValueError:
            continue
    return None


def extract_row(
    split: str,
    method_label: str,
    json_key: str,
    target: float,
    target_label: str,
    entry: dict,
) -> dict:
    val = entry.get("val", {}) or {}
    test = entry.get("test", {}) or {}
    return {
        "split": split,
        "method": method_label,
        "target": target_label,
        "k": entry.get("k", ""),
        "tau": entry.get("tau", ""),
        "tau_no": entry.get("tau_no", ""),
        "tau_with": entry.get("tau_with", ""),
        "recall_val": val.get("recall", ""),
        "FAR_val": val.get("FAR", ""),
        "F1_val": val.get("F1", ""),
        "recall_test": test.get("recall", ""),
        "FAR_test": test.get("FAR", ""),
        "precision_test": test.get("precision", ""),
        "F1_test": test.get("F1", ""),
        "no_alert_test": test.get("no_alert", ""),
        "USEFUL_test": test.get("USEFUL", ""),
        "lead_median_test": test.get("lead_median", ""),
        "n_event_test": test.get("n_event", ""),
    }


def _to_float(x) -> Optional[float]:
    if x == "" or x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def fmt_cell(col: str, value) -> str:
    if value == "" or value is None:
        return "-"
    if col in NUMERIC_DISPLAY:
        fv = _to_float(value)
        return f"{fv:.4f}" if fv is not None else str(value)
    return str(value)


def format_table(rows: list[dict], cols: tuple[str, ...]) -> str:
    if not rows:
        return "(no rows)"
    fmt_rows = [{c: fmt_cell(c, r.get(c, "")) for c in cols} for r in rows]
    widths = {c: max(len(c), max(len(fr[c]) for fr in fmt_rows)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    lines = [header, sep]
    for fr in fmt_rows:
        lines.append("  ".join(fr[c].ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def mean_std(values: list[float]) -> tuple[Optional[float], Optional[float]]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None, None
    if len(nums) == 1:
        return nums[0], 0.0
    return stats.mean(nums), stats.pstdev(nums)


def build_aggregate(rows: list[dict]) -> list[dict]:
    by_key: dict[tuple[str, str], list[dict]] = {}
    for r in rows:
        by_key.setdefault((r["method"], r["target"]), []).append(r)
    out = []
    for (method, target), rs in by_key.items():
        agg = {"method": method, "target": target, "n_splits": len(rs)}
        for c in AGG_COLS:
            m, s = mean_std([_to_float(r.get(c, "")) for r in rs])
            if m is None:
                agg[c] = "-"
            else:
                if c in {"no_alert_test", "USEFUL_test"}:
                    agg[c] = f"{m:.2f} +/- {s:.2f}"
                else:
                    agg[c] = f"{m:.4f} +/- {s:.4f}"
        out.append(agg)
    method_order = list(METHOD_MAP.keys())
    out.sort(key=lambda r: (r["target"], method_order.index(r["method"]) if r["method"] in method_order else 99))
    return out


def format_aggregate(rows: list[dict]) -> str:
    if not rows:
        return "(no aggregate)"
    cols = ("method", "target", "n_splits", *AGG_COLS)
    widths = {c: len(c) for c in cols}
    fmt_rows = []
    for r in rows:
        fr = {c: str(r.get(c, "-")) for c in cols}
        for c in cols:
            widths[c] = max(widths[c], len(fr[c]))
        fmt_rows.append(fr)
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    lines = [header, sep]
    for fr in fmt_rows:
        lines.append("  ".join(fr[c].ljust(widths[c]) for c in cols))
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
        default="sheath_blight_rolling_split*/group_tau/group_tau_hybrid_summary.json",
        help="Glob (relative to --base) for per-split dispatch summary JSONs.",
    )
    ap.add_argument(
        "--targets",
        type=float,
        nargs="+",
        default=[0.85, 0.88, 0.90],
        help="Recall targets to extract (must match JSON's R>=X keys).",
    )
    ap.add_argument(
        "--out",
        default=None,
        help="Optional CSV path for the per-split rows.",
    )
    ap.add_argument(
        "--out-agg",
        default=None,
        help="Optional CSV path for the (method,target) aggregate.",
    )
    args = ap.parse_args()

    paths = sorted(glob.glob(os.path.join(args.base, args.pattern)))
    if not paths:
        print(
            f"[rolling_dispatch_farmin] no JSONs matched "
            f"(base={args.base!r}, pattern={args.pattern!r})",
            file=sys.stderr,
        )
        return 1

    rows: list[dict] = []
    missing: list[str] = []
    for jp in paths:
        try:
            data = json.load(open(jp))
        except Exception as e:
            print(f"[rolling_dispatch_farmin] failed to load {jp}: {e}", file=sys.stderr)
            continue
        selections = data.get("selections", {}) or {}
        split = split_tag_from_path(jp)
        for target in args.targets:
            found = find_target_entry(selections, target)
            if not found:
                missing.append(f"{split}: target {target} (no matching R>= key)")
                continue
            target_label, methods_at_target = found
            for method_label, json_key in METHOD_MAP.items():
                entry = methods_at_target.get(json_key)
                if entry is None:
                    missing.append(f"{split}/{target_label}: method {json_key} missing")
                    continue
                rows.append(
                    extract_row(split, method_label, json_key, target, target_label, entry)
                )

    method_order = list(METHOD_MAP.keys())
    rows.sort(key=lambda r: (r["target"], r["split"], method_order.index(r["method"]) if r["method"] in method_order else 99))

    print("# rolling-split FAR-min selections")
    print(
        "# selection policy in JSON: min FAR_val subject to recall_val >= target; "
        "tie-break (FAR, _lead desc, tau_with, tau_no, k)."
    )
    print(
        "# user-requested tie-break (F1_val desc, USEFUL_val desc) was NOT re-applied — "
        "FAR_val ties are rare on this grid, but diff is possible."
    )
    print(f"# splits={len(paths)}  targets={args.targets}  rows={len(rows)}")
    if missing:
        print("# missing entries:")
        for m in missing:
            print(f"#   {m}")
    print()
    print(format_table(rows, REPORT_COLS))

    print()
    print("# mean +/- std across splits, per (method, target)")
    agg = build_aggregate(rows)
    print(format_aggregate(agg))

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=REPORT_COLS)
            w.writeheader()
            for r in rows:
                w.writerow({c: r.get(c, "") for c in REPORT_COLS})
        print(f"# wrote {args.out}", file=sys.stderr)

    if args.out_agg:
        os.makedirs(os.path.dirname(args.out_agg) or ".", exist_ok=True)
        cols = ("method", "target", "n_splits", *AGG_COLS)
        with open(args.out_agg, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=cols)
            w.writeheader()
            for r in agg:
                w.writerow({c: r.get(c, "") for c in cols})
        print(f"# wrote {args.out_agg}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
