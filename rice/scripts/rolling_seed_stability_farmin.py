"""Aggregate seed-stability FAR-min selections across rolling splits.

Reads per-(split, seed) outputs produced by ``scripts/run_rolling_seedstab.sh``:

  rice/outputs_stage1/seed_stability/sheath_blight_split{N}_v..._t..._seed{S}/
    A/useful_pareto/useful_sweep_A.csv          # full (k, tau) sweep, val+test
    D/useful_pareto/useful_sweep_D.csv
    group_tau/group_tau_hybrid_summary.json     # dispatch selections + eval

Selection policy applied here:
  - A_baseline, D_history:
      val recall >= target, FAR_val min,
      tie-break: F1_val desc, USEFUL_val desc, lead_mean_val desc.
      Selected directly from the useful_sweep CSV (val + test sweep both
      available, so test metrics come from the same row).
  - dispatch_group_tau:
      pulled from group_tau_hybrid_summary.json at each recall target.
      The JSON's internal policy is "FAR_val min s.t. recall_val >= target"
      with tie-break (FAR, _lead desc, tau_with, tau_no, k). Same primary
      criterion as the user's policy; tie-break differs slightly but FAR_val
      ties on this grid are rare.

Reports three tables on stdout (and optional CSV outputs):
  1) Full per-row table: split x seed x method x target
  2) (method, target) overall mean/std across 3 splits x 3 seeds = 9 points
  3) (split, method, target) seed mean/std across seeds 0/1/2
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


METHODS = ("A_baseline", "D_history", "dispatch_group_tau")
JSON_METHOD_KEY = {
    "A_baseline": "A_raw_global",
    "D_history": "D_raw_global",
    "dispatch_group_tau": "dispatch_group_tau",
}
USEFUL_SWEEP_REL = {
    "A_baseline": "A/useful_pareto/useful_sweep_A.csv",
    "D_history": "D/useful_pareto/useful_sweep_D.csv",
}

ROW_COLS = (
    "split", "seed", "method", "target",
    "k", "tau", "tau_no", "tau_with",
    "recall_val", "FAR_val", "F1_val",
    "recall_test", "FAR_test", "precision_test", "F1_test",
    "no_alert_test", "USEFUL_test", "lead_median_test", "n_event_test",
    "fallback",
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


def _to_float(x) -> Optional[float]:
    if x == "" or x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_split_seed(dirname: str) -> tuple[str, int]:
    m_s = re.search(r"(split\d+)", dirname)
    m_seed = re.search(r"seed(\d+)$", dirname)
    return (m_s.group(1) if m_s else "-", int(m_seed.group(1)) if m_seed else -1)


def find_target_entry(selections: dict, target: float) -> Optional[tuple[str, dict]]:
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


def select_ad_row(csv_path: str, target: float) -> tuple[Optional[dict], bool]:
    """User-policy FAR-min selection on a useful_sweep CSV."""
    with open(csv_path) as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return None, False
    feas = [r for r in rows if (_to_float(r.get("recall_val")) or -1.0) >= target]
    fallback = False
    if not feas:
        fallback = True
        feas = rows
        key = lambda r: (
            -(_to_float(r.get("recall_val")) or -1.0),
            -(_to_float(r.get("F1_val")) or -1.0),
            (_to_float(r.get("FAR_val")) or float("inf")),
        )
    else:
        # user tie-break: F1_val desc, USEFUL_val desc, lead_mean_val desc
        key = lambda r: (
            (_to_float(r.get("FAR_val")) or float("inf")),
            -(_to_float(r.get("F1_val")) or -1.0),
            -(_to_float(r.get("USEFUL_val")) or -1.0),
            -(_to_float(r.get("lead_mean_val")) or -1.0),
        )
    return min(feas, key=key), fallback


def row_from_ad_csv(
    split: str, seed: int, method: str, target: float, target_label: str,
    csv_path: str,
) -> Optional[dict]:
    chosen, fallback = select_ad_row(csv_path, target)
    if chosen is None:
        return None
    return {
        "split": split, "seed": seed, "method": method, "target": target_label,
        "k": chosen.get("k", ""),
        "tau": chosen.get("tau", ""),
        "tau_no": "", "tau_with": "",
        "recall_val": chosen.get("recall_val", ""),
        "FAR_val": chosen.get("FAR_val", ""),
        "F1_val": chosen.get("F1_val", ""),
        "recall_test": chosen.get("recall_test", ""),
        "FAR_test": chosen.get("FAR_test", ""),
        "precision_test": chosen.get("precision_test", ""),
        "F1_test": chosen.get("F1_test", ""),
        "no_alert_test": chosen.get("no_alert_test", ""),
        "USEFUL_test": chosen.get("USEFUL_test", ""),
        "lead_median_test": chosen.get("lead_median_test", ""),
        "n_event_test": chosen.get("n_event_test", ""),
        "fallback": "yes" if fallback else "no",
    }


def row_from_dispatch_json(
    split: str, seed: int, target: float, target_label: str, entry: dict,
) -> dict:
    val = entry.get("val", {}) or {}
    test = entry.get("test", {}) or {}
    return {
        "split": split, "seed": seed, "method": "dispatch_group_tau",
        "target": target_label,
        "k": entry.get("k", ""),
        "tau": "",
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
        "fallback": "no",
    }


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


def mean_std(values: list[Optional[float]]) -> tuple[Optional[float], Optional[float]]:
    nums = [v for v in values if v is not None]
    if not nums:
        return None, None
    if len(nums) == 1:
        return nums[0], 0.0
    return stats.mean(nums), stats.pstdev(nums)


def aggregate(rows: list[dict], group_keys: tuple[str, ...]) -> list[dict]:
    buckets: dict[tuple, list[dict]] = {}
    for r in rows:
        buckets.setdefault(tuple(r[k] for k in group_keys), []).append(r)
    out = []
    method_order = list(METHODS)
    for gkey, rs in buckets.items():
        agg = dict(zip(group_keys, gkey))
        agg["n"] = len(rs)
        for c in AGG_COLS:
            m, s = mean_std([_to_float(r.get(c)) for r in rs])
            if m is None:
                agg[c] = "-"
            elif c in {"no_alert_test", "USEFUL_test"}:
                agg[c] = f"{m:.2f} +/- {s:.2f}"
            else:
                agg[c] = f"{m:.4f} +/- {s:.4f}"
        out.append(agg)
    out.sort(key=lambda r: tuple(
        method_order.index(r["method"]) if k == "method" and r["method"] in method_order else r.get(k, "")
        for k in group_keys
    ))
    return out


def format_agg(rows: list[dict], group_keys: tuple[str, ...]) -> str:
    if not rows:
        return "(no aggregate)"
    cols = (*group_keys, "n", *AGG_COLS)
    fmt = [{c: str(r.get(c, "-")) for c in cols} for r in rows]
    widths = {c: max(len(c), max(len(fr[c]) for fr in fmt)) for c in cols}
    header = "  ".join(c.ljust(widths[c]) for c in cols)
    sep = "  ".join("-" * widths[c] for c in cols)
    lines = [header, sep]
    for fr in fmt:
        lines.append("  ".join(fr[c].ljust(widths[c]) for c in cols))
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--base",
        default="rice/outputs_stage1/seed_stability",
        help="Root directory containing per-(split,seed) run dirs.",
    )
    ap.add_argument(
        "--pattern",
        default="sheath_blight_split*_v*_t*_seed*",
        help="Glob for per-run dirs under --base.",
    )
    ap.add_argument(
        "--targets",
        type=float, nargs="+", default=[0.85, 0.88, 0.90],
        help="Recall targets to evaluate.",
    )
    ap.add_argument("--out-rows", default=None, help="Optional CSV for per-row table.")
    ap.add_argument("--out-overall", default=None, help="Optional CSV for overall mean/std.")
    ap.add_argument("--out-per-split", default=None, help="Optional CSV for split-level mean/std.")
    args = ap.parse_args()

    dirs = sorted(glob.glob(os.path.join(args.base, args.pattern)))
    if not dirs:
        print(f"[seedstab] no run dirs under {args.base!r} pattern={args.pattern!r}", file=sys.stderr)
        return 1

    rows: list[dict] = []
    missing: list[str] = []
    for d in dirs:
        split, seed = parse_split_seed(os.path.basename(d))
        # A/D from useful_sweep CSVs
        for method in ("A_baseline", "D_history"):
            csv_path = os.path.join(d, USEFUL_SWEEP_REL[method])
            if not os.path.isfile(csv_path):
                missing.append(f"{split}/seed{seed}/{method}: missing {csv_path}")
                continue
            for tgt in args.targets:
                tgt_label = f"R>={tgt:.2f}".rstrip("0").rstrip(".") if tgt != int(tgt) else f"R>={int(tgt)}"
                row = row_from_ad_csv(split, seed, method, tgt, tgt_label, csv_path)
                if row is None:
                    missing.append(f"{split}/seed{seed}/{method}/{tgt_label}: empty CSV")
                    continue
                rows.append(row)
        # dispatch from JSON
        jp = os.path.join(d, "group_tau", "group_tau_hybrid_summary.json")
        if not os.path.isfile(jp):
            missing.append(f"{split}/seed{seed}/dispatch_group_tau: missing {jp}")
            continue
        try:
            data = json.load(open(jp))
        except Exception as e:
            missing.append(f"{split}/seed{seed}/dispatch: load fail {e}")
            continue
        selections = data.get("selections", {}) or {}
        for tgt in args.targets:
            found = find_target_entry(selections, tgt)
            if not found:
                missing.append(f"{split}/seed{seed}/dispatch: no R>= key for {tgt}")
                continue
            tgt_label, methods_at = found
            entry = methods_at.get(JSON_METHOD_KEY["dispatch_group_tau"])
            if entry is None:
                missing.append(f"{split}/seed{seed}/{tgt_label}: dispatch_group_tau missing")
                continue
            rows.append(row_from_dispatch_json(split, seed, tgt, tgt_label, entry))

    method_order = list(METHODS)
    rows.sort(key=lambda r: (
        r["target"], r["split"], r["seed"],
        method_order.index(r["method"]) if r["method"] in method_order else 99,
    ))

    print("# rolling-split XGB seed stability  FAR-min selections")
    print("# A/D: user policy (FAR_val min s.t. recall_val>=target; tie F1_val,USEFUL_val,lead_mean)")
    print("# dispatch: JSON selection (same primary; tie-break slightly differs - FAR ties rare)")
    print(f"# splits/seeds detected: {len(dirs)}  rows: {len(rows)}  missing: {len(missing)}")
    for m in missing:
        print(f"#   missing: {m}")
    print()
    print("## (1) per-row table")
    print(format_table(rows, ROW_COLS))

    print()
    print("## (2) overall mean +/- std across (3 splits x 3 seeds = 9 points), per (method, target)")
    overall = aggregate(rows, ("method", "target"))
    print(format_agg(overall, ("method", "target")))

    print()
    print("## (3) seed mean +/- std per (split, method, target)")
    per_split = aggregate(rows, ("split", "method", "target"))
    print(format_agg(per_split, ("split", "method", "target")))

    def _write(path: str, fieldnames: tuple[str, ...], data: list[dict]) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=fieldnames)
            w.writeheader()
            for r in data:
                w.writerow({c: r.get(c, "") for c in fieldnames})
        print(f"# wrote {path}", file=sys.stderr)

    if args.out_rows:
        _write(args.out_rows, ROW_COLS, rows)
    if args.out_overall:
        _write(args.out_overall, ("method", "target", "n", *AGG_COLS), overall)
    if args.out_per_split:
        _write(args.out_per_split, ("split", "method", "target", "n", *AGG_COLS), per_split)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
