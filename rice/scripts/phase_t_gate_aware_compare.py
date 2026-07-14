"""
Phase T7e — Gate-aware Stage 1 selection across multiple ckpts.

Reads phase_t_useful_pareto wide-format sweep CSVs and selects gate-aware
operating points per ckpt:

  (1) recall_val >= 0.90 -> FAR_val minimum
  (2) recall_val >= 0.92 -> FAR_val minimum
  (3) recall_val >= 0.95 -> FAR_val minimum  [reference / upper-bound check]

  tie-break: lead_median_val shorter (Stage 2 prefers shorter alert lead),
             then lead_mean_val shorter, then tau higher, then k larger.

For each selected op, prints val + test:
  recall, FAR, precision, F1, n_alert, n_event
  no_alert / TOO_LATE / MISSED / TOO_EARLY / USEFUL counts and rates per n_event
  alert lead mean / median

Notes:
  - rule = first_crossing (matches input sweep CSVs)
  - USEFUL is reported for context but not used in selection
  - intended Stage 1 = gate; Stage 2 does the timing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


SELECTION_SORT_COLS = ["FAR_val", "lead_median_val", "lead_mean_val", "tau", "k"]
SELECTION_SORT_DIRS = [True, True, True, False, False]


def select_gate_aware(df: pd.DataFrame, target_recall: float) -> dict | None:
    cands = df[df["recall_val"] >= target_recall]
    if cands.empty:
        return None
    cands = cands.sort_values(SELECTION_SORT_COLS, ascending=SELECTION_SORT_DIRS)
    return cands.iloc[0].to_dict()


def _coerce(v):
    if isinstance(v, (np.floating, float)):
        return float(v) if not (isinstance(v, float) and np.isnan(v)) else None
    if isinstance(v, (np.integer, int)):
        return int(v)
    return v


def fmt_pct(c: int, n: int) -> str:
    return f"{c}({c/max(n,1)*100:.1f}%)"


def print_op(label: str, row: dict) -> None:
    print(f"  [{label}]")
    print(f"    op: k={int(row['k'])}  tau={float(row['tau']):.3f}")
    for split in ["val", "test"]:
        ne = int(row[f"n_event_{split}"])
        no_a = int(row[f"no_alert_{split}"])
        tl = int(row[f"TOO_LATE_{split}"])
        ms = int(row[f"MISSED_{split}"])
        te = int(row[f"TOO_EARLY_{split}"])
        us = int(row[f"USEFUL_{split}"])
        rec = float(row[f"recall_{split}"]); far = float(row[f"FAR_{split}"])
        pr = float(row[f"precision_{split}"]); f1 = float(row[f"F1_{split}"])
        n_al = int(row[f"n_alert_{split}"])
        lm = row[f"lead_mean_{split}"]
        lmd = row[f"lead_median_{split}"]
        print(f"    {split}: R={rec:.3f}  FAR={far:.3f}  P={pr:.3f}  F1={f1:.3f}  "
              f"n_alert={n_al}  n_event={ne}")
        print(f"          no_alert={fmt_pct(no_a, ne)}  TOO_LATE={fmt_pct(tl, ne)}  "
              f"MISSED={fmt_pct(ms, ne)}")
        print(f"          TOO_EARLY={fmt_pct(te, ne)}  USEFUL={fmt_pct(us, ne)}  [info]")
        if isinstance(lm, (float, np.floating)) and not np.isnan(lm):
            print(f"          alert lead mean={lm:.1f}  median={float(lmd):.1f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweeps", required=True,
                    help="comma-separated paths to useful_sweep CSVs")
    ap.add_argument("--labels", required=True,
                    help="comma-separated labels (same order as --sweeps)")
    ap.add_argument("--recall_targets", default="0.90,0.92,0.95")
    ap.add_argument("--out_json", default="")
    args = ap.parse_args()

    sweep_paths = [s.strip() for s in args.sweeps.split(",")]
    labels = [s.strip() for s in args.labels.split(",")]
    if len(sweep_paths) != len(labels):
        raise SystemExit("--sweeps and --labels must have same length")
    targets = [float(x) for x in args.recall_targets.split(",") if x.strip()]

    out = {"targets": targets, "selections": {}}
    # Per-ckpt selections
    for label, path in zip(labels, sweep_paths):
        df = pd.read_csv(path)
        print(f"\n========== {label}  (n_cells={len(df)}) ==========")
        out["selections"][label] = {}
        for tgt in targets:
            pick = select_gate_aware(df, tgt)
            key = f"recall>={tgt:.2f}_FARmin"
            if pick is None:
                print(f"\n  [{key}]  (no qualifying cell on val)")
                out["selections"][label][key] = None
                continue
            print()
            print_op(key, pick)
            out["selections"][label][key] = {k: _coerce(v) for k, v in pick.items()}

    # Side-by-side summary
    print(f"\n========== Side-by-side (test) ==========")
    print(f"  {'target':>14}  {'ckpt':>22}  {'k':>2} {'tau':>6} "
          f"{'t_R':>5} {'t_FAR':>6} {'t_P':>6} {'t_F1':>6} "
          f"{'t_noA':>5} {'t_TL':>4} {'t_MS':>4} {'t_TE':>4} {'t_USE':>5} "
          f"{'lead_med':>8} {'lead_mean':>9}")
    for tgt in targets:
        key = f"recall>={tgt:.2f}_FARmin"
        for label in labels:
            pick = out["selections"][label].get(key)
            if pick is None:
                print(f"  {f'R>={tgt:.2f}':>14}  {label:>22}  (no qualifying cell)")
                continue
            print(f"  {f'R>={tgt:.2f}':>14}  {label:>22}  "
                  f"{int(pick['k']):>2d} {float(pick['tau']):>6.3f} "
                  f"{float(pick['recall_test']):>5.3f} {float(pick['FAR_test']):>6.3f} "
                  f"{float(pick['precision_test']):>6.3f} {float(pick['F1_test']):>6.3f} "
                  f"{int(pick['no_alert_test']):>5d} {int(pick['TOO_LATE_test']):>4d} "
                  f"{int(pick['MISSED_test']):>4d} {int(pick['TOO_EARLY_test']):>4d} "
                  f"{int(pick['USEFUL_test']):>5d} "
                  f"{float(pick['lead_median_test'] or float('nan')):>8.1f} "
                  f"{float(pick['lead_mean_test'] or float('nan')):>9.1f}")

    if args.out_json:
        Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out_json).write_text(json.dumps(out, indent=2, default=str))
        print(f"\n[saved] {args.out_json}")


if __name__ == "__main__":
    main()
