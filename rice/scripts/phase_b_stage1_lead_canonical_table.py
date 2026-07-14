#!/usr/bin/env python3
"""Canonical Stage 1 best-gate metric table — split3 (val=2023 / test=2024).

Produces the PPT-ready table the user asked for, pulling test-side metrics
from the AUTHORITATIVE source for each gate type:

  * dispatch_group_tau  : group_tau_hybrid_summary.json (.selections.R>=0.88
                          .dispatch_group_tau.test.lead_{mean,median})
  * A_baseline          : useful_pareto/useful_sweep_A.csv at the selected
                          (k, tau) row — lead_{mean,median}_test
  * D_history           : useful_pareto/useful_sweep_D.csv at the selected
                          (k, tau) row — lead_{mean,median}_test

LEAD DEFINITION (held-out test cohort):
  For each EVENT site-year, the Stage 1 gate fires "alert" on the first day
  whose calibrated score crosses the gate's tau (k consecutive days for the
  hybrid rule). "lead" = (first_event_day L_dotted) - (alert day), in days.
  Positive lead = alert before event; negative = alert after.
  Median / mean are over event site-years where the gate fired (matched).

NOTE: Stage 1 training label = "lead14-45_ignore" — at TRAIN time, only days
within L-45..L-14 of an event are positive; other event days are ignored.
At INFERENCE time the alert is the first threshold crossing, which can be
much further from L than the label window (e.g. 60-80d lead is possible when
the model's calibrated probability rises early). The label window biases
WHERE the model has gradient signal, not WHERE the inference alert lives.

Columns:
  pest selected_method selected_run val_recall val_FAR test_recall test_FAR
  test_precision test_F1 test_lead_mean test_lead_median

Input:
  - existing gate_selection CSV (val-only selected gate per pest)
  - Stage 1 sweep CSVs / summary JSONs (one per pest × run × split3)

Output:
  - <out_csv> with the columns above
  - prints a formatted table to stdout
"""

from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import pandas as pd

DEFAULT_TARGET = "R>=0.88"


def s1_lead_from_json(json_path: Path, target: str, method_key: str) -> tuple[float, float]:
    """method_key in {'dispatch_group_tau','global_tau_hybrid','A_raw_global',
    'D_raw_global','OR_hybrid'} as stored in group_tau_hybrid_summary.json."""
    d = json.loads(json_path.read_text())
    sel = d["selections"][target][method_key]
    return float(sel["test"]["lead_mean"]), float(sel["test"]["lead_median"])


def s1_lead_from_sweep(sweep_csv: Path, k: int, tau: float) -> tuple[float, float]:
    """Pull test-side lead_{mean,median} from useful_pareto sweep CSV at the
    row matching (k, tau) exactly."""
    d = pd.read_csv(sweep_csv)
    m = d[(d["k"] == int(k)) & (abs(d["tau"] - float(tau)) < 1e-6)]
    if m.empty:
        return float("nan"), float("nan")
    r = m.iloc[0]
    return float(r["lead_mean_test"]), float(r["lead_median_test"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate_csv",
                    default="rice/outputs_stage2_batch_2024_bestgate/_summary/stage1_gate_selection_split3_2024.csv")
    ap.add_argument("--split", default="split3")
    ap.add_argument("--val_year", type=int, default=2023)
    ap.add_argument("--test_year", type=int, default=2024)
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--out_csv",
                    default="rice/outputs_stage2_batch_2024_bestgate/_summary/stage1_best_gate_canonical_ppt.csv")
    args = ap.parse_args()

    sel = pd.read_csv(args.gate_csv)
    rows = []
    for _, r in sel.iterrows():
        pest = r["pest"]
        method = str(r["selected_method"])
        run = int(r["selected_run"])
        k = int(r["k"])
        tau = r.get("tau"); tau_no = r.get("tau_no")
        base = (f"rice/outputs_stage1/batch_rolling/{pest}/run{run}/"
                f"{args.split}_v{args.val_year}_t{args.test_year}")
        if method == "A_baseline":
            csv = Path(base) / "A/useful_pareto/useful_sweep_A.csv"
            lead_mean, lead_median = s1_lead_from_sweep(csv, k, float(tau))
            source = f"useful_sweep_A.csv (k={k}, tau={float(tau):.4f})"
        elif method == "D_history":
            csv = Path(base) / "D/useful_pareto/useful_sweep_D.csv"
            lead_mean, lead_median = s1_lead_from_sweep(csv, k, float(tau))
            source = f"useful_sweep_D.csv (k={k}, tau={float(tau):.4f})"
        elif method == "dispatch_group_tau":
            j = Path(base) / "group_tau/group_tau_hybrid_summary.json"
            lead_mean, lead_median = s1_lead_from_json(j, args.target, "dispatch_group_tau")
            source = f"summary.json selections[{args.target}][dispatch_group_tau].test"
        else:
            lead_mean = lead_median = float("nan")
            source = f"UNKNOWN method={method}"

        rows.append({
            "pest": pest,
            "selected_method": method,
            "selected_run": run,
            "val_recall": float(r["val_recall"]),
            "val_FAR": float(r["val_FAR"]),
            "test_recall": float(r["test_recall"]),
            "test_FAR": float(r["test_FAR"]),
            "test_precision": float(r["test_precision"]),
            "test_F1": float(r["test_F1"]),
            "test_lead_mean": lead_mean,
            "test_lead_median": lead_median,
            "test_n_event": int(r["test_n_event"]),
            "lead_source": source,
        })

    out = pd.DataFrame(rows)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)

    print(f"[wrote] {args.out_csv}  rows={len(out)}")
    print()
    print(f"=== Stage 1 best gate per pest — {args.split} (val={args.val_year} test={args.test_year}, {args.target}) ===")
    show = ["pest", "selected_method", "selected_run",
            "val_recall", "val_FAR",
            "test_recall", "test_FAR", "test_precision", "test_F1",
            "test_lead_mean", "test_lead_median", "test_n_event"]
    with pd.option_context("display.width", 220, "display.max_columns", 20,
                            "display.float_format", "{:.3f}".format):
        print(out[show].to_string(index=False))
    print()
    print("Lead definition (test):")
    print("  For each EVENT site-year, lead = first_event_doy - first_alert_doy.")
    print("  Median/mean over event SYs where the gate fired (matched).")
    print("  Inference alert = first day score crosses tau for k consecutive days;")
    print("  it is NOT constrained to the train-label window L-45..L-14, so leads")
    print("  outside that window (e.g. 50-80d) are expected when calibrated scores")
    print("  rise early.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
