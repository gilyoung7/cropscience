"""Phase S11 Step 2 — per-sample oracle over (model, offset) and GO/NO-GO verdict.

Reads the Phase-S10-style sample_grid with two models (baseline 2-sided +
center-ra variant), each scored at 4 offsets, and computes:

    1. Single-best per (model, offset) IoU_overall (denom = N_TOTAL_TEST).
       The maximum across 8 combos is the "single-best" reference for selector
       judgement.
    2. Within-model oracle: for each model, sum_per_sample(max IoU over its
       4 offsets) / N_TOTAL_TEST.
    3. Across-model oracle: sum_per_sample(max IoU over all 8 (model, offset)
       combos) / N_TOTAL_TEST.
    4. Per-sample winner distribution: how many samples have each
       (model, offset) as their unique best.

Verdict:
    Δ_oracle_minus_best > +0.02 → GO for selector (Step 3 worth training)
    +0.01 < Δ ≤ +0.02          → MARGINAL (selector may help; report only)
    Δ ≤ +0.01                  → NO-GO (selector cannot exceed near-equal
                                          single-best; stop here)

Outputs (out_dir, default outputs/phase_s11/):
    oracle_summary.csv          — per (model, offset) IoU + within/across oracles
    winner_distribution.csv     — counts per (best_model, best_offset)
    per_sample_oracle.csv       — sample-level (best_model, best_offset, best_iou)
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from rice.scripts.phase_s_selector import N_TOTAL_TEST


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sample_grid", type=str,
                    default="rice/outputs/diag/phase_s10_sample_grid.csv",
                    help="Two-model sample_grid (baseline + center_ra variant).")
    ap.add_argument("--out_dir", type=str, default="rice/outputs/phase_s11/")
    ap.add_argument("--base_label_substr", type=str, default="(asym=25)",
                    help="Substring identifying the baseline row.")
    ap.add_argument("--alt_label_substr", type=str, default="center_ra",
                    help="Substring identifying the experimental row.")
    ap.add_argument("--n_total", type=int, default=N_TOTAL_TEST,
                    help="Denominator for IoU_overall (default 575).")
    args = ap.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.sample_grid):
        raise SystemExit(f"[abort] sample_grid not found: {args.sample_grid}")
    grid = pd.read_csv(args.sample_grid)
    models = list(grid["model"].unique())
    print(f"[input] {args.sample_grid}  rows={len(grid)}  models={models}", flush=True)

    base_match = [m for m in models if args.base_label_substr in m
                  and args.alt_label_substr not in m]
    alt_match = [m for m in models if args.alt_label_substr in m]
    if not base_match or not alt_match:
        raise SystemExit(
            f"[abort] need both baseline (substr={args.base_label_substr!r}) "
            f"and alt (substr={args.alt_label_substr!r}); got "
            f"base_match={base_match} alt_match={alt_match}")
    base_label, alt_label = base_match[0], alt_match[0]
    print(f"  baseline label : {base_label}", flush=True)
    print(f"  alt label      : {alt_label}", flush=True)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 220); pd.set_option("display.max_columns", 20)

    # ----- 1) Single-best per (model, offset) -------------------------------
    # Only matched rows contribute to the IoU sum; unmatched rows contribute 0.
    rows = []
    for (model, off), sub in grid.groupby(["model", "offset"]):
        matched = sub[sub["matched"] == True]
        iou_sum = float(matched["iou_matched"].sum())
        n_matched = int(len(matched))
        n_unique_samples = int(sub["sample_id"].nunique())
        rows.append({
            "model": model, "offset": int(off),
            "n_unique_samples": n_unique_samples,
            "n_matched": n_matched,
            "iou_sum": iou_sum,
            "IoU_overall": iou_sum / float(args.n_total),
        })
    single_df = pd.DataFrame(rows).sort_values(["model", "offset"]).reset_index(drop=True)
    print("\n=================== Single-best per (model, offset) ===================", flush=True)
    print(single_df.to_string(index=False), flush=True)

    best_row = single_df.loc[single_df["IoU_overall"].idxmax()]
    best_single_iou = float(best_row["IoU_overall"])
    print(f"\n  [single-best] {best_row['model']!r} off={int(best_row['offset'])} "
          f"→ IoU_overall = {best_single_iou:.4f}", flush=True)

    # ----- 2 & 3) Within-model and across-model oracles ---------------------
    # Build per-sample table: sample_id → list of (model, offset, iou).
    # Unmatched rows have iou=0; per phase_s3 convention they still count.
    # Sample is included if at least one (model, offset) is matched.
    all_sids = sorted(grid["sample_id"].astype(str).unique())
    sample_records = []
    within_model_iou = {base_label: 0.0, alt_label: 0.0}
    n_within = {base_label: 0, alt_label: 0}
    across_iou_sum = 0.0
    n_across = 0
    for sid in all_sids:
        sub = grid[grid["sample_id"].astype(str) == sid]
        # max iou per model (over its 4 offsets) — unmatched rows iou=0
        best_per_model = {}
        for m in (base_label, alt_label):
            sub_m = sub[sub["model"] == m]
            if sub_m.empty:
                continue
            # Use iou_matched (0 for unmatched rows by construction)
            iou_m = float(sub_m["iou_matched"].max())
            off_m = int(sub_m.loc[sub_m["iou_matched"].idxmax(), "offset"])
            # Match flag of the best row
            mat_m = bool(sub_m.loc[sub_m["iou_matched"].idxmax(), "matched"])
            best_per_model[m] = (iou_m, off_m, mat_m)
            if mat_m:
                within_model_iou[m] += iou_m
                n_within[m] += 1
        # Across-model oracle
        valid = [(m, iou, off, mat) for m, (iou, off, mat) in best_per_model.items() if mat]
        if not valid:
            continue
        m_star, iou_star, off_star, _ = max(valid, key=lambda t: t[1])
        across_iou_sum += iou_star
        n_across += 1
        sample_records.append({
            "sample_id": sid,
            "best_model": m_star, "best_offset": off_star, "best_iou": iou_star,
            "iou_base_best": best_per_model.get(base_label, (0.0, -1, False))[0],
            "off_base_best": best_per_model.get(base_label, (0.0, -1, False))[1],
            "iou_alt_best":  best_per_model.get(alt_label,  (0.0, -1, False))[0],
            "off_alt_best":  best_per_model.get(alt_label,  (0.0, -1, False))[1],
        })
    per_sample_df = pd.DataFrame(sample_records)

    within_overall = {m: within_model_iou[m] / float(args.n_total) for m in (base_label, alt_label)}
    across_overall = across_iou_sum / float(args.n_total)

    print("\n=================== Within-model oracle (best-of-4-offsets per sample) ===================", flush=True)
    for m in (base_label, alt_label):
        print(f"  {m:<40} n={n_within[m]:>4}  iou_sum={within_model_iou[m]:.2f}  "
              f"IoU_overall = {within_overall[m]:.4f}", flush=True)
    print("\n=================== Across-model oracle (best-of-8 (model, offset) per sample) ===================", flush=True)
    print(f"  union cohort  n={n_across:>4}  iou_sum={across_iou_sum:.2f}  "
          f"IoU_overall = {across_overall:.4f}", flush=True)

    # ----- 4) Winner distribution -------------------------------------------
    win_df = (per_sample_df.groupby(["best_model", "best_offset"])
              .size().reset_index(name="n_winner"))
    win_df["pct_of_cohort"] = win_df["n_winner"] / float(n_across) * 100.0
    win_df = win_df.sort_values(["best_model", "best_offset"]).reset_index(drop=True)
    print("\n=================== Per-sample winner distribution ===================", flush=True)
    print(win_df.to_string(index=False), flush=True)

    # ----- Output CSVs ------------------------------------------------------
    single_df.to_csv(out_dir / "oracle_summary.csv", index=False)
    win_df.to_csv(out_dir / "winner_distribution.csv", index=False)
    per_sample_df.to_csv(out_dir / "per_sample_oracle.csv", index=False)
    print(f"\n[csv] {out_dir}/oracle_summary.csv", flush=True)
    print(f"[csv] {out_dir}/winner_distribution.csv", flush=True)
    print(f"[csv] {out_dir}/per_sample_oracle.csv", flush=True)

    # ----- Verdict (GO / MARGINAL / NO-GO) ----------------------------------
    delta_oracle = across_overall - best_single_iou
    delta_within_base  = within_overall[base_label] - best_single_iou
    delta_within_alt   = within_overall[alt_label]  - best_single_iou
    print("\n=================== Phase S11 Step 2 verdict ===================", flush=True)
    print(f"  best single (model, offset) IoU = {best_single_iou:.4f}", flush=True)
    print(f"  within-model oracle (baseline)  = {within_overall[base_label]:.4f}  "
          f"Δ = {delta_within_base:+.4f}", flush=True)
    print(f"  within-model oracle (alt/S10)   = {within_overall[alt_label]:.4f}  "
          f"Δ = {delta_within_alt:+.4f}", flush=True)
    print(f"  across-model oracle (8 combos)  = {across_overall:.4f}  "
          f"Δ = {delta_oracle:+.4f}", flush=True)

    if delta_oracle > 0.02:
        verdict = "GO"
        rec = ("Across-model oracle exceeds single-best by > +0.02. "
               "Selector (Step 3, 8-way) is worth training.")
    elif delta_oracle > 0.01:
        verdict = "MARGINAL"
        rec = ("Oracle gain modest (+0.01 < Δ ≤ +0.02). Selector may help "
               "but accuracy floor is tight; train if accuracy budget allows.")
    else:
        verdict = "NO-GO"
        rec = ("Oracle gain ≤ +0.01. Selector cannot reliably beat single-best; "
               "STOP — do not run Step 3.")
    print(f"\n  ===== {verdict} =====", flush=True)
    print(f"  {rec}", flush=True)

    # Quick sanity for next-step decision
    if verdict in ("GO", "MARGINAL"):
        print(f"\n  Next step: run phase_s11_selector with --sample_grid {args.sample_grid}", flush=True)
    else:
        print(f"\n  Next step: skip Step 3, report Phase S11 as complete.", flush=True)


if __name__ == "__main__":
    main()
