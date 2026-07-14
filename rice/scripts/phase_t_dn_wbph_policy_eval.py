"""Multi-policy re-evaluation of the WBPH direct_neighbor offset grid (test 2024).

Reads wbph_offset_grid.csv (produced by phase_t_dn_wbph_offset_grid.py) and scores
baseline vs direct_neighbor under several offset-selection policies, separating
ORACLE/REFERENCE (peek at test) from DEPLOYABLE (val/selector only).

Policies (each maps sample_id -> one offset on TEST):
  oracle_per_sample   ORACLE   argmax IoU80 over offsets per sample (upper bound)
  oracle_fixed_60     REFERENCE const 60 (the per-pest TEST-oracle offset used in compare_eval)
  val_fixed_iou80     DEPLOY   const = argmax over offsets of mean val(2023) IoU80
  val_fixed_band      DEPLOY   const = argmax over offsets of mean val(2023) IoU_band (~lead_v3 style)
  deploy_q10/q20      DEPLOY   base=val_fixed_iou80, cap eval_doy by val lead quantile (no test labels)
  learned_v2_reg_coarse DEPLOY apply the EXISTING (baseline-trained) v2 regressor coarse
                               per-sample offset picks to BOTH variants

Metrics per (variant, policy): late_eval_rate, late_mu_rate, no_overlap_rate (80 & band),
IoU80, IoU_band, PI_hit_rate (80 & band), MAE_center, mean_offset, coverage.

NOTE: IoU80 = 80% shortest-mass interval IoU (matched_IoU80 convention).
      IoU_band = mu +/- 1.96*sigma fixed-band IoU (offset_constraint convention; sigma=5 FIXED).
Both are reported because the model is a fixed-sigma Gaussian head, so the "band" is
architecturally fixed-width; differences across policies come only from mu alignment.

Run:
    PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_dn_wbph_policy_eval --force
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from rice.configs.base import RICE_ROOT
from rice.scripts.phase_t_stage2_offset_constraint import (
    build_val_caps, lookup_deploy_cap, choose_offset, ANCHORS)

OUT_DIR = RICE_ROOT / "outputs/diag/stage2_direct_neighbor_wbph_2024"
GRID = OUT_DIR / "wbph_offset_grid.csv"
SELECTOR_PICKS = RICE_ROOT / "outputs/stage2/selector_cross_split/2024_WBPH/v2_per_sample_test_selections.csv"
VARIANTS = ["baseline", "direct_neighbor"]
BIN_W = 20


def metric_block(rows: pd.DataFrame, n_total: int) -> dict:
    """Aggregate metrics over the per-sample selected rows (matched only)."""
    n = len(rows)
    if n == 0:
        return {"n": 0, "coverage": 0.0}
    return {
        "n": int(n), "coverage": round(n / n_total, 4),
        "mean_offset": round(rows["offset"].mean(), 2),
        "late_eval_rate": round(rows["late_eval"].mean(), 4),
        "late_mu_rate": round(rows["late_mu"].mean(), 4),
        "no_overlap80_rate": round(rows["no_overlap80"].mean(), 4),
        "no_overlap_band_rate": round(rows["no_overlap_band"].mean(), 4),
        "IoU80": round(rows["iou80"].mean(), 4),
        "IoU_band": round(rows["iou_band"].mean(), 4),
        # coverage-weighted (unmatched samples count as 0): the honest deployable score
        "IoU80_overall": round(rows["iou80"].sum() / n_total, 4),
        "IoU_band_overall": round(rows["iou_band"].sum() / n_total, 4),
        "PI_hit80_rate": round(rows["PI_hit80"].mean(), 4),
        "PI_hit_band_rate": round(rows["PI_hit_band"].mean(), 4),
        "MAE_center": round(rows["MAE_center"].mean(), 4),
        "width80_mean": round(rows["width80"].mean(), 2),
    }


def select_rows(test: pd.DataFrame, pick: dict) -> pd.DataFrame:
    """pick: sample_id -> offset. Return the matching grid rows (drop unmatched)."""
    out = []
    for sid, off in pick.items():
        r = test[(test.sample_id == sid) & (test.offset == int(off))]
        if len(r):
            out.append(r.iloc[0])
    return pd.DataFrame(out) if out else pd.DataFrame(columns=test.columns)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    grid = pd.read_csv(GRID)

    # deployable lead-quantile caps (val pool, variant-independent: L/R/alert are ground truth)
    caps, _val_pool, caps_df = build_val_caps([0.10, 0.20], min_bin=20, min_pest=30, bin_w=BIN_W)

    # baseline-trained learned selector per-sample picks (coarse regressor)
    sel = pd.read_csv(SELECTOR_PICKS)
    sel_coarse = sel[sel.selector == "v2_regressor_coarse"]
    learned_pick = dict(zip(sel_coarse.sample_id, sel_coarse.offset.astype(int)))

    summary_rows = []
    persample_frames = []
    per_offset_rows = []

    for variant in VARIANTS:
        gv = grid[grid.variant == variant]
        val = gv[gv.split == "val"].copy()
        test = gv[gv.split == "test"].copy()
        n_total = test.sample_id.nunique()
        test_ids = sorted(test.sample_id.unique())

        # ---- per-offset summary (variant x offset) ----
        for off, sub in test.groupby("offset"):
            blk = metric_block(sub, n_total)
            per_offset_rows.append({"variant": variant, "offset": int(off), **blk})

        # ---- val-based fixed offsets ----
        n_val = val.sample_id.nunique()
        # matched-mean selection (ignores coverage -> can pick a degenerate late offset)
        off_val_iou80 = int(val.groupby("offset")["iou80"].mean().idxmax())
        # coverage-weighted / overall selection (unmatched val samples = 0): deployable-sensible
        val_overall = val.groupby("offset")["iou80"].sum() / n_val
        off_val_overall = int(val_overall.idxmax())

        # ---- policy -> per-sample offset pick ----
        picks = {}
        # oracle per-sample (argmax IoU80) — uses TEST labels
        op = {}
        for sid in test_ids:
            s = test[test.sample_id == sid]
            op[sid] = int(s.loc[s["iou80"].idxmax(), "offset"])
        picks[("oracle_per_sample", "ORACLE")] = op
        # oracle fixed 60 (reference; 60 chosen via test in compare_eval)
        picks[("oracle_fixed_60", "REFERENCE")] = {sid: 60 for sid in test_ids}
        # val fixed (deployable): matched-mean pick (coverage-blind) AND overall pick (coverage-weighted)
        picks[(f"val_fixed_matched(off{off_val_iou80})", "DEPLOY")] = {sid: off_val_iou80 for sid in test_ids}
        picks[(f"val_fixed_overall(off{off_val_overall})", "DEPLOY")] = {sid: off_val_overall for sid in test_ids}
        # deploy_q caps on top of val_fixed_overall base (deployable)
        for q, qname in [(0.10, "deploy_q10"), (0.20, "deploy_q20")]:
            dp = {}
            for sid in test_ids:
                alert = int(test[test.sample_id == sid]["alert_tstar"].iloc[0])
                cap, _lvl = lookup_deploy_cap(caps, q, "WBPH", alert, BIN_W)
                dp[sid] = int(choose_offset(off_val_overall, cap))
            picks[(qname + f"(base off{off_val_overall})", "DEPLOY")] = dp
        # learned selector (baseline-trained) applied to this variant
        picks[("learned_v2_reg_coarse", "DEPLOY")] = {sid: learned_pick[sid]
                                                      for sid in test_ids if sid in learned_pick}

        for (pol, cls), pick in picks.items():
            rows = select_rows(test, pick)
            blk = metric_block(rows, n_total)
            summary_rows.append({"variant": variant, "policy": pol, "class": cls, **blk})
            if len(rows):
                rr = rows.copy(); rr["policy"] = pol; rr["class"] = cls
                persample_frames.append(rr)

    summ = pd.DataFrame(summary_rows)
    per_off = pd.DataFrame(per_offset_rows)
    persample = pd.concat(persample_frames, ignore_index=True) if persample_frames else pd.DataFrame()

    def w(df, name):
        p = OUT_DIR / name
        if p.exists() and not args.force:
            raise SystemExit(f"Refuse to overwrite {p} (use --force)")
        df.to_csv(p, index=False); print(f"  wrote {p} ({len(df)} rows)")

    w(summ, "policy_summary.csv")
    w(per_off, "per_offset_summary.csv")
    w(persample, "per_sample_by_policy.csv")
    w(caps_df[caps_df.pest == "WBPH"], "deploy_caps_wbph.csv")

    # ---- console digest ----
    mcols = ["IoU80", "IoU80_overall", "IoU_band", "late_eval_rate", "late_mu_rate",
             "no_overlap80_rate", "PI_hit80_rate", "MAE_center", "mean_offset", "coverage", "n"]
    print("\n=== POLICY SUMMARY (baseline vs direct_neighbor) ===")
    print("  IoU80 = matched-only (coverage-blind); IoU80_overall = coverage-weighted (unmatched=0) = honest deployable score")
    for cls in ["ORACLE", "REFERENCE", "DEPLOY"]:
        sub = summ[summ["class"] == cls]
        if sub.empty:
            continue
        print(f"\n--- {cls} ---")
        piv = sub.pivot_table(index="policy", columns="variant", values=["IoU80", "IoU80_overall",
                              "coverage", "MAE_center"], aggfunc="first")
        print(piv.to_string())
    print("\n=== full policy table ===")
    print(summ[["variant", "policy", "class"] + mcols].to_string(index=False))

    print("\n=== PER-OFFSET (esp off45/off60) ===")
    print(per_off[["variant", "offset", "n", "IoU80", "IoU_band", "late_eval_rate",
                   "late_mu_rate", "no_overlap80_rate", "MAE_center", "width80_mean"]].to_string(index=False))

    print("\n[ref] lead_v3 baseline (band-IoU, val_best_offset=21): test IoU_matched=0.267 "
          "(from _summary/WBPH_selection.csv; band convention, NOT IoU80)")
    print(f"[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
