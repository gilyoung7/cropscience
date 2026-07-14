"""WBPH-only: ckpt-norm rebuild of the Stage-2 offset-candidate grid + v2 selector
retrain + deployable-policy comparison. Fixes the phase_r_oracle_iou normalization
bug (it recomputes SEASON-level norm; the model was trained with ckpt["norm_mean"]
= nowcast-expanded-train norm + raw dispatch channels). See reference_stage2_norm_paths.

This does NOT modify phase_r_oracle_iou. It:
  1. reads the validated ckpt-norm grid (rice/outputs/diag/stage2_direct_neighbor_wbph_2024/
     wbph_offset_grid.csv; mu/IoU80 match matched_eval.json bit-exact),
  2. emits ckpt-norm lead_v3_{val,test}_sample_grid CSVs (lead_v3 schema + dispatch feats),
  3. retrains the v2 selector (reusing phase_b_stage2_offset_selector_v2_ranking helpers
     UNCHANGED: build_candidate_rows / train_lgb_* / pick_offsets),
  4. compares policies: original_wrongnorm (existing selector picks, WRONG-norm trained),
     val_fixed, deploy_q10, deploy_q20, learned_v2_ckptnorm — for baseline AND direct_neighbor.

Selector internal IoU is the mu+/-1.96sigma BAND IoU (iou_from_mu); we ALSO report IoU80
(shortest-mass) by joining the validated grid. Wrong-norm artifacts are preserved; all new
files carry a _ckptnorm suffix and go to a separate folder.

Run:
    PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_dn_wbph_ckptnorm_selector --force
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rice.configs.base import RICE_ROOT
from rice.scripts.phase_b_stage2_offset_selector_v2_ranking import (
    build_candidate_rows, train_lgb_regressor, train_lgb_ranker, train_lgb_classifier,
    pick_offsets, COARSE_OFFSETS, DENSE_OFFSETS, SAMPLE_FEATURES)
from rice.scripts.phase_t_stage2_offset_constraint import (
    build_val_caps, lookup_deploy_cap, choose_offset)

PEST = "WBPH"
GRID = RICE_ROOT / "outputs/diag/stage2_direct_neighbor_wbph_2024/wbph_offset_grid.csv"
DISPATCH_CSV = RICE_ROOT / "outputs/stage2/batch_2024_bestgate/WBPH/gate_dispatch_group_tau_R088_features_per_sy.csv"
CLIM_CSV = RICE_ROOT / "outputs/stage2/batch_2024_bestgate/WBPH/climatology_train_stats.csv"
MATCHED_EVAL = RICE_ROOT / "outputs/stage2/compare_eval/WBPH/matched_eval.json"
# existing WRONG-NORM selector picks (phase_r-based), kept for contrast only
WRONGNORM_SELECTOR_PICKS = RICE_ROOT / "outputs/stage2/selector_cross_split/2024_WBPH/v2_per_sample_test_selections.csv"
OUT_DIR = RICE_ROOT / "outputs/diag/stage2_ckptnorm_selector_wbph"
VARIANTS = ["baseline", "direct_neighbor"]
BIN_W = 20


def emit_sample_grid(grid: pd.DataFrame, variant: str, split: str, disp: pd.DataFrame) -> pd.DataFrame:
    """lead_v3_*_sample_grid schema (ckpt-norm mu) + dispatch SAMPLE_FEATURES."""
    g = grid[(grid.variant == variant) & (grid.split == split)].copy()
    # alert_tstar comes from the dispatch table (disp) to avoid a merge name clash;
    # it is identical to the grid's alert_tstar (both absolute DOY).
    out = g[["sample_id", "offset", "mu", "L", "R", "sigma"]].copy()
    out = out.merge(disp, on="sample_id", how="left")
    return out


def metrics_from_picks(pick: dict, joined: pd.DataFrame, n_total: int) -> dict:
    """pick: sample_id->offset. joined: per-(sample_id,offset) grid rows w/ all metrics."""
    rows = []
    for sid, off in pick.items():
        r = joined[(joined.sample_id == sid) & (joined.offset == int(off))]
        if len(r):
            rows.append(r.iloc[0])
    if not rows:
        return {"n": 0, "coverage": 0.0}
    d = pd.DataFrame(rows)
    n = len(d)
    return {
        "n": int(n), "coverage": round(n / n_total, 4), "mean_offset": round(d["offset"].mean(), 2),
        "IoU80": round(d["iou80"].mean(), 4), "IoU80_overall": round(d["iou80"].sum() / n_total, 4),
        "IoU_band": round(d["iou_band"].mean(), 4), "IoU_band_overall": round(d["iou_band"].sum() / n_total, 4),
        "late_mu_rate": round(d["late_mu"].mean(), 4),
        "no_overlap80_rate": round(d["no_overlap80"].mean(), 4),
        "PI_hit80_rate": round(d["PI_hit80"].mean(), 4),
        "MAE_center": round(d["MAE_center"].mean(), 4),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    grid = pd.read_csv(GRID)
    clim_mid = float(pd.read_csv(CLIM_CSV).iloc[0]["mean_mid"])
    print(f"[init] clim_mid_doy={clim_mid:.2f}  grid rows={len(grid)}")

    disp_raw = pd.read_csv(DISPATCH_CSV)
    disp_raw["sample_id"] = disp_raw["site"].astype(str) + "-" + disp_raw["year"].astype(int).astype(str)
    disp = disp_raw[["sample_id"] + SAMPLE_FEATURES + ["dispatch_branch"]].drop_duplicates("sample_id")

    caps, _vp, caps_df = build_val_caps([0.10, 0.20], min_bin=20, min_pest=30, bin_w=BIN_W)

    wn = pd.read_csv(WRONGNORM_SELECTOR_PICKS)
    wn = wn[wn.selector == "v2_regressor_coarse"]
    wrongnorm_pick = dict(zip(wn.sample_id, wn.offset.astype(int)))

    def w(df, name):
        p = OUT_DIR / name
        if p.exists() and not args.force:
            raise SystemExit(f"Refuse to overwrite {p} (use --force)")
        df.to_csv(p, index=False); print(f"  wrote {p} ({len(df)} rows)")

    w(caps_df[caps_df.pest == PEST], "deploy_caps_wbph_ckptnorm.csv")

    summary = []
    val_offsets_used = {}
    for variant in VARIANTS:
        # 1) emit ckpt-norm lead_v3 sample grids
        val_sg = emit_sample_grid(grid, variant, "val", disp)
        test_sg = emit_sample_grid(grid, variant, "test", disp)
        val_p = OUT_DIR / f"lead_v3_val_sample_grid_{variant}_ckptnorm.csv"
        test_p = OUT_DIR / f"lead_v3_test_sample_grid_{variant}_ckptnorm.csv"
        w(val_sg, val_p.name); w(test_sg, test_p.name)

        # 2) candidate rows (reuse v2 helpers UNCHANGED) — selector iou = BAND iou
        val_dense = build_candidate_rows(val_p, DENSE_OFFSETS, clim_mid)
        test_coarse = build_candidate_rows(test_p, COARSE_OFFSETS, clim_mid)
        w(val_dense, f"v2_per_candidate_val_{variant}_ckptnorm.csv")

        # 3) retrain selector on ckpt-norm val
        reg = train_lgb_regressor(val_dense)
        learned = pick_offsets(reg, test_coarse, "regressor")
        learned_pick = dict(zip(learned.sample_id, learned.offset.astype(int)))

        # joined per-(sample,offset) metric table = the validated grid (test)
        joined = grid[(grid.variant == variant) & (grid.split == "test")].copy()
        n_total = joined.sample_id.nunique()
        test_ids = sorted(joined.sample_id.unique())

        # val-best fixed offset (coverage-weighted overall band IoU on val)
        vt = grid[(grid.variant == variant) & (grid.split == "val")]
        n_val = vt.sample_id.nunique()
        off_val = int((vt.groupby("offset")["iou_band"].sum() / n_val).idxmax())
        val_offsets_used[variant] = off_val

        policies = {
            ("learned_v2_ckptnorm", "DEPLOY"): learned_pick,
            ("original_wrongnorm", "DEPRECATED"): {s: wrongnorm_pick[s] for s in test_ids if s in wrongnorm_pick},
            (f"val_fixed(off{off_val})", "DEPLOY"): {s: off_val for s in test_ids},
        }
        for q, qn in [(0.10, "deploy_q10"), (0.20, "deploy_q20")]:
            dp = {}
            for s in test_ids:
                alert = int(joined[joined.sample_id == s]["alert_tstar"].iloc[0])
                cap, _ = lookup_deploy_cap(caps, q, PEST, alert, BIN_W)
                dp[s] = int(choose_offset(off_val, cap))
            policies[(qn + f"(base off{off_val})", "DEPLOY")] = dp

        for (pol, cls), pick in policies.items():
            summary.append({"variant": variant, "policy": pol, "class": cls,
                            **metrics_from_picks(pick, joined, n_total)})

    summ = pd.DataFrame(summary)
    w(summ, "policy_summary_ckptnorm.csv")

    # ---- validation echo: grid IoU80 vs matched_eval.json (already bit-exact at gen) ----
    sweep = {v: json.load(open(MATCHED_EVAL))[v]["sweep"] for v in VARIANTS}
    print("\n=== VALIDATION: ckpt-norm grid IoU80 vs matched_eval.json (per offset) ===")
    for variant in VARIANTS:
        t = grid[(grid.variant == variant) & (grid.split == "test")]
        diffs = []
        for off in COARSE_OFFSETS:
            sub = t[t.offset == off]
            if str(off) in sweep[variant] and len(sub):
                diffs.append(abs(sub["iou80"].mean() - sweep[variant][str(off)]["matched_IoU80"]))
        print(f"  {variant:16s} max|ΔIoU80|={max(diffs):.5f}  (0 => grid correct)")

    print("\n=== CKPT-NORM POLICY COMPARISON (WBPH 2024) ===")
    print("  IoU_band = selector's own metric (mu±1.96σ); IoU80 = shortest-mass (matched_eval convention)")
    print("  *_overall = coverage-weighted (unmatched=0). original_wrongnorm = WRONG-norm-trained selector picks.")
    cols = ["variant", "policy", "class", "n", "coverage", "mean_offset",
            "IoU_band", "IoU_band_overall", "IoU80", "IoU80_overall", "late_mu_rate", "MAE_center"]
    print(summ[cols].to_string(index=False))
    print(f"\n  val-selected fixed offset per variant: {val_offsets_used}")
    print(f"[done] outputs in {OUT_DIR}")


if __name__ == "__main__":
    main()
