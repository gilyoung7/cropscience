"""Generate WBPH per-sample offset grid (val 2023 + test 2024) for baseline AND
direct_neighbor Stage-2 models — the input artifact for multi-policy re-evaluation.

WHY: the direct_neighbor width/IoU80 gain reported earlier was measured at a
per-pest TEST-oracle fixed offset (WBPH=60), NOT a deployable selector. To judge
whether the gain survives a deployable policy we need a per-(sample, offset) grid
on BOTH val (for selecting offsets without test leakage) and test (for scoring).

This MIRRORS the proven inference path in phase_t_directN_interval_width.py
(reconstruct_samples + build_model handle neighbor + dispatch channels; the
grouped causal-tstar forward produces the gaussian mu and the conditional PMF).
NO model/loss/selector change. Pure inference + bookkeeping.

Per (variant, split, sample_id, offset) row we record:
  mu_DOY        gaussian center (model._last_mu_BK, +doy_start-1)   [continuous]
  pred_point    median of conditional PMF (DOY)
  band_L/band_R mu +/- 1.96*sigma  (fixed ~95% Gaussian band; sigma=5 fixed)
  iou_band      IoU of [band_L, band_R] vs true [L+1, R]            (offset_constraint convention)
  pred_L80/R80  80% shortest-mass interval (DOY)
  iou80         IoU of the 80% interval vs true                     (matched_IoU80 convention)
  width80       pred_R80 - pred_L80 + 1
  L_DOY/R_DOY   true interval (DOY); alert_tstar (DOY); eval_tstar = alert + offset

Validation: regenerating the baseline grid must reproduce the existing
lead_v3_test_sample_grid.csv mu and iou_matched (run with --validate).

Run:
    PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_dn_wbph_offset_grid --force
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rice.configs import config as C
from rice.configs.base import RICE_ROOT
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import (split_samples, build_stage2_nowcast_samples,
                              group_stage2_samples_by_site_year, GroupedIntervalEventDataset,
                              compute_norm_stats)
from rice.scripts.common import make_loader, collate_grouped_stage2
from rice.scripts.eval_s2n_direct_compare import reconstruct_samples, build_model
from rice.scripts.stage1_confidence_utils import load_dispatch_feature_table, DISPATCH_TOTAL_CHANNELS
from rice.src.train_eval import (hazard_to_pmf_cdf_logS, shortest_mass_interval_1d,
                                 overlap_metrics, CTYPE_INTERVAL)

PEST = "WBPH"
ANCHORS = [7, 14, 21, 30, 45, 60]            # coarse offset grid (deployable schema)
Z = 1.96
VARIANTS = {
    "baseline": "outputs/stage2/batch_2024_bestgate",
    "direct_neighbor": "outputs/stage2/direct_neighbor",
}
OUT_DIR = RICE_ROOT / "outputs/diag/stage2_direct_neighbor_wbph_2024"
SPLITS = ["val", "test"]   # val=2023, test=2024


def _collate6(batch):
    return collate_grouped_stage2(batch)[:6]


@torch.no_grad()
def _collect_rows(model, loader, source_groups, T, device, sigma):
    """Per (sample_id, relative tstar) interval rows with mu + 80% interval + band."""
    rows = {}
    group_idx = 0
    model.eval()
    for X, L, R, ctype, tstar, valid_mask in loader:
        X = X.to(device, non_blocking=True)
        tstar_t = tstar.to(device, non_blocking=True)
        valid_mask_t = valid_mask.to(device, non_blocking=True)
        hazard = model(X, tstar=tstar_t, valid_mask=valid_mask_t)
        B, K, T_h = hazard.shape
        pmf, _cdf, logS = hazard_to_pmf_cdf_logS(hazard.reshape(B * K, T_h),
                                                 tstar=tstar_t.reshape(B * K))
        pmf_np = pmf.cpu().numpy().reshape(B, K, T_h)
        logS_np = logS.cpu().numpy().reshape(B, K, T_h)
        mu_BK = model._last_mu_BK
        mu_np = mu_BK.detach().cpu().numpy() if mu_BK is not None else np.full((B, K), np.nan)
        L_np = L.cpu().numpy().astype(int)
        R_np = R.cpu().numpy().astype(int)
        ctype_np = ctype.cpu().numpy().astype(int)
        tstar_np = tstar.cpu().numpy().astype(int)
        valid_np = valid_mask.cpu().numpy().astype(bool)

        for bi in range(B):
            group = source_groups[group_idx + bi] if (group_idx + bi) < len(source_groups) else {"samples": []}
            srows = group.get("samples", [])
            for ki in range(K):
                if not valid_np[bi, ki] or int(ctype_np[bi, ki]) != int(CTYPE_INTERVAL):
                    continue
                meta = srows[ki] if ki < len(srows) else {}
                pmf_raw = pmf_np[bi, ki]
                tstar_val = int(tstar_np[bi, ki])
                if tstar_val > 0:
                    pmf_raw = pmf_raw.copy()
                    pmf_raw[:tstar_val] = 0.0
                total = float(np.sum(pmf_raw))
                pmf_cond = pmf_raw / total if total > 0 else pmf_raw
                cdf_cond = np.cumsum(pmf_cond)

                pL80, pR80, _ = shortest_mass_interval_1d(pmf_cond, target_mass=0.8,
                                                          Tend=T, normalize=False)
                pL80 = max(1, min(int(pL80), int(T)))
                pR80 = max(1, min(int(pR80), int(T)))
                if pL80 > pR80:
                    pL80, pR80 = pR80, pL80
                if total > 0 and cdf_cond[-1] >= 0.5:
                    p_point = int(np.searchsorted(cdf_cond, 0.5) + 1)
                else:
                    p_point = int(T)

                mu_rel = float(mu_np[bi, ki])   # relative 1..T units
                site = meta.get("site_id")
                year = meta.get("year")
                sid = f"{site}-{int(year)}" if site is not None and year is not None else f"{group_idx+bi}-{ki}"
                rows[(sid, tstar_val)] = {
                    "sample_id": sid, "site": site, "year": year,
                    "tstar_rel": tstar_val,
                    "true_L_rel": int(L_np[bi, ki]), "true_R_rel": int(R_np[bi, ki]),
                    "mu_rel": mu_rel, "p_point_rel": p_point,
                    "pred_L80_rel": pL80, "pred_R80_rel": pR80,
                }
        group_idx += B
    return rows


def collect_grid(ckpt_path: Path, split: str, device,
                 val_year: int = 2023, test_year: int = 2024, anchors=None) -> pd.DataFrame:
    """Per-(sample, offset) ckpt-norm grid. val_year/test_year set the year split
    (defaults preserve the original WBPH 2024 behavior); anchors overrides the
    offset set (defaults to module ANCHORS)."""
    anchors_use = list(anchors) if anchors is not None else ANCHORS
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    run = int(ckpt["run"]); _, get_feature_cols = resolve_pest(PEST)
    C.DOY_START = int(ckpt.get("doy_start", 60)); C.DOY_END = int(ckpt.get("doy_end", 300))
    if ckpt.get("d_model"): C.D_MODEL = int(ckpt["d_model"])
    if ckpt.get("n_head"): C.N_HEAD = int(ckpt["n_head"])
    if ckpt.get("n_layers"): C.N_LAYERS = int(ckpt["n_layers"])
    doy_start = int(C.DOY_START); T = C.DOY_END - C.DOY_START + 1
    sigma = float(ckpt.get("stage2_pmf_sigma", 5.0))

    samples, _ = reconstruct_samples(ckpt, run, get_feature_cols)
    train_s, val_s, test_s = split_samples(samples, val_frac=0.1, test_frac=0.1,
                                           seed=int(ckpt.get("split_seed", 42)), split_mode="year",
                                           val_year=int(val_year),
                                           test_year_min=int(test_year), test_year_max=int(test_year))
    split_s = {"val": val_s, "test": test_s}[split]
    if not split_s:
        return pd.DataFrame()

    conf = load_dispatch_feature_table(ckpt.get("stage2_dispatch_feature_csv"))
    alert_map = {f"{site}-{int(year)}": int(info["alert_tstar_doy"]) for (site, year), info in conf.items()}

    groups = group_stage2_samples_by_site_year(build_stage2_nowcast_samples(
        split_s, window=int(ckpt.get("stage2_nowcast_window", 28)),
        stride=int(ckpt.get("stage2_nowcast_stride", 1)),
        tstar_start=ckpt.get("stage2_nowcast_tstar_start", None),
        only_pre_event=bool(int(ckpt.get("stage2_nowcast_only_pre_event", 1))),
        event_time_proxy=str(ckpt.get("stage2_nowcast_event_time_proxy", "r")),
        require_tstar_before_L=bool(int(ckpt.get("stage2_nowcast_require_tstar_before_L", 0)))))
    # Normalization: use ckpt["norm_mean"]/["norm_std"] — the EXACT stats the model
    # was TRAINED with (run_train.py saves compute_norm_stats(nowcast-expanded train)
    # with dispatch channels forced RAW). Verified: ckpt norm == compute_norm_stats(
    # build_stage2_nowcast_samples(train))+raw, bit-exact. This matches eval_s2n_direct_compare
    # (the compare_eval/matched_eval source). NB: phase_r_oracle_iou recomputes norm over
    # SEASON-level train (NOT nowcast) -> mis-scales inputs and shifts mu ~20d; its
    # lead_v3_test_sample_grid.csv / selector_cross_split / v2 selector outputs are on the
    # WRONG norm. We deliberately do NOT follow phase_r here.
    loader = make_loader(GroupedIntervalEventDataset(groups, ckpt["norm_mean"], ckpt["norm_std"]),
                         C.BATCH_EVAL, shuffle=False, collate_fn=_collate6)
    model = build_model(ckpt, int(split_s[0]["X"].shape[1]), device)
    rows = _collect_rows(model, loader, groups, int(T), device, sigma)

    # key by (sid, absolute tstar DOY) for offset matching against alert (DOY)
    row_map = {(sid, t + doy_start - 1): r for (sid, t), r in rows.items()}

    out = []
    for off in anchors_use:
        for sid, alert_abs in alert_map.items():
            r = row_map.get((sid, int(alert_abs) + int(off)))
            if r is None:
                continue
            tLr, tRr = r["true_L_rel"], r["true_R_rel"]
            mu_rel, pp_rel = r["mu_rel"], r["p_point_rel"]
            # 80% interval IoU (shift-invariant -> compute in rel coords)
            iou80, rec80, prec80 = overlap_metrics(r["pred_L80_rel"], r["pred_R80_rel"], tLr, tRr)
            # fixed ~95% Gaussian band mu +/- 1.96 sigma (offset_constraint convention)
            bL_rel = int(round(mu_rel - Z * sigma)); bR_rel = int(round(mu_rel + Z * sigma))
            iou_band, recb, precb = overlap_metrics(bL_rel, bR_rel, tLr, tRr)
            # to DOY
            d = doy_start - 1
            L_DOY = tLr + d; R_DOY = tRr + d
            mu_DOY = mu_rel + d
            true_start = L_DOY + 1; true_mid = 0.5 * (L_DOY + R_DOY)
            eval_doy = int(alert_abs) + int(off)
            out.append({
                "pest": PEST, "sample_id": sid, "site": r["site"], "year": r["year"],
                "offset": int(off), "alert_tstar": int(alert_abs), "eval_tstar": eval_doy,
                "L": L_DOY, "R": R_DOY, "true_start": true_start, "true_mid": true_mid,
                "mu": mu_DOY, "sigma": sigma, "pred_point": pp_rel + d,
                "band_L": bL_rel + d, "band_R": bR_rel + d,
                "pred_L80": r["pred_L80_rel"] + d, "pred_R80": r["pred_R80_rel"] + d,
                "width80": r["pred_R80_rel"] - r["pred_L80_rel"] + 1,
                "iou80": float(iou80), "iou_band": float(iou_band),
                # metrics (eval-time scoring; no selection here)
                "late_eval": bool(eval_doy > true_start),
                "late_mu": bool(mu_DOY > true_mid),
                "no_overlap80": bool(iou80 == 0.0),
                "no_overlap_band": bool(iou_band == 0.0),
                "PI_hit_band": bool(bL_rel + d <= true_mid <= bR_rel + d),
                "PI_hit80": bool(r["pred_L80_rel"] + d <= true_mid <= r["pred_R80_rel"] + d),
                "MAE_center": float(abs(mu_DOY - true_mid)),          # |mu - true_mid| (offset_constraint convention)
                "MAE_point": float(abs((pp_rel + d) - true_mid)),     # |median - true_mid| (matched_compare convention)
            })
    return pd.DataFrame(out)


def write_no_overwrite(df, path: Path, force):
    if path.exists() and not force:
        raise SystemExit(f"Refuse to overwrite: {path} (use --force)")
    df.to_csv(path, index=False); print(f"  wrote {path} ({len(df)} rows)")


def validate_against_matched_eval(test_df: pd.DataFrame, variant: str):
    """Validate against compare_eval/WBPH/matched_eval.json (the CORRECT ckpt-norm path:
    eval_s2n_matched_compare). Compares per-offset IoU80, MAE_center(median), PI_hit."""
    import json
    jp = RICE_ROOT / "outputs/stage2/compare_eval" / PEST / "matched_eval.json"
    if not jp.exists():
        print(f"[validate] no matched_eval.json at {jp}"); return
    sweep = json.load(open(jp))[variant]["sweep"]
    print(f"[validate {variant}] per-offset vs matched_eval.json (ckpt-norm reference):")
    print(f"  {'off':>4} {'n_mine':>6} {'n_ref':>5} {'IoU80_mine':>10} {'IoU80_ref':>9} {'dIoU':>7} "
          f"{'MAEpt_mine':>10} {'MAEpt_ref':>9}")
    for off in ANCHORS:
        sub = test_df[test_df.offset == off]
        if str(off) not in sweep or len(sub) == 0:
            continue
        ref = sweep[str(off)]
        print(f"  {off:>4} {len(sub):>6} {ref['matched_event_count']:>5} "
              f"{sub['iou80'].mean():>10.4f} {ref['matched_IoU80']:>9.4f} "
              f"{sub['iou80'].mean()-ref['matched_IoU80']:>+7.4f} "
              f"{sub['MAE_point'].mean():>10.3f} {ref['MAE_center']:>9.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--validate", action="store_true", default=True)
    args = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen] WBPH offset grid (val 2023 + test 2024), device={device}. NO model change.\n")

    frames = []
    for variant, base in VARIANTS.items():
        ck = RICE_ROOT / base / PEST / "lead_v3_final" / "ckpt" / "checkpoint_run4.pt"
        for split in SPLITS:
            df = collect_grid(ck, split, device)
            df["variant"] = variant; df["split"] = split
            frames.append(df)
            print(f"[ok] {variant}/{split}: {len(df)} rows, "
                  f"{df.sample_id.nunique() if len(df) else 0} samples, "
                  f"offsets {sorted(df.offset.unique()) if len(df) else []}")
            if args.validate and split == "test" and len(df):
                validate_against_matched_eval(df, variant)
    allrows = pd.concat(frames, ignore_index=True)
    write_no_overwrite(allrows, OUT_DIR / "wbph_offset_grid.csv", args.force)
    print(f"\n[done] grid in {OUT_DIR}/wbph_offset_grid.csv")


if __name__ == "__main__":
    main()
