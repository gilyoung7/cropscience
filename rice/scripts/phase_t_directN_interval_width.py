"""direct_neighbor vs baseline: per-event 80%-mass interval WIDTH analysis (diagnosis only).

Question: is the direct_neighbor matched_IoU80 gain a real timing improvement, or does the
80% predicted interval just get WIDER (so it overlaps the truth more easily)?

Method (reuses the exact eval machinery that produced compare_eval):
  * load each pest's baseline + direct_neighbor checkpoint, run Stage-2 inference on TEST,
  * collect_interval_preds_grouped -> per (site-year, nowcast tstar) 80% shortest-mass interval
    [pred_L, pred_R] and IoU vs true [L+1, R],
  * match each true event at tstar == alert_tstar + offset (same as eval_s2n_matched_compare),
  * width = pred_R - pred_L + 1 (days).  Compare width & IoU between variants on the SAME events.

NO model/loss/selector change. Outputs to rice/outputs/diag/stage2_directN_width/.

Run:
    python -m rice.scripts.phase_t_directN_interval_width
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")   # run_viz_interval imports matplotlib

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rice.configs import config as C
from rice.configs.base import RICE_ROOT
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import (split_samples, build_stage2_nowcast_samples,
                              group_stage2_samples_by_site_year, GroupedIntervalEventDataset)
from rice.scripts.common import make_loader, collate_grouped_stage2
from rice.scripts.run_viz_interval import collect_interval_preds_grouped
from rice.scripts.eval_s2n_direct_compare import reconstruct_samples, build_model
from rice.scripts.stage1_confidence_utils import load_dispatch_feature_table

PROD_BASE = "outputs/stage2/batch_2024_bestgate"      # baseline (lead_v3, no neighbor)
DN_BASE = "outputs/stage2/direct_neighbor"            # direct_neighbor (lead_v3 + neighbor)
COMPARE = "outputs/stage2/compare_eval"
PESTS = ["WBPH", "BPH", "blast", "bacterial_blight", "brown_spot", "sheath_blight"]  # DN ckpts exist
OFFSETS = [0, 7, 14, 21, 30, 45, 60]


def _collate6(batch):
    return collate_grouped_stage2(batch)[:6]


def collect_variant(ckpt_path: Path, pest: str, device) -> pd.DataFrame:
    """Per-matched-event rows (one per true event x offset) with 80% interval width + IoU."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    run = int(ckpt["run"]); _, get_feature_cols = resolve_pest(pest)
    C.DOY_START = int(ckpt.get("doy_start", 60)); C.DOY_END = int(ckpt.get("doy_end", 300))
    if ckpt.get("d_model"): C.D_MODEL = int(ckpt["d_model"])
    if ckpt.get("n_head"): C.N_HEAD = int(ckpt["n_head"])
    if ckpt.get("n_layers"): C.N_LAYERS = int(ckpt["n_layers"])
    doy_start = int(C.DOY_START); T = C.DOY_END - C.DOY_START + 1

    samples, _ = reconstruct_samples(ckpt, run, get_feature_cols)
    _, _val, test_s = split_samples(samples, val_frac=0.1, test_frac=0.1,
                                    seed=int(ckpt.get("split_seed", 42)), split_mode="year",
                                    val_year=2023, test_year_min=2024, test_year_max=2024)
    conf = load_dispatch_feature_table(ckpt.get("stage2_dispatch_feature_csv"))
    alert_map = {f"{site}-{int(year)}": int(info["alert_tstar_doy"]) for (site, year), info in conf.items()}

    groups = group_stage2_samples_by_site_year(build_stage2_nowcast_samples(
        test_s, window=int(ckpt.get("stage2_nowcast_window", 28)),
        stride=int(ckpt.get("stage2_nowcast_stride", 1)),
        tstar_start=ckpt.get("stage2_nowcast_tstar_start", None),
        only_pre_event=bool(int(ckpt.get("stage2_nowcast_only_pre_event", 1))),
        event_time_proxy=str(ckpt.get("stage2_nowcast_event_time_proxy", "r")),
        require_tstar_before_L=bool(int(ckpt.get("stage2_nowcast_require_tstar_before_L", 0)))))
    loader = make_loader(GroupedIntervalEventDataset(groups, ckpt["norm_mean"], ckpt["norm_std"]),
                         C.BATCH_EVAL, shuffle=False, collate_fn=_collate6)
    model = build_model(ckpt, int(test_s[0]["X"].shape[1]), device)
    rows = collect_interval_preds_grouped(model, loader, groups, int(T), device,
                                          getattr(C, "PI_METHOD", "shortest"), 0.8, None, 10**8)
    # key by (sid, absolute tstar)
    row_map = {(str(r["sample_id"]), int(r["tstar"]) + doy_start - 1): r
               for r in rows if r.get("tstar") is not None}

    out = []
    for off in OFFSETS:
        for sid, alert_abs in alert_map.items():
            r = row_map.get((sid, int(alert_abs) + int(off)))
            if r is None:
                continue
            pL, pR = int(r["pred_L"]), int(r["pred_R"])
            tL, tR = int(r["true_L"]), int(r["true_R"])
            out.append({
                "pest": pest, "sample_id": sid, "offset": int(off),
                "alert_tstar": int(alert_abs), "eval_tstar": int(alert_abs) + int(off),
                "pred_width": pR - pL + 1, "true_width": tR - tL + 1,
                "iou": float(r["iou"]),
                "mae_center": abs(int(r["pred_point"]) - (tL + tR) / 2.0),
            })
    return pd.DataFrame(out)


WSTATS = ["mean", "median", "std", "q25", "q75", "q90", "q95", "min", "max"]


def wstats(s: pd.Series) -> dict:
    s = s.dropna()
    if s.empty:
        return {k: np.nan for k in WSTATS} | {"n": 0}
    return {"n": int(len(s)), "mean": round(s.mean(), 2), "median": round(s.median(), 2),
            "std": round(s.std(), 2), "q25": round(s.quantile(.25), 2), "q75": round(s.quantile(.75), 2),
            "q90": round(s.quantile(.90), 2), "q95": round(s.quantile(.95), 2),
            "min": round(s.min(), 2), "max": round(s.max(), 2)}


def write_no_overwrite(df, path: Path, force):
    if path.exists() and not force:
        raise SystemExit(f"Refuse to overwrite: {path} (use --force)")
    df.to_csv(path, index=False); print(f"  wrote {path} ({len(df)} rows)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default=str(RICE_ROOT / "outputs/diag/stage2_directN_width"))
    ap.add_argument("--pests", nargs="+", default=PESTS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[diag] direct_neighbor vs baseline 80% interval WIDTH (device={device}). "
          "NO model/selector change.\n")

    best_off = {}
    for p in args.pests:
        jf = RICE_ROOT / COMPARE / p / "matched_eval.json"
        best_off[p] = int(json.load(open(jf))["best_offset"]) if jf.exists() else 60

    frames = []
    for pest in args.pests:
        for tag, base in [("baseline", PROD_BASE), ("direct_neighbor", DN_BASE)]:
            ck = RICE_ROOT / base / pest / "lead_v3_final" / "ckpt" / "checkpoint_run4.pt"
            if not ck.exists():
                print(f"[skip] {pest}/{tag}: no ckpt"); continue
            df = collect_variant(ck, pest, device); df["variant"] = tag
            frames.append(df)
            print(f"[ok] {pest}/{tag}: {len(df)} matched rows over offsets {OFFSETS}")
    allrows = pd.concat(frames, ignore_index=True)
    write_no_overwrite(allrows, out_dir / "per_event_width.csv", args.force)

    # rows AT each pest's best_offset (the set behind the headline matched_IoU80)
    best = allrows[allrows.apply(lambda r: r["offset"] == best_off[r["pest"]], axis=1)].copy()

    # ---- Q1 overall (at best_offset) ----
    print("\n=== Q1. 80% interval WIDTH overall (at each pest's best_offset) ===")
    ov = []
    for tag in ["baseline", "direct_neighbor"]:
        s = best[best.variant == tag]["pred_width"]
        ov.append({"variant": tag, **wstats(s),
                   "mean_iou": round(best[best.variant == tag]["iou"].mean(), 4),
                   "mean_true_width": round(best[best.variant == tag]["true_width"].mean(), 2)})
    ovdf = pd.DataFrame(ov); print(ovdf.to_string(index=False))
    write_no_overwrite(ovdf, out_dir / "width_summary_overall.csv", args.force)

    # ---- Q2 per pest ----
    print("\n=== Q2. WIDTH per pest x variant (at best_offset) ===")
    rows = []
    for pest in args.pests:
        for tag in ["baseline", "direct_neighbor"]:
            s = best[(best.pest == pest) & (best.variant == tag)]
            rows.append({"pest": pest, "variant": tag, "best_offset": best_off[pest], **wstats(s["pred_width"]),
                         "mean_iou": round(s["iou"].mean(), 4) if len(s) else np.nan})
    bypest = pd.DataFrame(rows)
    print(bypest[["pest", "variant", "best_offset", "n", "mean", "median", "q90", "max", "mean_iou"]].to_string(index=False))
    write_no_overwrite(bypest, out_dir / "width_summary_by_pest.csv", args.force)

    # ---- Q3 threshold fractions ----
    print("\n=== Q3. fraction of events with WIDTH > threshold (at best_offset) ===")
    thr = []
    for tag in ["baseline", "direct_neighbor"]:
        s = best[best.variant == tag]["pred_width"]
        thr.append({"variant": tag, "n": int(len(s)),
                    "frac_gt30": round((s > 30).mean(), 4), "frac_gt45": round((s > 45).mean(), 4),
                    "frac_gt60": round((s > 60).mean(), 4), "frac_gt90": round((s > 90).mean(), 4)})
    thrdf = pd.DataFrame(thr); print(thrdf.to_string(index=False))
    write_no_overwrite(thrdf, out_dir / "width_threshold_fractions.csv", args.force)

    # ---- Q4 WBPH detail ----
    print("\n=== Q4. WBPH width — overall + offset 45/60 cohorts ===")
    wb = allrows[allrows.pest == "WBPH"]
    wrows = []
    for tag in ["baseline", "direct_neighbor"]:
        for off in [best_off.get("WBPH", 60), 45, 60]:
            s = wb[(wb.variant == tag) & (wb.offset == off)]
            tagn = f"off{off}" + ("(best)" if off == best_off.get("WBPH") else "")
            wrows.append({"variant": tag, "cohort": tagn, **wstats(s["pred_width"]),
                          "mean_iou": round(s["iou"].mean(), 4) if len(s) else np.nan})
    wbdf = pd.DataFrame(wrows).drop_duplicates(["variant", "cohort"])
    print(wbdf[["variant", "cohort", "n", "mean", "median", "q90", "max", "mean_iou"]].to_string(index=False))
    write_no_overwrite(wbdf, out_dir / "wbph_width_by_offset.csv", args.force)

    # WBPH: do high-IoU samples have larger width? (per variant, IoU quartile -> mean width)
    print("\n  WBPH width vs IoU (at offset 60): IoU quartile -> mean width / corr")
    for tag in ["baseline", "direct_neighbor"]:
        s = wb[(wb.variant == tag) & (wb.offset == 60)].copy()
        if len(s) < 4:
            print(f"   {tag}: n={len(s)} too few"); continue
        s["iou_q"] = pd.qcut(s["iou"].rank(method="first"), 4, labels=["Q1", "Q2", "Q3", "Q4"])
        mw = s.groupby("iou_q")["pred_width"].mean().round(1).to_dict()
        corr = s[["pred_width", "iou"]].corr().iloc[0, 1]
        print(f"   {tag:16s} corr(width,iou)={corr:+.3f}  width by IoU-quartile {mw}")

    # ---- Q5 / width-IoU relationship overall ----
    print("\n=== Q5/relationship. width-vs-IoU (all matched offsets pooled) ===")
    for tag in ["baseline", "direct_neighbor"]:
        s = allrows[allrows.variant == tag]
        corr = s[["pred_width", "iou"]].corr().iloc[0, 1]
        print(f"  {tag:16s} n={len(s):5d}  corr(width,iou)={corr:+.3f}  "
              f"mean_width={s.pred_width.mean():.1f}  mean_iou={s.iou.mean():.3f}")

    print(f"\n[done] outputs in {out_dir}")


if __name__ == "__main__":
    main()
