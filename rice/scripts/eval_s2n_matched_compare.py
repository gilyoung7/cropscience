"""Matched site-year eval (PPT unit) for baseline vs S2N-direct, all pests.

Reproduces the run_viz_interval gated-eval unit: for each dispatch-alerted event
site-year, take ONE Stage-2 interval prediction at the nowcast frame
tstar == alert_tstar_doy + offset, then summarize with the SAME
summarize_matched_interval_rows used by run_viz_interval (per-site-year IoU80,
interval-hit, MAE-center). This is the PPT 'Model_IoU' unit, NOT the per-frame
IoU_mean_interval_only used by eval_s2n_direct_compare.py.

No project file is modified; project functions are imported. Reuses
reconstruct_samples + build_model from eval_s2n_direct_compare.

For each pest the offset that maximizes the BASELINE matched IoU80 over a small
grid is chosen, and BOTH variants are reported at that same offset (fair A/B,
PPT-style tuned offset). The full per-offset sweep is stored in the per-pest JSON.

Usage (from cropscience/ root):
  PYTHONPATH=. .venv/bin/python rice/scripts/eval_s2n_matched_compare.py \
      --pests WBPH BPH blast bacterial_blight brown_spot sheath_blight
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
import numpy as np
import torch

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import (
    split_samples, build_stage2_nowcast_samples,
    group_stage2_samples_by_site_year, GroupedIntervalEventDataset,
)
from rice.scripts.common import make_loader, collate_grouped_stage2
from rice.scripts.run_viz_interval import (
    collect_interval_preds_grouped, summarize_matched_interval_rows,
)
from rice.scripts.eval_s2n_direct_compare import reconstruct_samples, build_model

PROD_BASE = "rice/outputs_stage2_batch_2024_bestgate"
DN_BASE = "rice/outputs_stage2_direct_neighbor"
OUT_BASE = "rice/outputs_stage2_compare_eval"
OFFSET_GRID = [0, 7, 14, 21, 30, 45, 60]


def _collate6(batch):
    # collate_grouped_stage2 returns 7 (… + pheno_pad); collect_interval_preds_grouped
    # unpacks 6. These models use phenology_bias_head=0 (pheno unused), so drop it.
    return collate_grouped_stage2(batch)[:6]


def matched_eval_one(ckpt_path, pest, device, offsets):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    run = int(ckpt["run"]); _, get_feature_cols = resolve_pest(pest)
    C.DOY_START = int(ckpt.get("doy_start", 60)); C.DOY_END = int(ckpt.get("doy_end", 300))
    if ckpt.get("d_model"): C.D_MODEL = int(ckpt["d_model"])
    if ckpt.get("n_head"): C.N_HEAD = int(ckpt["n_head"])
    if ckpt.get("n_layers"): C.N_LAYERS = int(ckpt["n_layers"])
    doy_start = int(C.DOY_START); T = C.DOY_END - C.DOY_START + 1

    samples, _ = reconstruct_samples(ckpt, run, get_feature_cols)
    _, _val_s, test_s = split_samples(samples, val_frac=0.1, test_frac=0.1,
                                      seed=int(ckpt.get("split_seed", 42)), split_mode="year",
                                      val_year=2023, test_year_min=2024, test_year_max=2024)
    # true event site-years in the (dispatch) test cohort
    n_true = len({(str(s["site_id"]), int(s["year"]))
                  for s in test_s if str(s.get("censor_type", "right")) != "right"})

    # alert_tstar (DOY) per site-year from the SAME dispatch CSV the model used
    from rice.scripts.stage1_confidence_utils import load_dispatch_feature_table
    conf = load_dispatch_feature_table(ckpt.get("stage2_dispatch_feature_csv"))
    alert_map_abs = {f"{site}-{int(year)}": int(info["alert_tstar_doy"])
                     for (site, year), info in conf.items()}

    # forward Stage-2 over test groups
    test_groups = group_stage2_samples_by_site_year(build_stage2_nowcast_samples(
        test_s, window=int(ckpt.get("stage2_nowcast_window", 28)),
        stride=int(ckpt.get("stage2_nowcast_stride", 1)),
        tstar_start=ckpt.get("stage2_nowcast_tstar_start", None),
        only_pre_event=bool(int(ckpt.get("stage2_nowcast_only_pre_event", 1))),
        event_time_proxy=str(ckpt.get("stage2_nowcast_event_time_proxy", "r")),
        require_tstar_before_L=bool(int(ckpt.get("stage2_nowcast_require_tstar_before_L", 0)))))
    loader = make_loader(GroupedIntervalEventDataset(test_groups, ckpt["norm_mean"], ckpt["norm_std"]),
                         C.BATCH_EVAL, shuffle=False, collate_fn=_collate6)
    model = build_model(ckpt, int(test_s[0]["X"].shape[1]), device)
    rows = collect_interval_preds_grouped(
        model, loader, test_groups, int(T), device,
        getattr(C, "PI_METHOD", "shortest"), 0.8, None, 100000000)
    row_map = {(str(r["sample_id"]), int(r["tstar"]) + doy_start - 1): r
               for r in rows if r.get("tstar") is not None}

    sweep = {}
    for off in offsets:
        matched = []
        for sid, alert_abs in alert_map_abs.items():
            r = row_map.get((sid, int(alert_abs) + int(off)))
            if r is None:
                continue
            mr = dict(r)
            for k in ("true_L", "true_R", "pred_L", "pred_R", "pred_point"):
                mr[k] = int(mr[k]) + doy_start - 1
            matched.append(mr)
        st = summarize_matched_interval_rows(matched, n_true=int(n_true), doy_start=doy_start)
        sweep[int(off)] = {
            "matched_IoU80": float(st["IoU80"]) if matched else float("nan"),
            "PI_hit_rate": float(st["interval_hit_recall"]),     # hits(iou>0)/n_true
            "interval_hit_f1": float(st["interval_hit_f1"]),
            "MAE_center": float(st["MAE_int"]) if matched else float("nan"),
            "matched_event_count": int(len(matched)),
            "n_true": int(n_true),
        }
    return {"ckpt": str(ckpt_path), "d_in": int(test_s[0]["X"].shape[1]),
            "n_true": int(n_true), "alert_cohort": len(alert_map_abs), "sweep": sweep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pests", nargs="+",
                    default=["WBPH", "BPH", "blast", "bacterial_blight", "brown_spot", "sheath_blight"])
    ap.add_argument("--offsets", type=int, nargs="+", default=OFFSET_GRID)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tsv_rows = [["pest", "variant", "matched_IoU80", "PI_hit_rate", "MAE_center",
                 "matched_event_count", "n_true", "offset"]]
    for pest in args.pests:
        jobs = [("baseline", f"{PROD_BASE}/{pest}/lead_v3_final/ckpt/checkpoint_run4.pt"),
                ("direct_neighbor", f"{DN_BASE}/{pest}/lead_v3_final/ckpt/checkpoint_run4.pt")]
        res = {}
        for tag, p in jobs:
            if not Path(p).exists():
                print(f"[skip] {pest}/{tag}: ckpt missing {p}"); continue
            try:
                res[tag] = matched_eval_one(p, pest, device, args.offsets)
            except Exception as e:
                print(f"[ERROR] {pest}/{tag}: {type(e).__name__}: {e}")
        if "baseline" not in res:
            print(f"[skip] {pest}: no baseline"); continue
        # choose offset maximizing baseline matched_IoU80 (ignore nan)
        bsw = res["baseline"]["sweep"]
        valid = [(o, v["matched_IoU80"]) for o, v in bsw.items() if np.isfinite(v["matched_IoU80"])]
        best_off = max(valid, key=lambda x: x[1])[0] if valid else args.offsets[0]
        out_dir = Path(OUT_BASE) / pest; out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "matched_eval.json").write_text(json.dumps({"best_offset": best_off, **res}, indent=2, default=str))
        for tag in ("baseline", "direct_neighbor"):
            if tag not in res:
                continue
            v = res[tag]["sweep"][best_off]
            tsv_rows.append([pest, tag, f"{v['matched_IoU80']:.4f}", f"{v['PI_hit_rate']:.4f}",
                             f"{v['MAE_center']:.4f}", str(v["matched_event_count"]),
                             str(v["n_true"]), str(best_off)])
        print(f"[{pest}] best_offset={best_off} | "
              + " | ".join(f"{t}:IoU80={res[t]['sweep'][best_off]['matched_IoU80']:.4f}"
                           for t in res if t in ("baseline", "direct_neighbor")))

    out_tsv = Path(OUT_BASE) / "all_pests_direct_neighbor_matched_compare.tsv"
    out_tsv.write_text("\n".join("\t".join(r) for r in tsv_rows) + "\n")
    print("\n===== MATCHED (site-year) COMPARISON =====")
    print("\n".join("\t".join(r) for r in tsv_rows))
    print(f"\n[saved] {out_tsv}")


if __name__ == "__main__":
    main()
