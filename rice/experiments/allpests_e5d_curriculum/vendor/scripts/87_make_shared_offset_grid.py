#!/usr/bin/env python
"""87 — offsets 1..75 grid from shared-encoder + offset-specific-head checkpoints (EXP D1|D2).

Mirrors 83: NO feature-channel change (input = unchanged DN 51ch, ckpt norm has 51
entries). The architectural change (shared band-causal encode + per-offset heads) is
rebuilt by injecting the D1/D2 kwargs (read back from the ckpt) into legacy build_model
via functools.partial over the vendored model class. One model forward over the causal
groups already yields mu for every candidate offset (the offset loop below only MATCHES
tstar_doy == alert + off -> no per-offset forward). Grid columns identical to A/B/C so the
existing diagnostic / selector / oracle scripts consume it unchanged.

Run from cropscience cwd:
  PY=/home/gpu4080/research/cropscience/.venv/bin/python
  cd /home/gpu4080/research/cropscience
  PYTHONPATH=/home/gpu4080/research/wbph_interval_perf_202607:/home/gpu4080/research/cropscience \
    $PY /home/gpu4080/research/wbph_interval_perf_202607/scripts/87_make_shared_offset_grid.py --exp D1 --force
"""
from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse, functools
from pathlib import Path
import numpy as np, pandas as pd, torch

WS = Path("/home/gpu4080/research/wbph_interval_perf_202607")
CS = Path("/home/gpu4080/research/cropscience")
import sys; sys.path.insert(0, str(WS)); sys.path.insert(0, str(CS))

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import (split_samples, build_stage2_nowcast_samples,
                              group_stage2_samples_by_site_year, GroupedIntervalEventDataset)
from rice.scripts.common import make_loader
from rice.scripts.stage1_confidence_utils import load_dispatch_feature_table
from rice.src.train_eval import overlap_metrics
import rice.scripts.eval_s2n_direct_compare as E
from src.vendor.model import HierarchicalCausalHazardTransformer as VModel
from rice.scripts.phase_t_dn_wbph_offset_grid import _collect_rows, _collate6

ANCHORS = list(range(1, 76))

# Pest under evaluation. Default WBPH keeps every pre-existing WBPH result bit-identical;
# scripts/allpests/ sets this module global to fan the same collect() over the other pests.
PEST = "WBPH"


def _shared_kwargs_from_ckpt(ckpt):
    """Rebuild geometry for the shared-encode family: D1/D2 (offset-specific heads),
    D3 (shared head), D4 (shared head + per-offset residual)."""
    return dict(
        use_shared_multi_offset=bool(int(ckpt.get("stage2_use_shared_multi_offset", 0))),
        mu_head_mode=str(ckpt.get("stage2_mu_head_mode", "shared")),
        candidate_offsets=[int(x) for x in ckpt.get("stage2_candidate_offsets", [])],
        shared_band_window=int(ckpt.get("stage2_shared_band_window",
                                        ckpt.get("stage2_nowcast_window", 28))),
        use_issue_doy_features=bool(int(ckpt.get("stage2_use_issue_doy_features", 0))),
        doy_period=float(ckpt.get("stage2_offset_doy_period", 365.0)),
        use_offset_residual=bool(int(ckpt.get("stage2_use_offset_residual", 0))),
        residual_hidden_dim=int(ckpt.get("stage2_residual_hidden_dim", 16)),
        residual_scale=float(ckpt.get("stage2_residual_scale", 1.0)),
        zero_init_residual=bool(int(ckpt.get("stage2_zero_init_residual", 1))),
    )


def collect(ckpt_path, year, split, device, vlabel):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    run = int(ckpt["run"]); _, gfc = resolve_pest(PEST)
    C.DOY_START = int(ckpt.get("doy_start", 60)); C.DOY_END = int(ckpt.get("doy_end", 300))
    if ckpt.get("d_model"): C.D_MODEL = int(ckpt["d_model"])
    if ckpt.get("n_head"): C.N_HEAD = int(ckpt["n_head"])
    if ckpt.get("n_layers"): C.N_LAYERS = int(ckpt["n_layers"])
    doy_start = int(C.DOY_START); T = C.DOY_END - C.DOY_START + 1
    sigma = float(ckpt.get("stage2_pmf_sigma", 5.0))

    sk = _shared_kwargs_from_ckpt(ckpt)
    if not sk["use_shared_multi_offset"]:
        raise SystemExit(f"[abort] ckpt is not a shared-multi-offset model: {ckpt_path}")
    E.HierarchicalCausalHazardTransformer = functools.partial(VModel, **sk)
    from rice.scripts.eval_s2n_direct_compare import reconstruct_samples, build_model

    samples, _ = reconstruct_samples(ckpt, run, gfc)   # base(30)+neighbor(6)+dispatch(15)=51 (UNCHANGED)
    conf = load_dispatch_feature_table(ckpt.get("stage2_dispatch_feature_csv"))
    alert_map = {f"{s2}-{int(y)}": int(info["alert_tstar_doy"]) for (s2, y), info in conf.items()}
    _, val_s, test_s = split_samples(samples, val_frac=0.1, test_frac=0.1,
                                     seed=int(ckpt.get("split_seed", 42)), split_mode="year",
                                     val_year=year - 1, test_year_min=year, test_year_max=year)
    split_s = {"val": val_s, "test": test_s}[split]
    if not split_s:
        return pd.DataFrame()
    assert int(split_s[0]["X"].shape[1]) == len(ckpt["norm_mean"]), \
        f"d_in {split_s[0]['X'].shape[1]} != ckpt norm {len(ckpt['norm_mean'])} (input dim must be UNCHANGED)"
    groups = group_stage2_samples_by_site_year(build_stage2_nowcast_samples(
        split_s, window=int(ckpt.get("stage2_nowcast_window", 28)),
        stride=int(ckpt.get("stage2_nowcast_stride", 1)),
        tstar_start=ckpt.get("stage2_nowcast_tstar_start", None),
        only_pre_event=bool(int(ckpt.get("stage2_nowcast_only_pre_event", 1))),
        event_time_proxy=str(ckpt.get("stage2_nowcast_event_time_proxy", "r")),
        require_tstar_before_L=bool(int(ckpt.get("stage2_nowcast_require_tstar_before_L", 0)))))
    loader = make_loader(GroupedIntervalEventDataset(groups, ckpt["norm_mean"], ckpt["norm_std"]),
                         C.BATCH_EVAL, shuffle=False, collate_fn=_collate6)
    model = build_model(ckpt, int(split_s[0]["X"].shape[1]), device)   # vendored + shared kwargs
    # explicit offset list/order guard (strict load already caught a length mismatch)
    model.assert_offsets_match([int(x) for x in ckpt.get("stage2_candidate_offsets", [])])
    rows = _collect_rows(model, loader, groups, int(T), device, sigma)  # ONE forward per batch
    row_map = {(sid, t + doy_start - 1): r for (sid, t), r in rows.items()}
    d = doy_start - 1
    out = []
    for off in ANCHORS:
        for sid, alert_abs in alert_map.items():
            r = row_map.get((sid, int(alert_abs) + int(off)))
            if r is None:
                continue
            tLr, tRr = r["true_L_rel"], r["true_R_rel"]
            iou80, _, _ = overlap_metrics(r["pred_L80_rel"], r["pred_R80_rel"], tLr, tRr)
            out.append({"pest": PEST, "variant": vlabel, "year": int(year), "split": split,
                        "sample_id": sid, "site": r["site"], "offset": int(off),
                        "alert_tstar": int(alert_abs), "eval_tstar": int(alert_abs) + int(off),
                        "L": tLr + d, "R": tRr + d, "true_center": 0.5 * (tLr + tRr) + d,
                        "mu": r["mu_rel"] + d, "sigma": sigma, "pred_point": r["p_point_rel"] + d,
                        "pred_L80": r["pred_L80_rel"] + d, "pred_R80": r["pred_R80_rel"] + d,
                        "iou80": float(iou80)})
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True, choices=["D1", "D2", "D3", "D4"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--smoke", action="store_true", help="read ckpt_smoke/ dir instead of ckpt/")
    args = ap.parse_args()
    vlabel = f"offset_cond_{args.exp}"
    # D1/D2 -> shared_offset_head/, D3/D4 -> shared_head/
    exp_root = "shared_offset_head" if args.exp in ("D1", "D2") else "shared_head"
    root = WS / f"outputs/feature_experiments/{exp_root}/{args.exp}"
    sub = "ckpt_smoke" if args.smoke else "ckpt"
    ckpt = {y: root / f"{sub}/{y}/ckpt/checkpoint_run4.pt" for y in (2022, 2023, 2024)}
    od = root / "grid"; od.mkdir(parents=True, exist_ok=True)
    outp = od / f"wbph_{vlabel}_grid_1to75{'_smoke' if args.smoke else ''}.csv"
    if outp.exists() and not args.force:
        raise SystemExit(f"Refuse to overwrite {outp} (use --force)")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = []
    for y in args.years:
        if not Path(ckpt[y]).exists():
            print(f"[skip] {y}: ckpt missing {ckpt[y]}"); continue
        for split in ("val", "test"):
            df = collect(ckpt[y], y, split, device, vlabel)
            frames.append(df)
            print(f"[ok] {y}/{split}: {len(df)} rows, {df.sample_id.nunique() if len(df) else 0} samples")
    if not frames:
        raise SystemExit("[abort] no rows (missing ckpt?)")
    out = pd.concat(frames, ignore_index=True); out.to_csv(outp, index=False)
    print(f"[87] wrote {outp} ({len(out)} rows) variant={vlabel}")


if __name__ == "__main__":
    main()
