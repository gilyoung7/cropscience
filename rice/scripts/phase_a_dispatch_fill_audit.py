"""Phase A.4 — causal-fill audit for dispatch confidence features.

Rebuilds the same Stage-2 nowcast cohort as run_train would (using the nowcast
settings stored in the Stage-2 ckpt meta), appends dispatch confidence features
via stage1_confidence_utils in the exact same mode as training, and reports:

  - per-split (train/val/test) season-level row counts: filled vs missing
  - per-(case_bucket) nowcast-window fill rates (pre_L / in_LR / post_R / right)
  - per-alerted-sy post-alert row-count distribution
  - no-alert-sy fill confirmation (should be zero filled rows under causal-fill)

Read-only diagnostic. Does NOT load Stage-2 weights or run any forward pass.
"""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples, build_stage2_nowcast_samples
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.stage1_confidence_utils import (
    DISPATCH_FEATURE_DIM,
    load_dispatch_feature_table,
    append_dispatch_confidence_to_samples,
    train_mean_features_from_table,
)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--ckpt", required=True,
                    help="Stage-2 ckpt with stage2_dispatch_features_added=True.")
    ap.add_argument("--dispatch_feature_csv", default=None,
                    help="Override CSV path; default uses path from ckpt meta.")
    ap.add_argument("--mode_override", default=None,
                    help="Override fill mode (default: ckpt meta).")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    default_doy_start = getattr(C, "DOY_START", 60)
    default_doy_end = getattr(C, "DOY_END", 300)
    doy_start = int(ckpt.get("doy_start", default_doy_start))
    C.DOY_START = doy_start
    C.DOY_END = int(ckpt.get("doy_end", default_doy_end))

    nc_window = int(ckpt.get("stage2_nowcast_window", 28))
    nc_stride = int(ckpt.get("stage2_nowcast_stride", 1))
    nc_tstart = ckpt.get("stage2_nowcast_tstar_start", None)
    nc_only_pre = bool(int(ckpt.get("stage2_nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("stage2_nowcast_event_time_proxy", "r"))
    nc_req = bool(int(ckpt.get("stage2_nowcast_require_tstar_before_L", 0)))

    csv_path = args.dispatch_feature_csv or ckpt.get("stage2_dispatch_feature_csv")
    if not csv_path:
        raise SystemExit("[abort] no dispatch feature CSV in ckpt and none provided")
    fill_mode = str(args.mode_override or ckpt.get("stage2_dispatch_feature_mode", "causal"))
    miss_val = float(ckpt.get("stage2_dispatch_feature_missing_value", 0.0))

    print(f"[audit] ckpt={Path(args.ckpt).name}")
    print(f"[audit] csv={csv_path}")
    print(f"[audit] fill_mode={fill_mode}  doy_start={doy_start}  "
          f"nowcast(window={nc_window}, stride={nc_stride}, only_pre_event={nc_only_pre}, "
          f"event_time_proxy={nc_proxy})")

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples = build_samples_for_run(args.run, get_feature_cols)

    conf_map = load_dispatch_feature_table(csv_path)
    train_mean = (train_mean_features_from_table(csv_path)
                  if fill_mode == "ablate_train_mean" else None)
    stats = append_dispatch_confidence_to_samples(
        samples, conf_map, doy_start=doy_start, mode=fill_mode,
        missing_value=miss_val, ablate_train_mean_features=train_mean,
    )
    print(f"[audit] all-samples append stats: {stats}")

    train_s, val_s, test_s = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )

    # (1) Per-split season-level rows
    print("\n## (1) per-split season-level row counts (last channel = missing indicator)")
    print(f"  {'split':>6} {'n_sy':>6} {'sy_with_filled':>16} {'sy_all_missing':>16} "
          f"{'rows_filled':>13} {'rows_missing':>13} {'fill_ratio':>11}")
    for tag, seas in [("train", train_s), ("val", val_s), ("test", test_s)]:
        n_sy = len(seas)
        n_with = n_no = 0
        rf = rm = 0
        for s in seas:
            miss = np.asarray(s["X"], dtype=np.float32)[:, -1]
            nf = int((miss == 0.0).sum())
            nm = int((miss == 1.0).sum())
            rf += nf
            rm += nm
            if nf > 0:
                n_with += 1
            else:
                n_no += 1
        ratio = rf / max(rf + rm, 1)
        print(f"  {tag:>6} {n_sy:>6} {n_with:>16} {n_no:>16} "
              f"{rf:>13} {rm:>13} {ratio:>11.4f}")

    # (2) Per-bucket nowcast-window fill (any t in the W-window has missing=0)
    print("\n## (2) nowcast-window fill by case_bucket "
          "(window 'filled' if ANY t in [tstar-W+1, tstar] has dispatch features)")
    nc_kw = dict(window=nc_window, stride=nc_stride, tstar_start=nc_tstart,
                 only_pre_event=nc_only_pre, event_time_proxy=nc_proxy,
                 require_tstar_before_L=nc_req)
    for tag, seas in [("train", train_s), ("val", val_s), ("test", test_s)]:
        nc_s = build_stage2_nowcast_samples(seas, **nc_kw)
        bucket_fill: dict = {}
        for sp in nc_s:
            bucket = str(sp.get("case_bucket", "?"))
            X = np.asarray(sp["X"], dtype=np.float32)
            miss = X[:, -1]
            any_filled = int((miss == 0.0).any())
            d = bucket_fill.setdefault(bucket, [0, 0])
            d[0] += any_filled
            d[1] += 1
        print(f"  {tag:>5} nowcast windows = {len(nc_s)}")
        for b in sorted(bucket_fill):
            nf, nt = bucket_fill[b]
            print(f"    bucket={b:>10s}  fill_ratio={nf/max(nt,1):.4f}  ({nf}/{nt})")

    # (3) Per-alerted-sy: post-alert row counts
    print("\n## (3) post-alert season-row counts per alerted sy "
          "(causal mode: rows >= alert_t_rel)")
    sy_to_T: dict = {(str(s["site_id"]), int(s["year"])): int(s["X"].shape[0])
                     for s in samples}
    post_counts = []
    for sy, info in conf_map.items():
        T_season = sy_to_T.get(sy)
        if T_season is None:
            continue
        alert_t_doy = int(info["alert_tstar_doy"])
        alert_t_rel = alert_t_doy - doy_start + 1
        idx = max(0, min(T_season, alert_t_rel))
        post_counts.append(T_season - idx)
    arr = np.asarray(post_counts, dtype=int) if post_counts else None
    if arr is not None and arr.size:
        print(f"  n_alerted_sy={arr.size}  mean={arr.mean():.1f}  "
              f"median={int(np.median(arr))}  min={int(arr.min())}  max={int(arr.max())}")
        print(f"  quantiles: q05={int(np.percentile(arr,5))}  "
              f"q25={int(np.percentile(arr,25))}  q50={int(np.percentile(arr,50))}  "
              f"q75={int(np.percentile(arr,75))}  q95={int(np.percentile(arr,95))}")
        zero_post = int((arr == 0).sum())
        lt7 = int((arr < 7).sum())
        lt28 = int((arr < 28).sum())
        print(f"  zero_post_rows={zero_post}  (<7 days)={lt7}  (<W=28 days)={lt28}")

    # (4) no-alert sy fill confirmation
    print("\n## (4) no-alert sy fill confirmation (should be all-missing under causal-fill)")
    n_no_alert = 0
    n_no_alert_filled = 0
    for s in samples:
        sy = (str(s["site_id"]), int(s["year"]))
        if sy in conf_map:
            continue
        n_no_alert += 1
        miss = np.asarray(s["X"], dtype=np.float32)[:, -1]
        if (miss == 0.0).any():
            n_no_alert_filled += 1
    print(f"  no-alert_sy={n_no_alert}  with_any_filled_row={n_no_alert_filled}  "
          f"(expected 0 under causal-fill; non-zero only valid for broadcast or ablation modes)")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
