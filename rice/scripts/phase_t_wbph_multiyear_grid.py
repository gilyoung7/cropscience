"""Generate ckpt-norm offset grids for WBPH across years (2022/2023/2024) for the
offset-grid ablation + year-stability study. Generates offsets 1..75 in ONE forward
pass per (variant, year) so the ablation script can subset any offset set (dense /
uniform-5day / 1-day) without re-running the model.

Per-year HELD-OUT checkpoints (test_year=Y, val_year=Y-1):
  baseline        2022,2023,2024  (batch_2022_baseline / batch_2023_baseline / batch_2024_bestgate)
  direct_neighbor 2024 ONLY       (no per-year DN ckpt exists)
Missing (variant, year) pairs are SKIPPED with a warning. Each ckpt carries its own
ckpt["norm_mean"] (correct) and its own dispatch CSV — collect_grid uses them.

ckpt-norm only. No wrong-norm / phase_r. Does not overwrite without --force.
Output: rice/outputs/diag/stage2_ckptnorm_selector_wbph/multiyear/wbph_grid_1to75_multiyear.csv

Example
-------
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_multiyear_grid --force
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_multiyear_grid --years 2024 --force
"""
from __future__ import annotations

import os
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse
from pathlib import Path

import pandas as pd
import torch

from rice.configs.base import RICE_ROOT
import rice.scripts.phase_t_dn_wbph_offset_grid as G

OUT_DIR = RICE_ROOT / "outputs/diag/stage2_ckptnorm_selector_wbph/multiyear"
MAX_OFFSET = 75   # generate 1..75; ablation subsets later

# (variant, test_year) -> ckpt path (val_year = test_year - 1)
CKPTS = {
    ("baseline", 2022): "outputs/stage2/batch_2022_baseline/WBPH/lead_v3_final/ckpt/checkpoint_run4.pt",
    ("baseline", 2023): "outputs/stage2/batch_2023_baseline/WBPH/lead_v3_final/ckpt/checkpoint_run4.pt",
    ("baseline", 2024): "outputs/stage2/batch_2024_bestgate/WBPH/lead_v3_final/ckpt/checkpoint_run4.pt",
    ("direct_neighbor", 2024): "outputs/stage2/direct_neighbor/WBPH/lead_v3_final/ckpt/checkpoint_run4.pt",
    # rolling DN ckpts for held-out 2022/2023 (produced by run_s2n_direct_rolling.sh;
    # skipped automatically until they exist). Each is trained val=Y-1/test=Y, NOT the 2024 ckpt.
    ("direct_neighbor", 2022): "outputs/stage2/direct_neighbor_rolling/2022/WBPH/lead_v3_final/ckpt/checkpoint_run4.pt",
    ("direct_neighbor", 2023): "outputs/stage2/direct_neighbor_rolling/2023/WBPH/lead_v3_final/ckpt/checkpoint_run4.pt",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=int, nargs="+", default=[2022, 2023, 2024])
    ap.add_argument("--variants", nargs="+", default=["baseline", "direct_neighbor"])
    ap.add_argument("--max-offset", type=int, default=MAX_OFFSET)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "wbph_grid_1to75_multiyear.csv"
    if out_csv.exists() and not args.force:
        raise SystemExit(f"Refuse to overwrite {out_csv} (use --force)")

    anchors = list(range(1, int(args.max_offset) + 1))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen] WBPH multiyear grids, offsets 1..{args.max_offset}, device={device} (ckpt-norm)\n")

    frames, skipped = [], []
    for variant in args.variants:
        for year in args.years:
            rel = CKPTS.get((variant, year))
            ck = (RICE_ROOT / rel) if rel else None
            if ck is None or not ck.exists():
                skipped.append((variant, year))
                print(f"[skip] {variant}/{year}: no ckpt (held-out per-year model unavailable)")
                continue
            for split in ("val", "test"):
                df = G.collect_grid(ck, split, device, val_year=year - 1, test_year=year, anchors=anchors)
                df["variant"] = variant; df["year"] = year; df["split"] = split
                frames.append(df)
                print(f"[ok] {variant}/{year}/{split}: {len(df)} rows, "
                      f"{df.sample_id.nunique() if len(df) else 0} samples")

    if not frames:
        raise SystemExit("[abort] no grids generated.")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(f"\n[done] wrote {out_csv} ({len(out)} rows)")
    if skipped:
        print(f"[note] SKIPPED (no ckpt): {skipped}  -> these (variant,year) cannot be evaluated.")
    print("[validate] 2024 only has a matched_eval.json reference; other years rely on the "
          "identical ckpt-norm inference path (no independent reference).")


if __name__ == "__main__":
    main()
