"""Exp 2a — generate a DENSER offset candidate grid for WBPH (ckpt-norm).

New offsets need a REAL model forward (mu at eval_tstar=alert+offset is not
faithfully interpolable). Reuses the validated generator collect_grid() from
phase_t_dn_wbph_offset_grid.py (ckpt["norm_mean"] norm), only widening the offset
set. Produces the same schema as wbph_offset_grid.csv so the analysis/selector code
consumes it unchanged.

Default offsets = coarse {7,14,21,30,45,60} + {28,35,42,49,56}. Use --offsets for a
full 5-day grid, e.g. --offsets 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75.

NO test labels. ckpt-norm only. Does not overwrite without --force.
Output: rice/outputs/diag/stage2_ckptnorm_selector_wbph/iou_improvement/wbph_dense_offset_grid.csv

Example
-------
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_dense_offset_grid --force
  PYTHONPATH=. .venv/bin/python -m rice.scripts.phase_t_wbph_dense_offset_grid \
      --offsets 5 10 15 20 25 30 35 40 45 50 55 60 65 70 75 --force
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

OUT_DIR = RICE_ROOT / "outputs/diag/stage2_ckptnorm_selector_wbph/iou_improvement"
DEFAULT_DENSE = sorted(set(G.ANCHORS + [28, 35, 42, 49, 56]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--offsets", type=int, nargs="+", default=DEFAULT_DENSE)
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--no-validate", action="store_true")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "wbph_dense_offset_grid.csv"
    if out_csv.exists() and not args.force:
        raise SystemExit(f"Refuse to overwrite {out_csv} (use --force)")

    # widen the offset set used inside collect_grid (validated ckpt-norm generator)
    G.ANCHORS = sorted(set(int(o) for o in args.offsets))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[gen] WBPH dense offsets={G.ANCHORS}  device={device}  (ckpt-norm)\n")

    frames = []
    for variant, base in G.VARIANTS.items():
        ck = RICE_ROOT / base / G.PEST / "lead_v3_final" / "ckpt" / "checkpoint_run4.pt"
        for split in G.SPLITS:
            df = G.collect_grid(ck, split, device)
            df["variant"] = variant; df["split"] = split
            frames.append(df)
            print(f"[ok] {variant}/{split}: {len(df)} rows, "
                  f"{df.sample_id.nunique() if len(df) else 0} samples, offsets {sorted(df.offset.unique()) if len(df) else []}")
            if (not args.no_validate) and split == "test" and len(df):
                G.validate_against_matched_eval(df, variant)  # checks overlap offsets vs matched_eval.json

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(out_csv, index=False)
    print(f"\n[done] wrote {out_csv} ({len(out)} rows)")


if __name__ == "__main__":
    main()
