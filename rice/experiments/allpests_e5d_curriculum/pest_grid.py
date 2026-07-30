#!/usr/bin/env python
"""Offsets 1..75 candidate grid for one pest, for the dev or the clean E5d checkpoints.

This IS the raw prediction artifact. Both the dev calibration eval and the clean fold-isolated
eval are computed from these CSVs, so preserving them is what makes requirement "compute both
protocols later" hold without retraining.

Reuses scripts/87_make_shared_offset_grid.collect() verbatim -- same forward, same row
semantics, same columns -- with its module-level PEST switched. Nothing about the model or the
row math is reimplemented here.

  cd /home/gpu4080/research/cropscience
  PYTHONPATH=$WS:$CS $PY $CS/rice/experiments/allpests_e5d/pest_grid.py --pest blast --mode dev --force
"""
from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse, importlib.util, sys
from pathlib import Path
import pandas as pd, torch

WS = Path("/home/gpu4080/research/wbph_interval_perf_202607")
CS = Path("/home/gpu4080/research/cropscience")
sys.path.insert(0, str(CS / "rice/experiments/allpests_e5d_curriculum"))
sys.path.insert(0, str(CS / "rice/experiments/allpests_e5d/vendor"))   # pinned deps only
import pest_paths as PP

_spec = importlib.util.spec_from_file_location("mk87", str(CS / "rice/experiments/allpests_e5d/vendor/scripts/87_make_shared_offset_grid.py"))
mk87 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(mk87)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", required=True)
    ap.add_argument("--mode", required=True, choices=["dev", "clean"])
    ap.add_argument("--years", type=int, nargs="+", default=PP.YEARS)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()

    mk87.PEST = a.pest                      # <- the only knob that makes collect() pest-aware
    vlabel = "E5d" if a.mode == "dev" else "E5d_clean"
    ckpt_of = PP.dev_ckpt if a.mode == "dev" else PP.clean_ckpt
    outp = PP.dev_grid(a.pest) if a.mode == "dev" else PP.clean_grid(a.pest)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists() and not a.force:
        raise SystemExit(f"[grid] refuse to overwrite {outp} (use --force)")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    frames = []
    for y in a.years:
        ck = ckpt_of(a.pest, y) / "ckpt/checkpoint_run4.pt"
        if not ck.exists():
            print(f"[skip] {a.pest}/{y}: ckpt missing {ck}"); continue
        for split in ("val", "test"):
            df = mk87.collect(ck, y, split, device, vlabel)
            frames.append(df)
            n = df.sample_id.nunique() if len(df) else 0
            print(f"[ok] {a.pest} {a.mode} {y}/{split}: {len(df)} rows, {n} samples")
    if not frames:
        raise SystemExit(f"[grid] no rows for {a.pest} ({a.mode}) -- train first")

    out = pd.concat(frames, ignore_index=True)
    if out.empty:
        raise SystemExit(f"[grid] empty grid for {a.pest} ({a.mode})")
    assert set(out["pest"].unique()) == {a.pest}, f"pest column leaked: {out['pest'].unique()}"
    out.to_csv(outp, index=False)
    print(f"[grid] wrote {outp} ({len(out)} rows, variant={vlabel})")


if __name__ == "__main__":
    main()
