#!/usr/bin/env python
"""E5d clean-selection training launcher.

Restricts the trainer's VALIDATION set to the fold's val_ckpt sample IDs so that the
checkpoint (best raw val_iou80) is selected on val_ckpt ONLY -- val_fit and val_cal are
never seen by training. Achieved by patching rice.src.dataset.split_samples BEFORE
src.vendor.run_train imports it (run_train does `from rice.src.dataset import split_samples`,
so the name is bound at import time to whatever the attribute is then). NO existing file is modified.

Usage (invoked by train_clean_fold.sh):
  python _patched_train.py --eval_year 2022 --assign /path/split_assignment.json -- <run_train args...>
"""
import argparse, json, runpy, sys
from pathlib import Path

WS = Path("/home/gpu4080/research/wbph_interval_perf_202607")
CS = Path("/home/gpu4080/research/cropscience")
sys.path.insert(0, str(WS)); sys.path.insert(0, str(CS))

ap = argparse.ArgumentParser()
ap.add_argument("--eval_year", type=int, required=True)
ap.add_argument("--assign", required=True, help="split_assignment.json from make_splits.py")
ap.add_argument("rest", nargs=argparse.REMAINDER, help="-- then the run_train args")
a = ap.parse_args()
rest = a.rest[1:] if a.rest and a.rest[0] == "--" else a.rest

assign = json.loads(Path(a.assign).read_text())[str(a.eval_year)]
KEEP = {s for s, b in assign.items() if b == "val_ckpt"}
if not KEEP:
    raise SystemExit(f"[abort] no val_ckpt ids for eval_year={a.eval_year}")

import rice.src.dataset as DS
_orig = DS.split_samples

def _sid(s):
    return f"{s['site_id']}-{int(s['year'])}"

def patched_split_samples(samples, *args, **kw):
    tr, va, te = _orig(samples, *args, **kw)
    va2 = [s for s in va if _sid(s) in KEEP]
    print(f"[clean-split] val restricted to val_ckpt: {len(va)} -> {len(va2)} samples "
          f"(eval_year={a.eval_year}, val_ckpt ids={len(KEEP)})", flush=True)
    if not va2:
        raise SystemExit("[abort] val_ckpt filter produced an EMPTY validation set")
    return tr, va2, te

DS.split_samples = patched_split_samples          # patch BEFORE run_train imports it

sys.argv = ["run_train"] + rest
print(f"[clean-train] eval_year={a.eval_year} val_ckpt={len(KEEP)} ids; argv={' '.join(rest[:6])} ...", flush=True)
runpy.run_module("src.vendor.run_train", run_name="__main__")
