#!/usr/bin/env python
"""Per-pest path resolution shared by the all-pest grid / split / eval steps.

configs/paths.yaml is WBPH-only and frozen (it is the provenance record for the WBPH result).
Rather than edit it, we synthesise the same dict shape per pest. src/io_utils.load_dispatch and
load_clim_mid take `paths` as an argument, so a synthetic dict is all they need -- no patching.

The dispatch filename is NOT constant: the best Stage-1 gate was chosen per (pest, year), so it
is one of gate_{dispatch_group_tau,A_baseline,D_history}_R088_features_per_sy.csv. We glob and
assert exactly one match instead of assuming a name.
"""
from __future__ import annotations
from pathlib import Path

WS = Path("/home/gpu4080/research/wbph_interval_perf_202607")
CS = Path("/home/gpu4080/research/cropscience")
OUT_ROOT = WS / "outputs/allpests_e5d"
YEARS = [2022, 2023, 2024]
OFFSETS = [3, 7, 14, 21, 28, 30, 35, 42, 45, 49, 56, 60]
SEEDS = [0, 1, 2, 3, 4]
SIGMA = 8.0
SHIFT_GRID = list(range(-12, 16))


def batch_dir(year: int) -> str:
    return "batch_2024_bestgate" if year == 2024 else f"batch_{year}_baseline"


def cell_dir(pest: str, year: int) -> Path:
    return CS / f"rice/outputs/stage2/{batch_dir(year)}/{pest}"


def dispatch_csv(pest: str, year: int) -> Path:
    hits = sorted(cell_dir(pest, year).glob("gate_*_R088_features_per_sy.csv"))
    if len(hits) != 1:
        raise SystemExit(f"[paths] expected exactly 1 dispatch CSV for {pest}/{year}, got {hits}")
    return hits[0]


def clim_csv(pest: str, year: int) -> Path:
    p = cell_dir(pest, year) / "climatology_train_stats.csv"
    if not p.exists():
        raise SystemExit(f"[paths] missing climatology for {pest}/{year}: {p}")
    return p


def synthetic_paths(pest: str) -> dict:
    """Same shape src/io_utils expects. Keys keep their legacy 'wbph_' prefix on purpose --
    io_utils indexes them by that literal name; renaming would fork io_utils for no gain."""
    return {"selected_inputs": {
        "wbph_stage1_dispatch":  {y: str(dispatch_csv(pest, y)) for y in YEARS},
        "wbph_clim_train_stats": {y: str(clim_csv(pest, y)) for y in YEARS},
    }}


def geometry(pest: str) -> tuple[int, int]:
    """(doy_start, T) straight from pests.tsv. BPH is the outlier: 140..270 -> T=131."""
    for line in (CS / "rice/experiments/allpests_e5d/pests.tsv").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        f = line.split()
        if f[0] == pest:
            return int(f[2]), int(f[3]) - int(f[2]) + 1
    raise SystemExit(f"[paths] {pest} not in pests.tsv")


# ---- output layout: everything for one pest lives under one dir, dev and clean kept apart ----
def pest_root(pest: str) -> Path:            return OUT_ROOT / pest
def dev_ckpt(pest, year):                    return pest_root(pest) / f"dev/ckpt/{year}"
def dev_grid(pest):                          return pest_root(pest) / f"dev/grid/{pest}_E5d_grid_1to75.csv"
def clean_root(pest):                        return pest_root(pest) / "clean"
def clean_ckpt(pest, year):                  return clean_root(pest) / f"ckpt/{year}"
def clean_grid(pest):                        return clean_root(pest) / f"grid/{pest}_E5d_clean_grid_1to75.csv"
def split_assignment(pest):                  return clean_root(pest) / "split_assignment.json"
def eval_dir(pest):                          return pest_root(pest) / "eval"
def log_dir(pest):                           return pest_root(pest) / "logs"
