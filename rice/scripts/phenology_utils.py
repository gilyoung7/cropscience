"""Fuzzy phenology feature utilities — appends 11 channels to base_X.

Channels (per (site, year, day)):
  Static (broadcast across T, from LONG2 obs csv):
    1  growing_start_doy
    2  growing_end_doy
    3  growing_mid_doy
    4  growing_len_days
    5  best_suitability
    6  best_months
    7  offset_days
    8  window_idx
  Dynamic (per-day, tstar-relative):
    9  phenology_progress              = (day_DOY - growing_start_doy) / growing_len_days
   10  days_to_growing_mid             = growing_mid_doy - day_DOY
   11  abs_days_to_growing_mid

(Note: days_since_growing_start_at_t / days_until_growing_end_at_t / is_growing_at_t
already live in base_X channels 12-14 from data_pipeline — not duplicated here.)

Missing phenology (rare; LONG2 csv null) -> zero padding for all 11 channels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from rice.configs import config as C


PHENO_STATIC_NAMES = [
    "growing_start_doy",
    "growing_end_doy",
    "growing_mid_doy",
    "growing_len_days",
    "best_suitability",
    "best_months",
    "offset_days",
    "window_idx",
]
PHENO_DYNAMIC_NAMES = [
    "phenology_progress",
    "days_to_growing_mid",
    "abs_days_to_growing_mid",
]
PHENO_FEATURE_NAMES = PHENO_STATIC_NAMES + PHENO_DYNAMIC_NAMES
PHENO_FEATURE_DIM = len(PHENO_FEATURE_NAMES)
_PHENO_COLS_LONG2 = PHENO_STATIC_NAMES  # all 8 static come from LONG2 csv


def load_pheno_map() -> dict:
    """resolve_pest(pest) must have been called so C.PATH_OBS is set."""
    df = pd.read_csv(C.PATH_OBS, encoding="utf-8-sig",
                     usecols=["site_id", "year"] + _PHENO_COLS_LONG2)
    df = df.drop_duplicates(["site_id", "year"]).copy()
    out = {}
    for r in df.itertuples(index=False):
        key = (str(r.site_id), int(r.year))
        out[key] = {nm: (float(getattr(r, nm)) if pd.notna(getattr(r, nm)) else None)
                    for nm in _PHENO_COLS_LONG2}
    return out


def append_pheno_to_samples(samples: list[dict], pheno_map: dict, doy_start: int) -> int:
    n_done = 0
    for s in samples:
        key = (str(s["site_id"]), int(s["year"]))
        info = pheno_map.get(key)
        X_old = np.asarray(s["X"], dtype=np.float32)
        T_season = int(X_old.shape[0])
        block = np.zeros((T_season, PHENO_FEATURE_DIM), dtype=np.float32)
        if info is None or info.get("growing_start_doy") is None or info.get("growing_len_days") is None:
            s["X"] = np.concatenate([X_old, block], axis=1).astype(np.float32)
            n_done += 1
            continue
        gs = float(info["growing_start_doy"])
        ge = float(info["growing_end_doy"]) if info.get("growing_end_doy") is not None else gs
        gm = float(info["growing_mid_doy"]) if info.get("growing_mid_doy") is not None else (gs + ge) / 2
        glen = float(info["growing_len_days"]) if info.get("growing_len_days") is not None else max(ge - gs, 1.0)
        bs = float(info["best_suitability"]) if info.get("best_suitability") is not None else 0.0
        bm = float(info["best_months"]) if info.get("best_months") is not None else 0.0
        offd = float(info["offset_days"]) if info.get("offset_days") is not None else 0.0
        widx = float(info["window_idx"]) if info.get("window_idx") is not None else 0.0
        static_vals = np.array([gs, ge, gm, glen, bs, bm, offd, widx], dtype=np.float32)
        block[:, :8] = np.tile(static_vals[None, :], (T_season, 1))
        doys = np.arange(T_season, dtype=np.float32) + float(doy_start)
        denom = glen if glen > 0 else 1.0
        block[:, 8] = (doys - gs) / denom
        dtgm = gm - doys
        block[:, 9] = dtgm
        block[:, 10] = np.abs(dtgm)
        s["X"] = np.concatenate([X_old, block], axis=1).astype(np.float32)
        n_done += 1
    return n_done
