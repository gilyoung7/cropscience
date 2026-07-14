"""Minimal derived weather utilities — appends 7 channels to base_X.

Derived from base_X (no raw daily CSV reload), using the run=4 channel layout
(channel indices in base_X):
  0  rain_7d_sum
  1  rain_7d_days
  2  tmean_7d_mean
  3  tmax_7d_max
  5  rh_7d_mean

Channels added:
  1 vpd_7d_mean        proxy = Magnus(tmean_7d_mean) * (1 - rh_7d_mean/100)
  2 vpd_7d_max         proxy = Magnus(tmax_7d_max)   * (1 - rh_7d_mean/100)
  3 rain_28d_sum       28-day rolling mean of rain_7d_sum (mm-equivalent)
  4 gdd_28d_sum        28-day rolling sum of max(tmean_7d_mean - 10, 0)
  5 rainy_days_streak  proxy: rain_7d_days (last-7d wet days)
  6 dry_streak         proxy: 7 - rain_7d_days
  7 humid_rain_proxy   rh_7d_mean * rain_7d_days
"""

from __future__ import annotations

import numpy as np


DERIVED_WEATHER_NAMES = [
    "vpd_7d_mean", "vpd_7d_max", "rain_28d_sum", "gdd_28d_sum",
    "rainy_days_streak", "dry_streak", "humid_rain_proxy",
]
DERIVED_WEATHER_DIM = len(DERIVED_WEATHER_NAMES)

_CH_RAIN_7D_SUM = 0
_CH_RAIN_7D_DAYS = 1
_CH_TMEAN_7D_MEAN = 2
_CH_TMAX_7D_MAX = 3
_CH_RH_7D_MEAN = 5


def _magnus_es_kpa(T: np.ndarray) -> np.ndarray:
    return 0.6108 * np.exp(17.27 * T / (T + 237.3 + 1e-8))


def _vpd_kpa(T: np.ndarray, rh_percent: np.ndarray) -> np.ndarray:
    return _magnus_es_kpa(T) * (1.0 - np.clip(rh_percent, 0.0, 100.0) / 100.0)


def _rolling_apply(arr: np.ndarray, window: int, reduce: str) -> np.ndarray:
    """Right-aligned causal rolling; same length as input."""
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    if reduce == "mean":
        for i in range(n):
            lo = max(0, i - window + 1)
            out[i] = float(np.nanmean(arr[lo:i + 1]))
    elif reduce == "sum":
        for i in range(n):
            lo = max(0, i - window + 1)
            out[i] = float(np.nansum(arr[lo:i + 1]))
    else:
        raise ValueError(reduce)
    return out


def append_derived_weather_to_samples(samples: list[dict]) -> int:
    n_done = 0
    for s in samples:
        X_old = np.asarray(s["X"], dtype=np.float32)
        T_season = int(X_old.shape[0])
        tmean = X_old[:, _CH_TMEAN_7D_MEAN]
        tmax = X_old[:, _CH_TMAX_7D_MAX]
        rh = X_old[:, _CH_RH_7D_MEAN]
        rain7 = X_old[:, _CH_RAIN_7D_SUM]
        rdays7 = X_old[:, _CH_RAIN_7D_DAYS]
        vpd_mean = _vpd_kpa(tmean, rh)
        vpd_max = _vpd_kpa(tmax, rh)
        rain_28 = _rolling_apply(rain7, 28, "mean")
        gdd_daily = np.maximum(tmean - 10.0, 0.0).astype(np.float32)
        gdd_28 = _rolling_apply(gdd_daily, 28, "sum")
        rainy_streak = rdays7.astype(np.float32).copy()
        dry_streak = (7.0 - rdays7).astype(np.float32)
        humid_rain = (rh * rdays7).astype(np.float32)
        block = np.stack(
            [vpd_mean.astype(np.float32),
             vpd_max.astype(np.float32),
             rain_28.astype(np.float32),
             gdd_28.astype(np.float32),
             rainy_streak, dry_streak, humid_rain],
            axis=1,
        )
        block = np.nan_to_num(block, nan=0.0, posinf=0.0, neginf=0.0)
        s["X"] = np.concatenate([X_old, block], axis=1).astype(np.float32)
        n_done += 1
    return n_done
