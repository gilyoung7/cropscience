"""Site-history feature utilities shared by training + evaluation.

11 channels appended to base_X (per-(site, year, day)):
  Static (broadcast across T):
    1  prev_year_L_doy_at_site               (0 if missing; see flag)
    2  prev_year_event_at_site               (1 if last year had event AT THIS SITE)
    3  site_avg_L_doy_recent3y               (0 if missing; see flag)
    4  years_since_last_event_at_site        (99 if no prior event)
    5  n_events_recent5y_at_site             (0..5)
    6  prev_year_L_miss                      (1 if prev_year_L missing)
    7  site_avg_L_recent3y_miss              (1 if avg3y missing)
  Dynamic (per-day, varies with DOY):
    8  days_to_prev_year_L                   = prev_year_L_doy - day_DOY  (0 if miss)
    9  abs_days_to_prev_year_L               (0 if miss)
   10  days_to_site_avg_L_recent3y           = avg3y - day_DOY  (0 if miss)
   11  abs_days_to_site_avg_L_recent3y       (0 if miss)

Policies:
  rolling      : history uses any year < target year  (operational rolling)
  strict_train : history capped at year <= train_year_max  (no val/test leak)
"""

from __future__ import annotations

import numpy as np
import pandas as pd


HISTORY_STATIC_NAMES = [
    "prev_year_L_doy_at_site",
    "prev_year_event_at_site",
    "site_avg_L_doy_recent3y",
    "years_since_last_event_at_site",
    "n_events_recent5y_at_site",
    "prev_year_L_miss",
    "site_avg_L_recent3y_miss",
]
HISTORY_DYNAMIC_NAMES = [
    "days_to_prev_year_L",
    "abs_days_to_prev_year_L",
    "days_to_site_avg_L_recent3y",
    "abs_days_to_site_avg_L_recent3y",
]
HISTORY_FEATURE_NAMES = HISTORY_STATIC_NAMES + HISTORY_DYNAMIC_NAMES
HISTORY_FEATURE_DIM = len(HISTORY_FEATURE_NAMES)


def compute_site_history(all_samples: list[dict], doy_start: int,
                         policy: str = "rolling",
                         train_year_max: int = 2021) -> dict:
    """Build per-(site, year) history dict from base seasonal samples.
    Returns {(site, year): {field: value, ...}}.
    """
    # 1. Collect interval events: (site, year) -> L_doy
    site_year_L = {}
    for s in all_samples:
        site = str(s["site_id"])
        year = int(s["year"])
        ctype = str(s.get("censor_type", "right"))
        if ctype != "right" and s.get("L") is not None and pd.notna(s["L"]):
            site_year_L[(site, year)] = int(s["L"]) + int(doy_start)
    # 2. Per site: sorted list of (year, L_doy)
    site_events = {}
    for (site, year), L_doy in site_year_L.items():
        site_events.setdefault(site, []).append((int(year), int(L_doy)))
    for site in site_events:
        site_events[site].sort()

    history = {}
    for s in all_samples:
        site = str(s["site_id"])
        year = int(s["year"])
        cap = (year - 1) if policy == "rolling" else min(year - 1, int(train_year_max))
        # events strictly before cap+1
        events_before = [(y, L) for y, L in site_events.get(site, []) if y <= cap]
        if events_before:
            last_y, last_L = events_before[-1]
            prev_year_L_doy = float(last_L) if last_y == year - 1 else None
            prev_year_event = 1 if last_y == year - 1 else 0
            recent3 = [L for y, L in events_before if y >= cap - 2]
            site_avg_L_recent3y = float(np.mean(recent3)) if recent3 else None
            years_since_last = int(year - last_y)
            recent5 = [L for y, L in events_before if y >= year - 5]
            n_events_recent5y = int(len(recent5))
        else:
            prev_year_L_doy = None
            prev_year_event = 0
            site_avg_L_recent3y = None
            years_since_last = None
            n_events_recent5y = 0

        miss_prev_L = 1 if prev_year_L_doy is None else 0
        miss_avg3y = 1 if site_avg_L_recent3y is None else 0
        prev_L_imp = float(prev_year_L_doy) if prev_year_L_doy is not None else 0.0
        avg3y_imp = float(site_avg_L_recent3y) if site_avg_L_recent3y is not None else 0.0
        years_since_imp = int(years_since_last) if years_since_last is not None else 99

        history[(site, year)] = {
            "prev_year_L_doy_at_site": prev_L_imp,
            "prev_year_event_at_site": int(prev_year_event),
            "site_avg_L_doy_recent3y": avg3y_imp,
            "years_since_last_event_at_site": int(years_since_imp),
            "n_events_recent5y_at_site": int(n_events_recent5y),
            "prev_year_L_miss": int(miss_prev_L),
            "site_avg_L_recent3y_miss": int(miss_avg3y),
        }
    return history


def build_history_channels(s: dict, h: dict, doy_start: int) -> np.ndarray:
    """Returns (T_season, 11) float32 channel block for one base sample."""
    X = s["X"]
    T_season = int(X.shape[0])
    static = np.array([
        h["prev_year_L_doy_at_site"],
        h["prev_year_event_at_site"],
        h["site_avg_L_doy_recent3y"],
        h["years_since_last_event_at_site"],
        h["n_events_recent5y_at_site"],
        h["prev_year_L_miss"],
        h["site_avg_L_recent3y_miss"],
    ], dtype=np.float32)
    static_chan = np.tile(static[None, :], (T_season, 1))
    doys = np.arange(T_season, dtype=np.float32) + float(doy_start)
    prev_L = float(h["prev_year_L_doy_at_site"])
    avg3y = float(h["site_avg_L_doy_recent3y"])
    miss_prev = float(h["prev_year_L_miss"])
    miss_avg = float(h["site_avg_L_recent3y_miss"])
    days_to_prev = (prev_L - doys) * (1.0 - miss_prev)
    abs_days_prev = np.abs(prev_L - doys) * (1.0 - miss_prev)
    days_to_avg = (avg3y - doys) * (1.0 - miss_avg)
    abs_days_avg = np.abs(avg3y - doys) * (1.0 - miss_avg)
    dynamic_chan = np.stack([days_to_prev, abs_days_prev, days_to_avg, abs_days_avg],
                            axis=1).astype(np.float32)
    return np.concatenate([static_chan, dynamic_chan], axis=1).astype(np.float32)


def append_history_to_samples(samples: list[dict], history: dict, doy_start: int) -> int:
    """In-place: append 11 history channels to each sample's X. Returns count."""
    n_done = 0
    for s in samples:
        key = (str(s["site_id"]), int(s["year"]))
        h = history.get(key)
        X_old = np.asarray(s["X"], dtype=np.float32)
        T_season = int(X_old.shape[0])
        if h is None:
            zero_pad = np.zeros((T_season, HISTORY_FEATURE_DIM), dtype=np.float32)
            zero_pad[:, HISTORY_STATIC_NAMES.index("prev_year_L_miss")] = 1.0
            zero_pad[:, HISTORY_STATIC_NAMES.index("site_avg_L_recent3y_miss")] = 1.0
            zero_pad[:, HISTORY_STATIC_NAMES.index("years_since_last_event_at_site")] = 99.0
            s["X"] = np.concatenate([X_old, zero_pad], axis=1).astype(np.float32)
        else:
            chan = build_history_channels(s, h, doy_start)
            s["X"] = np.concatenate([X_old, chan], axis=1).astype(np.float32)
        n_done += 1
    return n_done
