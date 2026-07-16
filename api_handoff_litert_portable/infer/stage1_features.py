"""Stage-1 base features — torch-free, layout-preserving.

Ports `api_handoff_transformer/infer/stage1.py`'s single-(site, year) feature path
with numpy + pandas only. Every function names the original it mirrors.

MEMORY LAYOUT — READ BEFORE "OPTIMIZING"
----------------------------------------
The deployed API builds base X as `X_df.to_numpy(dtype=np.float32)`
(stage1.py:720). Measured on pandas 3.0.3 / numpy 2.5.1, that returns an
**F-contiguous** array (strides (4, 524)), not C-contiguous. `_append_history`
then uses `np.concatenate`, which returns **C-contiguous** — so the layout
legitimately differs per branch:

    A branch (no history): F-order, strides (4, 524)
    D branch (+history)  : C-order, strides (92, 4)

This is not cosmetic. `_build_tabular` reduces with mean/std/slope over these
arrays, and float32 reduction order depends on layout:

    mean: max|C - F| = 2.98e-08     std: max|C - F| = 5.96e-08
    min/max: identical (order-independent)

~1e-8 on a feature can flip a probability across tau and move the alert DOY.
The deployed API's layout is the source of truth, so this module reproduces it
exactly and MUST NOT call np.ascontiguousarray / np.asfortranarray to "clean up".
tests/test_stage1_portable.py pins the flags and asserts that forcing C-order
changes the features (the regression this guards against).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# stage1.py::_build_tabular statistic order.
TABULAR_STATS = ("mean", "std", "min", "max", "first", "last", "slope")

# stage1.py::HISTORY_STATIC_NAMES / HISTORY_DYNAMIC_NAMES — order matters.
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
HISTORY_FEATURE_DIM = len(HISTORY_STATIC_NAMES) + len(HISTORY_DYNAMIC_NAMES)  # 11

COORD_COLS = ("좌표-위도", "좌표-경도")
PHENO_CANDIDATE_COLS = (
    "days_since_growing_start", "days_until_growing_end", "is_growing",
    "growing_start_doy", "growing_end_doy", "growing_mid_doy", "growing_len_days",
)


class Stage1FeatureError(ValueError):
    """Raised when Stage-1 inputs are missing/inconsistent. Never silent."""


def array_layout(a: np.ndarray) -> dict:
    """Record dtype/shape/strides/flags at a boundary (for tests + diagnostics)."""
    return {
        "dtype": str(a.dtype),
        "shape": list(a.shape),
        "strides": list(a.strides),
        "c_contiguous": bool(a.flags["C_CONTIGUOUS"]),
        "f_contiguous": bool(a.flags["F_CONTIGUOUS"]),
    }


def base_x_from_season(season: pd.DataFrame, feature_cols: list[str]) -> np.ndarray:
    """Port of stage1.py:712-720. Returns X in the SAME layout as the deployed API.

    The `[base..., base__miss...]` block order and the `to_numpy` call are both
    load-bearing — see the module docstring.
    """
    missing = [c for c in feature_cols if c not in season.columns]
    if missing:
        raise Stage1FeatureError(
            f"base feature columns missing after preprocessing: {missing}"
        )
    X_df = season[feature_cols].copy()
    for c in feature_cols:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
        miss = X_df[c].isna().astype(np.float32)
        X_df[c] = X_df[c].fillna(0.0)
        X_df[f"{c}__miss"] = miss
    # DO NOT wrap in ascontiguousarray — F-order here is the deployed behaviour.
    return X_df.to_numpy(dtype=np.float32)


def history_channels(X_T: int, h: dict, doy_start: int) -> np.ndarray:
    """Port of stage1.py::_history_channels."""
    static = np.array([
        h["prev_year_L_doy_at_site"], h["prev_year_event_at_site"],
        h["site_avg_L_doy_recent3y"], h["years_since_last_event_at_site"],
        h["n_events_recent5y_at_site"], h["prev_year_L_miss"],
        h["site_avg_L_recent3y_miss"],
    ], dtype=np.float32)
    static_chan = np.tile(static[None, :], (X_T, 1))
    doys = np.arange(X_T, dtype=np.float32) + float(doy_start)
    prev_L = float(h["prev_year_L_doy_at_site"])
    avg3y = float(h["site_avg_L_doy_recent3y"])
    miss_prev = float(h["prev_year_L_miss"])
    miss_avg = float(h["site_avg_L_recent3y_miss"])
    dyn = np.stack([
        (prev_L - doys) * (1.0 - miss_prev),
        np.abs(prev_L - doys) * (1.0 - miss_prev),
        (avg3y - doys) * (1.0 - miss_avg),
        np.abs(avg3y - doys) * (1.0 - miss_avg),
    ], axis=1).astype(np.float32)
    return np.concatenate([static_chan, dyn], axis=1).astype(np.float32)


def append_history(base_X: np.ndarray, site: str, year: int, history: dict,
                   doy_start: int) -> np.ndarray:
    """Port of stage1.py::_append_history.

    np.concatenate returns C-order — that C-order IS the deployed D-branch layout.
    """
    T = int(base_X.shape[0])
    h = (history or {}).get((str(site), int(year)))
    if h is None:
        pad = np.zeros((T, HISTORY_FEATURE_DIM), dtype=np.float32)
        pad[:, HISTORY_STATIC_NAMES.index("prev_year_L_miss")] = 1.0
        pad[:, HISTORY_STATIC_NAMES.index("site_avg_L_recent3y_miss")] = 1.0
        pad[:, HISTORY_STATIC_NAMES.index("years_since_last_event_at_site")] = 99.0
        return np.concatenate([base_X, pad], axis=1).astype(np.float32)
    return np.concatenate(
        [base_X, history_channels(T, h, doy_start)], axis=1
    ).astype(np.float32)


def build_nowcast_samples(samples: list[dict], window: int, stride: int,
                          only_pre_event: bool, event_time_proxy: str) -> list[dict]:
    """Port of stage1.py::_build_nowcast_samples.

    `x[(tstar-window):tstar, :]` with copy=False keeps a VIEW of the parent's
    layout — for the A branch that view is strided (C=0, F=0), which is exactly
    what the deployed API feeds the model.
    """
    out: list[dict] = []
    if not samples:
        return out
    T = int(samples[0]["X"].shape[0])
    t0 = min(int(window), T)
    for s in samples:
        x = np.asarray(s["X"], dtype=np.float32)
        ctype = str(s["censor_type"])
        has_event = ctype != "right"
        if has_event:
            L_time, R_time = int(s["L"]), int(s["R"])
            event_time = int((L_time + R_time) // 2) if event_time_proxy == "mid" else int(R_time)
        else:
            event_time = None
        for tstar in range(t0, T + 1, stride):
            if only_pre_event and has_event and event_time is not None and tstar >= event_time:
                continue
            y_event = 1 if (has_event and event_time is not None and event_time > tstar) else 0
            out.append({
                "site_id": s["site_id"], "year": int(s["year"]),
                "X": x[(tstar - window):tstar, :].astype(np.float32, copy=False),
                "y_event": int(y_event), "tstar": int(tstar), "season_length": int(T),
            })
    return out


def build_tabular(samples: list[dict], add_tstar_position_feature: bool) -> np.ndarray:
    """Port of stage1.py::_build_tabular — the layout-sensitive reduction."""
    feats = []
    for s in samples:
        x = np.asarray(s["X"], dtype=np.float32)
        t = np.arange(x.shape[0], dtype=np.float32)
        t_center = t - t.mean()
        t_var = float((t_center ** 2).sum()) + 1e-8
        mean = x.mean(axis=0); std = x.std(axis=0)
        xmin = x.min(axis=0); xmax = x.max(axis=0)
        xfirst = x[0]; xlast = x[-1]
        slope = ((x - mean) * t_center[:, None]).sum(axis=0) / t_var
        f = np.concatenate([mean, std, xmin, xmax, xfirst, xlast, slope], axis=0)
        if add_tstar_position_feature:
            season_length = max(int(s.get("season_length", x.shape[0])), 1)
            tstar = int(s.get("tstar", season_length))
            f = np.concatenate(
                [f, np.asarray([float(tstar) / float(season_length)], dtype=np.float32)]
            )
        feats.append(f)
    return (np.stack(feats, axis=0).astype(np.float32) if feats
            else np.zeros((0, 0), dtype=np.float32))


def apply_temperature(p: np.ndarray, temperature: float, eps: float = 1e-8) -> np.ndarray:
    """Port of stage1.py::_apply_temperature (fixed scalar from temperature.json)."""
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-(logit / float(temperature))))


def first_crossing_k(ts: np.ndarray, ps: np.ndarray, tau: float, k: int) -> int | None:
    """Port of stage1.py::_first_crossing_k — k consecutive days >= tau."""
    streak = 0
    for i in range(len(ts)):
        if ps[i] >= tau:
            streak += 1
            if streak >= k:
                return int(ts[i])
        else:
            streak = 0
    return None
