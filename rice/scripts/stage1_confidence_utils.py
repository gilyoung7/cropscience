"""Stage-2 dispatch confidence-feature utilities.

Mirrors ``site_history_utils.append_history_to_samples``: appends a fixed
block of channels to the per-(site, year) season-length X tensor BEFORE
nowcast slicing. Stage 2 dataset / model code therefore needs no change —
the input dim simply grows by ``DISPATCH_TOTAL_CHANNELS``.

The feature source is the per-(site, year) dispatch alert table produced by
``build_dispatch_feature_table.py`` from the full dispatch alert_map
(typically 987 site-years for sheath_blight R>=0.88), NOT the offset-specific
sample_grid (which is only a few hundred matched/interval rows).

Causal fill (``mode='causal'``):
    nowcast tstar t < alert_t_rel  -> zeros(14) + missing=1
    nowcast tstar t >= alert_t_rel -> features(14) + missing=0
Site-years that never alerted are zeros(14) + missing=1 for all t.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# Order matters; must match build_dispatch_feature_table.py column ordering.
DISPATCH_FEATURE_NAMES: list[str] = [
    "alert_tstar",
    "with_history",
    "dispatch_branch",
    "A_score_at_alert",
    "D_score_at_alert",
    "score_margin",
    "dispatch_score_at_alert",
    "dispatch_tau_used",
    "score_over_tau_margin",
    "recent_14d_mean_score",
    "recent_28d_mean_score",
    "score_above_tau_streak",
    "score_rolling_slope_14d",
    "p_mean_so_far_at_alert",
]
DISPATCH_FEATURE_DIM: int = len(DISPATCH_FEATURE_NAMES)            # 14
DISPATCH_MISSING_NAME: str = "dispatch_feature_missing"
DISPATCH_TOTAL_CHANNELS: int = DISPATCH_FEATURE_DIM + 1            # 15
DISPATCH_CHANNEL_NAMES: list[str] = DISPATCH_FEATURE_NAMES + [DISPATCH_MISSING_NAME]


def _branch_to_float(v) -> float:
    """Encode 'D' -> 1.0, 'A' (or anything else) -> 0.0. NaN-safe."""
    if isinstance(v, str):
        return 1.0 if v.strip().upper() == "D" else 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if not np.isfinite(f) else f


def load_dispatch_feature_table(
    csv_path: str,
    only_splits: tuple[str, ...] | None = None,
) -> dict:
    """Load per-(site, year) dispatch confidence features.

    Returns ``{(site_str, year_int): {'alert_tstar_doy': int, 'features': np.ndarray(14,)}}``.
    Missing/non-finite numeric cells become 0.0 in features (the missing-indicator
    is set by the appender at row level, not at feature level).

    Parameters
    ----------
    only_splits : optional tuple of {'train','val','test'}
        If provided AND the CSV has a 'split' column, only rows whose split
        is in this set are loaded. Default None loads all rows.
    """
    df = pd.read_csv(csv_path)
    if only_splits is not None and "split" in df.columns:
        df = df[df["split"].isin(list(only_splits))].copy()
    out: dict = {}
    for _, r in df.iterrows():
        sy = (str(r["site"]), int(r["year"]))
        feats = []
        for name in DISPATCH_FEATURE_NAMES:
            v = r.get(name)
            if name == "dispatch_branch":
                feats.append(_branch_to_float(v))
                continue
            try:
                fv = float(v)
            except (TypeError, ValueError):
                fv = 0.0
            feats.append(0.0 if not np.isfinite(fv) else fv)
        out[sy] = {
            "alert_tstar_doy": int(r["alert_tstar"]),
            "features": np.asarray(feats, dtype=np.float32),
        }
    return out


def train_mean_features_from_table(csv_path: str) -> np.ndarray:
    """Mean of the 14 numeric dispatch features over the train split rows of a
    feature CSV. Used by ablate_train_mean mode (Phase A.3).

    Falls back to mean over all rows if no 'split' column is present.
    Returns a (14,) float32 vector.
    """
    df = pd.read_csv(csv_path)
    if "split" in df.columns:
        df = df[df["split"] == "train"]
    if df.empty:
        return np.zeros(DISPATCH_FEATURE_DIM, dtype=np.float32)
    out = []
    for name in DISPATCH_FEATURE_NAMES:
        if name == "dispatch_branch":
            out.append(float(df[name].apply(_branch_to_float).mean()))
            continue
        v = pd.to_numeric(df[name], errors="coerce")
        m = float(v.mean()) if v.notna().any() else 0.0
        out.append(0.0 if not np.isfinite(m) else m)
    return np.asarray(out, dtype=np.float32)


def append_dispatch_confidence_to_samples(
    samples: list[dict],
    conf_map: dict,
    doy_start: int,
    mode: str = "causal",
    missing_value: float = 0.0,
    ablate_train_mean_features: np.ndarray | None = None,
) -> dict:
    """In-place: append 15 channels (14 features + 1 missing indicator) to each
    sample's season-length X.

    Parameters
    ----------
    samples : list of dicts each having 'site_id', 'year', and 'X' of shape
        (T_season, d_feat). X is replaced by (T_season, d_feat + 15).
    conf_map : output of ``load_dispatch_feature_table``.
    doy_start : doy index that maps t=0 -> DOY (relative coords).
        alert_t_rel = alert_tstar_doy - doy_start + 1.
    mode : one of
        - 'causal'             (default; production / training fill)
        - 'broadcast'          (leakage baseline; broadcast features to all rows)
        - 'ablate_missing'     (eval-only; force every row to missing-fill,
                                 i.e. (missing_value, ..., missing_value, 1.0).
                                 This is the *exact* fill that pre-alert rows
                                 already see under causal mode, so it stays
                                 inside the model's seen input distribution
                                 unlike a hard-zero override.)
        - 'ablate_train_mean'  (eval-only; replace the 14 feature slots in EVERY
                                 row with the train-split feature mean, keep
                                 missing indicator = 1. Requires
                                 ``ablate_train_mean_features``.)
    missing_value : fill value for the 14 feature slots when no feature applies
        (default 0.0; matches causal-mode zero padding).
    ablate_train_mean_features : (14,) np.ndarray, required for ablate_train_mean.

    Returns
    -------
    stats : dict with counts for logging.
    """
    valid_modes = ("causal", "broadcast", "ablate_missing", "ablate_train_mean")
    if mode not in valid_modes:
        raise ValueError(f"unknown mode {mode!r}; expected one of {valid_modes}")
    if mode == "ablate_train_mean" and ablate_train_mean_features is None:
        raise ValueError("mode='ablate_train_mean' requires ablate_train_mean_features")
    if not samples:
        return {
            "n_with_alert": 0, "n_no_alert": 0,
            "n_rows_with_feature": 0, "n_rows_missing": 0,
            "added_channels": DISPATCH_TOTAL_CHANNELS,
            "mode": mode,
        }
    n_with = 0
    n_no_alert = 0
    n_rows_with_feature = 0
    n_rows_missing = 0
    for s in samples:
        sy = (str(s["site_id"]), int(s["year"]))
        X_old = np.asarray(s["X"], dtype=np.float32)
        T_season = int(X_old.shape[0])
        new_chan = np.full(
            (T_season, DISPATCH_TOTAL_CHANNELS),
            float(missing_value),
            dtype=np.float32,
        )
        new_chan[:, DISPATCH_FEATURE_DIM] = 1.0  # default missing = 1

        if mode == "ablate_missing":
            # All rows already match the pre-alert/never-alert pattern from
            # causal training: features=missing_value, indicator=1. No further
            # branching; counted as missing for stats.
            n_no_alert += 0 if conf_map.get(sy) is None else 0
            n_rows_missing += T_season
            s["X"] = np.concatenate([X_old, new_chan], axis=1).astype(np.float32)
            continue

        if mode == "ablate_train_mean":
            new_chan[:, :DISPATCH_FEATURE_DIM] = ablate_train_mean_features
            new_chan[:, DISPATCH_FEATURE_DIM] = 1.0
            n_rows_missing += T_season  # indicator stays = 1 (still "missing")
            s["X"] = np.concatenate([X_old, new_chan], axis=1).astype(np.float32)
            continue

        info = conf_map.get(sy)
        if info is not None:
            n_with += 1
            alert_t_doy = int(info["alert_tstar_doy"])
            # relative tstar index: t=0 corresponds to doy_start
            alert_t_rel = alert_t_doy - int(doy_start) + 1
            if alert_t_rel >= T_season:
                # alert beyond season — treat as never alerted in-season
                n_rows_missing += T_season
            elif mode == "broadcast":
                new_chan[:, :DISPATCH_FEATURE_DIM] = info["features"]
                new_chan[:, DISPATCH_FEATURE_DIM] = 0.0
                n_rows_with_feature += T_season
            else:  # causal
                idx = max(0, alert_t_rel)
                new_chan[idx:, :DISPATCH_FEATURE_DIM] = info["features"]
                new_chan[idx:, DISPATCH_FEATURE_DIM] = 0.0
                n_rows_with_feature += T_season - idx
                n_rows_missing += idx
        else:
            n_no_alert += 1
            n_rows_missing += T_season

        s["X"] = np.concatenate([X_old, new_chan], axis=1).astype(np.float32)

    return {
        "n_with_alert": n_with,
        "n_no_alert": n_no_alert,
        "n_rows_with_feature": n_rows_with_feature,
        "n_rows_missing": n_rows_missing,
        "added_channels": DISPATCH_TOTAL_CHANNELS,
        "mode": mode,
    }
