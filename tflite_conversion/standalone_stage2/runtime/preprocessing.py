"""Standalone Stage-2 input builder — pure Python + NumPy + Pandas.

Reproduces `api_handoff_transformer/infer/preprocess.py::build_real_input`
without importing torch or anything from the API package. Each step below cites
the upstream function it mirrors; the parity test
(tests/test_preprocessing_parity.py) asserts bit-exactness against the original
on real data.

Differences from the original, and why they are not behaviour changes:
  * Site slicing / master-CSV grep is gone: this runtime takes the caller's daily
    frame for one site-year directly (upstream `_daily_year_from_frame` path).
  * The LONG observation CSV is gone: coordinates and phenology arrive as
    explicit request fields. Stage 1 is out of scope for this package, so — like
    alert_tstar and the dispatch features — these are inputs. They cannot be
    derived from weather: phenology is step data observed on a handful of DOYs
    (e.g. 8 rows for a 241-day season) and then forward-filled, so it is not a
    function of DOY.
  * torch tensors are replaced by float32 ndarrays.

Rolling-window caveat (important): upstream computes the 7/14-day rollings over
the whole year's rows and only then slices the season. A frame that starts at
doy_start would give different rollings at the season's first days. This module
therefore requires the daily frame to cover DOY 1..doy_end contiguously, which
is what the shipped master data provides, and errors otherwise rather than
silently producing different features.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import (
    COORD_COLS,
    DISPATCH_FEATURE_NAMES,
    DISPATCH_MISSING_NAME,
    PHENO_COLS,
    REQUIRED_DAILY_COLS,
    DispatchRequest,
    PestMetadata,
)

DISPATCH_FEATURE_DIM = len(DISPATCH_FEATURE_NAMES)          # 14
DISPATCH_TOTAL_CHANNELS = DISPATCH_FEATURE_DIM + 1          # 15

# Weather columns upstream imputes (interpolate -> ffill -> bfill).
# GDD10_since_gs is intentionally NOT imputed (matches upstream preprocess.py:67-79).
WEATHER_COLS = [
    "일강수량(mm)",
    "최고기온(°C)",
    "최저기온(°C)",
    "평균기온(°C)",
    "평균 풍속(m/s)",
    "최대 풍속(m/s)",
    "평균 상대습도(%)",
    "합계 일조시간(h)",
    "합계 일사량(MJ/m2)",
]


class PreprocessError(ValueError):
    """Raised when input is missing/inconsistent. Never silent, never filled."""


@dataclass
class BuiltInput:
    X: np.ndarray            # (1, 1, T, d_in) float32
    tstar: np.ndarray        # (1, 1) int64, 1-based season index
    valid_mask: np.ndarray   # (1, 1) bool
    alert_tstar_doy: int
    tstar_season_index: int
    year: int


# ---------------------------------------------------------------------------
# 1. daily load + impute  (mirrors preprocess._process_daily_year)
# ---------------------------------------------------------------------------
def load_daily_csv(path: Path) -> pd.DataFrame:
    p = Path(path)
    if not p.is_file():
        raise PreprocessError(f"daily CSV not found: {p}")
    # utf-8-sig strips the BOM so '지점ID'/'일시' match, as upstream does.
    return pd.read_csv(p, encoding="utf-8-sig")


def _prepare_daily(daily: pd.DataFrame, year: int | None) -> tuple[pd.DataFrame, int]:
    df = daily.rename(columns=lambda c: c.strip() if isinstance(c, str) else c).copy()
    if "지점ID" in df.columns:
        df = df.rename(columns={"지점ID": "site_id"})

    missing = [c for c in REQUIRED_DAILY_COLS if c not in df.columns]
    if missing:
        raise PreprocessError(
            f"daily data missing required column(s): {missing}\n"
            f"  required: {list(REQUIRED_DAILY_COLS)}\n"
            f"  got:      {list(df.columns)}"
        )

    df["date"] = pd.to_datetime(df["일시"], errors="coerce")
    n_bad = int(df["date"].isna().sum())
    if n_bad:
        raise PreprocessError(
            f"daily data has {n_bad} row(s) with an unparseable '일시' date"
        )
    df["year"] = df["date"].dt.year.astype(int)
    df["doy"] = df["date"].dt.dayofyear.astype(int)

    years = sorted(df["year"].unique().tolist())
    if year is None:
        if len(years) != 1:
            raise PreprocessError(
                f"daily data spans {len(years)} years {years}; pass 'year' in the "
                f"dispatch request to select one."
            )
        year = int(years[0])
    if year not in years:
        raise PreprocessError(f"daily data has no rows for year={year} (has {years})")
    sub = df[df["year"] == int(year)].copy()

    if "site_id" in sub.columns:
        sites = sub["site_id"].astype(str).unique().tolist()
        if len(sites) > 1:
            raise PreprocessError(
                f"daily data covers {len(sites)} sites {sites[:5]}; supply one site."
            )

    dup = sub["doy"].duplicated()
    if dup.any():
        d = sorted(sub.loc[dup, "doy"].unique().tolist())
        raise PreprocessError(f"daily data has duplicate DOY row(s) for year={year}: {d[:10]}")

    sub = sub.sort_values("doy").copy()
    for c in WEATHER_COLS:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    # Per (site, year) impute — upstream preprocess.py:191-193.
    for c in WEATHER_COLS:
        sub[c] = sub[c].interpolate(limit_direction="both").ffill().bfill()
    sub["일강수량(mm)"] = sub["일강수량(mm)"].fillna(0.0)
    sub["GDD10_since_gs"] = pd.to_numeric(sub["GDD10_since_gs"], errors="coerce")
    if "site_id" not in sub.columns:
        sub["site_id"] = "site"
    return sub, int(year)


def _check_doy_coverage(daily: pd.DataFrame, doy_end: int, year: int) -> None:
    """Require contiguous DOY 1..doy_end.

    Upstream computes rollings over the full year before slicing the season, so
    a short frame silently changes the first season days' features. Refuse that.
    """
    present = set(daily["doy"].astype(int).tolist())
    need = set(range(1, int(doy_end) + 1))
    gaps = sorted(need - present)
    if gaps:
        raise PreprocessError(
            f"daily data for year={year} is missing {len(gaps)} DOY in 1..{doy_end}: "
            f"{gaps[:10]}{'...' if len(gaps) > 10 else ''}\n"
            f"  The 7/14-day rolling features are computed over the whole year before "
            f"the season is sliced, so a partial year changes the result. Supply the "
            f"full year up to DOY {doy_end}."
        )


# ---------------------------------------------------------------------------
# 2. rolling/derived features  (mirrors preprocess._add_rolling_features)
# ---------------------------------------------------------------------------
def _add_rolling_features(daily: pd.DataFrame) -> pd.DataFrame:
    d = daily.sort_values(["site_id", "year", "doy"]).copy()
    g = d.groupby(["site_id", "year"], sort=False)

    d["trange"] = d["최고기온(°C)"] - d["최저기온(°C)"]

    d["rain_7d_sum"] = g["일강수량(mm)"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    d["rain_14d_sum"] = g["일강수량(mm)"].transform(lambda s: s.rolling(14, min_periods=1).sum())
    d["rain_7d_days"] = g["일강수량(mm)"].transform(
        lambda s: (s > 0).rolling(7, min_periods=1).sum()
    )

    d["tmean_7d_mean"] = g["평균기온(°C)"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    d["tmax_7d_max"] = g["최고기온(°C)"].transform(lambda s: s.rolling(7, min_periods=1).max())
    d["tmin_7d_min"] = g["최저기온(°C)"].transform(lambda s: s.rolling(7, min_periods=1).min())

    if "DD10" in d.columns:
        d["DD10_7d_sum"] = g["DD10"].transform(lambda s: s.rolling(7, min_periods=1).sum())

    d["rh_7d_mean"] = g["평균 상대습도(%)"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    d["rh_14d_mean"] = g["평균 상대습도(%)"].transform(lambda s: s.rolling(14, min_periods=1).mean())
    d["wind_7d_mean"] = g["평균 풍속(m/s)"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    d["wind_7d_max"] = g["최대 풍속(m/s)"].transform(lambda s: s.rolling(7, min_periods=1).max())
    d["sun_7d_sum"] = g["합계 일조시간(h)"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    d["rad_7d_sum"] = g["합계 일사량(MJ/m2)"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    d["trange_7d_mean"] = g["trange"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    return d


# ---------------------------------------------------------------------------
# 3. phenology merge  (mirrors preprocess._merge_pheno_daily_ffill)
# ---------------------------------------------------------------------------
def _merge_phenology(season: pd.DataFrame, pheno_rows: list[dict],
                     pheno_cols: list[str]) -> pd.DataFrame:
    if not pheno_cols:
        return season
    ph = pd.DataFrame(pheno_rows)
    if "obs_doy" not in ph.columns:
        raise PreprocessError("phenology records need an 'obs_doy' field")
    ph["doy"] = pd.to_numeric(ph["obs_doy"], errors="coerce").astype("Int64")
    ph = ph.dropna(subset=["doy"]).copy()
    ph["doy"] = ph["doy"].astype(int)
    ph = ph.drop(columns=["obs_doy"])
    missing = [c for c in pheno_cols if c not in ph.columns]
    if missing:
        raise PreprocessError(f"phenology records missing column(s): {missing}")
    for c in pheno_cols:
        ph[c] = pd.to_numeric(ph[c], errors="coerce")

    # last-wins per doy, except is_growing which takes max (upstream:359-364)
    agg = {c: "last" for c in pheno_cols}
    if "is_growing" in agg:
        agg["is_growing"] = "max"
    ph = ph[["doy"] + pheno_cols].sort_values(["doy"]).groupby(["doy"], as_index=False).agg(agg)

    out = season.merge(ph, on="doy", how="left").sort_values("doy").copy()
    out[pheno_cols] = out[pheno_cols].ffill()
    for c in pheno_cols:
        # upstream fills offset_days with -1.0; it is not a Stage-2 base channel
        # for any of the 8 pests, so only the 0.0 branch can be reached here.
        out[c] = out[c].fillna(0.0)
    return out


# ---------------------------------------------------------------------------
# 4. dispatch vector + causal append  (mirrors preprocess._dispatch_feature_vector
#    and _append_dispatch_causal)
# ---------------------------------------------------------------------------
def _branch_to_float(v) -> float:
    if isinstance(v, str):
        return 1.0 if v.strip().upper() == "D" else 0.0
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return 0.0 if not np.isfinite(f) else f


def _dispatch_feature_vector(features: dict, alert_doy_override: int) -> np.ndarray:
    out: list[float] = []
    for name in DISPATCH_FEATURE_NAMES:
        if name == "alert_tstar":
            out.append(float(alert_doy_override))
            continue
        v = features[name]  # presence already enforced by DispatchRequest.validate
        if name == "dispatch_branch":
            out.append(_branch_to_float(v))
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            raise PreprocessError(
                f"dispatch feature {name!r}={v!r} is not numeric"
            ) from None
        if not np.isfinite(fv):
            raise PreprocessError(
                f"dispatch feature {name!r}={v!r} is not finite. Supply a real value; "
                f"this runtime does not substitute 0."
            )
        out.append(fv)
    return np.asarray(out, dtype=np.float32)


def _append_dispatch_causal(X_base: np.ndarray, features: np.ndarray,
                            alert_doy: int, doy_start: int,
                            missing_value: float = 0.0) -> np.ndarray:
    T_season = int(X_base.shape[0])
    new_chan = np.full((T_season, DISPATCH_TOTAL_CHANNELS), float(missing_value), dtype=np.float32)
    new_chan[:, DISPATCH_FEATURE_DIM] = 1.0
    alert_t_rel = int(alert_doy) - int(doy_start) + 1
    if alert_t_rel < T_season:
        idx = max(0, alert_t_rel)
        new_chan[idx:, :DISPATCH_FEATURE_DIM] = features
        new_chan[idx:, DISPATCH_FEATURE_DIM] = 0.0
    return np.concatenate([X_base, new_chan], axis=1).astype(np.float32)


# ---------------------------------------------------------------------------
# 5. nowcast masking  (mirrors preprocess._mask_to_recent_window)
# ---------------------------------------------------------------------------
def _mask_to_recent_window(X: np.ndarray, tstar: int, window: int) -> np.ndarray:
    T, D = X.shape
    X_out = np.zeros_like(X, dtype=np.float32)
    if D > 1:
        X_out[:, 1::2] = 1.0
    start = max(1, int(tstar) - int(window) + 1)
    end = min(T, int(tstar))
    if end >= start:
        X_out[start - 1:end, :] = X[start - 1:end, :]
    return X_out


# ---------------------------------------------------------------------------
# top-level
# ---------------------------------------------------------------------------
def build_input(
    md: PestMetadata,
    daily: pd.DataFrame,
    req: DispatchRequest,
    norm_mean: np.ndarray,
    norm_std: np.ndarray,
) -> BuiltInput:
    """Build the (1, 1, T, d_in) normalized Stage-2 input. Mirrors build_real_input."""
    req.validate(md)

    base_cols = md.base_channels
    nbase = len(base_cols)
    D = md.d_in
    if norm_mean.shape != (D,) or norm_std.shape != (D,):
        raise PreprocessError(
            f"norm arrays {norm_mean.shape}/{norm_std.shape} != (d_in={D},)"
        )

    daily_y, year = _prepare_daily(daily, req.year)
    _check_doy_coverage(daily_y, md.doy_end, year)
    daily_y = _add_rolling_features(daily_y)

    season = daily_y[
        (daily_y["doy"] >= md.doy_start) & (daily_y["doy"] <= md.doy_end)
    ].copy()
    season = season.sort_values("doy").reset_index(drop=True)
    if len(season) != md.T:
        raise PreprocessError(
            f"season slice has {len(season)} rows, expected T={md.T} "
            f"(DOY {md.doy_start}..{md.doy_end}) for year={year}"
        )

    if md.requires_site_coords:
        season[COORD_COLS[0]] = float(req.site["lat"])
        season[COORD_COLS[1]] = float(req.site["lon"])
    if md.requires_phenology:
        pheno_cols = [c for c in PHENO_COLS if c in base_cols]
        season = _merge_phenology(season, req.phenology, pheno_cols)

    missing_base = [c for c in base_cols if c not in season.columns]
    if missing_base:
        raise PreprocessError(
            f"[{md.pest}] base channel(s) not produced by preprocessing: {missing_base}"
        )

    # base block: [base..., base__miss...]  (upstream:591-604)
    X_df = season[base_cols].copy()
    for c in base_cols:
        X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
        miss = X_df[c].isna().astype(np.float32)
        X_df[c] = X_df[c].fillna(0.0)
        X_df[f"{c}__miss"] = miss
    built = list(X_df.columns)
    expected = md.feature_names[: 2 * nbase]
    if built != expected:
        raise PreprocessError(
            f"[{md.pest}] base block order mismatch\n  got:      {built}\n"
            f"  expected: {expected}"
        )
    X_base = X_df.to_numpy(dtype=np.float32)

    feat_vec = _dispatch_feature_vector(req.dispatch_features, req.alert_tstar)
    X_full = _append_dispatch_causal(
        X_base, feat_vec, alert_doy=req.alert_tstar, doy_start=md.doy_start,
        missing_value=0.0,
    )
    if X_full.shape != (md.T, D):
        raise PreprocessError(f"[{md.pest}] assembled X {X_full.shape} != (T={md.T}, D={D})")

    tstar_idx = req.alert_tstar - md.doy_start + 1 + md.selected_offset
    tstar_idx = max(1, min(tstar_idx, md.T))
    X_masked = _mask_to_recent_window(X_full, tstar=tstar_idx, window=md.nowcast_window)

    std = np.where(norm_std < 1e-6, 1.0, norm_std).astype(np.float32)
    X_norm = (X_masked - norm_mean.astype(np.float32)) / std

    if not np.isfinite(X_norm).all():
        bad = [md.feature_names[i] for i in range(D) if not np.isfinite(X_norm[:, i]).all()]
        raise PreprocessError(f"[{md.pest}] non-finite values in normalized X at: {bad}")

    return BuiltInput(
        X=X_norm.reshape(1, 1, md.T, D).astype(np.float32),
        tstar=np.array([[tstar_idx]], dtype=np.int64),
        valid_mask=np.ones((1, 1), dtype=bool),
        alert_tstar_doy=int(req.alert_tstar),
        tstar_season_index=int(tstar_idx),
        year=int(year),
    )
