"""Season preprocessing shared by Stage 1 and Stage 2 — torch-free.

Ports the deployed daily -> rolling -> coords -> phenology chain
(api_handoff_transformer/infer/stage1.py + infer/preprocess.py) with numpy +
pandas only. The Stage-2 tensor builder itself is vendored verbatim in
_stage2_preprocessing.py (already proven bit-exact against build_real_input).

pandas is retained deliberately: the deployed API's base X comes from
`X_df.to_numpy(dtype=np.float32)`, whose F-contiguous layout is load-bearing for
float32 reduction order in Stage-1's tabular features. See stage1_features.py.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .stage1_features import COORD_COLS, PHENO_CANDIDATE_COLS, Stage1FeatureError

WEATHER_COLS = [
    "일강수량(mm)", "최고기온(°C)", "최저기온(°C)", "평균기온(°C)",
    "평균 풍속(m/s)", "최대 풍속(m/s)", "평균 상대습도(%)",
    "합계 일조시간(h)", "합계 일사량(MJ/m2)",
]
REQUIRED_DAILY_COLS = ["일시"] + WEATHER_COLS + ["GDD10_since_gs"]


class PreprocessError(ValueError):
    """Raised on missing/inconsistent daily input. Never silent."""


def daily_year_frame(daily: pd.DataFrame, site_id: str, year: int) -> pd.DataFrame:
    """Port of stage1.py::_process_daily_year — parse date, filter, impute."""
    df = daily.rename(columns=lambda c: c.strip() if isinstance(c, str) else c).copy()
    if "지점ID" in df.columns:
        df = df.rename(columns={"지점ID": "site_id"})
    missing = [c for c in REQUIRED_DAILY_COLS if c not in df.columns]
    if missing:
        raise PreprocessError(f"daily data missing required column(s): {missing}")
    if "site_id" in df.columns:
        df["site_id"] = df["site_id"].astype(str)
        df = df[df["site_id"] == str(site_id)].copy()
    else:
        df["site_id"] = str(site_id)
    df["date"] = pd.to_datetime(df["일시"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["year"] = df["date"].dt.year.astype(int)
    df["doy"] = df["date"].dt.dayofyear.astype(int)
    sub = df[df["year"] == int(year)].copy()
    if sub.empty:
        raise PreprocessError(
            f"no daily rows for site_id={site_id!r} year={year} in the supplied data"
        )
    if sub["doy"].duplicated().any():
        d = sorted(sub.loc[sub["doy"].duplicated(), "doy"].unique().tolist())
        raise PreprocessError(f"duplicate DOY rows for {site_id}/{year}: {d[:10]}")
    sub = sub.sort_values("doy").copy()
    for c in WEATHER_COLS:
        sub[c] = pd.to_numeric(sub[c], errors="coerce")
    for c in WEATHER_COLS:
        sub[c] = sub[c].interpolate(limit_direction="both").ffill().bfill()
    sub["일강수량(mm)"] = sub["일강수량(mm)"].fillna(0.0)
    # GDD10_since_gs is intentionally NOT imputed (matches upstream).
    sub["GDD10_since_gs"] = pd.to_numeric(sub["GDD10_since_gs"], errors="coerce")
    return sub


def add_rolling_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Port of stage1.py::_add_rolling_features / preprocess.py::_add_rolling_features."""
    d = daily.sort_values(["site_id", "year", "doy"]).copy()
    g = d.groupby(["site_id", "year"], sort=False)
    d["trange"] = d["최고기온(°C)"] - d["최저기온(°C)"]
    d["rain_7d_sum"] = g["일강수량(mm)"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    d["rain_14d_sum"] = g["일강수량(mm)"].transform(lambda s: s.rolling(14, min_periods=1).sum())
    d["rain_7d_days"] = g["일강수량(mm)"].transform(
        lambda s: (s > 0).rolling(7, min_periods=1).sum())
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


def merge_pheno_ffill(season: pd.DataFrame, pheno_rows: pd.DataFrame,
                      pheno_cols: list[str]) -> pd.DataFrame:
    """Port of stage1.py::_merge_pheno_daily_ffill — merge by DOY, ffill, fill 0."""
    if not pheno_cols:
        return season
    ph = pheno_rows.copy()
    missing = [c for c in pheno_cols if c not in ph.columns]
    if missing:
        raise PreprocessError(f"phenology rows missing column(s): {missing}")
    ph["doy"] = pd.to_numeric(ph["obs_doy"], errors="coerce").astype("Int64")
    ph = ph.dropna(subset=["doy"]).copy()
    ph["doy"] = ph["doy"].astype(int)
    for c in pheno_cols:
        ph[c] = pd.to_numeric(ph[c], errors="coerce")
    agg = {c: "last" for c in pheno_cols}
    if "is_growing" in agg:
        agg["is_growing"] = "max"
    ph = ph[["doy"] + pheno_cols].sort_values(["doy"]).groupby(["doy"], as_index=False).agg(agg)
    out = season.merge(ph, on="doy", how="left").sort_values("doy").copy()
    out[pheno_cols] = out[pheno_cols].ffill()
    for c in pheno_cols:
        if c == "offset_days":
            out[c] = out[c].fillna(-1.0)
            continue
        out[c] = out[c].fillna(0.0)
    return out


def build_season(daily: pd.DataFrame, site_id: str, year: int, doy_start: int,
                 doy_end: int, feature_cols: list[str],
                 site_meta=None, pheno=None) -> pd.DataFrame:
    """Season slice with rolling/coords/phenology attached.

    Mirrors stage1.py::_base_sample_from_frames:698-711. Returns exactly
    T = doy_end - doy_start + 1 rows or raises.
    """
    d = daily_year_frame(daily, site_id, year)
    d = add_rolling_features(d)
    season = d[(d["doy"] >= doy_start) & (d["doy"] <= doy_end)].copy()
    season = season.sort_values("doy").reset_index(drop=True)
    T = int(doy_end) - int(doy_start) + 1
    if len(season) != T:
        present = sorted(season["doy"].tolist())
        raise PreprocessError(
            f"daily season for site={site_id} year={year} has {len(season)} rows, "
            f"expected T={T} (DOY {doy_start}..{doy_end}); present range="
            f"({present[0] if present else None},{present[-1] if present else None})"
        )
    if any(c in COORD_COLS for c in feature_cols):
        if site_meta is None:
            raise PreprocessError(
                f"feature_cols need {list(COORD_COLS)} but no SiteMetadataProvider was given"
            )
        lat, lon = site_meta.latlon(site_id)
        season["좌표-위도"] = lat
        season["좌표-경도"] = lon
    pheno_cols = [c for c in PHENO_CANDIDATE_COLS if c in feature_cols]
    if pheno_cols:
        if pheno is None:
            raise PreprocessError(
                f"feature_cols need phenology {pheno_cols} but no PhenologyProvider was given"
            )
        season = merge_pheno_ffill(season, pheno.rows(site_id, year), pheno_cols)
    return season
