from __future__ import annotations

from pathlib import Path
import re
import numpy as np
import pandas as pd

from configs import config as C


# =========================
# Daily (joined) loader + fill + rolling features
# =========================
def load_daily(path: Path) -> pd.DataFrame:
    usecols_daily = [
        "지점ID", "일시",
        "일강수량(mm)", "최고기온(°C)", "최저기온(°C)", "평균기온(°C)",
        "평균 풍속(m/s)", "최대 풍속(m/s)",
        "DD10", "GDD10_cum",
        C.COUNT_COL,
    ]
    daily = pd.read_csv(path, usecols=usecols_daily)

    daily["date"] = pd.to_datetime(daily["일시"], errors="coerce")
    daily = daily.dropna(subset=["date"]).copy()

    daily = daily.rename(columns={"지점ID": "site_id"})
    daily["site_id"] = daily["site_id"].astype(str)

    daily["year"] = daily["date"].dt.year.astype(int)
    daily["doy"] = daily["date"].dt.dayofyear.astype(int)
    daily = daily.sort_values(["site_id", "date"]).reset_index(drop=True)

    # ---- daily 단계에서 기상 NaN 보간 ----
    weather_cols = [
        "일강수량(mm)",
        "최고기온(°C)",
        "최저기온(°C)",
        "평균기온(°C)",
        "평균 풍속(m/s)",
        "최대 풍속(m/s)",
    ]
    for c in weather_cols:
        daily[c] = pd.to_numeric(daily[c], errors="coerce")

    out = []
    for (site, year), sub in daily.groupby(["site_id", "year"], sort=False):
        sub = sub.sort_values("doy").copy()
        for c in weather_cols:
            sub[c] = sub[c].interpolate(limit_direction="both").ffill().bfill()
        sub["일강수량(mm)"] = sub["일강수량(mm)"].fillna(0.0)
        out.append(sub)

    daily = pd.concat(out, ignore_index=True)
    return daily


def load_gdd_since_db(gdd_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    """
    Read many per-site GDD csvs (filename like site_32458_56647_GDD_timeseries.csv)
    -> dataframe [site_id, date, GDD10_since_db]
    """
    dfs = []
    bad_files: list[str] = []

    for fp in sorted(gdd_dir.glob("*.csv")):
        m = re.search(r"site_(\d+_\d+)", fp.name)
        if not m:
            bad_files.append(fp.name)
            continue
        site_id = m.group(1)

        df = pd.read_csv(fp, usecols=["date", "GDD10_since_db"])
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        df["GDD10_since_db"] = pd.to_numeric(df["GDD10_since_db"], errors="coerce")
        df["site_id"] = site_id

        df = df.dropna(subset=["date"]).copy()
        dfs.append(df)

    if not dfs:
        raise RuntimeError(f"No GDD csv found under: {gdd_dir}")

    gdd_db = pd.concat(dfs, ignore_index=True)
    gdd_db = gdd_db.drop_duplicates(["site_id", "date"], keep="last")
    return gdd_db, bad_files


def merge_gdd_since_db(daily: pd.DataFrame, gdd_db: pd.DataFrame) -> pd.DataFrame:
    daily = daily.copy()
    daily["site_id"] = daily["site_id"].astype(str)
    daily["date"] = pd.to_datetime(daily["date"], errors="coerce")

    out = daily.merge(gdd_db, on=["site_id", "date"], how="left")
    out["GDD10_since_db"] = out["GDD10_since_db"].fillna(0.0)
    return out


def add_rolling_features(daily: pd.DataFrame) -> pd.DataFrame:
    """
    rolling/파생 feature 생성 (site-year 기준)
    """
    daily = daily.sort_values(["site_id", "year", "doy"]).copy()
    g = daily.groupby(["site_id", "year"], sort=False)

    # 파생: 일교차
    daily["trange"] = daily["최고기온(°C)"] - daily["최저기온(°C)"]

    # rain rolling
    daily["rain_7d_sum"]  = g["일강수량(mm)"].transform(lambda s: s.rolling(7,  min_periods=1).sum())
    daily["rain_14d_sum"] = g["일강수량(mm)"].transform(lambda s: s.rolling(14, min_periods=1).sum())
    daily["rain_7d_days"] = g["일강수량(mm)"].transform(lambda s: (s > 0).rolling(7, min_periods=1).sum())

    # temp rolling
    daily["tmean_7d_mean"] = g["평균기온(°C)"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    daily["tmax_7d_max"]   = g["최고기온(°C)"].transform(lambda s: s.rolling(7, min_periods=1).max())
    daily["tmin_7d_min"]   = g["최저기온(°C)"].transform(lambda s: s.rolling(7, min_periods=1).min())

    # DD/GDD rolling
    daily["DD10_7d_sum"] = g["DD10"].transform(lambda s: s.rolling(7, min_periods=1).sum())

    # trange rolling
    daily["trange_7d_mean"] = g["trange"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    return daily


# =========================
# OBS loader + aggregation + meta
# =========================
def load_obs(path: Path) -> pd.DataFrame:
    """
    LONG2 file loader. Requires:
    site_id, year, obs_doy, COUNT_COL, days_since_flowering, days_since_growing_start, is_growing, and lat/lon columns.
    """
    obs = pd.read_csv(path)

    need = [
        "site_id", "year", "obs_doy",
        C.COUNT_COL,
        "days_since_flowering", "days_since_growing_start", "is_growing",
        "좌표-위도", "좌표-경도",
    ]
    missing = [c for c in need if c not in obs.columns]
    if missing:
        raise ValueError(f"Missing columns in LONG2: {missing}")

    obs["site_id"] = obs["site_id"].astype(str)
    obs["year"] = pd.to_numeric(obs["year"], errors="coerce").astype(int)

    obs["obs_doy"] = pd.to_numeric(obs["obs_doy"], errors="coerce")
    obs = obs.dropna(subset=["obs_doy"]).copy()
    obs["obs_doy"] = obs["obs_doy"].astype(int)

    obs[C.COUNT_COL] = pd.to_numeric(obs[C.COUNT_COL], errors="coerce")
    for c in ["days_since_flowering", "days_since_growing_start", "is_growing"]:
        obs[c] = pd.to_numeric(obs[c], errors="coerce")

    obs["좌표-위도"] = pd.to_numeric(obs["좌표-위도"], errors="coerce")
    obs["좌표-경도"] = pd.to_numeric(obs["좌표-경도"], errors="coerce")

    obs = obs.sort_values(["site_id", "year", "obs_doy"]).reset_index(drop=True)
    return obs


def aggregate_obs_daily_max(obs: pd.DataFrame) -> pd.DataFrame:
    """
    같은 날 여러 번 관측이 있으면 COUNT_COL의 최대값으로 대표
    - count NaN은 제거(미관측 처리)
    returns: obs2 [site_id, year, obs_doy, COUNT_COL]
    """
    obs2 = obs.copy()

    obs2["obs_doy"] = pd.to_numeric(obs2["obs_doy"], errors="coerce")
    obs2[C.COUNT_COL] = pd.to_numeric(obs2[C.COUNT_COL], errors="coerce")

    obs2 = obs2.dropna(subset=["obs_doy", C.COUNT_COL]).copy()
    obs2["obs_doy"] = obs2["obs_doy"].astype(int)

    obs2 = obs2.groupby(["site_id", "year", "obs_doy"], as_index=False)[C.COUNT_COL].max()
    return obs2


def make_obs_meta(obs2: pd.DataFrame, doy_start: int, doy_end: int) -> pd.DataFrame:
    """
    관측 프로세스 feature: first_obs_doy, n_obs, max_gap (+ first_obs_season)
    based on obs2 within season [doy_start, doy_end]
    """
    obs2_season = obs2[(obs2["obs_doy"] >= doy_start) & (obs2["obs_doy"] <= doy_end)].copy()
    obs2_season = obs2_season.sort_values(["site_id", "year", "obs_doy"])

    rows = []
    for (site, year), sub in obs2_season.groupby(["site_id", "year"], sort=False):
        d = np.sort(sub["obs_doy"].to_numpy())
        n_obs = len(d)
        first_obs = int(d[0])
        if n_obs >= 2:
            max_gap = int(np.max(np.diff(d)))
        else:
            max_gap = int(doy_end - doy_start)  # 1회 관측이면 매우 불확실
        rows.append(
            {
                "site_id": site,
                "year": int(year),
                "first_obs_doy": first_obs,
                "n_obs": int(n_obs),
                "max_gap": int(max_gap),
            }
        )

    meta = pd.DataFrame(rows)
    T = doy_end - doy_start + 1
    meta["first_obs_season"] = (meta["first_obs_doy"] - doy_start + 1).clip(1, T).astype(int)
    return meta


# =========================
# Static + Phenology merge
# =========================
def add_site_static_latlon(train_df: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    site_static = obs[["site_id", "좌표-위도", "좌표-경도"]].drop_duplicates("site_id").copy()
    site_static["좌표-위도"] = pd.to_numeric(site_static["좌표-위도"], errors="coerce")
    site_static["좌표-경도"] = pd.to_numeric(site_static["좌표-경도"], errors="coerce")

    out = train_df.merge(site_static, on="site_id", how="left")
    out["좌표-위도"] = out["좌표-위도"].fillna(out["좌표-위도"].mean())
    out["좌표-경도"] = out["좌표-경도"].fillna(out["좌표-경도"].mean())
    return out


def merge_pheno_daily_ffill(train_df: pd.DataFrame, obs: pd.DataFrame) -> pd.DataFrame:
    """
    LONG2 phenology 컬럼을 daily(train_df)에 doy 기준으로 merge + ffill
    """
    pheno_cols = [
        "site_id",
        "year",
        "obs_doy",
        "days_since_flowering",
        "days_since_growing_start",
        "is_growing",
    ]
    pheno = obs[pheno_cols].copy()

    pheno["site_id"] = pheno["site_id"].astype(str)
    pheno["year"] = pheno["year"].astype(int)
    pheno["obs_doy"] = pd.to_numeric(pheno["obs_doy"], errors="coerce").astype("Int64")

    for c in ["days_since_flowering", "days_since_growing_start", "is_growing"]:
        pheno[c] = pd.to_numeric(pheno[c], errors="coerce")

    pheno = pheno.dropna(subset=["obs_doy"]).copy()
    pheno["doy"] = pheno["obs_doy"].astype(int)
    pheno = pheno.drop(columns=["obs_doy"])

    # 하루 1레코드로 압축 (중복 merge 방지)
    pheno = (
        pheno.sort_values(["site_id", "year", "doy"])
        .groupby(["site_id", "year", "doy"], as_index=False)
        .agg(
            {
                "days_since_flowering": "last",
                "days_since_growing_start": "last",
                "is_growing": "max",
            }
        )
    )

    out = train_df.merge(pheno, on=["site_id", "year", "doy"], how="left")
    out = out.sort_values(["site_id", "year", "doy"]).copy()

    cols_fill = ["days_since_flowering", "days_since_growing_start", "is_growing"]
    out[cols_fill] = out.groupby(["site_id", "year"], sort=False)[cols_fill].ffill()

    out["days_since_flowering"] = out["days_since_flowering"].fillna(0.0)
    out["days_since_growing_start"] = out["days_since_growing_start"].fillna(0.0)
    out["is_growing"] = out["is_growing"].fillna(0.0)
    return out


# =========================
# Final: build daily feature frame for training
# =========================
def make_daily_feature_frame(daily: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """
    Returns (daily_feat, feature_cols_base_plus_roll)
    daily_feat columns: [site_id, year, doy, date] + features...
    """
    feature_cols = [
        "일강수량(mm)",
        "최고기온(°C)",
        "최저기온(°C)",
        "평균기온(°C)",
        "평균 풍속(m/s)",
        "최대 풍속(m/s)",
        "DD10",
        "GDD10_since_db",
        # rolling/derived
        "rain_7d_sum",
        "rain_14d_sum",
        "rain_7d_days",
        "tmean_7d_mean",
        "tmax_7d_max",
        "tmin_7d_min",
        "DD10_7d_sum",
        "trange",
        "trange_7d_mean",
    ]
    daily_feat = daily[["site_id", "year", "doy", "date"] + feature_cols].copy()
    return daily_feat, feature_cols
