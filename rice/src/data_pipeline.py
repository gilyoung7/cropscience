from __future__ import annotations

from pathlib import Path
import hashlib
import json
import numpy as np
import pandas as pd

from rice.configs import config as C
from rice.configs.base import DAILY_CACHE_DIR as BASE_DAILY_CACHE_DIR


# =========================
# Daily (joined) loader + fill + rolling features
# =========================
def load_daily(path: Path) -> pd.DataFrame:
    required_cols = [
        "지점ID",
        "일시",
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
    optional_cols = [
        "DD10",
        "GDD10_cum",
        "GDD10_since_gs",
        C.COUNT_COL,
    ]
    header = pd.read_csv(path, nrows=0)
    cols = header.columns.tolist()
    missing = [c for c in required_cols if c not in cols]
    if missing:
        raise ValueError(f"Missing required daily columns: {missing}")

    usecols_daily = required_cols + [c for c in optional_cols if c in cols]
    daily = pd.read_csv(path, usecols=usecols_daily)

    daily["date"] = pd.to_datetime(daily["일시"], errors="coerce")
    daily = daily.dropna(subset=["date"]).copy()

    daily = daily.rename(columns={"지점ID": "site_id"})
    daily["site_id"] = daily["site_id"].astype(str)

    daily["year"] = daily["date"].dt.year.astype(int)
    daily["doy"] = daily["date"].dt.dayofyear.astype(int)
    if getattr(C, "YEAR_MIN", None) is not None:
        daily = daily[daily["year"] >= int(C.YEAR_MIN)].copy()
    if getattr(C, "YEAR_MAX", None) is not None:
        daily = daily[daily["year"] <= int(C.YEAR_MAX)].copy()
    daily = daily.sort_values(["site_id", "date"]).reset_index(drop=True)

    if "GDD10_since_gs" not in daily.columns:
        raise ValueError("Daily data must include GDD10_since_gs")

    # sanity: basic ranges
    n_sites = daily["site_id"].nunique()
    year_min = int(daily["year"].min()) if not daily.empty else None
    year_max = int(daily["year"].max()) if not daily.empty else None
    doy_min = int(daily["doy"].min()) if not daily.empty else None
    doy_max = int(daily["doy"].max()) if not daily.empty else None
    print(f"[daily] n_sites={n_sites} year_range=({year_min},{year_max}) doy_range=({doy_min},{doy_max})")

    # ---- daily 단계에서 기상 NaN 보간 ----
    weather_cols = [
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
    if "DD10" in daily.columns:
        daily["DD10_7d_sum"] = g["DD10"].transform(lambda s: s.rolling(7, min_periods=1).sum())

    # humidity rolling
    if "평균 상대습도(%)" in daily.columns:
        daily["rh_7d_mean"] = g["평균 상대습도(%)"].transform(lambda s: s.rolling(7, min_periods=1).mean())
        daily["rh_14d_mean"] = g["평균 상대습도(%)"].transform(lambda s: s.rolling(14, min_periods=1).mean())

    # sun/radiation rolling
    if "합계 일조시간(h)" in daily.columns:
        daily["sun_7d_sum"] = g["합계 일조시간(h)"].transform(lambda s: s.rolling(7, min_periods=1).sum())
    if "합계 일사량(MJ/m2)" in daily.columns:
        daily["rad_7d_sum"] = g["합계 일사량(MJ/m2)"].transform(lambda s: s.rolling(7, min_periods=1).sum())

    # trange rolling
    daily["trange_7d_mean"] = g["trange"].transform(lambda s: s.rolling(7, min_periods=1).mean())
    return daily


def _daily_cache_key(path_daily: Path) -> str:
    st = path_daily.stat()
    roll_config = {
        "rain_windows": [7, 14],
        "temp_windows": [7],
        "rh_windows": [7, 14],
        "sun_windows": [7],
        "rad_windows": [7],
        "dd_window": 7,
        "impute_policy": getattr(C, "IMPUTE_POLICY", "unknown"),
        "miss_indicator_policy": getattr(C, "MISS_INDICATOR_POLICY", "unknown"),
        "year_max": getattr(C, "YEAR_MAX", None),
        "year_min": getattr(C, "YEAR_MIN", None),
    }
    roll_hash = hashlib.sha1(json.dumps(roll_config, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    preproc_version = getattr(C, "PREPROC_VERSION", "v1.0")
    raw = "|".join(
        [
            str(path_daily.resolve()),
            str(st.st_mtime_ns),
            str(st.st_size),
            str(getattr(C, "YEAR_MAX", None)),
            str(getattr(C, "YEAR_MIN", None)),
            preproc_version,
            roll_hash,
        ]
    )
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]


def load_daily_preprocessed(path_daily: Path) -> pd.DataFrame:
    cache_dir = Path(getattr(C, "DAILY_CACHE_DIR", BASE_DAILY_CACHE_DIR))
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_key = _daily_cache_key(path_daily)
    cache_path = cache_dir / f"daily_preprocessed_{cache_key}.pkl"

    if cache_path.exists():
        print(f"[cache] hit: {cache_path}")
        return pd.read_pickle(cache_path)

    print(f"[cache] miss: {cache_path}")
    daily = load_daily(path_daily)
    daily = add_rolling_features(daily)
    daily.to_pickle(cache_path)
    print(f"[cache] saved: {cache_path}")
    return daily


# =========================
# OBS loader + aggregation + meta
# =========================
def load_obs(path: Path) -> pd.DataFrame:
    """
    LONG2 file loader. Requires:
    site_id, year, obs_doy, COUNT_COL, days_since_growing_start, days_until_growing_end, is_growing, and lat/lon columns.
    """
    # utf-8-sig handles BOM-prefixed headers (e.g., "\ufeffsite_id")
    obs = pd.read_csv(path, encoding="utf-8-sig")
    obs = obs.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)

    # Backward-compatible aliases for arrival guidance columns.
    if "tgt_doy_min" not in obs.columns and "tgt_doy_rule" in obs.columns:
        obs["tgt_doy_min"] = obs["tgt_doy_rule"]
    if "tgt_dst_min" not in obs.columns and "tgt_dst_rule" in obs.columns:
        obs["tgt_dst_min"] = obs["tgt_dst_rule"]
    if getattr(C, "APPLY_PEST_FILTER", False) and C.PEST_COL in obs.columns and C.TARGET_PEST is not None:
        obs = obs.loc[obs[C.PEST_COL] == C.TARGET_PEST].copy()

    if C.LABEL_COL in obs.columns:
        label_series = pd.to_numeric(obs[C.LABEL_COL], errors="coerce")
        label_unique = sorted({x for x in label_series.dropna().unique()})
        miss_rate = float(label_series.isna().mean())
        print(f"[obs] pre_year_filter rows={len(obs)} label_event_unique={label_unique} label_event_missing_rate={miss_rate:.4f}")
    else:
        print(f"[obs] pre_year_filter rows={len(obs)} label_event_unique=None label_event_missing_rate=None")

    need = [
        "site_id", "year", "obs_doy",
        C.COUNT_COL,
        "days_since_growing_start", "days_until_growing_end", "is_growing",
        "좌표-위도", "좌표-경도",
    ]
    missing = [c for c in need if c not in obs.columns]
    if missing:
        raise ValueError(f"Missing columns in LONG2: {missing}")

    obs["site_id"] = obs["site_id"].astype(str)
    obs["year"] = pd.to_numeric(obs["year"], errors="coerce").astype(int)
    if getattr(C, "YEAR_MIN", None) is not None:
        obs = obs[obs["year"] >= int(C.YEAR_MIN)].copy()
    if getattr(C, "YEAR_MAX", None) is not None:
        obs = obs[obs["year"] <= int(C.YEAR_MAX)].copy()
    year_min = int(obs["year"].min()) if not obs.empty else None
    year_max = int(obs["year"].max()) if not obs.empty else None
    print(f"[obs] post_year_filter rows={len(obs)} year_range=({year_min},{year_max})")

    obs["obs_doy"] = pd.to_numeric(obs["obs_doy"], errors="coerce")
    obs = obs.dropna(subset=["obs_doy"]).copy()
    obs["obs_doy"] = obs["obs_doy"].astype(int)

    obs[C.COUNT_COL] = pd.to_numeric(obs[C.COUNT_COL], errors="coerce")
    if C.LABEL_COL in obs.columns:
        obs[C.LABEL_COL] = pd.to_numeric(obs[C.LABEL_COL], errors="coerce")
    for c in ["days_since_growing_start", "days_until_growing_end", "is_growing"]:
        obs[c] = pd.to_numeric(obs[c], errors="coerce")

    # Optional target guidance columns from LONG
    for c in [
        "tgt_doy_min",
        "tgt_dst_min",
        "has_arrived",
        "days_since_arrival",
    ]:
        if c in obs.columns:
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
    if C.COUNT_COL in obs2.columns:
        obs2[C.COUNT_COL] = pd.to_numeric(obs2[C.COUNT_COL], errors="coerce")
    label_col = C.LABEL_COL if C.LABEL_COL in obs2.columns else None
    if label_col is not None:
        obs2[label_col] = pd.to_numeric(obs2[label_col], errors="coerce").fillna(0.0)

    drop_cols = ["obs_doy"]
    if label_col is None:
        drop_cols.append(C.COUNT_COL)
    obs2 = obs2.dropna(subset=drop_cols).copy()
    obs2["obs_doy"] = obs2["obs_doy"].astype(int)

    agg = {}
    if C.COUNT_COL in obs2.columns:
        agg[C.COUNT_COL] = "max"
    if label_col is not None:
        agg[label_col] = "max"
    obs2 = obs2.groupby(["site_id", "year", "obs_doy"], as_index=False).agg(agg)
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
    candidate_cols = [
        "days_since_growing_start",
        "days_until_growing_end",
        "is_growing",
        "growing_start_doy",
        "growing_end_doy",
        "growing_mid_doy",
        "growing_len_days",
        "tgt_doy_min",
        "tgt_dst_min",
        "has_arrived",
        "days_since_arrival",
    ]
    pheno_cols = [c for c in candidate_cols if c in obs.columns]
    if not pheno_cols:
        return train_df

    pheno = obs[["site_id", "year", "obs_doy"] + pheno_cols].copy()

    pheno["site_id"] = pheno["site_id"].astype(str)
    pheno["year"] = pheno["year"].astype(int)
    pheno["obs_doy"] = pd.to_numeric(pheno["obs_doy"], errors="coerce").astype("Int64")

    for c in pheno_cols:
        pheno[c] = pd.to_numeric(pheno[c], errors="coerce")

    pheno = pheno.dropna(subset=["obs_doy"]).copy()
    pheno["doy"] = pheno["obs_doy"].astype(int)
    pheno = pheno.drop(columns=["obs_doy"])

    # 하루 1레코드로 압축 (중복 merge 방지)
    agg = {c: "last" for c in pheno_cols}
    if "is_growing" in agg:
        agg["is_growing"] = "max"
    pheno = (
        pheno.sort_values(["site_id", "year", "doy"])
        .groupby(["site_id", "year", "doy"], as_index=False)
        .agg(agg)
    )

    out = train_df.merge(pheno, on=["site_id", "year", "doy"], how="left")
    out = out.sort_values(["site_id", "year", "doy"]).copy()

    out[pheno_cols] = out.groupby(["site_id", "year"], sort=False)[pheno_cols].ffill()
    for c in pheno_cols:
        # Keep optional arrival features as NaN so __miss indicators are meaningful.
        if c in {"tgt_doy_min", "tgt_dst_min", "days_since_arrival"}:
            continue
        out[c] = out[c].fillna(0.0)
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
        "평균 상대습도(%)",
        "합계 일조시간(h)",
        "합계 일사량(MJ/m2)",
        "GDD10_since_gs",
        # rolling/derived
        "rain_7d_sum",
        "rain_14d_sum",
        "rain_7d_days",
        "tmean_7d_mean",
        "tmax_7d_max",
        "tmin_7d_min",
        "rh_7d_mean",
        "rh_14d_mean",
        "sun_7d_sum",
        "rad_7d_sum",
        "trange",
        "trange_7d_mean",
    ]
    if "DD10" in daily.columns:
        feature_cols.insert(6, "DD10")
        feature_cols.insert(feature_cols.index("trange"), "DD10_7d_sum")

    cols_keep = ["site_id", "year", "doy", "date"] + [c for c in feature_cols if c in daily.columns]
    daily_feat = daily[cols_keep].copy()
    return daily_feat, [c for c in feature_cols if c in daily.columns]
