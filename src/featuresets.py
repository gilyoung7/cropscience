def get_feature_cols(run: int):
    base = [
        "일강수량(mm)", "최고기온(°C)", "최저기온(°C)", "평균기온(°C)",
        "평균 풍속(m/s)", "최대 풍속(m/s)",
        "DD10", "GDD10_since_db",
    ]
    meta = ["first_obs_season", "n_obs", "max_gap", "좌표-위도", "좌표-경도"]
    roll = [
        "rain_7d_sum", "rain_14d_sum", "rain_7d_days",
        "tmean_7d_mean", "tmax_7d_max", "tmin_7d_min",
        "DD10_7d_sum",
        "trange", "trange_7d_mean",
    ]
    pheno = ["days_since_flowering", "days_since_growing_start", "is_growing"]

    if run == 1:
        return base
    if run == 2:
        return base + meta
    if run == 3:
        return base + meta + roll
    if run == 4:
        return meta + roll + ["GDD10_since_db"]
    if run == 5:
        return meta + roll + ["GDD10_since_db"] + pheno
    raise ValueError("RUN must be 1..5")
