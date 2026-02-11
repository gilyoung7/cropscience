def get_feature_cols(run: int) -> list[str]:
    base = [
        "일강수량(mm)", "최고기온(°C)", "최저기온(°C)", "평균기온(°C)",
        "평균 풍속(m/s)", "최대 풍속(m/s)",
        "평균 상대습도(%)", "합계 일조시간(h)", "합계 일사량(MJ/m2)",
        "GDD10_since_gs",
    ]
    pheno = ["days_since_growing_start", "days_until_growing_end", "is_growing"]
    roll = [
        "rain_7d_sum", "rain_7d_days",
        "tmean_7d_mean", "tmax_7d_max", "tmin_7d_min",
        "rh_7d_mean",         "sun_7d_sum", "rad_7d_sum",
        "trange", "trange_7d_mean",
    ]
    meta = ["first_obs_season", "n_obs", "max_gap", "좌표-위도", "좌표-경도"]

    if run == 0:
        cols = base
    elif run == 1:
        cols = base + pheno
    elif run == 2:
        cols = base + pheno + roll
    elif run == 3:
        cols = base + pheno + roll + meta
    else:
        raise ValueError("run must be 0..3")
    return list(dict.fromkeys(cols))
