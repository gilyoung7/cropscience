def get_feature_cols(run: int) -> list[str]:
    base = [
        "일강수량(mm)", "최고기온(°C)", "최저기온(°C)", "평균기온(°C)",
        "평균 풍속(m/s)", "최대 풍속(m/s)",
        "GDD10_cum",
    ]
    pheno = ["days_since_growing_start", "days_until_growing_end", "is_growing"]
    roll = [
        "rain_7d_sum", "rain_7d_days",
        "tmean_7d_mean", "tmax_7d_max", "tmin_7d_min",
        "trange", "trange_7d_mean",
    ]
    meta = ["좌표-위도", "좌표-경도"]

    if run == 0:
        cols = roll + meta
    elif run == 1:
        cols = base + pheno
    elif run == 2:
        cols = base + pheno + roll
    elif run == 3:
        cols = base + pheno + roll + meta
    elif run == 4:
        cols = roll + meta + pheno
    else:
        raise ValueError("run must be 0..4")
    return list(dict.fromkeys(cols))
