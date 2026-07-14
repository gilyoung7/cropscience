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
    meta = ["좌표-위도", "좌표-경도"]
    # run=6 extensions (Phase A0/A audit, 2026-05-13):
    #   roll_ext: 3 extra rolling aggregates passing the missing-rate gate
    #   pheno_ext: 4 phenology fields from the LONG csv (site-leaning variance)
    # Excluded: rh_14d_mean (46% test miss), DD10_7d_sum (raw DD10 absent in daily csv).
    roll_ext = roll + ["rain_14d_sum", "wind_7d_mean", "wind_7d_max"]
    pheno_ext = pheno + ["best_suitability", "best_months", "offset_days", "window_idx"]

    if run == 0:
        cols = base
    elif run == 1:
        cols = base + pheno
    elif run == 2:
        cols = base + pheno + roll
    elif run == 3:
        cols = base + pheno + roll + meta
    elif run == 4:
        cols = roll + meta + pheno
    elif run == 5:
        cols = [c for c in (roll + meta + pheno) if c != "rad_7d_sum"]
    elif run == 6:
        # D = 22 (run=4 baseline 15 + 3 rolling + 4 phenology)
        cols = roll_ext + meta + pheno_ext
    elif run == 7:
        # D = 18 (run=4 baseline 15 + 3 rolling only; pheno_ext 4 excluded)
        cols = roll_ext + meta + pheno
    elif run == 8:
        # Phase S12: run=4 + GDD10_since_gs (thermal accumulation signal).
        # 16 features, identical to run=4 except for the GDD column.
        cols = roll + meta + pheno + ["GDD10_since_gs"]
    else:
        raise ValueError("run must be 0..8")
    return list(dict.fromkeys(cols))
