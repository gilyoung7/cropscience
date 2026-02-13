def get_feature_cols(run: int) -> list[str]:
    base = [
        "일강수량(mm)", "평균기온(°C)",
        "평균 풍속(m/s)", "최대 풍속(m/s)",
        "평균 상대습도(%)", "합계 일조시간(h)", "합계 일사량(MJ/m2)",
        "GDD10_since_gs",
    ]
    pheno = ["days_since_growing_start", "days_until_growing_end", "is_growing"]
    roll = [
        "rain_7d_sum", "rain_7d_days",
        "tmean_7d_mean",
        "rh_7d_mean", 
        "trange", "trange_7d_mean",
    ]
    meta = ["좌표-위도", "좌표-경도"]
    tgt = ["tgt_doy_min", "tgt_dst_min", "has_arrived", "days_since_arrival"]

    if run == 0:
        cols = base
    elif run == 1:
        cols = base + pheno
    elif run == 2:
        cols = base + pheno + roll
    elif run == 3:
        cols = base + pheno + roll + meta
    elif run == 4:
        cols = [c for c in base if c not in ["합계 일조시간(h)", "합계 일사량(MJ/m2)"]]
    elif run == 5:
        cols = roll + meta
    elif run == 6:
        cols = roll + meta + pheno
    elif run == 7:
        cols = roll + meta + pheno + [
            "GDD10_since_gs",
            "합계 일조시간(h)",
            "합계 일사량(MJ/m2)",
        ]
    elif run == 8:
        cols = roll + meta + pheno + tgt
    elif run == 9:
        cols = roll + meta + pheno + [
            "GDD10_since_gs",
            "합계 일조시간(h)",
            "합계 일사량(MJ/m2)",
        ] + tgt
    else:
        raise ValueError("run must be 0..9")
    return list(dict.fromkeys(cols))
