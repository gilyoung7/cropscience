from __future__ import annotations

from dataclasses import dataclass
import hashlib
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


CTYPE2ID = {"interval": 0, "right": 1, "left": 2}


def build_train_frame(
    daily_feat: pd.DataFrame,
    labels: pd.DataFrame,
    obs_meta: pd.DataFrame,
    T: int,
) -> pd.DataFrame:
    """
    daily_feat: [site_id, year, doy, date] + base/rolling features
    labels: [site_id, year, censor_type, L_doy, R_doy, ...]
    obs_meta: [site_id, year, first_obs_season, n_obs, max_gap, ...]
    """
    df = daily_feat.copy()

    # type unify
    df["site_id"] = df["site_id"].astype(str)
    df["year"] = df["year"].astype(int)
    labels = labels.copy()
    labels["site_id"] = labels["site_id"].astype(str)
    labels["year"] = labels["year"].astype(int)
    obs_meta = obs_meta.copy()
    obs_meta["site_id"] = obs_meta["site_id"].astype(str)
    obs_meta["year"] = obs_meta["year"].astype(int)

    # merge labels
    df = df.merge(labels, on=["site_id", "year"], how="inner")

    # merge obs_meta (관측 프로세스 feature)
    df = df.merge(
        obs_meta[["site_id", "year", "first_obs_season", "n_obs", "max_gap"]],
        on=["site_id", "year"],
        how="left",
    )

    # fill missing obs_meta
    df["first_obs_season"] = df["first_obs_season"].fillna(1).astype(int)
    df["n_obs"] = df["n_obs"].fillna(0).astype(int)
    df["max_gap"] = df["max_gap"].fillna(T).astype(int)

    return df


def slice_season(df: pd.DataFrame, doy_start: int, doy_end: int) -> pd.DataFrame:
    out = df[(df["doy"] >= doy_start) & (df["doy"] <= doy_end)].copy()

    # label clamp to season
    out["L_doy"] = out["L_doy"].clip(lower=doy_start, upper=doy_end)
    out["R_doy"] = out["R_doy"].clip(lower=doy_start, upper=doy_end)
    return out


def build_samples_season(
    df_season: pd.DataFrame,
    feature_cols: list[str],
    doy_start: int,
    doy_end: int,
    pheno_ext_cols: list[str] | None = None,
) -> tuple[list[dict], int, list[str]]:
    """
    Returns (samples, dropped_groups, feature_names)
    samples item: {"site_id","year","X","L","R","censor_type", optional "pheno_vec"}
      - X: (T,D) float32
      - L,R: season coordinates in 1..T
      - pheno_vec: (len(pheno_ext_cols),) site-year static vector, taken from
        the last DOY of the season (after merge_pheno_daily_ffill). Present
        only when `pheno_ext_cols` is provided and the columns exist in df_season.
    """
    T = doy_end - doy_start + 1
    samples: list[dict] = []
    dropped = 0
    printed_nan = False
    feature_names: list[str] = []

    for (site, year), sub in df_season.groupby(["site_id", "year"], sort=False):
        sub = sub.sort_values("doy")

        # 시즌 구간이 정확히 T개인지 확인 (빠진 day 있으면 drop)
        if len(sub) != T:
            dropped += 1
            continue

        # impute + missing indicator
        X_df = sub[feature_cols].copy()
        for c in feature_cols:
            if c in X_df.columns:
                X_df[c] = pd.to_numeric(X_df[c], errors="coerce")
                miss = X_df[c].isna().astype(np.float32)
                X_df[c] = X_df[c].fillna(0.0)
                X_df[f"{c}__miss"] = miss

        if not feature_names:
            feature_names = list(X_df.columns)

        X = X_df.to_numpy(dtype=np.float32)

        L = int(sub["L_doy"].iloc[0]) - doy_start + 1
        R = int(sub["R_doy"].iloc[0]) - doy_start + 1
        ctype = str(sub["censor_type"].iloc[0])

        # clamp to [1, T]
        L = min(max(L, 1), T)
        R = min(max(R, 1), T)

        # debug: detect non-finite
        if not printed_nan and not np.isfinite(X).all():
            bad = []
            for i, col in enumerate(X_df.columns):
                if not np.isfinite(X[:, i]).all():
                    bad.append(col)
            print(f"[nan_check] non-finite in site={site} year={year} cols={bad}")
            printed_nan = True

        rec = {"site_id": site, "year": int(year), "X": X, "L": L, "R": R, "censor_type": ctype}

        if pheno_ext_cols:
            present = [c for c in pheno_ext_cols if c in sub.columns]
            if present:
                last_row = sub.iloc[-1]
                vec = np.asarray(
                    [pd.to_numeric(last_row[c], errors="coerce") for c in pheno_ext_cols],
                    dtype=np.float32,
                )
                # Same fallbacks as merge_pheno_daily_ffill: offset_days -> -1, else 0.
                for i, c in enumerate(pheno_ext_cols):
                    if not np.isfinite(vec[i]):
                        vec[i] = -1.0 if c == "offset_days" else 0.0
                rec["pheno_vec"] = vec
        samples.append(rec)

    return samples, dropped, feature_names


def split_by_site(samples: list[dict], val_frac=0.1, test_frac=0.1, seed=42):
    rng = np.random.default_rng(seed)
    sites = sorted(list({s["site_id"] for s in samples}))
    rng.shuffle(sites)

    n = len(sites)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_sites = set(sites[:n_test])
    val_sites = set(sites[n_test : n_test + n_val])
    train_sites = set(sites[n_test + n_val :])

    train = [s for s in samples if s["site_id"] in train_sites]
    val = [s for s in samples if s["site_id"] in val_sites]
    test = [s for s in samples if s["site_id"] in test_sites]
    return train, val, test


def split_by_site_year(samples: list[dict], val_frac=0.1, test_frac=0.1, seed=42):
    """
    Random split by (site_id, year) pair.
    The same site-year never crosses splits, but different years from the same
    site may appear in train/val/test.
    """
    rng = np.random.default_rng(seed)
    pairs = sorted({(str(s["site_id"]), int(s["year"])) for s in samples})
    rng.shuffle(pairs)

    n = len(pairs)
    n_test = int(n * test_frac)
    n_val = int(n * val_frac)

    test_pairs = set(pairs[:n_test])
    val_pairs = set(pairs[n_test : n_test + n_val])
    train_pairs = set(pairs[n_test + n_val :])

    def _key(s: dict) -> tuple[str, int]:
        return str(s["site_id"]), int(s["year"])

    train = [s for s in samples if _key(s) in train_pairs]
    val = [s for s in samples if _key(s) in val_pairs]
    test = [s for s in samples if _key(s) in test_pairs]
    return train, val, test


def split_by_temporal(
    samples: list[dict],
    train_end_year: int = 2018,
    val_start_year: int = 2019,
    val_end_year: int = 2020,
    test_start_year: int = 2021,
    test_end_year: int = 2022,
):
    """
    Split by absolute year boundaries. Samples outside the explicit ranges are
    left out by design so boundary choices stay visible in sanity counts.
    """
    train = [s for s in samples if int(s["year"]) <= int(train_end_year)]
    val = [s for s in samples if int(val_start_year) <= int(s["year"]) <= int(val_end_year)]
    test = [s for s in samples if int(test_start_year) <= int(s["year"]) <= int(test_end_year)]
    return train, val, test


def split_by_year(
    samples: list[dict],
    val_year: int = 2022,
    test_year_min: int = 2023,
    test_year_max: int = 2024,
):
    """
    Split by absolute year boundaries (simple form):
      train: year <  val_year
      val:   year == val_year
      test:  test_year_min <= year <= test_year_max
    Samples outside any of these ranges are dropped by design.
    """
    train = [s for s in samples if int(s["year"]) < int(val_year)]
    val = [s for s in samples if int(s["year"]) == int(val_year)]
    test = [s for s in samples if int(test_year_min) <= int(s["year"]) <= int(test_year_max)]
    return train, val, test


def split_samples(
    samples: list[dict],
    val_frac=0.1,
    test_frac=0.1,
    seed=42,
    split_mode: str = "site",
    temporal_train_end_year: int = 2018,
    temporal_val_start_year: int = 2019,
    temporal_val_end_year: int = 2020,
    temporal_test_start_year: int = 2021,
    temporal_test_end_year: int = 2022,
    val_year: int | None = None,
    test_year_min: int | None = None,
    test_year_max: int | None = None,
):
    mode = str(split_mode).strip().lower()
    if mode == "site":
        return split_by_site(samples, val_frac=val_frac, test_frac=test_frac, seed=seed)
    if mode == "site_year":
        return split_by_site_year(samples, val_frac=val_frac, test_frac=test_frac, seed=seed)
    if mode == "temporal":
        return split_by_temporal(
            samples,
            train_end_year=temporal_train_end_year,
            val_start_year=temporal_val_start_year,
            val_end_year=temporal_val_end_year,
            test_start_year=temporal_test_start_year,
            test_end_year=temporal_test_end_year,
        )
    if mode == "year":
        if val_year is None or test_year_min is None or test_year_max is None:
            raise ValueError(
                "split_mode='year' requires val_year, test_year_min, test_year_max"
            )
        return split_by_year(
            samples,
            val_year=int(val_year),
            test_year_min=int(test_year_min),
            test_year_max=int(test_year_max),
        )
    raise ValueError(
        f"unknown split_mode={split_mode!r}; expected one of: site, site_year, temporal, year"
    )


def split_fingerprint(train: list[dict], val: list[dict], test: list[dict], sample_n: int = 5) -> dict:
    def _sites(samples: list[dict]) -> list[str]:
        return sorted({s["site_id"] for s in samples})

    def _hash(sites: list[str]) -> str:
        joined = "|".join(sites)
        return hashlib.sha1(joined.encode("utf-8")).hexdigest()

    train_sites = _sites(train)
    val_sites = _sites(val)
    test_sites = _sites(test)

    return {
        "train_sites_sample": train_sites[:sample_n],
        "val_sites_sample": val_sites[:sample_n],
        "test_sites_sample": test_sites[:sample_n],
        "train_sites_hash": _hash(train_sites),
        "val_sites_hash": _hash(val_sites),
        "test_sites_hash": _hash(test_sites),
        "n_train_sites": len(train_sites),
        "n_val_sites": len(val_sites),
        "n_test_sites": len(test_sites),
    }


def log_split_fingerprint(
    label: str,
    train: list[dict],
    val: list[dict],
    test: list[dict],
    sample_n: int = 5,
):
    fp = split_fingerprint(train, val, test, sample_n=sample_n)
    print(
        f"[split:{label}] train_sites={fp['n_train_sites']} val_sites={fp['n_val_sites']} test_sites={fp['n_test_sites']} "
        f"train_hash={fp['train_sites_hash']} val_hash={fp['val_sites_hash']} test_hash={fp['test_sites_hash']} "
        f"train_sample={fp['train_sites_sample']} val_sample={fp['val_sites_sample']} test_sample={fp['test_sites_sample']}"
    )


def split_sanity_summary(train: list[dict], val: list[dict], test: list[dict]) -> dict:
    def _sites(samples_: list[dict]) -> set[str]:
        return {str(s["site_id"]) for s in samples_}

    def _pairs(samples_: list[dict]) -> set[tuple[str, int]]:
        return {(str(s["site_id"]), int(s["year"])) for s in samples_}

    def _years(samples_: list[dict]) -> list[int]:
        return sorted({int(s["year"]) for s in samples_})

    def _counts(samples_: list[dict]) -> dict:
        counts = censor_type_counts(samples_)
        total = sum(counts.values())
        event = counts.get("interval", 0) + counts.get("left", 0)
        return {
            "interval": counts.get("interval", 0),
            "right": counts.get("right", 0),
            "left": counts.get("left", 0),
            "event": event,
            "total": total,
            "event_ratio": event / total if total else 0.0,
            "right_ratio": counts.get("right", 0) / total if total else 0.0,
        }

    train_sites, val_sites, test_sites = _sites(train), _sites(val), _sites(test)
    train_pairs, val_pairs, test_pairs = _pairs(train), _pairs(val), _pairs(test)

    train_years_by_site: dict[str, set[int]] = {}
    for s in train:
        train_years_by_site.setdefault(str(s["site_id"]), set()).add(int(s["year"]))

    test_hist = 0
    for s in test:
        years = train_years_by_site.get(str(s["site_id"]), set())
        if any(y < int(s["year"]) for y in years):
            test_hist += 1

    def _range(years: list[int]) -> tuple[int | None, int | None]:
        if not years:
            return None, None
        return years[0], years[-1]

    return {
        "train_sites": len(train_sites),
        "val_sites": len(val_sites),
        "test_sites": len(test_sites),
        "train_pairs": len(train_pairs),
        "val_pairs": len(val_pairs),
        "test_pairs": len(test_pairs),
        "train_test_site_intersection": len(train_sites & test_sites),
        "train_val_site_intersection": len(train_sites & val_sites),
        "val_test_site_intersection": len(val_sites & test_sites),
        "train_test_pair_intersection": len(train_pairs & test_pairs),
        "train_val_pair_intersection": len(train_pairs & val_pairs),
        "val_test_pair_intersection": len(val_pairs & test_pairs),
        "test_rows_with_historical_train_site_year": test_hist,
        "test_rows_total": len(test),
        "test_rows_historical_ratio": test_hist / len(test) if test else 0.0,
        "train_year_range": _range(_years(train)),
        "val_year_range": _range(_years(val)),
        "test_year_range": _range(_years(test)),
        "train_counts": _counts(train),
        "val_counts": _counts(val),
        "test_counts": _counts(test),
    }


def log_split_sanity(label: str, train: list[dict], val: list[dict], test: list[dict], split_mode: str):
    s = split_sanity_summary(train, val, test)
    print(
        f"[split_sanity:{label}] mode={split_mode} "
        f"sites train/val/test={s['train_sites']}/{s['val_sites']}/{s['test_sites']} "
        f"pairs train/val/test={s['train_pairs']}/{s['val_pairs']}/{s['test_pairs']}"
    )
    print(
        f"[split_sanity:{label}] site_intersections train∩test={s['train_test_site_intersection']} "
        f"train∩val={s['train_val_site_intersection']} val∩test={s['val_test_site_intersection']} | "
        f"pair_intersections train∩test={s['train_test_pair_intersection']} "
        f"train∩val={s['train_val_pair_intersection']} val∩test={s['val_test_pair_intersection']}"
    )
    print(
        f"[split_sanity:{label}] test_historical="
        f"{s['test_rows_with_historical_train_site_year']}/{s['test_rows_total']} "
        f"({100.0 * s['test_rows_historical_ratio']:.2f}%) "
        f"year_ranges train={s['train_year_range']} val={s['val_year_range']} test={s['test_year_range']}"
    )
    print(
        f"[split_sanity:{label}] counts "
        f"train={s['train_counts']} val={s['val_counts']} test={s['test_counts']}"
    )


def censor_type_counts(samples: list[dict]) -> dict[str, int]:
    counts = {"left": 0, "interval": 0, "right": 0}
    for s in samples:
        c = str(s.get("censor_type", ""))
        if c in counts:
            counts[c] += 1
    return counts


def _counts_to_probs(counts: dict[str, int]) -> dict[str, float]:
    total = sum(counts.values())
    if total == 0:
        return {k: 0.0 for k in counts}
    return {k: counts[k] / total for k in counts}


def split_seed_search_topk(
    samples: list[dict],
    val_frac: float,
    test_frac: float,
    seed_candidates: list[int],
    target_test_interval: int | None = None,
    tol_test_interval: int | None = None,
    topk: int = 1,
    split_mode: str = "site",
    temporal_train_end_year: int = 2018,
    temporal_val_start_year: int = 2019,
    temporal_val_end_year: int = 2020,
    temporal_test_start_year: int = 2021,
    temporal_test_end_year: int = 2022,
    val_year: int | None = None,
    test_year_min: int | None = None,
    test_year_max: int | None = None,
) -> dict:
    overall_counts = censor_type_counts(samples)
    overall_probs = _counts_to_probs(overall_counts)

    scored = []
    for seed in seed_candidates:
        train_s, val_s, test_s = split_samples(
            samples,
            val_frac=val_frac,
            test_frac=test_frac,
            seed=seed,
            split_mode=split_mode,
            temporal_train_end_year=temporal_train_end_year,
            temporal_val_start_year=temporal_val_start_year,
            temporal_val_end_year=temporal_val_end_year,
            temporal_test_start_year=temporal_test_start_year,
            temporal_test_end_year=temporal_test_end_year,
            val_year=val_year,
            test_year_min=test_year_min,
            test_year_max=test_year_max,
        )

        train_counts = censor_type_counts(train_s)
        val_counts = censor_type_counts(val_s)
        test_counts = censor_type_counts(test_s)

        train_probs = _counts_to_probs(train_counts)
        val_probs = _counts_to_probs(val_counts)
        test_probs = _counts_to_probs(test_counts)

        score = 0.0
        for t in ("left", "interval", "right"):
            score += (train_probs[t] - overall_probs[t]) ** 2
            score += (val_probs[t] - overall_probs[t]) ** 2
            score += (test_probs[t] - overall_probs[t]) ** 2

        meets_constraint = True
        if target_test_interval is not None and tol_test_interval is not None:
            test_int = test_counts.get("interval", 0)
            if abs(test_int - target_test_interval) > tol_test_interval:
                meets_constraint = False

        scored.append(
            {
                "seed": seed,
                "score": float(score),
                "split_mode": str(split_mode),
                "counts": {
                    "overall": overall_counts,
                    "train": train_counts,
                    "val": val_counts,
                    "test": test_counts,
                },
                "meets_constraint": meets_constraint,
            }
        )

    filtered = [s for s in scored if s["meets_constraint"]]
    used_fallback = False
    if not filtered:
        filtered = scored
        used_fallback = True

    filtered.sort(key=lambda x: x["score"])
    topk_list = filtered[: max(1, int(topk))]

    return {"topk": topk_list, "used_fallback": used_fallback}


def split_seed_search(
    samples: list[dict],
    val_frac: float,
    test_frac: float,
    seed_candidates: list[int],
    target_test_interval: int | None = None,
    tol_test_interval: int | None = None,
    split_mode: str = "site",
) -> dict:
    result = split_seed_search_topk(
        samples=samples,
        val_frac=val_frac,
        test_frac=test_frac,
        seed_candidates=seed_candidates,
        target_test_interval=target_test_interval,
        tol_test_interval=tol_test_interval,
        topk=1,
        split_mode=split_mode,
    )
    return {"topk": result["topk"], "used_fallback": result["used_fallback"]}


def _mask_to_recent_window(X: np.ndarray, tstar: int, window: int) -> np.ndarray:
    """
    Keep only recent [tstar-window+1, tstar] observations (1-based, inclusive).
    Equivalent 0-based slice is X[tstar-window : tstar], i.e. no future leakage.
    Everything else becomes missing: value=0, miss-indicator=1.
    Assumes miss indicators are appended after each base feature (odd dims).
    """
    T, D = X.shape
    X_out = np.zeros_like(X, dtype=np.float32)
    if D > 1:
        X_out[:, 1::2] = 1.0

    start = max(1, int(tstar) - int(window) + 1)
    end = min(T, int(tstar))
    if end >= start:
        i0 = start - 1
        i1 = end
        X_out[i0:i1, :] = X[i0:i1, :]
    return X_out


def build_stage2_nowcast_samples(
    samples: list[dict],
    window: int,
    stride: int,
    tstar_start: int | None = None,
    only_pre_event: bool = True,
    event_time_proxy: str = "r",
    require_tstar_before_L: bool = True,
) -> list[dict]:
    """
    Build Stage-2 nowcast samples.
    Output keeps season-length X (masked outside recent window) so hazard axis remains 1..T.

    Event-time handling:
      - proxy='r': event_time = R
      - proxy='mid': event_time = floor((L+R)/2)
    Interval ambiguity region L <= t* < R is retained and tracked by `case_bucket`.

    Label handling:
      - Always keep original interval [L, R] when future event exists.
    """
    out: list[dict] = []
    if not samples:
        return out
    if window <= 0:
        raise ValueError("window must be >= 1")
    if stride <= 0:
        raise ValueError("stride must be >= 1")
    if event_time_proxy not in ("r", "mid"):
        raise ValueError("event_time_proxy must be one of: r, mid")

    T = int(samples[0]["X"].shape[0])
    t0 = int(window if tstar_start is None else tstar_start)
    t0 = max(1, min(t0, T))

    for s in samples:
        X = np.asarray(s["X"], dtype=np.float32)
        ctype = str(s["censor_type"])
        has_event = ctype != "right"

        if has_event:
            L0 = int(s["L"])
            R0 = int(s["R"])
            if event_time_proxy == "mid":
                event_time = int((L0 + R0) // 2)
            else:
                event_time = int(R0)
        else:
            event_time = None

        for tstar in range(t0, T + 1, stride):
            if bool(require_tstar_before_L) and has_event and tstar >= int(s["L"]):
                continue
            if only_pre_event and has_event and event_time is not None and tstar >= event_time:
                continue

            if has_event and event_time is not None and event_time > tstar:
                L_new = int(s["L"])
                R_new = int(s["R"])
                c_new = "interval"
            else:
                # No future event after t* in season horizon.
                L_new = int(T)
                R_new = int(T)
                c_new = "right"

            rec = {
                "site_id": s["site_id"],
                "year": int(s["year"]),
                # store base X only; do masking on-the-fly in __getitem__
                "X": X,
                "L": L_new,
                "R": R_new,
                "censor_type": c_new,
                "tstar": int(tstar),
                "window": int(window),
                "event_time": int(event_time) if event_time is not None else None,
                "orig_L": int(s["L"]),
                "orig_R": int(s["R"]),
                "orig_censor_type": str(s["censor_type"]),
                "case_bucket": (
                    "right"
                    if not has_event
                    else ("pre_L" if tstar < int(s["L"]) else ("in_LR" if tstar < int(s["R"]) else "post_R"))
                ),
            }
            if "pheno_vec" in s:
                rec["pheno_vec"] = s["pheno_vec"]
            out.append(rec)

    return out


def compute_norm_stats(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Train samples only -> mean/std for each feature dim.
    Streaming reduction to avoid materializing gigantic (N,T,D) arrays.
    """
    # Optional memory logging (no extra deps)
    import os
    import resource

    mem_log = os.environ.get("RICE_MEM_LOG", "0") not in ("0", "", "false", "False")
    mem_every = int(os.environ.get("RICE_MEM_LOG_EVERY", "5000"))

    def _rss_gb() -> float:
        # ru_maxrss is KB on Linux
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0 * 1024.0)

    sum_ = None
    sumsq = None
    count = 0

    for i, s in enumerate(samples):
        x_base = np.asarray(s["X"], dtype=np.float32)
        if "tstar" in s and "window" in s:
            x = _mask_to_recent_window(x_base, tstar=int(s["tstar"]), window=int(s["window"]))
        else:
            x = x_base
        x2d = x.reshape(-1, x.shape[-1]).astype(np.float64, copy=False)
        if sum_ is None:
            sum_ = np.zeros(x2d.shape[1], dtype=np.float64)
            sumsq = np.zeros(x2d.shape[1], dtype=np.float64)
        sum_ += x2d.sum(axis=0)
        sumsq += np.square(x2d).sum(axis=0)
        count += x2d.shape[0]
        if mem_log and (i % mem_every == 0):
            print(f"[mem] compute_norm_stats i={i} rss={_rss_gb():.2f} GB")

    if sum_ is None or count == 0:
        raise ValueError("compute_norm_stats: empty samples")

    mean = sum_ / count
    var = np.maximum(sumsq / count - np.square(mean), 0.0)
    std = np.sqrt(var)
    std = np.where(std < 1e-6, 1.0, std)
    # keep missing indicators as 0/1 (no normalization)
    # missing indicators are appended after each base feature -> odd indices.
    miss_idx = np.arange(1, mean.shape[0], 2)
    if miss_idx.size > 0:
        mean[miss_idx] = 0.0
        std[miss_idx] = 1.0
    return mean.astype(np.float32), std.astype(np.float32)


class IntervalEventDataset(Dataset):
    def __init__(self, samples: list[dict], mean: np.ndarray, std: np.ndarray):
        self.samples = samples
        self.mean = mean
        self.std = std
        # Precompute missing-indicator template to avoid full zero-fill per sample.
        if samples:
            X0 = np.asarray(samples[0]["X"], dtype=np.float32)
            self._miss_template = np.zeros_like(X0, dtype=np.float32)
            if X0.shape[1] > 1:
                self._miss_template[:, 1::2] = 1.0
        else:
            self._miss_template = None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        X_base = s["X"]
        if "tstar" in s and "window" in s:
            # Reuse template to avoid full zero-fill cost each sample.
            if self._miss_template is None:
                X = _mask_to_recent_window(X_base, tstar=int(s["tstar"]), window=int(s["window"]))
            else:
                X = self._miss_template.copy()
                T = X_base.shape[0]
                tstar = int(s["tstar"])
                window = int(s["window"])
                start = max(1, tstar - window + 1)
                end = min(T, tstar)
                if end >= start:
                    i0 = start - 1
                    i1 = end
                    X[i0:i1, :] = X_base[i0:i1, :]
        else:
            X = X_base
        X = (X - self.mean) / self.std
        X = torch.from_numpy(X).float()  # (T,D)
        L = torch.tensor(int(s["L"]), dtype=torch.long)
        R = torch.tensor(int(s["R"]), dtype=torch.long)
        c = torch.tensor(CTYPE2ID[str(s["censor_type"])], dtype=torch.long)
        return X, L, R, c


def group_stage2_samples_by_site_year(samples: list[dict]) -> list[dict]:
    """
    Group flat stage-2 nowcast rows by (site_id, year), preserving t* order.
    Each group item has:
      {"site_id", "year", "samples": [row0, row1, ... sorted by tstar]}
    """
    grouped: dict[tuple[str, int], list[dict]] = {}
    for s in samples:
        key = (str(s["site_id"]), int(s["year"]))
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(s)

    out: list[dict] = []
    for (site_id, year), rows in grouped.items():
        rows_sorted = sorted(rows, key=lambda x: int(x.get("tstar", 0)))
        out.append({"site_id": site_id, "year": year, "samples": rows_sorted})
    return out


class GroupedIntervalEventDataset(Dataset):
    """
    site-year grouped dataset for hierarchical Stage-2 modeling.
    __getitem__ returns:
      X_seq:    (K,T,D)
      L_seq:    (K,)
      R_seq:    (K,)
      c_seq:    (K,)
      tstar_seq:(K,)
    where K is the number of t* rows in this site-year.
    """

    def __init__(self, groups: list[dict], mean: np.ndarray, std: np.ndarray):
        self.groups = groups
        self.mean = mean
        self.std = std
        self._miss_template_cache: dict[tuple[int, int], np.ndarray] = {}

    def __len__(self):
        return len(self.groups)

    def _masked_X(self, X_base: np.ndarray, tstar: int, window: int) -> np.ndarray:
        shape_key = (int(X_base.shape[0]), int(X_base.shape[1]))
        templ = self._miss_template_cache.get(shape_key)
        if templ is None:
            templ = np.zeros_like(X_base, dtype=np.float32)
            if X_base.shape[1] > 1:
                templ[:, 1::2] = 1.0
            self._miss_template_cache[shape_key] = templ

        X = templ.copy()
        T = int(X_base.shape[0])
        start = max(1, int(tstar) - int(window) + 1)
        end = min(T, int(tstar))
        if end >= start:
            i0 = start - 1
            i1 = end
            X[i0:i1, :] = X_base[i0:i1, :]
        return X

    def __getitem__(self, idx):
        g = self.groups[idx]
        rows = g["samples"]
        if not rows:
            raise ValueError("GroupedIntervalEventDataset: empty group encountered")

        X_seq = []
        L_seq = []
        R_seq = []
        c_seq = []
        tstar_seq = []
        for s in rows:
            X_base = np.asarray(s["X"], dtype=np.float32)
            if "tstar" in s and "window" in s:
                X = self._masked_X(X_base, tstar=int(s["tstar"]), window=int(s["window"]))
                tstar_val = int(s["tstar"])
            else:
                X = X_base
                tstar_val = int(X_base.shape[0])
            X = (X - self.mean) / self.std
            X_seq.append(X.astype(np.float32, copy=False))
            L_seq.append(int(s["L"]))
            R_seq.append(int(s["R"]))
            c_seq.append(int(CTYPE2ID[str(s["censor_type"])]))
            tstar_seq.append(tstar_val)

        X_arr = np.stack(X_seq, axis=0)  # (K,T,D)
        # pheno_vec is a site-year static vector; same across rows. Take from rows[0].
        if "pheno_vec" in rows[0]:
            pheno_vec = np.asarray(rows[0]["pheno_vec"], dtype=np.float32)
        else:
            pheno_vec = np.zeros(0, dtype=np.float32)
        return (
            torch.from_numpy(X_arr).float(),
            torch.tensor(L_seq, dtype=torch.long),
            torch.tensor(R_seq, dtype=torch.long),
            torch.tensor(c_seq, dtype=torch.long),
            torch.tensor(tstar_seq, dtype=torch.long),
            torch.from_numpy(pheno_vec).float(),
        )
