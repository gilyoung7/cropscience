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
) -> tuple[list[dict], int, list[str]]:
    """
    Returns (samples, dropped_groups, feature_names)
    samples item: {"site_id","year","X","L","R","censor_type"}
      - X: (T,D) float32
      - L,R: season coordinates in 1..T
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

        samples.append({"site_id": site, "year": int(year), "X": X, "L": L, "R": R, "censor_type": ctype})

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
) -> dict:
    overall_counts = censor_type_counts(samples)
    overall_probs = _counts_to_probs(overall_counts)

    scored = []
    for seed in seed_candidates:
        train_s, val_s, test_s = split_by_site(samples, val_frac=val_frac, test_frac=test_frac, seed=seed)

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
) -> dict:
    result = split_seed_search_topk(
        samples=samples,
        val_frac=val_frac,
        test_frac=test_frac,
        seed_candidates=seed_candidates,
        target_test_interval=target_test_interval,
        tol_test_interval=tol_test_interval,
        topk=1,
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
) -> list[dict]:
    """
    Build Stage-2 nowcast samples for "first event after t*" target.
    Output keeps season-length X (masked outside recent window) so hazard axis remains 1..T.

    Event-time handling:
      - proxy='r': event_time = R
      - proxy='mid': event_time = floor((L+R)/2)
    Interval ambiguity region L <= t* < R is retained and tracked by `case_bucket`.
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
            if only_pre_event and has_event and event_time is not None and tstar >= event_time:
                continue

            X_now = _mask_to_recent_window(X, tstar=tstar, window=window)

            if has_event and event_time is not None and event_time > tstar:
                # Convert to a point-like interval at first-future event proxy.
                R_new = int(event_time)
                L_new = int(max(1, R_new - 1))
                c_new = "interval"
            else:
                # No future event after t* in season horizon.
                L_new = int(T)
                R_new = int(T)
                c_new = "right"

            out.append(
                {
                    "site_id": s["site_id"],
                    "year": int(s["year"]),
                    "X": X_now.astype(np.float32, copy=False),
                    "L": L_new,
                    "R": R_new,
                    "censor_type": c_new,
                    "tstar": int(tstar),
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
            )

    return out


def compute_norm_stats(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Train samples only -> mean/std for each feature dim
    """
    X_all = np.concatenate([s["X"][None, :, :] for s in samples], axis=0)  # (N,T,D)
    mean = X_all.reshape(-1, X_all.shape[-1]).mean(axis=0)
    std = X_all.reshape(-1, X_all.shape[-1]).std(axis=0)
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

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        X = (s["X"] - self.mean) / self.std
        X = torch.from_numpy(X).float()  # (T,D)
        L = torch.tensor(int(s["L"]), dtype=torch.long)
        R = torch.tensor(int(s["R"]), dtype=torch.long)
        c = torch.tensor(CTYPE2ID[str(s["censor_type"])], dtype=torch.long)
        return X, L, R, c
