from __future__ import annotations

from dataclasses import dataclass
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
) -> tuple[list[dict], int]:
    """
    Returns (samples, dropped_groups)
    samples item: {"site_id","year","X","L","R","censor_type"}
      - X: (T,D) float32
      - L,R: season coordinates in 1..T
    """
    T = doy_end - doy_start + 1
    samples: list[dict] = []
    dropped = 0

    for (site, year), sub in df_season.groupby(["site_id", "year"], sort=False):
        sub = sub.sort_values("doy")

        # 시즌 구간이 정확히 T개인지 확인 (빠진 day 있으면 drop)
        if len(sub) != T:
            dropped += 1
            continue

        X = sub[feature_cols].to_numpy(dtype=np.float32)

        L = int(sub["L_doy"].iloc[0]) - doy_start + 1
        R = int(sub["R_doy"].iloc[0]) - doy_start + 1
        ctype = str(sub["censor_type"].iloc[0])

        # clamp to [1, T]
        L = min(max(L, 1), T)
        R = min(max(R, 1), T)

        samples.append({"site_id": site, "year": int(year), "X": X, "L": L, "R": R, "censor_type": ctype})

    return samples, dropped


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


def compute_norm_stats(samples: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Train samples only -> mean/std for each feature dim
    """
    X_all = np.concatenate([s["X"][None, :, :] for s in samples], axis=0)  # (N,T,D)
    mean = X_all.reshape(-1, X_all.shape[-1]).mean(axis=0)
    std = X_all.reshape(-1, X_all.shape[-1]).std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
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
