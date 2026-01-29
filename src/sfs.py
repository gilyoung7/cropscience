from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from configs import config as C
from src.dataset import (
    build_samples_season,
    split_by_site,
    compute_norm_stats,
    IntervalEventDataset,
)
from src.model import HazardTransformer
from src.train_eval import run_epoch_weighted, eval_nll_model


def make_loaders_for_features(
    train_df_season,
    trial_cols: list[str],
    Tend: int,
    doy_start: int,
    doy_end: int,
    seed: int,
):
    """
    trial_cols -> samples -> (train,val) split -> norm -> datasets -> loaders
    (test는 SFS에는 사용 안 함)
    """
    samples_trial, dropped = build_samples_season(train_df_season, trial_cols, doy_start, doy_end)
    if dropped > 0:
        # SFS에서는 빠르게 돌리므로 경고만
        pass

    train_s, val_s, _ = split_by_site(samples_trial, val_frac=0.1, test_frac=0.1, seed=seed)
    x_mean, x_std = compute_norm_stats(train_s)

    train_ds = IntervalEventDataset(train_s, x_mean, x_std)
    val_ds = IntervalEventDataset(val_s, x_mean, x_std)

    gen = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=C.BATCH_TRAIN,
        shuffle=True,
        drop_last=False,
        generator=gen,
        num_workers=C.NUM_WORKERS,
        pin_memory=C.PIN_MEMORY,
        persistent_workers=C.PERSISTENT_WORKERS,
        prefetch_factor=C.PREFETCH_FACTOR,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=C.BATCH_EVAL,
        shuffle=False,
        drop_last=False,
        num_workers=C.NUM_WORKERS,
        pin_memory=C.PIN_MEMORY,
        persistent_workers=C.PERSISTENT_WORKERS,
        prefetch_factor=C.PREFETCH_FACTOR,
    )
    return train_loader, val_loader


def sfs_topk(
    train_df_season,
    candidates: list[str],
    Tend: int,
    doy_start: int,
    doy_end: int,
    device,
    seed: int = 0,
    max_k: int = 12,
    max_epochs_fs: int = 25,
    patience_fs: int = 5,
    min_delta_fs: float = 1e-3,
    min_gain: float = 1e-4,
):
    """
    Sequential Forward Selection:
      start empty -> add best feature each step using quick early-stopping val NLL
    """
    selected: list[str] = []
    best_val = float("inf")

    for step in range(1, max_k + 1):
        best_feat = None
        best_val_step = best_val

        for f in candidates:
            if f in selected:
                continue

            trial = selected + [f]

            # seed 고정
            random.seed(seed)
            np.random.seed(seed)
            torch.manual_seed(seed)

            train_loader, val_loader = make_loaders_for_features(
                train_df_season=train_df_season,
                trial_cols=trial,
                Tend=Tend,
                doy_start=doy_start,
                doy_end=doy_end,
                seed=seed,
            )

            model = HazardTransformer(
                d_in=len(trial),
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=C.N_LAYERS,
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
            ).to(device)
            opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)

            best_local = float("inf")
            pat = 0

            for _epoch in range(1, max_epochs_fs + 1):
                _ = run_epoch_weighted(model, opt, train_loader, Tend=Tend, device=device, train=True)
                va = eval_nll_model(model, val_loader, Tend=Tend, device=device)

                if va < best_local - min_delta_fs:
                    best_local = float(va)
                    pat = 0
                else:
                    pat += 1
                    if pat >= patience_fs:
                        break

            if best_local < best_val_step:
                best_val_step = best_local
                best_feat = f

        gain = best_val - best_val_step
        if best_feat is None or gain < min_gain:
            print(f"[SFS] stop at step {step} | gain={gain:.6f}")
            break

        selected.append(best_feat)
        best_val = best_val_step
        print(f"[SFS] step {step:02d}: +{best_feat:25s} val_nll={best_val:.5f} gain={gain:.6f}")

    return selected, best_val
