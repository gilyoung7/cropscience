from __future__ import annotations

import random
import numpy as np
import torch
from torch.utils.data import DataLoader

from etc.configs import config as C
from etc.src.dataset import (
    build_samples_season,
    split_by_sample,
    compute_norm_stats,
    IntervalEventDataset,
    log_split_fingerprint,
)
from etc.src.model import HazardTransformer
from etc.src.train_eval import run_epoch_weighted, eval_nll_model


def make_loaders_for_features(
    train_df_season,
    trial_cols: list[str],
    Tend: int,
    doy_start: int,
    doy_end: int,
    split_seed: int,
    train_seed: int,
    log_split: bool = False,
):
    """
    trial_cols -> samples -> (train,val) split -> norm -> datasets -> loaders
    (test는 SFS에는 사용 안 함)
    """
    samples_trial, dropped, _ = build_samples_season(train_df_season, trial_cols, doy_start, doy_end)
    if dropped > 0:
        # SFS에서는 빠르게 돌리므로 경고만
        pass

    train_s, val_s, test_s = split_by_sample(samples_trial, val_frac=0.1, test_frac=0.1, seed=split_seed)
    if log_split:
        log_split_fingerprint("sfs", train_s, val_s, test_s)
    x_mean, x_std = compute_norm_stats(train_s)

    train_ds = IntervalEventDataset(train_s, x_mean, x_std)
    val_ds = IntervalEventDataset(val_s, x_mean, x_std)

    gen = torch.Generator().manual_seed(train_seed)
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
    train_seed: int = 42,
    split_seed: int = C.SPLIT_SEED,
    max_k: int = 12,
    max_epochs_fs: int = 25,
    patience_fs: int = 5,
    min_delta_fs: float = 1e-3,
    min_gain: float = 1e-4,
    elbow_patience: int = 2,
    elbow_ratio: float = 0.3,
    bad_patience: int = 1,
):
    """
    Sequential Forward Selection:
      start empty -> add best feature each step using quick early-stopping val NLL
    """
    selected: list[str] = []
    best_val = float("inf")
    history: list[dict] = []
    max_gain_so_far = None
    elbow_count = 0
    bad_count = 0

    split_logged = False
    for step in range(1, max_k + 1):
        prev_best_val = best_val
        best_feat = None
        best_val_step = best_val

        for f in candidates:
            if f in selected:
                continue

            trial = selected + [f]

            # seed 고정
            random.seed(train_seed)
            np.random.seed(train_seed)
            torch.manual_seed(train_seed)

            train_loader, val_loader = make_loaders_for_features(
                train_df_season=train_df_season,
                trial_cols=trial,
                Tend=Tend,
                doy_start=doy_start,
                doy_end=doy_end,
                split_seed=split_seed,
                train_seed=train_seed,
                log_split=not split_logged,
            )
            split_logged = True

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

        if best_feat is None:
            gain = None
        elif prev_best_val == float("inf"):
            gain = None
        else:
            gain = prev_best_val - best_val_step

        if gain is not None:
            if gain > 0:
                if max_gain_so_far is None:
                    max_gain_so_far = gain
                else:
                    max_gain_so_far = max(max_gain_so_far, gain)

            if gain <= 0:
                bad_count += 1
            else:
                bad_count = 0

            if max_gain_so_far is not None:
                if gain < max_gain_so_far * elbow_ratio:
                    elbow_count += 1
                else:
                    elbow_count = 0

        history.append(
            {
                "step": step,
                "added_feature": best_feat if best_feat is not None else "",
                "best_val_nll_after_step": float(best_val_step),
                "gain": None if gain is None else float(gain),
                "selected_features_so_far": str(selected + ([best_feat] if best_feat else [])),
            }
        )

        if best_feat is None:
            print(f"[SFS] stop at step {step} | no improvement candidate")
            break
        if gain is not None and bad_count >= bad_patience:
            print(f"[SFS] stop at step {step} | bad_count={bad_count} gain={gain:.6f}")
            break
        if gain is not None and gain < min_gain:
            print(f"[SFS] stop at step {step} | gain={gain:.6f}")
            break
        if gain is not None and elbow_count >= elbow_patience:
            print(f"[SFS] stop at step {step} | elbow_count={elbow_count} gain={gain:.6f}")
            break

        selected.append(best_feat)
        best_val = best_val_step
        if gain is None:
            print(f"[SFS] step {step:02d}: +{best_feat:25s} val_nll={best_val:.5f} gain=NA")
        else:
            print(f"[SFS] step {step:02d}: +{best_feat:25s} val_nll={best_val:.5f} gain={gain:.6f}")

    return selected, best_val, history
