from __future__ import annotations

import random
import numpy as np
import torch

from apple.configs import config as C
from apple.src.sfs import make_loaders_for_features
from apple.src.dataset import build_samples_season, split_by_site, log_split_fingerprint
from apple.src.model import HazardTransformer
from apple.src.train_eval import run_epoch_weighted, eval_nll_model


def train_trial_val_nll(
    train_df_season,
    trial_cols: list[str],
    Tend: int,
    doy_start: int,
    doy_end: int,
    device,
    train_seed: int,
    split_seed: int,
    max_epochs_fs: int = 25,
    patience_fs: int = 5,
    min_delta_fs: float = 1e-3,
):
    """
    Quick train + early stop -> best val NLL for given trial_cols.
    Returns best_local_val_nll
    """
    random.seed(train_seed)
    np.random.seed(train_seed)
    torch.manual_seed(train_seed)

    train_loader, val_loader = make_loaders_for_features(
        train_df_season=train_df_season,
        trial_cols=trial_cols,
        Tend=Tend,
        doy_start=doy_start,
        doy_end=doy_end,
        split_seed=split_seed,
        train_seed=train_seed,
    )

    model = HazardTransformer(
        d_in=len(trial_cols),
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

    return best_local


def backward_elimination(
    train_df_season,
    start_cols: list[str],
    Tend: int,
    doy_start: int,
    doy_end: int,
    device,
    train_seed: int = 0,
    split_seed: int = C.SPLIT_SEED,
    min_k: int = 12,
    max_drop: float = 0.002,
    max_epochs_fs: int = 25,
    patience_fs: int = 5,
    min_delta_fs: float = 1e-3,
    protected: list[str] | None = None,
):
    """
    Greedy backward elimination:
    - Start from start_cols
    - At each step, try removing each removable feature (not protected)
    - Pick removal that gives the lowest val_nll (or smallest increase)
    - Stop when:
        * len(cols) <= min_k, OR
        * best removal worsens val_nll by more than max_drop (val_nll increases too much), OR
        * no removable feature
    Returns:
        final_cols, history(list of dict)
    """
    if protected is None:
        protected = []

    cols = list(start_cols)
    protected_set = set(protected)

    # sanity: protect must exist
    protected_set = {c for c in protected_set if c in cols}

    # baseline with all features
    base_val = train_trial_val_nll(
        train_df_season,
        cols,
        Tend,
        doy_start,
        doy_end,
        device,
        train_seed=train_seed,
        split_seed=split_seed,
        max_epochs_fs=max_epochs_fs, patience_fs=patience_fs, min_delta_fs=min_delta_fs
    )

    samples_full, _ = build_samples_season(train_df_season, cols, doy_start, doy_end)
    train_s, val_s, test_s = split_by_site(samples_full, val_frac=0.1, test_frac=0.1, seed=split_seed)
    log_split_fingerprint("backward", train_s, val_s, test_s)

    history = []
    history.append({
        "step": 0,
        "action": "baseline",
        "removed": "",
        "n_features": len(cols),
        "val_nll": base_val,
        "delta_from_prev": 0.0,
        "features": "|".join(cols),
    })

    prev_best = base_val
    step = 0

    while len(cols) > min_k:
        step += 1

        removable = [c for c in cols if c not in protected_set]
        if not removable:
            history.append({
                "step": step,
                "action": "stop",
                "removed": "",
                "n_features": len(cols),
                "val_nll": prev_best,
                "delta_from_prev": 0.0,
                "features": "|".join(cols),
                "reason": "no removable features (all protected)",
            })
            break

        best_remove = None
        best_val = float("inf")
        initial_scores = []

        for r in removable:
            trial = [c for c in cols if c != r]
            va = train_trial_val_nll(
                train_df_season,
                trial,
                Tend,
                doy_start,
                doy_end,
                device,
                train_seed=train_seed,
                split_seed=split_seed,
                max_epochs_fs=max_epochs_fs, patience_fs=patience_fs, min_delta_fs=min_delta_fs
            )
            initial_scores.append((r, float(va)))

        initial_scores.sort(key=lambda x: x[1])
        top_k = [r for r, _ in initial_scores[: min(3, len(initial_scores))]]

        for r in top_k:
            trial = [c for c in cols if c != r]
            vals = []
            for s in C.SEEDS:
                va = train_trial_val_nll(
                    train_df_season,
                    trial,
                    Tend,
                    doy_start,
                    doy_end,
                    device,
                    train_seed=s,
                    split_seed=split_seed,
                    max_epochs_fs=max_epochs_fs,
                    patience_fs=patience_fs,
                    min_delta_fs=min_delta_fs,
                )
                vals.append(float(va))
            va_mean = sum(vals) / len(vals)
            if va_mean < best_val:
                best_val = va_mean
                best_remove = r

        delta = best_val - prev_best  # +면 악화, -면 개선

        # accept removal only if not too much worse
        if delta > max_drop:
            history.append({
                "step": step,
                "action": "stop",
                "removed": "",
                "n_features": len(cols),
                "val_nll": prev_best,
                "delta_from_prev": float(delta),
                "features": "|".join(cols),
                "reason": f"best removal worsens too much: delta={delta:.6f} > max_drop={max_drop}",
            })
            break

        # commit removal
        cols = [c for c in cols if c != best_remove]
        prev_best = best_val

        history.append({
            "step": step,
            "action": "remove",
            "removed": best_remove,
            "n_features": len(cols),
            "val_nll": float(best_val),
            "delta_from_prev": float(delta),
            "features": "|".join(cols),
        })

    return cols, history
