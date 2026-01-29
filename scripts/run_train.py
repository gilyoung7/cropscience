from __future__ import annotations

import copy
import random
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from configs import config as C
from src.featuresets import get_feature_cols
from src.data_pipeline import (
    load_daily,
    load_gdd_since_db,
    merge_gdd_since_db,
    add_rolling_features,
    load_obs,
    aggregate_obs_daily_max,
    make_obs_meta,
    add_site_static_latlon,
    merge_pheno_daily_ffill,
    make_daily_feature_frame,
)
from src.labels import build_interval_labels_from_doy, filter_labels_by_gap
from src.dataset import (
    build_train_frame,
    slice_season,
    build_samples_season,
    split_by_site,
    compute_norm_stats,
    IntervalEventDataset,
)
from src.model import HazardTransformer
from src.train_eval import run_epoch_weighted, eval_nll_model


def make_loader(ds, batch_size, shuffle, seed=None):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=C.NUM_WORKERS,
        pin_memory=C.PIN_MEMORY,
        persistent_workers=C.PERSISTENT_WORKERS,
        prefetch_factor=C.PREFETCH_FACTOR,
    )
    if shuffle and seed is not None:
        gen = torch.Generator().manual_seed(seed)
        return DataLoader(ds, generator=gen, **kwargs)
    return DataLoader(ds, **kwargs)

def resolve_out_path(run: int, out_root: str, out_path: str | None) -> Path:
    if out_path:
        out_path_resolved = Path(out_path)
        out_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        return out_path_resolved
    out_dir = Path(out_root) / "ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"checkpoint_run{run}.pt"


def main(run: int, out_root: str, out_path: str | None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # =========================
    # 1) DAILY: load + GDD merge + rolling
    # =========================
    daily = load_daily(C.PATH_DAILY)
    gdd_db, bad = load_gdd_since_db(C.GDD_DIR)
    if bad:
        print("GDD filename parse failed (first 5):", bad[:5])
    daily = merge_gdd_since_db(daily, gdd_db)
    daily = add_rolling_features(daily)

    # =========================
    # 2) OBS: load + aggregate
    # =========================
    obs = load_obs(C.PATH_OBS)
    obs2 = aggregate_obs_daily_max(obs)

    # =========================
    # 3) LABELS: interval/left/right + gap filter
    # =========================
    labels = build_interval_labels_from_doy(
        obs2,
        threshold=C.THRESHOLD,
        season_start_doy=C.SEASON_START_DOY,
        season_end_doy=C.SEASON_END_DOY,
    )
    labels = filter_labels_by_gap(labels, C.DOY_START, C.DOY_END, C.MAX_GAP)
    print("labels censor_type counts:\n", labels["censor_type"].value_counts())

    # =========================
    # 4) OBS META
    # =========================
    obs_meta = make_obs_meta(obs2, C.DOY_START, C.DOY_END)

    # =========================
    # 5) DAILY FEATURE FRAME + merge labels/meta/static/pheno
    # =========================
    daily_feat, _ = make_daily_feature_frame(daily)
    T = C.DOY_END - C.DOY_START + 1
    train_df = build_train_frame(daily_feat, labels, obs_meta, T=T)

    train_df = add_site_static_latlon(train_df, obs)
    train_df = merge_pheno_daily_ffill(train_df, obs)

    # =========================
    # 6) Feature set selection
    # =========================
    feature_cols = get_feature_cols(run)
    print(f"RUN={run} | D_in={len(feature_cols)}")
    # sanity: columns exist
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in train_df: {missing}")

    # =========================
    # 7) Season slice + samples
    # =========================
    df_season = slice_season(train_df, C.DOY_START, C.DOY_END)
    samples, dropped = build_samples_season(df_season, feature_cols, C.DOY_START, C.DOY_END)
    print("samples:", len(samples), "| dropped groups (len!=T):", dropped)

    # =========================
    # 8) Split + norm + datasets
    # =========================
    train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=42)
    print("split:", len(train_s), len(val_s), len(test_s), "| unique sites:", len({s["site_id"] for s in samples}))

    x_mean, x_std = compute_norm_stats(train_s)
    train_ds = IntervalEventDataset(train_s, x_mean, x_std)
    val_ds   = IntervalEventDataset(val_s,   x_mean, x_std)
    test_ds  = IntervalEventDataset(test_s,  x_mean, x_std)

    # =========================
    # 9) Multi-seed training (early stopping)
    # =========================
    trained_states = []
    for SEED in C.SEEDS:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        train_loader = make_loader(train_ds, C.BATCH_TRAIN, shuffle=True, seed=SEED)
        val_loader   = make_loader(val_ds,   C.BATCH_EVAL,  shuffle=False)
        # test_loader  = make_loader(test_ds,  C.BATCH_EVAL,  shuffle=False)  # eval script에서 사용

        torch.manual_seed(SEED)
        model = HazardTransformer(
            d_in=len(feature_cols),
            d_model=C.D_MODEL,
            nhead=C.N_HEAD,
            num_layers=C.N_LAYERS,
            dropout=C.DROPOUT,
            max_len=C.MAX_LEN,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)

        best_val = float("inf")
        best_state = None
        best_epoch = -1
        pat = 0

        for epoch in range(1, C.MAX_EPOCHS + 1):
            tr = run_epoch_weighted(model, opt, train_loader, Tend=T, device=device, train=True)
            va = eval_nll_model(model, val_loader, Tend=T, device=device)
            print(f"[seed {SEED}] epoch {epoch:02d} | train_nll {tr:.4f} | val_nll {va:.4f}")

            if va < best_val - C.MIN_DELTA:
                best_val = float(va)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                pat = 0
            else:
                pat += 1
                if pat >= C.PATIENCE:
                    break

        if best_state is None:
            raise RuntimeError(f"[seed {SEED}] best_state is None")

        trained_states.append(
            {
                "seed": SEED,
                "best_epoch": best_epoch,
                "best_val_nll": best_val,
                "state_dict": best_state,
            }
        )
        print(f"[seed {SEED}] DONE | best_epoch={best_epoch} | best_val_nll={best_val:.4f}\n")

    # =========================
    # 10) Save checkpoint bundle
    # =========================
    bundle = {
        "run": run,
        "feature_cols": feature_cols,
        "doy_start": C.DOY_START,
        "doy_end": C.DOY_END,
        "T": T,
        "norm_mean": x_mean,
        "norm_std": x_std,
        "trained_states": trained_states,
        "split_counts": {"train": len(train_s), "val": len(val_s), "test": len(test_s)},
    }
    out_path_resolved = resolve_out_path(run, out_root, out_path)
    torch.save(bundle, out_path_resolved)
    print("saved:", out_path_resolved)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--run", type=int, default=5)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--out_root", type=str, default="outputs")
    args = p.parse_args()
    main(args.run, args.out_root, args.out)
