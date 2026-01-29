from __future__ import annotations

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
from src.train_eval import eval_nll_model, eval_metrics_with_overlap


def make_loader(ds, batch_size, shuffle):
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,
        num_workers=C.NUM_WORKERS,
        pin_memory=C.PIN_MEMORY,
        persistent_workers=C.PERSISTENT_WORKERS,
        prefetch_factor=C.PREFETCH_FACTOR,
    )


def rebuild_datasets(run: int):
    """
    Rebuild datasets deterministically (same as training pipeline split seed=42).
    Norm stats are recomputed from train split.
    """
    # DAILY
    daily = load_daily(C.PATH_DAILY)
    gdd_db, _ = load_gdd_since_db(C.GDD_DIR)
    daily = merge_gdd_since_db(daily, gdd_db)
    daily = add_rolling_features(daily)

    # OBS
    obs = load_obs(C.PATH_OBS)
    obs2 = aggregate_obs_daily_max(obs)

    # LABELS
    labels = build_interval_labels_from_doy(
        obs2, threshold=C.THRESHOLD, season_start_doy=C.SEASON_START_DOY, season_end_doy=C.SEASON_END_DOY
    )
    labels = filter_labels_by_gap(labels, C.DOY_START, C.DOY_END, C.MAX_GAP)

    # META
    obs_meta = make_obs_meta(obs2, C.DOY_START, C.DOY_END)

    # FRAME
    daily_feat, _ = make_daily_feature_frame(daily)
    T = C.DOY_END - C.DOY_START + 1
    train_df = build_train_frame(daily_feat, labels, obs_meta, T=T)
    train_df = add_site_static_latlon(train_df, obs)
    train_df = merge_pheno_daily_ffill(train_df, obs)

    feature_cols = get_feature_cols(run)

    df_season = slice_season(train_df, C.DOY_START, C.DOY_END)
    samples, dropped = build_samples_season(df_season, feature_cols, C.DOY_START, C.DOY_END)
    if dropped > 0:
        print("WARNING: dropped groups (len!=T):", dropped)

    train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=42)
    x_mean, x_std = compute_norm_stats(train_s)

    train_ds = IntervalEventDataset(train_s, x_mean, x_std)
    val_ds   = IntervalEventDataset(val_s,   x_mean, x_std)
    test_ds  = IntervalEventDataset(test_s,  x_mean, x_std)

    return feature_cols, T, train_ds, val_ds, test_ds


def mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def resolve_ckpt_path(run: int, out_root: str, ckpt_path: str | None) -> Path:
    if ckpt_path:
        return Path(ckpt_path)
    return Path(out_root) / "ckpt" / f"checkpoint_run{run}.pt"


def resolve_out_csv(run: int, out_root: str, out_csv: str | None) -> str | None:
    if out_csv is None:
        out_dir = Path(out_root) / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / f"eval_run{run}.csv")
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    return out_csv


def main(ckpt_path: str | None, run: int, out_csv: str | None, out_root: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt_path_resolved = resolve_ckpt_path(run, out_root, ckpt_path)
    ckpt = torch.load(ckpt_path_resolved, map_location="cpu")
    trained_states = ckpt["trained_states"]
    print("loaded ckpt:", ckpt_path_resolved, "| seeds:", [d["seed"] for d in trained_states])

    feature_cols, T, train_ds, val_ds, test_ds = rebuild_datasets(run)
    print(f"RUN={run} | D_in={len(feature_cols)} | T={T}")

    val_loader  = make_loader(val_ds,  C.BATCH_EVAL, shuffle=False)
    test_loader = make_loader(test_ds, C.BATCH_EVAL, shuffle=False)

    records = []
    for d in trained_states:
        seed = int(d["seed"])

        model = HazardTransformer(
            d_in=len(feature_cols),
            d_model=C.D_MODEL,
            nhead=C.N_HEAD,
            num_layers=C.N_LAYERS,
            dropout=C.DROPOUT,
            max_len=C.MAX_LEN,
        ).to(device)
        model.load_state_dict(d["state_dict"])
        model.eval()

        val_nll  = float(eval_nll_model(model, val_loader, Tend=T, device=device))
        test_nll = float(eval_nll_model(model, test_loader, Tend=T, device=device))

        val_stats  = eval_metrics_with_overlap(model, val_loader, Tend=T, device=device, alpha=0.2)
        test_stats = eval_metrics_with_overlap(model, test_loader, Tend=T, device=device, alpha=0.2)

        rec = {
            "seed": seed,
            "best_epoch": int(d["best_epoch"]),
            "best_val_nll": float(d["best_val_nll"]),
            "VAL_NLL": val_nll,
            "TEST_NLL": test_nll,
            "TEST_IoU80(interval_only)": float(test_stats["IoU_mean_interval_only(80%)"]),
            "TEST_Prec80(interval_only)": float(test_stats["Precision_mean_interval_only(80%)"]),
            "TEST_Rec80(interval_only)": float(test_stats["Recall_mean_interval_only(80%)"]),
            "TEST_MAE_int(interval_only)": float(test_stats["mae_mid_mean_interval_only"]),
            "TEST_Mass_int(interval_only)": float(test_stats["mass_in_interval_mean_interval_only"]),
            "TEST_N_int": int(test_stats["N_interval_samples"]),
        }
        records.append(rec)

    df = pd.DataFrame(records).sort_values("seed").reset_index(drop=True)
    print("\n=== per-seed ===")
    print(df.to_string(index=False))

    cols = [
        "VAL_NLL",
        "TEST_NLL",
        "TEST_IoU80(interval_only)",
        "TEST_Prec80(interval_only)",
        "TEST_Rec80(interval_only)",
        "TEST_MAE_int(interval_only)",
        "TEST_Mass_int(interval_only)",
    ]
    print("\n=== mean ± std (seeds) ===")
    for col in cols:
        m, sd = mean_std(df[col].to_numpy())
        print(f"{col:>26s}: {m:.4f} ± {sd:.4f}")
    print("\nTEST interval-only N:", sorted(df["TEST_N_int"].unique().tolist()))

    out_csv_resolved = resolve_out_csv(run, out_root, out_csv)
    if out_csv_resolved:
        df.to_csv(out_csv_resolved, index=False)
        print("saved:", out_csv_resolved)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--run", type=int, default=5)
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--out_root", type=str, default="outputs")
    args = p.parse_args()
    main(args.ckpt, args.run, args.out_csv, args.out_root)
