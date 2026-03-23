from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import torch

from etc.configs import config as C
from etc.src.pest_resolver import resolve_pest, default_out_root, ensure_output_dirs
from etc.src.data_pipeline import (
    load_daily_preprocessed,
    load_obs,
    aggregate_obs_daily_max,
    make_obs_meta,
    add_site_static_latlon,
    merge_pheno_daily_ffill,
    make_daily_feature_frame,
)
from etc.src.labels import build_interval_labels_from_doy, filter_labels_by_gap
from etc.src.dataset import build_train_frame, slice_season
from etc.src.backward import backward_elimination


def rebuild_train_df_season():
    daily = load_daily_preprocessed(C.PATH_DAILY)

    obs = load_obs(C.PATH_OBS)
    obs2 = aggregate_obs_daily_max(obs)

    labels = build_interval_labels_from_doy(
        obs2, threshold=C.THRESHOLD, season_start_doy=C.SEASON_START_DOY, season_end_doy=C.SEASON_END_DOY
    )
    labels = filter_labels_by_gap(labels, C.DOY_START, C.DOY_END, C.MAX_GAP)

    obs_meta = make_obs_meta(obs2, C.DOY_START, C.DOY_END)

    daily_feat, _ = make_daily_feature_frame(daily)
    T = C.DOY_END - C.DOY_START + 1

    train_df = build_train_frame(daily_feat, labels, obs_meta, T=T)
    train_df = add_site_static_latlon(train_df, obs)
    train_df = merge_pheno_daily_ffill(train_df, obs)

    train_df_season = slice_season(train_df, C.DOY_START, C.DOY_END)
    return train_df_season, T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, required=True)
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--split_seed", type=int, default=C.SPLIT_SEED)

    p.add_argument("--min_k", type=int, default=12)
    p.add_argument("--max_drop", type=float, default=0.002)

    # quick-train params
    p.add_argument("--max_epochs_fs", type=int, default=25)
    p.add_argument("--patience_fs", type=int, default=5)
    p.add_argument("--min_delta_fs", type=float, default=1e-3)

    # protect some features from removal (repeatable)
    p.add_argument("--protect", nargs="*", default=[])

    args = p.parse_args()
    _, get_feature_cols = resolve_pest(args.pest)
    if not args.out_root:
        args.out_root = default_out_root(args.pest)
    ensure_output_dirs(args.out_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    out_root = Path(args.out_root)
    (out_root / "backward").mkdir(parents=True, exist_ok=True)

    # start cols from run feature set
    start_cols = get_feature_cols(args.run)
    print(f"RUN={args.run} | start D_in={len(start_cols)}")
    if args.protect:
        print("protected:", args.protect)

    train_df_season, T = rebuild_train_df_season()

    final_cols, history = backward_elimination(
        train_df_season=train_df_season,
        start_cols=start_cols,
        Tend=T,
        doy_start=C.DOY_START,
        doy_end=C.DOY_END,
        device=device,
        train_seed=args.seed,
        split_seed=args.split_seed,
        min_k=args.min_k,
        max_drop=args.max_drop,
        max_epochs_fs=args.max_epochs_fs,
        patience_fs=args.patience_fs,
        min_delta_fs=args.min_delta_fs,
        protected=args.protect,
    )

    print("\nFINAL_COLS (n=%d):" % len(final_cols))
    print(final_cols)

    # save
    out_txt = out_root / "backward" / f"backward_run{args.run}_best.txt"
    out_csv = out_root / "backward" / f"backward_run{args.run}_steps.csv"

    with open(out_txt, "w", encoding="utf-8") as f:
        for c in final_cols:
            f.write(c + "\n")
    df = pd.DataFrame(history)
    df.to_csv(out_csv, index=False)

    print("saved:", out_txt)
    print("saved:", out_csv)


if __name__ == "__main__":
    main()
