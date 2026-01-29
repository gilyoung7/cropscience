from __future__ import annotations

import argparse
from pathlib import Path
import torch

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
from src.dataset import build_train_frame, slice_season
from src.sfs import sfs_topk


def rebuild_train_df_season():
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

    # SEASON
    train_df_season = slice_season(train_df, C.DOY_START, C.DOY_END)
    return train_df_season, T


def resolve_topk_path(run: int, out_root: str, topk_path: str | None) -> Path:
    if topk_path:
        return Path(topk_path)
    return Path(out_root) / "pi" / f"pi_run{run}_topk.txt"


def resolve_out_txt(run: int, out_root: str, out_txt: str | None) -> str | None:
    if out_txt is None:
        out_dir = Path(out_root) / "sfs"
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / f"sfs_run{run}_best.txt")
    if out_txt:
        Path(out_txt).parent.mkdir(parents=True, exist_ok=True)
    return out_txt


def main(topk_path: str | None, seed: int, max_k: int, out_txt: str | None, run: int, out_root: str):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_df_season, T = rebuild_train_df_season()

    # candidates from file (TOPK from PI)
    topk_path_resolved = resolve_topk_path(run, out_root, topk_path)
    with open(topk_path_resolved, "r", encoding="utf-8") as f:
        candidates = [line.strip() for line in f if line.strip()]
    print("num candidates:", len(candidates))
    print("candidates:", candidates)

    best_feats, best_val = sfs_topk(
        train_df_season=train_df_season,
        candidates=candidates,
        Tend=T,
        doy_start=C.DOY_START,
        doy_end=C.DOY_END,
        device=device,
        seed=seed,
        max_k=max_k,
    )

    print("\nBEST_FEATS =", best_feats)
    print("BEST_VAL_NLL =", best_val)

    out_txt_resolved = resolve_out_txt(run, out_root, out_txt)
    if out_txt_resolved:
        with open(out_txt_resolved, "w", encoding="utf-8") as f:
            for x in best_feats:
                f.write(x + "\n")
        print("saved:", out_txt_resolved)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--topk_path", type=str, default=None)
    p.add_argument("--run", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--max_k", type=int, default=12)
    p.add_argument("--out_txt", type=str, default=None)
    p.add_argument("--out_root", type=str, default="outputs")
    args = p.parse_args()
    main(args.topk_path, args.seed, args.max_k, args.out_txt, args.run, args.out_root)
