from __future__ import annotations

import argparse
from pathlib import Path
import re
import torch
import pandas as pd

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


def resolve_steps_csv(run: int, out_root: str, steps_csv: str | None) -> str:
    if steps_csv:
        Path(steps_csv).parent.mkdir(parents=True, exist_ok=True)
        return steps_csv
    out_dir = Path(out_root) / "sfs"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"sfs_run{run}_steps.csv")


def infer_run_from_path(path: Path) -> int | None:
    match = re.search(r"run(\d+)", path.name)
    if not match:
        return None
    return int(match.group(1))


def check_topk_run_match(run: int, topk_path: Path, used_default_path: bool):
    if not used_default_path:
        return
    match = re.search(r"run(\d+)", topk_path.name)
    if not match:
        return
    file_run = int(match.group(1))
    if file_run != run:
        msg = (
            f"Run mismatch: --run={run} but topk file looks like run{file_run} ({topk_path}). "
            f"Pass --run {file_run} or set --topk_path explicitly.\n"
            f"Example: python -m scripts.run_sfs --run {file_run} --topk_path {topk_path}"
        )
        raise ValueError(msg)


def main(
    topk_path: str | None,
    seed: int,
    max_k: int,
    out_txt: str | None,
    run: int,
    out_root: str,
    min_gain: float,
    elbow_patience: int,
    elbow_ratio: float,
    bad_patience: int,
    steps_csv: str | None,
    split_seed: int,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_df_season, T = rebuild_train_df_season()

    # candidates from file (TOPK from PI)
    used_default_path = topk_path is None
    topk_path_resolved = resolve_topk_path(run, out_root, topk_path)
    check_topk_run_match(run, topk_path_resolved, used_default_path)
    with open(topk_path_resolved, "r", encoding="utf-8") as f:
        candidates = [line.strip() for line in f if line.strip()]
    print("num candidates:", len(candidates))
    print("candidates:", candidates)

    best_feats, best_val, history = sfs_topk(
        train_df_season=train_df_season,
        candidates=candidates,
        Tend=T,
        doy_start=C.DOY_START,
        doy_end=C.DOY_END,
        device=device,
        train_seed=seed,
        split_seed=split_seed,
        max_k=max_k,
        min_gain=min_gain,
        elbow_patience=elbow_patience,
        elbow_ratio=elbow_ratio,
        bad_patience=bad_patience,
    )

    print("\nBEST_FEATS =", best_feats)
    print("BEST_VAL_NLL =", best_val)

    out_txt_resolved = resolve_out_txt(run, out_root, out_txt)
    if out_txt_resolved:
        with open(out_txt_resolved, "w", encoding="utf-8") as f:
            for x in best_feats:
                f.write(x + "\n")
        print("saved:", out_txt_resolved)

    steps_csv_resolved = resolve_steps_csv(run, out_root, steps_csv)
    if history:
        df_steps = pd.DataFrame(history)
        df_steps.to_csv(steps_csv_resolved, index=False)
        print("saved:", steps_csv_resolved)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--topk_path", type=str, default=None)
    p.add_argument("--run", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--split_seed", type=int, default=C.SPLIT_SEED)
    p.add_argument("--max_k", type=int, default=12)
    p.add_argument("--min_gain", type=float, default=1e-4)
    p.add_argument("--elbow_patience", type=int, default=2)
    p.add_argument("--elbow_ratio", type=float, default=0.3)
    p.add_argument("--bad_patience", type=int, default=1)
    p.add_argument("--out_txt", type=str, default=None)
    p.add_argument("--out_root", type=str, default="outputs")
    p.add_argument("--steps_csv", type=str, default=None)
    args = p.parse_args()
    if args.topk_path is not None and args.run == 5:
        inferred_run = infer_run_from_path(Path(args.topk_path))
        if inferred_run is not None and inferred_run != args.run:
            print(
                f"WARNING: --run defaulted to 5 but topk_path looks like run{inferred_run}. "
                f"Using run={inferred_run} for outputs. To override, pass --run explicitly."
            )
            args.run = inferred_run
        elif inferred_run is None:
            print("WARNING: --run defaulted to 5 and topk_path has no run pattern; outputs will use run=5.")
    main(
        args.topk_path,
        args.seed,
        args.max_k,
        args.out_txt,
        args.run,
        args.out_root,
        args.min_gain,
        args.elbow_patience,
        args.elbow_ratio,
        args.bad_patience,
        args.steps_csv,
        args.split_seed,
    )
