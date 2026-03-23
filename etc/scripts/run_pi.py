from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from etc.configs import config as C
from etc.src.pest_resolver import resolve_pest, default_out_root, ensure_output_dirs
from etc.scripts.common import make_loader
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
from etc.src.dataset import (
    build_train_frame,
    slice_season,
    build_samples_season,
    split_by_sample,
    compute_norm_stats,
    IntervalEventDataset,
    log_split_fingerprint,
)
from etc.src.ckpt_schema import validate_ckpt_meta
from etc.src.model import HazardTransformer
from etc.src.interpret import permutation_importance_features, summarize_importance


def rebuild_val_dataset(run: int, get_feature_cols):
    # DAILY
    daily = load_daily_preprocessed(C.PATH_DAILY)

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
    samples, dropped, feature_names = build_samples_season(df_season, feature_cols, C.DOY_START, C.DOY_END)
    if dropped > 0:
        print("WARNING: dropped groups (len!=T):", dropped)

    # same split as train/eval
    train_s, val_s, test_s = split_by_sample(samples, val_frac=0.1, test_frac=0.1, seed=C.SPLIT_SEED)
    log_split_fingerprint("pi", train_s, val_s, test_s)
    x_mean, x_std = compute_norm_stats(train_s)

    val_ds = IntervalEventDataset(val_s, x_mean, x_std)
    return feature_cols, feature_names, T, val_ds


def resolve_ckpt_path(run: int, out_root: str, ckpt_path: str | None) -> Path:
    if ckpt_path:
        return Path(ckpt_path)
    return Path(out_root) / "ckpt" / f"checkpoint_run{run}.pt"


def resolve_out_prefix(run: int, out_root: str, out_prefix: str | None) -> Path:
    if out_prefix:
        out_prefix_path = Path(out_prefix)
        out_prefix_path.parent.mkdir(parents=True, exist_ok=True)
        return out_prefix_path
    out_dir = Path(out_root) / "pi"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"pi_run{run}"


def check_run_match(ckpt_run: int, cli_run: int, allow_run_mismatch: bool, ckpt_path: Path):
    if ckpt_run != cli_run:
        msg = (
            f"Run mismatch: ckpt has run={ckpt_run} but CLI --run={cli_run}. "
            f"Pass --run {ckpt_run} or specify --ckpt correctly.\n"
            f"Example: python -m scripts.run_pi --run {ckpt_run} --ckpt {ckpt_path}"
        )
        if allow_run_mismatch:
            print("WARNING:", msg)
        else:
            raise ValueError(msg)


def main(
    pest: str,
    ckpt_path: str | None,
    run: int,
    n_repeats: int,
    topk: int,
    out_prefix: str | None,
    out_root: str,
    seed_only: int | None,
    allow_run_mismatch: bool,
):
    _, get_feature_cols = resolve_pest(pest)
    if not out_root:
        out_root = default_out_root(pest)
    ensure_output_dirs(out_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt_path_resolved = resolve_ckpt_path(run, out_root, ckpt_path)
    ckpt = torch.load(ckpt_path_resolved, map_location="cpu")
    ckpt_run = int(ckpt.get("run", -1))
    check_run_match(ckpt_run, run, allow_run_mismatch, ckpt_path_resolved)
    states = ckpt["trained_states"]
    print("loaded ckpt:", ckpt_path_resolved, "| seeds:", [d["seed"] for d in states])

    feature_cols, feature_names, T, val_ds = rebuild_val_dataset(run, get_feature_cols)
    print(f"[features] n={len(feature_names)} head={feature_names[:5]} tail={feature_names[-5:]}")
    val_loader = make_loader(val_ds, C.BATCH_EVAL, shuffle=False)

    X0, *_ = next(iter(val_loader))
    D_in = int(X0.shape[-1])
    print(f"[D_in] computed_from_dataset={D_in}")
    validate_ckpt_meta(
        ckpt,
        pest=pest,
        run=run,
        d_in=D_in,
        feature_names=feature_names,
        allow_run_mismatch=allow_run_mismatch,
    )

    all_imps = []
    records = []

    for d in states:
        seed = int(d["seed"])
        if seed_only is not None and seed != seed_only:
            continue

        model = HazardTransformer(
            d_in=D_in,
            d_model=C.D_MODEL,
            nhead=C.N_HEAD,
            num_layers=C.N_LAYERS,
            dropout=C.DROPOUT,
            max_len=C.MAX_LEN,
        ).to(device)
        model.load_state_dict(d["state_dict"])
        model.eval()

        base, imps, ranked = permutation_importance_features(
            model,
            val_loader,
            Tend=T,
            device=device,
            feature_names=feature_names,
            n_repeats=n_repeats,
            seed=seed,
        )

        all_imps.append(imps)
        print(f"\n[seed {seed}] base_val_nll={base:.4f} TOP10:")
        for name, imp in ranked[:10]:
            print(f"  {name:30s}  ΔNLL={imp:.6f}")

        for name, imp in ranked:
            records.append({"seed": seed, "feature": name, "delta_nll": float(imp), "base_val_nll": float(base)})

    out_prefix_path = resolve_out_prefix(run, out_root, out_prefix)
    if out_prefix_path.suffix:
        out_prefix_path = out_prefix_path.with_suffix("")

    df_long = pd.DataFrame(records)
    out_long = out_prefix_path.with_name(out_prefix_path.name + "_long.csv")
    df_long.to_csv(out_long, index=False)
    print("saved:", out_long)

    ranked_mean, top_names = summarize_importance(all_imps, feature_names, topk=topk)
    df_mean = pd.DataFrame([{"feature": n, "mean_delta_nll": float(v)} for n, v in ranked_mean])
    out_mean = out_prefix_path.with_name(out_prefix_path.name + "_mean.csv")
    df_mean.to_csv(out_mean, index=False)
    print("saved:", out_mean)

    out_topk = out_prefix_path.with_name(out_prefix_path.name + "_topk.txt")
    with open(out_topk, "w", encoding="utf-8") as f:
        for n in top_names:
            f.write(n + "\n")
    print("saved:", out_topk)
    print("\nTOPK:", top_names)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--n_repeats", type=int, default=3)
    p.add_argument("--topk", type=int, default=20)
    p.add_argument("--out_prefix", type=str, default=None)
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--seed_only", type=int, default=None)
    p.add_argument("--allow_run_mismatch", action="store_true")
    args = p.parse_args()
    main(
        args.pest,
        args.ckpt,
        args.run,
        args.n_repeats,
        args.topk,
        args.out_prefix,
        args.out_root,
        args.seed_only,
        args.allow_run_mismatch,
    )
