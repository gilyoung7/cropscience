from __future__ import annotations

import argparse
import json
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rice.configs import config as C
from rice.src.featuresets import get_feature_cols
from rice.src.data_pipeline import (
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
from rice.src.labels import build_interval_labels_from_doy, filter_labels_by_gap
from rice.src.dataset import (
    build_train_frame,
    slice_season,
    build_samples_season,
    split_by_site,
    compute_norm_stats,
    IntervalEventDataset,
    log_split_fingerprint,
    split_fingerprint,
    split_seed_search_topk,
)
from rice.src.model import HazardTransformer
from rice.src.train_eval import eval_nll_model, eval_metrics_with_overlap


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


def build_samples_for_run(run: int):
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

    return feature_cols, T, samples


def parse_seed_candidates(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    if ":" in raw:
        start_s, end_s = raw.split(":", 1)
        return list(range(int(start_s), int(end_s)))
    return [int(x) for x in raw.split(",") if x.strip()]


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


def check_run_match(ckpt_run: int, cli_run: int, allow_run_mismatch: bool, ckpt_path: Path):
    if ckpt_run != cli_run:
        msg = (
            f"Run mismatch: ckpt has run={ckpt_run} but CLI --run={cli_run}. "
            f"Pass --run {ckpt_run} or specify --ckpt correctly.\n"
            f"Example: python -m scripts.run_eval --run {ckpt_run} --ckpt {ckpt_path}"
        )
        if allow_run_mismatch:
            print("WARNING:", msg)
        else:
            raise ValueError(msg)


def main(
    ckpt_path: str | None,
    run: int,
    out_csv: str | None,
    out_root: str,
    allow_run_mismatch: bool,
    split_seed: int,
    seeds: list[int] | None,
    auto_split_seed: bool,
    seed_candidates_raw: str | None,
    target_test_interval: int | None,
    tol_test_interval: int | None,
    auto_split_topk: int,
    split_seed_from_topk_idx: int | None,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt_path_resolved = resolve_ckpt_path(run, out_root, ckpt_path)
    print(f"Using checkpoint: {ckpt_path_resolved}")
    ckpt = torch.load(ckpt_path_resolved, map_location="cpu")
    ckpt_run = int(ckpt.get("run", -1))
    check_run_match(ckpt_run, run, allow_run_mismatch, ckpt_path_resolved)
    trained_states = ckpt["trained_states"]
    print("loaded ckpt:", ckpt_path_resolved, "| seeds:", [d["seed"] for d in trained_states])

    feature_cols, T, samples = build_samples_for_run(run)
    result = None
    chosen = None
    if auto_split_seed:
        candidates = parse_seed_candidates(seed_candidates_raw) or list(range(0, 200))
        result = split_seed_search_topk(
            samples,
            val_frac=0.1,
            test_frac=0.1,
            seed_candidates=candidates,
            target_test_interval=target_test_interval,
            tol_test_interval=tol_test_interval,
            topk=auto_split_topk,
        )
        topk_list = result["topk"]
        if not topk_list:
            raise ValueError("auto_split_seed produced no candidates")
        if split_seed_from_topk_idx is None:
            split_seed_from_topk_idx = 0
        if split_seed_from_topk_idx < 0 or split_seed_from_topk_idx >= len(topk_list):
            raise ValueError(f"--split_seed_from_topk_idx out of range (0..{len(topk_list)-1})")
        chosen = topk_list[split_seed_from_topk_idx]
        split_seed = int(chosen["seed"])
        train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)
        print(
            f"[auto_split] selected seed={split_seed} score={chosen['score']:.6f} "
            f"counts={chosen['counts']}"
        )
        if result.get("used_fallback"):
            print("[auto_split] WARNING: no seed met constraints; using best score fallback.")
        print("[auto_split] topk seeds (seed, score, test_counts):")
        for i, item in enumerate(topk_list):
            print(f"  [{i}] seed={item['seed']} score={item['score']:.6f} test={item['counts']['test']}")
    else:
        train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)

    log_split_fingerprint("eval", train_s, val_s, test_s)
    fp = split_fingerprint(train_s, val_s, test_s)
    split_out_dir = Path(out_root) / "splits"
    split_out_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_out_dir / f"selected_split_seed_run{run}.json"
    split_payload = {
        "seed": split_seed,
        "score": chosen["score"] if chosen is not None else None,
        "counts": chosen["counts"] if chosen is not None else None,
        "hashes": {
            "train": fp["train_sites_hash"],
            "val": fp["val_sites_hash"],
            "test": fp["test_sites_hash"],
        },
    }
    with open(split_path, "w", encoding="utf-8") as f:
        json.dump(split_payload, f, ensure_ascii=False, indent=2)
    print("saved:", split_path)

    if result is not None:
        topk_payload = {
            "target_test_interval": target_test_interval,
            "tol_test_interval": tol_test_interval,
            "seed_candidates": seed_candidates_raw,
            "topk": [],
        }
        for item in result["topk"]:
            seed_i = int(item["seed"])
            tr_i, va_i, te_i = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=seed_i)
            fp_i = split_fingerprint(tr_i, va_i, te_i)
            topk_payload["topk"].append(
                {
                    "seed": seed_i,
                    "score": item["score"],
                    "counts": item["counts"],
                    "hashes": {
                        "train": fp_i["train_sites_hash"],
                        "val": fp_i["val_sites_hash"],
                        "test": fp_i["test_sites_hash"],
                    },
                }
            )
        topk_path = split_out_dir / f"selected_split_seeds_run{run}.json"
        with open(topk_path, "w", encoding="utf-8") as f:
            json.dump(topk_payload, f, ensure_ascii=False, indent=2)
        print("saved:", topk_path)

    x_mean, x_std = compute_norm_stats(train_s)
    train_ds = IntervalEventDataset(train_s, x_mean, x_std)
    val_ds   = IntervalEventDataset(val_s,   x_mean, x_std)
    test_ds  = IntervalEventDataset(test_s,  x_mean, x_std)
    print(f"RUN={run} | D_in={len(feature_cols)} | T={T}")

    val_loader  = make_loader(val_ds,  C.BATCH_EVAL, shuffle=False)
    test_loader = make_loader(test_ds, C.BATCH_EVAL, shuffle=False)

    records = []
    for d in trained_states:
        seed = int(d["seed"])
        if seeds is not None and seed not in seeds:
            continue

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
    p.add_argument("--allow_run_mismatch", action="store_true")
    p.add_argument("--split_seed", type=int, default=C.SPLIT_SEED)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--auto_split_seed", action="store_true")
    p.add_argument("--auto_split_topk", type=int, default=1)
    p.add_argument("--split_seed_from_topk_idx", type=int, default=None)
    p.add_argument("--seed_candidates", type=str, default=None)
    p.add_argument("--target_test_interval", type=int, default=None)
    p.add_argument("--tol_test_interval", type=int, default=None)
    args = p.parse_args()
    main(
        args.ckpt,
        args.run,
        args.out_csv,
        args.out_root,
        args.allow_run_mismatch,
        args.split_seed,
        args.seeds,
        args.auto_split_seed,
        args.seed_candidates,
        args.target_test_interval,
        args.tol_test_interval,
        args.auto_split_topk,
        args.split_seed_from_topk_idx,
    )
