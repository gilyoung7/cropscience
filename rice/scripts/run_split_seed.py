from __future__ import annotations

import argparse
import json
from pathlib import Path

from rice.src.pest_resolver import resolve_pest, default_out_root, ensure_output_dirs
from rice.src.data_pipeline import (
    load_daily_preprocessed,
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
    split_fingerprint,
    split_seed_search_topk,
)


def parse_seed_candidates(raw: str | None) -> list[int] | None:
    if raw is None:
        return None
    if ":" in raw:
        start_s, end_s = raw.split(":", 1)
        return list(range(int(start_s), int(end_s)))
    return [int(x) for x in raw.split(",") if x.strip()]


def main(
    pest: str,
    run: int,
    out_root: str | None,
    auto_split_seed: bool,
    seed_candidates_raw: str | None,
    target_test_interval: int | None,
    tol_test_interval: int | None,
    auto_split_topk: int,
    split_seed_from_topk_idx: int | None,
    val_frac: float,
    test_frac: float,
):
    C, get_feature_cols = resolve_pest(pest)
    if not out_root:
        out_root = default_out_root(pest)
    ensure_output_dirs(out_root)

    # 1) Build samples with same preprocessing path as train/eval.
    daily = load_daily_preprocessed(C.PATH_DAILY)
    obs = load_obs(C.PATH_OBS)
    obs2 = aggregate_obs_daily_max(obs)
    labels = build_interval_labels_from_doy(
        obs2,
        threshold=C.THRESHOLD,
        season_start_doy=C.SEASON_START_DOY,
        season_end_doy=C.SEASON_END_DOY,
    )
    labels = filter_labels_by_gap(labels, C.DOY_START, C.DOY_END, C.MAX_GAP)
    obs_meta = make_obs_meta(obs2, C.DOY_START, C.DOY_END)
    daily_feat, _ = make_daily_feature_frame(daily)
    T = C.DOY_END - C.DOY_START + 1
    train_df = build_train_frame(daily_feat, labels, obs_meta, T=T)
    train_df = add_site_static_latlon(train_df, obs)
    train_df = merge_pheno_daily_ffill(train_df, obs)

    feature_cols = get_feature_cols(run)
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in train_df: {missing}")

    df_season = slice_season(train_df, C.DOY_START, C.DOY_END)
    samples, dropped, _ = build_samples_season(df_season, feature_cols, C.DOY_START, C.DOY_END)
    print("samples:", len(samples), "| dropped groups (len!=T):", dropped)

    result = None
    if not auto_split_seed:
        raise ValueError("run_split_seed requires --auto_split_seed to generate top-k split seeds.")
    candidates = parse_seed_candidates(seed_candidates_raw) or list(range(0, 200))
    result = split_seed_search_topk(
        samples,
        val_frac=val_frac,
        test_frac=test_frac,
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
    print(
        f"[auto_split] selected seed={split_seed} score={chosen['score']:.6f} "
        f"counts={chosen['counts']}"
    )
    if result.get("used_fallback"):
        print("[auto_split] WARNING: no seed met constraints; using best score fallback.")
    print("[auto_split] topk seeds (seed, score, test_counts):")
    for i, item in enumerate(topk_list):
        print(f"  [{i}] seed={item['seed']} score={item['score']:.6f} test={item['counts']['test']}")

    split_out_dir = Path(out_root) / "splits"
    split_out_dir.mkdir(parents=True, exist_ok=True)

    topk_payload = {
        "run": run,
        "selected_topk_idx": int(split_seed_from_topk_idx),
        "selected_seed": int(split_seed),
        "target_test_interval": target_test_interval,
        "tol_test_interval": tol_test_interval,
        "seed_candidates": seed_candidates_raw,
        "val_frac": val_frac,
        "test_frac": test_frac,
        "topk": [],
    }
    for item in result["topk"]:
        seed_i = int(item["seed"])
        tr_i, va_i, te_i = split_by_site(samples, val_frac=val_frac, test_frac=test_frac, seed=seed_i)
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
    topk_path = split_out_dir / "selected_split_seeds.json"
    with open(topk_path, "w", encoding="utf-8") as f:
        json.dump(topk_payload, f, ensure_ascii=False, indent=2)
    print("saved:", topk_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, required=True)
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--auto_split_seed", action="store_true")
    p.add_argument("--auto_split_topk", type=int, default=3)
    p.add_argument("--split_seed_from_topk_idx", type=int, default=0)
    p.add_argument("--seed_candidates", type=str, default=None)
    p.add_argument("--target_test_interval", type=int, default=None)
    p.add_argument("--tol_test_interval", type=int, default=None)
    p.add_argument("--val_frac", type=float, default=0.1)
    p.add_argument("--test_frac", type=float, default=0.1)
    args = p.parse_args()

    main(
        pest=args.pest,
        run=args.run,
        out_root=args.out_root,
        auto_split_seed=args.auto_split_seed,
        seed_candidates_raw=args.seed_candidates,
        target_test_interval=args.target_test_interval,
        tol_test_interval=args.tol_test_interval,
        auto_split_topk=args.auto_split_topk,
        split_seed_from_topk_idx=args.split_seed_from_topk_idx,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
    )
