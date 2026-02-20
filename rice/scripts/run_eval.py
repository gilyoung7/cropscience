from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest, default_out_root, ensure_output_dirs
from rice.scripts.common import make_loader, parse_seed_candidates
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
    compute_norm_stats,
    IntervalEventDataset,
    log_split_fingerprint,
    split_seed_search_topk,
    build_stage2_nowcast_samples,
)
from rice.src.ckpt_schema import validate_ckpt_meta
from rice.src.model import HazardTransformer
from rice.src.train_eval import eval_nll_model, eval_metrics_with_overlap

def build_samples_for_run(run: int, get_feature_cols, return_debug_stats: bool = False):
    """
    Rebuild datasets deterministically (same as training pipeline split seed=42).
    Norm stats are recomputed from train split.
    """
    # DAILY
    daily = load_daily_preprocessed(C.PATH_DAILY, C.GDD_DIR)

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
    group_len = (
        df_season.groupby(["site_id", "year"], sort=False)
        .size()
        .reset_index(name="n_rows")
    )
    dropped_groups_df = group_len[group_len["n_rows"] != T][["site_id", "year"]].copy()
    samples, dropped, feature_names = build_samples_season(df_season, feature_cols, C.DOY_START, C.DOY_END)
    if dropped > 0:
        print("WARNING: dropped groups (len!=T):", dropped)

    if not return_debug_stats:
        return feature_cols, feature_names, T, samples

    debug_stats = {
        "n_groups_total": int(len(group_len)),
        "n_groups_kept": int(len(samples)),
        "n_groups_dropped_len_mismatch": int(dropped),
        "drop_ratio_len_mismatch": float(dropped / max(int(len(group_len)), 1)),
        "dropped_site_year": [(str(r.site_id), int(r.year)) for r in dropped_groups_df.itertuples(index=False)],
    }
    return feature_cols, feature_names, T, samples, debug_stats


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


def resolve_split_seeds_json_path(out_root: str, split_seeds_json: str | None) -> Path:
    if split_seeds_json:
        return Path(split_seeds_json)
    return Path(out_root) / "splits" / "selected_split_seeds.json"


def load_split_seed_from_topk(split_seeds_json_path: Path, split_seed_from_topk_idx: int | None) -> tuple[int, int, dict, dict]:
    if not split_seeds_json_path.exists():
        raise FileNotFoundError(f"split seeds json not found: {split_seeds_json_path}")
    with open(split_seeds_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    topk_list = payload.get("topk", [])
    if not topk_list:
        raise ValueError(f"topk is empty in split seeds json: {split_seeds_json_path}")
    if split_seed_from_topk_idx is None:
        split_seed_from_topk_idx = payload.get("selected_topk_idx", 0)
    if split_seed_from_topk_idx < 0 or split_seed_from_topk_idx >= len(topk_list):
        raise ValueError(f"--split_seed_from_topk_idx out of range (0..{len(topk_list)-1})")
    chosen = topk_list[int(split_seed_from_topk_idx)]
    split_seed = int(chosen["seed"])
    return split_seed, int(split_seed_from_topk_idx), chosen, payload


def main(
    pest: str,
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
    split_seeds_json: str | None,
    stage2_nowcast: bool,
    stage2_nowcast_window: int,
    stage2_nowcast_stride: int,
    stage2_nowcast_tstar_start: int | None,
    stage2_nowcast_only_pre_event: int,
    stage2_nowcast_event_time_proxy: str,
):
    _, get_feature_cols = resolve_pest(pest)
    if not out_root:
        out_root = default_out_root(pest)
    ensure_output_dirs(out_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt_path_resolved = resolve_ckpt_path(run, out_root, ckpt_path)
    print(f"Using checkpoint: {ckpt_path_resolved}")
    ckpt = torch.load(ckpt_path_resolved, map_location="cpu")
    trained_states = ckpt["trained_states"]
    print("loaded ckpt:", ckpt_path_resolved, "| seeds:", [d["seed"] for d in trained_states])

    feature_cols, feature_names_eval, T, samples = build_samples_for_run(run, get_feature_cols)
    print(f"[features] n={len(feature_names_eval)} head={feature_names_eval[:5]} tail={feature_names_eval[-5:]}")
    result = None
    chosen = None
    if split_seeds_json is not None:
        split_seeds_json_path = resolve_split_seeds_json_path(out_root, split_seeds_json)
        split_seed, chosen_idx, chosen, _ = load_split_seed_from_topk(split_seeds_json_path, split_seed_from_topk_idx)
        train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)
        print(f"[split_seed_json] selected seed={split_seed} idx={chosen_idx} file={split_seeds_json_path}")
        print(f"[split_seed_json] counts={chosen.get('counts')}")
    elif auto_split_seed:
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

    stage2_nowcast = bool(ckpt.get("stage2_nowcast", stage2_nowcast))
    if stage2_nowcast:
        stage2_nowcast_window = int(ckpt.get("stage2_nowcast_window", stage2_nowcast_window))
        stage2_nowcast_stride = int(ckpt.get("stage2_nowcast_stride", stage2_nowcast_stride))
        stage2_nowcast_tstar_start = ckpt.get("stage2_nowcast_tstar_start", stage2_nowcast_tstar_start)
        stage2_nowcast_only_pre_event = int(ckpt.get("stage2_nowcast_only_pre_event", stage2_nowcast_only_pre_event))
        stage2_nowcast_event_time_proxy = str(ckpt.get("stage2_nowcast_event_time_proxy", stage2_nowcast_event_time_proxy))

        train_s = build_stage2_nowcast_samples(
            train_s,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
        )
        val_s = build_stage2_nowcast_samples(
            val_s,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
        )
        test_s = build_stage2_nowcast_samples(
            test_s,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
        )
        print(
            f"[stage2_nowcast] window={stage2_nowcast_window} stride={stage2_nowcast_stride} "
            f"tstar_start={stage2_nowcast_tstar_start} only_pre_event={bool(stage2_nowcast_only_pre_event)} "
            f"event_time_proxy={stage2_nowcast_event_time_proxy} | "
            f"samples train={len(train_s)} val={len(val_s)} test={len(test_s)}"
        )
        def _bucket_counts(ss: list[dict]) -> dict[str, int]:
            out = {"pre_L": 0, "in_LR": 0, "post_R": 0, "right": 0}
            for x in ss:
                b = str(x.get("case_bucket", ""))
                if b in out:
                    out[b] += 1
            return out
        print(
            f"[stage2_nowcast] case_bucket train={_bucket_counts(train_s)} "
            f"val={_bucket_counts(val_s)} test={_bucket_counts(test_s)}"
        )

    log_split_fingerprint("eval", train_s, val_s, test_s)

    x_mean, x_std = compute_norm_stats(train_s)
    train_ds = IntervalEventDataset(train_s, x_mean, x_std)
    val_ds   = IntervalEventDataset(val_s,   x_mean, x_std)
    test_ds  = IntervalEventDataset(test_s,  x_mean, x_std)
    D_in = int(test_ds[0][0].shape[-1])
    print(f"[D_in] computed_from_dataset={D_in}")
    print(f"RUN={run} | D_in={D_in} | T={T}")

    validate_ckpt_meta(
        ckpt,
        pest=pest,
        run=run,
        d_in=D_in,
        feature_names=feature_names_eval,
        allow_run_mismatch=allow_run_mismatch,
    )

    val_loader  = make_loader(val_ds,  C.BATCH_EVAL, shuffle=False)
    test_loader = make_loader(test_ds, C.BATCH_EVAL, shuffle=False)

    records = []
    by_tstar_rows = []
    for d in trained_states:
        seed = int(d["seed"])
        if seeds is not None and seed not in seeds:
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

        val_nll  = float(eval_nll_model(model, val_loader, Tend=T, device=device))
        test_nll = float(eval_nll_model(model, test_loader, Tend=T, device=device))

        val_stats = eval_metrics_with_overlap(
            model,
            val_loader,
            Tend=T,
            device=device,
            alpha=0.2,
            pi_method=getattr(C, "PI_METHOD", "shortest"),
        )
        test_stats = eval_metrics_with_overlap(
            model,
            test_loader,
            Tend=T,
            device=device,
            alpha=0.2,
            pi_method=getattr(C, "PI_METHOD", "shortest"),
        )

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

        if stage2_nowcast:
            for split_name, split_samples in (("val", val_s), ("test", test_s)):
                tvals = sorted({int(s.get("tstar", -1)) for s in split_samples if "tstar" in s})
                for tstar in tvals:
                    sub = [s for s in split_samples if int(s.get("tstar", -1)) == tstar]
                    if not sub:
                        continue
                    sub_ds = IntervalEventDataset(sub, x_mean, x_std)
                    sub_loader = make_loader(sub_ds, C.BATCH_EVAL, shuffle=False)
                    sub_stats = eval_metrics_with_overlap(
                        model,
                        sub_loader,
                        Tend=T,
                        device=device,
                        alpha=0.2,
                        pi_method=getattr(C, "PI_METHOD", "shortest"),
                    )
                    by_tstar_rows.append(
                        {
                            "seed": seed,
                            "split": split_name,
                            "tstar": int(tstar),
                            "n": int(len(sub)),
                            "N_total": int(len(sub)),
                            "N_interval_label": int(sum(1 for s in sub if str(s.get("censor_type", "")) == "interval")),
                            "N_right_label": int(sum(1 for s in sub if str(s.get("censor_type", "")) == "right")),
                            "N_preL": int(sum(1 for s in sub if str(s.get("case_bucket", "")) == "pre_L")),
                            "N_inLR": int(sum(1 for s in sub if str(s.get("case_bucket", "")) == "in_LR")),
                            "N_postR": int(sum(1 for s in sub if str(s.get("case_bucket", "")) == "post_R")),
                            "IoU80_interval_only": float(sub_stats["IoU_mean_interval_only(80%)"]),
                            "Precision80_interval_only": float(sub_stats["Precision_mean_interval_only(80%)"]),
                            "Recall80_interval_only": float(sub_stats["Recall_mean_interval_only(80%)"]),
                            "MAE_int_interval_only": float(sub_stats["mae_mid_mean_interval_only"]),
                            "Mass_int_interval_only": float(sub_stats["mass_in_interval_mean_interval_only"]),
                            "N_int": int(sub_stats["N_interval_samples"]),
                        }
                    )

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
        if stage2_nowcast and by_tstar_rows:
            by_tstar_df = pd.DataFrame(by_tstar_rows).sort_values(["seed", "split", "tstar"]).reset_index(drop=True)
            by_tstar_out = str(Path(out_csv_resolved).with_name(Path(out_csv_resolved).stem + "_by_tstar.csv"))
            by_tstar_df.to_csv(by_tstar_out, index=False)
            print("saved:", by_tstar_out)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--allow_run_mismatch", action="store_true")
    p.add_argument("--split_seed", type=int, default=C.SPLIT_SEED)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--auto_split_seed", action="store_true")
    p.add_argument("--auto_split_topk", type=int, default=1)
    p.add_argument("--split_seed_from_topk_idx", type=int, default=None)
    p.add_argument("--split_seeds_json", type=str, default=None)
    p.add_argument("--seed_candidates", type=str, default=None)
    p.add_argument("--target_test_interval", type=int, default=None)
    p.add_argument("--tol_test_interval", type=int, default=None)
    p.add_argument("--stage2_nowcast", action="store_true")
    p.add_argument("--stage2_nowcast_window", type=int, default=56)
    p.add_argument("--stage2_nowcast_stride", type=int, default=7)
    p.add_argument("--stage2_nowcast_tstar_start", type=int, default=None)
    p.add_argument("--stage2_nowcast_only_pre_event", type=int, default=1)
    p.add_argument("--stage2_nowcast_event_time_proxy", type=str, default="r", choices=["r", "mid"])
    # Stage-1 style aliases for pipeline consistency.
    p.add_argument("--nowcast_window", dest="stage2_nowcast_window", type=int)
    p.add_argument("--nowcast_stride", dest="stage2_nowcast_stride", type=int)
    p.add_argument("--nowcast_tstar_start", dest="stage2_nowcast_tstar_start", type=int)
    p.add_argument("--nowcast_only_pre_event", dest="stage2_nowcast_only_pre_event", type=int)
    p.add_argument("--nowcast_event_time_proxy", dest="stage2_nowcast_event_time_proxy", type=str, choices=["r", "mid"])
    args = p.parse_args()
    main(
        args.pest,
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
        args.split_seeds_json,
        args.stage2_nowcast,
        args.stage2_nowcast_window,
        args.stage2_nowcast_stride,
        args.stage2_nowcast_tstar_start,
        args.stage2_nowcast_only_pre_event,
        args.stage2_nowcast_event_time_proxy,
    )
