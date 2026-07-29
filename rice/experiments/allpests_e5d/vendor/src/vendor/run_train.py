from __future__ import annotations

import copy
import random
import argparse
from pathlib import Path
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest, default_out_root, ensure_output_dirs
from rice.scripts.common import (
    make_loader,
    collate_grouped_stage2,
    parse_seed_candidates,
    parse_tags,
    init_wandb_run,
    finish_wandb_run,
)
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
    split_samples,
    compute_norm_stats,
    IntervalEventDataset,
    GroupedIntervalEventDataset,
    log_split_fingerprint,
    log_split_sanity,
    split_seed_search_topk,
    build_stage2_nowcast_samples,
    group_stage2_samples_by_site_year,
)
from rice.src.ckpt_schema import build_ckpt_meta
from src.vendor.model import HazardTransformer, HierarchicalCausalHazardTransformer  # VENDORED (prior_residual_alert_bin branch; legacy untouched)
from rice.src.train_eval import (
    run_epoch_weighted,
    run_epoch_weighted_grouped,
    eval_nll_model,
    eval_nll_model_grouped,
    eval_metrics_with_overlap,
    eval_metrics_with_overlap_grouped,
    early_recall80_site_year,
)
from rice.scripts.run_event_train import (
    EventTransformer,
    build_nowcast_samples,
    build_tabular_from_samples,
)
from rice.scripts.run_event_eval import apply_temperature, fit_temperature_grid
from rice.scripts.run_viz_interval import (
    build_alert_map,
    build_alert_map_consecutive,
    collect_interval_preds,
    collect_interval_preds_grouped,
    load_stage1_eval_policy,
    summarize_matched_interval_rows,
)

DEBUG_LOADER_SETTINGS = False
DEBUG_PICKLE_DATASET = True
DEBUG_SAMPLE_CHECK = True
WANDB_ENTITY_DEFAULT = "gilyoung7-seoul-national-university"

def _split_stats(samples: list[dict]) -> dict:
    sites = {s["site_id"] for s in samples}
    n_sites = len(sites)
    n_samples = len(samples)
    if n_samples == 0:
        return {"n_sites": n_sites, "n_samples": 0, "event_rate": 0.0}
    n_event = sum(1 for s in samples if str(s.get("censor_type", "")) != "right")
    return {"n_sites": n_sites, "n_samples": n_samples, "event_rate": n_event / n_samples}

def _interval_len_stats(samples: list[dict]) -> tuple[int, float, float]:
    lens = [int(s["R"]) - int(s["L"]) for s in samples if str(s.get("censor_type", "")) == "interval"]
    if not lens:
        return 0, 0.0, 0.0
    arr = np.asarray(lens, dtype=float)
    return len(lens), float(arr.mean()), float(arr.var(ddof=0))


def parse_balance_ratio(raw: str | None) -> dict[str, float] | None:
    """
    raw format: 'right:interval:left', e.g. '1:1:1'
    """
    if raw is None:
        return None
    if str(raw).strip().lower() in ("none", "off", "false", "no", ""):
        return None
    parts = [p.strip() for p in raw.split(":")]
    if len(parts) != 3:
        raise ValueError("--train_balance_ratio must be in 'right:interval:left' format, e.g. 1:1:1")
    vals = [float(p) for p in parts]
    if any(v < 0 for v in vals) or sum(vals) <= 0:
        raise ValueError("--train_balance_ratio values must be non-negative and not all zero")
    s = sum(vals)
    return {
        "right": vals[0] / s,
        "interval": vals[1] / s,
        "left": vals[2] / s,
    }


def build_balanced_sampler(train_samples: list[dict], ratio: dict[str, float], seed: int) -> WeightedRandomSampler:
    counts = {"right": 0, "interval": 0, "left": 0}
    for s in train_samples:
        c = str(s.get("censor_type", ""))
        if c in counts:
            counts[c] += 1

    active = {k: v for k, v in ratio.items() if counts.get(k, 0) > 0 and v > 0}
    if not active:
        raise ValueError(f"no active classes for balanced sampling. counts={counts} ratio={ratio}")
    z = sum(active.values())
    active = {k: v / z for k, v in active.items()}

    class_w = {}
    for k in ("right", "interval", "left"):
        if counts.get(k, 0) > 0 and k in active:
            class_w[k] = active[k] / float(counts[k])
        else:
            class_w[k] = 0.0

    sample_w = [class_w.get(str(s.get("censor_type", "")), 0.0) for s in train_samples]
    gen = torch.Generator().manual_seed(seed)
    weights = torch.as_tensor(sample_w, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_samples),
        replacement=True,
        generator=gen,
    )


def _build_base_samples_from_preloaded(
    *,
    daily_feat: pd.DataFrame,
    obs: pd.DataFrame,
    obs2: pd.DataFrame,
    get_feature_cols,
    run: int,
    doy_start: int,
    doy_end: int,
) -> tuple[list[dict], list[str]]:
    labels = build_interval_labels_from_doy(
        obs2,
        threshold=C.THRESHOLD,
        season_start_doy=C.SEASON_START_DOY,
        season_end_doy=C.SEASON_END_DOY,
    )
    labels = filter_labels_by_gap(labels, int(doy_start), int(doy_end), C.MAX_GAP)
    obs_meta = make_obs_meta(obs2, int(doy_start), int(doy_end))
    T = int(doy_end) - int(doy_start) + 1
    train_df = build_train_frame(daily_feat, labels, obs_meta, T=T)
    train_df = add_site_static_latlon(train_df, obs)
    train_df = merge_pheno_daily_ffill(train_df, obs)
    feature_cols = get_feature_cols(run)
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in gated-val train_df: {missing}")
    df_season = slice_season(train_df, int(doy_start), int(doy_end))
    samples, dropped, feature_names = build_samples_season(df_season, feature_cols, int(doy_start), int(doy_end))
    if dropped:
        print(f"[gated_val] dropped stage1 groups (len!=T): {dropped}")
    return samples, feature_names


def _select_stage_state(ckpt: dict, seed: int) -> dict | None:
    for state in ckpt.get("trained_states", []):
        if int(state.get("seed", -1)) == int(seed):
            return state
    return None


def _prepare_gated_val_alert_maps(
    *,
    stage1_ckpt_path: str,
    stage1_eval_csv: str,
    seeds: list[int],
    split_seed: int,
    split_mode: str,
    run: int,
    daily_feat: pd.DataFrame,
    obs: pd.DataFrame,
    obs2: pd.DataFrame,
    get_feature_cols,
    device: torch.device,
) -> tuple[dict[int, dict[str, int]], int]:
    ckpt1 = torch.load(stage1_ckpt_path, map_location="cpu")
    stage1_doy_start = int(ckpt1.get("doy_start", C.DOY_START))
    stage1_doy_end = int(ckpt1.get("doy_end", C.DOY_END))
    samples1_base, feature_names1 = _build_base_samples_from_preloaded(
        daily_feat=daily_feat,
        obs=obs,
        obs2=obs2,
        get_feature_cols=get_feature_cols,
        run=run,
        doy_start=stage1_doy_start,
        doy_end=stage1_doy_end,
    )
    train_s1_base, val_s1_base, _test_s1_base = split_samples(
        samples1_base,
        val_frac=0.1,
        test_frac=0.1,
        seed=int(split_seed),
        split_mode=split_mode,
    )
    task_mode = str(ckpt1.get("task_mode", "season_complete"))
    if task_mode != "nowcast":
        raise ValueError("gated validation currently expects Stage1 task_mode='nowcast'")
    val_s1 = build_nowcast_samples(
        val_s1_base,
        window=int(ckpt1.get("nowcast_window", 28)),
        stride=int(ckpt1.get("nowcast_stride", 7)),
        tstar_start=ckpt1.get("nowcast_tstar_start", None),
        only_pre_event=bool(int(ckpt1.get("nowcast_only_pre_event", 1))),
        event_time_proxy=str(ckpt1.get("nowcast_event_time_proxy", "r")),
    )
    policy_by_seed = load_stage1_eval_policy(stage1_eval_csv)
    if not policy_by_seed:
        raise ValueError("--gated_val_stage1_eval_csv is required for gated validation checkpoint selection")

    alert_maps: dict[int, dict[str, int]] = {}
    model_kind = str(ckpt1.get("event_model", "transformer"))
    add_tstar_position_feature = bool(ckpt1.get("add_tstar_position_feature", False))
    for seed in seeds:
        policy = policy_by_seed.get(int(seed))
        if policy is None:
            raise ValueError(f"Stage1 eval CSV has no policy row for seed={seed}")
        state = _select_stage_state(ckpt1, int(seed))
        if state is None:
            raise ValueError(f"Stage1 checkpoint has no trained state for seed={seed}")

        if model_kind == "transformer":
            x_mean1, x_std1 = compute_norm_stats(train_s1_base)
            val_ds1 = IntervalEventDataset(val_s1, x_mean1, x_std1)
            val_loader1 = make_loader(val_ds1, C.BATCH_EVAL, shuffle=False)
            model1 = EventTransformer(
                d_in=len(feature_names1),
                d_model=int(ckpt1.get("d_model", C.D_MODEL)),
                nhead=int(ckpt1.get("n_head", C.N_HEAD)),
                num_layers=2,
                dropout=float(ckpt1.get("dropout", C.DROPOUT)),
                max_len=C.MAX_LEN,
            ).to(device)
            model1.load_state_dict(state["state_dict"])
            model1.eval()
            probs = []
            with torch.no_grad():
                for X, *_rest in val_loader1:
                    X = X.to(device, non_blocking=True)
                    probs.append(torch.sigmoid(model1(X)).detach().cpu().numpy())
            p_val_raw = np.concatenate(probs) if probs else np.asarray([], dtype=float)
        else:
            clf = state.get("sk_model")
            if clf is None:
                raise ValueError("Stage1 checkpoint missing sk_model")
            X_val_tab = build_tabular_from_samples(
                val_s1,
                add_tstar_position_feature=bool(add_tstar_position_feature),
            )
            if hasattr(clf, "predict_proba"):
                p_val_raw = clf.predict_proba(X_val_tab)[:, 1]
            else:
                p_val_raw = np.asarray(clf.predict(X_val_tab), dtype=float)

        y_val = np.asarray([0 if str(s.get("censor_type", "")) == "right" else 1 for s in val_s1], dtype=int)
        temp, _ = fit_temperature_grid(y_val, p_val_raw)
        p_val_cal = apply_temperature(p_val_raw, temp)
        tau = float(policy["tau"])
        k = int(policy.get("gate_consecutive_k", 1))
        ma = int(policy.get("gate_smooth_window", 1))
        use_t_alert_start = int(policy.get("gate_use_t_alert_start", 0))
        t_alert_start = None if use_t_alert_start == 0 else policy.get("t_alert_start", None)
        if k > 1 or ma > 1:
            alert_rel = build_alert_map_consecutive(
                val_s1,
                p_val_cal,
                tau,
                split_name="val",
                seed=int(seed),
                t_alert_start=t_alert_start,
                consecutive_k=k,
                smooth_window=ma,
            )
        else:
            alert_rel = build_alert_map(val_s1, p_val_cal, tau, t_alert_start)
        alert_maps[int(seed)] = {
            str(sample_id): int(tstar) + int(stage1_doy_start) - 1
            for sample_id, tstar in alert_rel.items()
            if tstar is not None
        }
        print(
            f"[gated_val] seed={seed} stage1_tau={tau:.6f} "
            f"policy={policy.get('gate_policy_name', '')} alerts={len(alert_maps[int(seed)])}"
        )
    return alert_maps, stage1_doy_start


@torch.no_grad()
def _eval_gated_validation_stats(
    *,
    model,
    loader,
    source_groups: list[dict],
    source_samples: list[dict],
    grouped_mode: bool,
    alert_map_abs: dict[str, int],
    stage2_doy_start: int,
    stage2_tstar_offsets: list[int],
    stage2_tstar_offset_weights: list[float],
    Tend: int,
    device: torch.device,
) -> dict:
    if grouped_mode:
        rows = collect_interval_preds_grouped(
            model,
            loader,
            source_groups=source_groups,
            Tend=int(Tend),
            device=device,
            pi_method=getattr(C, "PI_METHOD", "shortest"),
            max_samples=100000000,
        )
    else:
        rows = collect_interval_preds(
            model,
            loader,
            source_samples=source_samples,
            Tend=int(Tend),
            device=device,
            pi_method=getattr(C, "PI_METHOD", "shortest"),
            max_samples=100000000,
        )
    row_map = {}
    for r in rows:
        if r.get("tstar") is None:
            continue
        tstar_abs = int(r["tstar"]) + int(stage2_doy_start) - 1
        row_map[(str(r["sample_id"]), int(tstar_abs))] = r

    n_true = len({
        (s.get("site_id"), int(s.get("year")))
        for s in source_samples
        if str(s.get("censor_type", "")) == "interval"
    })
    matched_rows = []
    matched_by_offset = {int(offset): [] for offset in stage2_tstar_offsets}
    missed_by_offset = {int(offset): 0 for offset in stage2_tstar_offsets}
    for sample_id, alert_abs in alert_map_abs.items():
        for offset in stage2_tstar_offsets:
            stage2_tstar_abs = int(alert_abs) + int(offset)
            pred_row = row_map.get((str(sample_id), int(stage2_tstar_abs)))
            if pred_row is None:
                missed_by_offset[int(offset)] += 1
                continue
            pred_row = dict(pred_row)
            pred_row["tstar"] = int(stage2_tstar_abs)
            pred_row["true_L"] = int(pred_row["true_L"]) + int(stage2_doy_start) - 1
            pred_row["true_R"] = int(pred_row["true_R"]) + int(stage2_doy_start) - 1
            pred_row["pred_L"] = int(pred_row["pred_L"]) + int(stage2_doy_start) - 1
            pred_row["pred_R"] = int(pred_row["pred_R"]) + int(stage2_doy_start) - 1
            pred_row["pred_point"] = int(pred_row["pred_point"]) + int(stage2_doy_start) - 1
            pred_row["alert_tstar"] = int(alert_abs)
            pred_row["stage2_tstar"] = int(stage2_tstar_abs)
            pred_row["stage2_tstar_offset"] = int(offset)
            matched_rows.append(pred_row)
            matched_by_offset[int(offset)].append(pred_row)

    stats = summarize_matched_interval_rows(
        matched_rows,
        n_true=int(n_true),
        doy_start=int(stage2_doy_start),
    )
    early_rec, early_success, early_denom = early_recall80_site_year(matched_rows)
    stats["EarlyRecall80"] = float(early_rec)
    stats["early_recall80_success"] = int(early_success)
    stats["early_recall80_denominator"] = int(early_denom)
    stats["n_matched"] = int(len(matched_rows))
    stats["n_true"] = int(n_true)
    stats["offset_missed"] = int(sum(missed_by_offset.values()))
    stats["stage2_tstar_offsets"] = ",".join(str(int(x)) for x in stage2_tstar_offsets)

    per_offset_stats = {}
    for offset in stage2_tstar_offsets:
        off = int(offset)
        off_rows = matched_by_offset[off]
        off_stats = summarize_matched_interval_rows(
            off_rows,
            n_true=int(n_true),
            doy_start=int(stage2_doy_start),
        )
        off_early, off_success, off_denom = early_recall80_site_year(off_rows)
        off_stats["EarlyRecall80"] = float(off_early)
        off_stats["early_recall80_success"] = int(off_success)
        off_stats["early_recall80_denominator"] = int(off_denom)
        off_stats["n_matched"] = int(len(off_rows))
        off_stats["offset_missed"] = int(missed_by_offset[off])
        per_offset_stats[off] = off_stats
        for key, value in off_stats.items():
            if isinstance(value, (int, float, np.integer, np.floating)):
                stats[f"offset{off}_{key}"] = float(value)

    weight_arr = np.asarray(stage2_tstar_offset_weights, dtype=float)
    if weight_arr.size != len(stage2_tstar_offsets):
        raise ValueError("stage2_tstar_offset_weights must have same length as stage2_tstar_offsets")
    if not np.isfinite(weight_arr).all() or float(weight_arr.sum()) <= 0:
        raise ValueError("stage2_tstar_offset_weights must be finite and sum > 0")
    weight_arr = weight_arr / float(weight_arr.sum())
    stats["stage2_tstar_offset_weights"] = ",".join(f"{float(w):.6g}" for w in weight_arr)
    weighted_metric_keys = [
        "IoU80",
        "interval_hit_f1",
        "MAE_int",
        "Mass_int",
        "pred_width",
        "EarlyRecall80",
        "post_true_start_rate",
    ]
    for key in weighted_metric_keys:
        vals = []
        weights = []
        for offset, weight in zip(stage2_tstar_offsets, weight_arr):
            val = float(per_offset_stats[int(offset)].get(key, float("nan")))
            if np.isfinite(val):
                vals.append(val)
                weights.append(float(weight))
        if vals:
            w = np.asarray(weights, dtype=float)
            w = w / float(w.sum())
            stats[f"weighted_{key}"] = float(np.dot(w, np.asarray(vals, dtype=float)))
        else:
            stats[f"weighted_{key}"] = float("nan")
    return stats


def resolve_out_path(run: int, out_root: str, out_path: str | None) -> Path:
    if out_path:
        out_path_resolved = Path(out_path)
        out_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        return out_path_resolved
    out_dir = Path(out_root) / "ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"checkpoint_run{run}.pt"


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


def parse_int_list(raw: str | None) -> list[int]:
    if raw is None or str(raw).strip() == "":
        return []
    return [int(x.strip()) for x in str(raw).split(",") if x.strip() != ""]


def parse_float_list(raw: str | None) -> list[float]:
    if raw is None or str(raw).strip() == "":
        return []
    return [float(x.strip()) for x in str(raw).split(",") if x.strip() != ""]


def main(
    pest: str,
    run: int,
    out_root: str,
    out_path: str | None,
    split_seed: int,
    split_mode: str,
    seeds: list[int] | None,
    auto_split_seed: bool,
    seed_candidates_raw: str | None,
    target_test_interval: int | None,
    tol_test_interval: int | None,
    auto_split_topk: int,
    split_seed_from_topk_idx: int | None,
    split_seeds_json: str | None,
    dropout: float | None,
    weight_decay: float | None,
    lr: float | None,
    w_interval: float | None,
    w_left: float | None,
    w_right: float | None,
    lambda_mass: float,
    stage2_entropy_lambda: float,
    stage2_entropy_conditional: int,
    stage2_location_lambda: float,
    lambda_right_late: float,
    right_late_tau: float | None,
    train_balance_ratio: str | None,
    stage2_nowcast: bool,
    stage2_nowcast_window: int,
    stage2_nowcast_stride: int,
    stage2_nowcast_tstar_start: int | None,
    stage2_nowcast_only_pre_event: int,
    stage2_nowcast_event_time_proxy: str,
    stage2_nowcast_require_tstar_before_L: int,
    stage2_causal_tstar: bool,
    stage2_tstar_layers: int,
    stage2_use_tstar_scalar_pos: int,
    stage2_early_tstar_weight_min: float,
    stage2_site_year_mean_loss: int,
    stage2_time_chunk_size: int,
    stage2_conditional_survival: int,
    stage2_lead_weighting: int,
    target_lead_min: int,
    target_lead_max: int,
    support_lead_min: int,
    support_lead_max: int,
    lead_weight_min: float,
    stage2_mass_lead_weighting: int,
    stage2_warm_start_ckpt: str | None,
    stage2_warm_start_seed: int | None,
    stage2_lead_loss_mode: str,
    stage2_lead_min: int,
    stage2_lead_max: int,
    stage2_mid_lead_min: int,
    stage2_mid_lead_max: int,
    stage2_late_exclude_days: int,
    stage2_lead_weight_1_14: float,
    stage2_lead_weight_15_29: float,
    stage2_lead_weight_30_60: float,
    stage2_lead_weight_61_75: float,
    stage2_lead_weight_gt75: float,
    stage2_best_metric: str,
    gated_val_stage1_ckpt: str | None,
    gated_val_stage1_eval_csv: str | None,
    gated_val_stage2_tstar_offset: int,
    gated_val_stage2_tstar_offsets: str | None,
    gated_val_stage2_tstar_offset_weights: str | None,
    max_epochs_override: int | None,
    patience_override: int | None,
    num_workers_override: int | None,
    batch_train_override: int | None,
    batch_eval_override: int | None,
    doy_start_override: int | None,
    doy_end_override: int | None,
    d_model_override: int | None,
    amp: int,
    amp_dtype: str,
    use_wandb: bool,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_group: str | None,
    wandb_run_name: str | None,
    wandb_tags: str | None,
    wandb_job_type: str | None,
    save_epoch_checkpoints: int,
    stage2_sanity_only: int,
    stage2_sanity_batches: int,
    stage2_pmf_mode: str = "hazard",
    stage2_pmf_sigma: float = 5.0,
    stage2_pmf_mu_max: float = 0.0,
    stage2_pmf_asym_weight: float = 10.0,
    stage2_pmf_right_weight: float = 0.3,
    stage2_pmf_target_offset: float = 0.0,
    stage2_pmf_asym_weight_early: float = 0.0,
    stage2_pmf_target_early_offset: float = 30.0,
    stage2_pmf_target_mode: str = "l_offset",
    stage2_pmf_zone_late_weight: float = 0.0,
    stage2_pmf_zone_too_late_weight: float = 0.0,
    stage2_pmf_zone_missed_weight: float = 0.0,
    stage2_pmf_zone_too_early_weight: float = 0.0,
    stage2_pmf_zone_too_late_threshold: float = 15.0,
    stage2_pmf_zone_missed_threshold: float = 22.0,
    stage2_pmf_zone_too_early_threshold: float = 23.0,
    stage2_phenology_bias_head: int = 0,
    stage2_phenology_hidden: int = 8,
    stage2_pmf_long_lead_threshold: float = 0.0,
    stage2_pmf_long_lead_weight: float = 1.0,
    stage2_pmf_right_anchor: float = 0.0,
    val_year: int = 2022,
    test_year_min: int = 2023,
    test_year_max: int = 2024,
    stage2_dispatch_feature_csv: str | None = None,
    stage2_dispatch_feature_mode: str = "causal",
    stage2_dispatch_feature_missing_value: float = 0.0,
    stage2_cohort_dispatch_only: bool = False,
    stage2_pmf_mu_mode: str = "absolute",
    stage2_pmf_lead_min: float = 7.0,
    stage2_pmf_lead_max: float = 75.0,
    stage2_pmf_clim_mid: float = 0.0,
    stage2_pmf_delta_max: float = 60.0,
    stage2_reset_head_mu: bool = False,
    stage2_aux_lead_lambda: float = 0.0,
    stage2_aux_lead_huber_delta: float = 10.0,
    stage2_gaussian_loss_mode: str = "asym_mse",
    stage2_gaussian_interval_continuity_correction: int = 0,
    stage2_gaussian_interval_lambda: float = 0.1,
    stage2_add_neighbor_history: bool = False,
    stage2_neighbor_decay_km: float = 20.0,
):
    _, get_feature_cols = resolve_pest(pest)
    if not out_root:
        out_root = default_out_root(pest)
    ensure_output_dirs(out_root)
    if max_epochs_override is not None:
        C.MAX_EPOCHS = int(max_epochs_override)
    if patience_override is not None:
        C.PATIENCE = int(patience_override)
    if num_workers_override is not None:
        C.NUM_WORKERS = int(num_workers_override)
    if batch_train_override is not None:
        C.BATCH_TRAIN = int(batch_train_override)
    if batch_eval_override is not None:
        C.BATCH_EVAL = int(batch_eval_override)
    if doy_start_override is not None:
        C.DOY_START = int(doy_start_override)
    if doy_end_override is not None:
        C.DOY_END = int(doy_end_override)
    if int(C.DOY_START) > int(C.DOY_END):
        raise ValueError("--doy_start_override must be <= --doy_end_override")
    if d_model_override is not None:
        C.D_MODEL = int(d_model_override)
    if int(C.D_MODEL) <= 0:
        raise ValueError("D_MODEL must be >= 1")
    if int(C.D_MODEL) % int(C.N_HEAD) != 0:
        raise ValueError(f"D_MODEL ({C.D_MODEL}) must be divisible by N_HEAD ({C.N_HEAD})")
    gated_val_offsets = parse_int_list(gated_val_stage2_tstar_offsets)
    if not gated_val_offsets:
        gated_val_offsets = [int(gated_val_stage2_tstar_offset)]
    gated_val_offset_weights = parse_float_list(gated_val_stage2_tstar_offset_weights)
    if not gated_val_offset_weights:
        gated_val_offset_weights = [1.0 for _ in gated_val_offsets]
    if len(gated_val_offset_weights) != len(gated_val_offsets):
        raise ValueError("--gated_val_stage2_tstar_offset_weights length must match --gated_val_stage2_tstar_offsets")
    weight_sum = float(np.sum(np.asarray(gated_val_offset_weights, dtype=float)))
    if not np.isfinite(weight_sum) or weight_sum <= 0:
        raise ValueError("--gated_val_stage2_tstar_offset_weights must sum to a positive finite value")
    gated_val_offset_weights = [float(w) / weight_sum for w in gated_val_offset_weights]

    print(
        "Effective DataLoader config: "
        f"BATCH_TRAIN={C.BATCH_TRAIN}, BATCH_EVAL={C.BATCH_EVAL}, "
        f"NUM_WORKERS={C.NUM_WORKERS}, PIN_MEMORY={C.PIN_MEMORY}, "
        f"PERSISTENT_WORKERS={C.PERSISTENT_WORKERS}, PREFETCH_FACTOR={C.PREFETCH_FACTOR}"
    )
    print(
        f"Effective Model/Data config: D_MODEL={C.D_MODEL}, N_HEAD={C.N_HEAD}, N_LAYERS={C.N_LAYERS}, "
        f"DOY_START={C.DOY_START}, DOY_END={C.DOY_END}"
    )
    print(
        f"Effective Best-Epoch Metric: {stage2_best_metric}, "
        f"train_balance_ratio={train_balance_ratio}, PI_METHOD={getattr(C, 'PI_METHOD', 'shortest')}"
    )
    print(f"Effective Split config: split_mode={split_mode}, split_seed={split_seed}")
    if stage2_causal_tstar and (not stage2_nowcast):
        print("[stage2_causal_tstar] disabled because --stage2_nowcast is off")
    if stage2_tstar_layers < 0:
        raise ValueError("--stage2_tstar_layers must be >= 0")
    if stage2_early_tstar_weight_min <= 0.0 or stage2_early_tstar_weight_min > 1.0:
        raise ValueError("--stage2_early_tstar_weight_min must be in (0, 1]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    use_amp = bool(int(amp)) and (device.type == "cuda")
    if bool(int(amp)) and device.type != "cuda":
        print("[amp] requested but CUDA is not available; AMP disabled")
    if amp_dtype not in ("bf16", "fp16"):
        raise ValueError("--amp_dtype must be one of: bf16, fp16")
    print(f"[amp] enabled={int(use_amp)} dtype={amp_dtype}")
    grouped_mode = bool(stage2_nowcast and stage2_causal_tstar)
    if (not grouped_mode) and (
        stage2_tstar_layers != 1
        or int(stage2_use_tstar_scalar_pos) != 0
        or abs(float(stage2_early_tstar_weight_min) - 1.0) > 1e-12
        or int(stage2_site_year_mean_loss) != 0
    ):
        print("[stage2_causal_tstar] grouped-only tuning args are ignored because grouped mode is off")

    wandb_run = init_wandb_run(
        use_wandb=use_wandb,
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        group=wandb_group,
        job_type=wandb_job_type,
        tags=parse_tags(wandb_tags) + [f"pest:{pest}", "script:run_train"],
        config={
            "pest": pest,
            "run": run,
            "split_seed": split_seed,
            "split_mode": split_mode,
            "stage2_nowcast": bool(stage2_nowcast),
            "stage2_nowcast_window": int(stage2_nowcast_window),
            "stage2_nowcast_stride": int(stage2_nowcast_stride),
            "stage2_nowcast_tstar_start": None if stage2_nowcast_tstar_start is None else int(stage2_nowcast_tstar_start),
            "stage2_nowcast_only_pre_event": int(stage2_nowcast_only_pre_event),
            "stage2_nowcast_event_time_proxy": stage2_nowcast_event_time_proxy,
            "stage2_nowcast_require_tstar_before_L": int(stage2_nowcast_require_tstar_before_L),
            "stage2_causal_tstar": bool(grouped_mode),
            "stage2_tstar_layers": int(stage2_tstar_layers),
            "stage2_use_tstar_scalar_pos": int(stage2_use_tstar_scalar_pos),
            "stage2_early_tstar_weight_min": float(stage2_early_tstar_weight_min),
            "stage2_site_year_mean_loss": int(stage2_site_year_mean_loss),
            "stage2_conditional_survival": int(bool(stage2_conditional_survival)),
            "stage2_entropy_lambda": float(stage2_entropy_lambda),
            "stage2_entropy_conditional": int(stage2_entropy_conditional),
            "stage2_location_lambda": float(stage2_location_lambda),
            "doy_start": int(C.DOY_START),
            "doy_end": int(C.DOY_END),
            "d_model": int(C.D_MODEL),
            "n_head": int(C.N_HEAD),
            "n_layers": int(C.N_LAYERS),
            "amp": int(bool(amp)),
            "amp_dtype": str(amp_dtype),
            "max_epochs": int(C.MAX_EPOCHS),
            "patience": int(C.PATIENCE),
            "num_workers": int(C.NUM_WORKERS),
            "batch_train": int(C.BATCH_TRAIN),
            "batch_eval": int(C.BATCH_EVAL),
        },
    )

    # =========================
    # 1) DAILY: load + GDD merge + rolling
    # =========================
    t0 = time.perf_counter()
    daily = load_daily_preprocessed(C.PATH_DAILY)
    print(f"[time] load_daily+gdd+roll={time.perf_counter()-t0:.2f}s")

    # =========================
    # 2) OBS: load + aggregate
    # =========================
    t0 = time.perf_counter()
    obs = load_obs(C.PATH_OBS)
    obs2 = aggregate_obs_daily_max(obs)
    print(f"[time] load_obs+aggregate={time.perf_counter()-t0:.2f}s")

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
    t0 = time.perf_counter()
    df_season = slice_season(train_df, C.DOY_START, C.DOY_END)
    # Phenology-bias-head: pull static phenology vector per site-year (separate
    # from the time-series X). Columns must exist in train_df after merge_pheno_daily_ffill.
    pheno_ext_cols_default = ["best_suitability", "best_months", "offset_days", "window_idx"]
    pheno_ext_cols = (
        [c for c in pheno_ext_cols_default if c in df_season.columns]
        if bool(int(stage2_phenology_bias_head))
        else None
    )
    if pheno_ext_cols is not None:
        print(f"[phenobias] pheno_ext_cols loaded: {pheno_ext_cols}")
    samples, dropped, feature_names = build_samples_season(
        df_season, feature_cols, C.DOY_START, C.DOY_END,
        pheno_ext_cols=pheno_ext_cols,
    )
    print("samples:", len(samples), "| dropped groups (len!=T):", dropped)
    print(f"[time] build_samples_season={time.perf_counter()-t0:.2f}s")
    print(f"[features] n={len(feature_names)} head={feature_names[:5]} tail={feature_names[-5:]}")

    # ---- VENDORED: weather-anomaly channels (AFTER base, BEFORE neighbor) ----
    # anomaly = base - per-DOY climatology fit on TRAIN years (< val_year) only (leakage-safe).
    # Inserted here so neighbor/dispatch stay at the tail (norm raw-forcing unchanged) and the
    # anomaly channels get STANDARDIZED like base weather. Controlled by env WBPH_ANOMALY_COLS.
    import os as _os
    stage2_anomaly_idx0 = -1      # VENDORED: v2_scaled anomaly index range -> raw-forced in norm (-1 if unused)
    stage2_anomaly_n = 0
    if _os.environ.get("WBPH_ANOMALY_COLS"):
        _sel = _os.environ["WBPH_ANOMALY_COLS"].strip()
        _mode = _os.environ.get("WBPH_ANOMALY_MODE", "v1").strip()
        _Tan = int(samples[0]["X"].shape[0])
        _train_an = [s for s in samples if int(s["year"]) < int(val_year)]   # leakage-safe fit
        _d0 = int(samples[0]["X"].shape[1])
        if _mode == "station_ts":
            # station-level clim fit from FULL daily weather (year<val_year, ~23yr/station).
            # Tier A/B/C via WBPH_STATION_TIER; all channels CAUSAL, standardized+clipped, raw-forced.
            from src.features.station_ts_anomaly import (fit_station_clim, fit_tier_standardizer,
                                                         apply_tier, tier_channel_names, save_spec)
            _TEMP_RH = ["tmean_7d_mean", "tmax_7d_max", "tmin_7d_min", "rh_7d_mean"]
            _cols = (_TEMP_RH if _sel in ("temp_rh", "temp_rh_4ch", "core")
                     else [c.strip() for c in _sel.split(",")])
            _tier = _os.environ.get("WBPH_STATION_TIER", "A").strip().upper()
            _clip = float(_os.environ.get("WBPH_ANOMALY_CLIP", "5.0"))
            _names = tier_channel_names(_cols, _tier)
            _sites = {str(s["site_id"]) for s in samples}
            _spec = fit_station_clim(daily, _cols, int(val_year), int(C.DOY_START), int(C.DOY_END),
                                     site_ids=_sites)
            _spec = fit_tier_standardizer(_train_an, _spec, list(feature_names), _tier)
            _sout = _os.environ.get("WBPH_ANOMALY_SPEC_OUT", "")
            if _sout:
                import os as _os2; _os2.makedirs(_os2.path.dirname(_sout), exist_ok=True); save_spec(_spec, _sout)
            _stack = []; _nglob = 0
            for s in samples:
                if str(s["site_id"]) not in _spec["stations"]:
                    _nglob += 1
                a = apply_tier(s["X"], s["site_id"], _spec, list(feature_names), _tier, clip=_clip)
                s["X"] = np.concatenate([np.asarray(s["X"]), a.astype(np.asarray(s["X"]).dtype)], axis=1)
                _stack.append(a)
            feature_names = list(feature_names) + [f"st{_tier}_{n}" for n in _names]
            stage2_anomaly_idx0 = _d0; stage2_anomaly_n = len(_names)
            _A = np.concatenate(_stack, axis=0)
            _nan = int(np.isnan(_A).sum()); _inf = int(np.isinf(_A).sum()); _zmax = float(np.abs(_A).max())
            print(f"[station_ts] tier={_tier} appended {len(_names)} ch at idx[{_d0}:{_d0+len(_names)}] "
                  f"d_in {_d0}->{samples[0]['X'].shape[1]} cols={_cols} clip=±{_clip} clim=year<{int(val_year)} "
                  f"n_stations={len(_spec['stations'])} samples_via_global_fallback={_nglob}/{len(samples)} "
                  f"spec_out={_sout or '(not saved)'}")
            print(f"[station_ts] POST-SCALE (==post-norm, raw-forced) per-channel:")
            for _k, _n in enumerate(_names):
                _v = _A[:, _k]
                print(f"    [{_d0+_k:2d}] {_n:18s} mean={_v.mean():+.3f} std={_v.std():.3f} "
                      f"min={_v.min():+.2f} max={_v.max():+.2f} |z|max={np.abs(_v).max():.2f}")
            print(f"[station_ts] NaN={_nan} inf={_inf} global|z|max={_zmax:.2f} "
                  f"{'OK' if (_nan==0 and _inf==0 and _zmax<10.0) else '*** WARNING: check scale/NaN ***'}")
        elif _mode == "v2_scaled":
            # per-DOY clim (rain via log1p) -> TRAIN-standardize -> clip[-5,5]; raw-forced in norm.
            from src.features.weather_anomaly import (CORE_COLS, ALL_COLS,
                                                      fit_anomaly_spec, apply_anomaly_spec, save_spec)
            _cols = (CORE_COLS if _sel == "core" else ALL_COLS if _sel == "all"
                     else [c.strip() for c in _sel.split(",")])
            _clip = float(_os.environ.get("WBPH_ANOMALY_CLIP", "5.0"))
            _spec = fit_anomaly_spec(_train_an, list(feature_names), _cols, _Tan)
            _sout = _os.environ.get("WBPH_ANOMALY_SPEC_OUT", "")
            if _sout:
                import os as _os2; _os2.makedirs(_os2.path.dirname(_sout), exist_ok=True); save_spec(_spec, _sout)
            _stack = []
            for s in samples:
                a = apply_anomaly_spec(s["X"], _spec, list(feature_names), clip=_clip)
                s["X"] = np.concatenate([np.asarray(s["X"]), a.astype(np.asarray(s["X"]).dtype)], axis=1)
                _stack.append(a)
            feature_names = list(feature_names) + [f"anomz_{c}" for c in _cols]
            stage2_anomaly_idx0 = _d0; stage2_anomaly_n = len(_cols)
            _A = np.concatenate(_stack, axis=0)   # (N*T, n_cols) — already ~unit + clipped == post-norm (raw-forced)
            _nan = int(np.isnan(_A).sum()); _inf = int(np.isinf(_A).sum()); _zmax = float(np.abs(_A).max())
            print(f"[weather_anomaly:v2_scaled] appended {len(_cols)} ch at idx[{_d0}:{_d0+len(_cols)}] "
                  f"d_in {_d0}->{samples[0]['X'].shape[1]} cols={_cols} clip=±{_clip} clim_train_n={len(_train_an)} "
                  f"spec_out={_sout or '(not saved)'}")
            print(f"[weather_anomaly:v2_scaled] POST-SCALE (==post-norm, raw-forced) per-channel:")
            for _k, _c in enumerate(_cols):
                _v = _A[:, _k]
                print(f"    [{_d0+_k:2d}] {_c:16s} mean={_v.mean():+.3f} std={_v.std():.3f} "
                      f"min={_v.min():+.2f} max={_v.max():+.2f} |z|max={np.abs(_v).max():.2f}")
            print(f"[weather_anomaly:v2_scaled] NaN={_nan} inf={_inf} global|z|max={_zmax:.2f} "
                  f"{'OK' if (_nan==0 and _inf==0 and _zmax<10.0) else '*** WARNING: check scale/NaN ***'}")
        else:
            from src.features.weather_anomaly import (CORE_COLS, ALL_COLS,
                                                      fit_doy_climatology, compute_anomaly_channels, save_clim)
            _cols = (CORE_COLS if _sel == "core" else ALL_COLS if _sel == "all"
                     else [c.strip() for c in _sel.split(",")])
            _clim = fit_doy_climatology(_train_an, list(feature_names), _cols, _Tan)
            _cout = _os.environ.get("WBPH_ANOMALY_CLIM_OUT", "")
            if _cout:
                import os as _os2; _os2.makedirs(_os2.path.dirname(_cout), exist_ok=True); save_clim(_clim, _cols, _cout)
            for s in samples:
                a = compute_anomaly_channels(s["X"], _clim, list(feature_names), _cols)
                s["X"] = np.concatenate([np.asarray(s["X"]), a], axis=1)
            feature_names = list(feature_names) + [f"anom_{c}" for c in _cols]
            print(f"[weather_anomaly:v1] appended {len(_cols)} anomaly ch after base: d_in {_d0}->{samples[0]['X'].shape[1]} "
                  f"cols={_cols} clim_train_n={len(_train_an)} clim_out={_cout or '(not saved)'}")

    # ---- VENDORED: phenology / thermal-time GDD channels (AFTER base, BEFORE neighbor) ----
    # P0a raw_gdd (1ch, z-std cumulative GDD10_since_gs) | P1a station_gdd_clim (2ch, per-(station,DOY)
    # anomaly-z + empirical percentile). CAUSAL; clim/std fit on year<val_year; RAW-forced in norm via
    # stage2_anomaly_idx0/n (mutually exclusive with WBPH_ANOMALY_*). OFF by default (byte-identical when
    # WBPH_PHENO_MODE unset). Controlled by env WBPH_PHENO_MODE / WBPH_PHENO_CLIP / WBPH_PHENO_SPEC_OUT.
    if _os.environ.get("WBPH_PHENO_MODE"):
        if stage2_anomaly_idx0 >= 0:
            raise SystemExit("[abort] WBPH_PHENO_MODE and WBPH_ANOMALY_COLS are mutually exclusive")
        from src.features import phenology_relative as _PH
        _pm = _os.environ["WBPH_PHENO_MODE"].strip()
        _clip = float(_os.environ.get("WBPH_PHENO_CLIP", "5.0"))
        _d0 = int(samples[0]["X"].shape[1])
        _train_ph = [s for s in samples if int(s["year"]) < int(val_year)]   # leakage-safe fit
        _lk = _PH.build_gdd_lookup(daily, int(C.DOY_START), int(C.DOY_END))
        if _pm == "raw_gdd":
            _spec = _PH.fit_raw_gdd_standardizer(_train_ph, _lk)
            _apply = lambda s: _PH.apply_raw_gdd(str(s["site_id"]), int(s["year"]), _lk, _spec, clip=_clip)
        elif _pm == "station_gdd_clim":
            _sites = {str(s["site_id"]) for s in samples}
            _spec = _PH.fit_station_gdd_clim(daily, int(val_year), int(C.DOY_START), int(C.DOY_END), site_ids=_sites)
            _spec = _PH.fit_station_gdd_std(_train_ph, _spec, _lk)
            _apply = lambda s: _PH.apply_station_gdd(str(s["site_id"]), int(s["year"]), _lk, _spec, clip=_clip)
        else:
            raise SystemExit(f"[abort] unknown WBPH_PHENO_MODE={_pm} (raw_gdd|station_gdd_clim)")
        _names = _PH.pheno_channel_names(_pm)
        _stack = []
        for s in samples:
            a = _apply(s)
            s["X"] = np.concatenate([np.asarray(s["X"]), a.astype(np.asarray(s["X"]).dtype)], axis=1)
            _stack.append(a)
        feature_names = list(feature_names) + [f"ph_{n}" for n in _names]
        stage2_anomaly_idx0 = _d0; stage2_anomaly_n = len(_names)
        _sout = _os.environ.get("WBPH_PHENO_SPEC_OUT", "")
        if _sout:
            import os as _os3; _os3.makedirs(_os3.path.dirname(_sout), exist_ok=True)
            _PH.save_pheno_spec(_spec, _lk, _sout)
        _A = np.concatenate(_stack, axis=0)
        _nan = int(np.isnan(_A).sum()); _inf = int(np.isinf(_A).sum()); _zmax = float(np.abs(_A).max())
        print(f"[phenology:{_pm}] appended {len(_names)} ch at idx[{_d0}:{_d0+len(_names)}] "
              f"d_in {_d0}->{samples[0]['X'].shape[1]} names={_names} clip=±{_clip} clim=year<{int(val_year)} "
              f"spec_out={_sout or '(not saved)'}")
        print(f"[phenology:{_pm}] POST-SCALE (==post-norm, raw-forced) per-channel:")
        for _k, _n in enumerate(_names):
            _v = _A[:, _k]
            print(f"    [{_d0+_k:2d}] ph_{_n:16s} mean={_v.mean():+.3f} std={_v.std():.3f} "
                  f"min={_v.min():+.2f} max={_v.max():+.2f} |z|max={np.abs(_v).max():.2f}")
        print(f"[phenology:{_pm}] NaN={_nan} inf={_inf} global|z|max={_zmax:.2f} "
              f"{'OK' if (_nan==0 and _inf==0 and _zmax<10.0) else '*** WARNING: check scale/NaN ***'}")

    # ---- Optional Stage-2 DIRECT neighbor occurrence features (6 channels) ----
    # Appended right after build_samples_season and BEFORE dispatch features so
    # train + eval share one channel order: base -> neighbor -> dispatch. Uses the
    # same util + strict obs_doy < t leakage guard as Stage-1. OFF by default ->
    # production baseline is byte-identical when the flag is not passed.
    stage2_neighbor_added = False
    stage2_neighbor_feature_names: list[str] = []
    if bool(stage2_add_neighbor_history):
        from rice.scripts.neighbor_history_utils import (
            load_long_events, build_neighbor_index, append_neighbor_to_samples,
            NEIGHBOR_CHANNEL_NAMES, NEIGHBOR_FEATURE_DIM,
        )
        d_in_before_nb = int(samples[0]["X"].shape[1])
        ev_df, co_df, _nb_sy = load_long_events(
            C.PATH_OBS, label_col=getattr(C, "LABEL_COL", "label_event"),
            year_min=getattr(C, "YEAR_MIN", None), year_max=getattr(C, "YEAR_MAX", None),
        )
        nb_index = build_neighbor_index(ev_df, co_df)
        append_neighbor_to_samples(
            samples, nb_index, doy_start=int(C.DOY_START),
            decay_km=float(stage2_neighbor_decay_km),
        )
        feature_names = list(feature_names) + list(NEIGHBOR_CHANNEL_NAMES)
        stage2_neighbor_added = True
        stage2_neighbor_feature_names = list(NEIGHBOR_CHANNEL_NAMES)
        d_in_after_nb = int(samples[0]["X"].shape[1])
        print(f"[stage2_neighbor] before_d_in={d_in_before_nb}  added={NEIGHBOR_FEATURE_DIM}  "
              f"after_d_in={d_in_after_nb}  decay_km={float(stage2_neighbor_decay_km)}")
        print(f"[stage2_neighbor] channels={NEIGHBOR_CHANNEL_NAMES}")

    # ---- Optional Stage-1 dispatch confidence features (causal-fill, 15 chans)
    stage2_pmf_alert_tstar_feat_idx = -1
    if stage2_dispatch_feature_csv:
        from rice.scripts.stage1_confidence_utils import (
            load_dispatch_feature_table, append_dispatch_confidence_to_samples,
            DISPATCH_CHANNEL_NAMES, DISPATCH_TOTAL_CHANNELS,
        )
        conf_map = load_dispatch_feature_table(stage2_dispatch_feature_csv)
        # Phase B: optional cohort restriction to dispatch-alerted site-years.
        # Done BEFORE append so feature appending stays correct on the kept set.
        if bool(stage2_cohort_dispatch_only):
            sy_in_conf = set(conf_map.keys())
            n_before = len(samples)
            samples = [s for s in samples
                       if (str(s["site_id"]), int(s["year"])) in sy_in_conf]
            print(f"[cohort_dispatch_only] filtered samples {n_before} -> {len(samples)} "
                  f"(kept site-years in conf_map; needed for lead_from_alert)")
            if not samples:
                raise SystemExit("[abort] cohort_dispatch_only left 0 samples")
        # Capture base d_in BEFORE append so alert_tstar channel index is known.
        base_d_in = int(samples[0]["X"].shape[1])
        stats = append_dispatch_confidence_to_samples(
            samples,
            conf_map,
            doy_start=int(C.DOY_START),
            mode=str(stage2_dispatch_feature_mode),
            missing_value=float(stage2_dispatch_feature_missing_value),
        )
        # DISPATCH_FEATURE_NAMES[0] = 'alert_tstar' -> appended at channel index = base_d_in.
        stage2_pmf_alert_tstar_feat_idx = base_d_in
        feature_names = list(feature_names) + DISPATCH_CHANNEL_NAMES
        print(f"[dispatch_features] csv={stage2_dispatch_feature_csv}  "
              f"mode={stats['mode']}  n_with_alert={stats['n_with_alert']}  "
              f"n_no_alert={stats['n_no_alert']}  "
              f"added_channels={stats['added_channels']}  "
              f"rows_with_feature={stats['n_rows_with_feature']}  "
              f"rows_missing={stats['n_rows_missing']}  "
              f"new_feature_count={len(feature_names)}  "
              f"alert_tstar_feat_idx={stage2_pmf_alert_tstar_feat_idx}  "
              f"cohort_dispatch_only={bool(stage2_cohort_dispatch_only)}")
    elif bool(stage2_cohort_dispatch_only):
        raise SystemExit("[abort] --stage2_cohort_dispatch_only requires "
                          "--stage2_dispatch_feature_csv (need conf_map to filter)")
    # --- VENDORED: prior-residual alert_bin -> append a DEDICATED prior_mu channel (DOY) ---
    stage2_prior_mu_feat_idx = -1
    if str(stage2_pmf_mu_mode) == "prior_residual_alert_bin":
        import os as _os, json as _json
        _pt = _os.environ.get("WBPH_PRIOR_TABLE", "")
        if not _pt:
            raise SystemExit("[abort] prior_residual_alert_bin requires env WBPH_PRIOR_TABLE=<prior_table.json>")
        if not stage2_dispatch_feature_csv:
            raise SystemExit("[abort] prior_residual_alert_bin requires --stage2_dispatch_feature_csv (alert source)")
        _tab = _json.loads(open(_pt).read())
        _ABINS = [(-1e9,140,"<140"),(140,160,"140-160"),(160,180,"160-180"),(180,200,"180-200"),(200,1e9,">=200")]
        def _abin(a):
            for lo,hi,lab in _ABINS:
                if lo<=a<hi: return lab
            return ">=200"
        def _pmu(a):
            return float(a) + float(_tab["bin_lead"].get(_abin(float(a)), _tab["global_lead"]))
        stage2_prior_mu_feat_idx = int(samples[0]["X"].shape[1])   # append at tail (after dispatch)
        _nmiss = 0
        for s in samples:
            info = conf_map.get((str(s["site_id"]), int(s["year"])))
            if info is None:
                _nmiss += 1; alert_doy = float(C.DOY_START)
            else:
                alert_doy = float(info["alert_tstar_doy"])
            Xc = s["X"]
            col = np.full((Xc.shape[0], 1), _pmu(alert_doy), dtype=Xc.dtype)
            s["X"] = np.concatenate([Xc, col], axis=1)
        feature_names = list(feature_names) + ["prior_mu_alert_bin"]
        print(f"[prior_residual] prior_mu channel idx={stage2_prior_mu_feat_idx} table={_pt} "
              f"global_lead={_tab['global_lead']:.2f} n_missing_conf={_nmiss} new_d_in={samples[0]['X'].shape[1]}")

    if str(stage2_pmf_mu_mode) == "lead_from_alert" and stage2_pmf_alert_tstar_feat_idx < 0:
        raise SystemExit("[abort] --stage2_pmf_mu_mode=lead_from_alert requires "
                          "--stage2_dispatch_feature_csv (need alert_tstar channel)")
    if str(stage2_pmf_mu_mode) == "residual_clim" and float(stage2_pmf_clim_mid) <= 0.0:
        raise SystemExit("[abort] --stage2_pmf_mu_mode=residual_clim requires "
                          "--stage2_pmf_clim_mid > 0 (per-pest climatology mean_mid in DOY units)")

    # =========================
    # 8) Split + norm + datasets
    # =========================
    result = None
    chosen = None
    if split_seeds_json is not None:
        split_seeds_json_path = resolve_split_seeds_json_path(out_root, split_seeds_json)
        split_seed, chosen_idx, chosen, payload = load_split_seed_from_topk(split_seeds_json_path, split_seed_from_topk_idx)
        train_s, val_s, test_s = split_samples(
            samples,
            val_frac=0.1,
            test_frac=0.1,
            seed=split_seed,
            split_mode=split_mode,
            val_year=val_year,
            test_year_min=test_year_min,
            test_year_max=test_year_max,
        )
        print(
            f"[split_seed_json] selected seed={split_seed} idx={chosen_idx} "
            f"file={split_seeds_json_path}"
        )
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
            split_mode=split_mode,
            val_year=val_year,
            test_year_min=test_year_min,
            test_year_max=test_year_max,
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
        train_s, val_s, test_s = split_samples(
            samples,
            val_frac=0.1,
            test_frac=0.1,
            seed=split_seed,
            split_mode=split_mode,
            val_year=val_year,
            test_year_min=test_year_min,
            test_year_max=test_year_max,
        )
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
        train_s, val_s, test_s = split_samples(
            samples,
            val_frac=0.1,
            test_frac=0.1,
            seed=split_seed,
            split_mode=split_mode,
            val_year=val_year,
            test_year_min=test_year_min,
            test_year_max=test_year_max,
        )

    log_split_sanity("base", train_s, val_s, test_s, split_mode=split_mode)

    # split stats
    if stage2_nowcast:
        train_s = build_stage2_nowcast_samples(
            train_s,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
            require_tstar_before_L=bool(stage2_nowcast_require_tstar_before_L),
        )
        val_s = build_stage2_nowcast_samples(
            val_s,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
            require_tstar_before_L=bool(stage2_nowcast_require_tstar_before_L),
        )
        test_s = build_stage2_nowcast_samples(
            test_s,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
            require_tstar_before_L=bool(stage2_nowcast_require_tstar_before_L),
        )
        print(
            f"[stage2_nowcast] window={stage2_nowcast_window} stride={stage2_nowcast_stride} "
            f"tstar_start={stage2_nowcast_tstar_start} only_pre_event={bool(stage2_nowcast_only_pre_event)} "
            f"event_time_proxy={stage2_nowcast_event_time_proxy} label_mode=orig | "
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

    # split stats
    tr = _split_stats(train_s)
    va = _split_stats(val_s)
    te = _split_stats(test_s)
    print(
        f"[split] train sites={tr['n_sites']} samples={tr['n_samples']} event_rate={tr['event_rate']:.4f} | "
        f"val sites={va['n_sites']} samples={va['n_samples']} event_rate={va['event_rate']:.4f} | "
        f"test sites={te['n_sites']} samples={te['n_samples']} event_rate={te['event_rate']:.4f}"
    )
    if auto_split_seed:
        n_int, mean_int, var_int = _interval_len_stats(test_s)
        print(f"[split] test interval length: n={n_int} mean={mean_int:.2f} var={var_int:.2f}")

    log_split_fingerprint("train", train_s, val_s, test_s)
    print("split:", len(train_s), len(val_s), len(test_s), "| unique sites:", len({s["site_id"] for s in samples}))

    if dropout is not None:
        C.DROPOUT = float(dropout)
    if weight_decay is not None:
        C.WEIGHT_DECAY = float(weight_decay)
    if lr is not None:
        C.LR = float(lr)
    if w_interval is not None:
        C.W_INTERVAL = float(w_interval)
    if w_left is not None:
        C.W_LEFT = float(w_left)
    if w_right is not None:
        C.W_RIGHT = float(w_right)
    print(
        f"[hparams] dropout={C.DROPOUT} weight_decay={C.WEIGHT_DECAY} lr={C.LR} "
        f"w_interval={C.W_INTERVAL} w_left={C.W_LEFT} w_right={C.W_RIGHT} "
        f"lambda_mass={lambda_mass} stage2_entropy_lambda={stage2_entropy_lambda} "
        f"stage2_entropy_conditional={stage2_entropy_conditional} "
        f"stage2_location_lambda={stage2_location_lambda} "
        f"lambda_right_late={lambda_right_late} right_late_tau={right_late_tau}"
    )
    if grouped_mode:
        print(
            f"[stage2_causal_tstar] tstar_layers={int(stage2_tstar_layers)} "
            f"use_tstar_scalar_pos={int(bool(stage2_use_tstar_scalar_pos))} "
            f"early_tstar_weight_min={float(stage2_early_tstar_weight_min):.3f} "
            f"site_year_mean_loss={int(bool(stage2_site_year_mean_loss))} "
            f"conditional_survival={int(bool(stage2_conditional_survival))}"
        )

    hparams = {
        "run": run,
        "pest": pest,
        "split_seed": split_seed,
        "split_mode": str(split_mode),
        "split_seed_from_topk_idx": split_seed_from_topk_idx,
        "train_seeds": seeds if seeds is not None else C.SEEDS,
        "dropout": C.DROPOUT,
        "weight_decay": C.WEIGHT_DECAY,
        "lr": C.LR,
        "w_interval": C.W_INTERVAL,
        "w_left": C.W_LEFT,
        "w_right": C.W_RIGHT,
        "lambda_mass": lambda_mass,
        "stage2_entropy_lambda": float(stage2_entropy_lambda),
        "stage2_entropy_conditional": int(stage2_entropy_conditional),
        "stage2_location_lambda": float(stage2_location_lambda),
        "lambda_right_late": lambda_right_late,
        "right_late_tau": right_late_tau,
        "stage2_nowcast": bool(stage2_nowcast),
        "stage2_nowcast_window": int(stage2_nowcast_window),
        "stage2_nowcast_stride": int(stage2_nowcast_stride),
        "stage2_nowcast_tstar_start": None if stage2_nowcast_tstar_start is None else int(stage2_nowcast_tstar_start),
        "stage2_nowcast_only_pre_event": int(stage2_nowcast_only_pre_event),
        "stage2_nowcast_event_time_proxy": stage2_nowcast_event_time_proxy,
        "stage2_nowcast_require_tstar_before_L": int(stage2_nowcast_require_tstar_before_L),
        "stage2_causal_tstar": bool(grouped_mode),
        "stage2_tstar_layers": int(stage2_tstar_layers),
        "stage2_use_tstar_scalar_pos": int(stage2_use_tstar_scalar_pos),
        "stage2_early_tstar_weight_min": float(stage2_early_tstar_weight_min),
        "stage2_site_year_mean_loss": int(stage2_site_year_mean_loss),
        "stage2_time_chunk_size": int(stage2_time_chunk_size),
        "stage2_conditional_survival": int(bool(stage2_conditional_survival)),
        "stage2_lead_weighting": int(stage2_lead_weighting),
        "target_lead_min": int(target_lead_min),
        "target_lead_max": int(target_lead_max),
        "support_lead_min": int(support_lead_min),
        "support_lead_max": int(support_lead_max),
        "lead_weight_min": float(lead_weight_min),
        "stage2_mass_lead_weighting": int(stage2_mass_lead_weighting),
        "stage2_warm_start_ckpt": stage2_warm_start_ckpt,
        "stage2_warm_start_seed": None if stage2_warm_start_seed is None else int(stage2_warm_start_seed),
        "stage2_lead_loss_mode": str(stage2_lead_loss_mode),
        "stage2_lead_min": int(stage2_lead_min),
        "stage2_lead_max": int(stage2_lead_max),
        "stage2_mid_lead_min": int(stage2_mid_lead_min),
        "stage2_mid_lead_max": int(stage2_mid_lead_max),
        "stage2_late_exclude_days": int(stage2_late_exclude_days),
        "stage2_lead_weight_1_14": float(stage2_lead_weight_1_14),
        "stage2_lead_weight_15_29": float(stage2_lead_weight_15_29),
        "stage2_lead_weight_30_60": float(stage2_lead_weight_30_60),
        "stage2_lead_weight_61_75": float(stage2_lead_weight_61_75),
        "stage2_lead_weight_gt75": float(stage2_lead_weight_gt75),
        "stage2_best_metric": str(stage2_best_metric),
        "stage2_pmf_mode": str(stage2_pmf_mode),
        "stage2_pmf_sigma": float(stage2_pmf_sigma),
        "stage2_pmf_mu_max": float(stage2_pmf_mu_max),
        "stage2_pmf_asym_weight": float(stage2_pmf_asym_weight),
        "stage2_pmf_right_weight": float(stage2_pmf_right_weight),
        "stage2_pmf_target_offset": float(stage2_pmf_target_offset),
        "stage2_pmf_asym_weight_early": float(stage2_pmf_asym_weight_early),
        "stage2_pmf_target_early_offset": float(stage2_pmf_target_early_offset),
        "stage2_pmf_target_mode": str(stage2_pmf_target_mode),
        "stage2_pmf_zone_late_weight": float(stage2_pmf_zone_late_weight),
        "stage2_pmf_zone_too_late_weight": float(stage2_pmf_zone_too_late_weight),
        "stage2_pmf_zone_missed_weight": float(stage2_pmf_zone_missed_weight),
        "stage2_pmf_zone_too_early_weight": float(stage2_pmf_zone_too_early_weight),
        "stage2_pmf_zone_too_late_threshold": float(stage2_pmf_zone_too_late_threshold),
        "stage2_pmf_zone_missed_threshold": float(stage2_pmf_zone_missed_threshold),
        "stage2_pmf_zone_too_early_threshold": float(stage2_pmf_zone_too_early_threshold),
        "stage2_phenology_bias_head": int(stage2_phenology_bias_head),
        "stage2_phenology_hidden": int(stage2_phenology_hidden),
        "stage2_dispatch_features_added": bool(stage2_dispatch_feature_csv),
        "stage2_dispatch_feature_csv": (str(stage2_dispatch_feature_csv)
                                          if stage2_dispatch_feature_csv else None),
        "stage2_dispatch_feature_mode": str(stage2_dispatch_feature_mode),
        "stage2_dispatch_feature_missing_value": float(stage2_dispatch_feature_missing_value),
        "stage2_cohort_dispatch_only": bool(stage2_cohort_dispatch_only),
        "stage2_pmf_mu_mode": str(stage2_pmf_mu_mode),
        "stage2_pmf_lead_min": float(stage2_pmf_lead_min),
        "stage2_pmf_lead_max": float(stage2_pmf_lead_max),
        "stage2_pmf_clim_mid": float(stage2_pmf_clim_mid),
        "stage2_pmf_delta_max": float(stage2_pmf_delta_max),
        "stage2_pmf_alert_tstar_feat_idx": int(stage2_pmf_alert_tstar_feat_idx),
        "stage2_dispatch_channels_raw": bool(stage2_dispatch_feature_csv),
        "stage2_neighbor_history_added": bool(stage2_neighbor_added),
        "stage2_neighbor_feature_names": list(stage2_neighbor_feature_names),
        "stage2_neighbor_decay_km": float(stage2_neighbor_decay_km),
        "stage2_reset_head_mu": bool(stage2_reset_head_mu),
        "stage2_aux_lead_lambda": float(stage2_aux_lead_lambda),
        "stage2_aux_lead_huber_delta": float(stage2_aux_lead_huber_delta),
        "stage2_pmf_long_lead_threshold": float(stage2_pmf_long_lead_threshold),
        "stage2_pmf_long_lead_weight": float(stage2_pmf_long_lead_weight),
        "stage2_pmf_right_anchor": float(stage2_pmf_right_anchor),
        "stage2_gaussian_loss_mode": str(stage2_gaussian_loss_mode),
        "stage2_gaussian_interval_continuity_correction": int(stage2_gaussian_interval_continuity_correction),
        "stage2_gaussian_interval_lambda": float(stage2_gaussian_interval_lambda),
        "gated_val_stage1_ckpt": gated_val_stage1_ckpt,
        "gated_val_stage1_eval_csv": gated_val_stage1_eval_csv,
        "gated_val_stage2_tstar_offset": int(gated_val_stage2_tstar_offset),
        "gated_val_stage2_tstar_offsets": [int(x) for x in gated_val_offsets],
        "gated_val_stage2_tstar_offset_weights": [float(x) for x in gated_val_offset_weights],
        "doy_start": int(C.DOY_START),
        "doy_end": int(C.DOY_END),
        "d_model": int(C.D_MODEL),
        "n_head": int(C.N_HEAD),
        "n_layers": int(C.N_LAYERS),
        "amp": int(bool(amp)),
        "amp_dtype": str(amp_dtype),
    }
    hparams_path = Path(out_root) / "hparams.json"
    hparams_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hparams_path, "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)
    print("saved:", hparams_path)

    t0 = time.perf_counter()
    x_mean, x_std = compute_norm_stats(train_s)
    print(f"[time] compute_norm_stats={time.perf_counter()-t0:.2f}s")

    # Phase B fix: dispatch confidence channels (last DISPATCH_TOTAL_CHANNELS
    # slots, appended after build_samples_season) must stay RAW. The lead head
    # reads alert_tstar in absolute DOY (~109..180), so standardization would
    # collapse it to ~0 and break mu = alert_rel + lead. Force mean=0 / std=1
    # on those slots; other channels keep their standardization.
    if stage2_dispatch_feature_csv:
        from rice.scripts.stage1_confidence_utils import DISPATCH_TOTAL_CHANNELS
        # VENDORED: prior_residual appends 1 extra tail channel (prior_mu, DOY) that must ALSO
        # stay RAW; include it so the alert_tstar channel is not accidentally standardized.
        n_disp = int(DISPATCH_TOTAL_CHANNELS) + (1 if str(stage2_pmf_mu_mode) == "prior_residual_alert_bin" else 0)
        d_total = int(x_mean.shape[0])
        disp_start = d_total - n_disp
        if disp_start < 0:
            raise SystemExit(
                f"[abort] norm_stats has only {d_total} channels but expected "
                f">= {n_disp} dispatch channels at the tail"
            )
        x_mean[disp_start:] = 0.0
        x_std[disp_start:] = 1.0
        print(f"[norm_stats] dispatch channels [{disp_start}:{d_total}] "
              f"forced to RAW (mean=0, std=1) so lead head reads alert_tstar "
              f"in absolute DOY")

    # VENDORED: v2_scaled anomaly channels are already TRAIN-standardized + clipped[-5,5] at build
    # time -> force RAW here so post-norm == the printed scaled values (no double standardization,
    # scale stays bounded regardless of compute_norm_stats).
    if stage2_anomaly_idx0 >= 0 and stage2_anomaly_n > 0:
        a0 = int(stage2_anomaly_idx0); a1 = a0 + int(stage2_anomaly_n)
        x_mean[a0:a1] = 0.0
        x_std[a0:a1] = 1.0
        print(f"[norm_stats] anomaly channels [{a0}:{a1}] forced to RAW "
              f"(pre-standardized+clipped at build) -> post-norm == scaled")

    if grouped_mode:
        train_groups = group_stage2_samples_by_site_year(train_s)
        val_groups = group_stage2_samples_by_site_year(val_s)
        test_groups = group_stage2_samples_by_site_year(test_s)
        train_ds = GroupedIntervalEventDataset(train_groups, x_mean, x_std)
        val_ds = GroupedIntervalEventDataset(val_groups, x_mean, x_std)
        test_ds = GroupedIntervalEventDataset(test_groups, x_mean, x_std)
        max_k_train = max((len(g["samples"]) for g in train_groups), default=0)
        print(
            f"[stage2_causal_tstar] grouped site-year counts: "
            f"train={len(train_groups)} val={len(val_groups)} test={len(test_groups)} | "
            f"max_K_train={max_k_train}"
        )
    else:
        train_ds = IntervalEventDataset(train_s, x_mean, x_std)
        val_ds = IntervalEventDataset(val_s, x_mean, x_std)
        test_ds = IntervalEventDataset(test_s, x_mean, x_std)

    if DEBUG_PICKLE_DATASET:
        try:
            pickle.dumps(train_ds)
            print("[debug] pickle.dumps(train_ds): OK")
        except Exception as e:
            print("[debug] pickle.dumps(train_ds): FAIL")
            raise

    if DEBUG_SAMPLE_CHECK:
        idxs = [0, len(train_ds) // 2, len(train_ds) - 1]
        for idx in idxs:
            if grouped_mode:
                _item = train_ds[idx]
                if len(_item) == 6:
                    X_i, L_i, R_i, c_i, tstar_i, pheno_i = _item
                    pheno_shape = tuple(pheno_i.shape)
                else:
                    X_i, L_i, R_i, c_i, tstar_i = _item
                    pheno_shape = "(none)"
                print(
                    f"[debug] train_ds[{idx}] X.shape={tuple(X_i.shape)} X.dtype={X_i.dtype} "
                    f"L.shape={tuple(L_i.shape)} R.shape={tuple(R_i.shape)} c.shape={tuple(c_i.shape)} "
                    f"tstar.shape={tuple(tstar_i.shape)} pheno.shape={pheno_shape}"
                )
            else:
                X_i, L_i, R_i, c_i = train_ds[idx]
                print(
                    f"[debug] train_ds[{idx}] X.shape={tuple(X_i.shape)} X.dtype={X_i.dtype} "
                    f"L={int(L_i)} R={int(R_i)} c={int(c_i)}"
                )

    D_in = int(train_ds[0][0].shape[-1])
    print(f"[D_in] computed_from_dataset={D_in}")
    if stage2_neighbor_added:
        print(f"[stage2_neighbor] final d_in (model input) = {D_in}  "
              f"(includes {len(stage2_neighbor_feature_names)} neighbor channels)")

    train_seeds = seeds if seeds is not None else C.SEEDS
    gated_val_alert_maps: dict[int, dict[str, int]] = {}
    if stage2_best_metric != "val_iou80":
        if not grouped_mode:
            print("[gated_val] flat Stage2 mode is supported, but current production policy is grouped.")
        if not gated_val_stage1_ckpt or not gated_val_stage1_eval_csv:
            raise ValueError(
                "--gated_val_stage1_ckpt and --gated_val_stage1_eval_csv are required "
                "when --stage2_best_metric is a gated validation metric"
            )
        gated_val_alert_maps, _stage1_doy_start = _prepare_gated_val_alert_maps(
            stage1_ckpt_path=str(gated_val_stage1_ckpt),
            stage1_eval_csv=str(gated_val_stage1_eval_csv),
            seeds=[int(s) for s in train_seeds],
            split_seed=int(split_seed),
            split_mode=str(split_mode),
            run=int(run),
            daily_feat=daily_feat,
            obs=obs,
            obs2=obs2,
            get_feature_cols=get_feature_cols,
            device=device,
        )
        print(
            f"[gated_val] selection_metric={stage2_best_metric} "
            f"stage2_tstar_offsets={','.join(str(int(x)) for x in gated_val_offsets)} "
            f"weights={','.join(f'{float(x):.3g}' for x in gated_val_offset_weights)}"
        )

    # =========================
    # 9) Multi-seed training (early stopping)
    # =========================
    trained_states = []
    balance_ratio = parse_balance_ratio(train_balance_ratio)
    if grouped_mode and balance_ratio is not None:
        print("[stage2_causal_tstar] balanced sampler is disabled in grouped mode")
        balance_ratio = None
    # ---- Offset-aware conditioning toggles (env-gated; OFF by default so the model
    # is byte-identical to the DN baseline). Parsed once here (main scope) so both the
    # model constructor and the ckpt bundle can reference them. Established WBPH_*
    # pattern. Both signals are DERIVED inside model.forward() from `tstar` and the
    # alert_tstar channel -> no dataloader change, input dim unchanged.
    #   WBPH_OFFSET_COND_EMB=1        -> learnable offset embedding
    #   WBPH_OFFSET_COND_EMB_DIM=8    -> embedding dim
    #   WBPH_OFFSET_COND_MAX=240      -> max offset covered by the table
    #   WBPH_OFFSET_COND_MIN=0        -> min alerted offset (validation floor)
    #   WBPH_OFFSET_COND_ISSUE_DOY=1  -> issue_doy sin/cos features
    _oc_emb = _os.environ.get("WBPH_OFFSET_COND_EMB", "0").strip() not in ("", "0", "false", "False")
    _oc_issue = _os.environ.get("WBPH_OFFSET_COND_ISSUE_DOY", "0").strip() not in ("", "0", "false", "False")
    _oc_emb_dim = int(_os.environ.get("WBPH_OFFSET_COND_EMB_DIM", "8"))
    _oc_max = int(_os.environ.get("WBPH_OFFSET_COND_MAX", "240"))
    _oc_min = int(_os.environ.get("WBPH_OFFSET_COND_MIN", "0"))
    if _oc_emb or _oc_issue:
        print(f"[offset_cond] ENABLED emb={_oc_emb}(dim={_oc_emb_dim},max={_oc_max},min={_oc_min}) "
              f"issue_doy={_oc_issue}")
    # ---- D1/D2 shared-encode + offset-specific-head toggles (env-gated; OFF by
    # default -> A/B/C paths untouched). candidate_offsets defaults to dense+off3.
    #   WBPH_SHARED_MULTI_OFFSET=1              -> encode base once (band-causal), gather per-offset
    #   WBPH_MU_HEAD_MODE=offset_specific       -> per-offset mu heads (else 'shared')
    #   WBPH_CANDIDATE_OFFSETS="3,7,14,..."     -> offset head set (must match eval grid offsets)
    #   (D2 reuses WBPH_OFFSET_COND_ISSUE_DOY=1 for the issue_doy sin/cos in each head)
    _sm_on = _os.environ.get("WBPH_SHARED_MULTI_OFFSET", "0").strip() not in ("", "0", "false", "False")
    _sm_head_mode = _os.environ.get("WBPH_MU_HEAD_MODE", "shared").strip()
    _sm_offsets_env = _os.environ.get("WBPH_CANDIDATE_OFFSETS", "").strip()
    _sm_offsets = ([int(x) for x in _sm_offsets_env.split(",") if x.strip() != ""]
                   if _sm_offsets_env else [3, 7, 14, 21, 28, 30, 35, 42, 45, 49, 56, 60])
    _sm_band = int(stage2_nowcast_window)   # band-causal width == baseline recent-window
    # ---- D4: shared base head + small per-offset residual (requires MU_HEAD_MODE=shared).
    #   WBPH_USE_OFFSET_RESIDUAL=1 / WBPH_RESIDUAL_HIDDEN_DIM=16 / WBPH_RESIDUAL_SCALE=1.0 /
    #   WBPH_ZERO_INIT_RESIDUAL=1 (default; residual final layer 0-init -> D4==D3 at init)
    _sm_resid = _os.environ.get("WBPH_USE_OFFSET_RESIDUAL", "0").strip() not in ("", "0", "false", "False")
    _sm_resid_hid = int(_os.environ.get("WBPH_RESIDUAL_HIDDEN_DIM", "16"))
    _sm_resid_scale = float(_os.environ.get("WBPH_RESIDUAL_SCALE", "1.0"))
    _sm_resid_zero = _os.environ.get("WBPH_ZERO_INIT_RESIDUAL", "1").strip() not in ("", "0", "false", "False")
    if _sm_on:
        print(f"[shared_multi_offset] ENABLED head_mode={_sm_head_mode} band_window={_sm_band} "
              f"issue_doy={_oc_issue} residual={_sm_resid}(hid={_sm_resid_hid},scale={_sm_resid_scale},"
              f"zero_init={_sm_resid_zero}) candidate_offsets={_sm_offsets}")
    for SEED in train_seeds:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        t0 = time.perf_counter()
        train_loader_kwargs = dict(batch_size=C.BATCH_TRAIN, shuffle=True, seed=SEED)
        val_loader_kwargs = dict(batch_size=C.BATCH_EVAL, shuffle=False)
        if grouped_mode:
            train_loader_kwargs["collate_fn"] = collate_grouped_stage2
            val_loader_kwargs["collate_fn"] = collate_grouped_stage2
        if balance_ratio is not None:
            sampler = build_balanced_sampler(train_s, balance_ratio, seed=SEED)
            train_loader_kwargs["sampler"] = sampler
            train_loader_kwargs["shuffle"] = False
            train_loader_kwargs["seed"] = None
            print(f"[seed {SEED}] balanced sampler enabled: ratio(right,interval,left)="
                  f"{balance_ratio['right']:.3f}:{balance_ratio['interval']:.3f}:{balance_ratio['left']:.3f}")
        if DEBUG_LOADER_SETTINGS and C.NUM_WORKERS > 0:
            train_loader = make_loader(train_ds, **train_loader_kwargs, multiprocessing_context="spawn")
            val_loader = make_loader(val_ds, **val_loader_kwargs, multiprocessing_context="spawn")
        else:
            train_loader = make_loader(train_ds, **train_loader_kwargs)
            val_loader = make_loader(val_ds, **val_loader_kwargs)
        print(f"[time] make_loaders={time.perf_counter()-t0:.2f}s")

        t0 = time.perf_counter()
        _ = next(iter(train_loader))
        print(f"[time] first_batch={time.perf_counter()-t0:.2f}s")
        # test_loader  = make_loader(test_ds,  C.BATCH_EVAL,  shuffle=False)  # eval script에서 사용

        torch.manual_seed(SEED)
        if grouped_mode:
            model = HierarchicalCausalHazardTransformer(
                d_in=D_in,
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=C.N_LAYERS,
                num_tstar_layers=int(stage2_tstar_layers),
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
                max_tstar_len=512,
                use_tstar_scalar_pos=bool(stage2_use_tstar_scalar_pos),
                phenology_bias_head=bool(int(stage2_phenology_bias_head)),
                phenology_dim=4,
                phenology_hidden=int(stage2_phenology_hidden),
                use_offset_embedding=bool(_oc_emb),
                offset_embedding_dim=int(_oc_emb_dim),
                offset_max=int(_oc_max),
                offset_min=int(_oc_min),
                use_issue_doy_features=bool(_oc_issue),
                doy_period=365.0,
                use_shared_multi_offset=bool(_sm_on),
                mu_head_mode=str(_sm_head_mode),
                candidate_offsets=list(_sm_offsets),
                shared_band_window=int(_sm_band),
                use_offset_residual=bool(_sm_resid),
                residual_hidden_dim=int(_sm_resid_hid),
                residual_scale=float(_sm_resid_scale),
                zero_init_residual=bool(_sm_resid_zero),
            ).to(device)
            model.early_tstar_weight_min = float(stage2_early_tstar_weight_min)
            model.site_year_mean_loss = bool(stage2_site_year_mean_loss)
            model.time_chunk_size = int(stage2_time_chunk_size)
            model.conditional_survival = bool(stage2_conditional_survival)
            model.pmf_mode = str(stage2_pmf_mode)
            model.gaussian_sigma = float(stage2_pmf_sigma)
            model.gaussian_mu_max = float(stage2_pmf_mu_max)
            # Phase B: lead_from_alert mu head config (no-op when mu_mode='absolute')
            model.mu_mode = str(stage2_pmf_mu_mode)
            model.lead_min = float(stage2_pmf_lead_min)
            model.lead_max = float(stage2_pmf_lead_max)
            # Residual-from-climatology mu head config (no-op when mu_mode != 'residual_clim').
            # clim_mid is provided in DOY units (e.g. 198.7 for sheath_blight);
            # convert to 1-based season-index coords to match L/R semantics in
            # the model head.
            model.clim_mid_rel = float(stage2_pmf_clim_mid) - float(C.DOY_START) + 1.0
            model.delta_max = float(stage2_pmf_delta_max)
            model.alert_tstar_feat_idx = int(stage2_pmf_alert_tstar_feat_idx)
            model.prior_mu_feat_idx = int(stage2_prior_mu_feat_idx)   # VENDORED: prior-residual channel idx (-1 if unused)
            model.doy_start = int(C.DOY_START)
            model.offset_cond_strict = True   # raise on out-of-range alerted offset
            model.lead_strict_alert_check = True   # safety: catch std-bug fast
            model.lead_debug_once_pending = (str(stage2_pmf_mu_mode) in
                                              ("lead_from_alert", "residual_clim", "prior_residual_alert_bin"))
            # Phase B aux lead loss (no-op when lambda=0)
            model.aux_lead_lambda = float(stage2_aux_lead_lambda)
            model.aux_lead_huber_delta = float(stage2_aux_lead_huber_delta)
            model.asym_weight = float(stage2_pmf_asym_weight)
            model.right_weight = float(stage2_pmf_right_weight)
            model.target_offset = float(stage2_pmf_target_offset)
            model.asym_weight_early = float(stage2_pmf_asym_weight_early)
            model.target_early_offset = float(stage2_pmf_target_early_offset)
            model.target_mode = str(stage2_pmf_target_mode)
            model.zone_late_weight = float(stage2_pmf_zone_late_weight)
            model.zone_too_late_weight = float(stage2_pmf_zone_too_late_weight)
            model.zone_missed_weight = float(stage2_pmf_zone_missed_weight)
            model.zone_too_early_weight = float(stage2_pmf_zone_too_early_weight)
            model.zone_too_late_threshold = float(stage2_pmf_zone_too_late_threshold)
            model.zone_missed_threshold = float(stage2_pmf_zone_missed_threshold)
            model.zone_too_early_threshold = float(stage2_pmf_zone_too_early_threshold)
            # Phase S5: per-sample long-lead weighting (mu-loss multiplier).
            model.long_lead_threshold = float(stage2_pmf_long_lead_threshold)
            model.long_lead_weight = float(stage2_pmf_long_lead_weight)
            model._phase_s5_sw_logged = False
            # Phase S10: configurable right-cens anchor (0 → Tend fallback).
            model.right_anchor = float(stage2_pmf_right_anchor)
            # Gaussian interval NLL ablation: select loss mode for Gaussian PMF.
            #   asym_mse    (default) : legacy asymmetric_mu_loss (regression-style).
            #   interval_nll          : gaussian_interval_nll_loss
            #                            P(L < T ≤ R) on Gaussian with fixed sigma.
            #   mixed                 : asym_mse + lambda * interval_nll. Lambda
            #                            controlled via stage2_gaussian_interval_lambda.
            model.gaussian_loss_mode = str(stage2_gaussian_loss_mode)
            model.gaussian_interval_continuity_correction = bool(int(stage2_gaussian_interval_continuity_correction))
            model.gaussian_interval_lambda = float(stage2_gaussian_interval_lambda)
        else:
            model = HazardTransformer(
                d_in=D_in,
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=C.N_LAYERS,
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
            ).to(device)
        if stage2_warm_start_ckpt:
            warm_ckpt = torch.load(stage2_warm_start_ckpt, map_location="cpu")
            warm_seed = int(SEED if stage2_warm_start_seed is None else stage2_warm_start_seed)
            warm_state = _select_stage_state(warm_ckpt, warm_seed)
            if warm_state is None:
                raise ValueError(f"warm-start checkpoint has no trained state for seed={warm_seed}: {stage2_warm_start_ckpt}")

            # Shape-aware patch: when the new model has a larger LAST-dim than
            # the warm checkpoint (typical case: added input channels such as
            # dispatch confidence features), copy old weights into [:, :old_in]
            # and zero-init the new tail. Other shape mismatches are dropped so
            # load_state_dict can still run, and those tensors fall back to the
            # current random init.
            cur_sd = model.state_dict()
            warm_sd = dict(warm_state["state_dict"])
            # Phase B headreset: drop head_mu.* keys so they fall through to
            # the fresh model init. backbone / in_proj / encoder / phen_head
            # come from warm ckpt as usual. Only active in lead_from_alert mode
            # since absolute mode reuses head_mu meaningfully.
            if bool(stage2_reset_head_mu):
                if str(stage2_pmf_mu_mode) != "lead_from_alert":
                    print(f"[reset_head_mu] WARNING: --stage2_reset_head_mu set "
                          f"but mu_mode={stage2_pmf_mu_mode!r}; reset has no "
                          f"intended effect outside lead_from_alert. Proceeding "
                          f"anyway.")
                reset_keys = sorted(k for k in warm_sd if k.startswith("head_mu."))
                for k in reset_keys:
                    del warm_sd[k]
                print(f"[reset_head_mu] reinitialized (dropped from warm-start, "
                      f"kept current fresh init): {reset_keys}")
            patched = []
            dropped = []
            for k in list(warm_sd.keys()):
                w_old = warm_sd[k]
                w_cur = cur_sd.get(k)
                if w_cur is None:
                    continue  # unexpected key, load_state_dict will skip
                if w_old.shape == w_cur.shape:
                    continue
                if (w_old.dim() == w_cur.dim()
                        and w_old.dim() >= 1
                        and tuple(w_old.shape[:-1]) == tuple(w_cur.shape[:-1])
                        and int(w_old.shape[-1]) < int(w_cur.shape[-1])):
                    new_w = w_cur.detach().clone()  # current random init
                    new_w[..., :int(w_old.shape[-1])] = w_old.to(new_w.dtype)
                    new_w[..., int(w_old.shape[-1]):] = 0.0
                    warm_sd[k] = new_w
                    patched.append({
                        "key": k,
                        "old_shape": tuple(w_old.shape),
                        "new_shape": tuple(new_w.shape),
                        "copied_in_dim": int(w_old.shape[-1]),
                        "zeroed_in_dim": int(new_w.shape[-1] - w_old.shape[-1]),
                    })
                else:
                    dropped.append({
                        "key": k,
                        "warm_shape": tuple(w_old.shape),
                        "cur_shape": tuple(w_cur.shape),
                    })
                    del warm_sd[k]

            if patched:
                print(f"[seed {SEED}] warm-start shape-patched {len(patched)} tensor(s) "
                      f"(zero-padded new input dims):")
                for p in patched:
                    print(f"  {p['key']}  old={p['old_shape']} -> new={p['new_shape']}  "
                          f"copied_in_dim={p['copied_in_dim']}  "
                          f"zeroed_in_dim={p['zeroed_in_dim']}")
            if dropped:
                print(f"[seed {SEED}] warm-start dropped {len(dropped)} incompatible "
                      f"tensor(s) (current random init kept):")
                for d in dropped:
                    print(f"  {d['key']}  warm={d['warm_shape']} cur={d['cur_shape']}")

            missing, unexpected = model.load_state_dict(warm_sd, strict=False)
            print(f"[seed {SEED}] warm-started Stage2 from {stage2_warm_start_ckpt} seed={warm_seed}")
            if missing:
                print(f"[seed {SEED}] warm-start missing keys ({len(missing)}, will use random init): {missing[:8]}")
            if unexpected:
                print(f"[seed {SEED}] warm-start unexpected keys ({len(unexpected)}, ignored): {unexpected[:8]}")
        model.use_amp_eval = bool(use_amp)
        model.amp_dtype_eval = str(amp_dtype)

        opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
        scaler = None
        if use_amp and amp_dtype == "fp16":
            scaler = torch.cuda.amp.GradScaler()

        if grouped_mode and int(stage2_sanity_only):
            tr, tr_base, tr_mass, tr_late, tr_entropy, tr_location, tr_right_frac = run_epoch_weighted_grouped(
                model,
                opt,
                train_loader,
                Tend=T,
                device=device,
                train=False,
                lambda_mass=lambda_mass,
                lambda_right_late=lambda_right_late,
                right_late_tau=right_late_tau,
                early_tstar_weight_min=float(stage2_early_tstar_weight_min),
                site_year_mean_loss=bool(stage2_site_year_mean_loss),
                lead_weighting=bool(stage2_lead_weighting),
                target_lead_min=int(target_lead_min),
                target_lead_max=int(target_lead_max),
                support_lead_min=int(support_lead_min),
                support_lead_max=int(support_lead_max),
                lead_weight_min=float(lead_weight_min),
                mass_lead_weighting=bool(stage2_mass_lead_weighting),
                lead_loss_mode=str(stage2_lead_loss_mode),
                lead_loss_min=int(stage2_lead_min),
                lead_loss_max=int(stage2_lead_max),
                lead_loss_mid_min=int(stage2_mid_lead_min),
                lead_loss_mid_max=int(stage2_mid_lead_max),
                lead_loss_late_exclude_days=int(stage2_late_exclude_days),
                lead_loss_weight_1_14=float(stage2_lead_weight_1_14),
                lead_loss_weight_15_29=float(stage2_lead_weight_15_29),
                lead_loss_weight_30_60=float(stage2_lead_weight_30_60),
                lead_loss_weight_61_75=float(stage2_lead_weight_61_75),
                lead_loss_weight_gt75=float(stage2_lead_weight_gt75),
                entropy_lambda=float(stage2_entropy_lambda),
                entropy_conditional=bool(stage2_entropy_conditional),
                location_lambda=float(stage2_location_lambda),
                conditional_survival=bool(stage2_conditional_survival),
                log_mass=True,
                epoch_idx=0,
                return_parts=True,
                use_amp=bool(use_amp),
                amp_dtype=str(amp_dtype),
                scaler=None,
                max_batches=int(stage2_sanity_batches),
            )
            print(
                f"[stage2_sanity_only] total={tr:.6f} base={tr_base:.6f} mass={tr_mass:.6f} "
                f"late={tr_late:.6f} entropy={tr_entropy:.6f} location={tr_location:.6f} "
                f"right_frac={tr_right_frac:.6f} batches={int(stage2_sanity_batches)}"
            )
            continue

        best_val = float("inf")
        best_val_iou = float("-inf")
        best_select_score = float("-inf")
        best_gated_stats = {}
        best_state = None
        best_epoch = -1
        pat = 0

        for epoch in range(1, C.MAX_EPOCHS + 1):
            if grouped_mode:
                tr, tr_base, tr_mass, tr_late, tr_entropy, tr_location, tr_right_frac = run_epoch_weighted_grouped(
                    model,
                    opt,
                    train_loader,
                    Tend=T,
                    device=device,
                    train=True,
                    lambda_mass=lambda_mass,
                    lambda_right_late=lambda_right_late,
                    right_late_tau=right_late_tau,
                    early_tstar_weight_min=float(stage2_early_tstar_weight_min),
                    site_year_mean_loss=bool(stage2_site_year_mean_loss),
                    lead_weighting=bool(stage2_lead_weighting),
                    target_lead_min=int(target_lead_min),
                    target_lead_max=int(target_lead_max),
                    support_lead_min=int(support_lead_min),
                    support_lead_max=int(support_lead_max),
                    lead_weight_min=float(lead_weight_min),
                    mass_lead_weighting=bool(stage2_mass_lead_weighting),
                    lead_loss_mode=str(stage2_lead_loss_mode),
                    lead_loss_min=int(stage2_lead_min),
                    lead_loss_max=int(stage2_lead_max),
                    lead_loss_mid_min=int(stage2_mid_lead_min),
                    lead_loss_mid_max=int(stage2_mid_lead_max),
                    lead_loss_late_exclude_days=int(stage2_late_exclude_days),
                    lead_loss_weight_1_14=float(stage2_lead_weight_1_14),
                    lead_loss_weight_15_29=float(stage2_lead_weight_15_29),
                    lead_loss_weight_30_60=float(stage2_lead_weight_30_60),
                    lead_loss_weight_61_75=float(stage2_lead_weight_61_75),
                    lead_loss_weight_gt75=float(stage2_lead_weight_gt75),
                    entropy_lambda=float(stage2_entropy_lambda),
                    entropy_conditional=bool(stage2_entropy_conditional),
                    location_lambda=float(stage2_location_lambda),
                    conditional_survival=bool(stage2_conditional_survival),
                    log_mass=True,
                    epoch_idx=epoch,
                    return_parts=True,
                    use_amp=bool(use_amp),
                    amp_dtype=str(amp_dtype),
                    scaler=scaler,
                )
                va = eval_nll_model_grouped(model, val_loader, Tend=T, device=device)
                va_stats = eval_metrics_with_overlap_grouped(
                    model,
                    val_loader,
                    Tend=T,
                    device=device,
                    alpha=0.2,
                    pi_method=getattr(C, "PI_METHOD", "shortest"),
                )
            else:
                tr, tr_base, tr_mass, tr_late, tr_right_frac = run_epoch_weighted(
                    model,
                    opt,
                    train_loader,
                    Tend=T,
                    device=device,
                    train=True,
                    lambda_mass=lambda_mass,
                    lambda_right_late=lambda_right_late,
                    right_late_tau=right_late_tau,
                    log_mass=True,
                    epoch_idx=epoch,
                    return_parts=True,
                    use_amp=bool(use_amp),
                    amp_dtype=str(amp_dtype),
                    scaler=scaler,
                )
                tr_entropy = 0.0
                tr_location = 0.0
                va = eval_nll_model(model, val_loader, Tend=T, device=device)
                va_stats = eval_metrics_with_overlap(
                    model,
                    val_loader,
                    Tend=T,
                    device=device,
                    alpha=0.2,
                    pi_method=getattr(C, "PI_METHOD", "shortest"),
                )
            va_iou = float(va_stats["IoU_mean_interval_only(80%)"])
            if np.isnan(va_iou):
                va_iou = float("-inf")
            gated_stats = {}
            select_score = va_iou
            select_label = "val_iou80"
            if stage2_best_metric != "val_iou80":
                gated_stats = _eval_gated_validation_stats(
                    model=model,
                    loader=val_loader,
                    source_groups=val_groups if grouped_mode else [],
                    source_samples=val_s,
                    grouped_mode=bool(grouped_mode),
                    alert_map_abs=gated_val_alert_maps.get(int(SEED), {}),
                    stage2_doy_start=int(C.DOY_START),
                    stage2_tstar_offsets=[int(x) for x in gated_val_offsets],
                    stage2_tstar_offset_weights=[float(x) for x in gated_val_offset_weights],
                    Tend=T,
                    device=device,
                )
                if stage2_best_metric == "gated_val_iou80":
                    select_score = float(gated_stats.get("weighted_IoU80", float("nan")))
                elif stage2_best_metric == "gated_val_interval_hit_f1":
                    select_score = float(gated_stats.get("weighted_interval_hit_f1", float("nan")))
                elif stage2_best_metric == "gated_val_mae_int":
                    mae = float(gated_stats.get("weighted_MAE_int", float("nan")))
                    select_score = -mae if np.isfinite(mae) else float("-inf")
                else:
                    raise ValueError(f"Unknown --stage2_best_metric: {stage2_best_metric}")
                if not np.isfinite(select_score):
                    select_score = float("-inf")
                select_label = str(stage2_best_metric)
            if not np.isfinite(va):
                print(f"[seed {SEED}] epoch {epoch:02d} | val_nll is non-finite ({va}); forcing +inf for model selection")
                va = float("inf")
            msg = (
                f"[seed {SEED}] epoch {epoch:02d} | train_total {tr:.4f} | train_base {tr_base:.4f} "
                f"| train_mass {tr_mass:.4f} | train_late {tr_late:.4f} | train_entropy {tr_entropy:.4f} "
                f"| train_location {tr_location:.4f} "
                f"| train_right_frac {tr_right_frac:.4f} "
                f"| val_nll {va:.4f} | val_iou80 {va_iou:.4f}"
            )
            if gated_stats:
                msg += (
                    f" | gated_iou80_w {float(gated_stats.get('weighted_IoU80', float('nan'))):.4f}"
                    f" | gated_hit_f1_w {float(gated_stats.get('weighted_interval_hit_f1', float('nan'))):.4f}"
                    f" | gated_mae_w {float(gated_stats.get('weighted_MAE_int', float('nan'))):.4f}"
                    f" | gated_mass_w {float(gated_stats.get('weighted_Mass_int', float('nan'))):.4f}"
                    f" | gated_width_w {float(gated_stats.get('weighted_pred_width', float('nan'))):.4f}"
                    f" | gated_early_w {float(gated_stats.get('weighted_EarlyRecall80', float('nan'))):.4f}"
                    f" | gated_n {int(gated_stats.get('n_matched', 0))}"
                    f" | select({select_label}) {select_score:.4f}"
                )
                for off in gated_val_offsets:
                    msg += (
                        f" | off{int(off)}_n {int(gated_stats.get(f'offset{int(off)}_n_matched', 0))}"
                        f" off{int(off)}_iou {float(gated_stats.get(f'offset{int(off)}_IoU80', float('nan'))):.4f}"
                        f" off{int(off)}_hitf1 {float(gated_stats.get(f'offset{int(off)}_interval_hit_f1', float('nan'))):.4f}"
                        f" off{int(off)}_mae {float(gated_stats.get(f'offset{int(off)}_MAE_int', float('nan'))):.4f}"
                    )
            print(msg)
            if wandb_run is not None:
                log_payload = {
                    "seed": int(SEED),
                    "epoch": int(epoch),
                    "train/loss": float(tr),
                    "train/total_loss": float(tr),
                    "train/base_loss": float(tr_base),
                    "train/mass_loss": float(tr_mass),
                    "train/late_loss": float(tr_late),
                    "train/late_loss_mean": float(tr_late),
                    "train/entropy": float(tr_entropy),
                    "train/entropy_loss": float(tr_entropy),
                    "train/lambda_entropy": float(stage2_entropy_lambda),
                    "train/lambda_entropy_term": float(stage2_entropy_lambda) * float(tr_entropy),
                    "train/location": float(tr_location),
                    "train/location_loss": float(tr_location),
                    "train/lambda_location": float(stage2_location_lambda),
                    "train/lambda_location_term": float(stage2_location_lambda) * float(tr_location),
                    "train/right_frac": float(tr_right_frac),
                    "train/lr": float(opt.param_groups[0]["lr"]),
                    "val/loss": float(va),
                    "val/nll": float(va),
                    "val/iou80_interval_only": float(va_iou),
                    "val/selection_score": float(select_score),
                }
                if gated_stats:
                    log_payload.update(
                        {
                            "val/gated_iou80": float(gated_stats.get("weighted_IoU80", float("nan"))),
                            "val/gated_interval_hit_f1": float(gated_stats.get("weighted_interval_hit_f1", float("nan"))),
                            "val/gated_mae_int": float(gated_stats.get("weighted_MAE_int", float("nan"))),
                            "val/gated_mass_int": float(gated_stats.get("weighted_Mass_int", float("nan"))),
                            "val/gated_pred_width": float(gated_stats.get("weighted_pred_width", float("nan"))),
                            "val/gated_early_recall80": float(gated_stats.get("weighted_EarlyRecall80", float("nan"))),
                            "val/gated_post_true_start_rate": float(gated_stats.get("weighted_post_true_start_rate", float("nan"))),
                            "val/gated_n_matched": int(gated_stats.get("n_matched", 0)),
                            "val/gated_offset_missed": int(gated_stats.get("offset_missed", 0)),
                        }
                    )
                    for off in gated_val_offsets:
                        log_payload.update(
                            {
                                f"val/offset{int(off)}_iou80": float(gated_stats.get(f"offset{int(off)}_IoU80", float("nan"))),
                                f"val/offset{int(off)}_interval_hit_f1": float(gated_stats.get(f"offset{int(off)}_interval_hit_f1", float("nan"))),
                                f"val/offset{int(off)}_mae_int": float(gated_stats.get(f"offset{int(off)}_MAE_int", float("nan"))),
                                f"val/offset{int(off)}_mass_int": float(gated_stats.get(f"offset{int(off)}_Mass_int", float("nan"))),
                                f"val/offset{int(off)}_pred_width": float(gated_stats.get(f"offset{int(off)}_pred_width", float("nan"))),
                                f"val/offset{int(off)}_early_recall80": float(gated_stats.get(f"offset{int(off)}_EarlyRecall80", float("nan"))),
                                f"val/offset{int(off)}_post_true_start_rate": float(gated_stats.get(f"offset{int(off)}_post_true_start_rate", float("nan"))),
                                f"val/offset{int(off)}_n_matched": int(gated_stats.get(f"offset{int(off)}_n_matched", 0)),
                            }
                        )
                wandb_run.log(log_payload, step=int(epoch))
                pass

            if int(save_epoch_checkpoints):
                epoch_bundle = {
                    **build_ckpt_meta(
                        run=run,
                        pest=pest,
                        d_in=D_in,
                        feature_cols=feature_cols,
                        feature_names=feature_names,
                        year_max=getattr(C, "YEAR_MAX", None),
                    ),
                    "doy_start": C.DOY_START,
                    "doy_end": C.DOY_END,
                    "split_mode": str(split_mode),
                    "split_seed": int(split_seed),
                    "T": T,
                    "d_model": int(C.D_MODEL),
                    "n_head": int(C.N_HEAD),
                    "n_layers": int(C.N_LAYERS),
                    "stage2_nowcast": bool(stage2_nowcast),
                    "stage2_nowcast_window": int(stage2_nowcast_window),
                    "stage2_nowcast_stride": int(stage2_nowcast_stride),
                    "stage2_nowcast_tstar_start": None if stage2_nowcast_tstar_start is None else int(stage2_nowcast_tstar_start),
                    "stage2_nowcast_only_pre_event": int(stage2_nowcast_only_pre_event),
                    "stage2_nowcast_event_time_proxy": stage2_nowcast_event_time_proxy,
                    "stage2_nowcast_require_tstar_before_L": int(stage2_nowcast_require_tstar_before_L),
                    "stage2_causal_tstar": bool(grouped_mode),
                    "stage2_tstar_layers": int(stage2_tstar_layers),
                    "stage2_use_tstar_scalar_pos": int(stage2_use_tstar_scalar_pos),
                    "stage2_early_tstar_weight_min": float(stage2_early_tstar_weight_min),
                    "stage2_site_year_mean_loss": int(stage2_site_year_mean_loss),
                    "stage2_time_chunk_size": int(stage2_time_chunk_size),
                    "stage2_conditional_survival": int(bool(stage2_conditional_survival)),
                    "stage2_lead_weighting": int(stage2_lead_weighting),
                    "target_lead_min": int(target_lead_min),
                    "target_lead_max": int(target_lead_max),
                    "support_lead_min": int(support_lead_min),
                    "support_lead_max": int(support_lead_max),
                    "lead_weight_min": float(lead_weight_min),
                    "stage2_mass_lead_weighting": int(stage2_mass_lead_weighting),
                    "stage2_entropy_lambda": float(stage2_entropy_lambda),
                    "stage2_entropy_conditional": int(stage2_entropy_conditional),
                    "stage2_location_lambda": float(stage2_location_lambda),
                    "stage2_warm_start_ckpt": stage2_warm_start_ckpt,
                    "stage2_warm_start_seed": None if stage2_warm_start_seed is None else int(stage2_warm_start_seed),
                    "stage2_lead_loss_mode": str(stage2_lead_loss_mode),
                    "stage2_lead_min": int(stage2_lead_min),
                    "stage2_lead_max": int(stage2_lead_max),
                    "stage2_mid_lead_min": int(stage2_mid_lead_min),
                    "stage2_mid_lead_max": int(stage2_mid_lead_max),
                    "stage2_late_exclude_days": int(stage2_late_exclude_days),
                    "stage2_lead_weight_1_14": float(stage2_lead_weight_1_14),
                    "stage2_lead_weight_15_29": float(stage2_lead_weight_15_29),
                    "stage2_lead_weight_30_60": float(stage2_lead_weight_30_60),
                    "stage2_lead_weight_61_75": float(stage2_lead_weight_61_75),
                    "stage2_lead_weight_gt75": float(stage2_lead_weight_gt75),
                    "stage2_best_metric": str(stage2_best_metric),
                    "stage2_pmf_mode": str(stage2_pmf_mode),
                    "stage2_pmf_sigma": float(stage2_pmf_sigma),
                    "stage2_pmf_mu_max": float(stage2_pmf_mu_max),
                    "stage2_pmf_asym_weight": float(stage2_pmf_asym_weight),
                    "stage2_pmf_right_weight": float(stage2_pmf_right_weight),
                    "stage2_pmf_target_offset": float(stage2_pmf_target_offset),
                    "stage2_pmf_asym_weight_early": float(stage2_pmf_asym_weight_early),
                    "stage2_pmf_target_early_offset": float(stage2_pmf_target_early_offset),
                    "stage2_pmf_target_mode": str(stage2_pmf_target_mode),
                    "stage2_pmf_zone_late_weight": float(stage2_pmf_zone_late_weight),
                    "stage2_pmf_zone_too_late_weight": float(stage2_pmf_zone_too_late_weight),
                    "stage2_pmf_zone_missed_weight": float(stage2_pmf_zone_missed_weight),
                    "stage2_pmf_zone_too_early_weight": float(stage2_pmf_zone_too_early_weight),
                    "stage2_pmf_zone_too_late_threshold": float(stage2_pmf_zone_too_late_threshold),
                    "stage2_pmf_zone_missed_threshold": float(stage2_pmf_zone_missed_threshold),
                    "stage2_pmf_zone_too_early_threshold": float(stage2_pmf_zone_too_early_threshold),
                    "stage2_phenology_bias_head": int(stage2_phenology_bias_head),
                    "stage2_phenology_hidden": int(stage2_phenology_hidden),
                    "stage2_dispatch_features_added": bool(stage2_dispatch_feature_csv),
                    "stage2_dispatch_feature_csv": (str(stage2_dispatch_feature_csv)
                                                      if stage2_dispatch_feature_csv else None),
                    "stage2_dispatch_feature_mode": str(stage2_dispatch_feature_mode),
                    "stage2_dispatch_feature_missing_value": float(stage2_dispatch_feature_missing_value),
                    "stage2_cohort_dispatch_only": bool(stage2_cohort_dispatch_only),
                    "stage2_pmf_mu_mode": str(stage2_pmf_mu_mode),
                    "stage2_pmf_lead_min": float(stage2_pmf_lead_min),
                    "stage2_pmf_lead_max": float(stage2_pmf_lead_max),
                    "stage2_pmf_clim_mid": float(stage2_pmf_clim_mid),
                    "stage2_pmf_delta_max": float(stage2_pmf_delta_max),
                    "stage2_pmf_alert_tstar_feat_idx": int(stage2_pmf_alert_tstar_feat_idx),
        "stage2_dispatch_channels_raw": bool(stage2_dispatch_feature_csv),
        "stage2_neighbor_history_added": bool(stage2_neighbor_added),
        "stage2_neighbor_feature_names": list(stage2_neighbor_feature_names),
        "stage2_neighbor_decay_km": float(stage2_neighbor_decay_km),
        "stage2_reset_head_mu": bool(stage2_reset_head_mu),
        "stage2_aux_lead_lambda": float(stage2_aux_lead_lambda),
        "stage2_aux_lead_huber_delta": float(stage2_aux_lead_huber_delta),
                    "stage2_pmf_long_lead_threshold": float(stage2_pmf_long_lead_threshold),
                    "stage2_pmf_long_lead_weight": float(stage2_pmf_long_lead_weight),
                    "stage2_pmf_right_anchor": float(stage2_pmf_right_anchor),
                    "stage2_gaussian_loss_mode": str(stage2_gaussian_loss_mode),
                    "stage2_gaussian_interval_continuity_correction": int(stage2_gaussian_interval_continuity_correction),
                    "stage2_gaussian_interval_lambda": float(stage2_gaussian_interval_lambda),
                    "gated_val_stage1_ckpt": gated_val_stage1_ckpt,
                    "gated_val_stage1_eval_csv": gated_val_stage1_eval_csv,
                    "gated_val_stage2_tstar_offset": int(gated_val_stage2_tstar_offset),
                    "gated_val_stage2_tstar_offsets": [int(x) for x in gated_val_offsets],
                    "gated_val_stage2_tstar_offset_weights": [float(x) for x in gated_val_offset_weights],
                    "amp": int(bool(amp)),
                    "amp_dtype": str(amp_dtype),
                    "stage2_model_kind": "hierarchical_causal_tstar" if grouped_mode else "flat",
                    "stage2_nowcast_label_mode": "orig",
                    "norm_mean": x_mean,
                    "norm_std": x_std,
                    "trained_states": [
                        {
                            "seed": int(SEED),
                            "best_epoch": int(epoch),
                            "best_val_nll": float(va),
                            "best_val_iou80": float(va_iou),
                            "best_selection_score": float(select_score),
                            "best_gated_stats": dict(gated_stats),
                            "state_dict": copy.deepcopy(model.state_dict()),
                        }
                    ],
                    "split_counts": {"train": len(train_s), "val": len(val_s), "test": len(test_s)},
                    "epoch_checkpoint": True,
                    "epoch": int(epoch),
                }
                epoch_path = Path(out_root) / "ckpt" / f"checkpoint_run{run}_seed{int(SEED)}_epoch{int(epoch):02d}.pt"
                epoch_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(epoch_bundle, epoch_path)
                print(f"saved epoch checkpoint: {epoch_path}")

            if stage2_best_metric == "val_iou80":
                improved = va_iou > (best_val_iou + C.MIN_DELTA)
                tie_better_nll = (abs(va_iou - best_val_iou) <= C.MIN_DELTA) and (va < best_val - C.MIN_DELTA)
            else:
                improved = select_score > (best_select_score + C.MIN_DELTA)
                tie_better_nll = (abs(select_score - best_select_score) <= C.MIN_DELTA) and (va < best_val - C.MIN_DELTA)
            if improved or tie_better_nll:
                best_val = float(va)
                best_val_iou = float(va_iou)
                best_select_score = float(select_score)
                best_gated_stats = dict(gated_stats)
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
                "best_val_iou80": best_val_iou,
                "best_selection_score": best_select_score,
                "best_gated_stats": best_gated_stats,
                "state_dict": best_state,
            }
        )
        print(
            f"[seed {SEED}] DONE | best_epoch={best_epoch} | "
            f"best_val_iou80={best_val_iou:.4f} | best_val_nll={best_val:.4f} | "
            f"best_selection_score={best_select_score:.4f}\n"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "seed": int(SEED),
                    "best/best_epoch": int(best_epoch),
                    "best/best_val_nll": float(best_val),
                    "best/best_val_iou80": float(best_val_iou),
                    "best/best_selection_score": float(best_select_score),
                }
            )

    # =========================
    # 10) Save checkpoint bundle
    # =========================
    bundle = {
        **build_ckpt_meta(
            run=run,
            pest=pest,
            d_in=D_in,
            feature_cols=feature_cols,
            feature_names=feature_names,
            year_max=getattr(C, "YEAR_MAX", None),
        ),
        "doy_start": C.DOY_START,
        "doy_end": C.DOY_END,
        "split_mode": str(split_mode),
        "split_seed": int(split_seed),
        "T": T,
        "d_model": int(C.D_MODEL),
        "n_head": int(C.N_HEAD),
        "n_layers": int(C.N_LAYERS),
        "stage2_nowcast": bool(stage2_nowcast),
        "stage2_nowcast_window": int(stage2_nowcast_window),
        "stage2_nowcast_stride": int(stage2_nowcast_stride),
        "stage2_nowcast_tstar_start": None if stage2_nowcast_tstar_start is None else int(stage2_nowcast_tstar_start),
        "stage2_nowcast_only_pre_event": int(stage2_nowcast_only_pre_event),
        "stage2_nowcast_event_time_proxy": stage2_nowcast_event_time_proxy,
        "stage2_nowcast_require_tstar_before_L": int(stage2_nowcast_require_tstar_before_L),
        "stage2_causal_tstar": bool(grouped_mode),
        "stage2_tstar_layers": int(stage2_tstar_layers),
        "stage2_use_tstar_scalar_pos": int(stage2_use_tstar_scalar_pos),
        "stage2_early_tstar_weight_min": float(stage2_early_tstar_weight_min),
        "stage2_site_year_mean_loss": int(stage2_site_year_mean_loss),
        "stage2_time_chunk_size": int(stage2_time_chunk_size),
        "stage2_conditional_survival": int(bool(stage2_conditional_survival)),
        "stage2_lead_weighting": int(stage2_lead_weighting),
        "target_lead_min": int(target_lead_min),
        "target_lead_max": int(target_lead_max),
        "support_lead_min": int(support_lead_min),
        "support_lead_max": int(support_lead_max),
        "lead_weight_min": float(lead_weight_min),
                    "stage2_mass_lead_weighting": int(stage2_mass_lead_weighting),
                    "stage2_entropy_lambda": float(stage2_entropy_lambda),
                    "stage2_entropy_conditional": int(stage2_entropy_conditional),
                    "stage2_location_lambda": float(stage2_location_lambda),
                    "stage2_warm_start_ckpt": stage2_warm_start_ckpt,
        "stage2_warm_start_seed": None if stage2_warm_start_seed is None else int(stage2_warm_start_seed),
        "stage2_lead_loss_mode": str(stage2_lead_loss_mode),
        "stage2_lead_min": int(stage2_lead_min),
        "stage2_lead_max": int(stage2_lead_max),
        "stage2_mid_lead_min": int(stage2_mid_lead_min),
        "stage2_mid_lead_max": int(stage2_mid_lead_max),
                    "stage2_late_exclude_days": int(stage2_late_exclude_days),
                    "stage2_lead_weight_1_14": float(stage2_lead_weight_1_14),
                    "stage2_lead_weight_15_29": float(stage2_lead_weight_15_29),
                    "stage2_lead_weight_30_60": float(stage2_lead_weight_30_60),
                    "stage2_lead_weight_61_75": float(stage2_lead_weight_61_75),
                    "stage2_lead_weight_gt75": float(stage2_lead_weight_gt75),
        "stage2_best_metric": str(stage2_best_metric),
        "stage2_pmf_mode": str(stage2_pmf_mode),
        "stage2_pmf_sigma": float(stage2_pmf_sigma),
        "stage2_pmf_mu_max": float(stage2_pmf_mu_max),
        "stage2_pmf_asym_weight": float(stage2_pmf_asym_weight),
        "stage2_pmf_right_weight": float(stage2_pmf_right_weight),
        "stage2_pmf_target_offset": float(stage2_pmf_target_offset),
        "stage2_pmf_asym_weight_early": float(stage2_pmf_asym_weight_early),
        "stage2_pmf_target_early_offset": float(stage2_pmf_target_early_offset),
        "stage2_pmf_target_mode": str(stage2_pmf_target_mode),
        "stage2_pmf_zone_late_weight": float(stage2_pmf_zone_late_weight),
        "stage2_pmf_zone_too_late_weight": float(stage2_pmf_zone_too_late_weight),
        "stage2_pmf_zone_missed_weight": float(stage2_pmf_zone_missed_weight),
        "stage2_pmf_zone_too_early_weight": float(stage2_pmf_zone_too_early_weight),
        "stage2_pmf_zone_too_late_threshold": float(stage2_pmf_zone_too_late_threshold),
        "stage2_pmf_zone_missed_threshold": float(stage2_pmf_zone_missed_threshold),
        "stage2_pmf_zone_too_early_threshold": float(stage2_pmf_zone_too_early_threshold),
        "stage2_phenology_bias_head": int(stage2_phenology_bias_head),
        "stage2_phenology_hidden": int(stage2_phenology_hidden),
        "stage2_dispatch_features_added": bool(stage2_dispatch_feature_csv),
        "stage2_dispatch_feature_csv": (str(stage2_dispatch_feature_csv)
                                          if stage2_dispatch_feature_csv else None),
        "stage2_dispatch_feature_mode": str(stage2_dispatch_feature_mode),
        "stage2_dispatch_feature_missing_value": float(stage2_dispatch_feature_missing_value),
        "stage2_cohort_dispatch_only": bool(stage2_cohort_dispatch_only),
        "stage2_pmf_mu_mode": str(stage2_pmf_mu_mode),
        "stage2_pmf_lead_min": float(stage2_pmf_lead_min),
        "stage2_pmf_lead_max": float(stage2_pmf_lead_max),
        "stage2_pmf_clim_mid": float(stage2_pmf_clim_mid),
        "stage2_pmf_delta_max": float(stage2_pmf_delta_max),
        "stage2_pmf_alert_tstar_feat_idx": int(stage2_pmf_alert_tstar_feat_idx),
        "stage2_dispatch_channels_raw": bool(stage2_dispatch_feature_csv),
        "stage2_neighbor_history_added": bool(stage2_neighbor_added),
        "stage2_neighbor_feature_names": list(stage2_neighbor_feature_names),
        "stage2_neighbor_decay_km": float(stage2_neighbor_decay_km),
        "stage2_reset_head_mu": bool(stage2_reset_head_mu),
        "stage2_aux_lead_lambda": float(stage2_aux_lead_lambda),
        "stage2_aux_lead_huber_delta": float(stage2_aux_lead_huber_delta),
        "stage2_pmf_long_lead_threshold": float(stage2_pmf_long_lead_threshold),
        "stage2_pmf_long_lead_weight": float(stage2_pmf_long_lead_weight),
        "stage2_pmf_right_anchor": float(stage2_pmf_right_anchor),
        "stage2_gaussian_loss_mode": str(stage2_gaussian_loss_mode),
        "stage2_gaussian_interval_continuity_correction": int(stage2_gaussian_interval_continuity_correction),
        "stage2_gaussian_interval_lambda": float(stage2_gaussian_interval_lambda),
        "gated_val_stage1_ckpt": gated_val_stage1_ckpt,
        "gated_val_stage1_eval_csv": gated_val_stage1_eval_csv,
        "gated_val_stage2_tstar_offset": int(gated_val_stage2_tstar_offset),
        "gated_val_stage2_tstar_offsets": [int(x) for x in gated_val_offsets],
        "gated_val_stage2_tstar_offset_weights": [float(x) for x in gated_val_offset_weights],
        "amp": int(bool(amp)),
        "amp_dtype": str(amp_dtype),
        "stage2_model_kind": "hierarchical_causal_tstar" if grouped_mode else "flat",
        "stage2_nowcast_label_mode": "orig",
        # ---- Offset-aware conditioning config (read back by the grid script via
        # functools.partial to rebuild the exact head_mu / embedding geometry).
        # All-off -> baseline architecture (head_mu.0 = d_model->d_model).
        "stage2_use_offset_embedding": int(bool(_oc_emb)),
        "stage2_offset_embedding_dim": int(_oc_emb_dim),
        "stage2_offset_max": int(_oc_max),
        "stage2_offset_min": int(_oc_min),
        "stage2_use_issue_doy_features": int(bool(_oc_issue)),
        "stage2_offset_doy_period": 365.0,
        # ---- D1/D2 shared-encode + offset-specific heads (read back by the grid
        # script to rebuild the exact head/encoder geometry; all-off -> A/B/C).
        "stage2_use_shared_multi_offset": int(bool(_sm_on)),
        "stage2_mu_head_mode": str(_sm_head_mode),
        "stage2_candidate_offsets": [int(x) for x in _sm_offsets],
        "stage2_shared_band_window": int(_sm_band),
        "stage2_use_offset_residual": int(bool(_sm_resid)),
        "stage2_residual_hidden_dim": int(_sm_resid_hid),
        "stage2_residual_scale": float(_sm_resid_scale),
        "stage2_zero_init_residual": int(bool(_sm_resid_zero)),
        "norm_mean": x_mean,
        "norm_std": x_std,
        "trained_states": trained_states,
        "split_counts": {"train": len(train_s), "val": len(val_s), "test": len(test_s)},
    }
    out_path_resolved = resolve_out_path(run, out_root, out_path)
    torch.save(bundle, out_path_resolved)
    print("saved:", out_path_resolved)
    if wandb_run is not None:
        best_val_nlls = np.asarray([float(x["best_val_nll"]) for x in trained_states], dtype=float)
        best_val_iou80s = np.asarray([float(x["best_val_iou80"]) for x in trained_states], dtype=float)
        best_selection_scores = np.asarray([float(x["best_selection_score"]) for x in trained_states], dtype=float)
        wandb_run.summary["best_val_nll_mean"] = float(best_val_nlls.mean())
        wandb_run.summary["best_val_nll_std"] = float(best_val_nlls.std(ddof=1) if best_val_nlls.size > 1 else 0.0)
        wandb_run.summary["best_val_iou80_mean"] = float(best_val_iou80s.mean())
        wandb_run.summary["best_val_iou80_std"] = float(best_val_iou80s.std(ddof=1) if best_val_iou80s.size > 1 else 0.0)
        wandb_run.summary["best_selection_score_mean"] = float(best_selection_scores.mean())
        wandb_run.summary["best_selection_score_std"] = float(best_selection_scores.std(ddof=1) if best_selection_scores.size > 1 else 0.0)
        wandb_run.summary["checkpoint_path"] = str(out_path_resolved)
        wandb_run.summary["n_train_seeds"] = int(len(trained_states))
        wandb_run.save(str(out_path_resolved), policy="now")
    finish_wandb_run(wandb_run)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, required=True)
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--split_seed", type=int, default=C.SPLIT_SEED)
    p.add_argument("--split_mode", type=str, default="site", choices=["site", "site_year", "temporal", "year"])
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--auto_split_seed", action="store_true")
    p.add_argument("--auto_split_topk", type=int, default=1)
    p.add_argument("--split_seed_from_topk_idx", type=int, default=None)
    p.add_argument("--split_seeds_json", type=str, default=None)
    p.add_argument("--seed_candidates", type=str, default=None)
    p.add_argument("--target_test_interval", type=int, default=None)
    p.add_argument("--tol_test_interval", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--w_interval", type=float, default=None)
    p.add_argument("--w_left", type=float, default=None)
    p.add_argument("--w_right", type=float, default=None)
    p.add_argument("--lambda_mass", type=float, default=0.0)
    p.add_argument("--stage2_entropy_lambda", type=float, default=0.0)
    p.add_argument("--stage2_entropy_conditional", type=int, default=1)
    p.add_argument("--stage2_location_lambda", type=float, default=0.0)
    p.add_argument("--lambda_right_late", type=float, default=0.0)
    p.add_argument("--right_late_tau", type=float, default=220.0)
    p.add_argument("--train_balance_ratio", type=str, default="1:1:1")
    p.add_argument("--stage2_nowcast", action="store_true")
    p.add_argument("--stage2_nowcast_window", type=int, default=56)
    p.add_argument("--stage2_nowcast_stride", type=int, default=7)
    p.add_argument("--stage2_nowcast_tstar_start", type=int, default=None)
    p.add_argument("--stage2_nowcast_only_pre_event", type=int, default=1)
    p.add_argument("--stage2_nowcast_event_time_proxy", type=str, default="r", choices=["r", "mid"])
    p.add_argument("--stage2_nowcast_require_tstar_before_L", type=int, default=1)
    p.add_argument("--stage2_causal_tstar", action="store_true")
    p.add_argument("--stage2_tstar_layers", type=int, default=1)
    p.add_argument("--stage2_use_tstar_scalar_pos", type=int, default=0)
    p.add_argument("--stage2_early_tstar_weight_min", type=float, default=1.0)
    p.add_argument("--stage2_site_year_mean_loss", type=int, default=0)
    p.add_argument("--stage2_time_chunk_size", type=int, default=64)
    p.add_argument("--stage2_conditional_survival", type=int, default=1)
    p.add_argument("--stage2_lead_weighting", type=int, default=0)
    p.add_argument("--target_lead_min", type=int, default=30)
    p.add_argument("--target_lead_max", type=int, default=60)
    p.add_argument("--support_lead_min", type=int, default=15)
    p.add_argument("--support_lead_max", type=int, default=75)
    p.add_argument("--lead_weight_min", type=float, default=0.2)
    p.add_argument("--stage2_mass_lead_weighting", type=int, default=0)
    p.add_argument("--stage2_warm_start_ckpt", type=str, default=None)
    p.add_argument("--stage2_warm_start_seed", type=int, default=None)
    p.add_argument("--stage2_lead_loss_mode", type=str, default="none", choices=["none", "mask", "weighted"])
    p.add_argument("--stage2_lead_min", type=int, default=15)
    p.add_argument("--stage2_lead_max", type=int, default=75)
    p.add_argument("--stage2_mid_lead_min", type=int, default=30)
    p.add_argument("--stage2_mid_lead_max", type=int, default=60)
    p.add_argument("--stage2_late_exclude_days", type=int, default=14)
    p.add_argument("--stage2_lead_weight_1_14", type=float, default=0.0)
    p.add_argument("--stage2_lead_weight_15_29", type=float, default=0.7)
    p.add_argument("--stage2_lead_weight_30_60", type=float, default=1.5)
    p.add_argument("--stage2_lead_weight_61_75", type=float, default=1.0)
    p.add_argument("--stage2_lead_weight_gt75", type=float, default=0.25)
    p.add_argument("--stage2_best_metric", type=str, default="val_iou80", choices=["val_iou80", "gated_val_iou80", "gated_val_interval_hit_f1", "gated_val_mae_int"])
    p.add_argument("--gated_val_stage1_ckpt", type=str, default=None)
    p.add_argument("--gated_val_stage1_eval_csv", type=str, default=None)
    p.add_argument("--gated_val_stage2_tstar_offset", type=int, default=60)
    p.add_argument("--gated_val_stage2_tstar_offsets", type=str, default=None)
    p.add_argument("--gated_val_stage2_tstar_offset_weights", type=str, default=None)
    p.add_argument("--max_epochs_override", type=int, default=None)
    p.add_argument("--patience_override", type=int, default=None)
    p.add_argument("--num_workers_override", type=int, default=None)
    p.add_argument("--batch_train_override", type=int, default=None)
    p.add_argument("--batch_eval_override", type=int, default=None)
    p.add_argument("--doy_start_override", type=int, default=None)
    p.add_argument("--doy_end_override", type=int, default=None)
    p.add_argument("--d_model_override", type=int, default=None)
    p.add_argument("--amp", type=int, default=0)
    p.add_argument("--amp_dtype", type=str, default="bf16", choices=["bf16", "fp16"])
    p.add_argument("--save_epoch_checkpoints", type=int, default=0)
    p.add_argument("--stage2_sanity_only", type=int, default=0)
    p.add_argument("--stage2_sanity_batches", type=int, default=1)
    p.add_argument("--stage2_pmf_mode", type=str, default="hazard", choices=["hazard", "gaussian"])
    p.add_argument("--stage2_pmf_sigma", type=float, default=5.0)
    p.add_argument("--stage2_pmf_mu_max", type=float, default=0.0,
                   help="Upper bound on mu in DOY units; 0 means use Tend.")
    p.add_argument("--stage2_pmf_asym_weight", type=float, default=10.0,
                   help="Weight on (mu - L)^2 when mu > L (predicted later than the day before event).")
    p.add_argument("--stage2_pmf_right_weight", type=float, default=0.3,
                   help="Weight on right-censored MSE term.")
    p.add_argument("--stage2_pmf_target_offset", type=float, default=0.0,
                   help="Loss target = L + offset for event rows. e.g. +5 to bias mu into a 15-day interval.")
    p.add_argument("--stage2_pmf_asym_weight_early", type=float, default=0.0,
                   help="One-sided early MSE weight: penalize mu < L - target_early_offset. 0 disables.")
    p.add_argument("--stage2_pmf_target_early_offset", type=float, default=30.0,
                   help="Early lower bound = L - target_early_offset (days). mu below this is penalized.")
    p.add_argument("--stage2_pmf_target_mode", type=str, default="l_offset",
                   choices=["l_offset", "center"],
                   help="'l_offset': legacy target = L + target_offset with asym_weight. "
                        "'center': target = (L+R)/2, zone-aware soft penalties (zone_* hparams).")
    p.add_argument("--stage2_pmf_zone_late_weight", type=float, default=0.0,
                   help="Soft penalty weight for mu > mid (center mode). 0 disables.")
    p.add_argument("--stage2_pmf_zone_too_late_weight", type=float, default=0.0,
                   help="Soft penalty weight for mu > L + zone_too_late_threshold. 0 disables.")
    p.add_argument("--stage2_pmf_zone_missed_weight", type=float, default=0.0,
                   help="Soft penalty weight for mu > L + zone_missed_threshold (MISSED zone). 0 disables.")
    p.add_argument("--stage2_pmf_zone_too_early_weight", type=float, default=0.0,
                   help="Soft penalty weight for mu < L - zone_too_early_threshold. 0 disables.")
    p.add_argument("--stage2_pmf_zone_too_late_threshold", type=float, default=15.0,
                   help="Days after L marking USEFUL→TOO_LATE boundary (mu > L + this is too late).")
    p.add_argument("--stage2_pmf_zone_missed_threshold", type=float, default=22.0,
                   help="Days after L marking MISSED entry (mu > L + this is post-event).")
    p.add_argument("--stage2_pmf_zone_too_early_threshold", type=float, default=23.0,
                   help="Days before L marking TOO_EARLY entry (mu < L - this is too early).")
    p.add_argument("--stage2_phenology_bias_head", type=int, default=0,
                   help="1 = enable phenology bias head (Architecture A). "
                        "Uses 4 site-year static features (best_suitability, best_months, "
                        "offset_days, window_idx) routed through a small MLP and added to mu. "
                        "Only active when stage2_pmf_mode='gaussian'.")
    p.add_argument("--stage2_phenology_hidden", type=int, default=8,
                   help="Hidden width of phen_head MLP; 0 means linear (Linear(4,1)).")
    # Phase S5: long-lead per-sample weighting for the Gaussian mu-loss.
    p.add_argument("--stage2_pmf_long_lead_threshold", type=float, default=0.0,
                   help="Per-sample weight kicks in when (L+1)-tstar >= threshold (days). "
                        "0 disables; defaults are legacy unweighted behaviour.")
    p.add_argument("--stage2_pmf_long_lead_weight", type=float, default=1.0,
                   help="Multiplier on the event/early mu-loss for samples whose lead >= "
                        "stage2_pmf_long_lead_threshold. 1.0 is a no-op.")
    # Phase S10: right-cens loss anchor (replaces the implicit Tend target).
    p.add_argument("--stage2_dispatch_feature_csv", type=str, default=None,
                   help="Per-(site,year) dispatch confidence-feature CSV produced "
                        "by build_dispatch_feature_table.py. If set, 15 channels "
                        "(14 features + 1 missing indicator) are appended to X "
                        "before nowcast slicing.")
    p.add_argument("--stage2_dispatch_feature_mode", type=str,
                   default="causal", choices=["causal", "broadcast"],
                   help="'causal': rows with tstar < alert_t_rel are zero-padded "
                        "with missing=1; 'broadcast': all rows of an alerted sy "
                        "carry features (leakage; sanity baseline only).")
    p.add_argument("--stage2_dispatch_feature_missing_value", type=float,
                   default=0.0,
                   help="Fill value for the 14 feature slots when no feature "
                        "applies (alert not yet occurred / never occurred).")
    p.add_argument("--stage2_cohort_dispatch_only", action="store_true",
                   help="Phase B: restrict training cohort to dispatch-alerted "
                        "site-years (sy present in --stage2_dispatch_feature_csv). "
                        "Required when mu_mode=lead_from_alert is used so the "
                        "lead target is always defined.")
    p.add_argument("--stage2_add_neighbor_history", action="store_true",
                   help="DIRECT neighbor occurrence features: append 6 neighbor "
                        "channels to Stage-2 X right after build_samples_season "
                        "(before dispatch). OFF by default -> baseline unchanged.")
    p.add_argument("--stage2_neighbor_decay_km", type=float, default=20.0,
                   help="decay length (km) for neighbor_weighted_* channel "
                        "(default 20.0); only used with --stage2_add_neighbor_history.")
    p.add_argument("--stage2_pmf_mu_mode", type=str, default="absolute",
                   choices=["absolute", "lead_from_alert", "residual_clim", "prior_residual_alert_bin"],
                   help="absolute: existing sigmoid*T mu head (Phase A/old Best). "
                        "lead_from_alert: mu_DOY = alert_tstar + bounded-sigmoid "
                        "lead. Requires --stage2_dispatch_feature_csv to provide "
                        "the alert_tstar channel. "
                        "residual_clim: mu_DOY = clim_mid + delta_max * tanh(raw). "
                        "Requires --stage2_pmf_clim_mid (per-pest mean_mid DOY).")
    p.add_argument("--stage2_pmf_lead_min", type=float, default=7.0,
                   help="Lower bound (days) for bounded-sigmoid lead head.")
    p.add_argument("--stage2_pmf_lead_max", type=float, default=75.0,
                   help="Upper bound (days) for bounded-sigmoid lead head.")
    p.add_argument("--stage2_pmf_clim_mid", type=float, default=0.0,
                   help="Per-pest climatology mean_mid in DOY units (e.g. 198.7 "
                        "for sheath_blight). Required when "
                        "--stage2_pmf_mu_mode=residual_clim. Converted to "
                        "1-based season-index coords internally.")
    p.add_argument("--stage2_pmf_delta_max", type=float, default=60.0,
                   help="Half-range of tanh-bounded delta in residual_clim mode "
                        "(days). mu = clim_mid + delta_max * tanh(raw). Default 60.")
    p.add_argument("--stage2_reset_head_mu", action="store_true",
                   help="Phase B headreset: drop head_mu.* tensors from the "
                        "warm-start ckpt so the head is trained from a fresh "
                        "init. backbone/in_proj/encoder/phen_head still come "
                        "from warm-start. Intended for lead_from_alert mode "
                        "when probe shows z carries timing signal that the "
                        "absolute-DOY-trained head cannot read.")
    p.add_argument("--stage2_aux_lead_lambda", type=float, default=0.0,
                   help="Phase B auxiliary lead loss weight. When > 0, adds "
                        "lambda * Huber(mu - L, 0; delta=--stage2_aux_lead_huber_delta) "
                        "to the asymmetric mu loss on the SAME (interval & "
                        "lead_loss_mask) cell set. Unweighted (no sample_weight, "
                        "no asym). Default 0 = no aux loss.")
    p.add_argument("--stage2_aux_lead_huber_delta", type=float, default=10.0,
                   help="Huber delta for the aux lead loss (days). Default 10.")
    p.add_argument("--stage2_gaussian_loss_mode", type=str, default="asym_mse",
                   choices=["asym_mse", "interval_nll", "mixed"],
                   help="Stage 2 Gaussian PMF mu-head loss family. "
                        "asym_mse (default, backward-compatible): legacy "
                        "asymmetric_mu_loss (regression-style, optional zone/early "
                        "penalties). interval_nll: gaussian_interval_nll_loss "
                        "= -log P(L < T <= R) with fixed sigma=--stage2_pmf_sigma; "
                        "right-cens uses Gaussian survival -log P(T > C); ignores "
                        "asym_weight / zone_* / aux_lead_lambda. mixed: "
                        "asym_mse + lambda * interval_nll on the same mu output; "
                        "lambda controlled by --stage2_gaussian_interval_lambda. "
                        "Only active when --stage2_pmf_mode=gaussian.")
    p.add_argument("--stage2_gaussian_interval_lambda", type=float, default=0.1,
                   help="Mixed-loss weight on the interval_nll term when "
                        "--stage2_gaussian_loss_mode=mixed. Total Gaussian PMF "
                        "loss is asym_mse + lambda * interval_nll. Default 0.1 "
                        "(mild auxiliary signal). Ignored for other loss modes.")
    p.add_argument("--stage2_gaussian_interval_continuity_correction", type=int, default=0,
                   choices=[0, 1],
                   help="Continuity-correction toggle for the interval_nll loss. "
                        "0 (default): use raw L, R as the half-open Gaussian "
                        "boundary P(L < T <= R). 1: use (L+0.5, R+0.5) as the "
                        "continuous proxies for the day-inclusive [L+1, R] "
                        "discrete interval. Only used when "
                        "--stage2_gaussian_loss_mode=interval_nll.")
    p.add_argument("--stage2_pmf_right_anchor", type=float, default=0.0,
                   help="Target DOY anchoring the right-cens MSE: "
                        "loss_right = right_weight * mean((mu - right_anchor)^2). "
                        "0 disables (falls back to Tend, legacy behaviour); >0 lets "
                        "the right-cens pull aim at a less-extreme target "
                        "(e.g., 220 ≈ mid_max, 240 ≈ L_max).")
    # Stage-1 style aliases for pipeline consistency.
    p.add_argument("--nowcast_window", dest="stage2_nowcast_window", type=int)
    p.add_argument("--nowcast_stride", dest="stage2_nowcast_stride", type=int)
    p.add_argument("--nowcast_tstar_start", dest="stage2_nowcast_tstar_start", type=int)
    p.add_argument("--nowcast_only_pre_event", dest="stage2_nowcast_only_pre_event", type=int)
    p.add_argument("--nowcast_event_time_proxy", dest="stage2_nowcast_event_time_proxy", type=str, choices=["r", "mid"])
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="agro-rice")
    p.add_argument("--wandb_entity", type=str, default=WANDB_ENTITY_DEFAULT)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default=None)
    p.add_argument("--wandb_job_type", type=str, default="train")
    args = p.parse_args()
    main(
        args.pest,
        args.run,
        args.out_root,
        args.out,
        args.split_seed,
        args.split_mode,
        args.seeds,
        args.auto_split_seed,
        args.seed_candidates,
        args.target_test_interval,
        args.tol_test_interval,
        args.auto_split_topk,
        args.split_seed_from_topk_idx,
        args.split_seeds_json,
        args.dropout,
        args.weight_decay,
        args.lr,
        args.w_interval,
        args.w_left,
        args.w_right,
        args.lambda_mass,
        args.stage2_entropy_lambda,
        args.stage2_entropy_conditional,
        args.stage2_location_lambda,
        args.lambda_right_late,
        args.right_late_tau,
        args.train_balance_ratio,
        args.stage2_nowcast,
        args.stage2_nowcast_window,
        args.stage2_nowcast_stride,
        args.stage2_nowcast_tstar_start,
        args.stage2_nowcast_only_pre_event,
        args.stage2_nowcast_event_time_proxy,
        args.stage2_nowcast_require_tstar_before_L,
        args.stage2_causal_tstar,
        args.stage2_tstar_layers,
        args.stage2_use_tstar_scalar_pos,
        args.stage2_early_tstar_weight_min,
        args.stage2_site_year_mean_loss,
        args.stage2_time_chunk_size,
        args.stage2_conditional_survival,
        args.stage2_lead_weighting,
        args.target_lead_min,
        args.target_lead_max,
        args.support_lead_min,
        args.support_lead_max,
        args.lead_weight_min,
        args.stage2_mass_lead_weighting,
        args.stage2_warm_start_ckpt,
        args.stage2_warm_start_seed,
        args.stage2_lead_loss_mode,
        args.stage2_lead_min,
        args.stage2_lead_max,
        args.stage2_mid_lead_min,
        args.stage2_mid_lead_max,
        args.stage2_late_exclude_days,
        args.stage2_lead_weight_1_14,
        args.stage2_lead_weight_15_29,
        args.stage2_lead_weight_30_60,
        args.stage2_lead_weight_61_75,
        args.stage2_lead_weight_gt75,
        args.stage2_best_metric,
        args.gated_val_stage1_ckpt,
        args.gated_val_stage1_eval_csv,
        args.gated_val_stage2_tstar_offset,
        args.gated_val_stage2_tstar_offsets,
        args.gated_val_stage2_tstar_offset_weights,
        args.max_epochs_override,
        args.patience_override,
        args.num_workers_override,
        args.batch_train_override,
        args.batch_eval_override,
        args.doy_start_override,
        args.doy_end_override,
        args.d_model_override,
        args.amp,
        args.amp_dtype,
        args.use_wandb,
        args.wandb_project,
        args.wandb_entity,
        args.wandb_group,
        args.wandb_run_name,
        args.wandb_tags,
        args.wandb_job_type,
        args.save_epoch_checkpoints,
        args.stage2_sanity_only,
        args.stage2_sanity_batches,
        args.stage2_pmf_mode,
        args.stage2_pmf_sigma,
        args.stage2_pmf_mu_max,
        args.stage2_pmf_asym_weight,
        args.stage2_pmf_right_weight,
        args.stage2_pmf_target_offset,
        args.stage2_pmf_asym_weight_early,
        args.stage2_pmf_target_early_offset,
        args.stage2_pmf_target_mode,
        args.stage2_pmf_zone_late_weight,
        args.stage2_pmf_zone_too_late_weight,
        args.stage2_pmf_zone_missed_weight,
        args.stage2_pmf_zone_too_early_weight,
        args.stage2_pmf_zone_too_late_threshold,
        args.stage2_pmf_zone_missed_threshold,
        args.stage2_pmf_zone_too_early_threshold,
        args.stage2_phenology_bias_head,
        args.stage2_phenology_hidden,
        args.stage2_pmf_long_lead_threshold,
        args.stage2_pmf_long_lead_weight,
        args.stage2_pmf_right_anchor,
        args.val_year,
        args.test_year_min,
        args.test_year_max,
        args.stage2_dispatch_feature_csv,
        args.stage2_dispatch_feature_mode,
        args.stage2_dispatch_feature_missing_value,
        args.stage2_cohort_dispatch_only,
        args.stage2_pmf_mu_mode,
        args.stage2_pmf_lead_min,
        args.stage2_pmf_lead_max,
        args.stage2_pmf_clim_mid,
        args.stage2_pmf_delta_max,
        args.stage2_reset_head_mu,
        args.stage2_aux_lead_lambda,
        args.stage2_aux_lead_huber_delta,
        args.stage2_gaussian_loss_mode,
        args.stage2_gaussian_interval_continuity_correction,
        args.stage2_gaussian_interval_lambda,
        stage2_add_neighbor_history=args.stage2_add_neighbor_history,
        stage2_neighbor_decay_km=args.stage2_neighbor_decay_km,
    )
