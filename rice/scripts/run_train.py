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
from rice.src.model import HazardTransformer, HierarchicalCausalHazardTransformer
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
    samples, dropped, feature_names = build_samples_season(df_season, feature_cols, C.DOY_START, C.DOY_END)
    print("samples:", len(samples), "| dropped groups (len!=T):", dropped)
    print(f"[time] build_samples_season={time.perf_counter()-t0:.2f}s")
    print(f"[features] n={len(feature_names)} head={feature_names[:5]} tail={feature_names[-5:]}")

    # =========================
    # 8) Split + norm + datasets
    # =========================
    result = None
    chosen = None
    if split_seeds_json is not None:
        split_seeds_json_path = resolve_split_seeds_json_path(out_root, split_seeds_json)
        split_seed, chosen_idx, chosen, payload = load_split_seed_from_topk(split_seeds_json_path, split_seed_from_topk_idx)
        train_s, val_s, test_s = split_samples(samples, val_frac=0.1, test_frac=0.1, seed=split_seed, split_mode=split_mode)
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
        train_s, val_s, test_s = split_samples(samples, val_frac=0.1, test_frac=0.1, seed=split_seed, split_mode=split_mode)
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
        train_s, val_s, test_s = split_samples(samples, val_frac=0.1, test_frac=0.1, seed=split_seed, split_mode=split_mode)

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
                X_i, L_i, R_i, c_i, tstar_i = train_ds[idx]
                print(
                    f"[debug] train_ds[{idx}] X.shape={tuple(X_i.shape)} X.dtype={X_i.dtype} "
                    f"L.shape={tuple(L_i.shape)} R.shape={tuple(R_i.shape)} c.shape={tuple(c_i.shape)} "
                    f"tstar.shape={tuple(tstar_i.shape)}"
                )
            else:
                X_i, L_i, R_i, c_i = train_ds[idx]
                print(
                    f"[debug] train_ds[{idx}] X.shape={tuple(X_i.shape)} X.dtype={X_i.dtype} "
                    f"L={int(L_i)} R={int(R_i)} c={int(c_i)}"
                )

    D_in = int(train_ds[0][0].shape[-1])
    print(f"[D_in] computed_from_dataset={D_in}")

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
            ).to(device)
            model.early_tstar_weight_min = float(stage2_early_tstar_weight_min)
            model.site_year_mean_loss = bool(stage2_site_year_mean_loss)
            model.time_chunk_size = int(stage2_time_chunk_size)
            model.conditional_survival = bool(stage2_conditional_survival)
            model.pmf_mode = str(stage2_pmf_mode)
            model.gaussian_sigma = float(stage2_pmf_sigma)
            model.gaussian_mu_max = float(stage2_pmf_mu_max)
            model.asym_weight = float(stage2_pmf_asym_weight)
            model.right_weight = float(stage2_pmf_right_weight)
            model.target_offset = float(stage2_pmf_target_offset)
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
            missing, unexpected = model.load_state_dict(warm_state["state_dict"], strict=False)
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
    p.add_argument("--split_mode", type=str, default="site", choices=["site", "site_year", "temporal"])
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
    )
