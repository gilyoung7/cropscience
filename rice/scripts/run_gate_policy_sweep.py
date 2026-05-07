from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rice.configs import config as C
from rice.scripts.common import (
    collate_grouped_stage2,
    finish_wandb_run,
    init_wandb_run,
    make_loader,
    parse_tags,
)
from rice.scripts.run_event_eval import apply_temperature, best_tau_by_target, fit_temperature_grid
from rice.scripts.run_event_train import (
    EventTransformer,
    build_nowcast_samples,
    build_tabular_from_samples,
    make_event_labels,
)
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_viz_interval import (
    WANDB_ENTITY_DEFAULT,
    WANDB_PROJECT_DEFAULT,
    build_alert_map,
    collect_interval_preds,
    collect_interval_preds_grouped,
    predict_event_prob_event_model,
    resolve_ckpt_path,
)
from rice.src.dataset import (
    GroupedIntervalEventDataset,
    IntervalEventDataset,
    build_stage2_nowcast_samples,
    compute_norm_stats,
    group_stage2_samples_by_site_year,
    log_split_fingerprint,
    split_by_site,
)
from rice.src.model import HazardTransformer, HierarchicalCausalHazardTransformer
from rice.src.pest_resolver import default_out_root, ensure_output_dirs, resolve_pest
from rice.src.train_eval import early_recall80_site_year, hazard_to_pmf_cdf_logS, overlap_metrics


def parse_threshold_specs(raw: str) -> list[str | float]:
    out: list[str | float] = []
    for x in str(raw).split(","):
        s = x.strip()
        if not s:
            continue
        if s.lower() == "tau":
            out.append("tau")
        else:
            out.append(float(s))
    return out


def parse_int_list(raw: str) -> list[int]:
    return [int(x.strip()) for x in str(raw).split(",") if x.strip()]


def site_year_key(s: dict) -> str:
    return f"{str(s.get('site_id', ''))}-{int(s.get('year', -1))}"


def event_site_years(samples: list[dict]) -> set[str]:
    return {site_year_key(s) for s in samples if str(s.get("censor_type", "")) == "interval"}


def all_site_years(samples: list[dict]) -> set[str]:
    return {site_year_key(s) for s in samples}


def build_stage2_row_map(rows: list[dict]) -> dict[tuple[str, int], dict]:
    out = {}
    for r in rows:
        tstar = r.get("tstar")
        if tstar is None:
            continue
        out[(str(r["sample_id"]), int(tstar))] = r
    return out


def add_mass_in_true_interval(row: dict) -> float:
    pmf = np.asarray(row.get("pmf", []), dtype=float)
    if pmf.size == 0:
        return float("nan")
    l = max(1, int(row["true_L"]))
    r = min(int(row["true_R"]), int(pmf.size))
    if r < l:
        return 0.0
    return float(np.sum(pmf[l - 1 : r]))


def choose_stage2_tstar(alert_t: int, offset: int, available_tstars: list[int]) -> tuple[int | None, bool]:
    target = int(alert_t) + int(offset)
    later = [int(t) for t in available_tstars if int(t) >= target]
    if not later:
        return None, False
    chosen = min(later)
    return int(chosen), bool(chosen == target)


def compute_policy_metrics(
    *,
    split_name: str,
    split_seed: int,
    seed: int,
    threshold: float,
    threshold_policy: str,
    offset: int,
    anchor_threshold: str | float | None,
    anchor_probs: np.ndarray | None,
    samples1: list[dict],
    probs: np.ndarray,
    source_samples2: list[dict],
    stage2_rows: list[dict],
) -> dict:
    if anchor_threshold is None:
        alert_map = build_alert_map(samples1, probs, float(threshold), t_alert_start=None)
        alert_threshold = float(threshold)
    else:
        if anchor_probs is None:
            raise ValueError("anchor_probs must be provided when anchor_threshold is set")
        alert_map = build_alert_map(samples1, anchor_probs, float(anchor_threshold), t_alert_start=None)
        alert_threshold = float(anchor_threshold)
    row_map = build_stage2_row_map(stage2_rows)

    tstars_by_sid: dict[str, list[int]] = {}
    for r in stage2_rows:
        if r.get("tstar") is None:
            continue
        tstars_by_sid.setdefault(str(r["sample_id"]), []).append(int(r["tstar"]))
    for sid, vals in tstars_by_sid.items():
        tstars_by_sid[sid] = sorted(set(vals))

    event_sids = event_site_years(source_samples2)
    total_sids = all_site_years(source_samples2)

    matched_rows = []
    matched_event_sids = set()
    pred_event_sids = set()
    post_start_count = 0
    missing_stage2_count = 0
    exact_offset_count = 0

    for sid, alert_t in alert_map.items():
        available = tstars_by_sid.get(str(sid), [])
        stage2_tstar, exact_offset = choose_stage2_tstar(int(alert_t), int(offset), available)
        if stage2_tstar is None:
            missing_stage2_count += 1
            continue
        pred_event_sids.add(str(sid))
        exact_offset_count += int(exact_offset)
        row = row_map.get((str(sid), int(stage2_tstar)))
        if row is None:
            missing_stage2_count += 1
            continue
        row = dict(row)
        row["alert_tstar"] = int(alert_t)
        row["stage2_tstar"] = int(stage2_tstar)
        row["offset"] = int(offset)
        true_start = int(row["true_L"]) + 1
        row["is_post_true_start"] = bool(int(stage2_tstar) >= true_start)
        post_start_count += int(row["is_post_true_start"])
        matched_rows.append(row)
        matched_event_sids.add(str(sid))

    ious, recs, precs, maes, masses = [], [], [], [], []
    tp_overlap = 0
    for r in matched_rows:
        iou, rec, prec = overlap_metrics(r["pred_L"], r["pred_R"], r["true_L"], r["true_R"])
        ious.append(float(iou))
        recs.append(float(rec))
        precs.append(float(prec))
        pred_mid = (int(r["pred_L"]) + int(r["pred_R"])) / 2.0
        true_mid = (int(r["true_L"]) + int(r["true_R"])) / 2.0
        maes.append(float(abs(pred_mid - true_mid)))
        masses.append(add_mass_in_true_interval(r))
        tp_overlap += int(iou > 0.0)

    early_rows = []
    for r in matched_rows:
        rr = dict(r)
        # For offset policies, early warning requires the original Stage1 alert
        # and the Stage2 row used for interval prediction to both be pre-event.
        rr["tstar"] = max(int(rr["alert_tstar"]), int(rr["stage2_tstar"]))
        early_rows.append(rr)
    early_recall80, early_success, early_denom = early_recall80_site_year(early_rows)

    n_event_total = len(event_sids)
    n_total = len(total_sids)
    stage1_pred_pos = set(str(k) for k in alert_map.keys())
    stage1_tp = len(stage1_pred_pos & event_sids)
    stage1_precision = stage1_tp / len(stage1_pred_pos) if stage1_pred_pos else 0.0
    stage1_recall = stage1_tp / n_event_total if n_event_total > 0 else 0.0
    stage1_f1 = (
        2 * stage1_precision * stage1_recall / (stage1_precision + stage1_recall)
        if (stage1_precision + stage1_recall) > 0
        else 0.0
    )

    interval_precision = tp_overlap / len(matched_rows) if matched_rows else 0.0
    interval_recall = tp_overlap / n_event_total if n_event_total > 0 else 0.0
    interval_f1 = (
        2 * interval_precision * interval_recall / (interval_precision + interval_recall)
        if (interval_precision + interval_recall) > 0
        else 0.0
    )

    return {
        "split": split_name,
        "split_seed": int(split_seed),
        "seed": int(seed),
        "threshold": float(threshold),
        "threshold_policy": str(threshold_policy),
        "alert_threshold": float(alert_threshold),
        "offset": int(offset),
        "N_site_year": int(n_total),
        "N_event_site_year": int(n_event_total),
        "stage1_alert_site_year": int(len(stage1_pred_pos)),
        "stage1_gate_precision": float(stage1_precision),
        "stage1_gate_recall": float(stage1_recall),
        "stage1_gate_f1": float(stage1_f1),
        "matched_event_site_year": int(len(matched_event_sids)),
        "missing_stage2_count": int(missing_stage2_count),
        "exact_offset_count": int(exact_offset_count),
        "post_true_start_count": int(post_start_count),
        "post_true_start_rate": float(post_start_count / len(matched_rows)) if matched_rows else 0.0,
        "IoU80": float(np.mean(ious)) if ious else float("nan"),
        "Rec80": float(np.mean(recs)) if recs else float("nan"),
        "Prec80": float(np.mean(precs)) if precs else float("nan"),
        "MAE_int": float(np.mean(maes)) if maes else float("nan"),
        "Mass_int": float(np.nanmean(np.asarray(masses, dtype=float))) if masses else float("nan"),
        "interval_hit_precision": float(interval_precision),
        "interval_hit_recall": float(interval_recall),
        "interval_hit_f1": float(interval_f1),
        "EarlyRecall80": float(early_recall80),
        "EarlyRecall80_success": int(early_success),
        "EarlyRecall80_denominator": int(early_denom),
    }


def select_policy(val_df: pd.DataFrame, select_metric: str) -> pd.DataFrame:
    grouped = (
        val_df.groupby(["threshold_policy", "offset"], as_index=False)
        .agg(
            {
                "threshold": "mean",
                select_metric: "mean",
                "EarlyRecall80": "mean",
                "stage1_gate_recall": "mean",
                "interval_hit_f1": "mean",
                "IoU80": "mean",
            }
        )
    )
    grouped = grouped.sort_values(
        [select_metric, "EarlyRecall80", "stage1_gate_recall", "interval_hit_f1", "IoU80"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)
    grouped["rank"] = np.arange(1, len(grouped) + 1)
    return grouped


def main(
    pest: str,
    run: int,
    stage1_ckpt: str,
    stage2_ckpt: str,
    out_root: str,
    split_seed: int,
    seeds: list[int] | None,
    thresholds: list[str | float],
    offsets: list[int],
    anchor_threshold: float | None,
    tau_mode: str,
    tau_target_precision: float,
    tau_target_recall: float,
    select_metric: str,
    use_wandb: bool,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_group: str | None,
    wandb_run_name: str | None,
    wandb_tags: str | None,
):
    _, get_feature_cols = resolve_pest(pest)
    if not out_root:
        out_root = default_out_root(pest)
    ensure_output_dirs(out_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    wandb_run = init_wandb_run(
        use_wandb=use_wandb,
        project=wandb_project or WANDB_PROJECT_DEFAULT,
        entity=wandb_entity or WANDB_ENTITY_DEFAULT,
        run_name=wandb_run_name,
        group=wandb_group,
        job_type="gate_policy_sweep",
        tags=parse_tags(wandb_tags) + [f"pest:{pest}", "script:run_gate_policy_sweep"],
        config={
            "pest": pest,
            "run": int(run),
            "split_seed": int(split_seed),
            "thresholds": list(thresholds),
            "offsets": list(offsets),
            "anchor_threshold": None if anchor_threshold is None else float(anchor_threshold),
            "tau_mode": str(tau_mode),
            "tau_target_precision": float(tau_target_precision),
            "tau_target_recall": float(tau_target_recall),
            "select_metric": str(select_metric),
        },
    )

    stage1_path = resolve_ckpt_path(run, out_root, stage1_ckpt, "event_classifier_run{run}.pt")
    stage2_path = resolve_ckpt_path(run, out_root, stage2_ckpt, "checkpoint_run{run}.pt")
    print(f"Using stage1 ckpt: {stage1_path}")
    print(f"Using stage2 ckpt: {stage2_path}")
    ckpt1 = torch.load(stage1_path, map_location="cpu")
    ckpt2 = torch.load(stage2_path, map_location="cpu")

    C.DOY_START = int(ckpt2.get("doy_start", C.DOY_START))
    C.DOY_END = int(ckpt2.get("doy_end", C.DOY_END))
    C.D_MODEL = int(ckpt2.get("d_model", C.D_MODEL))
    C.N_HEAD = int(ckpt2.get("n_head", C.N_HEAD))
    C.N_LAYERS = int(ckpt2.get("n_layers", C.N_LAYERS))

    _feature_cols, feature_names_eval, T, samples = build_samples_for_run(run, get_feature_cols)
    print(f"[features] n={len(feature_names_eval)} head={feature_names_eval[:5]} tail={feature_names_eval[-5:]}")
    train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)

    task_mode = str(ckpt1.get("task_mode", "season_complete"))
    if task_mode != "nowcast":
        raise ValueError("Stage1 checkpoint must be nowcast")
    nowcast_window = int(ckpt1.get("nowcast_window", 28))
    nowcast_stride = int(ckpt1.get("nowcast_stride", 7))
    nowcast_tstar_start = ckpt1.get("nowcast_tstar_start", None)
    nowcast_only_pre_event = int(ckpt1.get("nowcast_only_pre_event", 1))
    nowcast_event_time_proxy = str(ckpt1.get("nowcast_event_time_proxy", "r"))
    add_tstar_position_feature = bool(ckpt1.get("add_tstar_position_feature", False))
    val_s1 = build_nowcast_samples(
        val_s,
        window=nowcast_window,
        stride=nowcast_stride,
        tstar_start=nowcast_tstar_start,
        only_pre_event=bool(nowcast_only_pre_event),
        event_time_proxy=nowcast_event_time_proxy,
    )
    test_s1 = build_nowcast_samples(
        test_s,
        window=nowcast_window,
        stride=nowcast_stride,
        tstar_start=nowcast_tstar_start,
        only_pre_event=bool(nowcast_only_pre_event),
        event_time_proxy=nowcast_event_time_proxy,
    )

    stage2_nowcast = bool(ckpt2.get("stage2_nowcast", False))
    if not stage2_nowcast:
        raise ValueError("Stage2 checkpoint must be nowcast for gate policy sweep")
    stage2_nowcast_window = int(ckpt2.get("stage2_nowcast_window", 56))
    stage2_nowcast_stride = int(ckpt2.get("stage2_nowcast_stride", 7))
    stage2_nowcast_tstar_start = ckpt2.get("stage2_nowcast_tstar_start", None)
    stage2_nowcast_only_pre_event = int(ckpt2.get("stage2_nowcast_only_pre_event", 1))
    stage2_nowcast_event_time_proxy = str(ckpt2.get("stage2_nowcast_event_time_proxy", "r"))
    val_s2 = build_stage2_nowcast_samples(
        val_s,
        window=stage2_nowcast_window,
        stride=stage2_nowcast_stride,
        tstar_start=stage2_nowcast_tstar_start,
        only_pre_event=bool(stage2_nowcast_only_pre_event),
        event_time_proxy=stage2_nowcast_event_time_proxy,
    )
    test_s2 = build_stage2_nowcast_samples(
        test_s,
        window=stage2_nowcast_window,
        stride=stage2_nowcast_stride,
        tstar_start=stage2_nowcast_tstar_start,
        only_pre_event=bool(stage2_nowcast_only_pre_event),
        event_time_proxy=stage2_nowcast_event_time_proxy,
    )
    log_split_fingerprint("gate_policy_sweep", train_s, val_s, test_s)

    x_mean, x_std = compute_norm_stats(train_s)
    grouped_mode = bool(
        ckpt2.get("stage2_causal_tstar", False)
        or str(ckpt2.get("stage2_model_kind", "flat")) == "hierarchical_causal_tstar"
    )
    stage2_tstar_layers = int(ckpt2.get("stage2_tstar_layers", 1))
    stage2_use_tstar_scalar_pos = int(ckpt2.get("stage2_use_tstar_scalar_pos", 0))
    if grouped_mode:
        val_groups2 = group_stage2_samples_by_site_year(val_s2)
        test_groups2 = group_stage2_samples_by_site_year(test_s2)
        val_ds2 = GroupedIntervalEventDataset(val_groups2, x_mean, x_std)
        test_ds2 = GroupedIntervalEventDataset(test_groups2, x_mean, x_std)
        val_loader2 = make_loader(val_ds2, 1, shuffle=False, collate_fn=collate_grouped_stage2)
        test_loader2 = make_loader(test_ds2, 1, shuffle=False, collate_fn=collate_grouped_stage2)
    else:
        val_groups2 = []
        test_groups2 = []
        val_ds2 = IntervalEventDataset(val_s2, x_mean, x_std)
        test_ds2 = IntervalEventDataset(test_s2, x_mean, x_std)
        val_loader2 = make_loader(val_ds2, C.BATCH_EVAL, shuffle=False)
        test_loader2 = make_loader(test_ds2, C.BATCH_EVAL, shuffle=False)

    y_val = make_event_labels(val_s1)
    all_rows = []
    for d2 in ckpt2["trained_states"]:
        seed = int(d2["seed"])
        if seeds is not None and seed not in seeds:
            continue

        d1 = next((s for s in ckpt1["trained_states"] if int(s["seed"]) == seed), None)
        if d1 is None:
            print(f"[seed {seed}] no stage1 state; skipping")
            continue

        model_kind = str(ckpt1.get("event_model", "transformer"))
        if model_kind == "transformer":
            model1 = EventTransformer(
                d_in=int(val_ds2[0][0].shape[-1]),
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=2,
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
            ).to(device)
            model1.load_state_dict(d1["state_dict"])
            model1.eval()
            val_loader1 = make_loader(IntervalEventDataset(val_s1, x_mean, x_std), C.BATCH_EVAL, shuffle=False)
            test_loader1 = make_loader(IntervalEventDataset(test_s1, x_mean, x_std), C.BATCH_EVAL, shuffle=False)
            p_val_raw = predict_event_prob_event_model(model1, val_loader1, device=device)
            p_test_raw = predict_event_prob_event_model(model1, test_loader1, device=device)
        else:
            clf = d1.get("sk_model")
            if clf is None:
                raise ValueError("stage1 checkpoint missing sk_model")
            X_val_tab = build_tabular_from_samples(val_s1, add_tstar_position_feature=add_tstar_position_feature)
            X_test_tab = build_tabular_from_samples(test_s1, add_tstar_position_feature=add_tstar_position_feature)
            if hasattr(clf, "predict_proba"):
                p_val_raw = clf.predict_proba(X_val_tab)[:, 1]
                p_test_raw = clf.predict_proba(X_test_tab)[:, 1]
            else:
                p_val_raw = np.asarray(clf.predict(X_val_tab), dtype=float)
                p_test_raw = np.asarray(clf.predict(X_test_tab), dtype=float)

        t_best, _ = fit_temperature_grid(y_val, p_val_raw)
        p_val_cal = apply_temperature(p_val_raw, t_best)
        p_test_cal = apply_temperature(p_test_raw, t_best)
        tau_policy, tau_val_f1, tau_val_prec, tau_val_rec = best_tau_by_target(
            y_val,
            p_val_cal,
            mode=tau_mode,
            target_precision=float(tau_target_precision),
            target_recall=float(tau_target_recall),
        )
        print(
            f"[seed {seed}] tau_policy={tau_policy:.6f} mode={tau_mode} "
            f"val_f1={tau_val_f1:.4f} val_prec={tau_val_prec:.4f} val_rec={tau_val_rec:.4f}"
        )

        if grouped_mode:
            model2 = HierarchicalCausalHazardTransformer(
                d_in=int(val_ds2[0][0].shape[-1]),
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=C.N_LAYERS,
                num_tstar_layers=stage2_tstar_layers,
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
                max_tstar_len=512,
                use_tstar_scalar_pos=bool(stage2_use_tstar_scalar_pos),
            ).to(device)
            model2.time_chunk_size = int(ckpt2.get("stage2_time_chunk_size", 64))
            model2.pmf_mode = str(ckpt2.get("stage2_pmf_mode", "hazard"))
            model2.gaussian_sigma = float(ckpt2.get("stage2_pmf_sigma", 5.0))
            model2.gaussian_mu_max = float(ckpt2.get("stage2_pmf_mu_max", 0.0))
        else:
            model2 = HazardTransformer(
                d_in=int(val_ds2[0][0].shape[-1]),
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=C.N_LAYERS,
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
            ).to(device)
        model2.load_state_dict(d2["state_dict"], strict=False)
        model2.eval()

        if grouped_mode:
            val_rows2 = collect_interval_preds_grouped(
                model2,
                val_loader2,
                source_groups=val_groups2,
                Tend=T,
                device=device,
                pi_method=getattr(C, "PI_METHOD", "shortest"),
                max_samples=100000000,
            )
            test_rows2 = collect_interval_preds_grouped(
                model2,
                test_loader2,
                source_groups=test_groups2,
                Tend=T,
                device=device,
                pi_method=getattr(C, "PI_METHOD", "shortest"),
                max_samples=100000000,
            )
        else:
            val_rows2 = collect_interval_preds(
                model2,
                val_loader2,
                source_samples=val_s2,
                Tend=T,
                device=device,
                pi_method=getattr(C, "PI_METHOD", "shortest"),
                max_samples=100000000,
            )
            test_rows2 = collect_interval_preds(
                model2,
                test_loader2,
                source_samples=test_s2,
                Tend=T,
                device=device,
                pi_method=getattr(C, "PI_METHOD", "shortest"),
                max_samples=100000000,
            )

        for threshold_spec in thresholds:
            threshold_policy = str(threshold_spec).lower() if str(threshold_spec).lower() == "tau" else f"{float(threshold_spec):.6g}"
            threshold = float(tau_policy) if threshold_policy == "tau" else float(threshold_spec)
            anchor = anchor_threshold
            if anchor is not None and str(anchor).lower() == "tau":
                anchor = float(tau_policy)
            for offset in offsets:
                all_rows.append(
                    compute_policy_metrics(
                        split_name="val",
                        split_seed=split_seed,
                        seed=seed,
                        threshold=threshold,
                        threshold_policy=threshold_policy,
                        offset=offset,
                        anchor_threshold=anchor,
                        anchor_probs=p_val_cal,
                        samples1=val_s1,
                        probs=p_val_cal,
                        source_samples2=val_s2,
                        stage2_rows=val_rows2,
                    )
                )
                all_rows.append(
                    compute_policy_metrics(
                        split_name="test",
                        split_seed=split_seed,
                        seed=seed,
                        threshold=threshold,
                        threshold_policy=threshold_policy,
                        offset=offset,
                        anchor_threshold=anchor,
                        anchor_probs=p_test_cal,
                        samples1=test_s1,
                        probs=p_test_cal,
                        source_samples2=test_s2,
                        stage2_rows=test_rows2,
                    )
                )

    df = pd.DataFrame(all_rows)
    if df.empty:
        raise RuntimeError("no sweep rows produced")
    out_dir = Path(out_root) / "gate_policy_sweep"
    out_dir.mkdir(parents=True, exist_ok=True)
    sweep_path = out_dir / f"gate_policy_sweep_run{run}_split{split_seed}.csv"
    df.to_csv(sweep_path, index=False)
    print("saved:", sweep_path)

    if select_metric not in df.columns:
        raise ValueError(f"--select_metric must be one of CSV columns; got {select_metric}")
    val_df = df[df["split"] == "val"].copy()
    policy_rank = select_policy(val_df, select_metric=select_metric)
    policy_path = out_dir / f"gate_policy_sweep_policy_rank_run{run}_split{split_seed}.csv"
    policy_rank.to_csv(policy_path, index=False)
    print("saved:", policy_path)

    best = policy_rank.iloc[0]
    best_threshold_policy = str(best["threshold_policy"])
    best_offset = int(best["offset"])
    selected_test = df[
        (df["split"] == "test")
        & (df["threshold_policy"].astype(str) == best_threshold_policy)
        & (df["offset"].astype(int) == best_offset)
    ].copy()
    selected_path = out_dir / f"gate_policy_sweep_selected_test_run{run}_split{split_seed}.csv"
    selected_test.to_csv(selected_path, index=False)
    print(f"[selected_policy] metric={select_metric} threshold_policy={best_threshold_policy} offset={best_offset}")
    print("saved:", selected_path)

    if wandb_run is not None:
        import wandb

        wandb_run.log({"sweep/all": wandb.Table(dataframe=df)})
        wandb_run.log({"sweep/policy_rank": wandb.Table(dataframe=policy_rank)})
        wandb_run.log({"sweep/selected_test": wandb.Table(dataframe=selected_test)})
        for col in ["IoU80", "Rec80", "Prec80", "EarlyRecall80", "stage1_gate_recall", "interval_hit_f1"]:
            if col in selected_test.columns:
                wandb_run.summary[f"selected_test/{col}_mean"] = float(selected_test[col].mean())
        wandb_run.summary["selected_policy/threshold_policy"] = str(best_threshold_policy)
        wandb_run.summary["selected_policy/threshold_mean"] = float(best["threshold"])
        wandb_run.summary["selected_policy/offset"] = int(best_offset)
        wandb_run.summary["sweep_csv_path"] = str(sweep_path)
        wandb_run.summary["policy_rank_csv_path"] = str(policy_path)
        wandb_run.summary["selected_test_csv_path"] = str(selected_path)
        wandb_run.save(str(sweep_path), policy="now")
        wandb_run.save(str(policy_path), policy="now")
        wandb_run.save(str(selected_path), policy="now")
    finish_wandb_run(wandb_run)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, required=True)
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--split_seed", type=int, default=C.SPLIT_SEED)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--thresholds", type=str, default="tau")
    p.add_argument("--offsets", type=str, default="0,7,14")
    p.add_argument("--anchor_threshold", type=str, default=None)
    p.add_argument("--tau_mode", type=str, default="f1", choices=["f1", "precision_target", "recall_target"])
    p.add_argument("--tau_target_precision", type=float, default=0.6)
    p.add_argument("--tau_target_recall", type=float, default=0.6)
    p.add_argument("--select_metric", type=str, default="interval_hit_f1")
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default=None)
    args = p.parse_args()
    main(
        pest=args.pest,
        run=args.run,
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        out_root=args.out_root,
        split_seed=args.split_seed,
        seeds=args.seeds,
        thresholds=parse_threshold_specs(args.thresholds),
        offsets=parse_int_list(args.offsets),
        anchor_threshold=None if args.anchor_threshold is None else (
            "tau" if str(args.anchor_threshold).lower() == "tau" else float(args.anchor_threshold)
        ),
        tau_mode=args.tau_mode,
        tau_target_precision=args.tau_target_precision,
        tau_target_recall=args.tau_target_recall,
        select_metric=args.select_metric,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
    )
