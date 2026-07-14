"""
Phase R — IoU-centric Oracle multi-offset analysis + σ tuning sweep.

Center metric: IoU(PI, [L, R])  (PI = [mu - 1.96σ, mu + 1.96σ])

Two analyses, both on the test interval cohort (n_total = 575):

(1) Oracle multi-offset (per-sample best offset, σ fixed)
    For each sample s and each offset o ∈ {60, 90, 105, 120}:
        mu_so   = stage2 mu at alert_tstar_s + o   (if matched, else NaN)
        IoU_so  = overlap_metrics(round(mu_so − 1.96σ), round(mu_so + 1.96σ),
                                  L_s, R_s)
    Oracle IoU_s = max_o IoU_so   (sample's best-attainable IoU)
    Report:
        - Oracle IoU_mean_matched : mean over samples with ≥1 matched offset
        - Oracle IoU_mean_overall : same numerator / n_total (treat unmatched as 0)
        - Histogram of which offset gives the per-sample max IoU

(2) σ tuning sweep, fixed offset
    For each offset ∈ {105, 120} and σ ∈ {2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0}:
        IoU_cell = mean over matched samples of overlap(PI_σ, [L, R])
        Also IoU_overall = IoU_cell × (n_match / n_total)
    Find best σ per (model, offset).

Output:
    | Model | Fixed best IoU (CSV) | Oracle IoU_matched | Oracle IoU_overall | best (offset, σ) |

No retraining; reuses 5 model ckpts.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import (
    GroupedIntervalEventDataset,
    build_stage2_nowcast_samples,
    compute_norm_stats,
    group_stage2_samples_by_site_year,
    split_samples,
)
from rice.src.model import HierarchicalCausalHazardTransformer
from rice.src.train_eval import overlap_metrics
from rice.scripts.common import collate_grouped_stage2, make_loader
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import build_nowcast_samples, build_tabular_from_samples
from rice.scripts.run_calibrate_event import apply_temperature, fit_temperature_grid

# Dispatch alert dependencies (read-only reuse; same first_crossing_k policy
# and same history-availability mask as phase_t_group_tau_hybrid).
from rice.scripts.phase_t_group_tau_hybrid import first_crossing_k
from rice.scripts.phase_t_history_subcohort_compare import make_history_mask


def best_tau_by_f1(y, p):
    taus = np.linspace(0.05, 0.95, 19)
    best_tau, best_f1 = 0.5, -1.0
    for t in taus:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        if (2 * tp + fp + fn) == 0:
            continue
        f1 = (2 * tp) / (2 * tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(t)
    return best_tau


def build_stage1_alert_map(stage1_ckpt_path, run, args, eval_split: str = "test"):
    ckpt = torch.load(stage1_ckpt_path, map_location="cpu")
    add_tpos = bool(ckpt.get("add_tstar_position_feature", False))
    nc_window = int(ckpt.get("nowcast_window", 28))
    nc_stride = int(ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("nowcast_event_time_proxy", "r"))

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples = build_samples_for_run(run, get_feature_cols)
    _, val_seas, test_seas = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )
    val_s = build_nowcast_samples(val_seas, window=nc_window, stride=nc_stride,
                                  only_pre_event=nc_only_pre, event_time_proxy=nc_proxy)
    test_s = build_nowcast_samples(test_seas, window=nc_window, stride=nc_stride,
                                   only_pre_event=nc_only_pre, event_time_proxy=nc_proxy)
    y_val = np.asarray([int(s["y_event"]) for s in val_s])
    X_val = build_tabular_from_samples(val_s, add_tstar_position_feature=add_tpos)
    X_test = build_tabular_from_samples(test_s, add_tstar_position_feature=add_tpos)
    clf = ckpt["trained_states"][0]["sk_model"]
    p_val_raw = clf.predict_proba(X_val)[:, 1]
    p_test_raw = clf.predict_proba(X_test)[:, 1]
    t_best, _ = fit_temperature_grid(y_val, p_val_raw)
    p_val_cal = apply_temperature(p_val_raw, t_best)
    p_test_cal = apply_temperature(p_test_raw, t_best)
    tau = best_tau_by_f1(y_val, p_val_cal)

    # Choose which split's samples to drop alerts on. Tau and temperature
    # scaling always come from the val split (out-of-sample for year-split).
    if str(eval_split) == "val":
        target_nowcast, target_probs, target_seas, denom_tag = (
            val_s, p_val_cal, val_seas, "n_interval_val"
        )
    else:
        target_nowcast, target_probs, target_seas, denom_tag = (
            test_s, p_test_cal, test_seas, "n_interval_test"
        )
    alert = {}
    for s, p_cal in zip(target_nowcast, target_probs):
        if p_cal < tau:
            continue
        key = (str(s["site_id"]), int(s["year"]))
        prev = alert.get(key)
        if prev is None or int(s["tstar"]) < prev:
            alert[key] = int(s["tstar"])
    n_interval_test = sum(1 for s in target_seas if str(s["censor_type"]) != "right")
    print(f"[stage1] tau={tau:.3f}  alert={len(alert)}  n_interval_test={n_interval_test}")
    return alert, n_interval_test


def _calibrated_probs_per_sy(ckpt_path, run, args,
                              include_splits: tuple[str, ...] = ("test",)):
    """Forward a Stage-1 ckpt and return per-(site, year) calibrated probability
    series for any subset of {train, val, test}.

    Temperature scaling is always fit on the val split (uncalibrated). The
    same scaling is applied to every requested split.

    Honors ckpt meta flags (site_history_added / phenology_added /
    derived_weather_added) by appending the matching channels before the
    nowcast split, mirroring phase_t_lead_aware_eval.build_probs.

    NOTE: Probabilities on the train split are *in-sample* (the model was
    trained on these site-years). The caller is responsible for surfacing
    this as a leakage warning when train probs are used for downstream
    feature construction.

    Returns
    -------
    per_sy : dict[(site, year), {"ts": np.ndarray, "ps": np.ndarray}]
        Concatenation of requested splits; rows sorted by tstar.
    doy_start : int
    n_interval_test : int
        Always counted over the test split (denominator for downstream).
    split_by_sy : dict[(site, year), str]
        'train' / 'val' / 'test' label per site-year (last writer wins on
        the unlikely case of duplicate sy across splits, which year-mode
        split_samples does not produce).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    default_doy_start = getattr(C, "DOY_START", 60)
    default_doy_end = getattr(C, "DOY_END", 300)
    ckpt_doy_start = ckpt.get("doy_start", None)
    ckpt_doy_end = ckpt.get("doy_end", None)
    print(f"  [calib] ckpt={Path(ckpt_path).name}  "
          f"ckpt.doy_start={ckpt_doy_start!r}  ckpt.doy_end={ckpt_doy_end!r}  "
          f"default=({default_doy_start},{default_doy_end})")
    C.DOY_START = int(ckpt_doy_start if ckpt_doy_start is not None else default_doy_start)
    C.DOY_END = int(ckpt_doy_end if ckpt_doy_end is not None else default_doy_end)
    add_tpos = bool(ckpt.get("add_tstar_position_feature", False))
    nc_window = int(ckpt.get("nowcast_window", 28))
    nc_stride = int(ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("nowcast_event_time_proxy", "r"))
    doy_start = int(C.DOY_START)

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples = build_samples_for_run(run, get_feature_cols)

    if bool(ckpt.get("site_history_added", False)):
        from rice.scripts.site_history_utils import (
            compute_site_history, append_history_to_samples,
        )
        history = compute_site_history(
            samples, doy_start=doy_start,
            policy=str(ckpt.get("site_history_policy", "rolling")),
            train_year_max=int(ckpt.get("history_train_year_max", 2021)),
        )
        append_history_to_samples(samples, history, doy_start=doy_start)
    if bool(ckpt.get("phenology_added", False)):
        from rice.scripts.phenology_utils import (
            load_pheno_map, append_pheno_to_samples,
        )
        pheno_map = load_pheno_map()
        append_pheno_to_samples(samples, pheno_map, doy_start=doy_start)
    if bool(ckpt.get("derived_weather_added", False)):
        from rice.scripts.derived_weather_utils import (
            append_derived_weather_to_samples,
        )
        append_derived_weather_to_samples(samples)
    # Stage-1 neighbor (N/DN ckpts): reconstruct the 6 neighbor channels the model
    # was trained with, appended LAST (same order as phase_t_lead_aware_train).
    # No-op for ckpts trained without neighbor.
    if bool(ckpt.get("neighbor_history_added", False)):
        from rice.scripts.neighbor_history_utils import (
            load_long_events, build_neighbor_index, append_neighbor_to_samples,
        )
        _nb_decay = float(ckpt.get("neighbor_decay_km", 20.0))
        _nb_ev, _nb_co, _ = load_long_events(
            C.PATH_OBS, label_col=getattr(C, "LABEL_COL", "label_event"),
            year_min=getattr(C, "YEAR_MIN", None), year_max=getattr(C, "YEAR_MAX", None),
        )
        append_neighbor_to_samples(samples, build_neighbor_index(_nb_ev, _nb_co),
                                   doy_start=doy_start, decay_km=_nb_decay)

    train_seas, val_seas, test_seas = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )

    nc_kw = dict(window=nc_window, stride=nc_stride,
                 only_pre_event=nc_only_pre, event_time_proxy=nc_proxy)

    clf = ckpt["trained_states"][0]["sk_model"]

    # Temperature scaling on val (out-of-sample for year-split).
    val_s = build_nowcast_samples(val_seas, **nc_kw)
    y_val = np.asarray([int(s["y_event"]) for s in val_s])
    X_val = build_tabular_from_samples(val_s, add_tstar_position_feature=add_tpos)
    p_val_raw = clf.predict_proba(X_val)[:, 1]
    t_best, _ = fit_temperature_grid(y_val, p_val_raw)

    per_sy: dict = {}
    split_by_sy: dict = {}

    def _accumulate(split_name: str, seas, nowcast=None):
        if nowcast is None:
            nowcast = build_nowcast_samples(seas, **nc_kw)
        X = build_tabular_from_samples(nowcast, add_tstar_position_feature=add_tpos)
        p_raw = clf.predict_proba(X)[:, 1]
        p_cal = apply_temperature(p_raw, t_best)
        for s, p in zip(nowcast, p_cal):
            sy = (str(s["site_id"]), int(s["year"]))
            d = per_sy.setdefault(sy, {"ts": [], "ps": []})
            d["ts"].append(int(s["tstar"]))
            d["ps"].append(float(p))
            split_by_sy[sy] = split_name

    if "train" in include_splits:
        _accumulate("train", train_seas)
    if "val" in include_splits:
        _accumulate("val", val_seas, nowcast=val_s)
    if "test" in include_splits:
        _accumulate("test", test_seas)

    for sy, d in per_sy.items():
        order = np.argsort(np.asarray(d["ts"], dtype=int))
        d["ts"] = np.asarray(d["ts"], dtype=int)[order]
        d["ps"] = np.asarray(d["ps"], dtype=float)[order]

    n_interval_test = sum(1 for s in test_seas if str(s["censor_type"]) != "right")
    print(f"  [calib] ckpt={Path(ckpt_path).name}  T*={t_best:.3f}  "
          f"splits={list(include_splits)}  per_sy={len(per_sy)}  "
          f"n_interval_test={n_interval_test}")
    if "train" in include_splits:
        print(f"  [calib] WARNING: train split probs are IN-SAMPLE "
              f"(model trained on these site-years). Downstream features may "
              f"overstate Stage-1 confidence on training years.")
    return per_sy, doy_start, n_interval_test, split_by_sy


def build_dispatch_alert_map(a_ckpt_path, d_ckpt_path, summary_json_path,
                              target_label, run, args,
                              include_splits: tuple[str, ...] = ("test",),
                              gate_method: str = "dispatch_group_tau"):
    """Build dispatch_group_tau alert map for the given recall target.

    Reads (k, tau_no, tau_with) from the year-split dispatch summary JSON
    selection at ``target_label`` (e.g. 'R>=0.88'), then applies the same
    first_crossing_k policy as phase_t_group_tau_hybrid:

        with_history sy  -> D probs + tau_with, first k-consecutive crossing
        no_history  sy   -> A probs + tau_no,   first k-consecutive crossing

    alert_t is the tstar of the k-th row that completes the window.

    Returns
    -------
    alert_map : dict[(site, year), int]
        Per site-year alert_t (relative tstar; absolute DOY = alert_t + doy_start - 1).
    n_total : int
        Number of interval-censored test site-years (denominator for downstream).
    ctx : dict
        Auxiliary context for downstream feature computation:
        per_sy_A, per_sy_D, miss_map, tau_no, tau_with, k, doy_start.
    """
    import json
    with open(summary_json_path) as fh:
        sel = json.load(fh).get("selections", {})
    if target_label not in sel:
        raise SystemExit(
            f"[dispatch] target {target_label!r} not in {summary_json_path}; "
            f"available={list(sel)}"
        )
    # gate_method controls how alert_t is derived. The summary JSON stores
    # selection entries for all three policies at each recall target.
    JSON_KEY = {
        "A_baseline": "A_raw_global",
        "D_history":  "D_raw_global",
        "dispatch_group_tau": "dispatch_group_tau",
    }
    if gate_method not in JSON_KEY:
        raise SystemExit(f"[dispatch] unknown gate_method={gate_method!r}; "
                          f"expected one of {list(JSON_KEY)}")
    sel_key = JSON_KEY[gate_method]
    dsel = sel[target_label].get(sel_key)
    if dsel is None:
        raise SystemExit(
            f"[dispatch] selection {sel_key!r} missing in selections[{target_label!r}]"
        )
    k = int(dsel["k"])
    if gate_method == "dispatch_group_tau":
        tau_no = float(dsel["tau_no"])
        tau_with = float(dsel["tau_with"])
        tau_single = None
    else:
        # A_raw_global / D_raw_global store a single 'tau' (single-method gate).
        tau_single = float(dsel["tau"])
        tau_no = tau_with = tau_single
    print(f"[dispatch] target={target_label}  gate_method={gate_method}  k={k}  "
          f"tau_single={tau_single}  tau_no={tau_no}  tau_with={tau_with}")

    print(f"[dispatch] building A (baseline) calibrated probs ... "
          f"include_splits={list(include_splits)}")
    per_sy_A, doy_start_A, n_total, split_by_sy_A = _calibrated_probs_per_sy(
        a_ckpt_path, run, args, include_splits=include_splits,
    )
    print(f"[dispatch] building D (history) calibrated probs ... "
          f"include_splits={list(include_splits)}")
    per_sy_D, doy_start_D, _, split_by_sy_D = _calibrated_probs_per_sy(
        d_ckpt_path, run, args, include_splits=include_splits,
    )
    if doy_start_A != doy_start_D:
        raise SystemExit(
            f"[dispatch] doy_start mismatch A={doy_start_A} D={doy_start_D}"
        )
    doy_start = int(doy_start_A)

    d_meta = torch.load(d_ckpt_path, map_location="cpu")
    pol = str(d_meta.get("site_history_policy", "rolling"))
    tyrmax = int(d_meta.get("history_train_year_max", 2021))
    print(f"[dispatch] history meta: policy={pol}  train_year_max={tyrmax}")
    miss_map = make_history_mask(args.pest, run, doy_start, pol, tyrmax)

    alert_map: dict = {}
    n_with = n_no = 0
    n_skipped = 0
    n_by_split: dict = {"train": 0, "val": 0, "test": 0}
    for sy in set(per_sy_A) | set(per_sy_D):
        A = per_sy_A.get(sy)
        D = per_sy_D.get(sy)
        if A is None or D is None:
            n_skipped += 1
            continue
        with_h = miss_map.get(sy, {"prev_year_L_miss": 1})["prev_year_L_miss"] == 0
        if gate_method == "A_baseline":
            at = first_crossing_k(A["ts"], A["ps"], tau_single, k)
        elif gate_method == "D_history":
            at = first_crossing_k(D["ts"], D["ps"], tau_single, k)
        else:  # dispatch_group_tau
            if with_h:
                at = first_crossing_k(D["ts"], D["ps"], tau_with, k)
            else:
                at = first_crossing_k(A["ts"], A["ps"], tau_no, k)
        if at is None:
            continue
        alert_map[sy] = int(at)
        if with_h:
            n_with += 1
        else:
            n_no += 1
        split_label = split_by_sy_A.get(sy) or split_by_sy_D.get(sy) or "unknown"
        if split_label in n_by_split:
            n_by_split[split_label] += 1

    print(f"[dispatch] alerts: total={len(alert_map)}  "
          f"with_history={n_with}  no_history={n_no}  "
          f"sy_missing_one_side={n_skipped}  n_interval_test={n_total}")
    if "train" in include_splits or "val" in include_splits:
        print(f"[dispatch] alerts by split: train={n_by_split['train']}  "
              f"val={n_by_split['val']}  test={n_by_split['test']}")
        if n_by_split["train"] > 0:
            print(f"[dispatch] WARNING: {n_by_split['train']} train alerts are "
                  f"derived from IN-SAMPLE A/D probabilities. Stage-1 confidence "
                  f"features for those site-years will be optimistically biased; "
                  f"OOF or rolling re-fit recommended for the final pipeline.")

    # split_by_sy_A and split_by_sy_D agree on label per sy because the split is
    # derived from year-mode split_samples (same boundaries on the same samples
    # list); we merge to give callers a single mapping.
    split_by_sy = dict(split_by_sy_A)
    split_by_sy.update(split_by_sy_D)

    ctx = {
        "per_sy_A": per_sy_A,
        "per_sy_D": per_sy_D,
        "miss_map": miss_map,
        "tau_no": tau_no,
        "tau_with": tau_with,
        "tau_single": tau_single,
        "k": k,
        "doy_start": doy_start,
        "split_by_sy": split_by_sy,
        "gate_method": gate_method,
    }
    return alert_map, n_total, ctx


def _dispatch_features_for_sy(sy, alert_t, ctx):
    """Compute the 14 dispatch-specific feature columns + alert_tstar(DOY).

    All score-window features use the dispatch_score series (D series if
    with_history else A series). recent_*d windows use DOY (not row counts):
    [alert_DOY - W, alert_DOY] inclusive on whatever rows actually exist.
    score_above_tau_streak counts consecutive dispatch_score >= tau rows ending
    at alert_t (must be >= k by construction).
    score_rolling_slope_14d is the OLS slope of dispatch_score vs DOY over the
    14d window; NaN if fewer than 2 rows are available.
    p_mean_so_far_at_alert averages dispatch_score over the rows whose tstar
    <= alert_t (i.e., season start through alert_t, only available rows).
    """
    per_sy_A = ctx["per_sy_A"]
    per_sy_D = ctx["per_sy_D"]
    miss_map = ctx["miss_map"]
    tau_no = ctx["tau_no"]
    tau_with = ctx["tau_with"]
    doy_start = ctx["doy_start"]
    gate_method = str(ctx.get("gate_method", "dispatch_group_tau"))
    tau_single = ctx.get("tau_single")

    A = per_sy_A.get(sy)
    D = per_sy_D.get(sy)
    with_h = miss_map.get(sy, {"prev_year_L_miss": 1})["prev_year_L_miss"] == 0

    def _at(series, t):
        if series is None:
            return float("nan")
        idx = np.where(series["ts"] == int(t))[0]
        return float(series["ps"][int(idx[0])]) if idx.size else float("nan")

    a_score = _at(A, alert_t)
    d_score = _at(D, alert_t)

    # gate_method picks which series the gate fired on. Stage 2 input features
    # are computed on THAT series so dispatch_score / streak / slope / etc. are
    # consistent with the alert. with_history is still reported as the true
    # history-availability flag for downstream analysis.
    if gate_method == "A_baseline":
        d_ts, d_ps, tau_used = A["ts"], A["ps"], float(tau_single)
        disp_score = a_score
        branch = "A"
    elif gate_method == "D_history":
        d_ts, d_ps, tau_used = D["ts"], D["ps"], float(tau_single)
        disp_score = d_score
        branch = "D"
    else:  # dispatch_group_tau
        if with_h:
            d_ts, d_ps, tau_used = D["ts"], D["ps"], tau_with
            disp_score = d_score
            branch = "D"
        else:
            d_ts, d_ps, tau_used = A["ts"], A["ps"], tau_no
            disp_score = a_score
            branch = "A"

    margin = (d_score - a_score) if (np.isfinite(d_score) and np.isfinite(a_score)) else float("nan")
    sot_margin = (disp_score - tau_used) if np.isfinite(disp_score) else float("nan")

    alert_doy = int(alert_t) + doy_start - 1
    doys = d_ts.astype(int) + doy_start - 1
    mask_14 = (doys >= alert_doy - 14) & (doys <= alert_doy)
    mask_28 = (doys >= alert_doy - 28) & (doys <= alert_doy)
    mean_14 = float(d_ps[mask_14].mean()) if mask_14.any() else float("nan")
    mean_28 = float(d_ps[mask_28].mean()) if mask_28.any() else float("nan")

    streak = 0
    idx_at = np.where(d_ts == int(alert_t))[0]
    if idx_at.size:
        i = int(idx_at[0])
        while i >= 0 and d_ps[i] >= tau_used:
            streak += 1
            i -= 1

    if mask_14.sum() >= 2:
        x = doys[mask_14].astype(float)
        y = d_ps[mask_14].astype(float)
        slope_14 = float(np.polyfit(x, y, 1)[0])
    else:
        slope_14 = float("nan")

    mask_cum = d_ts <= int(alert_t)
    p_mean_so_far = float(d_ps[mask_cum].mean()) if mask_cum.any() else float("nan")

    return {
        "alert_tstar": int(alert_doy),
        "dispatch_branch": branch,
        "with_history": int(with_h),
        "A_score_at_alert": a_score,
        "D_score_at_alert": d_score,
        "score_margin": margin,
        "dispatch_score_at_alert": disp_score,
        "dispatch_tau_used": float(tau_used),
        "score_over_tau_margin": sot_margin,
        "recent_14d_mean_score": mean_14,
        "recent_28d_mean_score": mean_28,
        "score_above_tau_streak": int(streak),
        "score_rolling_slope_14d": slope_14,
        "p_mean_so_far_at_alert": p_mean_so_far,
    }


def build_stage2_row_map(stage2_ckpt_path, run, args, device,
                          bypass_phen_head: bool = False, eval_split: str = "test"):
    """
    Forward Stage 2 and build per-(site, year, tstar) mu lookup.

    bypass_phen_head=True forces the model to ignore the phenology bias head at
    inference time (mu = mu_temporal only). Same weights as the original ckpt,
    different inference path → exposes how much of the model's mu variation
    actually comes from phen_bias vs from the encoder.
    """
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu")
    default_doy_start = getattr(C, "DOY_START", 60)
    default_doy_end = getattr(C, "DOY_END", 300)
    ckpt_doy_start = ckpt.get("doy_start", None)
    ckpt_doy_end = ckpt.get("doy_end", None)
    print(f"  [stage2] ckpt={Path(stage2_ckpt_path).name}  "
          f"ckpt.doy_start={ckpt_doy_start!r}  ckpt.doy_end={ckpt_doy_end!r}  "
          f"default=({default_doy_start},{default_doy_end})")
    C.DOY_START = int(ckpt_doy_start if ckpt_doy_start is not None else default_doy_start)
    C.DOY_END = int(ckpt_doy_end if ckpt_doy_end is not None else default_doy_end)
    doy_start = int(C.DOY_START)

    phen_bias_head_ckpt = bool(int(ckpt.get("stage2_phenology_bias_head", 0)))
    phen_bias_head = phen_bias_head_ckpt and (not bypass_phen_head)
    phen_hidden = int(ckpt.get("stage2_phenology_hidden", 8))
    pheno_ext_cols = (["best_suitability", "best_months", "offset_days", "window_idx"]
                      if phen_bias_head else None)
    print(f"  [phen_bias_head] ckpt_has={phen_bias_head_ckpt}  effective={phen_bias_head}  "
          f"bypass={bypass_phen_head}  hidden={phen_hidden}")

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples2 = build_samples_for_run(run, get_feature_cols, pheno_ext_cols=pheno_ext_cols)

    # If the Stage-2 ckpt was trained with dispatch confidence features appended,
    # mirror the exact same channel additions at evaluation time so X dim matches.
    if bool(ckpt.get("stage2_dispatch_features_added", False)):
        from rice.scripts.stage1_confidence_utils import (
            load_dispatch_feature_table, append_dispatch_confidence_to_samples,
            train_mean_features_from_table,
        )
        d_csv = ckpt.get("stage2_dispatch_feature_csv")
        # Optional override (multi-year aux sample_grid use-case).
        _csv_override = getattr(args, "stage2_dispatch_feature_csv_override", "") or ""
        if _csv_override:
            print(f"  [dispatch_features] OVERRIDE csv: "
                  f"ckpt={d_csv}  ->  override={_csv_override}")
            d_csv = _csv_override
        d_mode = str(ckpt.get("stage2_dispatch_feature_mode", "causal"))
        d_miss = float(ckpt.get("stage2_dispatch_feature_missing_value", 0.0))
        if not d_csv:
            raise SystemExit(
                "[abort] ckpt has stage2_dispatch_features_added=True but no "
                "stage2_dispatch_feature_csv path stored."
            )
        # Phase A.3: optional ablation override from args (eval-only).
        # Acceptable values: 'ablate_missing' or 'ablate_train_mean'. Any other
        # value falls back to ckpt-stored mode (production behavior).
        ablate_mode = getattr(args, "dispatch_ablation_mode", "") or ""
        ablate_csv = getattr(args, "dispatch_ablation_csv", "") or d_csv
        train_mean = None
        if ablate_mode in ("ablate_missing", "ablate_train_mean"):
            run_mode = ablate_mode
            if ablate_mode == "ablate_train_mean":
                train_mean = train_mean_features_from_table(ablate_csv)
                print(f"  [dispatch_features] ABLATION mode={ablate_mode}  "
                      f"train_mean_features={train_mean.tolist()}")
            else:
                print(f"  [dispatch_features] ABLATION mode={ablate_mode}  "
                      f"(rows pinned to missing-fill = inside-distribution)")
        else:
            run_mode = d_mode
        conf_map = load_dispatch_feature_table(ablate_csv if run_mode == "ablate_train_mean" else d_csv)
        stats = append_dispatch_confidence_to_samples(
            samples2, conf_map, doy_start=doy_start, mode=run_mode,
            missing_value=d_miss,
            ablate_train_mean_features=train_mean,
        )
        print(f"  [dispatch_features] eval-side append (mode={stats['mode']}): "
              f"n_with={stats['n_with_alert']}  n_no={stats['n_no_alert']}  "
              f"+{stats['added_channels']} channels  csv={d_csv}")

    train_s2_base, val_s2_base, test_s2_base = split_samples(
        samples2, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )
    nc_window = int(ckpt.get("stage2_nowcast_window", 28))
    nc_stride = int(ckpt.get("stage2_nowcast_stride", 1))
    nc_tstart = ckpt.get("stage2_nowcast_tstar_start", None)
    nc_only_pre = bool(int(ckpt.get("stage2_nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("stage2_nowcast_event_time_proxy", "r"))
    nc_req = bool(int(ckpt.get("stage2_nowcast_require_tstar_before_L", 0)))
    # eval_split selects which cohort's nowcast samples we forward through
    # Stage 2 for the row_map dump. Validation-side row_map is needed when
    # the caller wants to compute val-IoU and pick offsets on val (the only
    # honest way to avoid test-side offset cherry-picking).
    target_s2_base = val_s2_base if str(eval_split) == "val" else test_s2_base
    print(f"  [stage2] eval_split={eval_split}  target_sy={len(target_s2_base)}")
    test_s2 = build_stage2_nowcast_samples(
        target_s2_base, window=nc_window, stride=nc_stride, tstar_start=nc_tstart,
        only_pre_event=nc_only_pre, event_time_proxy=nc_proxy, require_tstar_before_L=nc_req,
    )

    x_mean, x_std = compute_norm_stats(train_s2_base)
    # Phase B fix: mirror run_train ONLY for ckpts that were trained with this
    # patch (stage2_dispatch_channels_raw=True). Pre-patch ckpts learned with
    # standardized dispatch channels and must keep that at eval to stay in
    # distribution.
    if bool(ckpt.get("stage2_dispatch_channels_raw", False)):
        from rice.scripts.stage1_confidence_utils import DISPATCH_TOTAL_CHANNELS
        n_disp = int(DISPATCH_TOTAL_CHANNELS)
        d_total = int(x_mean.shape[0])
        disp_start = d_total - n_disp
        if disp_start >= 0:
            x_mean[disp_start:] = 0.0
            x_std[disp_start:] = 1.0
            print(f"  [norm_stats] dispatch channels [{disp_start}:{d_total}] "
                  f"forced to RAW (matches training fix flag)")
        else:
            print(f"  [norm_stats] WARNING: cannot raw-pad dispatch channels "
                  f"(d_total={d_total} < n_disp={n_disp})")
    elif bool(ckpt.get("stage2_dispatch_features_added", False)):
        print(f"  [norm_stats] dispatch channels kept STANDARDIZED "
              f"(ckpt pre-dates raw-channel fix; honoring training distribution)")
    test_groups = group_stage2_samples_by_site_year(test_s2)
    ds = GroupedIntervalEventDataset(test_groups, x_mean, x_std)
    loader = make_loader(ds, 1, shuffle=False, collate_fn=collate_grouped_stage2)

    d_model = int(ckpt.get("d_model", C.D_MODEL))
    n_head = int(ckpt.get("n_head", C.N_HEAD))
    n_layers = int(ckpt.get("n_layers", C.N_LAYERS))
    model = HierarchicalCausalHazardTransformer(
        d_in=int(test_s2[0]["X"].shape[1]),
        d_model=d_model, nhead=n_head, num_layers=n_layers,
        num_tstar_layers=int(ckpt.get("stage2_tstar_layers", 1)),
        dropout=C.DROPOUT, max_len=C.MAX_LEN, max_tstar_len=512,
        use_tstar_scalar_pos=bool(int(ckpt.get("stage2_use_tstar_scalar_pos", 0))),
        phenology_bias_head=phen_bias_head, phenology_dim=4, phenology_hidden=phen_hidden,
    ).to(device)
    model.time_chunk_size = int(ckpt.get("stage2_time_chunk_size", 64))
    model.conditional_survival = bool(int(ckpt.get("stage2_conditional_survival", 0)))
    model.pmf_mode = "gaussian"
    model.gaussian_sigma = float(ckpt.get("stage2_pmf_sigma", 5.0))
    model.gaussian_mu_max = float(ckpt.get("stage2_pmf_mu_max", 0.0))
    # Phase B: lead_from_alert mu head — propagate ckpt meta to model attrs so
    # forward uses the same mu construction at eval. Backward compatible: when
    # ckpt has no meta or mu_mode='absolute', no behavior change.
    model.mu_mode = str(ckpt.get("stage2_pmf_mu_mode", "absolute"))
    model.lead_min = float(ckpt.get("stage2_pmf_lead_min", 7.0))
    model.lead_max = float(ckpt.get("stage2_pmf_lead_max", 75.0))
    # residual_clim mu head: clim_mid is stored in DOY; convert to 1-based
    # season-index coords (same convention as the model.forward branch).
    _clim_mid_doy = float(ckpt.get("stage2_pmf_clim_mid", 0.0))
    model.clim_mid_rel = _clim_mid_doy - float(doy_start) + 1.0 if _clim_mid_doy > 0.0 else 0.0
    model.delta_max = float(ckpt.get("stage2_pmf_delta_max", 60.0))
    model.alert_tstar_feat_idx = int(ckpt.get("stage2_pmf_alert_tstar_feat_idx", -1))
    model.doy_start = int(doy_start)
    model.lead_strict_alert_check = True
    model.lead_debug_once_pending = (model.mu_mode in ("lead_from_alert", "residual_clim"))
    if model.mu_mode == "lead_from_alert":
        print(f"  [mu_mode] lead_from_alert  lead_min={model.lead_min}  "
              f"lead_max={model.lead_max}  alert_tstar_feat_idx={model.alert_tstar_feat_idx}  "
              f"doy_start={model.doy_start}")
    elif model.mu_mode == "residual_clim":
        print(f"  [mu_mode] residual_clim  clim_mid_doy={_clim_mid_doy:.2f}  "
              f"clim_mid_rel={model.clim_mid_rel:.2f}  delta_max={model.delta_max:.1f}  "
              f"doy_start={model.doy_start}")
    model.load_state_dict(ckpt["trained_states"][0]["state_dict"], strict=False)
    model.eval()

    row_map = {}
    gi = 0
    is_lead_mode = (str(getattr(model, "mu_mode", "absolute")) == "lead_from_alert")
    n_skipped_pre_alert = 0
    with torch.no_grad():
        for _batch in loader:
            if len(_batch) == 7:
                X, L, R, ctype, tstar, valid_mask, pheno = _batch
            else:
                X, L, R, ctype, tstar, valid_mask = _batch
                pheno = None
            X = X.to(device); tstar_t = tstar.to(device); v_t = valid_mask.to(device)
            if pheno is not None:
                pheno = pheno.to(device)
            _ = model(X, tstar=tstar_t, valid_mask=v_t, pheno=pheno)
            mu_BK = getattr(model, "_last_mu_BK")
            mu_np = mu_BK.detach().cpu().numpy()
            v_np = valid_mask.cpu().numpy().astype(bool)
            L_np = L.cpu().numpy().astype(int)
            R_np = R.cpu().numpy().astype(int)
            c_np = ctype.cpu().numpy().astype(int)
            ts_np = tstar.cpu().numpy().astype(int)
            # Phase B: in lead_from_alert mode, only export post-alert cells.
            # pre-alert cells were forward-passed for context but their mu is
            # a dummy value that must not enter row_map.
            if is_lead_mode:
                lead_mask_BK = getattr(model, "_last_lead_loss_mask", None)
                lead_np = (lead_mask_BK.cpu().numpy().astype(bool)
                           if lead_mask_BK is not None
                           else np.ones_like(v_np))
            else:
                lead_np = None
            B, K = mu_np.shape
            for bi in range(B):
                g = test_groups[gi + bi]
                for ki in range(K):
                    if not v_np[bi, ki]:
                        continue
                    if lead_np is not None and not lead_np[bi, ki]:
                        n_skipped_pre_alert += 1
                        continue
                    key = (str(g["site_id"]), int(g["year"]), int(ts_np[bi, ki]))
                    row_map[key] = {
                        "mu": float(mu_np[bi, ki]),
                        "true_L": int(L_np[bi, ki]),
                        "true_R": int(R_np[bi, ki]),
                        "ctype": int(c_np[bi, ki]),
                    }
            gi += B
    if is_lead_mode:
        print(f"  [lead_from_alert] row_map cells: kept={len(row_map)}  "
              f"skipped_pre_alert={n_skipped_pre_alert}  "
              f"(only post-alert cells are exported for lookup)")
    return row_map, doy_start


def iou_for_mu(mu_abs, L_abs, R_abs, sigma):
    HW = 1.96 * float(sigma)
    pL = int(round(mu_abs - HW))
    pR = int(round(mu_abs + HW))
    iou, _, _ = overlap_metrics(pL, pR, int(L_abs), int(R_abs))
    return iou


def collect_per_sample_mu(alert_map, row_map, doy_start, offsets):
    """
    For each alerted interval sample, look up mu at alert_tstar + offset for
    every offset; return dict keyed by (site, year) with mu per offset, plus L,R.
    """
    out = {}
    for (site, year), alert_t in alert_map.items():
        per_off = {}
        L_abs = None
        R_abs = None
        for o in offsets:
            target = int(alert_t) + int(o)
            info = row_map.get((str(site), int(year), int(target)))
            if info is None or info["ctype"] != 0:
                continue
            mu_abs = info["mu"] + doy_start - 1
            per_off[int(o)] = float(mu_abs)
            L_abs = int(info["true_L"]) + doy_start - 1
            R_abs = int(info["true_R"]) + doy_start - 1
        if per_off and L_abs is not None:
            out[(str(site), int(year))] = {
                "per_off_mu": per_off, "L": L_abs, "R": R_abs, "alert_t": int(alert_t),
            }
    return out


def oracle_iou(samples_dict, offsets, sigma):
    per_sample_best_iou = []
    per_sample_best_off = []
    for key, info in samples_dict.items():
        best_iou = -1.0
        best_off = None
        for o in offsets:
            mu = info["per_off_mu"].get(int(o))
            if mu is None:
                continue
            iou = iou_for_mu(mu, info["L"], info["R"], sigma)
            if iou > best_iou:
                best_iou = iou
                best_off = int(o)
        if best_off is None:
            continue
        per_sample_best_iou.append(best_iou)
        per_sample_best_off.append(best_off)
    return np.asarray(per_sample_best_iou), np.asarray(per_sample_best_off)


def fixed_iou_at(samples_dict, offset, sigma):
    """Mean IoU and overall (n_total denom) IoU at a fixed (offset, σ)."""
    ious = []
    for key, info in samples_dict.items():
        mu = info["per_off_mu"].get(int(offset))
        if mu is None:
            continue
        ious.append(iou_for_mu(mu, info["L"], info["R"], sigma))
    n_match = len(ious)
    iou_match = float(np.mean(ious)) if ious else float("nan")
    return iou_match, n_match


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, default="sheath_blight")
    p.add_argument("--run", type=int, default=4)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--models", type=str, required=True)
    p.add_argument("--offsets", type=str, default="60,90,105,120")
    p.add_argument("--oracle_sigma", type=float, default=5.0,
                   help="σ at which Oracle IoU is computed (default 5.0)")
    p.add_argument("--sigma_sweep", type=str, default="2.5,3.0,3.5,4.0,4.5,5.0,6.0")
    p.add_argument("--sigma_sweep_offsets", type=str, default="105,120")
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--per_model_sigma", type=str, default="",
                   help="Comma-separated LABEL=σ pairs. e.g. "
                        "'D=15 baseline (asym=15)=4.0,D=15 phenobias=4.5'. "
                        "Used by sample_grid_csv. Missing labels use --oracle_sigma.")
    p.add_argument("--sample_grid_csv", type=str, default=None,
                   help="If set, dump per-(model, sample, offset) IoU grid to this CSV. "
                        "Uses --per_model_sigma if provided, else --oracle_sigma.")
    p.add_argument("--nohead_labels", type=str, default="",
                   help="Comma-separated model labels (matching --models) to evaluate with "
                        "the phenology bias head bypassed (mu = mu_temporal only). "
                        "Same ckpt, different inference path. Use to quantify how much "
                        "of mu_range comes from phen_bias vs the encoder.")
    # ---- Dispatch alert mode (opt-in) ----
    p.add_argument("--dispatch_summary_json", type=str, default="",
                   help="If set, use Stage-1 dispatch_group_tau alerts in place of "
                        "the default first-crossing alert_map. Path to the year-split "
                        "group_tau_hybrid_summary.json from phase_t_group_tau_hybrid.")
    p.add_argument("--dispatch_target_label", type=str, default="R>=0.88",
                   help="Selection key inside the summary JSON (e.g. 'R>=0.85', "
                        "'R>=0.88', 'R>=0.90').")
    p.add_argument("--dispatch_a_ckpt", type=str, default="",
                   help="Stage-1 A (baseline, no-history) ckpt for dispatch alert.")
    p.add_argument("--dispatch_d_ckpt", type=str, default="",
                   help="Stage-1 D (history) ckpt for dispatch alert.")
    p.add_argument("--dispatch_ablation_mode", type=str, default="",
                   choices=["", "ablate_missing", "ablate_train_mean"],
                   help="Phase A.3 ablation override at Stage-2 eval. "
                        "'ablate_missing' pins all rows to the in-distribution "
                        "pre-alert/missing fill (no hard-zero OOD). "
                        "'ablate_train_mean' replaces feature slots with train "
                        "mean and keeps missing=1. Empty (default) uses the "
                        "fill mode stored in the ckpt.")
    p.add_argument("--dispatch_ablation_csv", type=str, default="",
                   help="Override CSV for train-mean computation (defaults to "
                        "the CSV stored in the ckpt meta).")
    p.add_argument("--stage2_dispatch_feature_csv_override", type=str, default="",
                   help="Replace the ckpt-stored stage2_dispatch_feature_csv "
                        "path with this CSV. Use when re-evaluating an existing "
                        "Stage-2 ckpt against a dispatch CSV built for a "
                        "different val/test year split (e.g. multi-year "
                        "offset-selector aux grids). The override is wired "
                        "before any ablation logic so train_mean / load paths "
                        "all see the override.")
    p.add_argument("--eval_split", type=str, default="test",
                   choices=["test", "val"],
                   help="Which split's cohort to evaluate (Stage 1 alert + "
                        "Stage 2 forward + sample_grid). Use 'val' to produce "
                        "the validation sample_grid for offset selection; the "
                        "selected offset is then applied to the 'test' grid.")
    p.add_argument("--dispatch_gate_method", type=str,
                   default="dispatch_group_tau",
                   choices=["dispatch_group_tau", "A_baseline", "D_history"],
                   help="Per-pest Stage 1 gate policy used to derive alert_t. "
                        "Must match the gate used in build_dispatch_feature_table "
                        "(otherwise eval alerts diverge from training cohort).")
    p.add_argument("--per_model_extra_offsets", type=str, default="",
                   help="Per-label extra offsets appended to --offsets, written into "
                        "sample_grid_csv only. Format: 'LABEL=o1,o2,..;LABEL2=...'. "
                        "Other models still use the base --offsets. Useful for expanding "
                        "the action space of one model (e.g. selector) without re-running "
                        "every model. Stage 2 nowcast must include the resulting "
                        "alert_tstar+offset frame; values outside [window, T] are silently "
                        "unmatched.")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[abort] CUDA required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    _ = resolve_pest(args.pest)

    offsets = [int(x) for x in str(args.offsets).split(",") if x.strip()]
    sigmas_sweep = [float(x) for x in str(args.sigma_sweep).split(",") if x.strip()]
    sigma_sweep_offsets = [int(x) for x in str(args.sigma_sweep_offsets).split(",") if x.strip()]
    print(f"[config] offsets={offsets}  oracle_sigma={args.oracle_sigma}")
    print(f"[config] σ sweep: σ={sigmas_sweep}  offsets={sigma_sweep_offsets}")

    model_entries = []
    for part in str(args.models).split(";"):
        part = part.strip()
        if not part: continue
        label, ckpt = part.split("|", 1)
        model_entries.append((label.strip(), ckpt.strip()))
    print(f"[models] {len(model_entries)}")

    print("\n[stage1] alert_map…")
    dispatch_ctx = None
    if args.dispatch_summary_json:
        if not (args.dispatch_a_ckpt and args.dispatch_d_ckpt):
            raise SystemExit(
                "[dispatch] --dispatch_a_ckpt and --dispatch_d_ckpt are required "
                "when --dispatch_summary_json is set."
            )
        alert_map, n_total, dispatch_ctx = build_dispatch_alert_map(
            Path(args.dispatch_a_ckpt), Path(args.dispatch_d_ckpt),
            Path(args.dispatch_summary_json), args.dispatch_target_label,
            args.run, args,
            include_splits=(str(args.eval_split),),
            gate_method=str(args.dispatch_gate_method),
        )
        # Lead-shift diagnostic: compare dispatch alert_t vs the legacy
        # phase_r k=1 first-crossing alert_t (which historical Stage-1 reports
        # used as alert_tstar). dispatch alert_t is the k-th completion row, so
        # it is at most a few stride days later per the same series; the diff
        # in alert_DOY equals the negative shift in lead_days vs reported.
        try:
            legacy_alert_map, _ = build_stage1_alert_map(
                Path(args.stage1_ckpt), args.run, args,
                eval_split=str(args.eval_split),
            )
        except Exception as e:
            print(f"[lead-shift] legacy alert_map build failed: {e}  (skipping diagnostic)")
            legacy_alert_map = {}
        common = [sy for sy in alert_map if sy in legacy_alert_map]
        if common:
            diffs_doy = np.asarray(
                [int(alert_map[sy]) - int(legacy_alert_map[sy]) for sy in common],
                dtype=int,
            )
            lead_shift = -diffs_doy
            print(
                f"[lead-shift] common_sy={len(common)}  "
                f"alert_DOY_diff (dispatch - legacy): "
                f"mean={diffs_doy.mean():+.2f}  median={int(np.median(diffs_doy))}  "
                f"p05={int(np.percentile(diffs_doy, 5))}  "
                f"p95={int(np.percentile(diffs_doy, 95))}  "
                f"min={int(diffs_doy.min())}  max={int(diffs_doy.max())}  "
                f"|  lead_shift (dispatch - legacy): mean={lead_shift.mean():+.2f}"
            )
            only_dispatch = [sy for sy in alert_map if sy not in legacy_alert_map]
            only_legacy = [sy for sy in legacy_alert_map if sy not in alert_map]
            print(f"[lead-shift] only_dispatch={len(only_dispatch)}  "
                  f"only_legacy={len(only_legacy)}")
    else:
        alert_map, n_total = build_stage1_alert_map(
            Path(args.stage1_ckpt), args.run, args,
            eval_split=str(args.eval_split),
        )

    # Parse per-model sigma map for sample_grid dump.
    per_model_sigma = {}
    if args.per_model_sigma:
        for pair in str(args.per_model_sigma).split(","):
            pair = pair.strip()
            if not pair or "=" not in pair:
                continue
            label_pm, sig_pm = pair.rsplit("=", 1)
            per_model_sigma[label_pm.strip()] = float(sig_pm.strip())
    if per_model_sigma:
        print(f"[per_model_sigma] {per_model_sigma}")

    nohead_labels = set()
    if args.nohead_labels:
        nohead_labels = {s.strip() for s in str(args.nohead_labels).split(",") if s.strip()}
    if nohead_labels:
        print(f"[nohead_labels] {sorted(nohead_labels)}  (phen_head bypassed at inference)")

    per_model_extra_offsets: dict[str, list[int]] = {}
    if args.per_model_extra_offsets:
        for entry in str(args.per_model_extra_offsets).split(";"):
            entry = entry.strip()
            if not entry or "=" not in entry:
                continue
            lbl, off_str = entry.rsplit("=", 1)
            extras = [int(x) for x in off_str.split(",") if x.strip()]
            per_model_extra_offsets[lbl.strip()] = extras
    if per_model_extra_offsets:
        print(f"[per_model_extra_offsets] {per_model_extra_offsets}")

    oracle_rows = []
    sweep_rows = []
    sample_grid_rows = []

    for label, ckpt_path in model_entries:
        print(f"\n----- {label} -----")
        bypass_ph = (label in nohead_labels)
        row_map, doy_start = build_stage2_row_map(
            Path(ckpt_path), args.run, args, device,
            bypass_phen_head=bypass_ph, eval_split=str(args.eval_split),
        )
        samples_dict = collect_per_sample_mu(alert_map, row_map, doy_start, offsets)
        print(f"  samples_with_any_match = {len(samples_dict)}")

        # Oracle
        per_iou, per_off = oracle_iou(samples_dict, offsets, args.oracle_sigma)
        n_match_any = len(per_iou)
        oracle_iou_matched = float(per_iou.mean()) if n_match_any else float("nan")
        oracle_iou_overall = float(per_iou.sum() / n_total) if n_match_any else 0.0
        off_hist = pd.Series(per_off).value_counts().sort_index() if n_match_any else pd.Series(dtype=int)
        print(f"  [oracle σ={args.oracle_sigma}] n_match_any={n_match_any}  "
              f"IoU_matched={oracle_iou_matched:.4f}  IoU_overall={oracle_iou_overall:.4f}")
        if not off_hist.empty:
            hist_str = "  ".join(f"{int(o)}d:{int(c)}" for o, c in off_hist.items())
            print(f"  [oracle best-offset hist] {hist_str}")

        # Fixed IoU per offset (for σ=oracle_sigma)
        per_off_iou = {}
        for o in offsets:
            iou_m, n_m = fixed_iou_at(samples_dict, o, args.oracle_sigma)
            per_off_iou[o] = (iou_m, n_m)
            # mu stats at this offset
            mu_o = np.asarray([info["per_off_mu"][int(o)] for info in samples_dict.values()
                               if int(o) in info["per_off_mu"]], dtype=float)
            mu_mean = float(mu_o.mean()) if mu_o.size else float("nan")
            mu_std = float(mu_o.std(ddof=0)) if mu_o.size else float("nan")
            print(f"  fixed σ={args.oracle_sigma} off={o}: IoU_matched={iou_m:.4f}  "
                  f"n_match={n_m}  IoU_overall={iou_m * n_m / n_total:.4f}  "
                  f"mu_mean={mu_mean:.2f}  mu_std={mu_std:.2f}")
        # Per-sample mu_range across the requested offsets (matched only)
        ranges = []
        for info in samples_dict.values():
            vals = [info["per_off_mu"][int(o)] for o in offsets if int(o) in info["per_off_mu"]]
            if len(vals) >= 2:
                ranges.append(float(max(vals) - min(vals)))
        if ranges:
            ranges_arr = np.asarray(ranges, dtype=float)
            print(f"  [mu_range across {len(offsets)} offsets] n={len(ranges_arr)} "
                  f"mean={ranges_arr.mean():.2f} std={ranges_arr.std(ddof=0):.2f} "
                  f"min={ranges_arr.min():.2f} p50={np.median(ranges_arr):.2f} "
                  f"max={ranges_arr.max():.2f}")
        # Coverage-aware best selection: pick best by IoU_matched and by
        # IoU_overall separately. matched-best can be misleading when n_match
        # is small (an offset that matched only a few easy samples can look
        # great in matched terms while losing badly on overall).
        def _overall(o):
            iou_m, n_m = per_off_iou[o]
            return float(iou_m * n_m / n_total) if n_total > 0 else 0.0
        best_matched_off = max(per_off_iou, key=lambda o: per_off_iou[o][0])
        best_overall_off = max(per_off_iou, key=_overall)
        fixed_best_matched_iou = per_off_iou[best_matched_off][0]
        fixed_best_overall_iou_overall = _overall(best_overall_off)
        print(f"  [coverage-aware] best by IoU_matched: off={best_matched_off}  "
              f"IoU_matched={fixed_best_matched_iou:.4f}  "
              f"n_match={per_off_iou[best_matched_off][1]}  "
              f"coverage={per_off_iou[best_matched_off][1]/max(n_total,1):.4f}  "
              f"IoU_overall={_overall(best_matched_off):.4f}")
        print(f"  [coverage-aware] best by IoU_overall: off={best_overall_off}  "
              f"IoU_matched={per_off_iou[best_overall_off][0]:.4f}  "
              f"n_match={per_off_iou[best_overall_off][1]}  "
              f"coverage={per_off_iou[best_overall_off][1]/max(n_total,1):.4f}  "
              f"IoU_overall={fixed_best_overall_iou_overall:.4f}")

        oracle_rows.append({
            "model": label,
            "n_match_any": n_match_any,
            "n_total": int(n_total),
            # Legacy fields (matched-best) — kept for backward compat with
            # phase_s10/s11/s12 readers; they index 'fixed_best_offset' etc.
            "fixed_best_offset": best_matched_off,
            "fixed_best_IoU_matched": fixed_best_matched_iou,
            "fixed_best_n_match": per_off_iou[best_matched_off][1],
            "fixed_best_IoU_overall": fixed_best_matched_iou * per_off_iou[best_matched_off][1] / n_total,
            # NEW: explicit matched-best vs overall-best columns
            "fixed_best_matched_offset": best_matched_off,
            "fixed_best_matched_n_match": per_off_iou[best_matched_off][1],
            "fixed_best_matched_coverage": per_off_iou[best_matched_off][1] / max(n_total, 1),
            "fixed_best_matched_IoU_matched": fixed_best_matched_iou,
            "fixed_best_matched_IoU_overall": fixed_best_matched_iou * per_off_iou[best_matched_off][1] / n_total,
            "fixed_best_overall_offset": best_overall_off,
            "fixed_best_overall_n_match": per_off_iou[best_overall_off][1],
            "fixed_best_overall_coverage": per_off_iou[best_overall_off][1] / max(n_total, 1),
            "fixed_best_overall_IoU_matched": per_off_iou[best_overall_off][0],
            "fixed_best_overall_IoU_overall": fixed_best_overall_iou_overall,
            "oracle_IoU_matched": oracle_iou_matched,
            "oracle_IoU_overall": oracle_iou_overall,
            "delta_IoU_matched": oracle_iou_matched - fixed_best_matched_iou,
            "off_hist_60":  int(off_hist.get(60, 0)),
            "off_hist_90":  int(off_hist.get(90, 0)),
            "off_hist_105": int(off_hist.get(105, 0)),
            "off_hist_120": int(off_hist.get(120, 0)),
        })

        # σ sweep at offsets {105, 120}
        for off in sigma_sweep_offsets:
            for sig in sigmas_sweep:
                iou_m, n_m = fixed_iou_at(samples_dict, off, sig)
                sweep_rows.append({
                    "model": label,
                    "offset": int(off),
                    "sigma": float(sig),
                    "n_match": int(n_m),
                    "IoU_matched": float(iou_m),
                    "IoU_overall": float(iou_m * n_m / n_total) if np.isfinite(iou_m) else float("nan"),
                })

        # Sample-level grid dump (one row per (model, sample, offset)).
        # σ is the per-model best σ if --per_model_sigma provided, else --oracle_sigma.
        # If this label has extra offsets, run an extended forward at those frames
        # and write them into the same sample_grid (so phase_s3 sees the wider set).
        extra_offs = per_model_extra_offsets.get(label, [])
        offsets_for_grid = list(offsets) + [int(o) for o in extra_offs if int(o) not in offsets]
        if extra_offs:
            print(f"  [extra_offsets for {label!r}] +{extra_offs}  "
                  f"→ grid uses offsets={offsets_for_grid}")
            # Re-collect samples_dict to include extra offsets without re-running forward
            # (row_map already contains every (site, year, tstar_frame); only the lookup
            # set widens).
            samples_dict = collect_per_sample_mu(alert_map, row_map, doy_start, offsets_for_grid)

        if args.sample_grid_csv:
            model_sig = float(per_model_sigma.get(label, args.oracle_sigma))
            for (site, year), info in samples_dict.items():
                tstar_abs = int(info["alert_t"]) + doy_start - 1
                L_abs = int(info["L"])
                R_abs = int(info["R"])
                true_event_doy = float(L_abs + R_abs) * 0.5   # interval midpoint
                # Dispatch features (constant per sample across offsets).
                # When dispatch_ctx is None this is a no-op; existing column
                # set is preserved.
                dfeat = (
                    _dispatch_features_for_sy(
                        (str(site), int(year)), int(info["alert_t"]), dispatch_ctx,
                    )
                    if dispatch_ctx is not None else {}
                )
                for o in offsets_for_grid:
                    mu_o = info["per_off_mu"].get(int(o))
                    if mu_o is None:
                        row = {
                            "model": label,
                            "sample_id": f"{site}-{int(year)}",
                            "site": site, "year": int(year),
                            "t_star_doy": tstar_abs,
                            "true_event_doy": true_event_doy,
                            "L": L_abs, "R": R_abs,
                            "mu": float("nan"),
                            "sigma": model_sig,
                            "offset": int(o),
                            "iou_matched": 0.0,
                            "matched": False,
                        }
                    else:
                        iou_o = iou_for_mu(mu_o, L_abs, R_abs, model_sig)
                        row = {
                            "model": label,
                            "sample_id": f"{site}-{int(year)}",
                            "site": site, "year": int(year),
                            "t_star_doy": tstar_abs,
                            "true_event_doy": true_event_doy,
                            "L": L_abs, "R": R_abs,
                            "mu": float(mu_o),
                            "sigma": model_sig,
                            "offset": int(o),
                            "iou_matched": float(iou_o),
                            "matched": True,
                        }
                    if dfeat:
                        row.update(dfeat)
                    sample_grid_rows.append(row)

        del row_map
        torch.cuda.empty_cache()

    if args.sample_grid_csv and sample_grid_rows:
        sg = pd.DataFrame(sample_grid_rows)
        sg.to_csv(args.sample_grid_csv, index=False)
        print(f"\n[csv] sample-level grid → {args.sample_grid_csv}  ({len(sg)} rows)")

    df_oracle = pd.DataFrame(oracle_rows)
    df_sweep = pd.DataFrame(sweep_rows)

    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 30)
    pd.set_option("display.max_rows", 200)

    print(f"\n=================== ORACLE multi-offset (σ={args.oracle_sigma}) ===================")
    print(df_oracle.to_string(index=False))

    print(f"\n=================== σ × offset sweep (IoU) ===================")
    print(df_sweep.to_string(index=False))

    # Best σ per (model, offset)
    print(f"\n=================== Best σ per (model, offset) ===================")
    best_sigma_rows = []
    for (label, off), sub in df_sweep.groupby(["model", "offset"], sort=False):
        if sub["IoU_matched"].dropna().empty:
            continue
        idx = sub["IoU_matched"].idxmax()
        r = sub.loc[idx]
        print(f"  [{label}] offset={int(r['offset'])}: best σ={r['sigma']:.1f}  "
              f"IoU_matched={r['IoU_matched']:.4f}  IoU_overall={r['IoU_overall']:.4f}  "
              f"n_match={int(r['n_match'])}")
        best_sigma_rows.append({"model": label, "offset": int(r["offset"]),
                                 "best_sigma": float(r["sigma"]),
                                 "IoU_matched": float(r["IoU_matched"]),
                                 "IoU_overall": float(r["IoU_overall"])})

    # Per-model summary table (sample request 형식)
    print(f"\n=================== Per-model summary (IoU-centric) ===================")
    rows = []
    for _, r in df_oracle.iterrows():
        label = r["model"]
        # pick best σ across both sweep offsets for this model
        sub = df_sweep[df_sweep["model"] == label]
        idx_best = sub["IoU_matched"].idxmax() if not sub["IoU_matched"].dropna().empty else None
        if idx_best is not None:
            best_off = int(sub.loc[idx_best, "offset"])
            best_sig = float(sub.loc[idx_best, "sigma"])
            best_iou = float(sub.loc[idx_best, "IoU_matched"])
            best_iou_ov = float(sub.loc[idx_best, "IoU_overall"])
        else:
            best_off = None; best_sig = float("nan"); best_iou = float("nan"); best_iou_ov = float("nan")
        rows.append({
            "Model": label,
            "Fixed best IoU (σ=oracle)": r["fixed_best_IoU_matched"],
            "Fixed best (off)": int(r["fixed_best_offset"]),
            "Oracle IoU_matched": r["oracle_IoU_matched"],
            "Oracle IoU_overall": r["oracle_IoU_overall"],
            "Δ (Oracle − Fixed)": r["delta_IoU_matched"],
            "σ-tuned best IoU": best_iou,
            "σ-tuned best (off, σ)": f"{best_off},{best_sig:.1f}",
            "σ-tuned IoU_overall": best_iou_ov,
        })
    df_sum = pd.DataFrame(rows)
    print(df_sum.to_string(index=False))

    if args.out_csv:
        df_oracle.to_csv(args.out_csv, index=False)
        sweep_csv = args.out_csv.replace(".csv", "_sigma_sweep.csv")
        df_sweep.to_csv(sweep_csv, index=False)
        print(f"\n[csv] oracle → {args.out_csv}  ({len(df_oracle)} rows)")
        print(f"[csv] σ sweep → {sweep_csv}  ({len(df_sweep)} rows)")


if __name__ == "__main__":
    main()
