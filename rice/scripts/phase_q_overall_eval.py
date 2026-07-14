"""
Phase Q — Honest operational evaluation against the FULL 575 interval cohort.

All percentages are over n_total = 575 (full interval site-year count on test).
Samples that fall outside the cohort (alert_tstar + offset row missing, after_R,
no alert, no features) are counted as P_failed_overall — i.e. operational misses.

Two evaluation modes:

  (A) FIXED grid: model × offset × σ × shift = 5 × 5 × 2 × 11 = 550 cells
      offsets ∈ {30, 60, 90, 105, 120}
      σ ∈ {3.5, 5.0}
      shifts ∈ {20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40}

  (B) VARIABLE offset (GBM, σ=3.5, shift=30): for 3 selected models only
      (baseline, 2-sided, phenobias). Per-sample predicted offset.

All cells use:
  PI    = [mu - 1.96σ,        mu + 1.96σ]
  PI_op = [mu - shift - 1.96σ, mu - shift + 1.96σ]
  lead  = L - PI_op.end

Operational buckets (same as Phase N):
  MISSED      lead < 0
  TOO_LATE    0   ≤ lead < 7
  URGENT      7   ≤ lead < 14
  IDEAL       14  ≤ lead < 30
  ADVANCE     30  ≤ lead < 45
  TOO_EARLY   lead ≥ 45

P_*_overall = bucket_count / n_total          (n_total = 575)
P_failed_overall = (MISSED + TOO_LATE) / n_total + (n_total - n_match) / n_total
                 = "MISSED-equivalent + cohort-out" / n_total
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import GradientBoostingRegressor

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


WEATHER_NAMES = [
    "rain_7d_sum", "tmean_7d_mean", "tmax_7d_max", "tmin_7d_min",
    "rh_7d_mean", "sun_7d_sum", "trange_7d_mean",
]


def _trapz(arr):
    f = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if f is None:
        return float(np.sum((arr[:-1] + arr[1:]) * 0.5)) if len(arr) >= 2 else float("nan")
    return float(f(arr))


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


def build_stage1_alert_and_features(stage1_ckpt_path, run, args, want_test_features=False):
    """
    Returns (alert_map_test, n_interval_test, train_df, test_df, weather_present, tau)
    where train_df / test_df hold per-(site,year) features (4 score stats + alert
    + lat/lon/year + weather snapshot at alert_tstar + 90), used by Variable offset.
    """
    ckpt = torch.load(stage1_ckpt_path, map_location="cpu")
    add_tpos = bool(ckpt.get("add_tstar_position_feature", False))
    nc_window = int(ckpt.get("nowcast_window", 28))
    nc_stride = int(ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("nowcast_event_time_proxy", "r"))

    _, get_feature_cols = resolve_pest(args.pest)
    feature_cols = get_feature_cols(run)
    _, _, _T, samples = build_samples_for_run(run, get_feature_cols)
    train_seas, val_seas, test_seas = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )
    train_s = build_nowcast_samples(train_seas, window=nc_window, stride=nc_stride,
                                    only_pre_event=nc_only_pre, event_time_proxy=nc_proxy)
    val_s = build_nowcast_samples(val_seas, window=nc_window, stride=nc_stride,
                                  only_pre_event=nc_only_pre, event_time_proxy=nc_proxy)
    test_s = build_nowcast_samples(test_seas, window=nc_window, stride=nc_stride,
                                   only_pre_event=nc_only_pre, event_time_proxy=nc_proxy)

    y_val = np.asarray([int(s["y_event"]) for s in val_s])
    X_train_tab = build_tabular_from_samples(train_s, add_tstar_position_feature=add_tpos)
    X_val_tab = build_tabular_from_samples(val_s, add_tstar_position_feature=add_tpos)
    X_test_tab = build_tabular_from_samples(test_s, add_tstar_position_feature=add_tpos)
    clf = ckpt["trained_states"][0]["sk_model"]
    p_train = clf.predict_proba(X_train_tab)[:, 1]
    p_val_raw = clf.predict_proba(X_val_tab)[:, 1]
    p_test_raw = clf.predict_proba(X_test_tab)[:, 1]
    t_best, _ = fit_temperature_grid(y_val, p_val_raw)
    p_train_cal = apply_temperature(p_train, t_best)
    p_val_cal = apply_temperature(p_val_raw, t_best)
    p_test_cal = apply_temperature(p_test_raw, t_best)
    tau = best_tau_by_f1(y_val, p_val_cal)
    print(f"[stage1] T*={t_best:.3f}  tau={tau:.3f}")

    # alert_map (test only)
    alert_test = {}
    for s, p_cal in zip(test_s, p_test_cal):
        if p_cal < tau:
            continue
        key = (str(s["site_id"]), int(s["year"]))
        prev = alert_test.get(key)
        if prev is None or int(s["tstar"]) < prev:
            alert_test[key] = int(s["tstar"])
    n_interval_test = sum(1 for s in test_seas if str(s["censor_type"]) != "right")
    print(f"[stage1] alert_map(test) size = {len(alert_test)}  n_interval_test = {n_interval_test}")

    if not want_test_features:
        return alert_test, n_interval_test, None, None, None, tau

    lat_idx = feature_cols.index("좌표-위도") if "좌표-위도" in feature_cols else None
    lon_idx = feature_cols.index("좌표-경도") if "좌표-경도" in feature_cols else None
    weather_idx = [feature_cols.index(n) for n in WEATHER_NAMES if n in feature_cols]
    weather_present = [n for n in WEATHER_NAMES if n in feature_cols]

    def _build_group_df(samples_now, scores, seas_list, with_alert):
        idx_map = defaultdict(list)
        for i, s in enumerate(samples_now):
            idx_map[(str(s["site_id"]), int(s["year"]))].append(i)
        seas_lookup = {(str(s["site_id"]), int(s["year"])): s for s in seas_list}
        rows = []
        for (site, year), idxs in idx_map.items():
            ts = np.asarray([int(samples_now[i]["tstar"]) for i in idxs])
            order = np.argsort(ts)
            ts_sorted = ts[order]
            sc_sorted = np.asarray([float(scores[i]) for i in idxs])[order]
            n = len(sc_sorted)
            marker = float(ts_sorted[-1])
            score_at = float(sc_sorted[-1])
            last30 = sc_sorted[-min(30, n):]
            score_peak = float(np.max(last30))
            score_auc = _trapz(last30)
            last14 = sc_sorted[-min(14, n):]
            slope = float(np.polyfit(np.arange(len(last14)), last14, 1)[0]) if len(last14) >= 2 else float("nan")

            seas = seas_lookup.get((site, int(year)))
            if seas is None:
                continue
            L_val = int(seas["L"]) if str(seas["censor_type"]) != "right" else None
            R_val = int(seas["R"]) if str(seas["censor_type"]) != "right" else None
            X = np.asarray(seas["X"])
            lat = float(X[0, lat_idx]) if lat_idx is not None else float("nan")
            lon = float(X[0, lon_idx]) if lon_idx is not None else float("nan")
            alert_t = None
            if with_alert:
                for i in idxs:
                    if scores[i] >= float(tau):
                        ti = int(samples_now[i]["tstar"])
                        if alert_t is None or ti < alert_t:
                            alert_t = ti
            rec = {
                "site_id": site, "year": int(year),
                "score_at_tstar": score_at, "score_peak_30": score_peak,
                "score_slope_14": slope, "score_auc_30": auc if False else score_auc,
                "marker_tstar": marker, "L": L_val, "R": R_val,
                "lat": lat, "lon": lon, "year_feat": int(year),
                "alert_tstar": alert_t,
            }
            # weather snapshot
            target_frame = alert_t + 90 if alert_t is not None else None
            T = int(X.shape[0])
            if target_frame is not None and 1 <= target_frame <= T:
                row = X[int(target_frame) - 1]
                for n, idx in zip(weather_present, weather_idx):
                    rec[f"w_{n}"] = float(row[idx])
            else:
                for n in weather_present:
                    rec[f"w_{n}"] = float("nan")
            rows.append(rec)
        return pd.DataFrame(rows)

    train_df = _build_group_df(train_s, p_train_cal, train_seas, with_alert=True)
    test_df = _build_group_df(test_s, p_test_cal, test_seas, with_alert=True)
    print(f"[features] train rows={len(train_df)} (alerted={train_df['alert_tstar'].notna().sum()}) "
          f"test rows={len(test_df)} (alerted={test_df['alert_tstar'].notna().sum()})")
    return alert_test, n_interval_test, train_df, test_df, weather_present, tau


def attach_mu_at_default(df, row_map, doy_start, default_candidates=(90, 105, 120, 75, 60, 135)):
    mu_vals, used_off = [], []
    for _, r in df.iterrows():
        if r["alert_tstar"] is None or pd.isna(r["alert_tstar"]):
            mu_vals.append(float("nan")); used_off.append(-1); continue
        found = None
        for cand in default_candidates:
            target = int(r["alert_tstar"]) + int(cand)
            info = row_map.get((str(r["site_id"]), int(r["year"]), int(target)))
            if info is not None and info["ctype"] == 0:
                found = (info["mu"] + doy_start - 1, cand)
                break
        if found is None:
            mu_vals.append(float("nan")); used_off.append(-1)
        else:
            mu_vals.append(found[0]); used_off.append(found[1])
    df = df.copy()
    df["mu_at_default_off"] = mu_vals
    df["default_off"] = used_off
    return df


def build_stage2_row_map(stage2_ckpt_path: Path, run: int, args, device: torch.device):
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu")
    C.DOY_START = int(ckpt.get("doy_start", C.DOY_START))
    C.DOY_END = int(ckpt.get("doy_end", C.DOY_END))
    doy_start = int(ckpt.get("doy_start", C.DOY_START))

    phen_bias_head = bool(int(ckpt.get("stage2_phenology_bias_head", 0)))
    phen_hidden = int(ckpt.get("stage2_phenology_hidden", 8))
    pheno_ext_cols = (["best_suitability", "best_months", "offset_days", "window_idx"]
                      if phen_bias_head else None)
    print(f"  [phen_bias_head] enabled={phen_bias_head}  hidden={phen_hidden}")

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples2 = build_samples_for_run(run, get_feature_cols, pheno_ext_cols=pheno_ext_cols)
    train_s2_base, _, test_s2_base = split_samples(
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
    test_s2 = build_stage2_nowcast_samples(
        test_s2_base, window=nc_window, stride=nc_stride, tstar_start=nc_tstart,
        only_pre_event=nc_only_pre, event_time_proxy=nc_proxy, require_tstar_before_L=nc_req,
    )

    x_mean, x_std = compute_norm_stats(train_s2_base)
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
    model.asym_weight = float(ckpt.get("stage2_pmf_asym_weight", 15.0))
    model.right_weight = float(ckpt.get("stage2_pmf_right_weight", 0.3))
    model.target_offset = float(ckpt.get("stage2_pmf_target_offset", 5.0))
    model.asym_weight_early = float(ckpt.get("stage2_pmf_asym_weight_early", 0.0))
    model.target_early_offset = float(ckpt.get("stage2_pmf_target_early_offset", 30.0))
    model.target_mode = str(ckpt.get("stage2_pmf_target_mode", "l_offset"))
    model.zone_late_weight = float(ckpt.get("stage2_pmf_zone_late_weight", 0.0))
    model.zone_too_late_weight = float(ckpt.get("stage2_pmf_zone_too_late_weight", 0.0))
    model.zone_missed_weight = float(ckpt.get("stage2_pmf_zone_missed_weight", 0.0))
    model.zone_too_early_weight = float(ckpt.get("stage2_pmf_zone_too_early_weight", 0.0))
    model.load_state_dict(ckpt["trained_states"][0]["state_dict"], strict=False)
    model.eval()

    row_map = {}
    gi = 0
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
            B, K = mu_np.shape
            for bi in range(B):
                g = test_groups[gi + bi]
                for ki in range(K):
                    if not v_np[bi, ki]:
                        continue
                    key = (str(g["site_id"]), int(g["year"]), int(ts_np[bi, ki]))
                    row_map[key] = {
                        "mu": float(mu_np[bi, ki]),
                        "true_L": int(L_np[bi, ki]),
                        "true_R": int(R_np[bi, ki]),
                        "ctype": int(c_np[bi, ki]),
                    }
            gi += B

    # Also build train row_map (for Variable offset GBM fit) -- separately
    return row_map, doy_start, phen_bias_head


def build_stage2_row_map_train(stage2_ckpt_path: Path, run: int, args, device: torch.device):
    """Train-split row_map (used only for Variable offset best_offset target)."""
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu")
    C.DOY_START = int(ckpt.get("doy_start", C.DOY_START))
    C.DOY_END = int(ckpt.get("doy_end", C.DOY_END))
    doy_start = int(ckpt.get("doy_start", C.DOY_START))

    phen_bias_head = bool(int(ckpt.get("stage2_phenology_bias_head", 0)))
    phen_hidden = int(ckpt.get("stage2_phenology_hidden", 8))
    pheno_ext_cols = (["best_suitability", "best_months", "offset_days", "window_idx"]
                      if phen_bias_head else None)

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples2 = build_samples_for_run(run, get_feature_cols, pheno_ext_cols=pheno_ext_cols)
    train_s2_base, _, _ = split_samples(
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
    train_s2 = build_stage2_nowcast_samples(
        train_s2_base, window=nc_window, stride=nc_stride, tstar_start=nc_tstart,
        only_pre_event=nc_only_pre, event_time_proxy=nc_proxy, require_tstar_before_L=nc_req,
    )

    x_mean, x_std = compute_norm_stats(train_s2_base)
    groups = group_stage2_samples_by_site_year(train_s2)
    ds = GroupedIntervalEventDataset(groups, x_mean, x_std)
    loader = make_loader(ds, 1, shuffle=False, collate_fn=collate_grouped_stage2)

    d_model = int(ckpt.get("d_model", C.D_MODEL))
    n_head = int(ckpt.get("n_head", C.N_HEAD))
    n_layers = int(ckpt.get("n_layers", C.N_LAYERS))
    model = HierarchicalCausalHazardTransformer(
        d_in=int(train_s2[0]["X"].shape[1]),
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
    model.load_state_dict(ckpt["trained_states"][0]["state_dict"], strict=False)
    model.eval()

    row_map = {}
    gi = 0
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
            B, K = mu_np.shape
            for bi in range(B):
                g = groups[gi + bi]
                for ki in range(K):
                    if not v_np[bi, ki]:
                        continue
                    key = (str(g["site_id"]), int(g["year"]), int(ts_np[bi, ki]))
                    row_map[key] = {
                        "mu": float(mu_np[bi, ki]),
                        "true_L": int(L_np[bi, ki]),
                        "true_R": int(R_np[bi, ki]),
                        "ctype": int(c_np[bi, ki]),
                    }
            gi += B
    return row_map, doy_start


def buckets_from_leads(leads: np.ndarray):
    missed = leads < 0
    too_late = (leads >= 0) & (leads < 7)
    urgent = (leads >= 7) & (leads < 14)
    ideal = (leads >= 14) & (leads < 30)
    advance = (leads >= 30) & (leads < 45)
    too_early = leads >= 45
    return missed, too_late, urgent, ideal, advance, too_early


def overall_metrics(matched_df: pd.DataFrame, n_total: int) -> dict:
    """matched_df columns: mu, L, R, PI_lo, PI_hi, lead."""
    n_match = len(matched_df)
    out_cohort = n_total - n_match
    if n_match == 0:
        return {
            "n_match": 0, "n_out_cohort": out_cohort,
            "P_ideal_overall": 0.0,
            "P_useful_A_overall": 0.0,
            "P_useful_B_overall": 0.0,
            "P_missed_or_late_overall": 0.0,
            "P_too_early_overall": 0.0,
            "P_failed_overall": 100.0 * out_cohort / n_total,
            "P_ideal_conditional": float("nan"),
            "IoU_PI_LR": float("nan"),
        }
    leads = matched_df["lead"].to_numpy(dtype=float)
    missed, too_late, urgent, ideal, advance, too_early = buckets_from_leads(leads)
    n_ideal = int(ideal.sum())
    n_useful_A = int(((leads >= 0) & (leads < 45)).sum())
    n_useful_B = int(((leads >= 7) & (leads < 45)).sum())
    n_M_L = int(missed.sum() + too_late.sum())
    n_TE = int(too_early.sum())

    iou_pi_lr = []
    for _, r in matched_df.iterrows():
        pL_pi = int(round(r["PI_lo"]))
        pR_pi = int(round(r["PI_hi"]))
        iou, _, _ = overlap_metrics(pL_pi, pR_pi, int(r["L"]), int(r["R"]))
        iou_pi_lr.append(iou)

    return {
        "n_match": int(n_match),
        "n_out_cohort": int(out_cohort),
        "P_ideal_overall": 100.0 * n_ideal / n_total,
        "P_useful_A_overall": 100.0 * n_useful_A / n_total,
        "P_useful_B_overall": 100.0 * n_useful_B / n_total,
        "P_missed_or_late_overall": 100.0 * n_M_L / n_total,
        "P_too_early_overall": 100.0 * n_TE / n_total,
        "P_failed_overall": 100.0 * (n_M_L + out_cohort) / n_total,
        "P_ideal_conditional": 100.0 * n_ideal / n_match,
        "IoU_PI_LR": float(np.mean(iou_pi_lr)),
    }


def evaluate_fixed_cell(alert_map, row_map, doy_start, n_total, offset, sigma, shift):
    HW = 1.96 * float(sigma)
    matched_rows = []
    for (site, year), alert_t in alert_map.items():
        target = int(alert_t) + int(offset)
        info = row_map.get((str(site), int(year), int(target)))
        if info is None or info["ctype"] != 0:
            continue
        mu = float(info["mu"]) + doy_start - 1
        L = int(info["true_L"]) + doy_start - 1
        R = int(info["true_R"]) + doy_start - 1
        lead = float(L) - (mu + HW - float(shift))
        matched_rows.append({"mu": mu, "L": L, "R": R,
                             "PI_lo": mu - HW, "PI_hi": mu + HW, "lead": lead})
    df = pd.DataFrame(matched_rows)
    return overall_metrics(df, n_total)


def variable_offset_evaluate(label, run, stage1_ckpt, stage2_ckpt, n_total_test, device, args,
                             sigma=3.5, shift=30.0, target_lead=22.0,
                             offset_candidates=(60, 75, 90, 105, 120, 135),
                             alert_map_test=None, train_df=None, test_df=None,
                             weather_present=None, tau=None):
    """Returns single-row metrics dict for variable offset on test (n_total based)."""
    if train_df is None or test_df is None:
        # rebuild
        alert_map_test, _, train_df, test_df, weather_present, tau = \
            build_stage1_alert_and_features(Path(stage1_ckpt), run, args, want_test_features=True)

    print(f"\n--- VARIABLE OFFSET for {label} ---")
    row_map_train, doy_start = build_stage2_row_map_train(Path(stage2_ckpt), run, args, device)
    row_map_test, _, _ = build_stage2_row_map(Path(stage2_ckpt), run, args, device)

    train_df = attach_mu_at_default(train_df, row_map_train, doy_start)
    test_df = attach_mu_at_default(test_df, row_map_test, doy_start)

    feat_cols = (["score_at_tstar", "score_peak_30", "score_slope_14", "score_auc_30",
                  "alert_tstar", "lat", "lon", "year_feat", "mu_at_default_off"]
                 + [f"w_{n}" for n in weather_present])

    HW = 1.96 * sigma
    cohort_rows = []
    for _, r in train_df.iterrows():
        if r["alert_tstar"] is None or pd.isna(r["alert_tstar"]) or r["L"] is None or pd.isna(r["L"]):
            continue
        alert_t = int(r["alert_tstar"])
        leads = {}
        for off in offset_candidates:
            target = alert_t + int(off)
            info = row_map_train.get((str(r["site_id"]), int(r["year"]), int(target)))
            if info is None or info["ctype"] != 0:
                continue
            mu_abs = info["mu"] + doy_start - 1
            L_abs = info["true_L"] + doy_start - 1
            leads[int(off)] = float(L_abs) - (mu_abs + HW - shift)
        if not leads:
            continue
        best_off = min(leads.keys(), key=lambda o: abs(leads[o] - target_lead))
        rec = dict(r); rec["best_offset"] = int(best_off)
        cohort_rows.append(rec)
    cohort = pd.DataFrame(cohort_rows)
    cohort_f = cohort.dropna(subset=["best_offset"] + feat_cols).copy()
    if len(cohort_f) < 10:
        print(f"  [WARN] usable train rows = {len(cohort_f)} — too small")
        return None
    X = cohort_f[feat_cols].to_numpy(dtype=float)
    y = cohort_f["best_offset"].to_numpy(dtype=float)
    reg = GradientBoostingRegressor(n_estimators=200, max_depth=3, learning_rate=0.05,
                                    random_state=42).fit(X, y)
    print(f"  [GBM] n_train={len(cohort_f)}  R²={reg.score(X, y):.4f}")

    cand_min, cand_max = min(offset_candidates), max(offset_candidates)
    matched_rows = []
    for _, r in test_df.iterrows():
        if r["alert_tstar"] is None or pd.isna(r["alert_tstar"]):
            continue
        if r["L"] is None or pd.isna(r["L"]):
            continue
        if any(pd.isna(r.get(c)) for c in feat_cols):
            continue
        fv = np.asarray([r[c] for c in feat_cols], dtype=float).reshape(1, -1)
        pred_off = int(round(float(np.clip(reg.predict(fv)[0], cand_min, cand_max))))
        target = int(r["alert_tstar"]) + pred_off
        info = row_map_test.get((str(r["site_id"]), int(r["year"]), int(target)))
        if info is None or info["ctype"] != 0:
            continue
        mu_abs = info["mu"] + doy_start - 1
        L_abs = info["true_L"] + doy_start - 1
        R_abs = info["true_R"] + doy_start - 1
        lead = float(L_abs) - (mu_abs + HW - shift)
        matched_rows.append({"mu": mu_abs, "L": L_abs, "R": R_abs,
                             "PI_lo": mu_abs - HW, "PI_hi": mu_abs + HW, "lead": lead,
                             "pred_offset": pred_off})

    df_out = pd.DataFrame(matched_rows)
    m = overall_metrics(df_out, n_total_test)
    m["mode"] = "variable"
    m["model"] = label
    m["offset"] = "VAR"
    m["sigma"] = sigma
    m["shift"] = shift
    m["pred_offset_mean"] = float(df_out["pred_offset"].mean()) if not df_out.empty else float("nan")
    m["pred_offset_std"] = float(df_out["pred_offset"].std(ddof=0)) if not df_out.empty else float("nan")
    print(f"  [variable result] n_match={m['n_match']} P_ideal_overall={m['P_ideal_overall']:.2f}% "
          f"P_failed_overall={m['P_failed_overall']:.2f}% pred_off mean={m['pred_offset_mean']:.1f}")
    del row_map_train
    torch.cuda.empty_cache()
    return m


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, default="sheath_blight")
    p.add_argument("--run", type=int, default=4)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--models", type=str, required=True,
                   help="LABEL|CKPT;LABEL|CKPT;... (5 models)")
    p.add_argument("--variable_models", type=str, default="",
                   help="Subset of --models labels to also run Variable offset on, comma-separated.")
    p.add_argument("--offsets", type=str, default="30,60,90,105,120")
    p.add_argument("--sigmas", type=str, default="3.5,5.0")
    p.add_argument("--shifts", type=str, default="20,22,24,26,28,30,32,34,36,38,40")
    p.add_argument("--variable_sigma", type=float, default=3.5)
    p.add_argument("--variable_shift", type=float, default=30.0)
    p.add_argument("--variable_target_lead", type=float, default=22.0)
    p.add_argument("--variable_offset_candidates", type=str, default="60,75,90,105,120,135")
    p.add_argument("--out_csv", type=str, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[abort] CUDA required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    _ = resolve_pest(args.pest)

    offsets = [int(x) for x in str(args.offsets).split(",") if x.strip()]
    sigmas = [float(x) for x in str(args.sigmas).split(",") if x.strip()]
    shifts = [float(x) for x in str(args.shifts).split(",") if x.strip()]
    var_offsets = [int(x) for x in str(args.variable_offset_candidates).split(",") if x.strip()]
    var_labels = [s.strip() for s in str(args.variable_models).split(",") if s.strip()]
    print(f"[device] {device} ({torch.cuda.get_device_name(0)})  "
          f"free={torch.cuda.mem_get_info(0)[0]//1024**2} MB")
    print(f"[fixed grid] offsets={offsets}  sigmas={sigmas}  shifts={shifts}")
    print(f"[variable] models={var_labels}  σ={args.variable_sigma}  shift={args.variable_shift} "
          f"target_lead={args.variable_target_lead}  cands={var_offsets}")

    model_entries = []
    for part in str(args.models).split(";"):
        part = part.strip()
        if not part: continue
        if "|" not in part:
            raise SystemExit(f"Bad --models entry: {part!r}")
        label, ckpt = part.split("|", 1)
        model_entries.append((label.strip(), ckpt.strip()))
    print(f"[models] {len(model_entries)} ckpts:")
    for label, c in model_entries:
        print(f"  {label} <- {c}")

    print("\n[stage1] alert_map (test) + per-(site,year) features…")
    alert_map_test, n_total_test, train_df, test_df, weather_present, tau = \
        build_stage1_alert_and_features(Path(args.stage1_ckpt), args.run, args,
                                        want_test_features=bool(var_labels))

    # ============ FIXED GRID ============
    fixed_rows = []
    for label, ckpt_path in model_entries:
        print(f"\n----- FIXED forward: {label} -----")
        row_map, doy_start, _ = build_stage2_row_map(Path(ckpt_path), args.run, args, device)
        for offset in offsets:
            for sigma in sigmas:
                for shift in shifts:
                    m = evaluate_fixed_cell(alert_map_test, row_map, doy_start, n_total_test,
                                            offset=offset, sigma=sigma, shift=shift)
                    fixed_rows.append({"model": label, "mode": "fixed",
                                       "offset": int(offset), "sigma": float(sigma),
                                       "shift": float(shift), **m})
        del row_map
        torch.cuda.empty_cache()

    # ============ VARIABLE OFFSET ============
    var_rows = []
    if var_labels:
        for label, ckpt_path in model_entries:
            if label not in var_labels:
                continue
            m = variable_offset_evaluate(
                label, args.run, args.stage1_ckpt, ckpt_path, n_total_test, device, args,
                sigma=args.variable_sigma, shift=args.variable_shift,
                target_lead=args.variable_target_lead,
                offset_candidates=var_offsets,
                alert_map_test=alert_map_test, train_df=train_df.copy() if train_df is not None else None,
                test_df=test_df.copy() if test_df is not None else None,
                weather_present=weather_present, tau=tau,
            )
            if m is not None:
                var_rows.append(m)

    all_rows = fixed_rows + var_rows
    cols = ["model", "mode", "offset", "sigma", "shift", "n_match", "n_out_cohort",
            "P_ideal_overall", "P_useful_A_overall", "P_useful_B_overall",
            "P_missed_or_late_overall", "P_too_early_overall",
            "P_failed_overall", "P_ideal_conditional", "IoU_PI_LR"]
    df = pd.DataFrame(all_rows)
    for c in cols:
        if c not in df.columns:
            df[c] = float("nan")
    df = df[cols]

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\n[csv] saved {len(df)} rows → {args.out_csv}")

    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 600)

    print(f"\n=================== ALL ROWS ({len(df)}) ===================")
    print(df.to_string(index=False))

    print(f"\n=================== Per-model BEST P_ideal_overall ===================")
    summary = []
    for label, sub in df.groupby("model", sort=False):
        s = sub.dropna(subset=["P_ideal_overall"])
        if s.empty:
            continue
        i = s["P_ideal_overall"].idxmax()
        r = s.loc[i]
        print(f"\n[{label}]")
        print(f"  best P_ideal_overall : mode={r['mode']} offset={r['offset']} σ={r['sigma']} shift={r['shift']:.0f} | "
              f"P_ideal={r['P_ideal_overall']:.2f}%  P_useful_B={r['P_useful_B_overall']:.2f}%  "
              f"P_failed={r['P_failed_overall']:.2f}%  n_match={int(r['n_match'])}")
        summary.append({
            "model": label,
            "best_mode": r["mode"],
            "best_setting": f"off={r['offset']},σ={r['sigma']},sh={r['shift']:.0f}",
            "P_ideal_overall": float(r["P_ideal_overall"]),
            "P_useful_B_overall": float(r["P_useful_B_overall"]),
            "P_failed_overall": float(r["P_failed_overall"]),
            "P_ideal_conditional": float(r["P_ideal_conditional"]),
            "n_match": int(r["n_match"]),
        })

    print(f"\n=================== 5-way + variable summary ===================")
    print(pd.DataFrame(summary).to_string(index=False))


if __name__ == "__main__":
    main()
