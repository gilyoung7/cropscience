"""
Phase P — Variable offset v3 on the 2-sided D=15 model (inference + GBM fit only).

The 2-sided model's mu actually moves with t* (mu_std ~ 5.9; ~9-day mu shift
between offset=60 and offset=120). v1/v2 (on baseline) failed because baseline
mu was t*-insensitive. v3 leverages 2-sided's t* responsiveness.

Procedure (no Stage 2 retraining):

1) Stage 1 forward on train/val/test nowcast samples; calibrate; build per
   (site, year):
       4 score stats (score_at_tstar, peak_30, slope_14, auc_30)
       alert_tstar (first t* with score >= tau, F1 on val)
       lat / lon / year / mu_at_default
       weather snapshot at alert_tstar + 90 (rolling features from the season X)

2) Stage 2 forward on train + test; build
       row_map[(site, year, tstar_frame)] = mu (DOY-relative frame index)

3) Build train cohort (interval-only, alerted):
       for each sample, evaluate lead_i(offset) over offset_candidates
       lead_i = L_i - mu_i(offset) - 1.96*σ + shift
       best_offset_i = argmin_offset |lead_i - target_lead|     # default target=22

4) Fit GBM (n_estimators=200, max_depth=3, lr=0.05) on (features → best_offset)

5) Test apply:
       predicted_offset_i = round(reg.predict(features_i))  → clip to candidate range
       look up mu_i at alert_tstar_i + predicted_offset_i
       lead_i = L_i - mu_i - 1.96*σ + shift
       compute P_ideal (lead ∈ [14, 30]), P_useful_A/B, etc.

6) Report comparison vs fixed offset=120 baseline.

No retraining; CUDA required for Stage 2 forwards.
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


# Rolling weather features to pull at alert_tstar + 90 (D=15 run=4 ordering).
# Indices into feature_cols, not the (raw + miss) X column order;
# X column order is feature_cols then __miss columns, so the raw col index
# in X equals feature_cols.index(name) directly.
WEATHER_NAMES = [
    "rain_7d_sum", "tmean_7d_mean", "tmax_7d_max", "tmin_7d_min",
    "rh_7d_mean", "sun_7d_sum", "trange_7d_mean",
]


def _trapz(arr):
    f = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if f is None:
        return float(np.sum((arr[:-1] + arr[1:]) * 0.5)) if len(arr) >= 2 else float("nan")
    return float(f(arr))


def extract_group_features(scores: np.ndarray, tstars: np.ndarray):
    n = len(scores)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    marker = float(tstars[-1])
    score_at = float(scores[-1])
    last30 = scores[-min(30, n):]
    score_peak = float(np.max(last30))
    score_auc = _trapz(last30)
    last14 = scores[-min(14, n):]
    if len(last14) >= 2:
        slope = float(np.polyfit(np.arange(len(last14)), last14, 1)[0])
    else:
        slope = np.nan
    return score_at, score_peak, slope, score_auc, marker


def best_tau_by_f1(y: np.ndarray, p: np.ndarray) -> float:
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


def build_seas_meta(seas_list, feature_cols):
    lat_idx = feature_cols.index("좌표-위도") if "좌표-위도" in feature_cols else None
    lon_idx = feature_cols.index("좌표-경도") if "좌표-경도" in feature_cols else None
    weather_idx = [feature_cols.index(n) for n in WEATHER_NAMES if n in feature_cols]
    weather_present = [n for n in WEATHER_NAMES if n in feature_cols]
    out = {}
    for s in seas_list:
        key = (str(s["site_id"]), int(s["year"]))
        out[key] = {
            "X": np.asarray(s["X"]),
            "L": (int(s["L"]) if str(s["censor_type"]) != "right" else None),
            "R": (int(s["R"]) if str(s["censor_type"]) != "right" else None),
            "censor_type": str(s["censor_type"]),
        }
    return out, lat_idx, lon_idx, weather_idx, weather_present


def group_and_features(samples_now, scores, seas_meta, tau, with_alert=True):
    idx_map = defaultdict(list)
    for i, s in enumerate(samples_now):
        idx_map[(str(s["site_id"]), int(s["year"]))].append(i)
    rows = []
    for (site, year), idxs in idx_map.items():
        ts = np.asarray([int(samples_now[i]["tstar"]) for i in idxs])
        order = np.argsort(ts)
        ts_sorted = ts[order]
        sc_sorted = np.asarray([float(scores[i]) for i in idxs])[order]
        feat = extract_group_features(sc_sorted, ts_sorted)
        score_at, score_peak, slope, auc, marker = feat

        info = seas_meta.get((site, int(year)))
        if info is None:
            continue
        L_val = info["L"]
        R_val = info["R"]
        ctype = info["censor_type"]

        alert_t = None
        if with_alert:
            for i in idxs:
                if scores[i] >= float(tau):
                    ti = int(samples_now[i]["tstar"])
                    if alert_t is None or ti < alert_t:
                        alert_t = ti

        rows.append({
            "site_id": site, "year": int(year),
            "score_at_tstar": score_at, "score_peak_30": score_peak,
            "score_slope_14": slope, "score_auc_30": auc,
            "marker_tstar": marker,
            "L": L_val, "R": R_val, "censor_type": ctype,
            "alert_tstar": alert_t,
        })
    return pd.DataFrame(rows)


def attach_static_and_weather(df: pd.DataFrame, seas_meta: dict, lat_idx, lon_idx,
                              weather_idx, weather_present, weather_at_tstar_offset: int = 90,
                              T: int = 300):
    lats, lons, years = [], [], []
    weather_cols = {n: [] for n in weather_present}
    for _, r in df.iterrows():
        key = (r["site_id"], int(r["year"]))
        info = seas_meta.get(key)
        X = info["X"] if info is not None else None
        lat = float(X[0, lat_idx]) if (X is not None and lat_idx is not None) else float("nan")
        lon = float(X[0, lon_idx]) if (X is not None and lon_idx is not None) else float("nan")
        lats.append(lat)
        lons.append(lon)
        years.append(int(r["year"]))

        # weather snapshot at alert_tstar + offset (in DOY frame, 1-indexed → X is 0-indexed)
        target_frame = (int(r["alert_tstar"]) + int(weather_at_tstar_offset)
                        if r["alert_tstar"] is not None and not pd.isna(r["alert_tstar"])
                        else float("nan"))
        if X is None or pd.isna(target_frame) or not (1 <= target_frame <= T):
            for n in weather_present:
                weather_cols[n].append(float("nan"))
        else:
            row = X[int(target_frame) - 1]
            for n, idx in zip(weather_present, weather_idx):
                weather_cols[n].append(float(row[idx]))

    df = df.copy()
    df["lat"] = lats
    df["lon"] = lons
    df["year_feat"] = years
    for n in weather_present:
        df[f"w_{n}"] = weather_cols[n]
    return df


def stage1_pipeline(stage1_ckpt_path: Path, run: int, args):
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

    seas_train_meta, lat_idx, lon_idx, weather_idx, weather_present = build_seas_meta(train_seas, feature_cols)
    seas_test_meta, _, _, _, _ = build_seas_meta(test_seas, feature_cols)
    print(f"[features] weather_present = {weather_present}")
    print(f"[features] lat_idx={lat_idx} lon_idx={lon_idx} weather_idx={weather_idx}")

    train_df = group_and_features(train_s, p_train_cal, seas_train_meta, tau, with_alert=True)
    test_df = group_and_features(test_s, p_test_cal, seas_test_meta, tau, with_alert=True)
    train_df = attach_static_and_weather(train_df, seas_train_meta, lat_idx, lon_idx,
                                         weather_idx, weather_present)
    test_df = attach_static_and_weather(test_df, seas_test_meta, lat_idx, lon_idx,
                                        weather_idx, weather_present)

    print(f"[features] train groups={len(train_df)} (interval={train_df['L'].notna().sum()} "
          f"alerted={train_df['alert_tstar'].notna().sum()})  "
          f"test groups={len(test_df)} (interval={test_df['L'].notna().sum()} "
          f"alerted={test_df['alert_tstar'].notna().sum()})")
    return train_df, test_df, weather_present


def build_stage2_row_map_for_split(stage2_ckpt: Path, run: int, args, device, which_split: str):
    """which_split in {'train','test'}"""
    ckpt = torch.load(stage2_ckpt, map_location="cpu")
    C.DOY_START = int(ckpt.get("doy_start", C.DOY_START))
    C.DOY_END = int(ckpt.get("doy_end", C.DOY_END))
    doy_start = int(ckpt.get("doy_start", C.DOY_START))

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples2 = build_samples_for_run(run, get_feature_cols)
    train_s2_base, _, test_s2_base = split_samples(
        samples2, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )
    base = train_s2_base if which_split == "train" else test_s2_base
    nc_window = int(ckpt.get("stage2_nowcast_window", 28))
    nc_stride = int(ckpt.get("stage2_nowcast_stride", 1))
    nc_tstart = ckpt.get("stage2_nowcast_tstar_start", None)
    nc_only_pre = bool(int(ckpt.get("stage2_nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("stage2_nowcast_event_time_proxy", "r"))
    nc_req = bool(int(ckpt.get("stage2_nowcast_require_tstar_before_L", 0)))
    sample_list = build_stage2_nowcast_samples(
        base, window=nc_window, stride=nc_stride, tstar_start=nc_tstart,
        only_pre_event=nc_only_pre, event_time_proxy=nc_proxy, require_tstar_before_L=nc_req,
    )

    x_mean, x_std = compute_norm_stats(train_s2_base)
    groups = group_stage2_samples_by_site_year(sample_list)
    ds = GroupedIntervalEventDataset(groups, x_mean, x_std)
    loader = make_loader(ds, 1, shuffle=False, collate_fn=collate_grouped_stage2)

    d_model = int(ckpt.get("d_model", C.D_MODEL))
    n_head = int(ckpt.get("n_head", C.N_HEAD))
    n_layers = int(ckpt.get("n_layers", C.N_LAYERS))
    model = HierarchicalCausalHazardTransformer(
        d_in=int(sample_list[0]["X"].shape[1]),
        d_model=d_model, nhead=n_head, num_layers=n_layers,
        num_tstar_layers=int(ckpt.get("stage2_tstar_layers", 1)),
        dropout=C.DROPOUT, max_len=C.MAX_LEN, max_tstar_len=512,
        use_tstar_scalar_pos=bool(int(ckpt.get("stage2_use_tstar_scalar_pos", 0))),
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
    print(f"[stage2 fwd:{which_split}] rows={len(sample_list)}  d_model={d_model}")

    row_map: dict = {}
    gi = 0
    with torch.no_grad():
        for X, L, R, ctype, tstar, valid_mask in loader:
            X = X.to(device); tstar_t = tstar.to(device); v_t = valid_mask.to(device)
            _ = model(X, tstar=tstar_t, valid_mask=v_t)
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


def attach_mu_at_default(df: pd.DataFrame, row_map: dict, doy_start: int,
                          default_candidates=(90, 105, 120, 75, 60, 135)) -> pd.DataFrame:
    """
    For each row (alerted site-year), look up mu at alert_tstar + default_offset
    where default_offset is the FIRST candidate from `default_candidates` that
    yields an interval-typed row in row_map. Adds:
        df["mu_at_default_off"]  : mu in absolute DOY (or NaN)
        df["default_off"]        : which offset was used (-1 if none)
    Right-censored / no-alert / unmatched site-years get NaN / -1.
    """
    mu_vals, used_off = [], []
    n_no_alert = n_unmatched = n_ok = 0
    for _, r in df.iterrows():
        if r["alert_tstar"] is None or pd.isna(r["alert_tstar"]):
            mu_vals.append(float("nan")); used_off.append(-1); n_no_alert += 1; continue
        found = None
        for cand in default_candidates:
            target = int(r["alert_tstar"]) + int(cand)
            info = row_map.get((str(r["site_id"]), int(r["year"]), int(target)))
            if info is not None and info["ctype"] == 0:
                found = (info["mu"] + doy_start - 1, cand)
                break
        if found is None:
            mu_vals.append(float("nan")); used_off.append(-1); n_unmatched += 1
        else:
            mu_vals.append(found[0]); used_off.append(found[1]); n_ok += 1
    df = df.copy()
    df["mu_at_default_off"] = mu_vals
    df["default_off"] = used_off
    print(f"  [attach_mu_at_default] ok={n_ok}  no_alert={n_no_alert}  unmatched={n_unmatched}")
    return df


def build_train_cohort(train_df: pd.DataFrame, row_map_train: dict, doy_start: int,
                       offset_candidates, sigma: float, shift: float,
                       target_lead: float = 22.0):
    """
    train_df must already have `mu_at_default_off` (via attach_mu_at_default).
    Adds best_offset / best_lead / n_offset_cands per interval+alerted row.
    """
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
            lead = float(L_abs) - (mu_abs + HW - shift)
            leads[int(off)] = lead
        if not leads:
            continue
        best_off = min(leads.keys(), key=lambda o: abs(leads[o] - target_lead))
        rec = dict(r)
        rec["best_offset"] = int(best_off)
        rec["best_lead"] = float(leads[best_off])
        rec["n_offset_cands"] = len(leads)
        cohort_rows.append(rec)
    return pd.DataFrame(cohort_rows)


def apply_to_test(test_df: pd.DataFrame, row_map_test: dict, doy_start: int,
                  reg: GradientBoostingRegressor, feat_cols: list,
                  offset_candidates, sigma: float, shift: float):
    HW = 1.96 * sigma
    cand_min = min(offset_candidates)
    cand_max = max(offset_candidates)
    results = []
    n_no_features = 0
    n_no_alert = 0
    n_no_match = 0
    n_used = 0
    nan_feature_counts = {c: 0 for c in feat_cols}
    missing_cols_seen = set()
    for _, r in test_df.iterrows():
        if r["alert_tstar"] is None or pd.isna(r["alert_tstar"]):
            n_no_alert += 1
            continue
        feats = {}
        for c in feat_cols:
            if c not in r.index:
                missing_cols_seen.add(c)
                feats[c] = float("nan")
            else:
                feats[c] = r[c]
        feat_vec = np.asarray([feats[c] for c in feat_cols], dtype=float).reshape(1, -1)
        if not np.isfinite(feat_vec).all():
            # which feature(s) NaN?
            for i, c in enumerate(feat_cols):
                if not np.isfinite(feat_vec[0, i]):
                    nan_feature_counts[c] += 1
            n_no_features += 1
            continue
        # for "mu_at_default_off" feature, fill via the same default rule as train cohort
        pred_off = float(reg.predict(feat_vec)[0])
        pred_off = int(round(np.clip(pred_off, cand_min, cand_max)))

        alert_t = int(r["alert_tstar"])
        target = alert_t + pred_off
        info = row_map_test.get((str(r["site_id"]), int(r["year"]), int(target)))
        if info is None or info["ctype"] != 0:
            n_no_match += 1
            continue
        mu_abs = info["mu"] + doy_start - 1
        L_abs = info["true_L"] + doy_start - 1
        R_abs = info["true_R"] + doy_start - 1
        lead = float(L_abs) - (mu_abs + HW - shift)
        results.append({
            "site_id": r["site_id"], "year": int(r["year"]),
            "alert_tstar": alert_t, "pred_offset": pred_off,
            "mu_abs": mu_abs, "L_abs": L_abs, "R_abs": R_abs,
            "PI_lo": mu_abs - HW, "PI_hi": mu_abs + HW,
            "PI_op_hi": mu_abs + HW - shift, "lead": lead,
        })
        n_used += 1
    return pd.DataFrame(results), {
        "n_no_alert": n_no_alert,
        "n_no_features": n_no_features,
        "n_no_match": n_no_match,
        "n_used": n_used,
        "nan_feature_counts": {c: int(v) for c, v in nan_feature_counts.items() if v > 0},
        "missing_cols_in_test_df": sorted(missing_cols_seen),
    }


def bucket_metrics(df: pd.DataFrame, n_interval_total: int) -> dict:
    if df.empty:
        return {"n_match": 0, "P_useful_A": float("nan"), "P_useful_B": float("nan"),
                "P_ideal": float("nan"), "P_missed_or_late": float("nan"),
                "P_too_early": float("nan"),
                "lead_mean_UB": float("nan"), "lead_median_UB": float("nan"),
                "IoU_PI_LR": float("nan")}
    leads = df["lead"].to_numpy(dtype=float)
    missed = leads < 0
    too_late = (leads >= 0) & (leads < 7)
    urgent = (leads >= 7) & (leads < 14)
    ideal = (leads >= 14) & (leads < 30)
    advance = (leads >= 30) & (leads < 45)
    too_early = leads >= 45
    n = len(leads)
    pct = lambda mask: 100.0 * float(mask.sum()) / n
    P_useful_A = pct((leads >= 0) & (leads < 45))
    P_useful_B = pct((leads >= 7) & (leads < 45))
    P_ideal = pct(ideal)
    iou_pi_lr = []
    for _, r in df.iterrows():
        pL_pi = int(round(r["PI_lo"]))
        pR_pi = int(round(r["PI_hi"]))
        iou, _, _ = overlap_metrics(pL_pi, pR_pi, int(r["L_abs"]), int(r["R_abs"]))
        iou_pi_lr.append(iou)
    return {
        "n_match": n,
        "P_useful_A": P_useful_A,
        "P_useful_B": P_useful_B,
        "P_ideal": P_ideal,
        "P_ideal_given_useful_A": (P_ideal / P_useful_A) if P_useful_A > 0 else float("nan"),
        "P_ideal_given_useful_B": (P_ideal / P_useful_B) if P_useful_B > 0 else float("nan"),
        "P_missed_or_late": pct(missed) + pct(too_late),
        "P_too_early": pct(too_early),
        "lead_mean_UB": float(leads[(leads >= 7) & (leads < 45)].mean())
            if ((leads >= 7) & (leads < 45)).any() else float("nan"),
        "lead_median_UB": float(np.median(leads[(leads >= 7) & (leads < 45)]))
            if ((leads >= 7) & (leads < 45)).any() else float("nan"),
        "IoU_PI_LR": float(np.mean(iou_pi_lr)),
        "n_interval_total": int(n_interval_total),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, default="sheath_blight")
    p.add_argument("--run", type=int, default=4)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument("--sigma", type=float, default=3.5)
    p.add_argument("--shift", type=float, default=30.0)
    p.add_argument("--target_lead", type=float, default=22.0,
                   help="best_offset = argmin_off |lead(off) - target_lead|")
    p.add_argument("--offset_candidates", type=str, default="60,75,90,105,120,135")
    p.add_argument("--n_estimators", type=int, default=200)
    p.add_argument("--max_depth", type=int, default=3)
    p.add_argument("--learning_rate", type=float, default=0.05)
    p.add_argument("--out_csv", type=str, default=None)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[abort] CUDA required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    _ = resolve_pest(args.pest)
    print(f"[device] {device} ({torch.cuda.get_device_name(0)})  "
          f"free={torch.cuda.mem_get_info(0)[0]//1024**2} MB")
    print(f"[config] σ={args.sigma}  shift={args.shift}  target_lead={args.target_lead}")
    offsets = [int(x) for x in str(args.offset_candidates).split(",") if x.strip()]
    print(f"[config] offset_candidates={offsets}")

    print("\n[1/4] Stage 1 features + alerts (train + test)")
    train_df, test_df, weather_present = stage1_pipeline(Path(args.stage1_ckpt), args.run, args)

    print("\n[2/4] Stage 2 forward (train, then test)")
    row_map_train, doy_start_train = build_stage2_row_map_for_split(
        Path(args.stage2_ckpt), args.run, args, device, "train")
    row_map_test, doy_start_test = build_stage2_row_map_for_split(
        Path(args.stage2_ckpt), args.run, args, device, "test")
    assert doy_start_train == doy_start_test
    doy_start = int(doy_start_train)

    print("\n[2.5/4] attach mu_at_default_off to train_df + test_df")
    print("  train_df:")
    train_df = attach_mu_at_default(train_df, row_map_train, doy_start)
    print("  test_df:")
    test_df = attach_mu_at_default(test_df, row_map_test, doy_start)

    print("\n[3/4] Build train cohort (best_offset per sample) + GBM fit")
    cohort = build_train_cohort(train_df, row_map_train, doy_start,
                                offsets, args.sigma, args.shift,
                                target_lead=float(args.target_lead))
    print(f"[cohort] train rows = {len(cohort)}")
    if cohort.empty:
        raise SystemExit("Empty train cohort — check stage1/stage2 paths and split.")

    feat_cols = (["score_at_tstar", "score_peak_30", "score_slope_14", "score_auc_30",
                  "alert_tstar", "lat", "lon", "year_feat", "mu_at_default_off"]
                 + [f"w_{n}" for n in weather_present])
    # drop NaN in features/target
    cohort_f = cohort.dropna(subset=["best_offset"] + feat_cols).copy()
    print(f"[cohort:fit] usable rows = {len(cohort_f)}  (features={len(feat_cols)})")

    X = cohort_f[feat_cols].to_numpy(dtype=float)
    y = cohort_f["best_offset"].to_numpy(dtype=float)
    reg = GradientBoostingRegressor(
        n_estimators=int(args.n_estimators),
        max_depth=int(args.max_depth),
        learning_rate=float(args.learning_rate),
        random_state=42,
    ).fit(X, y)
    y_pred = reg.predict(X)
    r2 = reg.score(X, y)
    mae = float(np.mean(np.abs(y_pred - y)))
    print(f"[GBM] R²={r2:.4f}  MAE={mae:.2f}  intercept_used=False  n_estimators={args.n_estimators}")
    fi = sorted(zip(feat_cols, reg.feature_importances_), key=lambda x: -x[1])
    print("[GBM] feature importance:")
    for c, w in fi:
        print(f"  {c:<22} = {w:.4f}")

    print(f"\n[best_offset stats on train cohort]")
    print(f"  mean={float(cohort_f['best_offset'].mean()):.2f}  std={float(cohort_f['best_offset'].std(ddof=0)):.2f}  "
          f"min={int(cohort_f['best_offset'].min())}  max={int(cohort_f['best_offset'].max())}")
    bins = pd.cut(cohort_f["best_offset"], bins=[59,74,89,104,119,134,150],
                  labels=["60-74","75-89","90-104","105-119","120-134","135+"])
    print("  distribution:")
    print(bins.value_counts().sort_index().to_string())

    print("\n[4/4] Apply regressor to test (variable predicted_offset)")
    test_results, stats = apply_to_test(test_df, row_map_test, doy_start, reg, feat_cols,
                                        offsets, args.sigma, args.shift)
    print(f"[test] used={stats['n_used']}  no_alert={stats['n_no_alert']}  "
          f"no_features={stats['n_no_features']}  no_match={stats['n_no_match']}")
    if stats.get("missing_cols_in_test_df"):
        print(f"[test:diag] columns NOT present on test_df: {stats['missing_cols_in_test_df']}")
    if stats.get("nan_feature_counts"):
        print(f"[test:diag] NaN counts per feature (only features with NaNs shown):")
        for c, v in sorted(stats["nan_feature_counts"].items(), key=lambda kv: -kv[1]):
            print(f"    {c:<22} = {v}")
    if not test_results.empty:
        print(f"[test] pred_offset mean={test_results['pred_offset'].mean():.2f}  "
              f"std={test_results['pred_offset'].std(ddof=0):.2f}  "
              f"min={int(test_results['pred_offset'].min())} max={int(test_results['pred_offset'].max())}")
        po_bins = pd.cut(test_results["pred_offset"], bins=[59,74,89,104,119,134,150],
                         labels=["60-74","75-89","90-104","105-119","120-134","135+"])
        print("  test pred_offset distribution:")
        print(po_bins.value_counts().sort_index().to_string())

    n_interval_test = int(test_df["L"].notna().sum())
    m_var = bucket_metrics(test_results, n_interval_test)

    # Also compute fixed offset=120 reference on test (same σ/shift)
    HW = 1.96 * args.sigma
    fixed_rows = []
    for _, r in test_df.iterrows():
        if r["alert_tstar"] is None or pd.isna(r["alert_tstar"]) or r["L"] is None or pd.isna(r["L"]):
            continue
        target = int(r["alert_tstar"]) + 120
        info = row_map_test.get((str(r["site_id"]), int(r["year"]), int(target)))
        if info is None or info["ctype"] != 0:
            continue
        mu_abs = info["mu"] + doy_start - 1
        L_abs = info["true_L"] + doy_start - 1
        R_abs = info["true_R"] + doy_start - 1
        lead = float(L_abs) - (mu_abs + HW - args.shift)
        fixed_rows.append({
            "site_id": r["site_id"], "year": int(r["year"]),
            "L_abs": L_abs, "R_abs": R_abs,
            "PI_lo": mu_abs - HW, "PI_hi": mu_abs + HW,
            "PI_op_hi": mu_abs + HW - args.shift, "lead": lead,
        })
    fixed_df = pd.DataFrame(fixed_rows)
    m_fixed = bucket_metrics(fixed_df, n_interval_test)

    # Fair comparison: restrict variable cohort to the fixed=120 site-years.
    fixed_keys = set((str(r["site_id"]), int(r["year"]))
                     for _, r in fixed_df.iterrows())
    if not test_results.empty:
        mask_fc = test_results.apply(
            lambda r: (str(r["site_id"]), int(r["year"])) in fixed_keys, axis=1)
        var_in_fixed_cohort = test_results[mask_fc].copy()
    else:
        var_in_fixed_cohort = test_results.copy()
    m_var_fc = bucket_metrics(var_in_fixed_cohort, n_interval_test)

    # Also: variable cohort - fixed cohort (samples variable picked up but fixed=120 missed)
    if not test_results.empty:
        var_outside = test_results[~mask_fc].copy()
    else:
        var_outside = test_results.copy()
    m_var_outside = bucket_metrics(var_outside, n_interval_test)

    print("\n=================== Variable vs Fixed offset comparison ===================")
    rows = [
        {"Setup": f"fixed offset=120 σ={args.sigma} shift={args.shift}", **m_fixed},
        {"Setup": f"variable (all alerted) σ={args.sigma} shift={args.shift}", **m_var},
        {"Setup": f"variable ∩ fixed=120 cohort (fair, n_keys={len(fixed_keys)})", **m_var_fc},
        {"Setup": f"variable \\ fixed=120 (extra samples only)", **m_var_outside},
    ]
    df_out = pd.DataFrame(rows)
    cols = ["Setup", "n_match", "P_useful_A", "P_useful_B", "P_ideal",
            "P_ideal_given_useful_A", "P_ideal_given_useful_B",
            "P_missed_or_late", "P_too_early",
            "lead_mean_UB", "lead_median_UB", "IoU_PI_LR"]
    df_out = df_out[cols]
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    pd.set_option("display.width", 240)
    pd.set_option("display.max_columns", 30)
    print(df_out.to_string(index=False))

    # Side-analysis: predicted_offset distribution split by in/out fixed_cohort
    if not test_results.empty:
        in_fc = test_results[mask_fc]
        out_fc = test_results[~mask_fc]
        print("\n--- predicted_offset distribution by cohort ---")
        print(f"  in  fixed=120 cohort (n={len(in_fc)}): "
              f"mean={in_fc['pred_offset'].mean():.1f}  std={in_fc['pred_offset'].std(ddof=0):.1f}  "
              f"min={int(in_fc['pred_offset'].min())}  max={int(in_fc['pred_offset'].max())}")
        if len(out_fc) > 0:
            print(f"  out fixed=120 cohort (n={len(out_fc)}): "
                  f"mean={out_fc['pred_offset'].mean():.1f}  std={out_fc['pred_offset'].std(ddof=0):.1f}  "
                  f"min={int(out_fc['pred_offset'].min())}  max={int(out_fc['pred_offset'].max())}")

    if args.out_csv and not test_results.empty:
        test_results.to_csv(args.out_csv, index=False)
        print(f"\n[csv] per-sample test predictions saved to {args.out_csv}  rows={len(test_results)}")


if __name__ == "__main__":
    main()
