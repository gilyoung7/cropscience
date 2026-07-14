"""
Phase I — Variable offset via Stage 1 score → predicted_L linear regression.

Procedure (no retraining):
  1. Stage 1 forward on train/val/test nowcast samples (XGBoost predict_proba).
  2. Per (site, year) group, extract 4 features from the calibrated score series:
         score_at_tstar, score_peak_30, score_slope_14, score_auc_30
     (marker t* = group's max tstar).
  3. Fit linear regression on train interval rows: 4 features → L.
  4. Apply regressor to test, then for each alerted site-year:
         offset_i = clip(pred_L_i - alert_tstar_i - SAFETY, OFFSET_MIN, OFFSET_MAX)
  5. Stage 2 forward → row_map (site, year, tstar_frame) → mu.
  6. Look up mu at (alert_tstar_i + offset_i), PI = [mu - HW, mu + HW].
  7. Compute matched-cohort metrics: IoU80, EarlyRecall80, Precision, Recall, F1.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression

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
from rice.src.train_eval import early_recall80_site_year, overlap_metrics
from rice.scripts.common import collate_grouped_stage2, make_loader
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import build_nowcast_samples, build_tabular_from_samples
from rice.scripts.run_calibrate_event import apply_temperature, fit_temperature_grid


def _trapz(arr: np.ndarray) -> float:
    f = getattr(np, "trapezoid", None) or getattr(np, "trapz", None)
    if f is None:
        return float(np.sum((arr[:-1] + arr[1:]) * 0.5)) if len(arr) >= 2 else float("nan")
    return float(f(arr))


def extract_group_features(scores: np.ndarray, tstars: np.ndarray):
    n = len(scores)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, np.nan
    marker_tstar = float(tstars[-1])
    score_at = float(scores[-1])
    last30 = scores[-min(30, n):]
    score_peak = float(np.max(last30))
    score_auc = _trapz(last30)
    last14 = scores[-min(14, n):]
    if len(last14) >= 2:
        slope = float(np.polyfit(np.arange(len(last14)), last14, 1)[0])
    else:
        slope = np.nan
    return score_at, score_peak, slope, score_auc, marker_tstar


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


def group_and_features(samples_now: list[dict], scores: np.ndarray,
                       with_alert: bool = False, tau: float | None = None) -> pd.DataFrame:
    idx_map: dict[tuple[str, int], list[int]] = defaultdict(list)
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
        L_val = None
        for i in idxs:
            Li = samples_now[i].get("L")
            if Li is not None:
                L_val = int(Li)
                break
        row = {
            "site_id": site, "year": int(year),
            "score_at_tstar": score_at, "score_peak_30": score_peak,
            "score_slope_14": slope, "score_auc_30": auc,
            "marker_tstar": marker, "L": L_val,
        }
        if with_alert:
            alert_t = None
            for j, i in enumerate(idxs):
                pass
            for i in idxs:
                if scores[i] >= float(tau):
                    ti = int(samples_now[i]["tstar"])
                    if alert_t is None or ti < alert_t:
                        alert_t = ti
            row["alert_tstar"] = alert_t
        rows.append(row)
    return pd.DataFrame(rows)


def stage1_pipeline(stage1_ckpt_path: Path, run: int, args) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    ckpt = torch.load(stage1_ckpt_path, map_location="cpu")
    add_tpos = bool(ckpt.get("add_tstar_position_feature", False))
    nc_window = int(ckpt.get("nowcast_window", 28))
    nc_stride = int(ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("nowcast_event_time_proxy", "r"))

    _, get_feature_cols = resolve_pest(args.pest)
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
    X_train = build_tabular_from_samples(train_s, add_tstar_position_feature=add_tpos)
    X_val = build_tabular_from_samples(val_s, add_tstar_position_feature=add_tpos)
    X_test = build_tabular_from_samples(test_s, add_tstar_position_feature=add_tpos)

    clf = ckpt["trained_states"][0]["sk_model"]
    p_train = clf.predict_proba(X_train)[:, 1]
    p_val_raw = clf.predict_proba(X_val)[:, 1]
    p_test_raw = clf.predict_proba(X_test)[:, 1]
    t_best, _ = fit_temperature_grid(y_val, p_val_raw)
    p_train_cal = apply_temperature(p_train, t_best)
    p_val_cal = apply_temperature(p_val_raw, t_best)
    p_test_cal = apply_temperature(p_test_raw, t_best)
    tau = best_tau_by_f1(y_val, p_val_cal)
    print(f"[stage1] T*={t_best:.3f}  tau (F1 on val) = {tau:.3f}")

    train_df = group_and_features(train_s, p_train_cal)
    test_df = group_and_features(test_s, p_test_cal, with_alert=True, tau=tau)
    print(f"[features] train groups={len(train_df)} (interval={train_df['L'].notna().sum()})  "
          f"test groups={len(test_df)} (interval={test_df['L'].notna().sum()}, "
          f"alerted={test_df['alert_tstar'].notna().sum()})")
    return train_df, test_df, tau


def fit_regressor(train_df: pd.DataFrame) -> LinearRegression:
    feat_cols = ["score_at_tstar", "score_peak_30", "score_slope_14", "score_auc_30"]
    df = train_df.dropna(subset=["L"] + feat_cols).copy()
    X = df[feat_cols].to_numpy()
    y = df["L"].to_numpy(dtype=float)
    reg = LinearRegression().fit(X, y)
    y_pred = reg.predict(X)
    r2 = reg.score(X, y)
    mae = float(np.mean(np.abs(y_pred - y)))
    print(f"[regressor] n_train={len(df)}  R^2={r2:.4f}  MAE={mae:.2f} days  "
          f"intercept={reg.intercept_:.2f}")
    for c, w in zip(feat_cols, reg.coef_):
        print(f"  coef[{c:<16}] = {w:+.3f}")
    return reg


def stage2_row_map(stage2_ckpt_path: Path, run: int, args, device: torch.device) -> tuple[dict, int, int]:
    ckpt = torch.load(stage2_ckpt_path, map_location="cpu")
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
    ).to(device)
    model.time_chunk_size = int(ckpt.get("stage2_time_chunk_size", 64))
    model.conditional_survival = bool(int(ckpt.get("stage2_conditional_survival", 0)))
    model.pmf_mode = "gaussian"
    model.gaussian_sigma = float(ckpt.get("stage2_pmf_sigma", 5.0))
    model.gaussian_mu_max = float(ckpt.get("stage2_pmf_mu_max", 0.0))
    model.asym_weight = float(ckpt.get("stage2_pmf_asym_weight", 15.0))
    model.right_weight = float(ckpt.get("stage2_pmf_right_weight", 0.3))
    model.target_offset = float(ckpt.get("stage2_pmf_target_offset", 5.0))
    model.load_state_dict(ckpt["trained_states"][0]["state_dict"], strict=False)
    model.eval()
    print(f"[stage2 model] d_model={d_model} n_head={n_head} n_layers={n_layers}  test_rows={len(test_s2)}")

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

    n_interval = sum(1 for s in test_s2_base if str(s["censor_type"]) != "right")
    return row_map, doy_start, n_interval


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, default="sheath_blight")
    p.add_argument("--run", type=int, required=True,
                   help="run id for both Stage 1 ckpt and Stage 2 ckpt")
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    p.add_argument("--safety", type=float, default=10.0,
                   help="days before predicted_L to anchor mu (so mu predicts ~10 days early)")
    p.add_argument("--offset_min", type=int, default=30)
    p.add_argument("--offset_max", type=int, default=120)
    p.add_argument("--pi_halfwidth", type=float, default=10.0)
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[abort] CUDA required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    _ = resolve_pest(args.pest)
    print(f"[device] {device} ({torch.cuda.get_device_name(0)})  "
          f"free={torch.cuda.mem_get_info(0)[0]//1024**2} MB")
    print(f"[config] safety={args.safety}  offset_clip=[{args.offset_min},{args.offset_max}]  PI=±{args.pi_halfwidth}")

    print("\n[1/4] Stage 1 forward + features per site-year + alert_tstar")
    train_df, test_df, tau = stage1_pipeline(Path(args.stage1_ckpt), args.run, args)

    print("\n[2/4] Fit linear regression (4 features → L) on train interval rows")
    reg = fit_regressor(train_df)

    print("\n[3/4] Apply regressor to test, compute variable offset_i per site-year")
    feat_cols = ["score_at_tstar", "score_peak_30", "score_slope_14", "score_auc_30"]
    mask_t = test_df[feat_cols].notna().all(axis=1) & test_df["alert_tstar"].notna()
    test_use = test_df[mask_t].copy()
    test_use["pred_L"] = reg.predict(test_use[feat_cols].to_numpy())
    test_use["raw_offset"] = test_use["pred_L"] - test_use["alert_tstar"] - float(args.safety)
    test_use["offset"] = (test_use["raw_offset"]
                          .clip(lower=int(args.offset_min), upper=int(args.offset_max))
                          .round().astype(int))
    n_clip_lo = int((test_use["raw_offset"] < int(args.offset_min)).sum())
    n_clip_hi = int((test_use["raw_offset"] > int(args.offset_max)).sum())
    print(f"[apply] alerted+features rows = {len(test_use)}  "
          f"clipped_low={n_clip_lo}  clipped_high={n_clip_hi}")
    print(f"[offset] mean={test_use['offset'].mean():.1f}  median={int(test_use['offset'].median())}  "
          f"min={int(test_use['offset'].min())}  max={int(test_use['offset'].max())}  "
          f"std={test_use['offset'].std():.1f}")
    print(f"[pred_L] mean={test_use['pred_L'].mean():.1f}  std={test_use['pred_L'].std():.1f}")

    print("\n[4/4] Stage 2 forward → row_map → evaluate at variable offsets")
    row_map, doy_start, n_interval = stage2_row_map(Path(args.stage2_ckpt), args.run, args, device)

    HW = float(args.pi_halfwidth)
    Tend = int(C.DOY_END - C.DOY_START + 1)
    matched = []
    offset_missed = 0
    after_R = 0
    after_Tend = 0
    not_interval = 0
    for _, r in test_use.iterrows():
        alert_t = int(r["alert_tstar"])
        offset = int(r["offset"])
        target = alert_t + offset
        key = (str(r["site_id"]), int(r["year"]), int(target))
        info = row_map.get(key)
        if info is None:
            offset_missed += 1
            if target > Tend:
                after_Tend += 1
            else:
                after_R += 1
            continue
        if info["ctype"] != 0:
            not_interval += 1
            continue
        mu = float(info["mu"]) + doy_start - 1
        true_L = int(info["true_L"]) + doy_start - 1
        true_R = int(info["true_R"]) + doy_start - 1
        alert_abs = alert_t + doy_start - 1
        stage2_abs = target + doy_start - 1
        matched.append({
            "sample_id": f"{r['site_id']}-{int(r['year'])}",
            "tstar": stage2_abs, "alert_tstar_abs": alert_abs, "stage2_tstar_abs": stage2_abs,
            "mu": mu,
            "pred_L": int(round(mu - HW)), "pred_R": int(round(mu + HW)),
            "true_L": true_L, "true_R": true_R, "offset": offset,
        })
    print(f"[match] matched={len(matched)}  not_interval={not_interval}  "
          f"offset_missed={offset_missed} (after_R={after_R}, after_Tend={after_Tend})")

    if not matched:
        print("\n[abort] no matched rows; check ckpt paths and offset clip")
        return

    ious = []
    tp = 0
    for m in matched:
        iou, _, _ = overlap_metrics(m["pred_L"], m["pred_R"], m["true_L"], m["true_R"])
        ious.append(iou)
        hit = (min(m["pred_R"], m["true_R"]) - max(m["pred_L"], m["true_L"])) > 0
        if hit:
            tp += 1
    precision = tp / len(matched)
    recall = tp / max(n_interval, 1)
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    er80, _, _ = early_recall80_site_year(matched)
    mu_arr = np.asarray([m["mu"] for m in matched])
    L_arr = np.asarray([m["true_L"] for m in matched])

    print("\n=================== VARIABLE OFFSET RESULT ===================")
    print(f"  n_match              = {len(matched)}")
    print(f"  offset mean/std      = {test_use['offset'].mean():.1f} / {test_use['offset'].std():.1f}")
    print(f"  offset min/median/max = {int(test_use['offset'].min())} / "
          f"{int(test_use['offset'].median())} / {int(test_use['offset'].max())}")
    print(f"  mu_mean / mu_std     = {mu_arr.mean():.2f} / {mu_arr.std(ddof=0):.2f}")
    print(f"  L_mean  / L_std      = {L_arr.mean():.2f} / {L_arr.std(ddof=0):.2f}")
    print(f"  mean(mu - L)         = {(mu_arr - L_arr).mean():+.2f}")
    print(f"  mu_std / L_std       = {mu_arr.std(ddof=0) / max(L_arr.std(ddof=0), 1e-9):.3f}")
    print(f"  IoU80                = {np.mean(ious):.4f}")
    print(f"  EarlyRecall80        = {er80:.4f}")
    print(f"  Precision / Recall   = {precision:.4f} / {recall:.4f}")
    print(f"  F1                   = {f1:.4f}")

    print("\n--- comparison reference (D=15, phase_h fixed offsets) ---")
    print(f"  fixed offset=60  : F1=0.540  IoU80=0.257  EarlyRecall=0.621")
    print(f"  fixed offset=105 : F1=0.628  IoU80=0.324  EarlyRecall=0.750   ← current best F1")
    print(f"  fixed offset=120 : F1=0.589  IoU80=0.420  EarlyRecall=0.438   ← best IoU but n drops")


if __name__ == "__main__":
    main()
