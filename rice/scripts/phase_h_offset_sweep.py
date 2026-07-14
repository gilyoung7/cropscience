"""
Phase H — Stage 2 offset sweep on already-trained ckpts (no retraining).

For each (stage1_ckpt, stage2_ckpt) pair (baseline D=15 + new D=22), run a
single forward pass per stage to build:
  alert_map:  (site, year) -> first t* where stage1 score >= tau
  row_map:    (site, year, tstar_frame) -> {mu, true_L, true_R, ctype}

Then sweep stage2_tstar_offset ∈ {30,45,60,75,90,105,120}, match
target_tstar = alert_tstar + offset against row_map, and report metrics on the
matched interval cohort. PI = [mu - HW, mu + HW] (default HW=10, ~95% PI for
sigma=5).

The goal: see whether the +54 day mu shift in D=22 is in fact correct for a
larger offset, i.e., the D=22 model targets a later DOY than the D=15 baseline
because alert_tstar+60 is no longer the right anchor.
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
from rice.src.train_eval import early_recall80_site_year, overlap_metrics
from rice.scripts.common import collate_grouped_stage2, make_loader
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import build_nowcast_samples, build_tabular_from_samples
from rice.scripts.run_calibrate_event import apply_temperature, fit_temperature_grid


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


def build_stage1_alert_map(stage1_ckpt_path: Path, run: int, args) -> tuple[dict, int, int]:
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

    alert = {}
    for s, p_cal in zip(test_s, p_test_cal):
        if p_cal < tau:
            continue
        key = (str(s["site_id"]), int(s["year"]))
        prev = alert.get(key)
        if prev is None or int(s["tstar"]) < prev:
            alert[key] = int(s["tstar"])

    n_total = len(test_seas)
    n_interval = sum(1 for s in test_seas if str(s["censor_type"]) != "right")
    print(f"  [stage1] T*={t_best:.3f}  tau={tau:.3f}  alert_map={len(alert)}  "
          f"(test groups: total={n_total}, interval={n_interval})")
    return alert, n_interval, n_total


def build_stage2_row_map(stage2_ckpt_path: Path, run: int, args, device: torch.device) -> tuple[dict, int]:
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
    print(f"  [stage2 model] d_model={d_model} n_head={n_head} n_layers={n_layers}  "
          f"nowcast window={nc_window} stride={nc_stride} test_rows={len(test_s2)}")

    row_map = {}
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
    return row_map, doy_start


def sweep_one_model(label, run, stage1_ckpt, stage2_ckpt, offsets, HW, args, device) -> pd.DataFrame:
    print(f"\n----- {label}  (run={run}) -----")
    print(f"  stage1_ckpt = {stage1_ckpt}")
    print(f"  stage2_ckpt = {stage2_ckpt}")
    alert_map, n_interval, n_total = build_stage1_alert_map(Path(stage1_ckpt), run, args)
    row_map, doy_start = build_stage2_row_map(Path(stage2_ckpt), run, args, device)
    Tend = int(C.DOY_END - C.DOY_START + 1)

    rows = []
    for offset in offsets:
        matched = []
        offset_missed = 0
        for (site, year), alert_t in alert_map.items():
            target = int(alert_t) + int(offset)
            info = row_map.get((str(site), int(year), int(target)))
            if info is None or info["ctype"] != 0:
                offset_missed += 1
                continue
            mu = float(info["mu"])
            mu_abs = mu + doy_start - 1
            alert_abs = int(alert_t) + doy_start - 1
            stage2_abs = int(target) + doy_start - 1
            true_L_abs = int(info["true_L"]) + doy_start - 1
            true_R_abs = int(info["true_R"]) + doy_start - 1
            pred_L = int(round(mu_abs - HW))
            pred_R = int(round(mu_abs + HW))
            matched.append({
                "sample_id": f"{site}-{int(year)}",
                "tstar": int(stage2_abs),
                "alert_tstar_abs": alert_abs,
                "stage2_tstar_abs": stage2_abs,
                "mu": mu_abs,
                "pred_L": pred_L, "pred_R": pred_R, "pred_point": int(round(mu_abs)),
                "true_L": true_L_abs, "true_R": true_R_abs,
            })

        if not matched:
            rows.append({
                "model": label, "offset": offset, "n_match": 0, "n_offset_missed": offset_missed,
                "mu_mean": float("nan"), "mu_std": float("nan"),
                "L_mean": float("nan"), "L_std": float("nan"),
                "mean_mu_minus_L": float("nan"), "mu_std_over_L_std": float("nan"),
                "IoU80": float("nan"), "EarlyRecall80": float("nan"),
                "precision": float("nan"), "recall": float("nan"), "f1": float("nan"),
            })
            continue

        ious = []
        tp = 0
        for r in matched:
            iou, _, _ = overlap_metrics(r["pred_L"], r["pred_R"], r["true_L"], r["true_R"])
            ious.append(iou)
            hit = (min(r["pred_R"], r["true_R"]) - max(r["pred_L"], r["true_L"])) > 0
            if hit:
                tp += 1
        precision = tp / len(matched)
        recall = tp / max(n_interval, 1)
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        er80, _, _ = early_recall80_site_year(matched)
        mu_arr = np.asarray([r["mu"] for r in matched], dtype=float)
        L_arr = np.asarray([r["true_L"] for r in matched], dtype=float)
        rows.append({
            "model": label, "offset": int(offset),
            "n_match": len(matched), "n_offset_missed": int(offset_missed),
            "mu_mean": float(mu_arr.mean()), "mu_std": float(mu_arr.std(ddof=0)),
            "L_mean": float(L_arr.mean()), "L_std": float(L_arr.std(ddof=0)),
            "mean_mu_minus_L": float((mu_arr - L_arr).mean()),
            "mu_std_over_L_std": float(mu_arr.std(ddof=0) / max(L_arr.std(ddof=0), 1e-9)),
            "IoU80": float(np.mean(ious)),
            "EarlyRecall80": float(er80),
            "precision": float(precision), "recall": float(recall), "f1": float(f1),
        })
    return pd.DataFrame(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, default="sheath_blight")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    p.add_argument("--offsets", type=str, default="30,45,60,75,90,105,120")
    p.add_argument("--pi_halfwidth", type=float, default=10.0)
    p.add_argument("--baseline_run", type=int, default=4)
    p.add_argument("--baseline_label", type=str, default="D=15 (baseline)")
    p.add_argument("--baseline_stage1_ckpt", type=str, required=True)
    p.add_argument("--baseline_stage2_ckpt", type=str, required=True)
    p.add_argument("--new_run", type=int, default=6)
    p.add_argument("--new_label", type=str, default="D=22 (new)")
    p.add_argument("--new_stage1_ckpt", type=str, required=True)
    p.add_argument("--new_stage2_ckpt", type=str, required=True)
    p.add_argument("--mid_run", type=int, default=None,
                   help="optional third model run id (e.g. 7 for D=18)")
    p.add_argument("--mid_label", type=str, default="D=18 (mid)")
    p.add_argument("--mid_stage1_ckpt", type=str, default=None)
    p.add_argument("--mid_stage2_ckpt", type=str, default=None)
    args = p.parse_args()

    _ = resolve_pest(args.pest)
    offsets = [int(x) for x in str(args.offsets).split(",") if x.strip()]
    HW = float(args.pi_halfwidth)
    if not torch.cuda.is_available():
        raise SystemExit("[abort] CUDA not available — this script requires GPU. "
                         "Check `nvidia-smi` and PyTorch CUDA build.")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    print(f"[config] offsets={offsets}  PI half-width=±{HW}")
    print(f"[device] {device}  ({torch.cuda.get_device_name(0)})  "
          f"free={torch.cuda.mem_get_info(0)[0] // 1024**2} MB")

    df_b = sweep_one_model(args.baseline_label, args.baseline_run,
                           args.baseline_stage1_ckpt, args.baseline_stage2_ckpt,
                           offsets, HW, args, device)
    df_m = None
    if args.mid_run is not None and args.mid_stage1_ckpt and args.mid_stage2_ckpt:
        df_m = sweep_one_model(args.mid_label, int(args.mid_run),
                               args.mid_stage1_ckpt, args.mid_stage2_ckpt,
                               offsets, HW, args, device)
    df_n = sweep_one_model(args.new_label, args.new_run,
                           args.new_stage1_ckpt, args.new_stage2_ckpt,
                           offsets, HW, args, device)

    parts = [df_b] + ([df_m] if df_m is not None else []) + [df_n]
    df = pd.concat(parts, ignore_index=True)
    cols = ["model", "offset", "n_match", "n_offset_missed",
            "mu_mean", "mean_mu_minus_L", "mu_std", "mu_std_over_L_std",
            "IoU80", "EarlyRecall80", "precision", "recall", "f1"]
    df = df[cols]
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print("\n=================== OFFSET SWEEP ===================")
    print(df.to_string(index=False))

    # Highlight best mean_mu_minus_L for each model
    print("\n--- best |mean(mu-L)| per model ---")
    for label, sub in df.groupby("model", sort=False):
        sub_ok = sub.dropna(subset=["mean_mu_minus_L"])
        if sub_ok.empty:
            continue
        idx_min_bias = sub_ok["mean_mu_minus_L"].abs().idxmin()
        idx_max_iou = sub_ok["IoU80"].idxmax()
        r1 = sub_ok.loc[idx_min_bias]
        r2 = sub_ok.loc[idx_max_iou]
        print(f"  [{label}]  bias-min @ offset={int(r1['offset'])}: "
              f"mean(mu-L)={r1['mean_mu_minus_L']:+.2f}  IoU80={r1['IoU80']:.3f}")
        print(f"  [{label}]  IoU-max  @ offset={int(r2['offset'])}: "
              f"mean(mu-L)={r2['mean_mu_minus_L']:+.2f}  IoU80={r2['IoU80']:.3f}")


if __name__ == "__main__":
    main()
