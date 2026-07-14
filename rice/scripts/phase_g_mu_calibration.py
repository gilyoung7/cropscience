"""
Phase G — Global mu calibration ablation (inference only).

Given the trained D=22 final_aw15 Stage 2 ckpt, run inference, build matched
rows (Stage 1 alert + offset=60), then apply a constant shift to mu:
    mu_corrected = mu - DELTA  (default DELTA=68)
    PI_corrected = [mu_corr - HW, mu_corr + HW]  (HW default 10, ~95% PI for sigma=5)

Compare gating metrics before/after correction against the baseline D=15 ref.
No retraining.
"""

from __future__ import annotations

import argparse
import sys
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


def stage1_alert_map(stage1_ckpt_path: Path, args) -> tuple[dict[tuple[str, int], int], int]:
    ckpt = torch.load(stage1_ckpt_path, map_location="cpu")
    add_tstar_position_feature = bool(ckpt.get("add_tstar_position_feature", False))
    nc_window = int(ckpt.get("nowcast_window", 28))
    nc_stride = int(ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("nowcast_event_time_proxy", "r"))

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples = build_samples_for_run(args.run, get_feature_cols)
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
    X_val = build_tabular_from_samples(val_s, add_tstar_position_feature=add_tstar_position_feature)
    X_test = build_tabular_from_samples(test_s, add_tstar_position_feature=add_tstar_position_feature)

    clf = ckpt["trained_states"][0]["sk_model"]
    p_val_raw = clf.predict_proba(X_val)[:, 1]
    p_test_raw = clf.predict_proba(X_test)[:, 1]
    t_best, _ = fit_temperature_grid(y_val, p_val_raw)
    p_val_cal = apply_temperature(p_val_raw, t_best)
    p_test_cal = apply_temperature(p_test_raw, t_best)
    tau = best_tau_by_f1(y_val, p_val_cal)
    print(f"[stage1] T*={t_best:.3f}  tau (F1 on val) = {tau:.3f}")

    alert_first = {}
    for s, p_cal in zip(test_s, p_test_cal):
        if p_cal < tau:
            continue
        key = (str(s["site_id"]), int(s["year"]))
        prev = alert_first.get(key)
        if prev is None or int(s["tstar"]) < prev:
            alert_first[key] = int(s["tstar"])
    n_interval = sum(1 for s in test_seas if str(s["censor_type"]) != "right")
    return alert_first, n_interval


def build_stage2_model(ckpt: dict, d_in: int, device: torch.device):
    d_model = int(ckpt.get("d_model", C.D_MODEL))
    n_head = int(ckpt.get("n_head", C.N_HEAD))
    n_layers = int(ckpt.get("n_layers", C.N_LAYERS))
    model = HierarchicalCausalHazardTransformer(
        d_in=d_in,
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
    d2 = ckpt["trained_states"][0]
    model.load_state_dict(d2["state_dict"], strict=False)
    model.eval()
    return model, d_model, n_head, n_layers


@torch.no_grad()
def stage2_mu_per_tstar(model, loader, groups: list[dict], device: torch.device) -> dict:
    """
    Build map: (site_id, year, tstar_frame) -> {"mu": float, "true_L": int, "true_R": int, "ctype": int}
    where tstar_frame is 1-based frame index (same as stored in samples).
    """
    out = {}
    gi = 0
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
                out[key] = {
                    "mu": float(mu_np[bi, ki]),
                    "true_L": int(L_np[bi, ki]),
                    "true_R": int(R_np[bi, ki]),
                    "ctype": int(c_np[bi, ki]),
                }
        gi += B
    return out


def compute_metrics(matched_rows: list[dict], n_true: int) -> dict:
    if not matched_rows:
        return {"IoU80": float("nan"), "EarlyRecall80": float("nan"),
                "precision": float("nan"), "recall": float("nan"),
                "f1": float("nan"), "pred_pos": 0, "tp": 0,
                "lead_mean": float("nan"), "stage2_lead_mean": float("nan")}
    tp = 0
    ious = []
    for r in matched_rows:
        iou, _, _ = overlap_metrics(int(r["pred_L"]), int(r["pred_R"]),
                                    int(r["true_L"]), int(r["true_R"]))
        ious.append(iou)
        hit = (min(int(r["pred_R"]), int(r["true_R"])) -
               max(int(r["pred_L"]), int(r["true_L"]))) > 0
        if hit:
            tp += 1
    pred_pos = len(matched_rows)
    precision = tp / pred_pos if pred_pos else 0.0
    recall = tp / n_true if n_true else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
    er80, n_succ, denom = early_recall80_site_year(matched_rows)
    leads = [int(r["true_L"]) + 1 - int(r["alert_tstar_abs"]) for r in matched_rows]
    s2_leads = [int(r["true_L"]) + 1 - int(r["stage2_tstar_abs"]) for r in matched_rows]
    return {
        "IoU80": float(np.mean(ious)),
        "EarlyRecall80": float(er80),
        "EarlyRecall80_n": (int(n_succ), int(denom)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "pred_pos": int(pred_pos),
        "tp": int(tp),
        "lead_mean": float(np.mean(leads)),
        "stage2_lead_mean": float(np.mean(s2_leads)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, default="sheath_blight")
    p.add_argument("--run", type=int, default=6)
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    p.add_argument("--stage2_tstar_offset", type=int, default=60)
    p.add_argument("--delta", type=float, default=68.0,
                   help="constant mu shift to apply (mu_corrected = mu - delta)")
    p.add_argument("--pi_halfwidth", type=float, default=10.0,
                   help="PI = [mu - HW, mu + HW] (default 10 ~ 95%% PI for sigma=5)")
    args = p.parse_args()

    _ = resolve_pest(args.pest)

    print("\n[step 1/3] Stage 1 alert_map (test, F1 tau)…")
    alert_map, n_interval_groups = stage1_alert_map(Path(args.stage1_ckpt), args)
    print(f"[stage1] alert_map size = {len(alert_map)}  (interval site-year denom = {n_interval_groups})")

    print("\n[step 2/3] Stage 2 forward → mu per (site, year, tstar)…")
    ckpt2 = torch.load(args.stage2_ckpt, map_location="cpu")
    C.DOY_START = int(ckpt2.get("doy_start", C.DOY_START))
    C.DOY_END = int(ckpt2.get("doy_end", C.DOY_END))
    doy_start = int(ckpt2.get("doy_start", C.DOY_START))

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples2 = build_samples_for_run(args.run, get_feature_cols)
    train_s2_base, _, test_s2_base = split_samples(
        samples2, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )
    nc_window = int(ckpt2.get("stage2_nowcast_window", 28))
    nc_stride = int(ckpt2.get("stage2_nowcast_stride", 1))
    nc_tstart = ckpt2.get("stage2_nowcast_tstar_start", None)
    nc_only_pre = bool(int(ckpt2.get("stage2_nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt2.get("stage2_nowcast_event_time_proxy", "r"))
    nc_req = bool(int(ckpt2.get("stage2_nowcast_require_tstar_before_L", 0)))
    test_s2 = build_stage2_nowcast_samples(
        test_s2_base, window=nc_window, stride=nc_stride, tstar_start=nc_tstart,
        only_pre_event=nc_only_pre, event_time_proxy=nc_proxy, require_tstar_before_L=nc_req,
    )
    print(f"[stage2 nowcast] test rows = {len(test_s2)}")

    x_mean, x_std = compute_norm_stats(train_s2_base)
    test_groups = group_stage2_samples_by_site_year(test_s2)
    ds = GroupedIntervalEventDataset(test_groups, x_mean, x_std)
    loader = make_loader(ds, 1, shuffle=False, collate_fn=collate_grouped_stage2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, d_model, n_head, n_layers = build_stage2_model(
        ckpt2, d_in=int(test_s2[0]["X"].shape[1]), device=device
    )
    print(f"[stage2 model] d_model={d_model} n_head={n_head} n_layers={n_layers}")

    row_map = stage2_mu_per_tstar(model, loader, test_groups, device)
    print(f"[stage2 forward] rows = {len(row_map)}")

    print(f"\n[step 3/3] Match alerts at tstar + offset = {args.stage2_tstar_offset} (frame), "
          f"build matched_rows…")

    matched = []
    offset_missed = 0
    after_R = 0
    after_Tend = 0
    Tend = int(C.DOY_END - C.DOY_START + 1)
    for (site, year), alert_t in alert_map.items():
        target = int(alert_t) + int(args.stage2_tstar_offset)
        key = (str(site), int(year), int(target))
        info = row_map.get(key)
        if info is None or info["ctype"] != 0:  # not interval
            offset_missed += 1
            if target > Tend:
                after_Tend += 1
            continue
        mu = float(info["mu"])
        # convert to absolute DOY
        alert_abs = int(alert_t) + doy_start - 1
        stage2_abs = int(target) + doy_start - 1
        mu_abs = mu + doy_start - 1
        true_L_abs = int(info["true_L"]) + doy_start - 1
        true_R_abs = int(info["true_R"]) + doy_start - 1
        matched.append({
            "sample_id": f"{site}-{int(year)}",
            "site_id": site, "year": int(year),
            "alert_tstar_abs": alert_abs,
            "stage2_tstar_abs": stage2_abs,
            "tstar": int(stage2_abs),
            "mu_uncorr": float(mu_abs),
            "true_L": true_L_abs,
            "true_R": true_R_abs,
        })
    print(f"[match] matched={len(matched)}  offset_missed={offset_missed}  (after_Tend within={after_Tend})")

    HW = float(args.pi_halfwidth)
    DELTA = float(args.delta)

    rows_uncorr = []
    rows_corr = []
    for r in matched:
        mu = r["mu_uncorr"]
        ru = dict(r); ru["pred_L"] = int(round(mu - HW)); ru["pred_R"] = int(round(mu + HW)); ru["pred_point"] = int(round(mu))
        rows_uncorr.append(ru)
        rc = dict(r); mu_c = mu - DELTA
        rc["pred_L"] = int(round(mu_c - HW)); rc["pred_R"] = int(round(mu_c + HW)); rc["pred_point"] = int(round(mu_c))
        rows_corr.append(rc)

    m_un = compute_metrics(rows_uncorr, n_true=n_interval_groups)
    m_co = compute_metrics(rows_corr, n_true=n_interval_groups)

    def mu_stats(rows, key):
        if not rows:
            return (float("nan"),) * 4
        mu = np.asarray([float(r[key]) for r in rows], dtype=float)
        L = np.asarray([float(r["true_L"]) for r in rows], dtype=float)
        return (
            float(mu.mean()), float(mu.std(ddof=0)),
            float((mu - L).mean()),
            float(mu.std(ddof=0) / max(L.std(ddof=0), 1e-9)),
        )

    mu_u_mean, mu_u_std, du, ru = mu_stats(rows_uncorr, "pred_point")
    mu_c_mean, mu_c_std, dc, rc = mu_stats(rows_corr, "pred_point")

    BASELINE_REF = {
        "IoU80": 0.147, "EarlyRecall80": 0.960, "precision": 0.373, "recall": 0.273,
        "f1": 0.315, "mu_mean": 187.20, "mu_std": 5.05,
        "mean_mu_minus_L": -14.26, "mu_std_over_L_std": 0.280,
    }

    rows = [
        {"setup": "uncorrected (D=22 new)", **m_un,
         "mu_mean": mu_u_mean, "mu_std": mu_u_std,
         "mean(mu-L)": du, "mu_std/L_std": ru},
        {"setup": f"corrected (-{int(DELTA)})", **m_co,
         "mu_mean": mu_c_mean, "mu_std": mu_c_std,
         "mean(mu-L)": dc, "mu_std/L_std": rc},
        {"setup": "[ref] baseline D=15", "IoU80": BASELINE_REF["IoU80"],
         "EarlyRecall80": BASELINE_REF["EarlyRecall80"], "precision": BASELINE_REF["precision"],
         "recall": BASELINE_REF["recall"], "f1": BASELINE_REF["f1"],
         "pred_pos": 543, "tp": int(round(BASELINE_REF["precision"] * 543)),
         "lead_mean": float("nan"), "stage2_lead_mean": float("nan"),
         "mu_mean": BASELINE_REF["mu_mean"], "mu_std": BASELINE_REF["mu_std"],
         "mean(mu-L)": BASELINE_REF["mean_mu_minus_L"],
         "mu_std/L_std": BASELINE_REF["mu_std_over_L_std"],
         "EarlyRecall80_n": "—"},
    ]
    df = pd.DataFrame(rows)
    cols = ["setup", "pred_pos", "tp", "IoU80", "EarlyRecall80", "precision", "recall", "f1",
            "mu_mean", "mu_std", "mean(mu-L)", "mu_std/L_std", "lead_mean", "stage2_lead_mean"]
    df = df[cols]
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    pd.set_option("display.width", 240); pd.set_option("display.max_columns", 30)
    print("\n=================== SUMMARY ===================")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
