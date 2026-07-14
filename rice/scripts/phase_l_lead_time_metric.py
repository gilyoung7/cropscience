"""
Phase L — Operational lead-time decomposition (inference only).

For each (model, offset, sigma) cell:
    PI_end_i = mu_i + 1.96 * sigma                # 95% PI upper bound
    lead_i   = L_i - PI_end_i                     # days from PI_end to event

Bin (agronomic for sheath blight):
    MISSED      : lead <   0    (alert after event — useless)
    TOO_LATE    : 0   ≤ lead < 7    (insufficient time to spray)
    URGENT      : 7   ≤ lead < 14   (1-2 weeks; spray immediately)
    IDEAL       : 14  ≤ lead < 30   (prevention sweet spot)
    ADVANCE     : 30  ≤ lead < 45   (just before chemistry fade)
    TOO_EARLY   : lead ≥ 45     (chemistry expires; needs re-spray)

Summary metrics:
    P_ideal           = share(IDEAL)
    P_useful          = share(URGENT) + share(IDEAL) + share(ADVANCE)
    P_missed_or_late  = share(MISSED) + share(TOO_LATE)
    P_too_early       = share(TOO_EARLY)
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
from rice.scripts.common import collate_grouped_stage2, make_loader
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import build_nowcast_samples, build_tabular_from_samples
from rice.scripts.run_calibrate_event import apply_temperature, fit_temperature_grid


BIN_BOUNDARIES = [
    ("MISSED",    -np.inf,  0.0),
    ("TOO_LATE",   0.0,     7.0),
    ("URGENT",     7.0,    14.0),
    ("IDEAL",     14.0,    30.0),
    ("ADVANCE",   30.0,    45.0),
    ("TOO_EARLY", 45.0,     np.inf),
]
BIN_NAMES = [b[0] for b in BIN_BOUNDARIES]


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


def build_stage1_alert_map(stage1_ckpt_path: Path, run: int, args):
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
    print(f"[stage1] T*={t_best:.3f}  tau (F1 on val) = {tau:.3f}")

    alert = {}
    for s, p_cal in zip(test_s, p_test_cal):
        if p_cal < tau:
            continue
        key = (str(s["site_id"]), int(s["year"]))
        prev = alert.get(key)
        if prev is None or int(s["tstar"]) < prev:
            alert[key] = int(s["tstar"])
    n_interval = sum(1 for s in test_seas if str(s["censor_type"]) != "right")
    print(f"[stage1] alert_map size = {len(alert)}  (interval site-year denom = {n_interval})")
    return alert, n_interval


def build_stage2_row_map(stage2_ckpt_path: Path, run: int, args, device: torch.device):
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
    model.asym_weight_early = float(ckpt.get("stage2_pmf_asym_weight_early", 0.0))
    model.target_early_offset = float(ckpt.get("stage2_pmf_target_early_offset", 30.0))
    model.load_state_dict(ckpt["trained_states"][0]["state_dict"], strict=False)
    model.eval()
    print(f"[stage2 model] d_model={d_model} n_head={n_head} n_layers={n_layers} test_rows={len(test_s2)}")

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
    return row_map, doy_start


def classify_lead(lead: float) -> str:
    for name, lo, hi in BIN_BOUNDARIES:
        if lo <= lead < hi:
            return name
    return "TOO_EARLY"


def evaluate_lead_cell(alert_map: dict, row_map: dict, doy_start: int,
                       offset: int, sigma: float) -> dict:
    """
    For each alerted interval site-year:
        PI_end = mu + 1.96 * sigma   (DOY absolute)
        lead   = L - PI_end
    Bin lead into the agronomic categories above.
    """
    MULT = 1.96
    half = MULT * float(sigma)
    leads = []
    bins = []
    n_offset_missed = 0
    for (site, year), alert_t in alert_map.items():
        target = int(alert_t) + int(offset)
        info = row_map.get((str(site), int(year), int(target)))
        if info is None or info["ctype"] != 0:
            n_offset_missed += 1
            continue
        mu_abs = float(info["mu"]) + doy_start - 1
        L_abs  = int(info["true_L"]) + doy_start - 1
        PI_end = mu_abs + half
        lead = float(L_abs) - float(PI_end)
        leads.append(lead)
        bins.append(classify_lead(lead))

    n = len(leads)
    if n == 0:
        out = {"n_match": 0, "n_offset_missed": n_offset_missed,
               "lead_mean": float("nan"), "lead_median": float("nan"),
               "PI_end_mean_minus_L_mean": float("nan")}
        for name in BIN_NAMES:
            out[f"{name}_pct"] = float("nan")
        out["P_ideal"] = float("nan")
        out["P_useful"] = float("nan")
        out["P_missed_or_late"] = float("nan")
        out["P_too_early"] = float("nan")
        return out

    leads_arr = np.asarray(leads, dtype=float)
    counts = {name: 0 for name in BIN_NAMES}
    for b in bins:
        counts[b] += 1
    out = {
        "n_match": n, "n_offset_missed": n_offset_missed,
        "lead_mean": float(leads_arr.mean()),
        "lead_median": float(np.median(leads_arr)),
        "PI_end_mean_minus_L_mean": float(-leads_arr.mean()),
    }
    for name in BIN_NAMES:
        out[f"{name}_pct"] = 100.0 * counts[name] / n
    out["P_ideal"] = 100.0 * counts["IDEAL"] / n
    out["P_useful"] = 100.0 * (counts["URGENT"] + counts["IDEAL"] + counts["ADVANCE"]) / n
    out["P_missed_or_late"] = 100.0 * (counts["MISSED"] + counts["TOO_LATE"]) / n
    out["P_too_early"] = 100.0 * counts["TOO_EARLY"] / n
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, default="sheath_blight")
    p.add_argument("--run", type=int, default=4)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    p.add_argument("--stage1_ckpt", type=str, required=True,
                   help="shared Stage 1 ckpt (both models use the same D=15 stage1)")
    p.add_argument("--baseline_label", type=str, default="D=15 baseline (asym=15, 1-sided)")
    p.add_argument("--baseline_stage2_ckpt", type=str, required=True)
    p.add_argument("--new_label", type=str, default="D=15 new (asym=25, 2-sided)")
    p.add_argument("--new_stage2_ckpt", type=str, required=True)
    p.add_argument("--offsets", type=str, default="60,90,105,120")
    p.add_argument("--sigmas", type=str, default="3.5,5.0")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[abort] CUDA required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    _ = resolve_pest(args.pest)

    offsets = [int(x) for x in str(args.offsets).split(",") if x.strip()]
    sigmas = [float(x) for x in str(args.sigmas).split(",") if x.strip()]
    print(f"[device] {device} ({torch.cuda.get_device_name(0)})  "
          f"free={torch.cuda.mem_get_info(0)[0]//1024**2} MB")
    print(f"[grid] offsets={offsets}  sigmas={sigmas}  (PI_end = mu + 1.96 σ)")
    print(f"[bins] {', '.join(f'{b[0]}[{b[1]},{b[2]})' for b in BIN_BOUNDARIES)}")

    print("\n[stage1] alert_map (shared between both Stage 2 models)…")
    alert_map, _ = build_stage1_alert_map(Path(args.stage1_ckpt), args.run, args)

    rows: list[dict] = []
    for label, ckpt_path in [(args.baseline_label, args.baseline_stage2_ckpt),
                             (args.new_label, args.new_stage2_ckpt)]:
        print(f"\n----- model: {label} -----")
        print(f"  ckpt = {ckpt_path}")
        row_map, doy_start = build_stage2_row_map(Path(ckpt_path), args.run, args, device)
        for offset in offsets:
            for sigma in sigmas:
                m = evaluate_lead_cell(alert_map, row_map, doy_start, offset, sigma)
                rows.append({"model": label, "offset": int(offset), "sigma": float(sigma), **m})
        # free row_map between models
        del row_map
        torch.cuda.empty_cache()

    cols = ["model", "offset", "sigma", "n_match", "lead_mean", "lead_median",
            "MISSED_pct", "TOO_LATE_pct", "URGENT_pct", "IDEAL_pct",
            "ADVANCE_pct", "TOO_EARLY_pct",
            "P_ideal", "P_useful", "P_missed_or_late", "P_too_early"]
    df = pd.DataFrame(rows)[cols]
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    pd.set_option("display.width", 260); pd.set_option("display.max_columns", 30)
    print("\n=================== LEAD-TIME BINS (%) ===================")
    print(df.to_string(index=False))

    # Best cells (operational value)
    print("\n--- best cells by operational metric ---")
    idx_ideal = df["P_ideal"].idxmax()
    idx_useful = df["P_useful"].idxmax()
    idx_low_miss = df["P_missed_or_late"].idxmin()
    r1 = df.loc[idx_ideal]
    r2 = df.loc[idx_useful]
    r3 = df.loc[idx_low_miss]
    print(f"  best P_ideal           : {r1['model']}  offset={int(r1['offset'])} σ={r1['sigma']:.1f}  "
          f"P_ideal={r1['P_ideal']:.1f}%  P_useful={r1['P_useful']:.1f}%  "
          f"P_missed_or_late={r1['P_missed_or_late']:.1f}%")
    print(f"  best P_useful (7~45d)  : {r2['model']}  offset={int(r2['offset'])} σ={r2['sigma']:.1f}  "
          f"P_ideal={r2['P_ideal']:.1f}%  P_useful={r2['P_useful']:.1f}%  "
          f"P_missed_or_late={r2['P_missed_or_late']:.1f}%")
    print(f"  lowest P_missed_or_late: {r3['model']}  offset={int(r3['offset'])} σ={r3['sigma']:.1f}  "
          f"P_ideal={r3['P_ideal']:.1f}%  P_useful={r3['P_useful']:.1f}%  "
          f"P_missed_or_late={r3['P_missed_or_late']:.1f}%")

    # Per-model best
    print("\n--- per-model best P_useful ---")
    for label, sub in df.groupby("model", sort=False):
        idx = sub["P_useful"].idxmax()
        r = df.loc[idx]
        print(f"  [{label}] offset={int(r['offset'])} σ={r['sigma']:.1f}  "
              f"P_ideal={r['P_ideal']:.1f}%  P_useful={r['P_useful']:.1f}%  "
              f"P_missed_or_late={r['P_missed_or_late']:.1f}%  lead_mean={r['lead_mean']:.1f}")


if __name__ == "__main__":
    main()
