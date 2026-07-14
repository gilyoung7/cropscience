"""
Phase N — 4-way operational evaluation (inference only, post-hoc shift).

Grid: 4 models × 4 offsets × 2 σ × 4 shifts = 128 cells.

For each cell on the matched interval cohort (alert_tstar + offset row exists):
    PI     = [mu - 1.96σ,        mu + 1.96σ]                   # academic, shift=0
    PI_op  = [mu - shift - 1.96σ, mu - shift + 1.96σ]          # operational
    lead   = L - PI_op.end = L - (mu + 1.96σ - shift)

Buckets (interval samples):
    MISSED      lead < 0
    TOO_LATE    0   ≤ lead < 7
    URGENT      7   ≤ lead < 14
    IDEAL       14  ≤ lead < 30
    ADVANCE     30  ≤ lead < 45
    TOO_EARLY   lead ≥ 45

Metrics:
    Academic   : IoU(PI, [L, R]) at shift=0   (PI vs ground truth interval)
                 mean(mu - mid), mu_std
    Operational: P_useful_A = lead ∈ [0, 45]
                 P_useful_B = lead ∈ [7, 45]
                 P_ideal    = lead ∈ [14, 30]
                 P_ideal_given_useful_A, P_ideal_given_useful_B
                 P_missed_or_late, P_too_early
                 lead_mean_useful_B, lead_median_useful_B

Stage 1 alert_map is built once (same D=15 ckpt).
Stage 2 forward is run once per model (4 forwards total).
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

    # Phenology bias head support: ckpt records whether the model was trained
    # with phen_head. If so, push phenology cols through build_samples_for_run
    # so each Stage 2 nowcast sample carries a `pheno_vec`.
    phen_bias_head = bool(int(ckpt.get("stage2_phenology_bias_head", 0)))
    phen_hidden = int(ckpt.get("stage2_phenology_hidden", 8))
    pheno_ext_cols = (
        ["best_suitability", "best_months", "offset_days", "window_idx"]
        if phen_bias_head else None
    )
    print(f"  [phen_bias_head] enabled={phen_bias_head}  hidden={phen_hidden}  "
          f"pheno_ext_cols={pheno_ext_cols}")

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
        phenology_bias_head=phen_bias_head,
        phenology_dim=4,
        phenology_hidden=phen_hidden,
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
    print(f"[stage2 model] d_model={d_model} n_head={n_head} n_layers={n_layers} "
          f"target_mode={model.target_mode} test_rows={len(test_s2)}")

    row_map: dict = {}
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
    return row_map, doy_start


def evaluate_cell(alert_map: dict, row_map: dict, doy_start: int,
                  offset: int, sigma: float, shift: float) -> dict:
    HW = 1.96 * float(sigma)
    sh = float(shift)
    matched = []
    for (site, year), alert_t in alert_map.items():
        target = int(alert_t) + int(offset)
        info = row_map.get((str(site), int(year), int(target)))
        if info is None or info["ctype"] != 0:
            continue
        mu = float(info["mu"]) + doy_start - 1
        true_L = int(info["true_L"]) + doy_start - 1
        true_R = int(info["true_R"]) + doy_start - 1
        mid = 0.5 * (true_L + true_R)
        PI_lo = mu - HW
        PI_hi = mu + HW
        PI_op_hi = mu - sh + HW
        lead = float(true_L) - float(PI_op_hi)
        matched.append({
            "mu": mu, "mid": mid, "L": true_L, "R": true_R,
            "PI_lo": PI_lo, "PI_hi": PI_hi, "lead": lead,
        })
    if not matched:
        out = {"n_match": 0, "IoU_PI_LR": float("nan"),
               "mu_mean": float("nan"), "mu_std": float("nan"),
               "mean_mu_minus_mid": float("nan"),
               "P_useful_A": float("nan"), "P_useful_B": float("nan"),
               "P_ideal": float("nan"),
               "P_ideal_given_useful_A": float("nan"),
               "P_ideal_given_useful_B": float("nan"),
               "P_missed_or_late": float("nan"),
               "P_too_early": float("nan"),
               "lead_mean_useful_B": float("nan"),
               "lead_median_useful_B": float("nan")}
        for n in ("MISSED", "TOO_LATE", "URGENT", "IDEAL", "ADVANCE", "TOO_EARLY"):
            out[f"{n}_pct"] = float("nan")
        return out

    leads = np.asarray([m["lead"] for m in matched], dtype=float)
    mus = np.asarray([m["mu"] for m in matched], dtype=float)
    mids = np.asarray([m["mid"] for m in matched], dtype=float)
    Ls = np.asarray([m["L"] for m in matched], dtype=int)
    Rs = np.asarray([m["R"] for m in matched], dtype=int)

    iou_pi_lr = []
    for m in matched:
        # ground-truth interval = [L, R]; PI = [round(PI_lo), round(PI_hi)]
        pL_pi = int(round(m["PI_lo"]))
        pR_pi = int(round(m["PI_hi"]))
        # treat overlap_metrics convention: true_L2 = L + 1 internally
        iou, _, _ = overlap_metrics(pL_pi, pR_pi, int(m["L"]), int(m["R"]))
        iou_pi_lr.append(iou)
    iou_pi_lr = float(np.mean(iou_pi_lr))

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

    out = {
        "n_match": n,
        "IoU_PI_LR": iou_pi_lr,
        "mu_mean": float(mus.mean()),
        "mu_std": float(mus.std(ddof=0)),
        "mean_mu_minus_mid": float((mus - mids).mean()),
        "MISSED_pct": pct(missed),
        "TOO_LATE_pct": pct(too_late),
        "URGENT_pct": pct(urgent),
        "IDEAL_pct": pct(ideal),
        "ADVANCE_pct": pct(advance),
        "TOO_EARLY_pct": pct(too_early),
        "P_useful_A": P_useful_A,
        "P_useful_B": P_useful_B,
        "P_ideal": P_ideal,
        "P_ideal_given_useful_A": (P_ideal / P_useful_A) if P_useful_A > 0 else float("nan"),
        "P_ideal_given_useful_B": (P_ideal / P_useful_B) if P_useful_B > 0 else float("nan"),
        "P_missed_or_late": pct(missed) + pct(too_late),
        "P_too_early": pct(too_early),
        "lead_mean_useful_B": float(leads[(leads >= 7) & (leads < 45)].mean())
            if ((leads >= 7) & (leads < 45)).any() else float("nan"),
        "lead_median_useful_B": float(np.median(leads[(leads >= 7) & (leads < 45)]))
            if ((leads >= 7) & (leads < 45)).any() else float("nan"),
    }
    return out


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
                   help="Comma list of LABEL=CKPT pairs (use | to separate label and ckpt, "
                        "and ; to separate models). e.g. 'baseline|path1;new|path2'")
    p.add_argument("--offsets", type=str, default="60,90,105,120")
    p.add_argument("--sigmas", type=str, default="3.5,5.0")
    p.add_argument("--shifts", type=str, default="0,14,22,30")
    p.add_argument("--out_csv", type=str, default=None,
                   help="optional CSV path for the full 128-row grid")
    args = p.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("[abort] CUDA required")
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    _ = resolve_pest(args.pest)

    offsets = [int(x) for x in str(args.offsets).split(",") if x.strip()]
    sigmas = [float(x) for x in str(args.sigmas).split(",") if x.strip()]
    shifts = [float(x) for x in str(args.shifts).split(",") if x.strip()]
    print(f"[device] {device} ({torch.cuda.get_device_name(0)})  "
          f"free={torch.cuda.mem_get_info(0)[0]//1024**2} MB")
    print(f"[grid] offsets={offsets}  sigmas={sigmas}  shifts={shifts}")

    # Parse models
    model_entries = []
    for part in str(args.models).split(";"):
        part = part.strip()
        if not part:
            continue
        if "|" not in part:
            raise SystemExit(f"Bad --models entry: {part!r} (must be LABEL|CKPT)")
        label, ckpt = part.split("|", 1)
        model_entries.append((label.strip(), ckpt.strip()))
    if len(model_entries) < 2:
        raise SystemExit("Need at least 2 models in --models")
    print(f"[models] {len(model_entries)} ckpts")
    for label, ckpt in model_entries:
        print(f"  {label} <- {ckpt}")

    print("\n[stage1] alert_map (shared)…")
    alert_map, n_interval = build_stage1_alert_map(Path(args.stage1_ckpt), args.run, args)

    rows: list[dict] = []
    for label, ckpt_path in model_entries:
        print(f"\n----- model: {label} -----")
        row_map, doy_start = build_stage2_row_map(Path(ckpt_path), args.run, args, device)
        for offset in offsets:
            for sigma in sigmas:
                for shift in shifts:
                    m = evaluate_cell(alert_map, row_map, doy_start,
                                      offset=offset, sigma=sigma, shift=shift)
                    rows.append({
                        "model": label,
                        "offset": int(offset),
                        "sigma": float(sigma),
                        "shift": float(shift),
                        **m,
                    })
        del row_map
        torch.cuda.empty_cache()

    df = pd.DataFrame(rows)
    cols = ["model", "offset", "sigma", "shift", "n_match",
            "IoU_PI_LR", "mu_mean", "mu_std", "mean_mu_minus_mid",
            "MISSED_pct", "TOO_LATE_pct", "URGENT_pct", "IDEAL_pct",
            "ADVANCE_pct", "TOO_EARLY_pct",
            "P_useful_A", "P_useful_B", "P_ideal",
            "P_ideal_given_useful_A", "P_ideal_given_useful_B",
            "P_missed_or_late", "P_too_early",
            "lead_mean_useful_B", "lead_median_useful_B"]
    df = df[cols]

    if args.out_csv:
        df.to_csv(args.out_csv, index=False)
        print(f"\n[csv] full grid saved to {args.out_csv}  rows={len(df)}")

    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    pd.set_option("display.width", 260)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)

    print("\n=================== FULL GRID (128 rows) ===================")
    print(df.to_string(index=False))

    print("\n=================== Per-model best cells ===================")
    summary_rows = []
    for label, sub in df.groupby("model", sort=False):
        s = sub.dropna(subset=["P_ideal"])
        if s.empty:
            continue
        i_ideal = s["P_ideal"].idxmax()
        i_cond = s["P_ideal_given_useful_B"].idxmax()
        i_iou = s["IoU_PI_LR"].idxmax()
        print(f"\n[{label}]")
        for tag, idx in [("best P_ideal           ", i_ideal),
                         ("best P_ideal|useful_B  ", i_cond),
                         ("best IoU(PI,[L,R])     ", i_iou)]:
            r = s.loc[idx]
            print(f"  {tag}: offset={int(r['offset'])} σ={r['sigma']:.1f} shift={r['shift']:.0f} | "
                  f"IoU={r['IoU_PI_LR']:.3f} | "
                  f"P_useful_A={r['P_useful_A']:.1f}% P_useful_B={r['P_useful_B']:.1f}% "
                  f"P_ideal={r['P_ideal']:.1f}% (|UB={r['P_ideal_given_useful_B']:.3f}) "
                  f"P_M+L={r['P_missed_or_late']:.1f}% P_TE={r['P_too_early']:.1f}% "
                  f"lead_med_UB={r['lead_median_useful_B']:.1f}")
        summary_rows.append({
            "model": label,
            "best_ideal_cell": f"off={int(s.loc[i_ideal,'offset'])},σ={s.loc[i_ideal,'sigma']:.1f},sh={s.loc[i_ideal,'shift']:.0f}",
            "best_P_ideal": float(s.loc[i_ideal, "P_ideal"]),
            "best_iou_cell": f"off={int(s.loc[i_iou,'offset'])},σ={s.loc[i_iou,'sigma']:.1f},sh={s.loc[i_iou,'shift']:.0f}",
            "best_IoU_PI_LR": float(s.loc[i_iou, "IoU_PI_LR"]),
            "P_useful_B_at_best_ideal": float(s.loc[i_ideal, "P_useful_B"]),
            "P_missed_or_late_at_best_ideal": float(s.loc[i_ideal, "P_missed_or_late"]),
        })

    print("\n=================== 4-way summary (each model best by P_ideal) ===================")
    print(pd.DataFrame(summary_rows).to_string(index=False))


if __name__ == "__main__":
    main()
