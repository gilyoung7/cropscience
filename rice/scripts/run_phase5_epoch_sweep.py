"""
Phase 5 Gaussian PMF: epoch-by-epoch sweep evaluation.

For each saved epoch checkpoint under --ckpt_dir:
  - rebuild the Stage-2 hierarchical model with pmf_mode/sigma from the ckpt
  - run val_loader, collect (mu, hazard, true_L/R, tstar, ctype) per t* row
  - for PI in {60, 80, 95}: derive (pL, pR) via shortest-mass interval
  - compute IoU, recall, width per row
  - bin by lead = (true_L + 1 - tstar): {1-14, 15-29, 30-45, 46-60, 61-75, >75}
  - EarlyRecall = mean over interval rows of (pred_L <= true_L)

Outputs (under --out_dir):
  - summary.csv: one row per (epoch, PI level, lead bin)
  - mu_stats.csv: per-epoch mu summary stats
  - mu_vs_L.csv: per-row mu and true_L (event rows only) for scatter plots
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from rice.configs import config as C
from rice.scripts.common import make_loader, collate_grouped_stage2
from rice.scripts.run_eval import build_samples_for_run
from rice.src.dataset import (
    GroupedIntervalEventDataset,
    build_stage2_nowcast_samples,
    compute_norm_stats,
    group_stage2_samples_by_site_year,
    split_samples,
)
from rice.src.model import HierarchicalCausalHazardTransformer
from rice.src.pest_resolver import resolve_pest
from rice.src.train_eval import (
    CTYPE_INTERVAL,
    hazard_to_pmf_cdf_logS,
    shortest_mass_interval_1d,
)


def _iou_rec_width(pred_l: int, pred_r: int, true_l: int, true_r: int):
    true_start = true_l + 1
    true_end = true_r
    inter_l = max(pred_l, true_start)
    inter_r = min(pred_r, true_end)
    inter = max(0, inter_r - inter_l + 1)
    pred_len = max(1, pred_r - pred_l + 1)
    true_len = max(1, true_end - true_start + 1)
    union_l = min(pred_l, true_start)
    union_r = max(pred_r, true_end)
    union = max(1, union_r - union_l + 1)
    iou = inter / union
    rec = inter / true_len
    width = pred_r - pred_l + 1
    return float(iou), float(rec), int(width)


def _lead_bin(lead: int) -> str:
    if lead <= 0:
        return "lead_le0"
    if lead <= 14:
        return "lead_1_14"
    if lead <= 29:
        return "lead_15_29"
    if lead <= 45:
        return "lead_30_45"
    if lead <= 60:
        return "lead_46_60"
    if lead <= 75:
        return "lead_61_75"
    return "lead_gt75"


LEAD_BINS = ["lead_1_14", "lead_15_29", "lead_30_45", "lead_46_60", "lead_61_75", "lead_gt75", "lead_le0"]


@torch.no_grad()
def collect_rows_for_epoch(ckpt_path: Path, val_loader, Tend: int, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = ckpt["trained_states"][0]["state_dict"]
    epoch = int(ckpt.get("epoch", -1))

    D_in = state_dict["in_proj.weight"].shape[1]
    d_model = int(ckpt.get("d_model", C.D_MODEL))
    n_head = int(ckpt.get("n_head", C.N_HEAD))
    n_layers = int(ckpt.get("n_layers", C.N_LAYERS))
    pmf_mode = str(ckpt.get("stage2_pmf_mode", "hazard"))
    sigma = float(ckpt.get("stage2_pmf_sigma", 5.0))
    mu_max = float(ckpt.get("stage2_pmf_mu_max", 0.0))
    tstar_layers = int(ckpt.get("stage2_tstar_layers", 1))
    use_tstar_scalar_pos = bool(int(ckpt.get("stage2_use_tstar_scalar_pos", 0)))
    time_chunk = int(ckpt.get("stage2_time_chunk_size", 64))
    conditional_survival = bool(int(ckpt.get("stage2_conditional_survival", 0)))

    model = HierarchicalCausalHazardTransformer(
        d_in=D_in,
        d_model=d_model,
        nhead=n_head,
        num_layers=n_layers,
        num_tstar_layers=tstar_layers,
        dropout=C.DROPOUT,
        max_len=C.MAX_LEN,
        max_tstar_len=512,
        use_tstar_scalar_pos=use_tstar_scalar_pos,
    ).to(device)
    model.time_chunk_size = time_chunk
    model.pmf_mode = pmf_mode
    model.gaussian_sigma = sigma
    model.gaussian_mu_max = mu_max
    model.conditional_survival = conditional_survival
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print(f"  [{ckpt_path.name}] missing keys ({len(missing)}): {missing[:5]}")
    if unexpected:
        print(f"  [{ckpt_path.name}] unexpected keys ({len(unexpected)}): {unexpected[:5]}")
    model.eval()

    rows: list[dict] = []
    for X, L, R, ctype, tstar, valid_mask in val_loader:
        X = X.to(device, non_blocking=True)
        tstar_t = tstar.to(device, non_blocking=True)
        valid_mask_t = valid_mask.to(device, non_blocking=True)

        hazard = model(X, tstar=tstar_t, valid_mask=valid_mask_t)
        B, K, T_h = hazard.shape

        mu_BK = getattr(model, "_last_mu_BK", None)
        mu_np = mu_BK.detach().cpu().numpy() if mu_BK is not None else None

        nll_tstar = tstar_t.reshape(B * K) if conditional_survival else None
        pmf, _, _ = hazard_to_pmf_cdf_logS(hazard.reshape(B * K, T_h), tstar=nll_tstar)
        pmf_np = pmf.cpu().numpy().reshape(B, K, T_h)

        L_np = L.cpu().numpy().astype(int)
        R_np = R.cpu().numpy().astype(int)
        ctype_np = ctype.cpu().numpy().astype(int)
        tstar_np = tstar.cpu().numpy().astype(int)
        valid_np = valid_mask.cpu().numpy().astype(bool)

        for bi in range(B):
            for ki in range(K):
                if not valid_np[bi, ki]:
                    continue
                if int(ctype_np[bi, ki]) != int(CTYPE_INTERVAL):
                    continue
                true_l = int(L_np[bi, ki])
                true_r = int(R_np[bi, ki])
                tstar_val = int(tstar_np[bi, ki])
                lead = (true_l + 1) - tstar_val
                rows.append({
                    "epoch": epoch,
                    "tstar": tstar_val,
                    "true_l": true_l,
                    "true_r": true_r,
                    "lead": lead,
                    "lead_bin": _lead_bin(lead),
                    "mu": float(mu_np[bi, ki]) if mu_np is not None else float("nan"),
                    "pmf": pmf_np[bi, ki].astype(np.float32),
                })
    return rows, epoch


def summarize_for_pi(rows: list[dict], pi_level: int, Tend: int):
    target_mass = float(pi_level) / 100.0
    bin_buckets: dict[str, dict[str, list]] = {b: {"iou": [], "rec": [], "width": [], "early": []} for b in LEAD_BINS + ["all"]}
    for r in rows:
        pL, pR, _ = shortest_mass_interval_1d(r["pmf"], target_mass=target_mass, Tend=Tend)
        iou, rec, width = _iou_rec_width(int(pL), int(pR), int(r["true_l"]), int(r["true_r"]))
        early = 1.0 if int(pL) <= int(r["true_l"]) else 0.0
        for key in [r["lead_bin"], "all"]:
            if key not in bin_buckets:
                continue
            bin_buckets[key]["iou"].append(iou)
            bin_buckets[key]["rec"].append(rec)
            bin_buckets[key]["width"].append(width)
            bin_buckets[key]["early"].append(early)
    out = []
    for bin_name, vals in bin_buckets.items():
        n = len(vals["iou"])
        out.append({
            "pi_level": int(pi_level),
            "lead_bin": bin_name,
            "n": n,
            "iou_mean": float(np.mean(vals["iou"])) if n else float("nan"),
            "rec_mean": float(np.mean(vals["rec"])) if n else float("nan"),
            "early_recall": float(np.mean(vals["early"])) if n else float("nan"),
            "width_mean": float(np.mean(vals["width"])) if n else float("nan"),
            "width_median": float(np.median(vals["width"])) if n else float("nan"),
        })
    return out


def mu_distribution_stats(rows: list[dict]) -> dict:
    mus = np.array([r["mu"] for r in rows if np.isfinite(r["mu"])])
    Ls = np.array([r["true_l"] for r in rows if np.isfinite(r["mu"])])
    if mus.size == 0:
        return {}
    delta = mus - Ls.astype(float)
    return {
        "mu_n": int(mus.size),
        "mu_min": float(mus.min()),
        "mu_q05": float(np.quantile(mus, 0.05)),
        "mu_q25": float(np.quantile(mus, 0.25)),
        "mu_median": float(np.median(mus)),
        "mu_mean": float(mus.mean()),
        "mu_q75": float(np.quantile(mus, 0.75)),
        "mu_q95": float(np.quantile(mus, 0.95)),
        "mu_max": float(mus.max()),
        "mu_std": float(mus.std()),
        "mu_minus_L_mean": float(delta.mean()),
        "mu_minus_L_abs_mean": float(np.abs(delta).mean()),
        "mu_pos_frac": float(np.mean(delta > 0)),
        "L_mean": float(Ls.mean()),
        "L_std": float(Ls.std()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--split_seed", type=int, default=54)
    ap.add_argument("--split_mode", default="site_year")
    ap.add_argument("--ckpt_dir", required=True, help="dir containing checkpoint_run*_seed*_epoch*.pt")
    ap.add_argument("--seed", type=int, default=0, help="train seed to filter epoch checkpoints")
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--pi_levels", default="60,80,95")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pi_levels = [int(x) for x in args.pi_levels.split(",")]

    # apply_pest_config populates DOY_START / D_MODEL / DROPOUT / MAX_LEN etc. on C.
    _, get_feature_cols = resolve_pest(args.pest)

    # Pull DOY/D_MODEL etc. from one checkpoint to align config
    sample_ckpts = sorted(Path(args.ckpt_dir).glob(f"checkpoint_run*_seed{int(args.seed)}_epoch*.pt"))
    if not sample_ckpts:
        raise RuntimeError(f"no epoch checkpoints under {args.ckpt_dir}")
    head_ckpt = torch.load(sample_ckpts[0], map_location="cpu", weights_only=False)
    C.DOY_START = int(head_ckpt.get("doy_start", C.DOY_START))
    C.DOY_END = int(head_ckpt.get("doy_end", C.DOY_END))
    C.D_MODEL = int(head_ckpt.get("d_model", C.D_MODEL))
    C.N_HEAD = int(head_ckpt.get("n_head", C.N_HEAD))
    C.N_LAYERS = int(head_ckpt.get("n_layers", C.N_LAYERS))

    nowcast_window = int(head_ckpt.get("stage2_nowcast_window", 28))
    nowcast_stride = int(head_ckpt.get("stage2_nowcast_stride", 1))
    nowcast_only_pre_event = int(head_ckpt.get("stage2_nowcast_only_pre_event", 1))
    nowcast_event_time_proxy = str(head_ckpt.get("stage2_nowcast_event_time_proxy", "r"))
    nowcast_require_tstar_before_L = int(head_ckpt.get("stage2_nowcast_require_tstar_before_L", 0))
    nowcast_tstar_start = head_ckpt.get("stage2_nowcast_tstar_start", None)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    feature_cols, feature_names, T, samples = build_samples_for_run(int(args.run), get_feature_cols)
    print(f"[features] n={len(feature_names)} T={T} samples={len(samples)}")

    train_s, val_s, _ = split_samples(samples, val_frac=0.1, test_frac=0.1, seed=int(args.split_seed), split_mode=args.split_mode)
    print(f"[split] mode={args.split_mode} seed={args.split_seed} train={len(train_s)} val={len(val_s)}")

    train_s2 = build_stage2_nowcast_samples(
        train_s,
        window=nowcast_window,
        stride=nowcast_stride,
        tstar_start=nowcast_tstar_start,
        only_pre_event=bool(nowcast_only_pre_event),
        event_time_proxy=nowcast_event_time_proxy,
        require_tstar_before_L=bool(nowcast_require_tstar_before_L),
    )
    val_s2 = build_stage2_nowcast_samples(
        val_s,
        window=nowcast_window,
        stride=nowcast_stride,
        tstar_start=nowcast_tstar_start,
        only_pre_event=bool(nowcast_only_pre_event),
        event_time_proxy=nowcast_event_time_proxy,
        require_tstar_before_L=bool(nowcast_require_tstar_before_L),
    )
    print(f"[nowcast] train={len(train_s2)} val={len(val_s2)} window={nowcast_window} stride={nowcast_stride}")

    print("[norm] computing norm stats from train_s2 ...")
    x_mean, x_std = compute_norm_stats(train_s2)
    val_groups = group_stage2_samples_by_site_year(val_s2)
    val_ds = GroupedIntervalEventDataset(val_groups, x_mean, x_std)
    val_loader = make_loader(val_ds, C.BATCH_EVAL, shuffle=False, collate_fn=collate_grouped_stage2)
    print(f"[loader] val_groups={len(val_groups)} batch={C.BATCH_EVAL}")

    summary_rows = []
    mu_stats_rows = []
    mu_vs_L_rows = []

    for ckpt_path in sample_ckpts:
        print(f"--- {ckpt_path.name} ---")
        rows, ep = collect_rows_for_epoch(ckpt_path, val_loader, Tend=T, device=device)
        print(f"  collected {len(rows)} interval rows")
        for pi in pi_levels:
            for r in summarize_for_pi(rows, pi_level=pi, Tend=T):
                r["epoch"] = ep
                summary_rows.append(r)
        mu_stats = mu_distribution_stats(rows)
        mu_stats["epoch"] = ep
        mu_stats_rows.append(mu_stats)
        for r in rows:
            mu_vs_L_rows.append({
                "epoch": ep,
                "tstar": r["tstar"],
                "true_l": r["true_l"],
                "true_r": r["true_r"],
                "lead": r["lead"],
                "lead_bin": r["lead_bin"],
                "mu": r["mu"],
            })

    def write_csv(path: Path, rows: list[dict]):
        if not rows:
            return
        keys = sorted({k for r in rows for k in r.keys()})
        ordered = ["epoch"] + [k for k in keys if k != "epoch"]
        with path.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=ordered)
            w.writeheader()
            for r in rows:
                w.writerow({k: r.get(k, "") for k in ordered})
        print(f"saved {path} ({len(rows)} rows)")

    write_csv(out_dir / "summary.csv", summary_rows)
    write_csv(out_dir / "mu_stats.csv", mu_stats_rows)
    write_csv(out_dir / "mu_vs_L.csv", mu_vs_L_rows)


if __name__ == "__main__":
    main()
