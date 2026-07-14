"""
Phase F — Feature-group ablation on Stage 2 mu (inference only).

For each test (site, year, t*) nowcast row, replace the values of a feature group
(phenology_4 or rolling_3) with the train mean of that feature and clear the
corresponding __miss indicator, then re-run Stage 2 forward inference. Compare
the resulting mu distributions against the unmodified baseline.

No retraining.
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


PHENO_4 = ["best_suitability", "best_months", "offset_days", "window_idx"]
ROLL_3  = ["rain_14d_sum", "wind_7d_mean", "wind_7d_max"]


def feature_indices(feature_cols: list[str], names: list[str]) -> list[int]:
    return [feature_cols.index(n) for n in names if n in feature_cols]


def compute_train_means(samples: list[dict], indices: list[int]) -> dict[int, float]:
    out: dict[int, float] = {}
    if not samples:
        return out
    for idx in indices:
        vals = np.concatenate([np.asarray(s["X"])[:, idx] for s in samples])
        out[idx] = float(vals.mean())
    return out


def apply_ablation(samples: list[dict], indices: list[int], means: dict[int, float], D_raw: int) -> list[dict]:
    out = []
    for s in samples:
        X = np.asarray(s["X"]).copy()
        for idx in indices:
            X[:, idx] = float(means[idx])
            if X.shape[1] > idx + D_raw:
                X[:, idx + D_raw] = 0.0
        s2 = dict(s)
        s2["X"] = X
        out.append(s2)
    return out


def build_stage2_model(ckpt: dict, d_in: int, device: torch.device) -> HierarchicalCausalHazardTransformer:
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
    model.pmf_mode = str(ckpt.get("stage2_pmf_mode", "gaussian"))
    model.gaussian_sigma = float(ckpt.get("stage2_pmf_sigma", 5.0))
    model.gaussian_mu_max = float(ckpt.get("stage2_pmf_mu_max", 0.0))
    model.asym_weight = float(ckpt.get("stage2_pmf_asym_weight", 15.0))
    model.right_weight = float(ckpt.get("stage2_pmf_right_weight", 0.3))
    model.target_offset = float(ckpt.get("stage2_pmf_target_offset", 5.0))
    d2 = ckpt["trained_states"][0]
    model.load_state_dict(d2["state_dict"], strict=False)
    model.eval()
    return model


@torch.no_grad()
def forward_extract_mu(model, loader, groups: list[dict], device: torch.device) -> pd.DataFrame:
    rows = []
    gi = 0
    for X, L, R, ctype, tstar, valid_mask in loader:
        X = X.to(device)
        tstar_t = tstar.to(device)
        valid_t = valid_mask.to(device)
        _ = model(X, tstar=tstar_t, valid_mask=valid_t)
        mu_BK = getattr(model, "_last_mu_BK")
        mu_np = mu_BK.detach().cpu().numpy()
        v_np = valid_mask.cpu().numpy().astype(bool)
        L_np = L.cpu().numpy().astype(int)
        c_np = ctype.cpu().numpy().astype(int)
        B, K = mu_np.shape
        for bi in range(B):
            g = groups[gi + bi]
            mu_vals = mu_np[bi][v_np[bi]]
            if len(mu_vals) == 0:
                continue
            L_event = None
            for ki in range(K):
                if v_np[bi, ki] and c_np[bi, ki] == 0:
                    L_event = int(L_np[bi, ki])
                    break
            rows.append({
                "site_id": str(g["site_id"]),
                "year": int(g["year"]),
                "mu_mean": float(mu_vals.mean()),
                "mu_first": float(mu_vals[0]),
                "mu_last": float(mu_vals[-1]),
                "n_tstar": int(len(mu_vals)),
                "L": float(L_event) if L_event is not None else float("nan"),
                "censor": "interval" if L_event is not None else "right",
            })
        gi += B
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, label: str) -> dict:
    interval = df[df["censor"] == "interval"].copy()
    mu = interval["mu_mean"].to_numpy(dtype=float)
    L = interval["L"].to_numpy(dtype=float)
    mask = np.isfinite(mu) & np.isfinite(L)
    if mask.sum() < 3:
        return {"label": label}
    mu_v, L_v = mu[mask], L[mask]
    return {
        "label": label,
        "n_interval": int(mask.sum()),
        "mu_mean": float(mu_v.mean()),
        "mu_std": float(mu_v.std(ddof=0)),
        "L_mean": float(L_v.mean()),
        "L_std": float(L_v.std(ddof=0)),
        "mean_mu_minus_L": float((mu_v - L_v).mean()),
        "median_mu_minus_L": float(np.median(mu_v - L_v)),
        "mean_abs_mu_minus_L": float(np.abs(mu_v - L_v).mean()),
        "mu_std_over_L_std": float(mu_v.std(ddof=0) / max(L_v.std(ddof=0), 1e-9)),
    }


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, default="sheath_blight")
    p.add_argument("--run", type=int, default=6)
    p.add_argument("--stage2_ckpt", type=str, required=True)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    args = p.parse_args()

    _, get_feature_cols = resolve_pest(args.pest)
    feature_cols = get_feature_cols(args.run)
    D_raw = len(feature_cols)
    print(f"[features] run={args.run}  D_raw={D_raw}  feature_cols={feature_cols}")

    pheno_idx = feature_indices(feature_cols, PHENO_4)
    roll_idx  = feature_indices(feature_cols, ROLL_3)
    print(f"[ablation] PHENO_4 idx={pheno_idx}   ROLL_3 idx={roll_idx}")

    ckpt2 = torch.load(args.stage2_ckpt, map_location="cpu")
    C.DOY_START = int(ckpt2.get("doy_start", C.DOY_START))
    C.DOY_END = int(ckpt2.get("doy_end", C.DOY_END))

    _, _, T, samples2 = build_samples_for_run(args.run, get_feature_cols)
    train_s2_base, val_s2_base, test_s2_base = split_samples(
        samples2, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )
    print(f"[split] train={len(train_s2_base)} val={len(val_s2_base)} test={len(test_s2_base)}")

    # train means computed on raw season-level X
    pheno_means = compute_train_means(train_s2_base, pheno_idx)
    roll_means  = compute_train_means(train_s2_base, roll_idx)
    print("[train_means PHENO_4]:")
    for c, idx in zip(PHENO_4, pheno_idx):
        print(f"  {c:<18} idx={idx}  train_mean={pheno_means[idx]:.4f}")
    print("[train_means ROLL_3]:")
    for c, idx in zip(ROLL_3, roll_idx):
        print(f"  {c:<18} idx={idx}  train_mean={roll_means[idx]:.4f}")

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
    print(f"[nowcast test rows] = {len(test_s2)}")

    x_mean, x_std = compute_norm_stats(train_s2_base)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_stage2_model(ckpt2, d_in=int(test_s2[0]["X"].shape[1]), device=device)

    def run_inference(samples, label):
        groups = group_stage2_samples_by_site_year(samples)
        ds = GroupedIntervalEventDataset(groups, x_mean, x_std)
        loader = make_loader(ds, 1, shuffle=False, collate_fn=collate_grouped_stage2)
        df = forward_extract_mu(model, loader, groups, device)
        return df, summarize(df, label)

    print("\n===== [1/3] BASELINE inference (no ablation) =====")
    df_base, s_base = run_inference(test_s2, "BASELINE (no ablation)")

    print("\n===== [2/3] PHENOLOGY-4 → train_mean =====")
    test_s2_pheno = apply_ablation(test_s2, pheno_idx, pheno_means, D_raw)
    df_pheno, s_pheno = run_inference(test_s2_pheno, "PHENO_4 → train_mean")

    print("\n===== [3/3] ROLLING-3 → train_mean =====")
    test_s2_roll = apply_ablation(test_s2, roll_idx, roll_means, D_raw)
    df_roll, s_roll = run_inference(test_s2_roll, "ROLL_3 → train_mean")

    rows = [s_base, s_pheno, s_roll]
    # External reference numbers (baseline D=15 reference per user)
    rows.append({
        "label": "[ref] baseline D=15 (year-split, full)",
        "n_interval": 842,
        "mu_mean": 187.20, "mu_std": 5.05,
        "L_mean": 201.47, "L_std": 18.03,
        "mean_mu_minus_L": -14.26, "median_mu_minus_L": -13.00,
        "mean_abs_mu_minus_L": 18.19,
        "mu_std_over_L_std": 0.280,
    })

    out_df = pd.DataFrame(rows)
    cols = ["label","n_interval","mu_mean","mu_std","L_mean","L_std",
            "mean_mu_minus_L","median_mu_minus_L","mean_abs_mu_minus_L","mu_std_over_L_std"]
    out_df = out_df[cols]
    pd.set_option("display.float_format", lambda v: f"{v:.3f}")
    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 30)
    print("\n=================== SUMMARY ===================")
    print(out_df.to_string(index=False))


if __name__ == "__main__":
    main()
