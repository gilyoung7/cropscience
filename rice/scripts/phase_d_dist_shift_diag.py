"""
Phase D — Train vs Test distribution shift + mu × feature correlation diag.

Measurement only. No training.

Step 1: Build year-split season samples (run=6 sheath_blight), extract per
(site_id, year) values for the run-6 phenology and rolling-extension features:
    pheno_4 = best_suitability, best_months, offset_days, window_idx
    roll_3  = rain_14d_sum, wind_7d_mean, wind_7d_max
- Per site-year value: last DOY value for phenology (~static after ffill);
  mean over season for rolling.

Step 2: Run Stage 2 (final_aw15) forward inference on the test set's grouped
nowcast samples to obtain mu_BK; aggregate to per-site-year (mean mu over
valid t* candidates).

Step 3: Report
    - train vs test KS test for each feature
    - Pearson r + Spearman ρ of test mu against each feature
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import ks_2samp, pearsonr, spearmanr

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import (
    GroupedIntervalEventDataset,
    build_stage2_nowcast_samples,
    compute_norm_stats,
    group_stage2_samples_by_site_year,
    split_samples,
)
from rice.scripts.common import collate_grouped_stage2, make_loader
from rice.src.data_pipeline import (
    add_site_static_latlon,
    aggregate_obs_daily_max,
    load_daily_preprocessed,
    load_obs,
    make_daily_feature_frame,
    make_obs_meta,
    merge_pheno_daily_ffill,
)
from rice.src.dataset import build_train_frame, slice_season
from rice.src.labels import build_interval_labels_from_doy, filter_labels_by_gap
from rice.src.model import HierarchicalCausalHazardTransformer


PHENO_4 = ["best_suitability", "best_months", "offset_days", "window_idx"]
ROLL_3 = ["rain_14d_sum", "wind_7d_mean", "wind_7d_max"]


def per_site_year_table(train_df: pd.DataFrame, doy_start: int, doy_end: int) -> pd.DataFrame:
    """
    Build (site_id, year) -> {feature_value} table.
    Phenology: last value in season (after ffill). Rolling: season mean.
    """
    df = slice_season(train_df, doy_start, doy_end).copy()
    rows = []
    for (site, year), sub in df.groupby(["site_id", "year"], sort=False):
        sub = sub.sort_values("doy")
        rec = {"site_id": str(site), "year": int(year)}
        for c in PHENO_4:
            rec[c] = float(sub[c].iloc[-1]) if c in sub.columns and len(sub) else float("nan")
        for c in ROLL_3:
            rec[c] = float(sub[c].mean(skipna=True)) if c in sub.columns and len(sub) else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def build_split_tables(args, doy_start: int, doy_end: int):
    daily = load_daily_preprocessed(C.PATH_DAILY)
    obs = load_obs(C.PATH_OBS)
    obs2 = aggregate_obs_daily_max(obs)
    labels = build_interval_labels_from_doy(
        obs2, threshold=C.THRESHOLD, season_start_doy=C.SEASON_START_DOY, season_end_doy=C.SEASON_END_DOY
    )
    labels = filter_labels_by_gap(labels, doy_start, doy_end, C.MAX_GAP)
    obs_meta = make_obs_meta(obs2, doy_start, doy_end)
    daily_feat, _ = make_daily_feature_frame(daily)
    T = doy_end - doy_start + 1
    train_df = build_train_frame(daily_feat, labels, obs_meta, T=T)
    train_df = add_site_static_latlon(train_df, obs)
    train_df = merge_pheno_daily_ffill(train_df, obs)

    table = per_site_year_table(train_df, doy_start, doy_end)
    # Year-based assignment matching split_samples(split_mode='year')
    val_year = args.val_year
    tmin, tmax = args.test_year_min, args.test_year_max
    table["split"] = "train"
    table.loc[table["year"] == val_year, "split"] = "val"
    table.loc[(table["year"] >= tmin) & (table["year"] <= tmax), "split"] = "test"
    # Drop years not in [DOY year range, e.g. ymin=2002]; we restrict to >=2002
    table = table[table["year"] >= 2002].copy()
    # Restrict train to <= val_year-1 (no leakage)
    table = table[~((table["split"] == "train") & (table["year"] >= val_year))].copy()
    return table


def stage2_mu_per_site_year(args, ckpt_path: Path, device: torch.device) -> pd.DataFrame:
    from rice.scripts.run_eval import build_samples_for_run
    _, get_feature_cols = resolve_pest(args.pest)
    _, _, T2, samples2 = build_samples_for_run(args.run, get_feature_cols)

    ckpt2 = torch.load(ckpt_path, map_location="cpu")
    C.DOY_START = int(ckpt2.get("doy_start", C.DOY_START))
    C.DOY_END = int(ckpt2.get("doy_end", C.DOY_END))

    train_s2_base, val_s2_base, test_s2_base = split_samples(
        samples2, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )
    print(f"[split] train={len(train_s2_base)} val={len(val_s2_base)} test={len(test_s2_base)}")

    nc_window = int(ckpt2.get("stage2_nowcast_window", 28))
    nc_stride = int(ckpt2.get("stage2_nowcast_stride", 1))
    nc_tstart = ckpt2.get("stage2_nowcast_tstar_start", None)
    nc_only_pre = bool(int(ckpt2.get("stage2_nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt2.get("stage2_nowcast_event_time_proxy", "r"))
    nc_req = bool(int(ckpt2.get("stage2_nowcast_require_tstar_before_L", 0)))

    test_s2 = build_stage2_nowcast_samples(
        test_s2_base, window=nc_window, stride=nc_stride,
        tstar_start=nc_tstart, only_pre_event=nc_only_pre,
        event_time_proxy=nc_proxy, require_tstar_before_L=nc_req,
    )
    print(f"[nowcast] test samples (per-tstar) = {len(test_s2)}")

    x_mean, x_std = compute_norm_stats(train_s2_base)
    test_groups = group_stage2_samples_by_site_year(test_s2)
    test_ds = GroupedIntervalEventDataset(test_groups, x_mean, x_std)
    loader = make_loader(test_ds, 1, shuffle=False, collate_fn=collate_grouped_stage2)

    stage2_pmf_mode = str(ckpt2.get("stage2_pmf_mode", "hazard"))
    if stage2_pmf_mode != "gaussian":
        raise SystemExit(f"phase_d expects gaussian Stage2 ckpt; got pmf_mode={stage2_pmf_mode}")

    d_model = int(ckpt2.get("d_model", C.D_MODEL))
    n_head = int(ckpt2.get("n_head", C.N_HEAD))
    n_layers = int(ckpt2.get("n_layers", C.N_LAYERS))
    model = HierarchicalCausalHazardTransformer(
        d_in=int(test_ds[0][0].shape[-1]),
        d_model=d_model, nhead=n_head, num_layers=n_layers,
        num_tstar_layers=int(ckpt2.get("stage2_tstar_layers", 1)),
        dropout=C.DROPOUT, max_len=C.MAX_LEN, max_tstar_len=512,
        use_tstar_scalar_pos=bool(int(ckpt2.get("stage2_use_tstar_scalar_pos", 0))),
    ).to(device)
    print(f"[stage2 model] d_model={d_model} n_head={n_head} n_layers={n_layers}")
    model.time_chunk_size = int(ckpt2.get("stage2_time_chunk_size", 64))
    model.conditional_survival = bool(int(ckpt2.get("stage2_conditional_survival", 0)))
    model.pmf_mode = "gaussian"
    model.gaussian_sigma = float(ckpt2.get("stage2_pmf_sigma", 5.0))
    model.gaussian_mu_max = float(ckpt2.get("stage2_pmf_mu_max", 0.0))
    model.asym_weight = float(ckpt2.get("stage2_pmf_asym_weight", 15.0))
    model.right_weight = float(ckpt2.get("stage2_pmf_right_weight", 0.3))
    model.target_offset = float(ckpt2.get("stage2_pmf_target_offset", 5.0))

    d2 = ckpt2["trained_states"][0]
    model.load_state_dict(d2["state_dict"], strict=False)
    model.eval()

    rows = []
    with torch.no_grad():
        gi = 0
        for X, L, R, ctype, tstar, valid_mask in loader:
            X = X.to(device)
            tstar_t = tstar.to(device)
            valid_t = valid_mask.to(device)
            _ = model(X, tstar=tstar_t, valid_mask=valid_t)
            mu_BK = getattr(model, "_last_mu_BK", None)
            if mu_BK is None:
                raise SystemExit("model._last_mu_BK is None after forward (gaussian path expected)")
            mu_BK_np = mu_BK.detach().cpu().numpy()
            v_np = valid_mask.cpu().numpy().astype(bool)
            L_np = L.cpu().numpy().astype(int)
            ctype_np = ctype.cpu().numpy().astype(int)
            B, K = mu_BK_np.shape
            for bi in range(B):
                g = test_groups[gi + bi]
                mu_vals = mu_BK_np[bi][v_np[bi]]
                if len(mu_vals) == 0:
                    continue
                L_event = None
                for ki in range(K):
                    if v_np[bi, ki] and ctype_np[bi, ki] == 0:
                        L_event = int(L_np[bi, ki])
                        break
                rows.append({
                    "site_id": str(g["site_id"]),
                    "year": int(g["year"]),
                    "mu_mean": float(mu_vals.mean()),
                    "mu_median": float(np.median(mu_vals)),
                    "mu_first": float(mu_vals[0]),
                    "mu_last": float(mu_vals[-1]),
                    "n_tstar": int(len(mu_vals)),
                    "L": (float(L_event) if L_event is not None else float("nan")),
                    "censor": ("interval" if L_event is not None else "right"),
                })
            gi += B
    return pd.DataFrame(rows)


def ks_block(tab: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    rows = []
    train = tab[tab["split"] == "train"]
    test = tab[tab["split"] == "test"]
    for c in feature_cols:
        a = train[c].to_numpy(dtype=float)
        b = test[c].to_numpy(dtype=float)
        a = a[np.isfinite(a)]
        b = b[np.isfinite(b)]
        if len(a) < 3 or len(b) < 3:
            rows.append({"feature": c, "n_train": len(a), "n_test": len(b),
                         "train_mean": np.nan, "train_std": np.nan,
                         "test_mean": np.nan, "test_std": np.nan,
                         "mean_shift": np.nan, "KS_D": np.nan, "KS_p": np.nan})
            continue
        ks = ks_2samp(a, b)
        rows.append({
            "feature": c,
            "n_train": len(a), "n_test": len(b),
            "train_mean": float(np.mean(a)), "train_std": float(np.std(a, ddof=0)),
            "test_mean": float(np.mean(b)),  "test_std": float(np.std(b, ddof=0)),
            "train_q25": float(np.quantile(a, 0.25)), "train_q75": float(np.quantile(a, 0.75)),
            "test_q25": float(np.quantile(b, 0.25)),  "test_q75": float(np.quantile(b, 0.75)),
            "mean_shift": float(np.mean(b) - np.mean(a)),
            "KS_D": float(ks.statistic), "KS_p": float(ks.pvalue),
        })
    return pd.DataFrame(rows)


def corr_block(merged: pd.DataFrame, feature_cols: list[str], mu_col: str = "mu_mean") -> pd.DataFrame:
    rows = []
    mu = merged[mu_col].to_numpy(dtype=float)
    for c in feature_cols:
        f = merged[c].to_numpy(dtype=float)
        mask = np.isfinite(mu) & np.isfinite(f)
        n = int(mask.sum())
        if n < 5 or np.std(mu[mask]) < 1e-12 or np.std(f[mask]) < 1e-12:
            rows.append({"feature": c, "n": n, "pearson_r": np.nan, "spearman_rho": np.nan})
            continue
        r, _ = pearsonr(mu[mask], f[mask])
        rho, _ = spearmanr(mu[mask], f[mask])
        rows.append({"feature": c, "n": n, "pearson_r": float(r), "spearman_rho": float(rho), "abs_r": abs(float(r))})
    return pd.DataFrame(rows)


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

    # Apply pest config (sets C.DOY_START / C.DOY_END / paths)
    _ = resolve_pest(args.pest)

    # Match the trained DOY range
    ckpt2 = torch.load(args.stage2_ckpt, map_location="cpu")
    doy_start = int(ckpt2.get("doy_start", C.DOY_START))
    doy_end = int(ckpt2.get("doy_end", C.DOY_END))
    print(f"[ckpt] DOY range = [{doy_start}, {doy_end}]")

    print("[step 1/3] Build per-site-year feature table…")
    table = build_split_tables(args, doy_start, doy_end)
    print(table.groupby("split").size())

    feature_cols = PHENO_4 + ROLL_3
    ks_df = ks_block(table, feature_cols)
    print("\n=== Train vs Test KS Test (per site-year) ===")
    pd.set_option("display.float_format", lambda x: f"{x:.4f}")
    pd.set_option("display.width", 240)
    print(ks_df.to_string(index=False))

    print("\n[step 2/3] Stage 2 forward → per-site-year mu (test set)…")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mu_df = stage2_mu_per_site_year(args, Path(args.stage2_ckpt), device)
    print(f"[mu] rows={len(mu_df)}  interval={int((mu_df['censor']=='interval').sum())} "
          f"right={int((mu_df['censor']=='right').sum())}")

    print("\n[step 3/3] mu × feature correlation (test)…")
    test_tab = table[table["split"] == "test"][["site_id", "year"] + feature_cols]
    merged = mu_df.merge(test_tab, on=["site_id", "year"], how="inner")
    print(f"[merge] rows={len(merged)}")

    print("\n=== mu_mean × feature correlations (test, ALL site-years) ===")
    all_corr = corr_block(merged, feature_cols, mu_col="mu_mean")
    print(all_corr.to_string(index=False))

    interval_only = merged[merged["censor"] == "interval"].copy()
    print(f"\n=== mu_mean × feature correlations (test, interval-only n={len(interval_only)}) ===")
    int_corr = corr_block(interval_only, feature_cols, mu_col="mu_mean")
    print(int_corr.to_string(index=False))

    # also compare mu vs L to give Δ context
    if len(interval_only) > 0:
        mu = interval_only["mu_mean"].to_numpy(dtype=float)
        L = interval_only["L"].to_numpy(dtype=float)
        mask = np.isfinite(mu) & np.isfinite(L)
        if mask.sum() >= 3:
            r, _ = pearsonr(mu[mask], L[mask])
            print(
                f"\n[ref] mu_mean vs true_L (interval, n={int(mask.sum())}): "
                f"Pearson r={r:+.4f}  mean(mu-L)={(mu[mask]-L[mask]).mean():+.2f}  "
                f"mu_std={mu[mask].std(ddof=0):.2f}  L_std={L[mask].std(ddof=0):.2f}  "
                f"ratio={mu[mask].std(ddof=0)/max(L[mask].std(ddof=0),1e-9):.3f}"
            )


if __name__ == "__main__":
    main()
