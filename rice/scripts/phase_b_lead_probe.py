"""Phase B linear-probe diagnostic on encoder representation z.

For each given Stage-2 ckpt:
  1. Rebuilds the train+val+test cohort the same way run_train would (using
     nowcast / dispatch / norm settings from ckpt meta).
  2. Forwards Stage 2 and captures the post-encoder, pre-head representation
     z (stashed as model._last_z_BKD in model.forward).
  3. Picks one z per (site, year): the cell at the alert_t row (smallest
     tstar >= alert_t_rel from the dispatch feature CSV). Pre-alert cells
     are skipped — only post-alert cells carry the lead-target signal.
  4. Fits a Ridge regression z -> lead_to_L on train+val, evaluates on test.
  5. Compares against three baselines:
       - constant baseline (predict train mean lead)
       - the model's own predicted_lead (lead_from_alert ckpts only)
       - the model's own mu vs true_L_DOY

Decision rule:
  - probe corr >> model corr: encoder has timing signal that the head
    cannot read -> head reset is a plausible next step.
  - probe corr ~ 0 too: the representation itself lacks per-sample lead
    signal -> head reset / uncond full retraining unlikely to help.

Usage:
    .venv/bin/python -m rice.scripts.phase_b_lead_probe \\
        --ckpts lead_v3=<path>,uncond=<path> \\
        --dispatch_feature_csv outputs_dispatch_R088_features_per_sy.csv \\
        --out_table outputs_phase_b_lead_probe_table.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score

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
from rice.scripts.stage1_confidence_utils import (
    DISPATCH_FEATURE_NAMES,
    DISPATCH_TOTAL_CHANNELS,
    append_dispatch_confidence_to_samples,
    load_dispatch_feature_table,
)


# Subgroup feature indices into the per-sy features ndarray (14 dims).
_FEAT_IDX = {name: i for i, name in enumerate(DISPATCH_FEATURE_NAMES)}


def _corr(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return float("nan"), float("nan")
    if float(np.std(x[m])) == 0.0 or float(np.std(y[m])) == 0.0:
        return float("nan"), float("nan")
    p = float(pearsonr(x[m], y[m])[0])
    s = float(spearmanr(x[m], y[m]).correlation)
    return p, s


def extract_z_table(ckpt_path: Path, label: str, args, device) -> tuple[pd.DataFrame, int]:
    """Forward train+val+test for one ckpt; capture z at the post-alert cell
    closest to alert_t_rel. Returns (DataFrame, d_model).
    """
    print(f"\n========== probe ckpt: label={label!r}  path={ckpt_path}")
    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)

    default_doy_start = getattr(C, "DOY_START", 60)
    default_doy_end = getattr(C, "DOY_END", 300)
    doy_start = int(ckpt.get("doy_start", default_doy_start))
    C.DOY_START = doy_start
    C.DOY_END = int(ckpt.get("doy_end", default_doy_end))

    nc_window = int(ckpt.get("stage2_nowcast_window", 28))
    nc_stride = int(ckpt.get("stage2_nowcast_stride", 1))
    nc_tstart = ckpt.get("stage2_nowcast_tstar_start", None)
    nc_only_pre = bool(int(ckpt.get("stage2_nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("stage2_nowcast_event_time_proxy", "r"))
    nc_req = bool(int(ckpt.get("stage2_nowcast_require_tstar_before_L", 0)))

    print(f"  [probe] doy_start={doy_start}  nowcast(W={nc_window}, stride={nc_stride}, "
          f"only_pre_event={nc_only_pre}, proxy={nc_proxy})")

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples = build_samples_for_run(args.run, get_feature_cols)

    has_dispatch = bool(ckpt.get("stage2_dispatch_features_added", False))
    if has_dispatch:
        d_csv = ckpt.get("stage2_dispatch_feature_csv") or args.dispatch_feature_csv
        d_mode = str(ckpt.get("stage2_dispatch_feature_mode", "causal"))
        d_miss = float(ckpt.get("stage2_dispatch_feature_missing_value", 0.0))
        conf_map_ckpt = load_dispatch_feature_table(d_csv) if d_csv else {}
        append_dispatch_confidence_to_samples(
            samples, conf_map_ckpt, doy_start=doy_start,
            mode=d_mode, missing_value=d_miss,
        )
        print(f"  [probe] dispatch features appended; "
              f"d_in={int(samples[0]['X'].shape[1])}  csv={d_csv}")

    # Cohort + alert_t lookup (always from --dispatch_feature_csv so all ckpts
    # share the same cohort and the same alert_t per (site, year)).
    cohort_map = load_dispatch_feature_table(args.dispatch_feature_csv)
    alert_t_rel_by_sy: dict = {
        sy: int(v["alert_tstar_doy"]) - doy_start + 1
        for sy, v in cohort_map.items()
    }
    print(f"  [probe] cohort_map size = {len(cohort_map)} (alerted site-years across all splits)")

    train_s, val_s, test_s = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )

    nc_kw = dict(window=nc_window, stride=nc_stride, tstar_start=nc_tstart,
                 only_pre_event=nc_only_pre, event_time_proxy=nc_proxy,
                 require_tstar_before_L=nc_req)

    x_mean, x_std = compute_norm_stats(train_s)
    if has_dispatch and bool(ckpt.get("stage2_dispatch_channels_raw", False)):
        n_disp = int(DISPATCH_TOTAL_CHANNELS)
        d_total = int(x_mean.shape[0])
        disp_start = d_total - n_disp
        if disp_start >= 0:
            x_mean[disp_start:] = 0.0
            x_std[disp_start:] = 1.0
            print(f"  [probe] norm RAW on dispatch ch [{disp_start}:{d_total}]")

    # Build the model with the same shape as ckpt
    d_in_full = int(samples[0]["X"].shape[1])
    d_model = int(ckpt.get("d_model", C.D_MODEL))
    n_head = int(ckpt.get("n_head", C.N_HEAD))
    n_layers = int(ckpt.get("n_layers", C.N_LAYERS))
    phen_bias_head = bool(int(ckpt.get("stage2_phenology_bias_head", 0)))
    if phen_bias_head:
        print(f"  [probe] WARNING: ckpt has stage2_phenology_bias_head=1 but probe "
              f"does not load pheno features; ignoring phen_head (mu=mu_temporal).")
    model = HierarchicalCausalHazardTransformer(
        d_in=d_in_full, d_model=d_model, nhead=n_head, num_layers=n_layers,
        num_tstar_layers=int(ckpt.get("stage2_tstar_layers", 1)),
        dropout=C.DROPOUT, max_len=C.MAX_LEN, max_tstar_len=512,
        use_tstar_scalar_pos=bool(int(ckpt.get("stage2_use_tstar_scalar_pos", 0))),
        phenology_bias_head=False, phenology_dim=4,
        phenology_hidden=int(ckpt.get("stage2_phenology_hidden", 8)),
    ).to(device)
    model.time_chunk_size = int(ckpt.get("stage2_time_chunk_size", 64))
    model.conditional_survival = bool(int(ckpt.get("stage2_conditional_survival", 0)))
    model.pmf_mode = "gaussian"
    model.gaussian_sigma = float(ckpt.get("stage2_pmf_sigma", 5.0))
    model.gaussian_mu_max = float(ckpt.get("stage2_pmf_mu_max", 0.0))
    model.mu_mode = str(ckpt.get("stage2_pmf_mu_mode", "absolute"))
    model.lead_min = float(ckpt.get("stage2_pmf_lead_min", 7.0))
    model.lead_max = float(ckpt.get("stage2_pmf_lead_max", 75.0))
    model.alert_tstar_feat_idx = int(ckpt.get("stage2_pmf_alert_tstar_feat_idx", -1))
    model.doy_start = doy_start
    model.lead_strict_alert_check = False  # probe — never abort on empty batch
    model.load_state_dict(ckpt["trained_states"][0]["state_dict"], strict=False)
    model.eval()
    print(f"  [probe] model: d_in={d_in_full}  d_model={d_model}  "
          f"mu_mode={model.mu_mode}")

    out_rows: list = []
    n_skip_no_post = 0
    for tag, seas in [("train", train_s), ("val", val_s), ("test", test_s)]:
        seas_alerted = [s for s in seas
                        if (str(s["site_id"]), int(s["year"])) in cohort_map]
        if not seas_alerted:
            continue
        nc_s = build_stage2_nowcast_samples(seas_alerted, **nc_kw)
        groups = group_stage2_samples_by_site_year(nc_s)
        ds = GroupedIntervalEventDataset(groups, x_mean, x_std)
        loader = make_loader(ds, 1, shuffle=False, collate_fn=collate_grouped_stage2)
        n_split_kept = 0
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
                z_BKD = model._last_z_BKD.detach().cpu().numpy()    # (B, K, d_model)
                mu_BK = model._last_mu_BK.detach().cpu().numpy()    # (B, K)
                lead_BK = getattr(model, "_last_lead_BK", None)
                lead_BK = lead_BK.cpu().numpy() if lead_BK is not None else None
                v_np = valid_mask.cpu().numpy().astype(bool)
                L_np = L.cpu().numpy().astype(int)
                R_np = R.cpu().numpy().astype(int)
                c_np = ctype.cpu().numpy().astype(int)
                ts_np = tstar.cpu().numpy().astype(int)
                B, K = mu_BK.shape
                for bi in range(B):
                    g = groups[gi + bi]
                    sy = (str(g["site_id"]), int(g["year"]))
                    alert_t_rel = alert_t_rel_by_sy.get(sy)
                    if alert_t_rel is None:
                        continue
                    # Pick the smallest tstar >= alert_t_rel among valid cells
                    best_ki = None
                    best_diff = None
                    for ki in range(K):
                        if not v_np[bi, ki]:
                            continue
                        t_here = int(ts_np[bi, ki])
                        if t_here < alert_t_rel:
                            continue
                        diff = t_here - alert_t_rel
                        if best_diff is None or diff < best_diff:
                            best_diff = diff
                            best_ki = ki
                    if best_ki is None:
                        n_skip_no_post += 1
                        continue
                    ki = best_ki
                    L_abs = int(L_np[bi, ki]) + doy_start - 1
                    R_abs = int(R_np[bi, ki]) + doy_start - 1
                    alert_doy = alert_t_rel + doy_start - 1
                    lead_to_L = float(L_abs - alert_doy)
                    # Look up subgroup features from the dispatch CSV.
                    feats = cohort_map[sy]["features"]
                    sg_with_history = float(feats[_FEAT_IDX["with_history"]])
                    sg_branch = float(feats[_FEAT_IDX["dispatch_branch"]])
                    sg_score_margin = float(feats[_FEAT_IDX["score_margin"]])
                    sg_sot_margin = float(feats[_FEAT_IDX["score_over_tau_margin"]])
                    row = {
                        "split": tag,
                        "site": sy[0], "year": sy[1],
                        "ctype": int(c_np[bi, ki]),
                        "alert_doy": int(alert_doy),
                        "tstar_picked": int(ts_np[bi, ki]),
                        "L": int(L_abs), "R": int(R_abs),
                        "lead_to_L": lead_to_L,
                        "mu_abs": float(mu_BK[bi, ki]) + doy_start - 1,
                        "lead_pred": (float(lead_BK[bi, ki])
                                       if lead_BK is not None else float("nan")),
                        "sg_with_history": sg_with_history,
                        "sg_branch": sg_branch,
                        "sg_score_margin": sg_score_margin,
                        "sg_score_over_tau_margin": sg_sot_margin,
                    }
                    z_vec = z_BKD[bi, ki, :]
                    for di in range(d_model):
                        row[f"z_{di}"] = float(z_vec[di])
                    out_rows.append(row)
                    n_split_kept += 1
                gi += B
        print(f"  [probe] split={tag:>5}: kept={n_split_kept} / alerted_sy={len(seas_alerted)}")
    if n_skip_no_post:
        print(f"  [probe] skipped {n_skip_no_post} sy: no valid cell with tstar >= alert_t_rel")
    df = pd.DataFrame(out_rows)
    df["ckpt_label"] = label
    print(f"  [probe] total rows={len(df)}")
    return df, d_model


def _probe_block(tr: pd.DataFrame, te: pd.DataFrame, d_model: int,
                  ridge_alpha: float, tag: str) -> dict:
    """Compute Ridge probe + constant baseline + model-head reference on one
    (train+val, test) slice. Returns a dict of metrics; also prints them.
    """
    z_cols = [f"z_{i}" for i in range(d_model)]
    out = {"slice": tag, "n_trv": int(len(tr)), "n_te": int(len(te))}
    print(f"\n  -- {tag}  n_train+val={len(tr)}  n_test={len(te)}")
    if len(tr) < 5 or len(te) < 5:
        print("    [skip] too few rows for stable probe")
        return out
    Xtr = tr[z_cols].astype(float).values
    ytr = tr["lead_to_L"].astype(float).values
    Xte = te[z_cols].astype(float).values
    yte = te["lead_to_L"].astype(float).values
    Lte = te["L"].astype(float).values
    alert_te = te["alert_doy"].astype(float).values

    out["lead_te_mean"] = float(yte.mean())
    out["lead_te_std"] = float(yte.std(ddof=0))
    print(f"    target lead_to_L test: mean={out['lead_te_mean']:.2f}  "
          f"std={out['lead_te_std']:.2f}")

    rm = Ridge(alpha=ridge_alpha)
    rm.fit(Xtr, ytr)
    pred_lead = rm.predict(Xte)
    p_pl, s_pl = _corr(pred_lead, yte)
    mu_probe = alert_te + pred_lead
    p_mu, s_mu = _corr(mu_probe, Lte)
    mae = float(mean_absolute_error(yte, pred_lead))
    r2 = float(r2_score(yte, pred_lead))
    out.update({
        "probe_corr_pred_lead_pearson": p_pl,
        "probe_corr_pred_lead_spearman": s_pl,
        "probe_corr_alert_plus_pred_L_pearson": p_mu,
        "probe_corr_alert_plus_pred_L_spearman": s_mu,
        "probe_MAE": mae, "probe_R2": r2,
    })
    print(f"    [probe Ridge] corr(pred_lead, true_lead)={p_pl:+.4f}  "
          f"corr(alert+pred, L)={p_mu:+.4f}  MAE={mae:.2f}  R^2={r2:+.4f}")

    mean_tr = float(ytr.mean())
    mae_c = float(np.mean(np.abs(yte - mean_tr)))
    mu_c = alert_te + mean_tr
    p_muc, _ = _corr(mu_c, Lte)
    out.update({
        "const_train_mean_lead": mean_tr,
        "const_MAE": mae_c,
        "const_corr_alert_plus_const_L_pearson": p_muc,
    })
    print(f"    [const]       train_mean_lead={mean_tr:.2f}  MAE={mae_c:.2f}  "
          f"corr(alert+const, L)={p_muc:+.4f}")

    if te["lead_pred"].notna().any():
        ml = te["lead_pred"].astype(float).values
        p_ml, s_ml = _corr(ml, yte)
        mae_ml = float(np.nanmean(np.abs(yte - ml)))
        out.update({
            "model_lead_corr_pearson": p_ml,
            "model_lead_corr_spearman": s_ml,
            "model_lead_MAE": mae_ml,
        })
        print(f"    [model head]  corr(model_lead, true_lead)={p_ml:+.4f}  "
              f"MAE={mae_ml:.2f}   (reference only)")
    mu_model = te["mu_abs"].astype(float).values
    p_mm, _ = _corr(mu_model, Lte)
    out["model_mu_corr_L_pearson"] = p_mm
    print(f"    [model mu]    corr(model_mu, L)={p_mm:+.4f}   (reference only)")
    return out


def probe_subgroups(df: pd.DataFrame, d_model: int, label: str,
                     ridge_alpha: float) -> list[dict]:
    """Phase B subgroup probe. Fixes cohort to interval-censored only.
    For each subgroup definition, fits Ridge on the train+val slice OF THAT
    SUBGROUP, evaluates on test slice OF THAT SUBGROUP.
    """
    print(f"\n## subgroup probe  ckpt={label!r}  (interval-censored cohort only)")
    df = df[df["ctype"] == 0].copy()
    if df.empty:
        print("  [skip] no interval-censored rows")
        return []
    tr_all = df[df["split"].isin(["train", "val"])]
    te_all = df[df["split"] == "test"]
    print(f"  fixed cohort: ctype=0 (interval)  n_trv={len(tr_all)}  n_te={len(te_all)}")

    results: list[dict] = []

    # (1) with_history
    print("\n  --- subgroup: with_history ---")
    for v, name in [(1.0, "with_history=1"), (0.0, "with_history=0")]:
        tr = tr_all[tr_all["sg_with_history"] == v]
        te = te_all[te_all["sg_with_history"] == v]
        r = _probe_block(tr, te, d_model, ridge_alpha, name)
        r.update({"ckpt": label, "subgroup_dim": "with_history",
                   "subgroup_value": ("1" if v == 1.0 else "0")})
        results.append(r)

    # (2) dispatch_branch  (encoded as 1.0=D / 0.0=A in features array)
    print("\n  --- subgroup: dispatch_branch ---")
    for v, name in [(1.0, "branch=D"), (0.0, "branch=A")]:
        tr = tr_all[tr_all["sg_branch"] == v]
        te = te_all[te_all["sg_branch"] == v]
        r = _probe_block(tr, te, d_model, ridge_alpha, name)
        r.update({"ckpt": label, "subgroup_dim": "dispatch_branch",
                   "subgroup_value": ("D" if v == 1.0 else "A")})
        results.append(r)

    # (3) score_over_tau_margin median split — median over (train+val) only
    sot_med = float(tr_all["sg_score_over_tau_margin"].median())
    print(f"\n  --- subgroup: score_over_tau_margin  (median={sot_med:.4f}, "
          f"computed on train+val) ---")
    for cmp, name in [(">=", f"sot_high (>= {sot_med:.4f})"),
                       ("<",  f"sot_low  (< {sot_med:.4f})")]:
        if cmp == ">=":
            tr = tr_all[tr_all["sg_score_over_tau_margin"] >= sot_med]
            te = te_all[te_all["sg_score_over_tau_margin"] >= sot_med]
            sval = "high"
        else:
            tr = tr_all[tr_all["sg_score_over_tau_margin"] < sot_med]
            te = te_all[te_all["sg_score_over_tau_margin"] < sot_med]
            sval = "low"
        r = _probe_block(tr, te, d_model, ridge_alpha, name)
        r.update({"ckpt": label, "subgroup_dim": "score_over_tau_margin",
                   "subgroup_value": sval, "split_median_trv": sot_med})
        results.append(r)

    # (4) score_margin (=D_score - A_score) median split — train+val median
    sm_med = float(tr_all["sg_score_margin"].median())
    print(f"\n  --- subgroup: score_margin (=D-A)  (median={sm_med:.4f}, "
          f"computed on train+val) ---")
    for cmp, name in [(">=", f"score_margin_high (>= {sm_med:.4f})"),
                       ("<",  f"score_margin_low  (< {sm_med:.4f})")]:
        if cmp == ">=":
            tr = tr_all[tr_all["sg_score_margin"] >= sm_med]
            te = te_all[te_all["sg_score_margin"] >= sm_med]
            sval = "high"
        else:
            tr = tr_all[tr_all["sg_score_margin"] < sm_med]
            te = te_all[te_all["sg_score_margin"] < sm_med]
            sval = "low"
        r = _probe_block(tr, te, d_model, ridge_alpha, name)
        r.update({"ckpt": label, "subgroup_dim": "score_margin",
                   "subgroup_value": sval, "split_median_trv": sm_med})
        results.append(r)

    return results


def probe_one(df: pd.DataFrame, d_model: int, label: str, ridge_alpha: float,
               cohort_filter: str = "all") -> None:
    """Run Ridge probe + baselines on a single ckpt's df.
    cohort_filter='all' uses every row; 'interval' restricts to interval-censored
    (true L observed) which is the only cohort where lead_to_L is well-defined
    in the strict sense.
    """
    if cohort_filter == "interval":
        df = df[df["ctype"] == 0].copy()
    z_cols = [f"z_{i}" for i in range(d_model)]
    tr = df[df["split"].isin(["train", "val"])]
    te = df[df["split"] == "test"]
    print(f"\n## probe  ckpt={label!r}  cohort_filter={cohort_filter}  "
          f"d_model={d_model}  n_train+val={len(tr)}  n_test={len(te)}")
    if tr.empty or te.empty:
        print("  [skip] empty cohort")
        return
    Xtr = tr[z_cols].astype(float).values
    ytr = tr["lead_to_L"].astype(float).values
    Xte = te[z_cols].astype(float).values
    yte = te["lead_to_L"].astype(float).values
    Lte = te["L"].astype(float).values
    alert_te = te["alert_doy"].astype(float).values

    print(f"  [target] lead_to_L test: mean={yte.mean():.2f}  std={yte.std(ddof=0):.2f}  "
          f"min={yte.min():.1f}  max={yte.max():.1f}")

    # Ridge probe
    rm = Ridge(alpha=ridge_alpha)
    rm.fit(Xtr, ytr)
    pred_lead = rm.predict(Xte)
    p_pl, s_pl = _corr(pred_lead, yte)
    mu_probe = alert_te + pred_lead
    p_mu, s_mu = _corr(mu_probe, Lte)
    mae = float(mean_absolute_error(yte, pred_lead))
    r2 = float(r2_score(yte, pred_lead))
    print(f"  [probe Ridge(alpha={ridge_alpha})]")
    print(f"    corr(pred_lead, true_lead): pearson={p_pl:+.4f}  spearman={s_pl:+.4f}")
    print(f"    corr(alert+pred_lead, L):   pearson={p_mu:+.4f}  spearman={s_mu:+.4f}")
    print(f"    MAE(pred_lead)={mae:.2f}  R^2={r2:+.4f}")

    # Constant baseline
    mean_lead_tr = float(ytr.mean())
    mae_c = float(np.mean(np.abs(yte - mean_lead_tr)))
    mu_c = alert_te + mean_lead_tr
    p_muc, s_muc = _corr(mu_c, Lte)
    print(f"  [constant baseline]")
    print(f"    train_mean_lead={mean_lead_tr:.2f}  MAE={mae_c:.2f}")
    print(f"    corr(alert+const, L):       pearson={p_muc:+.4f}  spearman={s_muc:+.4f}")

    # Model's own lead head
    if te["lead_pred"].notna().any():
        ml = te["lead_pred"].astype(float).values
        p_ml, s_ml = _corr(ml, yte)
        mae_ml = float(np.nanmean(np.abs(yte - ml)))
        print(f"  [model lead head]")
        print(f"    corr(model_lead, true_lead): pearson={p_ml:+.4f}  spearman={s_ml:+.4f}")
        print(f"    MAE(model_lead)={mae_ml:.2f}")

    # Model's mu vs L
    mu_model = te["mu_abs"].astype(float).values
    p_mm, s_mm = _corr(mu_model, Lte)
    print(f"  [model mu]")
    print(f"    corr(model_mu, L):           pearson={p_mm:+.4f}  spearman={s_mm:+.4f}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--ckpts", required=True,
                    help="Comma-separated label=path entries.")
    ap.add_argument("--dispatch_feature_csv", required=True,
                    help="Per-(site,year) dispatch confidence CSV — cohort + alert_t lookup.")
    ap.add_argument("--ridge_alpha", type=float, default=1.0)
    ap.add_argument("--out_table", default=None,
                    help="Optional CSV with one row per (ckpt, site, year).")
    ap.add_argument("--subgroups", action="store_true",
                    help="Run subgroup probe (interval-censored only): "
                         "with_history, dispatch_branch, score_over_tau_margin "
                         "(median split on train+val), score_margin (median "
                         "split on train+val).")
    ap.add_argument("--out_subgroup_csv", default=None,
                    help="Optional CSV to dump subgroup probe metrics.")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[probe] device={device}  ridge_alpha={args.ridge_alpha}")

    entries = []
    for part in str(args.ckpts).split(","):
        if "=" not in part:
            continue
        label, path = part.split("=", 1)
        entries.append((label.strip(), Path(path.strip())))
    if not entries:
        raise SystemExit("[abort] no valid --ckpts entries")

    results = []
    for label, path in entries:
        if not path.exists():
            print(f"[skip] ckpt missing: {path}")
            continue
        df, d_model = extract_z_table(path, label, args, device)
        results.append((label, df, d_model))

    print("\n" + "=" * 78)
    print("REPORT — all cells (any ctype)")
    print("=" * 78)
    for label, df, d_model in results:
        probe_one(df, d_model, label, args.ridge_alpha, cohort_filter="all")

    print("\n" + "=" * 78)
    print("REPORT — interval-censored only (true L observed)")
    print("=" * 78)
    for label, df, d_model in results:
        probe_one(df, d_model, label, args.ridge_alpha, cohort_filter="interval")

    if args.subgroups:
        print("\n" + "=" * 78)
        print("SUBGROUP PROBE — interval-censored cohort, fitted/evaluated within each subgroup")
        print("=" * 78)
        sg_rows: list[dict] = []
        for label, df, d_model in results:
            sg_rows.extend(probe_subgroups(df, d_model, label, args.ridge_alpha))
        if args.out_subgroup_csv and sg_rows:
            sg_df = pd.DataFrame(sg_rows)
            front = ["ckpt", "subgroup_dim", "subgroup_value", "slice",
                     "n_trv", "n_te", "lead_te_mean", "lead_te_std",
                     "probe_corr_pred_lead_pearson", "probe_corr_alert_plus_pred_L_pearson",
                     "probe_MAE", "probe_R2",
                     "const_MAE", "const_corr_alert_plus_const_L_pearson",
                     "model_lead_corr_pearson", "model_lead_MAE", "model_mu_corr_L_pearson"]
            cols = front + [c for c in sg_df.columns if c not in front]
            sg_df = sg_df[cols]
            sg_df.to_csv(args.out_subgroup_csv, index=False)
            print(f"\n# wrote subgroup probe table -> {args.out_subgroup_csv}",
                  file=sys.stderr)

    if args.out_table:
        combined = pd.concat([df for _, df, _ in results], ignore_index=True)
        combined.to_csv(args.out_table, index=False)
        print(f"\n# wrote table -> {args.out_table}  rows={len(combined)}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
