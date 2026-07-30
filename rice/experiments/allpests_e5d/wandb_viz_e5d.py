#!/usr/bin/env python
"""Log one E5d dev-protocol (pest, eval_year) result to W&B, reusing the legacy viz code.

CONTRACT (traced from rice/scripts/run_viz_interval_selector.py + run_viz_selector_all_pests_years.sh,
2026-07-29). Every key below is the legacy key, unchanged:

  config      pest, year, selector_name, sigma, sample_grid, selector_offsets   (+ E5d additions)
  images      viz/{offset_histogram, iou_sorted, lead_bias, calibration_iou, calibration_density,
                   lead_bin_bars, top_interval, worst_interval, random_interval,
                   random_interval_grid, top_pmf, worst_pmf, random_pmf}        13 keys
  tables      table/per_sample, table/metrics_by_lead_bin
  summary     final/{n_total, mean_iou, median_iou, frac_iou_gt_0_2, frac_early_or_inside_30,
                     mu_minus_mid_mean, mu_minus_mid_median, lead_days_mean, lead_days_median,
                     pest, year, selector_name}
  artifacts   run.save(per_sample_csv), run.save(lead_csv)  policy="now", non-fatal
  order       run.log(payload) -> run.summary -> run.save -> run.finish
  job_type    interval_viz_selector

SEED STRUCTURE (verified, not assumed):
  Stage-2 training  = single seed 0 (hparams train_seeds=[0]; one checkpoint_run4.pt).
  Legacy selector   = single fit, seed 0 (phase_b_stage2_offset_selector_v2_ranking.py:405-409
                      calls the trainers with no seed argument; no --seed CLI exists there).
  E5d selector      = 5 fits, seeds 0..4 -- the ONE axis the legacy structure never had.
  viz random seed   = figure sample selection only (default 0, grid uses seed+1); unrelated
                      to any model seed.

So: 1 run per (pest, eval_year) -- 24 total, matching the legacy run count. All 5 selector
seeds live inside that single run. Each of the 13 legacy image keys carries a LIST of 5
images, one per selector seed, captioned. Nothing is averaged into a "mean selected offset".
final/* is the 5-seed mean, final_std/* the 5-seed std, and raw Stage-2 (pre-selector,
uncalibrated, training seed 0) goes to final_raw/*.

W&B is opt-in and never fatal: no project -> local figures only; any upload error is caught
and the caller continues. No API key is read or stored here -- wandb picks up WANDB_API_KEY
or ~/.netrc itself.
"""
from __future__ import annotations
import importlib.util, sys, traceback
from pathlib import Path
import numpy as np, pandas as pd

from repo_paths import AP, CS, VENDOR, WS        # roots derived from this file's location
sys.path.insert(0, str(AP))
sys.path.insert(0, str(VENDOR))   # pinned deps only
import pest_paths as PP

# reuse the legacy figure/table code verbatim -- do not reimplement any of it
_LEGACY = CS / "rice/scripts/run_viz_interval_selector.py"
_spec = importlib.util.spec_from_file_location("legacy_viz", str(_LEGACY))
VZ = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(VZ)

# the 13 legacy image keys, in the legacy order
VIZ_KEYS = ["offset_histogram", "iou_sorted", "lead_bias", "calibration_iou",
            "calibration_density", "lead_bin_bars", "top_interval", "worst_interval",
            "random_interval", "random_interval_grid", "top_pmf", "worst_pmf", "random_pmf"]


def to_legacy_per_sample(pr: pd.DataFrame, selector_name: str) -> pd.DataFrame:
    """Map E5d picked rows onto the legacy per-sample schema (same 16 columns, same meaning).

    Legacy definitions preserved exactly: true_L_plus_1 = L+1, lead_days = (L+1) - mu,
    early_or_inside_30 = mu in [L+1-30, R].
    """
    d = pr.copy()
    mid = 0.5 * (d["L"] + d["R"])
    tL = d["L"].astype(float) + 1.0
    out = pd.DataFrame({
        "sample_id": d["sample_id"].astype(str),
        "site": d["sample_id"].astype(str).str.rsplit("-", n=1).str[0],
        "year": pd.to_numeric(d["sample_id"].astype(str).str.rsplit("-", n=1).str[-1],
                              errors="coerce").fillna(-1).astype(int),
        "alert_tstar": d["alert_tstar"].astype(float),
        "selected_offset": d["offset"].astype(int),
        "selector": selector_name,
        "mu": d["pred_mu"].astype(float),
        "pred_L": d["pred_L80"].astype(float).round().astype("Int64"),
        "pred_R": d["pred_R80"].astype(float).round().astype("Int64"),
        "true_L_plus_1": tL.astype(int),
        "true_R": d["R"].astype(float).astype(int),
        "true_mid": mid.astype(float),
        "iou": d["iou80_real"].astype(float) if "iou80_real" in d else np.nan,
        "mu_minus_mid": (d["pred_mu"] - mid).astype(float),
        "lead_days": (tL - d["pred_mu"]).astype(float),
    })
    out["early_or_inside_30"] = ((out["mu"] >= (out["true_L_plus_1"] - 30)) &
                                 (out["mu"] <= out["true_R"]))
    return out


def legacy_summary(df: pd.DataFrame) -> dict:
    """Byte-for-byte the legacy summary dict (run_viz_interval_selector.py:138-148)."""
    n = len(df)
    return {
        "n_total": int(n),
        "mean_iou": float(df["iou"].mean()) if n else 0.0,
        "median_iou": float(df["iou"].median()) if n else 0.0,
        "frac_iou_gt_0_2": float((df["iou"] > 0.2).mean()) if n else 0.0,
        "frac_early_or_inside_30": float(df["early_or_inside_30"].mean()) if n else 0.0,
        "mu_minus_mid_mean": float(df["mu_minus_mid"].mean()) if n else float("nan"),
        "mu_minus_mid_median": float(df["mu_minus_mid"].median()) if n else float("nan"),
        "lead_days_mean": float(df["lead_days"].mean()) if n else float("nan"),
        "lead_days_median": float(df["lead_days"].median()) if n else float("nan"),
    }


def build_figures(df: pd.DataFrame, lead_df: pd.DataFrame, title: str, out_dir: Path,
                  seed_tag: str, sigma: float, Tend: int, viz_seed: int,
                  topk=10, worstk=10, randomk=10, grid_n=50, grid_cols=2) -> dict[str, Path]:
    """Same 13 figures, same constructors, same sample-selection call as the legacy script."""
    import matplotlib.pyplot as plt
    top, worst, rand_small, rand_grid = VZ.select_top_worst_random(
        df, topk, worstk, randomk, grid_n, seed=viz_seed)
    figs = {
        "offset_histogram":     VZ.fig_offset_histogram(df, title),
        "iou_sorted":           VZ.fig_iou_sorted(df, title),
        "lead_bias":            VZ.fig_lead_bias(df, title),
        "calibration_iou":      VZ.fig_calibration_scatter(df, title, color_mode="iou"),
        "calibration_density":  VZ.fig_calibration_scatter(df, title, color_mode="density"),
        "lead_bin_bars":        VZ.fig_lead_bin_bars(lead_df, title + "  — Lead-bin IoU / MAE"),
        "top_interval":         VZ.fig_interval_rows(top, title=f"{title}  — Top-{len(top)} IoU", Tend=Tend),
        "worst_interval":       VZ.fig_interval_rows(worst, title=f"{title}  — Worst-{len(worst)} IoU", Tend=Tend),
        "random_interval":      VZ.fig_interval_rows(rand_small, title=f"{title}  — Random-{len(rand_small)}", Tend=Tend),
        "random_interval_grid": VZ.fig_interval_grid(rand_grid, title=f"{title}  — Random-{len(rand_grid)} grid",
                                                     Tend=Tend, n_cols=grid_cols),
        "top_pmf":              VZ.fig_pmf_rows(top, title=f"{title}  — Top-{len(top)} PMF", sigma=sigma, Tend=Tend),
        "worst_pmf":            VZ.fig_pmf_rows(worst, title=f"{title}  — Worst-{len(worst)} PMF", sigma=sigma, Tend=Tend),
        "random_pmf":           VZ.fig_pmf_rows(rand_small, title=f"{title}  — Random-{len(rand_small)} PMF",
                                                sigma=sigma, Tend=Tend),
    }
    paths = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, fig in figs.items():
        p = out_dir / f"fig_{name}__{seed_tag}.png"
        fig.savefig(p, dpi=130, bbox_inches="tight")
        plt.close(fig)
        paths[name] = p
    return paths


def build_dev_year(pest: str, eval_year: int, g: pd.DataFrame, DISP: dict,
                   doy_start: int, T: int):
    """Re-run the dev protocol for one year and return (per_seed, raw_metrics, cfg).

    Mirrors pest_eval.run_dev exactly -- same 50/50 val halves, same 5 selector seeds, same
    shift grid -- so the logged numbers are the ones the eval CSVs already contain.
    """
    import pest_eval as PE
    from src.selector_utils import train_selector, pick_offsets, picked_rows
    from src import diagnostics as D

    disp, clim = DISP[eval_year]
    vids = set(g[(g.year == eval_year) & (g.split == "val")].sample_id.unique())
    fit, cal = PE.half(vids, "val_fit"), PE.half(vids, "val_cal")
    if not fit or not cal:
        return [], {}, {}
    sels = [train_selector(PE.make_cand(PE.frames(g, eval_year, "val", fit), disp, clim,
                                        doy_start, T), sd) for sd in PP.SEEDS]
    best, bestv = 0, -1.0
    for dlt in PP.SHIFT_GRID:
        vc = PE.make_cand(PE.frames(g, eval_year, "val", cal, dlt), disp, clim, doy_start, T)
        n = vc.sample_id.nunique()
        v = float(np.mean([D.summarize(D.enrich(picked_rows(vc, pick_offsets(r, vc))),
                                       min(PP.OFFSETS), n)["IoU80_overall_tol0"] for r in sels]))
        if v > bestv:
            best, bestv = dlt, v

    per_seed, raw = [], {}
    for tag, dlt in (("raw", 0), ("calibrated", best)):
        tc = PE.make_cand(PE.frames(g, eval_year, "test", None, dlt), disp, clim, doy_start, T)
        m = PE.metrics_at(tc, sels, tc.sample_id.nunique())
        if tag == "raw":
            raw = {k: v for k, v in m.items() if isinstance(v, (int, float))}
            continue
        for sd, reg in zip(PP.SEEDS, sels):
            pr = D.enrich(picked_rows(tc, pick_offsets(reg, tc)))
            per_seed.append(dict(seed=sd,
                                 per_sample=to_legacy_per_sample(pr, f"coverage_aware_lgbm_seed{sd}"),
                                 metrics=m))
    cfg = dict(eval_year=eval_year,
               split=f"train<={eval_year-2} / val {eval_year-1} / test {eval_year}",
               stage2_train_seed=0, selector_seeds=list(PP.SEEDS), viz_random_seed=0,
               protocol="dev", model="E5d",
               doy_start=doy_start, doy_end=doy_start + T - 1, T=T,
               selector="coverage_aware_lightgbm", selector_n_seeds=len(PP.SEEDS),
               calibration=f"global additive mu shift, grid {PP.SHIFT_GRID[0]}..{PP.SHIFT_GRID[-1]},"
                           f" selected on val_cal; delta*={best:+d}",
               calibration_shift=best, sigma_eval=PP.SIGMA)
    f = AP / "gate_policy_observed.csv"
    if f.exists():
        gp = pd.read_csv(f)
        r = gp[(gp.pest == pest) & (gp.year == eval_year)]
        if not r.empty:
            cfg.update(stage1_gate=str(r.iloc[0]["gate"]), stage1_tau=str(r.iloc[0]["tau"]),
                       stage1_k=str(r.iloc[0]["k"]))
    return per_seed, raw, cfg


def log_e5d_dev_year(pest, eval_year, g, DISP, doy_start, T, outd, project,
                     entity=None, group=None, log=print) -> bool:
    """Convenience wrapper used by pest_eval: build + upload one (pest, eval_year) run."""
    gg = g[(g.variant == "E5d") & (g.offset.isin(PP.OFFSETS))]
    per_seed, raw, cfg = build_dev_year(pest, eval_year, gg, DISP, doy_start, T)
    if not per_seed:
        log(f"[wandb] {pest}/{eval_year}: no evaluable dev fold -- skipped")
        return False
    return log_run(pest, eval_year, per_seed, raw, cfg,
                   Path(outd) / f"wandb_{pest}_{eval_year}",
                   project=project, entity=entity, group=group)


def log_run(pest: str, eval_year: int, per_seed: list[dict], raw_metrics: dict,
            cfg_extra: dict, out_dir: Path, project: str | None, entity: str | None = None,
            group: str | None = None, sigma: float = PP.SIGMA, Tend: int = 366,
            viz_seed: int = 0) -> bool:
    """One W&B run for one (pest, eval_year), carrying all 5 selector seeds.

    per_seed: [{"seed": int, "per_sample": DataFrame(legacy schema), "metrics": dict}, ...]
    raw_metrics: the pre-selector / uncalibrated Stage-2 numbers (training seed 0).
    Returns True if the upload completed; False on any failure (never raises).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{pest}_{eval_year}"

    # --- always build locally, W&B or not (legacy behaviour: figures + CSV on disk) ---
    all_ps, all_lead, fig_paths = [], [], {}
    for rec in per_seed:
        sd = int(rec["seed"])
        df = rec["per_sample"].copy()
        lead_df = VZ.compute_lead_bin_table(df)
        df.insert(0, "selector_seed", sd)
        lead_df.insert(0, "selector_seed", sd)
        all_ps.append(df); all_lead.append(lead_df)
        title = f"{pest} {eval_year} — E5d dev — selector_seed={sd}"
        fig_paths[sd] = build_figures(rec["per_sample"], VZ.compute_lead_bin_table(rec["per_sample"]),
                                      title, out_dir, f"seed{sd}", sigma, Tend, viz_seed)
    ps = pd.concat(all_ps, ignore_index=True)
    ld = pd.concat(all_lead, ignore_index=True)
    per_sample_csv = out_dir / f"per_sample_{tag}.csv"
    lead_csv = out_dir / f"metrics_by_lead_bin_{tag}.csv"
    ps.to_csv(per_sample_csv, index=False); ld.to_csv(lead_csv, index=False)
    print(f"[viz] wrote {per_sample_csv} ({len(ps)} rows, {ps.selector_seed.nunique()} seeds)")

    if not project:
        print("[viz] no W&B project -> local figures + CSV only")
        return False

    try:
        import wandb
    except ImportError:
        print("[viz] wandb not installed; skipping upload (non-fatal)")
        return False

    run = None
    try:
        seeds = sorted(int(r["seed"]) for r in per_seed)
        selector_name = f"coverage_aware_lgbm_seed{min(seeds)}-{max(seeds)}"
        config = {
            # legacy config keys, unchanged
            "pest": pest, "year": eval_year, "selector_name": selector_name,
            "sigma": sigma,
            "sample_grid": str(PP.dev_grid(pest)),
            "selector_offsets": "(E5d coverage-aware selector; picks computed in-process)",
        }
        config.update(cfg_extra)
        run = wandb.init(project=project, entity=entity or None,
                         name=f"{pest}_{eval_year}_E5d_dev_selector_viz",
                         group=group or None, job_type="interval_viz_selector",
                         config=config, reinit=True)

        payload = {}
        for name in VIZ_KEYS:                       # 13 legacy keys, each a 5-image list
            payload[f"viz/{name}"] = [
                wandb.Image(str(fig_paths[sd][name]),
                            caption=f"{pest} {eval_year} | selector_seed={sd} | {name}")
                for sd in seeds if name in fig_paths[sd]]
        payload["table/per_sample"] = wandb.Table(dataframe=ps.assign(
            early_or_inside_30=ps["early_or_inside_30"].astype(int)))
        payload["table/metrics_by_lead_bin"] = wandb.Table(dataframe=ld)
        run.log(payload)

        # final/* = 5-seed mean of the legacy summary; final_std/* = 5-seed std
        sums = [legacy_summary(r["per_sample"]) for r in per_seed]
        for k in sums[0]:
            vals = np.array([s[k] for s in sums], dtype=float)
            run.summary[f"final/{k}"] = float(np.nanmean(vals))
            run.summary[f"final_std/{k}"] = float(np.nanstd(vals, ddof=0))
        run.summary["final/pest"] = pest
        run.summary["final/year"] = eval_year
        run.summary["final/selector_name"] = selector_name
        # selector-applied E5d metrics (5-seed mean already) and pre-selector raw Stage-2
        for k, v in (per_seed[0].get("metrics") or {}).items():
            if isinstance(v, (int, float)):
                run.summary[f"final_selector/{k}"] = float(v)
        for k, v in (raw_metrics or {}).items():
            if isinstance(v, (int, float)):
                run.summary[f"final_raw/{k}"] = float(v)

        try:
            run.save(str(per_sample_csv), policy="now")
            run.save(str(lead_csv), policy="now")
        except Exception as e:                                        # noqa: BLE001
            print(f"[wandb] save() failed (non-fatal): {e}")
        run.finish()
        print(f"[wandb] logged {len(VIZ_KEYS)} viz keys x {len(seeds)} seeds -> {project}")
        return True
    except Exception as e:                                            # noqa: BLE001
        print(f"[wandb] upload FAILED (non-fatal, evaluation unaffected): {type(e).__name__}: {e}")
        traceback.print_exc(limit=2)
        try:
            if run is not None:
                run.finish(exit_code=1)
        except Exception:                                             # noqa: BLE001
            pass
        return False
