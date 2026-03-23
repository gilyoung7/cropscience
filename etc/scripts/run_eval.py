from __future__ import annotations

import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch

from etc.configs import config as C
from etc.src.pest_resolver import resolve_pest, default_out_root, ensure_output_dirs
from etc.scripts.common import (
    make_loader,
    parse_seed_candidates,
    parse_tags,
    init_wandb_run,
    finish_wandb_run,
)
from etc.src.data_pipeline import (
    load_daily_preprocessed,
    load_obs,
    aggregate_obs_daily_max,
    make_obs_meta,
    add_site_static_latlon,
    merge_pheno_daily_ffill,
    make_daily_feature_frame,
)
from etc.src.labels import build_interval_labels_from_doy, filter_labels_by_gap
from etc.src.dataset import (
    build_train_frame,
    slice_season,
    build_samples_season,
    split_by_sample,
    compute_norm_stats,
    IntervalEventDataset,
    log_split_fingerprint,
    split_seed_search_topk,
    build_stage2_nowcast_samples,
)
from etc.src.ckpt_schema import validate_ckpt_meta
from etc.src.model import HazardTransformer
from etc.src.train_eval import (
    CTYPE_INTERVAL,
    eval_nll_model,
    eval_metrics_with_overlap,
    hazard_to_pmf_cdf_logS,
    quantile_from_cdf_1d,
    shortest_mass_interval_1d,
)

WANDB_ENTITY_DEFAULT = "gilyoung7-seoul-national-university"

def build_samples_for_run(run: int, get_feature_cols, return_debug_stats: bool = False):
    """
    Rebuild datasets deterministically (same as training pipeline split seed=42).
    Norm stats are recomputed from train split.
    """
    # DAILY
    daily = load_daily_preprocessed(C.PATH_DAILY)

    # OBS
    obs = load_obs(C.PATH_OBS)
    obs2 = aggregate_obs_daily_max(obs)

    # LABELS
    labels = build_interval_labels_from_doy(
        obs2, threshold=C.THRESHOLD, season_start_doy=C.SEASON_START_DOY, season_end_doy=C.SEASON_END_DOY
    )
    labels = filter_labels_by_gap(labels, C.DOY_START, C.DOY_END, C.MAX_GAP)

    # META
    obs_meta = make_obs_meta(obs2, C.DOY_START, C.DOY_END)

    # FRAME
    daily_feat, _ = make_daily_feature_frame(daily)
    T = C.DOY_END - C.DOY_START + 1
    train_df = build_train_frame(daily_feat, labels, obs_meta, T=T)
    train_df = add_site_static_latlon(train_df, obs)
    train_df = merge_pheno_daily_ffill(train_df, obs)

    feature_cols = get_feature_cols(run)

    df_season = slice_season(train_df, C.DOY_START, C.DOY_END)
    group_len = (
        df_season.groupby(["site_id", "year"], sort=False)
        .size()
        .reset_index(name="n_rows")
    )
    dropped_groups_df = group_len[group_len["n_rows"] != T][["site_id", "year"]].copy()
    samples, dropped, feature_names = build_samples_season(df_season, feature_cols, C.DOY_START, C.DOY_END)
    if dropped > 0:
        print("WARNING: dropped groups (len!=T):", dropped)

    if not return_debug_stats:
        return feature_cols, feature_names, T, samples

    debug_stats = {
        "n_groups_total": int(len(group_len)),
        "n_groups_kept": int(len(samples)),
        "n_groups_dropped_len_mismatch": int(dropped),
        "drop_ratio_len_mismatch": float(dropped / max(int(len(group_len)), 1)),
        "dropped_site_year": [(str(r.site_id), int(r.year)) for r in dropped_groups_df.itertuples(index=False)],
    }
    return feature_cols, feature_names, T, samples, debug_stats


def mean_std(arr):
    arr = np.asarray(arr, dtype=float)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


@torch.no_grad()
def collect_interval_width_rows(
    model,
    loader,
    source_samples: list[dict],
    Tend: int,
    device,
    alpha: float = 0.2,
    pi_method: str = "shortest",
    sample_id_prefix: str = "",
    seed: int | None = None,
):
    q_lo = alpha / 2
    q_hi = 1 - alpha / 2
    target_mass = 1.0 - float(alpha)
    rows: list[dict] = []
    sample_idx = 0

    model.eval()
    for X, L, R, ctype in loader:
        X = X.to(device, non_blocking=True)
        hazard = model(X)
        pmf, cdf, _ = hazard_to_pmf_cdf_logS(hazard)

        L_np = L.cpu().numpy().astype(int)
        R_np = R.cpu().numpy().astype(int)
        ctype_np = ctype.cpu().numpy().astype(int)
        cdf_np = cdf.cpu().numpy()
        pmf_np = pmf.cpu().numpy()

        for b in range(len(L_np)):
            if int(ctype_np[b]) != int(CTYPE_INTERVAL):
                continue
            if pi_method == "shortest":
                pL, pR, _ = shortest_mass_interval_1d(pmf_np[b], target_mass=target_mass, Tend=Tend)
            elif pi_method == "quantile":
                pL = quantile_from_cdf_1d(cdf_np[b], q_lo, Tend)
                pR = quantile_from_cdf_1d(cdf_np[b], q_hi, Tend)
            else:
                raise ValueError(f"Unknown pi_method: {pi_method}. expected 'shortest' or 'quantile'")

            pL = max(1, min(int(pL), int(Tend)))
            pR = max(1, min(int(pR), int(Tend)))
            if pL > pR:
                pL, pR = pR, pL

            true_l = int(L_np[b])
            true_r = int(R_np[b])
            pred_width = int(pR - pL)
            true_width = int(true_r - true_l)
            pred_mid = float((pL + pR) / 2.0)
            true_mid = float((true_l + true_r) / 2.0)
            abs_mid_error = float(abs(pred_mid - true_mid))
            overlap_left = max(pL, true_l)
            overlap_right = min(pR, true_r)
            overlap_len = int(max(0, overlap_right - overlap_left))
            union_left = min(pL, true_l)
            union_right = max(pR, true_r)
            union_len = float(max(0, union_right - union_left))
            if union_len > 0:
                overlap_ratio = float(overlap_len / union_len)
            else:
                overlap_ratio = 1.0 if (pL == true_l and pR == true_r) else 0.0

            sidx = sample_idx + b
            sample_meta = source_samples[sidx] if sidx < len(source_samples) else {}
            site = sample_meta.get("site_id")
            year = sample_meta.get("year")

            if site is not None and year is not None:
                sample_id = f"{site}-{int(year)}"
            else:
                sample_id = f"{int(sidx)}"

            rows.append(
                {
                    "sample_id": sample_id,
                    "sample_idx": int(sidx),
                    "seed": None if seed is None else int(seed),
                    "site": None if site is None else str(site),
                    "year": None if year is None else int(year),
                    "pred_l": int(pL),
                    "pred_r": int(pR),
                    "true_l": int(true_l),
                    "true_r": int(true_r),
                    "pred_width": int(pred_width),
                    "true_width": int(true_width),
                    "pred_mid": float(pred_mid),
                    "true_mid": float(true_mid),
                    "abs_mid_error": float(abs_mid_error),
                    "overlap_len": int(overlap_len),
                    "overlap_ratio": float(overlap_ratio),
                    "pmf": pmf_np[b].astype(float).tolist(),
                }
            )
        sample_idx += len(L_np)
    return rows


def log_interval_width_diagnostics_to_wandb(wandb_run, rows: list[dict]):
    if wandb_run is None or not rows:
        return
    import wandb
    import matplotlib.pyplot as plt

    pred_width = np.asarray([r["pred_width"] for r in rows], dtype=float)
    true_width = np.asarray([r["true_width"] for r in rows], dtype=float)
    abs_mid_error = np.asarray([r["abs_mid_error"] for r in rows], dtype=float)

    wandb_run.summary["mean_pred_interval_width"] = float(np.mean(pred_width))
    wandb_run.summary["median_pred_interval_width"] = float(np.median(pred_width))
    wandb_run.summary["mean_true_interval_width"] = float(np.mean(true_width))
    wandb_run.summary["median_true_interval_width"] = float(np.median(true_width))
    wandb_run.summary["mean_abs_mid_error"] = float(np.mean(abs_mid_error))

    table_cols = [
        "sample_id",
        "sample_idx",
        "seed",
        "site",
        "year",
        "pred_l",
        "pred_r",
        "true_l",
        "true_r",
        "pred_width",
        "true_width",
        "pred_mid",
        "true_mid",
        "abs_mid_error",
    ]
    table_data = [[r[c] for c in table_cols] for r in rows]
    table = wandb.Table(columns=table_cols, data=table_data)
    wandb_run.log({"eval/interval_width_table": table})

    def _label_for_row(r: dict) -> str:
        site = r.get("site")
        year = r.get("year")
        if site is not None and year is not None:
            return f"{site}-{year}"
        return str(r.get("sample_id"))

    grouped: dict[str, dict] = {}
    for r in rows:
        skey = str(r["sample_id"])
        g = grouped.setdefault(
            skey,
            {
                "sample_id": skey,
                "sample_idx": int(r.get("sample_idx", -1)),
                "site": r.get("site"),
                "year": r.get("year"),
                "true_l": int(r["true_l"]),
                "true_r": int(r["true_r"]),
                "rows_by_seed": {},
            },
        )
        g["rows_by_seed"][int(r["seed"])] = r

    if not grouped:
        return

    seeds = sorted({int(r["seed"]) for r in rows if r.get("seed") is not None})
    if not seeds:
        return

    sample_stats = []
    for skey, g in grouped.items():
        seed_rows = list(g["rows_by_seed"].values())
        ov = np.asarray([float(x["overlap_ratio"]) for x in seed_rows], dtype=float)
        ae = np.asarray([float(x["abs_mid_error"]) for x in seed_rows], dtype=float)
        sample_stats.append(
            {
                "sample_id": str(skey),
                "min_overlap_ratio": float(np.min(ov)),
                "mean_overlap_ratio": float(np.mean(ov)),
                "max_overlap_ratio": float(np.max(ov)),
                "mean_abs_mid_error": float(np.mean(ae)),
            }
        )

    def _sample_label(g: dict) -> str:
        if g.get("site") is not None and g.get("year") is not None:
            return f"{g['site']}-{g['year']}"
        return f"sample_{g['sample_id']}"

    def _plot_interval_comparison_by_sample_ids(selected_sample_ids: list[str], title: str):
        if not selected_sample_ids:
            return None
        n = len(selected_sample_ids)
        fig_h = max(6.0, min(18.0, 0.35 * n + 1.5))
        fig, ax = plt.subplots(figsize=(12, fig_h))

        colors = plt.cm.get_cmap("tab10", max(len(seeds), 3))
        offsets = np.linspace(-0.20, 0.20, num=len(seeds)) if len(seeds) > 1 else np.array([0.0])

        yticks = []
        ylabels = []
        for i, skey in enumerate(selected_sample_ids):
            g = grouped[skey]
            y = i + 1
            y_true = y - 0.30
            true_mid = (int(g["true_l"]) + int(g["true_r"])) / 2.0
            ax.hlines(y_true, xmin=g["true_l"], xmax=g["true_r"], color="black", linewidth=2.5)
            ax.plot(true_mid, y_true, "o", color="black", markersize=3)

            for j, sd in enumerate(seeds):
                rr = g["rows_by_seed"].get(int(sd))
                if rr is None:
                    continue
                y_pred = y + float(offsets[j])
                c = colors(j)
                ax.hlines(y_pred, xmin=rr["pred_l"], xmax=rr["pred_r"], color=c, linewidth=2)
                ax.plot(rr["pred_mid"], y_pred, "o", color=c, markersize=3)

            yticks.append(y)
            ylabels.append(_sample_label(g))

        ax.set_xlabel("timestep")
        ax.set_ylabel("sample")
        ax.set_title(f"{title} | seeds={seeds}")
        ax.set_yticks(yticks)
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.grid(axis="x", alpha=0.25)
        ax.invert_yaxis()

        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], color="black", lw=2.5, label="true interval")]
        for j, sd in enumerate(seeds):
            handles.append(Line2D([0], [0], color=colors(j), lw=2, label=f"pred seed {sd}"))
        ax.legend(
            handles=handles,
            loc="best",
        )
        fig.tight_layout()
        return fig

    def _plot_pmf_by_sample_ids(selected_sample_ids: list[str], title: str, max_samples: int = 5):
        if not selected_sample_ids:
            return None
        picked = selected_sample_ids[:max_samples]
        n = len(picked)
        fig, axes = plt.subplots(n, 1, figsize=(12, max(2.5 * n, 4.0)), sharex=True)
        if n == 1:
            axes = [axes]
        colors = plt.cm.get_cmap("tab10", max(len(seeds), 3))

        for i, skey in enumerate(picked):
            g = grouped[skey]
            ax = axes[i]
            ax.axvspan(g["true_l"], g["true_r"], color="gray", alpha=0.20, label="true interval")
            for j, sd in enumerate(seeds):
                rr = g["rows_by_seed"].get(int(sd))
                if rr is None:
                    continue
                pmf = np.asarray(rr.get("pmf", []), dtype=float)
                if pmf.size == 0:
                    continue
                x = np.arange(1, pmf.size + 1)
                ax.plot(x, pmf, color=colors(j), lw=1.2, label=f"seed {sd}")
            ax.set_ylabel("pmf")
            ax.set_title(_sample_label(g))
            ax.grid(axis="x", alpha=0.2)
            if i == 0:
                ax.legend(loc="upper right", fontsize=8)
        axes[-1].set_xlabel("timestep")
        fig.suptitle(f"{title} | seeds={seeds}", y=0.98)
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.subplots_adjust(top=0.90)
        return fig

    for k_target in (50, 5):
        k = min(k_target, len(sample_stats))
        if k <= 0:
            continue

        top = sorted(
            sample_stats,
            key=lambda x: (float(x["min_overlap_ratio"]), -float(x["mean_abs_mid_error"])),
            reverse=True,
        )[:k]
        top_ids = [str(x["sample_id"]) for x in top]
        fig_top = _plot_interval_comparison_by_sample_ids(top_ids, f"Sample Interval Comparison (Top-{k} min_overlap)")
        if fig_top is not None:
            wandb_run.log({f"eval/sample_interval_comparison_top{k}": wandb.Image(fig_top)})
            plt.close(fig_top)
        fig_top_pmf = _plot_pmf_by_sample_ids(top_ids, f"Sample PMF Comparison (Top-{k} min_overlap)")
        if fig_top_pmf is not None:
            wandb_run.log({f"eval/sample_pmf_comparison_top{min(k,5)}": wandb.Image(fig_top_pmf)})
            plt.close(fig_top_pmf)

        worst = sorted(
            sample_stats,
            key=lambda x: (float(x["min_overlap_ratio"]), float(x["mean_abs_mid_error"])),
        )[:k]
        worst_ids = [str(x["sample_id"]) for x in worst]
        fig_worst = _plot_interval_comparison_by_sample_ids(worst_ids, f"Sample Interval Comparison (Worst-{k} min_overlap)")
        if fig_worst is not None:
            wandb_run.log({f"eval/sample_interval_comparison_worst{k}": wandb.Image(fig_worst)})
            plt.close(fig_worst)
        fig_worst_pmf = _plot_pmf_by_sample_ids(worst_ids, f"Sample PMF Comparison (Worst-{k} min_overlap)")
        if fig_worst_pmf is not None:
            wandb_run.log({f"eval/sample_pmf_comparison_worst{min(k,5)}": wandb.Image(fig_worst_pmf)})
            plt.close(fig_worst_pmf)


def resolve_ckpt_path(run: int, out_root: str, ckpt_path: str | None) -> Path:
    if ckpt_path:
        return Path(ckpt_path)
    return Path(out_root) / "ckpt" / f"checkpoint_run{run}.pt"


def resolve_out_csv(run: int, out_root: str, out_csv: str | None) -> str | None:
    if out_csv is None:
        out_dir = Path(out_root) / "eval"
        out_dir.mkdir(parents=True, exist_ok=True)
        return str(out_dir / f"eval_run{run}.csv")
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    return out_csv


def resolve_split_seeds_json_path(out_root: str, split_seeds_json: str | None) -> Path:
    if split_seeds_json:
        return Path(split_seeds_json)
    return Path(out_root) / "splits" / "selected_split_seeds.json"


def load_split_seed_from_topk(split_seeds_json_path: Path, split_seed_from_topk_idx: int | None) -> tuple[int, int, dict, dict]:
    if not split_seeds_json_path.exists():
        raise FileNotFoundError(f"split seeds json not found: {split_seeds_json_path}")
    with open(split_seeds_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    topk_list = payload.get("topk", [])
    if not topk_list:
        raise ValueError(f"topk is empty in split seeds json: {split_seeds_json_path}")
    if split_seed_from_topk_idx is None:
        split_seed_from_topk_idx = payload.get("selected_topk_idx", 0)
    if split_seed_from_topk_idx < 0 or split_seed_from_topk_idx >= len(topk_list):
        raise ValueError(f"--split_seed_from_topk_idx out of range (0..{len(topk_list)-1})")
    chosen = topk_list[int(split_seed_from_topk_idx)]
    split_seed = int(chosen["seed"])
    return split_seed, int(split_seed_from_topk_idx), chosen, payload


def main(
    pest: str,
    ckpt_path: str | None,
    run: int,
    out_csv: str | None,
    out_root: str,
    allow_run_mismatch: bool,
    split_seed: int,
    seeds: list[int] | None,
    auto_split_seed: bool,
    seed_candidates_raw: str | None,
    target_test_interval: int | None,
    tol_test_interval: int | None,
    auto_split_topk: int,
    split_seed_from_topk_idx: int | None,
    split_seeds_json: str | None,
    stage2_nowcast: bool,
    stage2_nowcast_window: int,
    stage2_nowcast_stride: int,
    stage2_nowcast_tstar_start: int | None,
    stage2_nowcast_only_pre_event: int,
    stage2_nowcast_event_time_proxy: str,
    use_wandb: bool,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_group: str | None,
    wandb_run_name: str | None,
    wandb_tags: str | None,
    wandb_job_type: str | None,
):
    _, get_feature_cols = resolve_pest(pest)
    if not out_root:
        out_root = default_out_root(pest)
    ensure_output_dirs(out_root)
    if stage2_nowcast_window is None:
        stage2_nowcast_window = int(getattr(C, "STAGE2_NOWCAST_WINDOW", 56))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    wandb_run = init_wandb_run(
        use_wandb=use_wandb,
        project=wandb_project,
        entity=wandb_entity,
        run_name=wandb_run_name,
        group=wandb_group,
        job_type=wandb_job_type,
        tags=parse_tags(wandb_tags) + [f"pest:{pest}", "script:run_eval"],
        config={
            "pest": pest,
            "run": run,
            "split_seed": split_seed,
            "stage2_nowcast": bool(stage2_nowcast),
            "stage2_nowcast_window": int(stage2_nowcast_window),
            "stage2_nowcast_stride": int(stage2_nowcast_stride),
            "stage2_nowcast_tstar_start": None if stage2_nowcast_tstar_start is None else int(stage2_nowcast_tstar_start),
            "stage2_nowcast_only_pre_event": int(stage2_nowcast_only_pre_event),
            "stage2_nowcast_event_time_proxy": stage2_nowcast_event_time_proxy,
        },
    )

    ckpt_path_resolved = resolve_ckpt_path(run, out_root, ckpt_path)
    print(f"Using checkpoint: {ckpt_path_resolved}")
    ckpt = torch.load(ckpt_path_resolved, map_location="cpu")
    trained_states = ckpt["trained_states"]
    print("loaded ckpt:", ckpt_path_resolved, "| seeds:", [d["seed"] for d in trained_states])

    feature_cols, feature_names_eval, T, samples = build_samples_for_run(run, get_feature_cols)
    print(f"[features] n={len(feature_names_eval)} head={feature_names_eval[:5]} tail={feature_names_eval[-5:]}")
    result = None
    chosen = None
    if split_seeds_json is not None:
        split_seeds_json_path = resolve_split_seeds_json_path(out_root, split_seeds_json)
        split_seed, chosen_idx, chosen, _ = load_split_seed_from_topk(split_seeds_json_path, split_seed_from_topk_idx)
        train_s, val_s, test_s = split_by_sample(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)
        print(f"[split_seed_json] selected seed={split_seed} idx={chosen_idx} file={split_seeds_json_path}")
        print(f"[split_seed_json] counts={chosen.get('counts')}")
    elif auto_split_seed:
        candidates = parse_seed_candidates(seed_candidates_raw) or list(range(0, 200))
        result = split_seed_search_topk(
            samples,
            val_frac=0.1,
            test_frac=0.1,
            seed_candidates=candidates,
            target_test_interval=target_test_interval,
            tol_test_interval=tol_test_interval,
            topk=auto_split_topk,
        )
        topk_list = result["topk"]
        if not topk_list:
            raise ValueError("auto_split_seed produced no candidates")
        if split_seed_from_topk_idx is None:
            split_seed_from_topk_idx = 0
        if split_seed_from_topk_idx < 0 or split_seed_from_topk_idx >= len(topk_list):
            raise ValueError(f"--split_seed_from_topk_idx out of range (0..{len(topk_list)-1})")
        chosen = topk_list[split_seed_from_topk_idx]
        split_seed = int(chosen["seed"])
        train_s, val_s, test_s = split_by_sample(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)
        print(
            f"[auto_split] selected seed={split_seed} score={chosen['score']:.6f} "
            f"counts={chosen['counts']}"
        )
        if result.get("used_fallback"):
            print("[auto_split] WARNING: no seed met constraints; using best score fallback.")
        print("[auto_split] topk seeds (seed, score, test_counts):")
        for i, item in enumerate(topk_list):
            print(f"  [{i}] seed={item['seed']} score={item['score']:.6f} test={item['counts']['test']}")
    else:
        train_s, val_s, test_s = split_by_sample(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)

    stage2_nowcast = bool(ckpt.get("stage2_nowcast", stage2_nowcast))
    if stage2_nowcast:
        stage2_nowcast_window = int(ckpt.get("stage2_nowcast_window", stage2_nowcast_window))
        stage2_nowcast_stride = int(ckpt.get("stage2_nowcast_stride", stage2_nowcast_stride))
        stage2_nowcast_tstar_start = ckpt.get("stage2_nowcast_tstar_start", stage2_nowcast_tstar_start)
        stage2_nowcast_only_pre_event = int(ckpt.get("stage2_nowcast_only_pre_event", stage2_nowcast_only_pre_event))
        stage2_nowcast_event_time_proxy = str(ckpt.get("stage2_nowcast_event_time_proxy", stage2_nowcast_event_time_proxy))

        train_s = build_stage2_nowcast_samples(
            train_s,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
        )
        val_s = build_stage2_nowcast_samples(
            val_s,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
        )
        test_s = build_stage2_nowcast_samples(
            test_s,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
        )
        print(
            f"[stage2_nowcast] window={stage2_nowcast_window} stride={stage2_nowcast_stride} "
            f"tstar_start={stage2_nowcast_tstar_start} only_pre_event={bool(stage2_nowcast_only_pre_event)} "
            f"event_time_proxy={stage2_nowcast_event_time_proxy} label_mode=orig | "
            f"samples train={len(train_s)} val={len(val_s)} test={len(test_s)}"
        )
        def _bucket_counts(ss: list[dict]) -> dict[str, int]:
            out = {"pre_L": 0, "in_LR": 0, "post_R": 0, "right": 0}
            for x in ss:
                b = str(x.get("case_bucket", ""))
                if b in out:
                    out[b] += 1
            return out
        print(
            f"[stage2_nowcast] case_bucket train={_bucket_counts(train_s)} "
            f"val={_bucket_counts(val_s)} test={_bucket_counts(test_s)}"
        )

    log_split_fingerprint("eval", train_s, val_s, test_s)

    x_mean, x_std = compute_norm_stats(train_s)
    train_ds = IntervalEventDataset(train_s, x_mean, x_std)
    val_ds   = IntervalEventDataset(val_s,   x_mean, x_std)
    test_ds  = IntervalEventDataset(test_s,  x_mean, x_std)
    D_in = int(test_ds[0][0].shape[-1])
    print(f"[D_in] computed_from_dataset={D_in}")
    print(f"RUN={run} | D_in={D_in} | T={T}")

    validate_ckpt_meta(
        ckpt,
        pest=pest,
        run=run,
        d_in=D_in,
        feature_names=feature_names_eval,
        allow_run_mismatch=allow_run_mismatch,
    )

    val_loader  = make_loader(val_ds,  C.BATCH_EVAL, shuffle=False)
    test_loader = make_loader(test_ds, C.BATCH_EVAL, shuffle=False)

    records = []
    by_tstar_rows = []
    interval_width_rows = []
    for d in trained_states:
        seed = int(d["seed"])
        if seeds is not None and seed not in seeds:
            continue

        model = HazardTransformer(
            d_in=D_in,
            d_model=C.D_MODEL,
            nhead=C.N_HEAD,
            num_layers=C.N_LAYERS,
            dropout=C.DROPOUT,
            max_len=C.MAX_LEN,
        ).to(device)
        model.load_state_dict(d["state_dict"])
        model.eval()

        val_nll  = float(eval_nll_model(model, val_loader, Tend=T, device=device))
        test_nll = float(eval_nll_model(model, test_loader, Tend=T, device=device))

        val_stats = eval_metrics_with_overlap(
            model,
            val_loader,
            Tend=T,
            device=device,
            alpha=0.2,
            pi_method=getattr(C, "PI_METHOD", "shortest"),
        )
        test_stats = eval_metrics_with_overlap(
            model,
            test_loader,
            Tend=T,
            device=device,
            alpha=0.2,
            pi_method=getattr(C, "PI_METHOD", "shortest"),
        )

        rec = {
            "seed": seed,
            "best_epoch": int(d["best_epoch"]),
            "best_val_nll": float(d["best_val_nll"]),
            "VAL_NLL": val_nll,
            "TEST_NLL": test_nll,
            "TEST_IoU80(interval_only)": float(test_stats["IoU_mean_interval_only(80%)"]),
            "TEST_Prec80(interval_only)": float(test_stats["Precision_mean_interval_only(80%)"]),
            "TEST_Rec80(interval_only)": float(test_stats["Recall_mean_interval_only(80%)"]),
            "TEST_MAE_int(interval_only)": float(test_stats["mae_mid_mean_interval_only"]),
            "TEST_Mass_int(interval_only)": float(test_stats["mass_in_interval_mean_interval_only"]),
            "TEST_N_int": int(test_stats["N_interval_samples"]),
        }
        records.append(rec)
        if wandb_run is not None:
            wandb_run.log(
                {
                    "seed": int(seed),
                    "eval/val_nll": float(val_nll),
                    "eval/test_nll": float(test_nll),
                    "eval/test_iou80_interval_only": float(rec["TEST_IoU80(interval_only)"]),
                    "eval/test_prec80_interval_only": float(rec["TEST_Prec80(interval_only)"]),
                    "eval/test_rec80_interval_only": float(rec["TEST_Rec80(interval_only)"]),
                    "eval/test_mae_int_interval_only": float(rec["TEST_MAE_int(interval_only)"]),
                    "eval/test_mass_int_interval_only": float(rec["TEST_Mass_int(interval_only)"]),
                    "eval/test_n_interval": int(rec["TEST_N_int"]),
                    "eval/best_epoch": int(rec["best_epoch"]),
                    "eval/best_val_nll_from_ckpt": float(rec["best_val_nll"]),
                }
            )
            interval_width_rows.extend(
                collect_interval_width_rows(
                    model=model,
                    loader=test_loader,
                    source_samples=test_s,
                    Tend=T,
                    device=device,
                    alpha=0.2,
                    pi_method=getattr(C, "PI_METHOD", "shortest"),
                    sample_id_prefix=f"seed{seed}_",
                    seed=seed,
                )
            )

        if stage2_nowcast:
            for split_name, split_samples in (("val", val_s), ("test", test_s)):
                tvals = sorted({int(s.get("tstar", -1)) for s in split_samples if "tstar" in s})
                for tstar in tvals:
                    sub = [s for s in split_samples if int(s.get("tstar", -1)) == tstar]
                    if not sub:
                        continue
                    sub_ds = IntervalEventDataset(sub, x_mean, x_std)
                    sub_loader = make_loader(sub_ds, C.BATCH_EVAL, shuffle=False)
                    sub_stats = eval_metrics_with_overlap(
                        model,
                        sub_loader,
                        Tend=T,
                        device=device,
                        alpha=0.2,
                        pi_method=getattr(C, "PI_METHOD", "shortest"),
                    )
                    by_tstar_rows.append(
                        {
                            "seed": seed,
                            "split": split_name,
                            "tstar": int(tstar),
                            "n": int(len(sub)),
                            "N_total": int(len(sub)),
                            "N_interval_label": int(sum(1 for s in sub if str(s.get("censor_type", "")) == "interval")),
                            "N_right_label": int(sum(1 for s in sub if str(s.get("censor_type", "")) == "right")),
                            "N_preL": int(sum(1 for s in sub if str(s.get("case_bucket", "")) == "pre_L")),
                            "N_inLR": int(sum(1 for s in sub if str(s.get("case_bucket", "")) == "in_LR")),
                            "N_postR": int(sum(1 for s in sub if str(s.get("case_bucket", "")) == "post_R")),
                            "IoU80_interval_only": float(sub_stats["IoU_mean_interval_only(80%)"]),
                            "Precision80_interval_only": float(sub_stats["Precision_mean_interval_only(80%)"]),
                            "Recall80_interval_only": float(sub_stats["Recall_mean_interval_only(80%)"]),
                            "MAE_int_interval_only": float(sub_stats["mae_mid_mean_interval_only"]),
                            "Mass_int_interval_only": float(sub_stats["mass_in_interval_mean_interval_only"]),
                            "N_int": int(sub_stats["N_interval_samples"]),
                        }
                    )

    df = pd.DataFrame(records).sort_values("seed").reset_index(drop=True)
    print("\n=== per-seed ===")
    print(df.to_string(index=False))

    cols = [
        "VAL_NLL",
        "TEST_NLL",
        "TEST_IoU80(interval_only)",
        "TEST_Prec80(interval_only)",
        "TEST_Rec80(interval_only)",
        "TEST_MAE_int(interval_only)",
        "TEST_Mass_int(interval_only)",
    ]
    print("\n=== mean ± std (seeds) ===")
    for col in cols:
        m, sd = mean_std(df[col].to_numpy())
        print(f"{col:>26s}: {m:.4f} ± {sd:.4f}")
        if wandb_run is not None:
            key = col.replace("(", "").replace(")", "").replace("%", "").replace(" ", "_")
            wandb_run.summary[f"{key}_mean"] = float(m)
            wandb_run.summary[f"{key}_std"] = float(sd)
    print("\nTEST interval-only N:", sorted(df["TEST_N_int"].unique().tolist()))
    if wandb_run is not None:
        wandb_run.summary["n_eval_seeds"] = int(len(df))
        wandb_run.summary["test_n_int_unique"] = ",".join(map(str, sorted(df["TEST_N_int"].unique().tolist())))
        log_interval_width_diagnostics_to_wandb(wandb_run, interval_width_rows)

    out_csv_resolved = resolve_out_csv(run, out_root, out_csv)
    if out_csv_resolved:
        df.to_csv(out_csv_resolved, index=False)
        print("saved:", out_csv_resolved)
        if wandb_run is not None:
            wandb_run.summary["eval_csv_path"] = str(out_csv_resolved)
            wandb_run.save(str(out_csv_resolved), policy="now")
        if stage2_nowcast and by_tstar_rows:
            by_tstar_df = pd.DataFrame(by_tstar_rows).sort_values(["seed", "split", "tstar"]).reset_index(drop=True)
            by_tstar_out = str(Path(out_csv_resolved).with_name(Path(out_csv_resolved).stem + "_by_tstar.csv"))
            by_tstar_df.to_csv(by_tstar_out, index=False)
            print("saved:", by_tstar_out)
            if wandb_run is not None:
                wandb_run.summary["eval_by_tstar_csv_path"] = str(by_tstar_out)
                wandb_run.save(str(by_tstar_out), policy="now")
    finish_wandb_run(wandb_run)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--allow_run_mismatch", action="store_true")
    p.add_argument("--split_seed", type=int, default=C.SPLIT_SEED)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--auto_split_seed", action="store_true")
    p.add_argument("--auto_split_topk", type=int, default=1)
    p.add_argument("--split_seed_from_topk_idx", type=int, default=None)
    p.add_argument("--split_seeds_json", type=str, default=None)
    p.add_argument("--seed_candidates", type=str, default=None)
    p.add_argument("--target_test_interval", type=int, default=None)
    p.add_argument("--tol_test_interval", type=int, default=None)
    p.add_argument("--stage2_nowcast", action="store_true")
    p.add_argument("--stage2_nowcast_window", type=int, default=None)
    p.add_argument("--stage2_nowcast_stride", type=int, default=7)
    p.add_argument("--stage2_nowcast_tstar_start", type=int, default=None)
    p.add_argument("--stage2_nowcast_only_pre_event", type=int, default=1)
    p.add_argument("--stage2_nowcast_event_time_proxy", type=str, default="r", choices=["r", "mid"])
    # Stage-1 style aliases for pipeline consistency.
    p.add_argument("--nowcast_window", dest="stage2_nowcast_window", type=int)
    p.add_argument("--nowcast_stride", dest="stage2_nowcast_stride", type=int)
    p.add_argument("--nowcast_tstar_start", dest="stage2_nowcast_tstar_start", type=int)
    p.add_argument("--nowcast_only_pre_event", dest="stage2_nowcast_only_pre_event", type=int)
    p.add_argument("--nowcast_event_time_proxy", dest="stage2_nowcast_event_time_proxy", type=str, choices=["r", "mid"])
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default="agro-rice")
    p.add_argument("--wandb_entity", type=str, default=WANDB_ENTITY_DEFAULT)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default=None)
    p.add_argument("--wandb_job_type", type=str, default="eval")
    args = p.parse_args()
    main(
        args.pest,
        args.ckpt,
        args.run,
        args.out_csv,
        args.out_root,
        args.allow_run_mismatch,
        args.split_seed,
        args.seeds,
        args.auto_split_seed,
        args.seed_candidates,
        args.target_test_interval,
        args.tol_test_interval,
        args.auto_split_topk,
        args.split_seed_from_topk_idx,
        args.split_seeds_json,
        args.stage2_nowcast,
        args.stage2_nowcast_window,
        args.stage2_nowcast_stride,
        args.stage2_nowcast_tstar_start,
        args.stage2_nowcast_only_pre_event,
        args.stage2_nowcast_event_time_proxy,
        args.use_wandb,
        args.wandb_project,
        args.wandb_entity,
        args.wandb_group,
        args.wandb_run_name,
        args.wandb_tags,
        args.wandb_job_type,
    )
