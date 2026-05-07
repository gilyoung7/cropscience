from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rice.configs import config as C
from rice.scripts.common import collate_grouped_stage2, make_loader
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_viz_interval import collect_interval_preds, collect_interval_preds_grouped
from rice.src.dataset import (
    GroupedIntervalEventDataset,
    IntervalEventDataset,
    build_stage2_nowcast_samples,
    compute_norm_stats,
    group_stage2_samples_by_site_year,
    split_by_site,
)
from rice.src.model import HazardTransformer, HierarchicalCausalHazardTransformer
from rice.src.pest_resolver import default_out_root, ensure_output_dirs, resolve_pest
from rice.src.train_eval import overlap_metrics


DEFAULT_BINS = [
    ("post_or_in", -10_000, 0),
    ("1-14", 1, 14),
    ("15-30", 15, 30),
    ("31-45", 31, 45),
    ("46-60", 46, 60),
    ("61-90", 61, 90),
    ("91-120", 91, 120),
    ("121-150", 121, 150),
    ("151-180", 151, 180),
    ("181+", 181, 10_000),
]


def lead_bin(x: int) -> str:
    for name, lo, hi in DEFAULT_BINS:
        if int(lo) <= int(x) <= int(hi):
            return name
    return "unknown"


def mass_in_interval(row: dict) -> float:
    pmf = np.asarray(row.get("pmf", []), dtype=float)
    if pmf.size == 0:
        return float("nan")
    l = max(1, int(row["true_L"]))
    r = min(int(row["true_R"]), int(pmf.size))
    if r < l:
        return 0.0
    return float(np.sum(pmf[l - 1 : r]))


def rows_to_summary(rows: list[dict], *, split_seed: int, seed: int) -> pd.DataFrame:
    out = []
    for r in rows:
        true_start = int(r["true_L"]) + 1
        tstar = int(r["tstar"])
        lead = int(true_start - tstar)
        iou, rec, prec = overlap_metrics(r["pred_L"], r["pred_R"], r["true_L"], r["true_R"])
        out.append(
            {
                "split_seed": int(split_seed),
                "seed": int(seed),
                "sample_id": str(r["sample_id"]),
                "tstar": int(tstar),
                "true_L": int(r["true_L"]),
                "true_R": int(r["true_R"]),
                "true_start": int(true_start),
                "lead_to_start": int(lead),
                "lead_bin": lead_bin(lead),
                "pred_L": int(r["pred_L"]),
                "pred_R": int(r["pred_R"]),
                "pred_point": int(r["pred_point"]),
                "pred_width": int(r["pred_R"]) - int(r["pred_L"]) + 1,
                "IoU80": float(iou),
                "Rec80": float(rec),
                "Prec80": float(prec),
                "MAE_int": float(abs(((int(r["pred_L"]) + int(r["pred_R"])) / 2.0) - ((int(r["true_L"]) + int(r["true_R"])) / 2.0))),
                "Mass_int": mass_in_interval(r),
            }
        )
    row_df = pd.DataFrame(out)
    if row_df.empty:
        return row_df
    summary = (
        row_df.groupby(["split_seed", "seed", "lead_bin"], as_index=False)
        .agg(
            n=("sample_id", "size"),
            lead_mean=("lead_to_start", "mean"),
            tstar_mean=("tstar", "mean"),
            true_start_mean=("true_start", "mean"),
            pred_L_mean=("pred_L", "mean"),
            pred_R_mean=("pred_R", "mean"),
            pred_point_mean=("pred_point", "mean"),
            pred_width_mean=("pred_width", "mean"),
            IoU80=("IoU80", "mean"),
            Rec80=("Rec80", "mean"),
            Prec80=("Prec80", "mean"),
            MAE_int=("MAE_int", "mean"),
            Mass_int=("Mass_int", "mean"),
        )
    )
    summary["lead_bin"] = pd.Categorical(summary["lead_bin"], [x[0] for x in DEFAULT_BINS], ordered=True)
    return summary.sort_values(["split_seed", "seed", "lead_bin"]).reset_index(drop=True)


def main(pest: str, run: int, out_root: str, split_seed: int, ckpt: str, seeds: list[int] | None):
    _, get_feature_cols = resolve_pest(pest)
    if not out_root:
        out_root = default_out_root(pest)
    ensure_output_dirs(out_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    ckpt_path = Path(ckpt)
    print("Using checkpoint:", ckpt_path)
    ckpt_obj = torch.load(ckpt_path, map_location="cpu")
    C.DOY_START = int(ckpt_obj.get("doy_start", C.DOY_START))
    C.DOY_END = int(ckpt_obj.get("doy_end", C.DOY_END))
    C.D_MODEL = int(ckpt_obj.get("d_model", C.D_MODEL))
    C.N_HEAD = int(ckpt_obj.get("n_head", C.N_HEAD))
    C.N_LAYERS = int(ckpt_obj.get("n_layers", C.N_LAYERS))

    _, feature_names, T, samples = build_samples_for_run(run, get_feature_cols)
    print(f"[features] n={len(feature_names)}")
    train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)

    if not bool(ckpt_obj.get("stage2_nowcast", False)):
        raise ValueError("checkpoint is not stage2_nowcast")
    test_s2 = build_stage2_nowcast_samples(
        test_s,
        window=int(ckpt_obj.get("stage2_nowcast_window", 28)),
        stride=int(ckpt_obj.get("stage2_nowcast_stride", 1)),
        tstar_start=ckpt_obj.get("stage2_nowcast_tstar_start", None),
        only_pre_event=bool(int(ckpt_obj.get("stage2_nowcast_only_pre_event", 1))),
        event_time_proxy=str(ckpt_obj.get("stage2_nowcast_event_time_proxy", "r")),
    )
    x_mean, x_std = compute_norm_stats(train_s)
    grouped_mode = bool(
        ckpt_obj.get("stage2_causal_tstar", False)
        or str(ckpt_obj.get("stage2_model_kind", "flat")) == "hierarchical_causal_tstar"
    )
    if grouped_mode:
        test_groups = group_stage2_samples_by_site_year(test_s2)
        test_ds = GroupedIntervalEventDataset(test_groups, x_mean, x_std)
        test_loader = make_loader(test_ds, 1, shuffle=False, collate_fn=collate_grouped_stage2)
    else:
        test_groups = []
        test_ds = IntervalEventDataset(test_s2, x_mean, x_std)
        test_loader = make_loader(test_ds, C.BATCH_EVAL, shuffle=False)

    D_in = int(test_ds[0][0].shape[-1])
    all_summary = []
    all_rows = []
    for d in ckpt_obj["trained_states"]:
        seed = int(d["seed"])
        if seeds is not None and seed not in seeds:
            continue
        if grouped_mode:
            model = HierarchicalCausalHazardTransformer(
                d_in=D_in,
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=C.N_LAYERS,
                num_tstar_layers=int(ckpt_obj.get("stage2_tstar_layers", 1)),
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
                max_tstar_len=512,
                use_tstar_scalar_pos=bool(int(ckpt_obj.get("stage2_use_tstar_scalar_pos", 0))),
            ).to(device)
            model.time_chunk_size = int(ckpt_obj.get("stage2_time_chunk_size", 64))
            model.pmf_mode = str(ckpt_obj.get("stage2_pmf_mode", "hazard"))
            model.gaussian_sigma = float(ckpt_obj.get("stage2_pmf_sigma", 5.0))
            model.gaussian_mu_max = float(ckpt_obj.get("stage2_pmf_mu_max", 0.0))
            model.load_state_dict(d["state_dict"], strict=False)
            model.eval()
            rows = collect_interval_preds_grouped(
                model,
                test_loader,
                source_groups=test_groups,
                Tend=T,
                device=device,
                pi_method=getattr(C, "PI_METHOD", "shortest"),
                max_samples=100000000,
            )
        else:
            model = HazardTransformer(
                d_in=D_in,
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=C.N_LAYERS,
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
            ).to(device)
            model.load_state_dict(d["state_dict"], strict=False)
            model.eval()
            rows = collect_interval_preds(
                model,
                test_loader,
                source_samples=test_s2,
                Tend=T,
                device=device,
                pi_method=getattr(C, "PI_METHOD", "shortest"),
                max_samples=100000000,
            )
        row_df = pd.DataFrame(rows)
        row_summary = rows_to_summary(rows, split_seed=split_seed, seed=seed)
        all_summary.append(row_summary)
        if not row_df.empty:
            row_df["split_seed"] = int(split_seed)
            row_df["seed"] = int(seed)
            row_df["true_start"] = row_df["true_L"].astype(int) + 1
            row_df["lead_to_start"] = row_df["true_start"].astype(int) - row_df["tstar"].astype(int)
            row_df["lead_bin"] = row_df["lead_to_start"].map(lead_bin)
            all_rows.append(row_df)
        print(f"[seed {seed}] rows={len(rows)}")

    out_dir = Path(out_root) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_df = pd.concat(all_summary, ignore_index=True) if all_summary else pd.DataFrame()
    rows_df = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    summary_path = out_dir / f"stage2_lead_bin_eval_run{run}_split{split_seed}.csv"
    rows_path = out_dir / f"stage2_lead_bin_rows_run{run}_split{split_seed}.csv"
    summary_df.to_csv(summary_path, index=False)
    rows_df.to_csv(rows_path, index=False)
    print("saved:", summary_path)
    print("saved:", rows_path)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pest", required=True)
    p.add_argument("--run", type=int, default=4)
    p.add_argument("--out_root", required=True)
    p.add_argument("--split_seed", type=int, required=True)
    p.add_argument("--ckpt", required=True)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    args = p.parse_args()
    main(args.pest, args.run, args.out_root, args.split_seed, args.ckpt, args.seeds)
