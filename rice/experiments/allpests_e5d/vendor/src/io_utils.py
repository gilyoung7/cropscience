"""IO helpers for the WBPH interval-perf workspace. All legacy paths come from configs/paths.yaml
and are treated READ-ONLY. Nothing here writes to legacy locations."""
from __future__ import annotations
from pathlib import Path
import yaml
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parents[1]   # workspace root

# Stage-1 dispatch sample-level features (selector inputs) + branch. Order per legacy CSV.
SAMPLE_FEATURES = [
    "alert_tstar", "A_score_at_alert", "D_score_at_alert", "score_margin",
    "dispatch_score_at_alert", "dispatch_tau_used", "score_over_tau_margin",
    "recent_14d_mean_score", "recent_28d_mean_score",
    "score_above_tau_streak", "score_rolling_slope_14d",
    "p_mean_so_far_at_alert", "with_history",
]


def load_paths(cfg_path: str | Path = None) -> dict:
    cfg_path = Path(cfg_path) if cfg_path else HERE / "configs" / "paths.yaml"
    with open(cfg_path) as f:
        return yaml.safe_load(f)


def load_experiment(cfg_name: str) -> dict:
    with open(HERE / "configs" / cfg_name) as f:
        return yaml.safe_load(f)


def load_grid(paths: dict) -> pd.DataFrame:
    """The ckpt-norm multiyear offset grid (baseline+DN, 2022/2023/2024, offsets 1..75)."""
    return pd.read_csv(paths["selected_inputs"]["wbph_grid_multiyear"])


def load_dispatch(paths: dict, year: int) -> pd.DataFrame:
    p = paths["selected_inputs"]["wbph_stage1_dispatch"][year]
    d = pd.read_csv(p)
    d["sample_id"] = d["site"].astype(str) + "-" + d["year"].astype(int).astype(str)
    keep = ["sample_id"] + [c for c in SAMPLE_FEATURES if c in d.columns] + ["dispatch_branch"]
    return d[keep].drop_duplicates("sample_id")


def load_clim_mid(paths: dict, year: int) -> float:
    p = paths["selected_inputs"]["wbph_clim_train_stats"][year]
    return float(pd.read_csv(p).iloc[0]["mean_mid"])


def read_ckpt_norm(ckpt_path: str | Path) -> dict:
    """Read ONLY norm_mean/norm_std + a few meta fields from a ckpt (map_location cpu). Read-only."""
    import torch
    c = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    return {
        "norm_mean": np.asarray(c.get("norm_mean")) if c.get("norm_mean") is not None else None,
        "norm_std": np.asarray(c.get("norm_std")) if c.get("norm_std") is not None else None,
        "d_in": len(c["norm_mean"]) if c.get("norm_mean") is not None else None,
        "stage2_pmf_sigma": c.get("stage2_pmf_sigma"),
        "stage2_pmf_mu_mode": c.get("stage2_pmf_mu_mode"),
        "stage2_neighbor_history_added": c.get("stage2_neighbor_history_added"),
        "doy_start": c.get("doy_start"), "doy_end": c.get("doy_end"),
    }


def out_dir(paths: dict, key: str) -> Path:
    d = HERE / paths["outputs"][key]
    d.mkdir(parents=True, exist_ok=True)
    return d
