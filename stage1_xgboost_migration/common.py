"""Shared helpers for the Stage-1 XGBoost portable-model migration.

Read-only with respect to every original artefact: the source .pt checkpoints are
opened with torch.load and never written back.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

SCHEMA_VERSION = "stage1-xgb-portable/1.0.0"

REPO_ROOT = Path(__file__).resolve().parent.parent
MIGRATION_ROOT = Path(__file__).resolve().parent

# Checkpoints are read from an extraction of the handoff zip into an ignored
# workdir, so the originals outside the repo are never touched.
SOURCE_ZIP = REPO_ROOT.parent / "api_handoff_transformer_batch_20260708.zip"
WORK_ROOT = MIGRATION_ROOT / "_work" / "api_handoff_transformer"
ARTIFACTS_ROOT = MIGRATION_ROOT / "artifacts"
FIXTURES_ROOT = MIGRATION_ROOT / "fixtures"
REPORTS_ROOT = MIGRATION_ROOT / "reports"

# The live API package (source of the preprocessing code used for real-data parity).
API_ROOT = REPO_ROOT.parent / "api_handoff_transformer"

PESTS = [
    "BPH",
    "WBPH",
    "bacterial_blight",
    "blast",
    "brown_spot",
    "rice_stem_borer_1",
    "rice_stem_borer_2",
    "sheath_blight",
]
BRANCHES = ["A", "D"]

# _build_tabular (infer/stage1.py) concatenates these per-channel statistics in
# this exact order, then optionally appends the tstar position feature.
TABULAR_STATS = ["mean", "std", "min", "max", "first", "last", "slope"]
TSTAR_FEATURE_NAME = "tstar_pos"

# Site-history channels appended by infer/stage1.py::_append_history when the
# checkpoint carries site_history_added=True (the D branch). Copied verbatim from
# HISTORY_STATIC_NAMES + HISTORY_DYNAMIC_NAMES; _history_channels emits them in
# exactly this order (static block first, then the dynamic block).
HISTORY_STATIC_NAMES = [
    "prev_year_L_doy_at_site",
    "prev_year_event_at_site",
    "site_avg_L_doy_recent3y",
    "years_since_last_event_at_site",
    "n_events_recent5y_at_site",
    "prev_year_L_miss",
    "site_avg_L_recent3y_miss",
]
HISTORY_DYNAMIC_NAMES = [
    "days_to_prev_year_L",
    "abs_days_to_prev_year_L",
    "days_to_site_avg_L_recent3y",
    "abs_days_to_site_avg_L_recent3y",
]
HISTORY_NAMES = HISTORY_STATIC_NAMES + HISTORY_DYNAMIC_NAMES  # 11


def ckpt_path(pest: str, branch: str) -> Path:
    return (WORK_ROOT / "assets" / "stage1" / pest / f"{branch}_ckpt"
            / f"event_xgb_w28_lead14-45_{branch}.pt")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def env_versions() -> dict:
    import numpy
    import sklearn
    import torch
    import xgboost
    return {
        "python": sys.version.split()[0],
        "python_full": sys.version.replace("\n", " "),
        "torch": torch.__version__,
        "xgboost": xgboost.__version__,
        "scikit_learn": sklearn.__version__,
        "numpy": numpy.__version__,
        "executable": sys.executable,
    }


def channel_names(ckpt: dict) -> list[str]:
    """Per-timestep input channels for the window fed to _build_tabular.

    Base channels are ckpt["feature_names"]. When ckpt["site_history_added"] is
    set (the D branch), infer/stage1.py::_calibrated_per_sy appends the 11
    site-history channels via _append_history before tabularisation.
    """
    channels = [str(c) for c in ckpt["feature_names"]]
    if bool(ckpt.get("site_history_added", False)):
        channels = channels + list(HISTORY_NAMES)
    return channels


def derive_feature_names(ckpt: dict) -> list[str]:
    """Reconstruct the exact predict_proba input column order.

    The order is not stored in the checkpoint; it is defined by
    infer/stage1.py::_build_tabular, which emits, for a window of shape
    (T, n_channels): mean, std, min, max, first, last, slope -- each a vector of
    n_channels -- concatenated in that order, followed by tstar/season_length
    when add_tstar_position_feature is set.

    The caller must cross-check the resulting length against
    sk_model.n_features_in_; nothing here is guessed.
    """
    channels = channel_names(ckpt)
    names = [f"{stat}__{ch}" for stat in TABULAR_STATS for ch in channels]
    if bool(ckpt.get("add_tstar_position_feature", False)):
        names.append(TSTAR_FEATURE_NAME)
    return names


def load_checkpoint(path: Path) -> dict:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2, ensure_ascii=False, sort_keys=False)
        fh.write("\n")
