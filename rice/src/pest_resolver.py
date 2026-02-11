from __future__ import annotations

import importlib
from pathlib import Path

from rice.configs import config as C
from rice.configs.base import RICE_ROOT, DAILY_CACHE_DIR

PEST_SLUG_MAP = {
    "sbw": "벼멸구",
    "rbw": "흰등멸구",
    "sheath_blight": "잎집무늬마름병",
    "blast": "잎도열병",
    "bacterial_blight": "흰잎마름병",
    "brown_spot": "깨씨무늬병",
    "rice_stem_borer_1": "이화명나방1화기",
    "rice_stem_borer_2": "이화명나방2화기",
}


def available_pest_slugs() -> list[str]:
    pests_dir = Path(__file__).resolve().parents[1] / "pests"
    slugs = []
    for p in sorted(pests_dir.iterdir()):
        if not p.is_dir():
            continue
        if p.name.startswith("_"):
            continue
        if (p / "config.py").exists() and (p / "features.py").exists():
            slugs.append(p.name)
    return slugs


def resolve_pest(pest_slug: str):
    available = available_pest_slugs()
    if pest_slug not in available:
        mapping = ", ".join(f"{k}:{v}" for k, v in PEST_SLUG_MAP.items() if k in available)
        raise ValueError(
            f"Unknown pest slug: {pest_slug}. available={available}. slug_map={mapping}"
        )
    C.apply_pest_config(pest_slug)
    features_mod = importlib.import_module(f"rice.pests.{pest_slug}.features")
    if not hasattr(features_mod, "get_feature_cols"):
        raise ValueError(f"pest features module has no get_feature_cols: {pest_slug}")
    return C, features_mod.get_feature_cols


def default_out_root(pest_slug: str) -> str:
    return str(RICE_ROOT / "outputs" / pest_slug)


def ensure_output_dirs(out_root: str):
    root = Path(out_root)
    for name in ("ckpt", "logs", "eval", "pi", "sfs", "backward", "splits"):
        (root / name).mkdir(parents=True, exist_ok=True)
    DAILY_CACHE_DIR.mkdir(parents=True, exist_ok=True)
