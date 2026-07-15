"""Per-pest Stage-2 facts for the TFLite conversion pipeline.

IMPORTANT — what is authoritative here and what is not
-----------------------------------------------------
The **checkpoint is the source of truth** for every architectural value
(d_in, T, doy range, alert_idx, lead bounds, sigma, ...). `checkpoint.py` reads
all of them out of the .pt and never takes them from this file.

The `EXPECTED` table below is a *cross-check*, not an input: it is the config we
believe each pest has, and `checkpoint.py` raises if a checkpoint disagrees.
That way a silently-reshipped checkpoint fails loudly instead of converting into
a wrong-but-plausible .tflite.

The other tables (LONG filenames, smoke records, offsets) are provenance-tagged
below — each says where it came from.
"""

from __future__ import annotations

from dataclasses import dataclass

PESTS: tuple[str, ...] = (
    "BPH",
    "WBPH",
    "bacterial_blight",
    "blast",
    "brown_spot",
    "rice_stem_borer_1",
    "rice_stem_borer_2",
    "sheath_blight",
)


@dataclass(frozen=True)
class ExpectedConfig:
    """Cross-check values. Verified against all 8 checkpoints on 2026-07-15."""

    d_in: int
    T: int
    doy_start: int
    doy_end: int
    alert_idx: int
    # shared by all 8
    d_model: int = 48
    n_head: int = 4
    n_layers: int = 3
    tstar_layers: int = 1
    lead_min: float = 7.0
    lead_max: float = 75.0
    sigma: float = 5.0
    pmf_mode: str = "gaussian"
    mu_mode: str = "lead_from_alert"


# BPH is the outlier (shorter season, 6 base channels); the other 7 are uniform.
_WIDE = dict(d_in=45, T=241, doy_start=60, doy_end=300, alert_idx=30)

EXPECTED: dict[str, ExpectedConfig] = {
    "BPH": ExpectedConfig(d_in=27, T=131, doy_start=140, doy_end=270, alert_idx=12),
    **{p: ExpectedConfig(**_WIDE) for p in PESTS if p != "BPH"},
}

# LONG observation filenames.
# Source: rice/pests/<pest>/config.py PATH_OBS + TARGET_PEST (authoritative).
# NOTE: api_handoff_transformer/infer/batch.py:48 PEST_TO_KOREAN maps BOTH
# rice_stem_borer_1 and _2 to "이화명나방" and so cannot disambiguate them; it is
# used there for a different purpose (matching a pest column). Do not use it to
# pick a LONG file.
LONG_FILENAME: dict[str, str] = {
    "BPH": "RICE_LONG_벼멸구.csv",
    "WBPH": "RICE_LONG_흰등멸구.csv",
    "bacterial_blight": "RICE_LONG_흰잎마름병.csv",
    "blast": "RICE_LONG_잎도열병.csv",
    "brown_spot": "RICE_LONG_깨씨무늬병.csv",
    "rice_stem_borer_1": "RICE_LONG_이화명나방1화기.csv",
    "rice_stem_borer_2": "RICE_LONG_이화명나방2화기.csv",
    "sheath_blight": "RICE_LONG_잎집무늬마름병.csv",
}


@dataclass(frozen=True)
class SmokeRecord:
    """A real (site, year) with the value the API recorded for it.

    Source: api_handoff_transformer/README.md §10 "Smoke test record", which
    states all 8 were verified end-to-end with stage1_live and
    zero_placeholder_used=false. mu_doy is quoted there to 2 decimals.
    """

    site: str
    year: int
    alert_doy: int
    mu_doy: float


SMOKE: dict[str, SmokeRecord] = {
    "BPH": SmokeRecord("33210_56298", 2004, 176, 234.32),
    "WBPH": SmokeRecord("33908_67063", 2011, 171, 245.82),
    "bacterial_blight": SmokeRecord("30247_65595", 2010, 206, 268.32),
    "blast": SmokeRecord("36582_63441", 2017, 125, 198.08),
    "brown_spot": SmokeRecord("31522_54338", 2022, 157, 224.93),
    "rice_stem_borer_1": SmokeRecord("35474_56809", 2018, 125, 184.17),
    "rice_stem_borer_2": SmokeRecord("31959_58947", 2014, 186, 260.97),
    "sheath_blight": SmokeRecord("35694_60137", 2004, 118, 192.72),
}

# selected_fixed_offset per pest.
# Source: api_handoff_transformer/configs/fallback_policy.yaml (schema v4).
# Read at runtime by checkpoint.py rather than trusted from here; this copy is
# only a fallback/cross-check.
SELECTED_OFFSET: dict[str, int] = {
    "BPH": 30,
    "WBPH": 45,
    "bacterial_blight": 7,
    "blast": 7,
    "brown_spot": 14,
    "rice_stem_borer_1": 7,
    "rice_stem_borer_2": 14,
    "sheath_blight": 45,
}

# Which pests actually serve the Transformer as the official answer.
# Source: fallback_policy.yaml recommended_source / learned_output_status.
# The other six return climatology and mark the learned output "experimental",
# so a .tflite for them optimizes a path production does not answer with.
LEARNED_IS_MAIN: frozenset[str] = frozenset({"BPH", "WBPH"})
