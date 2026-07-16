"""Metadata + request schema for the standalone Stage-2 LiteRT runtime.

Pure stdlib + numpy. No torch, no tensorflow, no api_handoff_transformer import.

Metadata is produced by build_package.py from the Stage-2 checkpoints and is the
only source of per-pest configuration at runtime — the runtime never guesses a
pest's shape, channel order, or normalization.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

METADATA_SCHEMA_VERSION = "1.0"

# The 14 Stage-1 dispatch features + the missing indicator, in channel order.
# Mirrors api_handoff_transformer/infer/preprocess.py DISPATCH_FEATURE_NAMES;
# build_package.py asserts this against every checkpoint's feature_names, so a
# drift fails the build rather than silently reordering channels here.
DISPATCH_FEATURE_NAMES: tuple[str, ...] = (
    "alert_tstar",
    "with_history",
    "dispatch_branch",
    "A_score_at_alert",
    "D_score_at_alert",
    "score_margin",
    "dispatch_score_at_alert",
    "dispatch_tau_used",
    "score_over_tau_margin",
    "recent_14d_mean_score",
    "recent_28d_mean_score",
    "score_above_tau_streak",
    "score_rolling_slope_14d",
    "p_mean_so_far_at_alert",
)
DISPATCH_MISSING_NAME = "dispatch_feature_missing"

COORD_COLS: tuple[str, ...] = ("좌표-위도", "좌표-경도")
PHENO_COLS: tuple[str, ...] = (
    "days_since_growing_start",
    "days_until_growing_end",
    "is_growing",
)

# Daily weather columns the caller must supply (Korean schema, as shipped).
REQUIRED_DAILY_COLS: tuple[str, ...] = (
    "일시",
    "일강수량(mm)",
    "최고기온(°C)",
    "최저기온(°C)",
    "평균기온(°C)",
    "평균 풍속(m/s)",
    "최대 풍속(m/s)",
    "평균 상대습도(%)",
    "합계 일조시간(h)",
    "합계 일사량(MJ/m2)",
    "GDD10_since_gs",
)


class SchemaError(ValueError):
    """Raised when metadata or a request violates the contract. Never silent."""


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass(frozen=True)
class PestMetadata:
    """One pest's model contract, loaded from models/<pest>/metadata.json."""

    raw: dict[str, Any]
    root: Path

    # --- identity -------------------------------------------------------
    @property
    def pest(self) -> str:
        return self.raw["pest"]

    @property
    def schema_version(self) -> str:
        return self.raw["metadata_schema_version"]

    # --- shape / architecture -------------------------------------------
    @property
    def T(self) -> int:
        return int(self.raw["T"])

    @property
    def d_in(self) -> int:
        return int(self.raw["d_in"])

    @property
    def doy_start(self) -> int:
        return int(self.raw["doy_start"])

    @property
    def doy_end(self) -> int:
        return int(self.raw["doy_end"])

    @property
    def alert_idx(self) -> int:
        return int(self.raw["alert_idx"])

    @property
    def nowcast_window(self) -> int:
        return int(self.raw["nowcast_window"])

    @property
    def selected_offset(self) -> int:
        return int(self.raw["selected_offset"])

    # --- mu / interval ---------------------------------------------------
    @property
    def sigma(self) -> float:
        return float(self.raw["gaussian_sigma"])

    @property
    def mu_mode(self) -> str:
        return self.raw["mu_mode"]

    @property
    def mu_semantics(self) -> str:
        return self.raw["mu_output_semantics"]

    # --- channels --------------------------------------------------------
    @property
    def feature_names(self) -> list[str]:
        return list(self.raw["feature_names"])

    @property
    def base_channels(self) -> list[str]:
        return list(self.raw["base_channels"])

    @property
    def dispatch_channels(self) -> list[str]:
        return list(self.raw["dispatch_channels"])

    @property
    def norm_bypass_indices(self) -> list[int]:
        return list(self.raw["norm_bypass_channel_indices"])

    @property
    def requires_site_coords(self) -> bool:
        return bool(self.raw["requires_site_coords"])

    @property
    def requires_phenology(self) -> bool:
        return bool(self.raw["requires_phenology"])

    # --- files -----------------------------------------------------------
    def model_path(self, variant: str) -> Path:
        key = f"model_{variant}"
        if key not in self.raw["files"]:
            raise SchemaError(
                f"[{self.pest}] no {variant!r} model in metadata; "
                f"have {sorted(k for k in self.raw['files'] if k.startswith('model_'))}"
            )
        return self.root / self.raw["files"][key]["filename"]

    def model_sha256(self, variant: str) -> str:
        return self.raw["files"][f"model_{variant}"]["sha256"]

    @property
    def normalization_path(self) -> Path:
        return self.root / self.raw["files"]["normalization"]["filename"]

    def verify_files(self, variants: tuple[str, ...] = ("fp16", "fp32")) -> None:
        """Re-hash the shipped files and compare with metadata."""
        for v in variants:
            p = self.model_path(v)
            if not p.is_file():
                raise SchemaError(f"[{self.pest}] missing model file: {p}")
            got = sha256_file(p)
            if got != self.model_sha256(v):
                raise SchemaError(
                    f"[{self.pest}] {v} sha256 mismatch\n  file: {got}\n  meta: "
                    f"{self.model_sha256(v)}"
                )
        n = self.normalization_path
        if not n.is_file():
            raise SchemaError(f"[{self.pest}] missing normalization: {n}")
        got = sha256_file(n)
        want = self.raw["files"]["normalization"]["sha256"]
        if got != want:
            raise SchemaError(
                f"[{self.pest}] normalization sha256 mismatch\n  file: {got}\n  meta: {want}"
            )


def load_metadata(models_root: Path, pest: str) -> PestMetadata:
    root = Path(models_root) / pest
    mp = root / "metadata.json"
    if not mp.is_file():
        avail = sorted(p.name for p in Path(models_root).iterdir() if p.is_dir())
        raise SchemaError(f"unknown pest {pest!r}: no {mp}. Available: {avail}")
    raw = json.loads(mp.read_text(encoding="utf-8"))
    ver = raw.get("metadata_schema_version")
    if ver != METADATA_SCHEMA_VERSION:
        raise SchemaError(
            f"[{pest}] metadata_schema_version={ver!r}, runtime expects "
            f"{METADATA_SCHEMA_VERSION!r}. Rebuild the package with build_package.py."
        )
    md = PestMetadata(raw=raw, root=root)
    _self_check(md)
    return md


def _self_check(md: PestMetadata) -> None:
    """Internal consistency of a single metadata blob."""
    e: list[str] = []
    if len(md.feature_names) != md.d_in:
        e.append(f"len(feature_names)={len(md.feature_names)} != d_in={md.d_in}")
    if md.T != md.doy_end - md.doy_start + 1:
        e.append(f"T={md.T} != doy_end-doy_start+1={md.doy_end - md.doy_start + 1}")
    if not (0 <= md.alert_idx < md.d_in):
        e.append(f"alert_idx={md.alert_idx} out of range for d_in={md.d_in}")
    elif md.feature_names[md.alert_idx] != "alert_tstar":
        e.append(
            f"feature_names[{md.alert_idx}]={md.feature_names[md.alert_idx]!r} != 'alert_tstar'"
        )
    if md.dispatch_channels != list(DISPATCH_FEATURE_NAMES) + [DISPATCH_MISSING_NAME]:
        e.append("dispatch_channels do not match the expected 15-channel order")
    nbase = len(md.base_channels)
    if 2 * nbase + len(md.dispatch_channels) != md.d_in:
        e.append(
            f"2*nbase({nbase}) + dispatch({len(md.dispatch_channels)}) != d_in={md.d_in}"
        )
    if md.mu_mode != "lead_from_alert":
        e.append(f"mu_mode={md.mu_mode!r} unsupported by this runtime")
    if e:
        raise SchemaError(f"[{md.pest}] metadata self-check failed:\n  - " + "\n  - ".join(e))


@dataclass
class DispatchRequest:
    """The Stage-1-derived context the standalone Stage-2 runner must be given.

    Stage 1 is out of scope for this package, so everything it would have
    produced is an explicit input. Nothing is defaulted to zero.
    """

    pest: str
    alert_tstar: int
    dispatch_features: dict[str, float]
    site: dict[str, float] = field(default_factory=dict)
    phenology: list[dict[str, float]] = field(default_factory=list)
    year: int | None = None

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "DispatchRequest":
        for k in ("pest", "alert_tstar", "dispatch_features"):
            if k not in d:
                raise SchemaError(f"dispatch request missing required key {k!r}")
        return DispatchRequest(
            pest=str(d["pest"]),
            alert_tstar=int(d["alert_tstar"]),
            dispatch_features=dict(d["dispatch_features"]),
            site=dict(d.get("site", {})),
            phenology=list(d.get("phenology", [])),
            year=(int(d["year"]) if d.get("year") is not None else None),
        )

    @staticmethod
    def from_json(path: Path) -> "DispatchRequest":
        p = Path(path)
        if not p.is_file():
            raise SchemaError(f"dispatch JSON not found: {p}")
        return DispatchRequest.from_dict(json.loads(p.read_text(encoding="utf-8")))

    def validate(self, md: PestMetadata) -> None:
        """Validate against the pest's metadata. Missing values are errors."""
        e: list[str] = []
        if self.pest != md.pest:
            e.append(f"pest={self.pest!r} does not match metadata pest={md.pest!r}")

        # --- dispatch features: exact set, no fill ------------------------
        want = set(DISPATCH_FEATURE_NAMES)
        got = set(self.dispatch_features)
        missing = sorted(want - got)
        extra = sorted(got - want)
        if missing:
            e.append(
                f"dispatch_features missing {len(missing)} required value(s): {missing}. "
                f"Supply every feature explicitly; they are not defaulted to 0."
            )
        if extra:
            e.append(f"dispatch_features has unknown key(s): {extra}")
        for k, v in self.dispatch_features.items():
            if k in want and k != "dispatch_branch" and not isinstance(v, (int, float)):
                e.append(f"dispatch_features[{k!r}]={v!r} is not numeric")

        # --- alert within the pest's season ------------------------------
        if not (md.doy_start <= self.alert_tstar <= md.doy_end):
            e.append(
                f"alert_tstar={self.alert_tstar} outside {md.pest} season "
                f"[{md.doy_start}, {md.doy_end}]"
            )

        # --- site coords / phenology, required per pest ------------------
        if md.requires_site_coords:
            for k in ("lat", "lon"):
                if k not in self.site:
                    e.append(
                        f"{md.pest} uses base channels {list(COORD_COLS)} -> "
                        f"site.{k} is required"
                    )
        if md.requires_phenology:
            if not self.phenology:
                e.append(
                    f"{md.pest} uses phenology base channels {list(PHENO_COLS)} -> "
                    f"'phenology' must be a non-empty list of "
                    f"{{obs_doy, {', '.join(PHENO_COLS)}}} records"
                )
            for i, row in enumerate(self.phenology):
                if "obs_doy" not in row:
                    e.append(f"phenology[{i}] missing 'obs_doy'")
                for c in PHENO_COLS:
                    if c in md.base_channels and c not in row:
                        e.append(f"phenology[{i}] missing {c!r}")
        if e:
            raise SchemaError(
                f"[{self.pest}] dispatch request invalid:\n  - " + "\n  - ".join(e)
            )
