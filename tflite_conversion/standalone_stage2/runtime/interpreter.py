"""Standalone Stage-2 LiteRT/TFLite runner — no torch, no tensorflow.

Loads a per-pest FP16 (default) or FP32 .tflite built by build_package.py, runs
the mu head, and reports mu_doy plus the 95% prediction interval using exactly
the API's rule (see `_prediction_interval`).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .preprocessing import BuiltInput, build_input, load_daily_csv
from .schema import DispatchRequest, PestMetadata, SchemaError, load_metadata

DEFAULT_VARIANT = "fp16"
VARIANTS = ("fp16", "fp32")


def package_root() -> Path:
    """Package root: the directory containing models/ and runtime/."""
    return Path(__file__).resolve().parents[1]


def models_root() -> Path:
    return package_root() / "models"


def _load_norm(md: PestMetadata) -> tuple[np.ndarray, np.ndarray]:
    """norm_mean/norm_std from normalization.npz (float32, no precision loss)."""
    with np.load(md.normalization_path) as z:
        mean = np.asarray(z["norm_mean"], dtype=np.float32)
        std = np.asarray(z["norm_std"], dtype=np.float32)
    if mean.shape != (md.d_in,) or std.shape != (md.d_in,):
        raise SchemaError(
            f"[{md.pest}] normalization shapes {mean.shape}/{std.shape} != (d_in={md.d_in},)"
        )
    return mean, std


class Stage2Model:
    """One pest + variant, kept loaded so repeated calls reuse the interpreter."""

    def __init__(self, pest: str, variant: str = DEFAULT_VARIANT,
                 models_dir: Path | None = None, verify_sha256: bool = True):
        if variant not in VARIANTS:
            raise SchemaError(f"variant must be one of {VARIANTS}, got {variant!r}")
        self.md = load_metadata(models_dir or models_root(), pest)
        self.variant = variant
        if verify_sha256:
            self.md.verify_files(variants=(variant,))
        self.norm_mean, self.norm_std = _load_norm(self.md)

        # Imported lazily so `import runtime.schema` works without the interpreter.
        from ai_edge_litert.interpreter import Interpreter

        path = self.md.model_path(variant)
        if not path.is_file():
            raise SchemaError(f"[{pest}] model file missing: {path}")
        self._interp = Interpreter(model_path=str(path))
        self._interp.allocate_tensors()
        self._check_io()

    def _check_io(self) -> None:
        want = {
            "X": ((1, 1, self.md.T, self.md.d_in), np.float32),
            "tstar": ((1, 1), np.int64),
            "valid_mask": ((1, 1), np.bool_),
        }
        details = self._interp.get_input_details()
        if len(details) != len(want):
            raise SchemaError(
                f"[{self.md.pest}] model has {len(details)} inputs, expected {len(want)}"
            )
        shapes = {tuple(int(x) for x in d["shape"]) for d in details}
        for name, (shape, _) in want.items():
            if shape not in shapes:
                raise SchemaError(
                    f"[{self.md.pest}] model has no input with shape {shape} (for {name}); "
                    f"model inputs: {sorted(shapes)}"
                )
        out = self._interp.get_output_details()
        if len(out) != 1:
            raise SchemaError(f"[{self.md.pest}] model has {len(out)} outputs, expected 1")

    def _invoke(self, built: BuiltInput) -> float:
        """Bind by dtype+shape, not by index.

        LiteRT does not guarantee input tensor order matches the Python
        signature, so match each declared input to the argument of the same
        dtype and shape.
        """
        args = {
            "X": built.X,
            "tstar": built.tstar,
            "valid_mask": built.valid_mask,
        }
        used: set[str] = set()
        for d in self._interp.get_input_details():
            shape, dtype = tuple(int(x) for x in d["shape"]), d["dtype"]
            match = None
            for name, arr in args.items():
                if name in used:
                    continue
                cand = arr.astype(dtype) if arr.dtype != dtype else arr
                if tuple(cand.shape) == shape:
                    match = (name, cand)
                    break
            if match is None:
                raise SchemaError(
                    f"[{self.md.pest}] cannot bind model input {d['name']!r} "
                    f"shape={shape} dtype={np.dtype(dtype).name}"
                )
            used.add(match[0])
            self._interp.set_tensor(d["index"], match[1])
        if len(used) != len(args):
            raise SchemaError(f"[{self.md.pest}] bound only {sorted(used)} of {sorted(args)}")
        self._interp.invoke()
        out_d = self._interp.get_output_details()[0]
        return float(np.asarray(self._interp.get_tensor(out_d["index"])).reshape(-1)[0])

    def predict(self, daily: pd.DataFrame, req: DispatchRequest) -> dict[str, Any]:
        built = build_input(self.md, daily, req, self.norm_mean, self.norm_std)
        mu_rel = self._invoke(built)
        return _format_result(self.md, self.variant, built, mu_rel)


def _prediction_interval(md: PestMetadata, mu_doy_temporal: float) -> dict[str, Any]:
    """Exactly the API's rule — api_handoff_transformer/run_predict.py:490-498.

        sigma = ckpt sigma (5.0)
        half  = round(1.96 * sigma, 1)          -> 9.8
        mu_doy = round(mu_doy_temporal, 2)
        lower  = int(round(mu_doy_temporal - half))
        upper  = int(round(mu_doy_temporal + half))

    Two details replicated deliberately:
      * the interval is derived from the UNROUNDED mu, not from the rounded
        mu_doy that is reported;
      * Python's round() is banker's rounding (round-half-to-even), so the
        values must be plain Python floats here, not numpy scalars, to match
        the API bit-for-bit at .5 boundaries.
    """
    sigma = float(md.sigma)
    half = round(1.96 * sigma, 1)
    mu = float(mu_doy_temporal)
    return {
        "mu_doy": round(mu, 2),
        "pi_95": [int(round(mu - half)), int(round(mu + half))],
        "sigma_days": sigma,
        "half_width_days": half,
    }


def _format_result(md: PestMetadata, variant: str, built: BuiltInput,
                   mu_rel: float) -> dict[str, Any]:
    # mu is a 1-based season index; DOY = mu + doy_start - 1 (run_predict.py:477)
    mu_doy_temporal = float(mu_rel) + md.doy_start - 1.0
    pi = _prediction_interval(md, mu_doy_temporal)
    return {
        "pest": md.pest,
        "backend": f"tflite_{variant}",
        "year": built.year,
        "alert_tstar": built.alert_tstar_doy,
        "mu_rel_season_index": round(float(mu_rel), 4),
        "mu_doy": pi["mu_doy"],
        "prediction_interval_95": pi["pi_95"],
        "sigma_days": pi["sigma_days"],
        "tstar_season_index": built.tstar_season_index,
        "selected_offset": md.selected_offset,
        "input_shape": list(built.X.shape),
        "model_sha256": md.model_sha256(variant),
        "metadata_schema_version": md.schema_version,
    }


_CACHE: dict[tuple[str, str, str], Stage2Model] = {}


def predict_stage2(
    pest: str,
    daily_data: pd.DataFrame | str | Path,
    alert_tstar: int,
    dispatch_features: dict[str, float],
    variant: str = DEFAULT_VARIANT,
    site: dict[str, float] | None = None,
    phenology: list[dict[str, float]] | None = None,
    year: int | None = None,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Run standalone Stage-2 for one (pest, site-year).

    Args:
      pest: one of the 8 packaged pests.
      daily_data: DataFrame, or a path to a daily weather CSV (Korean schema).
        Must cover DOY 1..doy_end of the target year contiguously.
      alert_tstar: alert DOY from Stage-1 (out of scope here — supply it).
      dispatch_features: all 14 Stage-1 dispatch features. No value is defaulted.
      variant: "fp16" (default) or "fp32".
      site: {"lat":..., "lon":...} — required for the 7 non-BPH pests.
      phenology: LONG-style records {obs_doy, days_since_growing_start,
        days_until_growing_end, is_growing} — required for the 7 non-BPH pests.
      year: needed only if daily_data spans multiple years.

    Returns a dict with mu_doy and prediction_interval_95.
    """
    daily = (
        daily_data if isinstance(daily_data, pd.DataFrame)
        else load_daily_csv(Path(daily_data))
    )
    req = DispatchRequest(
        pest=pest,
        alert_tstar=int(alert_tstar),
        dispatch_features=dict(dispatch_features),
        site=dict(site or {}),
        phenology=list(phenology or []),
        year=year,
    )
    key = (pest, variant, str(models_dir or models_root()))
    if key not in _CACHE:
        _CACHE[key] = Stage2Model(pest, variant=variant, models_dir=models_dir)
    return _CACHE[key].predict(daily, req)
