"""Stage-2 via LiteRT/TFLite — FP16 default, FP32 optional. No torch, no TF.

Reuses the metadata + normalization + tensor builder from the standalone Stage-2
package (proven bit-exact against the deployed build_real_input, 8/8 pests).
mu_doy and the prediction interval follow the deployed rule exactly.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ._stage2_preprocessing import build_input
from ._stage2_schema import DispatchRequest, PestMetadata, SchemaError, load_metadata

DEFAULT_VARIANT = "fp16"
VARIANTS = ("fp16", "fp32")


class Stage2Error(RuntimeError):
    """Stage-2 could not produce a learned prediction."""


def _load_norm(md: PestMetadata) -> tuple[np.ndarray, np.ndarray]:
    with np.load(md.normalization_path) as z:
        mean = np.asarray(z["norm_mean"], dtype=np.float32)
        std = np.asarray(z["norm_std"], dtype=np.float32)
    if mean.shape != (md.d_in,) or std.shape != (md.d_in,):
        raise Stage2Error(f"[{md.pest}] normalization shape != (d_in={md.d_in},)")
    return mean, std


def prediction_interval(sigma: float, mu_doy_temporal: float) -> dict:
    """EXACT port of api_handoff_transformer/run_predict.py:490-498.

        half   = round(1.96 * sigma, 1)
        mu_doy = round(mu_doy_temporal, 2)
        lower  = int(round(mu_doy_temporal - half))
        upper  = int(round(mu_doy_temporal + half))

    Two details are deliberate: the interval derives from the UNROUNDED mu, and
    Python's round() is round-half-to-even — so plain floats are used, never
    numpy scalars, to match at .5 boundaries.
    """
    sigma = float(sigma)
    half = round(1.96 * sigma, 1)
    mu = float(mu_doy_temporal)
    return {
        "mu_doy": round(mu, 2),
        "pi_95": {
            "lower_doy": int(round(mu - half)),
            "upper_doy": int(round(mu + half)),
            "sigma_days": sigma,
        },
    }


class Stage2Model:
    """One pest + variant; the interpreter is kept loaded across calls."""

    def __init__(self, pest: str, models_dir: Path, variant: str = DEFAULT_VARIANT,
                 verify_sha256: bool = False):
        if variant not in VARIANTS:
            raise Stage2Error(f"variant must be one of {VARIANTS}, got {variant!r}")
        try:
            self.md = load_metadata(Path(models_dir), pest)
        except SchemaError as e:
            raise Stage2Error(str(e)) from e
        self.variant = variant
        if verify_sha256:
            self.md.verify_files(variants=(variant,))
        self.norm_mean, self.norm_std = _load_norm(self.md)
        from ai_edge_litert.interpreter import Interpreter

        path = self.md.model_path(variant)
        if not path.is_file():
            raise Stage2Error(
                f"[{pest}] {variant} model missing: {path}. Rebuild with "
                f"build_package.py, or pass --stage2-variant fp32."
            )
        self._interp = Interpreter(model_path=str(path))
        self._interp.allocate_tensors()

    @property
    def sigma(self) -> float:
        return float(self.md.sigma)

    @property
    def doy_start(self) -> int:
        return int(self.md.doy_start)

    def _invoke(self, X, tstar, valid_mask) -> float:
        """Bind inputs by dtype+shape — LiteRT does not guarantee signature order."""
        args = {"X": X, "tstar": tstar, "valid_mask": valid_mask}
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
                raise Stage2Error(
                    f"[{self.md.pest}] cannot bind model input {d['name']!r} shape={shape}"
                )
            used.add(match[0])
            self._interp.set_tensor(d["index"], match[1])
        self._interp.invoke()
        out = self._interp.get_output_details()[0]
        return float(np.asarray(self._interp.get_tensor(out["index"])).reshape(-1)[0])

    def predict(self, daily: pd.DataFrame, pest: str, alert_tstar: int,
                dispatch_features: dict, site: dict, phenology: list[dict],
                year: int) -> dict:
        """Build the Stage-2 tensor and run mu. Returns mu_rel + diagnostics."""
        req = DispatchRequest(
            pest=pest, alert_tstar=int(alert_tstar),
            dispatch_features=dict(dispatch_features),
            site=dict(site or {}), phenology=list(phenology or []), year=int(year),
        )
        built = build_input(self.md, daily, req, self.norm_mean, self.norm_std)
        mu_rel = self._invoke(built.X, built.tstar, built.valid_mask)
        return {
            "mu_rel": mu_rel,
            "mu_doy_temporal": float(mu_rel) + self.doy_start - 1.0,
            "built": built,
        }
