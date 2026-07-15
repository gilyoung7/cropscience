"""Standalone Stage-2 pest-timing runtime (LiteRT/TFLite).

Runs the 8 deployed Stage-2 models with no PyTorch, no TensorFlow, no XGBoost
and no .pt checkpoints. Requires only: ai-edge-litert, numpy, pandas.

Stage 1, weather-API ingestion, and the existing API wiring are out of scope:
`alert_tstar`, the 14 dispatch features, and (for the 7 non-BPH pests) site
coordinates and phenology are explicit inputs.

    from runtime import predict_stage2
    out = predict_stage2(pest="BPH", daily_data="daily.csv", alert_tstar=176,
                         dispatch_features={...}, variant="fp16")
"""

from .interpreter import (  # noqa: F401
    DEFAULT_VARIANT,
    VARIANTS,
    Stage2Model,
    models_root,
    package_root,
    predict_stage2,
)
from .preprocessing import BuiltInput, PreprocessError, build_input, load_daily_csv  # noqa: F401
from .schema import (  # noqa: F401
    DISPATCH_FEATURE_NAMES,
    METADATA_SCHEMA_VERSION,
    DispatchRequest,
    PestMetadata,
    SchemaError,
    load_metadata,
)

__all__ = [
    "predict_stage2",
    "Stage2Model",
    "DispatchRequest",
    "PestMetadata",
    "load_metadata",
    "build_input",
    "load_daily_csv",
    "BuiltInput",
    "SchemaError",
    "PreprocessError",
    "DISPATCH_FEATURE_NAMES",
    "METADATA_SCHEMA_VERSION",
    "DEFAULT_VARIANT",
    "VARIANTS",
    "models_root",
    "package_root",
]
