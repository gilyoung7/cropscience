from pathlib import Path
import importlib
from rice.configs.base import (
    PATH_DAILY,
    YEAR_MAX,
    DAILY_CACHE_DIR,
    PREPROC_VERSION,
    IMPUTE_POLICY,
    MISS_INDICATOR_POLICY,
)
# -----------------------------
# Minimal common runtime skeleton
# Values below are expected to be overridden by --pest config.
# -----------------------------
PATH_OBS = None
GDD_DIR = None

COUNT_COL = "obs_value"
THRESHOLD = 0.0
LABEL_COL = "label_event"
PEST_COL = "pest"
TARGET_PEST = None
APPLY_PEST_FILTER = False

# Needed at import-time in several modules/CLI defaults.
SPLIT_SEED = 42
LEFT_WINDOW_DAYS = 15

# DataLoader defaults (can be overridden by pest config via apply_pest_config).
BATCH_TRAIN = 64
BATCH_EVAL = 128
NUM_WORKERS = 4
PIN_MEMORY = True
PERSISTENT_WORKERS = True
PREFETCH_FACTOR = 2

PEST_SLUG = None


def apply_pest_config(pest_slug: str):
    mod = importlib.import_module(f"rice.pests.{pest_slug}.config")
    for k in dir(mod):
        if k.isupper():
            globals()[k] = getattr(mod, k)
    globals()["PEST_SLUG"] = pest_slug
    # cache dir is globally fixed and must not be pest-specific
    from rice.configs.base import DAILY_CACHE_DIR as _FIXED_CACHE_DIR
    globals()["DAILY_CACHE_DIR"] = _FIXED_CACHE_DIR
