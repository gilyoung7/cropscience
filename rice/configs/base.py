import os
from pathlib import Path


# The daily union table lives outside the repo, so it cannot be derived from RICE_ROOT.
# RICE_DAILY_CSV relocates it on hosts that do not mirror the first server's layout; the
# default is unchanged, so a machine with /home/gpu4080/ygdata behaves exactly as before.
PATH_DAILY = Path(os.environ.get(
    "RICE_DAILY_CSV",
    "/home/gpu4080/ygdata/rice/1997_2024_RICE_union_all_sites_with_GDD10_since_gs.csv"))
YEAR_MIN = None
YEAR_MAX = 2024

# Directory holding the per-pest observation tables (RICE_LONG_<korean name>.csv). Like
# PATH_DAILY this lives outside the repo, so it cannot be derived from RICE_ROOT. Each pest
# config joins its own filename onto this; only the directory is relocatable, so the Korean
# pest name stays where it belongs -- in that pest's config. Default unchanged, so a host with
# /home/gpu4080/ygdata behaves exactly as before.
LONG_BY_PEST_DIR = Path(os.environ.get(
    "RICE_LONG_BY_PEST_DIR", "/home/gpu4080/ygdata/rice/LONG_by_pest"))

# Always shared across all pests to avoid duplicated caches.
RICE_ROOT = Path(__file__).resolve().parents[1]
DAILY_CACHE_DIR = RICE_ROOT / "outputs" / "cache"

# Cache/version controls for daily preprocessing.
PREPROC_VERSION = "v1.0"
IMPUTE_POLICY = "ffill_bfill_interpolate_fill0"
MISS_INDICATOR_POLICY = "enabled"
