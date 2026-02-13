from pathlib import Path


PATH_DAILY = Path("/home/gpu4080/ygdata/rice/1997_2024_RICE_union_all_sites_with_GDD10_since_gs.csv")
YEAR_MIN = None
YEAR_MAX = 2024

# Always shared across all pests to avoid duplicated caches.
RICE_ROOT = Path(__file__).resolve().parents[1]
DAILY_CACHE_DIR = RICE_ROOT / "outputs" / "cache"

# Cache/version controls for daily preprocessing.
PREPROC_VERSION = "v1.0"
IMPUTE_POLICY = "ffill_bfill_interpolate_fill0"
MISS_INDICATOR_POLICY = "enabled"
