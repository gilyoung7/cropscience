from pathlib import Path


PATH_DAILY = Path("/home/gpu4080/ygdata/etc/joined_SAMPLE_GDD_조명나방.csv")
YEAR_MIN = None
YEAR_MAX = 2025

# Always shared across all pests to avoid duplicated caches.
ETC_ROOT = Path(__file__).resolve().parents[1]
DAILY_CACHE_DIR = ETC_ROOT / "outputs" / "cache"

# Cache/version controls for daily preprocessing.
PREPROC_VERSION = "v1.0"
IMPUTE_POLICY = "ffill_bfill_interpolate_fill0"
MISS_INDICATOR_POLICY = "enabled"
