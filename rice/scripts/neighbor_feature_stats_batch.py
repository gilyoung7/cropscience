"""Batch neighbor-occurrence feature stats across all pests -> one TSV/CSV.

Runs the same computation as ``neighbor_history_utils.py`` for each pest LONG
CSV and writes one stats row per pest. STATS ONLY — does not touch any training
checkpoint or training pipeline.

Usage:
    cd /home/gpu4080/research/cropscience
    python rice/scripts/neighbor_feature_stats_batch.py
    python rice/scripts/neighbor_feature_stats_batch.py --pests BPH,WBPH --limit 2000
    python rice/scripts/neighbor_feature_stats_batch.py --out rice/outputs_stage1/neighbor_stats.csv

Output columns (in order):
    pest, sites, event_rows, evaluated_site_years,
    neighbor_any_7d_30km_mean,
    neighbor_count_14d_30km_mean, neighbor_count_14d_30km_positive_ratio,
    neighbor_count_30d_50km_mean, neighbor_count_30d_50km_positive_ratio,
    neighbor_weighted_14d_50km_mean, neighbor_weighted_14d_50km_max,
    neighbor_min_dist_14d_50km_miss_ratio
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.scripts.neighbor_history_utils import (
    DEFAULT_DECAY_KM,
    MIN_DIST_MISS_NAME,
    NEIGHBOR_CHANNEL_NAMES,
    NEIGHBOR_FEATURE_NAMES,
    build_neighbor_index,
    compute_neighbor_block,
    load_long_events,
)

# The 8 pests backed by a per-pest LONG_by_pest CSV (BPH2 is a BPH variant on a
# separate merged file and is excluded from the default sweep).
DEFAULT_PESTS = [
    "sheath_blight",
    "blast",
    "bacterial_blight",
    "brown_spot",
    "BPH",
    "WBPH",
    "rice_stem_borer_1",
    "rice_stem_borer_2",
]

OUTPUT_COLUMNS = [
    "pest",
    "sites",
    "event_rows",
    "evaluated_site_years",
    "neighbor_any_7d_30km_mean",
    "neighbor_count_14d_30km_mean",
    "neighbor_count_14d_30km_positive_ratio",
    "neighbor_count_30d_50km_mean",
    "neighbor_count_30d_50km_positive_ratio",
    "neighbor_weighted_14d_50km_mean",
    "neighbor_weighted_14d_50km_max",
    "neighbor_min_dist_14d_50km_miss_ratio",
]


def compute_pest_stats(pest: str, limit: int | None, decay_km: float) -> dict:
    """Resolve a pest config, compute neighbor features, return one stats row."""
    resolve_pest(pest)  # applies PATH_OBS / DOY_START / DOY_END / LABEL_COL / YEAR_* into C
    path = C.PATH_OBS
    doy_start = int(C.DOY_START)
    doy_end = int(C.DOY_END)
    T = doy_end - doy_start + 1
    label_col = getattr(C, "LABEL_COL", "label_event")
    year_min = getattr(C, "YEAR_MIN", None)
    year_max = getattr(C, "YEAR_MAX", None)

    events_df, coords_df, site_years = load_long_events(
        path, label_col=label_col, year_min=year_min, year_max=year_max
    )
    index = build_neighbor_index(events_df, coords_df)

    use = site_years if (limit is None or int(limit) <= 0) else site_years[: int(limit)]

    nfeat = len(NEIGHBOR_FEATURE_NAMES)
    sums = np.zeros(nfeat, dtype=np.float64)
    maxs = np.full(nfeat, -np.inf, dtype=np.float64)
    n_rows = 0
    pos14 = 0
    pos30 = 0
    miss_sum = 0.0
    i_c14 = NEIGHBOR_FEATURE_NAMES.index("neighbor_count_14d_30km")
    i_c30 = NEIGHBOR_FEATURE_NAMES.index("neighbor_count_30d_50km")
    miss_idx = NEIGHBOR_CHANNEL_NAMES.index(MIN_DIST_MISS_NAME)

    for site_id, year in use:
        block = compute_neighbor_block(site_id, year, doy_start, T, index, decay_km=decay_km)
        feat = block[:, :nfeat]  # exclude the miss indicator from mean/max table
        sums += feat.sum(axis=0)
        maxs = np.maximum(maxs, feat.max(axis=0))
        n_rows += feat.shape[0]
        pos14 += int((feat[:, i_c14] > 0).sum())
        pos30 += int((feat[:, i_c30] > 0).sum())
        miss_sum += float(block[:, miss_idx].sum())

    denom = max(n_rows, 1)
    means = sums / denom
    maxs = np.where(np.isfinite(maxs), maxs, 0.0)

    def mean_of(name: str) -> float:
        return float(means[NEIGHBOR_FEATURE_NAMES.index(name)])

    def max_of(name: str) -> float:
        return float(maxs[NEIGHBOR_FEATURE_NAMES.index(name)])

    print(
        f"[{pest}] sites={len(index.site_ids)} event_rows={len(events_df)} "
        f"site_years_eval={len(use)} day_rows={n_rows} "
        f"DOY {doy_start}-{doy_end} (T={T})"
    )

    return {
        "pest": pest,
        "sites": int(len(index.site_ids)),
        "event_rows": int(len(events_df)),
        "evaluated_site_years": int(len(use)),
        "neighbor_any_7d_30km_mean": mean_of("neighbor_any_7d_30km"),
        "neighbor_count_14d_30km_mean": mean_of("neighbor_count_14d_30km"),
        "neighbor_count_14d_30km_positive_ratio": pos14 / denom,
        "neighbor_count_30d_50km_mean": mean_of("neighbor_count_30d_50km"),
        "neighbor_count_30d_50km_positive_ratio": pos30 / denom,
        "neighbor_weighted_14d_50km_mean": mean_of("neighbor_weighted_14d_50km"),
        "neighbor_weighted_14d_50km_max": max_of("neighbor_weighted_14d_50km"),
        "neighbor_min_dist_14d_50km_miss_ratio": miss_sum / denom,
    }


def main():
    ap = argparse.ArgumentParser(description="Batch neighbor-feature stats across pests.")
    ap.add_argument(
        "--pests",
        default=",".join(DEFAULT_PESTS),
        help="comma-separated pest slugs, or 'all' for every available pest",
    )
    ap.add_argument("--limit", type=int, default=0,
                    help="max site-years per pest (0 = all)")
    ap.add_argument("--decay_km", type=float, default=DEFAULT_DECAY_KM)
    ap.add_argument("--out", default="rice/outputs_stage1/neighbor_feature_stats.tsv",
                    help="output path; .csv -> comma-separated, else tab-separated")
    args = ap.parse_args()

    if str(args.pests).strip().lower() == "all":
        from rice.src.pest_resolver import available_pest_slugs
        pests = available_pest_slugs()
    else:
        pests = [p.strip() for p in str(args.pests).split(",") if p.strip()]

    limit = None if int(args.limit) <= 0 else int(args.limit)
    print(f"[cfg] pests={pests} limit={limit} decay_km={args.decay_km}")

    rows = []
    failures = []
    for pest in pests:
        try:
            rows.append(compute_pest_stats(pest, limit, float(args.decay_km)))
        except Exception as e:  # keep going so one bad pest doesn't sink the sweep
            print(f"[WARN] pest={pest} failed: {type(e).__name__}: {e}", file=sys.stderr)
            failures.append((pest, str(e)))

    if not rows:
        raise SystemExit("[abort] no pest produced stats")

    df = pd.DataFrame(rows)[OUTPUT_COLUMNS]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sep = "," if out_path.suffix.lower() == ".csv" else "\t"
    df.to_csv(out_path, sep=sep, index=False)

    print(f"\n[ok] wrote {len(df)} rows -> {out_path}")
    with pd.option_context("display.width", 200, "display.max_columns", None):
        print(df.to_string(index=False))
    if failures:
        print(f"\n[note] {len(failures)} pest(s) skipped: "
              + ", ".join(f"{p}({msg})" for p, msg in failures))


if __name__ == "__main__":
    main()
