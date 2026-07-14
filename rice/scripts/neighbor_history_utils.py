"""Neighbor (spatial) occurrence features for Stage-1 — standalone util.

Mirrors the structure + leakage-prevention pattern of
``scripts/site_history_utils.py`` but, instead of the *same* site's past years,
it looks at *other* sites' occurrences earlier in the **same** season.

5 dynamic channels (+ 1 missing-indicator), per (site, year, day):
    1  neighbor_any_7d_30km        any other-site event in [t-7,  t) within 30 km  -> {0,1}
    2  neighbor_count_14d_30km     # other-site events in [t-14, t) within 30 km
    3  neighbor_count_30d_50km     # other-site events in [t-30, t) within 50 km
    4  neighbor_weighted_14d_50km  sum exp(-dist_km/decay) over [t-14, t) within 50 km
    5  neighbor_min_dist_14d_50km  min dist (km) among [t-14, t) within 50 km (else FILL)
  (+) neighbor_min_dist_14d_50km_miss  1.0 when no qualifying neighbor that day, else 0.0

Definition (for a sample at site=i, year=y, day=t):
  Use ONLY records with label_event > 0, from OTHER sites (site_id != i), in the
  SAME year y, with ``obs_doy < t`` (strict). ``obs_doy == t`` is excluded and
  ``obs_doy > t`` is impossible by construction -> no future-information leak.
  Each channel further restricts to its own (window_days, radius_km).

Distance: haversine in km, from per-site coordinates (좌표-위도/좌표-경도).
Weighted channel decay length defaults to 20 km: weight = exp(-dist_km / 20).

Performance: site<->site haversine distances are precomputed once into an
N x N matrix (N = #sites is small, ~hundreds). Per site-year the per-day block
is computed with a single (T x M) broadcast over that year's M events. (A
BallTree(metric='haversine') is an equivalent drop-in if N ever grows large.)

CLI sanity check (no training pipeline needed; reads the pest LONG CSV only):
    python scripts/neighbor_history_utils.py --pest sheath_blight --limit 1000
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Channel specs / names
# ---------------------------------------------------------------------------
DEFAULT_DECAY_KM = 20.0
# Sentinel distance used when a day has no qualifying neighbor (kept large and
# positive, in the spirit of site_history_utils' 99 / -1 sentinels). The paired
# *_miss indicator lets a model distinguish "far" from "none".
MIN_DIST_FILL = 999.0

# (name, window_days, radius_km, kind). kind in {any, count, weighted, min_dist}.
NEIGHBOR_FEATURE_SPECS = [
    {"name": "neighbor_any_7d_30km",        "window": 7,  "radius": 30.0, "kind": "any"},
    {"name": "neighbor_count_14d_30km",     "window": 14, "radius": 30.0, "kind": "count"},
    {"name": "neighbor_count_30d_50km",     "window": 30, "radius": 50.0, "kind": "count"},
    {"name": "neighbor_weighted_14d_50km",  "window": 14, "radius": 50.0, "kind": "weighted"},
    {"name": "neighbor_min_dist_14d_50km",  "window": 14, "radius": 50.0, "kind": "min_dist"},
]
MIN_DIST_SPEC_NAME = "neighbor_min_dist_14d_50km"
MIN_DIST_MISS_NAME = "neighbor_min_dist_14d_50km_miss"

NEIGHBOR_FEATURE_NAMES = [s["name"] for s in NEIGHBOR_FEATURE_SPECS]
NEIGHBOR_MISS_NAMES = [MIN_DIST_MISS_NAME]
# Order of appended channels in X (5 features + 1 miss indicator).
NEIGHBOR_CHANNEL_NAMES = NEIGHBOR_FEATURE_NAMES + NEIGHBOR_MISS_NAMES
NEIGHBOR_FEATURE_DIM = len(NEIGHBOR_CHANNEL_NAMES)

_LAT_COL = "좌표-위도"
_LON_COL = "좌표-경도"


# ---------------------------------------------------------------------------
# Distance helpers (haversine, km)
# ---------------------------------------------------------------------------
_EARTH_R_KM = 6371.0088


def haversine_km(lat1, lon1, lat2, lon2):
    """Vectorized haversine distance in km. Inputs in degrees."""
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lon1 = np.radians(np.asarray(lon1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    lon2 = np.radians(np.asarray(lon2, dtype=float))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * _EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


def _haversine_matrix(lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Full N x N haversine distance matrix (km) from site coordinate arrays."""
    lat_r = np.radians(lat)
    lon_r = np.radians(lon)
    dlat = lat_r[:, None] - lat_r[None, :]
    dlon = lon_r[:, None] - lon_r[None, :]
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_r)[:, None] * np.cos(lat_r)[None, :] * np.sin(dlon / 2.0) ** 2
    )
    return 2.0 * _EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))


# ---------------------------------------------------------------------------
# Index: site coordinates + per-year event lists
# ---------------------------------------------------------------------------
class NeighborIndex:
    """Precomputed spatial index for neighbor-feature computation.

    Attributes
    ----------
    site_ids : list[str]                      sorted site ids
    site_to_idx : dict[str, int]              site id -> row index in ``dist``
    dist : np.ndarray (N, N) float64          site<->site haversine km
    events_by_year : dict[int, dict]          year -> {"site_idx": (M,), "doy": (M,)}
                                              only label_event > 0 records.
    """

    __slots__ = ("site_ids", "site_to_idx", "dist", "events_by_year")

    def __init__(self, site_ids, site_to_idx, dist, events_by_year):
        self.site_ids = site_ids
        self.site_to_idx = site_to_idx
        self.dist = dist
        self.events_by_year = events_by_year


def build_neighbor_index(events_df: pd.DataFrame, coords_df: pd.DataFrame) -> NeighborIndex:
    """Build a :class:`NeighborIndex`.

    Parameters
    ----------
    events_df : DataFrame with columns [site_id, year, obs_doy]
        Already filtered to label_event > 0 (occurrence records only).
    coords_df : DataFrame with columns [site_id, lat, lon]
        One row per site.
    """
    coords_df = coords_df.dropna(subset=["lat", "lon"]).drop_duplicates("site_id").copy()
    coords_df["site_id"] = coords_df["site_id"].astype(str)
    coords_df = coords_df.sort_values("site_id").reset_index(drop=True)

    site_ids = coords_df["site_id"].tolist()
    site_to_idx = {sid: i for i, sid in enumerate(site_ids)}
    lat = coords_df["lat"].to_numpy(dtype=float)
    lon = coords_df["lon"].to_numpy(dtype=float)
    dist = _haversine_matrix(lat, lon)

    ev = events_df.copy()
    ev["site_id"] = ev["site_id"].astype(str)
    ev = ev[ev["site_id"].isin(site_to_idx)]
    events_by_year: dict[int, dict] = {}
    for year, sub in ev.groupby("year", sort=False):
        s_idx = sub["site_id"].map(site_to_idx).to_numpy(dtype=np.int64)
        doy = pd.to_numeric(sub["obs_doy"], errors="coerce").to_numpy()
        ok = np.isfinite(doy)
        events_by_year[int(year)] = {
            "site_idx": s_idx[ok],
            "doy": doy[ok].astype(np.int64),
        }
    return NeighborIndex(site_ids, site_to_idx, dist, events_by_year)


# ---------------------------------------------------------------------------
# Per-(site, year) feature block
# ---------------------------------------------------------------------------
def compute_neighbor_block(
    site_id,
    year: int,
    doy_start: int,
    T: int,
    index: NeighborIndex,
    decay_km: float = DEFAULT_DECAY_KM,
) -> np.ndarray:
    """Return a (T, NEIGHBOR_FEATURE_DIM) float32 channel block.

    Row k corresponds to absolute ``doy = doy_start + k`` (same convention as
    build_samples_season / site_history_utils).
    """
    block = np.zeros((T, NEIGHBOR_FEATURE_DIM), dtype=np.float32)
    midx = NEIGHBOR_CHANNEL_NAMES.index(MIN_DIST_SPEC_NAME)
    miss_idx = NEIGHBOR_CHANNEL_NAMES.index(MIN_DIST_MISS_NAME)
    # Defaults for the "no neighbor" case.
    block[:, midx] = MIN_DIST_FILL
    block[:, miss_idx] = 1.0

    si = index.site_to_idx.get(str(site_id))
    yev = index.events_by_year.get(int(year))
    if si is None or yev is None or yev["doy"].size == 0:
        return block

    e_site = yev["site_idx"]
    e_doy = yev["doy"]
    keep = e_site != si  # exclude the sample's own site
    if not np.any(keep):
        return block
    e_site = e_site[keep]
    e_doy = e_doy[keep]
    dist_e = index.dist[si, e_site]  # (M,)

    days = np.arange(T, dtype=np.int64) + int(doy_start)  # (T,)
    dd = days[:, None] - e_doy[None, :]                   # (T, M) = t - obs_doy
    # Strict future-leak guard: obs_doy < t  <=>  dd > 0. (dd == 0 -> excluded.)
    time_pos = dd > 0

    for spec in NEIGHBOR_FEATURE_SPECS:
        w = int(spec["window"])
        r = float(spec["radius"])
        mask = time_pos & (dd <= w) & (dist_e[None, :] <= r)  # (T, M)
        ch = NEIGHBOR_CHANNEL_NAMES.index(spec["name"])
        kind = spec["kind"]
        if kind == "any":
            block[:, ch] = mask.any(axis=1).astype(np.float32)
        elif kind == "count":
            block[:, ch] = mask.sum(axis=1).astype(np.float32)
        elif kind == "weighted":
            wt = np.where(mask, np.exp(-dist_e[None, :] / float(decay_km)), 0.0)
            block[:, ch] = wt.sum(axis=1).astype(np.float32)
        elif kind == "min_dist":
            d_masked = np.where(mask, dist_e[None, :], np.inf)  # (T, M)
            mind = d_masked.min(axis=1)                          # (T,)
            has = np.isfinite(mind)
            block[:, ch] = np.where(has, mind, MIN_DIST_FILL).astype(np.float32)
            block[:, miss_idx] = np.where(has, 0.0, 1.0).astype(np.float32)
        else:  # pragma: no cover - guarded by spec list
            raise ValueError(f"unknown neighbor feature kind: {kind}")
    return block


def append_neighbor_to_samples(
    samples: list[dict],
    index: NeighborIndex,
    doy_start: int,
    decay_km: float = DEFAULT_DECAY_KM,
) -> int:
    """In-place: append NEIGHBOR_FEATURE_DIM channels to each sample's X.

    Mirrors ``site_history_utils.append_history_to_samples``. Samples whose
    (site_id, year) is unknown to the index get the zero / FILL / miss=1
    defaults from :func:`compute_neighbor_block`. Returns count processed.
    """
    n_done = 0
    for s in samples:
        X_old = np.asarray(s["X"], dtype=np.float32)
        T = int(X_old.shape[0])
        block = compute_neighbor_block(
            s["site_id"], int(s["year"]), int(doy_start), T, index, decay_km=decay_km
        )
        s["X"] = np.concatenate([X_old, block], axis=1).astype(np.float32)
        n_done += 1
    return n_done


# ---------------------------------------------------------------------------
# LONG CSV loader (standalone path)
# ---------------------------------------------------------------------------
def load_long_events(
    path,
    label_col: str = "label_event",
    lat_col: str = _LAT_COL,
    lon_col: str = _LON_COL,
    year_min: int | None = None,
    year_max: int | None = None,
):
    """Read a pest LONG CSV -> (events_df, coords_df, site_years).

    events_df : [site_id, year, obs_doy]  (label_event > 0 only)
    coords_df : [site_id, lat, lon]       (median per site)
    site_years: sorted list of (site_id, year) present in the file (any row)
    """
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    need = ["site_id", "year", "obs_doy", label_col, lat_col, lon_col]
    missing = [c for c in need if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in LONG csv: {missing} (have={list(df.columns)[:20]})")

    df["site_id"] = df["site_id"].astype(str)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df["obs_doy"] = pd.to_numeric(df["obs_doy"], errors="coerce")
    df[label_col] = pd.to_numeric(df[label_col], errors="coerce").fillna(0.0)
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lon_col] = pd.to_numeric(df[lon_col], errors="coerce")
    df = df.dropna(subset=["year", "obs_doy"]).copy()
    df["year"] = df["year"].astype(int)
    df["obs_doy"] = df["obs_doy"].astype(int)
    if year_min is not None:
        df = df[df["year"] >= int(year_min)]
    if year_max is not None:
        df = df[df["year"] <= int(year_max)]

    coords_df = (
        df[["site_id", lat_col, lon_col]]
        .rename(columns={lat_col: "lat", lon_col: "lon"})
        .groupby("site_id", as_index=False)[["lat", "lon"]]
        .median()
    )
    events_df = df.loc[df[label_col] > 0, ["site_id", "year", "obs_doy"]].copy()
    site_years = sorted({(r.site_id, int(r.year)) for r in df.itertuples(index=False)})
    return events_df, coords_df, site_years


# ---------------------------------------------------------------------------
# Reference (independent) computation — used only for the leakage assert.
# ---------------------------------------------------------------------------
def _reference_block(site_id, year, doy_start, T, events_df, coords_df, decay_km):
    """Slow, obviously-correct re-implementation for cross-checking."""
    block = np.zeros((T, NEIGHBOR_FEATURE_DIM), dtype=np.float32)
    midx = NEIGHBOR_CHANNEL_NAMES.index(MIN_DIST_SPEC_NAME)
    miss_idx = NEIGHBOR_CHANNEL_NAMES.index(MIN_DIST_MISS_NAME)
    block[:, midx] = MIN_DIST_FILL
    block[:, miss_idx] = 1.0

    crow = coords_df.loc[coords_df["site_id"] == str(site_id)]
    if crow.empty:
        return block
    lat_i = float(crow["lat"].iloc[0])
    lon_i = float(crow["lon"].iloc[0])

    ev = events_df[(events_df["year"] == int(year)) & (events_df["site_id"] != str(site_id))].copy()
    ev = ev.merge(coords_df, on="site_id", how="left").dropna(subset=["lat", "lon"])
    if ev.empty:
        return block
    ev["dist"] = haversine_km(lat_i, lon_i, ev["lat"].to_numpy(), ev["lon"].to_numpy())

    e_doy = ev["obs_doy"].to_numpy()
    e_dist = ev["dist"].to_numpy()
    for k in range(T):
        t = int(doy_start) + k
        for spec in NEIGHBOR_FEATURE_SPECS:
            w = int(spec["window"])
            r = float(spec["radius"])
            sel = (e_doy < t) & (e_doy >= t - w) & (e_dist <= r)  # obs_doy < t (strict)
            ch = NEIGHBOR_CHANNEL_NAMES.index(spec["name"])
            d_sel = e_dist[sel]
            if spec["kind"] == "any":
                block[k, ch] = 1.0 if d_sel.size > 0 else 0.0
            elif spec["kind"] == "count":
                block[k, ch] = float(d_sel.size)
            elif spec["kind"] == "weighted":
                block[k, ch] = float(np.exp(-d_sel / float(decay_km)).sum()) if d_sel.size else 0.0
            elif spec["kind"] == "min_dist":
                if d_sel.size:
                    block[k, ch] = float(d_sel.min())
                    block[k, miss_idx] = 0.0
                else:
                    block[k, ch] = MIN_DIST_FILL
                    block[k, miss_idx] = 1.0
    return block


# ---------------------------------------------------------------------------
# Self-tests
# ---------------------------------------------------------------------------
def _selftest_leakage(decay_km: float = DEFAULT_DECAY_KM) -> None:
    """Synthetic 3-site check: obs_doy == t excluded, obs_doy > t never used."""
    coords = pd.DataFrame(
        {
            "site_id": ["A", "B", "C"],
            "lat": [36.0, 36.1, 36.0],     # B ~11 km N of A
            "lon": [127.0, 127.0, 127.2],  # C ~18 km E of A
        }
    )
    events = pd.DataFrame(
        {
            "site_id": ["B", "C"],
            "year": [2020, 2020],
            "obs_doy": [100, 105],
        }
    )
    index = build_neighbor_index(events, coords)
    doy_start = 95
    T = 20  # covers doy 95..114
    block = compute_neighbor_block("A", 2020, doy_start, T, index, decay_km=decay_km)
    cnt14_30 = block[:, NEIGHBOR_CHANNEL_NAMES.index("neighbor_count_14d_30km")]

    def at(doy):
        return cnt14_30[doy - doy_start]

    # t == 100: B's event is obs_doy == t -> excluded; C is in the future -> excluded.
    assert at(100) == 0.0, f"obs_doy==t must be excluded, got {at(100)}"
    # t == 101: B (doy100 < 101, within 14d, ~11km<30km) counted; C still future.
    assert at(101) == 1.0, f"expected 1 neighbor at t=101, got {at(101)}"
    # t == 105: C's own day -> excluded; only B counts.
    assert at(105) == 1.0, f"obs_doy==t (C) must be excluded, got {at(105)}"
    # t == 106: both B(doy100) and C(doy105) are strictly in the past & within window.
    assert at(106) == 2.0, f"expected 2 neighbors at t=106, got {at(106)}"
    # Before any event nothing is counted (no leakage from future events).
    assert at(99) == 0.0, f"no past events before t=99, got {at(99)}"
    print("[selftest] leakage / window / self-exclusion asserts passed.")


def _crosscheck_reference(index, events_df, coords_df, site_years, doy_start, T, decay_km, n=12):
    """Assert vectorized == independent reference on up to n event-bearing site-years."""
    years_with_events = set(index.events_by_year.keys())
    cand = [sy for sy in site_years if sy[1] in years_with_events]
    # Deterministic, spread-out sample (no RNG).
    if len(cand) > n:
        step = max(1, len(cand) // n)
        cand = cand[::step][:n]
    checked = 0
    for site_id, year in cand:
        fast = compute_neighbor_block(site_id, year, doy_start, T, index, decay_km=decay_km)
        ref = _reference_block(site_id, year, doy_start, T, events_df, coords_df, decay_km)
        assert np.allclose(fast, ref, atol=1e-5, equal_nan=True), (
            f"vectorized != reference for site={site_id} year={year}"
        )
        checked += 1
    print(f"[selftest] vectorized vs reference matched on {checked} site-years.")


def _sanity_main():
    import argparse

    ap = argparse.ArgumentParser(description="Neighbor-occurrence feature sanity check.")
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--limit", type=int, default=1000, help="max site-years to evaluate")
    ap.add_argument("--decay_km", type=float, default=DEFAULT_DECAY_KM)
    args = ap.parse_args()

    # Resolve pest -> applies PATH_OBS / DOY_START / DOY_END / LABEL_COL / YEAR_* into C.
    from rice.src.pest_resolver import resolve_pest
    from rice.configs import config as C

    resolve_pest(args.pest)
    path = C.PATH_OBS
    doy_start = int(C.DOY_START)
    doy_end = int(C.DOY_END)
    T = doy_end - doy_start + 1
    label_col = getattr(C, "LABEL_COL", "label_event")
    year_min = getattr(C, "YEAR_MIN", None)
    year_max = getattr(C, "YEAR_MAX", None)

    print(f"[cfg] pest={args.pest} path={path}")
    print(f"[cfg] DOY {doy_start}-{doy_end} (T={T}) label_col={label_col} "
          f"year_range=({year_min},{year_max}) decay_km={args.decay_km}")

    events_df, coords_df, site_years = load_long_events(
        path, label_col=label_col, year_min=year_min, year_max=year_max
    )
    index = build_neighbor_index(events_df, coords_df)
    print(f"[data] sites={len(index.site_ids)} event_rows={len(events_df)} "
          f"site_years={len(site_years)} years_with_events={sorted(index.events_by_year)}")

    # ---- leakage / correctness asserts (always run) ----
    _selftest_leakage(decay_km=args.decay_km)
    _crosscheck_reference(index, events_df, coords_df, site_years, doy_start, T, args.decay_km)

    # ---- aggregate feature stats over up to --limit site-years ----
    use = site_years[: int(args.limit)]
    n_ch = len(NEIGHBOR_FEATURE_NAMES)
    sums = np.zeros(n_ch, dtype=np.float64)
    maxs = np.full(n_ch, -np.inf, dtype=np.float64)
    n_rows = 0
    pos_count14 = 0  # rows with neighbor_count_14d_30km > 0
    pos_count30 = 0  # rows with neighbor_count_30d_50km > 0
    miss_rows = 0    # rows with min_dist missing
    idx_c14 = NEIGHBOR_FEATURE_NAMES.index("neighbor_count_14d_30km")
    idx_c30 = NEIGHBOR_FEATURE_NAMES.index("neighbor_count_30d_50km")
    miss_idx = NEIGHBOR_CHANNEL_NAMES.index(MIN_DIST_MISS_NAME)

    for site_id, year in use:
        block = compute_neighbor_block(site_id, year, doy_start, T, index, decay_km=args.decay_km)
        feat = block[:, :n_ch]  # exclude miss indicator from mean/max table
        sums += feat.sum(axis=0)
        maxs = np.maximum(maxs, feat.max(axis=0))
        n_rows += feat.shape[0]
        pos_count14 += int((feat[:, idx_c14] > 0).sum())
        pos_count30 += int((feat[:, idx_c30] > 0).sum())
        miss_rows += int(block[:, miss_idx].sum())

    means = sums / max(n_rows, 1)
    print(f"\n[stats] evaluated site_years={len(use)} day_rows={n_rows}")
    print(f"{'feature':32s} {'mean':>12s} {'max':>12s}")
    for j, name in enumerate(NEIGHBOR_FEATURE_NAMES):
        print(f"{name:32s} {means[j]:12.4f} {maxs[j]:12.4f}")
    if n_rows:
        print(f"\n[stats] rows with neighbor_count_14d_30km > 0: "
              f"{pos_count14}/{n_rows} ({100.0*pos_count14/n_rows:.2f}%)")
        print(f"[stats] rows with neighbor_count_30d_50km > 0: "
              f"{pos_count30}/{n_rows} ({100.0*pos_count30/n_rows:.2f}%)")
        print(f"[stats] rows with min_dist missing: "
              f"{miss_rows}/{n_rows} ({100.0*miss_rows/n_rows:.2f}%)")
    print(f"\n[ok] channels appended per sample: {NEIGHBOR_FEATURE_DIM} -> {NEIGHBOR_CHANNEL_NAMES}")


if __name__ == "__main__":
    _sanity_main()
