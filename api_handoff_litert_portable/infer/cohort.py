"""Full-site (cohort) prediction — the training/evaluation data flow, portable.

This is a backend swap of the pipeline that produced the project's Stage-1 /
Stage-2 results, NOT a new design. The lineage of every step below is:

  rice/src/data_pipeline.load_daily                     -> load_daily_cohort (chunked)
  rice/scripts/run_eval.build_samples_for_run           -> build_base_samples_cohort
  rice/src/data_pipeline.aggregate_obs_daily_max        -> aggregate_obs_daily_max
  rice/src/labels.build_interval_labels_from_doy        -> build_interval_labels
  rice/src/labels.filter_labels_by_gap                  -> filter_labels_by_gap
  rice/scripts/phase_r_oracle_iou._calibrated_probs_per_sy -> forward_cohort
  rice/scripts/phase_r_oracle_iou.build_dispatch_alert_map -> run_cohort (alert loop)
  rice/scripts/phase_r_oracle_iou._dispatch_features_for_sy-> dispatch_features_for_sy
  rice/scripts/phase_t_group_tau_hybrid.first_crossing_k   -> first_crossing_k

all of which were already vendored (with torch/sklearn) into
api_handoff_transformer/infer/stage1.py::compute_stage1_table. This module is
that same cohort flow with only the two backends replaced:

  Stage-1  pickled sklearn XGBClassifier in .pt  ->  xgboost.Booster + model.json
  Stage-2  PyTorch lead_v3 checkpoint            ->  LiteRT FP16

Everything else — feature construction, A/D branch, array dtype/shape/memory
layout, temperature, tau/k, threshold + alert_tstar, the 14 dispatch features,
interval labels, fallback/climatology policy, output schema, site ordering and
dedup — is unchanged.

Two things are deliberately taken from the DEPLOYED API rather than the research
scripts, because the deployed behaviour is what the A/B baseline validated:

  * temperature is the fixed per-branch value in calibration.json, not re-fit on
    the val split at runtime (stage1.py:_forward_one, not _calibrated_per_sy);
  * site_history comes from the shipped site_history.json keyed "site|year"
    (stage1.py::compute_alert_single_sy), not recomputed over every year.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from .preprocessing import (
    PreprocessError,
    REQUIRED_DAILY_COLS,
    add_rolling_features,
    merge_pheno_ffill,
)
from .stage1_features import (
    COORD_COLS,
    PHENO_CANDIDATE_COLS,
    Stage1FeatureError,
    append_history,
    apply_temperature,
    base_x_from_season,
    build_nowcast_samples,
    build_tabular,
)
from .stage1_portable import (
    PortableBranch,
    Stage1Error,
    dispatch_features_for_sy,
    first_crossing_k,
)

# --- Label/season config. Vendored verbatim from stage1.py:59-70, itself taken
#     from rice/pests/<pest>/config.py + rice/configs/base.py. DOY_START/END
#     come from each Stage-1 model's metadata, never from here.
COUNT_COL = "obs_value"
LABEL_COL = "label_event"
THRESHOLD = 0.0
SEASON_START_DOY = 1
SEASON_END_DOY = 365
MAX_GAP = 30
LEFT_WINDOW_DAYS = 15
YEAR_MIN = 2002
YEAR_MAX = 2024

# Forward the cohort in blocks of this many site-years so the full set of
# (window, D) nowcast slices is never materialized at once (stage1.py:_FORWARD_CHUNK).
FORWARD_CHUNK = 300

DAILY_CHUNKSIZE = 1_000_000


# ===========================================================================
# Labels  (src/labels.py, src/data_pipeline.aggregate_obs_daily_max)
# ===========================================================================
def aggregate_obs_daily_max(obs: pd.DataFrame) -> pd.DataFrame:
    """Port of stage1.py::_aggregate_obs_daily_max."""
    obs2 = obs.copy()
    obs2["obs_doy"] = pd.to_numeric(obs2["obs_doy"], errors="coerce")
    if COUNT_COL in obs2.columns:
        obs2[COUNT_COL] = pd.to_numeric(obs2[COUNT_COL], errors="coerce")
    label_col = LABEL_COL if LABEL_COL in obs2.columns else None
    if label_col is not None:
        obs2[label_col] = pd.to_numeric(obs2[label_col], errors="coerce").fillna(0.0)
    drop_cols = ["obs_doy"]
    if label_col is None:
        drop_cols.append(COUNT_COL)
    obs2 = obs2.dropna(subset=drop_cols).copy()
    obs2["obs_doy"] = obs2["obs_doy"].astype(int)
    agg = {}
    if COUNT_COL in obs2.columns:
        agg[COUNT_COL] = "max"
    if label_col is not None:
        agg[label_col] = "max"
    return obs2.groupby(["site_id", "year", "obs_doy"], as_index=False).agg(agg)


def build_interval_labels(obs2: pd.DataFrame) -> pd.DataFrame:
    """Port of stage1.py::_build_interval_labels."""
    rows = []
    for (site, year), sub in obs2.groupby(["site_id", "year"], sort=False):
        sub = sub.sort_values("obs_doy")
        t = sub["obs_doy"].to_numpy().astype(int)
        if LABEL_COL in sub.columns:
            y = pd.to_numeric(sub[LABEL_COL], errors="coerce").fillna(0.0).to_numpy()
            above = y > 0
        else:
            y = sub[COUNT_COL].to_numpy()
            above = y > THRESHOLD
        if above.any():
            idx_R = int(np.argmax(above))
            R_doy = int(t[idx_R])
            left_L = max(int(SEASON_START_DOY), int(R_doy) - LEFT_WINDOW_DAYS)
            if idx_R == 0:
                L_doy = left_L
            else:
                idx_Ls = np.where(~above[:idx_R])[0]
                L_doy = left_L if len(idx_Ls) == 0 else int(t[idx_Ls[-1]])
            rows.append({"site_id": site, "year": int(year), "censor_type": "interval",
                         "L_doy": int(L_doy), "R_doy": int(R_doy)})
        else:
            rows.append({"site_id": site, "year": int(year), "censor_type": "right",
                         "L_doy": int(SEASON_START_DOY), "R_doy": int(SEASON_END_DOY)})
    return pd.DataFrame(rows)


def filter_labels_by_gap(labels: pd.DataFrame, doy_start: int,
                         doy_end: int) -> pd.DataFrame:
    """Port of stage1.py::_filter_labels_by_gap."""
    if labels.empty:
        return labels
    lf = labels.copy()
    L_cl = lf["L_doy"].clip(doy_start, doy_end)
    R_cl = lf["R_doy"].clip(doy_start, doy_end)
    gap = R_cl - L_cl
    keep = (lf["censor_type"] != "interval") | ((gap >= 1) & (gap <= MAX_GAP))
    return lf.loc[keep].copy()


def load_obs_for_stage1(obs_csv: Path) -> pd.DataFrame:
    """Port of stage1.py::_load_obs_for_stage1."""
    obs_csv = Path(obs_csv)
    if not obs_csv.is_file():
        raise PreprocessError(f"LONG obs CSV not found for Stage-1: {obs_csv}")
    obs = pd.read_csv(obs_csv, encoding="utf-8-sig", low_memory=False)
    obs = obs.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
    obs["site_id"] = obs["site_id"].astype(str)
    obs["year"] = pd.to_numeric(obs["year"], errors="coerce").astype("Int64")
    obs = obs.dropna(subset=["year"]).copy()
    obs["year"] = obs["year"].astype(int)
    obs = obs[(obs["year"] >= YEAR_MIN) & (obs["year"] <= YEAR_MAX)].copy()
    return obs


# ===========================================================================
# Daily weather — ONE chunked pass over the master for the whole cohort
# ===========================================================================
def load_daily_cohort(master_daily_csv: Path, sites: set[str],
                      years: set[int]) -> dict[str, pd.DataFrame]:
    """Stream the daily master ONCE and return {site_id: frame} for `sites`.

    This is stage1.py::_prewarm_daily_cache without the on-disk cache: the master
    is read in chunks, each chunk filtered to the requested sites and years, and
    the survivors concatenated per site. A cohort has hundreds to thousands of
    sites, so the per-site scan the single-request path uses would re-read the
    1.7 GB master once per site.

    Filtering by year here (not just by site) is what keeps this cheap: a full
    pest cohort for one season is ~300k rows, not 28M.
    """
    master_daily_csv = Path(master_daily_csv)
    if not master_daily_csv.is_file():
        raise PreprocessError(f"daily master not found: {master_daily_csv}")
    header = pd.read_csv(master_daily_csv, nrows=0, encoding="utf-8-sig")
    cols = [c.strip() for c in header.columns.tolist()]
    missing = [c for c in REQUIRED_DAILY_COLS if c not in cols and c != "지점ID"]
    if missing:
        raise PreprocessError(
            f"daily master missing required columns: {missing}. file={master_daily_csv}")
    site_col = cols[0]
    parts: list[pd.DataFrame] = []
    reader = pd.read_csv(master_daily_csv, encoding="utf-8-sig",
                         dtype={site_col: str}, chunksize=DAILY_CHUNKSIZE)
    for chunk in reader:
        chunk.columns = cols
        sub = chunk[chunk[site_col].isin(sites)]
        if sub.empty:
            continue
        yr = pd.to_datetime(sub["일시"], errors="coerce").dt.year
        sub = sub[yr.isin(years)]
        if not sub.empty:
            parts.append(sub)
    if not parts:
        return {}
    df = pd.concat(parts, ignore_index=True)
    return {str(s): d for s, d in df.groupby(site_col, sort=False)}


# ===========================================================================
# Operational (as-of) Stage-2 input
# ===========================================================================
def pad_daily_to_season(daily: pd.DataFrame, year: int, as_of_doy: int,
                        doy_end: int) -> pd.DataFrame:
    """Keep DOY <= as_of_doy and pad the rest of the season with empty rows.

    Stage-2 was trained with `_mask_to_recent_window`, which zeroes every row
    outside [tstar-window+1, tstar] and sets that block's miss indicators to 1.
    tstar is alert_tstar_doy + selected_offset, so no row after that DOY can
    reach the output — the padded rows exist only to satisfy the fixed season
    length T, and their values are discarded before the model sees them.
    Verified empirically: garbling every post-cutoff row leaves mu_doy bit-exact
    (reports/stage2_causality.json).

    The padding is NaN rather than a copy of the last observation so it cannot
    be mistaken for data, and it is added AFTER the cutoff so it never enters an
    interpolation that a real row depends on — except across a missing run that
    straddles the cutoff, which is exactly the case `as_of_reproduces_full_year`
    reports on.
    """
    d = daily.copy()
    dates = pd.to_datetime(d["일시"], errors="coerce")
    d = d[dates.dt.dayofyear <= int(as_of_doy)].copy()
    if d.empty:
        return d
    have = set(pd.to_datetime(d["일시"], errors="coerce").dt.dayofyear.tolist())
    need = [t for t in range(int(as_of_doy) + 1, int(doy_end) + 1) if t not in have]
    if not need:
        return d
    site_col = "지점ID" if "지점ID" in d.columns else "site_id"
    base = {c: np.nan for c in d.columns}
    rows = []
    for t in need:
        r = dict(base)
        r[site_col] = d[site_col].iloc[0]
        r["일시"] = (pd.Timestamp(year=int(year), month=1, day=1)
                     + pd.Timedelta(days=int(t) - 1)).strftime("%Y-%m-%d")
        rows.append(r)
    return pd.concat([d, pd.DataFrame(rows)], ignore_index=True)


def as_of_reproduces_full_year(daily_full: pd.DataFrame, as_of_doy: int,
                               tstar_doy: int, window: int) -> tuple[bool, str]:
    """Can an as-of run reproduce the full-year Stage-2 input exactly?

    Two conditions:
      1. as_of_doy >= tstar_doy — otherwise the model's window is not yet
         observed at all.
      2. no missing run inside the window extends past as_of_doy. The training
         imputation is `interpolate(limit_direction="both")`, so a gap that
         straddles the cutoff is filled from the first valid value AFTER it; an
         operational run does not have that value and will impute differently.
    """
    if int(as_of_doy) < int(tstar_doy):
        return False, (f"as_of_doy={as_of_doy} is before tstar_doy={tstar_doy}; "
                       f"the nowcast window is not fully observed yet")
    d = daily_full.copy()
    doy = pd.to_datetime(d["일시"], errors="coerce").dt.dayofyear
    lo = int(tstar_doy) - int(window) + 1
    from .preprocessing import WEATHER_COLS

    for c in WEATHER_COLS:
        if c not in d.columns:
            continue
        vals = pd.to_numeric(d[c], errors="coerce")
        miss_in_win = doy.between(lo, int(tstar_doy)) & vals.isna()
        if not miss_in_win.any():
            continue
        # the gap is only safe if a valid value exists at or before as_of_doy
        after = vals[(doy > int(tstar_doy)) & (doy <= int(as_of_doy))].notna()
        if not after.any():
            return False, (f"column {c!r} has a missing run inside the window "
                           f"[{lo},{tstar_doy}] with no observed value before "
                           f"as_of_doy={as_of_doy}; the full-year run would "
                           f"impute it from a later observation")
    return True, ""


# ===========================================================================
# Base samples for the cohort  (run_eval.build_samples_for_run)
# ===========================================================================
def build_base_samples_cohort(feature_cols: list[str], daily_by_site: dict,
                              obs: pd.DataFrame, labels: pd.DataFrame,
                              doy_start: int, doy_end: int,
                              site_meta, pheno) -> tuple[list[dict], dict]:
    """One base sample per labeled (site, year) with a complete daily season.

    Port of stage1.py::_build_base_samples. Sites whose season is incomplete are
    dropped with the same rule as training (`len(season) != T`), and the reason
    is returned so the caller can report it instead of silently losing the site.
    """
    T = int(doy_end) - int(doy_start) + 1
    lab = labels.copy()
    lab["site_id"] = lab["site_id"].astype(str)
    lab["year"] = lab["year"].astype(int)
    lab_map = {(r.site_id, int(r.year)): r for r in lab.itertuples(index=False)}

    samples: list[dict] = []
    dropped: dict[str, str] = {}
    for (site, year) in sorted(lab_map.keys()):
        daily = daily_by_site.get(str(site))
        if daily is None:
            dropped[f"{site}|{year}"] ="no daily rows for the requested year"
            continue
        try:
            season = _season_for(daily, site, year, doy_start, doy_end,
                                 feature_cols, site_meta, pheno, T)
        except (PreprocessError, Stage1FeatureError) as e:
            dropped[f"{site}|{year}"] =f"{type(e).__name__}: {e}"
            continue
        try:
            X = base_x_from_season(season, feature_cols)
        except Stage1FeatureError as e:
            dropped[f"{site}|{year}"] =f"{type(e).__name__}: {e}"
            continue
        r = lab_map[(site, int(year))]
        L = min(max(int(r.L_doy) - doy_start + 1, 1), T)
        R = min(max(int(r.R_doy) - doy_start + 1, 1), T)
        samples.append({
            "site_id": str(site), "year": int(year), "X": X,
            "L": L, "R": R, "censor_type": str(r.censor_type),
        })
    return samples, dropped


def _season_for(daily: pd.DataFrame, site: str, year: int, doy_start: int,
                doy_end: int, feature_cols: list[str], site_meta, pheno,
                T: int) -> pd.DataFrame:
    """Season slice for one site-year out of an already-loaded frame.

    Same body as preprocessing.build_season, but takes the cohort frame that is
    already filtered to this site instead of re-filtering a master.
    """
    from .preprocessing import daily_year_frame

    d = daily_year_frame(daily, site, year)
    d = add_rolling_features(d)
    season = d[(d["doy"] >= doy_start) & (d["doy"] <= doy_end)].copy()
    season = season.sort_values("doy").reset_index(drop=True)
    if len(season) != T:
        raise PreprocessError(
            f"incomplete daily season for site={site} year={year}: "
            f"{len(season)} rows, expected T={T} (DOY {doy_start}..{doy_end})")
    if any(c in COORD_COLS for c in feature_cols):
        lat, lon = site_meta.latlon(site)
        season["좌표-위도"] = lat
        season["좌표-경도"] = lon
    pheno_cols = [c for c in PHENO_CANDIDATE_COLS if c in feature_cols]
    if pheno_cols:
        season = merge_pheno_ffill(season, pheno.rows(site, year), pheno_cols)
    return season


# ===========================================================================
# Cohort forward  (phase_r_oracle_iou._calibrated_probs_per_sy)
# ===========================================================================
def forward_cohort(branch: PortableBranch, base_samples: list[dict],
                   history: dict, season_length: int | None = None) -> dict:
    """Forward one Stage-1 branch over the whole cohort, in blocks.

    Port of stage1.py::_calibrated_per_sy with the fixed deployed temperature.
    One Booster.predict per block, not one per site — that is the whole point of
    the cohort path. Returns {(site, year): {"ts", "ps", "raw"}}.

    `season_length` pins the denominator of the tstar-position feature
    (build_tabular: tstar / season_length). Operational runs hold only a PREFIX
    of the season's rows, but the season they are a prefix OF is still the
    model's full DOY window — so the denominator must stay the full T. Letting
    it shrink with the available rows rewrites that feature for every tstar and
    moves the alert (measured: sheath_blight 2004 as-of 07-15 fired at DOY 107
    instead of 135). With it pinned, a truncated run reproduces the full-season
    series exactly, for every tstar it can evaluate.
    """
    per_sy: dict = {}
    for i0 in range(0, len(base_samples), FORWARD_CHUNK):
        block = base_samples[i0:i0 + FORWARD_CHUNK]
        if branch.site_history_added:
            block = [dict(s, X=append_history(s["X"], s["site_id"], s["year"],
                                              history, branch.doy_start))
                     for s in block]
        nc = build_nowcast_samples(block, branch.window, branch.stride,
                                   branch.only_pre, branch.proxy)
        if not nc:
            continue
        if season_length is not None:
            for s in nc:
                s["season_length"] = int(season_length)
        X = build_tabular(nc, branch.add_tpos)
        p_raw = branch.predict_raw(X)
        p_cal = apply_temperature(p_raw, branch.temperature)
        for s, praw, pcal in zip(nc, p_raw, p_cal):
            sy = (str(s["site_id"]), int(s["year"]))
            d = per_sy.setdefault(sy, {"ts": [], "ps": [], "raw": []})
            d["ts"].append(int(s["tstar"]))
            d["ps"].append(float(pcal))
            d["raw"].append(float(praw))
    out = {}
    for sy, d in per_sy.items():
        ts = np.asarray(d["ts"], dtype=int)
        order = np.argsort(ts)
        out[sy] = {
            "ts": ts[order],
            "ps": np.asarray(d["ps"], dtype=float)[order],
            "raw": np.asarray(d["raw"], dtype=float)[order],
        }
    return out


# ===========================================================================
# Cohort Stage-1  (phase_r_oracle_iou.build_dispatch_alert_map)
# ===========================================================================
def stage1_cohort(paths, pest: str, year: int, sites: list[str],
                  daily_by_site: dict, obs_frame: pd.DataFrame,
                  site_meta, pheno, gate: dict, site_history: dict,
                  log: list[str], as_of_doy: int | None = None) -> tuple[dict, dict]:
    """Stage-1 for the whole cohort in one pass.

    Returns ({"site|year": {"alert_tstar_doy", "dispatch_features"}},
             {"site|year": reason}).
    A site absent from the alert map simply fired no alert — the caller applies
    the unchanged fallback policy.

    `as_of_doy` (operational mode) truncates each season to DOY <= as_of_doy.
    Because the nowcast series is generated as tstar = window..T ascending and
    the alert is the FIRST crossing, a truncated season yields a strict PREFIX
    of the full-season probability series: any alert that would fire on or
    before as_of_doy fires identically, and later ones simply have not happened
    yet. That is the operational semantic, not an approximation of it.
    """
    branches = {b: PortableBranch(pest, b, paths.stage1_dir) for b in ("A", "D")}
    A, D = branches["A"], branches["D"]
    if A.feature_cols != D.feature_cols:
        raise Stage1Error(f"Stage-1 A/D feature_cols differ for {pest}")
    doy_start, doy_end = A.doy_start, A.doy_end
    if as_of_doy is not None:
        if int(as_of_doy) < doy_start + A.window:
            log.append(f"stage1_cohort: as_of_doy={as_of_doy} is before the first "
                       f"evaluable tstar ({doy_start + A.window}); no site can "
                       f"fire yet")
            return {}, {}
        doy_end = min(doy_end, int(as_of_doy))
        log.append(f"stage1_cohort: operational mode, season truncated to "
                   f"DOY {doy_start}..{doy_end}")

    # labels for the requested year (grouped per site-year, so filtering the
    # LONG frame to the year yields the identical labels training would build)
    obs_year = obs_frame[obs_frame["year"] == int(year)].copy()
    obs_year = obs_year[obs_year["site_id"].astype(str).isin(set(sites))]

    if as_of_doy is not None:
        # OPERATIONAL: the season's own observations are the thing being
        # predicted, so they must not be read. The deployed single-request path
        # makes the same call (run_predict.run_stage1 hardcodes L=1, R=1,
        # censor_type="right"): no interval label, therefore no pre-event
        # truncation of the nowcast series and no label-based cohort filtering.
        # Using the real label here would be look-ahead — and it is observable:
        # filter_labels_by_gap clips R_doy to the truncated doy_end, so a site
        # whose event falls after as_of_date silently drops out of the cohort.
        sites_with_daily = [s for s in sites if str(s) in daily_by_site]
        labels = pd.DataFrame([
            {"site_id": str(s), "year": int(year), "censor_type": "right",
             "L_doy": int(SEASON_START_DOY), "R_doy": int(SEASON_END_DOY)}
            for s in sites_with_daily])
        log.append(f"stage1_cohort: operational labelling (right-censored, no "
                   f"season observations used) for {len(labels)} sites")
    else:
        if obs_year.empty:
            log.append(f"stage1_cohort: no LONG observations for year={year}; "
                       f"every site falls back")
            return {}, {}
        labels = filter_labels_by_gap(
            build_interval_labels(aggregate_obs_daily_max(obs_year)),
            doy_start, doy_end)
        log.append(f"stage1_cohort: labeled_site_years={len(labels)} "
                   f"(doy {doy_start}..{doy_end})")
    if labels.empty:
        return {}, {}

    base_samples, dropped = build_base_samples_cohort(
        A.feature_cols, daily_by_site, obs_year, labels, doy_start, doy_end,
        site_meta, pheno)
    log.append(f"stage1_cohort: base_samples={len(base_samples)} dropped={len(dropped)}")
    if not base_samples:
        return {}, dropped

    # site_history: shipped asset, exactly as the deployed single-request path.
    # An absent site-year is not fatal for a cohort — that one site falls back
    # while the rest of the batch proceeds (the deployed batch behaves the same
    # way, since a per-site Stage-1 error becomes that row's fallback).
    from .stage1_portable import resolve_with_history

    history: dict = {}
    with_h_map: dict = {}
    for s in base_samples:
        key = (str(s["site_id"]), int(s["year"]))
        try:
            with_h, hrow = resolve_with_history(site_history, s["site_id"], s["year"])
        except Stage1Error as e:
            dropped.setdefault(f"{s['site_id']}|{s['year']}", str(e))
            continue
        with_h_map[key] = with_h
        history[key] = hrow
    usable = [s for s in base_samples
              if (str(s["site_id"]), int(s["year"])) in history]
    if len(usable) != len(base_samples):
        log.append(f"stage1_cohort: {len(base_samples) - len(usable)} site-years "
                   f"lack a shipped site_history entry -> fallback")
    if not usable:
        return {}, dropped

    # Full-season T from the model metadata — NOT len(season), which is shorter
    # in an operational (as_of_doy) run.
    full_T = int(A.doy_end) - int(A.doy_start) + 1
    t0 = time.perf_counter()
    per_sy_A = forward_cohort(A, usable, history, season_length=full_T)
    per_sy_D = forward_cohort(D, usable, history, season_length=full_T)
    log.append(f"stage1_cohort: forward A+D over {len(usable)} site-years in "
               f"{time.perf_counter() - t0:.1f}s")

    tau_single = gate.get("tau")
    tau_no = gate.get("tau_no") if gate.get("tau_no") is not None else tau_single
    tau_with = gate.get("tau_with") if gate.get("tau_with") is not None else tau_single
    k = int(gate["k"])
    method = gate["method"]

    alerts: dict = {}
    for sy in sorted(set(per_sy_A) | set(per_sy_D)):
        a, d = per_sy_A.get(sy), per_sy_D.get(sy)
        if a is None or d is None:
            continue
        with_h = with_h_map.get(sy, False)
        if method == "A_baseline":
            at = first_crossing_k(a["ts"], a["ps"], tau_single, k)
        elif method == "D_history":
            at = first_crossing_k(d["ts"], d["ps"], tau_single, k)
        else:
            at = (first_crossing_k(d["ts"], d["ps"], tau_with, k) if with_h
                  else first_crossing_k(a["ts"], a["ps"], tau_no, k))
        if at is None:
            continue
        feats = dispatch_features_for_sy(at, a, d, with_h, doy_start, method,
                                         tau_no, tau_with, tau_single)
        alerts[f"{sy[0]}|{sy[1]}"] = {"alert_tstar_doy": int(feats["alert_tstar"]),
                                      "dispatch_features": feats}
    log.append(f"stage1_cohort: alerts_fired={len(alerts)} / {len(usable)} evaluated")
    return alerts, dropped
