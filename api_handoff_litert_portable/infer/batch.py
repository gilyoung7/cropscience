"""Batch mode for the lightweight portable API.

Ports `api_handoff_transformer/infer/batch.py` onto the portable Stage-1
(XGBoost JSON) + Stage-2 (LiteRT) engine. The single-request pipeline is reused
per row exactly as the deployed batch reuses `run_stage2_transformer` — no
duplicated inference logic, no model conversion, no policy change.

Two input shapes are supported:

  1. REP-CSV BATCH — the deployed contract (analysis/BATCH_IMPLEMENTATION_PLAN.md §1):
         {"mode":"batch", "pest":"BPH", "year":2004,
          "include_diagnostics":false,
          "representative_sites_path":"...",   # optional, alias representative_sites_csv
          "max_sites":50}                      # optional, ops/testing only
     One pest + one year; sites come from the representative-site CSV, filtered
     to those that also have LONG and daily data for the year, then `sorted()`.

  2. GENERIC CSV BATCH — additive extension:
         {"mode":"batch", "input_csv":"rows.csv", "stage2_variant":"fp16"}
     One row per prediction with columns pest,site_id,year[,alert_tstar_doy].
     Rows may mix pests/years/sites, and **input row order is preserved**.

Error policy follows the deployed contract exactly:
  * whole-batch failure (bad request/assets) -> all three files still written,
    exit code 2;
  * per-row failure -> an error row, and the batch continues.
"""

from __future__ import annotations

import csv
import json
import time
import traceback
from pathlib import Path
from typing import Any

import pandas as pd

from .fallback import climatology_variant, compute_climatology, load_policy, per_pest_policy
from .inputs import existing_outputs, resolve_daily, resolve_obs, write_atomic
from .paths import MODEL_VERSION, VALID_PESTS, Paths
from .providers import FrameWeatherProvider, LongObsProvider
from .schemas import FLAT_COLS, flatten_response

DEFAULT_REP_FILENAME = "representative_site_ids_2002_2024.csv"

# batch.py:48-57 — rice_stem_borer_1 and _2 share the single 이화명나방 rep list.
PEST_TO_KOREAN: dict[str, str] = {
    "brown_spot": "깨씨무늬병",
    "BPH": "벼멸구",
    "rice_stem_borer_1": "이화명나방",
    "rice_stem_borer_2": "이화명나방",
    "blast": "잎도열병",
    "sheath_blight": "잎집무늬마름병",
    "WBPH": "흰등멸구",
    "bacterial_blight": "흰잎마름병",
}

# batch.py:168-173 — appended AFTER the 16 single-mode columns, never before.
BATCH_EXTRA_COLS = ["status", "error_reason"]
DIAG_COLS = [
    "alert_source", "stage1_method", "stage1_alert_tstar_doy",
    "tstar_season_index", "mu_rel_season_index", "base_channels_status",
    "input_X_shape", "d_in", "ckpt_pest_field",
]
GENERIC_EXTRA_COLS = ["row_index"]


class BatchRequestError(ValueError):
    """Whole-batch failure: the request itself is unusable."""


def normalize_pest(pest: Any) -> str | None:
    """batch.py:60-66 — batch is case-INSENSITIVE (single mode is not)."""
    if not isinstance(pest, str):
        return None
    return {p.lower(): p for p in VALID_PESTS}.get(pest.strip().lower())


def resolve_rep_csv(explicit: str | None, input_dir: Path, pkg_root: Path) -> Path:
    """batch.py:69-101 — explicit (abs as-is; rel vs cwd/input_dir/pkg_root),
    else auto-discover DEFAULT_REP_FILENAME in input_dir/cwd/pkg_root."""
    if explicit:
        p = Path(explicit)
        candidates = [p] if p.is_absolute() else [Path.cwd() / p, input_dir / p, pkg_root / p]
    else:
        candidates = [input_dir / DEFAULT_REP_FILENAME,
                      Path.cwd() / DEFAULT_REP_FILENAME,
                      pkg_root / DEFAULT_REP_FILENAME]
    for c in candidates:
        if c.is_file():
            return c
    raise BatchRequestError(
        "representative-site CSV not found. Supply it via request "
        "'representative_sites_path' or --representative-sites, or place "
        f"'{DEFAULT_REP_FILENAME}' in the input dir (--input-dir) or the current "
        "working directory. Tried: " + ", ".join(str(c) for c in candidates)
    )


def representative_sites(rep_csv: Path, pest: str) -> list[str]:
    """batch.py:104-122 — match by Korean name; dedup; sorted."""
    korean = PEST_TO_KOREAN.get(pest)
    if korean is None:
        raise BatchRequestError(f"no Korean rep-name mapping for pest={pest!r}")
    df = pd.read_csv(rep_csv, encoding="utf-8-sig", dtype=str)
    cols = {c.strip(): c for c in df.columns}
    if "pest" not in cols or "site_id" not in cols:
        raise BatchRequestError(
            f"representative-site CSV must have 'pest' and 'site_id' columns; "
            f"got {list(df.columns)}"
        )
    sub = df[df[cols["pest"]].astype(str).str.strip() == korean]
    sites = sub[cols["site_id"]].astype(str).str.strip()
    return sorted(dict.fromkeys(sites.tolist()))


def resolve_input_csv(explicit: str, input_dir: Path, pkg_root: Path) -> Path:
    """Resolve the generic-batch CSV. Same rule as resolve_rep_csv.

    Absolute paths are used as given; relative paths are tried against the
    caller's cwd, then --input-dir, then the package root. Previously this file
    alone was resolved with a bare `Path(...)`, so a relative name only worked
    when the process happened to run from the right directory.
    """
    p = Path(explicit).expanduser()
    candidates = ([p] if p.is_absolute()
                  else [Path.cwd() / p, input_dir / p, pkg_root / p])
    for c in candidates:
        if c.is_file():
            return c
    raise BatchRequestError(
        f"batch input_csv not found: {explicit}. Tried: "
        + ", ".join(str(c) for c in candidates)
    )


# Request-level settings a generic-batch row inherits when it does not carry
# its own. Without this the per-row request was rebuilt from scratch, so a
# top-level daily_weather_path / long_observation_path never reached the
# pipeline and every row failed with "daily weather CSV not found".
INHERITED_ROW_FIELDS = (
    "daily_weather_path",
    "long_observation_path",
    "representative_sites_path",
    "stage2_variant",
    "include_diagnostics",
)


def _inherit_request_fields(req: dict, raw_request: dict, raw_row: dict) -> None:
    """Fill `req` from the row first, then the top-level request.

    A value on the row always wins, so a mixed-pest CSV can point individual
    rows at their own LONG file while the rest inherit one shared master.
    Blank CSV cells ("", "nan") count as absent — pandas gives every column a
    value for every row, so an empty cell must not shadow the request default.
    """
    for key in INHERITED_ROW_FIELDS:
        row_val = raw_row.get(key)
        # An empty CSV cell reaches us as float NaN even with dtype=str, so a
        # bare `is not None` test would hand NaN to Path() further down.
        if row_val is not None and not isinstance(row_val, (str, bool, int)):
            row_val = None if pd.isna(row_val) else str(row_val)
        if isinstance(row_val, str):
            row_val = row_val.strip()
            if row_val.lower() in ("", "nan", "none"):
                row_val = None
        if row_val is not None:
            req[key] = row_val
        elif raw_request.get(key) is not None:
            req[key] = raw_request[key]
    # CSV cells arrive as strings, and a non-empty string is always truthy --
    # "false" would switch diagnostics ON. Coerce this one back to a real bool.
    v = req.get("include_diagnostics")
    if isinstance(v, str):
        req["include_diagnostics"] = v.strip().lower() in ("1", "true", "yes", "y")


def load_generic_rows(input_csv: Path) -> list[dict]:
    """Read a generic batch CSV. Row order is preserved.

    Required columns: pest, site_id, year. Optional: alert_tstar_doy.
    Missing required columns is a whole-batch failure (the file is unusable);
    a bad VALUE in one row is a per-row error.
    """
    if not Path(input_csv).is_file():
        raise BatchRequestError(f"batch input_csv not found: {input_csv}")
    df = pd.read_csv(input_csv, encoding="utf-8-sig", dtype=str)
    df.columns = [c.strip() for c in df.columns]
    required = ["pest", "site_id", "year"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise BatchRequestError(
            f"batch input_csv missing required column(s): {missing}. "
            f"Required: {required} (optional: alert_tstar_doy). Got: {list(df.columns)}"
        )
    return df.to_dict("records")


def classify(learned: dict | None, learned_err: str | None) -> tuple[str, str]:
    """batch.py:235-243.

    The deployed version detects the no-alert case with a substring match on the
    error prose ("fired no alert"). That is kept for compatibility, but the
    portable pipeline also emits an explicit marker, checked first.
    """
    if learned is not None:
        return "success", ""
    if learned_err and ("fired no alert" in learned_err or "no_alert" in learned_err):
        return "fallback", learned_err
    # Operational blocks are deliberate policy outcomes, not failures: the site
    # is answered from climatology and the batch carries on. They are marked
    # with an explicit status so they are never confused with a real error.
    if learned_err and (
            "stage2_window_crosses_unresolved_missing_run" in learned_err
            or "stage2_pending_window_not_yet_observed" in learned_err):
        return "fallback", learned_err
    return "error", (learned_err or "unknown Stage-2 failure")


def error_row(pest: str, site: str, year: Any, climatology: dict, reason: str) -> dict:
    """batch.py:202-218 — a row for a site that raised before a response existed.
    Climatology still fills final_* so the row stays usable downstream."""
    return {
        "pest": pest, "site_id": site, "year": year, "model_version": MODEL_VERSION,
        "final_source": "climatology_error",
        "final_mu_doy": (climatology or {}).get("mu_doy"),
        "final_pi95_lower": ((climatology or {}).get("pi_95") or {}).get("lower_doy"),
        "final_pi95_upper": ((climatology or {}).get("pi_95") or {}).get("upper_doy"),
        "learned_mu_doy": None, "learned_selected_offset": None,
        "learned_output_status": None,
        "climatology_mu_doy": (climatology or {}).get("mu_doy"),
        "climatology_variant": (climatology or {}).get("variant"),
        "recommended_source": None, "fallback_triggered": True,
        "alert_tstar_doy": None, "status": "error", "error_reason": reason,
    }


def diag_row(diag: dict) -> dict:
    """batch.py:221-232 — input_X_shape stringified, everything else passed through."""
    return {
        "alert_source": diag.get("alert_source"),
        "stage1_method": diag.get("stage1_method"),
        "stage1_alert_tstar_doy": diag.get("stage1_alert_tstar_doy"),
        "tstar_season_index": diag.get("tstar_season_index"),
        "mu_rel_season_index": diag.get("mu_rel_season_index"),
        "base_channels_status": diag.get("base_channels_status"),
        "input_X_shape": (str(diag.get("input_X_shape"))
                          if diag.get("input_X_shape") is not None else None),
        "d_in": diag.get("d_in"),
        "ckpt_pest_field": diag.get("ckpt_pest_field"),
    }


def write_predictions_csv(path: Path, rows: list[dict], include_diag: bool,
                          generic: bool) -> None:
    """batch.py:467-475 — 16 single cols + status/error_reason [+ diag] and, for
    generic mode only, a leading row_index. newline='' -> CRLF."""
    fieldnames = (list(GENERIC_EXTRA_COLS) if generic else []) + \
        list(FLAT_COLS) + list(BATCH_EXTRA_COLS)
    if include_diag:
        fieldnames += list(DIAG_COLS)
    import io

    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=fieldnames, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: r.get(k) for k in fieldnames})
    write_atomic(path, buf.getvalue(), newline="")


def write_run_log(path: Path, raw_request: dict, lines: list[str]) -> None:
    """batch.py:478-495 — batch-specific header + asset-load note + events."""
    from datetime import datetime, timezone

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body = [
        "run_predict.py log (Stage-1 XGBoost JSON + Stage-2 LiteRT — BATCH mode)",
        f"timestamp_utc: {ts}",
        f"model_version: {MODEL_VERSION}",
        f"request: {json.dumps(raw_request, ensure_ascii=False)}",
        "",
        "NOTE: per-pest assets (Stage-2 LiteRT model + normalization, Stage-1 A/D",
        "Booster models, gate.json, site_history.json, climatology) are loaded once",
        "per pest and reused across rows — no per-row reload.",
        "",
        "events:",
        *[f"  - {ln}" for ln in lines],
    ]
    write_atomic(path, "\n".join(body) + "\n")


def fail_batch(output_dir: Path, raw_request: dict, reason: str, t0: float,
               generic: bool = False) -> int:
    """batch.py:498-527 — whole-batch failure STILL writes all three files, exit 2."""
    import sys

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"[run_predict:batch] ERROR (whole batch): {reason}", file=sys.stderr)
    # A failed run must never replace results a previous successful run left
    # here. If any output file already exists, keep it and report instead.
    already = existing_outputs(output_dir)
    if already:
        print(
            f"[run_predict:batch] existing results preserved ({', '.join(already)}) "
            f"— the failure was NOT written to {output_dir}",
            file=sys.stderr)
        return 2
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "batch",
        "pest": raw_request.get("pest"),      # raw, un-normalized (deployed behaviour)
        "year": raw_request.get("year"),
        "error": reason,
        "requested_count": 0, "success_count": 0, "fallback_count": 0,
        "error_count": 0, "elapsed_seconds": elapsed, "results": [],
    }
    write_atomic(output_dir / "response.json",
                 json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_predictions_csv(output_dir / "predictions.csv", [], include_diag=False,
                          generic=generic)
    write_run_log(output_dir / "run_log.txt", raw_request,
                  [f"WHOLE-BATCH FAILURE: {reason}"])
    return 2


class _PestContext:
    """Per-pest assets, loaded once and reused across rows (deployed batch does
    the same with the .pt assets).

    In cohort mode the context also carries, for the whole batch:
      weather / obs  — providers over the ONE chunked pass of the daily master
                       and the ONE read of the LONG file;
      alert_map      — {site_id: {alert_tstar_doy, dispatch_features}} produced
                       by a single vectorized Stage-1 pass over every site;
      stage1_notes   — {site_id: reason} for sites Stage-1 could not evaluate.
    When these are set, run_pipeline reuses them instead of re-reading the master
    and re-running Stage-1 per site.
    """

    def __init__(self, paths: Paths, pest: str, policy: dict, variant: str):
        from .stage2_litert import Stage2Model

        self.pest = pest
        self.policy_pp = per_pest_policy(policy, pest)
        self.climatology = compute_climatology(
            paths.climatology_dir, pest, climatology_variant(policy, pest))
        self.model = Stage2Model(pest, paths.stage2_dir, variant=variant)
        # cohort extras (None => per-row behaviour, unchanged)
        self.weather = None
        self.obs = None
        self.alert_map: dict | None = None
        self.stage1_notes: dict = {}


def _prepare_cohort(paths: Paths, pest: str, year: int, sites: list[str],
                    policy: dict, variant: str, include_diag: bool,
                    log: list[str], daily_by_site: dict | None = None,
                    obs_provider=None, years: list[int] | None = None,
                    as_of_doy: int | None = None,
                    request: dict | None = None) -> dict:
    """Build the per-pest context with the cohort pre-pass already done.

    `daily_by_site` is the per-site season map from the site-selection scan; it
    is passed in so the daily master is read exactly ONCE per request. Returns
    {pest: _PestContext} so the caller can seed its ctx cache.
    """
    import time as _t

    from .cohort import load_daily_cohort, load_obs_for_stage1, stage1_cohort
    from .providers import FrameWeatherProvider, LongObsProvider
    from .stage1_portable import load_gate

    t0 = _t.perf_counter()
    daily_csv = (resolve_daily(request, paths.input_dir, paths.pkg_root)
                 or paths.input_dir / "daily_weather.csv")
    obs_csv = _obs_path(paths, pest, request)

    if daily_by_site is None:
        daily_by_site = load_daily_cohort(daily_csv, set(map(str, sites)), {int(year)})
        log.append(f"cohort daily: ONE chunked pass -> {len(daily_by_site)} sites")
    else:
        log.append(f"cohort daily: reusing the selection scan "
                   f"({len(daily_by_site)} sites) — master read once per request")
    n_rows = sum(len(d) for d in daily_by_site.values())
    log.append(f"cohort daily rows={n_rows}")

    # Labels always come from load_obs_for_stage1, never from the provider's
    # frame: it applies the YEAR_MIN..YEAR_MAX window and the int year dtype the
    # label builder was written against (stage1.py::_load_obs_for_stage1).
    obs_frame = load_obs_for_stage1(obs_csv)
    log.append(f"cohort LONG: {len(obs_frame)} rows (label frame)")

    ctx = _PestContext(paths, pest, policy, variant)

    # Operational mode: hand Stage-2 the observed prefix only, padded to the
    # fixed season length. The padded rows are masked out before the model sees
    # them, so this is the same tensor a full-year run would build — see
    # cohort.pad_daily_to_season.
    if as_of_doy is not None:
        from .cohort import pad_daily_to_season

        md_doy_end = None
        try:
            md_doy_end = int(json.loads(
                (paths.stage2_dir / pest / "metadata.json")
                .read_text(encoding="utf-8"))["doy_end"])
        except Exception:
            pass
        if md_doy_end is not None:
            daily_by_site = {
                s: pad_daily_to_season(d, int(year), int(as_of_doy), md_doy_end)
                for s, d in daily_by_site.items()}
            daily_by_site = {s: d for s, d in daily_by_site.items() if len(d)}
            log.append(f"operational: daily truncated to DOY<={as_of_doy} and "
                       f"padded to season end {md_doy_end} for {len(daily_by_site)} sites")

    cohort_daily = (pd.concat(daily_by_site.values(), ignore_index=True)
                    if daily_by_site else pd.DataFrame())
    ctx.weather = FrameWeatherProvider(cohort_daily) if len(cohort_daily) else None
    ctx.obs = LongObsProvider(obs_frame)

    gate = load_gate(pest, paths.stage1_dir)
    ctx.gate_method = gate["method"]
    ctx.as_of_doy = as_of_doy
    site_history = json.loads(
        (paths.stage1_pest(pest) / "site_history.json").read_text(encoding="utf-8"))

    alerts: dict = {}
    notes: dict = {}
    for y in (years or [int(year)]):
        a, n = stage1_cohort(
            paths, pest, int(y), [str(s) for s in sites], daily_by_site, obs_frame,
            ctx.obs, ctx.obs, gate, site_history, log, as_of_doy=as_of_doy)
        alerts.update(a)
        notes.update(n)
    ctx.alert_map = alerts
    ctx.stage1_notes = notes
    ctx.cohort_daily_by_site = daily_by_site
    log.append(f"cohort pre-pass done in {_t.perf_counter() - t0:.1f}s "
               f"(Stage-2 will run for {len(alerts)} alerted site-years)")
    return {(pest, variant): ctx}


def run_batch(paths: Paths, raw_request: dict, run_row, variant: str) -> int:
    """Batch entry point. `run_row(paths, request, variant, ctx)` is the caller's
    single-request pipeline, injected to avoid a circular import and to guarantee
    batch and single share one engine.

    Returns the process exit code: 0 = batch ran, 2 = wholly-invalid request.
    """
    t0 = time.perf_counter()
    log: list[str] = []
    include_diag = bool(raw_request.get("include_diagnostics", False))
    output_dir = paths.output_dir
    generic = raw_request.get("input_csv") is not None
    # Set by the representative-site path so the cohort pre-pass can reuse the
    # one daily-master scan instead of doing a second one.
    prescanned_daily: dict | None = None
    prescanned_obs = None
    prescanned_years: list[int] = []
    prescanned_as_of_doy: int | None = None

    try:
        policy = load_policy(paths.configs_dir / "fallback_policy.yaml")
    except Exception as e:
        return fail_batch(output_dir, raw_request,
                          f"policy load failed: {type(e).__name__}: {e}", t0, generic)

    # ---- build the work list -------------------------------------------
    try:
        if generic:
            csv_path = resolve_input_csv(
                str(raw_request["input_csv"]), paths.input_dir, paths.pkg_root)
            rows_in = load_generic_rows(csv_path)
            log.append(f"BATCH generic input_csv={csv_path} "
                       f"rows={len(rows_in)} include_diagnostics={include_diag}")
        else:
            pest = normalize_pest(raw_request.get("pest"))
            if pest is None:
                return fail_batch(
                    output_dir, raw_request,
                    f"invalid pest {raw_request.get('pest')!r}; must be one of "
                    f"{sorted(VALID_PESTS)}", t0, generic)
            # Mode A (historical): `year`, or `start_year`/`end_year` for a span
            #   such as 2002-2022 / 2023 / 2024 (the project's split boundaries).
            # Mode B (operational): `as_of_date` = YYYY-MM-DD; the year comes
            #   from the date and the season is truncated to that DOY.
            as_of_date = raw_request.get("as_of_date")
            as_of_doy = None
            if as_of_date is not None:
                ts = pd.to_datetime(str(as_of_date), errors="coerce")
                if pd.isna(ts):
                    return fail_batch(
                        output_dir, raw_request,
                        f"invalid as_of_date {as_of_date!r}; expected YYYY-MM-DD",
                        t0, generic)
                years = [int(ts.year)]
                as_of_doy = int(ts.dayofyear)
            elif raw_request.get("start_year") is not None or \
                    raw_request.get("end_year") is not None:
                sy, ey = raw_request.get("start_year"), raw_request.get("end_year")
                for nm, v in (("start_year", sy), ("end_year", ey)):
                    if not isinstance(v, int) or isinstance(v, bool):
                        return fail_batch(
                            output_dir, raw_request,
                            f"invalid {nm} {v!r}; must be an integer", t0, generic)
                if int(ey) < int(sy):
                    return fail_batch(
                        output_dir, raw_request,
                        f"end_year {ey} is before start_year {sy}", t0, generic)
                years = list(range(int(sy), int(ey) + 1))
            else:
                year_raw = raw_request.get("year")
                if not isinstance(year_raw, int) or isinstance(year_raw, bool):
                    return fail_batch(
                        output_dir, raw_request,
                        f"invalid year {year_raw!r}; must be an integer "
                        f"(or supply start_year/end_year, or as_of_date)",
                        t0, generic)
                years = [int(year_raw)]
            year = years[0]
            rep_csv = resolve_rep_csv(
                raw_request.get("representative_sites_path")
                or raw_request.get("representative_sites_csv"),
                paths.input_dir, paths.pkg_root)
            rep_sites = representative_sites(rep_csv, pest)
            log.append(f"BATCH pest={pest} (rep-name={PEST_TO_KOREAN[pest]}) year={year} "
                       f"include_diagnostics={include_diag}")
            log.append(f"representative_sites_path={rep_csv}")
            log.append(f"representative_site_count={len(rep_sites)}")

            # Keep every representative site that has EITHER a LONG observation
            # OR daily weather for the year — the union, matching the deployed
            # API (run_predict batch.py: rep_set & (long_year_sites |
            # daily_year_sites)).
            #
            # An intersection here would silently drop the representative sites
            # that have weather but were not surveyed that year. Survey coverage
            # is the sparse side: for sheath_blight 2004 only 135 of 858
            # representative sites carry a LONG row, so an intersection returns
            # 135 instead of 858. Those 723 sites are not a data error — Stage-1
            # simply fires no alert for them and the fallback policy answers with
            # climatology, which is exactly what a forecast for an unsurveyed
            # site should be. A forward-looking request (a season with no
            # observations yet) would intersect to zero sites and return an empty
            # batch.
            daily_csv = (resolve_daily(raw_request, paths.input_dir, paths.pkg_root)
                         or paths.input_dir / "daily_weather.csv")
            obs_csv = _obs_path(paths, pest, raw_request)
            if not daily_csv.is_file():
                raise BatchRequestError(f"batch daily_weather.csv not found: {daily_csv}")
            if not Path(obs_csv).is_file():
                raise BatchRequestError(f"batch LONG observation CSV not found: {obs_csv}")
            obs = LongObsProvider(obs_csv)
            # ONE chunked pass over the daily master, filtered to the
            # representative sites and every requested year. This both decides
            # which sites have weather and produces the per-site seasons the
            # cohort Stage-1 pass needs, so the master is never loaded whole and
            # never re-read per site or per year.
            from .cohort import load_daily_cohort

            rep_set = set(rep_sites)
            daily_by_site = load_daily_cohort(daily_csv, rep_set, set(years))
            log.append(f"cohort daily scan: {len(daily_by_site)} sites over "
                       f"years={years[0]}..{years[-1]}")

            obs_year_col = pd.to_numeric(obs.frame["year"], errors="coerce")
            rows_in = []
            per_year_target: dict[int, list[str]] = {}
            year_available_count = 0
            max_sites = raw_request.get("max_sites")
            for y in years:
                long_y = set(obs.frame[obs_year_col == y]["site_id"].astype(str))
                daily_y = {s for s, d in daily_by_site.items()
                           if (pd.to_datetime(d["일시"], errors="coerce").dt.year == y).any()}
                # rep & (long | daily); daily_y is already within rep_set
                target_y = sorted((rep_set & long_y) | daily_y)
                year_available_count += len(target_y)
                if isinstance(max_sites, int) and max_sites >= 0 and len(target_y) > max_sites:
                    log.append(f"year={y} max_sites={max_sites} -> truncating {len(target_y)}")
                    target_y = target_y[:max_sites]
                per_year_target[y] = target_y
                rows_in.extend({"pest": pest, "site_id": s, "year": y} for s in target_y)
                log.append(f"year={y}: rep_in_long={len(rep_set & long_y)} "
                           f"rep_in_daily={len(daily_y)} union={len(target_y)}")
            log.append(f"year_available_site_count={year_available_count} "
                       f"requested_count={len(rows_in)}")
            prescanned_daily = daily_by_site
            prescanned_obs = obs
            prescanned_years = years
            prescanned_as_of_doy = as_of_doy
    except BatchRequestError as e:
        return fail_batch(output_dir, raw_request, str(e), t0, generic)
    except Exception as e:
        return fail_batch(output_dir, raw_request,
                          f"input resolution failed: {type(e).__name__}: {e}", t0, generic)

    # ---- per-row loop (assets cached per pest; a row never aborts the batch) --
    rows: list[dict] = []
    results: list[dict] = []
    counts = {"success": 0, "fallback": 0, "error": 0}
    # Keyed by (pest, variant): a generic-batch row may override stage2_variant,
    # and each variant needs its own Stage2Model. Carrying the field without
    # keying on it would make a row-level override silently inert.
    ctx_cache: dict[tuple[str, str], _PestContext] = {}

    # ---- cohort pre-pass: ONE daily-master scan + ONE vectorized Stage-1 ------
    # This is the training/evaluation data flow (phase_r_oracle_iou.
    # build_dispatch_alert_map, vendored as stage1.py::compute_stage1_table):
    # load the season for every site once, forward the whole cohort through the
    # A and D models in blocks, then read the alert off each site's series.
    # Stage-2 still runs per site, but only for the sites that actually fired.
    cohort_ctx: dict[str, _PestContext] = {}
    if not generic and rows_in:
        try:
            cohort_ctx = _prepare_cohort(
                paths, pest, year, [r["site_id"] for r in rows_in], policy,
                variant, include_diag, log,
                daily_by_site=prescanned_daily, obs_provider=prescanned_obs,
                years=prescanned_years, as_of_doy=prescanned_as_of_doy,
                request=raw_request)
        except Exception as e:  # never fatal: fall back to the per-row path
            log.append(f"cohort pre-pass unavailable ({type(e).__name__}: {e}); "
                       f"falling back to per-row Stage-1")
            cohort_ctx = {}
    ctx_cache.update(cohort_ctx)

    for i, raw_row in enumerate(rows_in):
        row_pest = normalize_pest(raw_row.get("pest"))
        site = str(raw_row.get("site_id", "")).strip()
        clim_for_row: dict = {}
        try:
            if row_pest is None:
                raise BatchRequestError(
                    f"invalid pest {raw_row.get('pest')!r}; must be one of "
                    f"{sorted(VALID_PESTS)}")
            if not site:
                raise BatchRequestError("site_id must be a non-empty string")
            try:
                row_year = int(str(raw_row.get("year")).strip())
            except (TypeError, ValueError):
                raise BatchRequestError(
                    f"invalid year {raw_row.get('year')!r}; must be an integer") from None

            # Build the per-row request FIRST: the inherited fields decide which
            # Stage-2 variant this row needs, so the context cannot be chosen
            # before they are resolved.
            req = {"pest": row_pest, "site_id": site, "year": row_year,
                   "include_diagnostics": include_diag}
            _inherit_request_fields(req, raw_request, raw_row)
            alert = raw_row.get("alert_tstar_doy")
            if alert not in (None, "", "nan"):
                req["alert_tstar_doy"] = int(float(alert))

            from .stage2_litert import VARIANTS as VARIANT_NAMES

            row_variant = str(req.get("stage2_variant") or variant)
            if row_variant not in VARIANT_NAMES:
                raise BatchRequestError(
                    f"invalid stage2_variant {row_variant!r}; must be one of "
                    f"{sorted(VARIANT_NAMES)}")
            ck = (row_pest, row_variant)
            if ck not in ctx_cache:
                ctx_cache[ck] = _PestContext(paths, row_pest, policy, row_variant)
                log.append(f"ASSET LOAD (once) for pest={row_pest} "
                           f"variant={row_variant}")
            ctx = ctx_cache[ck]
            clim_for_row = ctx.climatology

            resp, err = run_row(paths, req, row_variant, ctx)
            status, reason = classify(resp["stage2"]["learned_stage2"], err)
            row = flatten_response(resp)
            row["status"] = status
            row["error_reason"] = reason
            if include_diag:
                diag = (resp.get("diagnostics") or {}).get("transformer") or {}
                row.update(diag_row(diag))
            results.append({
                "row_index": i, "pest": row_pest, "site_id": site, "year": row_year,
                "status": status, "error_reason": reason,
                "final_source": resp["final_prediction"]["source"],
                "final_mu_doy": resp["final_prediction"]["mu_doy"],
                "final_pi95": resp["final_prediction"]["pi_95"],
                "learned_mu_doy": (resp["stage2"]["learned_stage2"] or {}).get("mu_doy"),
                "alert_tstar_doy": resp["stage1"]["alert_tstar_doy"],
                **({"diagnostics": (resp.get("diagnostics") or {})} if include_diag else {}),
            })
        except Exception as e:  # a row NEVER aborts the batch (batch.py:405)
            reason = f"{type(e).__name__}: {e}"
            status = "error"
            row = error_row(raw_row.get("pest"), site, raw_row.get("year"),
                            clim_for_row, reason)
            if include_diag:
                row.update(diag_row({}))
            results.append({"row_index": i, "site_id": site, "status": status,
                            "error_reason": reason,
                            "traceback": traceback.format_exc().splitlines()[-3:]})
        row["row_index"] = i
        counts[status] += 1
        rows.append(row)

    elapsed = round(time.perf_counter() - t0, 2)
    log.append(f"per-row loop done: success={counts['success']} "
               f"fallback={counts['fallback']} error={counts['error']} "
               f"elapsed_seconds={elapsed}")

    summary: dict = {
        "mode": "batch",
        "input_kind": "generic_csv" if generic else "representative_sites",
        "model_version": MODEL_VERSION,
        "stage2_variant": variant,
        "requested_count": len(rows_in),
        "success_count": counts["success"],
        "fallback_count": counts["fallback"],
        "error_count": counts["error"],
        "elapsed_seconds": elapsed,
        "results": results,
    }
    if not generic:
        pest = rows_in[0]["pest"] if rows_in else normalize_pest(raw_request.get("pest"))
        summary.update({
            "pest": pest,
            "pest_korean": PEST_TO_KOREAN.get(pest),
            "year": raw_request.get("year"),
            "representative_site_count": len(rep_sites),
            "year_available_site_count": year_available_count,
            "recommended_source": per_pest_policy(policy, pest).get(
                "recommended_source", "climatology") if pest else None,
        })
        # Top-level climatology block, restored to match the deployed batch
        # summary (run_predict batch.py). It is the per-pest constant every
        # fallback row is filled from, so the summary is self-describing.
        clim = None
        if pest:
            ctx_for_clim = next((c for (p_, _v), c in ctx_cache.items()
                             if p_ == pest), None)
            clim = (ctx_for_clim.climatology if ctx_for_clim is not None
                    else compute_climatology(paths.climatology_dir, pest,
                                             climatology_variant(policy, pest)))
        if clim:
            summary["climatology"] = {
                "mu_doy": clim["mu_doy"],
                "pi_95": clim["pi_95"],
                "variant": clim["variant"],
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(output_dir / "response.json",
                 json.dumps(summary, indent=2, ensure_ascii=False) + "\n")
    write_predictions_csv(output_dir / "predictions.csv", rows, include_diag, generic)
    write_run_log(output_dir / "run_log.txt", raw_request, log)

    print(f"[run_predict:batch] OK — rows={len(rows_in)} success={counts['success']} "
          f"fallback={counts['fallback']} error={counts['error']} elapsed={elapsed}s")
    print(f"[run_predict:batch] wrote: {output_dir}/response.json, predictions.csv, run_log.txt")
    return 0


def _obs_path(paths: Paths, pest: str, request: dict | None = None) -> Path:
    """long_observation_path from the request, else Layout A, else Layout B."""
    p = resolve_obs(request, paths.input_dir, paths.pkg_root, pest)
    if p is not None:
        return p
    return paths.input_dir / "LONG_by_pest" / f"RICE_LONG_{pest}.csv"
