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
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in fieldnames})


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
    path.write_text("\n".join(body) + "\n", encoding="utf-8")


def fail_batch(output_dir: Path, raw_request: dict, reason: str, t0: float,
               generic: bool = False) -> int:
    """batch.py:498-527 — whole-batch failure STILL writes all three files, exit 2."""
    import sys

    elapsed = round(time.perf_counter() - t0, 2)
    print(f"[run_predict:batch] ERROR (whole batch): {reason}", file=sys.stderr)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "mode": "batch",
        "pest": raw_request.get("pest"),      # raw, un-normalized (deployed behaviour)
        "year": raw_request.get("year"),
        "error": reason,
        "requested_count": 0, "success_count": 0, "fallback_count": 0,
        "error_count": 0, "elapsed_seconds": elapsed, "results": [],
    }
    (output_dir / "response.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_predictions_csv(output_dir / "predictions.csv", [], include_diag=False,
                          generic=generic)
    write_run_log(output_dir / "run_log.txt", raw_request,
                  [f"WHOLE-BATCH FAILURE: {reason}"])
    return 2


class _PestContext:
    """Per-pest assets, loaded once and reused across rows (deployed batch does
    the same with the .pt assets)."""

    def __init__(self, paths: Paths, pest: str, policy: dict, variant: str):
        from .stage2_litert import Stage2Model

        self.pest = pest
        self.policy_pp = per_pest_policy(policy, pest)
        self.climatology = compute_climatology(
            paths.climatology_dir, pest, climatology_variant(policy, pest))
        self.model = Stage2Model(pest, paths.stage2_dir, variant=variant)


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

    try:
        policy = load_policy(paths.configs_dir / "fallback_policy.yaml")
    except Exception as e:
        return fail_batch(output_dir, raw_request,
                          f"policy load failed: {type(e).__name__}: {e}", t0, generic)

    # ---- build the work list -------------------------------------------
    try:
        if generic:
            rows_in = load_generic_rows(Path(raw_request["input_csv"]))
            log.append(f"BATCH generic input_csv={raw_request['input_csv']} "
                       f"rows={len(rows_in)} include_diagnostics={include_diag}")
        else:
            pest = normalize_pest(raw_request.get("pest"))
            if pest is None:
                return fail_batch(
                    output_dir, raw_request,
                    f"invalid pest {raw_request.get('pest')!r}; must be one of "
                    f"{sorted(VALID_PESTS)}", t0, generic)
            year_raw = raw_request.get("year")
            if not isinstance(year_raw, int) or isinstance(year_raw, bool):
                return fail_batch(output_dir, raw_request,
                                  f"invalid year {year_raw!r}; must be an integer",
                                  t0, generic)
            year = int(year_raw)
            rep_csv = resolve_rep_csv(
                raw_request.get("representative_sites_path")
                or raw_request.get("representative_sites_csv"),
                paths.input_dir, paths.pkg_root)
            rep_sites = representative_sites(rep_csv, pest)
            log.append(f"BATCH pest={pest} (rep-name={PEST_TO_KOREAN[pest]}) year={year} "
                       f"include_diagnostics={include_diag}")
            log.append(f"representative_sites_path={rep_csv}")
            log.append(f"representative_site_count={len(rep_sites)}")

            # Restrict to sites that actually have BOTH LONG and daily for the year.
            daily_csv = paths.input_dir / "daily_weather.csv"
            obs_csv = _obs_path(paths, pest)
            if not daily_csv.is_file():
                raise BatchRequestError(f"batch daily_weather.csv not found: {daily_csv}")
            if not Path(obs_csv).is_file():
                raise BatchRequestError(f"batch LONG observation CSV not found: {obs_csv}")
            obs = LongObsProvider(obs_csv)
            long_year = set(obs.frame[obs.frame["year"] == year]["site_id"]
                            .astype(str).tolist())
            weather = FrameWeatherProvider(daily_csv)
            wf = weather.frame
            site_col = "지점ID" if "지점ID" in wf.columns else wf.columns[0]
            dt = pd.to_datetime(wf["일시"], errors="coerce")
            daily_year = set(wf[dt.dt.year == year][site_col].astype(str).tolist())
            rep_set = set(rep_sites)
            target = sorted(rep_set & long_year & daily_year)
            year_available_count = len(target)
            max_sites = raw_request.get("max_sites")
            if isinstance(max_sites, int) and max_sites >= 0 and len(target) > max_sites:
                log.append(f"max_sites={max_sites} -> truncating {len(target)}")
                target = target[:max_sites]
            log.append(f"year_available_site_count={year_available_count} "
                       f"requested_count={len(target)}")
            rows_in = [{"pest": pest, "site_id": s, "year": year} for s in target]
    except BatchRequestError as e:
        return fail_batch(output_dir, raw_request, str(e), t0, generic)
    except Exception as e:
        return fail_batch(output_dir, raw_request,
                          f"input resolution failed: {type(e).__name__}: {e}", t0, generic)

    # ---- per-row loop (assets cached per pest; a row never aborts the batch) --
    rows: list[dict] = []
    results: list[dict] = []
    counts = {"success": 0, "fallback": 0, "error": 0}
    ctx_cache: dict[str, _PestContext] = {}

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

            if row_pest not in ctx_cache:
                ctx_cache[row_pest] = _PestContext(paths, row_pest, policy, variant)
                log.append(f"ASSET LOAD (once) for pest={row_pest}")
            ctx = ctx_cache[row_pest]
            clim_for_row = ctx.climatology

            req = {"pest": row_pest, "site_id": site, "year": row_year,
                   "include_diagnostics": include_diag}
            alert = raw_row.get("alert_tstar_doy")
            if alert not in (None, "", "nan"):
                req["alert_tstar_doy"] = int(float(alert))

            resp, err = run_row(paths, req, variant, ctx)
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

    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "response.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_predictions_csv(output_dir / "predictions.csv", rows, include_diag, generic)
    write_run_log(output_dir / "run_log.txt", raw_request, log)

    print(f"[run_predict:batch] OK — rows={len(rows_in)} success={counts['success']} "
          f"fallback={counts['fallback']} error={counts['error']} elapsed={elapsed}s")
    print(f"[run_predict:batch] wrote: {output_dir}/response.json, predictions.csv, run_log.txt")
    return 0


def _obs_path(paths: Paths, pest: str) -> Path:
    """Layout A wins, else Layout B (mirrors run_predict._obs_path)."""
    local = paths.input_dir / "long_observation.csv"
    if local.is_file():
        return local
    return paths.input_dir / "LONG_by_pest" / f"RICE_LONG_{pest}.csv"
