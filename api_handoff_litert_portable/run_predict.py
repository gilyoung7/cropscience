"""Lightweight portable API — Stage-1 XGBoost JSON + Stage-2 LiteRT FP16.

No PyTorch. No TensorFlow. No .pt checkpoints. Output contract identical to
api_handoff_transformer/run_predict.py (see docs/lightweight_api_integration_plan.md).

    python run_predict.py --input-dir IN --output-dir OUT [--stage2-variant fp16|fp32]

Exit codes mirror the deployed API:
    0  success, AND any Stage-2 failure (the deployed code returns 0 either way)
    1  bad request / missing config
    2  missing input files
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
if str(PKG_ROOT) not in sys.path:
    sys.path.insert(0, str(PKG_ROOT))

from infer.fallback import (  # noqa: E402
    PolicyError, climatology_variant, compute_climatology, load_policy, per_pest_policy,
)
from infer.inputs import (  # noqa: E402
    InputResolutionError, OutputCollisionError, plan_output_dir, release_claim,
    resolve_daily, resolve_obs, write_atomic,
)
from infer.paths import MODEL_VERSION, Paths  # noqa: E402
from infer.preprocessing import PreprocessError, build_season  # noqa: E402
from infer.providers import FrameWeatherProvider, LongObsProvider, ProviderError  # noqa: E402
from infer.schemas import (  # noqa: E402
    FLAT_COLS, RequestError, build_response, flatten_response, load_request,
)
from infer.stage1_features import Stage1FeatureError  # noqa: E402
from infer.stage1_portable import (  # noqa: E402
    PortableBranch, Stage1Error, alert_from_series, load_gate, resolve_with_history,
)
from infer.stage2_litert import DEFAULT_VARIANT, VARIANTS, Stage2Error, Stage2Model, \
    prediction_interval  # noqa: E402


def fail(msg: str, code: int = 1, exc: Exception | None = None):
    """Port of run_predict.py:215-219 — stderr + exit, nothing written."""
    print(f"[run_predict] ERROR: {msg}", file=sys.stderr)
    if exc is not None:
        traceback.print_exception(exc, file=sys.stderr)
    sys.exit(code)


def check_input_files(paths: Paths, request: dict | None) -> None:
    """Port of run_predict.py:170-212 — exit 2 with the required-files list.

    Honours the optional request fields daily_weather_path /
    long_observation_path; with neither, the historical input_dir filenames are
    used unchanged.
    """
    missing = []
    if request is None:
        missing.append("request.json")
    try:
        if _daily_path(paths, request) is None:
            missing.append("daily_weather.csv")
        if _obs_path(paths, request) is None:
            missing.append("long_observation.csv (or LONG_by_pest/RICE_LONG_<pest>.csv)")
    except InputResolutionError as e:
        print(f"[run_predict] ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    if missing:
        print(
            "[run_predict] ERROR: missing required input file(s): "
            + ", ".join(missing)
            + f"\n  input dir: {paths.input_dir}\n"
            "  required: request.json, daily_weather.csv, long_observation.csv\n"
            "  (or name them explicitly via daily_weather_path / "
            "long_observation_path in request.json)",
            file=sys.stderr,
        )
        sys.exit(2)


def _daily_path(paths: Paths, request: dict | None) -> Path | None:
    """daily_weather_path from the request, else input_dir/daily_weather.csv."""
    return resolve_daily(request, paths.input_dir, paths.pkg_root)


def _obs_path(paths: Paths, request: dict | None) -> Path | None:
    """long_observation_path from the request, else Layout A, else Layout B."""
    return resolve_obs(request, paths.input_dir, paths.pkg_root,
                       (request or {}).get("pest"))


def run_stage1(paths: Paths, pest: str, site_id: str, year: int, weather,
               site_meta, pheno, diag: dict) -> tuple[dict | None, str | None]:
    """Live Stage-1 over the portable JSON assets. Returns (result, error)."""
    try:
        gate = load_gate(pest, paths.stage1_dir)
        diag["stage1_method"] = gate["method"]
        sh_path = paths.stage1_pest(pest) / "site_history.json"
        if not sh_path.is_file():
            raise Stage1Error(f"[{pest}] site_history.json missing: {sh_path}")
        site_history = json.loads(sh_path.read_text(encoding="utf-8"))
        with_h, _hrow = resolve_with_history(site_history, site_id, year)

        series = {}
        for branch in ("A", "D"):
            pb = PortableBranch(pest, branch, paths.stage1_dir)
            season = build_season(weather.frame, site_id, year, pb.doy_start,
                                  pb.doy_end, pb.feature_cols, site_meta, pheno)
            # Layout is set here and must not be normalized — see stage1_features.py.
            from infer.stage1_features import array_layout, base_x_from_season

            X = base_x_from_season(season, pb.feature_cols)
            diag.setdefault("stage1_base_x_layout", {})[branch] = array_layout(X)
            hist = {(str(site_id), int(year)): _hrow} if _hrow is not None else {}
            base = [{"site_id": str(site_id), "year": int(year), "X": X,
                     "L": 1, "R": 1, "censor_type": "right"}]
            per_sy = pb.forward_one(base, hist)
            series[branch] = per_sy.get((str(site_id), int(year)))

        res = alert_from_series(gate, series["A"], series["D"], with_h,
                                _first_doy_start(paths, pest))
        if res is None:
            return None, None  # no-alert is a valid outcome, not an error
        return res, None
    except (Stage1Error, PreprocessError, Stage1FeatureError, ProviderError) as e:
        return None, f"{type(e).__name__}: {e}"


def _first_doy_start(paths: Paths, pest: str) -> int:
    pb = PortableBranch(pest, "A", paths.stage1_dir)
    return pb.doy_start


def run_pipeline(paths: Paths, request: dict, variant: str,
                 ctx=None) -> tuple[dict, str | None]:
    """Stage-1 -> Stage-2 -> response. Never raises for Stage-2 failures.

    `ctx` is an optional per-pest asset cache (infer.batch._PestContext). When
    given, the policy/climatology/Stage-2 model are reused instead of reloaded —
    this is what lets batch amortize asset loading while running the EXACT same
    code path as a single request. When None the behaviour is unchanged.
    """
    pest, site_id, year = request["pest"], request["site_id"], int(request["year"])
    policy = load_policy(paths.configs_dir / "fallback_policy.yaml")
    pp = per_pest_policy(policy, pest)
    climatology = (ctx.climatology if ctx is not None
                   else compute_climatology(paths.climatology_dir, pest,
                                            climatology_variant(policy, pest)))

    diag: dict = {"stage1_backend": "xgboost_json", "stage2_backend": f"litert_{variant}"}
    run_log: list[str] = [f"pest={pest} site={site_id} year={year}",
                          f"climatology mu_doy={climatology['mu_doy']} "
                          f"variant={climatology['variant']}"]

    daily_csv = _daily_path(paths, request) or (paths.input_dir / "daily_weather.csv")
    obs_csv = _obs_path(paths, request) or (
        paths.input_dir / "LONG_by_pest" / f"RICE_LONG_{pest}.csv")
    # In cohort batch mode the providers were built once, over a single chunked
    # pass of the daily master; reuse them instead of re-reading it per site.
    cohort_weather = getattr(ctx, "weather", None) if ctx is not None else None
    cohort_obs = getattr(ctx, "obs", None) if ctx is not None else None
    try:
        weather = cohort_weather if cohort_weather is not None else FrameWeatherProvider(daily_csv)
        obs = cohort_obs if cohort_obs is not None else LongObsProvider(obs_csv)
    except ProviderError as e:
        diag["input_error"] = str(e)
        return build_response(request, None, climatology, policy, diag, str(e),
                              _backends(variant)), str(e)
    diag["daily_input"] = daily_csv.name
    diag["long_observation_input"] = obs_csv.name
    diag["input_mode"] = "per_site" if obs_csv.name == "long_observation.csv" else "full_bundle"

    # ---- Stage 1 ------------------------------------------------------
    # Cohort mode: the alert + 14 dispatch features for every site were computed
    # in one vectorized pass before this loop (infer/cohort.stage1_cohort), so
    # here we only look this site up. A site absent from the map fired no alert
    # — the same outcome the per-site path produces, reached the same way.
    manual_alert = request.get("alert_tstar_doy")
    alert_map = getattr(ctx, "alert_map", None) if ctx is not None else None
    if alert_map is not None:
        diag["stage1_method"] = getattr(ctx, "gate_method", None)
        key = f"{site_id}|{int(year)}"
        hit = alert_map.get(key)
        s1 = hit
        s1_err = None if hit else (getattr(ctx, "stage1_notes", {}) or {}).get(key)
        diag["stage1_source"] = "cohort_prepass"
    else:
        s1, s1_err = run_stage1(paths, pest, site_id, year, weather, obs, obs, diag)
    s1_alert = s1["alert_tstar_doy"] if s1 else None
    dispatch_override = s1["dispatch_features"] if s1 else None
    if s1_err:
        alert_source = "stage1_error"
    elif s1_alert is not None:
        alert_source = "stage1_live"
    else:
        alert_source = "stage1_no_alert"
    if manual_alert is not None:
        alert_source = "manual_request" if s1_alert is None else "manual_request_over_stage1"
    alert_for_build = manual_alert if manual_alert is not None else s1_alert
    diag.update({"alert_source": alert_source, "stage1_alert_tstar_doy": s1_alert,
                 "stage1_error": s1_err})
    run_log.append(f"Stage-1: method={diag.get('stage1_method')} "
                   f"alert_source={alert_source} alert={s1_alert}"
                   + (f" error={s1_err}" if s1_err else ""))

    if alert_for_build is None or dispatch_override is None:
        # Worded to match the deployed API verbatim (run_predict.py:406-409),
        # including the trailing space before the appended Stage-1 error, so
        # error_reason in predictions.csv compares equal string-for-string.
        _gate = diag.get("stage1_method") or "?"
        err = (f"Stage-1 fired no alert for pest={pest} site={site_id} "
               f"year={year} (gate={_gate}); no manual "
               f"alert_tstar_doy supplied. " + (s1_err or ""))
        run_log.append(err)
        return build_response(request, None, climatology, policy, diag, err,
                             _backends(variant)), err

    # ---- Stage 2 (LiteRT) ---------------------------------------------
    selected_offset = pp.get("selected_fixed_offset")
    if selected_offset is None:
        err = f"selected_fixed_offset missing in fallback_policy.yaml for pest={pest}"
        return build_response(request, None, climatology, policy, diag, err,
                             _backends(variant)), err
    # Operational guard: Stage-2 reads the window [tstar-window+1, tstar] with
    # tstar = alert + selected_offset. Before that DOY is observed, part of the
    # window is padding, and running anyway would silently return a number built
    # from imputed filler. Fall back instead, and say why.
    as_of_doy = getattr(ctx, "as_of_doy", None) if ctx is not None else None
    if as_of_doy is not None and alert_for_build is not None:
        need_doy = int(alert_for_build) + int(selected_offset)
        if int(as_of_doy) < need_doy:
            status = "stage2_pending_window_not_yet_observed"
            err = (f"Stage-2 not yet evaluable for pest={pest} site={site_id} "
                   f"year={year}: needs weather through DOY {need_doy} "
                   f"(alert {alert_for_build} + offset {selected_offset}), "
                   f"as_of_date is DOY {as_of_doy} [{status}]")
            diag["stage2_output_status"] = status
            diag["stage2_needs_doy"] = need_doy
            run_log.append(err)
            return build_response(request, None, climatology, policy, diag, err,
                                  _backends(variant), stage2_output_status=status), err

        # The window IS observed, but the training imputation
        # (interpolate(limit_direction="both")) fills a gap from the first valid
        # value AFTER it. If a missing run inside the window is still unresolved
        # at as_of_date, an operational run would impute it differently from the
        # full-season run — so decline rather than emit a number we cannot stand
        # behind. Historical mode never reaches here.
        from infer.cohort import as_of_reproduces_full_year

        try:
            site_daily_for_check = weather.daily(site_id, year)
        except ProviderError:
            site_daily_for_check = None
        if site_daily_for_check is not None:
            reproducible, why = as_of_reproduces_full_year(
                site_daily_for_check, int(as_of_doy), need_doy,
                int(ctx.model.md.nowcast_window) if ctx is not None else 28)
            if not reproducible:
                status = "stage2_window_crosses_unresolved_missing_run"
                err = (f"Stage-2 withheld for pest={pest} site={site_id} "
                       f"year={year}: {why} [{status}]")
                diag["stage2_output_status"] = status
                diag["stage2_block_detail"] = why
                run_log.append(err)
                return build_response(request, None, climatology, policy, diag,
                                      err, _backends(variant),
                                      stage2_output_status=status), err

    try:
        model = ctx.model if ctx is not None else Stage2Model(
            pest, paths.stage2_dir, variant=variant)
        diag.update({"d_in": model.md.d_in, "T_full_season": model.md.T,
                     "doy_start": model.md.doy_start, "doy_end": model.md.doy_end,
                     "sigma_days": model.sigma, "ckpt_loaded": True})
        site_d, pheno_rows = {}, []
        if model.md.requires_site_coords:
            lat, lon = obs.latlon(site_id)
            site_d = {"lat": lat, "lon": lon}
        if model.md.requires_phenology:
            cols = [c for c in ("days_since_growing_start", "days_until_growing_end",
                                "is_growing") if c in model.md.base_channels]
            pheno_rows = obs.rows(site_id, year)[["obs_doy"] + cols].to_dict("records")
        # Stage-2's tensor builder is per-site by contract and rejects a
        # multi-site frame. The daily input may legitimately hold many sites
        # (a batch run, or the full master handed to a single request), so
        # filter to this site first — the deployed API does the same via
        # load_input_daily(site_id=...) / _daily_year_from_frame.
        site_daily = weather.daily(site_id, year)
        out = model.predict(site_daily, pest, int(alert_for_build),
                            dispatch_override, site_d, pheno_rows, year)
    except Exception as e:  # Stage-2 never propagates — matches the deployed contract
        err = f"Stage-2 failed for pest={pest}: {type(e).__name__}: {e}"
        run_log.append(err)
        return build_response(request, None, climatology, policy, diag, err,
                             _backends(variant)), err

    built = out["built"]
    diag.update({"model_forward_succeeded": True,
                 "input_X_shape": list(built.X.shape),
                 "alert_tstar_doy_used": int(built.alert_tstar_doy),
                 "tstar_season_index": int(built.tstar_season_index),
                 "base_channels_status": "real_preprocessing",
                 "zero_placeholder_used": False,
                 "mu_rel_season_index": round(float(out["mu_rel"]), 4)})
    pi = prediction_interval(model.sigma, out["mu_doy_temporal"])
    learned = {
        "mu_doy": pi["mu_doy"],
        "pi_95": pi["pi_95"],
        "selected_offset": int(selected_offset),
        "output_status": pp.get("learned_output_status", "experimental"),
        "model_kind": "lead_v3",
    }
    run_log.append(f"Stage-2 OK: mu_doy={pi['mu_doy']} backend=litert_{variant}")
    return build_response(request, learned, climatology, policy, diag, None,
                          _backends(variant)), None


def _backends(variant: str) -> dict:
    """Additive-only optional metadata (permitted by the brief)."""
    return {"stage1_backend": "xgboost_json", "stage2_backend": f"litert_{variant}"}


def write_outputs(paths: Paths, response: dict) -> None:
    """Port of run_predict.py:593-622 — response.json + predictions.csv.

    Written atomically (temp file + os.replace) so an interrupted run leaves the
    previous results intact rather than a truncated file.
    """
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    write_atomic(paths.output_dir / "response.json",
                 json.dumps(response, indent=2, ensure_ascii=False) + "\n")
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=FLAT_COLS)
    w.writeheader()
    w.writerow(flatten_response(response))
    write_atomic(paths.output_dir / "predictions.csv", buf.getvalue(), newline="")


def write_run_log(paths: Paths, request: dict, lines: list[str]) -> None:
    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    body = [
        "run_predict.py log (Stage-1 XGBoost JSON + Stage-2 LiteRT)",
        f"timestamp_utc: {ts}",
        f"model_version: {MODEL_VERSION}",
        f"request: {json.dumps(request, ensure_ascii=False)}",
        "",
        "events:",
        *[f"  - {ln}" for ln in lines],
    ]
    write_atomic(paths.output_dir / "run_log.txt", "\n".join(body) + "\n")


def main(argv: list[str] | None = None) -> int:
    """Entry point. Wraps _main so the output-directory claim is always released,
    whether the run succeeds, fails, or exits via sys.exit."""
    claimed: list[Path] = []
    try:
        return _main(argv, claimed)
    finally:
        for d in claimed:
            release_claim(d)


def _main(argv: list[str] | None, claimed: list[Path]) -> int:
    ap = argparse.ArgumentParser(description="Lightweight portable pest-timing API")
    ap.add_argument("--input-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--stage2-variant", choices=VARIANTS, default=DEFAULT_VARIANT,
                    help=f"Stage-2 model precision (default: {DEFAULT_VARIANT})")
    ap.add_argument("--representative-sites", default=None,
                    help="path to representative_site_ids_2002_2024.csv (batch mode)")
    ap.add_argument("--overwrite", action="store_true",
                    help="replace existing results in --output-dir "
                         "(default: refuse, so concurrent runs cannot clobber "
                         "each other)")
    ap.add_argument("--unique-output-subdir", action="store_true",
                    help="write into a fresh per-run subfolder of --output-dir "
                         "instead of the directory itself")
    args = ap.parse_args(argv)

    paths = Paths.from_root(PKG_ROOT, args.input_dir, args.output_dir)
    req_path = paths.input_dir / "request.json"
    if not req_path.is_file():
        check_input_files(paths, None)

    # Peek at `mode` BEFORE single-mode validation, exactly as the deployed API
    # does (run_predict.py:792-797) — batch has its own request schema, so
    # validating it as a single request would reject it with a misleading
    # "site_id must be a non-empty string".
    try:
        raw_request = json.loads(req_path.read_text(encoding="utf-8"))
    except Exception as e:
        fail(f"request.json is not valid JSON: {type(e).__name__}: {e}")
    if not isinstance(raw_request, dict):
        fail("request.json must contain a JSON object")

    # Decide where this run may write BEFORE doing any work, so a collision is
    # reported immediately and nothing is half-produced. The request may set the
    # same two switches as the CLI flags.
    try:
        out_dir, out_note = plan_output_dir(
            paths.output_dir, raw_request,
            overwrite=bool(args.overwrite or raw_request.get("overwrite")),
            unique_subdir=bool(args.unique_output_subdir
                               or raw_request.get("unique_output_subdir")))
    except OutputCollisionError as e:
        fail(str(e), code=1)
    claimed.append(out_dir)
    paths = Paths(pkg_root=paths.pkg_root, input_dir=paths.input_dir,
                  output_dir=out_dir)
    print(f"[run_predict] {out_note}", file=sys.stderr)

    if str(raw_request.get("mode", "single")).lower() == "batch":
        from infer.batch import run_batch

        variant = str(raw_request.get("stage2_variant") or args.stage2_variant)
        if variant not in VARIANTS:
            fail(f"invalid stage2_variant {variant!r}; must be one of {list(VARIANTS)}")
        if args.representative_sites and not raw_request.get("representative_sites_path"):
            raw_request["representative_sites_path"] = args.representative_sites
        # Batch needs EITHER a rep-CSV selection (pest + a time spec) OR a
        # generic input_csv. The time spec is `year` (one season),
        # `start_year`/`end_year` (a span such as 2002-2022), or `as_of_date`
        # (operational run for the season containing that date).
        _has_time = any(raw_request.get(k) is not None
                        for k in ("year", "start_year", "end_year", "as_of_date"))
        if raw_request.get("input_csv") is None and (
            raw_request.get("pest") is None or not _has_time
        ):
            fail(
                "batch request is missing its input specification. Supply either:\n"
                "  (a) representative-site batch: \"pest\" plus one of \"year\", "
                "\"start_year\"+\"end_year\", or \"as_of_date\" (plus optional "
                "\"representative_sites_path\"), or\n"
                "  (b) generic CSV batch: \"input_csv\" pointing at a CSV with columns "
                "pest,site_id,year[,alert_tstar_doy].\n"
                f"  got keys: {sorted(raw_request)}",
                code=2,
            )
        return run_batch(paths, raw_request, run_pipeline, variant)

    try:
        request = load_request(req_path)
    except RequestError as e:
        fail(str(e))
    check_input_files(paths, request)

    try:
        response, err = run_pipeline(paths, request, args.stage2_variant)
    except PolicyError as e:
        fail(str(e))
    except Exception as e:  # noqa: BLE001 - surface unexpected faults loudly
        fail(f"unexpected failure: {type(e).__name__}: {e}", exc=e)

    paths.output_dir.mkdir(parents=True, exist_ok=True)
    write_outputs(paths, response)
    write_run_log(paths, request, [f"outcome: {'ok' if err is None else err}"])
    if err:
        print(f"[run_predict] Stage-2 not used: {err}", file=sys.stderr)
    print(json.dumps(response, indent=2, ensure_ascii=False))
    # run_predict.py:848 is `return 0 if learned_err is None else 0` — always 0.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
