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
    """Port of run_predict.py:170-212 — exit 2 with the required-files list."""
    missing = []
    if request is None:
        missing.append("request.json")
    if not (paths.input_dir / "daily_weather.csv").is_file():
        missing.append("daily_weather.csv")
    if not _obs_path(paths, request).is_file():
        missing.append("long_observation.csv (or LONG_by_pest/RICE_LONG_<pest>.csv)")
    if missing:
        print(
            "[run_predict] ERROR: missing required input file(s): "
            + ", ".join(missing)
            + f"\n  input dir: {paths.input_dir}\n"
            "  required: request.json, daily_weather.csv, long_observation.csv",
            file=sys.stderr,
        )
        sys.exit(2)


def _obs_path(paths: Paths, request: dict | None) -> Path:
    """Port of run_predict.py:157-167 — Layout A wins, else Layout B."""
    local = paths.input_dir / "long_observation.csv"
    if local.is_file():
        return local
    pest = (request or {}).get("pest", "UNKNOWN")
    return paths.input_dir / "LONG_by_pest" / f"RICE_LONG_{pest}.csv"


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

    daily_csv = paths.input_dir / "daily_weather.csv"
    obs_csv = _obs_path(paths, request)
    try:
        weather = FrameWeatherProvider(daily_csv)
        obs = LongObsProvider(obs_csv)
    except ProviderError as e:
        diag["input_error"] = str(e)
        return build_response(request, None, climatology, policy, diag, str(e),
                              _backends(variant)), str(e)
    diag["daily_input"] = daily_csv.name
    diag["long_observation_input"] = obs_csv.name
    diag["input_mode"] = "per_site" if obs_csv.name == "long_observation.csv" else "full_bundle"

    # ---- Stage 1 (live, portable JSON) --------------------------------
    manual_alert = request.get("alert_tstar_doy")
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
        err = (f"Stage-1 fired no alert for pest={pest} site={site_id} year={year}; "
               f"no dispatch features available. " + (s1_err or ""))
        run_log.append(err)
        return build_response(request, None, climatology, policy, diag, err,
                             _backends(variant)), err

    # ---- Stage 2 (LiteRT) ---------------------------------------------
    selected_offset = pp.get("selected_fixed_offset")
    if selected_offset is None:
        err = f"selected_fixed_offset missing in fallback_policy.yaml for pest={pest}"
        return build_response(request, None, climatology, policy, diag, err,
                             _backends(variant)), err
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
    """Port of run_predict.py:593-622 — response.json + predictions.csv."""
    paths.output_dir.mkdir(parents=True, exist_ok=True)
    (paths.output_dir / "response.json").write_text(
        json.dumps(response, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    flat = flatten_response(response)
    with open(paths.output_dir / "predictions.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FLAT_COLS)
        w.writeheader()
        w.writerow(flat)


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
    (paths.output_dir / "run_log.txt").write_text("\n".join(body) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Lightweight portable pest-timing API")
    ap.add_argument("--input-dir", type=Path, default=None)
    ap.add_argument("--output-dir", type=Path, default=None)
    ap.add_argument("--stage2-variant", choices=VARIANTS, default=DEFAULT_VARIANT,
                    help=f"Stage-2 model precision (default: {DEFAULT_VARIANT})")
    ap.add_argument("--representative-sites", default=None,
                    help="path to representative_site_ids_2002_2024.csv (batch mode)")
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

    if str(raw_request.get("mode", "single")).lower() == "batch":
        from infer.batch import run_batch

        variant = str(raw_request.get("stage2_variant") or args.stage2_variant)
        if variant not in VARIANTS:
            fail(f"invalid stage2_variant {variant!r}; must be one of {list(VARIANTS)}")
        if args.representative_sites and not raw_request.get("representative_sites_path"):
            raw_request["representative_sites_path"] = args.representative_sites
        # Batch needs EITHER a rep-CSV selection (pest+year) OR a generic input_csv.
        if raw_request.get("input_csv") is None and (
            raw_request.get("pest") is None or raw_request.get("year") is None
        ):
            fail(
                "batch request is missing its input specification. Supply either:\n"
                "  (a) representative-site batch: \"pest\" and \"year\" (plus optional "
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
