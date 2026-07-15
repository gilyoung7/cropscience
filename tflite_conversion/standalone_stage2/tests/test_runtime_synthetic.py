"""Synthetic end-to-end test of the standalone package. NO torch required.

Generates a deterministic synthetic year of daily weather (real schema, made-up
values), runs every packaged pest through the full standalone path
(preprocessing -> LiteRT -> mu_doy + PI), and exercises the error contract.

This is the test that must pass inside the torch-free runtime venv, so it
imports only the packaged runtime + numpy/pandas.

    python tests/test_runtime_synthetic.py                    # uses dist/ package
    python tests/test_runtime_synthetic.py --package /path/to/stage2_litert

Values are NOT physically meaningful; this exercises the plumbing, not the
science. Real-data correctness is tests/test_output_parity.py.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DEFAULT_PKG = HERE.parent / "dist" / "stage2_litert"


def synthetic_daily(year: int = 2024, seed: int = 0) -> pd.DataFrame:
    """A full deterministic year in the real Korean daily schema."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range(f"{year}-01-01", f"{year}-12-31", freq="D")
    n = len(dates)
    doy = np.arange(1, n + 1)
    # smooth seasonal signal + noise so rolling features are non-degenerate
    seas = np.sin((doy - 100) / 365.0 * 2 * np.pi)
    tmean = 12.0 + 12.0 * seas + rng.normal(0, 1.0, n)
    return pd.DataFrame({
        "지점ID": "SYNTH_0001",
        "일시": dates.strftime("%Y-%m-%d"),
        "일강수량(mm)": np.clip(rng.gamma(0.6, 4.0, n) - 1.0, 0, None),
        "최고기온(°C)": tmean + 5.0 + rng.normal(0, 0.5, n),
        "최저기온(°C)": tmean - 5.0 + rng.normal(0, 0.5, n),
        "평균기온(°C)": tmean,
        "평균 풍속(m/s)": np.abs(rng.normal(2.0, 0.6, n)),
        "최대 풍속(m/s)": np.abs(rng.normal(5.0, 1.2, n)),
        "평균 상대습도(%)": np.clip(70.0 + 10.0 * seas + rng.normal(0, 4, n), 5, 100),
        "합계 일조시간(h)": np.clip(rng.normal(6.0, 2.0, n), 0, 14),
        "합계 일사량(MJ/m2)": np.clip(rng.normal(14.0, 4.0, n), 0, 33),
        "GDD10_since_gs": np.cumsum(np.clip(tmean - 10.0, 0, None)),
    })


def synthetic_dispatch(alert: int) -> dict[str, float]:
    """All 14 dispatch features, explicitly supplied (nothing defaulted)."""
    return {
        "alert_tstar": float(alert),
        "with_history": 0.0,
        "dispatch_branch": "D",
        "A_score_at_alert": 0.65,
        "D_score_at_alert": 0.64,
        "score_margin": -0.01,
        "dispatch_score_at_alert": 0.64,
        "dispatch_tau_used": 0.6,
        "score_over_tau_margin": 0.04,
        "recent_14d_mean_score": 0.46,
        "recent_28d_mean_score": 0.46,
        "score_above_tau_streak": 3.0,
        "score_rolling_slope_14d": 0.055,
        "p_mean_so_far_at_alert": 0.46,
    }


def synthetic_phenology(doy_start: int, doy_end: int) -> list[dict]:
    """LONG-style step records: a few observed DOYs, ffilled by the runtime."""
    gs, ge = doy_start + 20, doy_end - 20
    out = []
    for d in range(doy_start + 5, doy_end - 5, 30):
        out.append({
            "obs_doy": int(d),
            "days_since_growing_start": float(d - gs),
            "days_until_growing_end": float(ge - d),
            "is_growing": 1.0 if gs <= d <= ge else 0.0,
        })
    return out


def run(pkg: Path) -> int:
    sys.path.insert(0, str(pkg))
    from runtime.interpreter import Stage2Model  # noqa: E402
    from runtime.preprocessing import PreprocessError  # noqa: E402
    from runtime.schema import DispatchRequest, SchemaError, load_metadata  # noqa: E402

    manifest = json.loads((pkg / "manifest.json").read_text())
    pests = manifest["pests"]
    daily = synthetic_daily()
    rows, failures = [], []

    for pest in pests:
        try:
            md = load_metadata(pkg / "models", pest)
            alert = md.doy_start + 30
            req = DispatchRequest(
                pest=pest, alert_tstar=alert,
                dispatch_features=synthetic_dispatch(alert),
                site={"lat": 35.5, "lon": 128.5} if md.requires_site_coords else {},
                phenology=(synthetic_phenology(md.doy_start, md.doy_end)
                           if md.requires_phenology else []),
                year=2024,
            )
            for variant in ("fp16", "fp32"):
                res = Stage2Model(pest, variant=variant,
                                  models_dir=pkg / "models").predict(daily, req)
                assert res["pest"] == pest
                assert res["backend"] == f"tflite_{variant}"
                assert res["alert_tstar"] == alert
                assert isinstance(res["mu_doy"], float)
                lo, hi = res["prediction_interval_95"]
                assert isinstance(lo, int) and isinstance(hi, int) and lo < hi, res
                assert list(res["input_shape"]) == [1, 1, md.T, md.d_in]
                if variant == "fp16":
                    rows.append({
                        "pest": pest, "mu_doy": res["mu_doy"],
                        "pi_95": res["prediction_interval_95"],
                        "input_shape": res["input_shape"],
                        "sha256_prefix": res["model_sha256"][:12],
                    })
            print(f"  OK   {pest:<20} mu_doy={rows[-1]['mu_doy']:>8.2f} "
                  f"pi={rows[-1]['pi_95']}  shape={rows[-1]['input_shape'][2:]}")
        except Exception as e:
            failures.append({"pest": pest, "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()})
            print(f"  FAIL {pest:<20} {type(e).__name__}: {str(e)[:100]}")

    # --- error contract: these MUST raise, not silently fill -----------
    print("\n  error contract:")
    checks = []
    md = load_metadata(pkg / "models", pests[0])
    good_alert = md.doy_start + 30

    def expect_raise(label, fn, exc):
        try:
            fn()
        except exc as e:
            print(f"    OK   {label:<34} -> {type(e).__name__}")
            checks.append(True)
            return
        except Exception as e:
            print(f"    FAIL {label:<34} -> wrong type {type(e).__name__}: {e}")
            checks.append(False)
            return
        print(f"    FAIL {label:<34} -> no error raised")
        checks.append(False)

    model = Stage2Model(pests[0], variant="fp16", models_dir=pkg / "models")

    def _missing_dispatch():
        f = synthetic_dispatch(good_alert)
        del f["D_score_at_alert"]
        model.predict(daily, DispatchRequest(
            pest=pests[0], alert_tstar=good_alert, dispatch_features=f,
            site={"lat": 35.5, "lon": 128.5}, year=2024,
            phenology=synthetic_phenology(md.doy_start, md.doy_end)))

    def _alert_out_of_range():
        model.predict(daily, DispatchRequest(
            pest=pests[0], alert_tstar=md.doy_end + 5,
            dispatch_features=synthetic_dispatch(md.doy_end + 5),
            site={"lat": 35.5, "lon": 128.5}, year=2024,
            phenology=synthetic_phenology(md.doy_start, md.doy_end)))

    def _missing_doy():
        d = daily.drop(index=range(100, 110))
        model.predict(d, DispatchRequest(
            pest=pests[0], alert_tstar=good_alert,
            dispatch_features=synthetic_dispatch(good_alert),
            site={"lat": 35.5, "lon": 128.5}, year=2024,
            phenology=synthetic_phenology(md.doy_start, md.doy_end)))

    def _duplicate_doy():
        d = pd.concat([daily, daily.iloc[[150]]], ignore_index=True)
        model.predict(d, DispatchRequest(
            pest=pests[0], alert_tstar=good_alert,
            dispatch_features=synthetic_dispatch(good_alert),
            site={"lat": 35.5, "lon": 128.5}, year=2024,
            phenology=synthetic_phenology(md.doy_start, md.doy_end)))

    def _missing_column():
        d = daily.drop(columns=["평균 상대습도(%)"])
        model.predict(d, DispatchRequest(
            pest=pests[0], alert_tstar=good_alert,
            dispatch_features=synthetic_dispatch(good_alert),
            site={"lat": 35.5, "lon": 128.5}, year=2024,
            phenology=synthetic_phenology(md.doy_start, md.doy_end)))

    def _unknown_pest():
        load_metadata(pkg / "models", "not_a_pest")

    expect_raise("missing dispatch feature", _missing_dispatch, SchemaError)
    expect_raise("alert outside season", _alert_out_of_range, SchemaError)
    expect_raise("missing DOY in daily", _missing_doy, PreprocessError)
    expect_raise("duplicate DOY in daily", _duplicate_doy, PreprocessError)
    expect_raise("missing weather column", _missing_column, PreprocessError)
    expect_raise("unknown pest", _unknown_pest, SchemaError)

    ok = (not failures) and len(rows) == len(pests) and all(checks)
    print(f"\nsynthetic runtime: {'PASS' if ok else 'FAIL'} "
          f"({len(rows)}/{len(pests)} pests, {sum(checks)}/{len(checks)} error checks)")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, default=DEFAULT_PKG)
    args = ap.parse_args()
    if not (args.package / "manifest.json").is_file():
        print(f"error: no package at {args.package} (run build_package.py first)",
              file=sys.stderr)
        return 2
    print(f"synthetic runtime test — package: {args.package}")
    return run(args.package)


if __name__ == "__main__":
    raise SystemExit(main())
