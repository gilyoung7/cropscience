"""C. Model output parity on the 8 real smoke cases.

Compares mu_doy from four paths on the same real (site, year):
  1. original PyTorch model            (api_handoff_transformer/infer/model.py)
  2. existing common TFLite check      (tflite_conversion/stage2 artifacts)
  3. standalone package FP32
  4. standalone package FP16

and cross-checks against the site/year/alert/mu recorded in
api_handoff_transformer/README.md §10 (via pest_configs.SMOKE).

Tolerances (days): FP32 <= 1e-3, FP16 <= 0.1 — same bounds as the common
pipeline; see tflite_conversion/stage2/validate_all.py for the rationale.

Needs torch. BUILD-side test.

    ../../.venv-tflite/bin/python tests/test_output_parity.py \
        --daily-master "$DAILY_MASTER" --long-dir "$LONG_DIR"
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO_ROOT = PKG.parents[1]
API = REPO_ROOT / "api_handoff_transformer"
DIST = PKG / "dist" / "stage2_litert"
STAGE2 = REPO_ROOT / "tflite_conversion" / "stage2"

sys.path.insert(0, str(PKG))
sys.path.insert(0, str(STAGE2))
sys.path.insert(0, str(API))

from test_preprocessing_parity import api_built, standalone_request  # noqa: E402

from runtime.interpreter import Stage2Model, _prediction_interval  # noqa: E402
from runtime.schema import load_metadata  # noqa: E402

ATOL = {"fp32": 1e-3, "fp16": 0.1}


def run_pest(pest: str, site: str, year: int, daily_csv: Path, obs_csv: Path,
             cache_dir: Path, smoke) -> dict:
    from checkpoint import original_mu

    # --- 1. original PyTorch, on the API's own tensor -------------------
    loaded, ref = api_built(pest, site, year, daily_csv, obs_csv, cache_dir)
    mu_t = original_mu(loaded, ref.X, ref.tstar, ref.valid_mask)
    mu_rel_torch = float(mu_t.flatten()[0])
    to_doy = lambda m: float(m) + loaded.doy_start - 1.0
    doy_torch = to_doy(mu_rel_torch)

    # --- 2. existing common TFLite validation path ----------------------
    from validate_all import _interp, tflite_mu

    p_common = STAGE2 / "artifacts" / pest / f"{pest}_stage2_fp32.tflite"
    mu_common = float(np.asarray(
        tflite_mu(_interp(p_common), ref.X, ref.tstar, ref.valid_mask)
    ).reshape(-1)[0])
    doy_common = to_doy(mu_common)

    # --- 3/4. standalone package, from its own preprocessing ------------
    md = load_metadata(DIST / "models", pest)
    req = standalone_request(pest, site, year, obs_csv, ref)
    daily = pd.read_csv(cache_dir / f"daily_site_{site}.csv", encoding="utf-8-sig")

    out: dict = {
        "pest": pest, "site": site, "year": year,
        "alert_expected": smoke.alert_doy,
        "readme_mu_doy": smoke.mu_doy,
        "mu_doy_torch": doy_torch,
        "mu_doy_common_tflite_fp32": doy_common,
        "common_vs_torch_days": abs(doy_common - doy_torch),
    }
    for v in ("fp32", "fp16"):
        model = Stage2Model(pest, variant=v, models_dir=DIST / "models")
        res = model.predict(daily, req)
        out[f"standalone_{v}_mu_doy"] = res["mu_doy"]
        out[f"standalone_{v}_pi95"] = res["prediction_interval_95"]
        out[f"standalone_{v}_alert"] = res["alert_tstar"]
        # compare on the unrounded torch DOY vs the runtime's reported mu_doy
        # (which is round(.,2)); use the runtime's raw season index for the
        # numeric comparison so rounding is not counted as model error.
        raw_doy = float(res["mu_rel_season_index"]) + md.doy_start - 1.0
        out[f"standalone_{v}_days_vs_torch"] = abs(raw_doy - doy_torch)
        out[f"standalone_{v}_pass"] = out[f"standalone_{v}_days_vs_torch"] <= ATOL[v]

    # PI from the API's own rule applied to the torch mu, vs the runtime's PI
    ref_pi = _prediction_interval(md, doy_torch)
    out["reference_pi95_from_torch"] = ref_pi["pi_95"]
    out["pi_match_fp32"] = out["standalone_fp32_pi95"] == ref_pi["pi_95"]
    out["pi_match_fp16"] = out["standalone_fp16_pi95"] == ref_pi["pi_95"]

    out["alert_ok"] = out["standalone_fp32_alert"] == smoke.alert_doy
    out["readme_delta_days"] = abs(doy_torch - smoke.mu_doy)
    out["readme_match"] = out["readme_delta_days"] <= 0.005
    out["passed"] = bool(
        out["alert_ok"] and out["readme_match"]
        and out["standalone_fp32_pass"] and out["standalone_fp16_pass"]
        and out["common_vs_torch_days"] <= ATOL["fp32"]
        and out["pi_match_fp32"] and out["pi_match_fp16"]
    )
    return out


def main() -> int:
    from pest_configs import LONG_FILENAME, PESTS, SMOKE

    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-master", required=True, type=Path)
    ap.add_argument("--long-dir", required=True, type=Path)
    ap.add_argument("--pests", nargs="*", default=list(PESTS))
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(tempfile.gettempdir()) / "standalone_parity_cache")
    args = ap.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    rows, failures, skipped = [], [], []
    for pest in args.pests:
        smoke = SMOKE[pest]
        obs = args.long_dir / LONG_FILENAME[pest]
        if not obs.is_file():
            skipped.append({"pest": pest, "reason": f"LONG CSV not found: {obs}"})
            continue
        try:
            rows.append(run_pest(pest, smoke.site, smoke.year, args.daily_master,
                                 obs, args.cache_dir, smoke))
        except Exception as e:
            failures.append({"pest": pest, "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()})
            print(f"  FAIL {pest}: {type(e).__name__}: {str(e)[:140]}")

    print(f"\n{'pest':<19}{'torch':>10}{'common':>10}{'sa_fp32':>10}{'sa_fp16':>10}"
          f"{'fp32_d':>9}{'fp16_d':>9}{'README':>9}{'PI':>4}{'res':>6}")
    for r in rows:
        pi_ok = "Y" if (r["pi_match_fp32"] and r["pi_match_fp16"]) else "N"
        print(f"{r['pest']:<19}{r['mu_doy_torch']:>10.4f}"
              f"{r['mu_doy_common_tflite_fp32']:>10.4f}"
              f"{r['standalone_fp32_mu_doy']:>10.2f}{r['standalone_fp16_mu_doy']:>10.2f}"
              f"{r['standalone_fp32_days_vs_torch']:>9.2e}"
              f"{r['standalone_fp16_days_vs_torch']:>9.2e}"
              f"{r['readme_delta_days']:>9.4f}{pi_ok:>4}"
              f"{'PASS' if r['passed'] else 'FAIL':>6}")
    for s in skipped:
        print(f"{s['pest']:<19}  SKIPPED — {s['reason']}")

    ok = bool(rows) and all(r["passed"] for r in rows) and not failures
    print(f"\ntolerances: fp32 <= {ATOL['fp32']} d, fp16 <= {ATOL['fp16']} d")
    print(f"C. output parity: {'PASS' if ok else 'FAIL'} ({len(rows)}/{len(args.pests)})")
    (DIST.parent / "test_output_parity.json").write_text(
        json.dumps({"rows": rows, "failures": failures, "skipped": skipped}, indent=2)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
