# Lightweight portable pest-timing API

Same behaviour and output contract as the deployed API, with **no PyTorch, no
TensorFlow, no scikit-learn, and no `.pt` checkpoints**.

```
daily weather CSV + LONG observations + request
  → Stage 1: XGBoost JSON (xgboost.Booster) → alert_tstar + 14 dispatch features
  → Stage 2: LiteRT FP16 → mu_doy + 95% prediction interval
  → fallback policy (unchanged) → response.json / predictions.csv / run_log.txt
```

| | measured |
|---|---:|
| package assets (FP16 only) | **29.47 MB** |
| runtime venv | **259.48 MB** |
| **package + runtime** | **288.26 MB** |
| production archive (`.tar.gz`) | **8.6 MB** |

Model generation: **`lead_v3_final` (the deployed models)**. The newer DN model is
deliberately excluded.

## Install (no `.pt`, no torch)

```bash
cd /Users/doyoung-gil/cropscience
python3 -m venv .venv-lightweight-api
./.venv-lightweight-api/bin/python -m pip install -r api_handoff_litert_portable/requirements-runtime.txt
```

Installs 14 packages: `xgboost, ai-edge-litert, numpy, pandas, PyYAML` + 9
transitive. No torch, no tensorflow, no scikit-learn.

## Build the package (assets are not committed)

```bash
# needs the reference API unzipped + the Stage-2 LiteRT models built
./.venv-tflite/bin/python api_handoff_litert_portable/build_package.py --clean
./.venv-tflite/bin/python api_handoff_litert_portable/build_package.py --archive        # dist/*.tar.gz (FP16)
./.venv-tflite/bin/python api_handoff_litert_portable/build_package.py --include-fp32   # validation build
```

The build fails if any `.pt`/`.pth`, torch/tensorflow/sklearn import, or (in a
production build) an FP32 model reaches the package.

## Run

```bash
python run_predict.py --input-dir IN --output-dir OUT [--stage2-variant fp16|fp32]
```

`--stage2-variant` defaults to **fp16**. FP32 is available only in a build made
with `--include-fp32`.

### Input directory

| file | notes |
|---|---|
| `request.json` | `{"pest","site_id","year"[,"alert_tstar_doy","include_diagnostics"]}` |
| `daily_weather.csv` | Korean schema; may hold many sites (filtered per request) |
| `long_observation.csv` | LONG rows for the site (Layout A) |
| `LONG_by_pest/RICE_LONG_<pest>.csv` | Layout B, used only if Layout A is absent |

`pest` is **case-sensitive** (matches the deployed single-mode contract).

### Daily CSV schema

`일시`, `일강수량(mm)`, `최고기온(°C)`, `최저기온(°C)`, `평균기온(°C)`,
`평균 풍속(m/s)`, `최대 풍속(m/s)`, `평균 상대습도(%)`, `합계 일조시간(h)`,
`합계 일사량(MJ/m2)`, `GDD10_since_gs` (+ optional `지점ID`).

`GDD10_since_gs` is **not** imputed (matches upstream). The season must be
complete for the pest's DOY window, with no duplicate dates.

### Example

```bash
python run_predict.py --input-dir /tmp/in --output-dir /tmp/out
```
```json
{
  "pest": "BPH", "site_id": "33210_56298", "year": 2004,
  "model_version": "v0-transformer-real-ckpt",
  "stage1": {"alert_fired": true, "alert_tstar_doy": 176,
             "gate_method": "D_history", "alert_source": "stage1_live",
             "wiring_status": "stage1_xgboost_live"},
  "stage2": {"learned_stage2": {"mu_doy": 234.33,
                                "pi_95": {"lower_doy": 225, "upper_doy": 244, "sigma_days": 5.0},
                                "selected_offset": 30, "output_status": "main",
                                "model_kind": "lead_v3"},
             "climatology": {"mu_doy": 147.55, "pi_95": {...}, "variant": "mean_mid"},
             "recommended_source": "learned_stage2"},
  "final_prediction": {"source": "learned_stage2", "mu_doy": 234.33, ...},
  "backends": {"stage1_backend": "xgboost_json", "stage2_backend": "litert_fp16"}
}
```

`backends` is the only addition to the deployed schema — additive optional
metadata. No existing field is removed, renamed or re-typed.

### Batch

**Not implemented in this package.** The deployed `infer/batch.py` needs the
880-representative-site CSV and emits a different summary schema; it is out of
scope for this step. Loop `run_predict.py` per site, or see the report §"next
steps" for wiring it.

## What you must still supply

| need | why |
|---|---|
| daily weather CSV | no weather API is wired yet (`WeatherProvider` is the seam) |
| LONG observations | supplies coordinates + phenology + Stage-1 history fallback |
| representative-site ↔ ASOS mapping | **not implemented** — 880 sites vs 105 ASOS stations is unresolved |

7 of 8 pests need site coordinates and phenology; only BPH runs from weather
alone. These come from the LONG file via `LongObsProvider`.

## Errors

Nothing is ever zero-filled. Exit codes mirror the deployed API:

| exit | when |
|---:|---|
| 0 | success — **and any Stage-2 failure** (the deployed code returns 0 either way; check `stage2.learned_stage2 == null` or `diagnostics.transformer_error`) |
| 1 | bad request / missing config |
| 2 | missing input files |

Common messages: missing dispatch feature; alert outside the season; missing DOY
in the required range; duplicate DOY; missing weather column; `site_history.json`
has no entry for the site-year.

## FP16 policy

FP16 is the production default: ~1.56× smaller than FP32 at indistinguishable
speed, worst real-data mu error 3.6e-03 days vs PyTorch. FP32 is a validation /
fallback option and is excluded from production builds. FP16's reported `mu_doy`
can differ in the 2nd decimal (BPH 234.33 vs 234.32); the prediction interval —
what the API acts on — is identical.

## Tests

```bash
PY=../.venv-tflite/bin/python                  # build-side (has torch, for reference comparisons)
RPY=../.venv-lightweight-api-test/bin/python   # runtime-only (no torch)
DAILY_MASTER=/path/to/daily_weather.csv
LONG_DIR=/path/to/long_by_pest

$PY  tests/test_stage1_portable.py                                     # incl. memory-layout regression
$RPY tests/test_no_heavy_dependencies.py                               # must run in the torch-free venv
$RPY tests/test_smoke_cases.py --daily-master "$DAILY_MASTER" --long-dir "$LONG_DIR"
```

Results (2026-07-16): Stage-1 6/6 · no-heavy-deps PASS · 8-pest smoke **8/8**
(alerts computed live; `mu_doy` vs golden reference).

## Rebuild after a model change

1. Re-export Stage-1 portable JSON (`stage1_xgboost_migration/`) and/or Stage-2
   LiteRT (`tflite_conversion/standalone_stage2/build_package.py --clean`).
2. `python build_package.py --clean --archive`.
3. Re-run the three test commands above; all must pass.

Full write-up: [`docs/lightweight_api_integration_report.md`](../docs/lightweight_api_integration_report.md)
Design/analysis: [`docs/lightweight_api_integration_plan.md`](../docs/lightweight_api_integration_plan.md)
