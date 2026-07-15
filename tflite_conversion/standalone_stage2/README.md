# Standalone Stage-2 LiteRT package

Runs all 8 deployed Stage-2 pest-timing models with **no PyTorch, no TensorFlow,
no XGBoost and no `.pt` checkpoints**. Runtime dependencies are
`ai-edge-litert`, `numpy`, `pandas` — 11 packages total.

```
daily weather CSV/DataFrame + pest + alert_tstar + dispatch features
  → standalone preprocessing + normalization
  → FP16 TFLite (default; FP32 optional)
  → mu_doy + 95% prediction interval
```

## Scope — what is NOT included

| out of scope | you must supply it |
|---|---|
| **Stage 1** (XGBoost gate) | `alert_tstar` + all 14 dispatch features |
| **Weather API ingestion** | the daily weather CSV/DataFrame |
| **Existing API wiring** | — this package is standalone |

**7 of the 8 pests additionally need site coordinates and phenology.** Only BPH
runs from weather alone. See "Per-pest input differences".

## Build (needs torch — build only)

The build reads the `.pt` checkpoints once; nothing torch-related reaches the
package. `build_package.py` fails the build if it does.

```bash
cd /Users/doyoung-gil/cropscience
# prerequisite: the API package unzipped at the repo root, and the FP16/FP32
# .tflite already exported by tflite_conversion/stage2/export_all.py
./.venv-tflite/bin/python tflite_conversion/standalone_stage2/build_package.py --clean
# if the .tflite are missing, let the build regenerate them:
#   ... build_package.py --clean --export-missing-tflite
```

Output: `tflite_conversion/standalone_stage2/dist/stage2_litert/` (8.31 MB).
Build deps: `requirements-build.txt`.

## Install the runtime (no `.pt`, no torch)

```bash
python3 -m venv .venv-runtime
./.venv-runtime/bin/python -m pip install -r dist/stage2_litert/requirements-runtime.txt
```

Installs exactly: `ai-edge-litert==2.1.5`, `numpy==2.5.1`, `pandas==3.0.3`
(+ 8 transitive). Verified torch-free on macOS 26.5.1 / Apple M3 / Python 3.12.2.

## Python API

```python
import sys; sys.path.insert(0, "dist/stage2_litert")
from runtime import predict_stage2

out = predict_stage2(
    pest="BPH",
    daily_data="bph_daily_2004.csv",   # path or DataFrame
    alert_tstar=176,
    dispatch_features={               # all 14 required; nothing is defaulted
        "alert_tstar": 176.0, "with_history": 0.0, "dispatch_branch": "D",
        "A_score_at_alert": 0.6559, "D_score_at_alert": 0.6433,
        "score_margin": -0.0126, "dispatch_score_at_alert": 0.6433,
        "dispatch_tau_used": 0.6, "score_over_tau_margin": 0.0433,
        "recent_14d_mean_score": 0.4632, "recent_28d_mean_score": 0.4632,
        "score_above_tau_streak": 3.0, "score_rolling_slope_14d": 0.0551,
        "p_mean_so_far_at_alert": 0.4632,
    },
    variant="fp16",                   # or "fp32"
    year=2004,
    # 7 non-BPH pests also need:
    # site={"lat": 35.5, "lon": 128.5},
    # phenology=[{"obs_doy": 150, "days_since_growing_start": -6.0,
    #             "days_until_growing_end": 127.0, "is_growing": 0.0}, ...],
)
print(out["mu_doy"], out["prediction_interval_95"])
```

## CLI

```bash
python dist/stage2_litert/predict.py \
  --pest BPH \
  --daily-csv sample_daily.csv \
  --dispatch-json sample_dispatch.json \
  --variant fp16

python dist/stage2_litert/predict.py --list-pests
```

Output:

```json
{
  "pest": "BPH",
  "backend": "tflite_fp16",
  "year": 2004,
  "alert_tstar": 176,
  "mu_rel_season_index": 95.3252,
  "mu_doy": 234.33,
  "prediction_interval_95": [225, 244],
  "sigma_days": 5.0,
  "tstar_season_index": 67,
  "selected_offset": 30,
  "input_shape": [1, 1, 131, 27],
  "model_sha256": "a63c718b...",
  "metadata_schema_version": "1.0"
}
```

## Input: daily weather CSV schema

Korean column names, exactly as the API expects (BOM-safe, `utf-8-sig`):

| column | notes |
|---|---|
| `일시` | date, parseable by `pd.to_datetime` |
| `일강수량(mm)` | daily precipitation |
| `최고기온(°C)`, `최저기온(°C)`, `평균기온(°C)` | temperatures |
| `평균 풍속(m/s)`, `최대 풍속(m/s)` | wind |
| `평균 상대습도(%)` | humidity |
| `합계 일조시간(h)`, `합계 일사량(MJ/m2)` | sunshine, radiation |
| `GDD10_since_gs` | growing-degree-days; **not** imputed (matches upstream) |
| `지점ID` | optional; if present it must be a single site |

**Coverage requirement: DOY 1..`doy_end` of the target year, contiguous, no gaps
and no duplicates.** The 7/14-day rolling features are computed over the whole
year *before* the season is sliced, so a frame that starts at `doy_start` would
silently produce different features at the season's first days. The runtime
rejects that instead of guessing. `doy_end` is 270 for BPH, 300 for the rest.

If the frame spans multiple years, pass `year`.

## Input: dispatch JSON schema

```json
{
  "pest": "BPH",
  "year": 2004,
  "alert_tstar": 176,
  "dispatch_features": {
    "alert_tstar": 176.0,
    "with_history": 0.0,
    "dispatch_branch": "D",
    "A_score_at_alert": 0.6559762778214119,
    "D_score_at_alert": 0.643326509919207,
    "score_margin": -0.012649767902204845,
    "dispatch_score_at_alert": 0.643326509919207,
    "dispatch_tau_used": 0.6,
    "score_over_tau_margin": 0.04332650991920706,
    "recent_14d_mean_score": 0.4632082032263717,
    "recent_28d_mean_score": 0.4632082032263717,
    "score_above_tau_streak": 3,
    "score_rolling_slope_14d": 0.05507638107993647,
    "p_mean_so_far_at_alert": 0.4632082032263717
  },
  "site": {"lat": 35.5, "lon": 128.5},
  "phenology": [
    {"obs_doy": 150, "days_since_growing_start": -6.0,
     "days_until_growing_end": 127.0, "is_growing": 0.0}
  ]
}
```

All 14 `dispatch_features` are **required**. Missing values raise an error — they
are never defaulted to 0. `dispatch_branch` accepts `"D"`/`"A"` or a number.
`site` and `phenology` are required only for the pests that use them.

`phenology` records are LONG-style **step data**: a handful of observed DOYs
(often <10 for a 241-day season), merged by DOY and forward-filled, exactly as
upstream does. They are not a function of DOY and cannot be derived from weather,
which is why they are an input.

## Per-pest input differences

| pest | T | d_in | DOY range | alert_idx | needs site+phenology |
|---|---:|---:|---|---:|:--:|
| BPH | 131 | 27 | 140–270 | 12 | **no** |
| WBPH | 241 | 45 | 60–300 | 30 | yes |
| bacterial_blight | 241 | 45 | 60–300 | 30 | yes |
| blast | 241 | 45 | 60–300 | 30 | yes |
| brown_spot | 241 | 45 | 60–300 | 30 | yes |
| rice_stem_borer_1 | 241 | 45 | 60–300 | 30 | yes |
| rice_stem_borer_2 | 241 | 45 | 60–300 | 30 | yes |
| sheath_blight | 241 | 45 | 60–300 | 30 | yes |

BPH's 6 base channels are raw weather; the other 7 use 15 base channels
(rolling aggregates + lat/lon + 3 phenology). Every value above is read from
`models/<pest>/metadata.json`, which the build derives from the checkpoint.

## FP16 (default) vs FP32

FP16 is the default: ~1.56× smaller (403 KB vs 629 KB per pest) at
indistinguishable speed, with a worst real-data error of 3.6e-03 days (~5 min)
against PyTorch — far below the 1-day rounding of the prediction interval.

Choose FP32 (`variant="fp32"`, `--variant fp32`) when you want the closest
possible match to the PyTorch reference (worst real-data error 5.5e-05 days).

Caveat: because the reported `mu_doy` is rounded to 2 decimals, FP16 can differ
from PyTorch in the last decimal (e.g. BPH 234.33 vs 234.32) while the
`prediction_interval_95` is identical. The interval is what the API acts on.

## Errors

The runtime raises rather than filling or falling back. `SchemaError` and
`PreprocessError` both exit the CLI with code 2 and an actionable message:

| condition | error |
|---|---|
| missing/extra dispatch feature | `SchemaError` |
| `alert_tstar` outside the pest's season | `SchemaError` |
| unknown pest | `SchemaError` |
| model/normalization sha256 mismatch | `SchemaError` |
| `site`/`phenology` missing for a pest that needs them | `SchemaError` |
| missing DOY in the required range | `PreprocessError` |
| duplicate DOY | `PreprocessError` |
| missing weather column | `PreprocessError` |
| multi-site or multi-year frame (unresolvable) | `PreprocessError` |
| non-finite value after normalization | `PreprocessError` |

## Tests

```bash
cd tflite_conversion/standalone_stage2
PY=../../.venv-tflite/bin/python           # build-side (needs torch)
DAILY_MASTER=/path/to/daily_weather.csv
LONG_DIR=/path/to/long_by_pest

$PY tests/test_metadata.py
$PY tests/test_preprocessing_parity.py --daily-master "$DAILY_MASTER" --long-dir "$LONG_DIR"
$PY tests/test_output_parity.py         --daily-master "$DAILY_MASTER" --long-dir "$LONG_DIR"

# runtime-side (must be a torch-free venv)
../../.venv-runtime-test/bin/python tests/test_runtime_synthetic.py
../../.venv-runtime-test/bin/python tests/test_no_torch_dependency.py
```

Results (2026-07-15): metadata 8/8, preprocessing parity **8/8 bit-exact**,
output parity 8/8, torch-free runtime PASS.

Full write-up: [`docs/standalone_stage2_litert_package_report.md`](../../docs/standalone_stage2_litert_package_report.md)

## Notes

- `dist/`, `.venv-runtime-test/` and `tests/fixtures_real/` are git-ignored:
  build output, a venv, and fixtures derived from non-redistributable real data.
- Running the package creates `runtime/__pycache__` inside `dist/`; that is
  normal CPython behaviour, not shipped content. The build purges it and fails
  if it is present at build time.
- Only **BPH and WBPH** currently serve the learned output as the API's final
  answer; the other six return climatology with the Transformer marked
  `experimental` (`configs/fallback_policy.yaml`). Each `metadata.json` records
  this under `api_policy`.
