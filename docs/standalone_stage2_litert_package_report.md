# Standalone Stage-2 LiteRT package — report

**Date:** 2026-07-15
**Branch:** `feature/tflite-conversion`
**Host:** MacBook, Apple M3 (arm64), macOS 26.5.1, Python 3.12.2
**Outcome:** ✅ 8/8 pests packaged and running with **no PyTorch, no TensorFlow, no XGBoost, no `.pt`**. Preprocessing is **bit-exact** against the API; output parity holds on all 8 real smoke cases; verified in a genuinely torch-free venv.

Predecessors: [`bph_stage2_tflite_conversion_report.md`](bph_stage2_tflite_conversion_report.md),
[`all_pests_stage2_tflite_conversion_report.md`](all_pests_stage2_tflite_conversion_report.md).

---

## 1. Why the existing `.tflite` alone could not run

The all-8 conversion produced working `.tflite` files, but they are **only the
mu head**. Three things lived outside the graph, so a `.tflite` on its own could
not answer a request:

1. **Normalization stats live inside the `.pt`.** `norm_mean`/`norm_std` are
   checkpoint keys, and `infer/ckpt.py:137-142` additionally forces the last 15
   (dispatch) channels to mean=0/std=1 so they bypass normalization. Without the
   checkpoint there was nothing to normalize with.
2. **The input tensor is not the weather.** The model wants a normalized
   `(1, 1, T, d_in)` block — impute → 7/14-day rollings → coords → phenology →
   `[base…, base__miss…]` → causal dispatch append → nowcast masking. All of that
   was in `infer/preprocess.py`, which imports torch.
3. **`mu` is not the answer.** `mu` is a 1-based season index; the API converts
   it (`mu + doy_start - 1`) and derives the 95% interval with a specific
   rounding rule. That logic was in `run_predict.py`.

Plus the per-pest contract (T, d_in, DOY range, alert channel index) only existed
inside the checkpoints.

This package moves all four out of the `.pt` and into data + torch-free code.

## 2. Metadata extracted from the checkpoints

Build-time only (`build_package.py`), one read per checkpoint, into
`models/<pest>/metadata.json` (schema `1.0`):

| group | fields |
|---|---|
| identity | `pest`, `model_family`, `checkpoint_variant`, `source_checkpoint{path, sha256, bytes}` |
| architecture | `T`, `d_in`, `d_model`, `n_head`, `n_layers`, `tstar_layers`, `doy_start`, `doy_end`, `alert_idx`, `nowcast_window`, `selected_offset` |
| mu / interval | `mu_mode`, `lead_min`, `lead_max`, `gaussian_sigma`, `mu_output_semantics`, `prediction_interval_rule{source, half_width_days, formula, note}` |
| channels | `feature_names` (exact order), `base_channels`, `miss_channels`, `dispatch_channels`, `norm_bypass_channel_indices`, `norm_bypass_mask`, `requires_site_coords`, `requires_phenology`, `input_daily_doy_coverage_required` |
| io | `inputs[]` / `outputs[]` with shape + dtype |
| files | `model_fp16`, `model_fp32`, `normalization` — each `{filename, bytes, sha256}` |
| policy | `api_policy{recommended_source, learned_output_status}` |

`mu_output_semantics` records explicitly that **mu is a 1-based season index,
not absolute DOY** (`DOY = mu + doy_start - 1`).

`norm_mean`/`norm_std` go to `normalization.npz` as float32 — the checkpoint's
exact dtype, so no precision is lost. The build asserts the npz round-trip is
value-identical, and `test_metadata.py` re-compares it against the checkpoint
with `np.array_equal`.

**Nothing is guessed.** `build_package.py` calls
`tflite_conversion/stage2/checkpoint.py::load_pest`, which validates each
checkpoint against the expected contract *and* against the real `state_dict`
tensor shapes (`in_proj.in_features`, `self_attn.num_heads/embed_dim`, layer
counts), then re-derives the feature-block layout and the norm-bypass set from
the arrays themselves and fails the build on any mismatch. No mismatch was found.

| pest | d_in | T | DOY | alert_idx | needs site+pheno |
|---|---:|---:|---|---:|:--:|
| BPH | 27 | 131 | 140–270 | 12 | no |
| the other 7 | 45 | 241 | 60–300 | 30 | yes |

## 3. Making preprocessing standalone

`runtime/preprocessing.py` reimplements `build_real_input` in pure
NumPy/Pandas, step for step, each citing its upstream counterpart: impute
(interpolate→ffill→bfill, precipitation→0, GDD **not** imputed) → rolling
features → coords → phenology merge+ffill → `[base…, base__miss…]` block →
causal dispatch append → `_mask_to_recent_window` (including the `1::2` stride)
→ normalize with the ckpt stats and the `std<1e-6→1.0` guard.

Three deliberate departures:

1. **No master-CSV slicing.** The caller passes one site-year frame (the
   upstream `_daily_year_from_frame` path). A multi-site frame is rejected.
2. **No LONG observation file.** Site coordinates and phenology become explicit
   request fields — see §4.
3. **No torch.** Tensors are float32 ndarrays.

One safeguard the original does not have: the rolling features are computed over
the whole year *before* the season slice, so a frame starting at `doy_start`
would silently change the first season days. The runtime therefore **requires
contiguous DOY 1..`doy_end`** and errors otherwise.

### The finding that changed the input contract

The task specified the input as *daily weather + pest + alert_tstar + dispatch
features*. That is sufficient for **BPH only**. The other 7 pests' base channels
include `좌표-위도`, `좌표-경도`, `days_since_growing_start`,
`days_until_growing_end`, `is_growing` — which upstream reads from the per-pest
LONG observation CSV, not from weather.

Phenology is **step data**: for the WBPH smoke case there are 8 observation rows
for a 241-day season, merged by DOY and forward-filled. It is not a function of
DOY (at doy 161..167 the value stays at doy 160's), so it cannot be recomputed
from weather or from `growing_start_doy` alone.

By the same principle the task already applies to `alert_tstar` and the dispatch
features, these became explicit inputs (`site`, `phenology`) — required only for
the pests whose metadata says so, and never defaulted to 0.

## 4. Runtime dependencies

`requirements-runtime.txt` — exactly three direct pins, verified installable and
importable in a clean venv:

```
ai-edge-litert==2.1.5
numpy==2.5.1
pandas==3.0.3
```

`ai-edge-litert` is Google's standalone LiteRT (formerly TF Lite) interpreter
wheel; `pip show` confirms it requires only `backports.strenum, flatbuffers,
numpy, protobuf, tqdm, typing-extensions` — **it does not depend on the
`tensorflow` package**. Full closure in the test venv is 11 packages:

```
ai-edge-litert, backports.strenum, flatbuffers, numpy, pandas, pip,
protobuf, python-dateutil, six, tqdm, typing_extensions
```

Build-side (`requirements-build.txt`) keeps torch 2.12.1 + PyYAML 6.0.3 (+ the
litert-torch chain for `--export-missing-tflite`). None of it reaches the
package.

## 5. Torch-free verification (D)

Run in a venv created fresh and installed **only** from
`requirements-runtime.txt`. The test refuses to pass if torch is importable, so
a stray torch in `.venv-tflite` or conda cannot fake it.

| check | result |
|---|---|
| `import torch` / `tensorflow` / `xgboost` | **not importable** ✅ |
| `.pt`/`.pth`/`.ckpt`/`.onnx` in package | **none** (16 `.tflite` present) ✅ |
| 8/8 metadata load + sha256 verify | PASS |
| synthetic inference, 8/8 pests, fp16 + fp32 | PASS |
| error contract, 6/6 | PASS |
| CLI `--list-pests` | PASS |
| **real smoke inference via CLI (BPH)** | PASS |

Real smoke run inside the torch-free venv:

```
$ .venv-runtime-test/bin/python dist/stage2_litert/predict.py \
    --pest BPH --daily-csv bph_daily_2004.csv --dispatch-json bph_dispatch.json --variant fp16
{ "pest": "BPH", "backend": "tflite_fp16", "alert_tstar": 176,
  "mu_doy": 234.33, "prediction_interval_95": [225, 244], ... }
```

## 6. Parity results

### B. Preprocessing — bit-exact, 8/8

Standalone vs the API's `build_real_input`, same real inputs:

| pest | tensor | dtype | max abs | mean abs | bit-exact |
|---|---|---|---:|---:|:--:|
| BPH | (1,1,131,27) | float32 | 0.000e+00 | 0.000e+00 | ✅ |
| WBPH | (1,1,241,45) | float32 | 0.000e+00 | 0.000e+00 | ✅ |
| bacterial_blight | (1,1,241,45) | float32 | 0.000e+00 | 0.000e+00 | ✅ |
| blast | (1,1,241,45) | float32 | 0.000e+00 | 0.000e+00 | ✅ |
| brown_spot | (1,1,241,45) | float32 | 0.000e+00 | 0.000e+00 | ✅ |
| rice_stem_borer_1 | (1,1,241,45) | float32 | 0.000e+00 | 0.000e+00 | ✅ |
| rice_stem_borer_2 | (1,1,241,45) | float32 | 0.000e+00 | 0.000e+00 | ✅ |
| sheath_blight | (1,1,241,45) | float32 | 0.000e+00 | 0.000e+00 | ✅ |

Shape, dtype, `tstar_season_index` and `alert_tstar` all match. Per-channel max
error is 0 for every channel, so no explanation of residual difference is needed.

The test feeds the standalone runtime the same per-site slice the API cached,
so the comparison isolates preprocessing rather than site selection.

### C. Model output — 8/8, four paths

`mu_doy` on the real smoke cases: (1) original PyTorch, (2) the existing common
TFLite check, (3) standalone FP32, (4) standalone FP16.

| pest | torch | common fp32 | standalone fp32 | standalone fp16 | fp32 Δd | fp16 Δd | README Δ | PI |
|---|---:|---:|---:|---:|---:|---:|---:|:--:|
| BPH | 234.3240 | 234.3240 | 234.32 | 234.33 | 2.0e-05 | 1.2e-03 | 0.0040 | ✅ |
| WBPH | 245.8158 | 245.8158 | 245.82 | 245.82 | 2.6e-05 | 2.6e-05 | 0.0042 | ✅ |
| bacterial_blight | 268.3161 | 268.3161 | 268.32 | 268.32 | 2.9e-05 | 6.3e-04 | 0.0039 | ✅ |
| blast | 198.0840 | 198.0841 | 198.08 | 198.08 | 5.5e-05 | 2.6e-04 | 0.0040 | ✅ |
| brown_spot | 224.9314 | 224.9314 | 224.93 | 224.93 | 3.4e-05 | 1.3e-03 | 0.0014 | ✅ |
| rice_stem_borer_1 | 184.1720 | 184.1720 | 184.17 | 184.17 | 2.9e-06 | 3.6e-03 | 0.0020 | ✅ |
| rice_stem_borer_2 | 260.9706 | 260.9706 | 260.97 | 260.97 | 4.2e-05 | 4.2e-05 | 0.0006 | ✅ |
| sheath_blight | 192.7225 | 192.7225 | 192.72 | 192.72 | 1.2e-05 | 8.8e-05 | 0.0025 | ✅ |

Tolerances met with large margin: FP32 worst 5.5e-05 d (bound 1e-3, 18× inside),
FP16 worst 3.6e-03 d (bound 0.1, 28× inside). Every alert DOY matches README §10,
and every `mu_doy` matches its record to within that document's 2-decimal
rounding.

**Prediction interval matches on all 8**, for both variants, against the interval
computed by the API's own rule on the torch mu. The rule was read from
`run_predict.py:490-498`, not guessed:

```python
half   = round(1.96 * sigma, 1)              # 9.8 for sigma=5
mu_doy = round(mu_doy_temporal, 2)
pi_95  = [int(round(mu_doy_temporal - half)), int(round(mu_doy_temporal + half))]
```

Two details replicated deliberately: the interval derives from the **unrounded**
mu (not the reported rounded `mu_doy`), and Python's `round()` is
round-half-to-even, so plain floats are used rather than numpy scalars.

### A. Metadata — 8/8

feature_names length == d_in, norm shapes + exact values vs checkpoint, bypass
mask == derived-from-arrays, TFLite input shape == metadata, checkpoint config ==
metadata, all sha256 match.

## 7. Package size

**Total 8,310,385 B (8.31 MB)** for 8 pests × 2 variants.

| pest | total | fp16 | fp32 | normalization |
|---|---:|---:|---:|---:|
| BPH | 990,646 | 380,232 | 604,596 | 736 |
| WBPH | 1,039,477 | 403,328 | 629,172 | 880 |
| bacterial_blight | 1,039,553 | 403,328 | 629,172 | 880 |
| blast | 1,039,487 | 403,328 | 629,172 | 880 |
| brown_spot | 1,039,518 | 403,328 | 629,172 | 880 |
| rice_stem_borer_1 | 1,039,559 | 403,328 | 629,172 | 880 |
| rice_stem_borer_2 | 1,039,560 | 403,328 | 629,172 | 880 |
| sheath_blight | 1,039,536 | 403,328 | 629,172 | 880 |

Models are 8,267,336 B; runtime code + manifest + README are the remaining ~43 KB.
Shipping FP16 only would roughly halve it (≈3.2 MB).

## 8. Remaining limitations

- **Stage 1 is not included** — `alert_tstar` and the 14 dispatch features are
  inputs. (Independently: the shipped Stage-1 XGBoost checkpoints SIGSEGV under
  xgboost ≥ 2 because the booster is a legacy binary blob, so the live gate can't
  run here at all. Unrelated to this package but blocking for the full pipeline.)
- **Site coords + phenology are inputs** for the 7 non-BPH pests (§3).
- **No weather ingestion** — the caller supplies the daily frame, which must
  cover DOY 1..`doy_end` contiguously.
- **B=K=1 only.** The graphs are fixed at a single sample; batching needs a
  re-export.
- **FP16's reported `mu_doy` can differ from PyTorch in the 2nd decimal**
  (BPH 234.33 vs 234.32) purely from rounding at the boundary; the prediction
  interval — what the API acts on — is identical. Use FP32 if exact agreement of
  the printed `mu_doy` matters.
- **Climatology and the fallback policy are not implemented.** Only BPH and WBPH
  currently serve the learned output as the API's final answer; the other six
  return climatology upstream, with the Transformer marked `experimental`. Each
  metadata records this under `api_policy`, but the package always returns the
  learned value — the caller applies policy.
- Real-data fixtures are not committed (not redistributable); tests take paths.

## 9. Connecting a weather API later

1. Fetch daily weather for `(site, year)` and map it to the schema in
   `README.md` (Korean column names; `GDD10_since_gs` must be supplied or
   derived — upstream does **not** impute it).
2. Guarantee contiguous DOY 1..`doy_end` (270 BPH / 300 others). The runtime
   raises `PreprocessError` listing the gaps, which is a usable health check.
3. Hand the frame to `predict_stage2(daily_data=df, ...)` — a DataFrame is
   accepted directly, no file needed.
4. Provide `site`/`phenology` for the 7 non-BPH pests from the same source the
   LONG table came from.
5. Cache per site-year: preprocessing is deterministic and the interpreter is
   reused across calls.

## 10. Connecting Stage 1 and the full API later

1. **Fix the Stage-1 xgboost pin first** — pin the exact 1.x that wrote the
   checkpoints, or re-serialize the boosters with `Booster.save_model()` on the
   GPU server. Nothing Stage-1 runs today on a fresh env.
2. Have Stage-1 emit the dispatch JSON this package already accepts: it produces
   exactly `alert_tstar` + the 14 features (`infer/stage1.py::get_alert_single_sy`
   returns `alert_tstar_doy` + `dispatch_features`).
3. Add a backend switch in `run_predict.py` (`--stage2-backend torch|litert`)
   rather than replacing the torch path; keep torch as the reference default.
4. Re-add the climatology + `fallback_policy.yaml` layer around this package, or
   keep it in the API and call the package only for the learned branch.
5. Gate on parity in CI: `test_preprocessing_parity.py` and `test_output_parity.py`
   both exit non-zero on regression.

## 11. Swapping in a newer DN model

The package is regenerated, never hand-edited:

1. Land the new checkpoint at
   `api_handoff_transformer/assets/stage2/<pest>/lead_v3_final_checkpoint_run4.pt`
   (or update the path in `tflite_conversion/stage2/checkpoint.py`).
2. If the architecture or channel layout changed, update
   `tflite_conversion/stage2/pest_configs.py::EXPECTED` — the loader **fails
   loudly** on drift rather than converting something wrong.
3. Re-export: `cd tflite_conversion/stage2 && python export_all.py --pests <pest>`.
4. Rebuild: `python tflite_conversion/standalone_stage2/build_package.py --clean`
   — metadata, normalization, sha256 and manifest are all re-derived from the new
   checkpoint.
5. Re-run A/B/C/D. Preprocessing parity must stay bit-exact; if the new model
   changes the feature set, `runtime/preprocessing.py` needs the matching change
   and the parity test is what proves it.
6. Bump `METADATA_SCHEMA_VERSION` in `runtime/schema.py` if the metadata shape
   changes — the runtime refuses to load a mismatched schema version.

## 12. Repo hygiene

- No API file, checkpoint, or `fallback_policy.yaml` was modified. All 8
  checkpoints were verified byte-identical (SHA-256) to the shipped zip after
  the work.
- The existing `tflite_conversion/stage2/` artifacts are untouched; this package
  copies them.
- `.gitignore` gained three entries, all build/test output:
  `tflite_conversion/standalone_stage2/dist/`, `.venv-runtime-test/`,
  `tflite_conversion/standalone_stage2/tests/fixtures_real/`. Verified with
  `git check-ignore`, and `git ls-files -i -c` confirms no tracked file became
  ignored.
- Nothing staged, committed or pushed.
