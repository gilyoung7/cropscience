# Lightweight API integration report

**Date:** 2026-07-16 · **Branch:** `feature/tflite-conversion` · **Host:** MacBook, Apple M3 (arm64), macOS 26.5.1, Python 3.12.2
**Outcome:** ✅ `api_handoff_litert_portable/` runs all 8 pests end-to-end with **no PyTorch, no TensorFlow, no scikit-learn, no `.pt`** — **8/8 alerts computed live match the golden reference**, output contract preserved.
**Scope:** deployed `lead_v3_final` only. The newer DN model is excluded by design.

Companion: [`lightweight_api_integration_plan.md`](lightweight_api_integration_plan.md) (call-flow trace, dead-field audit, layout analysis).

---

## 1. Why the old environment was heavy

The deployed API needs the full training stack purely to *read* its assets:

| dependency | why it was there | needed now? |
|---|---|---|
| **torch** | Stage-2 `.pt` is a torch checkpoint; `HierarchicalCausalHazardTransformer` is an `nn.Module`. Stage-1 `.pt` also used torch **only as a pickle container**. | **no** — Stage-2 is LiteRT; Stage-1 is `model.json` |
| **scikit-learn** | Stage-1 `.pt` unpickles an `XGBClassifier` (a sklearn estimator) | **no** — `xgboost.Booster` loads `model.json` directly |
| xgboost | Stage-1 gate | yes |
| pandas / numpy | preprocessing | yes (see §6) |

So ~9–10 GB of ML runtime existed to load a 164k-parameter transformer and 16
gradient-boosted trees.

**Reduction claim, stated honestly:** the ~9–10 GB figure is **the size you
reported for the existing environment**; I could not find that environment on
this machine and did **not** measure it. The closest measured comparison is this
repo's dev env `.venv-tflite` (torch + litert-torch + jax + transformers), which
is **1,800.32 MB**. Against the reported 9–10 GB, 288.26 MB is a ~97% reduction;
against the measured 1.8 GB dev env, ~84%. Both framings are given rather than
quoting a single unverified number.

---

## 2. Stage-1: `.pt` → portable JSON

Assets: `model.json` + `calibration.json` + `metadata.json` per pest/branch (16),
`gate.json` per pest (8), plus `site_history.json` copied from the reference API
(the migration flagged it as an external asset it had not copied).

**`xgboost.Booster`, not `XGBClassifier`** — adopted only after proving equality:

| check | result |
|---|---|
| Booster.predict vs XGBClassifier.predict_proba[:,1] | **bit-exact, max\|diff\| = 0** |
| coverage | **48/48** = 16 models × {C-contiguous, F-contiguous, strided view} |

This drops scikit-learn entirely **and fixes a real breakage**:
`XGBClassifier.load_model()` raises `TypeError: _estimator_type undefined` under
xgboost 2.1.4 + scikit-learn 1.9.0, so the existing
`stage1_xgboost_migration/portable_stage1.py` (which uses XGBClassifier) cannot
load these models on a current environment. The new package does not have that
failure mode.

Gate parameters come from the frozen `gate.json` (the values
`resolve_gate_params` reads out of `group_tau_hybrid_summary.json`), **not** from
`stage1_selected_gates.yaml`, whose k/τ have drifted for two pests —
bacterial_blight (yaml k=1/τ=0.575 vs **k=3/τ=0.525**) and rice_stem_borer_1
(yaml k=2/τ=0.575 vs **k=3/τ=0.55**). `tests/test_stage1_portable.py` asserts the
non-drifted values are the ones shipped.

---

## 3. Stage-2: `.pt` → LiteRT FP16

Reuses the standalone Stage-2 runtime (metadata, `normalization.npz`, the tensor
builder already proven bit-exact against `build_real_input` on 8/8 pests). FP16
is the production default; FP32 exists behind `--stage2-variant fp32` and
`build_package.py --include-fp32`, and is excluded from production builds.

`mu_doy` and the prediction interval are ported exactly from
`run_predict.py:490-498`, including two details that are easy to get wrong: the
interval derives from the **unrounded** mu, and Python's `round()` is
round-half-to-even (so plain floats are used, never numpy scalars).

---

## 4. Array memory layout — the subtle one

The deployed API builds Stage-1 base X as `X_df.to_numpy(dtype=np.float32)`
(stage1.py:720). **Measured** (pandas 3.0.3 / numpy 2.5.1):

| boundary | dtype | strides | C | F |
|---|---|---|:--:|:--:|
| base X (`to_numpy`) — **A branch** | float32 | (4, 524) | 0 | **1** |
| after `append_history` (`np.concatenate`) — **D branch** | float32 | (92, 4) | **1** | 0 |
| nowcast window — A branch | float32 | (4, 524) | 0 | 0 |
| nowcast window — D branch | float32 | (92, 4) | 1 | 0 |
| tabular features (`np.stack`) | float32 | C-order | 1 | 0 |

So the layout **legitimately differs per branch**. It is not cosmetic — float32
reduction order depends on it:

```
mean : max|C − F| = 2.98e-08      std : max|C − F| = 5.96e-08
min / max : identical (order-independent)
```

`_build_tabular` reduces with mean/std/slope, so an `np.ascontiguousarray`
"cleanup" perturbs features by ~1e-8 — enough to flip a probability across τ and
move the alert DOY.

**The deployed layout is the source of truth and is reproduced exactly.** The
package builds base X through pandas `to_numpy` (F-order) and appends history via
`np.concatenate` (C-order), and never normalizes layout.
`tests/test_stage1_portable.py` pins dtype/shape/strides/flags at each boundary
for all 8 pests × both branches, **and asserts that forcing C-order changes the
features** (measured `max|F−C| = 2.384e-07`) — if that assertion ever reports "no
difference", the guard has gone vacuous and must be investigated, not relaxed.

---

## 5. API integration structure

```
api_handoff_litert_portable/
  run_predict.py          CLI + response/CSV/log writers (contract-compatible)
  infer/
    paths.py              asset locations, VALID_PESTS, MODEL_VERSION
    schemas.py            request validation + response assembly (exact key order)
    stage1_features.py    layout-preserving base X, history, nowcast, tabular
    stage1_portable.py    Booster load, temperature, gate, dispatch features
    preprocessing.py      daily → rolling → coords → phenology (shared)
    stage2_litert.py      FP16/FP32 LiteRT + mu_doy + PI
    fallback.py           climatology CSV + policy (unchanged)
    providers.py          WeatherProvider / SiteMetadataProvider / PhenologyProvider
    _stage2_*.py          vendored, already-proven Stage-2 tensor builder
  assets/{stage1,stage2,configs,climatology}   (built, not committed)
  tests/  build_package.py  requirements-{runtime,test}.txt  manifest.json
```

`providers.py` is the seam for the weather API: inference asks a provider and
never touches a path itself, so attaching a live feed later does not modify
`stage1_*`/`stage2_*`.

---

## 6. Dependency removal — evidence, not assumption

| package | verdict | evidence |
|---|---|---|
| **torch** | **removed** | Stage-2 is LiteRT; Stage-1 is `model.json`. Absent from the runtime venv; 8/8 smoke passes without it. |
| **tensorflow / keras** | **removed** | `ai-edge-litert` is a standalone wheel (`Requires: backports.strenum, flatbuffers, numpy, protobuf, tqdm, typing-extensions`). |
| **scikit-learn** | **removed** | Booster ≡ XGBClassifier bit-exact 48/48. |
| **pandas** | **KEPT — removal is unsafe** | The deployed base X comes from `X_df.to_numpy()`, and its F-order changes float32 mean/std by ~3e-08/6e-08 (§4). A numpy-only rebuild would have to reproduce that layout *and* pandas' `to_numeric`/`fillna` semantics; the parity risk is a moved alert DOY. Measured, not assumed. 70.2 MB. |
| **scipy** | **CANNOT be removed** | `pip show xgboost` → `Requires: numpy, scipy`. It is pulled in transitively, is never imported by our code, and is **the single largest package at 98.0 MB**. Removing it means replacing xgboost. |
| litert-torch / ai-edge-torch | **removed** | conversion-only; not in `requirements-runtime.txt`. |
| jupyter / notebook / pytest | **absent** | never installed; `test_no_heavy_dependencies.py` asserts it. |

Also excluded from production: **FP32 Stage-2 models** (build fails if present),
tests, `build_package.py`, dev reports.

---

## 7. Torch-free verification (D)

Run in `.venv-lightweight-api-test`, created fresh and installed from
`requirements-runtime.txt` **alone**. The test refuses to pass if a banned
framework is importable, so a stray torch elsewhere cannot fake it.

| check | result |
|---|---|
| `import torch` / `tensorflow` / `keras` / `sklearn` / `litert_torch` / `jupyter` / `pytest` | **all absent** ✅ |
| `.pt`/`.pth`/`.ckpt`/`.onnx` in package | **0** ✅ |
| CUDA/cuDNN artifacts | **0** ✅ |
| API imports | PASS |
| 8-pest assets load (16 Stage-1 branches + 8 Stage-2 FP16) | PASS |
| single prediction | PASS |
| CLI | PASS |
| **8-pest real smoke, inside this venv** | **8/8 PASS** |

---

## 8. Parity results — 8/8

Live lightweight run vs the recorded golden reference (README §10). **Alerts are
computed live** through the full portable chain (daily → season → F-order base X
→ Booster → temperature → k/τ gate).

| pest | alert (live) | alert (golden) | mu_doy (live, fp16) | mu_doy (golden) | Δ | final_source | ok |
|---|---:|---:|---:|---:|---:|---|:--:|
| BPH | **176** | 176 | 234.33 | 234.32 | 0.010 | learned_stage2 | ✅ |
| WBPH | **171** | 171 | 245.82 | 245.82 | 0.000 | learned_stage2 | ✅ |
| bacterial_blight | **206** | 206 | 268.32 | 268.32 | 0.000 | climatology | ✅ |
| blast | **125** | 125 | 198.08 | 198.08 | 0.000 | climatology | ✅ |
| brown_spot | **157** | 157 | 224.93 | 224.93 | 0.000 | climatology | ✅ |
| rice_stem_borer_1 | **125** | 125 | 184.17 | 184.17 | 0.000 | climatology | ✅ |
| rice_stem_borer_2 | **186** | 186 | 260.97 | 260.97 | 0.000 | climatology | ✅ |
| sheath_blight | **118** | 118 | 192.72 | 192.72 | 0.000 | climatology | ✅ |

Every alert DOY matches exactly. BPH's 0.010 is fp16 rounding at the 2-dp
boundary (raw 234.3252 vs 234.3240); its prediction interval is identical.

### What is and is not a live comparison

| item | status |
|---|---|
| Stage-1 raw probability | **golden reference** — proven bit-exact (max\|diff\|=0, 16/16 models) by `stage1_xgboost_migration/reports/alert_parity_report.md` on the server, not re-run here |
| Stage-1 calibrated probability | **golden reference** — same source, bit-exact |
| τ/k decision, alert_tstar, dispatch | **live here** for the final alert; the exhaustive per-sample comparison is the migration report's |
| Stage-2 tensor | **live-equivalent** — the vendored builder is bit-exact vs `build_real_input` on 8/8 (Stage-2 report) |
| Stage-2 mu_doy / PI | **live here**, compared to the recorded README values |
| fallback + final JSON | **live here**, compared to the deployed literals (code-read, plan §3) |

**The original API cannot run on this MacBook**: its Stage-1 `.pt` stores the
booster as a ~1 MB legacy binary blob that xgboost ≥ 2 rejects by **SIGSEGV (exit
139)** — reproduced on 2.1.4 and 2.1.4/3.1.2. Nothing above is invented; where a
live comparison was impossible it is labelled a golden reference.

**To re-compare against the original on the server** (where the legacy xgboost
exists):

```bash
# on the GPU server, in the ORIGINAL API's environment
cd ~/research/cropscience/api_handoff_transformer
for P in BPH WBPH bacterial_blight blast brown_spot rice_stem_borer_1 rice_stem_borer_2 sheath_blight; do
  python run_predict.py --input-dir /path/to/input_$P --output-dir /tmp/ref_$P
done
# then, with this package:
for P in ...; do
  python api_handoff_litert_portable/run_predict.py --input-dir /path/to/input_$P --output-dir /tmp/lw_$P
done
# and diff, ignoring the additive `backends` key:
python - <<'EOF'
import json, pathlib
for p in pathlib.Path('/tmp').glob('ref_*'):
    a = json.loads((p/'response.json').read_text())
    b = json.loads((pathlib.Path(str(p).replace('ref_','lw_'))/'response.json').read_text())
    b.pop('backends', None); b.pop('diagnostics', None); a.pop('diagnostics', None)
    print(p.name, 'IDENTICAL' if a == b else f'DIFF: {a} != {b}')
EOF
```

---

## 9. Fallback policy — unchanged

`fallback_policy.yaml` is copied verbatim; the selection logic is a line-for-line
port of `run_predict.py:540-566`. BPH/WBPH keep `learned_stage2` as the official
`final_prediction`; the other 6 keep `climatology` with the learned output as an
`experimental` auxiliary field. **No threshold, gate or policy was changed, and no
newer research finding was applied.** Verified live: all 8 `final_source` values
match the golden reference.

Deployed quirks reproduced deliberately rather than fixed (they are the contract):
`alert_fired` tracks `alert_tstar_doy_used`; `climatology_no_alert` is emitted for
any Stage-2 failure on learned-recommended pests; exit code is always 0 in single
mode; pest is case-sensitive; `year` accepts `bool`.

---

## 10. Size measurement

**Configuration B — production (FP16 only, runtime deps only). This is the recommendation.**

| component | size |
|---|---:|
| assets/stage1 (16 `model.json` + gates + `site_history.json`) | 26,188 KB |
| assets/stage2 (8 FP16 `.tflite` + normalization) | 3,240 KB |
| assets/configs + climatology | 44 KB |
| **package assets total** | **29,472 KB (29.47 MB)** |
| runtime code (`.py`) | 104 KB |
| **runtime venv** | **265,708 KB (259.48 MB)** |
| **package + runtime** | **288.26 MB** |
| production archive (`.tar.gz`) | **8.6 MB** |

Top packages in the runtime venv:

| package | size |
|---|---:|
| **scipy** | **98.0 MB** (transitive via xgboost; never imported by us) |
| **pandas** | **70.2 MB** (required for layout parity, §4/§6) |
| numpy | 33.6 MB |
| ai_edge_litert | 30.4 MB |
| pip | 11.8 MB (removable from a production image) |
| xgboost | 7.6 MB |
| protobuf (`google`) | 2.7 MB |
| PyYAML, dateutil, tqdm, flatbuffers, six, typing_extensions | < 1 MB each |

**Configuration A — full validation (FP16 + FP32 + test deps):** assets grow by
the 8 FP32 models (+3.9 MB → ~33.4 MB) and the venv adds scikit-learn (~40 MB)
for the Booster-equivalence test. Not recommended for deployment.

**Comparison:** vs the **reported** ~9–10 GB → **~97% smaller**. Vs the **measured**
1,800.32 MB dev env (`.venv-tflite`) → **~84% smaller**. The 9–10 GB was not
measured by me (§1).

---

## 11. Remaining bottlenecks

1. **scipy 98 MB** — the largest single item, pulled in by xgboost and never used
   by our code. The only way to drop it is to stop using xgboost (e.g. a
   numpy tree-walker over `model.json`), which risks Stage-1 parity. Not attempted.
2. **pandas 70.2 MB** — removable only by reproducing the F-order layout *and*
   pandas' numeric coercion in numpy. The parity risk (a moved alert DOY) is not
   worth 70 MB without a dedicated bit-exactness campaign.
3. **`site_history.json` = 26.2 MB of the 29.5 MB assets** — 7.3k site-year keys
   per pest, most of which a single request never touches. A per-site index or a
   compact binary would cut package size by ~80% with no model change.
4. **ai-edge-litert 30.4 MB** — the interpreter; nothing to trim.
5. **pip 11.8 MB** — excludable from a production container image.

Realistic floor without changing Stage-1's engine: ~**190 MB** (dropping pip and
slimming `site_history.json`); ~**90 MB** if xgboost/scipy were replaced.

---

## 12. Connecting the weather API and representative sites

**Not implemented in this step, by instruction.** The seam is `infer/providers.py`:

1. Implement `WeatherProvider.daily(site_id, year)` against the weather API, mapping
   to the Korean daily schema. `GDD10_since_gs` must be supplied or derived —
   upstream does **not** impute it.
2. The 880-representative-site ↔ 105-ASOS mapping becomes a resolver *in front of*
   the provider (`site_id → ASOS station → weather`). `notebooks/map_unique_sites_to_nearest_asos.py`
   exists in the repo as a starting point and was **not** used or validated here.
3. `SiteMetadataProvider`/`PhenologyProvider` currently read the LONG file; the 7
   non-BPH pests need coordinates + phenology, and phenology is **step data**
   (a handful of observed DOYs, forward-filled) that cannot be derived from weather.
4. Nothing in `stage1_*`/`stage2_*` changes.

## 13. Connecting Stage 1 and the full API

Stage 1 is already live in this package. Remaining to reach full deployed parity:

1. **Batch mode** — not implemented; needs the representative-site CSV and a
   different summary schema (plan §10).
2. **`site_history` fallback** — the deployed API re-derives history from the
   request's own prior-year obs when the site-year is missing; this package raises
   instead. Port `stage1.py:806-829` if that path matters.
3. **Deprecated `--input`/`--output` zip staging** — omitted deliberately.
4. **Server-side A/B** against the original (§8 commands) before cutover.

## 14. Swapping in the newer DN model later

1. Land the new Stage-2 checkpoint; re-export LiteRT
   (`tflite_conversion/standalone_stage2/build_package.py --clean`) — the loader
   fails loudly on architecture drift rather than converting something wrong.
2. If Stage-1 changed, re-export the portable JSON via `stage1_xgboost_migration/`.
3. `python api_handoff_litert_portable/build_package.py --clean --archive`.
4. Re-run all three test commands; **the memory-layout regression must still pass**
   — if the feature set changes, `stage1_features.py` needs the matching change and
   that test is what proves it.
5. Update `fallback_policy.yaml` only if the researcher promotes a pest; this work
   changed no policy.

## 15. Repo hygiene

- No API file, checkpoint, or `fallback_policy.yaml` was modified; the 8 Stage-2
  checkpoints and the reference API remain byte-identical.
- `.gitignore` gained the lightweight package's build outputs (`dist/`, `assets/`,
  `input/`, `output/`, test reports) and `.venv-lightweight-api-test/`, all verified
  with `git check-ignore`; `git ls-files -i -c` confirms no tracked file became ignored.
- Nothing staged, committed or pushed.
