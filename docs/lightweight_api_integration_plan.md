# Lightweight API integration plan

**Date:** 2026-07-16 · **Branch:** `feature/tflite-conversion`
**Goal:** reproduce the deployed API's behaviour and output contract with no PyTorch, no TensorFlow, no `.pt`.
**Scope:** existing deployed model `lead_v3_final` only. The newer DN model is explicitly excluded.

Reference API (read-only): `api_handoff_transformer/`, unzipped from
`api_handoff_transformer_batch_20260710.zip`
(sha256 `665e1b85769d45d27e5e7ed0a9c8b9f686068bf88d9397d3e9388be18a0446b4`, 136 files).

---

## 1. Real call flow (traced from code, not docs)

`main` (run_predict.py:718-848) runs in this order — note several checks precede request validation:

1. `Paths.from_root` → `load_data_sources` (**exit 1** if `data_sources.yaml` missing) — before any request check
2. `request.json` absent → `check_input_files(request=None)` → **exit 2**
3. raw `json.load` peek for `mode`; `mode == "batch"` → `infer/batch.py::run_batch` and **returns** (batch never runs single's validation)
4. `load_request` → `check_input_files` → `load_policy` → `load_stage1_gates` → `compute_climatology`
5. `run_stage2_transformer` (never raises; returns `(None, diag, err)`)
6. `build_response` → `write_outputs` → `write_run_log`

### Stage 1

`_gate_config_for` (run_predict.py:130-144) reads **only** `per_pest.<pest>.method` and top-level `target_label` from `stage1_selected_gates.yaml`. **k/tau come from `assets/stage1/<pest>/group_tau/group_tau_hybrid_summary.json`** → `selections["R>=0.88"][{A_raw_global|D_raw_global|dispatch_group_tau}]` (stage1.py:124-141).

The yaml k/tau have **drifted for 2 pests** — the JSON wins:

| pest | method | yaml k/tau | **JSON (authoritative)** |
|---|---|---|---|
| bacterial_blight | D_history | k=1, τ=0.575 | **k=3, τ=0.525** |
| rice_stem_borer_1 | A_baseline | k=2, τ=0.575 | **k=3, τ=0.55** |

Then `get_alert_single_sy` → `compute_alert_single_sy` (stage1.py:752-855):
`_base_sample_from_frames` → `_forward_one` (history → nowcast → tabular → `predict_proba` →
fixed temperature from `assets/stage1/<pest>/temperature.json`) → `_first_crossing_k` →
`_dispatch_features_for_sy` → `{alert_tstar_doy, dispatch_features(14)}` or `None`.

`with_h` comes from `site_history.json` (`"<site>|<year>" → 7 floats`); on miss it is re-derived
from the site's prior-year obs rows, and raises if none exist.

`alert_source` ∈ `{stage1_live, stage1_no_alert, stage1_error, manual_request, manual_request_over_stage1}`.
A manual `alert_tstar_doy` overrides **only** the alert channel; dispatch still comes from Stage-1.

### Stage 2

`build_real_input` (preprocess.py:487-648) → forward → **`model._last_mu_BK`** (the forward return
value is discarded, run_predict.py:468-471).
`tstar_season_index = alert_tstar_doy - doy_start + 1 + selected_offset`, clamped to `[1, T]`.

`dispatch_csv` is passed but **only read when `dispatch_row_override is None`** — i.e. on the
Stage-1-failed + manual-alert path. In the normal live path the 8 shipped dispatch CSVs are unused.

### mu_doy + prediction interval

```python
mu_rel           = float(mu_BK[0, 0])            # 1-based season index
mu_doy_temporal  = mu_rel + doy_start - 1.0
sigma            = ckpt["stage2_pmf_sigma"]      # learned: from the CKPT
half             = round(1.96 * sigma, 1)        # 9.8
mu_doy           = round(mu_doy_temporal, 2)
pi_95            = [int(round(mu_doy_temporal - half)), int(round(mu_doy_temporal + half))]
```

Two **different** sigma sources: learned uses the ckpt's sigma with `half` computed live;
climatology uses module constants `SIGMA_DAYS_DEFAULT = 5.0` / `PI95_HALFWIDTH = 9.8`
(run_predict.py:67-68). The interval derives from the **unrounded** mu. `round()` is
banker's rounding.

### Climatology

`compute_climatology` (run_predict.py:246-263) reads
`configs/climatology/<pest>_climatology_train_stats.csv`, row 0, column = `variant`
(always `mean_mid`). **The inline `mean_L/mean_mid/mean_R` in `fallback_policy.yaml` are dead** —
they merely happen to agree with the CSV.

### Fallback policy (unchanged by this work)

BPH + WBPH → `recommended_source: learned_stage2`; the other 6 → `climatology` with the learned
output as an `experimental` auxiliary field.

---

## 2. Dead config surface (present but never read)

`fallback_policy.yaml`

| field | status |
|---|---|
| `selected_fixed_offset`, `learned_output_status`, `recommended_source`, `climatology.variant` | **READ** |
| `version`, `schema_change_note`, `default_pi_quantile`, `default_sigma_days` | DEAD (sigma/PI hardcoded) |
| `run_stage2_learned` | DEAD — Stage-2 is attempted unconditionally |
| `final_prediction_source` | DEAD — `recommended_source` is used (they agree today) |
| `fallback_to_climatology_if_no_alert` | DEAD — fallback is unconditional |
| `selector_used` | DEAD — diagnostics hardcodes `False` |
| `climatology.mean_L/.mean_mid/.mean_R`, `.note`, `note` | DEAD — CSV wins |
| `no_alert_response.*`, `inference_flow_v0.*`, `resolved`, `outstanding_todos` | DEAD |

`stage1_selected_gates.yaml`: only `target_label` + `per_pest.*.method` are read. `k`, `tau`,
`tau_no`, `tau_with`, `run`, `sweep_csv*`, `summary_json` are DEAD, and `ckpt_A`/`ckpt_D` are DEAD
**and point at `assets/stage1/per_pest/...`, which does not exist** (the real layout is
`assets/stage1/<pest>/A_ckpt/...`; paths are constructed in code).

`configs/input_schema.json` / `output_schema.json` are **never referenced by any code**;
`today_doy` and `policy_override` are declared but unread.

---

## 3. Output contract to reproduce

`response.json` — `json.dumps(response, indent=2, ensure_ascii=False) + "\n"`:

```
pest, site_id, year, model_version="v0-transformer-real-ckpt",
stage1{alert_fired, alert_tstar_doy, gate_method, alert_source, wiring_status="stage1_xgboost_live"},
stage2{learned_stage2{mu_doy, pi_95{lower_doy, upper_doy, sigma_days}, selected_offset,
                      output_status, model_kind="lead_v3"} | null,
       climatology{mu_doy, pi_95{...}, variant}, recommended_source},
final_prediction{source, mu_doy, pi_95, selected_offset, fallback_triggered}
[, diagnostics{selector_used, climatology_train_stats_path,
               selected_fixed_offset_from_policy, transformer, transformer_error}]
```

`predictions.csv`: 16 fixed columns, `csv.DictWriter`, **CRLF**, `None`→empty, `True/False` as
Python repr. `run_log.txt`: fixed header + `events:` list.

Quirks that must be preserved verbatim:

1. **`stage1.alert_fired` tracks `alert_tstar_doy_used`**, set only *after* `build_real_input`
   succeeds — so a fired alert with a failed input build reports `alert_fired: false`.
2. **`climatology_no_alert` is emitted for *any* Stage-2 failure**, not only no-alert, and only for
   the 2 learned-recommended pests; the other 6 emit plain `climatology` with
   `fallback_triggered: false` — indistinguishable from a healthy run without diagnostics.
3. **Exit code is always 0 in single mode** — run_predict.py:848 is
   `return 0 if learned_err is None else 0`, a tautology. Stage-2 failure does not change it.
4. On any `fail()` path **nothing is written**; exit 1 (default) or 2 (missing inputs).
5. No pest normalization in single mode (case-sensitive); batch normalizes case-insensitively.
6. `year` accepts `bool` in single mode (`isinstance(True, int)` is True).

These are reproduced, not fixed — matching the deployed contract is the requirement.

---

## 4. Stage-1 array memory layout (measured, not assumed)

The original builds base X as `X_df.to_numpy(dtype=np.float32)` (stage1.py:512 cohort / :720
single). **Measured on this machine (pandas 3.0.3 / numpy 2.5.1):**

| boundary | dtype | strides | C | F |
|---|---|---|:--:|:--:|
| base X = `X_df.to_numpy()` — **A branch** | float32 | (4, 524) | 0 | **1** |
| after `_append_history` (`np.concatenate`) — **D branch** | float32 | (92, 4) | **1** | 0 |
| nowcast window slice — A branch | float32 | (4, 524) | 0 | 0 |
| nowcast window slice — D branch | float32 | (92, 4) | 1 | 0 |
| tabular features (`np.stack`) — both | float32 | C-contiguous | 1 | 0 |

So **the layout differs per branch**: A stays Fortran-ordered from `to_numpy`; D becomes
C-contiguous because `np.concatenate` returns C-order.

This matters. On the same values, layout changes float32 reduction order:

```
mean  identical across C/F layouts: False   max|d| = 2.980e-08
std   identical across C/F layouts: False   max|d| = 5.960e-08
min / max                         : True    (order-independent)
```

`_build_tabular` computes `mean`/`std`/`slope` over these arrays, so an `np.ascontiguousarray`
"cleanup" would perturb features by ~1e-8, which can flip a probability across τ and move the
alert DOY.

**Decision: the deployed API's layout is the source of truth and is reproduced exactly** — build
base X through pandas `to_numpy` (F-order) and append history via `np.concatenate` (C-order).
The lightweight package must not normalize layout. `tests/test_stage1_portable.py` pins
dtype/shape/strides/flags at each boundary and asserts an F→C "cleanup" changes the features
(the regression this guards).

---

## 5. Backend decisions (each verified before adoption)

| decision | evidence |
|---|---|
| **Stage 1: `xgboost.Booster`, not `XGBClassifier`** | Booster ≡ XGBClassifier bit-exact on **48/48** combos (16 models × {C, F, strided-view}), max\|diff\| = 0. Drops scikit-learn entirely. Also fixes a real breakage: `XGBClassifier.load_model()` raises `TypeError: _estimator_type undefined` under xgboost 2.1.4 + scikit-learn 1.9.0, so the existing `stage1_xgboost_migration/portable_stage1.py` cannot load models in that env. |
| **Stage 2: FP16 LiteRT** | `tflite_conversion/standalone_stage2` — preprocessing bit-exact 8/8; mu worst 3.6e-03 d vs PyTorch; PI identical 8/8. |
| **pandas: KEEP** | Removing it changes the array layout (§4) and the float32 features. Not worth the parity risk. Measured, not assumed. |
| **scikit-learn / scipy / torch / tensorflow: REMOVE** | Not needed by Booster + LiteRT + numpy/pandas. |

---

## 6. Package plan

```
api_handoff_litert_portable/
  run_predict.py            CLI, response/CSV/log writers (contract-compatible)
  infer/
    paths.py                asset locations
    schemas.py              request validation + response assembly (exact key order)
    stage1_features.py      base X (layout-preserving) + history + nowcast + tabular
    stage1_portable.py      Booster load, temperature, gate, dispatch features
    stage2_litert.py        FP16/FP32 LiteRT + mu_doy + PI
    preprocessing.py        Stage-2 tensor (from the standalone runtime)
    fallback.py             climatology CSV + fallback_policy (policy unchanged)
    providers.py            WeatherProvider / SiteMetadataProvider / PhenologyProvider
  assets/{stage1,stage2,configs,climatology}
  tests/…                   see §16 of the brief
  build_package.py          assemble + manifest + contamination scan + archive
```

Data-supply interfaces (`providers.py`) are split out now so the weather API and the
880-representative-site ↔ 105-ASOS mapping can be attached later without touching inference.
CSV/DataFrame providers are implemented in this step; no network calls.

---

## 7. Verification plan

A. Stage-1 portable load · B. Stage-1 alert parity · C. Stage-2 FP16 parity · D. 8-pest metadata ·
E. fallback policy · F. response schema · G. torch/TF-free env · H. contamination scan ·
I. size measurement · J. build reproducibility · K. **memory-layout regression**.

The original API **cannot run on this MacBook** — its Stage-1 `.pt` checkpoints store the booster
as a legacy binary blob that xgboost ≥ 2 rejects by SIGSEGV (exit 139, reproduced on 2.1.4 and
3.1.2). Anything that would require executing it is therefore reported as a **golden reference
comparison** against recorded values (README §10 smoke records, `stage1_xgboost_migration` parity
report, Stage-2 parity reports), never as a live result. The exact server-side commands to
re-compare against the original are given in the final report.
