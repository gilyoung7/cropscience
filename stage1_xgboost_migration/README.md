# Stage-1 XGBoost portable model migration

Exports the Stage-1 event-gate XGBClassifier out of each `.pt` checkpoint into a
standard **XGBoost JSON** model, so the models load on a modern XGBoost without
depending on the pickled sklearn object inside the checkpoint.

**Nothing here modifies the original checkpoints, retrains anything, or changes any
threshold / gate / k / tau.** The live API still reads the `.pt` files; wiring it to
the JSON models is deliberately left as future work (see *Remaining work*).

## Why

The 16 Stage-1 `.pt` files hold a **pickled** `xgboost.sklearn.XGBClassifier` under
`trained_states[0]["sk_model"]`. Unpickling couples loading to the exact
xgboost/scikit-learn/numpy versions that wrote it — the failure mode this migration
removes. XGBoost's own JSON format is version-stable and forward-compatible, and
carries no pickle payload.

## Source of truth

| item | value |
| --- | --- |
| checkpoint source | `../../api_handoff_transformer_batch_20260708.zip` |
| zip SHA-256 | `665e1b85769d45d27e5e7ed0a9c8b9f686068bf88d9397d3e9388be18a0446b4` |
| extracted to | `_work/` (gitignored, `chmod a-w`) |
| source env | `../../api_handoff_transformer/.venv` |

> There is no `api_handoff_transformer_batch_20260710.zip` on this server. The `…20260708.zip`
> above (mtime 2026-07-10 17:47) is the only batch zip present, and its 16 Stage-1
> checkpoints are byte-identical (SHA-256) to the extracted `../../api_handoff_transformer/`
> tree, so both candidate sources agree.

Source env versions: python 3.12.3, torch 2.12.0+cu130, xgboost 3.3.0,
scikit-learn 1.9.0, numpy 2.4.6 (full freeze in `requirements-source-env.txt`,
recorded for provenance — you do **not** need it to use the exported models).

## Layout

```
stage1_xgboost_migration/
  common.py                     shared helpers, feature-order derivation
  export_portable_models.py     .pt -> artifacts/<pest>/<A|D>/model.json + metadata.json
  validate_server_parity.py     source .pt model vs reloaded JSON, in the source env
  validate_modern_xgboost.py    replays fixtures on a modern XGBoost (MacBook)
  migration_manifest.json       what was exported, with hashes  [generated]
  requirements-source-env.txt   frozen source env, provenance only
  requirements-modern.txt       minimal deps to run the portable models
  artifacts/<pest>/<A|D>/       model.json + metadata.json       [committable]
  fixtures/                     inputs + reference predictions   [gitignored]
  reports/                      parity reports                   [gitignored]
  _work/                        zip extraction                   [gitignored]
```

## Feature order — read this before wiring the API to JSON

The checkpoints do **not** store the 85/162/211/288-column order the model is
called with; `feature_names` in the `.pt` is the **per-timestep channel** list, not
the flat feature vector. The order is defined by `infer/stage1.py::_build_tabular`
and is reconstructed in `common.py::derive_feature_names`:

```
channels = ckpt["feature_names"]                       # 12 or 30
if ckpt["site_history_added"]:                         # true on the D branch only
    channels += HISTORY_STATIC_NAMES + HISTORY_DYNAMIC_NAMES   # +11, via _append_history

features = [f"{stat}__{ch}" for stat in [mean, std, min, max, first, last, slope]
                            for ch in channels]        # 7 x n_channels
if ckpt["add_tstar_position_feature"]:                 # true everywhere
    features += ["tstar_pos"]                          # tstar / season_length
```

This reproduces every observed width exactly, and the export **asserts**
`len(features) == n_features_in_` per model rather than trusting the derivation —
an export fails loudly instead of writing a guessed order:

| branch | channels | width |
| --- | --- | --- |
| BPH A | 12 | 12x7+1 = **85** |
| BPH D | 12+11 | 23x7+1 = **162** |
| other A | 30 | 30x7+1 = **211** |
| other D | 30+11 | 41x7+1 = **288** |

`model.json` stores no feature names (`booster.feature_names` is `None` in the
source, so column *order* is the only contract). Callers must build columns in this
exact order. The full list is in each `metadata.json` under `feature_names`, with
provenance under `feature_name_order`.

## Reproduce on the server

```bash
cd /home/gpu4080/research/cropscience/stage1_xgboost_migration
V=../../api_handoff_transformer/.venv/bin/python

# 1. extract checkpoints read-only from the zip (originals untouched)
unzip -o -q ../../api_handoff_transformer_batch_20260708.zip \
    'api_handoff_transformer/assets/stage1/*' -d _work
chmod -R a-w _work/api_handoff_transformer/assets/stage1

# 2. export 16 portable models + metadata + manifest
$V export_portable_models.py

# 3. parity + regenerate fixtures (writes reports/server_parity_report.json)
$V validate_server_parity.py
$V validate_server_parity.py --skip-real     # synthetic only, much faster
```

## Status

All 16 models exported and verified bit-exact (`max|Δ| = 0.0`, class agreement 1.000):

| check | env | result |
| --- | --- | --- |
| source `.pt` model vs reloaded `model.json` | xgboost 3.3.0 (source env) | **16/16 bit-exact** |
| `model.json` on a *different* xgboost, no torch/pickle | xgboost 3.0.5 (`../.venv`) | **16/16 bit-exact** |

Real-data parity ran for **sheath_blight A and D** — the only pest with an
observation CSV under `api_handoff_transformer/input/LONG_by_pest/`. The other 7
pests are synthetic-only; `reports/server_parity_report.json` records the exact
reason per pest under `real_data_notes`.

The cross-version run is the point of the exercise: unpickling `sk_model` from the
`.pt` emits `"configuration generated by an older version of XGBoost, please export
the model by calling Booster.save_model from that version first"`, and that path
breaks as versions drift. Loading `model.json` needs neither torch nor
scikit-learn — only xgboost + numpy.

## Verify on the MacBook / modern XGBoost

Copy `stage1_xgboost_migration/` across **including `fixtures/`** (gitignored, so
move it out of band — see *Fixtures* below). Then:

```bash
cd stage1_xgboost_migration
python3 -m venv .venv
./.venv/bin/pip install -r requirements-modern.txt
./.venv/bin/python validate_modern_xgboost.py
```

Exit code 0 means every model loaded on the modern XGBoost and reproduced the
server's predictions bit-for-bit on the stored fixtures. It verifies each
`model.json` against the SHA-256 in its `metadata.json` first, so a truncated or
corrupted copy fails loudly rather than silently comparing the wrong bytes.

If a future XGBoost changes float handling and bit-exactness is lost, quantify it
rather than assuming it is benign:

```bash
./.venv/bin/python validate_modern_xgboost.py --tolerance 1e-7
```

## Fixtures

`fixtures/` holds the parity inputs and the server's reference `predict_proba`
outputs as `.npy`. It is **gitignored**: the `real` matrices derive from the
observation CSVs under `api_handoff_transformer/input/`, whose redistribution status
is not established. The `synthetic` matrices are seeded noise and carry no data.

Fixtures are reproducible: `make_synthetic()` uses `numpy.random.default_rng` with a
fixed per-model seed and explicit `float32`, so re-running step 3 regenerates
byte-identical inputs. Every fixture's SHA-256 is recorded in
`reports/server_parity_report.json` (`input_sha256`, `reference_proba_sha256`), and
`validate_modern_xgboost.py` re-hashes what it loads.

Each fixture is 64 rows x n_features of `float32` — **1.3 MB for all 16 models**, small
enough to move by hand. The multi-GB weather preprocessing cache lives under
`_work/cache/`, deliberately *not* under `fixtures/`, so copying `fixtures/` never
drags it along.

## Calibration and gate — how the alert is actually produced

Traced through the code, not assumed. There are **two** Stage-1 paths and they
calibrate differently:

| | cohort/research path | **API production path** |
| --- | --- | --- |
| entry | `compute_stage1_table` → `_calibrated_per_sy` | `get_alert_single_sy` → `compute_alert_single_sy` |
| temperature | **re-fit at runtime** by `_fit_temperature_grid` on VAL_YEAR=2023 raw probs | **read frozen** from `assets/stage1/<pest>/temperature.json` via `load_stage1_reference` |
| needs labels? | yes (NLL needs `y_event`) | no |

The API that ships never re-fits. `temperature.json` holds `temperature_A` and
`temperature_D`, so temperature is **per (pest, branch)** — one scalar each, fitted
once on the split3 val year. That is all the calibration there is:

```
p_cal = sigmoid(logit(clip(p_raw, 1e-8, 1-1e-8)) / temperature)   # _apply_temperature
```

so a **single scalar per model is sufficient**, and it is copied verbatim into
`calibration.json`. Nothing is re-fitted or re-tuned here.

**Freeze vs re-fit — recommendation: freeze (option 1).** Re-fitting at JSON runtime
is not merely undesirable, it is impossible in deployment: `_fit_temperature_grid`
needs `y_event` labels for the whole 2023 val cohort, which a prediction request does
not have. Freezing also *is* the status quo — it is what the shipped API already
does — so it changes no behaviour, and it makes the value auditable by hash.

### The gate: the YAML k/tau are dead and drifted

`run_predict.py::_gate_config_for` takes **only `method`** from
`configs/stage1_selected_gates.yaml`. `k`/`tau`/`tau_no`/`tau_with` come from
`assets/stage1/<pest>/group_tau/group_tau_hybrid_summary.json`, which
`infer/stage1.py` marks authoritative — *"the yaml copy has drifted for some pests
… do not trust yaml k/tau"*. Verified: it has drifted for exactly two pests.

| pest | yaml k/tau | authoritative JSON k/tau |
| --- | --- | --- |
| bacterial_blight | k=1, tau=0.575 | **k=3, tau=0.525** |
| rice_stem_borer_1 | k=2, tau=0.575 | **k=3, tau=0.55** |

`gate.json` is resolved the same way the API resolves it, and records the yaml copy
under `yaml_copy_for_reference` with a `drifted_vs_authoritative` flag.

**Config fields the API reads:** `per_pest.<pest>.method`, top-level `target_label`.
**Dead fields it never reads:** `k`, `tau`, `tau_no`, `tau_with`, `run`, `ckpt_A`,
`ckpt_D`, `sweep_csv*`, `summary_json` (ckpt/summary paths are rebuilt from
`stage1_dir`; the yaml's `assets/stage1/per_pest/...` paths do not even exist), plus
`inference_notes.temperature_scaling` (still says `TODO Q1`) and
`inference_notes.ckpt_load_pattern` (claims `ck['event_model']` holds the model
bytes — it actually holds the string `"xgb"`).

### Portable assets per model

```
artifacts/<pest>/gate.json               method + authoritative k/tau
artifacts/<pest>/<A|D>/model.json        XGBoost native model
artifacts/<pest>/<A|D>/metadata.json     feature order, shapes, hashes
artifacts/<pest>/<A|D>/calibration.json  temperature + formula + provenance
```

`site_history.json` (~7.3k site-year keys per pest) is **still required** for the D
branch and for `dispatch_group_tau` routing — it supplies the 11 history channels and
the `with_history` flag that picks `tau_no` vs `tau_with`. It is already a plain JSON
asset, so it is portable as-is and is referenced by hash from `calibration.json`
rather than copied.

## A real sharp edge: Stage-1 is sensitive to array memory layout

Found while validating, and worth knowing before touching the loader.
`_build_tabular` takes float32 `mean`/`std`/`slope` reductions down the season axis,
and numpy's accumulation order follows the array's strides. The API's base X comes
from `DataFrame.to_numpy()` and is **F-contiguous**; a **C-contiguous** array holding
bit-identical values gives `mean` differing by up to **3.3e-4**, ~1e-3 after the trees
and temperature — enough to move an alert by days when the score sits near tau
(observed: DOY 172 vs 168).

This is not caused by the migration: both the `.pt` and JSON paths consume the same
array and agree bit-exactly. But it means **any change that alters the layout of the
base X — a pandas upgrade, a `copy()`, a `reshape`, an `np.stack` — can shift alerts
with no model change at all.** The fixtures normalise X to C-contiguous and compute
their reference on those same arrays.

## Remaining work before the API can read JSON instead of .pt

Not started — this migration deliberately stops at conversion + verification.

1. `infer/stage1.py::_load_stage1_ckpt` still does `torch.load(...)["trained_states"][0]["sk_model"]`.
   It would need to `XGBClassifier().load_model(model.json)` and read `doy_start`,
   `doy_end`, `nowcast_window`, `nowcast_stride`, `add_tstar_position_feature`,
   `site_history_added`, `site_history_policy`, `history_train_year_max`, and
   `feature_cols` from `metadata.json` instead of the checkpoint. `feature_cols` is
   the one field `metadata.json` carries that the JSON model cannot supply.
2. Decide whether `_calibrated_per_sy` (the re-fitting cohort path) stays. It is the
   only remaining consumer of `.pt` + labels. The production path needs only
   `calibration.json`.
3. Keep `site_history.json` shipping alongside; the D branch cannot run without it.
4. Keep `requirements.txt`'s `torch`/`scikit-learn` pins only if Stage 2 still needs
   them — Stage 1 no longer does.
5. Pin the base-X memory layout (e.g. `np.ascontiguousarray` at one chokepoint) before
   any refactor of `_build_base_samples`, or alerts may move silently. See above.
6. Re-run the API's own `tools/regression_test.py` end-to-end against the JSON path
   before switching. Parity here covers Stage-1 up to `alert_tstar` + dispatch
   features; it does not cover Stage 2 or the fallback/climatology policy.
