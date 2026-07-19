# Batch implementation plan — analysis of the deployed contract

**Date:** 2026-07-16 · **Branch:** `feature/tflite-conversion`
**Source of truth:** `api_handoff_transformer/infer/batch.py` (528 lines, read-only), plus its call site `run_predict.py:794-797`.
**Purpose:** port batch to `api_handoff_litert_portable/` without breaking the single-mode contract.

---

## 0. ⚠️ The request schema differs from the task description

The task sketched this request:

```json
{"mode": "batch", "input_csv": "...", "output_dir": "...",
 "stage2_variant": "fp16", "include_diagnostics": false}
```

**None of `input_csv`, `output_dir`, or `stage2_variant` exist in the deployed
contract.** The real one is *one pest + one year per request*, with the site list
coming from the representative-site CSV:

```json
{"mode": "batch", "pest": "BPH", "year": 2004,
 "include_diagnostics": false,
 "representative_sites_path": "...",   // optional; alias: representative_sites_csv
 "max_sites": 50}                      // optional; batch.py:345 "testing/ops only; not part of the API spec"
```

`output_dir` comes from the `--output-dir` CLI arg, not the request
(batch.py:260). Per instruction ("실제 코드 계약을 기준으로 적용하고 차이를
보고해줘") the deployed shape is implemented as primary.

**Resolution:** support both.

| mode | trigger | order | rationale |
|---|---|---|---|
| **rep-CSV batch** (deployed contract) | `mode=batch` + `pest` + `year` | `sorted(site_id)` — batch.py:341 | byte-compatible with the deployed API |
| **generic CSV batch** (additive) | `mode=batch` + `input_csv` | **input row order preserved** | satisfies "일반 batch CSV 입력 지원" and "입력 순서 유지"; lets one run mix pests/years/sites |

The generic mode is a strict superset — it never changes rep-CSV behaviour.

---

## 1. Input format (`run_batch`, batch.py:246-464)

Entered from `run_predict.py:794-797` on a raw `json.load` peek, **before**
single-mode validation — so batch never runs `load_request`:

```python
if str(raw_request.get("mode", "single")).lower() == "batch":
    from infer.batch import run_batch
    return run_batch(paths, raw_request, data_sources, _sys.modules[__name__], args)
```

| key | required | default | read at |
|---|---|---|---|
| `mode` | yes (`"batch"`, case-insensitive) | `"single"` | run_predict.py:794 |
| `pest` | **yes** | — | batch.py:264 |
| `year` | **yes**, int, `bool` rejected | — | batch.py:271-274 |
| `include_diagnostics` | no | `False` | batch.py:256 |
| `representative_sites_path` / `representative_sites_csv` | no | auto-discover | batch.py:297-301 |
| `max_sites` | no | none | batch.py:345 |

`alert_tstar_doy` is **ignored** — batch hardcodes `None` per site (batch.py:375).

**Pest normalization: batch IS case-insensitive** (`_normalize_pest`, batch.py:60-66)
while single mode is case-sensitive. `{"pest":"bph"}` succeeds in batch, fails in
single. This asymmetry is deployed behaviour and is preserved.

---

## 2. Representative-site CSV schema

Resolution priority (`_resolve_rep_csv`, batch.py:69-101): explicit request field
/ CLI arg (absolute as-is; relative tried against cwd → input_dir → pkg_root),
else auto-discover `representative_site_ids_2002_2024.csv` in input_dir → cwd →
pkg_root. Not found → whole-batch failure.

Required columns (`_representative_sites`, batch.py:104-122): **`pest`** (Korean
name) and **`site_id`**. Read `encoding="utf-8-sig"`, `dtype=str`, column names
stripped. Sites are deduped and **`sorted()`**.

Verified against the local file
(`.../관측소 메타데이터/representative_site_ids_2002_2024.csv`, 6,152 rows):

| column | required | note |
|---|---|---|
| `pest` | **yes** | Korean name |
| `site_id` | **yes** | kept as string |
| `시도`, `시군구`, `읍면동` | no | present, never read |

7 distinct Korean names ≈ 856–862 sites each: 깨씨무늬병 862, 벼멸구 856,
이화명나방 859, 잎도열병 861, 잎집무늬마름병 858, 흰등멸구 859, 흰잎마름병 860.
`rice_stem_borer_1` and `_2` **share** 이화명나방 (batch.py:51-52), so the 8 pests
map onto 7 lists.

## 3. Which sites actually run

```python
target_sites = sorted(rep_set & long_year_sites & daily_year_sites)   # batch.py:341
```

A representative site is skipped **silently** unless it has *both* LONG rows for
the year *and* daily weather for the year. The counts are reported
(`representative_site_count`, `year_available_site_count`, `requested_count`) but
skipped sites produce **no row** in `predictions.csv`. Worth knowing: a run over
860 rep sites can legitimately emit far fewer rows.

## 4. Per-row processing and single-request reuse

**Yes — batch reuses the single-request engine.** It calls
`rp.run_stage2_transformer(...)` then `rp.build_response(...)` per site
(batch.py:379-389), passing preloaded assets so nothing reloads:

```python
loaded = load_stage2_model(...); a_meta/d_meta = _load_stage1_ckpt(...)
temp, site_history_static = load_stage1_reference(...)      # batch.py:356-359, ONCE
```

The per-site request is synthesized as
`{pest, site_id, year, alert_tstar_doy: None, include_diagnostics}`.

## 5. Error policy — two distinct levels

| level | trigger | effect | exit |
|---|---|---|---|
| **whole-batch** | bad pest, bad year, asset/config load, rep-CSV resolution, LONG load, daily load | `_fail_batch` (batch.py:498-527) — **still writes all 3 files** with an error summary and a header-only CSV | **2** |
| **per-row** | any exception inside the site loop (batch.py:405) | caught; `_error_flat` row with `status="error"`; **batch continues** | 0 |

So: a bad request kills the batch; a bad *site* never does.

`_classify` (batch.py:235-243):

| status | condition |
|---|---|
| `success` | `learned is not None` |
| `fallback` | `learned_err` contains the substring **`"fired no alert"`** |
| `error` | anything else |

That substring match is coupled to the exact wording at run_predict.py:421 —
brittle, and noted as a porting hazard.

## 6. Output contract

### `predictions.csv` (`_write_predictions_csv`, batch.py:467-475)

`_SINGLE_FLAT_COLS` (the same 16 as single mode) **+ `["status","error_reason"]`**
**+ 9 `_DIAG_COLS` only when `include_diagnostics`**:
`alert_source, stage1_method, stage1_alert_tstar_doy, tstar_season_index,
mu_rel_season_index, base_channels_status, input_X_shape (stringified), d_in,
ckpt_pest_field`.

`csv.DictWriter(..., extrasaction="ignore")`, `newline=""` → **CRLF**.
One row per target site. Header-only on whole-batch failure.

So the extra columns are appended **after** the existing 16 — exactly the "기존
컬럼 뒤에만 추가" rule.

### `response.json` — a completely different schema from single mode

Root keys (batch.py:422-441): `mode, pest, pest_korean, year, model_version,
representative_site_count, year_available_site_count, requested_count,
success_count, fallback_count, error_count, elapsed_seconds, recommended_source,
climatology, results`. **No `stage1`/`stage2`/`final_prediction` at root.**

`results[]` has **two shapes**: the normal entry (`site_id, status, error_reason,
final_source, final_mu_doy, final_pi95, learned_mu_doy, alert_tstar_doy[,
diagnostics]`) and the **crash** entry (`site_id, status, error_reason,
traceback` — batch.py:412-413), which lacks `final_*`. Consumers must handle both.

`json.dumps(..., indent=2, ensure_ascii=False) + "\n"` (trailing newline).

### `run_log.txt` (batch.py:478-495)
Batch-specific header + a fixed 4-line asset-load note + `events:` list.

### Output directory
Files go directly into `paths.output_dir` (`mkdir(parents=True, exist_ok=True)`;
existing files never cleared). No per-site subdirectories. `output.zip` only on
the deprecated `--output`-without-`--output-dir` path.

## 7. diagnostics / parallelism

- **diagnostics: supported** — `include_diagnostics` adds 9 CSV columns and a
  `diagnostics` key per `results[]` entry.
- **parallelism: none.** Zero occurrences of multiprocessing / concurrent.futures /
  threads / joblib / asyncio. A plain sequential `for` loop; the speedup comes
  entirely from loading assets once.

## 8. Known deployed quirks (reproduce, do not "fix")

1. `_classify` fallback detection is a **substring match** on an error string.
2. `batch.py:408` passes `site` as the `pest` arg to `_error_flat`, masked one
   line later by `row["pest"] = pest`.
3. `"climatology": climatology.get("variant") and {...}` (batch.py:436) — an `and`
   short-circuit that would emit the variant string instead of the object if the
   variant were falsy.
4. Whole-batch failure writes files (exit 2) while single-mode `fail()` writes
   nothing (exit 1/2) — opposite conventions.
5. `_fail_batch` reports the **raw, un-normalized** pest.

## 9. Port plan for the lightweight package

**Changed files (reported before editing, per instruction):**

| file | change | risk to single mode |
|---|---|---|
| `infer/batch.py` | **new** | none |
| `run_predict.py` | add a `mode` peek before `load_request` + `--representative-sites` arg | **must be proven zero** — regression test 15 |
| `infer/schemas.py` | add `batch_flat_row` / column constants | additive only |
| `tests/test_batch.py` | **new** | none |
| `tests/test_output_contract.py` | **new** (covers `select_final`, no-alert, key order, CRLF) | none |

**Reused unchanged:** Stage-1 (`stage1_portable`, `stage1_features`), Stage-2
(`stage2_litert`), `fallback.select_final`, `preprocessing`, `providers`. No model
conversion, no retraining, no policy change.

**Deliberate divergences from the deployed batch, and why:**

| deployed | lightweight | reason |
|---|---|---|
| preloads Stage-2 `.pt` + Stage-1 `.pt` metas | preloads LiteRT model + Booster branches | the whole point of this package |
| substring `"fired no alert"` | explicit no-alert flag from the pipeline, with the substring kept as a fallback | keeps `_classify` semantics without depending on prose |
| rep-CSV only | rep-CSV **and** generic CSV | task requirement; additive |

**Out of scope this step (TODO only):** live weather provider, 880↔105 ASOS
mapping, phenology generation, `site_history` re-derivation, server A/B against
the original, release/LFS packaging.
