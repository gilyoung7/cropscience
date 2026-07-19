# Batch implementation report

**Date:** 2026-07-20 · **Branch:** `feature/tflite-conversion` · **Host:** MacBook, Apple M3 (arm64)
**Outcome:** ✅ batch ported to `api_handoff_litert_portable/`; **single mode byte-identical**; all suites green (6/6, 19/19, 8/8, 23/23, no-heavy-deps PASS); new archive verified in isolation.
**Not committed** — this run stops before commit/push, as instructed.

Companion: [`BATCH_IMPLEMENTATION_PLAN.md`](BATCH_IMPLEMENTATION_PLAN.md) (contract analysis).

---

## 1. Implemented

| capability | status |
|---|---|
| **rep-CSV batch** (deployed contract: one pest + one year) | ✅ verified against the real 856-site 벼멸구 list |
| **generic CSV batch** (`input_csv`, mixed pest/site/year, order preserved) | ✅ |
| per-row reuse of the single-request pipeline | ✅ same `run_pipeline`, no duplicated inference |
| per-pest asset caching (LiteRT model + climatology + policy loaded once) | ✅ |
| all 8 pests | ✅ |
| `fp16` default / `fp32` selectable (`stage2_variant` or `--stage2-variant`) | ✅ |
| `include_diagnostics` → 9 extra CSV columns + per-row diagnostics | ✅ |
| per-row failure isolation (a bad row never aborts the batch) | ✅ |
| whole-batch failure → all 3 files still written, exit 2 | ✅ |
| `predictions.csv` / `response.json` / `run_log.txt` | ✅ |
| actionable error for a malformed batch request | ✅ (replaces the misleading `site_id` message) |

Files: **new** `infer/batch.py`, `tests/test_batch.py`, `tests/test_output_contract.py`,
`examples/*`; **modified** `run_predict.py` (mode peek + `--representative-sites` +
`ctx` reuse + a multi-site fix), `build_package.py` (`--archive-name`, refuses to
overwrite).

## 2. Identical to the deployed API

- 16 single-mode CSV columns, then `status,error_reason`, then the 9 diag columns —
  **appended only after** the existing ones.
- `_classify` semantics: `success` / `fallback` / `error`.
- Whole-batch vs per-row error levels and their exit codes (2 / 0).
- Batch is **case-insensitive** on pest (`bph` → `BPH`) while single mode is
  case-sensitive — the deployed asymmetry, preserved.
- rep-CSV resolution order, the `pest`+`site_id` schema, Korean-name matching
  (`rice_stem_borer_1`/`_2` share 이화명나방), dedup + `sorted()`.
- Target set = rep ∩ LONG-year ∩ daily-year; sites missing either are skipped
  silently (counted, not rowed).
- `max_sites`, `representative_sites_path` / `representative_sites_csv` aliases.
- CRLF in `predictions.csv`, trailing newline + `ensure_ascii=False` in JSON.
- `_fail_batch` reports the **raw, un-normalized** pest.
- Fallback policy untouched (BPH/WBPH learned; the other 6 climatology).

## 3. Deliberate divergences (and why)

| # | deployed | here | reason |
|---|---|---|---|
| 1 | request = `pest` + `year` only | **also** `input_csv` | the task asked for "일반 batch CSV 입력 지원" + input-order preservation, which the rep-CSV shape cannot express. Additive; rep-CSV behaviour unchanged. |
| 2 | rows in `sorted(site_id)` order | rep-CSV: sorted (unchanged) · generic: **input order** | matches each mode's contract |
| 3 | — | `row_index` column, **generic mode only** | required "요청 식별자 또는 row index"; absent from rep-CSV output so that stays byte-compatible |
| 4 | summary has `pest`,`year`,`pest_korean`,… | same, **plus** `input_kind`, `stage2_variant` | additive; rep-CSV keys all still present |
| 5 | fallback detected by substring `"fired no alert"` | same substring **plus** an explicit marker | keeps deployed semantics without depending solely on prose |
| 6 | preloads `.pt` assets | preloads LiteRT + Booster | the point of this package |

**Request-shape correction.** The task sketched `{"mode":"batch","input_csv",
"output_dir","stage2_variant"}`. The deployed contract has **none** of those keys
— it is one pest + one year, with `output_dir` coming from `--output-dir`. Both
shapes are now accepted; `output_dir` remains a CLI concern.

## 4. Bug found and fixed (affected single mode too)

The rep-CSV run initially returned **6/6 errors**:

```
Stage-2 failed for pest=BPH: PreprocessError: daily data covers 6 sites [...]; supply one site.
```

`run_pipeline` passed the **unfiltered** daily frame to Stage-2, whose tensor
builder is per-site by contract. Stage-1 was unaffected (it filters internally),
so per-site test fixtures had masked it — **single mode had the same latent bug**:
handing it the full master daily would have failed identically.

Fix: `run_predict.py` now passes `weather.daily(site_id, year)` (the provider's
site-filtered frame), mirroring the deployed `load_input_daily(site_id=...)`.
After the fix: **6/6 success**, each site with its own alert and mu:

| site | alert | mu_doy |
|---|---:|---:|
| 29608_53769 | 173 | 224.97 |
| 29851_55024 | 176 | 229.07 |
| 30144_56664 | 169 | 240.74 |
| 30236_55575 | 177 | 230.08 |
| 30247_65595 | 171 | 222.64 |
| 30304_62145 | 188 | 243.57 |

## 5. Single-prediction regression — none

| check | result |
|---|---|
| `test_batch.py` case R: single response before vs after batch | **byte-identical** (`json.dumps(sort_keys=True)` equal) |
| 8-pest smoke (alerts + mu vs golden reference) | **8/8 PASS**, unchanged |
| Stage-1 memory-layout regression | **6/6 PASS**, unchanged |

## 6. Test results

| suite | result |
|---|---|
| `test_stage1_portable.py` (incl. layout regression) | **PASS 6/6** |
| `test_output_contract.py` (**new**) | **PASS 19/19** |
| `test_batch.py` (**new**) | **PASS 23/23** |
| `test_smoke_cases.py` (8 pests) | **PASS 8/8** |
| `test_no_heavy_dependencies.py` | **PASS** |

Newly covered branches that previously had **zero** tests: `select_final` (all
four outcomes), no-alert, `climatology_no_alert`, `learned_stage2`,
`climatology`, `fallback_triggered`, error-row serialization, response key order,
CSV column order, CRLF, null→empty-field, trailing newlines.

**One test expectation was wrong, not the code.** I initially asserted an
unknown-site row would be `error`; it is `fallback`, because the Stage-1 failure
ends at the bail-out whose message starts "Stage-1 fired no alert …" and
`_classify` matches that substring. Verified the deployed API does exactly the
same (run_predict.py:419-424 + batch.py:241), then corrected the test.

## 7. Isolated deployment verification

New archive **`dist/api_handoff_litert_portable_batch_v2.tar.gz` (8,634,261 B)** —
the previous archive was **not** overwritten (the builder now refuses to).

Extracted to a temp dir outside the repo and run with the runtime-only venv:

| check | result |
|---|---|
| torch / tensorflow / sklearn / keras importable | **no** (all four absent) |
| xgboost sklearn wrapper used | **no** — `Booster` only |
| `.pt`/`.pth` in archive | **0** |
| FP32 models in the production archive | **0** |
| Stage-1 `model.json` / Stage-2 fp16 | **16/16 · 8/8** |
| `infer/batch.py` shipped, `tests/` excluded, `examples/` included | ✅ / ✅ (0 files) / 7 files |
| **single prediction** | alert **176**, mu_doy **234.33**, `learned_stage2` |
| **batch, 4 rows with failures** | 2 success · 1 fallback · 1 error; 3 output files written |

## 8. Deliberately NOT implemented (TODO)

- live weather API provider
- 880 ↔ 105 ASOS automatic mapping
- phenology auto-generation
- `site_history` re-derivation from prior-year observations
- server A/B against the original `.pt` API
- release / LFS packaging
- parallelism (the deployed batch is sequential too; assets-loaded-once is the
  only speedup)

## 9. Remaining operational work

1. **Weather + LONG per site.** Batch still needs `daily_weather.csv` and LONG
   covering every target site-year. With ~856 rep sites, only those present in
   **both** are predicted (this run: 133 of 856 had LONG for 2004).
2. **ASOS mapping** — required before "880 sites, one command" is real.
3. **Coordinates + phenology** for the 7 non-BPH pests still come from the LONG
   file.
4. **Server A/B** against the original before cutover.
5. Optional: `site_history.json` slimming (26.2 MB of 29.5 MB assets).

## 10. Reproduce

```bash
cd api_handoff_litert_portable
PY=../.venv-tflite/bin/python ; RPY=../.venv-lightweight-api-test/bin/python
DAILY_MASTER=/path/to/daily_weather.csv ; LONG_DIR=/path/to/long_by_pest

$RPY tests/test_output_contract.py
$RPY tests/test_batch.py       --daily-master "$DAILY_MASTER" --long-dir "$LONG_DIR"
$RPY tests/test_smoke_cases.py --daily-master "$DAILY_MASTER" --long-dir "$LONG_DIR"
$PY  tests/test_stage1_portable.py
$RPY tests/test_no_heavy_dependencies.py

$PY build_package.py --clean --archive --archive-name api_handoff_litert_portable_batch_v2
```

Batch run:

```bash
# generic CSV
$RPY run_predict.py --input-dir IN --output-dir OUT   # request.json: {"mode":"batch","input_csv":"rows.csv"}
# representative sites
$RPY run_predict.py --input-dir IN --output-dir OUT \
    --representative-sites /path/to/representative_site_ids_2002_2024.csv
```

Examples: `examples/batch_rows.csv`, `examples/request_batch_generic.json`,
`examples/request_batch_representative.json`, `examples/sample_batch_predictions.csv`,
`examples/sample_batch_response.json`, `examples/sample_batch_run_log.txt`.
