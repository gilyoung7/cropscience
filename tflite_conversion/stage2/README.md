# Stage-2 → TensorFlow Lite (all 8 pests)

Converts every deployed Stage-2 `lead_v3` model
(`HierarchicalCausalHazardTransformer`, mu head only) to FP32 and FP16 `.tflite`,
and validates each against the untouched PyTorch API model on synthetic **and**
real data.

**Nothing under `api_handoff_transformer/` is modified.** The original model,
loader, and preprocessing are imported as-is; all artifacts go to `artifacts/`.

Supersedes `../bph_stage2/` (BPH-only prototype), which is kept for reference —
see "Relationship to bph_stage2" below.

---

## Quickstart

```bash
cd /Users/doyoung-gil/cropscience

# 0. the API package must be unzipped at the repo root (not committed)
unzip -o api_handoff_transformer_batch_20260710.zip -d .

# 1. dedicated venv
python3 -m venv .venv-tflite
./.venv-tflite/bin/python -m pip install --upgrade pip
./.venv-tflite/bin/python -m pip install -r tflite_conversion/stage2/requirements-tflite.txt

cd tflite_conversion/stage2
PY=../../.venv-tflite/bin/python

# 2. convert all 8 pests -> fp32 + fp16
$PY export_all.py

# 3. synthetic parity (original vs wrapper vs fp32 vs fp16)
$PY validate_all.py --seeds 8

# 4. real-data validation (paths below are this MacBook's)
$PY validate_real.py \
  --daily-master "/Users/doyoung-gil/연구실/d/1997_2024_RICE_union_all_sites_with_GDD10_since_gs.csv" \
  --long-dir     "/Users/doyoung-gil/Downloads/LONG_by_pest"

# 5. benchmark
$PY benchmark_all.py --iters 200 --warmup 20 --repeats 3

# 6. join everything into the summary table
$PY summarize.py
```

Subsets: `--pests BPH WBPH`, `--variants fp32`.

## Layout

| file | role |
|---|---|
| `pest_configs.py` | pest list, expected-config cross-check table, LONG filenames, README §10 smoke records, policy offsets |
| `checkpoint.py` | load via the untouched `infer/ckpt.py` + validate the ckpt against the contract; synthetic input builder |
| `inference_model.py` | export-safe mu-only wrapper (one class, all 8 pests) |
| `export_all.py` | torch.export → litert-torch → `.tflite` (fp32/fp16), sha256, continue-on-failure |
| `validate_all.py` | synthetic parity + wrapper-assumption proofs |
| `validate_real.py` | real preprocessing → mu, vs README §10 records |
| `benchmark_all.py` | latency/size per pest |
| `summarize.py` | joins all stage JSON → `conversion_summary.json` + markdown tables |
| `artifacts/<pest>/` | `.tflite` + meta JSON (**not committed**) |

## Results (2026-07-15, Apple M3, single-threaded)

**8/8 FP32 and 8/8 FP16 converted; 8/8 wrapper parity bit-exact; 8/8 validated on real data.**

| pest | T×d_in | fp32 max err (d) | fp16 max err (d) | fp32 size | fp16 size | torch ms | fp32 ms | fp16 ms | rec |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| BPH | 131×27 | 7.63e-06 | 2.62e-02 | 604,596 | 380,232 | 1.403 | 1.036 | 1.045 | fp16 |
| WBPH | 241×45 | 1.53e-05 | 8.47e-04 | 629,172 | 403,328 | 3.280 | 2.682 | 2.621 | fp16 |
| bacterial_blight | 241×45 | 0.00e+00 | 1.43e-03 | 629,172 | 403,328 | 3.132 | 2.634 | 2.599 | fp16 |
| blast | 241×45 | 0.00e+00 | 1.14e-03 | 629,172 | 403,328 | 3.018 | 2.535 | 2.680 | fp16 |
| brown_spot | 241×45 | 0.00e+00 | 6.10e-04 | 629,172 | 403,328 | 3.149 | 2.531 | 2.553 | fp16 |
| rice_stem_borer_1 | 241×45 | 1.53e-05 | 1.53e-03 | 629,172 | 403,328 | 3.022 | 2.598 | 2.508 | fp16 |
| rice_stem_borer_2 | 241×45 | 1.53e-05 | 1.53e-05 | 629,172 | 403,328 | 3.071 | 2.530 | 2.555 | fp16 |
| sheath_blight | 241×45 | 1.53e-05 | 1.83e-04 | 629,172 | 403,328 | 3.026 | 2.526 | 2.505 | fp16 |

BPH is ~2.5× faster than the rest because its input is 131×27 = 3,537 elements
vs 241×45 = 10,845 (3.07×). Absolute ms drifts with machine load — trust the
ratio (TFLite ≈ 1.16–1.35× faster), not the absolute number.

Dynamic-range (int8) quantization is **not** built here: it shifted BPH's mu by
up to 0.93 days in the earlier prototype, failing tolerance.

## Tolerances

| variant | atol (days) | why |
|---|---:|---|
| fp32 | 1e-3 | mu is O(100) and float32 has ~7 digits → eps-level error ≈1e-5. 1e-3 d (~86 s) is 4 orders below the model's own σ=5 d and far below the API's 1-day PI rounding, yet tight enough to catch a real fault. |
| fp16 | 0.1 | fp16 has ~3 digits; the bounded-sigmoid lead head amplifies weight rounding over a 68-day span. 0.1 d (~2.4 h) is still 50× below σ=5 d and cannot change a rounded response, while rejecting the 0.93 d dynamic-range failure mode. |

`summarize.py` additionally prefers fp16 only when its **real-data** error stays
under 0.01 d (~15 min); above that it recommends fp32 and flags the pest.

## What the models expect

```
inputs   X          float32 (1, 1, T, d_in)   normalized, block layout
         tstar      int64   (1, 1)            1-based season index of t*
         valid_mask bool    (1, 1)
output   mu         float32 (1, 1)            1-based season index
DOY = mu + doy_start - 1
```

| | BPH | other 7 |
|---|---|---|
| d_in / T | 27 / 131 | 45 / 241 |
| DOY range | [140, 270] | [60, 300] |
| alert_idx | 12 | 30 |

Shared: `d_model=48, n_head=4, n_layers=3, tstar_layers=1, σ=5, lead=[7,75],
mu_mode=lead_from_alert`. Verified against all 8 checkpoints; `checkpoint.py`
raises on any drift.

`mu` alone is not the API response. A caller still needs
`infer/preprocess.build_real_input` for the tensor — normalization stats live
*inside* the `.pt`, and the last 15 (dispatch) channels bypass normalization
(`infer/ckpt.py:137-142`) — plus the 95% PI `[round(mu_doy-9.8), round(mu_doy+9.8)]`.

## Relationship to `bph_stage2/`

`../bph_stage2/` was the BPH-only prototype, superseded by this package. Its
scripts are not tracked in git; only a short notice remains there. The refactor
was verified: the old and new wrappers produce bit-identical mu for BPH over 18
(seed, alert) cases — max |old − new| = 0.0. Use this directory for all work,
including BPH.

## Known limitations

- **Stage-1 cannot run in this environment.** The shipped Stage-1 XGBoost ckpts
  SIGSEGV (exit 139) under xgboost 2.1.4 and 3.1.2 — the booster is a legacy
  binary blob that xgboost ≥ 2 refuses. Stage-2 never imports xgboost;
  `validate_real.py` sources the alert + 14 dispatch features from the shipped
  reference `configs/dispatch/<pest>_dispatch.csv` instead. This is a real
  problem for the API recipient, independent of TFLite.
- Exported graphs are fixed at **B=K=1** (production's shape). K>1 needs a
  re-export with the causal/padding masks reinstated.
- Only `mu` is exported; the hazard/PMF tail is dead code at inference.
- Only **BPH and WBPH** actually serve the learned output as the API's final
  answer; the other six return climatology and mark the Transformer output
  `experimental` (`configs/fallback_policy.yaml`). Their `.tflite` files are
  ready but optimize a path production does not currently answer with.

Full write-up: [`docs/all_pests_stage2_tflite_conversion_report.md`](../../docs/all_pests_stage2_tflite_conversion_report.md)
