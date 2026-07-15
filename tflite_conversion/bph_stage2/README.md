# BPH Stage-2 → TensorFlow Lite

> **SUPERSEDED — use [`../stage2/`](../stage2/) instead.**
>
> This was the BPH-only prototype. The common pipeline in `../stage2/` covers
> all 8 pests with the same approach and no per-pest duplication. This directory
> is kept intact for reference and its artifacts are preserved; it receives no
> further work.
>
> The refactor was verified equivalent: the old wrapper here and the new common
> wrapper produce **bit-identical** mu for BPH across 18 (seed, alert) cases
> (max |old − new| = 0.0).
>
> Note the two packages' synthetic inputs differ by design — this one defaults
> to `alert_doy=176`, the common one to the middle of each pest's DOY range — so
> their *synthetic* mu values are not directly comparable. Real-data results are.

Converts the BPH Stage-2 `lead_v3` model (`HierarchicalCausalHazardTransformer`,
mu head only) to `.tflite`, and validates it against the untouched PyTorch API
model on both synthetic and real input.

**Nothing under `api_handoff_transformer/` is modified.** The original model,
loader, preprocessing and runner are imported as-is; all new code lives here and
all artifacts are written to `artifacts/`.

---

## 0. Prerequisites

The API package must be unzipped at the repo root (it is not committed):

```bash
cd /Users/doyoung-gil/cropscience
unzip -o api_handoff_transformer_batch_20260710.zip -d .
```

This must exist afterwards:
`api_handoff_transformer/assets/stage2/BPH/lead_v3_final_checkpoint_run4.pt`

## 1. Environment (dedicated venv — never the system/anaconda env)

```bash
cd /Users/doyoung-gil/cropscience
python3 -m venv .venv-tflite
./.venv-tflite/bin/python -m pip install --upgrade pip
./.venv-tflite/bin/python -m pip install -r tflite_conversion/bph_stage2/requirements-tflite.txt
```

Verified on macOS 26.5.1 / Apple M3 (arm64) / Python 3.12.2, torch 2.12.1,
litert-torch 0.9.1. Exact pins: `requirements-tflite.txt`.

## 2. Convert

```bash
cd tflite_conversion/bph_stage2
../../.venv-tflite/bin/python export_tflite.py                  # -> artifacts/bph_stage2_fp32.tflite
../../.venv-tflite/bin/python export_tflite.py --quantize fp16     # -> artifacts/bph_stage2_fp16.tflite
../../.venv-tflite/bin/python export_tflite.py --quantize dynamic  # -> artifacts/bph_stage2_dynamic.tflite
```

`export_tflite.py` refuses to write an artifact unless the wrapper's `mu` is
bit-identical to the original model's on the sample input.

## 3. Validate (synthetic, deterministic)

```bash
../../.venv-tflite/bin/python validate_parity.py --tag fp32
../../.venv-tflite/bin/python validate_parity.py --tag fp16
../../.venv-tflite/bin/python validate_parity.py --tag dynamic
```

Compares original PyTorch vs wrapper vs TFLite over 8 seeds, and re-proves the
two equivalences the wrapper relies on (K=1 mask dropping; time-encoder
chunking). Exit code 0 = pass.

## 4. Validate (real data)

The weather/observation CSVs are **not** in the repo. On this MacBook they are at:

| file | location |
|---|---|
| daily master (1.6 GB, all sites/years) | `/Users/doyoung-gil/연구실/d/1997_2024_RICE_union_all_sites_with_GDD10_since_gs.csv` |
| BPH LONG observations (BPH = 벼멸구) | `/Users/doyoung-gil/Downloads/LONG_by_pest/RICE_LONG_벼멸구.csv` |
| representative sites | `/Users/doyoung-gil/연구실/데이터/관측소 메타데이터/representative_site_ids_2002_2024.csv` |

Slice the one site-year out of the master first (avoids loading 1.6 GB):

```bash
MASTER="/Users/doyoung-gil/연구실/d/1997_2024_RICE_union_all_sites_with_GDD10_since_gs.csv"
awk -F, 'NR==1{print; next} $1=="33210_56298" && $2 ~ /^2004/ {print}' "$MASTER" > /tmp/bph_daily.csv

cd tflite_conversion/bph_stage2
../../.venv-tflite/bin/python validate_real.py \
  --daily /tmp/bph_daily.csv \
  --obs   "/Users/doyoung-gil/Downloads/LONG_by_pest/RICE_LONG_벼멸구.csv" \
  --cache-dir /tmp/bph_cache \
  --tag   fp32
```

Reproduces the README §10 smoke record for BPH `33210_56298 / 2004`:
alert DOY **176**, learned `mu_doy` **234.324** (README records 234.32).

## 5. Benchmark

```bash
../../.venv-tflite/bin/python benchmark.py --iters 300 --warmup 30
```

---

## Results (2026-07-15, Apple M3, single-threaded)

| build | size | mu error vs PyTorch (synthetic) | mu error (real) | vs torch speed |
|---|---:|---:|---:|---:|
| PyTorch wrapper | 680,290 B* | 0 (bit-exact) | 0 | 1.00× |
| **TFLite fp32** | **605,868 B** | **1.53e-05 d** | **0.0** | **1.30–1.40×** |
| TFLite fp16 | 381,576 B | 2.25e-02 d | 1.19e-03 d | 1.29–1.39× |
| TFLite dynamic | 283,432 B | **9.32e-01 d** ❌ | 4.61e-01 d | 1.60–1.72× |

\* the `.pt` is the whole checkpoint (99 meta keys, norm stats, unused hazard
head); the `.tflite` files hold only the mu graph. Not a like-for-like size.

Speed is given as a range over 4 runs: absolute latency drifts with machine load
(PyTorch measured 0.85–1.43 ms), so only the ratio is meaningful. fp32 and fp16
are within noise of each other.

**Recommendation: fp32 or fp16.** `dynamic` (int8 weights) shifts `mu` by up to
0.93 days on synthetic input — it fails the 0.5-day tolerance and is not worth
0.2 ms against a model whose own σ is 5 days.

## What the model expects

```
inputs   X          float32 (1, 1, 131, 27)   normalized, block layout
         tstar      int64   (1, 1)            1-based season index of t*
         valid_mask bool    (1, 1)
output   mu         float32 (1, 1)            1-based season index
DOY = mu + doy_start - 1,  doy_start = 140
```

`mu` alone is not the API response. The caller still needs
`infer/preprocess.build_real_input` for the tensor (normalization stats live
*inside* the .pt, and dispatch channels 12..26 bypass normalization), plus the
95% PI `[round(mu_doy - 9.8), round(mu_doy + 9.8)]` from σ=5.

## Files

| file | role |
|---|---|
| `inference_model.py` | export-safe mu-only wrapper + ckpt config assertions + synthetic input |
| `export_tflite.py` | torch.export → litert-torch → `.tflite` (+ fp16/dynamic) |
| `validate_parity.py` | 3-way synthetic parity + wrapper-assumption proofs |
| `validate_real.py` | real-data BPH smoke test vs README §10 |
| `benchmark.py` | size / latency / peak-memory comparison |
| `artifacts/` | `.tflite` files + parity/benchmark JSON (**not committed**) |

Full write-up: [`docs/bph_stage2_tflite_conversion_report.md`](../../docs/bph_stage2_tflite_conversion_report.md)

## Known limitations

- **Stage-1 cannot run in this environment.** The shipped Stage-1 XGBoost ckpts
  SIGSEGV (exit 139) under xgboost 2.1.4 and 3.1.2. Stage-2 does not need
  xgboost; `validate_real.py` sources the alert + 14 dispatch features from the
  shipped reference `configs/dispatch/BPH_dispatch.csv` instead.
- The exported graph is fixed at **B=K=1** (production's shape). K>1 requires a
  re-export and reinstating the causal/padding masks.
- Only `mu` is exported. The hazard/PMF tail is dead code at inference — the API
  reads `model._last_mu_BK`, never the returned hazard.
