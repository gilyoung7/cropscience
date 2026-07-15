# BPH Stage-2 → TensorFlow Lite conversion report

**Date:** 2026-07-15
**Branch:** `feature/tflite-conversion`
**Host:** MacBook, Apple M3 (arm64), macOS 26.5.1
**Outcome:** ✅ `.tflite` produced and validated against PyTorch on both synthetic and **real** data.

---

## 1. Why BPH Stage-2 was chosen

From the prior investigation of the repo:

1. **Stage-1 is not a neural network.** It is `xgboost.sklearn.XGBClassifier`
   (`infer/stage1.py:531-545`, inference via `predict_proba`); `.pt` is only a
   pickle container. TFLite is irrelevant to it.
2. **Only BPH and WBPH actually serve the Transformer.** Per
   `configs/fallback_policy.yaml` (schema v4), BPH and WBPH have
   `recommended_source: learned_stage2` / `learned_output_status: main`. The
   other six pests return climatology as `final_prediction` and mark the
   Transformer output `experimental`. Converting sheath_blight would optimize a
   model that production does not answer with.
3. **BPH is the smallest.** `d_in=27, T=131` vs `d_in=45, T=241` for the other
   seven — the smallest tensor, so the fastest iteration.
4. **The architecture is shared.** All 8 pests are `d_model=48, n_head=4,
   n_layers=3, tstar_layers=1, sigma=5.0, lead=[7,75], mu_mode=lead_from_alert`.
   Whatever works for BPH generalizes to the rest.

## 2. Original model structure

`HierarchicalCausalHazardTransformer` (`infer/model.py:71`), 163,010 params
total. Checkpoint: `assets/stage2/BPH/lead_v3_final_checkpoint_run4.pt`
(680,290 B, 99 top-level keys, 60 state_dict tensors).

| key | BPH value |
|---|---|
| `d_in` / `d_model` / `n_head` / `n_layers` / `stage2_tstar_layers` | 27 / 48 / 4 / 3 / 1 |
| `T` / `doy_start` / `doy_end` | 131 / 140 / 270 |
| `stage2_pmf_mode` / `stage2_pmf_mu_mode` | `gaussian` / `lead_from_alert` |
| `stage2_pmf_sigma` / `lead_min` / `lead_max` | 5.0 / 7.0 / 75.0 |
| `stage2_pmf_alert_tstar_feat_idx` | 12 |
| `stage2_phenology_bias_head` | 0 (head absent) |
| `dropout` / `max_len` | **absent** → loader defaults 0.2 / 400 |

Channel layout (27) = 6 base weather + 6 `__miss` + 15 dispatch.
`feature_names[12] == 'alert_tstar'` — this **resolves Q2 of
`api_handoff_report.md`**, which had flagged the channel-12/30 semantic as
unverified. It is a literal `alert_tstar` DOY channel.

Forward (mu path):

```
X (1,1,131,27) → reshape (1,131,27)
  → in_proj(27→48) → PositionalEncoding → time_encoder (3 layers, gelu)  → h (1,131,48)
  → gather at tstar-1                                                     → z (1,1,48)
  → tstar_pos → tstar_encoder (1 layer, causal+padding mask) → nan_to_num → z*valid_mask
  → head_mu(48→48→1)                                                      → mu_logit (1,1)
alert_doy = X[...,12].amax(dim=2);  alert_rel = alert_doy - 140 + 1
lead = 7 + 68*sigmoid(mu_logit);   mu = clamp(alert_rel + lead, 0, 130)
```

Two structural facts that shape the conversion:

- **The returned hazard is dead compute.** `run_predict.py:468-471` discards
  `forward()`'s return and reads the side-effect attribute `model._last_mu_BK`.
  The Gaussian PMF → hazard tail (`model.py:405-409`) never influences the API
  response.
- **Normalization stats live inside the .pt** (`norm_mean`/`norm_std`, numpy
  arrays), and `ckpt.py:137-142` forces the last 15 (dispatch) entries to
  mean=0/std=1, so those channels bypass normalization. There is no external
  norm-stats file.

## 3. What blocked `torch.export`

Two independent blockers. The first was known from the investigation; **the
second was found only by attempting the export**, and was the one that actually
bit under torch 2.12.1.

### Blocker 1 — training branches in the model (`infer/model.py`)

| line | code | problem |
|---|---|---|
| 300 | `if not lead_loss_mask.any():` | `bool()` on a tensor → `.item()` → unbacked symint `u0` → `GuardOnDataDependentSymNode: Eq(u0, 1)` |
| 293-299 | `if strict and lead_loss_mask.any()` + `alert_doy_abs[lead_loss_mask]` | boolean-mask index → data-dependent output shape |
| 310-349 | one-shot debug block | `float(...)` / `.item()` throughout |
| 405-409 | Gaussian PMF → hazard | not needed for `mu` |

### Blocker 2 — inside PyTorch itself

Even after removing all of the above, export still failed with the *same* error
signature:

```
torch/nn/modules/transformer.py:535 in forward:
    is_causal = _detect_is_causal_mask(mask, is_causal, seq_len)
→ item: "Sym(Eq(u0, 1))" = torch.ops.aten.item.default(ne)
→ GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u0, 1)
```

Passing `mask=` to `nn.TransformerEncoder` sends it through
`_detect_is_causal_mask`, which evaluates `bool((mask == causal_comparison).all())`.
That is a `.item()` inside PyTorch's own code — unreachable by editing our model.

**Resolution:** at K=1 both masks are provably no-ops, so the wrapper drops them
on a `K == 1` branch (K comes from `X.shape` and is static at export, so the
branch is not data-dependent):

- causal mask `triu(ones(1,1), diagonal=1) == [[False]]` — masks nothing;
- an all-masked padding mask yields NaN, which `nan_to_num` + the subsequent
  `z * valid_mask` collapse to zero — the same zero the unmasked path produces.

`validate_parity.py` re-proves this every run for `valid=True` **and**
`valid=False`, all exact.

## 4. Wrapper design

`tflite_conversion/bph_stage2/inference_model.py` →
`BPHStage2InferenceModel(nn.Module)`.

- Constructed **from an already-loaded model** and holds *references to the same
  submodules* (`in_proj`, `pos`, `time_encoder`, `tstar_pos`, `tstar_encoder`,
  `head_mu`). No weight is copied or re-mapped, so mis-mapping is structurally
  impossible; `load_bph()` additionally asserts every wrapper parameter is the
  same object (`id()`) as the checkpoint model's.
- Loading is delegated to the **untouched** `infer/ckpt.py`, which already raises
  on missing/unexpected state_dict keys.
- `load_bph()` asserts the full BPH contract (d_in, T, d_model, doy range,
  alert_idx, lead bounds, sigma, and `feature_names[12] == 'alert_tstar'`) and
  raises on any drift.
- Rejects checkpoints with `phenology_bias_head` or `use_tstar_scalar_pos`
  enabled rather than silently ignoring those terms.
- Returns `mu` explicitly — no `_last_mu_BK` dependency.
- The chunked time-encoder loop (`model.py:202-208`) is replaced by one call;
  the encoder is batch-independent, so this is a no-op (re-proved each run).

Wrapper param count is 116,833 vs the checkpoint's 163,010 — the difference is
the unused hazard `head`, which is not on the mu path.

## 5. Conversion path actually used

**Priority 1 succeeded, with a correction.** `ai-edge-torch` 0.7.2 installs
fine on Apple Silicon — the "Linux-only" concern did not materialize — but it is
now a **deprecation shim**: the project was renamed to `litert-torch`, and
`ai_edge_torch.convert` no longer exists (`AttributeError`). The real entry point
is `litert_torch.convert`, which still runs `torch.export` underneath. So this is
the same official PyTorch → TFLite path under its current name. TensorFlow is
**not** required.

```
inference wrapper → torch.export (235 nodes) → litert_torch.convert → .tflite
```

Priorities 2 and 3 (alternative routes; TF/Keras re-implementation with manual
weight transfer) were **not needed**.

Estimated cost of the exported graph: **35.7 M ops ≈ 17.9 M MACs**.

### Installed versions (`.venv-tflite`, 75 packages)

| package | version |
|---|---|
| torch | 2.12.1 |
| ai-edge-torch | 0.7.2 (shim) |
| **litert-torch** | **0.9.1** |
| litert-converter | 0.2.0 |
| ai-edge-litert | 2.1.5 |
| ai-edge-quantizer | 0.7.0 |
| numpy | 2.5.1 |
| pandas / PyYAML | 3.0.3 / 6.0.3 |
| xgboost / scikit-learn (Stage-1 only, unusable — see §8) | 2.1.4 / 1.9.0 |

System/anaconda Python was **not** modified.

## 6. PyTorch vs TFLite output

Signature (fixed at production's B=K=1):

```
in : X float32 (1,1,131,27) | tstar int64 (1,1) | valid_mask bool (1,1)
out: mu float32 (1,1)        DOY = mu + 139
```

### A. Synthetic (seeded, real shape/dtype, 8 seeds, alert DOY 150…185)

| comparison | max abs | mean abs | max rel | verdict |
|---|---:|---:|---:|---|
| original vs wrapper | **0.0** | 0.0 | 0.0 | bit-exact |
| original vs TFLite fp32 | **1.53e-05 d** | 5.72e-06 | 2.05e-07 | PASS (atol 1e-3) |
| original vs TFLite fp16 | 2.25e-02 d | 1.56e-02 | 2.44e-04 | PASS (atol 0.5) |
| original vs TFLite dynamic | **9.32e-01 d** | 4.32e-01 | 1.04e-02 | **FAIL (atol 0.5)** |

**FP32 tolerance rationale (atol = 1e-3 days).** `mu` is O(100) (a season
index), and float32 carries ~7 significant digits, so eps-level relative error
lands near 1e-5 absolute. Reassociation differences between ATen and LiteRT
kernels (layernorm/softmax/matmul tiling) accumulate a few ulps. 1e-3 days ≈ 86
seconds: four orders of magnitude below the model's own σ = 5 days, and far
below the 1-day rounding the API applies to the 95% PI — operationally lossless,
yet tight enough to catch a genuine numerical fault. Observed error was 65×
inside this bound.

### B. Real data — README §10 smoke test reproduced ✅

`pest=BPH site_id=33210_56298 year=2004`, real weather + real observations,
`zero_placeholder_used=False`, `base_channels_status=real_preprocessing`:

| quantity | value |
|---|---|
| alert_tstar_doy used | **176** (README: 176 ✓) |
| tstar_season_index | 67 |
| original PyTorch mu_doy | **234.3240** (README: 234.32 ✓) |
| wrapper mu_doy | 234.3240 (Δ = 0) |
| **TFLite fp32 mu_doy** | **234.3240 (Δ = 0.0, exact)** |
| TFLite fp16 mu_doy | 234.3252 (Δ = 1.19e-03 d) |
| TFLite dynamic mu_doy | 234.7854 (Δ = 4.61e-01 d) |

The alert DOY was **not** assumed: the shipped reference
`configs/dispatch/BPH_dispatch.csv` independently carries
`alert_tstar=176, dispatch_branch=D, dispatch_tau_used=0.6, with_history=0` for
this site-year, consistent with BPH's `D_history` gate at τ=0.6.

## 7. Benchmark (Apple M3, single-threaded, 300 iters after 30 warm-up)

Representative run; the table below is one of four runs, with the spread across
runs given in the last column.

| model | size | mean ms | std | median | p90 | vs torch | vs torch across 4 runs |
|---|---:|---:|---:|---:|---:|---:|---:|
| PyTorch wrapper fp32 | 680,290 B* | 1.355 | 0.043 | 1.351 | 1.373 | 1.00× | — |
| **TFLite fp32** | 605,868 B | **1.039** | 0.031 | 1.026 | 1.089 | **1.30×** | 1.30–1.40× |
| TFLite fp16 | 381,576 B | 1.049 | 0.028 | 1.039 | 1.066 | 1.29× | 1.29–1.39× |
| TFLite dynamic | 283,432 B | 0.840 | 0.006 | 0.839 | 0.846 | 1.61× | 1.60–1.72× |

**Run-to-run variance is real and larger than the fp32/fp16 gap.** Absolute
latencies drifted (PyTorch 0.85–1.43 ms across runs) with machine load/thermal
state, so single absolute numbers should not be quoted. The *ratio* is stable:
TFLite fp32 ≈ **1.3–1.4× faster** than PyTorch. fp32 and fp16 are within noise
of each other and cannot be separated on speed by this measurement.

\* not like-for-like: the `.pt` is the entire checkpoint (99 meta keys, norm
stats, unused hazard head); the `.tflite` files carry only the mu graph.

`torch.set_num_threads(1)` pins PyTorch to match LiteRT's single-threaded
default interpreter, so the gap is not a thread-count artifact. TFLite is also
markedly more consistent (std 0.008 vs 0.041 ms). Peak memory via `tracemalloc`
is a Python-allocator figure only — it cannot see LiteRT's C++ arena or ATen's
buffers, so it is reported as a relative lower bound, not a real RSS number.

## 8. Quantization

| build | size | vs fp32 | mu error (synthetic) | mu error (real) | verdict |
|---|---:|---:|---:|---:|---|
| fp16 | 381,576 B | 0.63× (1.6× smaller) | 2.25e-02 d | 1.19e-03 d | ✅ recommended |
| dynamic (int8 weights) | 283,432 B | 0.47× (2.1× smaller) | **9.32e-01 d** | 4.61e-01 d | ❌ not recommended |

**fp16** is essentially free: 1.6× smaller for ~0.02 days (~30 min) of drift,
which is noise next to σ=5 days.

**dynamic range** shifts `mu` by up to **0.93 days** on synthetic input, failing
the 0.5-day tolerance, and 0.46 days on the real sample. Buying 0.2 ms and 98 KB
with ~half a day of timing error is a bad trade for a pest-timing forecast.
(Caveat: the synthetic input is Gaussian noise, i.e. out-of-distribution, which
likely exaggerates the error — but the real sample's 0.46 d is already 400×
the fp16 error.)

## 9. Failures and problems encountered

| # | attempt | exact error | resolution |
|---|---|---|---|
| 1 | `ai_edge_torch.convert(...)` | `AttributeError: module 'ai_edge_torch' has no attribute 'convert'` | package renamed → use `litert_torch.convert` |
| 2 | `torch.export` on wrapper (masks passed) | `GuardOnDataDependentSymNode: Eq(u0, 1)` at `transformer.py:535 _detect_is_causal_mask` | drop provably no-op masks at K=1 |
| 3 | `run_predict.py` (anaconda, xgboost 3.1.2) | **exit 139 SIGSEGV**, no output | — |
| 4 | same with xgboost 2.1.4 | **exit 139 SIGSEGV** inside `torch.load` | unresolved; bypassed Stage-1 |
| 5 | `build_real_input(..., cache_dir=None)` | `AttributeError: 'NoneType' object has no attribute 'mkdir'` | pass an explicit scratch `--cache-dir` |

**Unresolved: Stage-1 checkpoints cannot be loaded on this machine.** Stub
unpickling shows the booster stored as a ~1 MB **legacy binary blob** under
`_Booster.handle` (not the modern JSON/UBJ format). xgboost ≥ 2 rejects that
format by crashing rather than raising, so `except Exception` cannot catch it.
Loading them needs the xgboost 1.x that produced them; the version is not
recorded in the checkpoint. This does not affect Stage-2, which never imports
xgboost — but it **does** block running the full `run_predict.py` pipeline here,
and it is a live risk for the API recipient, since `requirements.txt` pins only
`xgboost>=2.0`, which cannot load the shipped Stage-1 assets.

## 10. Extending to the other 7 pests

The architecture is identical; only `d_in`, `T`, `doy_start/end` and `alert_idx`
differ (27/131/140/12 for BPH; 45/241/60/30 for the rest). The wrapper already
reads all of these from the checkpoint.

1. Generalize `ExpectedConfig` into a per-pest table (or derive it from the ckpt
   and assert `feature_names[alert_idx] == 'alert_tstar'`, which holds for all 8).
2. Parameterize `load_bph()` / `BPH_CKPT` by pest name; the path pattern is
   uniform: `assets/stage2/<pest>/lead_v3_final_checkpoint_run4.pt`.
3. Loop `export_tflite.py` over the 8 pests → `bph_stage2_fp32.tflite` becomes
   `<pest>_stage2_fp32.tflite`.
4. Re-run `validate_parity.py` per pest. The K=1 mask no-op argument is
   shape-independent and holds unchanged.
5. **Prioritize WBPH.** It is the only other pest whose learned output is
   `main`; the remaining six currently return climatology, so converting them
   optimizes a code path production does not answer with. Promoting any of them
   means flipping `recommended_source` in `configs/fallback_policy.yaml` first.

Expect the 45×241 pests to be ~3.4× the MACs of BPH (241/131 × 45/27).

## 11. Next steps to wire TFLite into the API

1. **The .tflite is only the mu head.** A TFLite backend still needs
   `infer/preprocess.build_real_input` for the tensor — normalization stats live
   *inside* the `.pt`, and dispatch channels 12..26 bypass normalization
   (`ckpt.py:137-142`). Ship `norm_mean`/`norm_std` alongside the `.tflite`, or
   keep reading them from the checkpoint.
2. **Add a backend switch** in `run_predict.py` (e.g. `--stage2-backend
   torch|tflite`) rather than replacing the torch path; keep torch as the
   reference and default.
3. **Reconstruct the response fields** the hazard tail no longer provides. mu is
   enough for `mu_doy` and the 95% PI `[round(mu_doy-9.8), round(mu_doy+9.8)]`
   (σ=5 fixed), but confirm nothing downstream consumes the PMF/hazard.
4. **Decide on fp16.** Recommended: 1.6× smaller, error 400× below dynamic's.
5. **Wire a parity gate into CI** — `validate_parity.py` exits non-zero on
   regression.
6. **Fix the Stage-1 xgboost pin independently.** This blocks the API recipient
   today regardless of TFLite: pin the exact xgboost 1.x that wrote the ckpts, or
   re-serialize the boosters to the modern format with `Booster.save_model()`.
7. Weigh the payoff honestly: 1.3× on a 1.3 ms model saves ~0.3 ms/request.
   The real motivations are dependency footprint (dropping the torch runtime) and
   mobile/edge deployment — not server throughput.

## 12. Reproducing

This report documents the BPH **prototype**, whose scripts are no longer tracked
in git. The prototype was superseded by the common pipeline covering all 8 pests
(BPH included); reproduce BPH from there:

```bash
cd tflite_conversion/stage2
../../.venv-tflite/bin/python export_all.py --pests BPH
../../.venv-tflite/bin/python validate_all.py --pests BPH
```

See [`tflite_conversion/stage2/README.md`](../tflite_conversion/stage2/README.md)
for setup and the full command list, and
[`all_pests_stage2_tflite_conversion_report.md`](all_pests_stage2_tflite_conversion_report.md)
for the all-8 results.

Note the prototype and the common pipeline build synthetic inputs differently
(prototype: `alert_doy=176`; common: the middle of each pest's DOY range), so
the *synthetic* mu figures in this report are not reproduced verbatim by the
commands above. The real-data results (§6B) are.

## 13. Repo hygiene

- `api_handoff_transformer/infer/model.py`, `run_predict.py` and every
  checkpoint are **byte-identical** to the shipped zip (verified by SHA-256
  before and after).
- New files only under `tflite_conversion/bph_stage2/` and `docs/`.
- Nothing staged, committed or pushed. `.tflite` artifacts and the unzipped
  package are untracked and were not added to git.
