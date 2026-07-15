# Stage-2 → TensorFlow Lite: all 8 pests

**Date:** 2026-07-15
**Branch:** `feature/tflite-conversion`
**Host:** MacBook, Apple M3 (arm64), macOS 26.5.1, Python 3.12.2
**Outcome:** ✅ 8/8 FP32 and 8/8 FP16 `.tflite` produced; 8/8 wrapper parity bit-exact; **8/8 validated on real data** against the API's own recorded smoke results.

Predecessor: [`bph_stage2_tflite_conversion_report.md`](bph_stage2_tflite_conversion_report.md) (BPH prototype).

---

## 1. Summary

| metric | result |
|---|---|
| FP32 exported | **8 / 8** |
| FP16 exported | **8 / 8** |
| Wrapper parity vs original PyTorch | **8 / 8 bit-exact (0.0)** |
| Synthetic parity within tolerance | **8 / 8** |
| Real-data validated | **8 / 8** |
| Recommended variant | **fp16 for all 8** |
| Conversion failures | none |

## 2. Refactor: one pipeline, no per-pest duplication

The BPH prototype (`tflite_conversion/bph_stage2/`) was generalized into
`tflite_conversion/stage2/`. Nothing is hardcoded per pest — every architectural
value is read from the checkpoint.

| file | role |
|---|---|
| `pest_configs.py` | pest list; expected-config **cross-check** table; LONG filenames; README §10 smoke records; policy offsets |
| `checkpoint.py` | load via untouched `infer/ckpt.py` + validate ckpt against the contract; synthetic input |
| `inference_model.py` | one export-safe mu-only wrapper for all 8 |
| `export_all.py` | torch.export → litert-torch → `.tflite`, sha256, continue-on-failure |
| `validate_all.py` | synthetic parity + wrapper-assumption proofs |
| `validate_real.py` | real preprocessing → mu vs README §10 |
| `benchmark_all.py` | latency/size per pest |
| `summarize.py` | joins stage JSON → `conversion_summary.json` + markdown |

**BPH-specific things that were removed:** a hardcoded `ExpectedConfig` (27/131/
140/270/12), `BPH_CKPT` path constant, `load_bph()`, `alert_doy=176` default,
`BPHStage2InferenceModel` name, and flat `artifacts/bph_stage2_*.tflite` paths
(now `artifacts/<pest>/<pest>_stage2_<variant>.tflite`).

**Refactor verified safe.** Old vs new wrapper on BPH, identical inputs, 18
(seed, alert) cases: max |old − new| = **0.0**, and max |original − new| = **0.0**.
`bph_stage2/` is superseded: its scripts are untracked and only a short pointer
notice remains there.

## 3. Checkpoints are the source of truth

All 8 checkpoints were surveyed before any code was written. `checkpoint.py`
reads every value from the `.pt` and cross-checks it against `pest_configs.EXPECTED`,
against internal consistency, **and against the actual weight shapes**
(`in_proj.in_features`, `self_attn.num_heads/embed_dim`, encoder layer counts,
`head_mu` output width). Any disagreement raises `ConfigMismatch` listing every
discrepancy. No mismatch was found.

| pest | d_in | T | DOY | alert_idx | params |
|---|---:|---:|---|---:|---:|
| BPH | 27 | 131 | [140, 270] | 12 | 163,010 |
| WBPH, bacterial_blight, blast, brown_spot, rice_stem_borer_1/2, sheath_blight | 45 | 241 | [60, 300] | 30 | 163,874 |

Verified identical across all 8: `d_model=48`, `n_head=4`, `n_layers=3`,
`tstar_layers=1`, `pmf_mode=gaussian`, `mu_mode=lead_from_alert`, `σ=5.0`,
`lead=[7,75]`, `dispatch_features_added=True`, `dispatch_channels_raw=True`,
`phenology_bias_head=0`, `use_tstar_scalar_pos=0`, and — importantly — the
**state_dict key set is identical**, so one wrapper class covers all 8.

Also verified per pest: `len(feature_names) == d_in`,
`T == doy_end - doy_start + 1`, `norm_mean/std` dims == `d_in`, and
`feature_names[alert_idx] == 'alert_tstar'` — the last of which **resolves Q2 of
`api_handoff_report.md`**, which had flagged the alert-channel semantic as
unverified, for all 8 pests rather than just one.

`dropout` and `max_len` are **absent** from every checkpoint; the loader's
defaults (0.2 / 400) apply. Dropout is inert in eval mode.

## 4. Wrapper design

`Stage2InferenceModel` re-implements only the mu path:

```
X (1,1,T,d_in) → in_proj → PosEnc → time_encoder(3) → h
  → gather at tstar-1 → z → tstar_pos → tstar_encoder(1) → nan_to_num → z*valid_mask
  → head_mu → mu_logit
alert_doy = X[...,alert_idx].amax(dim=2);  alert_rel = alert_doy - doy_start + 1
lead = 7 + 68*sigmoid(mu_logit);  mu = clamp(alert_rel + lead, 0, T-1)
```

Removed vs the original: the Gaussian PMF → hazard tail (dead compute — the API
discards `forward()`'s return and reads `_last_mu_BK` at `run_predict.py:468-471`),
the strict-check branch, the empty-batch warning branch, and the debug block.
`mu` is returned explicitly.

**Weight safety.** The wrapper copies no weights: it is built from an
already-loaded model and holds references to the *same submodule objects*, so
mis-mapping is structurally impossible. `build_wrapper` additionally asserts
every wrapper parameter is the same object (by `id`) as one of the source
model's. Loading is delegated to the untouched `infer/ckpt.py`, which raises on
missing/unexpected state_dict keys. The wrapper also *refuses* checkpoints with
`phenology_bias_head` or `use_tstar_scalar_pos` enabled rather than silently
ignoring those terms.

Wrapper params (116,833 BPH / 117,697 others) are below the checkpoint totals
because the unused hazard `head` is excluded — it is not on the mu path.

## 5. What blocked `torch.export` (unchanged from the BPH prototype)

1. **Training/debug branches in `infer/model.py`** — `model.py:300`
   `if not lead_loss_mask.any():` (bool() on a tensor → `.item()` → unbacked
   symint → `GuardOnDataDependentSymNode: Eq(u0, 1)`), the boolean-mask index at
   `293-299`, and the `float(...)`/`.item()` debug block at `310-349`.

2. **Inside PyTorch itself** — passing `mask=` to `nn.TransformerEncoder` routes
   through `_detect_is_causal_mask` (`torch/nn/modules/transformer.py:535`),
   which evaluates `bool((mask == causal_comparison).all())`: another `.item()`,
   unreachable by editing our model.

**Resolution (reused):** at K=1 both masks are provably no-ops — the causal mask
is `triu(ones(1,1),1) == [[False]]`, and an all-masked padding mask yields NaN
that `nan_to_num` + `z * valid_mask` collapse to the same zero the unmasked path
gives. K comes from `X.shape` and is static at export, so the branch is not
data-dependent. `validate_all.py` **re-proves this per pest** for `valid=True`
and `valid=False` (all exact), and also re-proves that the chunked time-encoder
loop is a no-op along the batch axis.

## 6. Conversion path

```
wrapper → torch.export → litert_torch.convert → .tflite
```

`ai-edge-torch` 0.7.2 is a **deprecation shim**: the project was renamed to
`litert-torch` and `ai_edge_torch.convert` no longer exists. `litert_torch.convert`
still runs `torch.export` underneath — the same official path under its current
name. TensorFlow is not required. Installs cleanly on Apple Silicon.

Versions (`.venv-tflite`): torch 2.12.1, litert-torch 0.9.1, litert-converter
0.2.0, ai-edge-litert 2.1.5, ai-edge-quantizer 0.7.0, numpy 2.5.1, pandas 3.0.3,
PyYAML 6.0.3. System/anaconda Python untouched.

## 7. Results table

The machine-readable form, including the SHA-256 of every artifact, is
`tflite_conversion/stage2/artifacts/conversion_summary.json`. That directory is
build output and is **not committed** — regenerate it with `summarize.py` after
running the pipeline (see `tflite_conversion/stage2/README.md`).

| pest | ckpt | wrapper | fp32 | fp32 max err (d) | fp32 size | fp32 ms | fp16 | fp16 max err (d) | fp16 size | fp16 ms | real data | recommended |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| BPH | OK | exact | OK | 7.63e-06 | 604,596 | 1.036 | OK | 2.62e-02 | 380,232 | 1.045 | PASS | **fp16** |
| WBPH | OK | exact | OK | 1.53e-05 | 629,172 | 2.682 | OK | 8.47e-04 | 403,328 | 2.621 | PASS | **fp16** |
| bacterial_blight | OK | exact | OK | 0.00e+00 | 629,172 | 2.634 | OK | 1.43e-03 | 403,328 | 2.599 | PASS | **fp16** |
| blast | OK | exact | OK | 0.00e+00 | 629,172 | 2.535 | OK | 1.14e-03 | 403,328 | 2.680 | PASS | **fp16** |
| brown_spot | OK | exact | OK | 0.00e+00 | 629,172 | 2.531 | OK | 6.10e-04 | 403,328 | 2.553 | PASS | **fp16** |
| rice_stem_borer_1 | OK | exact | OK | 1.53e-05 | 629,172 | 2.598 | OK | 1.53e-03 | 403,328 | 2.508 | PASS | **fp16** |
| rice_stem_borer_2 | OK | exact | OK | 1.53e-05 | 629,172 | 2.530 | OK | 1.53e-05 | 403,328 | 2.555 | PASS | **fp16** |
| sheath_blight | OK | exact | OK | 1.53e-05 | 629,172 | 2.526 | OK | 1.83e-04 | 403,328 | 2.505 | PASS | **fp16** |

SHA-256 of every artifact is recorded in `conversion_summary.json` and in each
`artifacts/<pest>/<pest>_stage2_<variant>.meta.json`, alongside the SHA-256 of
the source checkpoint.

### Tolerances

| variant | atol (days) | rationale |
|---|---:|---|
| fp32 | 1e-3 | mu is O(100); float32 ~7 digits → eps-level error ≈1e-5 absolute. Kernel reassociation between ATen and LiteRT costs a few ulps. 1e-3 d (~86 s) is 4 orders below σ=5 d and far below the API's 1-day PI rounding — operationally lossless, still tight enough to catch a real fault. |
| fp16 | 0.1 | fp16 ~3 digits; the bounded-sigmoid lead head amplifies weight rounding across a 68-day span. 0.1 d (~2.4 h) is 50× below σ=5 d and cannot change a rounded response, while being ~9× tighter than the 0.93 d dynamic-range failure this bound exists to reject. |

These are the thresholds requested (fp32 ≤ 1e-3 d, fp16 ≤ 0.1 d); both were kept
unchanged. Observed worst case was 1.53e-05 d (fp32) and 2.62e-02 d (fp16), i.e.
65× and 3.8× inside their bounds.

Validation used 8 seeds per pest, with the alert DOY swept across each pest's own
valid range (`linspace(doy_start+5, doy_end-30)`) — not a single random input.

## 8. Real-data validation — all 8 reproduce the API's recorded results

Driven through the **real** preprocessing chain
(`infer/preprocess.build_real_input`, unmodified) over real daily weather + real
LONG observations. No feature tensor was hand-crafted. Every pest reported
`base_channels_status=real_preprocessing` and `zero_placeholder_used=False`.

| pest | site / year | alert ok | mu_doy (PyTorch) | README §10 | delta | fp32 err (d) | fp16 err (d) |
|---|---|:--:|---:|---:|---:|---:|---:|
| BPH | 33210_56298 / 2004 | ✓ 176 | 234.3240 | 234.32 | 0.0040 | 0.00e+00 | 1.19e-03 |
| WBPH | 33908_67063 / 2011 | ✓ 171 | 245.8158 | 245.82 | 0.0042 | 0.00e+00 | 3.05e-05 |
| bacterial_blight | 30247_65595 / 2010 | ✓ 206 | 268.3161 | 268.32 | 0.0039 | 0.00e+00 | 6.10e-04 |
| blast | 36582_63441 / 2017 | ✓ 125 | 198.0840 | 198.08 | 0.0040 | 1.53e-05 | 2.75e-04 |
| brown_spot | 31522_54338 / 2022 | ✓ 157 | 224.9314 | 224.93 | 0.0014 | 0.00e+00 | 1.31e-03 |
| rice_stem_borer_1 | 35474_56809 / 2018 | ✓ 125 | 184.1720 | 184.17 | 0.0020 | 7.63e-06 | 3.63e-03 |
| rice_stem_borer_2 | 31959_58947 / 2014 | ✓ 186 | 260.9706 | 260.97 | 0.0006 | 0.00e+00 | 0.00e+00 |
| sheath_blight | 35694_60137 / 2004 | ✓ 118 | 192.7225 | 192.72 | 0.0025 | 1.53e-05 | 1.07e-04 |

Every alert DOY matches README §10 exactly, and every `mu_doy` matches to within
that document's 2-decimal rounding (max delta 0.0042). This independently
reproduces the whole shipped smoke record for Stage-2.

**How Stage-1 was bypassed honestly.** `run_predict.py` runs the live Stage-1
XGBoost gate first, which SIGSEGVs here (§10). Stage-1 is not the conversion
target and Stage-2 never imports xgboost. Everything Stage-1 would have produced
— the alert DOY and the 14 dispatch features — already exists as **real
reference values** in the shipped `configs/dispatch/<pest>_dispatch.csv`, which
README §9 records as reproduced by the live gate at ~100% (BPH) / ~99% (others).
`build_real_input` reads that table itself when `alert_tstar_doy=None`. The alert
values were therefore *read from shipped reference data*, not assumed — and they
independently agree with README §10 for all 8.

Data used (this MacBook; none of it is in the repo):

| file | path |
|---|---|
| daily master (1.6 GB) | `/Users/doyoung-gil/연구실/d/1997_2024_RICE_union_all_sites_with_GDD10_since_gs.csv` |
| LONG per pest (8 files) | `/Users/doyoung-gil/Downloads/LONG_by_pest/RICE_LONG_*.csv` |
| representative sites | `/Users/doyoung-gil/연구실/데이터/관측소 메타데이터/representative_site_ids_2002_2024.csv` (not needed for this run) |

Pest → LONG filename mapping was taken from `rice/pests/<pest>/config.py`
(`PATH_OBS`/`TARGET_PEST`), **not** from `infer/batch.py:48 PEST_TO_KOREAN` —
that map sends both `rice_stem_borer_1` and `_2` to `"이화명나방"` and so cannot
disambiguate the `1화기`/`2화기` files.

## 9. Benchmark (Apple M3, 1 thread, 200 iters × 3 repeats)

| pest | T×d_in | elements | torch ms | fp32 ms | fp16 ms | fp32 speedup | fp16 speedup |
|---|---|---:|---:|---:|---:|---:|---:|
| BPH | 131×27 | 3,537 | 1.403 | 1.036 | 1.045 | 1.35× | 1.34× |
| WBPH | 241×45 | 10,845 | 3.280 | 2.682 | 2.621 | 1.22× | 1.25× |
| bacterial_blight | 241×45 | 10,845 | 3.132 | 2.634 | 2.599 | 1.19× | 1.21× |
| blast | 241×45 | 10,845 | 3.018 | 2.535 | 2.680 | 1.19× | 1.13× |
| brown_spot | 241×45 | 10,845 | 3.149 | 2.531 | 2.553 | 1.24× | 1.23× |
| rice_stem_borer_1 | 241×45 | 10,845 | 3.022 | 2.598 | 2.508 | 1.16× | 1.20× |
| rice_stem_borer_2 | 241×45 | 10,845 | 3.071 | 2.530 | 2.555 | 1.21× | 1.20× |
| sheath_blight | 241×45 | 10,845 | 3.026 | 2.526 | 2.505 | 1.20× | 1.21× |

**BPH vs the other 7.** BPH's input is 3,537 elements vs 10,845 (3.07×), because
its season is shorter (131 vs 241 days) and it has 6 base weather channels
instead of 15. It runs ~2.2–2.4× faster in wall clock — less than 3.07× because
per-call overhead (interpreter dispatch, the tstar encoder at K=1, the mu head)
does not scale with T.

**fp16 is not faster than fp32** (1.13–1.34× vs 1.16–1.35×, overlapping). The
weights are stored fp16 but compute is still float; the win is file size, not
speed. Absolute ms drifts with machine load (measured spread up to ±0.4 ms on a
3 ms model), so only ratios should be quoted. `torch.set_num_threads(1)` pins
PyTorch to match LiteRT's single-threaded default interpreter.

## 10. Failures / open problems

| # | issue | status |
|---|---|---|
| 1 | Stage-1 XGBoost ckpts SIGSEGV (exit 139) under xgboost 2.1.4 **and** 3.1.2 | **open** — see below |
| 2 | `ai_edge_torch.convert` missing (`AttributeError`) | resolved — use `litert_torch.convert` |
| 3 | `torch.export` guard failure via `_detect_is_causal_mask` | resolved — drop provably no-op masks at K=1 |
| 4 | `build_real_input(cache_dir=None)` → `AttributeError: 'NoneType' has no attribute 'mkdir'` | resolved — pass an explicit scratch cache dir |

**No pest failed conversion.** Zero export failures, zero validation failures.

**Open — Stage-1 (out of scope here, but a live risk for the API recipient).**
Stub unpickling shows the booster stored as a ~1 MB **legacy binary blob** under
`_Booster.handle`, not the modern JSON/UBJ format. xgboost ≥ 2 rejects it by
crashing rather than raising, so `except Exception` cannot catch it. Loading
needs the xgboost 1.x that produced the ckpts; that version is not recorded in
the checkpoint. `api_handoff_transformer/requirements.txt` pins only
`xgboost>=2.0`, which **cannot load the shipped Stage-1 assets** — so the full
`run_predict.py` pipeline cannot run on a fresh environment today, regardless of
TFLite. Fix by pinning the exact 1.x, or re-serializing the boosters with
`Booster.save_model()` on the GPU server.

## 11. Recommended variant

**fp16 for all 8 pests.** Every pest passes both tolerances; fp16 is ~1.56×
smaller (403,328 vs 629,172 B; 380,232 vs 604,596 B for BPH) at indistinguishable
speed, and its worst real-data error is 3.63e-03 days (~5 minutes) — far below
the API's 1-day PI rounding.

`summarize.py` encodes the rule: prefer fp16 only when its **real-data** error is
under 0.01 d (~15 min), else recommend fp32 and flag the pest. All 8 clear it.

Two caveats worth carrying forward:

- **BPH's fp16 synthetic error (2.62e-02 d) is ~30× the other pests'** (~1e-3 d),
  though its real-data error (1.19e-03 d) is unremarkable. The synthetic input is
  Gaussian noise — out of distribution — which likely exaggerates it, and BPH's
  narrower input may concentrate rounding. It passes comfortably, but BPH is the
  pest to re-check first if fp16 is ever suspected.
- **Deployment value is currently limited to BPH and WBPH.** Only those two have
  `recommended_source: learned_stage2` in `configs/fallback_policy.yaml`; the
  other six return climatology and mark the Transformer output `experimental`.
  Their `.tflite` files are validated and ready, but converting them optimizes a
  path production does not answer with until someone flips that policy.

## 12. Extending / next steps to wire TFLite into the API

1. **The `.tflite` is only the mu head.** A backend still needs
   `infer/preprocess.build_real_input` for the tensor — normalization stats live
   *inside* the `.pt` and the last 15 dispatch channels bypass normalization
   (`ckpt.py:137-142`). Ship `norm_mean`/`norm_std` next to the `.tflite`, or
   keep reading them from the checkpoint.
2. **Add a backend switch** (`--stage2-backend torch|tflite`) rather than
   replacing the torch path; keep torch as the reference default.
3. **Reconstruct response fields.** mu suffices for `mu_doy` and the 95% PI
   `[round(mu_doy-9.8), round(mu_doy+9.8)]` (σ=5 fixed); confirm nothing
   downstream consumes the PMF/hazard.
4. **Wire `validate_all.py` + `validate_real.py` into CI** — both exit non-zero
   on regression.
5. **Fix the Stage-1 xgboost pin** (§10). This blocks the recipient today,
   independent of this work.
6. **Weigh the payoff honestly.** 1.2× on a 3 ms model saves ~0.5 ms/request.
   The real motivations are dependency footprint (dropping the torch runtime) and
   mobile/edge deployment — not server throughput.
7. **K>1 batching** would need a re-export with the masks reinstated and a
   different way around `_detect_is_causal_mask` (e.g. passing `is_causal`
   explicitly), since the K=1 no-op argument no longer holds.

## 13. Repo hygiene

- `api_handoff_transformer/infer/model.py`, `run_predict.py`, and the BPH
  checkpoint are **byte-identical** to the shipped zip (SHA-256 verified before
  and after). All 8 checkpoints are read-only inputs; none was modified or
  re-saved.
- `configs/fallback_policy.yaml` and Stage-1 assets untouched.
- `bph_stage2/` local files preserved on disk; its README is now a pointer here.
- New files under `tflite_conversion/stage2/` and `docs/` only.
- Nothing staged, committed, or pushed. `.venv-tflite/`, `api_handoff_transformer/`
  and all `.tflite` artifacts remain untracked.
