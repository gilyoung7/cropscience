# API / Inference Handoff Report — Rice Pest/Disease 2-Stage Pipeline

**Generated**: 2026-05-26
**Repository**: `/home/gpu4080/research/cropscience`
**Branch**: `feat/stage2-causal-tstar`
**Scope**: Read-only inspection. No files were modified, moved, or deleted.
**Only new files created by this task**: `api_handoff_report.md` (this file) and `api_handoff_manifest_draft.yaml`.

---

## Pre-state — `git status --short` (BEFORE)

```
 M rice/pests/BPH2/config.py
 M rice/pests/WBPH/config.py
 M rice/pests/bacterial_blight/config.py
 M rice/pests/blast/config.py
 M rice/pests/brown_spot/config.py
 M rice/pests/rice_stem_borer_1/config.py
 M rice/pests/rice_stem_borer_2/config.py
 M rice/pests/sheath_blight/features.py
 M rice/scripts/common.py
 M rice/scripts/run_eval.py
 M rice/scripts/run_event_eval.py
 M rice/scripts/run_event_train.py
 M rice/scripts/run_split_seed.py
 M rice/scripts/run_stage1_oof.py
 M rice/scripts/run_stage1b_cascade_v2.py
 M rice/scripts/run_train.py
 M rice/scripts/run_viz_interval.py
 M rice/src/data_pipeline.py
 M rice/src/dataset.py
 M rice/src/model.py
 M rice/src/train_eval.py
?? .ai-workflow/
?? 0.40
?? R
?? lead_variants.txt
?? rice/scripts/build_dispatch_feature_table.py
?? rice/scripts/derived_weather_utils.py
?? rice/scripts/merge_pest_batch_farmin.py
?? rice/scripts/phase1_dispatch_sample_grid_report.py
?? rice/scripts/phase2_dispatch_mu_diag.py
?? rice/scripts/phase_a0_missing_audit.py
?? rice/scripts/phase_a_dispatch_fill_audit.py
?? rice/scripts/phase_a_phenology_variance.py
?? rice/scripts/phase_b_canonical_summary.py
?? rice/scripts/phase_b_climatology_baseline.py
?? rice/scripts/phase_b_lead_probe.py
?? rice/scripts/phase_b_stage1_lead_canonical_table.py
?? rice/scripts/phase_b_stage1_operational_review.py
?? rice/scripts/phase_b_stage1_pptx_report.py
?? rice/scripts/phase_b_stage2_baseline_cross_split_summary.py
?? rice/scripts/phase_b_stage2_mu_distribution_diag.py
?? rice/scripts/phase_b_stage2_offset_selector_diagnostic.py
?? rice/scripts/phase_b_stage2_offset_selector_v2_ranking.py
?? rice/scripts/phase_b_stage2_selector_cross_split_driver.py
... [+ many other phase_* untracked scripts]
?? scripts/run_viz_selector_all_pests_years.sh
... [+ several stage2 output trees, viz dirs, sample_grid CSVs, logs]
```

(Full output captured in Appendix. Many `M` files = staged work; many `??` files = new untracked work that came in during this project.)

---

## A. Repository structure summary

```
/home/gpu4080/research/cropscience/
├── rice/
│   ├── configs/                      # base config (paths, season, hyperparams)
│   ├── pests/                        # per-pest config + features.py (Korean names)
│   │   ├── BPH/ WBPH/ bacterial_blight/ blast/ brown_spot/
│   │   ├── rice_stem_borer_1/ rice_stem_borer_2/ sheath_blight/
│   │   ├── BPH2/  (excluded from current pipeline — short history)
│   │   └── _template/
│   ├── src/                          # model + dataset + train_eval + pipeline
│   │   ├── data_pipeline.py          # daily CSV load, rolling features, pheno merge
│   │   ├── dataset.py                # season slicing, nowcast samples, split
│   │   ├── model.py                  # HazardTransformer + HierarchicalCausalHazardTransformer
│   │   ├── train_eval.py             # interval NLL, gaussian PMF losses, eval
│   │   ├── pest_resolver.py          # pest → config router
│   │   ├── ckpt_schema.py            # ckpt versioning
│   │   └── labels.py                 # interval/right/left censoring helpers
│   ├── scripts/                      # ~115 scripts: phase_*, run_*, *_utils
│   │   ├── run_event_train.py        # Stage-1 XGBoost (event_xgb) trainer
│   │   ├── run_event_eval.py         # Stage-1 alert builder, tau picker, temperature scaler
│   │   ├── run_train.py              # Stage-2 trainer entry point (~2400 LOC, all knobs)
│   │   ├── phase_s5_train.py         # Stage-2 lead_v3 pilot+final driver
│   │   ├── phase_r_oracle_iou.py     # Stage-2 forward + sample_grid generator (canonical eval)
│   │   ├── build_dispatch_feature_table.py   # builds per-(site,year) gate alert+features CSV
│   │   ├── stage1_confidence_utils.py # appends 14 dispatch features + 1 missing-indicator to X
│   │   ├── site_history_utils.py     # 11 history channels (per-site prev-year stats)
│   │   ├── phenology_utils.py        # phenology channels (best_suitability, etc.)
│   │   ├── derived_weather_utils.py  # VPD / 28d aggregates / streaks (7-ch extension)
│   │   ├── phase_b_climatology_baseline.py  # per-pest mean_mid baseline generator
│   │   ├── phase_b_stage2_offset_selector_v2_ranking.py  # offset selector (LightGBM)
│   │   ├── phase_b_stage2_selector_cross_split_driver.py # 8×3 driver
│   │   └── run_viz_interval_selector.py     # W&B viz w/ selector
│   ├── outputs_stage1/
│   │   └── batch_rolling/             # canonical 3-split × 8-pest × 3-run XGBoost
│   │       ├── _summary/              # cross-pest aggregates
│   │       ├── <pest>/run{0,1,2}/split{1,2,3}_v{val}_t{test}/
│   │       │   ├── A/ckpt/event_xgb_w28_lead14-45_A.pt    # A_baseline gate
│   │       │   ├── D/ckpt/event_xgb_w28_lead14-45_D.pt    # D_history gate
│   │       │   ├── A/useful_pareto/useful_sweep_A.csv     # (k, tau) sweep + test metrics
│   │       │   ├── D/useful_pareto/useful_sweep_D.csv
│   │       │   └── group_tau/group_tau_hybrid_summary.json # dispatch gate per target
│   │       └── _summary/pest_batch_farmin_all.csv         # 8×3×3×3 (pest×run×split×method) rows
│   ├── outputs_stage2_<pest>_uncond/          # Stage-2 uncond (hazard) backbones (split3)
│   ├── outputs_stage2_<pest>_uncond_split{1,2}/  # strict-mode per-split backbones
│   ├── outputs_stage2_batch_2024_bestgate/    # canonical split3 (val=2023 / test=2024) per-pest
│   ├── outputs_stage2_batch_2022_baseline/    # split1 (val=2021 / test=2022) per-pest
│   ├── outputs_stage2_batch_2023_baseline/    # split2 (val=2022 / test=2023) per-pest
│   └── outputs_stage2_selector_cross_split/   # V2 ranking selector outputs (8×3 cells)
└── scripts/                          # project-root shell drivers
    ├── run_stage1_batch_pests.sh
    ├── run_stage2_split_pest_best_gate_batch.sh        # split-parameterized bestgate runner
    ├── run_stage2_all_pests_baseline_2022_2023.sh      # overnight 8×2 driver
    ├── run_stage2_dispatch_sample_grid.sh              # phase_r_oracle_iou wrapper
    ├── make_aux_sample_grids_sheath_blight.sh          # aux val=2021/2022 grids for sheath_blight
    └── run_viz_selector_all_pests_years.sh             # W&B viz driver
```

External data path (NOT in repo, referenced via `rice/configs/base.py`):
- `PATH_DAILY = /home/gpu4080/ygdata/rice/1997_2024_RICE_union_all_sites_with_GDD10_since_gs.csv`
- `PATH_OBS   = /home/gpu4080/ygdata/rice/LONG_by_pest/RICE_LONG_*.csv`

→ **API must mount or replicate this raw data path** (or accept a thin upstream that produces equivalent features at the same dim).

---

## B. Stage-1 asset candidates

### B.1 Per-pest, per-split, per-run ckpt triplet (canonical location)

For each (pest, run ∈ {0,1,2}, split ∈ {split1_v2021_t2022, split2_v2022_t2023, split3_v2023_t2024}):
```
rice/outputs_stage1/batch_rolling/<pest>/run<R>/<SPLIT>/
├── A/ckpt/event_xgb_w28_lead14-45_A.pt          # XGBoost A_baseline (no history)
├── D/ckpt/event_xgb_w28_lead14-45_D.pt          # XGBoost D_history (+11 history channels)
├── A/useful_pareto/useful_sweep_A.csv           # (k, tau) Pareto sweep + per-row val+test metrics
├── D/useful_pareto/useful_sweep_D.csv           # same for D
└── group_tau/group_tau_hybrid_summary.json      # dispatch (group_tau) selections per target {R≥0.85, 0.88, 0.90}
```

**Ckpt content (verified for sheath_blight split3 run0)**: PyTorch dict with these inference-relevant keys:
- `model_type`, `event_model` (XGBoost model bytes)
- `feature_cols` (15 raw weather/pheno cols)
- `feature_names` (30 cols incl. nowcast/history/position extensions)
- `nowcast_window=28`, `nowcast_stride`, `nowcast_label_horizon`
- `add_tstar_position_feature=True`
- `history_feature_names` (11 cols), `history_feature_dim=11` — only for `D` ckpt
- `doy_start=60`, `doy_end=300`, `T=241`
- `run`, `pest`, `split_seed`, `split_mode`, `year_max`

→ **Sufficient for inference IF**:
- raw daily preprocessing produces the 15 base features (same Korean column names)
- `add_rolling_features` produces the 15 rolling features ([rice/src/data_pipeline.py:97](rice/src/data_pipeline.py#L97))
- `site_history_utils.append_history_to_samples` produces the 11 history channels (D only)
- The corresponding `(k, tau)` from the selection CSV is applied

### B.2 Per-pest selected (method, run, k, tau) — split3 canonical

[`rice/outputs_stage2_batch_2024_bestgate/_summary/stage1_gate_selection_split3_2024.csv`](rice/outputs_stage2_batch_2024_bestgate/_summary/stage1_gate_selection_split3_2024.csv) (val-only selection):

| pest | selected_method | selected_run | k | tau | tau_no | tau_with |
|---|---|---:|---:|---:|---:|---:|
| BPH | D_history | 2 | 3 | 0.600 | – | – |
| WBPH | dispatch_group_tau | 2 | 3 | – | 0.725 | 0.700 |
| bacterial_blight | D_history | 0 | 1 | 0.575 | – | – |
| blast | D_history | 0 | 3 | 0.600 | – | – |
| brown_spot | D_history | 2 | 3 | 0.500 | – | – |
| rice_stem_borer_1 | A_baseline | 2 | 2 | 0.575 | – | – |
| rice_stem_borer_2 | D_history | 2 | 3 | 0.575 | – | – |
| sheath_blight | dispatch_group_tau | 0 | 3 | – | 0.550 | 0.550 |

- For `A_baseline` / `D_history`: single tau (no history dispatch). Alert = first day where smoothed probability ≥ τ for k consecutive days.
- For `dispatch_group_tau`: hybrid two-tau rule (`tau_no` for no-history sites, `tau_with` for sites with prior history). Defined in [`rice/scripts/phase_t_group_tau_hybrid.py`](rice/scripts/phase_t_group_tau_hybrid.py).

Equivalent canonical PPT table (test-side metrics + lead_mean/median) at:
[`rice/outputs_stage2_batch_2024_bestgate/_summary/stage1_best_gate_canonical_ppt.csv`](rice/outputs_stage2_batch_2024_bestgate/_summary/stage1_best_gate_canonical_ppt.csv).

Same selection CSV exists per split:
- `rice/outputs_stage2_batch_2022_baseline/_summary/stage1_gate_selection_split1_2022.csv`
- `rice/outputs_stage2_batch_2023_baseline/_summary/stage1_gate_selection_split2_2023.csv`

### B.3 Stage-1 inference code

| file | purpose |
|---|---|
| [rice/scripts/run_event_train.py](rice/scripts/run_event_train.py) | Defines `EventTransformer` (PyTorch class) used **only** as baseline; the production gate is XGBoost in the ckpt's `event_model`. Also contains `build_tabular_from_samples`, `build_nowcast_samples`, `make_event_labels`. |
| [rice/scripts/run_event_eval.py](rice/scripts/run_event_eval.py) | Production-relevant gate logic: `apply_temperature`, `best_tau_by_target`, `compute_t_alert_start`, `build_alert_rows`. |
| [rice/scripts/build_dispatch_feature_table.py](rice/scripts/build_dispatch_feature_table.py) | For Stage-2 input: builds per-(site, year) row containing `alert_tstar` + 14 dispatch features for the selected gate. Required when calling Stage-2 lead_v3 inference. |
| [rice/scripts/stage1_confidence_utils.py](rice/scripts/stage1_confidence_utils.py) | Defines the 14 dispatch feature schema; `load_dispatch_feature_table`, `append_dispatch_confidence_to_samples`. |
| [rice/scripts/site_history_utils.py](rice/scripts/site_history_utils.py) | 11 site-history channels for D_history gate. |
| [rice/scripts/run_stage1_oof.py](rice/scripts/run_stage1_oof.py) | OOF prediction generator (training-time; not used at inference). |
| [rice/scripts/phase_t_group_tau_hybrid.py](rice/scripts/phase_t_group_tau_hybrid.py) | `dispatch_group_tau` two-tau hybrid rule logic. |

### B.4 Stage-1 sufficiency / missing

| item | status |
|---|---|
| Per-pest, per-split A.pt + D.pt + summary.json | ✅ all 8 pests × 3 splits × 3 runs present (verified via filesystem) |
| Per-pest selected (method, run, k, tau) | ✅ in `stage1_gate_selection_split*_*.csv` (one per split) |
| Raw feature → ckpt feature pipeline | ✅ via `rice/src/data_pipeline.py` (load_daily + add_rolling_features) + site_history_utils + phenology_utils |
| Temperature scaling parameters | ⚠️ check ckpt — `temperature` value is fitted at eval time via `fit_temperature_grid`. **TODO**: confirm whether `temperature` is stored in the saved ckpt or recomputed |
| Alert window aggregation logic | ✅ `run_event_eval.build_alert_rows` (k-consecutive smoothing + tau) |
| Stage-1 model class for XGBoost reload | ⚠️ ckpt stores `event_model` (xgboost bytes). API must `xgb.Booster()` and `load_model(bytes)`. **TODO**: confirm exact byte format |

---

## C. Stage-2 asset candidates

### C.1 Per-pest final ckpt (split3-canonical = test=2024)

```
rice/outputs_stage2_batch_2024_bestgate/<pest>/lead_v3_final/ckpt/checkpoint_run4.pt
```

All 8 pests present (verified). Per-split variants also exist:
- `rice/outputs_stage2_batch_2022_baseline/<pest>/lead_v3_final/ckpt/checkpoint_run4.pt`  (split1, test=2022)
- `rice/outputs_stage2_batch_2023_baseline/<pest>/lead_v3_final/ckpt/checkpoint_run4.pt`  (split2, test=2023)
- `rice/outputs_stage2_batch_2024_bestgate/<pest>/lead_v3_final/ckpt/checkpoint_run4.pt`  (split3, test=2024)

→ Production deploy should use the **most-recent split (split3 = test=2024)** by default.

### C.2 Ckpt contents (verified for sheath_blight split3)

Inference-relevant keys (full list ~150):

| key | value | meaning |
|---|---|---|
| `d_in` | 45 | total input channel count = 15 raw + 15 dispatch + 15(? incl miss) — see C.5 |
| `d_model` | 48 | transformer hidden dim |
| `T` | 241 | season-index length (`doy_end - doy_start + 1`) |
| `doy_start` | 60 | DOY=60 ≈ Mar 1 |
| `doy_end` | 300 | DOY=300 ≈ Oct 27 |
| `stage2_pmf_mode` | `'gaussian'` | μ-head + fixed σ Gaussian PMF |
| `stage2_pmf_sigma` | 5.0 | **fixed σ** in days for the Gaussian |
| `stage2_pmf_mu_mode` | `'lead_from_alert'` | μ_DOY = alert_tstar + bounded sigmoid lead (NOT absolute DOY) |
| `stage2_pmf_lead_min` | 7.0 | lead lower bound (days post alert) |
| `stage2_pmf_lead_max` | 75.0 | lead upper bound |
| `stage2_pmf_alert_tstar_feat_idx` | 30 | which channel of X holds alert_tstar (DOY) |
| `stage2_dispatch_features_added` | True | model expects the 15-ch dispatch block in X |
| `stage2_dispatch_feature_csv` | `rice/outputs_stage2_batch_2024_bestgate/sheath_blight/gate_dispatch_group_tau_R088_features_per_sy.csv` | per-(site, year) Stage-1 alert+feature lookup |
| `stage2_nowcast_window` | 28 | 28-day rolling window per sample |
| `stage2_nowcast_stride` | 1 | sample stride |
| `stage2_nowcast_require_tstar_before_L` | True | drop event rows where t* ≥ L |
| `feature_cols` | list(len=15) | base raw feature column names |
| `feature_names` | list(len=45) | full ordered channel names (incl. extensions + dispatch + history) |
| `norm_mean` | np.ndarray(45,) | input normalization mean |
| `norm_std` | np.ndarray(45,) | input normalization std |
| `trained_states[0].state_dict` | dict | PyTorch model weights |

→ Single ckpt **fully self-describes** model architecture + input schema (feature_cols, norm_mean/std, d_in) + Gaussian PMF parameters (σ=5, lead_min/max, alert_tstar_feat_idx). Inference only needs the ckpt + the dispatch CSV referenced in `stage2_dispatch_feature_csv` + raw daily preprocessing.

### C.3 Stage-2 inference / forward code

| file | purpose |
|---|---|
| [rice/src/model.py](rice/src/model.py) | `HazardTransformer` + `HierarchicalCausalHazardTransformer` model classes. The `forward()` method handles mu_mode=lead_from_alert: μ_DOY = alert_rel + lead_min + (lead_max − lead_min) × sigmoid(head(z)) (line 235-353). Gaussian PMF: `_gaussian_pmf_from_mu` + `_pmf_to_hazard` (lines 44-69). |
| [rice/src/train_eval.py](rice/src/train_eval.py) | `interval_nll_per_sample`, `asymmetric_mu_loss` (training loss). For inference: `hazard_to_pmf_cdf_logS`, `shortest_mass_interval_1d`, `quantile_from_cdf_1d`, `overlap_metrics`. |
| [rice/src/dataset.py](rice/src/dataset.py) | `build_stage2_nowcast_samples` (line 550), `group_stage2_samples_by_site_year` (line 740), `GroupedIntervalEventDataset` (line 760). Compute_norm_stats stored in ckpt so inference applies the stored norm. |
| [rice/scripts/run_train.py](rice/scripts/run_train.py) | Model instantiation, warm-start loading (lines 1300-1340 — full attr setup pattern). Inference can crib these lines. |
| [rice/scripts/phase_r_oracle_iou.py](rice/scripts/phase_r_oracle_iou.py) | **The canonical "load ckpt → forward → output mu per (sample, offset)" path** (`build_stage2_row_map`, line 509). 95% PI = `±1.96σ` rounded. Already supports `--stage2_dispatch_feature_csv_override` for swap-in scenarios. |
| [scripts/run_stage2_dispatch_sample_grid.sh](scripts/run_stage2_dispatch_sample_grid.sh) | Bash wrapper of phase_r_oracle_iou.py producing `lead_v3_test_sample_grid.csv`. |

### C.4 Sample_grid output schema (Stage-2 prediction format)

`<root>/<pest>/lead_v3_test_sample_grid.csv` (one row per (sample_id, offset)):

| column | meaning |
|---|---|
| `model` | e.g. `sheath_blight_lead_v3` |
| `sample_id` | `<site_id>-<year>` |
| `site`, `year` | identifier |
| `t_star_doy` | Stage-1 alert DOY |
| `true_event_doy` | (L+R)/2 — for eval only |
| `L`, `R` | true interval bounds (last non-event DOY, first event DOY) |
| `mu` | predicted μ (in DOY) at this offset |
| `sigma` | 5.0 |
| `offset` | activation offset in {7, 14, 21, 30, 45, 60} (coarse) |
| `iou_matched`, `matched` | for eval only |
| `alert_tstar` | = `t_star_doy` |
| `dispatch_branch`, `with_history` | dispatch metadata |
| `A_score_at_alert`, `D_score_at_alert` | Stage-1 scores at alert |
| `score_margin`, `dispatch_score_at_alert`, `dispatch_tau_used`, `score_over_tau_margin` | dispatch features |
| `recent_14d_mean_score`, `recent_28d_mean_score`, `score_above_tau_streak`, `score_rolling_slope_14d`, `p_mean_so_far_at_alert` | extra features |

**At inference time:**
1. mu (DOY) is what the API returns
2. 95% PI = [round(μ − 1.96 × 5), round(μ + 1.96 × 5)] = `[μ − 9.8, μ + 9.8]` days
3. Selector chooses an offset in {7, 14, 21, 30, 45, 60}; the μ at that offset is the operational answer

### C.5 d_in=45 channel layout (decoded from feature_names)

Per [rice/scripts/stage1_confidence_utils.py](rice/scripts/stage1_confidence_utils.py):
- 0..14  : 15 base+rolling features (matching Stage-1 feature_cols)
- 15..29 : 15 phenology/history extension channels (run-specific, ≤15 ext)
- 30..43 : 14 dispatch features (DISPATCH_FEATURE_NAMES)
- 44     : 1 `dispatch_feature_missing` indicator

Note: `stage2_pmf_alert_tstar_feat_idx=30` ⇒ index 30 is `alert_tstar` (first dispatch channel = A_score_at_alert? — **verify**). The model reads `X[..., alert_idx].amax(dim=2)` to get per-(B,K) absolute alert DOY ([model.py:255](rice/src/model.py#L255)). **TODO**: confirm channel-30 semantic.

### C.6 Stage-2 sufficiency / missing

| item | status |
|---|---|
| Per-pest final ckpt | ✅ all 8 pests × 3 splits present |
| Architecture self-described in ckpt | ✅ d_in, d_model, T, doy bounds, σ, μ-head config all stored |
| Feature normalization | ✅ `norm_mean` + `norm_std` in ckpt |
| Stage-1 → Stage-2 dispatch CSV (input) | ✅ co-located: `<root>/<pest>/gate_<METHOD>_R088_features_per_sy.csv` (1 per pest, method varies per pest as selected) |
| Inference model code | ✅ `rice/src/model.py` + `rice/src/dataset.py` + `phase_r_oracle_iou.py` |
| 95% PI formula | ✅ hard-coded `1.96 × σ` ([phase_r_oracle_iou.py:736](rice/scripts/phase_r_oracle_iou.py#L736)) |
| Lightweight API harness (no training imports) | ❌ **missing**. All current entry points (phase_r_oracle_iou, etc.) drag in the training pipeline. Need a slim `infer_stage2(features, ckpt)` function. |

---

## D. Selector asset candidates

### D.1 Selector code

| file | purpose |
|---|---|
| [rice/scripts/phase_b_stage2_offset_selector_v2_ranking.py](rice/scripts/phase_b_stage2_offset_selector_v2_ranking.py) | **V2 ranking selector** — trains LightGBM regressor + ranker + classifier on val sample_grid, picks per-sample best offset on test. |
| [rice/scripts/phase_b_stage2_selector_cross_split_driver.py](rice/scripts/phase_b_stage2_selector_cross_split_driver.py) | 8 pests × 3 years loop driver. |
| [rice/scripts/phase_b_stage2_offset_selector_diagnostic.py](rice/scripts/phase_b_stage2_offset_selector_diagnostic.py) | V1 (rules + decision tree). Best learned variant for some cells. |

### D.2 Candidate offsets

```python
COARSE_OFFSETS = [7, 14, 21, 30, 45, 60]            # operational, matches sample_grid
DENSE_OFFSETS  = list(range(1, 76))                  # 1..75, via μ linear interp (diagnostic)
```

API should use **coarse** by default (matches what the Stage-2 model is actually evaluated at).

### D.3 Per-pest, per-year selected per-sample offsets

```
rice/outputs_stage2_selector_cross_split/<year>_<pest>/
├── v2_per_sample_test_selections.csv     # per-sample × selector-variant: chosen offset + IoU + score
├── v2_per_candidate_val.csv              # val candidate rows (features × offsets) used for training
├── v2_test_results_summary.csv           # selector × overall test IoU (in policy-space)
└── v2_summary_for_ppt.txt
```

All 24 (year, pest) cells present (verified).

Aggregated cross-split table:
[`rice/outputs_stage2_selector_cross_split/selector_by_pest_year.csv`](rice/outputs_stage2_selector_cross_split/selector_by_pest_year.csv)

Columns: `year, pest, fixed_iou, best_selector_name, best_selector_iou, best_climatology_iou, oracle_iou_coarse, selector_minus_fixed, selector_minus_clim, beats_fixed, beats_clim, sel__v2_regressor_coarse, sel__v2_regressor_dense, sel__v2_ranker_coarse, sel__v2_ranker_dense, sel__v2_classifier_coarse, sel__v2_classifier_dense, sel__v1_bin_rule_*`

### D.4 Selector features (per-sample, offset-independent)

From [phase_b_stage2_offset_selector_v2_ranking.py:31](rice/scripts/phase_b_stage2_offset_selector_v2_ranking.py#L31):
```python
SAMPLE_FEATURES = [
    "alert_tstar",
    "A_score_at_alert", "D_score_at_alert", "score_margin",
    "dispatch_score_at_alert", "dispatch_tau_used", "score_over_tau_margin",
    "recent_14d_mean_score", "recent_28d_mean_score",
    "score_above_tau_streak", "score_rolling_slope_14d",
    "p_mean_so_far_at_alert", "with_history",
]
```
plus offset-dependent features added per candidate row: `offset`, `pred_mu`, `pred_lead = pred_mu - alert_tstar`, `mu_minus_clim_mid`, `tstar_minus_clim_mid`, `dispatch_branch_is_D`.

### D.5 Best per-pest/year selector (operational pick)

From [selector_by_pest_year.csv](rice/outputs_stage2_selector_cross_split/selector_by_pest_year.csv) (`best_selector_name` column):

| pest | 2022 | 2023 | 2024 |
|---|---|---|---|
| BPH | v2_ranker_dense | v2_classifier_coarse | v2_regressor_dense |
| WBPH | v2_classifier_dense | v2_ranker_coarse | v2_ranker_dense |
| bacterial_blight | v2_regressor_coarse | v2_regressor_dense | v1_bin_rule[...] |
| blast | v2_regressor_coarse | v2_ranker_coarse | v2_ranker_dense |
| brown_spot | v1_bin_rule[...] | v1_bin_rule[...] | v2_regressor_dense |
| rice_stem_borer_1 | v1_bin_rule[...] | v2_ranker_coarse | v2_ranker_dense |
| rice_stem_borer_2 | v2_regressor_coarse | v2_regressor_coarse | v2_classifier_coarse |
| sheath_blight | v1_bin_rule[...] | v2_ranker_dense | v2_regressor_dense |

→ **No single selector wins universally**. For API: either (a) use v2_regressor_dense (most-common winner) for simplicity, or (b) maintain per-pest best lookup.

### D.6 Selector sufficiency / missing — ⚠️ CRITICAL

| item | status |
|---|---|
| Per-sample selected offset (historical test data) | ✅ in `v2_per_sample_test_selections.csv` |
| **Persisted LightGBM model files** | ❌ **NOT persisted**. The selector is trained on-the-fly per (pest, year) call inside `phase_b_stage2_offset_selector_v2_ranking.py`. For NEW (unseen) site-years, the API has 3 options: (1) **re-train selector at deploy time** with fixed seed (deterministic but loads val sample_grid CSV); (2) **modify the V2 script to dump `lgb.Booster.save_model()` of the chosen selector per (pest, year)**; (3) **use the V1 bin_rule** which is interpretable + reproducible from edges+bin→offset mapping ([phase_b_stage2_offset_selector_v2_ranking.py:256](rice/scripts/phase_b_stage2_offset_selector_v2_ranking.py#L256)). |
| V1 bin_rule parameters | ⚠️ computed at runtime, edges + bin→offset mapping not persisted as a standalone file. Easy to extract & dump. |
| Candidate offset list | ✅ hard-coded `[7, 14, 21, 30, 45, 60]` |
| Oracle offset | ✅ marked LEAKY in `v2_test_results_summary.csv`. **Must not be used for API**. |
| `best_selector_name` lookup table | ✅ in `selector_by_pest_year.csv` (per pest×year) |

---

## E. Climatology / fallback candidates

### E.1 Per-pest train stats

```
<root>/<pest>/climatology_train_stats.csv
```

All 8 pests × 3 splits present (verified). Schema (single row):
```
pest, n_train_total_sy, n_train_alerted_sy, n_train_alerted_interval_sy,
mean_L, mean_R, mean_mid, median_L, median_R, median_mid, std_L, std_R, std_mid
```

Example (sheath_blight, split3):
```
mean_L=196.32  mean_R=211.01  mean_mid=203.66
median_L=197   median_R=213   median_mid=205
std_L=17.95    std_R=18.80    std_mid=18.33
```

### E.2 Climatology baseline sample_grids (per offset, for IoU eval)

```
<root>/<pest>/climatology_mean_L_test_sample_grid.csv
<root>/<pest>/climatology_mean_mid_test_sample_grid.csv
<root>/<pest>/climatology_mean_R_test_sample_grid.csv
```

These predict `μ = climatology_mean_{L,mid,R}` (constant per pest) for every sample. Best of the 3 per pest is the `best_climatology_*` in selector summaries. For most pests `clim_mean_mid` wins; check `best_climatology_name` column in `selector_by_pest_year.csv` for the exact winner per (pest, year).

### E.3 Climatology generator code

[`rice/scripts/phase_b_climatology_baseline.py`](rice/scripts/phase_b_climatology_baseline.py) — produces `climatology_train_stats.csv` + the 3 sample_grid CSVs from raw observation labels (train years).

### E.4 Fallback policy (recommended for API)

Based on results in [`rice/outputs_stage2_selector_cross_split/selector_summary_for_ppt.txt`](rice/outputs_stage2_selector_cross_split/selector_summary_for_ppt.txt):

| pest | learned beats clim (avg over 3 years) | recommended fallback |
|---|---|---|
| BPH | 3/3 ✅ | learned selector (clim=0 for BPH) |
| WBPH | 2/3 ✅ | learned selector |
| sheath_blight | 0/3 ❌ | **climatology** |
| blast | 1/3 | **climatology** (averages better) |
| brown_spot | 1/3 | **climatology** |
| bacterial_blight | 0/3 ❌ | **climatology** |
| rice_stem_borer_1 | 0/3 ❌ | **climatology** |
| rice_stem_borer_2 | 0/3 ❌ | **climatology** |

→ API should support a `policy = {learned, climatology}` switch per pest, with the above defaults.

### E.5 Climatology sufficiency / missing

| item | status |
|---|---|
| Per-pest train stats CSV | ✅ |
| Per-pest sample_grid CSVs (mean_L / mid / R) | ✅ |
| Best clim variant per pest (mean_L vs mid vs R) | ✅ in `selector_by_pest_year.csv` `best_climatology_name` |
| Fallback policy spec | ❌ informal — needs to be coded into the API config |

---

## F. Preprocessing / feature-generation code

### F.1 Daily weather preprocessing

| file | role |
|---|---|
| [rice/src/data_pipeline.py](rice/src/data_pipeline.py) `load_daily` (line 16) | Loads CSV, enforces required Korean column names |
| [rice/src/data_pipeline.py](rice/src/data_pipeline.py) `add_rolling_features` (line 97) | 7-day rolling (rain_7d_sum/days, tmean/tmax/tmin_7d_*, rh_7d_mean, sun/rad_7d_sum, trange/trange_7d_mean) |
| [rice/src/data_pipeline.py](rice/src/data_pipeline.py) `load_daily_preprocessed` (line 174) | Cached version (hash key for invalidation, stored under `rice/outputs/cache/`) |
| [rice/src/data_pipeline.py](rice/src/data_pipeline.py) `make_daily_feature_frame` (line 412) | Assembles the daily feature DataFrame |
| `IMPUTE_POLICY="ffill_bfill_interpolate_fill0"` (configs/base.py) | Missing-fill policy |

### F.2 Nowcast feature / sample generation

| file | role |
|---|---|
| [rice/src/dataset.py](rice/src/dataset.py) `build_stage2_nowcast_samples` (line 550) | 28-day rolling window samples per (site, year, t*) for Stage-2 |
| [rice/src/dataset.py](rice/src/dataset.py) `group_stage2_samples_by_site_year` (line 740) | Groups samples by site-year for grouped batching |
| [rice/scripts/run_event_train.py](rice/scripts/run_event_train.py) `build_nowcast_samples` (line 291) | Stage-1 nowcast variant |
| [rice/scripts/run_event_train.py](rice/scripts/run_event_train.py) `build_tabular_from_samples` (line 226) | Stage-1 XGBoost tabularization (flattens window) |

### F.3 History feature generation

[`rice/scripts/site_history_utils.py`](rice/scripts/site_history_utils.py) — 11 channels:
- **Static (broadcast)**: prev_year_L_doy_at_site, prev_year_event_at_site, site_avg_L_doy_recent3y, years_since_last_event_at_site, n_events_recent5y_at_site, prev_year_L_miss, site_avg_L_recent3y_miss
- **Dynamic (per-day)**: days_to_prev_year_L, abs_days_to_prev_year_L, days_to_site_avg_L_recent3y, abs_days_to_site_avg_L_recent3y

Policies: `rolling` (operational; uses any year < target) vs `strict_train` (capped at train year).

→ **D_history Stage-1 model and Stage-2 lead_v3 use `rolling`**.

### F.4 Phenology / GDD feature generation

| file | role |
|---|---|
| [rice/scripts/phenology_utils.py](rice/scripts/phenology_utils.py) | Phenology channels (best_suitability, best_months, offset_days, window_idx) |
| [rice/src/data_pipeline.py](rice/src/data_pipeline.py) `merge_pheno_daily_ffill` (line 344) | Joins phenology table to daily, forward-fill |
| `add_pheno_to_samples` (in dataset.py / build_samples_season) | Adds 3 base phenology channels: days_since_growing_start, days_until_growing_end, is_growing |
| GDD10 | Already in raw daily CSV as `GDD10_since_gs` column; loaded by `load_daily` |
| [rice/scripts/derived_weather_utils.py](rice/scripts/derived_weather_utils.py) | 7 derived channels (VPD 7d mean/max, 28d aggregates, streaks); appended to base_X for select runs |

### F.5 Stage-1 → Stage-2 dispatch features

[`rice/scripts/stage1_confidence_utils.py`](rice/scripts/stage1_confidence_utils.py) — 14 channels:
```
A_score_at_alert, D_score_at_alert, score_margin,
dispatch_score_at_alert, dispatch_tau_used, score_over_tau_margin,
recent_14d_mean_score, recent_28d_mean_score, score_above_tau_streak,
score_rolling_slope_14d, p_mean_so_far_at_alert,
... (3 more) ...
+ 1 dispatch_feature_missing  → 15 channels total
```

Generated by [`rice/scripts/build_dispatch_feature_table.py`](rice/scripts/build_dispatch_feature_table.py) given a Stage-1 ckpt + (k, tau, method) → per-(site, year) row.
Causal-fill mode: pre-alert rows get zeros + `missing=1`; post-alert rows get the actual features.

### F.6 Preprocessing sufficiency / missing

| item | status |
|---|---|
| Daily CSV loader + 7-day rollings | ✅ |
| Caching layer | ✅ (`rice/outputs/cache/` with hash key — needs writable mount or fresh recompute) |
| Phenology join | ✅ (depends on per-site phenology table referenced in pest config) |
| History generator (rolling) | ✅ |
| Dispatch feature generator | ✅ |
| **Single end-to-end "raw daily + pest → Stage-2 input X" function** | ❌ scattered across 5 files; needs a thin orchestrator for the API |
| Per-pest Korean column names | ⚠️ daily CSV uses Korean columns (e.g. `일강수량(mm)`); API must preserve them exactly |

---

## G. Suggested `asset_manifest.yaml` draft

Saved as a separate file: [`api_handoff_manifest_draft.yaml`](api_handoff_manifest_draft.yaml).

Key structure:
```yaml
version: 1
preprocessing:
  daily_csv_path: TODO  # external; not in repo
  cache_dir: rice/outputs/cache
  feature_pipeline:
    - rice/src/data_pipeline.py:load_daily
    - rice/src/data_pipeline.py:add_rolling_features
    - rice/scripts/derived_weather_utils.py  # optional, run-specific
    - rice/src/data_pipeline.py:merge_pheno_daily_ffill
    - rice/scripts/site_history_utils.py:append_history_to_samples
    - rice/scripts/stage1_confidence_utils.py:append_dispatch_confidence_to_samples

pests:
  - name: sheath_blight
    pest_dir: rice/pests/sheath_blight
    stage1:
      method: dispatch_group_tau
      run: 0
      split: split3_v2023_t2024
      ckpt_A: rice/outputs_stage1/batch_rolling/sheath_blight/run0/split3_v2023_t2024/A/ckpt/event_xgb_w28_lead14-45_A.pt
      ckpt_D: rice/outputs_stage1/batch_rolling/sheath_blight/run0/split3_v2023_t2024/D/ckpt/event_xgb_w28_lead14-45_D.pt
      dispatch_summary: rice/outputs_stage1/batch_rolling/sheath_blight/run0/split3_v2023_t2024/group_tau/group_tau_hybrid_summary.json
      gate_target: "R>=0.88"
      k: 3
      tau_no: 0.55
      tau_with: 0.55
      temperature: TODO  # confirm with researcher
    stage2:
      ckpt: rice/outputs_stage2_batch_2024_bestgate/sheath_blight/lead_v3_final/ckpt/checkpoint_run4.pt
      dispatch_feature_csv: rice/outputs_stage2_batch_2024_bestgate/sheath_blight/gate_dispatch_group_tau_R088_features_per_sy.csv
      sigma: 5.0
      mu_mode: lead_from_alert
      lead_min: 7
      lead_max: 75
      pi_quantile: 0.95  # ±1.96σ
      coarse_offsets: [7, 14, 21, 30, 45, 60]
    selector:
      v1_offsets_lookup: rice/outputs_stage2_selector_cross_split/2024_sheath_blight/v2_per_sample_test_selections.csv
      best_selector_name: v2_regressor_dense  # per selector_by_pest_year.csv
      best_selector_persisted_model: TODO  # NOT currently persisted; see Section H
    climatology:
      train_stats: rice/outputs_stage2_batch_2024_bestgate/sheath_blight/climatology_train_stats.csv
      best_clim_variant: clim_mean_mid  # from selector_by_pest_year.csv best_climatology_name
      fallback_default: climatology  # per Section E.4

# ... repeat for the other 7 pests ...

inference_code:
  stage1_model_loader: TODO  # confirm xgb.Booster load path
  stage1_alert_builder: rice/scripts/run_event_eval.py:build_alert_rows
  stage2_forward: rice/scripts/phase_r_oracle_iou.py:build_stage2_row_map
  iou_for_mu: rice/scripts/phase_r_oracle_iou.py:iou_for_mu
  pi_formula: |
    pred_L = round(mu - 1.96 * sigma)
    pred_R = round(mu + 1.96 * sigma)
```

The draft file marks every uncertain or TODO item explicitly.

---

## H. Missing files / ambiguous points — questions for the researcher

| # | Question | Why it matters |
|---|---|---|
| 1 | **Where is Stage-1 temperature scaling stored?** `run_event_eval.fit_temperature_grid` fits it, but I don't see it persisted in the XGBoost ckpt. Is it (a) baked into the saved scores, (b) recomputed at eval each time from val set, or (c) stored somewhere I missed? | API needs the same calibration to reproduce alerts. |
| 2 | **Confirm `stage2_pmf_alert_tstar_feat_idx=30` channel semantic.** The model reads `X[..., 30].amax(dim=2)` as absolute alert DOY. Is index 30 the `alert_tstar` channel after dispatch append (or a different layout)? Need exact channel name → idx map. | Wrong channel = catastrophic μ. |
| 3 | **No persisted selector model files.** V2 selectors are trained on-the-fly per call. For API deployment we need either (a) freeze + persist LightGBM models per (pest, year) ckpt, (b) re-train deterministically at deploy with fixed seed using val sample_grid as training data (loads the per-pest CSV), or (c) fall back to V1 bin_rule (interpretable). Which option is preferred? | This is the single biggest blocker for reproducible selector inference. |
| 4 | **Selector input contract** — selector features include several columns produced only by Stage-2 sample_grid generation (`pred_mu`, `pred_lead`, etc.). API needs to compute Stage-2 mu at all 6 coarse offsets first, then run selector, then return mu at the chosen offset. Confirm this 2-pass flow is acceptable. | Defines API latency / call structure. |
| 5 | **Which split to use in production?** All 3 splits (test=2022, 2023, 2024) have ckpts. The split3 (test=2024) ckpt is trained on data through 2023. For predictions in 2026 should we (a) use split3 as-is, (b) retrain on data through 2025, or (c) keep ensemble? | Affects model freshness policy. |
| 6 | **External data dependency:** Production assumes daily weather CSV at `/home/gpu4080/ygdata/rice/1997_2024_RICE_union_all_sites_with_GDD10_since_gs.csv`. Will API serve receive (a) the same CSV daily, (b) a per-site time-series API, or (c) batch upserts? | Drives preprocessing layer design. |
| 7 | **Phenology table source** — Korean column names + per-site rows. Path/format? | Currently referenced via `merge_pheno_daily_ffill` but I don't see a clear stored phenology CSV path. |
| 8 | **Stage-1 → Stage-2 dispatch CSV regeneration** at inference: `build_dispatch_feature_table.py` reads Stage-1 ckpts + scores + computes per-(site, year) row. For online inference we'd run this per new site-year. Is that the intended flow, or should we precompute a dispatch lookup for known sites? | Latency. |
| 9 | **BPH2 status** — `rice/pests/BPH2/` exists with full ckpts but cross-split summary excludes it. Confirm BPH2 is NOT part of the API contract. | Per CLAUDE memory context BPH2 is excluded (short history), but config files are still there. |
| 10 | **W&B / wandb dependency** — many scripts import `wandb`. API should NOT need it. Inference code paths should be importable without W&B init. Currently `run_train.py` and `phase_r_oracle_iou.py` have wandb hooks that may need to be no-op for production. | Avoid runtime breakage on systems w/o W&B credentials. |
| 11 | **Selector fallback when Stage-1 doesn't alert** — if Stage-1 gate doesn't fire for a site-year, there is no `alert_tstar` → Stage-2 mu undefined → selector input incomplete. What's the API response in this case? (climatology fallback?) | Defines the no-alert API contract. |
| 12 | **Sigma scaling for selector dense interpolation** — `--stage2_dispatch_feature_csv_override` exists in `phase_r_oracle_iou.py` for cohort swaps, but the dense oracle (1..75) uses μ linear interp between coarse anchors. For API operational policy we should restrict to **coarse offsets only**. Confirm. | Already noted in Section D; just confirm. |

---

## Post-state — `git status --short` (AFTER)

Captured after writing this report. **The only NEW files relative to BEFORE are `api_handoff_report.md` and `api_handoff_manifest_draft.yaml`** (allowed). No previously-existing file was modified, moved, or deleted (verified: count of ` M` rows = 21, identical to BEFORE).

```
[delta vs BEFORE — only new items]
?? api_handoff_report.md
?? api_handoff_manifest_draft.yaml

[unchanged from BEFORE: 21x ' M' modified rows + all prior '??' untracked entries]
```

Full diff = 0 modifications to existing tracked files. Read-only inspection succeeded.

---

## Appendix — Exact shell commands used

```bash
# Pre-state
git status --short
git branch --show-current
git log --oneline -10

# Repo structure
ls -la /home/gpu4080/research/cropscience/
ls -d rice/*/
ls -d rice/outputs_stage1/*/
ls -d rice/outputs_stage2_*/

# Stage-1
ls rice/outputs_stage1/batch_rolling/
ls rice/outputs_stage1/batch_rolling/_summary/
find rice/outputs_stage1/batch_rolling/sheath_blight/run0/split3_v2023_t2024 -maxdepth 3 -type f
ls rice/outputs_stage1/batch_rolling/sheath_blight/run0/split3_v2023_t2024/A/ckpt/
.venv/bin/python -c "import torch; ck = torch.load(...); print(list(ck.keys())[:30])"   # ckpt inspect
cat rice/outputs_stage2_batch_2024_bestgate/_summary/stage1_gate_selection_split3_2024.csv | head -10
.venv/bin/python -c "import json; s = json.load(open('rice/.../group_tau_hybrid_summary.json')); ..."

# Stage-1 availability matrix (per pest x run x split)
.venv/bin/python <<'PY'
import os
pests = ['BPH','WBPH','bacterial_blight','blast','brown_spot','rice_stem_borer_1','rice_stem_borer_2','sheath_blight']
splits = ['split1_v2021_t2022','split2_v2022_t2023','split3_v2023_t2024']
for p in pests:
    for r in [0,1,2]:
        for sp in splits:
            base = f"rice/outputs_stage1/batch_rolling/{p}/run{r}/{sp}"
            a = os.path.exists(f"{base}/A/ckpt/event_xgb_w28_lead14-45_A.pt")
            d = os.path.exists(f"{base}/D/ckpt/event_xgb_w28_lead14-45_D.pt")
            j = os.path.exists(f"{base}/group_tau/group_tau_hybrid_summary.json")
            if not (a and d and j):
                print(f"  {p:22s}  {r}  {sp:20s}  A={a} D={d} J={j}")
PY

# Stage-2 ckpt inspection
.venv/bin/python -c "import torch; ck = torch.load('rice/outputs_stage2_batch_2024_bestgate/sheath_blight/lead_v3_final/ckpt/checkpoint_run4.pt', map_location='cpu', weights_only=False); ..."

# Stage-2 availability matrix
.venv/bin/python <<'PY'
import os
roots = {2022:'rice/outputs_stage2_batch_2022_baseline', 2023:'rice/outputs_stage2_batch_2023_baseline', 2024:'rice/outputs_stage2_batch_2024_bestgate'}
pests = [...]
for p in pests:
    for y in (2022,2023,2024):
        print(os.path.exists(f"{roots[y]}/{p}/lead_v3_final/ckpt/checkpoint_run4.pt"))
PY

# Selector outputs
ls rice/outputs_stage2_selector_cross_split/
head -10 rice/outputs_stage2_selector_cross_split/selector_by_pest_year.csv
ls rice/outputs_stage2_selector_cross_split/2024_sheath_blight/
grep -n "predict\|pick_offsets" rice/scripts/phase_b_stage2_offset_selector_v2_ranking.py | head
find rice/outputs_stage2_selector_cross_split -name "*.lgb" -o -name "*.pkl"  # selector model persistence check

# Climatology
ls rice/outputs_stage2_batch_2024_bestgate/sheath_blight/climatology*
head -2 rice/outputs_stage2_batch_2024_bestgate/sheath_blight/climatology_train_stats.csv
.venv/bin/python <<'PY' # presence check per (pest, split) ...
PY

# Preprocessing / feature code
head -20 rice/src/data_pipeline.py
grep -n "^def \|^class " rice/src/data_pipeline.py | head -30
grep -n "^def \|^class " rice/src/dataset.py | head -25
grep -n "^def \|^class " rice/scripts/run_event_eval.py | head
grep -n "^def \|^class " rice/scripts/run_event_train.py | head
head -30 rice/scripts/site_history_utils.py
head -30 rice/scripts/derived_weather_utils.py
grep -n "^def \|^class " rice/scripts/stage1_confidence_utils.py | head

# Post-state
git status --short
```

---

*End of report.*
