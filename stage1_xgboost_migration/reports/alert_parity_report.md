# Stage-1 alert parity: original `.pt` path vs portable JSON path

Generated: `2026-07-16T03:23:42.007155+00:00`  
Source env: python 3.12.3, torch 2.12.0+cu130, xgboost 3.3.0, scikit-learn 1.9.0, numpy 2.4.6

**8/8 pests fully match** end-to-end (raw proba -> temperature -> tau/k gate -> alert_tstar -> dispatch features).

## Per-pest end-to-end result

| pest | data | site-years | fired (orig/portable) | alert DOY | dispatch | status |
| --- | --- | ---: | --- | --- | --- | --- |
| BPH | synthetic | 6 | 5 / 5 | match | match | ok |
| WBPH | real_weather_borrowed_labels | 7321 | 3281 / 3281 | match | match | ok |
| bacterial_blight | real_weather_borrowed_labels | 7321 | 4330 / 4330 | match | match | ok |
| blast | real_weather_borrowed_labels | 7321 | 4743 / 4743 | match | match | ok |
| brown_spot | real_weather_borrowed_labels | 7321 | 5665 / 5665 | match | match | ok |
| rice_stem_borer_1 | real_weather_borrowed_labels | 7321 | 3647 / 3647 | match | match | ok |
| rice_stem_borer_2 | real_weather_borrowed_labels | 7321 | 3403 / 3403 | match | match | ok |
| sheath_blight | real | 7321 | 5599 / 5599 | match | match | ok |

## Probability parity per model (16)

| pest | branch | T | raw max abs diff | raw bit-exact | calibrated max abs diff | calibrated bit-exact | tau | threshold decisions |
| --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- |
| BPH | A | 1.95 | 0 | True | 0 | True | 0.6 | all match (68/501 pass) |
| BPH | D | 1.5 | 0 | True | 0 | True | 0.6 | all match (95/501 pass) |
| WBPH | A | 2.0 | 0 | True | 0 | True | 0.725 | all match (73854/1124361 pass) |
| WBPH | D | 2.05 | 0 | True | 0 | True | 0.7 | all match (139814/1124361 pass) |
| bacterial_blight | A | 2.5999999999999996 | 0 | True | 0 | True | 0.525 | all match (129028/1124361 pass) |
| bacterial_blight | D | 2.5999999999999996 | 0 | True | 0 | True | 0.525 | all match (189920/1124361 pass) |
| blast | A | 2.65 | 0 | True | 0 | True | 0.6 | all match (135223/1124361 pass) |
| blast | D | 2.5 | 0 | True | 0 | True | 0.6 | all match (162699/1124361 pass) |
| brown_spot | A | 4.0 | 0 | True | 0 | True | 0.5 | all match (185330/1124361 pass) |
| brown_spot | D | 4.0 | 0 | True | 0 | True | 0.5 | all match (249006/1124361 pass) |
| rice_stem_borer_1 | A | 1.525 | 0 | True | 0 | True | 0.55 | all match (126014/1124361 pass) |
| rice_stem_borer_1 | D | 1.475 | 0 | True | 0 | True | 0.55 | all match (148676/1124361 pass) |
| rice_stem_borer_2 | A | 2.425 | 0 | True | 0 | True | 0.575 | all match (81542/1124361 pass) |
| rice_stem_borer_2 | D | 2.425 | 0 | True | 0 | True | 0.575 | all match (87834/1124361 pass) |
| sheath_blight | A | 5.0 | 0 | True | 0 | True | 0.55 | all match (229140/1124361 pass) |
| sheath_blight | D | 5.0 | 0 | True | 0 | True | 0.55 | all match (235606/1124361 pass) |

## Gate parameters actually used

`k`/`tau` come from each pest's `group_tau_hybrid_summary.json`, which `infer/stage1.py` marks authoritative; the copy in `stage1_selected_gates.yaml` has drifted and is **not** used.

| pest | method | k | tau | tau_no | tau_with |
| --- | --- | ---: | ---: | ---: | ---: |
| BPH | D_history | 3 | 0.6 | - | - |
| WBPH | dispatch_group_tau | 3 | - | 0.725 | 0.7 |
| bacterial_blight | D_history | 3 | 0.525 | - | - |
| blast | D_history | 3 | 0.6 | - | - |
| brown_spot | D_history | 3 | 0.5 | - | - |
| rice_stem_borer_1 | A_baseline | 3 | 0.55 | - | - |
| rice_stem_borer_2 | D_history | 3 | 0.575 | - | - |
| sheath_blight | dispatch_group_tau | 3 | - | 0.55 | 0.55 |

## Gate boundary unit parity

24 crafted series x k in {1,2,3} at tau=0.6 -- probability exactly at tau, one ULP below tau, streak of exactly k-1 vs k, NaN present, all-above, all-below. **all match: True**

## Shipped temperature vs a fresh runtime re-fit

The API's production path (`compute_alert_single_sy`) reads the frozen `temperature.json` and does not re-fit. The cohort path (`_calibrated_per_sy`) re-fits on VAL_YEAR=2023. This compares the two.

Only valid where the pest has its OWN labels -- `_fit_temperature_grid` minimises NLL against `y_event`, so a re-fit against borrowed labels would be meaningless and is skipped rather than reported as a mismatch.

| pest | branch | shipped | re-fit now | match |
| --- | --- | ---: | ---: | --- |
| WBPH | - | - | - | skipped: labels borrowed from another pest; a re-fit against them would be meaningless |
| bacterial_blight | - | - | - | skipped: labels borrowed from another pest; a re-fit against them would be meaningless |
| blast | - | - | - | skipped: labels borrowed from another pest; a re-fit against them would be meaningless |
| brown_spot | - | - | - | skipped: labels borrowed from another pest; a re-fit against them would be meaningless |
| rice_stem_borer_1 | - | - | - | skipped: labels borrowed from another pest; a re-fit against them would be meaningless |
| rice_stem_borer_2 | - | - | - | skipped: labels borrowed from another pest; a re-fit against them would be meaningless |
| sheath_blight | A | 5.0 | 5.0 | True |
| sheath_blight | D | 5.0 | 5.0 | True |

## Data provenance

- **BPH**: no compatible observation csv
- **WBPH**: no RICE_LONG_WBPH.csv; used real weather features at real sites with site-years/censoring borrowed from sheath_blight (identical feature_cols, doy range, window, stride, proxy, only_pre). Valid for numerical parity; NOT this pest's real cohort.
- **bacterial_blight**: no RICE_LONG_bacterial_blight.csv; used real weather features at real sites with site-years/censoring borrowed from sheath_blight (identical feature_cols, doy range, window, stride, proxy, only_pre). Valid for numerical parity; NOT this pest's real cohort.
- **blast**: no RICE_LONG_blast.csv; used real weather features at real sites with site-years/censoring borrowed from sheath_blight (identical feature_cols, doy range, window, stride, proxy, only_pre). Valid for numerical parity; NOT this pest's real cohort.
- **brown_spot**: no RICE_LONG_brown_spot.csv; used real weather features at real sites with site-years/censoring borrowed from sheath_blight (identical feature_cols, doy range, window, stride, proxy, only_pre). Valid for numerical parity; NOT this pest's real cohort.
- **rice_stem_borer_1**: no RICE_LONG_rice_stem_borer_1.csv; used real weather features at real sites with site-years/censoring borrowed from sheath_blight (identical feature_cols, doy range, window, stride, proxy, only_pre). Valid for numerical parity; NOT this pest's real cohort.
- **rice_stem_borer_2**: no RICE_LONG_rice_stem_borer_2.csv; used real weather features at real sites with site-years/censoring borrowed from sheath_blight (identical feature_cols, doy range, window, stride, proxy, only_pre). Valid for numerical parity; NOT this pest's real cohort.

## Portable pipeline (torch-free)

Ran under python 3.12.3, xgboost 3.0.5, numpy 2.4.4.

- torch imported: **False**
- `.pt` opened / unpickled: **none**
- pulled in by xgboost itself (not by the pipeline): ['pandas', 'scipy', 'sklearn']

| pest | site-years | fired (portable/reference) | mismatches | status |
| --- | ---: | --- | ---: | --- |
| BPH | 6 | 5 / 5 | 0 | ok |
| WBPH | 40 | 18 / 18 | 0 | ok |
| bacterial_blight | 40 | 20 / 20 | 0 | ok |
| blast | 40 | 20 / 20 | 0 | ok |
| brown_spot | 40 | 20 / 20 | 0 | ok |
| rice_stem_borer_1 | 40 | 18 / 18 | 0 | ok |
| rice_stem_borer_2 | 40 | 20 / 20 | 0 | ok |
| sheath_blight | 40 | 22 / 22 | 0 | ok |

