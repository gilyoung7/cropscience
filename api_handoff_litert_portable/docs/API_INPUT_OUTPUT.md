# API 입력·출력 명세 (사용자 관점)

대상: `api_handoff_litert_portable_patched` (Stage-1 XGBoost `model.json` + Stage-2 FP16 LiteRT)

이 문서는 추측이 아니라 현재 구현된 코드에서 직접 확인한 내용입니다. 근거 위치를
각 항목에 표기했습니다 (`schemas.py`, `run_predict.py`, `infer/batch.py`,
`infer/cohort.py`, `infer/inputs.py`, `infer/fallback.py`,
`assets/configs/fallback_policy.yaml`).

> **문서 역할 분담**
> [`../README.md`](../README.md) — 설치부터 첫 실행까지의 빠른 시작.
> 이 문서 — 요청 필드, 출력 컬럼, fallback 사례, 운영 상세의 전체 레퍼런스.
> 빠른 시작만 필요하면 README로 충분합니다.

---

## 0. 실행 방법

```bash
python run_predict.py --input-dir IN --output-dir OUT \
    [--stage2-variant fp16|fp32] \
    [--representative-sites PATH] \
    [--overwrite] [--unique-output-subdir]
```

### CLI 옵션

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--input-dir` | `<pkg>/input` | `request.json`을 찾는 디렉터리 |
| `--output-dir` | `<pkg>/output` | 결과를 쓸 디렉터리 |
| `--stage2-variant` | `fp16` | Stage-2 정밀도 |
| `--representative-sites` | 없음 | 대표지역 CSV 경로 (batch). 요청 필드가 우선 |
| `--overwrite` | 꺼짐 | 기존 결과를 **덮어씀**. 기본은 거부 |
| `--unique-output-subdir` | 꺼짐 | `--output-dir` 아래 **요청별 고유 하위 폴더**를 만들어 씀 |

`--overwrite`와 `--unique-output-subdir`는 `request.json`의 `"overwrite": true` /
`"unique_output_subdir": true`로도 지정할 수 있습니다.

### 입력 파일 배치 — 두 가지 방식

**(a) 디렉터리 배치 (기존 방식)** — 고정 파일명을 `--input-dir`에 둡니다.

| 파일 | 필수 | 설명 |
|---|---|---|
| `request.json` | **필수** | 단일 또는 batch 요청 |
| `daily_weather.csv` | **필수** | 한국어 스키마 일별 기상. 여러 site/연도 포함 가능 |
| `long_observation.csv` | 택1 | Layout A — 해당 site만 |
| `LONG_by_pest/RICE_LONG_<pest>.csv` | 택1 | Layout B — 해당 병해충 전체 site |

`long_observation.csv`가 있으면 그것이 우선하고, 없으면 `LONG_by_pest/`를 씁니다
(`inputs.resolve_obs`). 둘 다 없으면 **exit 2**.

**(b) 경로 지정 (운영 권장)** — `request.json`에 파일 위치를 직접 씁니다.
이 경우 입력 디렉터리에는 **`request.json` 하나만** 있으면 됩니다.

| 요청 필드 | 대체하는 파일 |
|---|---|
| `daily_weather_path` | `daily_weather.csv` |
| `long_observation_path` | `long_observation.csv` / `LONG_by_pest/…` |
| `representative_sites_path` | `representative_site_ids_2002_2024.csv` (batch) |

절대경로는 그대로 사용하고, 상대경로는 **현재 작업 디렉터리 → `--input-dir` →
패키지 루트** 순으로 찾습니다 (`inputs.resolve_input_path`).
필드를 생략하면 (a) 방식으로 동작하므로 **기존 사용법은 그대로 유지됩니다.**

지정한 경로의 파일이 없으면 시도한 경로를 모두 나열하고 **exit 2**로 종료합니다.

### ZIP은 필요하지 않습니다

요청을 ZIP으로 묶을 필요가 없고, 결과도 ZIP으로 만들지 않습니다.
코드에 `zipfile` / `tarfile` / `extract` / `unpack` 호출이 **하나도 없습니다.**
배포용 `.tar.gz`는 프로그램을 서버로 옮기기 위한 것으로, 한 번 풀면 그것으로 끝입니다.

입력 파일은 **복사하지 않고 원본 경로에서 직접 읽으며, 절대 수정하지 않습니다**
(`shutil` / `tempfile` 호출 없음). 대용량 daily 마스터를 요청마다 복제할 필요가
없습니다.

---

## 1. 입력

### 1-1. 단일 예측 (`mode` 없음 또는 `"single"`)

근거: `schemas.validate_request`

| 필드 | 자료형 | 필수 | 기본값 | 의미 |
|---|---|---|---|---|
| `pest` | string | **필수** | — | 8종 중 하나. **대소문자 구분** (`BPH`, `WBPH`, `bacterial_blight`, `blast`, `brown_spot`, `rice_stem_borer_1`, `rice_stem_borer_2`, `sheath_blight`) |
| `site_id` | string | **필수** | — | `"33908_67063"` 형태의 격자 ID. 빈 문자열 불가 |
| `year` | int | **필수** | — | 대상 연도 |
| `alert_tstar_doy` | int \| null | 선택 | `null` | Stage-1 alert를 수동 지정(DOY). 주면 Stage-1 결과를 **덮어씀** |
| `include_diagnostics` | bool | 선택 | `false` | 응답에 `diagnostics` 블록 포함 |
| `daily_weather_path` | string | 선택 | `input_dir/daily_weather.csv` | 일별 기상 CSV 경로 |
| `long_observation_path` | string | 선택 | Layout A → Layout B | LONG 관측 CSV 경로 |
| `overwrite` | bool | 선택 | `false` | 기존 결과 덮어쓰기 허용 |
| `unique_output_subdir` | bool | 선택 | `false` | 요청별 고유 출력 하위 폴더 생성 |

잘못된 `pest`/`site_id`/`year` → **exit 1**.

### 1-2. Batch (`"mode": "batch"`)

근거: `run_predict.main`, `batch.run_batch`

배치는 **대표지역 방식** 또는 **generic CSV 방식** 중 하나입니다.

#### (a) 대표지역 batch

| 필드 | 자료형 | 필수 | 기본값 | 의미 |
|---|---|---|---|---|
| `mode` | string | **필수** | — | `"batch"` |
| `pest` | string | **필수** | — | 병해충 1종 |
| `year` | int | 시간지정 택1 | — | **모드 A** — 단일 연도 |
| `start_year` / `end_year` | int | 시간지정 택1 | — | **모드 A** — 연도 범위 (예: 2002–2022) |
| `as_of_date` | string | 시간지정 택1 | — | **모드 B** — `"YYYY-MM-DD"` 실제 날짜 문자열 |
| `representative_sites_path` | string | 선택 | 입력 디렉터리에서 자동 탐색 | 대표지역 CSV 경로 (별칭 `representative_sites_csv`) |
| `daily_weather_path` | string | 선택 | `input_dir/daily_weather.csv` | 일별 기상 CSV 경로 |
| `long_observation_path` | string | 선택 | Layout A → Layout B | LONG 관측 CSV 경로 |
| `max_sites` | int | 선택 | 없음(전체) | 연도별 상한. 운영/테스트용 |
| `include_diagnostics` | bool | 선택 | `false` | 행별 진단 컬럼 추가 |
| `stage2_variant` | string | 선택 | `fp16` | `fp16` \| `fp32`. **`--stage2-variant`보다 우선** (`run_predict.py:425`). fp32는 `--include-fp32` 빌드에만 존재 |
| `overwrite` | bool | 선택 | `false` | 기존 결과 덮어쓰기 허용 |
| `unique_output_subdir` | bool | 선택 | `false` | 요청별 고유 출력 하위 폴더 생성 |

`year`, `start_year`/`end_year`, `as_of_date` 중 **하나도 없으면 exit 2**.

#### (b) generic CSV batch

| 필드 | 자료형 | 필수 | 의미 |
|---|---|---|---|
| `mode` | string | **필수** | `"batch"` |
| `input_csv` | string | **필수** | `pest,site_id,year[,alert_tstar_doy]` 컬럼 CSV |
| `stage2_variant` | string | 선택 | `fp16` \| `fp32` |

generic 모드는 코호트 pre-pass를 쓰지 않고 행 단위로 처리합니다
(`batch.run_batch`의 `if not generic` 분기).

`input_csv`는 다른 경로 필드와 **동일한 규칙**으로 해석됩니다 — 절대경로는 그대로,
상대경로는 CWD → `--input-dir` → 패키지 루트 순 (`batch.resolve_input_csv`).

#### 행 단위 필드 상속

최상위 요청의 다음 필드는 각 행에 상속됩니다 (`batch.INHERITED_ROW_FIELDS`):

`daily_weather_path`, `long_observation_path`, `representative_sites_path`,
`stage2_variant`, `include_diagnostics`

**행에 같은 이름의 컬럼이 있으면 행 값이 우선**하고, 없거나 빈 셀이면 최상위 값을
씁니다 (`batch._inherit_request_fields`). 빈 셀은 pandas에서 `NaN`으로 오므로
"값 없음"으로 정규화됩니다.

```csv
pest,site_id,year,long_observation_path
BPH,33210_56298,2004,                                  # 최상위 값 상속
WBPH,33908_67063,2011,/srv/pest/assets/LONG_by_pest/RICE_LONG_WBPH.csv
```

행이 `stage2_variant`를 덮어쓰면 그 행은 해당 정밀도의 Stage-2 모델로 실행됩니다.
컨텍스트 캐시가 `(pest, variant)`로 키잉되어 있어 변형별 모델이 각각 한 번만
로드됩니다.

### 1-3. 날짜 표기 — DOY인가 날짜 문자열인가

**입력에서 섞이므로 주의가 필요합니다.**

| 항목 | 표기 |
|---|---|
| `as_of_date` (입력) | **실제 날짜 문자열** `"2004-07-15"` |
| `alert_tstar_doy` (입력) | **DOY 정수** (1–365/366) |
| 출력의 모든 `*_doy` | **DOY 정수** |

`as_of_date`는 내부에서 `pd.to_datetime(...).dayofyear`로 DOY(`as_of_doy`)로 변환됩니다
(`batch.run_batch`). 사용자가 직접 `as_of_doy`를 넣는 필드는 없습니다.

### 1-4. historical(모드 A) vs operational(모드 B) 입력 차이

| | historical | operational |
|---|---|---|
| 시간 지정 | `year` 또는 `start_year`/`end_year` | `as_of_date` |
| 그 해 관측(LONG) 사용 | **사용** — interval label 생성 | **사용 안 함** (look-ahead 방지) |
| 기상 사용 범위 | 시즌 전체 | `as_of_date`까지, 이후는 padding |
| 결측 경계 가드 | **미적용** | **적용** |

operational에서 그 해 관측을 쓰지 않는 이유: 라벨은 예측 대상 그 자체이고,
`filter_labels_by_gap`이 절단된 `doy_end`로 `R_doy`를 클립하면서 event가 `as_of` 이후인
site가 코호트에서 조용히 탈락하기 때문입니다. 배포 API가 `L=1, R=1, censor_type="right"`를
쓰는 것과 같은 규약입니다 (`cohort.stage1_cohort`).

### 1-5. 대표지역 site 선택 (union)

근거: `batch.run_batch`

```
target = sorted( rep_sites ∩ (LONG_year ∪ daily_year) )
```

- 그 해 **관측이 없어도 기상이 있으면 포함**됩니다. Stage-1이 alert를 못 내고
  climatology fallback 행이 생성됩니다.
- 중복 `site_id`는 첫 항목만 유지 후 정렬 (`representative_sites`의
  `sorted(dict.fromkeys(...))`).
- 출력 행 순서는 **`site_id` 오름차순**입니다.

### 1-6. 결측 기상자료 허용 형태

근거: `preprocessing.daily_year_frame`

| 상황 | 처리 |
|---|---|
| 개별 셀 결측(빈칸/NaN) | **허용**. `interpolate(limit_direction="both")` → `ffill` → `bfill` |
| `일강수량(mm)` 잔여 결측 | `0.0`으로 채움 |
| `GDD10_since_gs` | **보간하지 않음** (학습과 동일) |
| 특정 컬럼 전 구간 결측 | 허용. 보간 불가 → 0 + miss 지시자 = 1 |
| 시즌 DOY 누락(행 자체 없음) | **오류** — 해당 site는 코호트에서 제외 |
| 같은 DOY 중복 행 | **오류** |
| 필수 컬럼 자체 누락 | **오류** |

필수 컬럼: `일시`, `일강수량(mm)`, `최고기온(°C)`, `최저기온(°C)`, `평균기온(°C)`,
`평균 풍속(m/s)`, `최대 풍속(m/s)`, `평균 상대습도(%)`, `합계 일조시간(h)`,
`합계 일사량(MJ/m2)`, `GDD10_since_gs` (+ `지점ID`).

**운영 모드 주의**: 셀 결측 자체는 허용되지만, 그 결측 구간이 Stage-2 window 안에 있고
`as_of` 시점까지 해소되지 않으면 해당 site는 Stage-2가 차단됩니다 (§4-1).

---

## 2. 출력

`--output-dir`에 4개 파일이 생성됩니다: `response.json`, `predictions.csv`,
`run_log.txt` (+ 단일 모드는 stdout에 JSON).

### 2-1. `response.json` — 단일 예측

근거: `schemas.build_response`

```
pest, site_id, year, model_version   : 요청 에코 + 모델 버전
stage1  : { alert_fired, alert_tstar_doy, gate_method, alert_source, wiring_status }
stage2  : { learned_stage2, climatology, recommended_source [, output_status] }
final_prediction : { source, mu_doy, pi_95, selected_offset, fallback_triggered }
backends : { stage1_backend, stage2_backend }     ← 신규 API 추가 필드
diagnostics : {...}                                ← include_diagnostics=true 일 때만
```

| 필드 | 자료형 | 의미 |
|---|---|---|
| `stage1.alert_fired` | bool | Stage-2 입력 빌드까지 성공했는지. **alert가 떠도 Stage-2가 실패하면 false** |
| `stage1.alert_tstar_doy` | int \| null | **Stage-1 결과** — 경보 발생 DOY |
| `stage1.gate_method` | string | `A_baseline` / `D_history` / `dispatch_group_tau` |
| `stage1.alert_source` | string \| null | `stage1_live` / `stage1_no_alert` / `stage1_error` / `manual_request` |
| `stage2.learned_stage2` | object \| null | **Stage-2 결과**. 실패·차단 시 `null` |
| `stage2.learned_stage2.mu_doy` | float | 모델이 예측한 발생 시점 DOY |
| `stage2.learned_stage2.pi_95` | object | `{lower_doy:int, upper_doy:int, sigma_days:float}` |
| `stage2.learned_stage2.selected_offset` | int | 정책상 고정 offset (일) |
| `stage2.learned_stage2.output_status` | string | `main`(BPH/WBPH) 또는 `experimental`(나머지 6종) |
| `stage2.learned_stage2.model_kind` | string | `lead_v3` |
| `stage2.climatology` | object | `{mu_doy, pi_95, variant}` — 병해충별 상수 |
| `stage2.output_status` | string | **운영 차단 시에만 존재**. 차단 사유 코드 |
| `stage2.recommended_source` | string | `learned_stage2` 또는 `climatology` |
| `final_prediction.source` | string | `learned_stage2` / `climatology` / `climatology_no_alert` |
| `final_prediction.mu_doy` | float | **최종 예측값** |
| `final_prediction.pi_95` | object | **최종 예측구간** |
| `final_prediction.selected_offset` | int \| null | learned일 때만 값 |
| `final_prediction.fallback_triggered` | bool | learned 권장인데 learned가 없을 때 true |

### 2-2. `predictions.csv` — 한 행이 의미하는 것

**한 행 = 하나의 `site_id` × 하나의 대상 연도.**

한 번의 batch 요청은 **병해충 1종**을 처리하고, 그 병해충의 대표 site마다 한 행을
생성합니다.

| 요청 | 대표 site | 행 수 | 실측 |
|---|---:|---:|---|
| `"year": 2004` | 858 | **858** | success 134 + fallback 724 |
| `"start_year": 2023, "end_year": 2024` | 858 | **1716** (858 × 2) | 2023년 858행 + 2024년 858행 |
| `"as_of_date": "2004-08-10"` | 858 | **858** | 아직 평가 불가한 site는 fallback 행 |

- **fallback site도 제거하지 않고 행으로 남깁니다.** 경보가 없거나, 운영상 아직
  이르거나, 결측 때문에 보류된 site는 climatology 값이 채워진 행이 됩니다.
  858개를 요청하면 결과도 858행입니다 — 행이 줄어들면 그것이 이상 신호입니다.
- 여러 연도 요청은 site가 연도마다 반복되므로 고유 `site_id` 수는 그대로 858입니다.
- `max_sites`를 주면 연도별로 그 수만큼 잘립니다(운영/테스트용).
- generic CSV batch(`input_csv`)는 입력 CSV의 행 수·순서를 그대로 따릅니다.

행 수가 예상과 다르면 `run_log.txt`의
`year_available_site_count` / `requested_count`를 확인하세요.

### 2-2-1. 컬럼 — 16 + 2

근거: `schemas.FLAT_COLS`, `batch.write_predictions_csv`

| # | 컬럼 | 자료형 | 의미 |
|---|---|---|---|
| 1 | `pest` | string | 병해충 |
| 2 | `site_id` | string | 격자 ID |
| 3 | `year` | int | 연도 |
| 4 | `model_version` | string | `v0-transformer-real-ckpt` |
| 5 | `final_source` | string | 최종값 출처 |
| 6 | **`final_mu_doy`** | float | **최종 예측 DOY** |
| 7 | **`final_pi95_lower`** | int | **최종 구간 하한 DOY** |
| 8 | **`final_pi95_upper`** | int | **최종 구간 상한 DOY** |
| 9 | `learned_mu_doy` | float \| 빈칸 | Stage-2 원값 (최종값이 아닐 수 있음) |
| 10 | `learned_selected_offset` | int \| 빈칸 | offset |
| 11 | `learned_output_status` | string \| 빈칸 | `main`/`experimental` 또는 **차단 사유 코드** |
| 12 | `climatology_mu_doy` | float | 기후평년 DOY |
| 13 | `climatology_variant` | string | `mean_mid` (8종 전부) |
| 14 | `recommended_source` | string | 정책상 권장 출처 |
| 15 | `fallback_triggered` | bool | fallback 여부 |
| 16 | `alert_tstar_doy` | int \| 빈칸 | Stage-1 경보 DOY |
| 17 | `status` | string | `success` / `fallback` / `error` (batch 전용) |
| 18 | `error_reason` | string | 사유 문구 (batch 전용) |

### 2-3. Stage-1 / Stage-2 결과 기록 위치

| | `response.json` | `predictions.csv` |
|---|---|---|
| **Stage-1** | `stage1.*` | `alert_tstar_doy` |
| **Stage-2** | `stage2.learned_stage2.*` | `learned_mu_doy`, `learned_selected_offset`, `learned_output_status` |
| **최종 결정** | `final_prediction.*` | `final_mu_doy`, `final_pi95_*`, `final_source` |

### 2-4. sigma / variant

- `sigma_days`는 **항상 5.0** (`fallback.SIGMA_DAYS_DEFAULT`). 모델이 site별로
  예측하는 값이 아니라 고정 상수입니다.
- `pi_95`는 `mu ± 9.8`일 (`PI95_HALFWIDTH = round(1.96 × 5.0, 1)`) 후 정수 반올림.
- `variant`는 8종 모두 `mean_mid` — 기후평년 계산에 쓰인 통계 종류입니다.

### 2-5. exit code

| code | 조건 |
|---|---|
| 0 | 정상. **Stage-2 실패·차단도 0** (climatology로 응답하므로) |
| 1 | 잘못된 요청 (pest/site_id/year, 정책 누락), **출력 디렉터리 충돌** |
| 2 | 입력 파일 누락, 지정 경로 없음, batch 요청 명세 부족 |

### 2-6. 출력 디렉터리 보호

출력 파일명은 `response.json` / `predictions.csv` / `run_log.txt`로 **고정**입니다.
따라서 두 요청이 같은 `--output-dir`를 쓰면 서로를 덮어씁니다. 이를 막기 위해
세 가지 장치가 있습니다 (`infer/inputs.py`).

**(1) 완료된 결과 보호** — 위 3개 중 하나라도 이미 있으면 **exit 1로 거부**합니다.

```
[run_predict] ERROR: output directory already contains results: response.json, ...
  Refusing to overwrite — another request may have written these.
  Choose one:
    * point --output-dir at a fresh directory (recommended for concurrent runs), or
    * pass --unique-output-subdir ... to auto-create a per-run subfolder, or
    * pass --overwrite ... to replace them.
```

**(2) 실행 중인 요청 보호 (claim)** — 존재 검사만으로는 **동시 실행을 막지 못합니다.**
세 프로세스가 빈 디렉터리를 동시에 보면 셋 다 검사를 통과하고, 마지막 하나만
살아남습니다 (실측: 3건 모두 exit 0, 결과는 1벌).
그래서 시작 시 `os.open(O_CREAT|O_EXCL)`로 `.run_claim` 파일을 **원자적으로**
만듭니다. 이미 있으면 다른 실행이 사용 중이므로 exit 1로 거부합니다.

측정 결과 — 같은 output-dir로 3건 동시 실행:

| | 개선 전 | 개선 후 |
|---|---|---|
| 종료 코드 | 0, 0, 0 | **0, 1, 1** |
| 남은 결과 | 1벌 (2건 조용히 소실) | 1벌 (**2건은 거부됐음을 명시**) |

**(3) 실패가 성공을 덮어쓰지 않음** — 배치 전체 실패(`fail_batch`) 시 기존 출력이
있으면 **건드리지 않고** stderr로만 알립니다. 재실행이 실패해도 직전 성공 결과가
보존됩니다 (실측: 실패 전후 `predictions.csv` SHA-256 동일).

모든 출력은 임시 파일에 쓴 뒤 `os.replace`로 교체합니다(`inputs.write_atomic`).
중간에 프로세스가 죽어도 잘린 파일이 남지 않습니다.

`.run_claim`은 정상·실패·예외 종료 모두에서 `try/finally`로 해제됩니다.

#### SIGKILL 등으로 `.run_claim`이 남았을 때

`kill -9`나 OOM으로 프로세스가 강제 종료되면 `finally`가 실행되지 못해 파일이
남을 수 있습니다. 이후 실행은 이렇게 거부됩니다.

```
[run_predict] ERROR: output directory is claimed by another run
  (pid=12345 claimed_utc=2026-07-22T04:11:07+00:00).
  ...
  If that run died, delete <dir>/.run_claim and retry.
```

처리 방법은 셋 중 하나입니다.

1. **해당 PID가 살아있는지 확인** — `ps -p <pid>`. 살아있으면 기다립니다.
2. **죽었으면 파일 삭제** — `rm <output-dir>/.run_claim`
3. **`--overwrite`로 실행** — 오래된 claim을 해제하고 새로 잡습니다

`--unique-output-subdir`를 쓰면 매번 새 폴더라 이 상황 자체가 생기지 않습니다.

---

## 3. 정상 learned 예측 예시

`recommended_source = learned_stage2`인 **BPH·WBPH**만 최종값이 Stage-2에서 옵니다.

**입력** (`examples/01_learned_request.json`)

```json
{"pest": "WBPH", "site_id": "33908_67063", "year": 2011, "include_diagnostics": true}
```

**출력** (`examples/01_learned_response.json`, 발췌)

```json
{
  "pest": "WBPH", "site_id": "33908_67063", "year": 2011,
  "model_version": "v0-transformer-real-ckpt",
  "stage1": {
    "alert_fired": true, "alert_tstar_doy": 171,
    "gate_method": "dispatch_group_tau", "alert_source": "stage1_live",
    "wiring_status": "stage1_xgboost_live"
  },
  "stage2": {
    "learned_stage2": {
      "mu_doy": 245.82,
      "pi_95": {"lower_doy": 236, "upper_doy": 256, "sigma_days": 5.0},
      "selected_offset": 45, "output_status": "main", "model_kind": "lead_v3"
    },
    "climatology": {
      "mu_doy": 203.5,
      "pi_95": {"lower_doy": 194, "upper_doy": 213, "sigma_days": 5.0},
      "variant": "mean_mid"
    },
    "recommended_source": "learned_stage2"
  },
  "final_prediction": {
    "source": "learned_stage2", "mu_doy": 245.82,
    "pi_95": {"lower_doy": 236, "upper_doy": 256, "sigma_days": 5.0},
    "selected_offset": 45, "fallback_triggered": false
  },
  "backends": {"stage1_backend": "xgboost_json", "stage2_backend": "litert_fp16"}
}
```

`final_mu_doy = learned_mu_doy = 245.82` — 이 경우에만 둘이 같습니다.

---

## 4. climatology fallback 사례

### 4-1. Stage-2 window가 미해소 결측 구간을 가로지름 (operational 전용)

**입력** (`examples/04a_missingrun_request.json`)

```json
{"mode": "batch", "pest": "sheath_blight", "as_of_date": "2004-06-18",
 "representative_sites_path": "representative_site_ids_2002_2024.csv",
 "include_diagnostics": false}
```

**출력** (`examples/04a_missingrun_predictions.csv`, 해당 행)

| 컬럼 | 값 |
|---|---|
| `site_id` | `29851_55024` |
| `status` | **`fallback`** |
| `final_source` | `climatology` |
| `final_mu_doy` | `203.66` |
| `learned_mu_doy` | (빈칸) |
| `learned_output_status` | **`stage2_window_crosses_unresolved_missing_run`** |
| `climatology_variant` | `mean_mid` |
| `error_reason` | `Stage-2 withheld for pest=sheath_blight site=29851_55024 year=2004: column '평균기온(°C)' has a missing run inside the window [142,169] with no observed value before as_of_doy=170 ... [stage2_window_crosses_unresolved_missing_run]` |

- `final_mu_doy` 출처: **climatology** (`sheath_blight_climatology_train_stats.csv`의 `mean_mid`)
- batch: **중단되지 않음.** 같은 batch의 다른 site는 정상 `success` 유지
- historical 모드에서는 이 가드가 **작동하지 않음**

### 4-2. as_of 시점에 offset window 미관측 (operational 전용)

**입력** (`examples/04b_pending_request.json`)

```json
{"mode": "batch", "pest": "WBPH", "as_of_date": "2011-07-20",
 "representative_sites_path": "representative_site_ids_2002_2024.csv",
 "include_diagnostics": false}
```

**출력** (`examples/04b_pending_predictions.csv`)

| 컬럼 | 값 |
|---|---|
| `site_id` | `33908_67063` |
| `status` | **`fallback`** |
| `final_source` | **`climatology_no_alert`** |
| `final_mu_doy` | `203.5` |
| `learned_mu_doy` | (빈칸) |
| `learned_output_status` | **`stage2_pending_window_not_yet_observed`** |
| `fallback_triggered` | `True` |
| `error_reason` | `Stage-2 not yet evaluable for pest=WBPH site=33908_67063 year=2011: needs weather through DOY 216 (alert 171 + offset 45), as_of_date is DOY 201 [stage2_pending_window_not_yet_observed]` |

`final_source`가 `climatology_no_alert`이고 `fallback_triggered=true`인 이유는 WBPH가
`recommended_source=learned_stage2`이기 때문입니다 (`fallback.select_final`).
날짜가 DOY 216을 지나면 자동으로 learned 값이 나옵니다.

### 4-3. Stage-1 alert 없음

**입력** (`examples/04c_noalert_request.json`)

```json
{"pest": "sheath_blight", "site_id": "24950_66616", "year": 2018, "include_diagnostics": true}
```

**출력** (`examples/04c_noalert_response.json`, 발췌)

```json
{
  "stage1": {"alert_fired": false, "alert_tstar_doy": null,
             "gate_method": "dispatch_group_tau", "alert_source": "stage1_error"},
  "stage2": {"learned_stage2": null,
             "climatology": {"mu_doy": 203.66, "variant": "mean_mid", ...},
             "recommended_source": "climatology"},
  "final_prediction": {"source": "climatology", "mu_doy": 203.66,
                       "selected_offset": null, "fallback_triggered": false}
}
```

| 항목 | 값 |
|---|---|
| `final_mu_doy` 출처 | climatology |
| `variant` | `mean_mid` |
| `learned_output_status` | 빈칸 (차단 코드 아님) |
| `error_reason` | `Stage-1 fired no alert for pest=... (gate=dispatch_group_tau); no manual alert_tstar_doy supplied. ...` |
| batch 동작 | `status=fallback`, 계속 진행 |

### 4-4. learned 모델 대상이 아닌 pest

**8종 중 6종(`blast`, `bacterial_blight`, `brown_spot`, `rice_stem_borer_1`,
`rice_stem_borer_2`, `sheath_blight`)은 `recommended_source=climatology`입니다.**
Stage-2가 **성공해도** 최종값은 climatology입니다.

**입력** (`examples/04d_nonlearned_request.json`)

```json
{"pest": "sheath_blight", "site_id": "35694_60137", "year": 2004, "include_diagnostics": true}
```

**출력** (`examples/04d_nonlearned_response.json`, 발췌)

```json
{
  "stage2": {
    "learned_stage2": {"mu_doy": 192.72, "pi_95": {"lower_doy": 183, "upper_doy": 203,
                        "sigma_days": 5.0}, "selected_offset": 45,
                        "output_status": "experimental", "model_kind": "lead_v3"},
    "climatology": {"mu_doy": 203.66, "variant": "mean_mid", ...},
    "recommended_source": "climatology"
  },
  "final_prediction": {"source": "climatology", "mu_doy": 203.66,
                       "selected_offset": null, "fallback_triggered": false}
}
```

| 항목 | 값 |
|---|---|
| `final_mu_doy` 출처 | **climatology (203.66)** — learned 192.72는 참고값 |
| `variant` | `mean_mid` |
| `learned_output_status` | **`experimental`** |
| `error_reason` | 빈칸 |
| `status` | **`success`** (실패가 아님) |
| batch 동작 | 정상 진행 |

**주의**: 이 경우 `status=success`인데 `final_source=climatology`입니다.
`status`만 보고 "learned 값이 쓰였다"고 판단하면 안 됩니다.

### 사례 요약

| 사례 | `status` | `final_source` | `learned_mu_doy` | `learned_output_status` | batch |
|---|---|---|---|---|---|
| 정상 learned (BPH/WBPH) | `success` | `learned_stage2` | 값 있음 | `main` | 계속 |
| 비-learned pest | `success` | `climatology` | 값 있음(미사용) | `experimental` | 계속 |
| Stage-1 alert 없음 | `fallback` | `climatology` / `climatology_no_alert` | 빈칸 | 빈칸 | 계속 |
| 미해소 결측 구간 | `fallback` | `climatology` | 빈칸 | `stage2_window_crosses_unresolved_missing_run` | 계속 |
| offset window 미관측 | `fallback` | `climatology_no_alert` | 빈칸 | `stage2_pending_window_not_yet_observed` | 계속 |
| 실제 오류 | `error` | `climatology` | 빈칸 | 빈칸 | 계속 |

**어떤 경우에도 batch 전체는 중단되지 않습니다** (`batch.run_batch`의 per-row
try/except). 전체 중단은 정책/자산 로드 실패 같은 배치 단위 오류(exit 2)뿐입니다.

---

## 5. historical vs operational 비교

같은 `site_id` / `year` 입력 기준:

| 항목 | historical (모드 A) | operational (모드 B) |
|---|---|---|
| 요청 필드 | `year` 또는 `start_year`+`end_year` | `as_of_date` |
| 기상 사용 범위 | 시즌 전체 (`doy_start`..`doy_end`) | `doy_start`..`as_of_doy`, 이후 NaN padding |
| 그 해 LONG 관측 | **사용** (interval label 생성) | **사용 안 함** |
| Stage-1 라벨 | 실제 interval/right-censored | 항상 `right`, `L=1`, `R=365` |
| Stage-1 코호트 필터 | `filter_labels_by_gap` 적용 | 미적용 (기상 있는 site 전부) |
| Stage-2 입력 | 실제 시즌 → mask | prefix + padding → 동일 mask |
| `as_of < alert+offset` 가드 | **없음** | **있음** (`stage2_pending_window_not_yet_observed`) |
| 미해소 결측 구간 가드 | **없음** | **있음** (`stage2_window_crosses_unresolved_missing_run`) |
| 용도 | 과거 재현·평가 (2002–2022 / 2023 / 2024) | 매일 갱신되는 운영 예보 |

**Stage-2가 미래 기상을 쓰지 않는 근거**: 학습 시 `_mask_to_recent_window`가
`[tstar-window+1, tstar]` (28일) 밖을 전부 0 + miss=1로 채웁니다.
`tstar = alert_tstar_doy - doy_start + 1 + selected_offset`. 즉
`alert + offset` 이후 기상은 모델에 도달할 수 없습니다.
실측 검증: 8종 중 7종이 미래 기상을 난수로 교란해도 `mu_doy` bit-exact,
나머지 1종의 차이는 모델이 아니라 결측 보간 경계 때문이며 그것이 §4-1 가드의 대상입니다
(`reports/stage2_causality.json`).

---

## 6. 데이터 흐름

```
1. 입력                  request.json + daily_weather.csv + LONG(site 좌표·생육)
2. 기상자료 조회·정렬     master를 청크 1회 스캔 → 대표지역 ∩ 요청연도로 필터 → site별 정렬
                         (operational: as_of_doy까지 자르고 시즌 끝까지 NaN padding)
3. Stage-1 feature 생성   결측 보간 → rolling(7/14일) → 좌표·생육 병합 → base X (T, 2·nbase)
                         → site_history 11채널 append(D분기) → 28일 nowcast window → tabular
4. Stage-1 alert 판정     A/D 두 Booster로 코호트 일괄 추론 → temperature 보정
                         → tau·k 연속 교차 첫 시점 = alert_tstar_doy + dispatch 14개 피처
5. Stage-2 window 구성    base X + dispatch 15채널(causal) → tstar = alert - doy_start + 1 + offset
                         → tstar 기준 28일만 남기고 마스킹 → ckpt 통계로 정규화 → (1,1,T,D)
6. learned 또는 fallback  alert 있고 가드 통과 → LiteRT FP16 추론 → mu_rel → mu_doy
                         아니면 learned_stage2 = null (사유 코드 기록)
7. final 결정             recommended_source가 learned_stage2이고 learned가 있으면 learned 채택
                         아니면 climatology(mean_mid) 채택 → PI = mu ± 9.8일
8. 최종 응답              response.json + predictions.csv + run_log.txt (exit 0)
```

---

## 6-A. 매일 운영 권장 방식

목표: 기상 데이터가 하루씩 추가되고, `as_of_date`를 오늘로 두어 병해충 하나에 대해
대표지역 전체를 매일 예측.

### 권장 구조 — 서버 고정 입력 + 요청당 `request.json` 하나

```
/srv/pest/
├── api/                                  ← 압축 푼 패키지 (한 번만)
│   ├── run_predict.py
│   ├── infer/
│   └── assets/                           ← 모델·site_history·climatology·정책 (내장)
├── assets/
│   ├── daily_weather.csv                 ← 매일 갱신되는 마스터
│   ├── LONG_by_pest/RICE_LONG_*.csv      ← 갱신 빈도 낮음
│   └── representative_site_ids_2002_2024.csv
├── runs/<날짜>/<병해충>/request.json      ← 요청당 이 파일 하나만
└── out/<날짜>/<병해충>/                   ← 요청별 고유 출력
```

```bash
#!/bin/bash
set -euo pipefail
ASSETS=/srv/pest/assets
PEST=sheath_blight
TODAY=$(date +%F)
REQ=/srv/pest/runs/$TODAY/$PEST
OUT=/srv/pest/out/$TODAY/$PEST

mkdir -p "$REQ" "$OUT"
cat > "$REQ/request.json" <<EOF
{"mode": "batch",
 "pest": "$PEST",
 "as_of_date": "$TODAY",
 "daily_weather_path": "$ASSETS/daily_weather.csv",
 "long_observation_path": "$ASSETS/LONG_by_pest/RICE_LONG_$PEST.csv",
 "representative_sites_path": "$ASSETS/representative_site_ids_2002_2024.csv",
 "include_diagnostics": false}
EOF

CUDA_VISIBLE_DEVICES="" python /srv/pest/api/run_predict.py \
    --input-dir "$REQ" --output-dir "$OUT"
```

핵심은 **요청마다 `--output-dir`를 날짜/병해충으로 분리**하는 것입니다.
이것만 지키면 여러 병해충을 동시에 돌려도 안전하고, claim 거부도 발생하지 않습니다.

### 여러 병해충 동시 실행

출력 루트를 공유해야 한다면 `--unique-output-subdir`를 쓰세요.

```bash
for PEST in BPH WBPH sheath_blight blast; do
  python /srv/pest/api/run_predict.py \
      --input-dir /srv/pest/runs/$TODAY/$PEST \
      --output-dir /srv/pest/out/$TODAY \
      --unique-output-subdir &
done
wait
# → /srv/pest/out/<날짜>/BPH_<날짜>_<UTC타임스탬프>_pid<PID>/ ... 4벌 모두 보존
```

### 여러 연도 batch (historical, 평가용)

프로젝트의 분할 경계(2002–2022 / 2023 / 2024)를 한 번에 재현할 때 씁니다.

```bash
ASSETS=/srv/pest/assets
REQ=/tmp/req_span; OUT=/tmp/out_span; mkdir -p $REQ $OUT

cat > $REQ/request.json <<EOF
{"mode": "batch",
 "pest": "sheath_blight",
 "start_year": 2023,
 "end_year": 2024,
 "daily_weather_path": "$ASSETS/daily_weather.csv",
 "long_observation_path": "$ASSETS/LONG_by_pest/RICE_LONG_sheath_blight.csv",
 "representative_sites_path": "$ASSETS/representative_site_ids_2002_2024.csv"}
EOF

python run_predict.py --input-dir $REQ --output-dir $OUT
wc -l $OUT/predictions.csv        # 1717 = 헤더 1 + 858 site × 2 연도
```

daily 마스터는 요청당 **한 번만** 스캔되며, 요청한 모든 연도를 한 번에 걸러냅니다
(`cohort.load_daily_cohort`). 연도를 늘려도 마스터를 다시 읽지 않습니다.

### 동일 output-dir 충돌 시 동작

```bash
OUT=/tmp/out_collide; mkdir -p $OUT
python run_predict.py --input-dir $REQ --output-dir $OUT   # 1회차 → exit 0
python run_predict.py --input-dir $REQ --output-dir $OUT   # 2회차 → exit 1
```

2회차 출력:

```
[run_predict] ERROR: output directory already contains results: response.json, predictions.csv, run_log.txt
  dir: /tmp/out_collide
  Refusing to overwrite — another request may have written these.
  Choose one:
    * point --output-dir at a fresh directory (recommended for concurrent runs), or
    * pass --unique-output-subdir (or "unique_output_subdir": true) to auto-create a per-run subfolder, or
    * pass --overwrite (or "overwrite": true) to replace them.
```

**거부된 실행은 아무것도 쓰지 않습니다** — 1회차 결과는 그대로 보존됩니다
(실측: `predictions.csv` SHA-256 불변).

동시 실행이면 메시지가 달라집니다. 세 프로세스가 빈 디렉터리를 동시에 봐도
`.run_claim` 선점 덕분에 **1건만 통과하고 2건은 exit 1**입니다.

```
[run_predict] ERROR: output directory is claimed by another run (pid=12345 ...).
```

### 여러 병해충 동시 실행 (`--unique-output-subdir`)

```bash
ASSETS=/srv/pest/assets
TODAY=$(date +%F)
OUT=/srv/pest/out/$TODAY; mkdir -p $OUT

for PEST in BPH WBPH sheath_blight; do
  REQ=/srv/pest/runs/$TODAY/$PEST; mkdir -p $REQ
  cat > $REQ/request.json <<EOF
{"mode": "batch", "pest": "$PEST", "as_of_date": "$TODAY",
 "daily_weather_path": "$ASSETS/daily_weather.csv",
 "long_observation_path": "$ASSETS/LONG_by_pest/RICE_LONG_$PEST.csv",
 "representative_sites_path": "$ASSETS/representative_site_ids_2002_2024.csv"}
EOF
  python run_predict.py --input-dir $REQ --output-dir $OUT --unique-output-subdir &
done
wait

ls $OUT
# BPH_20260722_20260722T041121268219_pid1940846
# WBPH_20260722_20260722T041121267409_pid1940847
# sheath_blight_20260722_20260722T041121268809_pid1940848
```

폴더명은 `<pest>_<시간지정>_<UTC타임스탬프(마이크로초)>_pid<PID>`이며, 시간지정의
하이픈은 제거됩니다(`2026-07-22` → `20260722`, `inputs.unique_run_dir`).
같은 마이크로초에 시작해도 PID가 달라 충돌하지 않습니다. 3벌 모두 보존됩니다.

### 하지 말아야 할 것

| 하지 말 것 | 이유 |
|---|---|
| `--output-dir` 생략 | 전 요청이 `<pkg>/output`을 공유 → claim 거부 또는 덮어쓰기 |
| 여러 요청에 같은 `--output-dir` | 1건만 성공하고 나머지는 exit 1 |
| daily 마스터를 요청마다 복사 | 불필요. 경로만 지정하면 원본을 직접 읽음 |
| 요청·결과를 ZIP으로 포장 | 지원하지 않으며 필요 없음 |
| 실패 후 습관적으로 `--overwrite` | 직전 성공 결과를 지울 수 있음. 원인부터 확인 |

---

## 7. 운영자 주의사항

### 7-1. `learned_mu_doy` ≠ `final_mu_doy`

가장 흔한 오해입니다. **8종 중 6종은 Stage-2가 성공해도 최종값이 climatology입니다.**

- `learned_mu_doy` — Stage-2 모델의 원출력. 참고값
- `final_mu_doy` — 정책(`recommended_source`)이 실제로 채택한 값. **사용자가 쓸 값**

`recommended_source=learned_stage2`인 병해충: **BPH, WBPH 뿐**.

### 7-2. fallback은 오류가 아님

`status=fallback`은 정상 동작입니다. 경보가 없거나, 운영 시점상 아직 이르거나,
결측 때문에 신뢰할 수 없는 경우 **숫자를 만들어내지 않고** 기후평년으로 답합니다.
`status=error`만 실제 문제입니다.

### 7-3. FP16 0.01일 표시 차이

Stage-2가 FP16이라 기존 PyTorch 대비 `mu_doy`가 소수 둘째 자리에서 0.01일 다를 수
있습니다 (예: 192.75 vs 192.74). 예측구간(`pi_95`)과 정책 결정은 동일합니다.
운영상 무시해도 되는 수준이며, 이 차이로 경보 여부나 구간이 바뀌지 않습니다.

### 7-4. 결측 자료의 batch 영향 범위

**해당 site에만 영향을 줍니다.** 한 site의 결측·오류가 다른 site의 결과를 바꾸거나
batch를 중단시키지 않습니다 (`batch.run_batch` per-row try/except, 인공 사례로 검증됨).
단, 대표지역 site 선택 자체가 바뀌면 행 구성이 달라지므로,
`year_available_site_count`와 `requested_count`를 `run_log.txt`에서 확인하세요.

### 7-5. 출력 디렉터리는 요청마다 분리하세요

API는 요청별 하위 폴더를 **자동으로 만들지 않습니다** (`--unique-output-subdir`를
주지 않는 한). 파일명이 고정이므로 같은 디렉터리를 재사용하면 exit 1로 거부되고,
`--overwrite`를 주면 이전 결과가 사라집니다.

- 정상 종료 시 `.run_claim`은 자동 삭제됩니다
- exit 1(충돌 거부)은 **아무것도 쓰지 않고** 종료합니다 — 기존 결과는 안전합니다
- 배치 전체 실패도 기존 결과를 보존합니다

### 7-6. status로 learned / fallback / error 구분

```
status == "success"  and final_source == "learned_stage2"  → Stage-2 값이 최종
status == "success"  and final_source == "climatology"     → Stage-2는 돌았지만 정책상 미채택
status == "fallback"                                       → 정상 fallback (learned_output_status로 사유 확인)
status == "error"                                          → 실제 문제. error_reason 확인
```

단일 예측(`response.json`)에는 `status`가 없으므로
`final_prediction.source`와 `stage2.learned_stage2`의 null 여부로 판단하세요.

---

## 사용자가 실제로 확인해야 할 핵심 필드 5개

| 우선순위 | 필드 | 왜 |
|---|---|---|
| 1 | **`final_mu_doy`** | 실제로 사용할 예측 시점(DOY). `learned_mu_doy`가 아님 |
| 2 | **`final_pi95_lower` / `final_pi95_upper`** | 예측구간. 단일 시점보다 이 범위로 판단 |
| 3 | **`status`** | `success` / `fallback` / `error` 구분 |
| 4 | **`final_source`** | 값의 출처 (`learned_stage2` / `climatology` / `climatology_no_alert`) |
| 5 | **`alert_tstar_doy`** | Stage-1 경보 시점. 방제 의사결정의 시작점 |

문제 추적이 필요할 때만 `learned_output_status`와 `error_reason`을 보면 됩니다.
