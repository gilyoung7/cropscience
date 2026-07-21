# Lightweight portable pest-timing API

Same behaviour and output contract as the deployed API, with **no PyTorch, no
TensorFlow, no scikit-learn, and no `.pt` checkpoints**.

```
daily weather CSV + LONG observations + request
  → Stage 1: XGBoost JSON (xgboost.Booster) → alert_tstar + 14 dispatch features
  → Stage 2: LiteRT FP16 → mu_doy + 95% prediction interval
  → fallback policy (unchanged) → response.json / predictions.csv / run_log.txt
```

Model generation: **`lead_v3_final` (the deployed models)**. The newer DN model is
deliberately excluded.

**Full input/output reference: [`docs/API_INPUT_OUTPUT.md`](docs/API_INPUT_OUTPUT.md)**
— every request field, every output column, fallback cases, operator notes.
This README is the quick start.

| | measured |
|---|---:|
| package assets (FP16 only) | **29.47 MB** |
| runtime venv — macOS / arm64 | **259 MB** |
| runtime venv — Linux / x86_64 | **~1.1 GB** (xgboost pulls `nvidia-nccl-cu12`; unused at inference) |
| production archive (`.tar.gz`) | **8.6 MB** |

---

## Install

```bash
cd /srv/pest                       # anywhere you keep the package
python3 -m venv .venv-lightweight-api
./.venv-lightweight-api/bin/python -m pip install \
    -r api_handoff_litert_portable/requirements-runtime.txt
```

Installs 14 packages: `xgboost, ai-edge-litert, numpy, pandas, PyYAML` + 9
transitive. No torch, no tensorflow, no scikit-learn.

---

## Run

```bash
python run_predict.py --input-dir IN --output-dir OUT \
    [--stage2-variant fp16|fp32] [--representative-sites PATH] \
    [--overwrite] [--unique-output-subdir]
```

| 옵션 | 기본값 | 의미 |
|---|---|---|
| `--input-dir` | `<pkg>/input` | `request.json`을 찾는 디렉터리 |
| `--output-dir` | `<pkg>/output` | 결과를 쓸 디렉터리. **요청마다 분리 권장** |
| `--stage2-variant` | `fp16` | Stage-2 정밀도. FP32는 `--include-fp32` 빌드에만 존재 |
| `--representative-sites` | 없음 | 대표지역 CSV 경로 (batch). 요청 필드가 우선 |
| `--overwrite` | 꺼짐 | 기존 결과를 덮어씀. **기본은 거부** |
| `--unique-output-subdir` | 꺼짐 | `--output-dir` 아래 요청별 고유 폴더 생성 |

뒤의 두 옵션은 `request.json`의 `"overwrite": true` /
`"unique_output_subdir": true`로도 지정할 수 있습니다.

### ZIP은 필요 없습니다

요청을 ZIP으로 묶지 않고, 결과도 ZIP으로 만들지 않습니다. 입력 파일은 **복사하지
않고 원본 경로에서 직접 읽으며 수정하지 않습니다.** 대용량 daily 마스터를 요청마다
복제할 필요가 없습니다.

---

## 빠른 시작 1 — 단일 site 예측

```bash
IN=/tmp/in; OUT=/tmp/out; mkdir -p $IN $OUT
cat > $IN/request.json <<'EOF'
{"pest": "WBPH", "site_id": "33908_67063", "year": 2011}
EOF

python run_predict.py --input-dir $IN --output-dir $OUT
```

`daily_weather_path` / `long_observation_path`를 생략했으므로
`$IN/daily_weather.csv`와 `$IN/long_observation.csv`(없으면
`$IN/LONG_by_pest/RICE_LONG_WBPH.csv`)를 찾습니다. 아래 "입력 두 방식" 참고.

응답 요지:

```json
{"stage1": {"alert_fired": true, "alert_tstar_doy": 171, ...},
 "stage2": {"learned_stage2": {"mu_doy": 245.82,
              "pi_95": {"lower_doy": 236, "upper_doy": 256, "sigma_days": 5.0},
              "selected_offset": 45, "output_status": "main"},
            "recommended_source": "learned_stage2"},
 "final_prediction": {"source": "learned_stage2", "mu_doy": 245.82, ...},
 "backends": {"stage1_backend": "xgboost_json", "stage2_backend": "litert_fp16"}}
```

`backends`가 배포 스키마 대비 유일한 추가 필드입니다. 기존 필드는 삭제·개명·형변경이
없습니다.

---

## 빠른 시작 2 — 대표지역 전체 batch (운영에서 쓰는 형태)

**한 번의 batch 요청은 병해충 1종을 처리합니다.** 여러 병해충은 요청을 나눠 실행합니다.

```bash
ASSETS=/srv/pest/assets            # 서버 고정 입력
REQ=/tmp/req; OUT=/tmp/out2; mkdir -p $REQ $OUT

cat > $REQ/request.json <<EOF
{"mode": "batch",
 "pest": "sheath_blight",
 "year": 2004,
 "daily_weather_path": "$ASSETS/daily_weather.csv",
 "long_observation_path": "$ASSETS/LONG_by_pest/RICE_LONG_sheath_blight.csv",
 "representative_sites_path": "$ASSETS/representative_site_ids_2002_2024.csv"}
EOF

python run_predict.py --input-dir $REQ --output-dir $OUT
```

경로 3개를 지정했으므로 **입력 디렉터리에는 `request.json` 하나만** 있으면 됩니다.

### batch 출력이 의미하는 것

> **대표 site 하나 = `predictions.csv` 한 행.**
> `sheath_blight`의 대표 site가 **858개**면 `predictions.csv`는 **858행**입니다.

- 한 행 = 하나의 `site_id` + 하나의 대상 연도
- **fallback site도 행이 유지됩니다.** 경보가 없거나 운영상 아직 이른 site는
  삭제되지 않고 climatology 값으로 채워진 행이 남습니다
  (실측: 858행 = success 134 + fallback 724)
- 여러 연도를 실행하면 **site × year** 만큼 생성됩니다
  (실측: `start_year=2023, end_year=2024` → **1716행** = 858 × 2)

### 시간 지정 3가지

| 필드 | 모드 | 예 |
|---|---|---|
| `year` | historical — 한 시즌 | `"year": 2004` |
| `start_year` + `end_year` | historical — 연도 범위 | `"start_year": 2002, "end_year": 2022` |
| `as_of_date` | **operational** — 그날까지의 기상으로 예측 | `"as_of_date": "2026-07-22"` |

셋 중 **하나는 반드시** 있어야 합니다 (없으면 exit 2).
`as_of_date`만 실제 날짜 문자열이고, 나머지 `*_doy` 값은 전부 DOY 정수입니다.
`as_of_date` 모드는 그 시즌의 관측을 읽지 않습니다(look-ahead 방지) — 상세는
[`docs/API_INPUT_OUTPUT.md`](docs/API_INPUT_OUTPUT.md) §5.

---

## 빠른 시작 3 — 매일 운영

```bash
ASSETS=/srv/pest/assets
PEST=sheath_blight
TODAY=$(date +%F)
REQ=/srv/pest/runs/$TODAY/$PEST
OUT=/srv/pest/out/$TODAY/$PEST          # 요청별로 분리
mkdir -p "$REQ" "$OUT"

cat > "$REQ/request.json" <<EOF
{"mode": "batch", "pest": "$PEST", "as_of_date": "$TODAY",
 "daily_weather_path": "$ASSETS/daily_weather.csv",
 "long_observation_path": "$ASSETS/LONG_by_pest/RICE_LONG_$PEST.csv",
 "representative_sites_path": "$ASSETS/representative_site_ids_2002_2024.csv"}
EOF

python run_predict.py --input-dir "$REQ" --output-dir "$OUT"
```

여러 병해충을 동시에 돌리면서 출력 루트를 공유해야 한다면:

```bash
for PEST in BPH WBPH sheath_blight blast; do
  python run_predict.py --input-dir /srv/pest/runs/$TODAY/$PEST \
      --output-dir /srv/pest/out/$TODAY --unique-output-subdir &
done
wait
# → out/<날짜>/BPH_<날짜>_<UTC타임스탬프>_pid<PID>/ ... 4벌 모두 보존
```

---

## 입력 두 방식

**(a) 디렉터리 배치** — 고정 파일명을 `--input-dir`에 둡니다.

| 파일 | 비고 |
|---|---|
| `request.json` | 필수 |
| `daily_weather.csv` | 한국어 스키마. 여러 site/연도 포함 가능 |
| `long_observation.csv` | Layout A — 해당 site만 |
| `LONG_by_pest/RICE_LONG_<pest>.csv` | Layout B — Layout A가 없을 때 사용 |

**(b) 경로 지정** — `request.json`에 위치를 직접 씁니다 (운영 권장).

| 필드 | 대체하는 파일 |
|---|---|
| `daily_weather_path` | `daily_weather.csv` |
| `long_observation_path` | `long_observation.csv` / `LONG_by_pest/…` |
| `representative_sites_path` | `representative_site_ids_2002_2024.csv` |

절대경로는 그대로, 상대경로는 **CWD → `--input-dir` → 패키지 루트** 순으로 찾습니다.
**필드를 생략하면 (a) 방식으로 동작하므로 기존 사용법은 그대로 유지됩니다.**

`pest`는 단일 모드에서 **대소문자를 구분**합니다 (batch는 구분하지 않음).

### 일별 CSV 스키마

`일시`, `일강수량(mm)`, `최고기온(°C)`, `최저기온(°C)`, `평균기온(°C)`,
`평균 풍속(m/s)`, `최대 풍속(m/s)`, `평균 상대습도(%)`, `합계 일조시간(h)`,
`합계 일사량(MJ/m2)`, `GDD10_since_gs` (+ `지점ID`).

`GDD10_since_gs`는 보간하지 않습니다(학습과 동일). 시즌은 해당 병해충의 DOY 구간을
빠짐없이 덮어야 하고, 같은 날짜가 중복되면 안 됩니다. 개별 셀 결측은 허용됩니다.

---

## 출력

| 파일 | 내용 |
|---|---|
| `response.json` | 단일: 전체 응답 / batch: 요약 + `results[]` |
| `predictions.csv` | **site당 한 행.** 단일 16컬럼, batch 18컬럼(`status`, `error_reason` 추가) |
| `run_log.txt` | 실행 이벤트 — 선택된 site 수, 코호트 통계, 소요 시간 |

운영 시 주로 볼 필드:

| 필드 | 의미 |
|---|---|
| `alert_tstar_doy` | Stage-1 경보 DOY. 방제 판단의 시작점 |
| `final_mu_doy` | **실제로 사용할 예측 시점(DOY)** |
| `final_pi95_lower` / `final_pi95_upper` | 예측구간. 단일 시점보다 이 범위로 판단 |
| `final_source` | 값의 출처 — `learned_stage2` / `climatology` / `climatology_no_alert` |
| `status` | `success` / `fallback` / `error` (batch 전용) |
| `learned_output_status` | `main` / `experimental`, 또는 운영 차단 사유 코드 |

> **`status=success`는 "API 실행이 성공했다"는 뜻이지 "모델값이 채택됐다"는 뜻이
> 아닙니다.** 모델값 채택 여부는 **`final_source`** 로 확인하세요.
> 8종 중 `recommended_source=learned_stage2`인 것은 **BPH·WBPH 2종뿐**이고,
> 나머지 6종은 Stage-2가 성공해도 `final_source=climatology`입니다.

---

## 출력 디렉터리 보호

출력 파일명이 고정이라 두 요청이 같은 `--output-dir`를 쓰면 서로를 덮어씁니다.
이를 막는 장치가 있습니다.

- **기존 결과가 있으면 기본적으로 거부** (exit 1). 조용히 덮어쓰지 않습니다.
- **동시 실행은 `.run_claim`으로 차단.** 시작 시 `O_CREAT|O_EXCL`로 디렉터리를
  원자적으로 선점하므로, 같은 output-dir로 3건을 동시에 돌리면 **1건만 실행되고
  나머지 2건은 exit 1로 거부**됩니다. (존재 검사만으로는 셋 다 빈 디렉터리를 보고
  통과해 마지막 하나만 남습니다.)
- **실패가 성공을 덮어쓰지 않습니다.** 배치 전체 실패 시 기존 출력은 보존됩니다.
- 모든 쓰기는 임시 파일 + `os.replace`라 중간에 죽어도 잘린 파일이 남지 않습니다.

권장: **요청마다 고유한 `--output-dir`**, 또는 `--unique-output-subdir`.

`kill -9` 등으로 `.run_claim`이 남으면 `ps -p <pid>`로 확인 후
`rm <output-dir>/.run_claim` 하거나 `--overwrite`로 실행하세요.

---

## Errors

값을 0으로 채워 만들어내는 일은 없습니다.

| exit | 조건 |
|---:|---|
| 0 | 성공 — **Stage-2 실패·차단도 0** (climatology로 응답). `stage2.learned_stage2 == null` 또는 `diagnostics.transformer_error` 확인 |
| 1 | 잘못된 요청(pest/site_id/year, 정책 누락), **출력 디렉터리 충돌** |
| 2 | 입력 파일 누락, 지정 경로 없음, batch 시간 지정 누락 |

자주 보는 메시지: dispatch feature 누락, 시즌 밖 alert, DOY 누락/중복, 기상 컬럼
누락, `site_history.json`에 해당 site-year 없음.

---

## 아직 직접 준비해야 하는 것

| 필요한 것 | 이유 |
|---|---|
| daily weather CSV | 기상 API가 아직 연결되지 않음 (`WeatherProvider`가 연결 지점) |
| LONG 관측 | 좌표 + 생육 + Stage-1 history의 출처 |

8종 중 7종이 site 좌표와 생육 정보를 필요로 하고, BPH만 기상만으로 동작합니다.
둘 다 LONG 파일에서 `LongObsProvider`가 읽습니다.

대표지역 site는 daily 마스터와 **동일한 격자 ID 체계**를 씁니다 — `sheath_blight`
기준 대표 858개가 daily에 **858/858(100%)** 존재하며, 별도 관측소 매핑 없이
그대로 예측됩니다.

---

## Batch 요청 두 형태

```jsonc
// (a) 대표지역 — 병해충 1종 + 시간 지정(year | start_year+end_year | as_of_date)
{"mode":"batch","pest":"BPH","year":2004,
 "representative_sites_path":"/srv/pest/assets/representative_site_ids_2002_2024.csv",
 "max_sites":50}

// (b) generic CSV — 병해충/site/연도 혼합 가능, 입력 행 순서 보존
{"mode":"batch","input_csv":"/srv/pest/req/rows.csv","stage2_variant":"fp16"}
```

`rows.csv`에는 `pest,site_id,year`가 필요합니다(선택: `alert_tstar_doy`).

`input_csv`도 다른 경로 필드와 동일하게 **절대경로는 그대로, 상대경로는
CWD → `--input-dir` → 패키지 루트** 순으로 찾습니다.

generic 모드에서도 최상위 `daily_weather_path` / `long_observation_path` /
`representative_sites_path` / `stage2_variant` / `include_diagnostics`가 각 행에
**상속**됩니다. **행에 같은 이름의 컬럼이 있으면 행 값이 우선**하므로, 병해충이
섞인 CSV에서 행마다 다른 LONG 파일을 지정할 수 있습니다.

```csv
pest,site_id,year,long_observation_path
BPH,33210_56298,2004,                                  # 최상위 값을 상속
WBPH,33908_67063,2011,/srv/pest/assets/LONG_by_pest/RICE_LONG_WBPH.csv
```

빈 셀은 "값 없음"으로 처리되어 최상위 값을 상속합니다.

`max_sites`는 운영/테스트용 상한이며 API 계약의 일부가 아닙니다.

한 행이 실패해도 batch는 중단되지 않고 error 행이 됩니다. **요청 자체**가 잘못되면
(CSV 없음, 컬럼 없음, 알 수 없는 pest) 전체가 exit 2로 실패합니다.
상세: [`analysis/BATCH_IMPLEMENTATION_REPORT.md`](../analysis/BATCH_IMPLEMENTATION_REPORT.md)

---

## FP16 정책

FP16이 운영 기본값입니다. FP32 대비 ~1.56배 작고 속도는 사실상 같으며, 실데이터
기준 최대 mu 오차는 PyTorch 대비 3.6e-03일입니다. FP32는 검증용이라 운영 빌드에
포함되지 않습니다.

FP16의 `mu_doy`는 소수 둘째 자리에서 0.01일 다를 수 있습니다(예: 192.75 vs
192.74). **예측구간과 정책 결정은 동일**하므로 운영상 무시해도 됩니다.

---

## 패키지 빌드 (assets는 커밋되지 않음)

```bash
# 참조 API가 풀려 있고 Stage-2 LiteRT 모델이 빌드돼 있어야 함
python build_package.py --clean
python build_package.py --archive        # dist/*.tar.gz (FP16)
python build_package.py --include-fp32   # 검증용 빌드
```

`.pt`/`.pth`, torch/tensorflow/sklearn import, (운영 빌드에서) FP32 모델이 하나라도
패키지에 들어가면 빌드가 실패합니다.

## Tests

### 새 환경에서 처음 실행할 때

패키지 디렉터리에서 두 줄이면 됩니다.

```bash
python -m pip install -r requirements-test.txt
pytest -q
```

`requirements-test.txt`는 런타임 의존성(`-r requirements-runtime.txt`)에
`pytest`와 `scikit-learn`을 더한 것입니다. **운영 환경에는
`requirements-runtime.txt`만 설치**하세요 — pytest는 들어가지 않습니다.

기대 출력:

```
62 passed, 3 deselected in 1.2s
```

모델·자산·기상 파일 없이 **1초대에** 끝납니다. 배치 요청 필드 상속, `input_csv`
경로 해석, boolean 변환, 출력 충돌 보호, 출력 계약을 검증합니다.
`3 deselected`는 아래 integration 테스트입니다(기본 제외).

### Integration 테스트 (실데이터 필요)

실제 기상 마스터·LONG·대표지역 CSV를 환경변수로 주고 명시적으로 실행합니다.

```bash
PEST_DAILY_MASTER=/srv/pest/assets/daily_weather.csv \
PEST_LONG_DIR=/srv/pest/assets/LONG_by_pest \
PEST_REP_CSV=/srv/pest/assets/representative_site_ids_2002_2024.csv \
pytest -q -m integration
```

대표지역 batch의 site당 1행 생성, 여러 연도의 site × year 행 수, 출력 충돌 거부를
실제 실행으로 확인합니다(약 30초). 환경변수가 없으면 자동으로 skip됩니다.

### 개별 스크립트 (독립 실행도 가능)

일부 테스트는 pytest 없이 단독 실행도 됩니다.

```bash
RPY=/path/to/.venv-lightweight-api/bin/python   # 런타임 전용 venv(torch 없음)
DAILY_MASTER=/srv/pest/assets/daily_weather.csv
LONG_DIR=/srv/pest/assets/LONG_by_pest

$RPY tests/test_no_heavy_dependencies.py  # torch 없는 venv에서 실행해야 의미 있음
$RPY tests/test_smoke_cases.py --daily-master "$DAILY_MASTER" --long-dir "$LONG_DIR"
$RPY tests/test_stage1_portable.py        # 메모리 레이아웃 회귀 (참조 비교는 빌드용 venv 필요)
```

## 모델 교체 후

1. Stage-1 portable JSON(`stage1_xgboost_migration/`) 또는 Stage-2 LiteRT
   (`tflite_conversion/standalone_stage2/build_package.py --clean`) 재생성
2. `python build_package.py --clean --archive`
3. 위 테스트 3종 전부 통과 확인

---

전체 기록: [`docs/lightweight_api_integration_report.md`](../docs/lightweight_api_integration_report.md)
· 설계/분석: [`docs/lightweight_api_integration_plan.md`](../docs/lightweight_api_integration_plan.md)
· **입출력 상세: [`docs/API_INPUT_OUTPUT.md`](docs/API_INPUT_OUTPUT.md)**
