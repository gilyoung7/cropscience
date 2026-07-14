#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
대표 site_id ↔ 최근접 ASOS 관측소(1개) 매핑 (D 방식, K=1).

- 대표 site 목록의 전체 고유 site_id를 추출
- site 대표 좌표는 병해충 LONG(좌표-위도/경도)에서 6자리 반올림 + median
- ASOS 후보는 META_ASOS를 2024-12-31 기준 활성 이력으로 정리(관측소당 1좌표)
- 하버사인(R=6,371,000 m, 입력순서 lat,lon)으로 각 site에 최근접 ASOS 1개 매핑
- 동거리 tie-break: 관측소 ID 숫자 오름차순(작은 ID)
- 원본 파일은 절대 수정하지 않음(read-only)
"""

from pathlib import Path
import sys
import numpy as np
import pandas as pd

# ===== 경로 =====
NB_DIR       = Path("/home/gpu4080/research/cropscience/notebooks")
REP_CSV      = NB_DIR / "representative_site_ids_2002_2024.csv"
META_CSV     = NB_DIR / "META_ASOS.csv"
LONG_DIR     = Path("/home/gpu4080/ygdata/rice/LONG_by_pest")

OUT_CSV      = NB_DIR / "unique_representative_site_nearest_asos.csv"
DIAG_CSV     = NB_DIR / "unique_representative_site_nearest_asos_diagnostics.csv"
MISSING_CSV  = NB_DIR / "sites_missing_coordinates.csv"

# ===== 상수 =====
EARTH_R_M = 6_371_000.0                 # 기존 로직과 동일한 지구 반경(m)
REF_DATE  = pd.Timestamp("2024-12-31")  # 활성 관측소 판단 기준일
LAT_RANGE = (30.0, 40.0)
LON_RANGE = (120.0, 135.0)

# 대표 목록에 대응하는 site 좌표 소스: LONG_by_pest 전체(좌표는 site 단위 속성).
# 이화명나방은 1화기/2화기 2개 파일이므로 8개 파일 모두 좌표원으로 사용한다.
LONG_FILES = sorted(LONG_DIR.glob("RICE_LONG_*.csv"))


def haversine_m(lat1, lon1, lat2, lon2):
    """벡터화 하버사인 거리(m). 입력 순서 latitude, longitude (degrees)."""
    lat1 = np.radians(lat1); lon1 = np.radians(lon1)
    lat2 = np.radians(lat2); lon2 = np.radians(lon2)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return 2.0 * EARTH_R_M * np.arcsin(np.sqrt(a))


def main():
    print("=" * 78)
    print("대표 site → 최근접 ASOS 매핑 (K=1, D 방식)")
    print("=" * 78)

    # -------------------------------------------------------------------------
    # 1. 대표 site 로드 & 고유 site_id 추출
    # -------------------------------------------------------------------------
    rep = pd.read_csv(REP_CSV, encoding="utf-8-sig", dtype=str)
    rep.columns = [c.strip() for c in rep.columns]
    for c in ["pest", "시도", "시군구", "읍면동", "site_id"]:
        if c not in rep.columns:
            raise SystemExit(f"[에러] 대표 파일에 '{c}' 컬럼 없음. 실제 컬럼={list(rep.columns)}")
    rep["site_id"] = rep["site_id"].astype(str).str.strip()

    n_rep_rows = len(rep)
    unique_sids = sorted(rep["site_id"].unique().tolist())
    n_unique = len(unique_sids)

    # 여러 병해충에서 중복 사용된 site_id 수
    pest_per_sid = rep.groupby("site_id")["pest"].nunique()
    n_multi_pest = int((pest_per_sid > 1).sum())

    # 동일 site_id에 시도/시군구/읍면동이 다르게 붙은 충돌 탐지
    loc_conflicts = []
    loc_group = rep.groupby("site_id")[["시도", "시군구", "읍면동"]]
    for sid, g in loc_group:
        uniq = g.drop_duplicates()
        if len(uniq) > 1:
            loc_conflicts.append((sid, uniq.values.tolist()))

    # -------------------------------------------------------------------------
    # 2. 대표 좌표 생성 (LONG에서 site_id별 median, 6자리 반올림)
    # -------------------------------------------------------------------------
    sid_set = set(unique_sids)
    coord_parts = []
    used_long = []
    for fp in LONG_FILES:
        try:
            df = pd.read_csv(fp, encoding="utf-8-sig",
                             usecols=["site_id", "좌표-위도", "좌표-경도"], dtype=str)
        except ValueError:
            # 컬럼명이 다르면 헤더만 확인 후 스킵
            hdr = pd.read_csv(fp, encoding="utf-8-sig", nrows=0).columns.tolist()
            print(f"[경고] {fp.name}: 기대 컬럼 없음 → 스킵. 실제 컬럼={hdr}")
            continue
        df["site_id"] = df["site_id"].astype(str).str.strip()
        df = df[df["site_id"].isin(sid_set)].copy()
        used_long.append((str(fp), len(df)))
        if len(df):
            coord_parts.append(df)

    if not coord_parts:
        raise SystemExit("[에러] LONG에서 좌표를 하나도 읽지 못함.")

    allc = pd.concat(coord_parts, ignore_index=True)
    allc["lat"] = pd.to_numeric(allc["좌표-위도"], errors="coerce")
    allc["lon"] = pd.to_numeric(allc["좌표-경도"], errors="coerce")
    allc = allc.dropna(subset=["lat", "lon"])                 # 변환불가/NaN 제거
    allc["lat"] = allc["lat"].round(6)
    allc["lon"] = allc["lon"].round(6)

    site_coords = (allc.groupby("site_id", as_index=False)
                        .agg(site_lat=("lat", "median"),
                             site_lon=("lon", "median")))

    have = set(site_coords["site_id"])
    missing = sorted(sid_set - have)
    n_have = len(have & sid_set)
    n_missing = len(missing)

    # 좌표 범위/스왑 의심 검사 (자동수정 금지, 보고만)
    range_flags = site_coords[
        (site_coords["site_lat"] < LAT_RANGE[0]) | (site_coords["site_lat"] > LAT_RANGE[1]) |
        (site_coords["site_lon"] < LON_RANGE[0]) | (site_coords["site_lon"] > LON_RANGE[1])
    ].copy()
    # 위/경도 뒤바뀜 의심: lat가 경도범위에 들어가고 lon이 위도범위에 들어가는 경우
    swap_susp = site_coords[
        (site_coords["site_lat"] >= LON_RANGE[0]) & (site_coords["site_lat"] <= LON_RANGE[1]) &
        (site_coords["site_lon"] >= LAT_RANGE[0]) & (site_coords["site_lon"] <= LAT_RANGE[1])
    ].copy()

    # -------------------------------------------------------------------------
    # 3. META_ASOS 이력 정리 (2024-12-31 기준 활성, 관측소당 1행)
    # -------------------------------------------------------------------------
    # 첫 줄이 빈 줄이라 헤더 위치를 자동 탐지
    raw = pd.read_csv(META_CSV, encoding="utf-8-sig", header=None, dtype=str)
    hdr_idx = None
    for i in range(len(raw)):
        row = raw.iloc[i].tolist()
        if any(isinstance(x, str) and x.strip() == "지점" for x in row):
            hdr_idx = i
            break
    if hdr_idx is None:
        raise SystemExit("[에러] META_ASOS에서 '지점' 헤더행을 찾지 못함.")
    meta = pd.read_csv(META_CSV, encoding="utf-8-sig", header=hdr_idx, dtype=str)
    meta.columns = [c.strip() for c in meta.columns]
    meta = meta.dropna(how="all").reset_index(drop=True)
    for c in ["지점", "시작일", "종료일", "위도", "경도", "지점명"]:
        if c not in meta.columns:
            raise SystemExit(f"[에러] META_ASOS에 '{c}' 컬럼 없음. 실제 컬럼={list(meta.columns)}")

    meta_n_rows = len(meta)
    meta_n_unique = meta["지점"].astype(str).str.strip().nunique()

    meta["지점"] = meta["지점"].astype(str).str.strip()
    meta["시작일_dt"] = pd.to_datetime(meta["시작일"], format="%Y-%m-%d", errors="coerce")
    meta["종료일_dt"] = pd.to_datetime(meta["종료일"], format="%Y-%m-%d", errors="coerce")
    meta["_orig"] = np.arange(len(meta))

    # 활성 조건: 시작일 <= 기준일  AND  (종료일 공란 OR 종료일 >= 기준일)
    active_mask = (meta["시작일_dt"] <= REF_DATE) & (
        meta["종료일_dt"].isna() | (meta["종료일_dt"] >= REF_DATE)
    )
    active = meta[active_mask].copy()

    # 관측소당 1행 선택: 시작일 최신 → 동률이면 원본순서 뒤
    active = active.sort_values(["지점", "시작일_dt", "_orig"],
                                ascending=[True, True, True])
    chosen = active.groupby("지점", as_index=False).tail(1).copy()

    # 좌표 숫자화 & 결측 제외
    chosen["station_lat"] = pd.to_numeric(chosen["위도"], errors="coerce")
    chosen["station_lon"] = pd.to_numeric(chosen["경도"], errors="coerce")
    before_coord = chosen["지점"].nunique()
    chosen = chosen.dropna(subset=["station_lat", "station_lon"]).copy()
    after_coord = chosen["지점"].nunique()
    n_coord_dropped = before_coord - after_coord

    n_active_stations = before_coord           # 활성(좌표결측 제외 전) 관측소 수
    n_dedup_removed = int(active_mask.sum()) - before_coord  # 활성행 중 중복이력 제거 수

    stn_ids  = chosen["지점"].to_numpy()
    stn_idn  = chosen["지점"].astype(int).to_numpy()   # tie-break용 숫자 ID
    stn_name = chosen["지점명"].astype(str).to_numpy()
    stn_lat  = chosen["station_lat"].to_numpy(dtype=float)
    stn_lon  = chosen["station_lon"].to_numpy(dtype=float)
    stn_id_set = set(stn_ids.tolist())

    # -------------------------------------------------------------------------
    # 4. 최근접 ASOS 매핑 (K=1)
    # -------------------------------------------------------------------------
    map_rows = []
    for r in site_coords.itertuples(index=False):
        d = haversine_m(r.site_lat, r.site_lon, stn_lat, stn_lon)   # (N_station,)
        dmin = d.min()
        cand = np.where(d == dmin)[0]                # 동거리 후보
        # tie-break: 관측소 ID 숫자 오름차순(작은 ID)
        pick = cand[np.argmin(stn_idn[cand])]
        map_rows.append({
            "site_id": r.site_id,
            "site_lat": r.site_lat,
            "site_lon": r.site_lon,
            "관측소_id": stn_ids[pick],
            "관측소명": stn_name[pick],
            "station_lat": float(stn_lat[pick]),
            "station_lon": float(stn_lon[pick]),
            "distance_m": float(d[pick]),
        })
    mapping = pd.DataFrame(map_rows)

    # -------------------------------------------------------------------------
    # 5/6. 출력 (최종 + 진단). 좌표 미확보 site는 별도 저장 & 검증 실패 처리.
    # -------------------------------------------------------------------------
    if n_missing > 0:
        miss_df = rep[rep["site_id"].isin(missing)][
            ["site_id", "시도", "시군구", "읍면동"]
        ].drop_duplicates("site_id").sort_values("site_id")
        miss_df.to_csv(MISSING_CSV, index=False, encoding="utf-8-sig")

    # 진단용 대표 행정구역: 대표 파일 첫 등장 행 사용
    rep_loc = rep.drop_duplicates("site_id")[["site_id", "시도", "시군구", "읍면동"]]
    diag = mapping.merge(rep_loc, on="site_id", how="left")
    diag = diag[["site_id", "시도", "시군구", "읍면동",
                 "site_lat", "site_lon",
                 "관측소_id", "관측소명", "station_lat", "station_lon", "distance_m"]]
    SORT_KEY = "distance_m (오름차순)"
    diag = diag.sort_values("distance_m").reset_index(drop=True)

    final = mapping[["site_id", "관측소_id"]].drop_duplicates("site_id").sort_values("site_id")

    # -------------------------------------------------------------------------
    # 7. 검증
    # -------------------------------------------------------------------------
    checks = {}
    checks["1_고유site수==최종행수"] = (n_unique == len(final)) and (n_missing == 0)
    checks["2_최종CSV_site_id중복없음"] = final["site_id"].is_unique
    checks["3_모든site에관측소1개"] = final["관측소_id"].notna().all() and len(final) == final["site_id"].nunique()
    checks["4_좌표미확보site_0"] = (n_missing == 0)
    checks["5_선택관측소_모두활성후보에존재"] = set(final["관측소_id"]).issubset(stn_id_set)
    checks["6_distance_유효(no NaN/inf/음수)"] = bool(
        np.isfinite(mapping["distance_m"]).all() and (mapping["distance_m"] >= 0).all()
    )
    checks["7_최종컬럼==[site_id,관측소_id]"] = list(final.columns) == ["site_id", "관측소_id"]
    checks["8_원본대표에서누락site_0"] = set(final["site_id"]) == sid_set if n_missing == 0 else False
    # 9는 아래에서 원본 mtime 비교

    all_pass = all(checks.values())

    # 좌표 미확보 site가 있으면 최종 파일 저장하지 않고 실패 처리
    if n_missing == 0 and all_pass:
        final.to_csv(OUT_CSV, index=False, encoding="utf-8-sig")
        diag.to_csv(DIAG_CSV, index=False, encoding="utf-8-sig")
        saved = True
    else:
        saved = False

    # -------------------------------------------------------------------------
    # 리포트
    # -------------------------------------------------------------------------
    dist = mapping["distance_m"].to_numpy(dtype=float)
    def km(x): return x / 1000.0

    print("\n----- [1] 대표 site -----")
    print(f"입력 전체 행 수                 : {n_rep_rows}")
    print(f"고유 site_id 수                 : {n_unique}")
    print(f"여러 병해충 중복 사용 site_id 수: {n_multi_pest}")
    print(f"행정구역 충돌 site_id 수        : {len(loc_conflicts)}")
    if loc_conflicts:
        print("  [충돌 상세] (site_id → [ [시도,시군구,읍면동], ... ])")
        for sid, combos in loc_conflicts[:50]:
            print(f"    {sid} → {combos}")
        if len(loc_conflicts) > 50:
            print(f"    ... 외 {len(loc_conflicts)-50}건")

    print("\n----- [2] 좌표 소스 & 확보 -----")
    print("LONG 파일별 사용 경로 (매칭행수):")
    for path, nmatch in used_long:
        print(f"  {path}  (matched={nmatch})")
    print(f"좌표 확보 site 수  : {n_have}")
    print(f"좌표 미확보 site 수: {n_missing}")
    if n_missing:
        print(f"  → 미확보 목록 저장: {MISSING_CSV}")
        print(f"  → 미확보 site 예시: {missing[:10]}")
    if len(range_flags):
        print(f"[보고] 좌표 범위 이탈(lat30-40/lon120-135) site: {len(range_flags)}건")
        print(range_flags.head(20).to_string(index=False))
    else:
        print("[보고] 좌표 범위 이탈 site: 0건")
    if len(swap_susp):
        print(f"[보고] 위/경도 뒤바뀜 의심 site: {len(swap_susp)}건 (자동수정 안 함)")
        print(swap_susp.head(20).to_string(index=False))
    else:
        print("[보고] 위/경도 뒤바뀜 의심 site: 0건")

    print("\n----- [3] META_ASOS 정리 -----")
    print(f"META_ASOS 전체 행 수            : {meta_n_rows}")
    print(f"META_ASOS 고유 관측소 ID 수     : {meta_n_unique}")
    print(f"2024-12-31 기준 활성 관측소 수  : {n_active_stations}")
    print(f"중복 이력 제거 수(활성행-관측소): {n_dedup_removed}")
    print(f"좌표 결측 제외 관측소 수        : {n_coord_dropped}")
    print(f"실제 매핑에 사용된 활성 ASOS 수 : {after_coord}")

    print("\n----- [4] 최근접 거리 통계 -----")
    print(f"실제 매핑에 쓰인 고유 ASOS 수   : {final['관측소_id'].nunique()}")
    print(f"최소   : {dist.min():.1f} m ({km(dist.min()):.3f} km)")
    print(f"평균   : {dist.mean():.1f} m ({km(dist.mean()):.3f} km)")
    print(f"중앙값 : {np.median(dist):.1f} m ({km(np.median(dist)):.3f} km)")
    print(f"최대   : {dist.max():.1f} m ({km(dist.max()):.3f} km)")
    print(f"10km 초과 site 수: {int((dist > 10_000).sum())}")
    print(f"20km 초과 site 수: {int((dist > 20_000).sum())}")
    print(f"30km 초과 site 수: {int((dist > 30_000).sum())}")

    print("\n----- [5/6] 출력 -----")
    print(f"진단 정렬 기준: {SORT_KEY}")
    print(f"최종 CSV 절대경로: {OUT_CSV}  (saved={saved})")
    print(f"진단 CSV 절대경로: {DIAG_CSV}  (saved={saved})")

    print("\n----- [7] 검증 -----")
    for k, v in checks.items():
        print(f"  [{'PASS' if v else 'FAIL'}] {k}")
    print(f"\n모든 검증 통과 여부: {'PASS' if all_pass else 'FAIL'}")
    if not saved:
        print("※ 검증 실패 또는 좌표 미확보 → 최종/진단 CSV를 저장하지 않았습니다.")

    return 0 if all_pass else 1


if __name__ == "__main__":
    sys.exit(main())
