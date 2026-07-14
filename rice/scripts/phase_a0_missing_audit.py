"""
Phase A0 — Pre-train missingness audit for sheath_blight feature candidates.

Scope: report missing rates for raw columns that back the candidate features.
Splits are aligned with the year-split config (train 2002–2021, val 2022, test 2023–24).
We measure raw missingness BEFORE the data_pipeline interpolation step, because
interpolation can only rescue gaps within a (site, year) that has at least one valid row.
"""

from __future__ import annotations

from pathlib import Path
import pandas as pd
import numpy as np

DAILY_PATH = Path("/home/gpu4080/ygdata/rice/1997_2024_RICE_union_all_sites_with_GDD10_since_gs.csv")
LONG_PATH = Path("/home/gpu4080/ygdata/rice/LONG_by_pest/RICE_LONG_잎집무늬마름병.csv")
YEAR_MIN = 2002

TRAIN_YEARS = range(2002, 2022)   # 2002..2021
VAL_YEARS = [2022]
TEST_YEARS = [2023, 2024]

DAILY_CANDIDATES = {
    # raw column name in daily csv -> derived feature(s) it backs
    "평균 상대습도(%)": ["rh_7d_mean", "rh_14d_mean"],
    "일강수량(mm)":    ["rain_14d_sum"],
    "평균 풍속(m/s)":  ["wind_7d_mean"],
    "최대 풍속(m/s)":  ["wind_7d_max"],
}
# DD10 may or may not be present; check separately.
DAILY_DD = "DD10"

LONG_CANDIDATES = ["best_suitability", "best_months", "offset_days", "window_idx"]


def split_of(year: int) -> str:
    if year in VAL_YEARS:
        return "val"
    if year in TEST_YEARS:
        return "test"
    if year in TRAIN_YEARS:
        return "train"
    return "out"


def summarize_overall(s: pd.Series, name: str) -> dict:
    return {
        "feature": name,
        "n": len(s),
        "miss_rate": float(s.isna().mean()),
    }


def summarize_by_split(df: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    for sp, g in df.groupby("split", sort=False):
        rows.append({
            "feature": col,
            "split": sp,
            "n": len(g),
            "miss_rate": float(g[col].isna().mean()),
        })
    return pd.DataFrame(rows)


def summarize_per_site_year(df: pd.DataFrame, col: str, group_cols: list[str]) -> pd.DataFrame:
    """Per (site, year) miss rate distribution, for systematic-vs-random judgment."""
    g = df.groupby(group_cols)[col].apply(lambda s: float(s.isna().mean()))
    return g.rename("miss_rate").reset_index()


def fully_missing_groups(per_group: pd.DataFrame, threshold: float = 1.0) -> pd.DataFrame:
    return per_group[per_group["miss_rate"] >= threshold].copy()


def verdict(test_rate: float, train_rate: float, overall_rate: float) -> str:
    if test_rate > 0.50:
        return "EXCLUDE (test miss > 50%)"
    if overall_rate < 0.10 and test_rate < 0.10:
        return "ADD (all miss < 10%)"
    if test_rate < 0.20 and train_rate > 0.30:
        return "ADD with mask (train high, test low)"
    if test_rate < 0.20:
        return "ADD (test miss < 20%)"
    if 0.10 <= overall_rate <= 0.30:
        return "ADD + explicit fillna"
    if 0.30 < overall_rate <= 0.50:
        return "MAYBE — needs mask review"
    return "UNCERTAIN"


def main() -> None:
    print("=" * 90)
    print("Phase A0 — Missingness audit for sheath_blight candidate features")
    print(f"daily : {DAILY_PATH}")
    print(f"long  : {LONG_PATH}")
    print(f"splits: train={list(TRAIN_YEARS)[0]}..{list(TRAIN_YEARS)[-1]}, val={VAL_YEARS}, test={TEST_YEARS}")
    print("=" * 90)

    # ---- DAILY ----
    daily = pd.read_csv(DAILY_PATH)
    daily.columns = [c.lstrip("﻿") for c in daily.columns]
    daily["date"] = pd.to_datetime(daily["일시"], errors="coerce")
    daily = daily.dropna(subset=["date"]).copy()
    daily["year"] = daily["date"].dt.year.astype(int)
    daily = daily.rename(columns={"지점ID": "site_id"})
    daily["site_id"] = daily["site_id"].astype(str)
    daily = daily[daily["year"] >= YEAR_MIN].copy()
    daily["split"] = daily["year"].apply(split_of)
    daily = daily[daily["split"] != "out"].copy()

    print(f"\n[daily] rows={len(daily):,}  n_sites={daily['site_id'].nunique()}  year_range=({daily['year'].min()},{daily['year'].max()})")
    print(f"[daily] split sizes:")
    print(daily["split"].value_counts().to_string())

    daily_cols = list(DAILY_CANDIDATES.keys())
    if DAILY_DD in daily.columns:
        daily_cols.append(DAILY_DD)
    else:
        print(f"\n[daily] NOTE: '{DAILY_DD}' column missing in daily csv → DD10_7d_sum cannot be sourced from raw.")

    # daily summary tables
    print("\n" + "-" * 90)
    print("DAILY raw columns — overall + per-split miss rates")
    print("-" * 90)
    rows = []
    for col in daily_cols:
        if col not in daily.columns:
            rows.append({"feature": col, "split": "MISSING_COL", "n": 0, "miss_rate": float("nan")})
            continue
        n = len(daily)
        overall = float(daily[col].isna().mean())
        rows.append({"feature": col, "split": "overall", "n": n, "miss_rate": overall})
        for sp in ["train", "val", "test"]:
            sub = daily[daily["split"] == sp]
            rows.append({
                "feature": col, "split": sp, "n": len(sub),
                "miss_rate": float(sub[col].isna().mean()) if len(sub) else float("nan"),
            })
    daily_table = pd.DataFrame(rows)
    pivot = daily_table.pivot_table(index="feature", columns="split", values="miss_rate", aggfunc="first")
    # column order
    col_order = [c for c in ["overall", "train", "val", "test"] if c in pivot.columns]
    pivot = pivot[col_order]
    print(pivot.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "NaN"))

    # per (site, year) breakdown — find fully-missing groups
    print("\n" + "-" * 90)
    print("DAILY — per (site, year) fully-missing groups (miss_rate == 1.0)")
    print("-" * 90)
    for col in daily_cols:
        if col not in daily.columns:
            continue
        per = summarize_per_site_year(daily, col, ["site_id", "year"])
        full = fully_missing_groups(per, 1.0)
        print(f"\n[{col}] fully-missing (site, year) count = {len(full)} / {len(per)}")
        if len(full):
            by_year = full.groupby("year").size().rename("n_sites_fully_missing")
            by_site = full.groupby("site_id").size().rename("n_years_fully_missing").sort_values(ascending=False).head(10)
            print(f"  by_year (top):\n{by_year.to_string()}")
            print(f"  by_site (top 10):\n{by_site.to_string()}")

    # ---- LONG ----
    print("\n" + "=" * 90)
    print("LONG (sheath_blight phenology) — overall + per-split miss rates")
    print("=" * 90)
    long_df = pd.read_csv(LONG_PATH, encoding="utf-8-sig")
    long_df.columns = [c.strip() for c in long_df.columns]
    long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce").astype("Int64")
    long_df = long_df.dropna(subset=["year"]).copy()
    long_df["year"] = long_df["year"].astype(int)
    long_df = long_df[long_df["year"] >= YEAR_MIN].copy()
    long_df["site_id"] = long_df["site_id"].astype(str)
    long_df["split"] = long_df["year"].apply(split_of)
    long_df = long_df[long_df["split"] != "out"].copy()

    print(f"[long] rows={len(long_df):,}  n_sites={long_df['site_id'].nunique()}  n_(site,year)={long_df.groupby(['site_id','year']).ngroups}")
    print(f"[long] split sizes:")
    print(long_df["split"].value_counts().to_string())

    rows = []
    for col in LONG_CANDIDATES:
        if col not in long_df.columns:
            rows.append({"feature": col, "split": "MISSING_COL", "n": 0, "miss_rate": float("nan")})
            continue
        overall = float(long_df[col].isna().mean())
        rows.append({"feature": col, "split": "overall", "n": len(long_df), "miss_rate": overall})
        for sp in ["train", "val", "test"]:
            sub = long_df[long_df["split"] == sp]
            rows.append({
                "feature": col, "split": sp, "n": len(sub),
                "miss_rate": float(sub[col].isna().mean()) if len(sub) else float("nan"),
            })
    long_table = pd.DataFrame(rows)
    pivot_long = long_table.pivot_table(index="feature", columns="split", values="miss_rate", aggfunc="first")
    col_order_l = [c for c in ["overall", "train", "val", "test"] if c in pivot_long.columns]
    pivot_long = pivot_long[col_order_l]
    print(pivot_long.to_string(float_format=lambda x: f"{x:.4f}" if pd.notna(x) else "NaN"))

    # per (site, year) for long
    print("\n" + "-" * 90)
    print("LONG — per (site, year) fully-missing groups")
    print("-" * 90)
    for col in LONG_CANDIDATES:
        if col not in long_df.columns:
            continue
        per = summarize_per_site_year(long_df, col, ["site_id", "year"])
        full = fully_missing_groups(per, 1.0)
        print(f"\n[{col}] fully-missing (site, year) count = {len(full)} / {len(per)}")
        if len(full):
            by_year = full.groupby("year").size().rename("n_sites_fully_missing")
            print(f"  by_year:\n{by_year.to_string()}")

    # ---- VERDICTS ----
    print("\n" + "=" * 90)
    print("VERDICTS")
    print("=" * 90)
    verdict_rows = []
    for col in daily_cols:
        if col not in daily.columns:
            verdict_rows.append((col, "n/a", "n/a", "n/a", "COLUMN MISSING IN DAILY"))
            continue
        overall = float(daily[col].isna().mean())
        train = float(daily.loc[daily["split"] == "train", col].isna().mean())
        test = float(daily.loc[daily["split"] == "test", col].isna().mean())
        verdict_rows.append((col, f"{overall:.4f}", f"{train:.4f}", f"{test:.4f}", verdict(test, train, overall)))
    for col in LONG_CANDIDATES:
        if col not in long_df.columns:
            verdict_rows.append((col, "n/a", "n/a", "n/a", "COLUMN MISSING IN LONG"))
            continue
        overall = float(long_df[col].isna().mean())
        train = float(long_df.loc[long_df["split"] == "train", col].isna().mean())
        test = float(long_df.loc[long_df["split"] == "test", col].isna().mean())
        verdict_rows.append((col, f"{overall:.4f}", f"{train:.4f}", f"{test:.4f}", verdict(test, train, overall)))

    print(f"\n{'feature':<22} {'overall':>10} {'train':>10} {'test':>10}  verdict")
    print("-" * 90)
    for row in verdict_rows:
        print(f"{row[0]:<22} {row[1]:>10} {row[2]:>10} {row[3]:>10}  {row[4]}")


if __name__ == "__main__":
    main()
