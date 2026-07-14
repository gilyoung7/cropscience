"""
Phase A — Phenology variance / encoding audit for sheath_blight.

For each candidate phenology column (best_suitability, best_months, offset_days, window_idx):
  - dtype + inferred encoding (continuous / ordinal / categorical)
  - unique value count + frequency distribution
  - within-site across-year variance (does it change for the same site over years?)
  - within-year across-site variance (does it differ between sites in the same year?)
  - (site, year) variance decomposition: site_effect / year_effect / residual shares
  - 0-fill semantics check for offset_days (is "0" a natural meaning, or a fillna sentinel?)
  - encoding recommendation (raw / ordinal-as-is / one-hot)
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

LONG_PATH = Path("/home/gpu4080/ygdata/rice/LONG_by_pest/RICE_LONG_잎집무늬마름병.csv")
YEAR_MIN = 2002

PHENO_COLS = ["best_suitability", "best_months", "offset_days", "window_idx"]


def load_long_unique_siteyear() -> pd.DataFrame:
    df = pd.read_csv(LONG_PATH, encoding="utf-8-sig")
    df.columns = [c.strip() for c in df.columns]
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df = df.dropna(subset=["year"]).copy()
    df["year"] = df["year"].astype(int)
    df = df[df["year"] >= YEAR_MIN].copy()
    df["site_id"] = df["site_id"].astype(str)

    # Phenology columns are static per (site, year) — collapse.
    keep = ["site_id", "year"] + [c for c in PHENO_COLS if c in df.columns]
    sy = df[keep].drop_duplicates(subset=["site_id", "year"]).copy()
    # Numeric coercion (best_months stored as string-list might fail — we'll inspect raw)
    return sy


def infer_encoding(s: pd.Series) -> str:
    """Best-effort dtype guess: 'continuous' / 'ordinal' / 'categorical' / 'string'."""
    raw = s.dropna()
    if raw.empty:
        return "empty"
    # Try numeric
    numeric = pd.to_numeric(raw, errors="coerce")
    n_numeric = numeric.notna().sum()
    if n_numeric / len(raw) < 0.9:
        # Mostly non-numeric → string/categorical
        nunique = raw.nunique()
        if nunique <= 30:
            return f"string-categorical (k={nunique})"
        return f"string ({nunique} unique)"
    # Numeric: integer-only? small unique?
    raw_num = numeric.dropna()
    is_int = (raw_num % 1 == 0).all()
    nunique = raw_num.nunique()
    if is_int and nunique <= 15:
        return f"ordinal-or-categorical (int, k={nunique})"
    if is_int:
        return f"discrete integer (k={nunique})"
    return f"continuous (k={nunique})"


def variance_decomp_numeric(sy: pd.DataFrame, col: str) -> dict:
    s = sy.copy()
    s[col] = pd.to_numeric(s[col], errors="coerce")
    s = s.dropna(subset=[col])
    if s.empty:
        return {"col": col, "n": 0}
    grand = float(s[col].mean())
    site_means = s.groupby("site_id")[col].mean()
    year_means = s.groupby("year")[col].mean()
    sm = s.merge(site_means.rename("site_mean"), left_on="site_id", right_index=True)
    sm = sm.merge(year_means.rename("year_mean"), left_on="year", right_index=True)

    total_var = float(sm[col].var(ddof=0))
    site_var = float(((sm["site_mean"] - grand) ** 2).mean())
    year_var = float(((sm["year_mean"] - grand) ** 2).mean())
    resid = sm[col] - sm["site_mean"] - sm["year_mean"] + grand
    resid_var = float((resid ** 2).mean())

    within_site_std = float(s.groupby("site_id")[col].std(ddof=0).mean())
    within_year_std = float(s.groupby("year")[col].std(ddof=0).mean())

    return {
        "col": col,
        "n": len(s),
        "grand_mean": grand,
        "overall_std": float(s[col].std(ddof=0)),
        "total_var": total_var,
        "site_share": site_var / total_var if total_var > 0 else float("nan"),
        "year_share": year_var / total_var if total_var > 0 else float("nan"),
        "resid_share": resid_var / total_var if total_var > 0 else float("nan"),
        "within_site_yearly_std": within_site_std,
        "within_year_site_std": within_year_std,
    }


def encoding_recommendation(col: str, enc: str, decomp: dict, raw_series: pd.Series) -> str:
    if "string" in enc:
        nunique = raw_series.dropna().nunique()
        if nunique <= 12:
            return f"one-hot or label encode (string, k={nunique})"
        return f"high-cardinality string ({nunique}) — needs custom handling"
    # numeric / ordinal cases
    nunique = pd.to_numeric(raw_series, errors="coerce").dropna().nunique()
    if col == "best_months":
        # Calendar month is naturally cyclic.
        return f"if month-index (1..12): use sin/cos OR one-hot (k={nunique}); raw ordinal misrepresents cycle"
    if col == "window_idx":
        if nunique <= 8:
            return f"one-hot (categorical, k={nunique})"
        return f"ordinal raw OK if windows are ordered (k={nunique})"
    if col == "offset_days":
        return "raw numeric (true ordinal — days)"
    if col == "best_suitability":
        return "raw numeric (continuous)"
    return "raw numeric"


def offset_zero_semantics(sy: pd.DataFrame) -> None:
    """Check whether offset_days==0 occurs naturally in data (vs being only a fillna sentinel)."""
    col = "offset_days"
    if col not in sy.columns:
        print(f"\n[offset_zero] column missing")
        return
    s_all = pd.to_numeric(sy[col], errors="coerce")
    n_nan = int(s_all.isna().sum())
    n_zero = int((s_all == 0).sum())
    n_nonzero = int((s_all.notna() & (s_all != 0)).sum())
    n_total = int(len(s_all))
    print(f"\n[offset_zero semantics] total={n_total}  NaN={n_nan}  zero={n_zero}  nonzero={n_nonzero}")
    print(f"  → zero is {'natural (present alongside many nonzeros)' if n_zero > 0 and n_nonzero > 0 else 'rare or only sentinel'}")
    if n_zero > 0 and n_nonzero > 0:
        print(f"  → if fillna(0.0) is applied to {n_nan} NaNs, they will be indistinguishable from true zeros.")
        print(f"  → recommend mask channel OR fillna(-1) / fillna(median) instead of 0.0")


def main() -> None:
    print("=" * 90)
    print("Phase A — Phenology variance & encoding audit")
    print("=" * 90)

    sy = load_long_unique_siteyear()
    print(f"\nUnique (site, year) rows = {len(sy):,}")
    print(f"n_sites = {sy['site_id'].nunique()}   n_years = {sy['year'].nunique()}")

    print("\n" + "-" * 90)
    print("Per-column: dtype + encoding inference + value distribution")
    print("-" * 90)
    encs: dict[str, str] = {}
    for col in PHENO_COLS:
        if col not in sy.columns:
            print(f"\n[{col}] COLUMN MISSING")
            continue
        raw = sy[col]
        enc = infer_encoding(raw)
        encs[col] = enc
        s = raw.dropna()
        print(f"\n[{col}]  dtype={raw.dtype}  enc={enc}")
        nunique = s.nunique()
        if nunique <= 20:
            vc = s.value_counts(dropna=False).sort_index()
            print(vc.to_string())
        else:
            num = pd.to_numeric(s, errors="coerce").dropna()
            if not num.empty:
                print(f"  num.min={num.min()}  num.max={num.max()}  num.mean={num.mean():.4f}  num.std={num.std(ddof=0):.4f}")
                print(f"  q10={num.quantile(0.1):.4f}  q50={num.quantile(0.5):.4f}  q90={num.quantile(0.9):.4f}")
            print(f"  (showing top 10 most common values)")
            print(s.value_counts().head(10).to_string())

    offset_zero_semantics(sy)

    print("\n" + "-" * 90)
    print("Variance decomposition (share of total variance)")
    print("-" * 90)
    decomps: dict[str, dict] = {}
    for col in PHENO_COLS:
        if col not in sy.columns:
            continue
        d = variance_decomp_numeric(sy, col)
        decomps[col] = d
        if d.get("n", 0) == 0:
            print(f"\n[{col}]  (non-numeric or empty — skip decomp)")
            continue
        print(
            f"\n[{col}]  n={d['n']}  overall_std={d['overall_std']:.4f}  total_var={d['total_var']:.4f}"
            f"\n  site_share={d['site_share']:.3f}   year_share={d['year_share']:.3f}   resid_share={d['resid_share']:.3f}"
            f"\n  within_site_yearly_std={d['within_site_yearly_std']:.4f}  (avg σ across years for one site)"
            f"\n  within_year_site_std={d['within_year_site_std']:.4f}  (avg σ across sites for one year)"
        )

    print("\n" + "=" * 90)
    print("SUMMARY + RECOMMENDATIONS")
    print("=" * 90)
    print(f"\n{'feature':<22} {'encoding':<40} {'site/year/resid':<24} verdict")
    print("-" * 110)
    for col in PHENO_COLS:
        if col not in sy.columns:
            print(f"{col:<22} MISSING")
            continue
        enc = encs.get(col, "?")
        d = decomps.get(col, {})
        if d.get("n", 0) == 0:
            shares = "n/a (non-numeric)"
            verdict = "see encoding rec below"
        else:
            shares = f"{d['site_share']:.2f}/{d['year_share']:.2f}/{d['resid_share']:.2f}"
            # Verdict on signal
            if d["overall_std"] < 1e-6:
                verdict = "EXCLUDE (constant)"
            elif d["site_share"] > 0.5 and d["year_share"] < 0.2:
                verdict = "ADD (site-dominant — breaks calendar prior)"
            elif d["year_share"] > 0.5 and d["site_share"] < 0.2:
                verdict = "WEAK (year-only — overlaps calendar prior)"
            elif d["site_share"] > 0.3 and d["year_share"] > 0.3:
                verdict = "ADD (mixed signal)"
            elif d["site_share"] > 0.3:
                verdict = "ADD (site-leaning)"
            elif d["year_share"] > 0.3:
                verdict = "WEAK (year-leaning)"
            else:
                verdict = "AMBIGUOUS (mostly residual)"
        print(f"{col:<22} {enc:<40} {shares:<24} {verdict}")

    print("\nEncoding recommendations:")
    for col in PHENO_COLS:
        if col not in sy.columns:
            continue
        rec = encoding_recommendation(col, encs.get(col, ""), decomps.get(col, {}), sy[col])
        print(f"  {col:<22} → {rec}")


if __name__ == "__main__":
    main()
