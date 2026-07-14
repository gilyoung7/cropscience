"""
Phase M — R - L distribution (interval censoring window) per split.

Reports mean, std, min/max, percentiles (10/25/50/75/90/95/99),
and a histogram (configurable bin edges) for interval-censored samples
in train / val / test (year-split 2022 / 2023-24).

Also reports L / R / mid = (L+R)/2 marginal stats on the test set so we
can plan mu = mid target training.
"""

from __future__ import annotations

import argparse
import numpy as np
import pandas as pd

from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples
from rice.scripts.run_eval import build_samples_for_run


def summarize(arr: np.ndarray, label: str) -> dict:
    if arr.size == 0:
        return {"split": label, "n": 0}
    return {
        "split": label, "n": int(arr.size),
        "mean": float(arr.mean()), "std": float(arr.std(ddof=0)),
        "min": int(arr.min()), "max": int(arr.max()),
        "p10": float(np.quantile(arr, 0.10)),
        "p25": float(np.quantile(arr, 0.25)),
        "p50": float(np.quantile(arr, 0.50)),
        "p75": float(np.quantile(arr, 0.75)),
        "p90": float(np.quantile(arr, 0.90)),
        "p95": float(np.quantile(arr, 0.95)),
        "p99": float(np.quantile(arr, 0.99)),
    }


def histogram_rows(arr: np.ndarray, bins: list[float], label: str) -> list[dict]:
    if arr.size == 0:
        return []
    counts, _ = np.histogram(arr, bins=bins)
    n = arr.size
    rows = []
    for i in range(len(bins) - 1):
        rows.append({
            "split": label,
            "bin": f"[{int(bins[i])},{int(bins[i+1])})",
            "n": int(counts[i]),
            "pct": 100.0 * counts[i] / n,
        })
    return rows


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, default="sheath_blight")
    p.add_argument("--run", type=int, default=4)
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    p.add_argument("--bins", type=str, default="0,3,6,9,12,15,18,21,30,45,60,90,200",
                   help="comma-separated bin edges (days)")
    args = p.parse_args()

    _ = resolve_pest(args.pest)
    _, get_feature_cols = resolve_pest(args.pest)
    _, _, T, samples = build_samples_for_run(args.run, get_feature_cols)
    train, val, test = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )

    def _rl(split_list):
        return np.asarray(
            [int(s["R"]) - int(s["L"]) for s in split_list
             if str(s["censor_type"]) == "interval"],
            dtype=int,
        )

    rl_train = _rl(train)
    rl_val = _rl(val)
    rl_test = _rl(test)

    print("=================== R − L summary (interval samples) ===================")
    stats_df = pd.DataFrame([
        summarize(rl_train, "train"),
        summarize(rl_val, "val"),
        summarize(rl_test, "test"),
    ])
    pd.set_option("display.float_format", lambda v: f"{v:.2f}")
    pd.set_option("display.width", 200)
    print(stats_df.to_string(index=False))

    bins = [float(x) for x in str(args.bins).split(",") if x.strip()]
    print("\n=================== Histogram of R−L (%) ===================")
    rows = []
    rows.extend(histogram_rows(rl_train, bins, "train"))
    rows.extend(histogram_rows(rl_val, bins, "val"))
    rows.extend(histogram_rows(rl_test, bins, "test"))
    hist_df = pd.DataFrame(rows)
    pivot = hist_df.pivot(index="bin", columns="split", values="pct").reindex(
        [f"[{int(bins[i])},{int(bins[i+1])})" for i in range(len(bins)-1)]
    )
    # also raw counts
    counts_pivot = hist_df.pivot(index="bin", columns="split", values="n").reindex(
        [f"[{int(bins[i])},{int(bins[i+1])})" for i in range(len(bins)-1)]
    )
    print("---- % ----")
    print(pivot.round(2).to_string())
    print("\n---- counts ----")
    print(counts_pivot.fillna(0).astype(int).to_string())

    # Test set L / R / mid marginal stats (frame index)
    L_test = np.asarray([int(s["L"]) for s in test
                         if str(s["censor_type"]) == "interval"], dtype=int)
    R_test = np.asarray([int(s["R"]) for s in test
                         if str(s["censor_type"]) == "interval"], dtype=int)
    mid_test = (L_test + R_test) / 2.0
    print("\n=================== test interval L / R / mid (frame index, 1..300) ===================")
    print(f"  L_test  : n={len(L_test)}  mean={L_test.mean():.2f}  std={L_test.std(ddof=0):.2f}  "
          f"min={L_test.min()} max={L_test.max()}")
    print(f"  R_test  : n={len(R_test)}  mean={R_test.mean():.2f}  std={R_test.std(ddof=0):.2f}  "
          f"min={R_test.min()} max={R_test.max()}")
    print(f"  mid_test: n={len(mid_test)}  mean={mid_test.mean():.2f}  std={mid_test.std(ddof=0):.2f}  "
          f"min={mid_test.min():.1f} max={mid_test.max():.1f}")
    print(f"  mid - L : mean={(mid_test - L_test).mean():.2f}  std={(mid_test - L_test).std(ddof=0):.2f}  "
          f"(== (R-L)/2)")


if __name__ == "__main__":
    main()
