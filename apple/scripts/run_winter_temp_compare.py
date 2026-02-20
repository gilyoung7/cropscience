from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from apple.configs import config as C
from apple.src.data_pipeline import load_daily, load_obs, aggregate_obs_daily_max
from apple.src.labels import build_interval_labels_from_doy, filter_labels_by_gap

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


def _add_winter_year(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = df.copy()
    out["month"] = out[date_col].dt.month
    out = out[out["month"].isin([12, 1, 2])].copy()
    out["winter_year"] = np.where(out["month"] == 12, out["year"] + 1, out["year"]).astype(int)
    return out


def _longest_streak(mask: np.ndarray) -> int:
    max_len = 0
    cur = 0
    for v in mask:
        if bool(v):
            cur += 1
            if cur > max_len:
                max_len = cur
        else:
            cur = 0
    return int(max_len)


def build_winter_metrics(
    cold_threshold: float,
    streak_threshold: float,
    censor_types: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily = load_daily(C.PATH_DAILY)
    obs = load_obs(C.PATH_OBS)
    obs2 = aggregate_obs_daily_max(obs)

    labels = build_interval_labels_from_doy(
        obs2,
        threshold=C.THRESHOLD,
        season_start_doy=C.SEASON_START_DOY,
        season_end_doy=C.SEASON_END_DOY,
    )
    labels = filter_labels_by_gap(labels, C.DOY_START, C.DOY_END, C.MAX_GAP)
    labels = labels[labels["censor_type"].isin(censor_types)].copy()
    labels = labels[["site_id", "year", "censor_type"]].copy()

    winter = _add_winter_year(daily, date_col="date")
    winter["year"] = winter["winter_year"].astype(int)
    winter = winter.drop(columns=["winter_year"], errors="ignore")
    winter = winter.merge(labels, on=["site_id", "year"], how="inner")
    winter = winter.sort_values(["site_id", "year", "date"]).copy()

    tmin_col = "최저기온(°C)"
    if tmin_col not in winter.columns:
        raise ValueError(f"Column not found: {tmin_col}")

    winter[tmin_col] = pd.to_numeric(winter[tmin_col], errors="coerce")
    winter = winter.dropna(subset=[tmin_col]).copy()

    grp = winter.groupby(["site_id", "year", "censor_type"], sort=False)
    rows = []
    for (site_id, year, ctype), sub in grp:
        tmin = sub[tmin_col].to_numpy(dtype=float)
        below_cold = tmin <= float(cold_threshold)
        below_streak = tmin <= float(streak_threshold)

        # Heating index: accumulate deficit below threshold.
        # Example: threshold 0C and tmin -3C adds +3.
        heating_index = np.maximum(0.0, float(cold_threshold) - tmin).sum()

        rows.append(
            {
                "site_id": site_id,
                "year": int(year),
                "censor_type": ctype,
                "winter_days": int(len(tmin)),
                "winter_tmin_mean": float(np.mean(tmin)),
                "winter_tmin_min": float(np.min(tmin)),
                "winter_tmin_p10": float(np.percentile(tmin, 10)),
                "cold_days_count": int(below_cold.sum()),
                "heating_index": float(heating_index),
                "streak_threshold_days": int((below_streak).sum()),
                "max_consecutive_streak_days": _longest_streak(below_streak),
            }
        )

    metric_df = pd.DataFrame(rows)
    return winter, metric_df


def plot_histograms(
    winter_daily: pd.DataFrame,
    metric_df: pd.DataFrame,
    out_png: Path,
) -> None:
    if plt is None:
        raise RuntimeError("matplotlib is not installed")

    tmin_col = "최저기온(°C)"
    ctypes = ["right", "interval"]
    colors = {"right": "#1f77b4", "interval": "#d62728"}

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    ax = axes.ravel()

    for c in ctypes:
        d = winter_daily.loc[winter_daily["censor_type"] == c, tmin_col]
        if len(d) > 0:
            ax[0].hist(d, bins=40, alpha=0.45, label=f"{c} (n={len(d)})", color=colors[c])
    ax[0].set_title("Winter daily Tmin distribution")
    ax[0].set_xlabel("Tmin (C)")
    ax[0].set_ylabel("Count")
    ax[0].legend()

    for c in ctypes:
        d = metric_df.loc[metric_df["censor_type"] == c, "winter_tmin_mean"]
        if len(d) > 0:
            ax[1].hist(d, bins=25, alpha=0.45, label=f"{c} (n={len(d)})", color=colors[c])
    ax[1].set_title("Winter mean Tmin (site-year)")
    ax[1].set_xlabel("Mean Tmin (C)")
    ax[1].set_ylabel("Count")
    ax[1].legend()

    for c in ctypes:
        d = metric_df.loc[metric_df["censor_type"] == c, "heating_index"]
        if len(d) > 0:
            ax[2].hist(d, bins=25, alpha=0.45, label=f"{c} (n={len(d)})", color=colors[c])
    ax[2].set_title("Heating index (site-year)")
    ax[2].set_xlabel("Accumulated cold deficit")
    ax[2].set_ylabel("Count")
    ax[2].legend()

    for c in ctypes:
        d = metric_df.loc[metric_df["censor_type"] == c, "max_consecutive_streak_days"]
        if len(d) > 0:
            bins = np.arange(0, max(2, int(d.max()) + 2)) - 0.5
            ax[3].hist(d, bins=bins, alpha=0.45, label=f"{c} (n={len(d)})", color=colors[c])
    ax[3].set_title("Max consecutive cold days (site-year)")
    ax[3].set_xlabel("Days")
    ax[3].set_ylabel("Count")
    ax[3].legend()

    fig.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _print_ascii_hist_compare(
    values_a: np.ndarray,
    values_b: np.ndarray,
    label_a: str,
    label_b: str,
    title: str,
    bins: int = 20,
    bar_width: int = 22,
) -> None:
    if values_a.size == 0 and values_b.size == 0:
        print(f"\n[{title}] no data")
        return

    all_vals = np.concatenate([values_a, values_b]) if values_a.size and values_b.size else (
        values_a if values_a.size else values_b
    )
    vmin = float(np.min(all_vals))
    vmax = float(np.max(all_vals))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0

    edges = np.linspace(vmin, vmax, bins + 1)
    ca, _ = np.histogram(values_a, bins=edges)
    cb, _ = np.histogram(values_b, bins=edges)
    max_count = max(int(ca.max()) if ca.size else 0, int(cb.max()) if cb.size else 0, 1)

    print(f"\n[{title}]")
    print(f"{label_a}=#  {label_b}=*")
    for i in range(bins):
        lo = edges[i]
        hi = edges[i + 1]
        ba = int(round((ca[i] / max_count) * bar_width))
        bb = int(round((cb[i] / max_count) * bar_width))
        print(
            f"{lo:7.2f}..{hi:7.2f} | "
            f"{label_a:8s} {'#' * ba:<{bar_width}} ({int(ca[i]):4d}) | "
            f"{label_b:8s} {'*' * bb:<{bar_width}} ({int(cb[i]):4d})"
        )


def print_terminal_histograms(winter_daily: pd.DataFrame, metric_df: pd.DataFrame) -> None:
    right_daily = winter_daily.loc[winter_daily["censor_type"] == "right", "최저기온(°C)"].to_numpy(dtype=float)
    int_daily = winter_daily.loc[winter_daily["censor_type"] == "interval", "최저기온(°C)"].to_numpy(dtype=float)
    _print_ascii_hist_compare(
        right_daily,
        int_daily,
        label_a="right",
        label_b="interval",
        title="Winter daily Tmin distribution",
    )

    right_mean = metric_df.loc[metric_df["censor_type"] == "right", "winter_tmin_mean"].to_numpy(dtype=float)
    int_mean = metric_df.loc[metric_df["censor_type"] == "interval", "winter_tmin_mean"].to_numpy(dtype=float)
    _print_ascii_hist_compare(
        right_mean,
        int_mean,
        label_a="right",
        label_b="interval",
        title="Winter mean Tmin (site-year)",
    )

    right_hi = metric_df.loc[metric_df["censor_type"] == "right", "heating_index"].to_numpy(dtype=float)
    int_hi = metric_df.loc[metric_df["censor_type"] == "interval", "heating_index"].to_numpy(dtype=float)
    _print_ascii_hist_compare(
        right_hi,
        int_hi,
        label_a="right",
        label_b="interval",
        title="Heating index (site-year)",
    )

    right_streak = metric_df.loc[metric_df["censor_type"] == "right", "max_consecutive_streak_days"].to_numpy(dtype=float)
    int_streak = metric_df.loc[metric_df["censor_type"] == "interval", "max_consecutive_streak_days"].to_numpy(dtype=float)
    _print_ascii_hist_compare(
        right_streak,
        int_streak,
        label_a="right",
        label_b="interval",
        title="Max consecutive cold days (site-year)",
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--cold_threshold", type=float, default=0.0, help="Heating index threshold in C")
    p.add_argument("--streak_threshold", type=float, default=-2.0, help="Cold streak threshold in C")
    p.add_argument("--out_dir", type=str, default="apple/outputs/winter_compare")
    p.add_argument("--terminal_hist", action="store_true", help="Print ASCII histograms to terminal")
    p.add_argument(
        "--censor_types",
        type=str,
        default="right,interval",
        help="Comma-separated types to include. ex) right,interval",
    )
    args = p.parse_args()

    censor_types = [x.strip() for x in args.censor_types.split(",") if x.strip()]
    winter_daily, metric_df = build_winter_metrics(
        cold_threshold=args.cold_threshold,
        streak_threshold=args.streak_threshold,
        censor_types=censor_types,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    daily_csv = out_dir / "winter_daily_labeled.csv"
    metric_csv = out_dir / "winter_site_year_metrics.csv"
    out_png = out_dir / "winter_hist_compare.png"

    winter_daily.to_csv(daily_csv, index=False, encoding="utf-8-sig")
    metric_df.to_csv(metric_csv, index=False, encoding="utf-8-sig")
    if plt is None:
        print("matplotlib is not installed; skipping histogram export.")
    else:
        plot_histograms(winter_daily, metric_df, out_png)

    print(f"saved: {daily_csv}")
    print(f"saved: {metric_csv}")
    if plt is not None:
        print(f"saved: {out_png}")
    if len(metric_df) == 0:
        print("No rows after filtering. Check thresholds/data period.")
        return

    summary = (
        metric_df.groupby("censor_type")[[
            "winter_days",
            "winter_tmin_mean",
            "winter_tmin_min",
            "cold_days_count",
            "heating_index",
            "max_consecutive_streak_days",
        ]]
        .agg(["count", "mean", "median", "std"])
    )
    print("\n[summary by censor_type]")
    print(summary)
    if args.terminal_hist:
        print_terminal_histograms(winter_daily, metric_df)


if __name__ == "__main__":
    main()
