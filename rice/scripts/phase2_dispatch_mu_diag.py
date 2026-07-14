"""Phase 2 mu diagnostic on a dispatch sample_grid CSV.

Reads the dispatch sample_grid produced by phase_r_oracle_iou (with
--dispatch_summary_json) and computes, per offset:

  - corr(alert_tstar, mu)      pearson + spearman
  - corr(true_L_DOY, mu)       pearson + spearman   (true_L_DOY = L column)
  - mu_DOY  mean / std

The same per-offset block is repeated for subcohorts:
  - with_history = 1 vs 0
  - dispatch_branch = D vs A
  - score_over_tau_margin: high (>= overall median) vs low (< overall median)

No retraining; pure read of the existing CSV. matched=False rows (mu is NaN
because alert_tstar + offset fell outside the Stage-2 nowcast frame) are
dropped before correlation.
"""

from __future__ import annotations

import argparse
import sys

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def _corr(x: np.ndarray, y: np.ndarray) -> dict:
    """Pearson + Spearman with NaN-safe filtering."""
    m = np.isfinite(x) & np.isfinite(y)
    if int(m.sum()) < 3:
        return {"n": int(m.sum()), "pearson": float("nan"),
                "spearman": float("nan")}
    xv = x[m].astype(float)
    yv = y[m].astype(float)
    if np.std(xv) == 0 or np.std(yv) == 0:
        return {"n": int(m.sum()), "pearson": float("nan"),
                "spearman": float("nan")}
    return {
        "n": int(m.sum()),
        "pearson": float(np.corrcoef(xv, yv)[0, 1]),
        "spearman": float(spearmanr(xv, yv).correlation),
    }


def _fmt(v: float, nd: int = 4) -> str:
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "nan"
    return f"{v:.{nd}f}"


def _per_offset_table(sub: pd.DataFrame, offsets: list[int],
                       n_total_sample_ids: int) -> list[dict]:
    """Per-offset metrics: matched counts, coverage, IoU(matched|overall),
    mu stats, corr(alert,mu), corr(L,mu).
    coverage = n_matched / n_total_sample_ids (sy in the cohort), not per-offset.
    """
    rows = []
    for o in offsets:
        g = sub[sub["offset"] == o]
        g = g[g["matched"] == True]  # noqa: E712 (CSV booleans land as bool)
        n_row = len(g)
        if n_row == 0:
            rows.append({"offset": int(o), "n_matched": 0,
                         "coverage": 0.0,
                         "IoU_matched": float("nan"),
                         "IoU_overall": 0.0,
                         "mu_mean": float("nan"), "mu_std": float("nan"),
                         "corr_alert_mu_pearson": float("nan"),
                         "corr_alert_mu_spearman": float("nan"),
                         "corr_L_mu_pearson": float("nan"),
                         "corr_L_mu_spearman": float("nan")})
            continue
        mu = g["mu"].astype(float).values
        a = g["alert_tstar"].astype(float).values
        L = g["L"].astype(float).values
        iou_col = g["iou_matched"].astype(float).values
        ca = _corr(a, mu)
        cL = _corr(L, mu)
        iou_matched = float(np.mean(iou_col))
        coverage = float(n_row) / max(int(n_total_sample_ids), 1)
        iou_overall = iou_matched * coverage
        rows.append({
            "offset": int(o),
            "n_matched": int(n_row),
            "coverage": coverage,
            "IoU_matched": iou_matched,
            "IoU_overall": iou_overall,
            "mu_mean": float(np.mean(mu)),
            "mu_std": float(np.std(mu, ddof=0)),
            "corr_alert_mu_pearson": ca["pearson"],
            "corr_alert_mu_spearman": ca["spearman"],
            "corr_L_mu_pearson": cL["pearson"],
            "corr_L_mu_spearman": cL["spearman"],
        })
    return rows


def _print_table(title: str, rows: list[dict], extra: dict | None = None) -> None:
    print(f"\n## {title}")
    if extra:
        print("  " + "  ".join(f"{k}={v}" for k, v in extra.items()))
    cols = ("offset", "n_matched", "coverage",
            "IoU_matched", "IoU_overall",
            "mu_mean", "mu_std",
            "corr_alert_mu_pearson", "corr_alert_mu_spearman",
            "corr_L_mu_pearson", "corr_L_mu_spearman")
    int_cols = {"offset", "n_matched"}
    head = "  " + "  ".join(f"{c:>22s}" for c in cols)
    print(head)
    print("  " + "  ".join("-" * 22 for _ in cols))
    for r in rows:
        cells = []
        for c in cols:
            v = r.get(c)
            if c in int_cols:
                cells.append(f"{int(v):>22d}" if v is not None else f"{'-':>22s}")
            else:
                cells.append(f"{_fmt(v):>22s}")
        print("  " + "  ".join(cells))
    # Coverage-aware best summary
    valid = [r for r in rows if r.get("n_matched", 0) > 0]
    if valid:
        best_m = max(valid, key=lambda r: r["IoU_matched"])
        best_o = max(valid, key=lambda r: r["IoU_overall"])
        print(f"  [best by IoU_matched] off={int(best_m['offset'])}  "
              f"n={int(best_m['n_matched'])}  coverage={best_m['coverage']:.4f}  "
              f"IoU_matched={best_m['IoU_matched']:.4f}  "
              f"IoU_overall={best_m['IoU_overall']:.4f}")
        print(f"  [best by IoU_overall] off={int(best_o['offset'])}  "
              f"n={int(best_o['n_matched'])}  coverage={best_o['coverage']:.4f}  "
              f"IoU_matched={best_o['IoU_matched']:.4f}  "
              f"IoU_overall={best_o['IoU_overall']:.4f}")


def _consistency_block(df: pd.DataFrame, offsets: list[int],
                        ref_offset: int | None) -> None:
    """sample_id set equality across offsets + same-sample mu drift."""
    print("\n## (4) sample_id consistency + mu drift across offsets")
    matched_by_off = {
        o: set(df[(df["offset"] == o) & (df["matched"] == True)]["sample_id"])  # noqa: E712
        for o in offsets
    }
    union_ids = set().union(*matched_by_off.values()) if matched_by_off else set()
    common_ids = (set.intersection(*matched_by_off.values())
                  if matched_by_off else set())
    print(f"  matched_per_offset = "
          f"{ {o: len(s) for o, s in matched_by_off.items()} }")
    print(f"  union  (matched at ANY offset) = {len(union_ids)}")
    print(f"  inter  (matched at ALL offsets) = {len(common_ids)}")

    if ref_offset is None or ref_offset not in offsets:
        ref_offset = min(offsets)
    print(f"  ref_offset for mu drift = {ref_offset}")
    g_ref = df[(df["offset"] == ref_offset) & (df["matched"] == True)].set_index("sample_id")  # noqa: E712
    cols = ("offset", "n_common", "mean_diff", "mean_abs_diff",
            "std_diff", "min_diff", "max_diff")
    print("  " + "  ".join(f"{c:>14s}" for c in cols))
    print("  " + "  ".join("-" * 14 for _ in cols))
    for o in offsets:
        if o == ref_offset:
            print(f"  {o:>14d} {'(ref)':>16s}")
            continue
        g_o = df[(df["offset"] == o) & (df["matched"] == True)].set_index("sample_id")  # noqa: E712
        common = list(set(g_ref.index) & set(g_o.index))
        if not common:
            print(f"  {o:>14d} {0:>14d}  (no common)")
            continue
        diff = (g_o.loc[common, "mu"].astype(float).values
                - g_ref.loc[common, "mu"].astype(float).values)
        absd = np.abs(diff)
        print(f"  {o:>14d} {len(common):>14d} "
              f"{diff.mean():>+14.4f} {absd.mean():>14.4f} "
              f"{diff.std(ddof=0):>14.4f} {diff.min():>+14.2f} {diff.max():>+14.2f}")

    pivot = (df[df["matched"] == True]  # noqa: E712
             .pivot_table(index="sample_id", columns="offset",
                          values="mu", aggfunc="first"))
    full = pivot.dropna()
    if not full.empty:
        ranges = (full.max(axis=1) - full.min(axis=1))
        print(f"  per-sample mu_range across {len(offsets)} offsets "
              f"(samples matched at every offset = {len(full)}):")
        print(f"    mean={ranges.mean():.4f}  median={ranges.median():.4f}  "
              f"std={ranges.std(ddof=0):.4f}  "
              f"min={ranges.min():.4f}  max={ranges.max():.4f}")


def _correlation_block(df: pd.DataFrame, ref_offset: int) -> None:
    """Global per-sample correlations relevant to absolute-vs-lead target choice.

    corr(alert_tstar, true_L_DOY) is the freebie correlation floor a lead/
    residual head gets for free without learning anything.
    """
    print(f"\n## (5) global correlations (one row per sample_id, ref offset={ref_offset})")
    sub = df[(df["offset"] == ref_offset) & (df["matched"] == True)]  # noqa: E712
    sub = sub.drop_duplicates(subset=["sample_id"])
    if sub.empty or "alert_tstar" not in sub.columns or "L" not in sub.columns:
        print("  required columns missing or no matched rows; skipping")
        return
    a = sub["alert_tstar"].astype(float).values
    L = sub["L"].astype(float).values
    lead = L - a
    mu = sub["mu"].astype(float).values

    pairs = [
        ("alert_tstar vs true_L_DOY  [free baseline for lead target]", a, L),
        ("alert_tstar vs lead_to_L (=L-alert)                         ", a, lead),
        ("alert_tstar vs mu                                            ", a, mu),
        ("true_L_DOY  vs mu                                            ", L, mu),
        ("lead_to_L   vs mu                                            ", lead, mu),
    ]
    for name, x, y in pairs:
        c = _corr(x, y)
        print(f"  {name}  n={c['n']:>4d}  "
              f"pearson={_fmt(c['pearson'])}  spearman={_fmt(c['spearman'])}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sample_grid_csv", required=True)
    ap.add_argument("--model", default=None,
                    help="If multiple models in the CSV, restrict to this label. "
                         "Defaults to the first unique model in the file.")
    ap.add_argument("--out_csv", default=None,
                    help="Optional path to dump all per-(cohort, offset) rows.")
    ap.add_argument("--ref_offset", type=int, default=None,
                    help="Reference offset for same-sample mu-drift table and "
                         "correlation block. Default: smallest offset present.")
    args = ap.parse_args()

    df = pd.read_csv(args.sample_grid_csv)
    if args.model is None:
        model_label = sorted(df["model"].unique().tolist())[0]
    else:
        model_label = args.model
    df = df[df["model"] == model_label].copy()
    if df.empty:
        print(f"[abort] no rows for model={model_label!r}")
        return 2

    # alert_tstar overwrite check: must equal t_star_doy in dispatch mode
    if "alert_tstar" not in df.columns:
        print("[abort] sample_grid missing 'alert_tstar' column — "
              "is this a dispatch-mode grid?")
        return 2

    offsets = sorted(int(x) for x in df["offset"].unique().tolist())
    n_total = df["sample_id"].nunique()
    n_matched_any = df[df["matched"] == True]["sample_id"].nunique()  # noqa: E712
    print(f"# sample_grid: {args.sample_grid_csv}")
    print(f"# model       : {model_label}")
    print(f"# offsets     : {offsets}")
    print(f"# sample_id total = {n_total}  matched_any = {n_matched_any}")

    out_rows: list[dict] = []

    overall_rows = _per_offset_table(df, offsets, n_total)
    _print_table("OVERALL", overall_rows,
                 extra={"n_sample_id": n_total})
    for r in overall_rows:
        out_rows.append({"cohort": "OVERALL", **r})

    # with_history split
    for wh_val, label in [(1, "with_history=1"), (0, "with_history=0")]:
        sub = df[df["with_history"] == wh_val]
        n_sy = sub["sample_id"].nunique()
        rows = _per_offset_table(sub, offsets, n_sy)
        _print_table(label, rows, extra={"n_sample_id": n_sy})
        for r in rows:
            out_rows.append({"cohort": label, **r})

    # dispatch_branch split
    for br_val in ("D", "A"):
        sub = df[df["dispatch_branch"] == br_val]
        n_sy = sub["sample_id"].nunique()
        rows = _per_offset_table(sub, offsets, n_sy)
        _print_table(f"dispatch_branch={br_val}", rows,
                     extra={"n_sample_id": n_sy})
        for r in rows:
            out_rows.append({"cohort": f"dispatch_branch={br_val}", **r})

    # score_over_tau_margin median split (compute median over unique samples,
    # then apply mask).
    one_per_sy = df.drop_duplicates(subset=["sample_id"], keep="first")
    margin_med = float(one_per_sy["score_over_tau_margin"].median())
    print(f"\n[margin-split] median(score_over_tau_margin) = {margin_med:.6f}")
    hi_ids = set(one_per_sy[one_per_sy["score_over_tau_margin"] >= margin_med]["sample_id"])
    lo_ids = set(one_per_sy[one_per_sy["score_over_tau_margin"] <  margin_med]["sample_id"])
    for ids, label in [(hi_ids, "margin_high"), (lo_ids, "margin_low")]:
        sub = df[df["sample_id"].isin(ids)]
        n_sy = sub["sample_id"].nunique()
        rows = _per_offset_table(sub, offsets, n_sy)
        _print_table(f"{label}  (median split @ {margin_med:.4f})",
                     rows, extra={"n_sample_id": n_sy})
        for r in rows:
            out_rows.append({"cohort": label, **r})

    ref_off = args.ref_offset if args.ref_offset is not None else min(offsets)
    _consistency_block(df, offsets, ref_off)
    _correlation_block(df, ref_off)

    # ---- (6) Oracle best-offset diagnostic (per-sample best offset by IoU)
    # Upper-bound only: shows how many samples would prefer each offset under
    # an oracle policy. NOT to be used to build a real offset policy.
    print("\n## (6) ORACLE best-offset diagnostic  (per-sample best offset by IoU; "
          "test-set upper bound only — do NOT use to build a policy)")
    sub_m = df[df["matched"] == True]  # noqa: E712
    if sub_m.empty:
        print("  no matched rows")
    else:
        best_by_sy = (sub_m.sort_values("iou_matched", ascending=False)
                          .drop_duplicates(subset=["sample_id"], keep="first"))
        hist = best_by_sy["offset"].value_counts().sort_index().to_dict()
        n_sy = int(best_by_sy["sample_id"].nunique())
        mean_iou = float(best_by_sy["iou_matched"].mean())
        print(f"  n_sy_with_any_match = {n_sy}  oracle mean IoU = {mean_iou:.4f}")
        print(f"  best-offset histogram:")
        for o in offsets:
            c = int(hist.get(o, 0))
            frac = c / max(n_sy, 1)
            sub_o = best_by_sy[best_by_sy["offset"] == o]
            if len(sub_o) == 0:
                print(f"    off={o:>4}: n=0  frac=0.0000  (no samples prefer this offset)")
                continue
            # subgroup lead and alert distribution among samples that pick this offset
            lead_sub = (sub_o["L"].astype(float) - sub_o["alert_tstar"].astype(float)).values
            atstar = sub_o["alert_tstar"].astype(float).values
            print(f"    off={o:>4}: n={c:>4}  frac={frac:.4f}  "
                  f"lead_to_L mean={lead_sub.mean():.2f} median={np.median(lead_sub):.1f}  "
                  f"alert_DOY mean={atstar.mean():.2f}")

    if args.out_csv:
        pd.DataFrame(out_rows).to_csv(args.out_csv, index=False)
        print(f"\n# wrote per-(cohort, offset) table -> {args.out_csv}",
              file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
