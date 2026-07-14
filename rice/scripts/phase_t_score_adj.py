"""
Phase T4 — DOY-normalized score adjustment (no training, no test leakage).

score_adj(s, t) = p_cal(s, t) - mean_p_cal_by_DOY[DOY(t)]

mean_p_cal_by_DOY is computed on:
  - train OOF probs (preferred, if probs cache provides them)
  - else val (warning: in-sample for val Pareto selection)

Pareto sweep over (tau_adj, k_consecutive) using the same k-consecutive alert rule
as the original Pareto. Comparison vs original (absolute-score) Pareto at
matched val_recall thresholds.

Outputs:
  - mean_by_DOY table (smoothed and raw)
  - original Pareto sweep CSV (sanity-check, same as phase_t_stage1a_pareto)
  - adjusted Pareto sweep CSV
  - matched-recall comparison printed and saved to JSON
  - verdict: at the target recall levels, does FAR decrease on val? on test?

No retraining. Pure score post-processing.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import build_nowcast_samples, build_tabular_from_samples
from rice.scripts.run_calibrate_event import apply_temperature, fit_temperature_grid
from rice.scripts.run_stage1b_cascade_v2 import derive_alerts, metrics_from_alerts


def load_probs(args) -> dict:
    if args.probs_cache and Path(args.probs_cache).exists():
        with open(args.probs_cache, "rb") as f:
            cache = pickle.load(f)
        print(f"[cache] loaded {args.probs_cache}  has_train={'train_df' in cache}")
        return cache

    print("[cache miss] rebuilding val/test probs only")
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    add_tpos = bool(ckpt.get("add_tstar_position_feature", False))
    nc_window = int(ckpt.get("nowcast_window", 28))
    nc_stride = int(ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("nowcast_event_time_proxy", "r"))
    nc_label_mode = str(ckpt.get("nowcast_label_mode", "eventually"))
    nc_label_horizon = ckpt.get("nowcast_label_horizon", None)
    nc_tstart = ckpt.get("nowcast_tstar_start", None)

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples = build_samples_for_run(args.run, get_feature_cols)
    _, val_seas, test_seas = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )
    nc_kw = dict(window=nc_window, stride=nc_stride, tstar_start=nc_tstart,
                 only_pre_event=nc_only_pre, event_time_proxy=nc_proxy,
                 label_mode=nc_label_mode, label_horizon=nc_label_horizon)
    val_now = build_nowcast_samples(val_seas, **nc_kw)
    test_now = build_nowcast_samples(test_seas, **nc_kw)
    X_val = build_tabular_from_samples(val_now, add_tstar_position_feature=add_tpos)
    X_test = build_tabular_from_samples(test_now, add_tstar_position_feature=add_tpos)
    y_val = np.asarray([int(s["y_event"]) for s in val_now])

    clf = ckpt["trained_states"][0]["sk_model"]
    p_val_raw = clf.predict_proba(X_val)[:, 1]
    p_test_raw = clf.predict_proba(X_test)[:, 1]
    t_best, _ = fit_temperature_grid(y_val, p_val_raw)
    p_val_cal = apply_temperature(p_val_raw, t_best)
    p_test_cal = apply_temperature(p_test_raw, t_best)

    def _df(seas, nc, p, split_name):
        sy_meta = {(str(s["site_id"]), int(s["year"])): s for s in seas}
        rows = []
        for s, prob in zip(nc, p):
            key = (str(s["site_id"]), int(s["year"]))
            meta = sy_meta[key]
            ctype = str(meta["censor_type"])
            rows.append({
                "split": split_name, "site": key[0], "year": key[1],
                "tstar": int(s["tstar"]), "p_cal": float(prob),
                "y_event": int(s["y_event"]),
                "true_L": int(meta["L"]) if ctype != "right" else None,
                "true_R": int(meta["R"]) if ctype != "right" else None,
            })
        return pd.DataFrame(rows)

    return {
        "t_best": float(t_best),
        "val_df": _df(val_seas, val_now, p_val_cal, "val"),
        "test_df": _df(test_seas, test_now, p_test_cal, "test"),
    }


def compute_mean_by_doy(source_df: pd.DataFrame, doy_start: int, smooth_window: int) -> tuple[dict, pd.DataFrame]:
    df = source_df.copy()
    df["doy"] = df["tstar"].astype(int) + int(doy_start)
    means = df.groupby("doy")["p_cal"].agg(["mean", "count", "std"]).reset_index()
    means.columns = ["doy", "mean_p", "n", "std_p"]
    raw = means.set_index("doy")["mean_p"]
    if smooth_window > 1:
        smoothed = raw.rolling(window=smooth_window, center=True, min_periods=1).mean()
    else:
        smoothed = raw
    means["mean_p_smoothed"] = smoothed.values
    return smoothed.to_dict(), means


def apply_score_adj(probs_df: pd.DataFrame, mean_by_doy: dict, doy_start: int) -> pd.DataFrame:
    df = probs_df.copy()
    df["doy"] = df["tstar"].astype(int) + int(doy_start)
    global_mean = float(np.mean(list(mean_by_doy.values())))
    df["mean_doy"] = df["doy"].map(mean_by_doy).fillna(global_mean)
    df["p_cal_orig"] = df["p_cal"]
    df["p_cal"] = df["p_cal_orig"] - df["mean_doy"]
    return df


def sweep(probs_df: pd.DataFrame, tau_grid: np.ndarray, ks: list[int]) -> pd.DataFrame:
    rows = []
    for k in ks:
        for tau in tau_grid:
            a = derive_alerts(probs_df, float(tau), k_consecutive=k)
            m = metrics_from_alerts(a, "")
            rows.append({
                "k": int(k), "tau": float(tau),
                "recall": m["recall"], "FAR": m["FAR"],
                "precision": m["precision"], "F1": m["F1"],
                "n_alert": m["n_alert"], "tp": m["tp"], "fp": m["fp"], "fn": m["fn"],
            })
    return pd.DataFrame(rows)


def best_at_recall(val_sweep: pd.DataFrame, target_recall: float) -> dict | None:
    cands = val_sweep[val_sweep["recall"] >= target_recall]
    if cands.empty:
        return None
    best = cands.loc[cands["FAR"].idxmin()]
    return {"k": int(best["k"]), "tau": float(best["tau"]),
            "val_recall": float(best["recall"]), "val_FAR": float(best["FAR"]),
            "val_precision": float(best["precision"]), "val_F1": float(best["F1"]),
            "val_n_alert": int(best["n_alert"])}


def lookup_test(test_sweep: pd.DataFrame, k: int, tau: float) -> dict:
    """Find test row at the (k, tau) selected on val."""
    matches = test_sweep[(test_sweep["k"] == k) & (np.isclose(test_sweep["tau"], tau, atol=1e-6))]
    if matches.empty:
        # nearest tau on this k
        sub = test_sweep[test_sweep["k"] == k].copy()
        sub["d"] = (sub["tau"] - tau).abs()
        row = sub.loc[sub["d"].idxmin()]
    else:
        row = matches.iloc[0]
    return {"test_recall": float(row["recall"]), "test_FAR": float(row["FAR"]),
            "test_precision": float(row["precision"]), "test_F1": float(row["F1"]),
            "test_n_alert": int(row["n_alert"])}


def matched_recall_compare(orig_val: pd.DataFrame, orig_test: pd.DataFrame,
                           adj_val: pd.DataFrame, adj_test: pd.DataFrame,
                           targets: list[float]) -> list[dict]:
    out = []
    print(f"\n  {'target':>7} | {'kind':>5} {'k':>2} {'tau':>7} | "
          f"{'v_R':>6} {'v_F':>6} {'v_P':>6} {'v_F1':>6} | "
          f"{'t_R':>6} {'t_F':>6} {'t_P':>6} {'t_F1':>6}")
    for target in targets:
        for kind, vsweep, tsweep in [("orig", orig_val, orig_test), ("adj", adj_val, adj_test)]:
            picked = best_at_recall(vsweep, target)
            if picked is None:
                print(f"  {target:>7.2f} | {kind:>5} {'-':>2} {'-':>7} | (no point meets recall on val)")
                out.append({"target": target, "kind": kind, "picked": None})
                continue
            t = lookup_test(tsweep, picked["k"], picked["tau"])
            print(f"  {target:>7.2f} | {kind:>5} {picked['k']:>2d} {picked['tau']:>7.3f} | "
                  f"{picked['val_recall']:>6.3f} {picked['val_FAR']:>6.3f} {picked['val_precision']:>6.3f} {picked['val_F1']:>6.3f} | "
                  f"{t['test_recall']:>6.3f} {t['test_FAR']:>6.3f} {t['test_precision']:>6.3f} {t['test_F1']:>6.3f}")
            out.append({"target": float(target), "kind": kind, "picked": picked, "test": t})
    return out


def verdict(compare: list[dict], targets: list[float]) -> str:
    lines = []
    for target in targets:
        cells = [c for c in compare if c["target"] == target]
        orig = next((c for c in cells if c["kind"] == "orig" and c["picked"] is not None), None)
        adj = next((c for c in cells if c["kind"] == "adj" and c["picked"] is not None), None)
        if orig is None and adj is None:
            lines.append(f"  recall>={target:.2f}: neither operating point exists on val.")
            continue
        if adj is None:
            lines.append(f"  recall>={target:.2f}: orig val_FAR={orig['picked']['val_FAR']:.3f}; adj has no qualifying point.")
            continue
        if orig is None:
            lines.append(f"  recall>={target:.2f}: orig has no qualifying point; adj val_FAR={adj['picked']['val_FAR']:.3f} (likely too high tau too).")
            continue
        dv_FAR = adj["picked"]["val_FAR"] - orig["picked"]["val_FAR"]
        dt_FAR = adj["test"]["test_FAR"] - orig["test"]["test_FAR"]
        dt_R = adj["test"]["test_recall"] - orig["test"]["test_recall"]
        msg = (f"  recall>={target:.2f}: dFAR_val={dv_FAR:+.3f}  dFAR_test={dt_FAR:+.3f}  "
               f"dRecall_test={dt_R:+.3f}")
        if dv_FAR < -0.02 and dt_FAR < -0.02 and dt_R > -0.02:
            msg += "   -> adj helps (val + test FAR drop, recall maintained)"
        elif dv_FAR < -0.02 and dt_FAR >= -0.02:
            msg += "   -> adj helps val only; test does not generalize"
        elif dv_FAR >= -0.02:
            msg += "   -> no meaningful val improvement"
        else:
            msg += "   -> mixed signal"
        lines.append(msg)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--stage1_ckpt", required=True)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--probs_cache", default="")
    ap.add_argument("--mean_source", default="auto", choices=["auto", "train", "val"],
                    help="auto: prefer train if cache has it, else val")
    ap.add_argument("--smooth_window", type=int, default=7,
                    help="smoothing window (days) for mean_by_DOY")
    ap.add_argument("--tau_orig_min", type=float, default=0.05)
    ap.add_argument("--tau_orig_max", type=float, default=0.95)
    ap.add_argument("--tau_orig_step", type=float, default=0.025)
    ap.add_argument("--tau_adj_min", type=float, default=-0.30)
    ap.add_argument("--tau_adj_max", type=float, default=0.70)
    ap.add_argument("--tau_adj_step", type=float, default=0.025)
    ap.add_argument("--ks", default="1,2,3")
    ap.add_argument("--targets", default="0.85,0.88,0.90,0.92,0.95",
                    help="recall targets for matched comparison")
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(args.stage1_ckpt, map_location="cpu", weights_only=False)
    C.DOY_START = int(ckpt.get("doy_start", 60))
    C.DOY_END = int(ckpt.get("doy_end", 300))
    print(f"[cfg] DOY_START={C.DOY_START} DOY_END={C.DOY_END}")

    cache = load_probs(args)
    val_df = cache["val_df"]
    test_df = cache["test_df"]
    train_df = cache.get("train_df")

    # Mean source selection
    if args.mean_source == "train" or (args.mean_source == "auto" and train_df is not None):
        if train_df is None:
            raise SystemExit("--mean_source=train requested but probs cache has no train_df. "
                             "Run phase_t_stage1b_cascade.py first to generate it.")
        source_label = "train_oof"
        source_df = train_df
    else:
        source_label = "val"
        source_df = val_df
        if train_df is not None:
            print("[note] cache has train_df, but --mean_source=val explicitly chosen")
        else:
            print("[warn] no train_df in cache; using val for mean_by_DOY (in-sample for val Pareto)")

    print(f"\n========== Phase T4 score_adj ==========")
    print(f"[mean source] {source_label}  (n_rows={len(source_df)})")
    print(f"[smooth_window] {args.smooth_window} days")

    mean_map, mean_tbl = compute_mean_by_doy(source_df, C.DOY_START, args.smooth_window)
    mean_tbl.to_csv(out_dir / f"mean_by_doy_{source_label}.csv", index=False)
    print(f"[mean_by_DOY] DOYs covered: {int(mean_tbl['doy'].min())} - {int(mean_tbl['doy'].max())}")
    print(f"  global mean_p_cal={mean_tbl['mean_p'].mean():.3f}  "
          f"range[min={mean_tbl['mean_p'].min():.3f}, max={mean_tbl['mean_p'].max():.3f}]")
    print(f"[mean_by_DOY] preview (every 20 DOYs):")
    print(f"  {'DOY':>4} {'n':>4} {'mean_p':>7} {'smooth':>7}")
    for _, r in mean_tbl[::20].iterrows():
        print(f"  {int(r['doy']):>4d} {int(r['n']):>4d} {r['mean_p']:>7.3f} {r['mean_p_smoothed']:>7.3f}")

    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    tau_orig = np.arange(args.tau_orig_min, args.tau_orig_max + 1e-9, args.tau_orig_step)
    tau_adj = np.arange(args.tau_adj_min, args.tau_adj_max + 1e-9, args.tau_adj_step)
    print(f"\n[sweep] ks={ks}  tau_orig: {len(tau_orig)} pts  tau_adj: {len(tau_adj)} pts")

    # Original sweep (sanity-check, same as phase_t_stage1a_pareto)
    orig_val_sweep = sweep(val_df, tau_orig, ks)
    orig_test_sweep = sweep(test_df, tau_orig, ks)

    # Adjusted sweep
    val_adj = apply_score_adj(val_df, mean_map, C.DOY_START)
    test_adj = apply_score_adj(test_df, mean_map, C.DOY_START)
    adj_val_sweep = sweep(val_adj, tau_adj, ks)
    adj_test_sweep = sweep(test_adj, tau_adj, ks)

    orig_val_sweep.to_csv(out_dir / "orig_val_sweep.csv", index=False)
    orig_test_sweep.to_csv(out_dir / "orig_test_sweep.csv", index=False)
    adj_val_sweep.to_csv(out_dir / "adj_val_sweep.csv", index=False)
    adj_test_sweep.to_csv(out_dir / "adj_test_sweep.csv", index=False)

    targets = [float(x) for x in args.targets.split(",") if x.strip()]

    print(f"\n----- Matched val-recall comparison (orig vs adj) -----")
    print(f"  val: select FAR-min at recall>=target. test: lookup at same (k, tau).")
    compare = matched_recall_compare(orig_val_sweep, orig_test_sweep,
                                     adj_val_sweep, adj_test_sweep, targets)

    # Also print, for reference, the adjusted Pareto top-k by val_F1 per k
    print(f"\n----- Adjusted Pareto: top 5 by val_F1 per k -----")
    print(f"  {'k':>2} {'tau_adj':>8} | {'v_R':>6} {'v_F':>6} {'v_P':>6} {'v_F1':>6} | "
          f"{'t_R':>6} {'t_F':>6} {'t_P':>6} {'t_F1':>6}")
    for k in ks:
        sub_v = adj_val_sweep[adj_val_sweep["k"] == k].nlargest(5, "F1")
        sub_t = adj_test_sweep[adj_test_sweep["k"] == k]
        for _, r in sub_v.iterrows():
            t = lookup_test(sub_t, k, float(r["tau"]))
            print(f"  {k:>2d} {r['tau']:>8.3f} | "
                  f"{r['recall']:>6.3f} {r['FAR']:>6.3f} {r['precision']:>6.3f} {r['F1']:>6.3f} | "
                  f"{t['test_recall']:>6.3f} {t['test_FAR']:>6.3f} {t['test_precision']:>6.3f} {t['test_F1']:>6.3f}")

    print(f"\n----- Verdict (per recall target) -----")
    v = verdict(compare, targets)
    print(v)

    summary = {
        "mean_source": source_label,
        "smooth_window": args.smooth_window,
        "ks": ks,
        "targets": targets,
        "compare": compare,
        "verdict_text": v,
    }
    (out_dir / "score_adj_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[saved] sweep csvs + summary in {out_dir}")


if __name__ == "__main__":
    main()
