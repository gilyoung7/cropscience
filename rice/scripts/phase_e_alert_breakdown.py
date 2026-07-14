"""
Phase E — Alert pipeline breakdown: Stage 1 alert vs Stage 2 matched.

For a Stage 1 ckpt (year-split), compute on the test split:
  - n_total_groups        : unique (site,year) in test
  - n_interval_groups     : event-bearing groups
  - n_right_groups        : right-censored groups
  - n_alert_stage1        : groups with ANY t* score >= tau (tau by F1 on val)
  - n_alert_interval      : how many alerts are from interval groups
  - n_alert_right         : how many alerts are from right-censored groups (false alerts)
  - For interval-alerted groups, distribute the alert_tstar+offset matching outcome:
      n_match_ok          : stage2 nowcast row exists at alert_tstar+offset (matched)
      n_after_R           : alert_tstar+offset >= R(event_time) → row pruned by only_pre_event
      n_after_Tend        : alert_tstar+offset > T_end (season end)
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import build_nowcast_samples, build_tabular_from_samples
from rice.scripts.run_calibrate_event import apply_temperature, fit_temperature_grid


def best_tau_by_f1(y: np.ndarray, p: np.ndarray) -> float:
    taus = np.linspace(0.05, 0.95, 19)
    best_tau, best_f1 = 0.5, -1.0
    for t in taus:
        pred = (p >= t).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        if (2*tp + fp + fn) == 0:
            continue
        f1 = (2*tp) / (2*tp + fp + fn)
        if f1 > best_f1:
            best_f1, best_tau = f1, float(t)
    return best_tau


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, default="sheath_blight")
    p.add_argument("--run", type=int, required=True)
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--label", type=str, default="")
    p.add_argument("--split_seed", type=int, default=42)
    p.add_argument("--val_year", type=int, default=2022)
    p.add_argument("--test_year_min", type=int, default=2023)
    p.add_argument("--test_year_max", type=int, default=2024)
    p.add_argument("--stage2_tstar_offset", type=int, default=60)
    args = p.parse_args()

    _ = resolve_pest(args.pest)
    ckpt = torch.load(args.stage1_ckpt, map_location="cpu")
    trained_states = ckpt["trained_states"]
    C.DOY_START = int(ckpt.get("doy_start", C.DOY_START))
    C.DOY_END = int(ckpt.get("doy_end", C.DOY_END))
    add_tstar_position_feature = bool(ckpt.get("add_tstar_position_feature", False))

    nc_window = int(ckpt.get("nowcast_window", 28))
    nc_stride = int(ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("nowcast_event_time_proxy", "r"))

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, T, samples = build_samples_for_run(args.run, get_feature_cols)
    _, val_seas, test_seas = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )

    def _expand(seas):
        return build_nowcast_samples(
            seas, window=nc_window, stride=nc_stride,
            only_pre_event=nc_only_pre, event_time_proxy=nc_proxy,
        )

    val_s = _expand(val_seas)
    test_s = _expand(test_seas)

    label = args.label or Path(args.stage1_ckpt).parent.parent.name
    print(f"\n===== {label}  (run={args.run}, ckpt={Path(args.stage1_ckpt).name}) =====")
    print(f"[season] test groups = {len(test_seas)}  (interval = "
          f"{sum(1 for s in test_seas if str(s['censor_type'])!='right')}, "
          f"right = {sum(1 for s in test_seas if str(s['censor_type'])=='right')})")
    print(f"[nowcast] val_s={len(val_s)} test_s={len(test_s)}")

    y_val = np.asarray([int(s["y_event"]) for s in val_s])
    X_val = build_tabular_from_samples(val_s, add_tstar_position_feature=add_tstar_position_feature)
    X_test = build_tabular_from_samples(test_s, add_tstar_position_feature=add_tstar_position_feature)

    d0 = trained_states[0]
    seed = int(d0["seed"])
    clf = d0.get("sk_model")
    if clf is None:
        raise SystemExit("ckpt missing sk_model")
    p_val_raw = clf.predict_proba(X_val)[:, 1]
    p_test_raw = clf.predict_proba(X_test)[:, 1]
    t_best, _ = fit_temperature_grid(y_val, p_val_raw)
    p_val_cal = apply_temperature(p_val_raw, t_best)
    p_test_cal = apply_temperature(p_test_raw, t_best)
    tau = best_tau_by_f1(y_val, p_val_cal)
    print(f"[seed {seed}] T*={t_best:.3f}  tau (F1 on val) = {tau:.3f}")

    test_group_meta = {}
    for s in test_seas:
        key = (str(s["site_id"]), int(s["year"]))
        test_group_meta[key] = {
            "censor_type": str(s["censor_type"]),
            "L": int(s["L"]) if str(s["censor_type"]) != "right" else None,
            "R": int(s["R"]) if str(s["censor_type"]) != "right" else None,
        }

    n_total = len(test_group_meta)
    n_interval = sum(1 for v in test_group_meta.values() if v["censor_type"] != "right")
    n_right = n_total - n_interval

    alert_first_tstar = {}
    for s, p_cal in zip(test_s, p_test_cal):
        if p_cal < tau:
            continue
        key = (str(s["site_id"]), int(s["year"]))
        prev = alert_first_tstar.get(key)
        if prev is None or int(s["tstar"]) < prev:
            alert_first_tstar[key] = int(s["tstar"])

    n_alert = len(alert_first_tstar)
    n_alert_interval = sum(1 for k in alert_first_tstar if test_group_meta[k]["censor_type"] != "right")
    n_alert_right = n_alert - n_alert_interval

    n_match_ok = 0
    n_after_R = 0
    n_after_Tend = 0
    Tend = int(C.DOY_END - C.DOY_START + 1)
    offset = int(args.stage2_tstar_offset)
    matched_pos = 0
    for key, alert_t in alert_first_tstar.items():
        meta = test_group_meta[key]
        if meta["censor_type"] == "right":
            n_after_R += 1
            continue
        target_tstar = alert_t + offset
        R = int(meta["R"])
        if target_tstar > Tend:
            n_after_Tend += 1
        elif target_tstar >= R:
            n_after_R += 1
        else:
            n_match_ok += 1
            matched_pos += 1

    print(f"\n[test group counts]")
    print(f"  n_total            = {n_total}")
    print(f"  n_interval         = {n_interval}")
    print(f"  n_right            = {n_right}")
    print(f"\n[Stage 1 alert (score >= tau at any t*)]")
    print(f"  n_alert (any cens) = {n_alert}  ({n_alert/n_total*100:.1f}% of total)")
    print(f"  n_alert_interval   = {n_alert_interval}  ({n_alert_interval/max(n_interval,1)*100:.1f}% of interval)")
    print(f"  n_alert_right      = {n_alert_right}  ({n_alert_right/max(n_right,1)*100:.1f}% of right)  ← false alerts")
    print(f"\n[Stage 2 gate (alert_tstar+{offset})]")
    print(f"  n_match_ok         = {n_match_ok}  (interval alerts matched into stage2 nowcast row)")
    print(f"  offset_missed      = {n_alert - n_match_ok}  (sum below)")
    print(f"    - right alerts auto-miss      = {n_alert_right}")
    print(f"    - target_tstar >= R           = {n_after_R - n_alert_right}")
    print(f"    - target_tstar >  Tend        = {n_after_Tend}")
    if alert_first_tstar:
        ats = np.array(list(alert_first_tstar.values()))
        print(f"\n[alert_tstar dist] min={int(ats.min())} median={int(np.median(ats))} mean={ats.mean():.1f} max={int(ats.max())}")


if __name__ == "__main__":
    main()
