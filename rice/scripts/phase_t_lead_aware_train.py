"""
Phase T7-train — Stage 1 XGB retrain with lead-aware training labels.

Protocol matches the eventually-label w28 baseline (stride=1, event_time_proxy=mid,
add_tstar_position_feature, year split, XGB) — only training labels differ.

For event site-years (base_censor_type != 'right') with lead = L - tstar:
  - lead_min <= lead <= lead_max:                   y_train = 1
  - outside_policy='ignore':    other event rows -> dropped
  - outside_policy='negative':  other event rows -> y_train = 0
  - outside_policy='semi_negative':
       lead > --lead_neg_above           -> y_train = 0
       other rows outside [min,max]      -> dropped
Non-event rows: y_train = 0, kept.

Val/test samples are NOT filtered — y_event keeps eventually-label semantics so
downstream evaluators (alert Pareto, occurrence AUC, TP/FP DOY) work unchanged.

XGB hyper-params copied from --template_ckpt for fair comparison with baseline.
Saved ckpt is structurally compatible with run_event_train output.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import torch
from xgboost import XGBClassifier

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.src.dataset import split_samples
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import (
    build_nowcast_samples,
    build_tabular_from_samples,
    make_event_labels,
)


def filter_train_by_lead(train_now: list[dict], lead_min: int, lead_max: int,
                         outside_policy: str,
                         lead_neg_above: int | None = None) -> tuple[list[dict], dict]:
    out = []
    stats = {"n_pos_kept": 0, "n_nonevent_kept": 0,
             "n_event_dropped_ignore": 0, "n_event_relabel_neg": 0,
             "lead_min": int(lead_min), "lead_max": int(lead_max),
             "outside_policy": str(outside_policy),
             "lead_neg_above": (None if lead_neg_above is None else int(lead_neg_above))}
    if outside_policy == "semi_negative" and lead_neg_above is None:
        raise ValueError("outside_policy=semi_negative requires --lead_neg_above")
    for s in train_now:
        ctype = str(s.get("base_censor_type", "right"))
        if ctype == "right":
            s2 = dict(s); s2["y_event"] = 0
            out.append(s2)
            stats["n_nonevent_kept"] += 1
            continue
        L = s.get("L")
        if L is None:
            continue
        tstar = int(s["tstar"])
        lead = int(L) - tstar
        if lead_min <= lead <= lead_max:
            s2 = dict(s); s2["y_event"] = 1
            out.append(s2)
            stats["n_pos_kept"] += 1
        else:
            if outside_policy == "ignore":
                stats["n_event_dropped_ignore"] += 1
                continue
            elif outside_policy == "negative":
                s2 = dict(s); s2["y_event"] = 0
                out.append(s2)
                stats["n_event_relabel_neg"] += 1
            elif outside_policy == "semi_negative":
                if lead > int(lead_neg_above):
                    s2 = dict(s); s2["y_event"] = 0
                    out.append(s2)
                    stats["n_event_relabel_neg"] += 1
                else:
                    stats["n_event_dropped_ignore"] += 1
                    continue
            else:
                raise ValueError(f"unknown outside_policy: {outside_policy}")
    return out, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--template_ckpt", required=True,
                    help="XGB ckpt to copy hyper-params + nowcast settings from")
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--nowcast_window", type=int, default=28)
    ap.add_argument("--lead_min", type=int, required=True)
    ap.add_argument("--lead_max", type=int, required=True)
    ap.add_argument("--outside_policy", choices=["ignore", "negative", "semi_negative"], required=True)
    ap.add_argument("--lead_neg_above", type=int, default=None,
                    help="semi_negative mode: lead > this value is relabeled as negative")
    ap.add_argument("--add_site_history", action="store_true",
                    help="append 11 site-history channels to base X")
    ap.add_argument("--site_history_policy", default="rolling",
                    choices=["rolling", "strict_train"])
    ap.add_argument("--history_train_year_max", type=int, default=2021,
                    help="strict_train policy upper bound for history")
    ap.add_argument("--add_phenology", action="store_true",
                    help="append 11 fuzzy-phenology channels to base X")
    ap.add_argument("--add_derived_weather", action="store_true",
                    help="append 7 derived-weather channels (VPD, 28d rain/GDD, streaks)")
    ap.add_argument("--xgb_seed", type=int, default=None,
                    help="override XGB random_state for seed-stability runs")
    ap.add_argument("--add_neighbor_history", action="store_true",
                    help="append neighbor (other-site) occurrence channels to base X")
    ap.add_argument("--neighbor_decay_km", type=float, default=None,
                    help="decay length (km) for neighbor_weighted_* (default: util default 20)")
    ap.add_argument("--dry_run", action="store_true",
                    help="build samples + tabular shapes, print feature dim, then skip train/save")
    ap.add_argument("--out_ckpt", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    ckpt = torch.load(args.template_ckpt, map_location="cpu", weights_only=False)
    C.DOY_START = int(ckpt.get("doy_start", 60))
    C.DOY_END = int(ckpt.get("doy_end", 300))
    add_tpos = bool(ckpt.get("add_tstar_position_feature", True))
    nc_stride = int(ckpt.get("nowcast_stride", 1))
    nc_only_pre = bool(int(ckpt.get("nowcast_only_pre_event", 1)))
    nc_proxy = str(ckpt.get("nowcast_event_time_proxy", "mid"))
    nc_label_mode = str(ckpt.get("nowcast_label_mode", "eventually"))
    nc_label_horizon = ckpt.get("nowcast_label_horizon", None)
    nc_tstart = ckpt.get("nowcast_tstar_start", None)
    feature_cols = ckpt["feature_cols"]
    feature_names = ckpt["feature_names"]
    # Running list of per-channel names; extended in append order (history ->
    # pheno -> derived_weather -> neighbor) so the saved feature_names + d_in
    # exactly match samples[0]["X"].shape[1].
    full_feature_names = list(feature_names)

    template_clf = ckpt["trained_states"][0]["sk_model"]
    hyper = {k: v for k, v in template_clf.get_params().items()
             if v is not None and k in {
                 "n_estimators", "max_depth", "learning_rate", "subsample",
                 "colsample_bytree", "reg_lambda", "min_child_weight", "gamma",
                 "random_state", "eval_metric", "tree_method", "device", "objective",
             }}
    if args.xgb_seed is not None:
        hyper["random_state"] = int(args.xgb_seed)
        print(f"[xgb_seed override] random_state={hyper['random_state']}")
    print(f"[cfg] DOY_START={C.DOY_START} DOY_END={C.DOY_END} window={args.nowcast_window} "
          f"add_tpos={add_tpos}  proxy={nc_proxy}")
    print(f"[hyper] {hyper}")
    print(f"[lead] window=[{args.lead_min},{args.lead_max}]  policy={args.outside_policy}"
          f"  neg_above={args.lead_neg_above}")

    _, get_feature_cols = resolve_pest(args.pest)
    _, _, _T, samples = build_samples_for_run(args.run, get_feature_cols)

    history_meta = None
    if args.add_site_history:
        from rice.scripts.site_history_utils import (
            compute_site_history, append_history_to_samples,
            HISTORY_FEATURE_NAMES, HISTORY_FEATURE_DIM,
        )
        history = compute_site_history(
            samples, doy_start=int(C.DOY_START),
            policy=args.site_history_policy,
            train_year_max=int(args.history_train_year_max),
        )
        n_done = append_history_to_samples(samples, history, doy_start=int(C.DOY_START))
        full_feature_names.extend(list(HISTORY_FEATURE_NAMES))
        new_dim = int(samples[0]["X"].shape[1])
        print(f"[history] added {HISTORY_FEATURE_DIM} channels  policy={args.site_history_policy}  "
              f"applied_to={n_done} samples  X dim now {new_dim}")
        history_meta = {
            "site_history_added": True,
            "site_history_policy": args.site_history_policy,
            "history_train_year_max": int(args.history_train_year_max),
            "history_feature_names": HISTORY_FEATURE_NAMES,
            "history_feature_dim": HISTORY_FEATURE_DIM,
        }

    pheno_meta = None
    if args.add_phenology:
        from rice.scripts.phenology_utils import (
            load_pheno_map, append_pheno_to_samples,
            PHENO_FEATURE_NAMES, PHENO_FEATURE_DIM,
        )
        pheno_map = load_pheno_map()
        n_pheno = append_pheno_to_samples(samples, pheno_map, doy_start=int(C.DOY_START))
        full_feature_names.extend(list(PHENO_FEATURE_NAMES))
        new_dim = int(samples[0]["X"].shape[1])
        print(f"[phenology] added {PHENO_FEATURE_DIM} channels  applied_to={n_pheno} samples  "
              f"X dim now {new_dim}")
        pheno_meta = {
            "phenology_added": True,
            "phenology_feature_names": PHENO_FEATURE_NAMES,
            "phenology_feature_dim": PHENO_FEATURE_DIM,
        }

    weather_meta = None
    if args.add_derived_weather:
        from rice.scripts.derived_weather_utils import (
            append_derived_weather_to_samples,
            DERIVED_WEATHER_NAMES, DERIVED_WEATHER_DIM,
        )
        n_w = append_derived_weather_to_samples(samples)
        full_feature_names.extend(list(DERIVED_WEATHER_NAMES))
        new_dim = int(samples[0]["X"].shape[1])
        print(f"[derived_weather] added {DERIVED_WEATHER_DIM} channels  applied_to={n_w} samples  "
              f"X dim now {new_dim}")
        weather_meta = {
            "derived_weather_added": True,
            "derived_weather_feature_names": DERIVED_WEATHER_NAMES,
            "derived_weather_feature_dim": DERIVED_WEATHER_DIM,
        }

    neighbor_meta = None
    if args.add_neighbor_history:
        from rice.scripts.neighbor_history_utils import (
            load_long_events, build_neighbor_index, append_neighbor_to_samples,
            NEIGHBOR_CHANNEL_NAMES, NEIGHBOR_FEATURE_DIM, DEFAULT_DECAY_KM,
        )
        decay_km = (float(args.neighbor_decay_km)
                    if args.neighbor_decay_km is not None else DEFAULT_DECAY_KM)
        # Neighbor index is built from the pest LONG CSV directly (site_id, year,
        # obs_doy, label_event, lat/lon). Features use only OTHER sites' events in
        # the SAME year with obs_doy < t, so there is no future / train-test leak.
        ev_df, co_df, _site_years = load_long_events(
            C.PATH_OBS, label_col=getattr(C, "LABEL_COL", "label_event"),
            year_min=getattr(C, "YEAR_MIN", None), year_max=getattr(C, "YEAR_MAX", None),
        )
        nb_index = build_neighbor_index(ev_df, co_df)
        n_nb = append_neighbor_to_samples(
            samples, nb_index, doy_start=int(C.DOY_START), decay_km=decay_km)
        full_feature_names.extend(list(NEIGHBOR_CHANNEL_NAMES))
        new_dim = int(samples[0]["X"].shape[1])
        print(f"[neighbor] added {NEIGHBOR_FEATURE_DIM} channels  decay_km={decay_km}  "
              f"sites={len(nb_index.site_ids)}  event_rows={len(ev_df)}  "
              f"applied_to={n_nb} samples  X dim now {new_dim}")
        neighbor_meta = {
            "neighbor_history_added": True,
            "neighbor_decay_km": float(decay_km),
            "neighbor_feature_names": list(NEIGHBOR_CHANNEL_NAMES),
            "neighbor_feature_dim": int(NEIGHBOR_FEATURE_DIM),
        }

    # Variant tag from active augmentation flags (A / D / N / DN).
    _variant = ("D" if args.add_site_history else "") + ("N" if args.add_neighbor_history else "")
    _variant = _variant or "A"
    print(f"[variant] {_variant}  per_day_X_channels={int(samples[0]['X'].shape[1])}  "
          f"feature_names={len(full_feature_names)}")

    train_seas, val_seas, test_seas = split_samples(
        samples, val_frac=0.1, test_frac=0.1, seed=args.split_seed,
        split_mode="year", val_year=args.val_year,
        test_year_min=args.test_year_min, test_year_max=args.test_year_max,
    )

    nc_kw = dict(window=int(args.nowcast_window), stride=int(nc_stride), tstar_start=nc_tstart,
                 only_pre_event=bool(nc_only_pre), event_time_proxy=nc_proxy,
                 label_mode=nc_label_mode, label_horizon=nc_label_horizon)
    train_now = build_nowcast_samples(train_seas, **nc_kw)
    val_now = build_nowcast_samples(val_seas, **nc_kw)
    test_now = build_nowcast_samples(test_seas, **nc_kw)
    print(f"[nowcast] train={len(train_now)}  val={len(val_now)}  test={len(test_now)}")

    train_lead, stats = filter_train_by_lead(train_now, args.lead_min, args.lead_max,
                                              args.outside_policy, args.lead_neg_above)
    print(f"[lead-filter] {stats}")
    print(f"[train after filter] n={len(train_lead)}")
    if not train_lead:
        raise SystemExit("Empty training set after lead filter")

    X_tr = build_tabular_from_samples(train_lead, add_tstar_position_feature=add_tpos)
    y_tr = make_event_labels(train_lead)
    pos_w = (y_tr == 0).sum() / max((y_tr == 1).sum(), 1)
    print(f"[train] X={X_tr.shape}  pos_rate={y_tr.mean():.3f}  pos_weight={pos_w:.3f}")
    if args.dry_run:
        print(f"[dry_run] variant={_variant}  per_day_X_channels={int(samples[0]['X'].shape[1])}  "
              f"feature_names={len(full_feature_names)}  tabular_X_tr={X_tr.shape}  "
              f"(window={int(args.nowcast_window)}, add_tpos={add_tpos})")
        print("[dry_run] no training / no checkpoint written")
        return
    hyper_local = dict(hyper)
    hyper_local["scale_pos_weight"] = float(pos_w)
    clf = XGBClassifier(**hyper_local)
    t0 = time.perf_counter()
    clf.fit(X_tr, y_tr)
    print(f"[xgb] trained in {time.perf_counter()-t0:.1f}s")

    out_ckpt_path = Path(args.out_ckpt)
    out_ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {
        "run": int(args.run), "pest": str(args.pest),
        "d_in": int(samples[0]["X"].shape[1]),
        "feature_cols": feature_cols, "feature_names": full_feature_names,
        "year_max": ckpt.get("year_max"),
        "model_type": "event_tabular", "event_model": "xgb",
        "doy_start": int(C.DOY_START), "doy_end": int(C.DOY_END),
        "T": int(args.nowcast_window), "task_mode": "nowcast",
        "nowcast_window": int(args.nowcast_window), "nowcast_stride": int(nc_stride),
        "nowcast_tstar_start": None if nc_tstart is None else int(nc_tstart),
        "nowcast_only_pre_event": int(nc_only_pre),
        "nowcast_event_time_proxy": nc_proxy,
        "nowcast_label_mode": nc_label_mode,
        "nowcast_label_horizon": None if nc_label_horizon is None else int(nc_label_horizon),
        "add_tstar_position_feature": bool(add_tpos),
        "split_seed": int(args.split_seed), "split_mode": "year",
        "trained_states": [{"seed": 0, "best_epoch": None, "best_val_bce": None, "sk_model": clf}],
        "lead_aware_label": {
            "lead_min": int(args.lead_min), "lead_max": int(args.lead_max),
            "outside_policy": str(args.outside_policy),
            "lead_neg_above": (None if args.lead_neg_above is None else int(args.lead_neg_above)),
        },
        **(history_meta if history_meta else {"site_history_added": False}),
        **(pheno_meta if pheno_meta else {"phenology_added": False}),
        **(weather_meta if weather_meta else {"derived_weather_added": False}),
        **(neighbor_meta if neighbor_meta else {"neighbor_history_added": False}),
        "variant": _variant,
        "lead_filter_stats": stats,
    }
    torch.save(bundle, out_ckpt_path)
    print(f"[saved] {out_ckpt_path}")


if __name__ == "__main__":
    main()
