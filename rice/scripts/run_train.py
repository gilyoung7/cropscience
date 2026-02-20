from __future__ import annotations

import copy
import random
import argparse
from pathlib import Path
import json
import time
import pickle
import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest, default_out_root, ensure_output_dirs
from rice.scripts.common import make_loader, parse_seed_candidates
from rice.src.data_pipeline import (
    load_daily_preprocessed,
    load_obs,
    aggregate_obs_daily_max,
    make_obs_meta,
    add_site_static_latlon,
    merge_pheno_daily_ffill,
    make_daily_feature_frame,
)
from rice.src.labels import build_interval_labels_from_doy, filter_labels_by_gap
from rice.src.dataset import (
    build_train_frame,
    slice_season,
    build_samples_season,
    split_by_site,
    compute_norm_stats,
    IntervalEventDataset,
    log_split_fingerprint,
    split_seed_search_topk,
    build_stage2_nowcast_samples,
)
from rice.src.ckpt_schema import build_ckpt_meta
from rice.src.model import HazardTransformer
from rice.src.train_eval import run_epoch_weighted, eval_nll_model, eval_metrics_with_overlap

DEBUG_LOADER_SETTINGS = False
DEBUG_PICKLE_DATASET = True
DEBUG_SAMPLE_CHECK = True

def _split_stats(samples: list[dict]) -> dict:
    sites = {s["site_id"] for s in samples}
    n_sites = len(sites)
    n_samples = len(samples)
    if n_samples == 0:
        return {"n_sites": n_sites, "n_samples": 0, "event_rate": 0.0}
    n_event = sum(1 for s in samples if str(s.get("censor_type", "")) != "right")
    return {"n_sites": n_sites, "n_samples": n_samples, "event_rate": n_event / n_samples}

def _interval_len_stats(samples: list[dict]) -> tuple[int, float, float]:
    lens = [int(s["R"]) - int(s["L"]) for s in samples if str(s.get("censor_type", "")) == "interval"]
    if not lens:
        return 0, 0.0, 0.0
    arr = np.asarray(lens, dtype=float)
    return len(lens), float(arr.mean()), float(arr.var(ddof=0))


def parse_balance_ratio(raw: str | None) -> dict[str, float] | None:
    """
    raw format: 'right:interval:left', e.g. '1:1:1'
    """
    if raw is None:
        return None
    parts = [p.strip() for p in raw.split(":")]
    if len(parts) != 3:
        raise ValueError("--train_balance_ratio must be in 'right:interval:left' format, e.g. 1:1:1")
    vals = [float(p) for p in parts]
    if any(v < 0 for v in vals) or sum(vals) <= 0:
        raise ValueError("--train_balance_ratio values must be non-negative and not all zero")
    s = sum(vals)
    return {
        "right": vals[0] / s,
        "interval": vals[1] / s,
        "left": vals[2] / s,
    }


def build_balanced_sampler(train_samples: list[dict], ratio: dict[str, float], seed: int) -> WeightedRandomSampler:
    counts = {"right": 0, "interval": 0, "left": 0}
    for s in train_samples:
        c = str(s.get("censor_type", ""))
        if c in counts:
            counts[c] += 1

    active = {k: v for k, v in ratio.items() if counts.get(k, 0) > 0 and v > 0}
    if not active:
        raise ValueError(f"no active classes for balanced sampling. counts={counts} ratio={ratio}")
    z = sum(active.values())
    active = {k: v / z for k, v in active.items()}

    class_w = {}
    for k in ("right", "interval", "left"):
        if counts.get(k, 0) > 0 and k in active:
            class_w[k] = active[k] / float(counts[k])
        else:
            class_w[k] = 0.0

    sample_w = [class_w.get(str(s.get("censor_type", "")), 0.0) for s in train_samples]
    gen = torch.Generator().manual_seed(seed)
    weights = torch.as_tensor(sample_w, dtype=torch.double)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(train_samples),
        replacement=True,
        generator=gen,
    )

def resolve_out_path(run: int, out_root: str, out_path: str | None) -> Path:
    if out_path:
        out_path_resolved = Path(out_path)
        out_path_resolved.parent.mkdir(parents=True, exist_ok=True)
        return out_path_resolved
    out_dir = Path(out_root) / "ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"checkpoint_run{run}.pt"


def resolve_split_seeds_json_path(out_root: str, split_seeds_json: str | None) -> Path:
    if split_seeds_json:
        return Path(split_seeds_json)
    return Path(out_root) / "splits" / "selected_split_seeds.json"


def load_split_seed_from_topk(split_seeds_json_path: Path, split_seed_from_topk_idx: int | None) -> tuple[int, int, dict, dict]:
    if not split_seeds_json_path.exists():
        raise FileNotFoundError(f"split seeds json not found: {split_seeds_json_path}")
    with open(split_seeds_json_path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    topk_list = payload.get("topk", [])
    if not topk_list:
        raise ValueError(f"topk is empty in split seeds json: {split_seeds_json_path}")
    if split_seed_from_topk_idx is None:
        split_seed_from_topk_idx = payload.get("selected_topk_idx", 0)
    if split_seed_from_topk_idx < 0 or split_seed_from_topk_idx >= len(topk_list):
        raise ValueError(f"--split_seed_from_topk_idx out of range (0..{len(topk_list)-1})")
    chosen = topk_list[int(split_seed_from_topk_idx)]
    split_seed = int(chosen["seed"])
    return split_seed, int(split_seed_from_topk_idx), chosen, payload


def main(
    pest: str,
    run: int,
    out_root: str,
    out_path: str | None,
    split_seed: int,
    seeds: list[int] | None,
    auto_split_seed: bool,
    seed_candidates_raw: str | None,
    target_test_interval: int | None,
    tol_test_interval: int | None,
    auto_split_topk: int,
    split_seed_from_topk_idx: int | None,
    split_seeds_json: str | None,
    dropout: float | None,
    weight_decay: float | None,
    lr: float | None,
    w_interval: float | None,
    w_left: float | None,
    w_right: float | None,
    lambda_mass: float,
    lambda_right_late: float,
    right_late_tau: float | None,
    train_balance_ratio: str | None,
    stage2_nowcast: bool,
    stage2_nowcast_window: int,
    stage2_nowcast_stride: int,
    stage2_nowcast_tstar_start: int | None,
    stage2_nowcast_only_pre_event: int,
    stage2_nowcast_event_time_proxy: str,
):
    _, get_feature_cols = resolve_pest(pest)
    if not out_root:
        out_root = default_out_root(pest)
    ensure_output_dirs(out_root)

    print(
        "Effective DataLoader config: "
        f"BATCH_TRAIN={C.BATCH_TRAIN}, BATCH_EVAL={C.BATCH_EVAL}, "
        f"NUM_WORKERS={C.NUM_WORKERS}, PIN_MEMORY={C.PIN_MEMORY}, "
        f"PERSISTENT_WORKERS={C.PERSISTENT_WORKERS}, PREFETCH_FACTOR={C.PREFETCH_FACTOR}"
    )
    print(
        f"Effective Best-Epoch Metric: VAL_IoU80(interval-only), "
        f"train_balance_ratio={train_balance_ratio}, PI_METHOD={getattr(C, 'PI_METHOD', 'shortest')}"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    # =========================
    # 1) DAILY: load + GDD merge + rolling
    # =========================
    t0 = time.perf_counter()
    daily = load_daily_preprocessed(C.PATH_DAILY, C.GDD_DIR)
    print(f"[time] load_daily+gdd+roll={time.perf_counter()-t0:.2f}s")

    # =========================
    # 2) OBS: load + aggregate
    # =========================
    t0 = time.perf_counter()
    obs = load_obs(C.PATH_OBS)
    obs2 = aggregate_obs_daily_max(obs)
    print(f"[time] load_obs+aggregate={time.perf_counter()-t0:.2f}s")

    # =========================
    # 3) LABELS: interval/left/right + gap filter
    # =========================
    labels = build_interval_labels_from_doy(
        obs2,
        threshold=C.THRESHOLD,
        season_start_doy=C.SEASON_START_DOY,
        season_end_doy=C.SEASON_END_DOY,
    )
    labels = filter_labels_by_gap(labels, C.DOY_START, C.DOY_END, C.MAX_GAP)
    print("labels censor_type counts:\n", labels["censor_type"].value_counts())

    # =========================
    # 4) OBS META
    # =========================
    obs_meta = make_obs_meta(obs2, C.DOY_START, C.DOY_END)

    # =========================
    # 5) DAILY FEATURE FRAME + merge labels/meta/static/pheno
    # =========================
    daily_feat, _ = make_daily_feature_frame(daily)
    T = C.DOY_END - C.DOY_START + 1
    train_df = build_train_frame(daily_feat, labels, obs_meta, T=T)

    train_df = add_site_static_latlon(train_df, obs)
    train_df = merge_pheno_daily_ffill(train_df, obs)

    # =========================
    # 6) Feature set selection
    # =========================
    feature_cols = get_feature_cols(run)
    print(f"RUN={run} | D_in={len(feature_cols)}")
    # sanity: columns exist
    missing = [c for c in feature_cols if c not in train_df.columns]
    if missing:
        raise ValueError(f"Missing feature columns in train_df: {missing}")

    # =========================
    # 7) Season slice + samples
    # =========================
    t0 = time.perf_counter()
    df_season = slice_season(train_df, C.DOY_START, C.DOY_END)
    samples, dropped, feature_names = build_samples_season(df_season, feature_cols, C.DOY_START, C.DOY_END)
    print("samples:", len(samples), "| dropped groups (len!=T):", dropped)
    print(f"[time] build_samples_season={time.perf_counter()-t0:.2f}s")
    print(f"[features] n={len(feature_names)} head={feature_names[:5]} tail={feature_names[-5:]}")

    # =========================
    # 8) Split + norm + datasets
    # =========================
    result = None
    chosen = None
    if split_seeds_json is not None:
        split_seeds_json_path = resolve_split_seeds_json_path(out_root, split_seeds_json)
        split_seed, chosen_idx, chosen, payload = load_split_seed_from_topk(split_seeds_json_path, split_seed_from_topk_idx)
        train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)
        print(
            f"[split_seed_json] selected seed={split_seed} idx={chosen_idx} "
            f"file={split_seeds_json_path}"
        )
        print(f"[split_seed_json] counts={chosen.get('counts')}")
    elif auto_split_seed:
        candidates = parse_seed_candidates(seed_candidates_raw) or list(range(0, 200))
        result = split_seed_search_topk(
            samples,
            val_frac=0.1,
            test_frac=0.1,
            seed_candidates=candidates,
            target_test_interval=target_test_interval,
            tol_test_interval=tol_test_interval,
            topk=auto_split_topk,
        )
        topk_list = result["topk"]
        if not topk_list:
            raise ValueError("auto_split_seed produced no candidates")
        if split_seed_from_topk_idx is None:
            split_seed_from_topk_idx = 0
        if split_seed_from_topk_idx < 0 or split_seed_from_topk_idx >= len(topk_list):
            raise ValueError(f"--split_seed_from_topk_idx out of range (0..{len(topk_list)-1})")
        chosen = topk_list[split_seed_from_topk_idx]
        split_seed = int(chosen["seed"])
        train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)
        print(
            f"[auto_split] selected seed={split_seed} score={chosen['score']:.6f} "
            f"counts={chosen['counts']}"
        )
        if result.get("used_fallback"):
            print("[auto_split] WARNING: no seed met constraints; using best score fallback.")
        print("[auto_split] topk seeds (seed, score, test_counts):")
        for i, item in enumerate(topk_list):
            print(f"  [{i}] seed={item['seed']} score={item['score']:.6f} test={item['counts']['test']}")
    else:
        train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)

    # split stats
    if stage2_nowcast:
        train_s = build_stage2_nowcast_samples(
            train_s,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
        )
        val_s = build_stage2_nowcast_samples(
            val_s,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
        )
        test_s = build_stage2_nowcast_samples(
            test_s,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
        )
        print(
            f"[stage2_nowcast] window={stage2_nowcast_window} stride={stage2_nowcast_stride} "
            f"tstar_start={stage2_nowcast_tstar_start} only_pre_event={bool(stage2_nowcast_only_pre_event)} "
            f"event_time_proxy={stage2_nowcast_event_time_proxy} | "
            f"samples train={len(train_s)} val={len(val_s)} test={len(test_s)}"
        )
        def _bucket_counts(ss: list[dict]) -> dict[str, int]:
            out = {"pre_L": 0, "in_LR": 0, "post_R": 0, "right": 0}
            for x in ss:
                b = str(x.get("case_bucket", ""))
                if b in out:
                    out[b] += 1
            return out
        print(
            f"[stage2_nowcast] case_bucket train={_bucket_counts(train_s)} "
            f"val={_bucket_counts(val_s)} test={_bucket_counts(test_s)}"
        )

    # split stats
    tr = _split_stats(train_s)
    va = _split_stats(val_s)
    te = _split_stats(test_s)
    print(
        f"[split] train sites={tr['n_sites']} samples={tr['n_samples']} event_rate={tr['event_rate']:.4f} | "
        f"val sites={va['n_sites']} samples={va['n_samples']} event_rate={va['event_rate']:.4f} | "
        f"test sites={te['n_sites']} samples={te['n_samples']} event_rate={te['event_rate']:.4f}"
    )
    if auto_split_seed:
        n_int, mean_int, var_int = _interval_len_stats(test_s)
        print(f"[split] test interval length: n={n_int} mean={mean_int:.2f} var={var_int:.2f}")

    log_split_fingerprint("train", train_s, val_s, test_s)
    print("split:", len(train_s), len(val_s), len(test_s), "| unique sites:", len({s["site_id"] for s in samples}))

    if dropout is not None:
        C.DROPOUT = float(dropout)
    if weight_decay is not None:
        C.WEIGHT_DECAY = float(weight_decay)
    if lr is not None:
        C.LR = float(lr)
    if w_interval is not None:
        C.W_INTERVAL = float(w_interval)
    if w_left is not None:
        C.W_LEFT = float(w_left)
    if w_right is not None:
        C.W_RIGHT = float(w_right)
    print(
        f"[hparams] dropout={C.DROPOUT} weight_decay={C.WEIGHT_DECAY} lr={C.LR} "
        f"w_interval={C.W_INTERVAL} w_left={C.W_LEFT} w_right={C.W_RIGHT} "
        f"lambda_mass={lambda_mass} lambda_right_late={lambda_right_late} right_late_tau={right_late_tau}"
    )

    hparams = {
        "run": run,
        "pest": pest,
        "split_seed": split_seed,
        "split_seed_from_topk_idx": split_seed_from_topk_idx,
        "train_seeds": seeds if seeds is not None else C.SEEDS,
        "dropout": C.DROPOUT,
        "weight_decay": C.WEIGHT_DECAY,
        "lr": C.LR,
        "w_interval": C.W_INTERVAL,
        "w_left": C.W_LEFT,
        "w_right": C.W_RIGHT,
        "lambda_mass": lambda_mass,
        "lambda_right_late": lambda_right_late,
        "right_late_tau": right_late_tau,
        "stage2_nowcast": bool(stage2_nowcast),
        "stage2_nowcast_window": int(stage2_nowcast_window),
        "stage2_nowcast_stride": int(stage2_nowcast_stride),
        "stage2_nowcast_tstar_start": None if stage2_nowcast_tstar_start is None else int(stage2_nowcast_tstar_start),
        "stage2_nowcast_only_pre_event": int(stage2_nowcast_only_pre_event),
        "stage2_nowcast_event_time_proxy": stage2_nowcast_event_time_proxy,
    }
    hparams_path = Path(out_root) / "hparams.json"
    hparams_path.parent.mkdir(parents=True, exist_ok=True)
    with open(hparams_path, "w", encoding="utf-8") as f:
        json.dump(hparams, f, ensure_ascii=False, indent=2)
    print("saved:", hparams_path)

    t0 = time.perf_counter()
    x_mean, x_std = compute_norm_stats(train_s)
    print(f"[time] compute_norm_stats={time.perf_counter()-t0:.2f}s")

    train_ds = IntervalEventDataset(train_s, x_mean, x_std)
    val_ds   = IntervalEventDataset(val_s,   x_mean, x_std)
    test_ds  = IntervalEventDataset(test_s,  x_mean, x_std)

    if DEBUG_PICKLE_DATASET:
        try:
            pickle.dumps(train_ds)
            print("[debug] pickle.dumps(train_ds): OK")
        except Exception as e:
            print("[debug] pickle.dumps(train_ds): FAIL")
            raise

    if DEBUG_SAMPLE_CHECK:
        idxs = [0, len(train_ds) // 2, len(train_ds) - 1]
        for idx in idxs:
            X_i, L_i, R_i, c_i = train_ds[idx]
            print(
                f"[debug] train_ds[{idx}] X.shape={tuple(X_i.shape)} X.dtype={X_i.dtype} "
                f"L={int(L_i)} R={int(R_i)} c={int(c_i)}"
            )

    D_in = int(train_ds[0][0].shape[-1])
    print(f"[D_in] computed_from_dataset={D_in}")

    # =========================
    # 9) Multi-seed training (early stopping)
    # =========================
    trained_states = []
    train_seeds = seeds if seeds is not None else C.SEEDS
    balance_ratio = parse_balance_ratio(train_balance_ratio)
    for SEED in train_seeds:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)

        t0 = time.perf_counter()
        train_loader_kwargs = dict(batch_size=C.BATCH_TRAIN, shuffle=True, seed=SEED)
        val_loader_kwargs = dict(batch_size=C.BATCH_EVAL, shuffle=False)
        if balance_ratio is not None:
            sampler = build_balanced_sampler(train_s, balance_ratio, seed=SEED)
            train_loader_kwargs["sampler"] = sampler
            train_loader_kwargs["shuffle"] = False
            train_loader_kwargs["seed"] = None
            print(f"[seed {SEED}] balanced sampler enabled: ratio(right,interval,left)="
                  f"{balance_ratio['right']:.3f}:{balance_ratio['interval']:.3f}:{balance_ratio['left']:.3f}")
        if DEBUG_LOADER_SETTINGS and C.NUM_WORKERS > 0:
            train_loader = make_loader(train_ds, **train_loader_kwargs, multiprocessing_context="spawn")
            val_loader = make_loader(val_ds, **val_loader_kwargs, multiprocessing_context="spawn")
        else:
            train_loader = make_loader(train_ds, **train_loader_kwargs)
            val_loader = make_loader(val_ds, **val_loader_kwargs)
        print(f"[time] make_loaders={time.perf_counter()-t0:.2f}s")

        t0 = time.perf_counter()
        _ = next(iter(train_loader))
        print(f"[time] first_batch={time.perf_counter()-t0:.2f}s")
        # test_loader  = make_loader(test_ds,  C.BATCH_EVAL,  shuffle=False)  # eval script에서 사용

        torch.manual_seed(SEED)
        model = HazardTransformer(
            d_in=D_in,
            d_model=C.D_MODEL,
            nhead=C.N_HEAD,
            num_layers=C.N_LAYERS,
            dropout=C.DROPOUT,
            max_len=C.MAX_LEN,
        ).to(device)

        opt = torch.optim.AdamW(model.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)

        best_val = float("inf")
        best_val_iou = float("-inf")
        best_state = None
        best_epoch = -1
        pat = 0

        for epoch in range(1, C.MAX_EPOCHS + 1):
            tr, tr_base, tr_mass, tr_late = run_epoch_weighted(
                model,
                opt,
                train_loader,
                Tend=T,
                device=device,
                train=True,
                lambda_mass=lambda_mass,
                lambda_right_late=lambda_right_late,
                right_late_tau=right_late_tau,
                log_mass=True,
                epoch_idx=epoch,
                return_parts=True,
            )
            va = eval_nll_model(model, val_loader, Tend=T, device=device)
            va_stats = eval_metrics_with_overlap(
                model,
                val_loader,
                Tend=T,
                device=device,
                alpha=0.2,
                pi_method=getattr(C, "PI_METHOD", "shortest"),
            )
            va_iou = float(va_stats["IoU_mean_interval_only(80%)"])
            if np.isnan(va_iou):
                va_iou = float("-inf")
            if not np.isfinite(va):
                print(f"[seed {SEED}] epoch {epoch:02d} | val_nll is non-finite ({va}); forcing +inf for model selection")
                va = float("inf")
            print(
                f"[seed {SEED}] epoch {epoch:02d} | train_total {tr:.4f} | train_base {tr_base:.4f} "
                f"| train_mass {tr_mass:.4f} | train_late {tr_late:.4f} | val_nll {va:.4f} | val_iou80 {va_iou:.4f}"
            )

            improved_iou = va_iou > (best_val_iou + C.MIN_DELTA)
            tie_iou_better_nll = (abs(va_iou - best_val_iou) <= C.MIN_DELTA) and (va < best_val - C.MIN_DELTA)
            if improved_iou or tie_iou_better_nll:
                best_val = float(va)
                best_val_iou = float(va_iou)
                best_epoch = epoch
                best_state = copy.deepcopy(model.state_dict())
                pat = 0
            else:
                pat += 1
                if pat >= C.PATIENCE:
                    break

        if best_state is None:
            raise RuntimeError(f"[seed {SEED}] best_state is None")

        trained_states.append(
            {
                "seed": SEED,
                "best_epoch": best_epoch,
                "best_val_nll": best_val,
                "best_val_iou80": best_val_iou,
                "state_dict": best_state,
            }
        )
        print(
            f"[seed {SEED}] DONE | best_epoch={best_epoch} | "
            f"best_val_iou80={best_val_iou:.4f} | best_val_nll={best_val:.4f}\n"
        )

    # =========================
    # 10) Save checkpoint bundle
    # =========================
    bundle = {
        **build_ckpt_meta(
            run=run,
            pest=pest,
            d_in=D_in,
            feature_cols=feature_cols,
            feature_names=feature_names,
            year_max=getattr(C, "YEAR_MAX", None),
        ),
        "doy_start": C.DOY_START,
        "doy_end": C.DOY_END,
        "T": T,
        "stage2_nowcast": bool(stage2_nowcast),
        "stage2_nowcast_window": int(stage2_nowcast_window),
        "stage2_nowcast_stride": int(stage2_nowcast_stride),
        "stage2_nowcast_tstar_start": None if stage2_nowcast_tstar_start is None else int(stage2_nowcast_tstar_start),
        "stage2_nowcast_only_pre_event": int(stage2_nowcast_only_pre_event),
        "stage2_nowcast_event_time_proxy": stage2_nowcast_event_time_proxy,
        "norm_mean": x_mean,
        "norm_std": x_std,
        "trained_states": trained_states,
        "split_counts": {"train": len(train_s), "val": len(val_s), "test": len(test_s)},
    }
    out_path_resolved = resolve_out_path(run, out_root, out_path)
    torch.save(bundle, out_path_resolved)
    print("saved:", out_path_resolved)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, required=True)
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--split_seed", type=int, default=C.SPLIT_SEED)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--auto_split_seed", action="store_true")
    p.add_argument("--auto_split_topk", type=int, default=1)
    p.add_argument("--split_seed_from_topk_idx", type=int, default=None)
    p.add_argument("--split_seeds_json", type=str, default=None)
    p.add_argument("--seed_candidates", type=str, default=None)
    p.add_argument("--target_test_interval", type=int, default=None)
    p.add_argument("--tol_test_interval", type=int, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--w_interval", type=float, default=None)
    p.add_argument("--w_left", type=float, default=None)
    p.add_argument("--w_right", type=float, default=None)
    p.add_argument("--lambda_mass", type=float, default=0.0)
    p.add_argument("--lambda_right_late", type=float, default=0.0)
    p.add_argument("--right_late_tau", type=float, default=220.0)
    p.add_argument("--train_balance_ratio", type=str, default="1:1:1")
    p.add_argument("--stage2_nowcast", action="store_true")
    p.add_argument("--stage2_nowcast_window", type=int, default=56)
    p.add_argument("--stage2_nowcast_stride", type=int, default=7)
    p.add_argument("--stage2_nowcast_tstar_start", type=int, default=None)
    p.add_argument("--stage2_nowcast_only_pre_event", type=int, default=1)
    p.add_argument("--stage2_nowcast_event_time_proxy", type=str, default="r", choices=["r", "mid"])
    # Stage-1 style aliases for pipeline consistency.
    p.add_argument("--nowcast_window", dest="stage2_nowcast_window", type=int)
    p.add_argument("--nowcast_stride", dest="stage2_nowcast_stride", type=int)
    p.add_argument("--nowcast_tstar_start", dest="stage2_nowcast_tstar_start", type=int)
    p.add_argument("--nowcast_only_pre_event", dest="stage2_nowcast_only_pre_event", type=int)
    p.add_argument("--nowcast_event_time_proxy", dest="stage2_nowcast_event_time_proxy", type=str, choices=["r", "mid"])
    args = p.parse_args()
    main(
        args.pest,
        args.run,
        args.out_root,
        args.out,
        args.split_seed,
        args.seeds,
        args.auto_split_seed,
        args.seed_candidates,
        args.target_test_interval,
        args.tol_test_interval,
        args.auto_split_topk,
        args.split_seed_from_topk_idx,
        args.split_seeds_json,
        args.dropout,
        args.weight_decay,
        args.lr,
        args.w_interval,
        args.w_left,
        args.w_right,
        args.lambda_mass,
        args.lambda_right_late,
        args.right_late_tau,
        args.train_balance_ratio,
        args.stage2_nowcast,
        args.stage2_nowcast_window,
        args.stage2_nowcast_stride,
        args.stage2_nowcast_tstar_start,
        args.stage2_nowcast_only_pre_event,
        args.stage2_nowcast_event_time_proxy,
    )
