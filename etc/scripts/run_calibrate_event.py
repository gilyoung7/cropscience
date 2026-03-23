from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import torch

from etc.configs import config as C
from etc.src.pest_resolver import resolve_pest, default_out_root, ensure_output_dirs
from etc.scripts.common import make_loader, parse_seed_candidates
from etc.scripts.run_eval import (
    build_samples_for_run,
    resolve_ckpt_path,
    resolve_split_seeds_json_path,
    load_split_seed_from_topk,
)
from etc.src.dataset import (
    split_by_sample,
    compute_norm_stats,
    IntervalEventDataset,
    split_seed_search_topk,
    log_split_fingerprint,
)
from etc.src.ckpt_schema import validate_ckpt_meta
from etc.src.model import HazardTransformer
from etc.src.train_eval import hazard_to_pmf_cdf_logS


def auc_roc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    pos = y_true == 1
    neg = y_true == 0
    n_pos = int(pos.sum())
    n_neg = int(neg.sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")

    order = np.argsort(y_score)
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, len(y_score) + 1)
    sum_pos = float(ranks[pos].sum())
    return (sum_pos - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    return float(np.mean((y_prob - y_true) ** 2))


def binary_nll(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-8) -> float:
    y_true = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y_true * np.log(p) + (1.0 - y_true) * np.log(1.0 - p)))


def apply_temperature(p: np.ndarray, temperature: float, eps: float = 1e-8) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    z = logit / float(temperature)
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature_grid(y_true: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
    # Wide but simple grid; robust without extra deps.
    grid = np.concatenate(
        [
            np.linspace(0.2, 1.0, 81),
            np.linspace(1.05, 3.0, 79),
            np.array([4.0, 5.0, 7.5, 10.0]),
        ]
    )
    best_t = 1.0
    best_nll = math.inf
    for t in grid:
        p_cal = apply_temperature(p_val, float(t))
        nll = binary_nll(y_true, p_cal)
        if nll < best_nll:
            best_nll = nll
            best_t = float(t)
    return best_t, float(best_nll)


def best_tau_by_f1(y_true: np.ndarray, p: np.ndarray) -> tuple[float, float, float, float]:
    # Choose tau on validation by max F1.
    taus = np.linspace(0.05, 0.95, 181)
    y = np.asarray(y_true, dtype=int)
    best = (0.5, -1.0, 0.0, 0.0)  # tau, f1, precision, recall
    for tau in taus:
        pred = (p >= float(tau)).astype(int)
        tp = int(((pred == 1) & (y == 1)).sum())
        fp = int(((pred == 1) & (y == 0)).sum())
        fn = int(((pred == 0) & (y == 1)).sum())
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
        if f1 > best[1]:
            best = (float(tau), float(f1), float(prec), float(rec))
    return best


def pr_at_tau(y_true: np.ndarray, p: np.ndarray, tau: float) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int)
    pred = (p >= float(tau)).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


@torch.no_grad()
def predict_p_event_and_median(model, loader, Tend, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    p_out = []
    m_out = []
    for X, _, _, _ in loader:
        X = X.to(device, non_blocking=True)
        hazard = model(X)
        _, cdf, logS = hazard_to_pmf_cdf_logS(hazard)
        p_event = 1.0 - torch.exp(logS[:, -1])
        cdf_last = cdf[:, -1]
        arg = (cdf >= 0.5).float().argmax(dim=1) + 1
        median = torch.where(cdf_last >= 0.5, arg, torch.tensor(Tend, device=cdf.device))
        p_out.append(p_event.cpu().numpy())
        m_out.append(median.cpu().numpy())
    if not p_out:
        return np.zeros((0,), dtype=float), np.zeros((0,), dtype=int)
    return np.concatenate(p_out, axis=0), np.concatenate(m_out, axis=0).astype(int)


def prf_from_pred(y_true: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    y = np.asarray(y_true, dtype=int)
    p = np.asarray(pred, dtype=int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return float(prec), float(rec), float(f1)


def resolve_out_csv(run: int, out_root: str, out_csv: str | None) -> str:
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        return out_csv
    out_dir = Path(out_root) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"calibration_run{run}.csv")


def main(
    pest: str,
    ckpt_path: str | None,
    run: int,
    out_csv: str | None,
    out_root: str,
    allow_run_mismatch: bool,
    split_seed: int,
    seeds: list[int] | None,
    auto_split_seed: bool,
    seed_candidates_raw: str | None,
    target_test_interval: int | None,
    tol_test_interval: int | None,
    auto_split_topk: int,
    split_seed_from_topk_idx: int | None,
    split_seeds_json: str | None,
    late_gate_from_end: int | None,
):
    _, get_feature_cols = resolve_pest(pest)
    if not out_root:
        out_root = default_out_root(pest)
    ensure_output_dirs(out_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    ckpt_path_resolved = resolve_ckpt_path(run, out_root, ckpt_path)
    print(f"Using checkpoint: {ckpt_path_resolved}")
    ckpt = torch.load(ckpt_path_resolved, map_location="cpu")
    trained_states = ckpt["trained_states"]
    print("loaded ckpt:", ckpt_path_resolved, "| seeds:", [d["seed"] for d in trained_states])

    feature_cols, feature_names_eval, T, samples = build_samples_for_run(run, get_feature_cols)
    print(f"[features] n={len(feature_names_eval)} head={feature_names_eval[:5]} tail={feature_names_eval[-5:]}")

    chosen = None
    if split_seeds_json is not None:
        split_seeds_json_path = resolve_split_seeds_json_path(out_root, split_seeds_json)
        split_seed, chosen_idx, chosen, _ = load_split_seed_from_topk(split_seeds_json_path, split_seed_from_topk_idx)
        train_s, val_s, test_s = split_by_sample(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)
        print(f"[split_seed_json] selected seed={split_seed} idx={chosen_idx} file={split_seeds_json_path}")
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
        train_s, val_s, test_s = split_by_sample(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)
        print(
            f"[auto_split] selected seed={split_seed} score={chosen['score']:.6f} "
            f"counts={chosen['counts']}"
        )
    else:
        train_s, val_s, test_s = split_by_sample(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)

    log_split_fingerprint("calib", train_s, val_s, test_s)

    x_mean, x_std = compute_norm_stats(train_s)
    train_ds = IntervalEventDataset(train_s, x_mean, x_std)
    val_ds = IntervalEventDataset(val_s, x_mean, x_std)
    test_ds = IntervalEventDataset(test_s, x_mean, x_std)
    D_in = int(test_ds[0][0].shape[-1])
    print(f"[D_in] computed_from_dataset={D_in}")
    print(f"RUN={run} | D_in={D_in} | T={T}")

    validate_ckpt_meta(
        ckpt,
        pest=pest,
        run=run,
        d_in=D_in,
        feature_names=feature_names_eval,
        allow_run_mismatch=allow_run_mismatch,
    )

    val_loader = make_loader(val_ds, C.BATCH_EVAL, shuffle=False)
    test_loader = make_loader(test_ds, C.BATCH_EVAL, shuffle=False)

    y_val = np.array([0 if str(s["censor_type"]) == "right" else 1 for s in val_s], dtype=int)
    y_test = np.array([0 if str(s["censor_type"]) == "right" else 1 for s in test_s], dtype=int)
    print(f"[labels] val event_rate={y_val.mean():.4f} test event_rate={y_test.mean():.4f}")

    records = []
    for d in trained_states:
        seed = int(d["seed"])
        if seeds is not None and seed not in seeds:
            continue

        model = HazardTransformer(
            d_in=D_in,
            d_model=C.D_MODEL,
            nhead=C.N_HEAD,
            num_layers=C.N_LAYERS,
            dropout=C.DROPOUT,
            max_len=C.MAX_LEN,
        ).to(device)
        model.load_state_dict(d["state_dict"])
        model.eval()

        p_val_raw, m_val = predict_p_event_and_median(model, val_loader, Tend=T, device=device)
        p_test_raw, m_test = predict_p_event_and_median(model, test_loader, Tend=T, device=device)

        best_t, val_nll_cal = fit_temperature_grid(y_val, p_val_raw)
        p_val_cal = apply_temperature(p_val_raw, best_t)
        p_test_cal = apply_temperature(p_test_raw, best_t)

        tau, val_f1, val_prec, val_rec = best_tau_by_f1(y_val, p_val_cal)
        test_prec, test_rec, test_f1 = pr_at_tau(y_test, p_test_cal, tau)

        if late_gate_from_end is not None and int(late_gate_from_end) >= 0:
            gate_start = max(1, T - int(late_gate_from_end) + 1)
            late_val = m_val >= gate_start
            late_test = m_test >= gate_start
            pred_val_g = ((p_val_cal >= float(tau)) & (~late_val)).astype(int)
            pred_test_g = ((p_test_cal >= float(tau)) & (~late_test)).astype(int)
            val_prec_g, val_rec_g, val_f1_g = prf_from_pred(y_val, pred_val_g)
            test_prec_g, test_rec_g, test_f1_g = prf_from_pred(y_test, pred_test_g)
            val_gate_rate = float(late_val.mean())
            test_gate_rate = float(late_test.mean())
        else:
            gate_start = None
            val_prec_g = val_rec_g = val_f1_g = float("nan")
            test_prec_g = test_rec_g = test_f1_g = float("nan")
            val_gate_rate = test_gate_rate = float("nan")

        rec = {
            "seed": seed,
            "best_epoch": int(d["best_epoch"]),
            "temperature": float(best_t),
            "val_auc_raw": float(auc_roc_binary(y_val, p_val_raw)),
            "val_auc_cal": float(auc_roc_binary(y_val, p_val_cal)),
            "test_auc_raw": float(auc_roc_binary(y_test, p_test_raw)),
            "test_auc_cal": float(auc_roc_binary(y_test, p_test_cal)),
            "val_brier_raw": float(brier_score(y_val, p_val_raw)),
            "val_brier_cal": float(brier_score(y_val, p_val_cal)),
            "test_brier_raw": float(brier_score(y_test, p_test_raw)),
            "test_brier_cal": float(brier_score(y_test, p_test_cal)),
            "val_nll_raw": float(binary_nll(y_val, p_val_raw)),
            "val_nll_cal": float(val_nll_cal),
            "test_nll_raw": float(binary_nll(y_test, p_test_raw)),
            "test_nll_cal": float(binary_nll(y_test, p_test_cal)),
            "tau_from_val_f1": float(tau),
            "val_prec@tau": float(val_prec),
            "val_rec@tau": float(val_rec),
            "val_f1@tau": float(val_f1),
            "test_prec@tau": float(test_prec),
            "test_rec@tau": float(test_rec),
            "test_f1@tau": float(test_f1),
            "late_gate_from_end": int(late_gate_from_end) if late_gate_from_end is not None else None,
            "late_gate_start_rel": int(gate_start) if gate_start is not None else None,
            "val_gate_rate": float(val_gate_rate),
            "test_gate_rate": float(test_gate_rate),
            "val_prec@tau_gate": float(val_prec_g),
            "val_rec@tau_gate": float(val_rec_g),
            "val_f1@tau_gate": float(val_f1_g),
            "test_prec@tau_gate": float(test_prec_g),
            "test_rec@tau_gate": float(test_rec_g),
            "test_f1@tau_gate": float(test_f1_g),
            "test_mean_p_raw": float(np.mean(p_test_raw)),
            "test_mean_p_cal": float(np.mean(p_test_cal)),
            "test_event_mean_p_raw": float(np.mean(p_test_raw[y_test == 1])) if (y_test == 1).any() else float("nan"),
            "test_nonevent_mean_p_raw": float(np.mean(p_test_raw[y_test == 0])) if (y_test == 0).any() else float("nan"),
            "test_event_mean_p_cal": float(np.mean(p_test_cal[y_test == 1])) if (y_test == 1).any() else float("nan"),
            "test_nonevent_mean_p_cal": float(np.mean(p_test_cal[y_test == 0])) if (y_test == 0).any() else float("nan"),
        }
        records.append(rec)

    df = pd.DataFrame(records).sort_values("seed").reset_index(drop=True)
    print("\n=== per-seed calibration ===")
    print(df.to_string(index=False))

    summarize_cols = [
        "temperature",
        "test_auc_raw",
        "test_auc_cal",
        "test_brier_raw",
        "test_brier_cal",
        "test_nll_raw",
        "test_nll_cal",
        "test_prec@tau",
        "test_rec@tau",
        "test_f1@tau",
        "test_prec@tau_gate",
        "test_rec@tau_gate",
        "test_f1@tau_gate",
        "test_event_mean_p_raw",
        "test_nonevent_mean_p_raw",
        "test_event_mean_p_cal",
        "test_nonevent_mean_p_cal",
    ]
    print("\n=== mean ± std (seeds) ===")
    for col in summarize_cols:
        arr = df[col].to_numpy(dtype=float)
        print(f"{col:>24s}: {float(np.mean(arr)):.4f} ± {float(np.std(arr, ddof=1) if len(arr) > 1 else 0.0):.4f}")

    out_csv_resolved = resolve_out_csv(run, out_root, out_csv)
    df.to_csv(out_csv_resolved, index=False)
    print("saved:", out_csv_resolved)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--out_csv", type=str, default=None)
    p.add_argument("--out_root", type=str, default=None)
    p.add_argument("--allow_run_mismatch", action="store_true")
    p.add_argument("--split_seed", type=int, default=C.SPLIT_SEED)
    p.add_argument("--seeds", type=int, nargs="*", default=None)
    p.add_argument("--auto_split_seed", action="store_true")
    p.add_argument("--auto_split_topk", type=int, default=1)
    p.add_argument("--split_seed_from_topk_idx", type=int, default=None)
    p.add_argument("--split_seeds_json", type=str, default=None)
    p.add_argument("--seed_candidates", type=str, default=None)
    p.add_argument("--target_test_interval", type=int, default=None)
    p.add_argument("--tol_test_interval", type=int, default=None)
    p.add_argument("--late_gate_from_end", type=int, default=None)
    args = p.parse_args()
    main(
        pest=args.pest,
        ckpt_path=args.ckpt,
        run=args.run,
        out_csv=args.out_csv,
        out_root=args.out_root,
        allow_run_mismatch=args.allow_run_mismatch,
        split_seed=args.split_seed,
        seeds=args.seeds,
        auto_split_seed=args.auto_split_seed,
        seed_candidates_raw=args.seed_candidates,
        target_test_interval=args.target_test_interval,
        tol_test_interval=args.tol_test_interval,
        auto_split_topk=args.auto_split_topk,
        split_seed_from_topk_idx=args.split_seed_from_topk_idx,
        split_seeds_json=args.split_seeds_json,
        late_gate_from_end=args.late_gate_from_end,
    )
