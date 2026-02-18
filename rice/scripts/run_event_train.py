from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from torch.utils.data import Dataset

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest, default_out_root, ensure_output_dirs
from rice.scripts.common import make_loader, parse_seed_candidates
from rice.scripts.run_eval import build_samples_for_run
from rice.src.dataset import (
    split_by_site,
    compute_norm_stats,
    IntervalEventDataset,
    split_seed_search_topk,
    log_split_fingerprint,
)
from rice.src.ckpt_schema import build_ckpt_meta


class EventTransformer(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_model: int = 64,
        nhead: int = 4,
        num_layers: int = 2,
        dropout: float = 0.2,
        max_len: int = 400,
    ):
        super().__init__()
        self.in_proj = nn.Linear(d_in, d_model)
        self.pos_emb = nn.Parameter(torch.zeros(1, max_len, d_model))
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, 1),
        )

    def forward(self, x):
        # x: (B,T,D)
        b, t, _ = x.shape
        z = self.in_proj(x)
        z = z + self.pos_emb[:, :t, :]
        h = self.encoder(z)
        pooled = h.mean(dim=1)
        return self.head(pooled).squeeze(1)


def resolve_out_path(run: int, out_root: str, out_path: str | None) -> Path:
    if out_path:
        p = Path(out_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p
    out_dir = Path(out_root) / "ckpt"
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir / f"event_classifier_run{run}.pt"


def resolve_split_seeds_json_path(out_root: str, split_seeds_json: str | None) -> Path:
    if split_seeds_json:
        return Path(split_seeds_json)
    return Path(out_root) / "splits" / "selected_split_seeds.json"


def load_split_seed_from_topk(split_seeds_json_path: Path, split_seed_from_topk_idx: int | None):
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
    return int(chosen["seed"]), int(split_seed_from_topk_idx), chosen, payload


def run_epoch_event(model, loader, device, train, opt=None, pos_weight: float = 1.0):
    model.train(train)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(pos_weight), device=device))
    total = 0.0
    n = 0
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            X, _, _, ctype = batch
            X = X.to(device, non_blocking=True)
            # event=1 for left/interval, event=0 for right
            y = (ctype.to(device, non_blocking=True) != 1).float()
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            X, y = batch
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
        else:
            raise ValueError("unexpected batch format in run_epoch_event")
        logits = model(X)
        loss = crit(logits, y)
        if train:
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), C.GRAD_CLIP_NORM)
            opt.step()
        total += float(loss.item()) * X.size(0)
        n += X.size(0)
    return total / max(n, 1)


class EventBinaryDataset(Dataset):
    def __init__(self, samples: list[dict], mean: np.ndarray, std: np.ndarray):
        self.samples = samples
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        X = (s["X"] - self.mean) / self.std
        X = torch.from_numpy(X).float()
        y = torch.tensor(float(s["y_event"]), dtype=torch.float32)
        return X, y


def build_tabular_from_samples(samples: list[dict]) -> np.ndarray:
    """
    Build fixed-size tabular features from (T, D) sequences.
    Per channel stats: mean, std, min, max, first, last, slope.
    """
    feats = []
    for s in samples:
        x = np.asarray(s["X"], dtype=np.float32)  # (T,D)
        t = np.arange(x.shape[0], dtype=np.float32)
        t_center = t - t.mean()
        t_var = float((t_center ** 2).sum()) + 1e-8

        mean = x.mean(axis=0)
        std = x.std(axis=0)
        xmin = x.min(axis=0)
        xmax = x.max(axis=0)
        xfirst = x[0]
        xlast = x[-1]
        slope = ((x - mean) * t_center[:, None]).sum(axis=0) / t_var

        f = np.concatenate([mean, std, xmin, xmax, xfirst, xlast, slope], axis=0)
        feats.append(f)
    return np.stack(feats, axis=0).astype(np.float32) if feats else np.zeros((0, 0), dtype=np.float32)


def make_event_labels(samples: list[dict]) -> np.ndarray:
    if len(samples) > 0 and "y_event" in samples[0]:
        return np.array([int(s["y_event"]) for s in samples], dtype=np.int64)
    return np.array([0 if str(s["censor_type"]) == "right" else 1 for s in samples], dtype=np.int64)


def build_nowcast_samples(
    samples: list[dict],
    window: int,
    stride: int,
    tstar_start: int | None = None,
    only_pre_event: bool = True,
) -> list[dict]:
    out = []
    if not samples:
        return out
    if window <= 0:
        raise ValueError("--nowcast_window must be >= 1")
    if stride <= 0:
        raise ValueError("--nowcast_stride must be >= 1")

    T = int(samples[0]["X"].shape[0])
    t0 = int(window if tstar_start is None else tstar_start)
    t0 = max(t0, window)
    t0 = min(t0, T)

    for s in samples:
        x = np.asarray(s["X"], dtype=np.float32)
        ctype = str(s["censor_type"])
        has_event = ctype != "right"
        # Proxy event time for nowcasting target:
        # use R (first observed positive upper bound / earliest guaranteed occurrence by censoring def).
        event_time = int(s["R"]) if has_event else None

        for tstar in range(t0, T + 1, stride):
            if only_pre_event and has_event and event_time is not None and tstar >= event_time:
                continue
            y_event = 1 if (has_event and event_time is not None and event_time > tstar) else 0
            xw = x[(tstar - window):tstar, :]
            out.append(
                {
                    "site_id": s["site_id"],
                    "year": int(s["year"]),
                    "X": xw.astype(np.float32, copy=False),
                    "y_event": int(y_event),
                    "tstar": int(tstar),
                    "event_time": int(event_time) if event_time is not None else None,
                    "base_censor_type": ctype,
                }
            )
    return out


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
    lr: float | None,
    weight_decay: float | None,
    dropout: float | None,
    max_epochs: int,
    patience: int,
    min_delta: float,
    event_pos_weight: float,
    model: str,
    task_mode: str,
    nowcast_window: int,
    nowcast_stride: int,
    nowcast_tstar_start: int | None,
    nowcast_only_pre_event: int,
):
    _, get_feature_cols = resolve_pest(pest)
    if not out_root:
        out_root = default_out_root(pest)
    ensure_output_dirs(out_root)

    if lr is not None:
        C.LR = float(lr)
    if weight_decay is not None:
        C.WEIGHT_DECAY = float(weight_decay)
    if dropout is not None:
        C.DROPOUT = float(dropout)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    feature_cols, feature_names, T, samples = build_samples_for_run(run, get_feature_cols)
    print(f"[features] n={len(feature_names)} head={feature_names[:5]} tail={feature_names[-5:]}")

    if split_seeds_json is not None:
        split_seeds_json_path = resolve_split_seeds_json_path(out_root, split_seeds_json)
        split_seed, chosen_idx, chosen, _ = load_split_seed_from_topk(split_seeds_json_path, split_seed_from_topk_idx)
        train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)
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
        chosen = topk_list[split_seed_from_topk_idx]
        split_seed = int(chosen["seed"])
        train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)
        print(f"[auto_split] selected seed={split_seed} score={chosen['score']:.6f} counts={chosen['counts']}")
    else:
        train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)

    if task_mode == "nowcast":
        train_s = build_nowcast_samples(
            train_s,
            window=nowcast_window,
            stride=nowcast_stride,
            tstar_start=nowcast_tstar_start,
            only_pre_event=bool(nowcast_only_pre_event),
        )
        val_s = build_nowcast_samples(
            val_s,
            window=nowcast_window,
            stride=nowcast_stride,
            tstar_start=nowcast_tstar_start,
            only_pre_event=bool(nowcast_only_pre_event),
        )
        test_s = build_nowcast_samples(
            test_s,
            window=nowcast_window,
            stride=nowcast_stride,
            tstar_start=nowcast_tstar_start,
            only_pre_event=bool(nowcast_only_pre_event),
        )
        print(
            f"[nowcast] window={nowcast_window} stride={nowcast_stride} "
            f"tstar_start={nowcast_tstar_start} only_pre_event={bool(nowcast_only_pre_event)} | "
            f"samples train={len(train_s)} val={len(val_s)} test={len(test_s)}"
        )

    log_split_fingerprint("event_train", train_s, val_s, test_s)

    x_mean, x_std = compute_norm_stats(train_s)
    if task_mode == "nowcast":
        train_ds = EventBinaryDataset(train_s, x_mean, x_std)
        val_ds = EventBinaryDataset(val_s, x_mean, x_std)
    else:
        train_ds = IntervalEventDataset(train_s, x_mean, x_std)
        val_ds = IntervalEventDataset(val_s, x_mean, x_std)
    D_in = int(train_ds[0][0].shape[-1])
    model_T = int(nowcast_window if task_mode == "nowcast" else T)
    print(f"RUN={run} | D_in={D_in} | T={model_T} | model={model} | task_mode={task_mode}")

    # auto class imbalance weight
    if event_pos_weight <= 0:
        y_train = make_event_labels(train_s).astype(int)
        n_pos = int((y_train == 1).sum())
        n_neg = int((y_train == 0).sum())
        event_pos_weight = float(n_neg / max(n_pos, 1))
    print(f"[hparams] lr={C.LR} wd={C.WEIGHT_DECAY} dropout={C.DROPOUT} max_epochs={max_epochs} patience={patience} pos_weight={event_pos_weight:.4f}")

    trained_states = []
    tab_feature_dim = None
    train_seeds = seeds if seeds is not None else C.SEEDS
    for SEED in train_seeds:
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(SEED)

        if model == "transformer":
            train_loader = make_loader(train_ds, C.BATCH_TRAIN, shuffle=True, seed=SEED)
            val_loader = make_loader(val_ds, C.BATCH_EVAL, shuffle=False)

            model_obj = EventTransformer(
                d_in=D_in,
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=2,
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
            ).to(device)
            opt = torch.optim.AdamW(model_obj.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)

            best_val = float("inf")
            best_state = None
            best_epoch = -1
            pat = 0
            for epoch in range(1, max_epochs + 1):
                tr = run_epoch_event(model_obj, train_loader, device=device, train=True, opt=opt, pos_weight=event_pos_weight)
                va = run_epoch_event(model_obj, val_loader, device=device, train=False, pos_weight=event_pos_weight)
                print(f"[seed {SEED}] epoch {epoch:02d} | train_bce {tr:.4f} | val_bce {va:.4f}")
                if va < best_val - min_delta:
                    best_val = float(va)
                    best_epoch = epoch
                    best_state = copy.deepcopy(model_obj.state_dict())
                    pat = 0
                else:
                    pat += 1
                    if pat >= patience:
                        break

            if best_state is None:
                raise RuntimeError(f"[seed {SEED}] best_state is None")

            trained_states.append(
                {
                    "seed": SEED,
                    "best_epoch": int(best_epoch),
                    "best_val_bce": float(best_val),
                    "state_dict": best_state,
                }
            )
            print(f"[seed {SEED}] DONE | best_epoch={best_epoch} | best_val_bce={best_val:.4f}")
        else:
            X_tr = build_tabular_from_samples(train_s)
            X_va = build_tabular_from_samples(val_s)
            y_tr = make_event_labels(train_s)
            y_va = make_event_labels(val_s)
            tab_feature_dim = int(X_tr.shape[1]) if X_tr.size > 0 else 0

            if model == "logreg":
                cls_w = {0: 1.0, 1: float(event_pos_weight)}
                clf = LogisticRegression(
                    max_iter=2000,
                    class_weight=cls_w,
                    random_state=SEED,
                    solver="lbfgs",
                )
                clf.fit(X_tr, y_tr)
            elif model == "lgbm":
                try:
                    from lightgbm import LGBMClassifier
                except Exception as e:
                    raise ImportError("lightgbm is not installed. Install it to use --model lgbm.") from e
                clf = LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    random_state=SEED,
                    class_weight={0: 1.0, 1: float(event_pos_weight)},
                )
                clf.fit(X_tr, y_tr)
            elif model == "xgb":
                try:
                    from xgboost import XGBClassifier
                except Exception as e:
                    raise ImportError("xgboost is not installed. Install it to use --model xgb.") from e
                clf = XGBClassifier(
                    n_estimators=400,
                    max_depth=5,
                    learning_rate=0.05,
                    subsample=0.9,
                    colsample_bytree=0.9,
                    reg_lambda=1.0,
                    random_state=SEED,
                    eval_metric="logloss",
                    scale_pos_weight=float(event_pos_weight),
                )
                clf.fit(X_tr, y_tr)
            else:
                raise ValueError(f"unsupported --model: {model}")

            p_va = clf.predict_proba(X_va)[:, 1]
            eps = 1e-8
            p_va = np.clip(p_va, eps, 1.0 - eps)
            best_val = float(-np.mean(y_va * np.log(p_va) + (1.0 - y_va) * np.log(1.0 - p_va)))

            trained_states.append(
                {
                    "seed": SEED,
                    "best_epoch": 1,
                    "best_val_bce": float(best_val),
                    "sk_model": clf,
                }
            )
            print(f"[seed {SEED}] DONE | best_epoch=1 | best_val_bce={best_val:.4f}")

    bundle = {
        **build_ckpt_meta(
            run=run,
            pest=pest,
            d_in=D_in,
            feature_cols=feature_cols,
            feature_names=feature_names,
            year_max=getattr(C, "YEAR_MAX", None),
        ),
        "model_type": "event_transformer" if model == "transformer" else "event_tabular",
        "event_model": model,
        "doy_start": C.DOY_START,
        "doy_end": C.DOY_END,
        "T": model_T,
        "task_mode": task_mode,
        "nowcast_window": int(nowcast_window),
        "nowcast_stride": int(nowcast_stride),
        "nowcast_tstar_start": None if nowcast_tstar_start is None else int(nowcast_tstar_start),
        "nowcast_only_pre_event": int(nowcast_only_pre_event),
        "norm_mean": x_mean,
        "norm_std": x_std,
        "tab_feature_dim": tab_feature_dim,
        "trained_states": trained_states,
        "split_seed": int(split_seed),
        "split_counts": {"train": len(train_s), "val": len(val_s), "test": len(test_s)},
        "event_pos_weight": float(event_pos_weight),
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
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight_decay", type=float, default=None)
    p.add_argument("--dropout", type=float, default=None)
    p.add_argument("--max_epochs", type=int, default=60)
    p.add_argument("--patience", type=int, default=8)
    p.add_argument("--min_delta", type=float, default=1e-3)
    p.add_argument("--event_pos_weight", type=float, default=0.0)
    p.add_argument("--model", type=str, default="transformer", choices=["transformer", "logreg", "lgbm", "xgb"])
    p.add_argument("--task_mode", type=str, default="season_complete", choices=["season_complete", "nowcast"])
    p.add_argument("--nowcast_window", type=int, default=28)
    p.add_argument("--nowcast_stride", type=int, default=7)
    p.add_argument("--nowcast_tstar_start", type=int, default=None)
    p.add_argument("--nowcast_only_pre_event", type=int, default=1)
    args = p.parse_args()
    main(
        pest=args.pest,
        run=args.run,
        out_root=args.out_root,
        out_path=args.out,
        split_seed=args.split_seed,
        seeds=args.seeds,
        auto_split_seed=args.auto_split_seed,
        seed_candidates_raw=args.seed_candidates,
        target_test_interval=args.target_test_interval,
        tol_test_interval=args.tol_test_interval,
        auto_split_topk=args.auto_split_topk,
        split_seed_from_topk_idx=args.split_seed_from_topk_idx,
        split_seeds_json=args.split_seeds_json,
        lr=args.lr,
        weight_decay=args.weight_decay,
        dropout=args.dropout,
        max_epochs=args.max_epochs,
        patience=args.patience,
        min_delta=args.min_delta,
        event_pos_weight=args.event_pos_weight,
        model=args.model,
        task_mode=args.task_mode,
        nowcast_window=args.nowcast_window,
        nowcast_stride=args.nowcast_stride,
        nowcast_tstar_start=args.nowcast_tstar_start,
        nowcast_only_pre_event=args.nowcast_only_pre_event,
    )
