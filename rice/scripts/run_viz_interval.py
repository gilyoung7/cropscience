from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import torch

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest, default_out_root, ensure_output_dirs
from rice.scripts.common import make_loader, parse_seed_candidates, parse_tags, init_wandb_run, finish_wandb_run
from rice.scripts.run_eval import build_samples_for_run
from rice.scripts.run_event_train import (
    EventTransformer,
    build_nowcast_samples,
    build_tabular_from_samples,
    make_event_labels,
)
from rice.scripts.run_event_eval import (
    apply_temperature,
    best_tau_by_target,
    fit_temperature_grid,
)
from rice.src.dataset import (
    split_by_site,
    compute_norm_stats,
    IntervalEventDataset,
    split_seed_search_topk,
    log_split_fingerprint,
    build_stage2_nowcast_samples,
)
from rice.src.model import HazardTransformer
from rice.src.train_eval import (
    CTYPE_INTERVAL,
    hazard_to_pmf_cdf_logS,
    shortest_mass_interval_1d,
    quantile_from_cdf_1d,
    overlap_metrics,
)

WANDB_ENTITY_DEFAULT = "gilyoung7-seoul-national-university"
WANDB_PROJECT_DEFAULT = "agro-rice"


def resolve_ckpt_path(run: int, out_root: str, ckpt_path: str | None, fallback_name: str) -> Path:
    if ckpt_path:
        return Path(ckpt_path)
    return Path(out_root) / "ckpt" / fallback_name.format(run=run)


@torch.no_grad()
def predict_event_prob_event_model(model, loader, device) -> np.ndarray:
    model.eval()
    probs = []
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            X, _, _, ctype = batch
            X = X.to(device, non_blocking=True)
            y = (ctype.to(device, non_blocking=True) != 1).float()
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            X, y = batch
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True).float()
        else:
            raise ValueError("unexpected batch format in predict_event_prob_event_model")
        logits = model(X)
        p = torch.sigmoid(logits).detach().cpu().numpy()
        probs.append(p)
    if probs:
        return np.concatenate(probs)
    return np.asarray([], dtype=float)


@torch.no_grad()
def collect_interval_preds(
    model,
    loader,
    source_samples: list[dict],
    Tend: int,
    device,
    pi_method: str,
    max_samples: int,
):
    rows: list[dict] = []
    sample_idx = 0
    q_lo = 0.1
    q_hi = 0.9
    target_mass = 0.8

    model.eval()
    for X, L, R, ctype in loader:
        X = X.to(device, non_blocking=True)
        hazard = model(X)
        pmf, cdf, _ = hazard_to_pmf_cdf_logS(hazard)

        L_np = L.cpu().numpy().astype(int)
        R_np = R.cpu().numpy().astype(int)
        ctype_np = ctype.cpu().numpy().astype(int)
        cdf_np = cdf.cpu().numpy()
        pmf_np = pmf.cpu().numpy()

        for b in range(len(L_np)):
            if int(ctype_np[b]) != int(CTYPE_INTERVAL):
                continue
            if len(rows) >= max_samples:
                return rows

            sidx = sample_idx + b
            sample_meta = source_samples[sidx] if sidx < len(source_samples) else {}

            pmf_raw = pmf_np[b]
            # Enforce no event before t* (shift mass to t*+1..T)
            tstar_val = int(sample_meta.get("tstar", 0) or 0)
            if tstar_val > 0:
                pmf_raw = pmf_raw.copy()
                pmf_raw[:tstar_val] = 0.0  # pmf index 0 -> time 1
            total_mass = float(np.sum(pmf_raw))
            if total_mass > 0.0:
                pmf_cond = pmf_raw / total_mass
            else:
                pmf_cond = pmf_raw
            cdf_cond = np.cumsum(pmf_cond)

            if pi_method == "shortest":
                # use conditional pmf; already normalized -> normalize=False
                pL, pR, _ = shortest_mass_interval_1d(pmf_cond, target_mass=target_mass, Tend=Tend, normalize=False)
            elif pi_method == "quantile":
                pL = quantile_from_cdf_1d(cdf_cond, q_lo, Tend)
                pR = quantile_from_cdf_1d(cdf_cond, q_hi, Tend)
            else:
                raise ValueError(f"Unknown pi_method: {pi_method}. expected 'shortest' or 'quantile'")

            pL = max(1, min(int(pL), int(Tend)))
            pR = max(1, min(int(pR), int(Tend)))
            if pL > pR:
                pL, pR = pR, pL

            if total_mass > 0.0 and cdf_cond[-1] >= 0.5:
                p_point = int(np.searchsorted(cdf_cond, 0.5) + 1)
            else:
                p_point = int(Tend)

            true_L = int(L_np[b])
            true_R = int(R_np[b])
            iou, rec, prec = overlap_metrics(pL, pR, true_L, true_R)
            site = sample_meta.get("site_id")
            year = sample_meta.get("year")
            if site is not None and year is not None:
                sample_id = f"{site}-{int(year)}"
            else:
                sample_id = f"{int(sidx)}"

            rows.append(
                {
                    "sample_id": sample_id,
                    "true_L": true_L,
                    "true_R": true_R,
                    "pred_L": int(pL),
                    "pred_R": int(pR),
                    "pred_point": int(p_point),
                    "tstar": sample_meta.get("tstar"),
                    "iou": float(iou),
                }
            )
        sample_idx += len(L_np)
    return rows


def plot_interval_rows(rows: list[dict], Tend: int, title: str):
    import matplotlib.pyplot as plt

    if not rows:
        return None
    n = len(rows)
    fig_h = max(2.0, 1.1 * n)
    fig, axes = plt.subplots(n, 1, figsize=(10, fig_h), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, rows):
        ax.hlines(0, r["true_L"], r["true_R"], color="black", lw=6, alpha=0.25, label="true interval")
        ax.hlines(0, r["pred_L"], r["pred_R"], color="tab:blue", lw=3, label="pred interval")
        ax.plot(r["pred_point"], 0, marker="o", color="tab:blue", ms=5, label="pred point")
        alert = r.get("alert_tstar")
        if alert is not None:
            ax.axvline(int(alert), color="tab:red", lw=1.2, ls="--", label="early warning t*")
        ax.set_yticks([])
        ax.set_xlim(1, Tend)
        ax.set_title(f"{r['sample_id']} | IoU={r['iou']:.2f}")
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def pr_auc_binary(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(y_score, dtype=float)
    n_pos = int((y == 1).sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s)
    y_sorted = y[order]
    tp = np.cumsum(y_sorted == 1)
    fp = np.cumsum(y_sorted == 0)
    recall = tp / max(n_pos, 1)
    precision = tp / np.maximum(tp + fp, 1)
    recall = np.concatenate([[0.0], recall])
    precision = np.concatenate([[1.0], precision])
    return float(np.trapz(precision, recall))


def fp_rate_at_tau(y_true: np.ndarray, p: np.ndarray, tau: float) -> float:
    y = np.asarray(y_true, dtype=int)
    pred = (p >= float(tau)).astype(int)
    neg = (y == 0)
    if int(neg.sum()) == 0:
        return float("nan")
    fp = int(((pred == 1) & neg).sum())
    return float(fp / int(neg.sum()))


def recall_at_tau(y_true: np.ndarray, p: np.ndarray, tau: float) -> float:
    y = np.asarray(y_true, dtype=int)
    pred = (p >= float(tau)).astype(int)
    pos = (y == 1)
    if int(pos.sum()) == 0:
        return float("nan")
    tp = int(((pred == 1) & pos).sum())
    return float(tp / int(pos.sum()))


def compute_t_alert_start(
    y_true: np.ndarray,
    p_cal: np.ndarray,
    tstar: np.ndarray,
    tau: float,
    pr_auc_min: float = 0.70,
    fp_rate_max: float = 0.03,
    recall_min: float = 0.55,
    consecutive: int = 3,
) -> int | None:
    uniq = np.unique(tstar)
    if uniq.size == 0:
        return None
    cond = []
    for t in uniq:
        m = (tstar == t)
        y_t = y_true[m]
        p_t = p_cal[m]
        if y_t.size == 0:
            cond.append(False)
            continue
        pr_auc = pr_auc_binary(y_t, p_t)
        fp_rate = fp_rate_at_tau(y_t, p_t, tau)
        rec = recall_at_tau(y_t, p_t, tau)
        ok = (pr_auc >= pr_auc_min) and (fp_rate <= fp_rate_max) and (rec >= recall_min)
        cond.append(ok)
    cond = np.array(cond, dtype=bool)
    if cond.size < consecutive:
        return None
    for i in range(cond.size - consecutive + 1):
        if cond[i:i + consecutive].all():
            return int(uniq[i])
    return None


def build_alert_map(samples: list[dict], probs: np.ndarray, tau: float, t_alert_start: int | None) -> dict[str, int | None]:
    alert = {}
    for s, p in zip(samples, probs):
        tstar = int(s.get("tstar", -1))
        if t_alert_start is not None and tstar < int(t_alert_start):
            continue
        site = str(s.get("site_id", ""))
        year = int(s.get("year", -1))
        key = f"{site}-{year}"
        if p < float(tau):
            continue
        prev = alert.get(key)
        if prev is None or tstar < int(prev):
            alert[key] = tstar
    return alert


def main(
    pest: str,
    run: int,
    stage1_ckpt: str,
    stage2_ckpt: str,
    out_root: str,
    split_seed: int,
    seeds: list[int] | None,
    auto_split_seed: bool,
    seed_candidates_raw: str | None,
    target_test_interval: int | None,
    tol_test_interval: int | None,
    auto_split_topk: int,
    split_seed_from_topk_idx: int | None,
    split_seeds_json: str | None,
    split: str,
    topk: int,
    worstk: int,
    randomk: int,
    max_pool: int,
    tau_mode: str,
    tau_target_precision: float,
    tau_target_recall: float,
    use_wandb: bool,
    wandb_project: str | None,
    wandb_entity: str | None,
    wandb_group: str | None,
    wandb_run_name: str | None,
    wandb_tags: str | None,
    wandb_job_type: str | None,
):
    _, get_feature_cols = resolve_pest(pest)
    if not out_root:
        out_root = default_out_root(pest)
    ensure_output_dirs(out_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    wandb_run = init_wandb_run(
        use_wandb=use_wandb,
        project=wandb_project or WANDB_PROJECT_DEFAULT,
        entity=wandb_entity or WANDB_ENTITY_DEFAULT,
        run_name=wandb_run_name,
        group=wandb_group,
        job_type=wandb_job_type,
        tags=parse_tags(wandb_tags) + [f"pest:{pest}", "script:run_viz_interval"],
        config={
            "pest": pest,
            "run": run,
            "split_seed": int(split_seed),
            "split": split,
            "topk": int(topk),
            "worstk": int(worstk),
            "randomk": int(randomk),
            "max_pool": int(max_pool),
            "tau_mode": tau_mode,
            "tau_target_precision": float(tau_target_precision),
            "tau_target_recall": float(tau_target_recall),
        },
    )

    stage1_path = resolve_ckpt_path(run, out_root, stage1_ckpt, "event_classifier_run{run}.pt")
    stage2_path = resolve_ckpt_path(run, out_root, stage2_ckpt, "checkpoint_run{run}.pt")
    print(f"Using stage1 ckpt: {stage1_path}")
    print(f"Using stage2 ckpt: {stage2_path}")
    ckpt1 = torch.load(stage1_path, map_location="cpu")
    ckpt2 = torch.load(stage2_path, map_location="cpu")

    feature_cols, feature_names_eval, T, samples = build_samples_for_run(run, get_feature_cols)
    print(f"[features] n={len(feature_names_eval)} head={feature_names_eval[:5]} tail={feature_names_eval[-5:]}")

    if split_seeds_json is not None:
        from rice.scripts.run_eval import resolve_split_seeds_json_path, load_split_seed_from_topk
        split_seeds_json_path = resolve_split_seeds_json_path(out_root, split_seeds_json)
        split_seed, chosen_idx, chosen, _ = load_split_seed_from_topk(split_seeds_json_path, split_seed_from_topk_idx)
        train_s, val_s, test_s = split_by_site(samples, val_frac=0.1, test_frac=0.1, seed=split_seed)
        print(f"[split_seed_json] selected seed={split_seed} idx={chosen_idx} file={split_seeds_json_path}")
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

    # Stage1 nowcast samples (for alert t*)
    task_mode = str(ckpt1.get("task_mode", "season_complete"))
    nowcast_window = int(ckpt1.get("nowcast_window", 28))
    nowcast_stride = int(ckpt1.get("nowcast_stride", 7))
    nowcast_tstar_start = ckpt1.get("nowcast_tstar_start", None)
    nowcast_only_pre_event = int(ckpt1.get("nowcast_only_pre_event", 1))
    nowcast_event_time_proxy = str(ckpt1.get("nowcast_event_time_proxy", "r"))
    if task_mode == "nowcast":
        val_s1 = build_nowcast_samples(
            val_s,
            window=nowcast_window,
            stride=nowcast_stride,
            tstar_start=nowcast_tstar_start,
            only_pre_event=bool(nowcast_only_pre_event),
            event_time_proxy=nowcast_event_time_proxy,
        )
        test_s1 = build_nowcast_samples(
            test_s,
            window=nowcast_window,
            stride=nowcast_stride,
            tstar_start=nowcast_tstar_start,
            only_pre_event=bool(nowcast_only_pre_event),
            event_time_proxy=nowcast_event_time_proxy,
        )
    else:
        raise ValueError("Stage1 visualization expects nowcast task_mode")

    # Stage2 nowcast samples (for interval viz)
    stage2_nowcast = bool(ckpt2.get("stage2_nowcast", False))
    if stage2_nowcast:
        stage2_nowcast_window = int(ckpt2.get("stage2_nowcast_window", 56))
        stage2_nowcast_stride = int(ckpt2.get("stage2_nowcast_stride", 7))
        stage2_nowcast_tstar_start = ckpt2.get("stage2_nowcast_tstar_start", None)
        stage2_nowcast_only_pre_event = int(ckpt2.get("stage2_nowcast_only_pre_event", 1))
        stage2_nowcast_event_time_proxy = str(ckpt2.get("stage2_nowcast_event_time_proxy", "mid"))
        val_s2 = build_stage2_nowcast_samples(
            val_s,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
        )
        test_s2 = build_stage2_nowcast_samples(
            test_s,
            window=stage2_nowcast_window,
            stride=stage2_nowcast_stride,
            tstar_start=stage2_nowcast_tstar_start,
            only_pre_event=bool(stage2_nowcast_only_pre_event),
            event_time_proxy=stage2_nowcast_event_time_proxy,
        )
    else:
        val_s2 = val_s
        test_s2 = test_s

    log_split_fingerprint("viz_final", train_s, val_s, test_s)

    x_mean, x_std = compute_norm_stats(train_s)
    val_ds2 = IntervalEventDataset(val_s2, x_mean, x_std)
    test_ds2 = IntervalEventDataset(test_s2, x_mean, x_std)
    val_loader2 = make_loader(val_ds2, C.BATCH_EVAL, shuffle=False)
    test_loader2 = make_loader(test_ds2, C.BATCH_EVAL, shuffle=False)

    if split not in {"val", "test"}:
        raise ValueError("--split must be 'val' or 'test'")
    source_samples2 = val_s2 if split == "val" else test_s2
    loader2 = val_loader2 if split == "val" else test_loader2
    samples1 = val_s1 if split == "val" else test_s1

    # Stage1 labels for tau
    y_val = make_event_labels(val_s1)
    y_test = make_event_labels(test_s1)

    for d2 in ckpt2["trained_states"]:
        seed = int(d2["seed"])
        if seeds is not None and seed not in seeds:
            continue

        # Stage1: pick matching seed
        d1 = None
        for s in ckpt1["trained_states"]:
            if int(s["seed"]) == seed:
                d1 = s
                break
        if d1 is None:
            print(f"[seed {seed}] no stage1 state; skipping")
            continue

        # Stage1 model
        model_kind = str(ckpt1.get("event_model", "transformer"))
        if model_kind == "transformer":
            model1 = EventTransformer(
                d_in=int(val_ds2[0][0].shape[-1]),
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=2,
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
            ).to(device)
            model1.load_state_dict(d1["state_dict"])
            model1.eval()
            val_loader1 = make_loader(IntervalEventDataset(val_s1, x_mean, x_std), C.BATCH_EVAL, shuffle=False)
            test_loader1 = make_loader(IntervalEventDataset(test_s1, x_mean, x_std), C.BATCH_EVAL, shuffle=False)
            p_val_raw = predict_event_prob_event_model(model1, val_loader1, device=device)
            p_test_raw = predict_event_prob_event_model(model1, test_loader1, device=device)
        else:
            clf = d1.get("sk_model")
            if clf is None:
                raise ValueError("stage1 checkpoint missing sk_model")
            X_val_tab = build_tabular_from_samples(val_s1)
            X_test_tab = build_tabular_from_samples(test_s1)
            if hasattr(clf, "predict_proba"):
                p_val_raw = clf.predict_proba(X_val_tab)[:, 1]
                p_test_raw = clf.predict_proba(X_test_tab)[:, 1]
            else:
                p_val_raw = np.asarray(clf.predict(X_val_tab), dtype=float)
                p_test_raw = np.asarray(clf.predict(X_test_tab), dtype=float)

        t_best, _ = fit_temperature_grid(y_val, p_val_raw)
        p_val_cal = apply_temperature(p_val_raw, t_best)
        p_test_cal = apply_temperature(p_test_raw, t_best)

        tau, _val_f1, _val_prec, _val_rec = best_tau_by_target(
            y_val,
            p_val_cal,
            mode=tau_mode,
            target_precision=tau_target_precision,
            target_recall=tau_target_recall,
        )

        # Stage1 alerts: no t* condition (use first t* with p>=tau)
        t_alert_start = None
        alert_map = build_alert_map(
            samples1,
            p_val_cal if split == "val" else p_test_cal,
            tau,
            t_alert_start,
        )

        # Stage2 model
        model2 = HazardTransformer(
            d_in=int(val_ds2[0][0].shape[-1]),
            d_model=C.D_MODEL,
            nhead=C.N_HEAD,
            num_layers=C.N_LAYERS,
            dropout=C.DROPOUT,
            max_len=C.MAX_LEN,
        ).to(device)
        model2.load_state_dict(d2["state_dict"])
        model2.eval()

        rows = collect_interval_preds(
            model2,
            loader2,
            source_samples=source_samples2,
            Tend=T,
            device=device,
            pi_method=getattr(C, "PI_METHOD", "shortest"),
            max_samples=max_pool,
        )
        if not rows:
            print(f"[seed {seed}] no interval samples for visualization.")
            continue

        # Build lookup for stage2 rows by (site-year, tstar)
        row_map = {}
        for r in rows:
            tstar = r.get("tstar")
            if tstar is None:
                continue
            row_map[(r["sample_id"], int(tstar))] = r

        # Interval site-year denominator (unique)
        interval_site_year = {
            (s.get("site_id"), int(s.get("year")))
            for s in source_samples2
            if str(s.get("censor_type", "")) == "interval"
        }
        n_true = len(interval_site_year)

        # Attach alert t* and select stage2 prediction at same t*
        matched_rows = []
        tp = 0
        pred_pos = 0
        lead_times = []
        for sid, alert_t in alert_map.items():
            if alert_t is None:
                continue
            pred_row = row_map.get((sid, int(alert_t)))
            if pred_row is None:
                continue
            pred_row = dict(pred_row)
            pred_row["alert_tstar"] = int(alert_t)
            matched_rows.append(pred_row)
            pred_pos += 1
            hit = (min(pred_row["pred_R"], pred_row["true_R"]) - max(pred_row["pred_L"], pred_row["true_L"])) > 0
            if hit:
                tp += 1
            try:
                lt = int(pred_row["true_L"]) - int(alert_t)
                lead_times.append(float(lt))
            except Exception:
                pass

        precision = tp / pred_pos if pred_pos > 0 else 0.0
        recall = tp / n_true if n_true > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        lead_time_mean = float(np.mean(lead_times)) if lead_times else float("nan")
        lead_time_median = float(np.median(lead_times)) if lead_times else float("nan")

        rows_sorted = sorted(matched_rows, key=lambda r: r["iou"], reverse=True)
        top_rows = rows_sorted[:topk]
        worst_rows = list(reversed(rows_sorted[-worstk:])) if rows_sorted else []
        if rows_sorted:
            rng = np.random.default_rng(int(seed))
            rand_rows = rng.choice(rows_sorted, size=min(randomk, len(rows_sorted)), replace=False).tolist()
        else:
            rand_rows = []

        fig_top = plot_interval_rows(top_rows, Tend=T, title=f"{split.upper()} Top-{len(top_rows)} (seed {seed})")
        fig_worst = plot_interval_rows(worst_rows, Tend=T, title=f"{split.upper()} Worst-{len(worst_rows)} (seed {seed})")
        fig_rand = plot_interval_rows(rand_rows, Tend=T, title=f"{split.upper()} Random-{len(rand_rows)} (seed {seed})")

        if wandb_run is not None:
            import wandb

            log_payload = {"seed": int(seed), "split": split}
            if fig_top is not None:
                log_payload[f"viz/{split}_top_seed{seed}"] = wandb.Image(fig_top)
            if fig_worst is not None:
                log_payload[f"viz/{split}_worst_seed{seed}"] = wandb.Image(fig_worst)
            if fig_rand is not None:
                log_payload[f"viz/{split}_random_seed{seed}"] = wandb.Image(fig_rand)

            table_rows = []
            for r in (top_rows + worst_rows + rand_rows):
                table_rows.append(
                    [
                        r["sample_id"],
                        r["true_L"],
                        r["true_R"],
                        r["pred_L"],
                        r["pred_R"],
                        r["pred_point"],
                        r.get("alert_tstar"),
                        r["iou"],
                    ]
                )
            log_payload[f"viz/{split}_samples_seed{seed}"] = wandb.Table(
                columns=["sample_id", "true_L", "true_R", "pred_L", "pred_R", "pred_point", "alert_tstar", "iou"],
                data=table_rows,
            )
            log_payload["final/t_alert_start"] = None
            log_payload["final/tau"] = float(tau)
            log_payload["final/precision"] = float(precision)
            log_payload["final/recall"] = float(recall)
            log_payload["final/f1"] = float(f1)
            log_payload["final/pred_pos_rate"] = float(pred_pos / max(n_true, 1))
            log_payload["final/lead_time_mean"] = float(lead_time_mean)
            log_payload["final/lead_time_median"] = float(lead_time_median)
            wandb_run.log(log_payload)

        print(f"[seed {seed}] logged {len(top_rows)}/{len(worst_rows)}/{len(rand_rows)} rows")

        # Save final metrics locally
        out_dir = Path(out_root) / "final"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"final_viz_{pest}_run{run}_{split}_seed{seed}.csv"
        with open(out_path, "w", encoding="utf-8") as f:
            f.write("seed,split,precision,recall,f1,pred_pos_rate,lead_time_mean,lead_time_median,t_alert_start,tau\n")
            t_alert_str = "" if t_alert_start is None else str(int(t_alert_start))
            f.write(
                f"{seed},{split},{precision:.6f},{recall:.6f},{f1:.6f},"
                f"{(pred_pos / max(n_true, 1)):.6f},{lead_time_mean:.6f},{lead_time_median:.6f},"
                f"{t_alert_str},{tau:.6f}\n"
            )
        print("saved:", out_path)

        # Save sample-level table locally (same rows as W&B table)
        sample_out = out_dir / f"final_viz_{pest}_run{run}_{split}_seed{seed}_samples.csv"
        with open(sample_out, "w", encoding="utf-8") as f:
            f.write("sample_id,true_L,true_R,pred_L,pred_R,pred_point,alert_tstar,iou\n")
            for r in (top_rows + worst_rows + rand_rows):
                alert = "" if r.get("alert_tstar") is None else str(int(r.get("alert_tstar")))
                f.write(
                    f"{r['sample_id']},{r['true_L']},{r['true_R']},{r['pred_L']},{r['pred_R']},"
                    f"{r['pred_point']},{alert},{r['iou']:.6f}\n"
                )
        print("saved:", sample_out)

    finish_wandb_run(wandb_run)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, required=True)
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--stage1_ckpt", type=str, required=True)
    p.add_argument("--stage2_ckpt", type=str, required=True)
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
    p.add_argument("--split", type=str, default="val", choices=["val", "test"])
    p.add_argument("--topk", type=int, default=5)
    p.add_argument("--worstk", type=int, default=5)
    p.add_argument("--randomk", type=int, default=5)
    p.add_argument("--max_pool", type=int, default=300)
    p.add_argument("--tau_mode", type=str, default="f1", choices=["f1", "precision_target", "recall_target"])
    p.add_argument("--tau_target_precision", type=float, default=0.6)
    p.add_argument("--tau_target_recall", type=float, default=0.6)
    p.add_argument("--use_wandb", action="store_true")
    p.add_argument("--wandb_project", type=str, default=None)
    p.add_argument("--wandb_entity", type=str, default=None)
    p.add_argument("--wandb_group", type=str, default=None)
    p.add_argument("--wandb_run_name", type=str, default=None)
    p.add_argument("--wandb_tags", type=str, default=None)
    p.add_argument("--wandb_job_type", type=str, default=None)
    args = p.parse_args()
    main(
        pest=args.pest,
        run=args.run,
        stage1_ckpt=args.stage1_ckpt,
        stage2_ckpt=args.stage2_ckpt,
        out_root=args.out_root,
        split_seed=args.split_seed,
        seeds=args.seeds,
        auto_split_seed=args.auto_split_seed,
        seed_candidates_raw=args.seed_candidates,
        target_test_interval=args.target_test_interval,
        tol_test_interval=args.tol_test_interval,
        auto_split_topk=args.auto_split_topk,
        split_seed_from_topk_idx=args.split_seed_from_topk_idx,
        split_seeds_json=args.split_seeds_json,
        split=args.split,
        topk=args.topk,
        worstk=args.worstk,
        randomk=args.randomk,
        max_pool=args.max_pool,
        tau_mode=args.tau_mode,
        tau_target_precision=args.tau_target_precision,
        tau_target_recall=args.tau_target_recall,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        wandb_job_type=args.wandb_job_type,
    )
