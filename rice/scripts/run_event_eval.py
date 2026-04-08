from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest, default_stage1_out_root, ensure_output_dirs
from rice.scripts.common import make_loader, parse_seed_candidates, parse_tags, init_wandb_run, finish_wandb_run
from rice.scripts.run_eval import build_samples_for_run
from rice.src.dataset import (
    split_by_site,
    compute_norm_stats,
    IntervalEventDataset,
    split_seed_search_topk,
    log_split_fingerprint,
)
from rice.scripts.run_event_train import (
    EventTransformer,
    resolve_split_seeds_json_path,
    load_split_seed_from_topk,
    build_tabular_from_samples,
    EventBinaryDataset,
    build_nowcast_samples,
    make_event_labels,
)

WANDB_ENTITY_DEFAULT = "gilyoung7-seoul-national-university"
WANDB_PROJECT_DEFAULT = "agro-rice"
ALERT_ROW_COLUMNS = ["split", "seed", "sample_id", "site", "year", "true_L", "true_R", "alert_tstar"]
ALERT_OUTPUT_COLUMNS = ALERT_ROW_COLUMNS + ["is_event", "alerted", "lead_time"]
ALERT_SUMMARY_COLUMNS = [
    "split",
    "seed",
    "n_site_year",
    "n_event",
    "n_alerted",
    "alert_rate",
    "true_event_alert_rate",
    "false_alert_rate",
    "specificity",
    "alert_precision",
    "alert_recall",
    "alert_f1",
    "lead_time_mean",
    "lead_time_median",
    "window_precision",
    "window_recall",
    "window_f1",
    "early_alert_rate",
    "non_event_false_alert_rate",
    "t_alert_start",
    "tau_selected",
]
PROB_OUTPUT_COLUMNS = [
    "split",
    "seed",
    "sample_id",
    "site",
    "year",
    "tstar",
    "y_event",
    "true_L",
    "true_R",
    "p_raw",
    "p_cal",
]
GATE_POLICY_SUMMARY_COLUMNS = [
    "split",
    "seed",
    "gate_policy_name",
    "gate_consecutive_k",
    "gate_smooth_window",
    "gate_use_t_alert_start",
    "tau_selected",
    "tau_alert_fallback_mode",
    "alert_rate",
    "alert_precision",
    "alert_recall",
    "alert_f1",
    "false_alert_rate",
    "specificity",
    "lead_time_mean",
    "lead_time_median",
    "window_precision",
    "window_recall",
    "window_f1",
    "early_alert_rate",
    "non_event_false_alert_rate",
    "t_alert_start",
    "selected",
]


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
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    return float(np.mean((p - y) ** 2))


def binary_nll(y_true: np.ndarray, y_prob: np.ndarray, eps: float = 1e-8) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


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


def apply_temperature(p: np.ndarray, temperature: float, eps: float = 1e-8) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    z = logit / float(temperature)
    return 1.0 / (1.0 + np.exp(-z))


def fit_temperature_grid(y_true: np.ndarray, p_val: np.ndarray) -> tuple[float, float]:
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
    taus = np.linspace(0.05, 0.95, 181)
    y = np.asarray(y_true, dtype=int)
    best = (0.5, -1.0, 0.0, 0.0)
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


def best_tau_by_target(
    y_true: np.ndarray,
    p: np.ndarray,
    mode: str,
    target_precision: float,
    target_recall: float,
) -> tuple[float, float, float, float]:
    taus = np.linspace(0.05, 0.95, 181)
    rows = []
    for tau in taus:
        prec, rec, f1 = pr_at_tau(y_true, p, float(tau))
        rows.append((float(tau), float(f1), float(prec), float(rec)))

    if mode == "f1":
        return max(rows, key=lambda x: x[1])

    if mode == "precision_target":
        cand = [r for r in rows if r[2] >= float(target_precision)]
        if cand:
            # among feasible thresholds, maximize recall; tie-break by higher precision
            return max(cand, key=lambda x: (x[3], x[2], -x[0]))
        return max(rows, key=lambda x: (x[2], x[3], x[1]))

    if mode == "recall_target":
        cand = [r for r in rows if r[3] >= float(target_recall)]
        if cand:
            # among feasible thresholds, maximize precision; tie-break by larger tau
            return max(cand, key=lambda x: (x[2], x[3], x[0]))
        return max(rows, key=lambda x: (x[3], x[2], x[1]))

    raise ValueError(f"unsupported tau mode: {mode}")


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


def metrics_by_tstar(
    y_true: np.ndarray,
    p_raw: np.ndarray,
    p_cal: np.ndarray,
    tau: float,
    tstar: np.ndarray,
    split_name: str,
    seed: int,
) -> list[dict]:
    rows = []
    for t in sorted(np.unique(tstar)):
        m = (tstar == t)
        if int(m.sum()) == 0:
            continue
        y = y_true[m]
        pr = p_raw[m]
        pc = p_cal[m]
        prec, rec, f1 = pr_at_tau(y, pc, tau)
        rows.append(
            {
                "seed": int(seed),
                "split": split_name,
                "tstar": int(t),
                "n": int(m.sum()),
                "event_rate": float(np.mean(y)),
                "auc_raw": float(auc_roc_binary(y, pr)),
                "auc_cal": float(auc_roc_binary(y, pc)),
                "pr_auc_raw": float(pr_auc_binary(y, pr)),
                "pr_auc_cal": float(pr_auc_binary(y, pc)),
                "brier_raw": float(brier_score(y, pr)),
                "brier_cal": float(brier_score(y, pc)),
                "nll_raw": float(binary_nll(y, pr)),
                "nll_cal": float(binary_nll(y, pc)),
                "prec@tau": float(prec),
                "rec@tau": float(rec),
                "f1@tau": float(f1),
                "fp_rate@tau": float(fp_rate_at_tau(y, pc, tau)),
            }
        )
    return rows


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


def build_alert_rows(
    samples: list[dict],
    probs: np.ndarray,
    tau: float,
    split_name: str,
    seed: int,
    t_alert_start: int | None,
    consecutive_k: int = 1,
    smooth_window: int = 1,
) -> list[list]:
    """
    Build per-(site,year) rows with true interval [L,R] and earliest alert tstar.
    """
    rows_by_key = {}
    for s, p in zip(samples, probs):
        site = str(s.get("site_id", ""))
        year = int(s.get("year", -1))
        key = f"{site}-{year}"
        rec = rows_by_key.get(key)
        if rec is None:
            rec = {
                "sample_id": key,
                "site": site,
                "year": year,
                "true_L": s.get("L"),
                "true_R": s.get("R"),
                "items": [],
            }
            rows_by_key[key] = rec
        rec["items"].append((int(s.get("tstar", -1)), float(p)))

    rows = []
    for rec in rows_by_key.values():
        items = sorted(rec["items"], key=lambda x: x[0])
        tstars = np.asarray([t for t, _ in items], dtype=int)
        pvals = np.asarray([p for _, p in items], dtype=float)
        if int(smooth_window) > 1 and pvals.size > 0:
            smooth = np.empty_like(pvals, dtype=float)
            for i in range(pvals.size):
                lo = max(0, i - int(smooth_window) + 1)
                smooth[i] = float(np.mean(pvals[lo:i + 1]))
        else:
            smooth = pvals

        alert_tstar = None
        streak = 0
        for tstar, p_now in zip(tstars, smooth):
            if t_alert_start is not None and int(tstar) < int(t_alert_start):
                streak = 0
                continue
            if float(p_now) >= float(tau):
                streak += 1
                if streak >= int(consecutive_k):
                    alert_tstar = int(tstar)
                    break
            else:
                streak = 0
        rows.append(
            [
                split_name,
                int(seed),
                rec["sample_id"],
                rec["site"],
                rec["year"],
                rec["true_L"],
                rec["true_R"],
                alert_tstar,
            ]
        )
    return rows


def build_gate_policy_name(consecutive_k: int, smooth_window: int, use_t_alert_start: bool) -> str:
    gate_tag = "gate" if bool(use_t_alert_start) else "nogate"
    return f"k{int(consecutive_k)}_ma{int(smooth_window)}_{gate_tag}"


def iter_gate_policy_candidates(
    consecutive_ks: list[int],
    smooth_windows: list[int],
    use_t_alert_start_options: list[bool],
) -> list[dict]:
    out = []
    for k in consecutive_ks:
        for ma in smooth_windows:
            for use_gate in use_t_alert_start_options:
                out.append(
                    {
                        "gate_policy_name": build_gate_policy_name(k, ma, use_gate),
                        "gate_consecutive_k": int(k),
                        "gate_smooth_window": int(ma),
                        "gate_use_t_alert_start": bool(use_gate),
                    }
                )
    return out


def alert_rows_to_frame(rows: list[list]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=ALERT_ROW_COLUMNS)
    if df.empty:
        return pd.DataFrame(columns=ALERT_OUTPUT_COLUMNS)

    for col in ("seed", "year", "true_L", "true_R", "alert_tstar"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["is_event"] = (df["true_L"].notna() & df["true_R"].notna()).astype(int)
    df["alerted"] = df["alert_tstar"].notna().astype(int)
    df["lead_time"] = np.nan
    m = (df["is_event"] == 1) & (df["alerted"] == 1)
    if bool(m.any()):
        df.loc[m, "lead_time"] = df.loc[m, "true_L"] - df.loc[m, "alert_tstar"]

    for col in ("seed", "year", "true_L", "true_R", "alert_tstar"):
        df[col] = df[col].astype("Int64")
    return df[ALERT_OUTPUT_COLUMNS]


def summarize_alert_rows(
    rows: list[list],
    split_name: str,
    seed: int,
    t_alert_start: int | None,
    tau_selected: float,
) -> dict:
    df = alert_rows_to_frame(rows)
    n_site_year = int(len(df))
    n_event = int(df["is_event"].sum()) if not df.empty else 0
    n_alerted = int(df["alerted"].sum()) if not df.empty else 0
    n_non_event = max(n_site_year - n_event, 0)
    n_true_alert = int(((df["is_event"] == 1) & (df["alerted"] == 1)).sum()) if not df.empty else 0
    n_false_alert = int(((df["is_event"] == 0) & (df["alerted"] == 1)).sum()) if not df.empty else 0

    alert_precision = n_true_alert / n_alerted if n_alerted > 0 else 0.0
    alert_recall = n_true_alert / n_event if n_event > 0 else 0.0
    alert_f1 = (
        (2 * alert_precision * alert_recall) / (alert_precision + alert_recall)
        if (alert_precision + alert_recall) > 0
        else 0.0
    )
    alert_rate = n_alerted / n_site_year if n_site_year > 0 else 0.0
    true_event_alert_rate = n_true_alert / n_event if n_event > 0 else 0.0
    false_alert_rate = n_false_alert / n_non_event if n_non_event > 0 else float("nan")
    specificity = 1.0 - false_alert_rate if np.isfinite(false_alert_rate) else float("nan")

    lead = df.loc[(df["is_event"] == 1) & (df["alerted"] == 1), "lead_time"].to_numpy(dtype=float)
    lead = lead[np.isfinite(lead)]
    lead_time_mean = float(np.mean(lead)) if lead.size > 0 else float("nan")
    lead_time_median = float(np.median(lead)) if lead.size > 0 else float("nan")

    return {
        "split": split_name,
        "seed": int(seed),
        "n_site_year": int(n_site_year),
        "n_event": int(n_event),
        "n_alerted": int(n_alerted),
        "alert_rate": float(alert_rate),
        "true_event_alert_rate": float(true_event_alert_rate),
        "false_alert_rate": float(false_alert_rate),
        "specificity": float(specificity),
        "alert_precision": float(alert_precision),
        "alert_recall": float(alert_recall),
        "alert_f1": float(alert_f1),
        "lead_time_mean": float(lead_time_mean),
        "lead_time_median": float(lead_time_median),
        "window_precision": float("nan"),
        "window_recall": float("nan"),
        "window_f1": float("nan"),
        "early_alert_rate": float("nan"),
        "non_event_false_alert_rate": float("nan"),
        "t_alert_start": None if t_alert_start is None else int(t_alert_start),
        "tau_selected": float(tau_selected),
    }


def summarize_within_h_alert_rows(
    rows: list[list],
    samples: list[dict],
    split_name: str,
    seed: int,
    t_alert_start: int | None,
    tau_selected: float,
    label_horizon: int,
) -> dict:
    summary = summarize_alert_rows(
        rows,
        split_name=split_name,
        seed=seed,
        t_alert_start=t_alert_start,
        tau_selected=tau_selected,
    )
    if int(label_horizon) <= 0:
        raise ValueError("label_horizon must be >= 1")

    event_time_by_id = {}
    for s in samples:
        sample_id = f"{str(s.get('site_id', ''))}-{int(s.get('year', -1))}"
        if sample_id not in event_time_by_id:
            event_time_by_id[sample_id] = s.get("event_time")

    df = alert_rows_to_frame(rows)
    if df.empty:
        summary.update(
            {
                "window_precision": 0.0,
                "window_recall": 0.0,
                "window_f1": 0.0,
                "early_alert_rate": float("nan"),
                "non_event_false_alert_rate": float("nan"),
            }
        )
        return summary

    n_event = int(df["is_event"].sum())
    n_non_event = int((df["is_event"] == 0).sum())
    n_window_hit = 0
    n_early_alert = 0
    n_non_event_false_alert = 0

    for row in df.itertuples(index=False):
        sample_id = str(row.sample_id)
        is_event = int(row.is_event)
        alerted = int(row.alerted)
        alert_tstar = row.alert_tstar
        event_time = event_time_by_id.get(sample_id)

        if is_event == 0:
            if alerted == 1:
                n_non_event_false_alert += 1
            continue

        if event_time is None or alerted == 0 or pd.isna(alert_tstar):
            continue

        event_time = int(event_time)
        alert_tstar = int(alert_tstar)
        window_start = int(event_time - int(label_horizon))
        if window_start <= alert_tstar < event_time:
            n_window_hit += 1
        elif alert_tstar < window_start:
            n_early_alert += 1

    window_recall = n_window_hit / n_event if n_event > 0 else 0.0
    early_alert_rate = n_early_alert / n_event if n_event > 0 else float("nan")
    non_event_false_alert_rate = (
        n_non_event_false_alert / n_non_event if n_non_event > 0 else float("nan")
    )
    window_alert_total = n_window_hit + n_early_alert + n_non_event_false_alert
    window_precision = n_window_hit / window_alert_total if window_alert_total > 0 else 0.0
    window_f1 = (
        (2 * window_precision * window_recall) / (window_precision + window_recall)
        if (window_precision + window_recall) > 0
        else 0.0
    )
    summary.update(
        {
            "window_precision": float(window_precision),
            "window_recall": float(window_recall),
            "window_f1": float(window_f1),
            "early_alert_rate": float(early_alert_rate),
            "non_event_false_alert_rate": float(non_event_false_alert_rate),
        }
    )
    return summary


def select_best_alert_metric_row(
    rows: list[dict],
    *,
    target_recall: float | None,
    max_false_alert_rate: float | None,
    policy: str,
) -> dict:
    recall_feasible = rows
    if target_recall is not None:
        recall_feasible = [r for r in rows if float(r["alert_recall"]) >= float(target_recall)]

    feasible = recall_feasible
    if max_false_alert_rate is not None:
        feasible = [
            r
            for r in recall_feasible
            if np.isfinite(r["false_alert_rate"]) and float(r["false_alert_rate"]) <= float(max_false_alert_rate)
        ]

    fallback_mode = "strict"
    if not feasible:
        if recall_feasible:
            feasible = recall_feasible
            fallback_mode = "recall_only"
        else:
            feasible = rows
            fallback_mode = "best_available"

    def score_key(r: dict) -> tuple[float, float, float, float]:
        false_alert_rate = -float(r["false_alert_rate"]) if np.isfinite(r["false_alert_rate"]) else float("-inf")
        specificity = float(r["specificity"]) if np.isfinite(r["specificity"]) else float("-inf")
        if fallback_mode == "best_available":
            if policy == "min_false_alert_rate":
                return (
                    float(r["alert_recall"]),
                    false_alert_rate,
                    float(r["alert_f1"]),
                    float(r.get("tau_selected", 0.0)),
                )
            if policy == "false_alert_rate_then_f1":
                return (
                    float(r["alert_recall"]),
                    false_alert_rate,
                    float(r["alert_f1"]),
                    float(r.get("tau_selected", 0.0)),
                )
            return (
                float(r["alert_recall"]),
                float(r["alert_f1"]),
                false_alert_rate,
                float(r.get("tau_selected", 0.0)),
            )
        if policy == "min_false_alert_rate":
            return (
                false_alert_rate,
                float(r["alert_f1"]),
                float(r["alert_precision"]),
                float(r.get("tau_selected", 0.0)),
            )
        if policy == "false_alert_rate_then_f1":
            return (
                false_alert_rate,
                float(r["alert_f1"]),
                float(r["alert_precision"]),
                float(r.get("tau_selected", 0.0)),
            )
        if policy == "f1_then_false_alert_rate":
            return (
                float(r["alert_f1"]),
                false_alert_rate,
                float(r["alert_precision"]),
                float(r.get("tau_selected", 0.0)),
            )
        return (
            float(r["alert_f1"]),
            specificity,
            float(r["alert_precision"]),
            float(r.get("tau_selected", 0.0)),
        )

    best = dict(max(feasible, key=score_key))
    best["tau_alert_fallback_mode"] = fallback_mode
    return best


def select_best_within_h_metric_row(
    rows: list[dict],
    *,
    target_window_recall: float | None,
) -> dict:
    feasible = rows
    if target_window_recall is not None:
        feasible = [r for r in rows if float(r["window_recall"]) >= float(target_window_recall)]

    fallback_mode = "strict"
    if not feasible:
        feasible = rows
        fallback_mode = "best_available"

    def key_strict(r: dict) -> tuple[float, float, float, float]:
        early = -float(r["early_alert_rate"]) if np.isfinite(r["early_alert_rate"]) else float("-inf")
        non_event_far = (
            -float(r["non_event_false_alert_rate"])
            if np.isfinite(r["non_event_false_alert_rate"])
            else float("-inf")
        )
        return (
            early,
            non_event_far,
            float(r["window_f1"]),
            float(r.get("tau_selected", 0.0)),
        )

    def key_fallback(r: dict) -> tuple[float, float, float, float]:
        early = -float(r["early_alert_rate"]) if np.isfinite(r["early_alert_rate"]) else float("-inf")
        non_event_far = (
            -float(r["non_event_false_alert_rate"])
            if np.isfinite(r["non_event_false_alert_rate"])
            else float("-inf")
        )
        return (
            float(r["window_recall"]),
            early,
            non_event_far,
            float(r["window_f1"]),
        )

    best = dict(max(feasible, key=(key_strict if fallback_mode == "strict" else key_fallback)))
    best["tau_alert_fallback_mode"] = fallback_mode
    return best


def best_tau_by_alert_site_year(
    samples: list[dict],
    probs: np.ndarray,
    *,
    split_name: str,
    seed: int,
    y_true: np.ndarray,
    tstar: np.ndarray,
    gate_consecutive_k: int,
    gate_smooth_window: int,
    gate_use_t_alert_start: bool,
    target_recall: float | None,
    max_false_alert_rate: float | None,
    policy: str,
) -> dict:
    taus = np.linspace(0.05, 0.95, 181)
    rows = []
    for tau in taus:
        t_alert_start = None
        if gate_use_t_alert_start:
            t_alert_start = compute_t_alert_start(y_true, probs, tstar, float(tau))
        alert_rows = build_alert_rows(
            samples,
            probs,
            float(tau),
            split_name=split_name,
            seed=seed,
            t_alert_start=t_alert_start,
            consecutive_k=gate_consecutive_k,
            smooth_window=gate_smooth_window,
        )
        summary = summarize_alert_rows(
            alert_rows,
            split_name=split_name,
            seed=seed,
            t_alert_start=t_alert_start,
            tau_selected=float(tau),
        )
        summary["gate_consecutive_k"] = int(gate_consecutive_k)
        summary["gate_smooth_window"] = int(gate_smooth_window)
        summary["gate_use_t_alert_start"] = bool(gate_use_t_alert_start)
        summary["gate_policy_name"] = build_gate_policy_name(gate_consecutive_k, gate_smooth_window, gate_use_t_alert_start)
        rows.append(summary)

    return select_best_alert_metric_row(
        rows,
        target_recall=target_recall,
        max_false_alert_rate=max_false_alert_rate,
        policy=policy,
    )


def best_tau_by_within_h_alert_window(
    samples: list[dict],
    probs: np.ndarray,
    *,
    split_name: str,
    seed: int,
    y_true: np.ndarray,
    tstar: np.ndarray,
    gate_consecutive_k: int,
    gate_smooth_window: int,
    gate_use_t_alert_start: bool,
    target_window_recall: float | None,
    label_horizon: int,
) -> dict:
    taus = np.linspace(0.05, 0.95, 181)
    rows = []
    for tau in taus:
        t_alert_start = None
        if gate_use_t_alert_start:
            t_alert_start = compute_t_alert_start(y_true, probs, tstar, float(tau))
        alert_rows = build_alert_rows(
            samples,
            probs,
            float(tau),
            split_name=split_name,
            seed=seed,
            t_alert_start=t_alert_start,
            consecutive_k=gate_consecutive_k,
            smooth_window=gate_smooth_window,
        )
        summary = summarize_within_h_alert_rows(
            alert_rows,
            samples=samples,
            split_name=split_name,
            seed=seed,
            t_alert_start=t_alert_start,
            tau_selected=float(tau),
            label_horizon=int(label_horizon),
        )
        summary["gate_consecutive_k"] = int(gate_consecutive_k)
        summary["gate_smooth_window"] = int(gate_smooth_window)
        summary["gate_use_t_alert_start"] = bool(gate_use_t_alert_start)
        summary["gate_policy_name"] = build_gate_policy_name(gate_consecutive_k, gate_smooth_window, gate_use_t_alert_start)
        rows.append(summary)

    return select_best_within_h_metric_row(
        rows,
        target_window_recall=target_window_recall,
    )


def plot_alert_rows(rows: list[list], title: str, Tend: int):
    import matplotlib.pyplot as plt

    if not rows:
        return None
    n = min(len(rows), 25)
    rows = rows[:n]
    fig_h = max(2.0, 1.1 * n)
    fig, axes = plt.subplots(n, 1, figsize=(10, fig_h), sharex=True)
    if n == 1:
        axes = [axes]
    for ax, r in zip(axes, rows):
        # columns: split, seed, sample_id, site, year, true_L, true_R, alert_tstar
        true_L, true_R, alert = r[5], r[6], r[7]
        ax.hlines(0, true_L, true_R, color="black", lw=6, alpha=0.25, label="true interval")
        if alert is not None:
            ax.axvline(int(alert), color="tab:red", lw=1.2, ls="--", label="early warning t*")
        ax.set_yticks([])
        ax.set_xlim(1, Tend)
        ax.set_title(str(r[2]))
    fig.suptitle(title)
    fig.tight_layout()
    return fig


@torch.no_grad()
def predict_event_prob(model, loader, device):
    model.eval()
    out = []
    for batch in loader:
        if isinstance(batch, (tuple, list)) and len(batch) == 4:
            X = batch[0]
        elif isinstance(batch, (tuple, list)) and len(batch) == 2:
            X = batch[0]
        else:
            raise ValueError("unexpected batch format in predict_event_prob")
        X = X.to(device, non_blocking=True)
        logits = model(X)
        p = torch.sigmoid(logits)
        out.append(p.cpu().numpy())
    return np.concatenate(out, axis=0) if out else np.zeros((0,), dtype=float)


def resolve_ckpt_path(run: int, out_root: str, ckpt_path: str | None) -> Path:
    if ckpt_path:
        return Path(ckpt_path)
    return Path(out_root) / "ckpt" / f"event_classifier_run{run}.pt"


def resolve_out_csv(run: int, out_root: str, out_csv: str | None) -> str:
    if out_csv:
        Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
        return out_csv
    out_dir = Path(out_root) / "eval"
    out_dir.mkdir(parents=True, exist_ok=True)
    return str(out_dir / f"event_eval_run{run}.csv")


def save_by_tstar_rows(out_csv_resolved: str, tstar_rows: list[dict]) -> str | None:
    if not tstar_rows:
        return None
    tstar_df = pd.DataFrame(tstar_rows).sort_values(["split", "tstar"]).reset_index(drop=True)
    tstar_out = str(Path(out_csv_resolved).with_name(Path(out_csv_resolved).stem + "_by_tstar.csv"))
    tstar_df.to_csv(tstar_out, index=False)
    print("saved:", tstar_out)
    return tstar_out


def save_alert_outputs(out_csv_resolved: str, alert_rows: list[list], alert_summary_rows: list[dict]) -> tuple[str | None, str | None]:
    alerts_out = None
    alerts_summary_out = None

    if alert_rows:
        alerts_df = alert_rows_to_frame(alert_rows)
        alerts_out = str(Path(out_csv_resolved).with_name(Path(out_csv_resolved).stem + "_alerts.csv"))
        alerts_df.to_csv(alerts_out, index=False)
        print("saved:", alerts_out)

    if alert_summary_rows:
        alert_summary_df = pd.DataFrame(alert_summary_rows, columns=ALERT_SUMMARY_COLUMNS).sort_values(
            ["split", "seed"]
        ).reset_index(drop=True)
        alerts_summary_out = str(Path(out_csv_resolved).with_name(Path(out_csv_resolved).stem + "_alerts_summary.csv"))
        alert_summary_df.to_csv(alerts_summary_out, index=False)
        print("saved:", alerts_summary_out)

    return alerts_out, alerts_summary_out


def build_probability_rows(
    samples: list[dict],
    p_raw: np.ndarray,
    p_cal: np.ndarray,
    *,
    split_name: str,
    seed: int,
) -> list[dict]:
    rows = []
    for s, pr, pc in zip(samples, p_raw, p_cal):
        site = str(s.get("site_id", ""))
        year = int(s.get("year", -1))
        tstar = s.get("tstar")
        sample_id = f"{site}-{year}"
        rows.append(
            {
                "split": split_name,
                "seed": int(seed),
                "sample_id": sample_id,
                "site": site,
                "year": year,
                "tstar": None if tstar is None else int(tstar),
                "y_event": int(s.get("y_event", 0)),
                "true_L": s.get("L"),
                "true_R": s.get("R"),
                "p_raw": float(pr),
                "p_cal": float(pc),
            }
        )
    return rows


def save_probability_outputs(out_csv_resolved: str, prob_rows: list[dict]) -> str | None:
    if not prob_rows:
        return None
    probs_df = pd.DataFrame(prob_rows, columns=PROB_OUTPUT_COLUMNS)
    out_path = str(Path(out_csv_resolved).with_name(Path(out_csv_resolved).stem + "_probs.csv"))
    probs_df.to_csv(out_path, index=False)
    print("saved:", out_path)
    return out_path


def save_gate_policy_outputs(out_csv_resolved: str, gate_policy_rows: list[dict]) -> str | None:
    if not gate_policy_rows:
        return None
    gate_policy_df = pd.DataFrame(gate_policy_rows, columns=GATE_POLICY_SUMMARY_COLUMNS).sort_values(
        ["split", "seed", "selected", "gate_policy_name"],
        ascending=[True, True, False, True],
    ).reset_index(drop=True)
    out_path = str(Path(out_csv_resolved).with_name(Path(out_csv_resolved).stem + "_gate_policy_val.csv"))
    gate_policy_df.to_csv(out_path, index=False)
    print("saved:", out_path)
    return out_path


def save_meta_json(out_csv_resolved: str, payload: dict) -> str:
    out_json_resolved = str(Path(out_csv_resolved).with_suffix(".meta.json"))
    with open(out_json_resolved, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print("saved:", out_json_resolved)
    return out_json_resolved


def main(
    pest: str,
    ckpt_path: str | None,
    run: int,
    out_csv: str | None,
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
    gate_policy_search: bool,
    gate_consecutive_k: int,
    gate_smooth_window: int,
    gate_use_t_alert_start: int,
    gate_search_consecutive_ks: list[int] | None,
    gate_search_smooth_windows: list[int] | None,
    gate_search_use_t_alert_start_options: list[int] | None,
    tau_select_level: str,
    tau_mode: str,
    tau_target_precision: float,
    tau_target_recall: float,
    tau_alert_target_recall: float | None,
    tau_alert_max_false_alert_rate: float | None,
    tau_alert_policy: str,
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
        out_root = default_stage1_out_root(pest)
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
        tags=parse_tags(wandb_tags) + [f"pest:{pest}", "script:run_event_eval"],
        config={
            "pest": pest,
            "run": run,
            "split_seed": int(split_seed),
            "gate_policy_search": bool(gate_policy_search),
            "gate_consecutive_k": int(gate_consecutive_k),
            "gate_smooth_window": int(gate_smooth_window),
            "gate_use_t_alert_start": int(gate_use_t_alert_start),
            "tau_select_level": tau_select_level,
            "tau_mode": tau_mode,
            "tau_target_precision": float(tau_target_precision),
            "tau_target_recall": float(tau_target_recall),
            "tau_alert_target_recall": None if tau_alert_target_recall is None else float(tau_alert_target_recall),
            "tau_alert_max_false_alert_rate": None if tau_alert_max_false_alert_rate is None else float(tau_alert_max_false_alert_rate),
            "tau_alert_policy": tau_alert_policy,
        },
    )

    ckpt_path_resolved = resolve_ckpt_path(run, out_root, ckpt_path)
    print(f"Using checkpoint: {ckpt_path_resolved}")
    ckpt = torch.load(ckpt_path_resolved, map_location="cpu")
    trained_states = ckpt["trained_states"]
    print("loaded ckpt:", ckpt_path_resolved, "| seeds:", [d["seed"] for d in trained_states])

    _, feature_names_eval, T, samples, debug_stats = build_samples_for_run(run, get_feature_cols, return_debug_stats=True)
    print(f"[features] n={len(feature_names_eval)} head={feature_names_eval[:5]} tail={feature_names_eval[-5:]}")

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

    log_split_fingerprint("event_eval", train_s, val_s, test_s)

    task_mode = str(ckpt.get("task_mode", "season_complete"))
    nowcast_window = int(ckpt.get("nowcast_window", 28))
    nowcast_stride = int(ckpt.get("nowcast_stride", 7))
    nowcast_tstar_start = ckpt.get("nowcast_tstar_start", None)
    nowcast_only_pre_event = int(ckpt.get("nowcast_only_pre_event", 1))
    nowcast_event_time_proxy = str(ckpt.get("nowcast_event_time_proxy", "r"))
    nowcast_label_mode = str(ckpt.get("nowcast_label_mode", "eventually"))
    nowcast_label_horizon = ckpt.get("nowcast_label_horizon", None)
    add_tstar_position_feature = bool(ckpt.get("add_tstar_position_feature", False))
    if nowcast_label_mode == "within_h" and nowcast_label_horizon is None:
        raise ValueError("checkpoint is missing nowcast_label_horizon for nowcast_label_mode=within_h")
    if task_mode == "nowcast":
        val_s = build_nowcast_samples(
            val_s,
            window=nowcast_window,
            stride=nowcast_stride,
            tstar_start=nowcast_tstar_start,
            only_pre_event=bool(nowcast_only_pre_event),
            event_time_proxy=nowcast_event_time_proxy,
            label_mode=nowcast_label_mode,
            label_horizon=nowcast_label_horizon,
        )
        test_s = build_nowcast_samples(
            test_s,
            window=nowcast_window,
            stride=nowcast_stride,
            tstar_start=nowcast_tstar_start,
            only_pre_event=bool(nowcast_only_pre_event),
            event_time_proxy=nowcast_event_time_proxy,
            label_mode=nowcast_label_mode,
            label_horizon=nowcast_label_horizon,
        )
        print(
            f"[nowcast] window={nowcast_window} stride={nowcast_stride} "
            f"tstar_start={nowcast_tstar_start} only_pre_event={bool(nowcast_only_pre_event)} "
            f"event_time_proxy={nowcast_event_time_proxy} "
            f"label_mode={nowcast_label_mode} label_horizon={nowcast_label_horizon} "
            f"add_tstar_position_feature={bool(add_tstar_position_feature)} | "
            f"samples val={len(val_s)} test={len(test_s)}"
        )

    x_mean, x_std = compute_norm_stats(train_s)
    if task_mode == "nowcast":
        val_ds = EventBinaryDataset(val_s, x_mean, x_std)
        test_ds = EventBinaryDataset(test_s, x_mean, x_std)
    else:
        val_ds = IntervalEventDataset(val_s, x_mean, x_std)
        test_ds = IntervalEventDataset(test_s, x_mean, x_std)
    val_loader = make_loader(val_ds, C.BATCH_EVAL, shuffle=False)
    test_loader = make_loader(test_ds, C.BATCH_EVAL, shuffle=False)

    y_val = make_event_labels(val_s)
    y_test = make_event_labels(test_s)
    tstar_val = np.array([int(s.get("tstar", 0)) for s in val_s], dtype=int)
    tstar_test = np.array([int(s.get("tstar", 0)) for s in test_s], dtype=int)
    d_in = int(test_ds[0][0].shape[-1])
    model_T = int(ckpt.get("T", T))
    print(f"RUN={run} | D_in={d_in} | T={model_T} | task_mode={task_mode} | val_n={len(val_s)} test_n={len(test_s)}")
    if tau_select_level == "alert_site_year" and task_mode != "nowcast":
        raise ValueError("tau_select_level=alert_site_year requires task_mode=nowcast")
    if int(gate_consecutive_k) < 1:
        raise ValueError("--gate_consecutive_k must be >= 1")
    if int(gate_smooth_window) < 1:
        raise ValueError("--gate_smooth_window must be >= 1")

    model_kind = str(ckpt.get("event_model", "transformer"))
    records = []
    split_sites = {
        "train": {str(s["site_id"]) for s in train_s},
        "val": {str(s["site_id"]) for s in val_s},
        "test": {str(s["site_id"]) for s in test_s},
    }
    dropped_pairs = debug_stats.get("dropped_site_year", [])
    dropped_by_split = {"train": 0, "val": 0, "test": 0}
    for site_id, _year in dropped_pairs:
        if site_id in split_sites["train"]:
            dropped_by_split["train"] += 1
        elif site_id in split_sites["val"]:
            dropped_by_split["val"] += 1
        elif site_id in split_sites["test"]:
            dropped_by_split["test"] += 1
    tstar_rows = []
    alert_rows = []
    alert_summary_rows = []
    prob_rows = []
    gate_policy_rows = []
    for d in trained_states:
        seed = int(d["seed"])
        if seeds is not None and seed not in seeds:
            continue
        if model_kind == "transformer":
            model = EventTransformer(
                d_in=d_in,
                d_model=C.D_MODEL,
                nhead=C.N_HEAD,
                num_layers=2,
                dropout=C.DROPOUT,
                max_len=C.MAX_LEN,
            ).to(device)
            model.load_state_dict(d["state_dict"])
            model.eval()
            p_val_raw = predict_event_prob(model, val_loader, device=device)
            p_test_raw = predict_event_prob(model, test_loader, device=device)
        else:
            clf = d.get("sk_model")
            if clf is None:
                raise ValueError("checkpoint does not include sklearn model object for tabular event model")
            X_val_tab = build_tabular_from_samples(
                val_s,
                add_tstar_position_feature=bool(add_tstar_position_feature),
            )
            X_test_tab = build_tabular_from_samples(
                test_s,
                add_tstar_position_feature=bool(add_tstar_position_feature),
            )
            if hasattr(clf, "predict_proba"):
                p_val_raw = clf.predict_proba(X_val_tab)[:, 1]
                p_test_raw = clf.predict_proba(X_test_tab)[:, 1]
            else:
                p_val_raw = np.asarray(clf.predict(X_val_tab), dtype=float)
                p_test_raw = np.asarray(clf.predict(X_test_tab), dtype=float)

        t_best, val_nll_cal = fit_temperature_grid(y_val, p_val_raw)
        p_val_cal = apply_temperature(p_val_raw, t_best)
        p_test_cal = apply_temperature(p_test_raw, t_best)

        t_alert_start = None
        gate_policy_name = build_gate_policy_name(int(gate_consecutive_k), int(gate_smooth_window), bool(gate_use_t_alert_start))
        gate_consecutive_k_selected = int(gate_consecutive_k)
        gate_smooth_window_selected = int(gate_smooth_window)
        gate_use_t_alert_start_selected = bool(gate_use_t_alert_start)
        if tau_select_level == "alert_site_year":
            if gate_policy_search:
                candidate_rows = []
                candidates = iter_gate_policy_candidates(
                    gate_search_consecutive_ks or [1, 2, 3],
                    gate_search_smooth_windows or [1, 3],
                    [bool(x) for x in (gate_search_use_t_alert_start_options or [0, 1])],
                )
                for cand in candidates:
                    if nowcast_label_mode == "within_h":
                        picked = best_tau_by_within_h_alert_window(
                            val_s,
                            p_val_cal,
                            split_name="val",
                            seed=seed,
                            y_true=y_val,
                            tstar=tstar_val,
                            gate_consecutive_k=int(cand["gate_consecutive_k"]),
                            gate_smooth_window=int(cand["gate_smooth_window"]),
                            gate_use_t_alert_start=bool(cand["gate_use_t_alert_start"]),
                            target_window_recall=tau_alert_target_recall,
                            label_horizon=int(nowcast_label_horizon),
                        )
                    else:
                        picked = best_tau_by_alert_site_year(
                            val_s,
                            p_val_cal,
                            split_name="val",
                            seed=seed,
                            y_true=y_val,
                            tstar=tstar_val,
                            gate_consecutive_k=int(cand["gate_consecutive_k"]),
                            gate_smooth_window=int(cand["gate_smooth_window"]),
                            gate_use_t_alert_start=bool(cand["gate_use_t_alert_start"]),
                            target_recall=tau_alert_target_recall,
                            max_false_alert_rate=tau_alert_max_false_alert_rate,
                            policy=tau_alert_policy,
                        )
                    candidate_rows.append(picked)
                if nowcast_label_mode == "within_h":
                    selected_policy = select_best_within_h_metric_row(
                        candidate_rows,
                        target_window_recall=tau_alert_target_recall,
                    )
                else:
                    selected_policy = select_best_alert_metric_row(
                        candidate_rows,
                        target_recall=tau_alert_target_recall,
                        max_false_alert_rate=tau_alert_max_false_alert_rate,
                        policy="false_alert_rate_then_f1",
                    )
                for row in candidate_rows:
                    gate_policy_rows.append(
                        {
                            "split": "val",
                            "seed": int(seed),
                            "gate_policy_name": row["gate_policy_name"],
                            "gate_consecutive_k": int(row["gate_consecutive_k"]),
                            "gate_smooth_window": int(row["gate_smooth_window"]),
                            "gate_use_t_alert_start": bool(row["gate_use_t_alert_start"]),
                            "tau_selected": float(row["tau_selected"]),
                            "tau_alert_fallback_mode": str(row.get("tau_alert_fallback_mode", "")),
                            "alert_rate": float(row["alert_rate"]),
                            "alert_precision": float(row["alert_precision"]),
                            "alert_recall": float(row["alert_recall"]),
                            "alert_f1": float(row["alert_f1"]),
                            "false_alert_rate": float(row["false_alert_rate"]),
                            "specificity": float(row["specificity"]),
                            "lead_time_mean": float(row["lead_time_mean"]),
                            "lead_time_median": float(row["lead_time_median"]),
                            "window_precision": float(row["window_precision"]),
                            "window_recall": float(row["window_recall"]),
                            "window_f1": float(row["window_f1"]),
                            "early_alert_rate": float(row["early_alert_rate"]),
                            "non_event_false_alert_rate": float(row["non_event_false_alert_rate"]),
                            "t_alert_start": row.get("t_alert_start"),
                            "selected": int(row["gate_policy_name"] == selected_policy["gate_policy_name"]),
                        }
                    )
                tau_pick = selected_policy
            else:
                if nowcast_label_mode == "within_h":
                    tau_pick = best_tau_by_within_h_alert_window(
                        val_s,
                        p_val_cal,
                        split_name="val",
                        seed=seed,
                        y_true=y_val,
                        tstar=tstar_val,
                        gate_consecutive_k=int(gate_consecutive_k),
                        gate_smooth_window=int(gate_smooth_window),
                        gate_use_t_alert_start=bool(gate_use_t_alert_start),
                        target_window_recall=tau_alert_target_recall,
                        label_horizon=int(nowcast_label_horizon),
                    )
                else:
                    tau_pick = best_tau_by_alert_site_year(
                        val_s,
                        p_val_cal,
                        split_name="val",
                        seed=seed,
                        y_true=y_val,
                        tstar=tstar_val,
                        gate_consecutive_k=int(gate_consecutive_k),
                        gate_smooth_window=int(gate_smooth_window),
                        gate_use_t_alert_start=bool(gate_use_t_alert_start),
                        target_recall=tau_alert_target_recall,
                        max_false_alert_rate=tau_alert_max_false_alert_rate,
                        policy=tau_alert_policy,
                    )
            tau = float(tau_pick["tau_selected"])
            tau_alert_fallback_mode = str(tau_pick.get("tau_alert_fallback_mode", "strict"))
            t_alert_start = tau_pick.get("t_alert_start")
            gate_policy_name = str(tau_pick.get("gate_policy_name", gate_policy_name))
            gate_consecutive_k_selected = int(tau_pick.get("gate_consecutive_k", gate_consecutive_k))
            gate_smooth_window_selected = int(tau_pick.get("gate_smooth_window", gate_smooth_window))
            gate_use_t_alert_start_selected = bool(tau_pick.get("gate_use_t_alert_start", bool(gate_use_t_alert_start)))
        else:
            tau, _, _, _ = best_tau_by_target(
                y_val,
                p_val_cal,
                mode=tau_mode,
                target_precision=tau_target_precision,
                target_recall=tau_target_recall,
            )
            tau_alert_fallback_mode = None

        val_prec, val_rec, val_f1 = pr_at_tau(y_val, p_val_cal, tau)
        test_prec, test_rec, test_f1 = pr_at_tau(y_test, p_test_cal, tau)
        val_fp = fp_rate_at_tau(y_val, p_val_cal, tau)
        test_fp = fp_rate_at_tau(y_test, p_test_cal, tau)
        prob_rows.extend(build_probability_rows(val_s, p_val_raw, p_val_cal, split_name="val", seed=seed))
        prob_rows.extend(build_probability_rows(test_s, p_test_raw, p_test_cal, split_name="test", seed=seed))
        test_alert_summary = {
            "alert_rate": float("nan"),
            "alert_precision": float("nan"),
            "alert_recall": float("nan"),
            "alert_f1": float("nan"),
            "false_alert_rate": float("nan"),
            "specificity": float("nan"),
            "lead_time_mean": float("nan"),
            "lead_time_median": float("nan"),
            "window_precision": float("nan"),
            "window_recall": float("nan"),
            "window_f1": float("nan"),
            "early_alert_rate": float("nan"),
            "non_event_false_alert_rate": float("nan"),
            "t_alert_start": None,
        }
        if task_mode == "nowcast":
            tstar_rows.extend(metrics_by_tstar(y_val, p_val_raw, p_val_cal, tau, tstar_val, split_name="val", seed=seed))
            tstar_rows.extend(metrics_by_tstar(y_test, p_test_raw, p_test_cal, tau, tstar_test, split_name="test", seed=seed))
            if tau_select_level == "alert_site_year" and gate_use_t_alert_start_selected:
                t_alert_start = compute_t_alert_start(y_val, p_val_cal, tstar_val, tau)

            val_alert_rows = build_alert_rows(
                val_s,
                p_val_cal,
                tau,
                split_name="val",
                seed=seed,
                t_alert_start=t_alert_start,
                consecutive_k=gate_consecutive_k_selected,
                smooth_window=gate_smooth_window_selected,
            )
            test_alert_rows = build_alert_rows(
                test_s,
                p_test_cal,
                tau,
                split_name="test",
                seed=seed,
                t_alert_start=t_alert_start,
                consecutive_k=gate_consecutive_k_selected,
                smooth_window=gate_smooth_window_selected,
            )
            alert_rows.extend(val_alert_rows)
            alert_rows.extend(test_alert_rows)

            val_alert_summary = summarize_alert_rows(
                val_alert_rows,
                split_name="val",
                seed=seed,
                t_alert_start=t_alert_start,
                tau_selected=tau,
            )
            if nowcast_label_mode == "within_h":
                val_alert_summary = summarize_within_h_alert_rows(
                    val_alert_rows,
                    samples=val_s,
                    split_name="val",
                    seed=seed,
                    t_alert_start=t_alert_start,
                    tau_selected=tau,
                    label_horizon=int(nowcast_label_horizon),
                )
                test_alert_summary = summarize_within_h_alert_rows(
                    test_alert_rows,
                    samples=test_s,
                    split_name="test",
                    seed=seed,
                    t_alert_start=t_alert_start,
                    tau_selected=tau,
                    label_horizon=int(nowcast_label_horizon),
                )
            else:
                test_alert_summary = summarize_alert_rows(
                    test_alert_rows,
                    split_name="test",
                    seed=seed,
                    t_alert_start=t_alert_start,
                    tau_selected=tau,
                )
            alert_summary_rows.extend([val_alert_summary, test_alert_summary])

        records.append(
            {
                "seed": seed,
                "best_epoch": int(d["best_epoch"]),
                "best_val_bce": float(d.get("best_val_bce", np.nan)),
                "temperature": float(t_best),
                "gate_policy_name": gate_policy_name,
                "gate_consecutive_k": int(gate_consecutive_k_selected),
                "gate_smooth_window": int(gate_smooth_window_selected),
                "gate_use_t_alert_start": int(gate_use_t_alert_start_selected),
                "nowcast_label_mode": nowcast_label_mode,
                "nowcast_label_horizon": (
                    int(nowcast_label_horizon) if nowcast_label_horizon is not None else float("nan")
                ),
                "add_tstar_position_feature": int(add_tstar_position_feature),
                "tau_select_level": tau_select_level,
                "tau_mode": tau_mode,
                "tau_target_precision": float(tau_target_precision),
                "tau_target_recall": float(tau_target_recall),
                "tau_alert_target_recall": (
                    float(tau_alert_target_recall) if tau_alert_target_recall is not None else float("nan")
                ),
                "tau_alert_max_false_alert_rate": (
                    float(tau_alert_max_false_alert_rate) if tau_alert_max_false_alert_rate is not None else float("nan")
                ),
                "tau_alert_policy": tau_alert_policy,
                "tau_alert_fallback_mode": tau_alert_fallback_mode,
                "tau_selected": float(tau),
                "val_auc_raw": float(auc_roc_binary(y_val, p_val_raw)),
                "val_auc_cal": float(auc_roc_binary(y_val, p_val_cal)),
                "test_auc_raw": float(auc_roc_binary(y_test, p_test_raw)),
                "test_auc_cal": float(auc_roc_binary(y_test, p_test_cal)),
                "val_pr_auc_raw": float(pr_auc_binary(y_val, p_val_raw)),
                "val_pr_auc_cal": float(pr_auc_binary(y_val, p_val_cal)),
                "test_pr_auc_raw": float(pr_auc_binary(y_test, p_test_raw)),
                "test_pr_auc_cal": float(pr_auc_binary(y_test, p_test_cal)),
                "val_brier_raw": float(brier_score(y_val, p_val_raw)),
                "val_brier_cal": float(brier_score(y_val, p_val_cal)),
                "test_brier_raw": float(brier_score(y_test, p_test_raw)),
                "test_brier_cal": float(brier_score(y_test, p_test_cal)),
                "val_nll_raw": float(binary_nll(y_val, p_val_raw)),
                "val_nll_cal": float(val_nll_cal),
                "test_nll_raw": float(binary_nll(y_test, p_test_raw)),
                "test_nll_cal": float(binary_nll(y_test, p_test_cal)),
                "val_event_rate": float(np.mean(y_val)),
                "test_event_rate": float(np.mean(y_test)),
                "val_prec@tau": float(val_prec),
                "val_rec@tau": float(val_rec),
                "val_f1@tau": float(val_f1),
                "val_fp_rate@tau": float(val_fp),
                "test_prec@tau": float(test_prec),
                "test_rec@tau": float(test_rec),
                "test_f1@tau": float(test_f1),
                "test_fp_rate@tau": float(test_fp),
                "test_mean_p_raw": float(np.mean(p_test_raw)),
                "test_mean_p_cal": float(np.mean(p_test_cal)),
                "test_event_mean_p_raw": float(np.mean(p_test_raw[y_test == 1])) if (y_test == 1).any() else float("nan"),
                "test_nonevent_mean_p_raw": float(np.mean(p_test_raw[y_test == 0])) if (y_test == 0).any() else float("nan"),
                "test_event_mean_p_cal": float(np.mean(p_test_cal[y_test == 1])) if (y_test == 1).any() else float("nan"),
                "test_nonevent_mean_p_cal": float(np.mean(p_test_cal[y_test == 0])) if (y_test == 0).any() else float("nan"),
                "test_alert_rate": float(test_alert_summary["alert_rate"]),
                "test_alert_precision": float(test_alert_summary["alert_precision"]),
                "test_alert_recall": float(test_alert_summary["alert_recall"]),
                "test_alert_f1": float(test_alert_summary["alert_f1"]),
                "test_false_alert_rate": float(test_alert_summary["false_alert_rate"]),
                "test_lead_time_mean": float(test_alert_summary["lead_time_mean"]),
                "test_lead_time_median": float(test_alert_summary["lead_time_median"]),
                "test_window_precision": float(test_alert_summary["window_precision"]),
                "test_window_recall": float(test_alert_summary["window_recall"]),
                "test_window_f1": float(test_alert_summary["window_f1"]),
                "test_early_alert_rate": float(test_alert_summary["early_alert_rate"]),
                "test_non_event_false_alert_rate": float(test_alert_summary["non_event_false_alert_rate"]),
                "test_t_alert_start": test_alert_summary["t_alert_start"],
                "drop_ratio_len_mismatch_overall": float(debug_stats.get("drop_ratio_len_mismatch", float("nan"))),
                "drop_ratio_len_mismatch_train": float(
                    dropped_by_split["train"] / max(dropped_by_split["train"] + len(train_s), 1)
                ),
                "drop_ratio_len_mismatch_val": float(
                    dropped_by_split["val"] / max(dropped_by_split["val"] + len(val_s), 1)
                ),
                "drop_ratio_len_mismatch_test": float(
                    dropped_by_split["test"] / max(dropped_by_split["test"] + len(test_s), 1)
                ),
            }
        )

        if wandb_run is not None:
            import wandb

            val_pred_rate = float((p_val_cal >= tau).mean())
            test_pred_rate = float((p_test_cal >= tau).mean())
            val_prec_tau, val_rec_tau, val_f1_tau = pr_at_tau(y_val, p_val_cal, tau)
            test_prec_tau, test_rec_tau, test_f1_tau = pr_at_tau(y_test, p_test_cal, tau)

            wandb_run.log(
                {
                    "seed": int(seed),
                    "val/auroc": float(auc_roc_binary(y_val, p_val_cal)),
                    "val/auprc": float(pr_auc_binary(y_val, p_val_cal)),
                    "val/precision@tau": float(val_prec_tau),
                    "val/recall@tau": float(val_rec_tau),
                    "val/f1@tau": float(val_f1_tau),
                    "val/pred_pos_rate@tau": float(val_pred_rate),
                    "val/nll": float(val_nll_cal),
                    "test/auroc": float(auc_roc_binary(y_test, p_test_cal)),
                    "test/auprc": float(pr_auc_binary(y_test, p_test_cal)),
                    "test/precision@tau": float(test_prec_tau),
                    "test/recall@tau": float(test_rec_tau),
                    "test/f1@tau": float(test_f1_tau),
                    "test/pred_pos_rate@tau": float(test_pred_rate),
                    "test/nll": float(binary_nll(y_test, p_test_cal)),
                }
            )

            # PR curve (test, calibrated)
            try:
                pr_curve = wandb.plot.pr_curve(
                    y_true=y_test,
                    y_probas=p_test_cal,
                    labels=["neg", "pos"],
                )
                wandb_run.log({"test/pr_curve": pr_curve})
            except Exception:
                pass

            # Probability histogram (test, calibrated)
            wandb_run.log({"test/prob_hist": wandb.Histogram(p_test_cal)})

            # Confusion matrix at tau (test)
            pred_test = (p_test_cal >= float(tau)).astype(int)
            try:
                cm = wandb.plot.confusion_matrix(
                    y_true=y_test,
                    preds=pred_test,
                    class_names=["neg", "pos"],
                )
                wandb_run.log({"test/confusion_matrix@tau": cm})
            except Exception:
                pass

            # Threshold sweep (test, calibrated)
            taus = np.linspace(0.05, 0.95, 181)
            sweep_rows = []
            for t in taus:
                prec, rec, f1 = pr_at_tau(y_test, p_test_cal, float(t))
                pred_rate = float((p_test_cal >= float(t)).mean())
                sweep_rows.append([float(t), float(prec), float(rec), float(f1), pred_rate])
            sweep_table = wandb.Table(columns=["tau", "precision", "recall", "f1", "pred_pos_rate"], data=sweep_rows)
            wandb_run.log({"test/threshold_sweep": sweep_table})

            if task_mode == "nowcast" and alert_rows:
                alert_fig = plot_alert_rows(alert_rows, title="Alert vs True Interval (sample)", Tend=T)
                if alert_fig is not None:
                    wandb_run.log({"event/alert_vs_interval": wandb.Image(alert_fig)})

    df = pd.DataFrame(records).sort_values("seed").reset_index(drop=True)
    print("\n=== per-seed event eval ===")
    print(df.to_string(index=False))

    out_csv_resolved = resolve_out_csv(run, out_root, out_csv)
    df.to_csv(out_csv_resolved, index=False)
    print("saved:", out_csv_resolved)
    if task_mode == "nowcast":
        save_by_tstar_rows(out_csv_resolved, tstar_rows)
        save_alert_outputs(out_csv_resolved, alert_rows, alert_summary_rows)
        save_probability_outputs(out_csv_resolved, prob_rows)
        save_gate_policy_outputs(out_csv_resolved, gate_policy_rows)
    save_meta_json(
        out_csv_resolved,
        {
            "task_mode": task_mode,
            "nowcast_window": nowcast_window,
            "nowcast_stride": nowcast_stride,
            "nowcast_tstar_start": nowcast_tstar_start,
            "nowcast_only_pre_event": nowcast_only_pre_event,
            "nowcast_event_time_proxy": nowcast_event_time_proxy,
            "nowcast_label_mode": nowcast_label_mode,
            "nowcast_label_horizon": nowcast_label_horizon,
            "add_tstar_position_feature": bool(add_tstar_position_feature),
            "gate_policy_search": bool(gate_policy_search),
            "gate_consecutive_k": int(gate_consecutive_k),
            "gate_smooth_window": int(gate_smooth_window),
            "gate_use_t_alert_start": int(gate_use_t_alert_start),
            "gate_search_consecutive_ks": gate_search_consecutive_ks,
            "gate_search_smooth_windows": gate_search_smooth_windows,
            "gate_search_use_t_alert_start_options": gate_search_use_t_alert_start_options,
            "tau_select_level": tau_select_level,
            "tau_mode": tau_mode,
            "tau_target_precision": float(tau_target_precision),
            "tau_target_recall": float(tau_target_recall),
            "tau_alert_target_recall": None if tau_alert_target_recall is None else float(tau_alert_target_recall),
            "tau_alert_max_false_alert_rate": None if tau_alert_max_false_alert_rate is None else float(tau_alert_max_false_alert_rate),
            "tau_alert_policy": tau_alert_policy,
            "split_seed": int(split_seed),
            "n_records": int(len(df)),
        },
    )

    finish_wandb_run(wandb_run)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pest", type=str, required=True)
    p.add_argument("--ckpt", type=str, default=None)
    p.add_argument("--run", type=int, default=0)
    p.add_argument("--out_csv", type=str, default=None)
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
    p.add_argument("--gate_policy_search", action="store_true")
    p.add_argument("--gate_consecutive_k", type=int, default=1)
    p.add_argument("--gate_smooth_window", type=int, default=1)
    p.add_argument("--gate_use_t_alert_start", type=int, default=0)
    p.add_argument("--gate_search_consecutive_ks", type=int, nargs="*", default=None)
    p.add_argument("--gate_search_smooth_windows", type=int, nargs="*", default=None)
    p.add_argument("--gate_search_use_t_alert_start_options", type=int, nargs="*", default=None)
    p.add_argument("--tau_select_level", type=str, default="pooled", choices=["pooled", "alert_site_year"])
    p.add_argument("--tau_mode", type=str, default="f1", choices=["f1", "precision_target", "recall_target"])
    p.add_argument("--tau_target_precision", type=float, default=0.6)
    p.add_argument("--tau_target_recall", type=float, default=0.6)
    p.add_argument("--tau_alert_target_recall", type=float, default=None)
    p.add_argument("--tau_alert_max_false_alert_rate", type=float, default=None)
    p.add_argument("--tau_alert_policy", type=str, default="f1_then_false_alert_rate", choices=["f1", "min_false_alert_rate", "f1_then_false_alert_rate"])
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
        ckpt_path=args.ckpt,
        run=args.run,
        out_csv=args.out_csv,
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
        gate_policy_search=args.gate_policy_search,
        gate_consecutive_k=args.gate_consecutive_k,
        gate_smooth_window=args.gate_smooth_window,
        gate_use_t_alert_start=args.gate_use_t_alert_start,
        gate_search_consecutive_ks=args.gate_search_consecutive_ks,
        gate_search_smooth_windows=args.gate_search_smooth_windows,
        gate_search_use_t_alert_start_options=args.gate_search_use_t_alert_start_options,
        tau_select_level=args.tau_select_level,
        tau_mode=args.tau_mode,
        tau_target_precision=args.tau_target_precision,
        tau_target_recall=args.tau_target_recall,
        tau_alert_target_recall=args.tau_alert_target_recall,
        tau_alert_max_false_alert_rate=args.tau_alert_max_false_alert_rate,
        tau_alert_policy=args.tau_alert_policy,
        use_wandb=args.use_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_group=args.wandb_group,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags,
        wandb_job_type=args.wandb_job_type,
    )
