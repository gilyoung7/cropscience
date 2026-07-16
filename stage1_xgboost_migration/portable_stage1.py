"""Torch-free, pickle-free Stage-1 alert runtime.

Consumes only portable assets:
    artifacts/<pest>/<A|D>/model.json        XGBoost native model
    artifacts/<pest>/<A|D>/calibration.json  frozen temperature
    artifacts/<pest>/gate.json               frozen method/k/tau (from the summary JSON)

Imports numpy + xgboost only -- deliberately no torch, no pandas, no sklearn, and
it never opens a .pt. Every numeric routine below is a line-for-line port of
api_handoff_transformer/infer/stage1.py; the docstrings name the original so the
two can be diffed. Nothing here re-fits, re-tunes, or re-derives anything.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost

ROOT = Path(__file__).resolve().parent
ARTIFACTS_ROOT = ROOT / "artifacts"

# infer/stage1.py::_build_tabular statistic order.
TABULAR_STATS = ["mean", "std", "min", "max", "first", "last", "slope"]

# infer/stage1.py::HISTORY_STATIC_NAMES / HISTORY_DYNAMIC_NAMES -- order matters.
HISTORY_STATIC_NAMES = [
    "prev_year_L_doy_at_site",
    "prev_year_event_at_site",
    "site_avg_L_doy_recent3y",
    "years_since_last_event_at_site",
    "n_events_recent5y_at_site",
    "prev_year_L_miss",
    "site_avg_L_recent3y_miss",
]
HISTORY_DYNAMIC_NAMES = [
    "days_to_prev_year_L",
    "abs_days_to_prev_year_L",
    "days_to_site_avg_L_recent3y",
    "abs_days_to_site_avg_L_recent3y",
]
HISTORY_FEATURE_DIM = len(HISTORY_STATIC_NAMES) + len(HISTORY_DYNAMIC_NAMES)  # 11


# --- port of infer/stage1.py::_history_channels ----------------------------
def history_channels(X_T: int, h: dict, doy_start: int) -> np.ndarray:
    static = np.array([
        h["prev_year_L_doy_at_site"], h["prev_year_event_at_site"],
        h["site_avg_L_doy_recent3y"], h["years_since_last_event_at_site"],
        h["n_events_recent5y_at_site"], h["prev_year_L_miss"], h["site_avg_L_recent3y_miss"],
    ], dtype=np.float32)
    static_chan = np.tile(static[None, :], (X_T, 1))
    doys = np.arange(X_T, dtype=np.float32) + float(doy_start)
    prev_L = float(h["prev_year_L_doy_at_site"]); avg3y = float(h["site_avg_L_doy_recent3y"])
    miss_prev = float(h["prev_year_L_miss"]); miss_avg = float(h["site_avg_L_recent3y_miss"])
    dyn = np.stack([
        (prev_L - doys) * (1.0 - miss_prev),
        np.abs(prev_L - doys) * (1.0 - miss_prev),
        (avg3y - doys) * (1.0 - miss_avg),
        np.abs(avg3y - doys) * (1.0 - miss_avg),
    ], axis=1).astype(np.float32)
    return np.concatenate([static_chan, dyn], axis=1).astype(np.float32)


# --- port of infer/stage1.py::_append_history ------------------------------
def append_history(base_X: np.ndarray, site: str, year: int, history: dict,
                   doy_start: int) -> np.ndarray:
    T = int(base_X.shape[0])
    h = history.get((str(site), int(year)))
    if h is None:
        pad = np.zeros((T, HISTORY_FEATURE_DIM), dtype=np.float32)
        pad[:, HISTORY_STATIC_NAMES.index("prev_year_L_miss")] = 1.0
        pad[:, HISTORY_STATIC_NAMES.index("site_avg_L_recent3y_miss")] = 1.0
        pad[:, HISTORY_STATIC_NAMES.index("years_since_last_event_at_site")] = 99.0
        return np.concatenate([base_X, pad], axis=1).astype(np.float32)
    return np.concatenate([base_X, history_channels(T, h, doy_start)], axis=1).astype(np.float32)


# --- port of infer/stage1.py::_apply_temperature ---------------------------
def apply_temperature(p: np.ndarray, temperature: float, eps: float = 1e-8) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1.0 - eps)
    logit = np.log(p / (1.0 - p))
    return 1.0 / (1.0 + np.exp(-(logit / float(temperature))))


# --- port of infer/stage1.py::_first_crossing_k ----------------------------
def first_crossing_k(ts: np.ndarray, ps: np.ndarray, tau: float, k: int) -> int | None:
    streak = 0
    for i in range(len(ts)):
        if ps[i] >= tau:
            streak += 1
            if streak >= k:
                return int(ts[i])
        else:
            streak = 0
    return None


# --- port of infer/stage1.py::_build_tabular -------------------------------
def build_tabular(samples: list[dict], add_tstar_position_feature: bool) -> np.ndarray:
    feats = []
    for s in samples:
        x = np.asarray(s["X"], dtype=np.float32)
        t = np.arange(x.shape[0], dtype=np.float32)
        t_center = t - t.mean()
        t_var = float((t_center ** 2).sum()) + 1e-8
        mean = x.mean(axis=0); std = x.std(axis=0)
        xmin = x.min(axis=0); xmax = x.max(axis=0)
        xfirst = x[0]; xlast = x[-1]
        slope = ((x - mean) * t_center[:, None]).sum(axis=0) / t_var
        f = np.concatenate([mean, std, xmin, xmax, xfirst, xlast, slope], axis=0)
        if add_tstar_position_feature:
            season_length = max(int(s.get("season_length", x.shape[0])), 1)
            tstar = int(s.get("tstar", season_length))
            f = np.concatenate([f, np.asarray([float(tstar) / float(season_length)],
                                              dtype=np.float32)])
        feats.append(f)
    return np.stack(feats, axis=0).astype(np.float32) if feats else np.zeros((0, 0), np.float32)


# --- port of infer/stage1.py::_build_nowcast_samples -----------------------
def build_nowcast_samples(samples: list[dict], window: int, stride: int,
                          only_pre_event: bool, event_time_proxy: str) -> list[dict]:
    out = []
    if not samples:
        return out
    T = int(samples[0]["X"].shape[0])
    t0 = min(int(window), T)
    for s in samples:
        x = np.asarray(s["X"], dtype=np.float32)
        ctype = str(s["censor_type"]); has_event = ctype != "right"
        if has_event:
            L_time = int(s["L"]); R_time = int(s["R"])
            event_time = int((L_time + R_time) // 2) if event_time_proxy == "mid" else int(R_time)
        else:
            event_time = None
        for tstar in range(t0, T + 1, stride):
            if only_pre_event and has_event and event_time is not None and tstar >= event_time:
                continue
            y_event = 1 if (has_event and event_time is not None and event_time > tstar) else 0
            out.append({
                "site_id": s["site_id"], "year": int(s["year"]),
                "X": x[(tstar - window):tstar, :].astype(np.float32, copy=False),
                "y_event": int(y_event), "tstar": int(tstar), "season_length": int(T),
            })
    return out


# --- port of infer/stage1.py::_dispatch_features_for_sy --------------------
def dispatch_features_for_sy(sy, alert_t, per_sy_A, per_sy_D, with_h, doy_start,
                             gate_method, tau_no, tau_with, tau_single) -> dict:
    A = per_sy_A.get(sy); D = per_sy_D.get(sy)

    def _at(series, t):
        if series is None:
            return float("nan")
        idx = np.where(series["ts"] == int(t))[0]
        return float(series["ps"][int(idx[0])]) if idx.size else float("nan")

    a_score = _at(A, alert_t); d_score = _at(D, alert_t)
    if gate_method == "A_baseline":
        d_ts, d_ps, tau_used, disp_score, branch = A["ts"], A["ps"], float(tau_single), a_score, "A"
    elif gate_method == "D_history":
        d_ts, d_ps, tau_used, disp_score, branch = D["ts"], D["ps"], float(tau_single), d_score, "D"
    else:
        if with_h:
            d_ts, d_ps, tau_used, disp_score, branch = D["ts"], D["ps"], tau_with, d_score, "D"
        else:
            d_ts, d_ps, tau_used, disp_score, branch = A["ts"], A["ps"], tau_no, a_score, "A"

    margin = (d_score - a_score) if (np.isfinite(d_score) and np.isfinite(a_score)) else float("nan")
    sot_margin = (disp_score - tau_used) if np.isfinite(disp_score) else float("nan")
    alert_doy = int(alert_t) + doy_start - 1
    doys = d_ts.astype(int) + doy_start - 1
    mask_14 = (doys >= alert_doy - 14) & (doys <= alert_doy)
    mask_28 = (doys >= alert_doy - 28) & (doys <= alert_doy)
    mean_14 = float(d_ps[mask_14].mean()) if mask_14.any() else float("nan")
    mean_28 = float(d_ps[mask_28].mean()) if mask_28.any() else float("nan")
    streak = 0
    idx_at = np.where(d_ts == int(alert_t))[0]
    if idx_at.size:
        i = int(idx_at[0])
        while i >= 0 and d_ps[i] >= tau_used:
            streak += 1; i -= 1
    if mask_14.sum() >= 2:
        slope_14 = float(np.polyfit(doys[mask_14].astype(float), d_ps[mask_14].astype(float), 1)[0])
    else:
        slope_14 = float("nan")
    mask_cum = d_ts <= int(alert_t)
    p_mean_so_far = float(d_ps[mask_cum].mean()) if mask_cum.any() else float("nan")
    return {
        "alert_tstar": int(alert_doy), "dispatch_branch": branch, "with_history": int(with_h),
        "A_score_at_alert": a_score, "D_score_at_alert": d_score, "score_margin": margin,
        "dispatch_score_at_alert": disp_score, "dispatch_tau_used": float(tau_used),
        "score_over_tau_margin": sot_margin, "recent_14d_mean_score": mean_14,
        "recent_28d_mean_score": mean_28, "score_above_tau_streak": int(streak),
        "score_rolling_slope_14d": slope_14, "p_mean_so_far_at_alert": p_mean_so_far,
    }


class PortableBranch:
    """One pest/branch: model.json + calibration.json, loaded torch-free."""

    def __init__(self, pest: str, branch: str, root: Path = ARTIFACTS_ROOT):
        d = Path(root) / pest / branch
        self.pest, self.branch = pest, branch
        self.model = xgboost.XGBClassifier()
        self.model.load_model(str(d / "model.json"))
        self.calibration = json.loads((d / "calibration.json").read_text(encoding="utf-8"))
        self.metadata = json.loads((d / "metadata.json").read_text(encoding="utf-8"))
        self.temperature = float(self.calibration["temperature"])
        fo = self.metadata["feature_name_order"]
        self.add_tpos = bool(fo["tstar_position_feature_appended"])
        self.site_history_added = bool(fo["history_channels_appended"])
        ctx = self.metadata["checkpoint_context"]
        self.doy_start = int(ctx["doy_start"])
        self.doy_end = int(ctx["doy_end"])
        self.window = int(ctx["nowcast_window"])
        self.stride = int(ctx["nowcast_stride"])
        self.only_pre = bool(int(ctx["nowcast_only_pre_event"]))
        self.proxy = str(ctx["nowcast_event_time_proxy"])

    def forward_one(self, base_samples: list[dict], history: dict | None = None) -> dict:
        """Port of infer/stage1.py::_forward_one: history -> nowcast -> tabular ->
        proba -> temperature. Also returns the raw probs so callers can compare
        pre- and post-calibration separately."""
        if self.site_history_added:
            samples = [dict(s, X=append_history(s["X"], s["site_id"], s["year"],
                                                history, self.doy_start))
                       for s in base_samples]
        else:
            samples = base_samples
        nc = build_nowcast_samples(samples, self.window, self.stride, self.only_pre, self.proxy)
        if not nc:
            return {}
        X = build_tabular(nc, self.add_tpos)
        p_raw = self.model.predict_proba(X)[:, 1]
        p_cal = apply_temperature(p_raw, self.temperature)
        per_sy: dict = {}
        for s, praw, pcal in zip(nc, p_raw, p_cal):
            sy = (str(s["site_id"]), int(s["year"]))
            d = per_sy.setdefault(sy, {"ts": [], "ps": [], "raw": []})
            d["ts"].append(int(s["tstar"])); d["ps"].append(float(pcal))
            d["raw"].append(float(praw))
        out = {}
        for sy, d in per_sy.items():
            ts = np.asarray(d["ts"], dtype=int)
            order = np.argsort(ts)
            out[sy] = {"ts": ts[order],
                       "ps": np.asarray(d["ps"], dtype=float)[order],
                       "raw": np.asarray(d["raw"], dtype=float)[order]}
        return out


def load_gate(pest: str, root: Path = ARTIFACTS_ROOT) -> dict:
    return json.loads((Path(root) / pest / "gate.json").read_text(encoding="utf-8"))


def alert_for_sy(gate: dict, per_sy_A: dict, per_sy_D: dict, sy, with_h: bool,
                 doy_start: int) -> dict | None:
    """Port of the gate block in infer/stage1.py::compute_alert_single_sy."""
    A = per_sy_A.get(sy); D = per_sy_D.get(sy)
    if A is None or D is None:
        return None
    method = gate["method"]
    tau_single = gate["tau"]
    tau_no = gate["tau_no"] if gate["tau_no"] is not None else gate["tau"]
    tau_with = gate["tau_with"] if gate["tau_with"] is not None else gate["tau"]
    k = int(gate["k"])
    if method == "A_baseline":
        at = first_crossing_k(A["ts"], A["ps"], tau_single, k)
    elif method == "D_history":
        at = first_crossing_k(D["ts"], D["ps"], tau_single, k)
    else:
        at = (first_crossing_k(D["ts"], D["ps"], tau_with, k) if with_h
              else first_crossing_k(A["ts"], A["ps"], tau_no, k))
    if at is None:
        return None
    feats = dispatch_features_for_sy(sy, at, per_sy_A, per_sy_D, with_h, doy_start,
                                     method, tau_no, tau_with, tau_single)
    return {"alert_tstar_doy": int(feats["alert_tstar"]), "dispatch_features": feats}
