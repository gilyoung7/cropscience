"""Stage-1 alert gate — portable JSON assets, no .pt, no torch, no scikit-learn.

Loads `model.json` with `xgboost.Booster` directly rather than `XGBClassifier`,
which removes the scikit-learn dependency entirely.

Booster was adopted only after proving equivalence: Booster.predict ==
XGBClassifier.predict_proba[:,1] **bit-exact (max|diff| = 0) on 48/48 combos**
(16 models x {C-contiguous, F-contiguous, strided view}). See
docs/lightweight_api_integration_report.md.

It also fixes a real breakage: `XGBClassifier.load_model()` raises
`TypeError: _estimator_type undefined` under xgboost 2.1.4 + scikit-learn 1.9.0,
so the XGBClassifier path cannot load these models on a current environment.

Gate parameters (method/k/tau) come from the frozen authoritative gate JSON —
the same values `resolve_gate_params` reads out of
`group_tau_hybrid_summary.json` (stage1.py:124-141), NOT from
stage1_selected_gates.yaml, whose k/tau have drifted for bacterial_blight and
rice_stem_borer_1.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import xgboost

from .stage1_features import (
    HISTORY_STATIC_NAMES,
    Stage1FeatureError,
    append_history,
    apply_temperature,
    build_nowcast_samples,
    build_tabular,
    first_crossing_k,
)


class Stage1Error(RuntimeError):
    """Stage-1 could not produce an alert. Surfaced, never silent."""


class PortableBranch:
    """One pest/branch: model.json + calibration.json + metadata.json."""

    def __init__(self, pest: str, branch: str, root: Path):
        d = Path(root) / pest / branch
        for f in ("model.json", "calibration.json", "metadata.json"):
            if not (d / f).is_file():
                raise Stage1Error(f"[{pest}/{branch}] missing asset: {d / f}")
        self.pest, self.branch = pest, branch
        # Booster, not XGBClassifier -> no scikit-learn import.
        self.booster = xgboost.Booster()
        self.booster.load_model(str(d / "model.json"))
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
        # The raw daily columns the checkpoint was built from (before the
        # __miss block and the history append) — metadata key is
        # `checkpoint_feature_cols`, mirroring ckpt["feature_cols"].
        self.feature_cols = list(self.metadata["checkpoint_feature_cols"])

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw probability. Equivalent to XGBClassifier.predict_proba(X)[:, 1]."""
        n = self.booster.num_features()
        if X.shape[1] != n:
            raise Stage1Error(
                f"[{self.pest}/{self.branch}] feature width {X.shape[1]} != model {n}"
            )
        # DMatrix does not alter the caller's array; layout is preserved upstream.
        return np.asarray(self.booster.predict(xgboost.DMatrix(X)), dtype=float)

    def forward_one(self, base_samples: list[dict], history: dict | None) -> dict:
        """Port of stage1.py::_forward_one — history -> nowcast -> tabular ->
        raw proba -> fixed temperature. Returns per-(site, year) series including
        the raw probabilities so callers can compare pre/post calibration."""
        if self.site_history_added:
            samples = [
                dict(s, X=append_history(s["X"], s["site_id"], s["year"], history,
                                         self.doy_start))
                for s in base_samples
            ]
        else:
            samples = base_samples
        nc = build_nowcast_samples(samples, self.window, self.stride, self.only_pre,
                                   self.proxy)
        if not nc:
            return {}
        X = build_tabular(nc, self.add_tpos)
        p_raw = self.predict_raw(X)
        p_cal = apply_temperature(p_raw, self.temperature)
        per_sy: dict = {}
        for s, praw, pcal in zip(nc, p_raw, p_cal):
            sy = (str(s["site_id"]), int(s["year"]))
            d = per_sy.setdefault(sy, {"ts": [], "ps": [], "raw": []})
            d["ts"].append(int(s["tstar"]))
            d["ps"].append(float(pcal))
            d["raw"].append(float(praw))
        out = {}
        for sy, d in per_sy.items():
            ts = np.asarray(d["ts"], dtype=int)
            order = np.argsort(ts)
            out[sy] = {
                "ts": ts[order],
                "ps": np.asarray(d["ps"], dtype=float)[order],
                "raw": np.asarray(d["raw"], dtype=float)[order],
            }
        return out


def load_gate(pest: str, root: Path) -> dict:
    """Frozen authoritative method/k/tau (from group_tau_hybrid_summary.json)."""
    p = Path(root) / pest / "gate.json"
    if not p.is_file():
        raise Stage1Error(f"[{pest}] missing gate.json: {p}")
    g = json.loads(p.read_text(encoding="utf-8"))
    for key in ("method", "k"):
        if key not in g:
            raise Stage1Error(f"[{pest}] gate.json missing {key!r}")
    return g


def dispatch_features_for_sy(alert_t, per_sy_A, per_sy_D, with_h, doy_start,
                             gate_method, tau_no, tau_with, tau_single) -> dict:
    """Port of stage1.py::_dispatch_features_for_sy — the 14 dispatch features."""
    A, D = per_sy_A, per_sy_D

    def _at(series, t):
        if series is None:
            return float("nan")
        idx = np.where(series["ts"] == int(t))[0]
        return float(series["ps"][int(idx[0])]) if idx.size else float("nan")

    a_score, d_score = _at(A, alert_t), _at(D, alert_t)
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
            streak += 1
            i -= 1
    if mask_14.sum() >= 2:
        slope_14 = float(np.polyfit(doys[mask_14].astype(float),
                                    d_ps[mask_14].astype(float), 1)[0])
    else:
        slope_14 = float("nan")
    mask_cum = d_ts <= int(alert_t)
    p_mean_so_far = float(d_ps[mask_cum].mean()) if mask_cum.any() else float("nan")
    return {
        "alert_tstar": int(alert_doy), "dispatch_branch": branch,
        "with_history": int(with_h),
        "A_score_at_alert": a_score, "D_score_at_alert": d_score,
        "score_margin": margin, "dispatch_score_at_alert": disp_score,
        "dispatch_tau_used": float(tau_used), "score_over_tau_margin": sot_margin,
        "recent_14d_mean_score": mean_14, "recent_28d_mean_score": mean_28,
        "score_above_tau_streak": int(streak), "score_rolling_slope_14d": slope_14,
        "p_mean_so_far_at_alert": p_mean_so_far,
    }


def alert_from_series(gate: dict, A: dict, D: dict, with_h: bool,
                      doy_start: int) -> dict | None:
    """Port of the gate block in stage1.py::compute_alert_single_sy.

    Returns None for no-alert (the deployed no-alert contract), else
    {alert_tstar_doy, dispatch_features, gate_used}.
    """
    if A is None or D is None:
        return None
    method = gate["method"]
    tau_single = gate.get("tau")
    tau_no = gate["tau_no"] if gate.get("tau_no") is not None else gate.get("tau")
    tau_with = gate["tau_with"] if gate.get("tau_with") is not None else gate.get("tau")
    k = int(gate["k"])
    if method == "A_baseline":
        at = first_crossing_k(A["ts"], A["ps"], tau_single, k)
    elif method == "D_history":
        at = first_crossing_k(D["ts"], D["ps"], tau_single, k)
    elif method == "dispatch_group_tau":
        at = (first_crossing_k(D["ts"], D["ps"], tau_with, k) if with_h
              else first_crossing_k(A["ts"], A["ps"], tau_no, k))
    else:
        raise Stage1Error(f"unknown gate method {method!r}")
    if at is None:
        return None
    feats = dispatch_features_for_sy(at, A, D, with_h, doy_start, method,
                                     tau_no, tau_with, tau_single)
    return {
        "alert_tstar_doy": int(feats["alert_tstar"]),
        "dispatch_features": feats,
        "gate_used": {"method": method, "k": k, "tau": tau_single,
                      "tau_no": tau_no, "tau_with": tau_with},
    }


def resolve_with_history(site_history: dict, site_id: str, year: int) -> tuple[bool, dict | None]:
    """Port of stage1.py:802-831 site_history lookup + with_h derivation.

    Returns (with_h, history_row). Raises when the site-year is absent: the
    deployed path re-derives history from the request's own prior-year obs rows
    and raises if there are none. This package takes history as a provided asset,
    so an absent key is an explicit error rather than a silent zero-fill.
    """
    key = f"{site_id}|{int(year)}"
    if key not in site_history:
        raise Stage1Error(
            f"site_history.json has no entry for {key!r}. The deployed API would "
            f"re-derive history from the request's prior-year observations; this "
            f"package requires the shipped site_history asset to cover the "
            f"site-year. Nothing is defaulted to zero."
        )
    vals = site_history[key]
    if len(vals) != len(HISTORY_STATIC_NAMES):
        raise Stage1Error(
            f"site_history[{key!r}] has {len(vals)} values, expected "
            f"{len(HISTORY_STATIC_NAMES)}"
        )
    h = {n: v for n, v in zip(HISTORY_STATIC_NAMES, vals)}
    with_h = float(h["prev_year_L_miss"]) == 0.0
    return with_h, h
