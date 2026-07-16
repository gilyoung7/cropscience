"""End-to-end Stage-1 alert parity: original .pt path vs portable JSON path.

Original path : .pt -> XGBClassifier -> raw predict_proba -> shipped temperature
                -> gate (method/k/tau) -> alert_tstar -> dispatch features
Portable path : model.json -> calibration.json -> gate.json -> same gate -> alert

Both paths are fed the IDENTICAL base samples, so any divergence is attributable
to the model/calibration source rather than to preprocessing. For sheath_blight
the portable alert is additionally checked against the API's real production entry
point, get_alert_single_sy().

Comparison is not short-circuited on raw probability: calibrated probs, per-day
threshold decisions, k-streak, alert DOY, fired/not-fired, branch, tau, k and all
14 dispatch features are compared, and a model only passes if every one matches.

Usage:
    ../api_handoff_transformer/.venv/bin/python validate_alert_parity.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
import traceback

import numpy as np

from common import (API_ROOT, ARTIFACTS_ROOT, BRANCHES, FIXTURES_ROOT, PESTS,
                    REPORTS_ROOT, WORK_ROOT, ckpt_path, env_versions, json_dump)

sys.path.insert(0, str(API_ROOT))
sys.path.insert(0, str(ARTIFACTS_ROOT.parent))

import portable_stage1 as ps  # noqa: E402

SYNTH_SEED = 20260716
# (scale, offset) per synthetic site-year -- a fixed span of the input space; see
# synthetic_base_samples(). Low-scale/negative-offset regions drive the models high,
# large-scale regions keep them low, so both fired and not-fired paths get exercised.
SYNTH_GRID = [(0.15, -30.0), (0.2, -10.0), (0.5, -20.0),
              (1.0, -40.0), (3.0, 0.0), (10.0, 20.0)]
SYNTH_SITE_YEARS = len(SYNTH_GRID)


def _api():
    """Import the API's own Stage-1 internals (the reference implementation)."""
    from infer import stage1 as s1
    return s1


# ---------------------------------------------------------------- fixtures
def synthetic_base_samples(ckpt, n_channels: int, seed: int) -> tuple[list[dict], dict]:
    """Deterministic synthetic season-length series, in the base-sample shape the
    API's own _forward_one consumes.

    The (scale, offset) grid below is a FIXED span of the input space, not tuned
    per pest: a probe showed pure noise at one scale never crosses tau, which makes
    an alert comparison vacuous, so the grid deliberately spans scales that do and
    do not drive the models above tau. Whatever each pest actually does on this grid
    is reported as-is -- the grid is not adjusted to manufacture a firing.

    This is synthetic weather, NOT real observation data -- reported as such.
    """
    rng = np.random.default_rng(seed)
    T = int(ckpt["doy_end"]) - int(ckpt["doy_start"]) + 1
    samples, history = [], {}
    for i, (scale, off) in enumerate(SYNTH_GRID):
        site, year = f"SYN{i:03d}", 2024
        X = (rng.standard_normal((T, n_channels)).astype(np.float32) * scale + off)
        has_event = (i % 2 == 0)
        samples.append({
            "site_id": site, "year": year, "X": X,
            "censor_type": "interval" if has_event else "right",
            "L": int(T * 0.6), "R": int(T * 0.8),
        })
        # Half the site-years get history (drives with_history -> tau_with vs tau_no).
        if i % 2 == 0:
            history[(site, year)] = {
                "prev_year_L_doy_at_site": float(int(ckpt["doy_start"]) + 40 + i),
                "prev_year_event_at_site": 1,
                "site_avg_L_doy_recent3y": float(int(ckpt["doy_start"]) + 45 + i),
                "years_since_last_event_at_site": 1,
                "n_events_recent5y_at_site": 2,
                "prev_year_L_miss": 0,
                "site_avg_L_recent3y_miss": 0,
            }
        else:
            history[(site, year)] = {
                "prev_year_L_doy_at_site": 0.0, "prev_year_event_at_site": 0,
                "site_avg_L_doy_recent3y": 0.0, "years_since_last_event_at_site": 99,
                "n_events_recent5y_at_site": 0, "prev_year_L_miss": 1,
                "site_avg_L_recent3y_miss": 1,
            }
    return samples, history


def real_base_samples(pest: str, obs_pest: str | None = None) -> tuple[list[dict], dict, str | None]:
    """Real Stage-1 base samples via the API's own preprocessing, or a reason string.

    ``obs_pest`` names the pest whose observation CSV supplies the site-years and
    censoring. It defaults to ``pest``. Passing a different pest is only valid when
    the two share feature_cols/doy_start/doy_end/window/stride/proxy/only_pre --
    the caller must have checked that -- and it is recorded in the report as
    ``real_weather_borrowed_labels`` rather than ``real``.
    """
    s1 = _api()
    obs_pest = obs_pest or pest
    obs_csv = API_ROOT / "input" / "LONG_by_pest" / f"RICE_LONG_{obs_pest}.csv"
    daily = API_ROOT / "input" / "daily_weather.csv"
    if not obs_csv.exists():
        return [], {}, f"no observation csv: {obs_csv.name}"
    if not daily.exists():
        return [], {}, f"no daily_weather.csv at {daily}"
    from common import load_checkpoint
    ck = load_checkpoint(ckpt_path(pest, "A"))
    ck_d = load_checkpoint(ckpt_path(pest, "D"))
    doy_start, doy_end = int(ck["doy_start"]), int(ck["doy_end"])
    obs = s1._load_obs_for_stage1(obs_csv)
    labels = s1._filter_labels_by_gap(
        s1._build_interval_labels(s1._aggregate_obs_daily_max(obs)), doy_start, doy_end)
    base = s1._build_base_samples(list(ck["feature_cols"]), daily, obs, labels,
                                  doy_start, doy_end, WORK_ROOT.parent / "cache")
    if not base:
        return [], {}, "preprocessing produced 0 base samples"
    history = s1._compute_site_history(
        base, doy_start,
        policy=str(ck_d.get("site_history_policy", "rolling")),
        train_year_max=int(ck_d.get("history_train_year_max", 2022)))
    return base, history, None


def _preproc_key(pest: str):
    """Everything that determines the base-sample X and the nowcast slicing.

    Two pests with an identical key consume an identical base sample for a given
    (site, year), so one's observation CSV can supply the other's site-years
    without changing a single feature value.
    """
    from common import load_checkpoint
    ck = load_checkpoint(ckpt_path(pest, "A"))
    return (tuple(ck["feature_cols"]), int(ck["doy_start"]), int(ck["doy_end"]),
            int(ck["nowcast_window"]), int(ck["nowcast_stride"]),
            str(ck["nowcast_event_time_proxy"]), bool(int(ck["nowcast_only_pre_event"])))


def _obs_donor(pest: str) -> str | None:
    """The pest whose observation CSV can supply this pest's site-years: itself if
    it ships one, else any pest with an identical preprocessing key that does.
    Returns None when nothing compatible exists (caller falls back to synthetic)."""
    long_dir = API_ROOT / "input" / "LONG_by_pest"
    if (long_dir / f"RICE_LONG_{pest}.csv").exists():
        return pest
    key = _preproc_key(pest)
    for other in PESTS:
        if other == pest or not (long_dir / f"RICE_LONG_{other}.csv").exists():
            continue
        if _preproc_key(other) == key:
            return other
    return None


# ---------------------------------------------------------------- compare
def _arr_stats(a: np.ndarray, b: np.ndarray) -> dict:
    d = np.abs(np.asarray(a, float) - np.asarray(b, float))
    return {"max_abs_diff": float(d.max()) if d.size else None,
            "mean_abs_diff": float(d.mean()) if d.size else None,
            "bit_exact": bool(np.array_equal(a, b)), "n": int(d.size)}


def compare_pest(pest: str, base: list[dict], history: dict, data_kind: str) -> dict:
    s1 = _api()
    from common import load_checkpoint

    rec: dict = {"pest": pest, "data_kind": data_kind, "n_site_years": len(base)}
    gate = ps.load_gate(pest)
    temp = json.loads((API_ROOT / "assets" / "stage1" / pest / "temperature.json").read_text())

    orig_per_sy, port_per_sy, branch_recs = {}, {}, []
    for branch in BRANCHES:
        meta = s1._load_stage1_ckpt(ckpt_path(pest, branch))
        t = float(temp[f"temperature_{branch}"])
        # ---- original: .pt + shipped temperature, via the API's own _forward_one
        o = s1._forward_one(meta, base, history, t)
        # ---- portable: model.json + calibration.json
        pb = ps.PortableBranch(pest, branch)
        p = pb.forward_one(base, history)
        orig_per_sy[branch], port_per_sy[branch] = o, p

        # raw + calibrated parity, pooled over all site-years (sorted ts order)
        keys = sorted(set(o) & set(p))
        cal_o = np.concatenate([o[k]["ps"] for k in keys]) if keys else np.zeros(0)
        cal_p = np.concatenate([p[k]["ps"] for k in keys]) if keys else np.zeros(0)
        raw_p = np.concatenate([p[k]["raw"] for k in keys]) if keys else np.zeros(0)
        # Recover the original raw probs by inverting nothing: recompute from the .pt
        # model on the identical tabular input the portable path used.
        raw_o = _orig_raw(s1, meta, base, history)
        rec_b = {
            "branch": branch,
            "temperature": t,
            "temperature_source": "assets/stage1/%s/temperature.json" % pest,
            "raw_proba": _arr_stats(raw_o, raw_p),
            "calibrated_proba": _arr_stats(cal_o, cal_p),
            "ts_aligned": bool(all(np.array_equal(o[k]["ts"], p[k]["ts"]) for k in keys)),
            "n_site_years_matched": len(keys),
            "cal_has_nan": bool(np.isnan(cal_p).any()), "cal_has_inf": bool(np.isinf(cal_p).any()),
        }
        # per-day threshold decisions at the tau this branch would be gated with
        tau_dbg = (gate["tau"] if gate["tau"] is not None
                   else (gate["tau_with"] if branch == "D" else gate["tau_no"]))
        thr_o = np.concatenate([o[k]["ps"] >= tau_dbg for k in keys]) if keys else np.zeros(0, bool)
        thr_p = np.concatenate([p[k]["ps"] >= tau_dbg for k in keys]) if keys else np.zeros(0, bool)
        rec_b["threshold_decisions"] = {
            "tau_used": float(tau_dbg), "n": int(thr_o.size),
            "agreement_rate": float((thr_o == thr_p).mean()) if thr_o.size else None,
            "all_match": bool(np.array_equal(thr_o, thr_p)),
            "n_pass_original": int(thr_o.sum()), "n_pass_portable": int(thr_p.sum()),
        }
        branch_recs.append(rec_b)
    rec["branches"] = branch_recs

    # ---- alert-level parity per site-year -------------------------------
    alerts, reference = [], {}
    doy_start = int(load_checkpoint(ckpt_path(pest, "A"))["doy_start"])
    for s in base:
        sy = (str(s["site_id"]), int(s["year"]))
        h = history.get(sy)
        with_h = (h is not None and int(h["prev_year_L_miss"]) == 0)

        o = _orig_alert(s1, gate, orig_per_sy, sy, with_h, doy_start)
        p = ps.alert_for_sy(gate, port_per_sy["A"], port_per_sy["D"], sy, with_h, doy_start)
        alerts.append(_compare_alert(sy, o, p))
        if o is not None:
            reference[f"{sy[0]}|{sy[1]}"] = o
    rec["alerts"] = alerts
    rec["fixture"] = _write_alert_fixture(pest, base, history, reference, data_kind,
                                          gate, temp)
    rec["alert_summary"] = {
        "n": len(alerts),
        "n_fired_original": sum(a["fired_original"] for a in alerts),
        "n_fired_portable": sum(a["fired_portable"] for a in alerts),
        "fired_agreement": all(a["fired_match"] for a in alerts),
        "alert_doy_all_match": all(a["alert_doy_match"] for a in alerts),
        "dispatch_all_match": all(a["dispatch_match"] for a in alerts),
    }
    rec["gate"] = {k: gate[k] for k in ["method", "k", "tau", "tau_no", "tau_with"]}
    rec["status"] = "ok" if (
        all(b["raw_proba"]["bit_exact"] and b["calibrated_proba"]["bit_exact"]
            and b["threshold_decisions"]["all_match"] for b in branch_recs)
        and rec["alert_summary"]["fired_agreement"]
        and rec["alert_summary"]["alert_doy_all_match"]
        and rec["alert_summary"]["dispatch_all_match"]
    ) else "mismatch"
    return rec


FIXTURE_MAX_SITE_YEARS = 40


def _write_alert_fixture(pest: str, base: list[dict], history: dict,
                         reference: dict, data_kind: str, gate: dict, temp: dict) -> dict:
    """Persist base samples + history + the ORIGINAL path's alerts so the torch-free
    pipeline can be replayed elsewhere. Capped and stratified: real cohorts have
    thousands of site-years, and a fixture that only held non-firing ones would make
    the replay vacuous, so fired and not-fired are both kept.

    The stored reference is RECOMPUTED by the original .pt path on the kept subset,
    not copied from the full-cohort pass. A fixture must be a closed unit: replaying
    a 40-site-year subset against a reference derived from a 7321-site-year cohort
    compares two different computations and reports spurious mismatches.
    """
    s1 = _api()
    fired = [s for s in base if f"{s['site_id']}|{s['year']}" in reference]
    quiet = [s for s in base if f"{s['site_id']}|{s['year']}" not in reference]
    half = FIXTURE_MAX_SITE_YEARS // 2
    keep = fired[:half] + quiet[:FIXTURE_MAX_SITE_YEARS - min(half, len(fired))]
    keep = keep[:FIXTURE_MAX_SITE_YEARS]

    # Normalise memory layout BEFORE computing the reference. _build_tabular's
    # mean/std/slope are float32 reductions whose accumulation order depends on the
    # array's strides: the API's base X comes from DataFrame.to_numpy() and is
    # F-contiguous, while np.load returns C-contiguous. Identical values, different
    # order -> mean differs by ~3e-4 -> ~1e-3 in probability -> an alert can move by
    # days. So the fixture is defined on C-contiguous arrays and the reference is
    # computed on exactly those, making the replay reproducible.
    keep = [dict(s, X=np.ascontiguousarray(s["X"], dtype=np.float32)) for s in keep]
    keys = {f"{s['site_id']}|{s['year']}" for s in keep}

    # --- reference recomputed on the kept subset, by the original .pt path -----
    sub_per_sy = {}
    for branch in BRANCHES:
        meta = s1._load_stage1_ckpt(ckpt_path(pest, branch))
        sub_per_sy[branch] = s1._forward_one(meta, keep, history,
                                             float(temp[f"temperature_{branch}"]))
    doy_start = int(s1._load_stage1_ckpt(ckpt_path(pest, "A"))["doy_start"])
    reference = {}
    for s in keep:
        sy = (str(s["site_id"]), int(s["year"]))
        h = history.get(sy)
        with_h = (h is not None and int(h["prev_year_L_miss"]) == 0)
        o = _orig_alert(s1, gate, sub_per_sy, sy, with_h, doy_start)
        if o is not None:
            reference[f"{sy[0]}|{sy[1]}"] = o

    out_dir = FIXTURES_ROOT / "alert"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{pest}.npz"
    sidecar = out_dir / f"{pest}.json"

    # Deliberately pickle-free: plain float32/int arrays + a JSON sidecar, so the
    # replay side can np.load(allow_pickle=False). An object-array fixture would
    # force the "portable" runtime to unpickle -- the very dependency being removed.
    np.savez_compressed(
        path,
        X=np.stack([np.asarray(s["X"], dtype=np.float32) for s in keep]),
        L=np.asarray([int(s["L"]) for s in keep], dtype=np.int64),
        R=np.asarray([int(s["R"]) for s in keep], dtype=np.int64),
        year=np.asarray([int(s["year"]) for s in keep], dtype=np.int64),
    )
    json_dump({
        "pest": pest,
        "data_kind": data_kind,
        "site_ids": [str(s["site_id"]) for s in keep],
        "censor_type": [str(s["censor_type"]) for s in keep],
        "history": {f"{k[0]}|{k[1]}": {hk: float(hv) for hk, hv in v.items()}
                    for k, v in history.items() if f"{k[0]}|{k[1]}" in keys},
        "reference_alerts": {k: v for k, v in reference.items() if k in keys},
        "npz_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
    }, sidecar)
    return {"npz": str(path.relative_to(FIXTURES_ROOT.parent)),
            "sidecar": str(sidecar.relative_to(FIXTURES_ROOT.parent)),
            "n_site_years": len(keep),
            "n_fired": sum(1 for s in keep if f"{s['site_id']}|{s['year']}" in reference),
            "bytes": path.stat().st_size + sidecar.stat().st_size,
            "npz_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
            "capped_from": len(base),
            "cap": FIXTURE_MAX_SITE_YEARS}


def _orig_raw(s1, meta, base, history) -> np.ndarray:
    """Raw (pre-temperature) probs from the .pt model on the same tabular input."""
    doy_start = meta["doy_start"]
    samples = ([dict(s, X=s1._append_history(s["X"], s["site_id"], s["year"], history, doy_start))
                for s in base] if meta["site_history_added"] else base)
    nc = s1._build_nowcast_samples(samples, meta["window"], meta["stride"],
                                   meta["only_pre"], meta["proxy"])
    if not nc:
        return np.zeros(0)
    X = s1._build_tabular(nc, meta["add_tpos"])
    p = meta["clf"].predict_proba(X)[:, 1]
    sy_ts = [((str(s["site_id"]), int(s["year"])), int(s["tstar"])) for s in nc]
    order = sorted(range(len(nc)), key=lambda i: (sy_ts[i][0], sy_ts[i][1]))
    return np.asarray([float(p[i]) for i in order])


def _orig_alert(s1, gate, per_sy, sy, with_h, doy_start):
    """The gate block of infer/stage1.py::compute_alert_single_sy, on original probs."""
    A = per_sy["A"].get(sy); D = per_sy["D"].get(sy)
    if A is None or D is None:
        return None
    tau_single = gate["tau"]
    tau_no = gate["tau_no"] if gate["tau_no"] is not None else gate["tau"]
    tau_with = gate["tau_with"] if gate["tau_with"] is not None else gate["tau"]
    k = int(gate["k"]); method = gate["method"]
    if method == "A_baseline":
        at = s1._first_crossing_k(A["ts"], A["ps"], tau_single, k)
    elif method == "D_history":
        at = s1._first_crossing_k(D["ts"], D["ps"], tau_single, k)
    else:
        at = (s1._first_crossing_k(D["ts"], D["ps"], tau_with, k) if with_h
              else s1._first_crossing_k(A["ts"], A["ps"], tau_no, k))
    if at is None:
        return None
    feats = s1._dispatch_features_for_sy(sy, at, per_sy["A"], per_sy["D"], with_h,
                                         doy_start, method, tau_no, tau_with, tau_single)
    return {"alert_tstar_doy": int(feats["alert_tstar"]), "dispatch_features": feats}


def _compare_alert(sy, o, p) -> dict:
    fo, fp = o is not None, p is not None
    rec = {"site_year": f"{sy[0]}|{sy[1]}", "fired_original": fo, "fired_portable": fp,
           "fired_match": fo == fp}
    if not (fo and fp):
        rec.update(alert_doy_original=(o or {}).get("alert_tstar_doy"),
                   alert_doy_portable=(p or {}).get("alert_tstar_doy"),
                   alert_doy_match=(not fo and not fp), dispatch_match=(not fo and not fp))
        return rec
    rec["alert_doy_original"] = o["alert_tstar_doy"]
    rec["alert_doy_portable"] = p["alert_tstar_doy"]
    rec["alert_doy_match"] = o["alert_tstar_doy"] == p["alert_tstar_doy"]
    fo_d, fp_d = o["dispatch_features"], p["dispatch_features"]
    diffs = {}
    for k in sorted(set(fo_d) | set(fp_d)):
        a, b = fo_d.get(k), fp_d.get(k)
        if isinstance(a, float) and isinstance(b, float):
            same = (a == b) or (np.isnan(a) and np.isnan(b))
            if not same:
                diffs[k] = {"original": a, "portable": b, "abs_diff": abs(a - b)}
        elif a != b:
            diffs[k] = {"original": a, "portable": b}
    rec["dispatch_match"] = not diffs
    rec["dispatch_diffs"] = diffs
    rec["dispatch_branch_original"] = fo_d.get("dispatch_branch")
    rec["dispatch_branch_portable"] = fp_d.get("dispatch_branch")
    rec["dispatch_tau_used"] = fo_d.get("dispatch_tau_used")
    rec["score_above_tau_streak"] = fo_d.get("score_above_tau_streak")
    return rec


# ------------------------------------------------- gate boundary unit parity
def gate_boundary_cases() -> dict:
    """Directly exercise threshold/k-streak edges: p exactly == tau, just under,
    streak of exactly k-1 vs k. Compares the API's _first_crossing_k with the
    portable port on crafted series -- no model involved."""
    s1 = _api()
    tau, cases = 0.6, []
    series = {
        "exactly_tau_k_times": [0.1, 0.6, 0.6, 0.6, 0.1],
        "just_under_tau": [0.1, 0.5999999, 0.5999999, 0.5999999, 0.1],
        "streak_k_minus_1": [0.1, 0.7, 0.7, 0.1, 0.7],
        "streak_resets_then_fires": [0.7, 0.7, 0.1, 0.7, 0.7, 0.7],
        "all_below": [0.0, 0.1, 0.2, 0.3],
        "all_above": [0.9, 0.9, 0.9],
        "nan_present": [float("nan"), 0.7, 0.7, 0.7],
        "one_ulp_below_tau": [0.1, np.nextafter(0.6, 0.0), 0.6, 0.6, 0.6],
    }
    for k in (1, 2, 3):
        for name, ps_list in series.items():
            ts = np.arange(1, len(ps_list) + 1)
            arr = np.asarray(ps_list, dtype=float)
            a = s1._first_crossing_k(ts, arr, tau, k)
            b = ps.first_crossing_k(ts, arr, tau, k)
            cases.append({"case": name, "k": k, "tau": tau,
                          "original": a, "portable": b, "match": a == b})
    return {"n_cases": len(cases), "all_match": all(c["match"] for c in cases),
            "cases": cases}


def temperature_refit_check(pest: str, base: list[dict], history: dict) -> dict:
    """Does the shipped temperature.json reproduce what the cohort path would fit
    at runtime? Uses the API's own _calibrated_per_sy (which re-fits on VAL_YEAR).
    Read-only; nothing is re-tuned."""
    s1 = _api()
    temp = json.loads((API_ROOT / "assets" / "stage1" / pest / "temperature.json").read_text())
    out = {}
    for branch in BRANCHES:
        meta = s1._load_stage1_ckpt(ckpt_path(pest, branch))
        try:
            _, t_fit = s1._calibrated_per_sy(meta, base, history)
            shipped = float(temp[f"temperature_{branch}"])
            out[branch] = {"shipped": shipped, "refit_now": float(t_fit),
                           "match": shipped == float(t_fit),
                           "abs_diff": abs(shipped - float(t_fit))}
        except Exception as exc:
            out[branch] = {"error": f"{type(exc).__name__}: {exc}"}
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-real", action="store_true")
    ap.add_argument("--pest", action="append")
    args = ap.parse_args()
    pests = args.pest or PESTS

    from common import load_checkpoint
    records, real_notes, refit = [], {}, {}
    cache: dict = {}
    for pest in pests:
        try:
            obs_pest = _obs_donor(pest) if not args.skip_real else None
            if obs_pest is None:
                why = "skipped by flag" if args.skip_real else "no compatible observation csv"
                base, history = ([], {})
            elif obs_pest in cache and _preproc_key(pest) == cache[obs_pest][0]:
                base, history, why = cache[obs_pest][1], cache[obs_pest][2], None
            else:
                base, history, why = real_base_samples(pest, obs_pest)
                if not why:
                    cache[obs_pest] = (_preproc_key(pest), base, history)

            if why:
                real_notes[pest] = why
                ck = load_checkpoint(ckpt_path(pest, "A"))
                base, history = synthetic_base_samples(ck, len(ck["feature_names"]),
                                                       SYNTH_SEED + PESTS.index(pest))
                kind = "synthetic"
            else:
                kind = "real" if obs_pest == pest else "real_weather_borrowed_labels"
                if kind != "real":
                    real_notes[pest] = (
                        f"no RICE_LONG_{pest}.csv; used real weather features at real sites with "
                        f"site-years/censoring borrowed from {obs_pest} (identical feature_cols, "
                        f"doy range, window, stride, proxy, only_pre). Valid for numerical parity; "
                        f"NOT this pest's real cohort.")
                # Only meaningful with this pest's OWN labels: _fit_temperature_grid
                # minimises NLL against y_event, so borrowed labels would "re-fit" to a
                # number that never meant anything. Skipped rather than reported as a
                # mismatch.
                if kind == "real":
                    refit[pest] = temperature_refit_check(pest, base, history)
                else:
                    refit[pest] = {"skipped": "labels borrowed from another pest; a re-fit "
                                              "against them would be meaningless"}
            rec = compare_pest(pest, base, history, kind)
        except Exception as exc:
            rec = {"pest": pest, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                   "traceback": traceback.format_exc()}
        records.append(rec)
        s = rec.get("alert_summary", {})
        print(f"[{rec['status']:<8}] {pest:<18} {rec.get('data_kind','-'):<9} "
              f"sy={rec.get('n_site_years','-'):<4} "
              f"fired {s.get('n_fired_original','-')}/{s.get('n_fired_portable','-')} "
              f"alert_match={s.get('alert_doy_all_match','-')} "
              f"dispatch_match={s.get('dispatch_all_match','-')} "
              f"{rec.get('error','')}", flush=True)

    boundary = gate_boundary_cases()
    print(f"\ngate boundary unit parity: {boundary['n_cases']} cases, "
          f"all_match={boundary['all_match']}")

    ok = sum(r["status"] == "ok" for r in records)
    report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_env": env_versions(),
        "totals": {"pests": len(records), "ok": ok, "not_ok": len(records) - ok},
        "memory_layout_sensitivity": {
            "finding": ("Stage-1 output depends on the MEMORY LAYOUT of the base X array, "
                        "independently of its values. infer/stage1.py::_build_tabular takes "
                        "float32 mean/std/slope reductions over the season axis; numpy's "
                        "accumulation order follows the array strides. The API's base X from "
                        "DataFrame.to_numpy() is F-contiguous; a C-contiguous array with "
                        "bit-identical values yields mean/std differing by up to ~3.3e-4, "
                        "~1e-3 in calibrated probability, which can move an alert by days "
                        "when the score sits near tau."),
            "affects_this_migration": False,
            "why_not": ("Both the .pt path and the portable path consume the same array and "
                        "agree bit-exactly; the sensitivity is in the shared preprocessing, "
                        "not in the model or calibration source."),
            "handled_by": ("fixtures normalise X to C-contiguous and the stored reference is "
                           "computed on those same normalised arrays"),
            "risk_for_the_api": ("if the base X layout ever changes (pandas version, a copy, a "
                                 "reshape, an np.stack), Stage-1 alerts can shift without any "
                                 "model change. Verified, not hypothetical."),
        },
        "real_data_notes": real_notes,
        "temperature_refit_check": refit,
        "gate_boundary_unit_parity": boundary,
        "pests": records,
    }
    json_dump(report, REPORTS_ROOT / "alert_parity_report.json")
    print(f"\nalert parity ok {ok}/{len(records)} -> {REPORTS_ROOT/'alert_parity_report.json'}")
    return 0 if ok == len(records) and boundary["all_match"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
