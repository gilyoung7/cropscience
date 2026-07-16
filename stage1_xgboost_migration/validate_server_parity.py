"""Compare predict_proba between the XGBClassifier pickled inside each .pt and a
fresh XGBClassifier reloaded from the exported model.json, in the source env.

Also writes the fixtures + reference predictions that validate_modern_xgboost.py
replays on a newer XGBoost install.

Usage:
    ../api_handoff_transformer/.venv/bin/python validate_server_parity.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import traceback

import numpy as np

from common import (ARTIFACTS_ROOT, BRANCHES, FIXTURES_ROOT, PESTS, REPORTS_ROOT, WORK_ROOT,
                    SCHEMA_VERSION, ckpt_path, derive_feature_names, env_versions,
                    json_dump, load_checkpoint, sha256_file)

SYNTHETIC_ROWS = 64
SYNTHETIC_SEED = 20260716


def make_synthetic(n_features: int, seed: int, n_rows: int = SYNTHETIC_ROWS) -> np.ndarray:
    """Deterministic feature matrix. Fixed seed + explicit dtype so the exact same
    bytes are reproducible on any machine (see README for the regeneration cmd)."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_rows, n_features)).astype(np.float32) * 10.0
    # A few structured rows so the trees see edge inputs, not just noise.
    X[0, :] = 0.0
    X[1, :] = np.arange(n_features, dtype=np.float32)
    X[2, :] = -np.arange(n_features, dtype=np.float32)
    return np.ascontiguousarray(X, dtype=np.float32)


def compare(p_src: np.ndarray, p_json: np.ndarray) -> dict:
    diff = np.abs(p_src.astype(np.float64) - p_json.astype(np.float64))
    cls_src = p_src.argmax(axis=1)
    cls_json = p_json.argmax(axis=1)
    return {
        "shape_source": list(p_src.shape),
        "shape_json": list(p_json.shape),
        "dtype_source": str(p_src.dtype),
        "dtype_json": str(p_json.dtype),
        "shape_match": p_src.shape == p_json.shape,
        "dtype_match": p_src.dtype == p_json.dtype,
        "max_abs_diff": float(diff.max()) if diff.size else None,
        "mean_abs_diff": float(diff.mean()) if diff.size else None,
        "bit_exact": bool(np.array_equal(p_src, p_json)),
        "class_agreement_rate": float((cls_src == cls_json).mean()) if diff.size else None,
        "n_rows": int(p_src.shape[0]),
        "source_has_nan": bool(np.isnan(p_src).any()),
        "source_has_inf": bool(np.isinf(p_src).any()),
        "json_has_nan": bool(np.isnan(p_json).any()),
        "json_has_inf": bool(np.isinf(p_json).any()),
        "source_proba_head": p_src[:3].tolist(),
        "json_proba_head": p_json[:3].tolist(),
    }


def validate_one(pest: str, branch: str, real_X: np.ndarray | None) -> dict:
    import xgboost

    rec: dict = {"pest": pest, "branch": branch}
    model_json = ARTIFACTS_ROOT / pest / branch / "model.json"
    src = ckpt_path(pest, branch)
    if not model_json.exists():
        rec.update(status="skipped", error=f"missing export: {model_json}")
        return rec

    try:
        ckpt = load_checkpoint(src)
        source_model = ckpt["trained_states"][0]["sk_model"]
        feature_names = derive_feature_names(ckpt)
        n_features = int(source_model.n_features_in_)

        # Reload into a brand-new estimator, as the task requires.
        loaded = xgboost.XGBClassifier()
        loaded.load_model(str(model_json))
        rec["reload_ok"] = True
        rec["reloaded_n_features_in_"] = int(getattr(loaded, "n_features_in_", -1))
        rec["reloaded_classes_"] = [int(c) for c in getattr(loaded, "classes_", [])]
        rec["n_features_in_"] = n_features
        rec["classes_match"] = (rec["reloaded_classes_"]
                                == [int(c) for c in source_model.classes_])

        cases = {}
        X_syn = make_synthetic(n_features, SYNTHETIC_SEED + abs(hash((pest, branch))) % 1000)
        cases["synthetic"] = X_syn
        if real_X is not None and real_X.shape[1] == n_features:
            cases["real"] = real_X
        elif real_X is not None:
            rec["real_skip_reason"] = (f"real matrix has {real_X.shape[1]} cols, "
                                       f"model expects {n_features}")

        results = {}
        for name, X in cases.items():
            p_src = source_model.predict_proba(X)
            p_json = loaded.predict_proba(X)
            res = compare(p_src, p_json)
            res["input_sha256"] = None
            results[name] = res

            fx_dir = FIXTURES_ROOT / pest / branch
            fx_dir.mkdir(parents=True, exist_ok=True)
            np.save(fx_dir / f"X_{name}.npy", X)
            np.save(fx_dir / f"proba_source_{name}.npy", p_src)
            res["input_sha256"] = sha256_file(fx_dir / f"X_{name}.npy")
            res["reference_proba_sha256"] = sha256_file(fx_dir / f"proba_source_{name}.npy")
            res["fixture_X"] = str((fx_dir / f"X_{name}.npy").relative_to(FIXTURES_ROOT))
            res["n_features"] = n_features

        rec["feature_names_len"] = len(feature_names)
        rec["cases"] = results
        rec["status"] = "ok" if all(
            r["bit_exact"] for r in results.values()) else "mismatch"
    except Exception as exc:
        rec.update(status="failed", error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc())
    return rec


def build_real_matrices() -> dict:
    """Build a genuine Stage-1 tabular matrix per pest using the API's own
    preprocessing. Returns {pest: ndarray}; pests whose inputs are absent are
    simply missing from the dict (recorded, never fabricated)."""
    import sys
    from common import API_ROOT
    sys.path.insert(0, str(API_ROOT))
    out: dict = {}
    notes: dict = {}
    try:
        from infer.stage1 import (_aggregate_obs_daily_max, _build_base_samples,
                                  _build_interval_labels, _filter_labels_by_gap,
                                  _load_obs_for_stage1)
    except Exception as exc:
        return {"_error": f"cannot import API preprocessing: {type(exc).__name__}: {exc}"}

    long_dir = API_ROOT / "input" / "LONG_by_pest"
    daily = API_ROOT / "input" / "daily_weather.csv"
    if not daily.exists():
        return {"_error": f"no daily_weather.csv at {daily}"}

    for pest in PESTS:
        obs_csv = long_dir / f"RICE_LONG_{pest}.csv"
        if not obs_csv.exists():
            notes[pest] = f"no observation csv: {obs_csv.name}"
            continue
        try:
            # Mirrors compute_stage1_table's preprocessing sequence exactly.
            ckpt = load_checkpoint(ckpt_path(pest, "A"))
            doy_start, doy_end = int(ckpt["doy_start"]), int(ckpt["doy_end"])
            obs = _load_obs_for_stage1(obs_csv)
            obs2 = _aggregate_obs_daily_max(obs)
            labels = _filter_labels_by_gap(_build_interval_labels(obs2),
                                           doy_start, doy_end)
            # Cache lives under _work/, not fixtures/: it is a multi-GB
            # preprocessing scratch area and must not be swept up when fixtures/
            # is copied to another machine.
            base = _build_base_samples(list(ckpt["feature_cols"]), daily, obs, labels,
                                       doy_start, doy_end, WORK_ROOT.parent / "cache")
            if not base:
                notes[pest] = "preprocessing produced 0 base samples"
                continue
            out[pest] = {"base": base, "ckpt": ckpt}
        except Exception as exc:
            notes[pest] = f"{type(exc).__name__}: {exc}"
    out["_notes"] = notes
    return out


def real_matrix_for(realdata: dict, pest: str, branch: str) -> np.ndarray | None:
    """Tabularise the cached base samples for one pest/branch exactly as the API does."""
    import sys
    from common import API_ROOT
    sys.path.insert(0, str(API_ROOT))
    from infer.stage1 import (_append_history, _build_nowcast_samples, _build_tabular,
                              _compute_site_history)

    entry = realdata.get(pest)
    if not isinstance(entry, dict) or "base" not in entry:
        return None
    ckpt = load_checkpoint(ckpt_path(pest, branch))
    base = entry["base"]
    samples = base
    if bool(ckpt.get("site_history_added", False)):
        history = _compute_site_history(
            base, int(ckpt["doy_start"]),
            policy=str(ckpt.get("site_history_policy", "rolling")),
            train_year_max=int(ckpt.get("history_train_year_max", 2022)))
        samples = [dict(s, X=_append_history(s["X"], s["site_id"], s["year"], history,
                                             int(ckpt["doy_start"]))) for s in base]
    now = _build_nowcast_samples(samples, int(ckpt["nowcast_window"]),
                                 int(ckpt["nowcast_stride"]),
                                 bool(ckpt.get("nowcast_only_pre_event", False)),
                                 str(ckpt.get("nowcast_event_time_proxy", "mid")))
    if not now:
        return None
    X = _build_tabular(now, bool(ckpt.get("add_tstar_position_feature", False)))
    return X[:SYNTHETIC_ROWS] if X.shape[0] > SYNTHETIC_ROWS else X


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-real", action="store_true")
    args = ap.parse_args()

    realdata: dict = {}
    real_notes: dict = {}
    if not args.skip_real:
        realdata = build_real_matrices()
        real_notes = realdata.get("_notes", {})
        if "_error" in realdata:
            real_notes = {"_error": realdata["_error"]}
        print(f"real-data preprocessing: available for "
              f"{[p for p in realdata if not p.startswith('_')]}", flush=True)

    records = []
    for pest in PESTS:
        for branch in BRANCHES:
            real_X = None
            if pest in realdata and not args.skip_real:
                try:
                    real_X = real_matrix_for(realdata, pest, branch)
                except Exception as exc:
                    real_notes[f"{pest}/{branch}"] = f"{type(exc).__name__}: {exc}"
            rec = validate_one(pest, branch, real_X)
            records.append(rec)
            cases = rec.get("cases", {})
            summary = "  ".join(
                f"{k}: max|d|={v['max_abs_diff']:.3e} exact={v['bit_exact']} "
                f"agree={v['class_agreement_rate']:.3f}"
                for k, v in cases.items())
            print(f"[{rec['status']:<8}] {pest:<18} {branch}  {summary or rec.get('error','')}",
                  flush=True)

    ok = sum(r["status"] == "ok" for r in records)
    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "source_env": env_versions(),
        "synthetic": {"rows": SYNTHETIC_ROWS, "base_seed": SYNTHETIC_SEED,
                      "generator": "numpy.random.default_rng, float32"},
        "real_data_notes": real_notes,
        "totals": {"attempted": len(records), "bit_exact": ok,
                   "not_bit_exact": len(records) - ok},
        "models": records,
    }
    json_dump(report, REPORTS_ROOT / "server_parity_report.json")
    print(f"\nbit-exact {ok}/{len(records)} -> {REPORTS_ROOT/'server_parity_report.json'}")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
