"""Export the Stage-1 XGBClassifier out of each .pt checkpoint into a portable
XGBoost JSON model plus a metadata sidecar.

The source .pt checkpoints are opened read-only and never rewritten. No model is
re-serialised via pickle/joblib/torch.save: the only export path is
XGBClassifier.save_model(*.json).

Usage:
    ../api_handoff_transformer/.venv/bin/python export_portable_models.py
"""
from __future__ import annotations

import argparse
import datetime as dt
import traceback

from common import (ARTIFACTS_ROOT, BRANCHES, HISTORY_NAMES, MIGRATION_ROOT, PESTS,
                    SCHEMA_VERSION, SOURCE_ZIP, TABULAR_STATS, channel_names,
                    ckpt_path, derive_feature_names, env_versions, json_dump,
                    load_checkpoint, sha256_file)


def export_one(pest: str, branch: str, versions: dict, timestamp: str,
               zip_sha: str | None) -> dict:
    src = ckpt_path(pest, branch)
    rec: dict = {"pest": pest, "branch": branch, "source_checkpoint": str(src),
                 "source_zip_sha256": zip_sha}

    if not src.exists():
        rec.update(status="failed", stage="locate", error=f"checkpoint not found: {src}")
        return rec

    rec["source_checkpoint_sha256"] = sha256_file(src)

    # ---- load ----------------------------------------------------------
    try:
        ckpt = load_checkpoint(src)
    except Exception as exc:
        rec.update(status="failed", stage="torch_load",
                   error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc())
        return rec
    rec["load_ok"] = True

    # ---- extract + type-check -----------------------------------------
    try:
        model = ckpt["trained_states"][0]["sk_model"]
    except Exception as exc:
        rec.update(status="failed", stage="extract",
                   error=f"{type(exc).__name__}: {exc}")
        return rec

    cls = f"{type(model).__module__}.{type(model).__name__}"
    rec["model_class"] = cls
    if type(model).__name__ != "XGBClassifier":
        rec.update(status="failed", stage="type_check",
                   error=f"expected XGBClassifier, found {cls}")
        return rec

    # ---- feature order (derived from API code, cross-checked) ----------
    try:
        feature_names = derive_feature_names(ckpt)
        n_features_in = int(model.n_features_in_)
        if len(feature_names) != n_features_in:
            rec.update(status="failed", stage="feature_order",
                       error=("derived feature order length "
                              f"{len(feature_names)} != n_features_in_ {n_features_in}; "
                              "refusing to guess"))
            return rec
    except Exception as exc:
        rec.update(status="failed", stage="feature_order",
                   error=f"{type(exc).__name__}: {exc}")
        return rec

    # ---- export JSON ---------------------------------------------------
    out_dir = ARTIFACTS_ROOT / pest / branch
    out_dir.mkdir(parents=True, exist_ok=True)
    model_json = out_dir / "model.json"
    try:
        model.save_model(str(model_json))
    except Exception as exc:
        rec.update(status="failed", stage="save_model",
                   error=f"{type(exc).__name__}: {exc}",
                   traceback=traceback.format_exc())
        return rec

    rec["exported_json"] = str(model_json)
    rec["exported_json_sha256"] = sha256_file(model_json)
    rec["exported_json_bytes"] = model_json.stat().st_size

    booster = model.get_booster()
    classes = [int(c) for c in model.classes_]
    metadata = {
        "schema_version": SCHEMA_VERSION,
        "pest": pest,
        "branch": branch,
        "source_checkpoint": str(src),
        "source_checkpoint_sha256": rec["source_checkpoint_sha256"],
        "source_zip": str(SOURCE_ZIP),
        "source_zip_sha256": zip_sha,
        "exported_json": model_json.name,
        "exported_json_sha256": rec["exported_json_sha256"],
        "exported_json_bytes": rec["exported_json_bytes"],
        "source_env": versions,
        "model_class": cls,
        "classes_": classes,
        "n_classes": len(classes),
        "n_features_in_": n_features_in,
        "feature_names": feature_names,
        "feature_name_order": {
            "source": ("derived from infer/stage1.py::_build_tabular "
                       "(+ _append_history when site_history_added); "
                       "length cross-checked against n_features_in_"),
            "base_channels": [str(c) for c in ckpt["feature_names"]],
            "history_channels_appended": bool(ckpt.get("site_history_added", False)),
            "history_channels": (list(HISTORY_NAMES)
                                 if ckpt.get("site_history_added", False) else []),
            "channels": channel_names(ckpt),
            "n_channels": len(channel_names(ckpt)),
            "stats_in_order": TABULAR_STATS,
            "tstar_position_feature_appended": bool(
                ckpt.get("add_tstar_position_feature", False)),
            "layout": ("concat([stat(channel) for stat in stats_in_order "
                       "for channel in channels]) then tstar_pos if appended"),
        },
        "booster_feature_names": booster.feature_names,
        "estimator_params": {k: (v if isinstance(v, (int, float, str, bool, type(None)))
                                 else repr(v))
                             for k, v in model.get_params().items()},
        "objective": str(model.objective),
        "n_trees": len(booster.get_dump()),
        "n_estimators": int(model.n_estimators) if model.n_estimators else None,
        "expected_predict_proba_shape": ["n_rows", len(classes)],
        "checkpoint_context": {
            k: (ckpt.get(k) if isinstance(ckpt.get(k), (int, float, str, bool, type(None)))
                else repr(ckpt.get(k)))
            for k in ["run", "pest", "d_in", "year_max", "model_type", "event_model",
                      "doy_start", "doy_end", "T", "task_mode", "nowcast_window",
                      "nowcast_stride", "nowcast_only_pre_event",
                      "nowcast_event_time_proxy", "add_tstar_position_feature",
                      "site_history_added", "phenology_added", "derived_weather_added"]
        },
        "checkpoint_feature_cols": [str(c) for c in ckpt["feature_cols"]],
        "export_timestamp_utc": timestamp,
    }
    json_dump(metadata, out_dir / "metadata.json")
    rec.update(status="ok", metadata_json=str(out_dir / "metadata.json"),
               n_features_in=n_features_in, n_trees=metadata["n_trees"],
               classes=classes, objective=metadata["objective"])
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", action="append", help="limit to these pests")
    args = ap.parse_args()

    pests = args.pest or PESTS
    versions = env_versions()
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    zip_sha = sha256_file(SOURCE_ZIP) if SOURCE_ZIP.exists() else None

    records = []
    for pest in pests:
        for branch in BRANCHES:
            rec = export_one(pest, branch, versions, timestamp, zip_sha)
            records.append(rec)
            flag = "ok  " if rec["status"] == "ok" else "FAIL"
            detail = (f"{rec.get('n_features_in')} feat, {rec.get('n_trees')} trees, "
                      f"{rec.get('exported_json_bytes', 0)/1024:.0f} KiB"
                      if rec["status"] == "ok" else
                      f"[{rec.get('stage')}] {rec.get('error')}")
            print(f"[{flag}] {pest:<18} {branch}  {detail}", flush=True)

    ok = sum(r["status"] == "ok" for r in records)
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": timestamp,
        "source_zip": str(SOURCE_ZIP),
        "source_zip_sha256": zip_sha,
        "source_env": versions,
        "totals": {"attempted": len(records), "exported": ok,
                   "failed": len(records) - ok},
        "models": records,
    }
    json_dump(manifest, MIGRATION_ROOT / "migration_manifest.json")
    print(f"\nexported {ok}/{len(records)} -> {ARTIFACTS_ROOT}")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
