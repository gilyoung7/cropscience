"""Freeze the Stage-1 calibration (temperature) and gate (method/k/tau) into
portable per-model assets beside each model.json.

Sources -- copied, never recomputed, never re-tuned:
  temperature: assets/stage1/<pest>/temperature.json   (already a shipped asset)
  gate:        assets/stage1/<pest>/group_tau/group_tau_hybrid_summary.json
               + method from configs/stage1_selected_gates.yaml
               resolved exactly as run_predict.py::_gate_config_for does.

Usage:
    ../api_handoff_transformer/.venv/bin/python export_calibration_assets.py
"""
from __future__ import annotations

import datetime as dt
import json

from common import (API_ROOT, ARTIFACTS_ROOT, BRANCHES, MIGRATION_ROOT, PESTS,
                    env_versions, json_dump, sha256_file)

CAL_SCHEMA_VERSION = "stage1-xgb-calibration/1.0.0"
GATE_SCHEMA_VERSION = "stage1-xgb-gate/1.0.0"

# run_predict.py::_gate_config_for -> infer/stage1.py::_GATE_JSON_KEY
GATE_JSON_KEY = {
    "A_baseline": "A_raw_global",
    "D_history": "D_raw_global",
    "dispatch_group_tau": "dispatch_group_tau",
}


def resolve_gate(pest: str, gates: dict) -> dict:
    """Mirror of run_predict.py::_gate_config_for + stage1.py::resolve_gate_params.

    method comes from the yaml; k/tau come from the summary JSON, which
    infer/stage1.py documents as authoritative ("the yaml copy has drifted for
    some pests ... do not trust yaml k/tau").
    """
    g = gates["per_pest"][pest]
    method = str(g["method"])
    target_label = str(gates.get("target_label") or "R>=0.88")
    summary_json = API_ROOT / "assets" / "stage1" / pest / "group_tau" / "group_tau_hybrid_summary.json"
    sel = json.loads(summary_json.read_text())["selections"]
    if target_label not in sel:
        raise KeyError(f"target {target_label!r} not in {summary_json}")
    dsel = sel[target_label][GATE_JSON_KEY[method]]
    k = int(dsel["k"])
    if method == "dispatch_group_tau":
        tau, tau_no, tau_with = None, float(dsel["tau_no"]), float(dsel["tau_with"])
    else:
        tau, tau_no, tau_with = float(dsel["tau"]), None, None

    yaml_k = g.get("k")
    yaml_tau = {"tau": g.get("tau"), "tau_no": g.get("tau_no"), "tau_with": g.get("tau_with")}
    drifted = (yaml_k != k or yaml_tau["tau"] != tau
               or yaml_tau["tau_no"] != tau_no or yaml_tau["tau_with"] != tau_with)
    return {
        "schema_version": GATE_SCHEMA_VERSION,
        "pest": pest,
        "method": method,
        "k": k,
        "tau": tau,
        "tau_no": tau_no,
        "tau_with": tau_with,
        "target_label": target_label,
        "resolved_from": {
            "method_source": "configs/stage1_selected_gates.yaml : per_pest.<pest>.method",
            "k_tau_source": (f"assets/stage1/{pest}/group_tau/group_tau_hybrid_summary.json"
                             f" : selections['{target_label}']['{GATE_JSON_KEY[method]}']"),
            "summary_json_sha256": sha256_file(summary_json),
            "note": ("k/tau intentionally NOT taken from the yaml: infer/stage1.py marks the "
                     "summary JSON authoritative and the yaml copy drifted."),
        },
        "yaml_copy_for_reference": {"k": yaml_k, **yaml_tau, "drifted_vs_authoritative": drifted},
    }


def main() -> int:
    import yaml

    versions = env_versions()
    timestamp = dt.datetime.now(dt.timezone.utc).isoformat()
    gates = yaml.safe_load((API_ROOT / "configs" / "stage1_selected_gates.yaml").read_text())

    cal_records, gate_records = [], []
    for pest in PESTS:
        temp_path = API_ROOT / "assets" / "stage1" / pest / "temperature.json"
        hist_path = API_ROOT / "assets" / "stage1" / pest / "site_history.json"
        temp = json.loads(temp_path.read_text())
        temp_sha = sha256_file(temp_path)

        gate = resolve_gate(pest, gates)
        gate["source_temperature_json_sha256"] = temp_sha
        json_dump(gate, ARTIFACTS_ROOT / pest / "gate.json")
        gate_records.append({k: gate[k] for k in
                             ["pest", "method", "k", "tau", "tau_no", "tau_with"]}
                            | {"yaml_drifted": gate["yaml_copy_for_reference"]["drifted_vs_authoritative"]})

        for branch in BRANCHES:
            t = float(temp[f"temperature_{branch}"])
            meta_path = ARTIFACTS_ROOT / pest / branch / "metadata.json"
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            cal = {
                "schema_version": CAL_SCHEMA_VERSION,
                "pest": pest,
                "branch": branch,
                "calibration_method": "temperature_scaling",
                "temperature": t,
                "applies_to": "positive-class probability, i.e. predict_proba(X)[:, 1]",
                "formula": ("p_cal = sigmoid(logit(clip(p_raw, 1e-8, 1-1e-8)) / temperature); "
                            "port of infer/stage1.py::_apply_temperature"),
                "eps": 1e-8,
                "probability_before": "raw XGBoost predict_proba(X)[:, 1]",
                "probability_after": "calibrated probability compared against tau by the gate",
                "source": {
                    "temperature_json": str(temp_path),
                    "temperature_json_sha256": temp_sha,
                    "source_checkpoint_sha256": meta["source_checkpoint_sha256"],
                    "model_json_sha256": meta["exported_json_sha256"],
                    "doy_start_in_temperature_json": int(temp["doy_start"]),
                    "method_in_temperature_json": str(temp["method"]),
                },
                "provenance": {
                    "origin": ("frozen asset shipped with the API package; the API's production "
                               "path (infer/stage1.py::compute_alert_single_sy via "
                               "load_stage1_reference) reads this value and does NOT re-fit."),
                    "originally_fitted_by": ("infer/stage1.py::_fit_temperature_grid, minimising "
                                             "binary NLL over a fixed grid on VAL_YEAR=2023 raw "
                                             "probabilities (cohort path _calibrated_per_sy)."),
                    "refit_at_runtime": False,
                    "n_fit_samples": None,
                    "n_fit_samples_note": ("not recorded in the shipped temperature.json and not "
                                           "recoverable from the checkpoint; left null rather than "
                                           "guessed. See reports/alert_parity_report.md."),
                },
                "site_history_reference": {
                    "required_for_this_branch": bool(
                        meta["feature_name_order"]["history_channels_appended"]),
                    "path": str(hist_path),
                    "sha256": sha256_file(hist_path),
                    "note": ("site_history.json supplies the 11 history channels and the with_history "
                             "flag that selects tau_no vs tau_with; it is a required portable asset "
                             "for the D branch and for dispatch_group_tau routing."),
                },
                "export_timestamp_utc": timestamp,
                "source_env": versions,
            }
            json_dump(cal, ARTIFACTS_ROOT / pest / branch / "calibration.json")
            cal_records.append({"pest": pest, "branch": branch, "temperature": t})
            print(f"[ok  ] {pest:<18} {branch}  T={t:<8} gate={gate['method']} "
                  f"k={gate['k']} tau={gate['tau']} no={gate['tau_no']} with={gate['tau_with']}")

    manifest = {
        "schema_version": "stage1-xgb-portable-manifest/1.0.0",
        "generated_utc": timestamp,
        "source_env": versions,
        "portable_assets_per_model": ["model.json", "metadata.json", "calibration.json"],
        "portable_assets_per_pest": ["gate.json"],
        "external_assets_still_required": {
            "site_history.json": ("per-pest, ~7.3k site-year keys; supplies history channels + "
                                  "with_history routing. Not copied into artifacts/ (see README)."),
        },
        "calibration": cal_records,
        "gates": gate_records,
    }
    json_dump(manifest, MIGRATION_ROOT / "portable_manifest.json")
    print(f"\nwrote calibration.json x{len(cal_records)}, gate.json x{len(gate_records)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
