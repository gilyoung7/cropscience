"""Request validation + response assembly — byte-compatible with the deployed API.

Key order, dtypes, null behaviour and the deployed quirks are reproduced exactly
(see docs/lightweight_api_integration_plan.md §3). The ONLY additions are two
optional metadata keys under `backends`, which the brief permits; no existing
field is removed, renamed, or re-typed.
"""

from __future__ import annotations

import json
from pathlib import Path

from .fallback import per_pest_policy, select_final
from .paths import MODEL_VERSION, VALID_PESTS

# run_predict.py:506-511 — display fallback when Stage-1 never reported a method.
_GATE_METHOD = {
    "BPH": "D_history", "WBPH": "dispatch_group_tau",
    "bacterial_blight": "D_history", "blast": "D_history",
    "brown_spot": "D_history", "rice_stem_borer_1": "A_baseline",
    "rice_stem_borer_2": "D_history", "sheath_blight": "dispatch_group_tau",
}


class RequestError(ValueError):
    """Invalid request. Maps to exit 1, as in the deployed API."""


def load_request(path: Path) -> dict:
    """Port of run_predict.py:222-236.

    Reproduces the deployed validation exactly, including two quirks:
      * pest is case-SENSITIVE in single mode (no normalization upstream)
      * `year` accepts bool, since isinstance(True, int) is True
    """
    p = Path(path)
    if not p.is_file():
        raise RequestError(f"request.json not found: {p}")
    req = json.loads(p.read_text(encoding="utf-8"))
    return validate_request(req)


def validate_request(req: dict) -> dict:
    pest = req.get("pest")
    site_id = req.get("site_id")
    year = req.get("year")
    if pest not in VALID_PESTS:
        raise RequestError(f"invalid pest '{pest}'. Must be one of: {sorted(VALID_PESTS)}")
    if not isinstance(site_id, str) or not site_id:
        raise RequestError("request.site_id must be a non-empty string")
    if not isinstance(year, int):
        raise RequestError("request.year must be an integer")
    return req


def build_response(request: dict, learned: dict | None, climatology: dict,
                   policy: dict, transformer_diag: dict, learned_error: str | None,
                   backends: dict, stage2_output_status: str | None = None) -> dict:
    """Port of run_predict.py:514-590, key-for-key and in emission order.

    `stage2_output_status` names why Stage-2 produced nothing when the reason is
    a deliberate operational block rather than a failure. It is set ONLY by the
    operational_daily path, so the historical/batch response stays byte-identical
    to the deployed schema.
    """
    pest = request["pest"]
    pp = per_pest_policy(policy, pest)
    recommended = pp.get("recommended_source", "climatology")

    clim_block = {
        "mu_doy": climatology["mu_doy"],
        "pi_95": climatology["pi_95"],
        "variant": climatology["variant"],
    }

    # run_predict.py:825 — alert_fired tracks alert_tstar_doy_used, which is set
    # only AFTER the Stage-2 input build succeeds. A fired alert whose input build
    # failed reports alert_fired=false. Reproduced deliberately.
    alert_tstar_used = transformer_diag.get("alert_tstar_doy_used")

    stage1 = {
        "alert_fired": alert_tstar_used is not None,
        "alert_tstar_doy": alert_tstar_used,
        "gate_method": transformer_diag.get("stage1_method") or _GATE_METHOD.get(pest, "unknown"),
        "alert_source": transformer_diag.get("alert_source"),
        "wiring_status": "stage1_xgboost_live",
    }

    final = select_final(learned, climatology, recommended)

    response = {
        "pest": pest,
        "site_id": request["site_id"],
        "year": request["year"],
        "model_version": MODEL_VERSION,
        "stage1": stage1,
        "stage2": {
            "learned_stage2": learned,
            "climatology": clim_block,
            "recommended_source": recommended,
        },
        "final_prediction": final,
    }
    # Additive-only optional metadata (permitted by the brief). Nothing above changes.
    response["backends"] = backends
    if learned is None and stage2_output_status:
        # Operational block only. Carried on the stage2 block so predictions.csv
        # can report it through the EXISTING learned_output_status column — no
        # new column, so the CSV schema still matches the deployed API.
        response["stage2"]["output_status"] = stage2_output_status

    if request.get("include_diagnostics"):
        response["diagnostics"] = {
            "selector_used": False,
            "climatology_train_stats_path": climatology["_source_csv"],
            "selected_fixed_offset_from_policy": pp.get("selected_fixed_offset"),
            "transformer": transformer_diag,
            "transformer_error": learned_error,
        }
    return response


# run_predict.py:598-620 — exactly 16 columns, in this order.
FLAT_COLS = [
    "pest", "site_id", "year", "model_version", "final_source", "final_mu_doy",
    "final_pi95_lower", "final_pi95_upper", "learned_mu_doy",
    "learned_selected_offset", "learned_output_status", "climatology_mu_doy",
    "climatology_variant", "recommended_source", "fallback_triggered",
    "alert_tstar_doy",
]


def flatten_response(response: dict) -> dict:
    """Port of run_predict.py:598-620 — the predictions.csv row."""
    final = response["final_prediction"]
    learned = response["stage2"]["learned_stage2"]
    clim = response["stage2"]["climatology"]
    return {
        "pest": response["pest"],
        "site_id": response["site_id"],
        "year": response["year"],
        "model_version": response["model_version"],
        "final_source": final["source"],
        "final_mu_doy": final["mu_doy"],
        "final_pi95_lower": final["pi_95"]["lower_doy"],
        "final_pi95_upper": final["pi_95"]["upper_doy"],
        "learned_mu_doy": (learned or {}).get("mu_doy"),
        "learned_selected_offset": (learned or {}).get("selected_offset"),
        "learned_output_status": ((learned or {}).get("output_status")
                                  or response["stage2"].get("output_status")),
        "climatology_mu_doy": clim["mu_doy"],
        "climatology_variant": clim["variant"],
        "recommended_source": response["stage2"]["recommended_source"],
        "fallback_triggered": final["fallback_triggered"],
        "alert_tstar_doy": response["stage1"]["alert_tstar_doy"],
    }
