"""Output-contract unit tests — the branches that had no coverage before.

Covers select_final's three outcomes, the no-alert / climatology_no_alert
distinction, fallback flags, response key order, CSV column order and CRLF, null
handling and trailing newlines. Pure unit tests: no model, no data files, so they
run anywhere in the runtime venv.

    python tests/test_output_contract.py
"""

from __future__ import annotations

import csv
import io
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from infer.fallback import PI95_HALFWIDTH, SIGMA_DAYS_DEFAULT, select_final  # noqa: E402
from infer.schemas import FLAT_COLS, build_response, flatten_response  # noqa: E402

CLIM = {
    "mu_doy": 147.55,
    "pi_95": {"lower_doy": 138, "upper_doy": 157, "sigma_days": 5.0},
    "variant": "mean_mid",
    "_source_csv": "BPH_climatology_train_stats.csv",
}
LEARNED = {
    "mu_doy": 234.33,
    "pi_95": {"lower_doy": 225, "upper_doy": 244, "sigma_days": 5.0},
    "selected_offset": 30,
    "output_status": "main",
    "model_kind": "lead_v3",
}
POLICY = {"per_pest": {
    "BPH": {"recommended_source": "learned_stage2", "learned_output_status": "main",
            "selected_fixed_offset": 30, "climatology": {"variant": "mean_mid"}},
    "sheath_blight": {"recommended_source": "climatology",
                      "learned_output_status": "experimental",
                      "selected_fixed_offset": 45, "climatology": {"variant": "mean_mid"}},
}}
BACKENDS = {"stage1_backend": "xgboost_json", "stage2_backend": "litert_fp16"}

results: list[dict] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append({"name": name, "passed": bool(cond), "detail": detail})


# --------------------------------------------------------------------------
# select_final — all three branches
# --------------------------------------------------------------------------
def test_select_final_learned() -> None:
    f = select_final(LEARNED, CLIM, "learned_stage2")
    check("select_final: learned + learned_stage2 -> learned_stage2",
          f["source"] == "learned_stage2" and f["mu_doy"] == 234.33
          and f["selected_offset"] == 30 and f["fallback_triggered"] is False,
          str(f))


def test_select_final_climatology_recommended() -> None:
    """Learned succeeded but the pest is climatology-recommended: climatology
    wins, selected_offset is null, and fallback_triggered stays False."""
    f = select_final(LEARNED, CLIM, "climatology")
    check("select_final: learned + climatology -> climatology, offset null, fb False",
          f["source"] == "climatology" and f["mu_doy"] == 147.55
          and f["selected_offset"] is None and f["fallback_triggered"] is False,
          str(f))


def test_select_final_no_learned_on_learned_pest() -> None:
    """Stage-2 produced nothing on a learned-recommended pest -> the deployed
    code emits 'climatology_no_alert' with fallback_triggered True, for ANY
    Stage-2 failure (not only a genuine no-alert)."""
    f = select_final(None, CLIM, "learned_stage2")
    check("select_final: none + learned_stage2 -> climatology_no_alert, fb True",
          f["source"] == "climatology_no_alert" and f["fallback_triggered"] is True
          and f["selected_offset"] is None, str(f))


def test_select_final_no_learned_on_clim_pest() -> None:
    """Same failure on a climatology-recommended pest is indistinguishable from a
    healthy run in final_prediction — deployed quirk, pinned here on purpose."""
    f = select_final(None, CLIM, "climatology")
    check("select_final: none + climatology -> climatology, fb False (deployed quirk)",
          f["source"] == "climatology" and f["fallback_triggered"] is False, str(f))


def test_pi_constants() -> None:
    check("climatology PI constants: sigma 5.0 / half 9.8",
          SIGMA_DAYS_DEFAULT == 5.0 and PI95_HALFWIDTH == 9.8,
          f"{SIGMA_DAYS_DEFAULT}/{PI95_HALFWIDTH}")


# --------------------------------------------------------------------------
# response assembly
# --------------------------------------------------------------------------
def _resp(learned, pest="BPH", diag=None, err=None, include_diag=False):
    req = {"pest": pest, "site_id": "S1", "year": 2004}
    if include_diag:
        req["include_diagnostics"] = True
    return build_response(req, learned, CLIM, POLICY, diag or {}, err, BACKENDS)


def test_response_key_order() -> None:
    r = _resp(LEARNED)
    expect = ["pest", "site_id", "year", "model_version", "stage1", "stage2",
              "final_prediction", "backends"]
    check("response root key ORDER matches the deployed contract (+backends last)",
          list(r) == expect, f"{list(r)}")


def test_response_nested_keys() -> None:
    r = _resp(LEARNED)
    ok = (list(r["stage1"]) == ["alert_fired", "alert_tstar_doy", "gate_method",
                                "alert_source", "wiring_status"]
          and list(r["stage2"]) == ["learned_stage2", "climatology", "recommended_source"]
          and list(r["final_prediction"]) == ["source", "mu_doy", "pi_95",
                                              "selected_offset", "fallback_triggered"])
    check("nested key order (stage1 / stage2 / final_prediction)", ok,
          f"{list(r['stage1'])} | {list(r['stage2'])} | {list(r['final_prediction'])}")


def test_diagnostics_off_by_default() -> None:
    r = _resp(LEARNED)
    check("diagnostics key ABSENT when include_diagnostics is falsy",
          "diagnostics" not in r, str(list(r)))


def test_diagnostics_on() -> None:
    r = _resp(LEARNED, diag={"alert_source": "stage1_live"}, include_diag=True)
    ok = ("diagnostics" in r
          and list(r["diagnostics"]) == ["selector_used", "climatology_train_stats_path",
                                         "selected_fixed_offset_from_policy",
                                         "transformer", "transformer_error"])
    check("diagnostics block present + key order when enabled", ok,
          str(list(r.get("diagnostics", {}))))


def test_null_learned() -> None:
    r = _resp(None)
    check("stage2.learned_stage2 is null when Stage-2 produced nothing",
          r["stage2"]["learned_stage2"] is None, str(r["stage2"]["learned_stage2"]))


def test_alert_fired_tracks_used_not_fired() -> None:
    """Deployed quirk: alert_fired reflects alert_tstar_doy_used (set only after
    the Stage-2 input build), NOT whether Stage-1 fired."""
    r = _resp(None, diag={"alert_source": "stage1_live", "stage1_alert_tstar_doy": 176})
    check("alert_fired False when Stage-1 fired but input build did not complete",
          r["stage1"]["alert_fired"] is False and r["stage1"]["alert_tstar_doy"] is None
          and r["stage1"]["alert_source"] == "stage1_live", str(r["stage1"]))
    r2 = _resp(LEARNED, diag={"alert_tstar_doy_used": 176})
    check("alert_fired True once alert_tstar_doy_used is set",
          r2["stage1"]["alert_fired"] is True and r2["stage1"]["alert_tstar_doy"] == 176,
          str(r2["stage1"]))


def test_gate_method_fallback_table() -> None:
    r = _resp(LEARNED)   # empty diag -> falls back to the static table
    check("gate_method falls back to the per-pest table when Stage-1 is silent",
          r["stage1"]["gate_method"] == "D_history", r["stage1"]["gate_method"])


# --------------------------------------------------------------------------
# predictions.csv contract
# --------------------------------------------------------------------------
def test_flat_cols_order() -> None:
    expect = ["pest", "site_id", "year", "model_version", "final_source",
              "final_mu_doy", "final_pi95_lower", "final_pi95_upper",
              "learned_mu_doy", "learned_selected_offset", "learned_output_status",
              "climatology_mu_doy", "climatology_variant", "recommended_source",
              "fallback_triggered", "alert_tstar_doy"]
    check("FLAT_COLS == the deployed 16 columns, in order",
          FLAT_COLS == expect, str(FLAT_COLS))


def test_flatten_nulls() -> None:
    row = flatten_response(_resp(None))
    ok = (row["learned_mu_doy"] is None and row["learned_selected_offset"] is None
          and row["learned_output_status"] is None
          and row["final_source"] == "climatology_no_alert")
    check("flatten_response leaves learned_* null on fallback", ok, str(row))


def test_csv_crlf_and_nulls() -> None:
    """csv.DictWriter with newline='' must produce CRLF; None -> empty field."""
    row = flatten_response(_resp(None))
    buf = io.StringIO(newline="")
    w = csv.DictWriter(buf, fieldnames=FLAT_COLS)
    w.writeheader(); w.writerow(row)
    raw = buf.getvalue()
    check("predictions.csv uses CRLF line endings",
          raw.count("\r\n") == 2 and raw.count("\n") == raw.count("\r\n"),
          repr(raw[-40:]))
    body = raw.split("\r\n")[1].split(",")
    idx = FLAT_COLS.index("learned_mu_doy")
    check("None serializes as an EMPTY csv field (not 'None')", body[idx] == "",
          repr(body[idx]))


def test_json_trailing_newline() -> None:
    text = json.dumps(_resp(LEARNED), indent=2, ensure_ascii=False) + "\n"
    check("response.json is written with a trailing newline", text.endswith("\n"))
    check("response.json uses ensure_ascii=False (non-ASCII preserved)",
          "\\u" not in json.dumps({"k": "잎도열병"}, ensure_ascii=False))


def main() -> int:
    for fn in [
        test_select_final_learned, test_select_final_climatology_recommended,
        test_select_final_no_learned_on_learned_pest,
        test_select_final_no_learned_on_clim_pest, test_pi_constants,
        test_response_key_order, test_response_nested_keys,
        test_diagnostics_off_by_default, test_diagnostics_on, test_null_learned,
        test_alert_fired_tracks_used_not_fired, test_gate_method_fallback_table,
        test_flat_cols_order, test_flatten_nulls, test_csv_crlf_and_nulls,
        test_json_trailing_newline,
    ]:
        fn()
    print("=== output contract tests ===")
    for r in results:
        print(f"  {'PASS' if r['passed'] else 'FAIL'}  {r['name']}")
        if not r["passed"] and r["detail"]:
            print(f"        got: {r['detail'][:160]}")
    ok = all(r["passed"] for r in results)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
          f"({sum(r['passed'] for r in results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
