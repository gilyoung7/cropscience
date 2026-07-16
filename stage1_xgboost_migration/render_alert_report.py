"""Render reports/alert_parity_report.md from the JSON reports.

Reads reports/alert_parity_report.json (+ portable_pipeline_report.json when present)
and writes a human-readable summary. Pure formatting -- it never recomputes or
re-derives a number, so the markdown cannot disagree with the JSON.

Usage:
    ../api_handoff_transformer/.venv/bin/python render_alert_report.py
"""
from __future__ import annotations

import json

from common import MIGRATION_ROOT, REPORTS_ROOT


def _fmt(x, nd=3):
    if x is None:
        return "-"
    if isinstance(x, float):
        return f"{x:.{nd}e}" if (x != 0 and abs(x) < 1e-3) else f"{x:g}"
    return str(x)


def main() -> int:
    rep = json.loads((REPORTS_ROOT / "alert_parity_report.json").read_text())
    pp_path = REPORTS_ROOT / "portable_pipeline_report.json"
    pp = json.loads(pp_path.read_text()) if pp_path.exists() else None

    L: list[str] = []
    A = L.append
    A("# Stage-1 alert parity: original `.pt` path vs portable JSON path")
    A("")
    A(f"Generated: `{rep['generated_utc']}`  ")
    env = rep["source_env"]
    A(f"Source env: python {env['python']}, torch {env['torch']}, "
      f"xgboost {env['xgboost']}, scikit-learn {env['scikit_learn']}, numpy {env['numpy']}")
    A("")
    t = rep["totals"]
    A(f"**{t['ok']}/{t['pests']} pests fully match** end-to-end "
      f"(raw proba -> temperature -> tau/k gate -> alert_tstar -> dispatch features).")
    A("")

    A("## Per-pest end-to-end result")
    A("")
    A("| pest | data | site-years | fired (orig/portable) | alert DOY | dispatch | status |")
    A("| --- | --- | ---: | --- | --- | --- | --- |")
    for p in rep["pests"]:
        s = p.get("alert_summary", {})
        A(f"| {p['pest']} | {p.get('data_kind','-')} | {p.get('n_site_years','-')} | "
          f"{s.get('n_fired_original','-')} / {s.get('n_fired_portable','-')} | "
          f"{'match' if s.get('alert_doy_all_match') else 'DIFFER'} | "
          f"{'match' if s.get('dispatch_all_match') else 'DIFFER'} | {p['status']} |")
    A("")

    A("## Probability parity per model (16)")
    A("")
    A("| pest | branch | T | raw max abs diff | raw bit-exact | calibrated max abs diff "
      "| calibrated bit-exact | tau | threshold decisions |")
    A("| --- | --- | ---: | ---: | --- | ---: | --- | ---: | --- |")
    for p in rep["pests"]:
        for b in p.get("branches", []):
            td = b["threshold_decisions"]
            A(f"| {p['pest']} | {b['branch']} | {b['temperature']} | "
              f"{_fmt(b['raw_proba']['max_abs_diff'])} | {b['raw_proba']['bit_exact']} | "
              f"{_fmt(b['calibrated_proba']['max_abs_diff'])} | "
              f"{b['calibrated_proba']['bit_exact']} | {td['tau_used']} | "
              f"{'all match' if td['all_match'] else 'DIFFER'} "
              f"({td['n_pass_original']}/{td['n']} pass) |")
    A("")

    A("## Gate parameters actually used")
    A("")
    A("`k`/`tau` come from each pest's `group_tau_hybrid_summary.json`, which "
      "`infer/stage1.py` marks authoritative; the copy in `stage1_selected_gates.yaml` "
      "has drifted and is **not** used.")
    A("")
    A("| pest | method | k | tau | tau_no | tau_with |")
    A("| --- | --- | ---: | ---: | ---: | ---: |")
    for p in rep["pests"]:
        g = p.get("gate", {})
        A(f"| {p['pest']} | {g.get('method','-')} | {g.get('k','-')} | {_fmt(g.get('tau'))} | "
          f"{_fmt(g.get('tau_no'))} | {_fmt(g.get('tau_with'))} |")
    A("")

    b = rep["gate_boundary_unit_parity"]
    A("## Gate boundary unit parity")
    A("")
    A(f"{b['n_cases']} crafted series x k in {{1,2,3}} at tau=0.6 -- probability exactly "
      f"at tau, one ULP below tau, streak of exactly k-1 vs k, NaN present, all-above, "
      f"all-below. **all match: {b['all_match']}**")
    A("")

    if rep.get("temperature_refit_check"):
        A("## Shipped temperature vs a fresh runtime re-fit")
        A("")
        A("The API's production path (`compute_alert_single_sy`) reads the frozen "
          "`temperature.json` and does not re-fit. The cohort path (`_calibrated_per_sy`) "
          "re-fits on VAL_YEAR=2023. This compares the two.")
        A("")
        A("Only valid where the pest has its OWN labels -- `_fit_temperature_grid` "
          "minimises NLL against `y_event`, so a re-fit against borrowed labels would "
          "be meaningless and is skipped rather than reported as a mismatch.")
        A("")
        A("| pest | branch | shipped | re-fit now | match |")
        A("| --- | --- | ---: | ---: | --- |")
        for pest, d in rep["temperature_refit_check"].items():
            if "skipped" in d:
                A(f"| {pest} | - | - | - | skipped: {d['skipped']} |")
                continue
            for br, v in d.items():
                if "error" in v:
                    A(f"| {pest} | {br} | - | - | error: {v['error']} |")
                else:
                    A(f"| {pest} | {br} | {v['shipped']} | {v['refit_now']} | {v['match']} |")
        A("")

    if rep.get("real_data_notes"):
        A("## Data provenance")
        A("")
        for k, v in rep["real_data_notes"].items():
            A(f"- **{k}**: {v}")
        A("")

    if pp:
        A("## Portable pipeline (torch-free)")
        A("")
        e = pp["env"]; tf = pp["torch_free"]
        A(f"Ran under python {e['python']}, xgboost {e['xgboost']}, numpy {e['numpy']}.")
        A("")
        A(f"- torch imported: **{'torch' in tf.get('imported', [])}**")
        A(f"- `.pt` opened / unpickled: **{tf.get('checkpoint_or_pickle_violations') or 'none'}**")
        A(f"- pulled in by xgboost itself (not by the pipeline): "
          f"{tf.get('pulled_in_by_xgboost_itself')}")
        A("")
        A("| pest | site-years | fired (portable/reference) | mismatches | status |")
        A("| --- | ---: | --- | ---: | --- |")
        for p in pp["pests"]:
            A(f"| {p['pest']} | {p.get('n_site_years','-')} | "
              f"{p.get('n_fired_portable','-')} / {p.get('n_fired_reference','-')} | "
              f"{p.get('n_mismatch','-')} | {p['status']} |")
        A("")

    out = REPORTS_ROOT / "alert_parity_report.md"
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"wrote {out} ({out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
