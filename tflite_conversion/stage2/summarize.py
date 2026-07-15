"""Join export / synthetic-parity / real-data / benchmark results into one table.

Reads the JSON each stage wrote into artifacts/ and emits:
    artifacts/conversion_summary.json
    docs/all_pests_stage2_tflite_conversion_report.md  (table section only;
        the surrounding narrative is maintained by hand)

Usage (inside .venv-tflite, from this directory):
    python summarize.py
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path

from pest_configs import LEARNED_IS_MAIN, PESTS

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"
REPO_ROOT = HERE.parents[1]

# Recommendation thresholds, in days.
# FP16 is preferred when it is safe, because it is ~1.56x smaller at the same
# speed. "Safe" = real-data error stays far below the 1-day rounding the API
# applies to the 95% PI. 0.01 d (~15 min) is a deliberately strict bar: it is
# 500x below sigma=5 d, so an fp16 build clearing it cannot alter a rounded
# response. Above it we still pass validation (0.1 d) but prefer fp32 and flag
# the pest for a closer look.
FP16_REAL_PREFER_DAYS = 0.01


def _load(name: str) -> dict:
    p = ARTIFACTS / name
    if not p.is_file():
        return {}
    return json.loads(p.read_text())


def build_rows() -> list[dict]:
    exp = _load("export_results.json")
    syn = _load("validate_all.json")
    real = _load("validate_real.json")
    bench = _load("benchmark_all.json")

    by_exp = {(r["pest"], r["variant"]): r for r in exp.get("results", [])}
    exp_fail = {(f["pest"], f["variant"]): f for f in exp.get("failures", [])}
    by_syn = {r["pest"]: r for r in syn.get("rows", [])}
    by_real = {r["pest"]: r for r in real.get("rows", [])}
    skipped = {s["pest"]: s for s in real.get("skipped", [])}
    by_bench = (bench.get("rows") or {})

    rows = []
    for pest in PESTS:
        s = by_syn.get(pest, {})
        r = by_real.get(pest, {})
        b = by_bench.get(pest, {})
        med = b.get("median_ms", {})

        notes: list[str] = []
        if pest in LEARNED_IS_MAIN:
            notes.append("learned output is the API's final answer")
        else:
            notes.append("API returns climatology; learned output is experimental")
        if pest in skipped:
            notes.append(f"real data skipped: {skipped[pest]['reason']}")

        fp32_ok = (pest, "fp32") in by_exp
        fp16_ok = (pest, "fp16") in by_exp

        fp16_real = r.get("fp16_max_abs_days")
        fp32_real = r.get("fp32_max_abs_days")

        # Recommendation
        if not (fp32_ok and s.get("fp32_pass")):
            rec = "conversion_failed"
        elif not (fp16_ok and s.get("fp16_pass")):
            rec = "fp32"
            notes.append("fp16 failed synthetic tolerance")
        elif not r:
            rec = "needs_further_validation"
            notes.append("no real-data validation")
        elif fp16_real is not None and fp16_real > FP16_REAL_PREFER_DAYS:
            rec = "fp32"
            notes.append(f"fp16 real error {fp16_real:.2e} d > {FP16_REAL_PREFER_DAYS} d bar")
        else:
            rec = "fp16"

        rows.append({
            "pest": pest,
            "checkpoint_loaded": bool(s.get("checkpoint_loaded", False)),
            "d_in": s.get("d_in"), "T": s.get("T"),
            "wrapper_parity": ("exact" if s.get("wrapper_parity_exact") else "FAIL"),
            "wrapper_max_error_days": s.get("wrapper_max_abs_days"),
            "fp32_export": fp32_ok,
            "fp32_max_error_days": s.get("fp32_max_abs_days"),
            "fp32_mean_error_days": s.get("fp32_mean_abs_days"),
            "fp32_real_error_days": fp32_real,
            "fp32_size_bytes": by_exp.get((pest, "fp32"), {}).get("bytes"),
            "fp32_sha256": by_exp.get((pest, "fp32"), {}).get("sha256"),
            "fp32_latency_ms": med.get("fp32"),
            "fp16_export": fp16_ok,
            "fp16_max_error_days": s.get("fp16_max_abs_days"),
            "fp16_mean_error_days": s.get("fp16_mean_abs_days"),
            "fp16_real_error_days": fp16_real,
            "fp16_size_bytes": by_exp.get((pest, "fp16"), {}).get("bytes"),
            "fp16_sha256": by_exp.get((pest, "fp16"), {}).get("sha256"),
            "fp16_latency_ms": med.get("fp16"),
            "pytorch_latency_ms": med.get("pytorch"),
            "fp32_speedup": (b.get("speedup") or {}).get("fp32"),
            "fp16_speedup": (b.get("speedup") or {}).get("fp16"),
            "real_data_validated": bool(r.get("passed", False)),
            "real_site_year": (f"{r['site']}/{r['year']}" if r else None),
            "real_alert_ok": r.get("alert_ok"),
            "real_mu_doy_pytorch": r.get("mu_doy_original"),
            "readme_mu_doy": r.get("readme_mu_doy"),
            "readme_delta_days": r.get("readme_delta_days"),
            "export_error": exp_fail.get((pest, "fp32"), {}).get("error"),
            "recommended_variant": rec,
            "notes": "; ".join(notes),
        })
    return rows


def md_table(rows: list[dict]) -> str:
    hdr = ("| pest | ckpt | wrapper | fp32 | fp32 max err (d) | fp32 size | fp32 ms | "
           "fp16 | fp16 max err (d) | fp16 size | fp16 ms | real data | recommended |")
    sep = "|" + "---|" * 13
    out = [hdr, sep]
    for r in rows:
        out.append(
            f"| {r['pest']} | {'OK' if r['checkpoint_loaded'] else 'FAIL'} "
            f"| {r['wrapper_parity']} "
            f"| {'OK' if r['fp32_export'] else 'FAIL'} | {r['fp32_max_error_days']:.2e} "
            f"| {r['fp32_size_bytes']:,} | {r['fp32_latency_ms']:.3f} "
            f"| {'OK' if r['fp16_export'] else 'FAIL'} | {r['fp16_max_error_days']:.2e} "
            f"| {r['fp16_size_bytes']:,} | {r['fp16_latency_ms']:.3f} "
            f"| {'PASS' if r['real_data_validated'] else 'no'} "
            f"| **{r['recommended_variant']}** |"
        )
    return "\n".join(out)


def md_real_table(rows: list[dict]) -> str:
    out = ["| pest | site / year | alert | mu_doy (PyTorch) | README | delta | fp32 err (d) | fp16 err (d) |",
           "|" + "---|" * 8]
    for r in rows:
        if not r["real_site_year"]:
            out.append(f"| {r['pest']} | — | — | — | — | — | — | — |")
            continue
        out.append(
            f"| {r['pest']} | {r['real_site_year']} "
            f"| {'✓' if r['real_alert_ok'] else '✗'} "
            f"| {r['real_mu_doy_pytorch']:.4f} | {r['readme_mu_doy']:.2f} "
            f"| {r['readme_delta_days']:.4f} "
            f"| {r['fp32_real_error_days']:.2e} | {r['fp16_real_error_days']:.2e} |"
        )
    return "\n".join(out)


def main() -> int:
    rows = build_rows()
    summary = {
        "generated_from": ["export_results.json", "validate_all.json",
                           "validate_real.json", "benchmark_all.json"],
        "fp16_prefer_threshold_days": FP16_REAL_PREFER_DAYS,
        "counts": {
            "pests": len(rows),
            "fp32_exported": sum(1 for r in rows if r["fp32_export"]),
            "fp16_exported": sum(1 for r in rows if r["fp16_export"]),
            "wrapper_exact": sum(1 for r in rows if r["wrapper_parity"] == "exact"),
            "real_data_validated": sum(1 for r in rows if r["real_data_validated"]),
            "recommend_fp16": sum(1 for r in rows if r["recommended_variant"] == "fp16"),
            "recommend_fp32": sum(1 for r in rows if r["recommended_variant"] == "fp32"),
        },
        "rows": rows,
    }
    (ARTIFACTS / "conversion_summary.json").write_text(json.dumps(summary, indent=2))

    print(md_table(rows))
    print()
    print(md_real_table(rows))
    print()
    print(json.dumps(summary["counts"], indent=2))

    (ARTIFACTS / "summary_tables.md").write_text(
        "## Conversion summary\n\n" + md_table(rows)
        + "\n\n## Real-data validation\n\n" + md_real_table(rows) + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
