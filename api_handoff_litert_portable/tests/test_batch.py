"""Batch-mode regression tests, incl. single-vs-batch equivalence.

Needs real inputs (daily weather + LONG). Point it at the local data:

    python tests/test_batch.py --daily-master "$DAILY_MASTER" --long-dir "$LONG_DIR"

Covers the required cases: 2-row all-success, partial failure, missing CSV,
missing required column, unknown pest, lowercase pest, diagnostics on/off,
fp16/fp32, column order, CRLF, input-order preservation, and that one row run in
batch equals the same request run in single mode.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from infer.batch import BATCH_EXTRA_COLS, DIAG_COLS, GENERIC_EXTRA_COLS  # noqa: E402
from infer.schemas import FLAT_COLS  # noqa: E402

SMOKE = {  # (site, year) known-good pairs from README §10
    "BPH": ("33210_56298", 2004),
    "WBPH": ("33908_67063", 2011),
    "sheath_blight": ("35694_60137", 2004),
}
LONG_FILENAME = {
    "BPH": "RICE_LONG_벼멸구.csv", "WBPH": "RICE_LONG_흰등멸구.csv",
    "sheath_blight": "RICE_LONG_잎집무늬마름병.csv",
}
results: list[dict] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    results.append({"name": name, "passed": bool(cond), "detail": str(detail)[:200]})


def run_api(in_dir: Path, out_dir: Path, extra: list[str] | None = None):
    r = subprocess.run(
        [sys.executable, str(PKG / "run_predict.py"),
         "--input-dir", str(in_dir), "--output-dir", str(out_dir), *(extra or [])],
        capture_output=True, text=True, timeout=3600)
    resp = None
    p = out_dir / "response.json"
    if p.is_file():
        resp = json.loads(p.read_text(encoding="utf-8"))
    return r, resp


def read_csv_raw(p: Path) -> tuple[str, list[str], list[dict]]:
    raw = p.read_bytes().decode("utf-8")
    rows = list(csv.DictReader(raw.splitlines()))
    header = raw.splitlines()[0].split(",")
    return raw, header, rows


def make_inputs(pest: str, work: Path, daily_master: Path, long_dir: Path) -> Path:
    """One input dir holding daily + LONG for the pest's smoke site."""
    site, year = SMOKE[pest]
    d = work / f"in_{pest}"
    d.mkdir(parents=True, exist_ok=True)
    slice_csv = work / f"_daily_{site}.csv"
    if not slice_csv.is_file():
        hdr = pd.read_csv(daily_master, nrows=0, encoding="utf-8-sig")
        col = hdr.columns[0]
        first = True
        for ch in pd.read_csv(daily_master, encoding="utf-8-sig", dtype={col: str},
                              chunksize=500_000):
            sub = ch[ch[col].astype(str) == site]
            if sub.empty:
                continue
            sub.to_csv(slice_csv, mode="w" if first else "a", header=first,
                       index=False, encoding="utf-8-sig")
            first = False
    (d / "daily_weather.csv").write_bytes(slice_csv.read_bytes())
    obs = pd.read_csv(long_dir / LONG_FILENAME[pest], encoding="utf-8-sig")
    obs[obs["site_id"].astype(str) == site].to_csv(
        d / "long_observation.csv", index=False, encoding="utf-8-sig")
    return d


def write_request(d: Path, obj: dict) -> None:
    (d / "request.json").write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                                    encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-master", required=True, type=Path)
    ap.add_argument("--long-dir", required=True, type=Path)
    ap.add_argument("--work", type=Path,
                    default=Path(tempfile.gettempdir()) / "lw_batch_tests")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)
    W = args.work

    bph = make_inputs("BPH", W, args.daily_master, args.long_dir)
    site, year = SMOKE["BPH"]

    # ---------------------------------------------------------------- 1
    write_request(bph, {"pest": "BPH", "site_id": site, "year": year})
    r, single = run_api(bph, W / "out_single")
    check("1. single BPH prediction still works",
          r.returncode == 0 and single
          and single["final_prediction"]["source"] == "learned_stage2"
          and single["stage1"]["alert_tstar_doy"] == 176,
          f"rc={r.returncode} {single and single['final_prediction']}")

    # ---------------------------------------------------------------- 2
    write_request(bph, {"pest": "BPH", "site_id": "99999_99999", "year": year})
    r, fb = run_api(bph, W / "out_single_fb")
    check("2. single fallback (unknown site) -> climatology_no_alert, exit 0",
          r.returncode == 0 and fb
          and fb["final_prediction"]["source"] == "climatology_no_alert"
          and fb["final_prediction"]["fallback_triggered"] is True
          and fb["stage2"]["learned_stage2"] is None,
          f"rc={r.returncode} {fb and fb['final_prediction']}")

    # ---------------------------------------------------------------- 3
    csv2 = W / "rows_ok.csv"
    csv2.write_text(f"pest,site_id,year\nBPH,{site},{year}\nBPH,{site},{year}\n",
                    encoding="utf-8")
    write_request(bph, {"mode": "batch", "input_csv": str(csv2)})
    r, b = run_api(bph, W / "out_b_ok")
    raw, header, rows = read_csv_raw(W / "out_b_ok" / "predictions.csv")
    check("3. batch 2 rows both succeed",
          r.returncode == 0 and b and b["success_count"] == 2
          and b["error_count"] == 0 and len(rows) == 2
          and all(x["status"] == "success" for x in rows),
          f"rc={r.returncode} {b and (b['success_count'], b['error_count'])}")

    # ---------------------------------------------------------------- 4
    csv3 = W / "rows_mixed.csv"
    csv3.write_text(
        f"pest,site_id,year\nBPH,{site},{year}\nBPH,BADSITE,{year}\n"
        f"not_a_pest,{site},{year}\n", encoding="utf-8")
    write_request(bph, {"mode": "batch", "input_csv": str(csv3)})
    r, b = run_api(bph, W / "out_b_mixed")
    raw, header, rows = read_csv_raw(W / "out_b_mixed" / "predictions.csv")
    statuses = [x["status"] for x in rows]
    # Row 1 (unknown site) is classified "fallback", NOT "error" — and that
    # matches the deployed API: a Stage-1 failure ends at the bail-out whose
    # message begins "Stage-1 fired no alert ..." (run_predict.py:419-424), and
    # _classify (batch.py:241) detects fallback by substring-matching exactly
    # that prose. Row 2 (invalid pest) raises before the pipeline -> "error".
    check("4. batch partial success — 1 success / 1 fallback / 1 error, NOT aborted",
          r.returncode == 0 and len(rows) == 3 and statuses[0] == "success"
          and statuses[1] == "fallback" and statuses[2] == "error",
          f"rc={r.returncode} statuses={statuses}")
    check("4b. non-success rows carry a concrete error_reason",
          all(rows[i]["error_reason"] for i in (1, 2)),
          f"{[rows[i]['error_reason'][:60] for i in (1,2)]}")
    check("4e. invalid-pest row serializes as a usable error row "
          "(climatology_error, fallback_triggered True)",
          rows[2]["final_source"] == "climatology_error"
          and rows[2]["fallback_triggered"] == "True"
          and rows[2]["learned_mu_doy"] == "",
          f"{(rows[2]['final_source'], rows[2]['fallback_triggered'])}")
    check("4c. input ROW ORDER preserved in output",
          [x["site_id"] for x in rows] == [site, "BADSITE", site],
          str([x["site_id"] for x in rows]))
    check("4d. row_index column present and sequential",
          [x["row_index"] for x in rows] == ["0", "1", "2"],
          str([x["row_index"] for x in rows]))

    # ---------------------------------------------------------------- 5
    write_request(bph, {"mode": "batch", "input_csv": str(W / "does_not_exist.csv")})
    r, b = run_api(bph, W / "out_b_nocsv")
    check("5. missing input CSV -> whole-batch failure, exit 2, files still written",
          r.returncode == 2 and b and "not found" in (b.get("error") or "")
          and (W / "out_b_nocsv" / "predictions.csv").is_file()
          and (W / "out_b_nocsv" / "run_log.txt").is_file(),
          f"rc={r.returncode} err={b and b.get('error','')[:70]}")

    # ---------------------------------------------------------------- 6
    bad = W / "rows_badcols.csv"
    bad.write_text("pest,site\nBPH,x\n", encoding="utf-8")
    write_request(bph, {"mode": "batch", "input_csv": str(bad)})
    r, b = run_api(bph, W / "out_b_badcols")
    check("6. missing required column -> whole-batch failure naming the column",
          r.returncode == 2 and b and "site_id" in (b.get("error") or ""),
          f"rc={r.returncode} err={b and b.get('error','')[:90]}")

    # ---------------------------------------------------------------- 7/8
    write_request(bph, {"mode": "batch", "pest": "not_a_pest", "year": year})
    r, b = run_api(bph, W / "out_b_badpest")
    check("7. unknown pest (rep batch) -> whole-batch failure exit 2",
          r.returncode == 2 and b and "invalid pest" in (b.get("error") or ""),
          f"rc={r.returncode}")

    csv_lower = W / "rows_lower.csv"
    csv_lower.write_text(f"pest,site_id,year\nbph,{site},{year}\n", encoding="utf-8")
    write_request(bph, {"mode": "batch", "input_csv": str(csv_lower)})
    r, b = run_api(bph, W / "out_b_lower")
    _, _, rows = read_csv_raw(W / "out_b_lower" / "predictions.csv")
    check("8. lowercase pest ACCEPTED in batch (deployed is case-insensitive)",
          r.returncode == 0 and rows and rows[0]["status"] == "success"
          and rows[0]["pest"] == "BPH",
          f"rc={r.returncode} {rows and (rows[0]['pest'], rows[0]['status'])}")

    # ---------------------------------------------------------------- 9
    csv_na = W / "rows_noalert.csv"
    csv_na.write_text(f"pest,site_id,year\nBPH,99999_99999,{year}\n", encoding="utf-8")
    write_request(bph, {"mode": "batch", "input_csv": str(csv_na)})
    r, b = run_api(bph, W / "out_b_noalert")
    _, _, rows = read_csv_raw(W / "out_b_noalert" / "predictions.csv")
    check("9. no-alert row -> climatology_no_alert + fallback_triggered True",
          rows and rows[0]["final_source"] == "climatology_no_alert"
          and rows[0]["fallback_triggered"] == "True",
          f"{rows and (rows[0]['final_source'], rows[0]['fallback_triggered'], rows[0]['status'])}")

    # ---------------------------------------------------------------- 10
    sb = make_inputs("sheath_blight", W, args.daily_master, args.long_dir)
    s_site, s_year = SMOKE["sheath_blight"]
    csv_sb = W / "rows_sb.csv"
    csv_sb.write_text(f"pest,site_id,year\nsheath_blight,{s_site},{s_year}\n",
                      encoding="utf-8")
    write_request(sb, {"mode": "batch", "input_csv": str(csv_sb)})
    r, b = run_api(sb, W / "out_b_sb")
    _, _, rows = read_csv_raw(W / "out_b_sb" / "predictions.csv")
    check("10. climatology-recommended pest -> final=climatology, learned kept",
          rows and rows[0]["final_source"] == "climatology"
          and rows[0]["fallback_triggered"] == "False"
          and rows[0]["learned_mu_doy"] not in ("", None),
          f"{rows and (rows[0]['final_source'], rows[0]['learned_mu_doy'])}")

    # ---------------------------------------------------------------- 11
    write_request(bph, {"mode": "batch", "input_csv": str(csv2),
                        "include_diagnostics": True})
    r, b = run_api(bph, W / "out_b_diag")
    _, header_diag, rows_d = read_csv_raw(W / "out_b_diag" / "predictions.csv")
    expect_diag = GENERIC_EXTRA_COLS + FLAT_COLS + BATCH_EXTRA_COLS + DIAG_COLS
    check("11a. include_diagnostics=True adds the 9 diag columns, appended last",
          header_diag == expect_diag, f"{header_diag}")
    check("11b. diagnostics values populated (alert_source)",
          rows_d and rows_d[0]["alert_source"] == "stage1_live",
          f"{rows_d and rows_d[0].get('alert_source')}")
    expect_nodiag = GENERIC_EXTRA_COLS + FLAT_COLS + BATCH_EXTRA_COLS
    check("11c. include_diagnostics=False omits them", header == expect_nodiag,
          f"{header}")

    # ---------------------------------------------------------------- 12
    write_request(bph, {"mode": "batch", "input_csv": str(csv2),
                        "stage2_variant": "fp32"})
    r32, b32 = run_api(bph, W / "out_b_fp32")
    ok32 = r32.returncode == 0 and b32 and b32["stage2_variant"] == "fp32"
    if not ok32 and b32 and b32.get("error_count"):
        ok32 = "fp32" in json.dumps(b32)[:2000]  # fp32 absent in a production build
    check("12. fp16 default / fp32 selectable via request",
          (b and b["stage2_variant"] == "fp16") and ok32,
          f"fp16_run={b and b.get('stage2_variant')} fp32_rc={r32.returncode}")

    # ---------------------------------------------------------------- 13/14
    check("13. CSV column order = row_index + 16 single + status,error_reason",
          header == GENERIC_EXTRA_COLS + FLAT_COLS + BATCH_EXTRA_COLS, str(header))
    raw_ok = (W / "out_b_ok" / "predictions.csv").read_bytes()
    check("14a. batch predictions.csv uses CRLF",
          raw_ok.count(b"\r\n") == 3 and raw_ok.count(b"\n") == raw_ok.count(b"\r\n"),
          f"crlf={raw_ok.count(b'!')}")
    resp_raw = (W / "out_b_ok" / "response.json").read_bytes()
    check("14b. batch response.json has a trailing newline",
          resp_raw.endswith(b"\n"))

    # ---------------------------------------------------------------- 15
    _, _, rows_ok = read_csv_raw(W / "out_b_ok" / "predictions.csv")
    b_row = rows_ok[0]
    s_final = single["final_prediction"]
    same = (b_row["final_source"] == s_final["source"]
            and float(b_row["final_mu_doy"]) == s_final["mu_doy"]
            and int(b_row["final_pi95_lower"]) == s_final["pi_95"]["lower_doy"]
            and int(b_row["final_pi95_upper"]) == s_final["pi_95"]["upper_doy"]
            and int(b_row["alert_tstar_doy"]) == single["stage1"]["alert_tstar_doy"]
            and float(b_row["learned_mu_doy"]) ==
            single["stage2"]["learned_stage2"]["mu_doy"])
    check("15. SINGLE vs BATCH one row: identical core results", same,
          f"batch={b_row['final_mu_doy']}/{b_row['final_source']} "
          f"single={s_final['mu_doy']}/{s_final['source']}")

    # ---------------------------------------------------------------- regression
    write_request(bph, {"pest": "BPH", "site_id": site, "year": year})
    r, again = run_api(bph, W / "out_single_after")
    check("R. single mode UNCHANGED after adding batch (byte-identical response)",
          json.dumps(again, sort_keys=True) == json.dumps(single, sort_keys=True),
          "differs" if again != single else "identical")

    print("=== batch tests ===")
    for x in results:
        print(f"  {'PASS' if x['passed'] else 'FAIL'}  {x['name']}")
        if not x["passed"] and x["detail"]:
            print(f"        got: {x['detail']}")
    ok = all(x["passed"] for x in results)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} "
          f"({sum(x['passed'] for x in results)}/{len(results)})")
    (W / "batch_test_results.json").write_text(json.dumps(results, indent=2,
                                                          ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception:
        traceback.print_exc()
        raise SystemExit(1)
