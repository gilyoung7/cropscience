"""8-pest end-to-end smoke test against the golden reference values.

GOLDEN REFERENCE COMPARISON — the original API cannot run on this MacBook (its
Stage-1 .pt checkpoints hold a legacy binary booster that xgboost >= 2 rejects
with SIGSEGV, exit 139, reproduced on 2.1.4 and 3.1.2). So the expected values
below are NOT produced live from the original here; they are the recorded values
from api_handoff_transformer/README.md §10, which the deployed API emitted on the
server. Everything on the lightweight side IS computed live.

    python tests/test_smoke_cases.py --daily-master "$DAILY_MASTER" --long-dir "$LONG_DIR"
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO = PKG.parent

# README §10 smoke record (server-produced; golden reference).
GOLDEN = {
    "BPH": ("33210_56298", 2004, 176, 234.32, "learned_stage2"),
    "WBPH": ("33908_67063", 2011, 171, 245.82, "learned_stage2"),
    "bacterial_blight": ("30247_65595", 2010, 206, 268.32, "climatology"),
    "blast": ("36582_63441", 2017, 125, 198.08, "climatology"),
    "brown_spot": ("31522_54338", 2022, 157, 224.93, "climatology"),
    "rice_stem_borer_1": ("35474_56809", 2018, 125, 184.17, "climatology"),
    "rice_stem_borer_2": ("31959_58947", 2014, 186, 260.97, "climatology"),
    "sheath_blight": ("35694_60137", 2004, 118, 192.72, "climatology"),
}
LONG_FILENAME = {
    "BPH": "RICE_LONG_벼멸구.csv", "WBPH": "RICE_LONG_흰등멸구.csv",
    "bacterial_blight": "RICE_LONG_흰잎마름병.csv", "blast": "RICE_LONG_잎도열병.csv",
    "brown_spot": "RICE_LONG_깨씨무늬병.csv",
    "rice_stem_borer_1": "RICE_LONG_이화명나방1화기.csv",
    "rice_stem_borer_2": "RICE_LONG_이화명나방2화기.csv",
    "sheath_blight": "RICE_LONG_잎집무늬마름병.csv",
}
# fp16 rounding can move the reported 2-dp mu_doy by 0.01 vs the reference.
MU_TOL_DAYS = 0.02


def ensure_inputs(pest: str, daily_master: Path, long_dir: Path, work: Path) -> Path:
    site, year, *_ = GOLDEN[pest]
    d = work / pest
    d.mkdir(parents=True, exist_ok=True)
    if not (d / "daily_weather.csv").is_file():
        slice_csv = work / f"_daily_{site}.csv"
        if not slice_csv.is_file():
            hdr = pd.read_csv(daily_master, nrows=0, encoding="utf-8-sig")
            col = hdr.columns[0]
            first = True
            for ch in pd.read_csv(daily_master, encoding="utf-8-sig",
                                  dtype={col: str}, chunksize=500_000):
                sub = ch[ch[col].astype(str) == site]
                if sub.empty:
                    continue
                sub.to_csv(slice_csv, mode="w" if first else "a", header=first,
                           index=False, encoding="utf-8-sig")
                first = False
            if first:
                raise FileNotFoundError(f"site {site} not in {daily_master}")
        (d / "daily_weather.csv").write_bytes(slice_csv.read_bytes())
    if not (d / "long_observation.csv").is_file():
        obs = pd.read_csv(long_dir / LONG_FILENAME[pest], encoding="utf-8-sig")
        obs[obs["site_id"].astype(str) == site].to_csv(
            d / "long_observation.csv", index=False, encoding="utf-8-sig")
    (d / "request.json").write_text(json.dumps(
        {"pest": pest, "site_id": site, "year": year, "include_diagnostics": True},
        indent=2, ensure_ascii=False))
    return d


def run_one(pest: str, in_dir: Path, out_dir: Path, variant: str) -> dict:
    site, year, g_alert, g_mu, g_source = GOLDEN[pest]
    r = subprocess.run(
        [sys.executable, str(PKG / "run_predict.py"),
         "--input-dir", str(in_dir), "--output-dir", str(out_dir),
         "--stage2-variant", variant],
        capture_output=True, text=True, timeout=1800,
    )
    resp_p = out_dir / "response.json"
    if not resp_p.is_file():
        return {"pest": pest, "passed": False, "exit": r.returncode,
                "error": f"no response.json; stderr={r.stderr[-300:]}"}
    resp = json.loads(resp_p.read_text())
    learned = resp["stage2"]["learned_stage2"]
    final = resp["final_prediction"]
    alert = resp["stage1"]["alert_tstar_doy"]
    out = {
        "pest": pest, "site": site, "year": year, "variant": variant,
        "exit": r.returncode,
        "alert_live": alert, "alert_golden": g_alert, "alert_ok": alert == g_alert,
        "learned_mu_doy": (learned or {}).get("mu_doy"),
        "golden_mu_doy": g_mu,
        "mu_delta": (abs((learned or {}).get("mu_doy", float("nan")) - g_mu)
                     if learned else None),
        "final_source": final["source"], "golden_final_source": g_source,
        "source_ok": final["source"] == g_source,
        "alert_source": resp["stage1"]["alert_source"],
        "output_status": (learned or {}).get("output_status"),
        "predictions_csv": (out_dir / "predictions.csv").is_file(),
        "run_log": (out_dir / "run_log.txt").is_file(),
    }
    out["mu_ok"] = bool(learned and out["mu_delta"] <= MU_TOL_DAYS)
    out["passed"] = bool(out["alert_ok"] and out["mu_ok"] and out["source_ok"]
                         and r.returncode == 0 and out["predictions_csv"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-master", required=True, type=Path)
    ap.add_argument("--long-dir", required=True, type=Path)
    ap.add_argument("--variant", default="fp16", choices=["fp16", "fp32"])
    ap.add_argument("--pests", nargs="*", default=list(GOLDEN))
    ap.add_argument("--work", type=Path,
                    default=Path(tempfile.gettempdir()) / "lw_api_smoke")
    args = ap.parse_args()
    args.work.mkdir(parents=True, exist_ok=True)

    rows, failures = [], []
    for pest in args.pests:
        try:
            in_dir = ensure_inputs(pest, args.daily_master, args.long_dir, args.work)
            rows.append(run_one(pest, in_dir, args.work / f"out_{pest}", args.variant))
        except Exception as e:
            failures.append({"pest": pest, "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()})
            print(f"  FAIL {pest}: {type(e).__name__}: {str(e)[:140]}")

    print(f"\n=== 8-pest end-to-end ({args.variant}) — live vs golden reference ===")
    print(f"{'pest':<19}{'alert':>7}{'gold':>6}{'ok':>4}{'mu_live':>10}{'gold_mu':>9}"
          f"{'d':>7}{'final_source':>18}{'ok':>4}{'res':>6}")
    for r in rows:
        mu_s = "-" if r["learned_mu_doy"] is None else f"{r['learned_mu_doy']:.2f}"
        d_s = "-" if r["mu_delta"] is None else f"{r['mu_delta']:.3f}"
        print(f"{r['pest']:<19}{str(r['alert_live']):>7}{r['alert_golden']:>6}"
              f"{'Y' if r['alert_ok'] else 'N':>4}{mu_s:>10}{r['golden_mu_doy']:>9}"
              f"{d_s:>7}{r['final_source']:>18}{'Y' if r['source_ok'] else 'N':>4}"
              f"{'PASS' if r['passed'] else 'FAIL':>6}")

    ok = bool(rows) and all(r["passed"] for r in rows) and not failures
    print(f"\npassed {sum(r['passed'] for r in rows)}/{len(rows)}; "
          f"{len(failures)} errored")
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")
    (args.work / "smoke_results.json").write_text(
        json.dumps({"rows": rows, "failures": failures}, indent=2, ensure_ascii=False))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
