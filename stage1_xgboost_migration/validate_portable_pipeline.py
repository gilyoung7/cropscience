"""Run the Stage-1 alert pipeline from portable assets ONLY, and check it against
the alerts the original .pt path produced on the server.

Hard constraint, asserted at exit: this process must never import torch, sklearn or
pandas, and must never open a .pt. It consumes only:
    artifacts/<pest>/<A|D>/model.json
    artifacts/<pest>/<A|D>/calibration.json
    artifacts/<pest>/<A|D>/metadata.json
    artifacts/<pest>/gate.json
    fixtures/alert/<pest>/*.npz        base samples + history + reference alerts

The fixtures are written by validate_alert_parity.py on the server. This script is
what should run on the MacBook / any modern XGBoost.

Usage:
    ../.venv/bin/python validate_portable_pipeline.py          # xgboost 3.0.5, no torch
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import sys
from pathlib import Path

import numpy as np

# Audit hook must be installed before anything else can touch the filesystem: it
# makes "never opens a .pt / never unpickles" an enforced property of this run
# rather than a claim in a docstring.
_VIOLATIONS: list[str] = []


def _audit(event: str, payload):
    if event == "open":
        p = str(payload[0])
        if p.endswith(".pt"):
            _VIOLATIONS.append(f"opened checkpoint: {p}")
    elif event in ("pickle.find_class", "pickle.loads"):
        _VIOLATIONS.append(f"unpickled: {event} {payload!r:.80}")


sys.addaudithook(_audit)

import portable_stage1 as ps  # noqa: E402

ROOT = Path(__file__).resolve().parent
FIXTURES = ROOT / "fixtures" / "alert"
REPORTS = ROOT / "reports"
PESTS = ["BPH", "WBPH", "bacterial_blight", "blast", "brown_spot",
         "rice_stem_borer_1", "rice_stem_borer_2", "sheath_blight"]

# torch is the hard constraint: it is what loads the .pt and unpickles the
# XGBClassifier. sklearn/pandas/scipy are NOT forbidden -- importing xgboost pulls
# them in by itself (verified), so banning them would ban xgboost. They are
# reported for transparency, not enforced.
FORBIDDEN = ("torch",)
REPORTED_ONLY = ("sklearn", "pandas", "scipy")


def load_fixture(pest: str):
    """Load the replay fixture with allow_pickle=False -- the portable path must
    never unpickle anything, which is the whole point of the migration."""
    f = FIXTURES / f"{pest}.npz"
    sidecar = FIXTURES / f"{pest}.json"
    if not (f.exists() and sidecar.exists()):
        return None
    z = np.load(f, allow_pickle=False)
    meta = json.loads(sidecar.read_text(encoding="utf-8"))
    base = [{"site_id": sid, "year": int(y), "X": X,
             "censor_type": ct, "L": int(L), "R": int(R)}
            for sid, y, X, ct, L, R in zip(meta["site_ids"], z["year"], z["X"],
                                           meta["censor_type"], z["L"], z["R"])]
    history = {tuple(k.rsplit("|", 1)[:1] + [int(k.rsplit("|", 1)[1])]): v
               for k, v in meta["history"].items()}
    history = {(k[0], k[1]): v for k, v in history.items()}
    return {"base": base, "history": history,
            "reference": meta["reference_alerts"], "data_kind": meta["data_kind"]}


def run_pest(pest: str) -> dict:
    rec: dict = {"pest": pest}
    fx = load_fixture(pest)
    if fx is None:
        return dict(rec, status="no_fixture",
                    error=f"missing {FIXTURES/(pest+'.npz')}; regenerate on the server")
    gate = ps.load_gate(pest)
    A = ps.PortableBranch(pest, "A")
    D = ps.PortableBranch(pest, "D")
    doy_start = A.doy_start

    per_A = A.forward_one(fx["base"], fx["history"])
    per_D = D.forward_one(fx["base"], fx["history"])

    rows, mismatches = [], 0
    for s in fx["base"]:
        sy = (str(s["site_id"]), int(s["year"]))
        h = fx["history"].get(sy)
        with_h = (h is not None and int(h["prev_year_L_miss"]) == 0)
        got = ps.alert_for_sy(gate, per_A, per_D, sy, with_h, doy_start)
        want = fx["reference"].get(f"{sy[0]}|{sy[1]}")

        got_doy = got["alert_tstar_doy"] if got else None
        want_doy = want["alert_tstar_doy"] if want else None
        ok = got_doy == want_doy
        disp_ok = True
        if got and want:
            for k, v in want["dispatch_features"].items():
                g = got["dispatch_features"].get(k)
                if isinstance(v, float) and isinstance(g, float):
                    if not ((v == g) or (np.isnan(v) and np.isnan(g))):
                        disp_ok = False
                elif v != g:
                    disp_ok = False
        if not (ok and disp_ok):
            mismatches += 1
        rows.append({"site_year": f"{sy[0]}|{sy[1]}",
                     "alert_doy_portable": got_doy, "alert_doy_reference": want_doy,
                     "alert_match": ok, "dispatch_match": disp_ok,
                     "fired_portable": got is not None, "fired_reference": want is not None,
                     "dispatch_branch": (got or {}).get("dispatch_features", {}).get("dispatch_branch")})

    rec.update(
        status="ok" if mismatches == 0 else "mismatch",
        data_kind=fx["data_kind"],
        n_site_years=len(rows),
        n_fired_portable=sum(r["fired_portable"] for r in rows),
        n_fired_reference=sum(r["fired_reference"] for r in rows),
        n_mismatch=mismatches,
        gate={k: gate[k] for k in ["method", "k", "tau", "tau_no", "tau_with"]},
        temperature_A=A.temperature, temperature_D=D.temperature,
        site_years=rows,
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", action="append")
    args = ap.parse_args()

    records = [run_pest(p) for p in (args.pest or PESTS)]
    for r in records:
        print(f"[{r['status']:<10}] {r['pest']:<18} {r.get('data_kind','-'):<30} "
              f"sy={r.get('n_site_years','-'):<5} "
              f"fired {r.get('n_fired_portable','-')}/{r.get('n_fired_reference','-')} "
              f"mismatch={r.get('n_mismatch','-')} {r.get('error','')}")

    leaked = sorted(m for m in FORBIDDEN if m in sys.modules)
    import xgboost
    ok = sum(r["status"] == "ok" for r in records)
    clean = not leaked and not _VIOLATIONS
    report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "env": {"python": sys.version.split()[0], "xgboost": xgboost.__version__,
                "numpy": np.__version__, "executable": sys.executable},
        "torch_free": {
            "forbidden_modules": list(FORBIDDEN), "imported": leaked,
            "checkpoint_or_pickle_violations": _VIOLATIONS,
            "clean": clean,
            "pulled_in_by_xgboost_itself": sorted(
                m for m in REPORTED_ONLY if m in sys.modules),
            "note": ("sklearn/pandas/scipy are imported by xgboost itself, not by this "
                     "pipeline; torch is the constraint that matters and is absent."),
        },
        "totals": {"pests": len(records), "ok": ok, "not_ok": len(records) - ok},
        "pests": records,
    }
    REPORTS.mkdir(parents=True, exist_ok=True)
    (REPORTS / "portable_pipeline_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")

    print(f"\ntorch imported: {'torch' in sys.modules}   "
          f".pt opened / unpickled: {_VIOLATIONS or 'none'}")
    print(f"(xgboost itself pulled in: "
          f"{sorted(m for m in REPORTED_ONLY if m in sys.modules)})")
    print(f"portable pipeline ok {ok}/{len(records)}  (xgboost {xgboost.__version__})")
    return 0 if ok == len(records) and clean else 1


if __name__ == "__main__":
    raise SystemExit(main())
