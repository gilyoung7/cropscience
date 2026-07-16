"""Replay the server's Stage-1 predictions on a newer XGBoost install.

Run this on the MacBook (or any modern env) after copying stage1_xgboost_migration/
across. It loads each portable model.json into a fresh XGBClassifier, feeds the
fixture inputs, and compares against the reference predictions recorded on the
server. No torch, no .pt, no pickle involved.

Usage:
    python -m venv .venv && ./.venv/bin/pip install -r requirements-modern.txt
    ./.venv/bin/python validate_modern_xgboost.py
    ./.venv/bin/python validate_modern_xgboost.py --tolerance 1e-7   # if not bit-exact
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import xgboost

ROOT = Path(__file__).resolve().parent
ARTIFACTS = ROOT / "artifacts"
FIXTURES = ROOT / "fixtures"
REPORTS = ROOT / "reports"

PESTS = ["BPH", "WBPH", "bacterial_blight", "blast", "brown_spot",
         "rice_stem_borer_1", "rice_stem_borer_2", "sheath_blight"]
BRANCHES = ["A", "D"]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def check_one(pest: str, branch: str, tol: float) -> dict:
    rec = {"pest": pest, "branch": branch}
    mdir = ARTIFACTS / pest / branch
    model_json, meta_json = mdir / "model.json", mdir / "metadata.json"
    if not model_json.exists():
        return dict(rec, status="skipped", error=f"missing {model_json}")

    meta = json.loads(meta_json.read_text(encoding="utf-8"))
    actual_sha = sha256_file(model_json)
    rec["model_json_sha256_ok"] = (actual_sha == meta["exported_json_sha256"])
    if not rec["model_json_sha256_ok"]:
        return dict(rec, status="failed",
                    error=(f"model.json sha256 mismatch: file={actual_sha} "
                           f"metadata={meta['exported_json_sha256']}"))

    try:
        model = xgboost.XGBClassifier()
        model.load_model(str(model_json))
    except Exception as exc:
        return dict(rec, status="failed", stage="load_model",
                    error=f"{type(exc).__name__}: {exc}")
    rec["load_ok"] = True
    rec["n_features_in_"] = int(getattr(model, "n_features_in_", -1))
    rec["expected_n_features_in_"] = meta["n_features_in_"]
    rec["classes_"] = [int(c) for c in getattr(model, "classes_", [])]
    rec["n_features_match"] = rec["n_features_in_"] == meta["n_features_in_"]
    rec["classes_match"] = rec["classes_"] == meta["classes_"]

    cases = {}
    for name in ("synthetic", "real"):
        xf = FIXTURES / pest / branch / f"X_{name}.npy"
        pf = FIXTURES / pest / branch / f"proba_source_{name}.npy"
        if not (xf.exists() and pf.exists()):
            continue
        X = np.load(xf)
        p_ref = np.load(pf)
        p_new = model.predict_proba(X)
        diff = np.abs(p_ref.astype(np.float64) - p_new.astype(np.float64))
        cases[name] = {
            "n_rows": int(X.shape[0]),
            "input_sha256": sha256_file(xf),
            "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "bit_exact": bool(np.array_equal(p_ref, p_new)),
            "within_tolerance": bool(diff.max() <= tol),
            "class_agreement_rate": float(
                (p_ref.argmax(1) == p_new.argmax(1)).mean()),
            "shape_ref": list(p_ref.shape), "shape_new": list(p_new.shape),
            "dtype_ref": str(p_ref.dtype), "dtype_new": str(p_new.dtype),
            "has_nan": bool(np.isnan(p_new).any()),
            "has_inf": bool(np.isinf(p_new).any()),
        }
    rec["cases"] = cases
    if not cases:
        return dict(rec, status="no_fixture",
                    error="no fixtures found; regenerate them on the server "
                          "(see README) and copy fixtures/ across")
    rec["status"] = ("ok" if all(c["within_tolerance"] for c in cases.values())
                     else "mismatch")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tolerance", type=float, default=0.0,
                    help="max allowed abs diff vs server reference (default 0 = bit-exact)")
    args = ap.parse_args()

    records = [check_one(p, b, args.tolerance) for p in PESTS for b in BRANCHES]
    for r in records:
        summary = "  ".join(
            f"{k}: max|d|={v['max_abs_diff']:.3e} exact={v['bit_exact']}"
            for k, v in r.get("cases", {}).items())
        print(f"[{r['status']:<9}] {r['pest']:<18} {r['branch']}  "
              f"{summary or r.get('error', '')}")

    ok = sum(r["status"] == "ok" for r in records)
    REPORTS.mkdir(parents=True, exist_ok=True)
    report = {
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "tolerance": args.tolerance,
        "modern_env": {"python": sys.version.split()[0],
                       "xgboost": xgboost.__version__,
                       "numpy": np.__version__},
        "totals": {"attempted": len(records), "passed": ok,
                   "failed": len(records) - ok},
        "models": records,
    }
    out = REPORTS / "modern_xgboost_report.json"
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"\npassed {ok}/{len(records)}  (xgboost {xgboost.__version__}) -> {out}")
    return 0 if ok == len(records) else 1


if __name__ == "__main__":
    raise SystemExit(main())
