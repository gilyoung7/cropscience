"""Stage-1 portable tests, including the ARRAY MEMORY LAYOUT regression.

The layout test is the important one. The deployed API builds base X with
`X_df.to_numpy(dtype=np.float32)` (stage1.py:720), which is F-contiguous, and
then appends history with `np.concatenate`, which is C-contiguous. float32
mean/std depend on reduction order, so "normalizing" the layout silently shifts
features by ~1e-8 — enough to flip a probability across tau and move the alert
DOY. This pins the deployed layout and proves the difference is real.

    ../.venv-tflite/bin/python tests/test_stage1_portable.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from infer.stage1_features import (  # noqa: E402
    append_history, array_layout, base_x_from_season, build_nowcast_samples,
    build_tabular,
)
from infer.stage1_portable import PortableBranch, load_gate  # noqa: E402

PESTS = ["BPH", "WBPH", "bacterial_blight", "blast", "brown_spot",
         "rice_stem_borer_1", "rice_stem_borer_2", "sheath_blight"]

# Measured from the deployed API on this machine (pandas 3.0.3 / numpy 2.5.1).
# See docs/lightweight_api_integration_plan.md §4.
EXPECTED_BASE_LAYOUT = {"c_contiguous": False, "f_contiguous": True}
EXPECTED_AFTER_HISTORY = {"c_contiguous": True, "f_contiguous": False}


def _season(n_cols: int, T: int = 131) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = [f"c{i}" for i in range(n_cols)]
    return pd.DataFrame(rng.normal(size=(T, n_cols)).astype(np.float32), columns=cols)


def test_base_layout_is_fortran() -> dict:
    season = _season(6)
    X = base_x_from_season(season, [f"c{i}" for i in range(6)])
    lay = array_layout(X)
    ok = (lay["c_contiguous"] == EXPECTED_BASE_LAYOUT["c_contiguous"]
          and lay["f_contiguous"] == EXPECTED_BASE_LAYOUT["f_contiguous"])
    return {"name": "base X is F-contiguous (matches deployed to_numpy)",
            "layout": lay, "passed": ok}


def test_history_append_is_c() -> dict:
    season = _season(6)
    X = base_x_from_season(season, [f"c{i}" for i in range(6)])
    XD = append_history(X, "s", 2004, {}, 140)
    lay = array_layout(XD)
    ok = (lay["c_contiguous"] == EXPECTED_AFTER_HISTORY["c_contiguous"]
          and lay["f_contiguous"] == EXPECTED_AFTER_HISTORY["f_contiguous"])
    return {"name": "after append_history X is C-contiguous (np.concatenate)",
            "layout": lay, "passed": ok}


def test_layout_actually_changes_features() -> dict:
    """The regression this guards: forcing C-order changes the tabular features.

    If this ever reports 'no difference', the guard has become vacuous and the
    layout pinning above is no longer protecting anything — investigate before
    relaxing it.
    """
    season = _season(6)
    X_f = base_x_from_season(season, [f"c{i}" for i in range(6)])
    X_c = np.ascontiguousarray(X_f)          # the tempting "cleanup"
    s = {"site_id": "s", "year": 2004, "L": 1, "R": 1, "censor_type": "right"}
    nc_f = build_nowcast_samples([dict(s, X=X_f)], 28, 1, False, "mid")
    nc_c = build_nowcast_samples([dict(s, X=X_c)], 28, 1, False, "mid")
    t_f = build_tabular(nc_f, True)
    t_c = build_tabular(nc_c, True)
    same = bool(np.array_equal(t_f, t_c))
    maxd = float(np.abs(t_f - t_c).max())
    return {
        "name": "F->C 'cleanup' changes tabular features (guard is not vacuous)",
        "max_abs_diff": maxd, "identical": same,
        # We EXPECT a difference. Identical would mean the guard proves nothing.
        "passed": (not same) and maxd > 0.0,
    }


def test_branch_layouts_all_pests() -> dict:
    """Record the real per-branch layout for every pest/branch."""
    rows, ok = [], True
    for pest in PESTS:
        for branch in ("A", "D"):
            pb = PortableBranch(pest, branch, PKG / "assets" / "stage1")
            season = _season(len(pb.feature_cols), T=pb.doy_end - pb.doy_start + 1)
            season.columns = pb.feature_cols
            X = base_x_from_season(season, pb.feature_cols)
            base_lay = array_layout(X)
            if pb.site_history_added:
                X2 = append_history(X, "s", 2004, {}, pb.doy_start)
                fed_lay = array_layout(X2)
                exp = EXPECTED_AFTER_HISTORY
            else:
                fed_lay = base_lay
                exp = EXPECTED_BASE_LAYOUT
            good = (fed_lay["c_contiguous"] == exp["c_contiguous"]
                    and fed_lay["f_contiguous"] == exp["f_contiguous"])
            ok &= good
            rows.append({"pest": pest, "branch": branch,
                         "history_appended": pb.site_history_added,
                         "fed_layout": fed_lay, "passed": good})
    return {"name": "per-pest/branch fed layout matches the deployed contract",
            "rows": rows, "passed": ok}


def test_models_load_without_sklearn() -> dict:
    """All 16 models load via xgboost.Booster; scikit-learn must not be needed."""
    loaded, errs = 0, []
    for pest in PESTS:
        for branch in ("A", "D"):
            try:
                pb = PortableBranch(pest, branch, PKG / "assets" / "stage1")
                n = pb.booster.num_features()
                X = np.zeros((2, n), dtype=np.float32)
                p = pb.predict_raw(X)
                assert p.shape == (2,), p.shape
                loaded += 1
            except Exception as e:
                errs.append(f"{pest}/{branch}: {type(e).__name__}: {e}")
    return {"name": "16 models load + predict via Booster",
            "loaded": loaded, "errors": errs, "passed": loaded == 16 and not errs}


def test_gates_authoritative() -> dict:
    """gate.json must carry the authoritative k/tau (the JSON values, not the
    drifted yaml copy). Spot-checks the 2 pests where they disagree."""
    expect = {"bacterial_blight": {"k": 3, "tau": 0.525},
              "rice_stem_borer_1": {"k": 3, "tau": 0.55}}
    rows, ok = [], True
    for pest in PESTS:
        g = load_gate(pest, PKG / "assets" / "stage1")
        row = {"pest": pest, "method": g["method"], "k": g["k"], "tau": g.get("tau"),
               "tau_no": g.get("tau_no"), "tau_with": g.get("tau_with")}
        if pest in expect:
            good = (g["k"] == expect[pest]["k"]
                    and abs((g.get("tau") or -1) - expect[pest]["tau"]) < 1e-9)
            row["drift_check"] = good
            ok &= good
        rows.append(row)
    return {"name": "gate.json holds authoritative (non-drifted) k/tau",
            "rows": rows, "passed": ok}


def main() -> int:
    tests = [
        test_base_layout_is_fortran(),
        test_history_append_is_c(),
        test_layout_actually_changes_features(),
        test_branch_layouts_all_pests(),
        test_models_load_without_sklearn(),
        test_gates_authoritative(),
    ]
    print("=== Stage-1 portable tests ===")
    for t in tests:
        print(f"  {'PASS' if t['passed'] else 'FAIL'}  {t['name']}")
        if t["name"].startswith("base X"):
            print(f"          strides={t['layout']['strides']} "
                  f"C={t['layout']['c_contiguous']} F={t['layout']['f_contiguous']}")
        if t["name"].startswith("after append"):
            print(f"          strides={t['layout']['strides']} "
                  f"C={t['layout']['c_contiguous']} F={t['layout']['f_contiguous']}")
        if "cleanup" in t["name"]:
            print(f"          max|F-C| = {t['max_abs_diff']:.3e} "
                  f"(nonzero is REQUIRED — proves the layout matters)")
    ok = all(t["passed"] for t in tests)
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'} ({sum(t['passed'] for t in tests)}/{len(tests)})")
    (HERE / "_stage1_layout_report.json").write_text(json.dumps(tests, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
