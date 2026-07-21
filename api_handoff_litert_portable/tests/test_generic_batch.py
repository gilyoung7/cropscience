"""Generic-batch contract: request-field inheritance and input_csv resolution.

These are the behaviours that broke once and must not break again:

  * a top-level daily_weather_path / long_observation_path never reached the
    per-row request, so every row failed with "daily weather CSV not found";
  * an empty CSV cell arrives as float NaN even with dtype=str, and reached
    Path() as a float (TypeError);
  * include_diagnostics from a CSV cell is a STRING, and "false" is truthy —
    it switched diagnostics on;
  * a row-level stage2_variant was carried but never used, because the context
    cache was keyed on pest alone;
  * input_csv was the only path field resolved with a bare Path(), so a relative
    name worked only when the process happened to run from the right directory.

Everything here is a unit test: `run_batch` takes the per-row pipeline as an
argument, so a stub captures the requests it is handed and no model, no asset
and no weather file is touched. Integration coverage that needs the real data
lives in test_generic_batch_integration below, marked `integration` and skipped
unless the data paths are provided.

    pytest -q                       # unit only (default)
    pytest -q -m integration        # needs PEST_DAILY_MASTER / PEST_LONG_DIR
"""

from __future__ import annotations

import csv
import json
import os
import sys
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
sys.path.insert(0, str(PKG))

from infer import batch as batch_mod  # noqa: E402
from infer.batch import (  # noqa: E402
    BatchRequestError,
    _inherit_request_fields,
    resolve_input_csv,
    run_batch,
)
from infer.paths import Paths  # noqa: E402

CLIM = {
    "mu_doy": 147.55,
    "pi_95": {"lower_doy": 138, "upper_doy": 157, "sigma_days": 5.0},
    "variant": "mean_mid",
    "_source_csv": "BPH_climatology_train_stats.csv",
}
POLICY = {"per_pest": {
    "BPH": {"recommended_source": "climatology", "learned_output_status": "main",
            "selected_fixed_offset": 30, "climatology": {"variant": "mean_mid"}},
    "WBPH": {"recommended_source": "climatology", "learned_output_status": "main",
             "selected_fixed_offset": 45, "climatology": {"variant": "mean_mid"}},
}}


# ---------------------------------------------------------------------------
# 1-4. _inherit_request_fields — pure, no I/O
# ---------------------------------------------------------------------------
def _inherit(row: dict, top: dict) -> dict:
    req = {"pest": "BPH", "site_id": "s", "year": 2004,
           "include_diagnostics": bool(top.get("include_diagnostics", False))}
    _inherit_request_fields(req, top, row)
    return req


def test_row_inherits_all_top_level_fields():
    top = {
        "daily_weather_path": "/data/daily.csv",
        "long_observation_path": "/data/long.csv",
        "representative_sites_path": "/data/rep.csv",
        "stage2_variant": "fp32",
        "include_diagnostics": True,
    }
    req = _inherit({}, top)
    for k, v in top.items():
        assert req[k] == v, f"{k} was not inherited"


def test_row_value_wins_over_top_level():
    top = {"daily_weather_path": "/data/A.csv",
           "long_observation_path": "/data/longA.csv"}
    row = {"daily_weather_path": "/data/B.csv"}
    req = _inherit(row, top)
    assert req["daily_weather_path"] == "/data/B.csv"   # row wins
    assert req["long_observation_path"] == "/data/longA.csv"  # not overridden


@pytest.mark.parametrize("blank", ["", "   ", "nan", "NaN", "none", "None"])
def test_blank_string_cell_inherits_top_level(blank):
    req = _inherit({"daily_weather_path": blank}, {"daily_weather_path": "/data/A.csv"})
    assert req["daily_weather_path"] == "/data/A.csv"


def test_nan_float_cell_inherits_and_never_becomes_a_path():
    """pandas gives float('nan') for an empty cell even with dtype=str."""
    req = _inherit({"daily_weather_path": float("nan")},
                   {"daily_weather_path": "/data/A.csv"})
    assert req["daily_weather_path"] == "/data/A.csv"
    assert isinstance(req["daily_weather_path"], str)
    Path(req["daily_weather_path"])  # would raise TypeError on a float


def test_missing_top_level_field_is_simply_absent():
    req = _inherit({}, {})
    assert "daily_weather_path" not in req
    assert "long_observation_path" not in req


@pytest.mark.parametrize("cell,expected", [
    ("true", True), ("True", True), ("TRUE", True), ("1", True), ("yes", True),
    ("false", False), ("False", False), ("0", False), ("no", False),
])
def test_include_diagnostics_string_becomes_bool(cell, expected):
    """'false' is a non-empty string and would be truthy if left uncoerced."""
    req = _inherit({"include_diagnostics": cell}, {"include_diagnostics": False})
    assert req["include_diagnostics"] is expected, f"{cell!r} -> {req['include_diagnostics']!r}"


def test_include_diagnostics_blank_cell_inherits_top_level():
    for top_val in (True, False):
        req = _inherit({"include_diagnostics": ""}, {"include_diagnostics": top_val})
        assert req["include_diagnostics"] is top_val


# ---------------------------------------------------------------------------
# 6. resolve_input_csv — cwd -> input_dir -> pkg_root
# ---------------------------------------------------------------------------
def test_input_csv_absolute_path(tmp_path):
    f = tmp_path / "rows.csv"
    f.write_text("pest,site_id,year\n")
    assert resolve_input_csv(str(f), tmp_path / "in", tmp_path / "pkg") == f


def test_input_csv_relative_to_cwd(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "rows.csv").write_text("pest,site_id,year\n")
    monkeypatch.chdir(cwd)
    got = resolve_input_csv("rows.csv", tmp_path / "in", tmp_path / "pkg")
    assert got == cwd / "rows.csv"


def test_input_csv_relative_to_input_dir(tmp_path, monkeypatch):
    empty_cwd = tmp_path / "elsewhere"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "rows.csv").write_text("pest,site_id,year\n")
    assert resolve_input_csv("rows.csv", in_dir, tmp_path / "pkg") == in_dir / "rows.csv"


def test_input_csv_relative_to_pkg_root(tmp_path, monkeypatch):
    empty_cwd = tmp_path / "elsewhere"
    empty_cwd.mkdir()
    monkeypatch.chdir(empty_cwd)
    pkg = tmp_path / "pkg"
    pkg.mkdir()
    (pkg / "rows.csv").write_text("pest,site_id,year\n")
    assert resolve_input_csv("rows.csv", tmp_path / "in", pkg) == pkg / "rows.csv"


def test_input_csv_cwd_wins_over_input_dir(tmp_path, monkeypatch):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    (cwd / "rows.csv").write_text("cwd\n")
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    (in_dir / "rows.csv").write_text("indir\n")
    monkeypatch.chdir(cwd)
    assert resolve_input_csv("rows.csv", in_dir, tmp_path / "pkg") == cwd / "rows.csv"


def test_input_csv_missing_lists_every_candidate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(BatchRequestError) as e:
        resolve_input_csv("nope.csv", tmp_path / "in", tmp_path / "pkg")
    msg = str(e.value)
    assert "nope.csv" in msg
    assert "Tried:" in msg
    assert str(tmp_path / "in") in msg and str(tmp_path / "pkg") in msg


# ---------------------------------------------------------------------------
# 1/2/5. End-to-end through run_batch with a stubbed per-row pipeline
# ---------------------------------------------------------------------------
class _FakeCtx:
    """Stands in for _PestContext: records which (pest, variant) was built."""

    def __init__(self, paths, pest, policy, variant):
        self.pest = pest
        self.variant = variant
        self.climatology = CLIM
        self.policy_pp = POLICY["per_pest"].get(pest, {})
        self.model = None
        self.weather = None
        self.obs = None
        self.alert_map = None
        self.stage1_notes = {}


def _install_stubs(monkeypatch, seen_ctx: list, seen_req: list):
    monkeypatch.setattr(batch_mod, "load_policy", lambda p: POLICY)
    monkeypatch.setattr(batch_mod, "compute_climatology", lambda *a, **k: CLIM)

    def make_ctx(paths, pest, policy, variant):
        ctx = _FakeCtx(paths, pest, policy, variant)
        seen_ctx.append((pest, variant))
        return ctx

    monkeypatch.setattr(batch_mod, "_PestContext", make_ctx)

    from infer.schemas import build_response

    def fake_run_row(paths, request, variant, ctx):
        seen_req.append({"request": dict(request), "variant": variant,
                         "ctx_variant": getattr(ctx, "variant", None)})
        resp = build_response(request, None, CLIM, POLICY, {}, "no alert (stub)",
                              {"stage1_backend": "stub", "stage2_backend": "stub"})
        return resp, "Stage-1 fired no alert (stub)"

    return fake_run_row


def _write(tmp_path, rows_csv: str, request: dict):
    in_dir = tmp_path / "in"
    in_dir.mkdir(exist_ok=True)
    out_dir = tmp_path / "out"
    out_dir.mkdir(exist_ok=True)
    (in_dir / "rows.csv").write_text(rows_csv, encoding="utf-8")
    (in_dir / "request.json").write_text(json.dumps(request), encoding="utf-8")
    return Paths(pkg_root=PKG, input_dir=in_dir, output_dir=out_dir)


def test_generic_batch_propagates_paths_to_every_row(tmp_path, monkeypatch):
    seen_ctx: list = []
    seen_req: list = []
    run_row = _install_stubs(monkeypatch, seen_ctx, seen_req)
    paths = _write(
        tmp_path,
        "pest,site_id,year\nBPH,11_22,2004\nWBPH,33_44,2011\n",
        {"mode": "batch", "input_csv": "rows.csv",
         "daily_weather_path": "/data/daily.csv",
         "long_observation_path": "/data/long.csv",
         "representative_sites_path": "/data/rep.csv",
         "include_diagnostics": True},
    )
    rc = run_batch(paths, json.loads((paths.input_dir / "request.json").read_text()),
                   run_row, "fp16")
    assert rc == 0
    assert len(seen_req) == 2
    for got in seen_req:
        r = got["request"]
        assert r["daily_weather_path"] == "/data/daily.csv"
        assert r["long_observation_path"] == "/data/long.csv"
        assert r["representative_sites_path"] == "/data/rep.csv"
        assert r["include_diagnostics"] is True


def test_generic_batch_row_path_overrides_top_level(tmp_path, monkeypatch):
    seen_ctx: list = []
    seen_req: list = []
    run_row = _install_stubs(monkeypatch, seen_ctx, seen_req)
    paths = _write(
        tmp_path,
        "pest,site_id,year,daily_weather_path\n"
        "BPH,11_22,2004,\n"                       # blank -> inherit A
        "WBPH,33_44,2011,/data/B.csv\n",          # row -> B
        {"mode": "batch", "input_csv": "rows.csv",
         "daily_weather_path": "/data/A.csv"},
    )
    rc = run_batch(paths, json.loads((paths.input_dir / "request.json").read_text()),
                   run_row, "fp16")
    assert rc == 0
    assert seen_req[0]["request"]["daily_weather_path"] == "/data/A.csv"
    assert seen_req[1]["request"]["daily_weather_path"] == "/data/B.csv"


def test_generic_batch_row_variant_selects_its_own_context(tmp_path, monkeypatch):
    """Same pest, two variants -> two contexts, and the row runs with its own."""
    seen_ctx: list = []
    seen_req: list = []
    run_row = _install_stubs(monkeypatch, seen_ctx, seen_req)
    paths = _write(
        tmp_path,
        "pest,site_id,year,stage2_variant\n"
        "BPH,11_22,2004,fp16\n"
        "BPH,33_44,2004,fp32\n"
        "BPH,55_66,2004,\n",                      # blank -> batch default fp16
        {"mode": "batch", "input_csv": "rows.csv"},
    )
    rc = run_batch(paths, json.loads((paths.input_dir / "request.json").read_text()),
                   run_row, "fp16")
    assert rc == 0
    assert seen_ctx == [("BPH", "fp16"), ("BPH", "fp32")], seen_ctx
    assert [g["variant"] for g in seen_req] == ["fp16", "fp32", "fp16"]
    assert [g["ctx_variant"] for g in seen_req] == ["fp16", "fp32", "fp16"]


def test_generic_batch_rejects_unknown_variant_per_row(tmp_path, monkeypatch):
    seen_ctx: list = []
    seen_req: list = []
    run_row = _install_stubs(monkeypatch, seen_ctx, seen_req)
    paths = _write(
        tmp_path,
        "pest,site_id,year,stage2_variant\nBPH,11_22,2004,int8\n",
        {"mode": "batch", "input_csv": "rows.csv"},
    )
    rc = run_batch(paths, json.loads((paths.input_dir / "request.json").read_text()),
                   run_row, "fp16")
    assert rc == 0                      # a bad row never aborts the batch
    rows = list(csv.DictReader(
        (paths.output_dir / "predictions.csv").open(encoding="utf-8-sig")))
    assert rows[0]["status"] == "error"
    assert "int8" in rows[0]["error_reason"]


def test_generic_batch_preserves_row_order_and_row_count(tmp_path, monkeypatch):
    seen_ctx: list = []
    seen_req: list = []
    run_row = _install_stubs(monkeypatch, seen_ctx, seen_req)
    paths = _write(
        tmp_path,
        "pest,site_id,year\nWBPH,99_99,2011\nBPH,11_22,2004\nWBPH,33_44,2011\n",
        {"mode": "batch", "input_csv": "rows.csv"},
    )
    run_batch(paths, json.loads((paths.input_dir / "request.json").read_text()),
              run_row, "fp16")
    rows = list(csv.DictReader(
        (paths.output_dir / "predictions.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 3
    assert [r["site_id"] for r in rows] == ["99_99", "11_22", "33_44"]
    assert [r["row_index"] for r in rows] == ["0", "1", "2"]


def test_generic_batch_fallback_rows_are_kept(tmp_path, monkeypatch):
    """A row that produced no learned value still appears, filled from climatology."""
    seen_ctx: list = []
    seen_req: list = []
    run_row = _install_stubs(monkeypatch, seen_ctx, seen_req)
    paths = _write(tmp_path, "pest,site_id,year\nBPH,11_22,2004\n",
                   {"mode": "batch", "input_csv": "rows.csv"})
    run_batch(paths, json.loads((paths.input_dir / "request.json").read_text()),
              run_row, "fp16")
    rows = list(csv.DictReader(
        (paths.output_dir / "predictions.csv").open(encoding="utf-8-sig")))
    assert len(rows) == 1
    assert rows[0]["status"] == "fallback"
    assert rows[0]["learned_mu_doy"] == ""
    assert rows[0]["final_mu_doy"] == str(CLIM["mu_doy"])


def test_generic_batch_relative_csv_found_via_input_dir(tmp_path, monkeypatch):
    """CWD is elsewhere; the CSV lives in --input-dir. Used to fail."""
    seen_ctx: list = []
    seen_req: list = []
    run_row = _install_stubs(monkeypatch, seen_ctx, seen_req)
    paths = _write(tmp_path, "pest,site_id,year\nBPH,11_22,2004\n",
                   {"mode": "batch", "input_csv": "rows.csv"})
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    rc = run_batch(paths, json.loads((paths.input_dir / "request.json").read_text()),
                   run_row, "fp16")
    assert rc == 0
    assert len(seen_req) == 1


# ---------------------------------------------------------------------------
# 8. Output-collision protection (no model needed)
# ---------------------------------------------------------------------------
from infer.inputs import (  # noqa: E402
    CLAIM_FILE, OutputCollisionError, existing_outputs, plan_output_dir,
    release_claim,
)


def test_plan_output_dir_refuses_existing_results(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "predictions.csv").write_text("x")
    with pytest.raises(OutputCollisionError):
        plan_output_dir(out, {}, overwrite=False, unique_subdir=False)


def test_plan_output_dir_overwrite_allows_and_claims(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / "predictions.csv").write_text("x")
    d, note = plan_output_dir(out, {}, overwrite=True, unique_subdir=False)
    assert d == out and "overwrite" in note
    assert (out / CLAIM_FILE).is_file()
    release_claim(out)
    assert not (out / CLAIM_FILE).exists()


def test_second_run_refused_while_claim_is_held(tmp_path):
    out = tmp_path / "out"
    plan_output_dir(out, {}, overwrite=False, unique_subdir=False)
    try:
        with pytest.raises(OutputCollisionError) as e:
            plan_output_dir(out, {}, overwrite=False, unique_subdir=False)
        assert "claimed by another run" in str(e.value)
    finally:
        release_claim(out)


def test_stale_claim_broken_by_overwrite(tmp_path):
    out = tmp_path / "out"
    out.mkdir()
    (out / CLAIM_FILE).write_text("pid=999999\nclaimed_utc=2020-01-01T00:00:00+00:00\n")
    d, _ = plan_output_dir(out, {}, overwrite=True, unique_subdir=False)
    assert d == out
    release_claim(out)
    assert not (out / CLAIM_FILE).exists()


def test_unique_subdir_never_collides(tmp_path):
    out = tmp_path / "out"
    dirs = []
    for _ in range(3):
        d, _ = plan_output_dir(out, {"pest": "BPH", "year": 2004},
                               overwrite=False, unique_subdir=True)
        dirs.append(d)
    assert len({str(d) for d in dirs}) == 3
    for d in dirs:
        assert d.is_dir()
        release_claim(d)


def test_failed_batch_does_not_overwrite_existing_results(tmp_path, monkeypatch):
    """fail_batch must leave a previous run's output untouched."""
    monkeypatch.setattr(batch_mod, "load_policy", lambda p: POLICY)
    out = tmp_path / "out"
    out.mkdir()
    for name in ("response.json", "predictions.csv", "run_log.txt"):
        (out / name).write_text("PREVIOUS")
    before = {n: (out / n).read_text() for n in existing_outputs(out)}

    rc = batch_mod.fail_batch(out, {"mode": "batch"}, "simulated failure", 0.0)
    assert rc == 2
    after = {n: (out / n).read_text() for n in existing_outputs(out)}
    assert after == before, "a failed run overwrote previous results"


def test_claim_file_is_not_left_behind_by_run_batch(tmp_path, monkeypatch):
    seen_ctx: list = []
    seen_req: list = []
    run_row = _install_stubs(monkeypatch, seen_ctx, seen_req)
    paths = _write(tmp_path, "pest,site_id,year\nBPH,11_22,2004\n",
                   {"mode": "batch", "input_csv": "rows.csv"})
    run_batch(paths, json.loads((paths.input_dir / "request.json").read_text()),
              run_row, "fp16")
    assert not (paths.output_dir / CLAIM_FILE).exists()


# ---------------------------------------------------------------------------
# 7. Representative-site cohort batch — integration (needs real data + assets)
# ---------------------------------------------------------------------------
DAILY_MASTER = os.environ.get("PEST_DAILY_MASTER")
LONG_DIR = os.environ.get("PEST_LONG_DIR")
REP_CSV = os.environ.get("PEST_REP_CSV")

_have_data = all([DAILY_MASTER, LONG_DIR, REP_CSV]) and all(
    Path(p).exists() for p in (DAILY_MASTER or "", LONG_DIR or "", REP_CSV or ""))

integration = pytest.mark.skipif(
    not _have_data,
    reason="set PEST_DAILY_MASTER / PEST_LONG_DIR / PEST_REP_CSV to run",
)


@pytest.mark.integration
@integration
def test_cohort_batch_one_row_per_site(tmp_path):
    """Representative batch still emits one row per site, fallbacks included."""
    import subprocess

    pest, year, max_sites = "sheath_blight", 2004, 12
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    (in_dir / "request.json").write_text(json.dumps({
        "mode": "batch", "pest": pest, "year": year,
        "daily_weather_path": DAILY_MASTER,
        "long_observation_path": str(Path(LONG_DIR) / f"RICE_LONG_{pest}.csv"),
        "representative_sites_path": REP_CSV,
        "max_sites": max_sites,
    }), encoding="utf-8")

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, str(PKG / "run_predict.py"),
                        "--input-dir", str(in_dir), "--output-dir", str(out_dir)],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-2000:]

    rows = list(csv.DictReader(
        (out_dir / "predictions.csv").open(encoding="utf-8-sig")))
    assert len(rows) == max_sites, f"expected {max_sites} rows, got {len(rows)}"
    assert len({r_["site_id"] for r_ in rows}) == max_sites, "site_id must be unique"

    # Every row is answered, fallback or not — none are dropped.
    for r_ in rows:
        assert r_["status"] in ("success", "fallback", "error")
        assert r_["final_mu_doy"] not in ("", None), \
            f"{r_['site_id']} has no final_mu_doy"
        if r_["status"] == "fallback":
            assert r_["learned_mu_doy"] == ""
            assert r_["final_source"].startswith("climatology")

    # column contract: the 16 single-mode columns + status/error_reason,
    # and NO row_index (that prefix is generic-mode only)
    from infer.schemas import FLAT_COLS
    assert list(rows[0].keys()) == list(FLAT_COLS) + ["status", "error_reason"]

    assert not (out_dir / CLAIM_FILE).exists()


@pytest.mark.integration
@integration
def test_cohort_batch_multi_year_row_count(tmp_path):
    """start_year/end_year gives site x year rows."""
    import subprocess

    pest, max_sites = "sheath_blight", 6
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    (in_dir / "request.json").write_text(json.dumps({
        "mode": "batch", "pest": pest, "start_year": 2023, "end_year": 2024,
        "daily_weather_path": DAILY_MASTER,
        "long_observation_path": str(Path(LONG_DIR) / f"RICE_LONG_{pest}.csv"),
        "representative_sites_path": REP_CSV,
        "max_sites": max_sites,
    }), encoding="utf-8")

    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run([sys.executable, str(PKG / "run_predict.py"),
                        "--input-dir", str(in_dir), "--output-dir", str(out_dir)],
                       capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert r.returncode == 0, r.stderr[-2000:]

    rows = list(csv.DictReader(
        (out_dir / "predictions.csv").open(encoding="utf-8-sig")))
    assert len(rows) == max_sites * 2
    years = sorted({r_["year"] for r_ in rows})
    assert years == ["2023", "2024"]


@pytest.mark.integration
@integration
def test_output_collision_refused_on_second_run(tmp_path):
    import subprocess

    pest = "sheath_blight"
    in_dir = tmp_path / "in"
    in_dir.mkdir()
    out_dir = tmp_path / "out"
    (in_dir / "request.json").write_text(json.dumps({
        "mode": "batch", "pest": pest, "year": 2004,
        "daily_weather_path": DAILY_MASTER,
        "long_observation_path": str(Path(LONG_DIR) / f"RICE_LONG_{pest}.csv"),
        "representative_sites_path": REP_CSV, "max_sites": 3,
    }), encoding="utf-8")
    env = dict(os.environ, CUDA_VISIBLE_DEVICES="")
    cmd = [sys.executable, str(PKG / "run_predict.py"),
           "--input-dir", str(in_dir), "--output-dir", str(out_dir)]

    first = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert first.returncode == 0, first.stderr[-2000:]
    digest = (out_dir / "predictions.csv").read_bytes()

    second = subprocess.run(cmd, capture_output=True, text=True, env=env, cwd=str(tmp_path))
    assert second.returncode == 1
    assert "already contains results" in second.stderr
    assert (out_dir / "predictions.csv").read_bytes() == digest, \
        "a refused run modified the previous output"

    third = subprocess.run(cmd + ["--overwrite"], capture_output=True, text=True,
                           env=env, cwd=str(tmp_path))
    assert third.returncode == 0, third.stderr[-2000:]
    assert not (out_dir / CLAIM_FILE).exists()
