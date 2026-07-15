"""B. Preprocessing parity: standalone runtime vs the API's build_real_input.

Runs both on the SAME real inputs for all 8 pests and compares the built tensor.
This is the test that matters most: the standalone preprocessing is a
reimplementation, so anything less than bit-exact needs an explanation.

Needs torch + the API package (it calls the original). BUILD-side test.

    ../../.venv-tflite/bin/python tests/test_preprocessing_parity.py \
        --daily-master "$DAILY_MASTER" --long-dir "$LONG_DIR"
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO_ROOT = PKG.parents[1]
API = REPO_ROOT / "api_handoff_transformer"
DIST = PKG / "dist" / "stage2_litert"

sys.path.insert(0, str(PKG))
sys.path.insert(0, str(REPO_ROOT / "tflite_conversion" / "stage2"))
sys.path.insert(0, str(API))

from runtime.preprocessing import build_input  # noqa: E402
from runtime.schema import DispatchRequest, load_metadata  # noqa: E402


def api_built(pest: str, site: str, year: int, daily_csv: Path, obs_csv: Path,
              cache_dir: Path):
    """The original build_real_input, untouched."""
    from checkpoint import load_pest, selected_offset
    from infer.preprocess import build_real_input

    loaded, _ = load_pest(pest)
    return loaded, build_real_input(
        feature_names=loaded.feature_names,
        nowcast_window=loaded.nowcast_window,
        season_T=loaded.T,
        doy_start=loaded.doy_start,
        norm_mean=loaded.norm_mean,
        norm_std=loaded.norm_std,
        dispatch_csv=API / "configs" / "dispatch" / f"{pest}_dispatch.csv",
        site_id=site,
        year=year,
        alert_tstar_doy=None,
        selected_offset=selected_offset(pest),
        alert_tstar_feat_idx=loaded.alert_tstar_feat_idx,
        dispatch_row_override=None,
        master_daily_csv=daily_csv,
        obs_csv=obs_csv,
        cache_dir=cache_dir,
    )


def standalone_request(pest: str, site: str, year: int, obs_csv: Path,
                       built_api) -> DispatchRequest:
    """Build the standalone request from the SAME sources the API used.

    dispatch features come from the dispatch row the API resolved (so we compare
    preprocessing, not dispatch lookup); site coords + phenology come from the
    same LONG CSV the API read, because the standalone runtime takes them as
    explicit inputs instead of reading that file itself.
    """
    md = load_metadata(DIST / "models", pest)
    row = dict(built_api.dispatch_row_used)
    feats = {k: row.get(k) for k in
             ("alert_tstar", "with_history", "dispatch_branch", "A_score_at_alert",
              "D_score_at_alert", "score_margin", "dispatch_score_at_alert",
              "dispatch_tau_used", "score_over_tau_margin", "recent_14d_mean_score",
              "recent_28d_mean_score", "score_above_tau_streak",
              "score_rolling_slope_14d", "p_mean_so_far_at_alert")}

    site_d: dict = {}
    pheno: list[dict] = []
    if md.requires_site_coords or md.requires_phenology:
        obs = pd.read_csv(obs_csv, encoding="utf-8-sig")
        obs = obs.rename(columns=lambda c: c.strip() if isinstance(c, str) else c)
        obs["site_id"] = obs["site_id"].astype(str)
        obs["year"] = pd.to_numeric(obs["year"], errors="coerce")
        if md.requires_site_coords:
            # mirror _site_latlon: first non-null for the site, else global mean
            lat = pd.to_numeric(obs["좌표-위도"], errors="coerce")
            lon = pd.to_numeric(obs["좌표-경도"], errors="coerce")
            sl = lat[obs["site_id"] == site].dropna()
            so = lon[obs["site_id"] == site].dropna()
            site_d = {
                "lat": float(sl.iloc[0]) if not sl.empty else float(lat.mean()),
                "lon": float(so.iloc[0]) if not so.empty else float(lon.mean()),
            }
        if md.requires_phenology:
            cols = [c for c in ("days_since_growing_start", "days_until_growing_end",
                                "is_growing") if c in md.base_channels]
            sub = obs[(obs["site_id"] == site) & (obs["year"] == year)]
            pheno = sub[["obs_doy"] + cols].to_dict("records")

    return DispatchRequest(
        pest=pest, alert_tstar=int(built_api.alert_tstar_doy),
        dispatch_features=feats, site=site_d, phenology=pheno, year=year,
    )


def compare(pest: str, site: str, year: int, daily_csv: Path, obs_csv: Path,
            cache_dir: Path) -> dict:
    loaded, ref = api_built(pest, site, year, daily_csv, obs_csv, cache_dir)
    md = load_metadata(DIST / "models", pest)
    req = standalone_request(pest, site, year, obs_csv, ref)

    with np.load(md.normalization_path) as z:
        mean, std = z["norm_mean"], z["norm_std"]

    # Feed the standalone runtime the SAME rows the API used. The API sliced the
    # site out of the master itself and cached it at daily_site_<site>.csv; the
    # standalone runtime is per-site by design (it rejects a multi-site frame),
    # so reuse that exact slice rather than re-deriving it — this keeps the test
    # about preprocessing, not about site selection.
    site_slice = cache_dir / f"daily_site_{site}.csv"
    if not site_slice.is_file():
        raise FileNotFoundError(
            f"expected the API's per-site cache at {site_slice}; "
            f"api_built() should have created it"
        )
    daily = pd.read_csv(site_slice, encoding="utf-8-sig")
    got = build_input(md, daily, req, mean, std)

    a = ref.X.detach().cpu().numpy()
    b = got.X
    d = np.abs(a - b)
    per_channel = d.reshape(-1, d.shape[-1]).max(axis=0)
    worst = int(np.argmax(per_channel))
    return {
        "pest": pest, "site": site, "year": year,
        "api_shape": list(a.shape), "standalone_shape": list(b.shape),
        "shape_match": a.shape == b.shape,
        "api_dtype": str(a.dtype), "standalone_dtype": str(b.dtype),
        "dtype_match": a.dtype == b.dtype,
        "max_abs": float(d.max()), "mean_abs": float(d.mean()),
        "bit_exact": bool(np.array_equal(a, b)),
        "worst_channel_index": worst,
        "worst_channel_name": md.feature_names[worst],
        "worst_channel_max_abs": float(per_channel[worst]),
        "tstar_match": int(ref.tstar_season_index) == int(got.tstar_season_index),
        "alert_match": int(ref.alert_tstar_doy) == int(got.alert_tstar_doy),
        "passed": bool(np.array_equal(a, b))
        and a.shape == b.shape and a.dtype == b.dtype
        and int(ref.tstar_season_index) == int(got.tstar_season_index),
    }


def main() -> int:
    from pest_configs import LONG_FILENAME, PESTS, SMOKE

    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-master", required=True, type=Path)
    ap.add_argument("--long-dir", required=True, type=Path)
    ap.add_argument("--pests", nargs="*", default=list(PESTS))
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(tempfile.gettempdir()) / "standalone_parity_cache")
    args = ap.parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    rows, failures, skipped = [], [], []
    for pest in args.pests:
        rec = SMOKE[pest]
        obs = args.long_dir / LONG_FILENAME[pest]
        if not obs.is_file():
            skipped.append({"pest": pest, "reason": f"LONG CSV not found: {obs}"})
            continue
        try:
            rows.append(compare(pest, rec.site, rec.year, args.daily_master, obs,
                                args.cache_dir))
        except Exception as e:
            failures.append({"pest": pest, "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()})
            print(f"  FAIL {pest}: {type(e).__name__}: {str(e)[:140]}")

    print(f"\n{'pest':<19}{'shape':>16}{'dtype':>9}{'max_abs':>11}{'mean_abs':>11}"
          f"{'bit_exact':>11}{'res':>6}")
    for r in rows:
        print(f"{r['pest']:<19}{str(r['standalone_shape'][2:]):>16}"
              f"{r['standalone_dtype']:>9}{r['max_abs']:>11.3e}{r['mean_abs']:>11.3e}"
              f"{str(r['bit_exact']):>11}{'PASS' if r['passed'] else 'FAIL':>6}")
    for s in skipped:
        print(f"{s['pest']:<19}  SKIPPED — {s['reason']}")

    ok = bool(rows) and all(r["passed"] for r in rows) and not failures
    n_exact = sum(r["bit_exact"] for r in rows)
    print(f"\nbit-exact: {n_exact}/{len(rows)}")
    print(f"B. preprocessing parity: {'PASS' if ok else 'FAIL'}")
    (DIST.parent / "test_preprocessing_parity.json").write_text(
        json.dumps({"rows": rows, "failures": failures, "skipped": skipped}, indent=2)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
