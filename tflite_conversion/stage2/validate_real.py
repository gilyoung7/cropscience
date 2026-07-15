"""Real-data validation for all 8 pests: PyTorch vs FP32/FP16 TFLite.

For each pest, drives the REAL preprocessing chain
(`api_handoff_transformer/infer/preprocess.build_real_input`, imported unmodified)
over real daily weather + real LONG observations, then compares mu.

The (site, year) and the expected values come from
`api_handoff_transformer/README.md` §10 "Smoke test record" — see
pest_configs.SMOKE. No feature tensor is hand-crafted.

Why this does not call run_predict.py
-------------------------------------
`run_predict.py` runs the live Stage-1 XGBoost gate first. The shipped Stage-1
checkpoints store the booster as a ~1 MB *legacy binary* blob under
`_Booster.handle`; xgboost >= 2 rejects that format by SIGSEGV (exit 139) inside
torch.load rather than raising, so it cannot be caught in-process. Reproduced on
xgboost 3.1.2 and 2.1.4.

Stage-1 is not the conversion target, and Stage-2 never imports xgboost.
Everything Stage-1 would have produced — the alert DOY and the 14 dispatch
confidence features — is already present as real reference values in the shipped
`configs/dispatch/<pest>_dispatch.csv`, which README §9 records as reproduced by
the live gate at ~100% for BPH / ~99% for the others. build_real_input reads that
table itself when alert_tstar_doy=None.

Usage (inside .venv-tflite, from this directory):
    python validate_real.py \
        --daily-master "/Users/doyoung-gil/연구실/d/1997_2024_RICE_union_all_sites_with_GDD10_since_gs.csv" \
        --long-dir     "/Users/doyoung-gil/Downloads/LONG_by_pest"
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch

from checkpoint import PKG_ROOT, original_mu, selected_offset
from inference_model import load_and_wrap
from pest_configs import LONG_FILENAME, PESTS, SMOKE
from validate_all import ATOL, VARIANTS, _interp, tflite_mu

sys.path.insert(0, str(PKG_ROOT))
from infer.preprocess import build_real_input  # noqa: E402

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def ensure_daily_slices(master: Path, sites: set[str], out_dir: Path) -> dict[str, Path]:
    """Slice the needed sites out of the 1.6 GB master in ONE chunked pass.

    build_real_input can take the master directly, but it would rescan the whole
    file per site. One pass writing per-site CSVs is much cheaper and produces
    identical input (same rows, same header, same dtypes).
    """
    import pandas as pd

    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {s: out_dir / f"daily_{s}.csv" for s in sites}
    missing = [s for s, p in paths.items() if not p.is_file()]
    if not missing:
        return paths

    print(f"  slicing {len(missing)} site(s) from master (one pass, ~1.6 GB)...")
    header = pd.read_csv(master, nrows=0, encoding="utf-8-sig")
    site_col = header.columns[0]
    first = {s: True for s in missing}
    for chunk in pd.read_csv(
        master, encoding="utf-8-sig", dtype={site_col: str}, chunksize=500_000
    ):
        sub = chunk[chunk[site_col].astype(str).isin(missing)]
        if sub.empty:
            continue
        for s, g in sub.groupby(site_col):
            if s not in paths:
                continue
            g.to_csv(paths[s], mode="w" if first[s] else "a",
                     header=first[s], index=False, encoding="utf-8-sig")
            first[s] = False
    for s in missing:
        if first[s]:
            raise RuntimeError(f"site {s} not found in master {master}")
    return paths


def validate_pest(pest: str, daily_csv: Path, obs_csv: Path, cache_dir: Path) -> dict:
    loaded, model, cfg = load_and_wrap(pest)
    rec = SMOKE[pest]
    dispatch_csv = PKG_ROOT / "configs" / "dispatch" / f"{pest}_dispatch.csv"
    if not dispatch_csv.is_file():
        raise FileNotFoundError(f"dispatch csv missing: {dispatch_csv}")

    built = build_real_input(
        feature_names=loaded.feature_names,
        nowcast_window=loaded.nowcast_window,
        season_T=loaded.T,
        doy_start=loaded.doy_start,
        norm_mean=loaded.norm_mean,
        norm_std=loaded.norm_std,
        dispatch_csv=dispatch_csv,
        site_id=rec.site,
        year=rec.year,
        alert_tstar_doy=None,  # taken from the shipped reference dispatch row
        selected_offset=selected_offset(pest),
        alert_tstar_feat_idx=loaded.alert_tstar_feat_idx,
        dispatch_row_override=None,
        master_daily_csv=daily_csv,
        obs_csv=obs_csv,
        cache_dir=cache_dir,
    )
    X, tstar, vm = built.X, built.tstar, built.valid_mask
    o = original_mu(loaded, X, tstar, vm)
    with torch.no_grad():
        w = model(X, tstar, vm)

    to_doy = lambda m: float(np.asarray(m).flatten()[0]) + loaded.doy_start - 1
    out: dict = {
        "pest": pest, "site": rec.site, "year": rec.year,
        "alert_expected": rec.alert_doy,
        "alert_used": int(built.alert_tstar_doy),
        "alert_ok": int(built.alert_tstar_doy) == rec.alert_doy,
        "tstar_season_index": int(built.tstar_season_index),
        "base_channels_status": str(built.base_channels_status),
        "zero_placeholder_used": bool(built.zero_placeholder_used),
        "X_shape": list(X.shape),
        "mu_original": float(o.flatten()[0]),
        "mu_doy_original": to_doy(o),
        "mu_doy_wrapper": to_doy(w),
        "wrapper_max_abs_days": float((o - w).abs().max()),
        "readme_mu_doy": rec.mu_doy,
        "readme_delta_days": abs(to_doy(o) - rec.mu_doy),
    }
    for v in VARIANTS:
        p = ARTIFACTS / pest / f"{pest}_stage2_{v}.tflite"
        if not p.is_file():
            out[f"{v}_mu_doy"] = None
            out[f"{v}_max_abs_days"] = None
            out[f"{v}_pass"] = False
            continue
        t = torch.from_numpy(np.asarray(tflite_mu(_interp(p), X, tstar, vm)).reshape(o.shape))
        out[f"{v}_mu_doy"] = to_doy(t)
        out[f"{v}_max_abs_days"] = float((o - t).abs().max())
        out[f"{v}_pass"] = out[f"{v}_max_abs_days"] <= ATOL[v]

    # README quotes mu_doy to 2 dp, so allow half-a-unit-in-the-last-place.
    out["readme_match"] = out["readme_delta_days"] <= 0.005
    out["passed"] = (
        out["alert_ok"] and out["readme_match"]
        and out["wrapper_max_abs_days"] == 0.0
        and not out["zero_placeholder_used"]
        and all(out[f"{v}_pass"] for v in VARIANTS)
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--daily-master", required=True, type=Path)
    ap.add_argument("--long-dir", required=True, type=Path)
    ap.add_argument("--pests", nargs="*", default=list(PESTS))
    ap.add_argument("--cache-dir", type=Path,
                    default=Path(tempfile.gettempdir()) / "stage2_real_cache")
    args = ap.parse_args()

    if not args.daily_master.is_file():
        raise SystemExit(f"missing daily master: {args.daily_master}")
    args.cache_dir.mkdir(parents=True, exist_ok=True)
    slice_dir = args.cache_dir / "daily_slices"

    # Report missing per-pest inputs instead of failing the whole run.
    runnable, skipped = [], []
    for p in args.pests:
        obs = args.long_dir / LONG_FILENAME[p]
        if obs.is_file():
            runnable.append((p, obs))
        else:
            skipped.append({
                "pest": p, "reason": "LONG observation CSV not found locally",
                "needed_file": LONG_FILENAME[p],
                "expected_local_path": str(obs),
                "server_path": f"/home/gpu4080/ygdata/rice/LONG_by_pest/{LONG_FILENAME[p]}",
                "copy_command": (
                    f'scp gpu4080:/home/gpu4080/ygdata/rice/LONG_by_pest/'
                    f'"{LONG_FILENAME[p]}" "{args.long_dir}/"'
                ),
            })

    slices = ensure_daily_slices(
        args.daily_master, {SMOKE[p].site for p, _ in runnable}, slice_dir
    )

    rows, failures = [], []
    for pest, obs in runnable:
        try:
            rows.append(validate_pest(pest, slices[SMOKE[pest].site], obs, args.cache_dir))
        except Exception as e:
            failures.append({"pest": pest, "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()})
            print(f"  FAIL {pest}: {type(e).__name__}: {str(e)[:140]}")

    print(f"\n{'pest':<19}{'site':>13}{'yr':>6}{'alert':>7}{'ok':>4}"
          f"{'mu_doy_torch':>14}{'readme':>9}{'d':>8}{'fp32_d':>10}{'fp16_d':>10}{'res':>6}")
    for r in rows:
        print(f"{r['pest']:<19}{r['site']:>13}{r['year']:>6}{r['alert_used']:>7}"
              f"{'Y' if r['alert_ok'] else 'N':>4}{r['mu_doy_original']:>14.4f}"
              f"{r['readme_mu_doy']:>9.2f}{r['readme_delta_days']:>8.4f}"
              f"{r['fp32_max_abs_days']:>10.2e}{r['fp16_max_abs_days']:>10.2e}"
              f"{'PASS' if r['passed'] else 'FAIL':>6}")
    for s in skipped:
        print(f"{s['pest']:<19}  SKIPPED — {s['reason']}")

    ok = all(r["passed"] for r in rows) and not failures
    print(f"\nvalidated {len(rows)}/{len(args.pests)} pests on real data; "
          f"{len(skipped)} skipped, {len(failures)} failed")
    print(f"RESULT: {'PASS' if ok else 'FAIL'}")

    (ARTIFACTS / "validate_real.json").write_text(
        json.dumps({"rows": rows, "skipped": skipped, "failures": failures}, indent=2)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
