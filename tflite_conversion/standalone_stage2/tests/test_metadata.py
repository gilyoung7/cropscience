"""A. Metadata validation for all 8 pests.

Checks each built metadata.json against the source checkpoint and against the
shipped files. Run from the standalone_stage2 directory:

    ../../.venv-tflite/bin/python tests/test_metadata.py

Needs torch (it re-reads the checkpoints to cross-check) — this is a BUILD-side
test. The torch-free runtime proof is tests/test_no_torch_dependency.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PKG = HERE.parent
REPO_ROOT = PKG.parents[1]
DIST = PKG / "dist" / "stage2_litert"

sys.path.insert(0, str(PKG))
sys.path.insert(0, str(REPO_ROOT / "tflite_conversion" / "stage2"))

from runtime.schema import load_metadata, sha256_file  # noqa: E402


def check_pest(pest: str) -> dict:
    from checkpoint import load_pest  # build-side loader (validates ckpt itself)

    md = load_metadata(DIST / "models", pest)   # runs schema self-check
    loaded, cfg = load_pest(pest)

    errs: list[str] = []

    def eq(name, got, want):
        if got != want:
            errs.append(f"{name}: metadata={got!r} ckpt={want!r}")

    # --- metadata vs checkpoint config -------------------------------
    eq("d_in", md.d_in, loaded.d_in)
    eq("T", md.T, loaded.T)
    eq("doy_start", md.doy_start, loaded.doy_start)
    eq("doy_end", md.doy_end, loaded.doy_end)
    eq("alert_idx", md.alert_idx, loaded.alert_tstar_feat_idx)
    eq("sigma", md.sigma, loaded.sigma)
    eq("nowcast_window", md.nowcast_window, loaded.nowcast_window)
    eq("feature_names", md.feature_names, list(loaded.feature_names))
    eq("mu_mode", md.mu_mode, str(loaded.model.mu_mode))
    eq("lead_min", md.raw["lead_min"], loaded.lead_min)
    eq("lead_max", md.raw["lead_max"], loaded.lead_max)
    eq("d_model", md.raw["d_model"], loaded.model.in_proj.out_features)
    eq("n_head", md.raw["n_head"], loaded.model.time_encoder.layers[0].self_attn.num_heads)
    eq("n_layers", md.raw["n_layers"], len(loaded.model.time_encoder.layers))

    # --- feature_names length == d_in --------------------------------
    if len(md.feature_names) != md.d_in:
        errs.append(f"len(feature_names)={len(md.feature_names)} != d_in={md.d_in}")

    # --- norm arrays: shape + exact values vs checkpoint --------------
    with np.load(md.normalization_path) as z:
        mean, std = z["norm_mean"], z["norm_std"]
    ck_mean = loaded.norm_mean.detach().cpu().numpy().astype(np.float32)
    ck_std = loaded.norm_std.detach().cpu().numpy().astype(np.float32)
    if mean.shape != (md.d_in,) or std.shape != (md.d_in,):
        errs.append(f"norm shapes {mean.shape}/{std.shape} != (d_in={md.d_in},)")
    if not np.array_equal(mean, ck_mean):
        errs.append(f"norm_mean differs from ckpt (max|d|={np.abs(mean-ck_mean).max()})")
    if not np.array_equal(std, ck_std):
        errs.append(f"norm_std differs from ckpt (max|d|={np.abs(std-ck_std).max()})")
    if mean.dtype != np.float32 or std.dtype != np.float32:
        errs.append(f"norm dtype {mean.dtype}/{std.dtype} != float32")

    # --- dispatch bypass mask matches the actual arrays ---------------
    derived = [i for i in range(md.d_in) if mean[i] == 0.0 and std[i] == 1.0]
    if md.norm_bypass_indices != derived:
        errs.append(
            f"norm_bypass_channel_indices {md.norm_bypass_indices} != derived {derived}"
        )
    n_base = len(md.base_channels)
    dispatch_idx = list(range(2 * n_base, md.d_in))
    if not set(dispatch_idx).issubset(set(md.norm_bypass_indices)):
        errs.append(f"dispatch channels {dispatch_idx} not all in bypass set")

    # --- sha256 of shipped files -------------------------------------
    try:
        md.verify_files()
    except Exception as e:
        errs.append(f"file sha256 verify failed: {e}")

    # --- source checkpoint sha256 ------------------------------------
    ck_sha = sha256_file(loaded.ckpt_path)
    if md.raw["source_checkpoint"]["sha256"] != ck_sha:
        errs.append("source_checkpoint.sha256 does not match the checkpoint on disk")

    # --- TFLite input shape matches metadata --------------------------
    from ai_edge_litert.interpreter import Interpreter

    for v in ("fp16", "fp32"):
        it = Interpreter(model_path=str(md.model_path(v)))
        it.allocate_tensors()
        shapes = {tuple(int(x) for x in d["shape"]) for d in it.get_input_details()}
        want = (1, 1, md.T, md.d_in)
        if want not in shapes:
            errs.append(f"{v}: no input with shape {want}; got {sorted(shapes)}")
        out = it.get_output_details()
        if len(out) != 1 or tuple(int(x) for x in out[0]["shape"]) != (1, 1):
            errs.append(f"{v}: unexpected output details {out}")

    return {"pest": pest, "errors": errs, "passed": not errs}


def main() -> int:
    from pest_configs import PESTS

    rows = [check_pest(p) for p in PESTS]
    print(f"{'pest':<20}{'result':>8}  errors")
    for r in rows:
        print(f"{r['pest']:<20}{'PASS' if r['passed'] else 'FAIL':>8}  "
              f"{'-' if r['passed'] else r['errors'][0][:60]}")
        for e in r["errors"][1:]:
            print(f"{'':28}{e[:60]}")
    ok = all(r["passed"] for r in rows)
    print(f"\nA. metadata validation: {'PASS' if ok else 'FAIL'} "
          f"({sum(r['passed'] for r in rows)}/{len(rows)})")
    (DIST.parent / "test_metadata.json").write_text(json.dumps(rows, indent=2))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
