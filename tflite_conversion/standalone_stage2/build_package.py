"""Build the standalone Stage-2 LiteRT package.

BUILD-TIME ONLY: this reads the .pt checkpoints (needs torch). The package it
produces runs with no torch, no tensorflow, no xgboost and no .pt.

    python build_package.py                 # all 8 pests
    python build_package.py --pests BPH     # subset
    python build_package.py --export-missing-tflite

Reads (never modifies):
  api_handoff_transformer/assets/stage2/<pest>/lead_v3_final_checkpoint_run4.pt
  api_handoff_transformer/configs/fallback_policy.yaml
  tflite_conversion/stage2/artifacts/<pest>/<pest>_stage2_{fp16,fp32}.tflite

Writes only under dist/.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parents[1]
PKG_ROOT = REPO_ROOT / "api_handoff_transformer"
STAGE2_ARTIFACTS = REPO_ROOT / "tflite_conversion" / "stage2" / "artifacts"
DIST = HERE / "dist" / "stage2_litert"

sys.path.insert(0, str(REPO_ROOT / "tflite_conversion" / "stage2"))
sys.path.insert(0, str(HERE))

from runtime.schema import (  # noqa: E402
    DISPATCH_FEATURE_NAMES,
    DISPATCH_MISSING_NAME,
    METADATA_SCHEMA_VERSION,
    COORD_COLS,
    PHENO_COLS,
    sha256_file,
)

VARIANTS = ("fp16", "fp32")

# Files that must never appear in the built package.
FORBIDDEN_SUFFIXES = (".pt", ".pth", ".onnx", ".ckpt", ".pkl", ".pyc")
FORBIDDEN_DIR_NAMES = ("__pycache__", ".venv", ".venv-tflite", "assets", "site-packages")
# Top-level module names no packaged .py may import. Checked via the AST, not by
# text search: the runtime docstrings legitimately *cite* upstream paths like
# api_handoff_transformer/infer/preprocess.py, and a raw token scan would flag
# those citations while still missing an obfuscated real import.
FORBIDDEN_IMPORT_ROOTS = frozenset({
    "torch", "torchvision", "torchaudio", "tensorflow", "tensorflow_lite",
    "xgboost", "sklearn", "api_handoff_transformer", "infer",
})


class BuildError(RuntimeError):
    pass


def _load_policy_offset(pest: str) -> int:
    import yaml

    pol = yaml.safe_load((PKG_ROOT / "configs" / "fallback_policy.yaml").read_text())
    per = (pol.get("per_pest", {}) or {}).get(pest) or {}
    v = per.get("selected_fixed_offset")
    if v is None:
        raise BuildError(f"[{pest}] selected_fixed_offset missing from fallback_policy.yaml")
    return int(v)


def _policy_status(pest: str) -> dict:
    import yaml

    pol = yaml.safe_load((PKG_ROOT / "configs" / "fallback_policy.yaml").read_text())
    per = (pol.get("per_pest", {}) or {}).get(pest) or {}
    return {
        "recommended_source": per.get("recommended_source"),
        "learned_output_status": per.get("learned_output_status"),
    }


def extract_metadata(pest: str) -> tuple[dict, np.ndarray, np.ndarray]:
    """Read the checkpoint once and cross-check it against the shared loader.

    Uses tflite_conversion/stage2/checkpoint.py::load_pest, which already
    validates the ckpt against the expected per-pest contract AND against the
    real state_dict tensor shapes, raising ConfigMismatch on any drift.
    """
    from checkpoint import load_pest  # tflite_conversion/stage2/checkpoint.py

    loaded, cfg = load_pest(pest)
    ck_path = loaded.ckpt_path

    fn = list(loaded.feature_names)
    D = int(loaded.d_in)
    n_disp = 15
    nbase = (D - n_disp) // 2
    if 2 * nbase + n_disp != D:
        raise BuildError(f"[{pest}] feature layout: D={D} is not 2*nbase+{n_disp}")

    base = fn[:nbase]
    miss = fn[nbase:2 * nbase]
    disp = fn[2 * nbase:]
    if miss != [f"{c}__miss" for c in base]:
        raise BuildError(f"[{pest}] miss block mismatch:\n  {miss}")
    if disp != list(DISPATCH_FEATURE_NAMES) + [DISPATCH_MISSING_NAME]:
        raise BuildError(f"[{pest}] dispatch block mismatch:\n  {disp}")
    if fn[loaded.alert_tstar_feat_idx] != "alert_tstar":
        raise BuildError(
            f"[{pest}] feature_names[{loaded.alert_tstar_feat_idx}] != 'alert_tstar'"
        )

    # Normalization exactly as the production loader resolves it, including the
    # dispatch-channel bypass (infer/ckpt.py:137-142 forces the last 15 to
    # mean=0/std=1 so those channels pass through unnormalized).
    mean = loaded.norm_mean.detach().cpu().numpy().astype(np.float32)
    std = loaded.norm_std.detach().cpu().numpy().astype(np.float32)
    bypass = [i for i in range(D) if mean[i] == 0.0 and std[i] == 1.0]
    dispatch_idx = list(range(2 * nbase, D))
    if not set(dispatch_idx).issubset(set(bypass)):
        raise BuildError(
            f"[{pest}] dispatch channels {dispatch_idx} are not all norm-bypassed; "
            f"bypass set = {bypass}"
        )

    m = loaded.model
    meta = {
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "pest": pest,
        "model_family": "HierarchicalCausalHazardTransformer",
        "checkpoint_variant": "lead_v3_final / checkpoint_run4 / split3(val=2023,test=2024)",
        "source_checkpoint": {
            "path": str(ck_path.relative_to(REPO_ROOT)),
            "sha256": sha256_file(ck_path),
            "bytes": ck_path.stat().st_size,
        },
        # architecture
        "T": int(loaded.T),
        "d_in": D,
        "d_model": int(m.in_proj.out_features),
        "n_head": int(m.time_encoder.layers[0].self_attn.num_heads),
        "n_layers": int(len(m.time_encoder.layers)),
        "tstar_layers": int(len(m.tstar_encoder.layers)),
        "doy_start": int(loaded.doy_start),
        "doy_end": int(loaded.doy_end),
        "alert_idx": int(loaded.alert_tstar_feat_idx),
        "nowcast_window": int(loaded.nowcast_window),
        "selected_offset": _load_policy_offset(pest),
        # mu / interval
        "mu_mode": str(m.mu_mode),
        "lead_min": float(loaded.lead_min),
        "lead_max": float(loaded.lead_max),
        "gaussian_sigma": float(loaded.sigma),
        "mu_output_semantics": (
            "mu is a 1-based season index (NOT absolute DOY). "
            "absolute DOY = mu + doy_start - 1."
        ),
        "prediction_interval_rule": {
            "source": "api_handoff_transformer/run_predict.py:490-498",
            "half_width_days": round(1.96 * float(loaded.sigma), 1),
            "formula": (
                "half = round(1.96*sigma, 1); "
                "mu_doy = round(mu_doy_temporal, 2); "
                "pi_95 = [int(round(mu_doy_temporal - half)), "
                "int(round(mu_doy_temporal + half))]"
            ),
            "note": (
                "The interval is computed from the UNROUNDED mu_doy_temporal, not "
                "from the reported (rounded) mu_doy. Python round() is "
                "round-half-to-even; use plain floats to match the API at .5 "
                "boundaries."
            ),
        },
        # channels
        "feature_names": fn,
        "base_channels": base,
        "miss_channels": miss,
        "dispatch_channels": disp,
        "norm_bypass_channel_indices": bypass,
        "norm_bypass_mask": [bool(i in set(bypass)) for i in range(D)],
        "requires_site_coords": any(c in base for c in COORD_COLS),
        "requires_phenology": any(c in base for c in PHENO_COLS),
        "input_daily_doy_coverage_required": [1, int(loaded.doy_end)],
        # io
        "inputs": [
            {"name": "X", "shape": [1, 1, int(loaded.T), D], "dtype": "float32"},
            {"name": "tstar", "shape": [1, 1], "dtype": "int64"},
            {"name": "valid_mask", "shape": [1, 1], "dtype": "bool"},
        ],
        "outputs": [{"name": "mu", "shape": [1, 1], "dtype": "float32"}],
        "api_policy": _policy_status(pest),
        "files": {},
    }
    return meta, mean, std


def _export_missing(pest: str) -> None:
    script = REPO_ROOT / "tflite_conversion" / "stage2" / "export_all.py"
    cmd = [sys.executable, str(script), "--pests", pest]
    print(f"    running: {' '.join(cmd)}")
    r = subprocess.run(cmd, cwd=script.parent, capture_output=True, text=True)
    if r.returncode != 0:
        raise BuildError(
            f"[{pest}] export_all.py failed (rc={r.returncode}): {r.stderr[-400:]}"
        )


def build_pest(pest: str, export_missing: bool) -> dict:
    out_dir = DIST / "models" / pest
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. source .tflite must exist
    srcs = {v: STAGE2_ARTIFACTS / pest / f"{pest}_stage2_{v}.tflite" for v in VARIANTS}
    absent = [v for v, p in srcs.items() if not p.is_file()]
    if absent:
        if not export_missing:
            raise BuildError(
                f"[{pest}] missing TFLite artifact(s) {absent}: "
                f"{[str(srcs[v]) for v in absent]}\n"
                f"  Build them first:\n"
                f"    cd tflite_conversion/stage2 && python export_all.py --pests {pest}\n"
                f"  or re-run this script with --export-missing-tflite."
            )
        print(f"    {pest}: {absent} missing -> exporting")
        _export_missing(pest)
        absent = [v for v, p in srcs.items() if not p.is_file()]
        if absent:
            raise BuildError(f"[{pest}] still missing after export: {absent}")

    # 2. metadata + normalization from the checkpoint
    meta, mean, std = extract_metadata(pest)

    # 3. normalization.npz (float32, exact)
    npz = out_dir / "normalization.npz"
    np.savez(npz, norm_mean=mean, norm_std=std)
    with np.load(npz) as z:  # verify round-trip is lossless
        if not (np.array_equal(z["norm_mean"], mean) and np.array_equal(z["norm_std"], std)):
            raise BuildError(f"[{pest}] normalization.npz round-trip changed values")
    meta["files"]["normalization"] = {
        "filename": "normalization.npz",
        "arrays": {
            "norm_mean": {"shape": list(mean.shape), "dtype": str(mean.dtype)},
            "norm_std": {"shape": list(std.shape), "dtype": str(std.dtype)},
        },
        "bytes": npz.stat().st_size,
        "sha256": sha256_file(npz),
    }

    # 4. copy models
    for v in VARIANTS:
        dst = out_dir / f"model_{v}.tflite"
        shutil.copy2(srcs[v], dst)
        meta["files"][f"model_{v}"] = {
            "filename": dst.name,
            "bytes": dst.stat().st_size,
            "sha256": sha256_file(dst),
            "source": str(srcs[v].relative_to(REPO_ROOT)),
        }
        if sha256_file(dst) != sha256_file(srcs[v]):
            raise BuildError(f"[{pest}] {v} copy differs from source")

    (out_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return meta


def copy_runtime() -> list[str]:
    dst = DIST / "runtime"
    if dst.exists():
        shutil.rmtree(dst)
    dst.mkdir(parents=True)
    names = []
    for f in sorted((HERE / "runtime").glob("*.py")):
        shutil.copy2(f, dst / f.name)
        names.append(f.name)
    shutil.copy2(HERE / "requirements-runtime.txt", DIST / "requirements-runtime.txt")

    # predict.py entry point at the package root
    (DIST / "predict.py").write_text(
        '#!/usr/bin/env python3\n'
        '"""Standalone Stage-2 CLI. See README.md."""\n'
        'import sys\n'
        'from pathlib import Path\n'
        'sys.path.insert(0, str(Path(__file__).resolve().parent))\n'
        'from runtime.cli import main\n'
        '\n'
        'if __name__ == "__main__":\n'
        '    raise SystemExit(main())\n',
        encoding="utf-8",
    )
    return names


def _imported_roots(tree: ast.AST) -> set[tuple[str, int]]:
    """Every top-level module name imported by this file, with line numbers."""
    found: set[tuple[str, int]] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                found.add((a.name.split(".")[0], node.lineno))
        elif isinstance(node, ast.ImportFrom):
            # level>0 is a relative import (from .schema import ...) — always local
            if node.level == 0 and node.module:
                found.add((node.module.split(".")[0], node.lineno))
    return found


def check_no_contamination() -> list[str]:
    """Fail the build if anything forbidden reached dist/.

    Import checks parse the AST so that docstrings citing upstream files are not
    mistaken for imports, and a real import cannot hide behind formatting.
    """
    problems: list[str] = []
    for p in DIST.rglob("*"):
        rel = p.relative_to(DIST)
        if any(part in FORBIDDEN_DIR_NAMES for part in rel.parts):
            problems.append(f"forbidden directory in package: {rel}")
            continue
        if p.is_dir():
            continue
        if p.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden file type {p.suffix}: {rel}")
        if p.suffix == ".py":
            src = p.read_text(encoding="utf-8", errors="replace")
            try:
                tree = ast.parse(src, filename=str(p))
            except SyntaxError as e:
                problems.append(f"packaged file does not parse: {rel}: {e}")
                continue
            for root, lineno in sorted(_imported_roots(tree)):
                if root in FORBIDDEN_IMPORT_ROOTS:
                    problems.append(f"forbidden import {root!r} in {rel}:{lineno}")

    req = (DIST / "requirements-runtime.txt").read_text()
    for line in req.splitlines():
        ls = line.strip()
        if not ls or ls.startswith("#"):
            continue
        name = re.split(r"[=<>!~\[ ]", ls, maxsplit=1)[0].strip().lower()
        if name in {"torch", "torchvision", "torchaudio", "tensorflow", "xgboost"}:
            problems.append(f"requirements-runtime.txt requires {name!r}: {ls}")
    return problems


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pests", nargs="*", default=None)
    ap.add_argument("--export-missing-tflite", action="store_true",
                    help="run tflite_conversion/stage2/export_all.py for missing artifacts")
    ap.add_argument("--clean", action="store_true", help="remove dist/ first")
    args = ap.parse_args()

    from pest_configs import PESTS  # tflite_conversion/stage2/pest_configs.py

    pests = args.pests or list(PESTS)

    if args.clean and DIST.exists():
        shutil.rmtree(DIST.parent)
    DIST.mkdir(parents=True, exist_ok=True)

    print(f"building standalone package -> {DIST}")
    built, failures = [], []
    for pest in pests:
        try:
            meta = build_pest(pest, args.export_missing_tflite)
            built.append(meta)
            sz = dir_size(DIST / "models" / pest)
            print(f"  OK   {pest:<20} d_in={meta['d_in']:<3} T={meta['T']:<4} "
                  f"{sz:>9,} B")
        except Exception as e:
            failures.append({"pest": pest, "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()})
            print(f"  FAIL {pest:<20} {type(e).__name__}: {str(e)[:110]}")

    runtime_files = copy_runtime()
    if (HERE / "README.md").is_file():
        shutil.copy2(HERE / "README.md", DIST / "README.md")
    else:
        raise BuildError(f"README.md not found at {HERE / 'README.md'}; the package ships it")

    # Running the package leaves runtime/__pycache__ behind (normal CPython
    # behaviour). Purge it so the contamination check reflects what the build
    # actually produces rather than what a previous run left.
    for cache in DIST.rglob("__pycache__"):
        shutil.rmtree(cache, ignore_errors=True)

    problems = check_no_contamination()
    if problems:
        print("\nCONTAMINATION CHECK FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print(f"\ncontamination check: PASS (no .pt/.pth, no torch/tf/xgboost imports, "
          f"no __pycache__/venv/assets)")

    per_pest_sizes = {
        m["pest"]: {
            "total_bytes": dir_size(DIST / "models" / m["pest"]),
            "model_fp16_bytes": m["files"]["model_fp16"]["bytes"],
            "model_fp32_bytes": m["files"]["model_fp32"]["bytes"],
            "normalization_bytes": m["files"]["normalization"]["bytes"],
        }
        for m in built
    }
    manifest = {
        "manifest_version": "1.0",
        "metadata_schema_version": METADATA_SCHEMA_VERSION,
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package": "stage2_litert",
        "description": (
            "Standalone Stage-2 pest-timing inference. Runs on LiteRT/TFLite with "
            "no PyTorch, TensorFlow, XGBoost or .pt checkpoints. Stage 1, weather "
            "ingestion and API wiring are out of scope: alert_tstar, dispatch "
            "features and (for non-BPH pests) site coords + phenology are inputs."
        ),
        "default_variant": "fp16",
        "variants": list(VARIANTS),
        "pests": sorted(m["pest"] for m in built),
        "failures": failures,
        "runtime_files": runtime_files,
        "models": {
            m["pest"]: {
                "d_in": m["d_in"], "T": m["T"],
                "doy_start": m["doy_start"], "doy_end": m["doy_end"],
                "alert_idx": m["alert_idx"],
                "requires_site_coords": m["requires_site_coords"],
                "requires_phenology": m["requires_phenology"],
                "selected_offset": m["selected_offset"],
                "api_policy": m["api_policy"],
                "files": {k: {"filename": v["filename"], "sha256": v["sha256"],
                              "bytes": v["bytes"]}
                          for k, v in m["files"].items()},
                "source_checkpoint_sha256": m["source_checkpoint"]["sha256"],
            }
            for m in built
        },
        "sizes": {
            "package_total_bytes": dir_size(DIST),
            "models_total_bytes": dir_size(DIST / "models"),
            "per_pest": per_pest_sizes,
        },
    }
    (DIST / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    total = manifest["sizes"]["package_total_bytes"]
    print(f"\nbuilt {len(built)}/{len(pests)} pests; {len(failures)} failed")
    print(f"package total: {total:,} B ({total / 1e6:.2f} MB)  -> {DIST}")
    if failures:
        print("failures:")
        for f in failures:
            print(f"  {f['pest']}: {f['error']}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
