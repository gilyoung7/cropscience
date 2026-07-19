"""Build the lightweight portable API package.

Copies portable Stage-1 JSON assets, Stage-2 LiteRT models, configs and
climatology into assets/, writes a SHA-256 manifest, scans for forbidden
content, and (optionally) emits a production archive.

    python build_package.py                    # assemble assets/ + manifest
    python build_package.py --archive          # + dist/<name>.tar.gz (FP16 only)
    python build_package.py --include-fp32     # keep FP32 in assets (validation build)

Reads (never modifies): stage1_xgboost_migration/, api_handoff_transformer/,
tflite_conversion/standalone_stage2/dist/.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import shutil
import sys
import tarfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
API = REPO / "api_handoff_transformer"
S1_MIG = REPO / "stage1_xgboost_migration" / "artifacts"
S2_DIST = REPO / "tflite_conversion" / "standalone_stage2" / "dist" / "stage2_litert" / "models"

PESTS = ["BPH", "WBPH", "bacterial_blight", "blast", "brown_spot",
         "rice_stem_borer_1", "rice_stem_borer_2", "sheath_blight"]

FORBIDDEN_SUFFIXES = (".pt", ".pth", ".onnx", ".ckpt", ".pkl", ".pyc")
FORBIDDEN_DIRS = ("__pycache__", ".venv", "site-packages")
FORBIDDEN_IMPORTS = frozenset({
    "torch", "torchvision", "torchaudio", "tensorflow", "keras", "sklearn",
    "litert_torch", "ai_edge_torch", "jupyter", "notebook",
})


class BuildError(RuntimeError):
    pass


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)
    return h.hexdigest()


def dir_size(p: Path) -> int:
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file())


def copy_stage1(pest: str, include_fp32: bool) -> dict:
    """Portable JSON models + gate + temperature + site_history."""
    dst = HERE / "assets" / "stage1" / pest
    dst.mkdir(parents=True, exist_ok=True)
    files: dict = {}
    for branch in ("A", "D"):
        src_d = S1_MIG / pest / branch
        out_d = dst / branch
        out_d.mkdir(parents=True, exist_ok=True)
        for name in ("model.json", "calibration.json", "metadata.json"):
            s = src_d / name
            if not s.is_file():
                raise BuildError(f"[{pest}/{branch}] missing portable asset: {s}")
            shutil.copy2(s, out_d / name)
            files[f"{branch}/{name}"] = {"bytes": (out_d / name).stat().st_size,
                                         "sha256": sha256_file(out_d / name)}
    g = S1_MIG / pest / "gate.json"
    if not g.is_file():
        raise BuildError(f"[{pest}] missing gate.json: {g}")
    shutil.copy2(g, dst / "gate.json")
    files["gate.json"] = {"bytes": (dst / "gate.json").stat().st_size,
                          "sha256": sha256_file(dst / "gate.json")}
    # site_history.json is an external asset the migration did not copy.
    sh = API / "assets" / "stage1" / pest / "site_history.json"
    if not sh.is_file():
        raise BuildError(f"[{pest}] missing site_history.json: {sh}")
    shutil.copy2(sh, dst / "site_history.json")
    files["site_history.json"] = {"bytes": (dst / "site_history.json").stat().st_size,
                                  "sha256": sha256_file(dst / "site_history.json")}
    return files


def copy_stage2(pest: str, include_fp32: bool) -> dict:
    dst = HERE / "assets" / "stage2" / pest
    dst.mkdir(parents=True, exist_ok=True)
    src = S2_DIST / pest
    if not (src / "metadata.json").is_file():
        raise BuildError(
            f"[{pest}] Stage-2 LiteRT assets missing at {src}.\n"
            f"  Rebuild them first:\n"
            f"    python tflite_conversion/standalone_stage2/build_package.py --clean"
        )
    files: dict = {}
    wanted = ["metadata.json", "normalization.npz", "model_fp16.tflite"]
    if include_fp32:
        wanted.append("model_fp32.tflite")
    for name in wanted:
        s = src / name
        if not s.is_file():
            raise BuildError(f"[{pest}] missing Stage-2 asset: {s}")
        shutil.copy2(s, dst / name)
        files[name] = {"bytes": (dst / name).stat().st_size,
                       "sha256": sha256_file(dst / name)}
    if not include_fp32:
        stale = dst / "model_fp32.tflite"
        if stale.exists():
            stale.unlink()
    return files


def copy_configs() -> dict:
    cdst = HERE / "assets" / "configs"
    cdst.mkdir(parents=True, exist_ok=True)
    src = API / "configs" / "fallback_policy.yaml"
    if not src.is_file():
        raise BuildError(f"fallback_policy.yaml not found: {src}")
    shutil.copy2(src, cdst / "fallback_policy.yaml")
    out = {"fallback_policy.yaml": {"bytes": (cdst / "fallback_policy.yaml").stat().st_size,
                                    "sha256": sha256_file(cdst / "fallback_policy.yaml")}}
    kdst = HERE / "assets" / "climatology"
    kdst.mkdir(parents=True, exist_ok=True)
    for pest in PESTS:
        s = API / "configs" / "climatology" / f"{pest}_climatology_train_stats.csv"
        if not s.is_file():
            raise BuildError(f"climatology CSV missing: {s}")
        shutil.copy2(s, kdst / s.name)
        out[f"climatology/{s.name}"] = {"bytes": (kdst / s.name).stat().st_size,
                                        "sha256": sha256_file(kdst / s.name)}
    return out


def _imported_roots(tree: ast.AST) -> set[tuple[str, int]]:
    found = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Import):
            for a in n.names:
                found.add((a.name.split(".")[0], n.lineno))
        elif isinstance(n, ast.ImportFrom) and n.level == 0 and n.module:
            found.add((n.module.split(".")[0], n.lineno))
    return found


def contamination_scan(root: Path, allow_fp32: bool) -> list[str]:
    """Fail the build on forbidden content. AST-based for imports."""
    problems: list[str] = []
    for p in root.rglob("*"):
        rel = p.relative_to(root)
        if any(part in FORBIDDEN_DIRS for part in rel.parts):
            problems.append(f"forbidden directory: {rel}")
            continue
        if p.is_dir():
            continue
        if p.suffix.lower() in FORBIDDEN_SUFFIXES:
            problems.append(f"forbidden file type {p.suffix}: {rel}")
        if not allow_fp32 and p.name == "model_fp32.tflite":
            problems.append(f"FP32 model in production build: {rel}")
        if p.suffix == ".py":
            try:
                tree = ast.parse(p.read_text(encoding="utf-8", errors="replace"))
            except SyntaxError as e:
                problems.append(f"does not parse: {rel}: {e}")
                continue
            for r, ln in sorted(_imported_roots(tree)):
                if r in FORBIDDEN_IMPORTS:
                    problems.append(f"forbidden import {r!r} in {rel}:{ln}")
    req = (root / "requirements-runtime.txt")
    if req.is_file():
        for line in req.read_text().splitlines():
            ls = line.strip()
            if not ls or ls.startswith("#"):
                continue
            name = ls.split("==")[0].split(">=")[0].strip().lower()
            if name in {"torch", "tensorflow", "xgboost-cpu-cuda", "keras",
                        "scikit-learn", "scipy", "litert-torch", "ai-edge-torch"}:
                problems.append(f"requirements-runtime.txt requires {name!r}")
    return problems


def check_required(include_fp32: bool) -> list[str]:
    missing = []
    for pest in PESTS:
        s1 = HERE / "assets" / "stage1" / pest
        for f in ("gate.json", "site_history.json", "A/model.json", "D/model.json",
                  "A/calibration.json", "D/calibration.json"):
            if not (s1 / f).is_file():
                missing.append(f"stage1/{pest}/{f}")
        s2 = HERE / "assets" / "stage2" / pest
        need = ["metadata.json", "normalization.npz", "model_fp16.tflite"]
        if include_fp32:
            need.append("model_fp32.tflite")
        for f in need:
            if not (s2 / f).is_file():
                missing.append(f"stage2/{pest}/{f}")
        if not (HERE / "assets" / "climatology" /
                f"{pest}_climatology_train_stats.csv").is_file():
            missing.append(f"climatology/{pest}")
    if not (HERE / "assets" / "configs" / "fallback_policy.yaml").is_file():
        missing.append("configs/fallback_policy.yaml")
    return missing


def make_archive(include_fp32: bool, archive_name: str | None = None) -> Path:
    dist = HERE / "dist"
    dist.mkdir(parents=True, exist_ok=True)
    name = archive_name or (
        f"api_handoff_litert_portable{'_fp32' if include_fp32 else ''}.tar.gz")
    if not name.endswith(".tar.gz"):
        name += ".tar.gz"
    out = dist / name
    if out.exists():
        raise BuildError(
            f"refusing to overwrite an existing archive: {out}. "
            f"Pass --archive-name with a new name."
        )
    exclude_names = {"dist", "tests", "build_package.py", "requirements-test.txt",
                     "__pycache__", "input", "output"}

    def _filter(ti: tarfile.TarInfo):
        parts = Path(ti.name).parts
        if any(p in exclude_names for p in parts):
            return None
        if Path(ti.name).suffix.lower() in FORBIDDEN_SUFFIXES:
            return None
        if not include_fp32 and Path(ti.name).name == "model_fp32.tflite":
            return None
        return ti

    with tarfile.open(out, "w:gz") as tf:
        tf.add(HERE, arcname="api_handoff_litert_portable", filter=_filter)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pests", nargs="*", default=PESTS)
    ap.add_argument("--include-fp32", action="store_true",
                    help="keep FP32 Stage-2 models (validation build; not production)")
    ap.add_argument("--archive", action="store_true", help="emit dist/*.tar.gz")
    ap.add_argument("--archive-name", default=None,
                    help="archive filename (refuses to overwrite an existing one)")
    ap.add_argument("--clean", action="store_true")
    args = ap.parse_args()

    if args.clean:
        for d in ("stage1", "stage2"):
            shutil.rmtree(HERE / "assets" / d, ignore_errors=True)

    print(f"building lightweight package -> {HERE}")
    manifest_models: dict = {}
    failures: list[dict] = []
    for pest in args.pests:
        try:
            s1 = copy_stage1(pest, args.include_fp32)
            s2 = copy_stage2(pest, args.include_fp32)
            manifest_models[pest] = {"stage1": s1, "stage2": s2}
            sz = dir_size(HERE / "assets" / "stage1" / pest) + \
                dir_size(HERE / "assets" / "stage2" / pest)
            print(f"  OK   {pest:<20} {sz:>10,} B")
        except Exception as e:
            failures.append({"pest": pest, "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()})
            print(f"  FAIL {pest:<20} {type(e).__name__}: {str(e)[:100]}")

    cfg = copy_configs()

    missing = check_required(args.include_fp32)
    if missing:
        print("\nREQUIRED FILES MISSING:")
        for m in missing:
            print(f"  - {m}")
        return 1

    # Running the package leaves __pycache__ behind (normal CPython behaviour).
    # Purge it so the scan reflects what the build produces, not what a previous
    # run left. The archive filter drops it independently.
    for cache in HERE.rglob("__pycache__"):
        if "dist" not in cache.parts:
            shutil.rmtree(cache, ignore_errors=True)

    problems = contamination_scan(HERE, allow_fp32=args.include_fp32)
    if problems:
        print("\nCONTAMINATION SCAN FAILED:")
        for p in problems:
            print(f"  - {p}")
        return 1
    print("\ncontamination scan: PASS (no .pt/.pth, no torch/tf/sklearn imports, "
          f"{'fp32 allowed' if args.include_fp32 else 'fp16 only'})")

    sizes = {
        "assets_stage1_bytes": dir_size(HERE / "assets" / "stage1"),
        "assets_stage2_bytes": dir_size(HERE / "assets" / "stage2"),
        "assets_configs_bytes": dir_size(HERE / "assets" / "configs"),
        "assets_climatology_bytes": dir_size(HERE / "assets" / "climatology"),
        "assets_total_bytes": dir_size(HERE / "assets"),
        "code_bytes": sum(f.stat().st_size for f in HERE.rglob("*.py")
                          if "dist" not in f.parts and "__pycache__" not in f.parts),
        "per_pest": {
            p: {"stage1": dir_size(HERE / "assets" / "stage1" / p),
                "stage2": dir_size(HERE / "assets" / "stage2" / p)}
            for p in manifest_models
        },
    }
    manifest = {
        "manifest_version": "1.0",
        "built_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "package": "api_handoff_litert_portable",
        "model_generation": "lead_v3_final (deployed). Newer DN models excluded by design.",
        "stage1_backend": "xgboost_json (xgboost.Booster; no scikit-learn)",
        "stage2_backend": "litert_fp16" + (" + litert_fp32" if args.include_fp32 else ""),
        "default_stage2_variant": "fp16",
        "includes_fp32": bool(args.include_fp32),
        "source_reference_api": {
            "zip": "api_handoff_transformer_batch_20260710.zip",
            "sha256": "665e1b85769d45d27e5e7ed0a9c8b9f686068bf88d9397d3e9388be18a0446b4",
        },
        "pests": sorted(manifest_models),
        "failures": failures,
        "models": manifest_models,
        "configs": cfg,
        "sizes": sizes,
    }
    (HERE / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"\nassets total: {sizes['assets_total_bytes']:,} B "
          f"({sizes['assets_total_bytes']/1e6:.2f} MB)")
    print(f"  stage1 {sizes['assets_stage1_bytes']:>10,} B | "
          f"stage2 {sizes['assets_stage2_bytes']:>10,} B | "
          f"configs+clim {sizes['assets_configs_bytes']+sizes['assets_climatology_bytes']:>7,} B")

    if args.archive:
        arc = make_archive(args.include_fp32, args.archive_name)
        print(f"archive: {arc} ({arc.stat().st_size:,} B)")

    print(f"built {len(manifest_models)}/{len(args.pests)} pests; {len(failures)} failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
