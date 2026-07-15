"""D. Prove the package runs with no PyTorch.

Must be executed INSIDE a venv built from requirements-runtime.txt alone. It
refuses to report success if torch is importable, so a stray torch in
.venv-tflite or conda cannot make it pass.

    python3 -m venv .venv-runtime-test
    ./.venv-runtime-test/bin/python -m pip install -r requirements-runtime.txt
    ./.venv-runtime-test/bin/python tests/test_no_torch_dependency.py

Checks:
  * torch / tensorflow / xgboost are NOT importable
  * the package contains no .pt/.pth/checkpoint
  * all 8 metadata load
  * synthetic inference succeeds for all 8
  * the CLI runs
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
DEFAULT_PKG = HERE.parent / "dist" / "stage2_litert"

BANNED = ("torch", "torchvision", "torchaudio", "tensorflow", "xgboost")


def check_banned_absent() -> tuple[bool, list[str]]:
    notes = []
    ok = True
    for m in BANNED:
        try:
            mod = importlib.import_module(m)
        except ImportError:
            notes.append(f"    OK   {m:<14} not importable")
        else:
            ok = False
            notes.append(
                f"    FAIL {m:<14} IS importable ({getattr(mod, '__version__', '?')}) "
                f"at {getattr(mod, '__file__', '?')}"
            )
    return ok, notes


def check_no_checkpoints(pkg: Path) -> tuple[bool, list[str]]:
    bad = [p for p in pkg.rglob("*") if p.suffix.lower() in (".pt", ".pth", ".ckpt", ".onnx")]
    if bad:
        return False, [f"    FAIL found {len(bad)} checkpoint-like file(s): "
                       f"{[str(p.relative_to(pkg)) for p in bad[:5]]}"]
    n_tflite = len(list(pkg.rglob("*.tflite")))
    return True, [f"    OK   no .pt/.pth/.ckpt/.onnx in package ({n_tflite} .tflite present)"]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", type=Path, default=DEFAULT_PKG)
    args = ap.parse_args()
    pkg = args.package

    print("D. torch-free runtime verification")
    print(f"  python     : {sys.executable}")
    print(f"  package    : {pkg}\n")

    results: dict[str, bool] = {}

    print("  1. banned frameworks absent:")
    ok, notes = check_banned_absent()
    print("\n".join(notes))
    results["banned_absent"] = ok

    print("\n  2. no checkpoints in package:")
    ok, notes = check_no_checkpoints(pkg)
    print("\n".join(notes))
    results["no_checkpoints"] = ok

    print("\n  3. metadata loads for all pests:")
    sys.path.insert(0, str(pkg))
    try:
        from runtime.schema import load_metadata

        manifest = json.loads((pkg / "manifest.json").read_text())
        pests = manifest["pests"]
        for p in pests:
            md = load_metadata(pkg / "models", p)
            md.verify_files()
        print(f"    OK   {len(pests)}/{len(pests)} metadata loaded + sha256 verified")
        results["metadata"] = True
    except Exception as e:
        print(f"    FAIL {type(e).__name__}: {e}")
        results["metadata"] = False

    print("\n  4. synthetic inference for all pests:")
    try:
        sys.path.insert(0, str(HERE))
        from test_runtime_synthetic import run as run_synth

        rc = run_synth(pkg)
        results["synthetic"] = rc == 0
    except Exception as e:
        print(f"    FAIL {type(e).__name__}: {e}")
        results["synthetic"] = False

    print("\n  5. CLI runs:")
    try:
        r = subprocess.run(
            [sys.executable, str(pkg / "predict.py"), "--list-pests"],
            capture_output=True, text=True, timeout=120,
        )
        listed = json.loads(r.stdout)["pests"] if r.returncode == 0 else []
        ok = r.returncode == 0 and len(listed) == 8
        print(f"    {'OK  ' if ok else 'FAIL'} predict.py --list-pests -> rc={r.returncode}, "
              f"{len(listed)} pests")
        if not ok:
            print(f"      stderr: {r.stderr[:200]}")
        results["cli"] = ok
    except Exception as e:
        print(f"    FAIL {type(e).__name__}: {e}")
        results["cli"] = False

    all_ok = all(results.values())
    print("\n  summary:")
    for k, v in results.items():
        print(f"    {k:<18} {'PASS' if v else 'FAIL'}")
    print(f"\nD. torch-free runtime: {'PASS' if all_ok else 'FAIL'}")
    if not results["banned_absent"]:
        print("  NOTE: a banned framework was importable — this environment does NOT "
              "prove torch-independence. Re-run inside a venv built from "
              "requirements-runtime.txt alone.")
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
