"""Prove the lightweight API runs with no heavy ML stack.

Must run INSIDE a venv built from requirements-runtime.txt alone. It refuses to
pass if any banned framework is importable, so a stray torch in .venv-tflite or
conda cannot make it green.

    python3 -m venv .venv-lightweight-api-test
    ./.venv-lightweight-api-test/bin/python -m pip install -r requirements-runtime.txt
    ./.venv-lightweight-api-test/bin/python tests/test_no_heavy_dependencies.py
"""

from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
PKG = HERE.parent

# scipy is deliberately NOT banned: xgboost requires it (pip show xgboost ->
# Requires: numpy, scipy), so it is present transitively in any env that can run
# Stage-1. Our code never imports it. Banning it would make this test fail for a
# reason we cannot fix without replacing xgboost.
BANNED = ("torch", "torchvision", "torchaudio", "tensorflow", "keras",
          "sklearn", "litert_torch", "ai_edge_torch", "jupyter",
          "notebook", "pytest")
PESTS = ["BPH", "WBPH", "bacterial_blight", "blast", "brown_spot",
         "rice_stem_borer_1", "rice_stem_borer_2", "sheath_blight"]


def main() -> int:
    print("=== lightweight API: no-heavy-dependency verification ===")
    print(f"  python : {sys.executable}")
    print(f"  package: {PKG}\n")
    results: dict[str, bool] = {}

    print("  1. banned frameworks not importable:")
    ok = True
    for m in BANNED:
        try:
            mod = importlib.import_module(m)
        except ImportError:
            print(f"    OK   {m:<15} absent")
        else:
            ok = False
            print(f"    FAIL {m:<15} IMPORTABLE ({getattr(mod, '__version__', '?')})")
    results["banned_absent"] = ok

    print("\n  2. no .pt/.pth/CUDA artifacts in the package:")
    bad = [p for p in PKG.rglob("*")
           if p.suffix.lower() in (".pt", ".pth", ".ckpt", ".onnx")
           and "dist" not in p.parts]
    cuda = [p for p in PKG.rglob("*") if "cudnn" in p.name.lower() or "cuda" in p.name.lower()]
    print(f"    {'OK  ' if not bad else 'FAIL'} checkpoints: {len(bad)}")
    print(f"    {'OK  ' if not cuda else 'FAIL'} cuda/cudnn : {len(cuda)}")
    results["no_checkpoints"] = not bad and not cuda

    print("\n  3. API imports:")
    sys.path.insert(0, str(PKG))
    try:
        import run_predict  # noqa: F401
        from infer.stage1_portable import PortableBranch  # noqa: F401
        from infer.stage2_litert import Stage2Model  # noqa: F401
        print("    OK   run_predict + infer.stage1_portable + infer.stage2_litert")
        results["import"] = True
    except Exception as e:
        print(f"    FAIL {type(e).__name__}: {e}")
        results["import"] = False

    print("\n  4. 8-pest assets load (Stage-1 Booster + Stage-2 LiteRT metadata):")
    try:
        from infer.stage1_portable import PortableBranch, load_gate
        from infer.stage2_litert import Stage2Model

        n1 = n2 = 0
        for pest in PESTS:
            load_gate(pest, PKG / "assets" / "stage1")
            for br in ("A", "D"):
                PortableBranch(pest, br, PKG / "assets" / "stage1")
                n1 += 1
            Stage2Model(pest, PKG / "assets" / "stage2", variant="fp16")
            n2 += 1
        print(f"    OK   stage1 branches {n1}/16 · stage2 fp16 models {n2}/8")
        results["assets"] = n1 == 16 and n2 == 8
    except Exception as e:
        print(f"    FAIL {type(e).__name__}: {e}")
        results["assets"] = False

    print("\n  5. single prediction runs:")
    try:
        import numpy as np

        from infer.stage1_portable import PortableBranch

        pb = PortableBranch("BPH", "A", PKG / "assets" / "stage1")
        p = pb.predict_raw(np.zeros((2, pb.booster.num_features()), dtype=np.float32))
        print(f"    OK   Stage-1 Booster forward -> {p.shape}")
        results["predict"] = p.shape == (2,)
    except Exception as e:
        print(f"    FAIL {type(e).__name__}: {e}")
        results["predict"] = False

    print("\n  6. CLI runs:")
    try:
        r = subprocess.run([sys.executable, str(PKG / "run_predict.py"), "--help"],
                           capture_output=True, text=True, timeout=120)
        ok = r.returncode == 0 and "--stage2-variant" in r.stdout
        print(f"    {'OK  ' if ok else 'FAIL'} run_predict.py --help -> rc={r.returncode}")
        results["cli"] = ok
    except Exception as e:
        print(f"    FAIL {type(e).__name__}: {e}")
        results["cli"] = False

    all_ok = all(results.values())
    print("\n  summary:")
    for k, v in results.items():
        print(f"    {k:<16} {'PASS' if v else 'FAIL'}")
    print(f"\nRESULT: {'PASS' if all_ok else 'FAIL'}")
    if not results["banned_absent"]:
        print("  NOTE: a banned framework was importable — this environment does NOT "
              "prove independence. Re-run in a venv built from requirements-runtime.txt.")
    (HERE / "_no_heavy_deps_report.json").write_text(json.dumps(results, indent=2))
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
