"""Export all 8 Stage-2 models to FP32 + FP16 TFLite.

Path: torch.export -> litert_torch.convert -> .tflite

NOTE on package naming: `ai-edge-torch` is now a deprecation shim — the project
was renamed to `litert-torch` and `ai_edge_torch.convert` no longer exists. The
real entry point is `litert_torch.convert`, which still runs torch.export
underneath, i.e. the same official path under its current name.

Usage (inside .venv-tflite, from this directory):
    python export_all.py                      # all 8, fp32 + fp16
    python export_all.py --pests BPH WBPH     # subset
    python export_all.py --variants fp32      # one variant

Writes artifacts/<pest>/<pest>_stage2_<variant>.tflite. A failure on one pest is
recorded and the run continues with the rest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import traceback
from pathlib import Path

import torch

from checkpoint import ckpt_path, make_synthetic_input, original_mu
from inference_model import load_and_wrap
from pest_configs import PESTS

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"
VARIANTS = ("fp32", "fp16")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _quant_config(variant: str):
    if variant == "fp32":
        return None
    if variant != "fp16":
        raise ValueError(f"unsupported variant {variant!r}")
    from litert_torch.generative.quantize import quant_recipes
    from litert_torch.quantize import quant_config as qcfg

    recipe = quant_recipes.full_fp16_recipe()
    if isinstance(recipe, qcfg.QuantConfig):
        return recipe
    return qcfg.QuantConfig(generative_recipe=recipe)


def export_one(pest: str, variant: str) -> dict:
    """Convert one (pest, variant). Raises on failure; caller records it."""
    import litert_torch

    loaded, model, cfg = load_and_wrap(pest)
    X, tstar, valid_mask = make_synthetic_input(loaded, seed=0, K=1)
    sample = (X, tstar, valid_mask)

    # Refuse to emit an artifact from a wrapper that has drifted from the ckpt.
    ref = original_mu(loaded, X, tstar, valid_mask)
    with torch.no_grad():
        got = model(X, tstar, valid_mask)
    if not torch.equal(ref, got):
        raise RuntimeError(
            f"[{pest}] refusing to export: wrapper mu != original mu "
            f"(max|d|={(ref - got).abs().max().item():.3e})"
        )

    kwargs = {}
    qc = _quant_config(variant)
    if qc is not None:
        kwargs["quant_config"] = qc

    edge = litert_torch.convert(model, sample, **kwargs)
    out_dir = ARTIFACTS / pest
    out_dir.mkdir(parents=True, exist_ok=True)
    out = out_dir / f"{pest}_stage2_{variant}.tflite"
    edge.export(str(out))

    rec = {
        "pest": pest,
        "variant": variant,
        "path": str(out.relative_to(ARTIFACTS.parent)),
        "bytes": out.stat().st_size,
        "sha256": sha256(out),
        "ckpt": str(ckpt_path(pest)),
        "ckpt_sha256": sha256(ckpt_path(pest)),
        "d_in": loaded.d_in,
        "T": loaded.T,
        "doy_start": loaded.doy_start,
        "doy_end": loaded.doy_end,
        "alert_idx": loaded.alert_tstar_feat_idx,
        "input_shapes": {
            "X": list(X.shape), "tstar": list(tstar.shape),
            "valid_mask": list(valid_mask.shape),
        },
        "input_dtypes": {
            "X": str(X.dtype), "tstar": str(tstar.dtype),
            "valid_mask": str(valid_mask.dtype),
        },
        "output_shape": list(ref.shape),
        "reference_mu_pytorch": float(ref.flatten()[0]),
        "torch_version": torch.__version__,
        "note": "mu is a 1-based season index; DOY = mu + doy_start - 1",
    }
    (out_dir / f"{pest}_stage2_{variant}.meta.json").write_text(json.dumps(rec, indent=2))
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pests", nargs="*", default=list(PESTS))
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()

    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    results: list[dict] = []
    failures: list[dict] = []

    for pest in args.pests:
        for variant in args.variants:
            try:
                rec = export_one(pest, variant)
                results.append(rec)
                print(f"  OK   {pest:<20} {variant:<7} {rec['bytes']:>9,} B  "
                      f"sha256={rec['sha256'][:16]}...")
            except Exception as e:
                failures.append({
                    "pest": pest, "variant": variant,
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                })
                print(f"  FAIL {pest:<20} {variant:<7} {type(e).__name__}: {str(e)[:110]}")

    (ARTIFACTS / "export_results.json").write_text(
        json.dumps({"results": results, "failures": failures}, indent=2)
    )
    print(f"\nexported {len(results)} / {len(args.pests) * len(args.variants)}; "
          f"{len(failures)} failed")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
