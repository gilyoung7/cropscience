"""Synthetic parity for all 8 pests: original PyTorch vs wrapper vs FP32 vs FP16.

Deterministic seeded input at each checkpoint's real shape/dtype. Also re-proves,
per pest, the two equivalences the wrapper relies on:
  * dropping the K=1 causal / padding masks is a no-op (valid=True and False)
  * the chunked time-encoder loop is a no-op along the batch axis

Usage (inside .venv-tflite, from this directory):
    python validate_all.py
    python validate_all.py --pests BPH --seeds 12
"""

from __future__ import annotations

import argparse
import json
import traceback
from pathlib import Path

import numpy as np
import torch

from checkpoint import make_synthetic_input, original_mu
from inference_model import load_and_wrap
from pest_configs import PESTS

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"

# Tolerances, in DAYS (mu is a season index in day units).
#
# FP32 = 1e-3 d. mu is O(100) and float32 carries ~7 significant digits, so
# eps-level relative error lands near 1e-5 absolute. Reassociation differences
# between ATen and LiteRT kernels (layernorm/softmax/matmul tiling) accumulate a
# few ulps. 1e-3 d (~86 s) sits 4 orders below the model's own sigma=5 d and far
# below the 1-day rounding the API applies to the 95% PI, so it is operationally
# lossless yet tight enough to catch a real numerical fault.
#
# FP16 = 0.1 d. fp16 has ~3 decimal digits; weight rounding perturbs head_mu,
# and the bounded-sigmoid lead head amplifies that across a 68-day span. 0.1 d
# (~2.4 h) is still 50x below sigma=5 d and below the API's 1-day PI rounding,
# so it cannot change a rounded response, while being ~4x tighter than the worst
# error dynamic-range quantization produced on BPH (0.93 d) — which is exactly
# the failure mode this bound exists to reject.
ATOL = {"fp32": 1e-3, "fp16": 0.1}
VARIANTS = ("fp32", "fp16")


def _interp(path: Path):
    from ai_edge_litert.interpreter import Interpreter

    it = Interpreter(model_path=str(path))
    it.allocate_tensors()
    return it


def tflite_mu(interp, X, tstar, valid_mask) -> np.ndarray:
    """Bind inputs by dtype+shape, not by index.

    LiteRT does not guarantee input tensor order matches the Python signature.
    """
    want = {
        "X": X.numpy(),
        "tstar": tstar.numpy().astype(np.int64),
        "valid_mask": valid_mask.numpy(),
    }
    used: set[str] = set()
    for d in interp.get_input_details():
        shape, dtype = tuple(d["shape"]), d["dtype"]
        match = None
        for name, arr in want.items():
            if name in used:
                continue
            cand = arr.astype(dtype) if arr.dtype != dtype else arr
            if tuple(cand.shape) == shape:
                match = (name, cand)
                break
        if match is None:
            raise RuntimeError(f"cannot bind TFLite input {d['name']} {shape} {dtype}")
        used.add(match[0])
        interp.set_tensor(d["index"], match[1])
    if len(used) != len(want):
        raise RuntimeError(f"bound only {used} of {set(want)}")
    interp.invoke()
    out = interp.get_output_details()
    if len(out) != 1:
        raise RuntimeError(f"expected 1 output, got {len(out)}")
    return interp.get_tensor(out[0]["index"])


def check_wrapper_assumptions(loaded, model) -> dict:
    """Re-prove the K=1 mask no-op and the time-encoder chunking no-op."""
    rows = []
    enc = model.tstar_encoder
    dm = loaded.model.in_proj.out_features
    for seed in range(3):
        for valid in (True, False):
            vm = torch.tensor([[valid]])
            z = torch.randn(1, 1, dm, generator=torch.Generator().manual_seed(seed))
            cm = torch.triu(torch.ones(1, 1, dtype=torch.bool), diagonal=1)
            with torch.no_grad():
                a = enc(z, mask=cm, src_key_padding_mask=~vm)
                b = enc(z)
            f = lambda t: torch.nan_to_num(t, 0.0, 0.0, 0.0) * vm.unsqueeze(-1).to(t.dtype)
            a, b = f(a), f(b)
            rows.append({"seed": seed, "valid": valid, "exact": bool(torch.equal(a, b))})

    src = loaded.model
    X, _, _ = make_synthetic_input(loaded, seed=0, K=8)
    B, K, T, _ = X.shape
    x_flat = X.reshape(B * K, T, -1)
    with torch.no_grad():
        single = src._encode_time_chunk(x_flat)
        chunks = torch.cat(
            [src._encode_time_chunk(x_flat[i:i + 3]) for i in range(0, B * K, 3)], dim=0
        )
    return {
        "mask_drop_all_exact": all(r["exact"] for r in rows),
        "mask_drop_rows": rows,
        "chunking_exact": bool(torch.equal(single, chunks)),
        "chunking_max_abs": float((single - chunks).abs().max()),
    }


def validate_pest(pest: str, seeds: int) -> dict:
    loaded, model, cfg = load_and_wrap(pest)
    rec: dict = {
        "pest": pest,
        "checkpoint_loaded": True,
        "d_in": loaded.d_in, "T": loaded.T,
        "doy_start": loaded.doy_start, "doy_end": loaded.doy_end,
        "alert_idx": loaded.alert_tstar_feat_idx,
        "assumptions": check_wrapper_assumptions(loaded, model),
    }

    interps = {}
    for v in VARIANTS:
        p = ARTIFACTS / pest / f"{pest}_stage2_{v}.tflite"
        interps[v] = _interp(p) if p.is_file() else None
        rec[f"{v}_artifact_present"] = p.is_file()

    # Spread alerts across the pest's own valid DOY range.
    lo, hi = loaded.doy_start + 5, loaded.doy_end - 30
    alerts = np.linspace(lo, hi, seeds)

    d_ow: list[float] = []
    per_variant: dict[str, list[float]] = {v: [] for v in VARIANTS}
    sample = None
    for i in range(seeds):
        X, t, vm = make_synthetic_input(loaded, seed=i, K=1, alert_doy=float(alerts[i]))
        o = original_mu(loaded, X, t, vm)
        with torch.no_grad():
            w = model(X, t, vm)
        d_ow.append(float((o - w).abs().max()))
        for v in VARIANTS:
            if interps[v] is None:
                continue
            tt = torch.from_numpy(np.asarray(tflite_mu(interps[v], X, t, vm)).reshape(o.shape))
            per_variant[v].append(float((o - tt).abs().max()))
        if sample is None:
            sample = (X, t, vm, o)

    X, t, vm, o = sample
    rec["input_shape"] = list(X.shape)
    rec["input_dtypes"] = {"X": str(X.dtype), "tstar": str(t.dtype), "valid_mask": str(vm.dtype)}
    rec["output_shape"] = list(o.shape)
    rec["output_dtype"] = str(o.dtype)
    for v in VARIANTS:
        det = interps[v].get_output_details()[0] if interps[v] else None
        rec[f"{v}_tflite_output_shape"] = list(map(int, det["shape"])) if det else None
        rec[f"{v}_tflite_output_dtype"] = np.dtype(det["dtype"]).name if det else None

    rec["wrapper_max_abs_days"] = max(d_ow)
    rec["wrapper_parity_exact"] = max(d_ow) == 0.0
    for v in VARIANTS:
        vals = per_variant[v]
        if not vals:
            rec[f"{v}_max_abs_days"] = None
            rec[f"{v}_mean_abs_days"] = None
            rec[f"{v}_pass"] = False
            continue
        rec[f"{v}_max_abs_days"] = max(vals)
        rec[f"{v}_mean_abs_days"] = sum(vals) / len(vals)
        rec[f"{v}_atol_days"] = ATOL[v]
        rec[f"{v}_pass"] = max(vals) <= ATOL[v]
    rec["seeds"] = seeds
    rec["passed"] = (
        rec["wrapper_parity_exact"]
        and rec["assumptions"]["mask_drop_all_exact"]
        and rec["assumptions"]["chunking_exact"]
        and all(rec[f"{v}_pass"] for v in VARIANTS)
    )
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pests", nargs="*", default=list(PESTS))
    ap.add_argument("--seeds", type=int, default=8)
    args = ap.parse_args()

    rows, failures = [], []
    for pest in args.pests:
        try:
            rows.append(validate_pest(pest, args.seeds))
        except Exception as e:
            failures.append({"pest": pest, "error": f"{type(e).__name__}: {e}",
                             "traceback": traceback.format_exc()})
            print(f"  FAIL {pest}: {type(e).__name__}: {str(e)[:120]}")

    print(f"\n{'pest':<20}{'shape':>16}{'wrap':>7}{'fp32_max_d':>12}{'fp32':>6}"
          f"{'fp16_max_d':>12}{'fp16':>6}{'all':>5}")
    for r in rows:
        shape = f"{r['input_shape'][2]}x{r['input_shape'][3]}"
        print(f"{r['pest']:<20}{shape:>16}"
              f"{'OK' if r['wrapper_parity_exact'] else 'BAD':>7}"
              f"{r['fp32_max_abs_days']:>12.3e}{'PASS' if r['fp32_pass'] else 'FAIL':>6}"
              f"{r['fp16_max_abs_days']:>12.3e}{'PASS' if r['fp16_pass'] else 'FAIL':>6}"
              f"{'PASS' if r['passed'] else 'FAIL':>5}")

    ok = all(r["passed"] for r in rows) and not failures
    print(f"\ntolerances: fp32 <= {ATOL['fp32']} d, fp16 <= {ATOL['fp16']} d "
          f"| seeds per pest: {args.seeds}")
    print(f"RESULT: {'PASS' if ok else 'FAIL'}  ({len(rows)}/{len(args.pests)} pests validated)")

    (ARTIFACTS / "validate_all.json").write_text(
        json.dumps({"tolerances_days": ATOL, "rows": rows, "failures": failures}, indent=2)
    )
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
