"""Benchmark PyTorch wrapper vs FP32/FP16 TFLite for all 8 pests.

Same CPU, same process, same input tensor per pest. PyTorch is pinned to one
thread so the comparison against LiteRT's single-threaded default interpreter is
like-for-like rather than a thread-count artifact.

Absolute latency on a laptop drifts with load/thermal state, so `--repeats`
runs the whole suite N times and reports the per-pest spread. Trust the ratio,
not the absolute number.

Usage (inside .venv-tflite, from this directory):
    python benchmark_all.py
    python benchmark_all.py --iters 200 --warmup 20 --repeats 3
"""

from __future__ import annotations

import argparse
import json
import platform
import statistics
import time
from pathlib import Path

import torch

from checkpoint import make_synthetic_input
from inference_model import load_and_wrap
from pest_configs import PESTS
from validate_all import VARIANTS, _interp, tflite_mu

ARTIFACTS = Path(__file__).resolve().parent / "artifacts"


def time_it(fn, iters: int, warmup: int) -> dict:
    for _ in range(warmup):
        fn()
    s = []
    for _ in range(iters):
        t0 = time.perf_counter()
        fn()
        s.append((time.perf_counter() - t0) * 1000.0)
    s.sort()
    return {
        "mean_ms": statistics.fmean(s),
        "std_ms": statistics.pstdev(s),
        "median_ms": statistics.median(s),
        "p90_ms": s[int(0.90 * len(s)) - 1],
        "min_ms": s[0],
        "max_ms": s[-1],
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=200)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--pests", nargs="*", default=list(PESTS))
    args = ap.parse_args()

    torch.set_num_threads(1)
    env = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "iters": args.iters, "warmup": args.warmup, "repeats": args.repeats,
    }
    print("=== environment ===")
    for k, v in env.items():
        print(f"  {k:16s} {v}")

    rows: dict[str, dict] = {}
    for pest in args.pests:
        loaded, model, cfg = load_and_wrap(pest)
        X, t, vm = make_synthetic_input(loaded, seed=0, K=1)
        entry: dict = {
            "pest": pest, "d_in": loaded.d_in, "T": loaded.T,
            "elements": int(loaded.T * loaded.d_in),
            "ckpt_bytes": loaded.ckpt_path.stat().st_size,
            "runs": {"pytorch": [], **{v: [] for v in VARIANTS}},
            "sizes": {},
        }
        interps = {}
        for v in VARIANTS:
            p = ARTIFACTS / pest / f"{pest}_stage2_{v}.tflite"
            interps[v] = _interp(p) if p.is_file() else None
            entry["sizes"][v] = p.stat().st_size if p.is_file() else None

        for _ in range(args.repeats):
            with torch.no_grad():
                entry["runs"]["pytorch"].append(
                    time_it(lambda: model(X, t, vm), args.iters, args.warmup)["mean_ms"]
                )
            for v in VARIANTS:
                if interps[v] is None:
                    continue
                entry["runs"][v].append(
                    time_it(lambda it=interps[v]: tflite_mu(it, X, t, vm),
                            args.iters, args.warmup)["mean_ms"]
                )
        rows[pest] = entry

    print(f"\n=== latency (mean ms, median of {args.repeats} repeats) ===")
    print(f"{'pest':<19}{'TxD':>9}{'elems':>7}{'torch':>8}{'fp32':>8}{'fp16':>8}"
          f"{'fp32_x':>8}{'fp16_x':>8}{'fp32_B':>10}{'fp16_B':>10}")
    for pest in args.pests:
        e = rows[pest]
        med = {k: (statistics.median(v) if v else float("nan")) for k, v in e["runs"].items()}
        e["median_ms"] = med
        e["speedup"] = {v: (med["pytorch"] / med[v] if med.get(v) else None) for v in VARIANTS}
        dims = "{}x{}".format(e["T"], e["d_in"])
        print(f"{pest:<19}{dims:>9}{e['elements']:>7}"
              f"{med['pytorch']:>8.3f}{med['fp32']:>8.3f}{med['fp16']:>8.3f}"
              f"{e['speedup']['fp32']:>7.2f}x{e['speedup']['fp16']:>7.2f}x"
              f"{e['sizes']['fp32']:>10,}{e['sizes']['fp16']:>10,}")

    print("\nspread across repeats (min-max mean_ms):")
    for pest in args.pests:
        e = rows[pest]
        f = lambda k: (f"{min(e['runs'][k]):.3f}-{max(e['runs'][k]):.3f}" if e["runs"][k] else "-")
        print(f"  {pest:<19} torch {f('pytorch'):>15}   fp32 {f('fp32'):>15}   fp16 {f('fp16'):>15}")

    print("\nNotes:")
    print("  * BPH is 131x27 = 3,537 input elements; the other 7 are 241x45 = 10,845")
    print("    (3.07x more), which is why BPH is markedly faster.")
    print("  * Absolute ms drifts with machine load; the ratio is the stable figure.")

    (ARTIFACTS / "benchmark_all.json").write_text(
        json.dumps({"env": env, "rows": rows}, indent=2)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
