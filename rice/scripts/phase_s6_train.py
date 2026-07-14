"""Phase S6 — 2-sided model with target_offset = ε (interval mid-target).

Single-knob change vs. the existing 2-sided ckpt:
    target_offset: 0.0 → ε  (ε defaults to 7.5 = D/2 for D=15)

Everything else (asym_weight=25, asym_weight_early=5, target_early_offset=30,
σ=5, target_mode="l_offset", warm-start chain) is held fixed so the comparison
isolates the target-location effect on IoU ceiling.

Stage A (uncond) is re-used from an existing ckpt. Stage B (pilot) and
Stage C (final) are re-trained with the new target_offset; the pilot is also
re-trained because mu warmup matters when the target shifts by ε.

Output ckpts (default naming, "tc" = target-center):
    pilot:  rice/outputs_stage2_..._2sided_tc{tag}_pilot/ckpt/checkpoint_run4.pt
    final:  rice/outputs_stage2_..._2sided_tc{tag}/ckpt/checkpoint_run4.pt
    tag = "7p5" for ε=7.5  ("p" stands for the decimal point).
"""
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PY = str(REPO_ROOT / ".venv" / "bin" / "python")


def _check_ckpt(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"[abort] {label} ckpt not found: {path}")


def _run_train(stage_label: str, cmd: list[str], log_path: Path) -> None:
    """Stream subprocess output line-by-line to both stdout and a log file.

    Uses Popen instead of run() so the user can watch live progress in tmux.
    Inserts `python -u` so the child run_train.py flushes prints per line.
    """
    import sys
    if cmd and cmd[1] == "-m":
        cmd = [cmd[0], "-u", *cmd[1:]]   # unbuffered python
    print(f"\n========== {stage_label} START ==========")
    print(" ".join(cmd))
    print(f"  log → {log_path}")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    env = {**os.environ,
           "PYTHONUNBUFFERED": "1",
           "PYTORCH_CUDA_ALLOC_CONF":
               os.environ.get("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")}
    with log_path.open("w") as logf:
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, bufsize=1, env=env,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            logf.write(line)
            logf.flush()
        rc = proc.wait()
    if rc != 0:
        raise SystemExit(f"[abort] {stage_label} failed (rc={rc}); see {log_path}")
    print(f"========== {stage_label} DONE ==========")


def _eps_tag(eps: float) -> str:
    """Convert a float ε into a filesystem-safe slug (7.5 → '7p5')."""
    s = f"{eps:g}".replace(".", "p").replace("-", "neg")
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", type=str, default="sheath_blight")
    ap.add_argument("--run", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--uncond_ckpt", type=str,
                    default="rice/outputs_stage2_sheath_blight_d15_asym25_2sided_uncond/ckpt/checkpoint_run4.pt",
                    help="Reused Stage-A uncond hazard checkpoint (no retraining).")
    ap.add_argument("--target_offset", type=float, default=7.5,
                    help="ε for target = L + ε. 7.5 = D/2 for D=15 windows.")
    ap.add_argument("--out_pilot", type=str, default=None,
                    help="Override pilot out_root. Default derives from --target_offset.")
    ap.add_argument("--out_final", type=str, default=None,
                    help="Override final out_root. Default derives from --target_offset.")
    ap.add_argument("--skip_pilot", action="store_true",
                    help="Skip Stage-B and use --out_pilot as warm-start for Stage-C.")
    ap.add_argument("--skip_final", action="store_true",
                    help="Train only Stage-B (debug).")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--amp_dtype", type=str, default="bf16")
    ap.add_argument("--d_model_override", type=int, default=48)
    ap.add_argument("--max_epochs_override", type=int, default=None,
                    help="Cap epochs (debug). Default preserves the original D=15 budget.")
    args = ap.parse_args()

    tag = _eps_tag(args.target_offset)
    out_pilot = args.out_pilot or f"rice/outputs_stage2_{args.pest}_d15_asym25_2sided_tc{tag}_pilot"
    out_final = args.out_final or f"rice/outputs_stage2_{args.pest}_d15_asym25_2sided_tc{tag}"
    pilot_ckpt = f"{out_pilot}/ckpt/checkpoint_run{args.run}.pt"
    final_ckpt = f"{out_final}/ckpt/checkpoint_run{args.run}.pt"

    print("=" * 70)
    print(f"Phase S6 train  seed={args.seed}  target_offset(ε)={args.target_offset}")
    print(f"  uncond ckpt   : {args.uncond_ckpt}")
    print(f"  pilot out_root: {out_pilot}")
    print(f"  final out_root: {out_final}")
    print("=" * 70)

    _check_ckpt(args.uncond_ckpt, "uncond")

    # ---------- common training args shared between Stage B and Stage C -------
    # Mirrors the existing 2-sided final hparams.json exactly except for
    # --stage2_pmf_target_offset which we sweep here.
    common = [
        PY, "-m", "rice.scripts.run_train",
        "--pest", args.pest,
        "--run", str(args.run),
        "--seeds", str(args.seed),
        "--split_seed", str(args.split_seed),
        "--split_mode", "year",
        "--val_year", str(args.val_year),
        "--test_year_min", str(args.test_year_min),
        "--test_year_max", str(args.test_year_max),
        "--dropout", "0.2",
        "--weight_decay", "0.0001",
        "--lr", "0.0001",
        "--w_interval", "1.0",
        "--w_left", "0.5",
        "--w_right", "0.5",
        "--stage2_nowcast",
        "--stage2_nowcast_window", "28",
        "--stage2_nowcast_stride", "1",
        "--stage2_nowcast_only_pre_event", "1",
        "--stage2_nowcast_event_time_proxy", "r",
        "--stage2_nowcast_require_tstar_before_L", "0",
        "--stage2_causal_tstar",
        "--stage2_tstar_layers", "1",
        "--stage2_use_tstar_scalar_pos", "0",
        "--stage2_early_tstar_weight_min", "0.2",
        "--stage2_site_year_mean_loss", "0",
        "--stage2_time_chunk_size", "64",
        "--stage2_conditional_survival", "0",
        "--stage2_pmf_mode", "gaussian",
        "--stage2_pmf_sigma", "5.0",
        "--stage2_pmf_right_weight", "0.3",
        "--stage2_pmf_target_mode", "l_offset",
        "--stage2_pmf_target_offset", str(args.target_offset),
        "--stage2_pmf_target_early_offset", "30.0",
        "--stage2_best_metric", "val_iou80",
        "--amp", str(args.amp),
        "--amp_dtype", str(args.amp_dtype),
        "--d_model_override", str(args.d_model_override),
    ]
    if args.max_epochs_override is not None:
        common += ["--max_epochs_override", str(args.max_epochs_override)]

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Stage B (pilot) ------------------------------------------------
    if not args.skip_pilot:
        pilot_cmd = list(common) + [
            "--out_root", out_pilot,
            "--stage2_warm_start_ckpt", args.uncond_ckpt,
            "--stage2_warm_start_seed", str(args.seed),
            "--stage2_pmf_asym_weight", "15.0",
            "--stage2_pmf_asym_weight_early", "0.0",
        ]
        _run_train(f"Stage B (pilot, asym=15, ε={args.target_offset})",
                    pilot_cmd,
                    log_dir / f"phase_s6_train_pilot_tc{tag}.log")
        _check_ckpt(pilot_ckpt, "pilot")
    else:
        print(f"[skip_pilot] re-using existing pilot ckpt: {pilot_ckpt}")
        _check_ckpt(pilot_ckpt, "pilot (pre-existing)")

    # ---------- Stage C (final) ------------------------------------------------
    if not args.skip_final:
        final_cmd = list(common) + [
            "--out_root", out_final,
            "--stage2_warm_start_ckpt", pilot_ckpt,
            "--stage2_warm_start_seed", str(args.seed),
            "--stage2_pmf_asym_weight", "25.0",
            "--stage2_pmf_asym_weight_early", "5.0",
        ]
        _run_train(f"Stage C (final, asym=25, ε={args.target_offset})",
                    final_cmd,
                    log_dir / f"phase_s6_train_final_tc{tag}.log")
        _check_ckpt(final_ckpt, "final")
    else:
        print("[skip_final] stage C skipped")

    print("\n" + "=" * 70)
    print(f"Phase S6 train DONE  (ε={args.target_offset})")
    print(f"  pilot : {pilot_ckpt}")
    print(f"  final : {final_ckpt}   ← use this for inference / eval")
    print("=" * 70)


if __name__ == "__main__":
    main()
