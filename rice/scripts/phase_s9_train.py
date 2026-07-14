"""Phase S9 — center mode + right_weight = 0 diagnostic.

Hypothesis being tested:
    Phase S8 (target_mode='center', pure symmetric MSE) showed mu landing
    well after mid (mu − mid ≈ +39).  The most likely remaining culprit is
    the right-censored term: `loss_right = right_weight * mean((mu − Tend)^2)`
    pulls mu toward Tend (=300, late in the season) for right-cens samples.
    Even though right-cens samples don't share gradients with interval samples
    at the sample level, they DO share the head_mu parameters, so a heavy
    right_weight can bias the head's overall output distribution late.

Single-knob change vs. Phase S8:
    right_weight : 0.3 → 0.0    (CLI --right_weight)

Everything else (target_mode='center', zone_*=0, σ=5, 3-stage chain) is held
fixed.  Stage A (uncond) reuse, pilot + final fresh.

Output ckpts:
    pilot:  rice/outputs_stage2_..._2sided_center_rw0_pilot/ckpt/checkpoint_run4.pt
    final:  rice/outputs_stage2_..._2sided_center_rw0/ckpt/checkpoint_run4.pt
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
PY = str(REPO_ROOT / ".venv" / "bin" / "python")


def _check_ckpt(path: str, label: str) -> None:
    if not os.path.exists(path):
        raise SystemExit(f"[abort] {label} ckpt not found: {path}")


def _run_train(stage_label: str, cmd: list[str], log_path: Path) -> None:
    """Stream subprocess output line-by-line to both stdout and a log file."""
    if cmd and cmd[1] == "-m":
        cmd = [cmd[0], "-u", *cmd[1:]]
    print(f"\n========== {stage_label} START ==========", flush=True)
    print(" ".join(cmd), flush=True)
    print(f"  log → {log_path}", flush=True)
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
    print(f"========== {stage_label} DONE ==========", flush=True)


def _slug(x: float) -> str:
    return f"{x:g}".replace(".", "p")


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
    ap.add_argument("--target_mode", type=str, default="center",
                    choices=["l_offset", "center"])
    ap.add_argument("--right_weight", type=float, default=0.0,
                    help="Weight on right-cens MSE (mu − Tend)^2. 0.0 disables "
                         "the right-cens pull-toward-Tend force (Phase S9 knob).")
    ap.add_argument("--zone_late_weight", type=float, default=0.0)
    ap.add_argument("--zone_too_late_weight", type=float, default=0.0)
    ap.add_argument("--zone_missed_weight", type=float, default=0.0)
    ap.add_argument("--zone_too_early_weight", type=float, default=0.0)
    ap.add_argument("--zone_too_late_threshold", type=float, default=15.0)
    ap.add_argument("--zone_missed_threshold", type=float, default=22.0)
    ap.add_argument("--zone_too_early_threshold", type=float, default=23.0)
    ap.add_argument("--out_pilot", type=str, default=None)
    ap.add_argument("--out_final", type=str, default=None)
    ap.add_argument("--skip_pilot", action="store_true")
    ap.add_argument("--pilot_ckpt", type=str, default=None)
    ap.add_argument("--skip_final", action="store_true")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--amp_dtype", type=str, default="bf16")
    ap.add_argument("--d_model_override", type=int, default=48)
    ap.add_argument("--max_epochs_override", type=int, default=None)
    args = ap.parse_args()

    # Tag carries the right_weight value for clean ckpt naming.
    rw_tag = f"rw{_slug(args.right_weight)}"
    base_tag = "center" if args.target_mode == "center" else "lofs"
    nz = [k for k, v in (
        ("late", args.zone_late_weight),
        ("tl", args.zone_too_late_weight),
        ("ms", args.zone_missed_weight),
        ("te", args.zone_too_early_weight),
    ) if float(v) > 0.0]
    tag = f"{base_tag}_{rw_tag}"
    if nz:
        tag += "_" + "_".join(nz)
    out_pilot = args.out_pilot or f"rice/outputs_stage2_{args.pest}_d15_asym25_2sided_{tag}_pilot"
    out_final = args.out_final or f"rice/outputs_stage2_{args.pest}_d15_asym25_2sided_{tag}"
    pilot_ckpt = (args.pilot_ckpt
                  or f"{out_pilot}/ckpt/checkpoint_run{args.run}.pt")
    final_ckpt = f"{out_final}/ckpt/checkpoint_run{args.run}.pt"

    print("=" * 70, flush=True)
    print(f"Phase S9 train  seed={args.seed}  target_mode={args.target_mode}  "
          f"right_weight={args.right_weight}  "
          f"zones={'/'.join(nz) if nz else 'none'}", flush=True)
    print(f"  uncond ckpt   : {args.uncond_ckpt}", flush=True)
    print(f"  pilot out_root: {out_pilot}", flush=True)
    print(f"  final out_root: {out_final}", flush=True)
    print(f"  final ckpt    : {final_ckpt}", flush=True)
    print("=" * 70, flush=True)

    _check_ckpt(args.uncond_ckpt, "uncond")

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
        "--stage2_pmf_right_weight", str(args.right_weight),
        "--stage2_pmf_target_mode", args.target_mode,
        "--stage2_pmf_target_offset", "0.0",
        "--stage2_pmf_target_early_offset", "30.0",
        "--stage2_pmf_zone_late_weight", str(args.zone_late_weight),
        "--stage2_pmf_zone_too_late_weight", str(args.zone_too_late_weight),
        "--stage2_pmf_zone_missed_weight", str(args.zone_missed_weight),
        "--stage2_pmf_zone_too_early_weight", str(args.zone_too_early_weight),
        "--stage2_pmf_zone_too_late_threshold", str(args.zone_too_late_threshold),
        "--stage2_pmf_zone_missed_threshold", str(args.zone_missed_threshold),
        "--stage2_pmf_zone_too_early_threshold", str(args.zone_too_early_threshold),
        "--stage2_best_metric", "val_iou80",
        "--amp", str(args.amp),
        "--amp_dtype", str(args.amp_dtype),
        "--d_model_override", str(args.d_model_override),
    ]
    if args.max_epochs_override is not None:
        common += ["--max_epochs_override", str(args.max_epochs_override)]

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    if not args.skip_pilot:
        pilot_cmd = list(common) + [
            "--out_root", out_pilot,
            "--stage2_warm_start_ckpt", args.uncond_ckpt,
            "--stage2_warm_start_seed", str(args.seed),
            "--stage2_pmf_asym_weight", "15.0",
            "--stage2_pmf_asym_weight_early", "0.0",
        ]
        _run_train(f"Stage B (pilot, target_mode={args.target_mode}, "
                    f"right_weight={args.right_weight})",
                    pilot_cmd, log_dir / f"phase_s9_train_pilot_{tag}.log")
        _check_ckpt(pilot_ckpt, "pilot")
    else:
        print(f"[skip_pilot] re-using pilot ckpt: {pilot_ckpt}", flush=True)
        _check_ckpt(pilot_ckpt, "pilot (pre-existing)")

    if not args.skip_final:
        final_cmd = list(common) + [
            "--out_root", out_final,
            "--stage2_warm_start_ckpt", pilot_ckpt,
            "--stage2_warm_start_seed", str(args.seed),
            "--stage2_pmf_asym_weight", "25.0",
            "--stage2_pmf_asym_weight_early", "5.0",
        ]
        _run_train(f"Stage C (final, target_mode={args.target_mode}, "
                    f"right_weight={args.right_weight})",
                    final_cmd, log_dir / f"phase_s9_train_final_{tag}.log")
        _check_ckpt(final_ckpt, "final")
    else:
        print("[skip_final] stage C skipped", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"Phase S9 train DONE  (tag={tag})", flush=True)
    print(f"  pilot : {pilot_ckpt}", flush=True)
    print(f"  final : {final_ckpt}   ← use this for inference / eval", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
