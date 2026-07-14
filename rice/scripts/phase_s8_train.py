"""Phase S8 — target_mode="center" diagnostic.

Hypothesis being tested:
    The S6/S7 finding (mu stuck at L-20 regardless of asym strength) is caused
    by the asymmetric mu-loss formulation itself.  Switching to target_mode=
    "center" uses target = (L+R)/2 with a pure symmetric MSE, bypassing the
    asym_weight / asym_weight_early machinery entirely (both are ignored in
    'center' mode per train_eval.py:323-326,351).

Single-knob change vs. the baseline 2-sided:
    target_mode : "l_offset" → "center"      (CLI --target_mode)
    zone_* knobs : all 0                     (clean diagnostic, no soft asymmetry)
Everything else (σ=5, right_weight=0.3, 3-stage warm-start chain) is held
fixed.  Note that target_offset / target_early_offset / asym_weight /
asym_weight_early are silently ignored under target_mode="center", so the
pilot vs. final stage asym distinction only affects code paths that look at
those attributes (e.g., diagnostic prints) — the actual mu-loss is identical
between pilot and final.

Stage A (uncond) is re-used.  Stage B (pilot) and Stage C (final) are both
re-trained from scratch (the existing pilot ckpt was trained under l_offset
mode and is not directly compatible).

Output ckpts (default naming):
    pilot:  rice/outputs_stage2_..._2sided_center_pilot/ckpt/checkpoint_run4.pt
    final:  rice/outputs_stage2_..._2sided_center/ckpt/checkpoint_run4.pt
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
        cmd = [cmd[0], "-u", *cmd[1:]]   # unbuffered python
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
                    choices=["l_offset", "center"],
                    help="Target formulation. 'center' uses target=(L+R)/2 with "
                         "symmetric MSE; asym_weight/asym_weight_early/target_offset "
                         "are silently ignored.")
    ap.add_argument("--zone_late_weight", type=float, default=0.0,
                    help="Soft penalty for mu > mid (center mode). 0 = pure symmetric MSE.")
    ap.add_argument("--zone_too_late_weight", type=float, default=0.0,
                    help="Soft penalty for mu > L + zone_too_late_threshold.")
    ap.add_argument("--zone_missed_weight", type=float, default=0.0,
                    help="Soft penalty for mu > L + zone_missed_threshold (MISSED).")
    ap.add_argument("--zone_too_early_weight", type=float, default=0.0,
                    help="Soft penalty for mu < L - zone_too_early_threshold (TOO_EARLY).")
    ap.add_argument("--zone_too_late_threshold", type=float, default=15.0)
    ap.add_argument("--zone_missed_threshold", type=float, default=22.0)
    ap.add_argument("--zone_too_early_threshold", type=float, default=23.0)
    ap.add_argument("--out_pilot", type=str, default=None,
                    help="Override pilot out_root. Default derives from --target_mode.")
    ap.add_argument("--out_final", type=str, default=None,
                    help="Override final out_root. Default derives from --target_mode.")
    ap.add_argument("--skip_pilot", action="store_true",
                    help="Skip Stage-B and use --out_pilot / --pilot_ckpt as warm-start for Stage-C.")
    ap.add_argument("--pilot_ckpt", type=str, default=None,
                    help="Explicit pilot ckpt to use as warm-start when --skip_pilot is set. "
                         "Defaults to <out_pilot>/ckpt/checkpoint_run{run}.pt.")
    ap.add_argument("--skip_final", action="store_true",
                    help="Train only Stage-B (debug).")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--amp_dtype", type=str, default="bf16")
    ap.add_argument("--d_model_override", type=int, default=48)
    ap.add_argument("--max_epochs_override", type=int, default=None,
                    help="Cap epochs (debug). Default preserves the original D=15 budget.")
    args = ap.parse_args()

    # Tag includes target_mode and a digest of zone-knob activations for clean
    # ckpt naming when zones are turned on later.
    nz = [k for k, v in (
        ("late", args.zone_late_weight),
        ("tl", args.zone_too_late_weight),
        ("ms", args.zone_missed_weight),
        ("te", args.zone_too_early_weight),
    ) if float(v) > 0.0]
    tag = "center"
    if nz:
        tag = "center_" + "_".join(nz)
    out_pilot = args.out_pilot or f"rice/outputs_stage2_{args.pest}_d15_asym25_2sided_{tag}_pilot"
    out_final = args.out_final or f"rice/outputs_stage2_{args.pest}_d15_asym25_2sided_{tag}"
    pilot_ckpt = (args.pilot_ckpt
                  or f"{out_pilot}/ckpt/checkpoint_run{args.run}.pt")
    final_ckpt = f"{out_final}/ckpt/checkpoint_run{args.run}.pt"

    print("=" * 70, flush=True)
    print(f"Phase S8 train  seed={args.seed}  target_mode={args.target_mode}  "
          f"zones={'/'.join(nz) if nz else 'none (pure center MSE)'}", flush=True)
    print(f"  uncond ckpt   : {args.uncond_ckpt}", flush=True)
    print(f"  pilot out_root: {out_pilot}", flush=True)
    print(f"  final out_root: {out_final}", flush=True)
    print(f"  final ckpt    : {final_ckpt}", flush=True)
    print("=" * 70, flush=True)

    _check_ckpt(args.uncond_ckpt, "uncond")

    # ---------- common training args shared between Stage B and Stage C -------
    # Mirrors the existing 2-sided final hparams.json with only target_mode +
    # zone_* changed.  asym_weight / asym_weight_early / target_offset /
    # target_early_offset are still passed because run_train.py requires them,
    # but the loss ignores all four under target_mode="center".
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
        "--stage2_pmf_target_mode", args.target_mode,
        "--stage2_pmf_target_offset", "0.0",          # ignored under center
        "--stage2_pmf_target_early_offset", "30.0",   # ignored under center
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

    # ---------- Stage B (pilot) ------------------------------------------------
    # asym_weight values here are ignored by the loss (center mode), but the
    # training script still requires them.  We keep the legacy 3-stage values
    # (pilot=15, final=25) so the only effective hparam transition between
    # stages is the warm-start ckpt chain.
    if not args.skip_pilot:
        pilot_cmd = list(common) + [
            "--out_root", out_pilot,
            "--stage2_warm_start_ckpt", args.uncond_ckpt,
            "--stage2_warm_start_seed", str(args.seed),
            "--stage2_pmf_asym_weight", "15.0",
            "--stage2_pmf_asym_weight_early", "0.0",
        ]
        _run_train(f"Stage B (pilot, target_mode={args.target_mode})",
                    pilot_cmd, log_dir / f"phase_s8_train_pilot_{tag}.log")
        _check_ckpt(pilot_ckpt, "pilot")
    else:
        print(f"[skip_pilot] re-using pilot ckpt: {pilot_ckpt}", flush=True)
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
        _run_train(f"Stage C (final, target_mode={args.target_mode})",
                    final_cmd, log_dir / f"phase_s8_train_final_{tag}.log")
        _check_ckpt(final_ckpt, "final")
    else:
        print("[skip_final] stage C skipped", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"Phase S8 train DONE  (target_mode={args.target_mode}, tag={tag})", flush=True)
    print(f"  pilot : {pilot_ckpt}", flush=True)
    print(f"  final : {final_ckpt}   ← use this for inference / eval", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
