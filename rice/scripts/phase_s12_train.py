"""Phase S12 — GDD feature ablation: run=4 → run=8 (+ GDD10_since_gs).

run=8 (added in phases/sheath_blight/features.py): 15 → 16 features.
The only difference is the addition of GDD10_since_gs, a cumulative
thermal-time signal that may be the missing driver behind the mu drift
observed in S6-S10.

Stage 1 must be retrained too: its XGBoost model expects exactly the
features in its training feature_cols. Mixing a run=4 Stage 1 with a
run=8 Stage 2 dataset would cause a feature-dimension mismatch in
phase_r.  We therefore retrain ALL 4 stages from scratch:

    Stage 1 (alert XGB) : --run 8                  ~30 min
    Stage 2 Uncond      : pmf_mode=hazard          ~30-40 min
    Stage 2 Pilot       : warm-start from uncond   ~35 min
    Stage 2 Final       : warm-start from pilot    ~20-30 min
    -----------------------------------------------------
    Total                                          ~2-2.5 h

Stage 1 cohort may shift with GDD; this is part of the ablation.  Report
the (sample_id) cohort delta vs the existing baseline Stage 1.

Output ckpts:
    Stage 1 : rice/outputs_stage1/sheath_blight_yearsplit2023-24_gdd/ckpt/event_classifier_run8.pt
    Uncond  : rice/outputs_stage2_..._2sided_gdd_uncond/ckpt/checkpoint_run4.pt
    Pilot   : rice/outputs_stage2_..._2sided_gdd_pilot/ckpt/checkpoint_run4.pt
    Final   : rice/outputs_stage2_..._2sided_gdd/ckpt/checkpoint_run4.pt
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


def _run_subprocess(stage_label: str, cmd: list[str], log_path: Path) -> None:
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", type=str, default="sheath_blight")
    ap.add_argument("--run", type=int, default=8,
                    help="Feature run number (must include GDD; default 8).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)

    # Stage 1 hparams (mirror existing run=4 ckpt; verified via torch.load).
    ap.add_argument("--stage1_out_root", type=str,
                    default=None,
                    help="Stage 1 out_root. Default derives from --doy_start_override "
                         "(adds _doy{X} suffix when override != 60).")
    ap.add_argument("--stage1_out_path", type=str,
                    default=None,
                    help="Explicit Stage 1 ckpt path. Default: "
                         "<stage1_out_root>/ckpt/event_classifier_run{run}.pt")
    ap.add_argument("--skip_stage1", action="store_true",
                    help="Skip Stage 1 retraining; use --stage1_ckpt_override.")
    ap.add_argument("--stage1_ckpt_override", type=str, default=None,
                    help="Explicit Stage 1 ckpt path to reuse when --skip_stage1.")

    # Stage 2 hparams (mirror baseline 2-sided final).
    ap.add_argument("--out_uncond", type=str, default=None)
    ap.add_argument("--out_pilot", type=str, default=None)
    ap.add_argument("--out_final", type=str, default=None)
    ap.add_argument("--skip_uncond", action="store_true")
    ap.add_argument("--skip_pilot", action="store_true")
    ap.add_argument("--skip_final", action="store_true")
    ap.add_argument("--uncond_ckpt_override", type=str, default=None)
    ap.add_argument("--pilot_ckpt_override", type=str, default=None)

    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--amp_dtype", type=str, default="bf16")
    ap.add_argument("--d_model_override", type=int, default=48)
    ap.add_argument("--max_epochs_override", type=int, default=None)
    ap.add_argument("--doy_start_override", type=int, default=None,
                    help="Override DOY range start. None = use pest default. "
                         "Pass 1 to match the baseline 2-sided Stage 2 frame "
                         "(Phase S12b setup).")
    ap.add_argument("--doy_end_override", type=int, default=None,
                    help="Override DOY range end. Defaults to 300 if "
                         "doy_start_override is set, else pest default.")
    args = ap.parse_args()

    # Tag includes "doy{X}" only when doy_start_override is set, so the legacy
    # pest-default runs (doy_start=60) keep the existing ckpt path layout.
    if args.doy_start_override is not None and args.doy_start_override != 60:
        tag_doy_suffix = f"_doy{int(args.doy_start_override)}"
    else:
        tag_doy_suffix = ""
    tag = "gdd"
    out_uncond = args.out_uncond or f"rice/outputs_stage2_{args.pest}_d15_asym25_2sided_{tag}{tag_doy_suffix}_uncond"
    out_pilot = args.out_pilot or f"rice/outputs_stage2_{args.pest}_d15_asym25_2sided_{tag}{tag_doy_suffix}_pilot"
    out_final = args.out_final or f"rice/outputs_stage2_{args.pest}_d15_asym25_2sided_{tag}{tag_doy_suffix}"
    # run_train.resolve_out_path uses `checkpoint_run{run}.pt`; mirror that
    # so the warm-start chain finds the previous-stage ckpt under the same run.
    uncond_ckpt = args.uncond_ckpt_override or f"{out_uncond}/ckpt/checkpoint_run{args.run}.pt"
    pilot_ckpt = args.pilot_ckpt_override or f"{out_pilot}/ckpt/checkpoint_run{args.run}.pt"
    final_ckpt = f"{out_final}/ckpt/checkpoint_run{args.run}.pt"

    stage1_out_root = (args.stage1_out_root
                       or f"rice/outputs_stage1/{args.pest}_yearsplit2023-24_gdd{tag_doy_suffix}")
    stage1_out_path = (args.stage1_out_path
                       or f"{stage1_out_root}/ckpt/event_classifier_run{args.run}.pt")
    stage1_ckpt = args.stage1_ckpt_override if args.skip_stage1 else stage1_out_path

    print("=" * 70, flush=True)
    print(f"Phase S12 train  run={args.run}  seed={args.seed}", flush=True)
    print(f"  stage1 ckpt   : {stage1_ckpt}", flush=True)
    print(f"  uncond  root  : {out_uncond}", flush=True)
    print(f"  pilot   root  : {out_pilot}", flush=True)
    print(f"  final   root  : {out_final}", flush=True)
    print(f"  final ckpt    : {final_ckpt}", flush=True)
    print("=" * 70, flush=True)

    log_dir = REPO_ROOT / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ---------- Stage 1: XGBoost alert classifier ----------------------------
    if not args.skip_stage1:
        s1_cmd = [
            PY, "-m", "rice.scripts.run_event_train",
            "--pest", args.pest, "--run", str(args.run),
            "--out_root", stage1_out_root,
            "--out", stage1_out_path,
            "--seeds", str(args.seed),
            "--split_seed", str(args.split_seed),
            "--split_mode", "year",
            "--val_year", str(args.val_year),
            "--test_year_min", str(args.test_year_min),
            "--test_year_max", str(args.test_year_max),
            "--model", "xgb",
            "--task_mode", "nowcast",
            "--nowcast_window", "28",
            "--nowcast_stride", "1",
            "--nowcast_only_pre_event", "1",
            "--nowcast_event_time_proxy", "mid",
            "--nowcast_label_mode", "eventually",
            "--add_tstar_position_feature",
            "--doy_start_override", str(args.doy_start_override
                                          if args.doy_start_override is not None else 60),
            "--doy_end_override", str(args.doy_end_override
                                        if args.doy_end_override is not None else 300),
        ]
        _run_subprocess(f"Stage 1 (XGB alert, run={args.run} incl. GDD, "
                          f"doy_start={args.doy_start_override or 60})",
                         s1_cmd, log_dir / f"phase_s12_train_stage1_run{args.run}{tag_doy_suffix}.log")
        _check_ckpt(stage1_ckpt, "stage1")
    else:
        print(f"[skip_stage1] using ckpt: {stage1_ckpt}", flush=True)
        _check_ckpt(stage1_ckpt, "stage1 (pre-existing)")

    # ---------- Stage 2: common args ----------------------------------------
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
        "--stage2_pmf_sigma", "5.0",
        "--stage2_pmf_right_weight", "0.3",
        "--stage2_pmf_target_mode", "l_offset",
        "--stage2_pmf_target_offset", "0.0",
        "--stage2_pmf_target_early_offset", "30.0",
        "--stage2_best_metric", "val_iou80",
        "--amp", str(args.amp),
        "--amp_dtype", str(args.amp_dtype),
        "--d_model_override", str(args.d_model_override),
    ]
    if args.max_epochs_override is not None:
        common += ["--max_epochs_override", str(args.max_epochs_override)]
    if args.doy_start_override is not None:
        common += ["--doy_start_override", str(args.doy_start_override)]
    if args.doy_end_override is not None:
        common += ["--doy_end_override", str(args.doy_end_override)]

    # ---------- Stage 2 Uncond (hazard mode, no warm-start) -----------------
    if not args.skip_uncond:
        uncond_cmd = list(common) + [
            "--out_root", out_uncond,
            "--stage2_pmf_mode", "hazard",
            "--stage2_pmf_asym_weight", "10.0",
            "--stage2_pmf_asym_weight_early", "0.0",
        ]
        _run_subprocess("Stage 2 Uncond (hazard)",
                         uncond_cmd, log_dir / f"phase_s12_train_uncond_{tag}{tag_doy_suffix}.log")
        _check_ckpt(uncond_ckpt, "uncond")
    else:
        print(f"[skip_uncond] using ckpt: {uncond_ckpt}", flush=True)
        _check_ckpt(uncond_ckpt, "uncond (pre-existing)")

    # ---------- Stage 2 Pilot (gaussian, asym=15/0) -------------------------
    # Baseline 2-sided 3-stage convention: pilot uses asym=15 / asym_early=0,
    # final uses asym=25 / asym_early=5.  Phase S12 keeps the convention so
    # the only variable changed vs. baseline is the GDD feature.
    if not args.skip_pilot:
        pilot_cmd = list(common) + [
            "--out_root", out_pilot,
            "--stage2_pmf_mode", "gaussian",
            "--stage2_warm_start_ckpt", uncond_ckpt,
            "--stage2_warm_start_seed", str(args.seed),
            "--stage2_pmf_asym_weight", "15.0",
            "--stage2_pmf_asym_weight_early", "0.0",
        ]
        _run_subprocess("Stage 2 Pilot (gaussian, asym=15/0)",
                         pilot_cmd, log_dir / f"phase_s12_train_pilot_{tag}{tag_doy_suffix}.log")
        _check_ckpt(pilot_ckpt, "pilot")
    else:
        print(f"[skip_pilot] using ckpt: {pilot_ckpt}", flush=True)
        _check_ckpt(pilot_ckpt, "pilot (pre-existing)")

    # ---------- Stage 2 Final (gaussian, asym=25/5) -------------------------
    if not args.skip_final:
        final_cmd = list(common) + [
            "--out_root", out_final,
            "--stage2_pmf_mode", "gaussian",
            "--stage2_warm_start_ckpt", pilot_ckpt,
            "--stage2_warm_start_seed", str(args.seed),
            "--stage2_pmf_asym_weight", "25.0",
            "--stage2_pmf_asym_weight_early", "5.0",
        ]
        _run_subprocess("Stage 2 Final (gaussian, asym=25/5)",
                         final_cmd, log_dir / f"phase_s12_train_final_{tag}{tag_doy_suffix}.log")
        _check_ckpt(final_ckpt, "final")
    else:
        print("[skip_final] stage final skipped", flush=True)

    print("\n" + "=" * 70, flush=True)
    print(f"Phase S12 train DONE  (tag={tag}, run={args.run})", flush=True)
    print(f"  stage1 : {stage1_ckpt}", flush=True)
    print(f"  uncond : {uncond_ckpt}", flush=True)
    print(f"  pilot  : {pilot_ckpt}", flush=True)
    print(f"  final  : {final_ckpt}   ← use this for inference / eval", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
