"""Phase S5 — 2-sided model + long-lead per-sample weighting.

Drives two sequential `run_train` invocations (subprocess) for the 2-sided
ckpt family.  Stage A (uncond) is re-used from an existing ckpt.

    Stage B (pilot) : warm-start from uncond ckpt;
                      pmf_mode=gaussian, asym_weight=15;
                      sample_weight applied (long-lead bias).
    Stage C (final) : warm-start from pilot ckpt (this run);
                      asym_weight=25, asym_weight_early=5, target_early_offset=30;
                      sample_weight applied (long-lead bias).

Long-lead weight scheme (binary, sample-level):
    weight = 1.0                   if (L + 1) - t* < threshold
    weight = long_lead_weight      otherwise

Output ckpts (default naming):
    pilot:  rice/outputs_stage2_sheath_blight_d15_asym25_2sided_lw{W}_pilot/ckpt/checkpoint_run4.pt
    final:  rice/outputs_stage2_sheath_blight_d15_asym25_2sided_lw{W}/ckpt/checkpoint_run4.pt
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
    ap.add_argument("--long_lead_threshold", type=float, default=60.0,
                    help="Lead (days) at/above which sample weight switches to long-lead.")
    ap.add_argument("--long_lead_weight", type=float, default=3.0,
                    help="Multiplier on mu-loss for long-lead samples (1.0 = no-op).")
    ap.add_argument("--out_pilot", type=str, default=None,
                    help="Override pilot out_root. Default derives from W.")
    ap.add_argument("--out_final", type=str, default=None,
                    help="Override final out_root. Default derives from W.")
    ap.add_argument("--skip_pilot", action="store_true",
                    help="Skip Stage-B and use --out_pilot as warm-start for Stage-C.")
    ap.add_argument("--skip_final", action="store_true",
                    help="Train only Stage-B (debug).")
    ap.add_argument("--amp", type=int, default=1)
    ap.add_argument("--amp_dtype", type=str, default="bf16")
    ap.add_argument("--d_model_override", type=int, default=48)
    ap.add_argument("--max_epochs_override", type=int, default=None,
                    help="Cap epochs (debug). Defaults preserve the original D=15 budget.")
    ap.add_argument("--dispatch_feature_csv", type=str, default=None,
                    help="Stage-1 dispatch confidence-feature CSV produced by "
                         "build_dispatch_feature_table.py. Forwarded to both "
                         "pilot and final run_train invocations.")
    ap.add_argument("--dispatch_feature_mode", type=str,
                    default="causal", choices=["causal", "broadcast"])
    ap.add_argument("--dispatch_feature_missing_value", type=float, default=0.0)
    ap.add_argument("--cohort_dispatch_only", action="store_true",
                    help="Phase B: restrict training cohort to dispatch-alerted "
                         "site-years. Forwarded to both pilot and final.")
    ap.add_argument("--mu_mode", type=str, default="absolute",
                    choices=["absolute", "lead_from_alert", "residual_clim"],
                    help="Stage 2 mu head mode (Phase B). "
                         "residual_clim requires --clim_mid.")
    ap.add_argument("--lead_min", type=float, default=7.0)
    ap.add_argument("--lead_max", type=float, default=75.0)
    ap.add_argument("--clim_mid", type=float, default=None,
                    help="Per-pest climatology mean_mid (DOY units) for the "
                         "residual_clim mu head. Required when "
                         "--mu_mode=residual_clim. Forwarded as "
                         "--stage2_pmf_clim_mid.")
    ap.add_argument("--delta_max", type=float, default=60.0,
                    help="Half-range of tanh-bounded delta in residual_clim "
                         "mode (days). Default 60. Forwarded as "
                         "--stage2_pmf_delta_max.")
    ap.add_argument("--target_mode", type=str, default=None,
                    choices=[None, "l_offset", "center"],
                    help="Override the hardcoded --stage2_pmf_target_mode "
                         "('l_offset') used by both pilot and final stages. "
                         "'center' uses target = (L+R)/2 with symmetric MSE "
                         "(no asym_weight / no early floor) — recommended for "
                         "--mu_mode=residual_clim. None = keep legacy default.")
    ap.add_argument("--asym_weight_early", type=float, default=None,
                    help="Override the hardcoded --stage2_pmf_asym_weight_early "
                         "(5.0) used by the final stage. Set 0.0 to disable the "
                         "L-target_early_offset floor (which traps mu near L-30 "
                         "if mu starts above L). None = keep legacy default. "
                         "Note: target_mode=center already skips this term.")
    ap.add_argument("--reset_head_mu", action="store_true",
                    help="Phase B headreset alias: when set (and neither of the "
                         "stage-specific flags below is set), applies head_mu "
                         "reset to the PILOT stage only (uncond -> pilot). The "
                         "final stage keeps the pilot's lead head intact. To "
                         "override per-stage, use --reset_head_mu_pilot / "
                         "--reset_head_mu_final instead.")
    ap.add_argument("--reset_head_mu_pilot", action="store_true",
                    help="Force head_mu reset on the pilot warm-start "
                         "(uncond -> pilot). Default off; auto-on when "
                         "--reset_head_mu is set and --reset_head_mu_final is "
                         "not explicitly toggled.")
    ap.add_argument("--reset_head_mu_final", action="store_true",
                    help="Force head_mu reset on the final warm-start "
                         "(pilot -> final). Default OFF: keep the lead head "
                         "learned during pilot. Only set this if you intend to "
                         "wipe pilot's lead head and retrain from scratch in "
                         "the final stage.")
    ap.add_argument("--aux_lead_lambda", type=float, default=0.0,
                    help="Phase B aux lead loss weight. Forwarded to both "
                         "pilot and final via --stage2_aux_lead_lambda. "
                         "0 disables. Default 0.")
    ap.add_argument("--aux_lead_huber_delta", type=float, default=10.0,
                    help="Huber delta for aux lead loss (days). Default 10.")
    ap.add_argument("--require_tstar_before_L", type=int, default=1,
                    help="If 1 (default), drop event-row nowcasts with "
                         "tstar >= L (i.e. in_LR / post_R). Required by "
                         "conditional_survival to avoid 'event rows have "
                         "tstar >= L' assertion. Set to 0 only if you "
                         "specifically want in_LR rows kept.")
    ap.add_argument("--batch_train_override", type=int, default=None,
                    help="Override C.BATCH_TRAIN for low-memory runs.")
    ap.add_argument("--batch_eval_override", type=int, default=None,
                    help="Override C.BATCH_EVAL for low-memory runs.")
    ap.add_argument("--gaussian_loss_mode", type=str, default="asym_mse",
                    choices=["asym_mse", "interval_nll", "mixed"],
                    help="Stage 2 Gaussian PMF mu-head loss family. "
                         "asym_mse (default): legacy asymmetric_mu_loss "
                         "(regression-style). interval_nll: "
                         "gaussian_interval_nll_loss using -log P(L<T<=R). "
                         "mixed: asym_mse + lambda * interval_nll (lambda via "
                         "--gaussian_interval_lambda). Forwarded to BOTH pilot "
                         "and final via --stage2_gaussian_loss_mode.")
    ap.add_argument("--gaussian_interval_lambda", type=float, default=0.1,
                    help="Lambda weight for the interval_nll auxiliary term "
                         "when --gaussian_loss_mode=mixed. Default 0.1 "
                         "(mild auxiliary). Forwarded as "
                         "--stage2_gaussian_interval_lambda.")
    ap.add_argument("--gaussian_interval_continuity_correction", type=int, default=0,
                    choices=[0, 1],
                    help="Continuity-correction toggle for interval_nll. "
                         "0=raw L,R; 1=(L+0.5, R+0.5). Forwarded as "
                         "--stage2_gaussian_interval_continuity_correction.")
    ap.add_argument("--right_weight", type=float, default=None,
                    help="Override the hardcoded --stage2_pmf_right_weight (0.3) "
                         "in the common args. Set 0.0 for event-only fine-tuning "
                         "(right-cens rows contribute zero loss / gradient). "
                         "None = keep legacy default 0.3.")
    args = ap.parse_args()
    # Resolve per-stage reset policy:
    #   - if --reset_head_mu_pilot / --reset_head_mu_final are explicitly given,
    #     respect them as-is.
    #   - else if --reset_head_mu (alias) is set, route reset to PILOT only.
    #   - else: no reset anywhere.
    reset_pilot = bool(args.reset_head_mu_pilot or
                       (args.reset_head_mu and not args.reset_head_mu_final))
    reset_final = bool(args.reset_head_mu_final)
    print(f"[phase_s5] pilot reset_head_mu={reset_pilot}")
    print(f"[phase_s5] final reset_head_mu={reset_final}")

    # Default out paths keyed on the weight value (lw3, lw5, ...).
    w_tag = f"lw{int(round(args.long_lead_weight))}"
    out_pilot = args.out_pilot or f"rice/outputs_stage2_{args.pest}_d15_asym25_2sided_{w_tag}_pilot"
    out_final = args.out_final or f"rice/outputs_stage2_{args.pest}_d15_asym25_2sided_{w_tag}"
    pilot_ckpt = f"{out_pilot}/ckpt/checkpoint_run{args.run}.pt"
    final_ckpt = f"{out_final}/ckpt/checkpoint_run{args.run}.pt"

    print("=" * 70)
    print(f"Phase S5 train  seed={args.seed}  "
          f"long_lead_threshold={args.long_lead_threshold}  "
          f"long_lead_weight={args.long_lead_weight}")
    print(f"  uncond ckpt   : {args.uncond_ckpt}")
    print(f"  pilot out_root: {out_pilot}")
    print(f"  final out_root: {out_final}")
    rw_str = "default(0.3)" if args.right_weight is None else f"{float(args.right_weight):.3f}"
    lam_str = (f"  intnll_lambda={float(args.gaussian_interval_lambda):.3f}"
               if args.gaussian_loss_mode == "mixed" else "")
    print(f"  loss mode     : gaussian_loss_mode={args.gaussian_loss_mode}  "
          f"continuity_correction={int(args.gaussian_interval_continuity_correction)}  "
          f"right_weight={rw_str}{lam_str}")
    if args.mu_mode == "residual_clim":
        print(f"  mu head mode  : residual_clim  clim_mid={float(args.clim_mid):.2f}  "
              f"delta_max={float(args.delta_max):.1f}")
    elif args.mu_mode == "lead_from_alert":
        print(f"  mu head mode  : lead_from_alert  "
              f"lead_min={float(args.lead_min):.1f}  lead_max={float(args.lead_max):.1f}")
    tm_str = "default(l_offset)" if args.target_mode is None else args.target_mode
    awe_str = ("default(pilot=0, final=5.0)" if args.asym_weight_early is None
               else f"{float(args.asym_weight_early):.2f}")
    print(f"  loss overrides: target_mode={tm_str}  asym_weight_early={awe_str}")
    print("=" * 70)

    _check_ckpt(args.uncond_ckpt, "uncond")

    # ---------- common training args shared between Stage B and Stage C -------
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
        "--stage2_nowcast_require_tstar_before_L", str(int(args.require_tstar_before_L)),
        "--stage2_causal_tstar",
        "--stage2_tstar_layers", "1",
        "--stage2_use_tstar_scalar_pos", "0",
        "--stage2_early_tstar_weight_min", "0.2",
        "--stage2_site_year_mean_loss", "0",
        "--stage2_time_chunk_size", "64",
        "--stage2_conditional_survival", "0",
        "--stage2_pmf_mode", "gaussian",
        "--stage2_pmf_sigma", "5.0",
        "--stage2_pmf_right_weight",
            str(float(args.right_weight) if args.right_weight is not None else 0.3),
        "--stage2_pmf_target_mode", str(args.target_mode) if args.target_mode is not None else "l_offset",
        "--stage2_best_metric", "val_iou80",
        "--amp", str(args.amp),
        "--amp_dtype", str(args.amp_dtype),
        "--d_model_override", str(args.d_model_override),
        "--stage2_pmf_long_lead_threshold", str(args.long_lead_threshold),
        "--stage2_pmf_long_lead_weight", str(args.long_lead_weight),
        "--stage2_gaussian_loss_mode", str(args.gaussian_loss_mode),
        "--stage2_gaussian_interval_continuity_correction",
            str(int(args.gaussian_interval_continuity_correction)),
        "--stage2_gaussian_interval_lambda", str(float(args.gaussian_interval_lambda)),
    ]
    if args.max_epochs_override is not None:
        common += ["--max_epochs_override", str(args.max_epochs_override)]
    if args.batch_train_override is not None:
        common += ["--batch_train_override", str(int(args.batch_train_override))]
    if args.batch_eval_override is not None:
        common += ["--batch_eval_override", str(int(args.batch_eval_override))]
    if args.dispatch_feature_csv:
        common += [
            "--stage2_dispatch_feature_csv", str(args.dispatch_feature_csv),
            "--stage2_dispatch_feature_mode", str(args.dispatch_feature_mode),
            "--stage2_dispatch_feature_missing_value",
                str(args.dispatch_feature_missing_value),
        ]
    if args.cohort_dispatch_only:
        common += ["--stage2_cohort_dispatch_only"]
    if args.mu_mode != "absolute":
        common += [
            "--stage2_pmf_mu_mode", str(args.mu_mode),
            "--stage2_pmf_lead_min", str(args.lead_min),
            "--stage2_pmf_lead_max", str(args.lead_max),
        ]
    if args.mu_mode == "residual_clim":
        if args.clim_mid is None or float(args.clim_mid) <= 0.0:
            raise SystemExit("[abort] --mu_mode=residual_clim requires "
                             "--clim_mid > 0 (per-pest mean_mid in DOY units)")
        common += [
            "--stage2_pmf_clim_mid", str(float(args.clim_mid)),
            "--stage2_pmf_delta_max", str(float(args.delta_max)),
        ]
    if float(args.aux_lead_lambda) > 0.0:
        common += [
            "--stage2_aux_lead_lambda", str(args.aux_lead_lambda),
            "--stage2_aux_lead_huber_delta", str(args.aux_lead_huber_delta),
        ]
    # Per-stage head_mu reset: see resolution above. NOT in `common` so the
    # final stage does not inherit pilot's reset.
    pilot_extra: list[str] = []
    final_extra: list[str] = []
    if reset_pilot:
        pilot_extra.append("--stage2_reset_head_mu")
    if reset_final:
        final_extra.append("--stage2_reset_head_mu")

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
            "--stage2_pmf_target_offset", "0.0",
            "--stage2_pmf_target_early_offset", "30.0",
        ] + pilot_extra
        _run_train("Stage B (pilot, asym=15, +sw)",
                    pilot_cmd,
                    log_dir / f"phase_s5_train_pilot_{w_tag}.log")
        _check_ckpt(pilot_ckpt, "pilot")
    else:
        print(f"[skip_pilot] re-using existing pilot ckpt: {pilot_ckpt}")
        _check_ckpt(pilot_ckpt, "pilot (pre-existing)")

    # ---------- Stage C (final) ------------------------------------------------
    # target_offset is taken from the existing 2-sided final hparams.json
    # (target_offset=0.0) to keep the comparison apples-to-apples against the
    # IoU=0.304 baseline. Override via CLI if a different anchor is desired.
    if not args.skip_final:
        _final_asym_weight_early = (str(float(args.asym_weight_early))
                                    if args.asym_weight_early is not None
                                    else "5.0")
        final_cmd = list(common) + [
            "--out_root", out_final,
            "--stage2_warm_start_ckpt", pilot_ckpt,
            "--stage2_warm_start_seed", str(args.seed),
            "--stage2_pmf_asym_weight", "25.0",
            "--stage2_pmf_asym_weight_early", _final_asym_weight_early,
            "--stage2_pmf_target_offset", "0.0",
            "--stage2_pmf_target_early_offset", "30.0",
        ] + final_extra
        _run_train("Stage C (final, asym=25, +sw)",
                    final_cmd,
                    log_dir / f"phase_s5_train_final_{w_tag}.log")
        _check_ckpt(final_ckpt, "final")
    else:
        print("[skip_final] stage C skipped")

    print("\n" + "=" * 70)
    print(f"Phase S5 train DONE")
    print(f"  pilot : {pilot_ckpt}")
    print(f"  final : {final_ckpt}   ← use this for inference / eval")
    print("=" * 70)


if __name__ == "__main__":
    main()
