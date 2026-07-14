"""Multi-pest TEST eval: production baseline Stage-2 vs S2N-direct (Stage-2 +neighbor).

Reuses ONLY project functions (no existing file modified). For each pest it
reconstructs each ckpt's training-time X (base -> neighbor -> dispatch, driven by
ckpt metadata), rebuilds the grouped causal-tstar model, and runs the SAME grouped
interval metric used in training (eval_metrics_with_overlap_grouped) on test=2024.
val is recomputed only as a self-check against ckpt['best_val_iou80'].

Outputs:
  rice/outputs_stage2_compare_eval/<pest>/test_eval_{baseline,direct_neighbor}.json
  rice/outputs_stage2_compare_eval/all_pests_direct_neighbor_compare.tsv

Usage (from cropscience/ root):
  PYTHONPATH=. .venv/bin/python rice/scripts/eval_s2n_direct_compare.py \
      --pests WBPH BPH blast bacterial_blight brown_spot sheath_blight
"""
from __future__ import annotations
import json, argparse
from pathlib import Path
import numpy as np
import torch

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.scripts.run_eval import build_samples_for_run
from rice.src.dataset import (
    split_samples, build_stage2_nowcast_samples,
    group_stage2_samples_by_site_year, GroupedIntervalEventDataset,
)
from rice.scripts.common import make_loader, collate_grouped_stage2
from rice.src.model import HierarchicalCausalHazardTransformer
from rice.src.train_eval import eval_metrics_with_overlap_grouped

PROD_BASE = "rice/outputs_stage2_batch_2024_bestgate"
DN_BASE = "rice/outputs_stage2_direct_neighbor"
OUT_BASE = "rice/outputs_stage2_compare_eval"

METRIC_COLS = [
    ("IoU80", "IoU_mean_interval_only(80%)"),
    ("PI_hit", "point_cov_mean_interval_only"),
    ("mass_in_interval", "mass_in_interval_mean_interval_only"),
    ("MAE_center", "mae_mid_mean_interval_only"),
    ("Recall80", "Recall_mean_interval_only(80%)"),
    ("Prec80", "Precision_mean_interval_only(80%)"),
    ("N_interval", "N_interval_samples"),
]


def reconstruct_samples(ckpt, run, get_feature_cols):
    _, _, _T, samples = build_samples_for_run(run, get_feature_cols)
    log = {"base_d_in": int(samples[0]["X"].shape[1])}
    if bool(ckpt.get("stage2_neighbor_history_added", False)):
        from rice.scripts.neighbor_history_utils import (
            load_long_events, build_neighbor_index, append_neighbor_to_samples)
        decay = float(ckpt.get("stage2_neighbor_decay_km", 20.0))
        ev, co, _ = load_long_events(C.PATH_OBS, label_col=getattr(C, "LABEL_COL", "label_event"),
                                     year_min=getattr(C, "YEAR_MIN", None), year_max=getattr(C, "YEAR_MAX", None))
        append_neighbor_to_samples(samples, build_neighbor_index(ev, co),
                                   doy_start=int(C.DOY_START), decay_km=decay)
        log["after_neighbor_d_in"] = int(samples[0]["X"].shape[1]); log["neighbor_decay_km"] = decay
    if bool(ckpt.get("stage2_dispatch_features_added", False)):
        from rice.scripts.stage1_confidence_utils import (
            load_dispatch_feature_table, append_dispatch_confidence_to_samples)
        conf = load_dispatch_feature_table(ckpt.get("stage2_dispatch_feature_csv"))
        if bool(ckpt.get("stage2_cohort_dispatch_only", False)):
            keys = set(conf.keys()); n0 = len(samples)
            samples = [s for s in samples if (str(s["site_id"]), int(s["year"])) in keys]
            log["cohort_filter"] = f"{n0}->{len(samples)}"
        append_dispatch_confidence_to_samples(
            samples, conf, doy_start=int(C.DOY_START),
            mode=str(ckpt.get("stage2_dispatch_feature_mode", "causal")),
            missing_value=float(ckpt.get("stage2_dispatch_feature_missing_value", 0.0)))
        log["after_dispatch_d_in"] = int(samples[0]["X"].shape[1])
    return samples, log


def build_model(ckpt, d_in, device):
    m = HierarchicalCausalHazardTransformer(
        d_in=int(d_in), d_model=int(ckpt.get("d_model", C.D_MODEL)),
        nhead=int(ckpt.get("n_head", C.N_HEAD)), num_layers=int(ckpt.get("n_layers", C.N_LAYERS)),
        num_tstar_layers=int(ckpt.get("stage2_tstar_layers", 1)), dropout=float(ckpt.get("dropout", 0.2)),
        max_len=int(getattr(C, "MAX_LEN", 400)), max_tstar_len=512,
        use_tstar_scalar_pos=bool(int(ckpt.get("stage2_use_tstar_scalar_pos", 0))),
        phenology_bias_head=bool(int(ckpt.get("stage2_phenology_bias_head", 0))),
        phenology_dim=4, phenology_hidden=int(ckpt.get("stage2_phenology_hidden", 8)),
    ).to(device)
    m.time_chunk_size = int(ckpt.get("stage2_time_chunk_size", 64))
    m.conditional_survival = bool(int(ckpt.get("stage2_conditional_survival", 0)))
    m.pmf_mode = str(ckpt.get("stage2_pmf_mode", "hazard"))
    m.gaussian_sigma = float(ckpt.get("stage2_pmf_sigma", 5.0))
    m.gaussian_mu_max = float(ckpt.get("stage2_pmf_mu_max", 0.0))
    m.mu_mode = str(ckpt.get("stage2_pmf_mu_mode", "absolute"))
    m.lead_min = float(ckpt.get("stage2_pmf_lead_min", 7.0)); m.lead_max = float(ckpt.get("stage2_pmf_lead_max", 75.0))
    m.clim_mid_rel = float(ckpt.get("stage2_pmf_clim_mid", 0.0)) - float(C.DOY_START) + 1.0
    m.delta_max = float(ckpt.get("stage2_pmf_delta_max", 60.0))
    m.alert_tstar_feat_idx = int(ckpt.get("stage2_pmf_alert_tstar_feat_idx", -1))
    m.doy_start = int(C.DOY_START); m.lead_strict_alert_check = True; m.lead_debug_once_pending = False
    m.right_anchor = float(ckpt.get("stage2_pmf_right_anchor", 0.0))
    m.target_mode = str(ckpt.get("stage2_pmf_target_mode", "l_offset"))
    m.asym_weight = float(ckpt.get("stage2_pmf_asym_weight", 10.0)); m.right_weight = float(ckpt.get("stage2_pmf_right_weight", 0.3))
    m.target_offset = float(ckpt.get("stage2_pmf_target_offset", 0.0))
    m.asym_weight_early = float(ckpt.get("stage2_pmf_asym_weight_early", 0.0))
    m.target_early_offset = float(ckpt.get("stage2_pmf_target_early_offset", 30.0))
    m.gaussian_loss_mode = str(ckpt.get("stage2_gaussian_loss_mode", "asym_mse"))
    m.gaussian_interval_continuity_correction = bool(int(ckpt.get("stage2_gaussian_interval_continuity_correction", 0)))
    m.gaussian_interval_lambda = float(ckpt.get("stage2_gaussian_interval_lambda", 0.1))
    for z in ["zone_late_weight","zone_too_late_weight","zone_missed_weight","zone_too_early_weight",
              "zone_too_late_threshold","zone_missed_threshold","zone_too_early_threshold",
              "long_lead_threshold","long_lead_weight","aux_lead_lambda","aux_lead_huber_delta",
              "early_tstar_weight_min","site_year_mean_loss"]:
        setattr(m, z, float(ckpt.get(f"stage2_pmf_{z}", ckpt.get(f"stage2_{z}", 0.0)) or 0.0))
    m._phase_s5_sw_logged = True
    m.load_state_dict(ckpt["trained_states"][0]["state_dict"]); m.eval()
    return m


def eval_split(split_s, ckpt, model, x_mean, x_std, T, device):
    s = build_stage2_nowcast_samples(
        split_s, window=int(ckpt.get("stage2_nowcast_window", 28)),
        stride=int(ckpt.get("stage2_nowcast_stride", 1)),
        tstar_start=ckpt.get("stage2_nowcast_tstar_start", None),
        only_pre_event=bool(int(ckpt.get("stage2_nowcast_only_pre_event", 1))),
        event_time_proxy=str(ckpt.get("stage2_nowcast_event_time_proxy", "r")),
        require_tstar_before_L=bool(int(ckpt.get("stage2_nowcast_require_tstar_before_L", 0))))
    groups = group_stage2_samples_by_site_year(s)
    loader = make_loader(GroupedIntervalEventDataset(groups, x_mean, x_std),
                         C.BATCH_EVAL, shuffle=False, collate_fn=collate_grouped_stage2)
    stats = eval_metrics_with_overlap_grouped(model, loader, Tend=int(T), device=device, alpha=0.2,
                                              pi_method=getattr(C, "PI_METHOD", "shortest"))
    stats["_n_nowcast_samples"] = len(s); stats["_n_site_year_groups"] = len(groups)
    return stats


def run_ckpt(ckpt_path, pest, tag, device):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    run = int(ckpt["run"]); _, get_feature_cols = resolve_pest(pest)
    C.DOY_START = int(ckpt.get("doy_start", 60)); C.DOY_END = int(ckpt.get("doy_end", 300))
    if ckpt.get("d_model"): C.D_MODEL = int(ckpt["d_model"])
    if ckpt.get("n_head"): C.N_HEAD = int(ckpt["n_head"])
    if ckpt.get("n_layers"): C.N_LAYERS = int(ckpt["n_layers"])
    T = C.DOY_END - C.DOY_START + 1
    samples, rlog = reconstruct_samples(ckpt, run, get_feature_cols)
    d_in = int(samples[0]["X"].shape[1])
    assert d_in == int(ckpt["d_in"]), f"{pest}/{tag}: d_in {d_in} != ckpt {ckpt['d_in']}"
    _, val_s, test_s = split_samples(samples, val_frac=0.1, test_frac=0.1,
                                     seed=int(ckpt.get("split_seed", 42)), split_mode="year",
                                     val_year=2023, test_year_min=2024, test_year_max=2024)
    model = build_model(ckpt, d_in, device)
    val_stats = eval_split(val_s, ckpt, model, ckpt["norm_mean"], ckpt["norm_std"], T, device)
    test_stats = eval_split(test_s, ckpt, model, ckpt["norm_mean"], ckpt["norm_std"], T, device)
    ckpt_val = float(ckpt["trained_states"][0].get("best_val_iou80", float("nan")))
    recon_val = float(val_stats["IoU_mean_interval_only(80%)"])
    ok = abs(recon_val - ckpt_val) < 1e-3
    print(f"[{pest}/{tag}] d_in={d_in} recon={rlog} | VAL IoU80 recon={recon_val:.4f} ckpt={ckpt_val:.4f} "
          f"selfcheck={'OK' if ok else 'MISMATCH'}")
    return {"pest": pest, "tag": tag, "ckpt": str(ckpt_path), "d_in": d_in, "recon": rlog,
            "selfcheck_ok": bool(ok), "ckpt_best_val_iou80": ckpt_val,
            "val": val_stats, "test": test_stats}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pests", nargs="+",
                    default=["WBPH", "BPH", "blast", "bacterial_blight", "brown_spot", "sheath_blight"])
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = []
    for pest in args.pests:
        jobs = [("baseline", f"{PROD_BASE}/{pest}/lead_v3_final/ckpt/checkpoint_run4.pt"),
                ("direct_neighbor", f"{DN_BASE}/{pest}/lead_v3_final/ckpt/checkpoint_run4.pt")]
        out_dir = Path(OUT_BASE) / pest; out_dir.mkdir(parents=True, exist_ok=True)
        for tag, p in jobs:
            if not Path(p).exists():
                print(f"[skip] {pest}/{tag}: ckpt not found {p}"); continue
            try:
                res = run_ckpt(p, pest, tag, device)
            except Exception as e:
                print(f"[ERROR] {pest}/{tag}: {type(e).__name__}: {e}"); continue
            (out_dir / f"test_eval_{tag}.json").write_text(json.dumps(res, indent=2, default=str))
            rows.append(res)

    # combined long-format TSV
    header = ["pest", "variant", "d_in"] + [c for c, _ in METRIC_COLS] + ["n_nowcast", "selfcheck"]
    lines = ["\t".join(header)]
    for r in rows:
        t = r["test"]
        vals = [f"{float(t[k]):.4f}" if isinstance(t.get(k), (int, float)) else str(t.get(k)) for _, k in METRIC_COLS]
        lines.append("\t".join([r["pest"], r["tag"], str(r["d_in"])] + vals
                               + [str(t.get("_n_nowcast_samples")), "OK" if r["selfcheck_ok"] else "MISMATCH"]))
    out_tsv = Path(OUT_BASE) / "all_pests_direct_neighbor_compare.tsv"
    out_tsv.write_text("\n".join(lines) + "\n")
    print("\n===== ALL-PEST TEST COMPARISON =====\n" + "\n".join(lines))
    print(f"\n[saved] {out_tsv}")


if __name__ == "__main__":
    main()
