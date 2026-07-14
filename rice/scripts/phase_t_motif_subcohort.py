"""
Phase T8c — Sub-cohort motif diagnostic comparing baseline vs D history_rolling.

For each combination (ckpt, split, cohort):
  - ckpt:   baseline_no_history | D_history_rolling
  - split:  val | test
  - cohort: with_history | no_history | full

Computes:
  1. TP/FP per-feature causal motif AUC + KS (top 5 printed)
  2. Causal-only multivar LogReg AUC (in-sample + 5-fold GroupKFold CV by site)
  3. Recall-preserving FAR sweep (causal secondary filter, target_recall=0.90)
  4. corr(L_DOY, score_peak_DOY) and corr(L_DOY, alert_DOY): Pearson + Spearman
  5. Early/normal/late event group score curve (mean p_cal by DOY) -> CSV per cohort

History cohort definition uses the same policy as the D ckpt (rolling, train_year_max
from D ckpt meta).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rice.configs import config as C
from rice.src.pest_resolver import resolve_pest
from rice.scripts.phase_t_lead_aware_eval import build_probs
from rice.scripts.phase_t_motif_diagnosis import (
    compute_motifs, per_feature_tpfp, multivar_logreg,
    recall_preserving_sweep, build_doy_percentile_table,
    serialize_mvar, CAUSAL_FEATS,
)
from rice.scripts.phase_t_score_diagnosis import (
    per_site_year_records, assign_event_group,
    score_curve_by_group, correlations,
)
from rice.scripts.phase_t_history_subcohort_compare import (
    make_history_mask, filter_probs,
)


def run_one_cohort(probs_df: pd.DataFrame, label: str, tau: float, k: int,
                    doy_start: int, target_recall: float, out_dir: Path) -> dict:
    n_total = int(probs_df.drop_duplicates(["site", "year"]).shape[0])
    n_event = int(probs_df[probs_df.y_event == 1].drop_duplicates(["site", "year"]).shape[0])
    print(f"\n  ---- {label}  (n_site_years={n_total}, n_events={n_event}) ----")
    if n_total < 10 or n_event < 5:
        print("    (skipped: too small)")
        return {"skipped": True, "n_total": n_total, "n_event": n_event}

    pct = build_doy_percentile_table(probs_df, doy_start)
    motif_recs = compute_motifs(probs_df, tau, k, doy_start, pct)
    per_feat = per_feature_tpfp(motif_recs, CAUSAL_FEATS)
    mvar = multivar_logreg(motif_recs, CAUSAL_FEATS)
    sweep = recall_preserving_sweep(motif_recs, mvar, target_recall)
    n_tp = int((motif_recs.cls == "TP").sum())
    n_fp = int((motif_recs.cls == "FP").sum())
    print(f"    alerted: TP={n_tp}  FP={n_fp}")

    if mvar:
        print(f"    causal multivar LogReg AUC:  in={mvar['in_sample_AUC']:.3f}  "
              f"cv={mvar['cv_AUC_mean']:.3f} ± {mvar['cv_AUC_std']:.3f}  "
              f"(folds={mvar['n_cv_folds']})")
    else:
        print("    causal multivar LogReg: (skipped — too few)")
    if sweep:
        b = sweep["base"]; bp = sweep["best_at_target"]
        if bp:
            print(f"    recall-preserve @ R>={target_recall:.2f}:  "
                  f"base R={b['recall']:.3f} FAR={b['FAR']:.3f}  ->  "
                  f"R={bp['recall']:.3f} FAR={bp['FAR']:.3f}  "
                  f"(dFAR={bp['FAR']-b['FAR']:+.3f})  thr={bp['thr']:.3f}  "
                  f"src={sweep['score_source']}")
        else:
            print(f"    recall-preserve @ R>={target_recall:.2f}: no threshold meets target")
    if per_feat:
        top5 = sorted(per_feat.items(), key=lambda x: -x[1]["AUC_dir_agnostic"])[:5]
        print(f"    top-5 per-feature AUC:")
        for f, m in top5:
            print(f"      {f:>34}  AUC={m['AUC_dir_agnostic']:.3f}  "
                  f"KS={m['KS']:.3f}  p={m['p_KS']:.2e}")

    score_recs = per_site_year_records(probs_df, tau, k, doy_start)
    corrs = correlations(score_recs)
    for cn, vv in corrs.items():
        pe = vv.get("pearson"); sp = vv.get("spearman")
        pe_s = f"{pe:.3f}" if pe is not None and not (isinstance(pe, float) and np.isnan(pe)) else "N/A"
        sp_s = f"{sp:.3f}" if sp is not None and not (isinstance(sp, float) and np.isnan(sp)) else "N/A"
        print(f"    corr(L_DOY, {cn:>11s}):  pearson={pe_s}  spearman={sp_s}  (n={vv.get('n')})")

    score_recs_g, gstats = assign_event_group(score_recs)
    group_map = {f"{r.site}|{r.year}": r.group for r in score_recs_g.itertuples(index=False)}
    curve = score_curve_by_group(probs_df, group_map, doy_start)
    curve.to_csv(out_dir / f"curve_{label}.csv", index=False)
    print(f"    L_DOY quartiles: q25={gstats['L_DOY_q25']:.1f}  q75={gstats['L_DOY_q75']:.1f}  "
          f"groups: early={gstats['n_early']} normal={gstats['n_normal']} late={gstats['n_late']}")

    return {
        "n_site_years": n_total, "n_event": n_event,
        "n_TP": n_tp, "n_FP": n_fp,
        "per_feature": per_feat,
        "multivar": serialize_mvar(mvar),
        "recall_sweep": ({"base": sweep["base"],
                           "best_at_target": sweep["best_at_target"],
                           "target_recall": sweep["target_recall"],
                           "score_source": sweep["score_source"]} if sweep else None),
        "corrs": corrs,
        "L_DOY_group_stats": gstats,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", default="sheath_blight")
    ap.add_argument("--run", type=int, required=True)
    ap.add_argument("--baseline_ckpt", required=True)
    ap.add_argument("--d_ckpt", required=True)
    ap.add_argument("--split_seed", type=int, default=42)
    ap.add_argument("--val_year", type=int, default=2022)
    ap.add_argument("--test_year_min", type=int, default=2023)
    ap.add_argument("--test_year_max", type=int, default=2024)
    ap.add_argument("--tau", type=float, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--target_recall", type=float, default=0.90)
    ap.add_argument("--out_dir", required=True)
    args = ap.parse_args()

    _ = resolve_pest(args.pest)
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    class N: pass
    common = N()
    for fld in ["pest", "run", "split_seed", "val_year", "test_year_min", "test_year_max"]:
        setattr(common, fld, getattr(args, fld))

    print("\n========== building baseline probs ==========")
    common.stage1_ckpt = args.baseline_ckpt
    base_cache = build_probs(common)
    doy_start = int(C.DOY_START)

    print("\n========== building D probs ==========")
    common.stage1_ckpt = args.d_ckpt
    d_cache = build_probs(common)

    d_ckpt = torch.load(args.d_ckpt, map_location="cpu", weights_only=False)
    pol = str(d_ckpt.get("site_history_policy", "rolling"))
    tyrmax = int(d_ckpt.get("history_train_year_max", 2021))
    print(f"\n[history meta from D ckpt] policy={pol}  train_year_max={tyrmax}")
    miss_map = make_history_mask(args.pest, args.run, doy_start, pol, tyrmax)

    out = {"history_policy": pol, "train_year_max": tyrmax,
           "args": {"tau": args.tau, "k": args.k, "target_recall": args.target_recall},
           "results": {}}
    for split_name in ["val", "test"]:
        base_df = base_cache[f"{split_name}_df"]
        d_df = d_cache[f"{split_name}_df"]
        present = set(zip(base_df["site"], base_df["year"].astype(int)))
        with_h = {sy for sy in present
                  if miss_map.get(sy, {"prev_year_L_miss": 1})["prev_year_L_miss"] == 0}
        no_h = present - with_h
        print(f"\n\n========== {split_name}  (n_site_years={len(present)}  "
              f"with_history={len(with_h)}  no_history={len(no_h)}) ==========")
        for ckpt_label, probs in [("baseline_no_history", base_df),
                                    ("D_history_rolling", d_df)]:
            print(f"\n  ====== ckpt = {ckpt_label} ======")
            for cohort_name, cohort_set in [("with_history", with_h),
                                              ("no_history", no_h),
                                              ("full", present)]:
                sub = filter_probs(probs, cohort_set)
                label = f"{split_name}__{ckpt_label}__{cohort_name}"
                res = run_one_cohort(sub, label, args.tau, args.k, doy_start,
                                       args.target_recall, out_dir)
                (out["results"]
                    .setdefault(split_name, {})
                    .setdefault(ckpt_label, {})[cohort_name]) = res

    # Side-by-side summary table
    print("\n\n========== Side-by-side (test, target recall =", args.target_recall, ") ==========")
    print(f"  {'ckpt':>22} {'cohort':>14}  {'mvar_cv_AUC':>12}  {'corr_peak':>10} {'corr_alert':>11}  "
          f"{'base_FAR':>9} {'best_FAR':>9} {'dFAR':>7}")
    for ckpt_label in ["baseline_no_history", "D_history_rolling"]:
        for cohort_name in ["with_history", "no_history", "full"]:
            r = out["results"].get("test", {}).get(ckpt_label, {}).get(cohort_name, {})
            if r.get("skipped"): continue
            mv = r.get("multivar") or {}
            cv_auc = mv.get("cv_AUC_mean")
            cv_auc_s = f"{cv_auc:.3f}" if cv_auc is not None else "N/A"
            cr = r.get("corrs", {})
            cp = (cr.get("peak_DOY") or {}).get("pearson")
            ca = (cr.get("alert_DOY") or {}).get("pearson")
            cp_s = f"{cp:.3f}" if cp is not None and not (isinstance(cp, float) and np.isnan(cp)) else "N/A"
            ca_s = f"{ca:.3f}" if ca is not None and not (isinstance(ca, float) and np.isnan(ca)) else "N/A"
            rs = r.get("recall_sweep")
            if rs and rs.get("best_at_target"):
                bf = rs["base"]["FAR"]; nf = rs["best_at_target"]["FAR"]
                bf_s = f"{bf:.3f}"; nf_s = f"{nf:.3f}"; df_s = f"{nf-bf:+.3f}"
            elif rs:
                bf_s = f"{rs['base']['FAR']:.3f}"; nf_s = "—"; df_s = "—"
            else:
                bf_s = "—"; nf_s = "—"; df_s = "—"
            print(f"  {ckpt_label:>22} {cohort_name:>14}  {cv_auc_s:>12}  {cp_s:>10} {ca_s:>11}  "
                  f"{bf_s:>9} {nf_s:>9} {df_s:>7}")

    (out_dir / "motif_subcohort_summary.json").write_text(
        json.dumps(out, indent=2, default=str))
    print(f"\n[saved] {out_dir / 'motif_subcohort_summary.json'}")


if __name__ == "__main__":
    main()
