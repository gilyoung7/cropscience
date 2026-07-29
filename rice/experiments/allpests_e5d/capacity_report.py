#!/usr/bin/env python
"""Render the capacity table and join whatever dev/clean performance exists.

Performance is READ FROM RESULT FILES, never hard-coded, so re-running this after the 8-pest
sweep finishes fills the empty cells automatically. Today only WBPH has both protocols; the
other seven are blank because they have not been trained yet, NOT because they scored badly.

  $PY rice/experiments/allpests_e5d/capacity_report.py
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import numpy as np, pandas as pd

WS = Path("/home/gpu4080/research/wbph_interval_perf_202607")
CS = Path("/home/gpu4080/research/cropscience")
sys.path.insert(0, str(CS / "rice/experiments/allpests_e5d"))
sys.path.insert(0, str(CS / "rice/experiments/allpests_e5d/vendor"))   # pinned deps only
import pest_paths as PP

CAP = PP.OUT_ROOT / "_capacity/capacity_by_pest_year.csv"
OUT = PP.OUT_ROOT / "_capacity"
# WBPH's published results live outside the all-pest tree (they predate it).
WBPH_DEV = WS / "outputs/feature_experiments/target_asym_2x2/eval/target_asym_2x2_calibration_eval.csv"
WBPH_CLEAN_FOLD = WS / "outputs/feature_experiments/e5d_clean_selection_3fold_20260716/foldwise_shift_fix/fold_metrics.csv"
WBPH_CLEAN_POOL = WS / "outputs/feature_experiments/e5d_clean_selection_3fold_20260716/foldwise_shift_fix/pooled_metrics.csv"


def perf_rows() -> pd.DataFrame:
    """(pest, eval_year) -> dev/clean IoU80. Whatever is on disk; missing stays NaN."""
    rows = []
    if WBPH_DEV.exists():
        d = pd.read_csv(WBPH_DEV)
        d = d[(d.model == "E5d_onset_asym") & (d.calib == "calibrated")]
        for _, r in d.iterrows():
            if str(r["scope"]).startswith("year"):
                rows.append(dict(pest="WBPH", eval_year=int(str(r["scope"])[4:]),
                                 dev_IoU80=float(r["IoU80_overall_tol0"]),
                                 dev_shift=int(r["shift"])))
    if WBPH_CLEAN_FOLD.exists():
        c = pd.read_csv(WBPH_CLEAN_FOLD)
        for _, r in c.iterrows():
            rows.append(dict(pest="WBPH", eval_year=int(r["eval_year"]),
                             clean_IoU80=float(r["IoU80_overall_tol0"]),
                             clean_oracle=float(r["oracle_iou"]), clean_shift=int(r["shift"])))
    # per-pest results produced by the all-pest sweep (empty until it runs)
    for p in [r.split()[0] for r in (CS / "rice/experiments/allpests_e5d/pests.tsv").read_text().splitlines()
              if r.strip() and not r.lstrip().startswith("#")]:
        f = PP.eval_dir(p) / "clean_fold_metrics.csv"
        if f.exists():
            for _, r in pd.read_csv(f).iterrows():
                rows.append(dict(pest=p, eval_year=int(r["eval_year"]),
                                 clean_IoU80=float(r["IoU80_overall_tol0"]),
                                 clean_oracle=float(r["oracle_iou"]), clean_shift=int(r["shift"])))
        f = PP.eval_dir(p) / "dev_fold_metrics.csv"
        if f.exists():
            d = pd.read_csv(f)
            for _, r in d[d.calib == "calibrated"].iterrows():
                rows.append(dict(pest=p, eval_year=int(r["eval_year"]),
                                 dev_IoU80=float(r["IoU80_overall_tol0"]), dev_shift=int(r["shift"])))
    if not rows:
        return pd.DataFrame(columns=["pest", "eval_year"])
    return pd.DataFrame(rows).groupby(["pest", "eval_year"], as_index=False).first()


def main():
    if not CAP.exists():
        raise SystemExit(f"[report] missing {CAP} -- run capacity_table.py first")
    cap = pd.read_csv(CAP)
    perf = perf_rows()
    m = cap.merge(perf, on=["pest", "eval_year"], how="left") if len(perf) else cap.assign(
        dev_IoU80=np.nan, clean_IoU80=np.nan)
    m = m.sort_values(["pest", "eval_year"])
    m.to_csv(OUT / "capacity_and_performance.csv", index=False)

    def fmt(x, n=0):
        return "-" if pd.isna(x) else (f"{x:,.{n}f}" if n else f"{int(x):,}")

    L = ["# All-pest E5d — data-to-parameter accounting",
         "",
         "Counts are computed from the real sample tensors (not parsed from logs); WBPH/2024",
         "reproduces its training log exactly (2,072 site-years / 278,653 nowcast samples).",
         "",
         "## Column definitions — three different things get called \"event\"",
         "",
         "| column | meaning |",
         "|---|---|",
         "| `n_event_rows_corpus` | raw observation-level event records, whole corpus, all years. Context only; never used in a ratio. |",
         "| `n_site_year_event_train` | site-years in the train split carrying a finite interval [L,R]. |",
         "| `n_nowcast_pre_L_train` | nowcast samples (site-year x t*) before onset L. |",
         "| `n_nowcast_right_train` | nowcast samples that are right-censored (no event yet). |",
         "",
         "`n_param_effective` is measured by gradient flow through the mu path, not read off the",
         "source: dead in E5d are `tstar_encoder` (the shared path returns before it),",
         "the hazard `head` (gaussian mode) and the template `head_mu` (offset-specific routing).",
         "The probe places each of the 12 slots on its own candidate offset so all 12 heads fire.",
         "",
         "## Parameters", ""]

    pc = m.groupby("pest").first().reset_index()
    L += ["| pest | d_in (prod -> E5d) | T | DOY | total params | effective | dead |",
          "|---|---|---|---|---|---|---|"]
    for _, r in pc.iterrows():
        note = " **(다름)**" if r["doy_start"] != 60 else ""
        L.append(f"| {r['pest']} | {int(r['d_in_production'])} -> {int(r['d_in_e5d'])} | "
                 f"{int(r['T'])}{note} | {int(r['doy_start'])}-{int(r['doy_end'])} | "
                 f"{fmt(r['n_param_total'])} | {fmt(r['n_param_effective'])} | {fmt(r['n_param_dead'])} |")

    L += ["", "## Data and ratios (train split of each eval-year fold)", "",
          "| pest | yr | site-yr | sites | yrs | event site-yr | nowcast | pre-L | right-cens | K mean/med/max | site-yr : effparam | nowcast : effparam | pre-L : effparam |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for _, r in m.iterrows():
        L.append(
            f"| {r['pest']} | {int(r['eval_year'])} | {fmt(r['n_site_year_train'])} | "
            f"{fmt(r['n_site_train'])} | {int(r['n_years_train'])} | {fmt(r['n_site_year_event_train'])} | "
            f"{fmt(r['n_nowcast_total_train'])} | {fmt(r['n_nowcast_pre_L_train'])} | "
            f"{fmt(r['n_nowcast_right_train'])} | "
            f"{r['K_mean_train']:.0f}/{r['K_median_train']:.0f}/{int(r['K_max_train'])} | "
            f"**1 : {r['ratio_effparam_per_siteyear']:.0f}** | {r['ratio_nowcast_total_per_effparam']:.2f} : 1 | "
            f"{r['ratio_nowcast_preL_per_effparam']:.2f} : 1 |")

    L += ["", "## Combined with performance", "",
          "**Blank dev/clean = not yet trained, NOT a poor score.** Only WBPH has been run;",
          "the other seven are pending the 8-pest sweep. Re-run this script afterwards to fill them.",
          "",
          "| pest | yr | site-yr | nowcast | site-yr : effparam | dev IoU80 | clean IoU80 |",
          "|---|---|---|---|---|---|---|"]
    for _, r in m.iterrows():
        L.append(f"| {r['pest']} | {int(r['eval_year'])} | {fmt(r['n_site_year_train'])} | "
                 f"{fmt(r['n_nowcast_total_train'])} | 1 : {r['ratio_effparam_per_siteyear']:.0f} | "
                 f"{fmt(r.get('dev_IoU80'), 4)} | {fmt(r.get('clean_IoU80'), 4)} |")

    n_perf = int(m["clean_IoU80"].notna().sum()) if "clean_IoU80" in m else 0
    L += ["", "## Exploratory reading only", "",
          f"Pests with a clean score so far: **{len(m[m.get('clean_IoU80', pd.Series(dtype=float)).notna()]['pest'].unique()) if n_perf else 0} of 8**.",
          "",
          "With at most 8 pests (and 3 correlated year-folds each) any correlation between a",
          "capacity ratio and IoU is **descriptive, not causal**. The pests differ simultaneously in",
          "season length, feature width, Stage-1 gate variant, event density and batch size, so no",
          "single ratio is isolated. Read the numbers as a map of where the model is most",
          "under-determined, not as an explanation of why one pest scores higher.",
          "",
          "BPH additionally runs on DOY 140-270 (T=131) against 60-300 (T=241) for the other seven.",
          "Its IoU is not on the same axis; keep it in its own row and out of any cross-pest mean.",
          ""]

    (OUT / "CAPACITY_REPORT.md").write_text("\n".join(L))
    print("\n".join(L))
    print(f"\n[report] wrote {OUT/'CAPACITY_REPORT.md'} and capacity_and_performance.csv")


if __name__ == "__main__":
    main()
