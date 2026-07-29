#!/usr/bin/env python
"""Both protocol evaluations for one pest, from the preserved raw grids.

  dev   -- Selection-OK=FAIL. Checkpoint was selected on the WHOLE y-1 season; selector and
           shift are fit on a 50/50 split of that same season. Comparable ACROSS models, not
           quotable as absolute performance. (This is the protocol behind WBPH's 0.3956.)
  clean -- fold-isolated. val_ckpt 40% picked the checkpoint during training, val_fit 30% fits
           the selector, val_cal 30% picks that fold's shift. Eval year touched once.
           (This is the protocol behind WBPH's 0.309865.)

The clean half is a faithful port of foldwise_shift_fix.py -- same helpers, same shift grid,
same assertions, same pooling rule. It is a PORT rather than an import because that module
truncates its run_log.txt at import time, which would corrupt the WBPH artifact.

  cd /home/gpu4080/research/cropscience
  PYTHONPATH=$WS:$CS $PY $CS/rice/experiments/allpests_e5d/pest_eval.py --pest blast
"""
from __future__ import annotations
import argparse, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np, pandas as pd

WS = Path("/home/gpu4080/research/wbph_interval_perf_202607")
CS = Path("/home/gpu4080/research/cropscience")
sys.path.insert(0, str(CS / "rice/experiments/allpests_e5d"))
sys.path.insert(0, str(CS / "rice/experiments/allpests_e5d/vendor"))   # pinned deps only
import pest_paths as PP
from src.io_utils import load_dispatch, load_clim_mid
from src.selector_utils import build_candidates, train_selector, pick_offsets, picked_rows
from src import diagnostics as D

OFFSETS, SEEDS, SIGMA, SHIFT_GRID = PP.OFFSETS, PP.SEEDS, PP.SIGMA, PP.SHIFT_GRID


def frames(g, year, split, ids=None, shift=0.0):
    d = g[(g.year == year) & (g.split == split)].copy()
    if ids is not None:
        d = d[d.sample_id.isin(ids)]
    d = d[["sample_id", "offset", "alert_tstar", "L", "R", "mu"]].copy()
    d["mu"] = d["mu"] + float(shift)
    return d


def make_cand(fr, disp, clim, doy_start, T):
    offs = sorted(set(OFFSETS) & {int(o) for o in fr.offset.unique()})
    return build_candidates(fr, disp, clim, offs, SIGMA, doy_start, T)


def metrics_at(tc, sels, n):
    oracle = float(tc.groupby("sample_id")["iou80_real"].max().mean())
    acc = []
    for reg in sels:
        pr = D.enrich(picked_rows(tc, pick_offsets(reg, tc)))
        m = D.summarize(pr, min(OFFSETS), n)
        mid = 0.5 * (pr["L"] + pr["R"])
        m["center_bias"] = float((pr["pred_mu"] - mid).mean())
        m["onset_bias"] = float((pr["pred_mu"] - pr["L"]).mean())
        m["oracle_iou"] = oracle
        acc.append(m)
    am = {k: float(np.nanmean([a[k] for a in acc])) for k in acc[0]}
    am["selector_oracle_gap"] = am["oracle_iou"] - am["IoU80_overall_tol0"]
    return am


def pool(fm: pd.DataFrame, extra: dict) -> dict:
    """sample-weighted mean for rates/metrics, SUM for counts (the count-column bug fix)."""
    w = fm["n"].to_numpy(float)
    counts = ["late_n_total", "late_late", "late_structural", "late_selector_induced"]
    rec = dict(n_samples_total=int(w.sum()), sigma=SIGMA, **extra)
    for c in counts:
        if c in fm:
            rec[f"{c}_total"] = float(fm[c].sum())
    for c in fm.columns:
        if fm[c].dtype.kind == "f" and c not in counts + ["eval_year", "n", "shift", "sigma"]:
            rec[c] = float(np.average(fm[c].to_numpy(float), weights=w))
    return rec


def half(sample_ids, which):
    """Deterministic 50/50 split of the dev val season into val_fit / val_cal.

    Separate salt from the clean 40/30/30 hash on purpose: the two protocols must not share a
    partition, otherwise 'dev' and 'clean' would be correlated through the same fit set.
    """
    out = set()
    for s in sample_ids:
        u = int(hashlib.md5(f"dev50:{s}".encode()).hexdigest()[:12], 16) / float(16 ** 12)
        if (u < 0.5) == (which == "val_fit"):
            out.add(s)
    return out


def run_dev(pest, years, g, DISP, doy_start, T, outd, log):
    rows = []
    for y in years:
        disp, clim = DISP[y]
        vids = set(g[(g.year == y) & (g.split == "val")].sample_id.unique())
        fit, cal = half(vids, "val_fit"), half(vids, "val_cal")
        if not fit or not cal:
            log(f"[dev] {pest}/{y}: empty val half (fit={len(fit)} cal={len(cal)}) -- skipped"); continue
        sels = [train_selector(make_cand(frames(g, y, "val", fit), disp, clim, doy_start, T), sd)
                for sd in SEEDS]
        best, bestv = 0, -1.0
        for dlt in SHIFT_GRID:
            vc = make_cand(frames(g, y, "val", cal, dlt), disp, clim, doy_start, T)
            n = vc.sample_id.nunique()
            v = float(np.mean([D.summarize(D.enrich(picked_rows(vc, pick_offsets(r, vc))),
                                           min(OFFSETS), n)["IoU80_overall_tol0"] for r in sels]))
            if v > bestv:
                best, bestv = dlt, v
        for tag, dlt in (("uncalibrated", 0), ("calibrated", best)):
            tc = make_cand(frames(g, y, "test", None, dlt), disp, clim, doy_start, T)
            n = tc.sample_id.nunique()
            m = metrics_at(tc, sels, n)
            m.update(eval_year=y, n=n, shift=dlt, sigma=SIGMA, calib=tag)
            rows.append(m)
        log(f"[dev] {pest}/{y}: Δ*={best:+d} (val_cal IoU80={bestv:.4f}) "
            f"fit={len(fit)} cal={len(cal)}")
    if not rows:
        log(f"[dev] {pest}: no evaluable year"); return
    fm = pd.DataFrame(rows); fm.to_csv(outd / "dev_fold_metrics.csv", index=False)
    pooled = [pool(fm[fm.calib == t], dict(pest=pest, protocol="dev_SelectionFAIL", calib=t))
              for t in ("uncalibrated", "calibrated")]
    pd.DataFrame(pooled).to_csv(outd / "dev_pooled_metrics.csv", index=False)
    for r in pooled:
        log(f"[dev] {pest} pooled {r['calib']:12} IoU80={r['IoU80_overall_tol0']:.6f} "
            f"oracle={r['oracle_iou']:.4f} MAE={r['MAE_center']:.2f}")


def run_clean(pest, years, doy_start, T, DISP, outd, log):
    grid_p = PP.clean_grid(pest)
    if not grid_p.exists():
        log(f"[clean] {pest}: grid missing {grid_p} -- skipped"); return
    g = pd.read_csv(grid_p)
    g = g[(g.variant == "E5d_clean") & (g.offset.isin(OFFSETS))]
    assign = json.loads(PP.split_assignment(pest).read_text())
    ids = {y: {k: {s for s, b in assign[str(y)].items() if b == k}
               for k in ("val_ckpt", "val_fit", "val_cal")} for y in years}

    SEL, used, A = {}, {}, {}
    for y in years:
        disp, clim = DISP[y]
        vfc = make_cand(frames(g, y, "val", ids[y]["val_fit"]), disp, clim, doy_start, T)
        SEL[y] = [train_selector(vfc, sd) for sd in SEEDS]
        log(f"[clean] {pest} fold {y}: val_fit rows={len(vfc)} samples={vfc.sample_id.nunique()}")

    sweep, chosen, bestval = [], {}, {}
    for y in years:
        disp, clim = DISP[y]
        used[y] = set()
        for dlt in SHIFT_GRID:
            vcc = make_cand(frames(g, y, "val", ids[y]["val_cal"], dlt), disp, clim, doy_start, T)
            used[y] |= set(vcc.sample_id.unique())
            n = vcc.sample_id.nunique()
            v = float(np.mean([D.summarize(D.enrich(picked_rows(vcc, pick_offsets(r, vcc))),
                                           min(OFFSETS), n)["IoU80_overall_tol0"] for r in SEL[y]]))
            sweep.append(dict(eval_year=y, shift=dlt, n_valcal_samples=n,
                              n_valcal_rows=int(len(vcc)), val_cal_IoU80_tol0=v))
        sy = pd.DataFrame([r for r in sweep if r["eval_year"] == y])
        b = sy.loc[sy.val_cal_IoU80_tol0.idxmax()]
        chosen[str(y)] = int(b["shift"]); bestval[str(y)] = float(b["val_cal_IoU80_tol0"])
        log(f"[clean] {pest} fold {y}: Δ*={int(b['shift']):+d} val_cal IoU80={b.val_cal_IoU80_tol0:.4f}")
    pd.DataFrame(sweep).to_csv(outd / "clean_shift_sweep.csv", index=False)

    # ---- isolation assertions (generalised from the WBPH original to any year set) ----
    cross = {f"eval{y}_rows_from_fold{o}_valcal": len(used[y] & ids[o]["val_cal"])
             for y in years for o in years if o != y}
    yr_of = lambda S: sorted({int(s.split("-")[-1]) for s in S})
    ev_ov = {str(y): len(set(g[(g.year == y) & (g.split == "test")].sample_id.unique()) & used[y])
             for y in years}
    A["cross_fold_valcal_rows"] = cross
    A["no_cross_fold_valcal"] = all(v == 0 for v in cross.values())
    A["shift_selection_years"] = {str(y): yr_of(used[y]) for y in years}
    A["each_fold_uses_only_y_minus_1"] = all(yr_of(used[y]) == [y - 1] for y in years if used[y])
    A["eval_samples_in_shift_selection"] = ev_ov
    A["no_eval_in_shift_selection"] = all(v == 0 for v in ev_ov.values())
    A["sweep_rows"] = len(sweep)
    A["sweep_rows_expected"] = len(SHIFT_GRID) * len(years)
    A["ALL_PASS"] = bool(A["no_cross_fold_valcal"] and A["each_fold_uses_only_y_minus_1"]
                         and A["no_eval_in_shift_selection"]
                         and A["sweep_rows"] == A["sweep_rows_expected"])
    (outd / "clean_protocol_assertions.json").write_text(json.dumps(A, indent=2, default=str))
    log("[clean] assertions " + json.dumps({k: v for k, v in A.items()
                                            if k not in ("cross_fold_valcal_rows",)}, default=str))
    if not A["ALL_PASS"]:
        raise SystemExit(f"[abort] {pest}: clean protocol assertions FAILED -- metrics not reported")

    rows = []
    for y in years:
        disp, clim = DISP[y]
        tc = make_cand(frames(g, y, "test", None, chosen[str(y)]), disp, clim, doy_start, T)
        n = tc.sample_id.nunique()
        m = metrics_at(tc, SEL[y], n)
        m.update(eval_year=y, n=n, shift=chosen[str(y)], sigma=SIGMA)
        rows.append(m)
    fm = pd.DataFrame(rows); fm.to_csv(outd / "clean_fold_metrics.csv", index=False)
    rec = pool(fm, dict(pest=pest, protocol="clean_fold_isolated",
                        shift_by_year=json.dumps(chosen)))
    pd.DataFrame([rec]).to_csv(outd / "clean_pooled_metrics.csv", index=False)
    (outd / "clean_chosen_shifts.json").write_text(json.dumps(
        {"sigma": SIGMA, "grid": [SHIFT_GRID[0], SHIFT_GRID[-1]],
         "chosen_shift_by_eval_year": chosen, "best_val_cal_IoU80": bestval}, indent=2))
    log(f"[clean] {pest} POOLED IoU80={rec['IoU80_overall_tol0']:.6f} "
        f"oracle={rec['oracle_iou']:.4f} MAE={rec['MAE_center']:.2f} n={rec['n_samples_total']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pest", required=True)
    ap.add_argument("--years", type=int, nargs="+", default=PP.YEARS)
    ap.add_argument("--skip", choices=["dev", "clean"], default=None)
    # W&B is opt-in and never fatal. No API key is read here -- wandb resolves
    # WANDB_API_KEY / ~/.netrc itself. Omit the project to stay fully offline.
    ap.add_argument("--wandb_project", default=os.environ.get("WANDB_PROJECT") or None)
    ap.add_argument("--wandb_entity", default=os.environ.get("WANDB_ENTITY") or None)
    ap.add_argument("--wandb_group", default=None)
    a = ap.parse_args()

    outd = PP.eval_dir(a.pest); outd.mkdir(parents=True, exist_ok=True)
    lf = open(outd / "eval_log.txt", "w")
    def log(*x):
        m = " ".join(str(i) for i in x); print(m); lf.write(m + "\n"); lf.flush()

    t0 = time.time()
    doy_start, T = PP.geometry(a.pest)
    P = PP.synthetic_paths(a.pest)
    DISP = {y: (load_dispatch(P, y), load_clim_mid(P, y)) for y in a.years}
    log(f"=== {a.pest}: doy_start={doy_start} T={T} years={a.years} ===")

    if a.skip != "dev":
        gp = PP.dev_grid(a.pest)
        if gp.exists():
            gd = pd.read_csv(gp)
            run_dev(a.pest, a.years, gd[(gd.variant == "E5d") & (gd.offset.isin(OFFSETS))],
                    DISP, doy_start, T, outd, log)
            if a.wandb_project:
                # One run per (pest, eval_year), all 5 selector seeds inside it. Any failure
                # here is swallowed by log_run -- evaluation results are already on disk.
                try:
                    import wandb_viz_e5d as VW
                    for y in a.years:
                        VW.log_e5d_dev_year(a.pest, y, gd, DISP, doy_start, T,
                                            outd, a.wandb_project, a.wandb_entity,
                                            a.wandb_group, log)
                except Exception as e:                                # noqa: BLE001
                    log(f"[wandb] connector failed (non-fatal): {type(e).__name__}: {e}")
        else:
            log(f"[dev] grid missing {gp} -- skipped")
    if a.skip != "clean":
        run_clean(a.pest, a.years, doy_start, T, DISP, outd, log)

    log(f"=== {a.pest} eval done in {time.time()-t0:.0f}s -> {outd} ===")


if __name__ == "__main__":
    main()
