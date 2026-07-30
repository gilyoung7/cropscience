#!/usr/bin/env python
"""Per-pest data-to-parameter accounting for the E5d Stage-2 model.

Every count below is computed from the real sample tensors, not parsed from logs, so the
definitions are uniform across pests. The column names spell out which "event" is meant --
three different things are called that in this project and conflating them is the main way
this table could mislead:

  n_event_rows_corpus     raw observation-level event records for the pest (whole corpus,
                          all years, from neighbor_feature_stats.tsv). NOT used in ratios.
  n_site_year_event_*     site-years whose label is an interval event (has L,R) in that split.
  n_nowcast_pre_L_*       nowcast samples (site-year x t*) falling before onset L.
  n_nowcast_right_*       nowcast samples that are right-censored (no event yet).

Effective parameters are determined by GRADIENT FLOW, not by reading the source: we run one
forward + backward on the mu path and count only parameters that receive a non-None gradient.
In the E5d configuration the hazard head and the template head_mu are both dead.

  cd /home/gpu4080/research/cropscience
  PYTHONPATH=$WS:$CS $PY $CS/rice/experiments/allpests_e5d/capacity_table.py --out <dir>
"""
from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd, torch

WS = Path("/home/gpu4080/research/wbph_interval_perf_202607")
CS = Path("/home/gpu4080/research/cropscience")
sys.path.insert(0, str(CS / "rice/experiments/allpests_e5d_curriculum"))
sys.path.insert(0, str(CS / "rice/experiments/allpests_e5d/vendor")); sys.path.insert(0, str(CS))   # pinned deps only
import pest_paths as PP

OFFSETS = PP.OFFSETS
NEIGHBOR_CH = 6          # --stage2_add_neighbor_history appends 6 channels


def registry():
    rows = []
    for line in (CS / "rice/experiments/allpests_e5d/pests.tsv").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        f = line.split()
        rows.append(dict(pest=f[0], server=int(f[1]), doy_start=int(f[2]), doy_end=int(f[3]),
                         n_event_rows_corpus=int(f[4])))
    return rows


def param_counts(d_in: int, T: int, ckpt_meta: dict) -> dict:
    """Total vs gradient-reachable parameters for the E5d configuration."""
    from src.vendor.model import HierarchicalCausalHazardTransformer as V
    m = V(d_in, d_model=int(ckpt_meta.get("d_model", 48)),
          nhead=int(ckpt_meta.get("n_head", 4)),
          num_layers=int(ckpt_meta.get("n_layers", 3)),
          max_len=max(400, T + 8),
          use_shared_multi_offset=True, mu_head_mode="offset_specific",
          candidate_offsets=OFFSETS,
          shared_band_window=int(ckpt_meta.get("stage2_nowcast_window", 28)),
          use_issue_doy_features=False)
    m.pmf_mode = "gaussian"
    m.train()
    total = sum(p.numel() for p in m.parameters() if p.requires_grad)

    # The probe must exercise ALL 12 offset heads, otherwise they look dead. Routing is
    # offset = tstar - alert_rel with alert_rel = X[..., alert_idx].amax(T) - doy_start + 1
    # (model.py:_offset_specific_mu_logit). alert_tstar_feat_idx defaults to -1 and doy_start
    # to 1.0 on a freshly constructed model, so we can place slot k exactly on OFFSETS[k].
    B, K = 2, len(OFFSETS)
    alert_idx = int(getattr(m, "alert_tstar_feat_idx", -1))
    ds = float(getattr(m, "doy_start", 1.0))
    ts = T // 2
    X = torch.randn(B, K, T, d_in)
    for k, off in enumerate(OFFSETS):
        X[:, k, :, alert_idx] = ts - off + ds - 1.0
    tstar = torch.full((B, K), ts, dtype=torch.long)
    vm = torch.ones(B, K, dtype=torch.bool)
    m.zero_grad(set_to_none=True)
    out = m(X, tstar, vm, None)
    mu = getattr(m, "_last_mu_BK", None)
    # drive the loss through the mu path (what the E5d objective actually optimises)
    loss = (mu.float() ** 2).mean() if mu is not None else out.float().pow(2).mean()
    loss.backward()
    n_heads_hit = sum(1 for q in range(len(OFFSETS))
                      if any(p.grad is not None and p.grad.abs().sum().item() > 0
                             for p in m.offset_mu_heads[q].parameters()))

    eff, dead = 0, {}
    for name, p in m.named_parameters():
        if not p.requires_grad:
            continue
        if p.grad is None or not torch.isfinite(p.grad).any() or p.grad.abs().sum().item() == 0.0:
            dead[name] = p.numel()
        else:
            eff += p.numel()
    dead_by_mod: dict[str, int] = {}
    for n, c in dead.items():
        dead_by_mod[n.split(".")[0]] = dead_by_mod.get(n.split(".")[0], 0) + c
    return dict(n_param_total=total, n_param_effective=eff, n_param_dead=total - eff,
                n_offset_heads_exercised=n_heads_hit, dead_modules=json.dumps(dead_by_mod))


def data_counts(pest: str, year: int) -> dict | None:
    """Build the real samples for one (pest, eval_year) fold and count everything."""
    from rice.configs import config as C
    from rice.src.pest_resolver import resolve_pest
    from rice.src.dataset import (split_samples, build_stage2_nowcast_samples,
                                  group_stage2_samples_by_site_year)
    from rice.scripts.eval_s2n_direct_compare import reconstruct_samples

    ck_p = PP.cell_dir(pest, year) / "lead_v3_final/ckpt/checkpoint_run4.pt"
    ck = torch.load(ck_p, map_location="cpu", weights_only=False)
    _, gfc = resolve_pest(pest)
    C.DOY_START = int(ck.get("doy_start", 60)); C.DOY_END = int(ck.get("doy_end", 300))
    T = C.DOY_END - C.DOY_START + 1
    samples, _ = reconstruct_samples(ck, int(ck["run"]), gfc)
    tr, va, te = split_samples(samples, val_frac=0.1, test_frac=0.1,
                               seed=int(ck.get("split_seed", 42)), split_mode="year",
                               val_year=year - 1, test_year_min=year, test_year_max=year)

    win = int(ck.get("stage2_nowcast_window", 28)); strd = int(ck.get("stage2_nowcast_stride", 1))
    rec = dict(pest=pest, eval_year=year, doy_start=C.DOY_START, doy_end=C.DOY_END, T=T,
               d_in_production=int(samples[0]["X"].shape[1]) if samples else -1)
    rec["d_in_e5d"] = rec["d_in_production"] + NEIGHBOR_CH

    kk_all = []
    for tag, S in (("train", tr), ("val", va), ("test", te)):
        sy = {f'{s["site_id"]}-{int(s["year"])}' for s in S}
        rec[f"n_site_year_{tag}"] = len(sy)
        rec[f"n_site_{tag}"] = len({s["site_id"] for s in S})
        yrs = sorted({int(s["year"]) for s in S})
        rec[f"n_years_{tag}"] = len(yrs)
        rec[f"years_{tag}"] = f"{min(yrs)}-{max(yrs)}" if yrs else ""
        # An "event site-year" is one whose label is an INTERVAL. Every sample carries an L
        # field regardless (right-censored ones too), so censor_type is the discriminator --
        # this reproduces the trainer's [split_sanity] 'interval' count.
        rec[f"n_site_year_event_{tag}"] = sum(
            1 for s in S if str(s.get("censor_type")) == "interval")
        rec[f"n_site_year_rightcens_{tag}"] = sum(
            1 for s in S if str(s.get("censor_type")) == "right")

        ns = build_stage2_nowcast_samples(S, window=win, stride=strd)
        rec[f"n_nowcast_total_{tag}"] = len(ns)
        pre = sum(1 for x in ns if x.get("case_bucket") == "pre_L")
        rig = sum(1 for x in ns if x.get("case_bucket") == "right")
        rec[f"n_nowcast_pre_L_{tag}"] = pre
        rec[f"n_nowcast_right_{tag}"] = rig
        rec[f"n_nowcast_other_{tag}"] = len(ns) - pre - rig

        # returns a LIST of {site_id, year, samples}; K is len(samples), not len(dict)
        grp = group_stage2_samples_by_site_year(ns)
        ks = np.array([len(g["samples"]) for g in grp], dtype=float)
        if ks.size:
            rec[f"K_mean_{tag}"] = float(ks.mean())
            rec[f"K_median_{tag}"] = float(np.median(ks))
            rec[f"K_max_{tag}"] = int(ks.max())
            if tag == "train":
                kk_all = ks
        else:
            rec[f"K_mean_{tag}"] = rec[f"K_median_{tag}"] = float("nan"); rec[f"K_max_{tag}"] = 0

    rec["_ckpt_meta"] = {k: ck.get(k) for k in
                         ("d_model", "n_head", "n_layers", "stage2_nowcast_window")}
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(WS / "outputs/allpests_e5d/_capacity"))
    ap.add_argument("--years", type=int, nargs="+", default=PP.YEARS)
    ap.add_argument("--pests", nargs="+", default=None)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    reg = {r["pest"]: r for r in registry()}
    pests = a.pests or list(reg)
    rows = []
    for p in pests:
        for y in a.years:
            t0 = time.time()
            try:
                rec = data_counts(p, y)
            except Exception as e:                       # noqa: BLE001
                print(f"[skip] {p}/{y}: {type(e).__name__}: {str(e)[:160]}", flush=True)
                continue
            meta = rec.pop("_ckpt_meta")
            rec.update(param_counts(rec["d_in_e5d"], rec["T"], meta))
            rec["n_event_rows_corpus"] = reg[p]["n_event_rows_corpus"]
            rec["geometry_note"] = ("BPH_window_140_270_T131_NOT_COMPARABLE"
                                    if rec["doy_start"] != 60 else "standard_60_300_T241")
            e = rec["n_param_effective"]
            rec["ratio_siteyear_per_effparam"] = rec["n_site_year_train"] / e
            rec["ratio_effparam_per_siteyear"] = e / rec["n_site_year_train"]
            rec["ratio_nowcast_total_per_effparam"] = rec["n_nowcast_total_train"] / e
            rec["ratio_nowcast_preL_per_effparam"] = rec["n_nowcast_pre_L_train"] / e
            rows.append(rec)
            print(f"[ok] {p}/{y} in {time.time()-t0:.0f}s  "
                  f"sy_train={rec['n_site_year_train']} nowcast={rec['n_nowcast_total_train']} "
                  f"eff_param={e}", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(out / "capacity_by_pest_year.csv", index=False)
    print(f"\n[capacity] wrote {out/'capacity_by_pest_year.csv'} ({len(df)} rows)")


if __name__ == "__main__":
    main()
