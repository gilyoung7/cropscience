#!/usr/bin/env python
"""Pre-flight validation for the all-pest E5d run. Changes nothing; exits non-zero on any FAIL.

Run this before launching either server. It is the only thing standing between a typo and
48 wasted GPU-hours.

  --level fast  (default) static checks only, no data load, ~seconds
  --level full  additionally builds the real sample tensors and runs a model forward per pest

Usage (from the repo root; the roots are derived, so no absolute path is needed):
  cd <cropscience>
  .venv/bin/python rice/experiments/allpests_e5d/dry_run.py --level full

Set CROPSCIENCE_ROOT only if you are running a copy of this directory from outside the repo.
"""
from __future__ import annotations
import os
os.environ.setdefault("MPLBACKEND", "Agg")
import argparse, json, subprocess, sys
import pandas as pd
from pathlib import Path

from repo_paths import AP, CS, VENDOR, WS        # roots derived from this file's location
sys.path.insert(0, str(VENDOR)); sys.path.insert(0, str(CS))

YEARS = [2022, 2023, 2024]
OFFSETS = [3, 7, 14, 21, 28, 30, 35, 42, 45, 49, 56, 60]
ANCHOR_MAX = 75                      # scripts/87 ANCHORS = 1..75

FAILS: list[str] = []
WARNS: list[str] = []


def fail(msg): FAILS.append(msg); print(f"  FAIL  {msg}")
def warn(msg): WARNS.append(msg); print(f"  WARN  {msg}")
def ok(msg):   print(f"  ok    {msg}")
ok_ = ok


def registry() -> list[dict]:
    rows = []
    for line in (AP / "pests.tsv").read_text().splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        f = line.split()
        rows.append(dict(pest=f[0], server=int(f[1]), doy_start=int(f[2]),
                         doy_end=int(f[3]), events=int(f[4]), b16_cells=int(f[5]),
                         cost=int(f[6])))
    return rows


def batch_dir(y): return "batch_2024_bestgate" if y == 2024 else f"batch_{y}_baseline"


def hparams(pest, year) -> dict | None:
    p = CS / f"rice/outputs/stage2/{batch_dir(year)}/{pest}/lead_v3_final/hparams.json"
    return json.loads(p.read_text()) if p.exists() else None


# ---------------------------------------------------------------- static checks
def check_registry_vs_config(rows):
    """pests.tsv geometry must match both the pest config AND the production hparams."""
    print("\n[1] registry vs pest config vs production hparams")
    from rice.src.pest_resolver import resolve_pest, available_pest_slugs
    avail = available_pest_slugs()
    for r in rows:
        p = r["pest"]
        if p not in avail:
            fail(f"{p}: not a loadable slug (available={avail})"); continue
        C, _ = resolve_pest(p)
        if (int(C.DOY_START), int(C.DOY_END)) != (r["doy_start"], r["doy_end"]):
            fail(f"{p}: pests.tsv doy=({r['doy_start']},{r['doy_end']}) != config "
                 f"({C.DOY_START},{C.DOY_END})")
            continue
        bad = [y for y in YEARS
               if (h := hparams(p, y)) and
               (h.get("doy_start"), h.get("doy_end")) != (r["doy_start"], r["doy_end"])]
        if bad:
            fail(f"{p}: hparams geometry disagrees with config in years {bad}")
        else:
            ok(f"{p}: doy=({r['doy_start']},{r['doy_end']}) T={r['doy_end']-r['doy_start']+1} consistent")


def check_inputs_exist(rows):
    """Every cell needs a dispatch CSV, a climatology table and a production ckpt."""
    print("\n[2] per-cell input files")
    for r in rows:
        p = r["pest"]
        miss = []
        for y in YEARS:
            d = CS / f"rice/outputs/stage2/{batch_dir(y)}/{p}"
            if not list(d.glob("gate_*_R088_features_per_sy.csv")): miss.append(f"{y}:dispatch")
            if not (d / "climatology_train_stats.csv").exists():    miss.append(f"{y}:clim")
            if not (d / "lead_v3_final/ckpt/checkpoint_run4.pt").exists(): miss.append(f"{y}:ckpt")
        fail(f"{p}: missing {miss}") if miss else ok(f"{p}: dispatch+clim+ckpt present for all 3 years")


def check_arg_extraction(rows):
    """Drive the real extract_args() from common.sh -- the exact code the trainer will use."""
    print("\n[3] production arg extraction (via common.sh)")
    for r in rows:
        p = r["pest"]
        for y in YEARS:
            cmd = f'source {AP}/common.sh; extract_args {p} {y}'
            res = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True)
            if res.returncode != 0 or not res.stdout.strip():
                fail(f"{p}/{y}: extract_args failed rc={res.returncode} {res.stderr.strip()[:120]}")
                continue
            a = res.stdout
            need = {"--pest": p, "--stage2_pmf_asym_weight": "25.0",
                    "--stage2_pmf_asym_weight_early": "0.0",
                    "--val_year": str(y - 1), "--test_year_min": str(y), "--test_year_max": str(y)}
            toks = a.split()
            bad = [k for k, v in need.items() if k not in toks or toks[toks.index(k) + 1] != v]
            if bad:
                fail(f"{p}/{y}: wrong/missing args {bad}")
            elif "--out_root" in toks:
                fail(f"{p}/{y}: --out_root survived stripping (would write into production!)")
            elif "--stage2_add_neighbor_history" not in toks:
                fail(f"{p}/{y}: neighbor history flag missing")
            else:
                ok(f"{p}/{y}: args OK ({len(toks)} tokens)")


def check_blast_2023_fallback():
    """The blast/2023 log is empty; we borrow blast/2022's command. Prove that is still valid."""
    print("\n[4] blast/2023 fallback validity")
    a, b = hparams("blast", 2022), hparams("blast", 2023)
    if not a or not b:
        fail("blast hparams missing"); return
    skip = {"val_year", "test_year_min", "test_year_max", "out_root",
            "stage2_warm_start_ckpt", "stage2_dispatch_feature_csv", "pest"}
    diff = [k for k in (set(a) | set(b)) - skip if a.get(k) != b.get(k)]
    if diff:
        fail(f"blast 2023 hparams differ from 2022 on {diff} -> fallback INVALID, do not train blast/2023")
    else:
        ok(f"blast 2023 == 2022 on all {len(set(a)|set(b))-len(skip)} non-year keys -> fallback sound")


def check_offset_feasibility(rows):
    """T=doy_end-doy_start+1 bounds how many candidate offsets can ever be realised."""
    print("\n[5] offset feasibility vs season length")
    for r in rows:
        T = r["doy_end"] - r["doy_start"] + 1
        if T <= max(OFFSETS):
            fail(f"{r['pest']}: T={T} <= max candidate offset {max(OFFSETS)}")
        elif T < ANCHOR_MAX * 2:
            warn(f"{r['pest']}: T={T} is short (others 241); anchors 1..{ANCHOR_MAX} and "
                 f"offsets up to {max(OFFSETS)} will be sparsely feasible -> expect fewer grid rows")
        else:
            ok(f"{r['pest']}: T={T}, all {len(OFFSETS)} candidate offsets have room")


S1_ROOT = CS / "rice/outputs/stage1/batch_rolling"
SPLIT_OF = {2022: ("split1", 2021), 2023: ("split2", 2022), 2024: ("split3", 2023)}


def _s1_dir(pest, year):
    """The forward-chained Stage-1 run dir for this eval year, whichever run{N} holds it."""
    sp, val = SPLIT_OF[year]
    for run in (0, 1, 2):
        d = S1_ROOT / pest / f"run{run}" / f"{sp}_v{val}_t{year}"
        if (d / "A/ckpt/event_xgb_w28_lead14-45_A.pt").exists():
            return d, run
    return None, None


def check_stage1_binding(rows):
    """(1) Every cell must consume a Stage-1 model whose TEST year is its own eval year.

    The dir name encodes val/test years, so a cell wired to the wrong fold surfaces as a year
    mismatch. Also asserts the Stage-1 ckpt's doy window equals the pest geometry -- that
    equality is what makes `offset = issue - alert` well defined downstream.
    """
    print("\n[8] Stage-1 checkpoint binding (per cell, per year)")
    import torch
    for r in rows:
        p = r["pest"]
        for y in YEARS:
            d, run = _s1_dir(p, y)
            sp, val = SPLIT_OF[y]
            if d is None:
                fail(f"{p}/{y}: no Stage-1 run dir {sp}_v{val}_t{y}")
                continue
            if f"_v{val}_t{y}" not in d.name:
                fail(f"{p}/{y}: bound to {d.name}, expected _v{val}_t{y}")
                continue
            try:
                c = torch.load(d / "A/ckpt/event_xgb_w28_lead14-45_A.pt",
                               map_location="cpu", weights_only=False)
            except Exception as e:                                   # noqa: BLE001
                fail(f"{p}/{y}: ckpt unreadable {type(e).__name__}"); continue
            ds, de = c.get("doy_start"), c.get("doy_end")
            if (ds, de) != (r["doy_start"], r["doy_end"]):
                fail(f"{p}/{y}: Stage-1 doy ({ds},{de}) != pests.tsv "
                     f"({r['doy_start']},{r['doy_end']}) -> offset frame mismatch")
            else:
                ok(f"{p}/{y}: run{run}/{d.name} doy={ds}-{de} T={c.get('T')}")


def check_gate_policy(rows):
    """(2) Record the gate kind / tau / k each cell actually fired on.

    Production consumes the R>=0.88 selection (gate CSVs are named *_R088_*) and the batch
    driver sweeps only KS_DISPATCH="3", so k should be 3 everywhere -- but we read it rather
    than assume. group_tau carries TWO taus (no-history / with-history).
    """
    print("\n[9] Stage-1 gate policy (kind / tau / k)")
    recs = []
    for r in rows:
        p = r["pest"]
        for y in YEARS:
            d, _ = _s1_dir(p, y)
            g = sorted((CS / f"rice/outputs/stage2/{batch_dir(y)}/{p}").glob(
                "gate_*_R088_features_per_sy.csv"))
            gate = g[0].name.replace("gate_", "").replace("_R088_features_per_sy.csv", "") if g else "?"
            tau, k = "n/a", "n/a"
            js = (d / "group_tau/group_tau_hybrid_summary.json") if d else None
            if js and js.exists():
                sel = json.loads(js.read_text()).get("selections", {}).get("R>=0.88", {})
                node = sel.get(gate) or sel.get("dispatch_group_tau", {})
                k = node.get("k", "n/a")
                tau = (f"no={node['tau_no']},with={node['tau_with']}"
                       if "tau_no" in node else node.get("tau", "n/a"))
            recs.append(dict(pest=p, year=y, gate=gate, tau=tau, k=k))
            ok(f"{p}/{y}: gate={gate} tau={tau} k={k}")
    ks = {str(x["k"]) for x in recs if x["k"] != "n/a"}
    if ks - {"3"}:
        warn(f"k is NOT uniformly 3: {sorted(ks)}")
    else:
        ok(f"k == 3 in all cells with a recorded policy (KS_DISPATCH=3 in run_stage1_batch_pests.sh)")
    pd.DataFrame(recs).to_csv(AP / "gate_policy_observed.csv", index=False)


def check_stage1_freeze(rows):
    """7 pests reuse frozen Stage-1 output; BPH is scheduled for a DOY 60-300 regeneration."""
    print("\n[10] Stage-1 freeze policy")
    man = AP / "stage1_freeze_manifest.csv"
    if not man.exists():
        warn(f"freeze manifest absent ({man.name}) -- run make_stage1_freeze_manifest.py")
        return
    import hashlib

    def sha(p, chunk=1 << 20):
        h = hashlib.sha256()
        with open(p, "rb") as f:
            for b in iter(lambda: f.read(chunk), b""):
                h.update(b)
        return h.hexdigest()[:16]

    m = pd.read_csv(man)
    for r in rows:
        sub = m[m.pest == r["pest"]]
        if sub.empty:
            fail(f"{r['pest']}: absent from freeze manifest"); continue
        pol = str(sub.iloc[0]["policy"])
        # recompute live so a drifted artifact cannot hide behind a stale manifest column
        drift = sum(1 for _, x in sub.iterrows()
                    if x["exists"] and (CS / x["path"]).exists()
                    and sha(CS / x["path"]) != x["sha256_recorded"])
        if r["pest"] == "BPH" and pol != "REGENERATE":
            fail(f"BPH policy={pol}, expected REGENERATE")
        elif r["pest"] != "BPH" and pol != "FROZEN":
            fail(f"{r['pest']} policy={pol}, expected FROZEN")
        elif drift:
            fail(f"{r['pest']}: {drift} frozen artifact(s) changed since the manifest was written")
        else:
            ok(f"{r['pest']}: {pol} ({len(sub)} artifacts verified)")


def check_vendor():
    """(12) The pinned dependencies must be present and byte-identical to the manifest.

    This is the gate that makes two servers provably run the same code. A missing or drifted
    vendored file is a FAIL, which stops training. The external workspace is never imported;
    if it exists here its comparison is printed as information only.
    """
    print("\n[12] vendored dependency pin")
    import vendor_check
    ok, fails, notes = vendor_check.verify(quiet=True)
    if not ok:
        for f in fails:
            fail(f"vendor: {f}")
    else:
        ok_(f"all {len(notes)} pinned files match VENDOR_MANIFEST.csv")
    drift = [n for n in notes if n.startswith("DRIFT")]
    absent = [n for n in notes if n.startswith("absent")]
    if absent:
        ok_(f"external workspace not present for {len(absent)} file(s) -- expected on server 2")
    if drift:
        warn(f"external workspace has drifted from the pin on {len(drift)} file(s) "
             f"(informational; the pinned copy is authoritative): {drift[:2]}")


def check_uniformity(rows):
    """Knobs that MUST be identical across all 24 cells, and the ones that legitimately vary.

    The run is only a fair cross-pest comparison if the first group never drifts. The second
    group is inherited per-cell on purpose (see README "Uniformity") and is reported, not failed.
    """
    print("\n[7] cross-cell uniformity")
    MUST_MATCH = ["d_model_override", "stage2_pmf_sigma", "stage2_pmf_mu_mode", "split_mode",
                  "split_seed", "stage2_pmf_target_mode", "stage2_pmf_asym_weight",
                  "stage2_pmf_asym_weight_early", "stage2_nowcast_window"]
    seen: dict[str, dict[str, list[str]]] = {k: {} for k in MUST_MATCH}
    varies: dict[str, set] = {"batch_train_override": set(), "gate_csv_kind": set()}
    for r in rows:
        for y in YEARS:
            res = subprocess.run(["bash", "-c", f'source {AP}/common.sh; extract_args {r["pest"]} {y}'],
                                 capture_output=True, text=True)
            t = res.stdout.split()
            def val(k):
                return t[t.index(f"--{k}") + 1] if f"--{k}" in t else "<absent>"
            for k in MUST_MATCH:
                seen[k].setdefault(val(k), []).append(f'{r["pest"]}/{y}')
            varies["batch_train_override"].add(val("batch_train_override"))
            g = val("stage2_dispatch_feature_csv").rsplit("/", 1)[-1]
            varies["gate_csv_kind"].add(g.replace("_R088_features_per_sy.csv", ""))
    for k in MUST_MATCH:
        if len(seen[k]) == 1:
            ok(f"{k} = {next(iter(seen[k]))} across all 24 cells")
        else:
            detail = {v: (c[:3] + ["..."] if len(c) > 3 else c) for v, c in seen[k].items()}
            fail(f"{k} NOT uniform: {detail}")
    for k, v in varies.items():
        warn(f"{k} varies by design: {sorted(v)}")
    geo = {(r["doy_start"], r["doy_end"]) for r in rows}
    if len(geo) > 1:
        warn(f"DOY geometry NOT uniform: {sorted(geo)} -- "
             f"{[r['pest'] for r in rows if (r['doy_start'], r['doy_end']) != (60, 300)]} "
             f"differ; their IoU is not directly comparable to the 60-300 pests")


def check_server_balance(rows):
    print("\n[6] server split")
    for s in (1, 2):
        sub = [r for r in rows if r["server"] == s]
        ok(f"server{s}: {len(sub)} pests {[r['pest'] for r in sub]} "
           f"cost~{sum(r['cost'] for r in sub)}")
    c1 = sum(r["cost"] for r in rows if r["server"] == 1)
    c2 = sum(r["cost"] for r in rows if r["server"] == 2)
    if max(c1, c2) > 1.5 * min(c1, c2):
        warn(f"server cost imbalance {c1:.0f} vs {c2:.0f} (>1.5x)")
    if len(rows) != len({r["pest"] for r in rows}):
        fail("duplicate pest in pests.tsv")


# ---------------------------------------------------------------- full checks
def check_forward(rows, year=2024):
    """Build the real samples for one year, then forward the E5d-geometry model over them.

    Data spec comes from that pest's PRODUCTION ckpt, so this exercises the true per-pest
    feature width. The production ckpt has no neighbor channels, so the E5d model we build
    is widened by 6 -- that +6 is exactly what --stage2_add_neighbor_history will add.
    """
    print(f"\n[11] data build + model forward (year {year})")
    import functools, torch
    from rice.configs import config as C
    from rice.src.pest_resolver import resolve_pest
    import rice.scripts.eval_s2n_direct_compare as E
    from src.vendor.model import HierarchicalCausalHazardTransformer as VModel

    for r in rows:
        p = r["pest"]
        ckpt_p = CS / f"rice/outputs/stage2/{batch_dir(year)}/{p}/lead_v3_final/ckpt/checkpoint_run4.pt"
        try:
            ck = torch.load(ckpt_p, map_location="cpu", weights_only=False)
            _, gfc = resolve_pest(p)
            C.DOY_START = int(ck.get("doy_start", 60)); C.DOY_END = int(ck.get("doy_end", 300))
            if ck.get("d_model"): C.D_MODEL = int(ck["d_model"])
            T = C.DOY_END - C.DOY_START + 1
            from rice.scripts.eval_s2n_direct_compare import reconstruct_samples
            samples, _ = reconstruct_samples(ck, int(ck["run"]), gfc)
            if not samples:
                fail(f"{p}: reconstruct_samples returned 0 samples"); continue
            d_prod = int(samples[0]["X"].shape[1])
            if d_prod != len(ck["norm_mean"]):
                fail(f"{p}: d_in {d_prod} != ckpt norm {len(ck['norm_mean'])}"); continue
            d_e5d = d_prod + 6                      # + neighbor history

            model = VModel(d_e5d,
                           d_model=int(ck.get("d_model", 48)),
                           nhead=int(ck.get("n_head", 4)),
                           num_layers=int(ck.get("n_layers", 3)),
                           max_len=max(400, T + 8),
                           use_shared_multi_offset=True, mu_head_mode="offset_specific",
                           candidate_offsets=OFFSETS,
                           shared_band_window=int(ck.get("stage2_nowcast_window", 28)),
                           use_issue_doy_features=False)
            model.pmf_mode = "gaussian"
            model.eval()
            # contract at model.py:482 -- X (B,K,T,D), tstar (B,K) in 1..T, valid_mask (B,K)
            B, K = 2, len(OFFSETS)
            X = torch.zeros(B, K, T, d_e5d)
            tstar = torch.full((B, K), T // 2, dtype=torch.long)
            vm = torch.ones(B, K, dtype=torch.bool)
            with torch.no_grad():
                hazard = model(X, tstar, vm, None)
            mu = getattr(model, "_last_mu_BK", None)
            if hazard.shape != (B, K, T):
                fail(f"{p}: hazard shape {tuple(hazard.shape)} != {(B, K, T)}")
            elif not torch.isfinite(hazard).all():
                fail(f"{p}: hazard has non-finite values")
            elif mu is None or not torch.isfinite(mu).all():
                fail(f"{p}: mu head produced None/non-finite (_last_mu_BK)")
            else:
                ok(f"{p}: n={len(samples)} T={T} d_prod={d_prod}->d_e5d={d_e5d} "
                   f"hazard{tuple(hazard.shape)} mu{tuple(mu.shape)} finite")
        except Exception as e:                       # noqa: BLE001 - report, never abort the sweep
            fail(f"{p}: {type(e).__name__}: {str(e)[:200]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--level", choices=["fast", "full"], default="fast")
    ap.add_argument("--year", type=int, default=2024, help="year used for the forward check")
    a = ap.parse_args()

    rows = registry()
    print(f"=== all-pest E5d dry run (level={a.level}, {len(rows)} pests) ===")
    check_registry_vs_config(rows)
    check_inputs_exist(rows)
    check_arg_extraction(rows)
    check_blast_2023_fallback()
    check_offset_feasibility(rows)
    check_uniformity(rows)
    check_stage1_binding(rows)
    check_gate_policy(rows)
    check_stage1_freeze(rows)
    check_vendor()
    check_server_balance(rows)
    if a.level == "full":
        check_forward(rows, a.year)

    print(f"\n=== {len(FAILS)} FAIL / {len(WARNS)} WARN ===")
    for m in FAILS: print("  FAIL", m)
    for m in WARNS: print("  WARN", m)
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
