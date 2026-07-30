#!/usr/bin/env python
"""Can the existing uncond/pilot checkpoints warm-start an E5d final? Read-only.

S1/S2 require final to warm-start from pilot, and pilot from uncond, with EVERYTHING else
identical to the S0 scratch arm. That is only legitimate if the donor checkpoints are
structurally compatible with the E5d model. This compares, per eval year:

  state_dict keys + tensor shapes    (the actual blocker for load_state_dict)
  d_in / feature width and norm length
  train/val/test split fields
  DOY window
  Stage-1 dispatch CSV the checkpoint was built against
  loss/pmf configuration of each donor stage

Writes rice/outputs_e5d_ablation/WBPH/COMPAT_REPORT.{csv,md}. Loads nothing into a model,
trains nothing, modifies nothing.
"""
from __future__ import annotations
import json, sys
from pathlib import Path
import pandas as pd, torch

CS = Path("/home/gpu4080/research/cropscience")
S2 = CS / "rice/outputs/stage2"
S0 = CS / "rice/outputs_allpests_e5d/WBPH"
OUT = CS / "rice/outputs_e5d_ablation/WBPH"
YEARS = [2022, 2023, 2024]

# Donor chain candidates. direct_neighbor* is the only family trained WITH neighbor history,
# which is what E5d uses (--stage2_add_neighbor_history), so it is the only plausible donor.
UNCOND = {2022: S2 / "WBPH_uncond_split1/ckpt/checkpoint_run4.pt",
          2023: S2 / "WBPH_uncond_split2/ckpt/checkpoint_run4.pt",
          2024: S2 / "WBPH_uncond/ckpt/checkpoint_run4.pt"}
PILOT_DN = {2022: S2 / "direct_neighbor_rolling/2022/WBPH/lead_v3_pilot/ckpt/checkpoint_run4.pt",
            2023: S2 / "direct_neighbor_rolling/2023/WBPH/lead_v3_pilot/ckpt/checkpoint_run4.pt",
            2024: S2 / "direct_neighbor/WBPH/lead_v3_pilot/ckpt/checkpoint_run4.pt"}
PILOT_BATCH = {2022: S2 / "batch_2022_baseline/WBPH/lead_v3_pilot/ckpt/checkpoint_run4.pt",
               2023: S2 / "batch_2023_baseline/WBPH/lead_v3_pilot/ckpt/checkpoint_run4.pt",
               2024: S2 / "batch_2024_bestgate/WBPH/lead_v3_pilot/ckpt/checkpoint_run4.pt"}
TARGET = {y: S0 / f"dev/ckpt/{y}/ckpt/checkpoint_run4.pt" for y in YEARS}


def load(p: Path):
    return torch.load(p, map_location="cpu", weights_only=False) if p.is_file() else None


def shapes(ck) -> dict:
    sd = ck.get("model_state") or ck.get("state_dict") or ck.get("model_state_dict")
    if sd is None:
        for k, v in ck.items():
            if isinstance(v, dict) and v and all(hasattr(t, "shape") for t in list(v.values())[:3]):
                sd = v
                break
    return {k: tuple(t.shape) for k, t in sd.items()} if sd else {}


def meta(ck) -> dict:
    return {k: ck.get(k) for k in
            ("doy_start", "doy_end", "split_seed", "split_mode", "d_model", "n_head", "n_layers",
             "stage2_pmf_mode", "stage2_pmf_mu_mode", "stage2_pmf_asym_weight",
             "stage2_pmf_asym_weight_early", "stage2_nowcast_window",
             "stage2_use_shared_multi_offset", "stage2_mu_head_mode",
             "stage2_dispatch_feature_csv", "run")}


def compare(tag: str, donor_p: Path, tgt_p: Path, year: int) -> dict:
    d, t = load(donor_p), load(tgt_p)
    r = dict(eval_year=year, donor=tag, donor_path=str(donor_p.relative_to(CS)) if donor_p.is_file() else str(donor_p),
             donor_exists=donor_p.is_file(), target_exists=tgt_p.is_file())
    if d is None or t is None:
        r["verdict"] = "MISSING"
        return r
    ds, ts = shapes(d), shapes(t)
    dm, tm = meta(d), meta(t)
    r["donor_n_params"] = len(ds); r["target_n_params"] = len(ts)
    only_t = sorted(set(ts) - set(ds)); only_d = sorted(set(ds) - set(ts))
    mism = sorted(k for k in (set(ts) & set(ds)) if ts[k] != ds[k])
    r["missing_in_donor"] = len(only_t); r["unexpected_in_donor"] = len(only_d)
    r["shape_mismatch"] = len(mism)
    r["missing_example"] = "; ".join(only_t[:3])
    r["mismatch_example"] = "; ".join(f"{k} {ds[k]}!={ts[k]}" for k in mism[:3])
    r["donor_norm_len"] = len(d.get("norm_mean", []) or [])
    r["target_norm_len"] = len(t.get("norm_mean", []) or [])
    for k in ("doy_start", "doy_end", "split_seed", "split_mode", "d_model",
              "stage2_pmf_mode", "stage2_pmf_mu_mode", "stage2_use_shared_multi_offset",
              "stage2_mu_head_mode", "stage2_nowcast_window"):
        r[f"donor_{k}"] = dm.get(k); r[f"target_{k}"] = tm.get(k)
    dd = str(dm.get("stage2_dispatch_feature_csv") or "")
    td = str(tm.get("stage2_dispatch_feature_csv") or "")
    r["dispatch_same"] = (Path(dd).name == Path(td).name) if dd and td else False
    r["donor_dispatch"] = Path(dd).name if dd else ""
    r["target_dispatch"] = Path(td).name if td else ""

    blockers = []
    if r["donor_norm_len"] != r["target_norm_len"]:
        blockers.append(f"d_in {r['donor_norm_len']}!={r['target_norm_len']}")
    if r["missing_in_donor"]:
        blockers.append(f"{r['missing_in_donor']} keys absent in donor")
    if r["shape_mismatch"]:
        blockers.append(f"{r['shape_mismatch']} shape mismatches")
    if (dm.get("doy_start"), dm.get("doy_end")) != (tm.get("doy_start"), tm.get("doy_end")):
        blockers.append("DOY window differs")
    if dm.get("split_seed") != tm.get("split_seed") or dm.get("split_mode") != tm.get("split_mode"):
        blockers.append("split differs")
    if not r["dispatch_same"]:
        blockers.append("dispatch CSV differs")
    r["blockers"] = "; ".join(blockers)
    r["verdict"] = "COMPATIBLE" if not blockers else "INCOMPATIBLE"
    return r


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for y in YEARS:
        rows.append(compare("uncond", UNCOND[y], TARGET[y], y))
        rows.append(compare("pilot_direct_neighbor", PILOT_DN[y], TARGET[y], y))
        rows.append(compare("pilot_batch_baseline", PILOT_BATCH[y], TARGET[y], y))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "COMPAT_REPORT.csv", index=False)

    cols = ["eval_year", "donor", "donor_norm_len", "target_norm_len", "missing_in_donor",
            "unexpected_in_donor", "shape_mismatch", "dispatch_same", "verdict"]
    pd.set_option("display.width", 220)
    print(df[cols].to_string(index=False))
    print("\n=== blockers ===")
    for _, r in df.iterrows():
        print(f"  {r['eval_year']} {r['donor']:24} {r['verdict']:12} {r['blockers']}")
    print("\n=== donor stage configuration ===")
    for _, r in df.iterrows():
        print(f"  {r['eval_year']} {r['donor']:24} pmf={r['donor_stage2_pmf_mode']} "
              f"mu={r['donor_stage2_pmf_mu_mode']} shared_multi_offset={r['donor_stage2_use_shared_multi_offset']} "
              f"head={r['donor_stage2_mu_head_mode']} doy=({r['donor_doy_start']},{r['donor_doy_end']})")
    n_ok = int((df.verdict == "COMPATIBLE").sum())
    print(f"\nCOMPATIBLE {n_ok} / {len(df)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
