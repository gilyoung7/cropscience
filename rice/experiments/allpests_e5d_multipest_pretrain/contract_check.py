#!/usr/bin/env python
"""Gate for multi-pest pretraining: prove the pests share ONE input contract before mixing them.

A shared backbone is only meaningful if every pest presents the encoder with the same tensor.
This reads the PRODUCTION Stage-2 checkpoints -- the same artifacts the E5d scratch run consumes
-- and compares, per pest and year: DOY window, T, d_in, the ordered feature-name list, and the
architecture knobs the backbone shape depends on. Nothing is trained and nothing is written
outside --out.

A pest that disagrees on ANY of these is reported as INELIGIBLE for the shared backbone rather
than silently reshaped. Reshaping is what would make the comparison against the scratch baseline
meaningless.

  .venv/bin/python rice/experiments/allpests_e5d_multipest_pretrain/contract_check.py
  .venv/bin/python .../contract_check.py --out <dir>      # also write contract_report.csv
"""
from __future__ import annotations
import argparse, hashlib, json, sys
from pathlib import Path

import pandas as pd
import torch

AP = Path(__file__).resolve().parent
sys.path.insert(0, str(AP.parent / "allpests_e5d"))
from repo_paths import CS                                    # noqa: E402

YEARS = [2022, 2023, 2024]
# The 8 pests the multi-pest experiment targets, in pests.tsv order.
PESTS = ["WBPH", "brown_spot", "rice_stem_borer_2", "rice_stem_borer_1",
         "sheath_blight", "blast", "BPH", "bacterial_blight"]

# Knobs the backbone's parameter shapes depend on. A mismatch here means the state_dict cannot
# be transferred without reshaping, so it disqualifies the pest from the shared backbone.
ARCH_KEYS = ["d_model", "n_head", "n_layers", "stage2_nowcast_window"]


def batch_dir(y: int) -> str:
    return "batch_2024_bestgate" if y == 2024 else f"batch_{y}_baseline"


def ckpt_path(pest: str, year: int) -> Path:
    return (CS / f"rice/outputs/stage2/{batch_dir(year)}/{pest}"
            / "lead_v3_final/ckpt/checkpoint_run4.pt")


def feature_hash(names: list[str]) -> str:
    """Order-sensitive digest of the feature contract -- two pests match only if the names AND
    their positions agree, because position is what the input projection's columns mean."""
    return hashlib.sha256("|".join(names).encode()).hexdigest()[:16]


def read_contract(pest: str, year: int) -> dict:
    p = ckpt_path(pest, year)
    if not p.is_file():
        return dict(pest=pest, year=year, status="MISSING_CKPT", path=str(p))
    c = torch.load(p, map_location="cpu", weights_only=False)
    names = list(c.get("feature_names") or [])
    row = dict(pest=pest, year=year, status="ok",
               doy_start=int(c.get("doy_start", -1)), doy_end=int(c.get("doy_end", -1)),
               T=int(c.get("doy_end", 0)) - int(c.get("doy_start", 0)) + 1,
               d_in=len(c.get("norm_mean", [])), n_feature_names=len(names),
               n_feature_cols=len(c.get("feature_cols") or []),
               feature_hash=feature_hash(names))
    for k in ARCH_KEYS:
        row[k] = c.get(k)
    return row


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="directory for contract_report.csv")
    ap.add_argument("--reference", default="WBPH", help="pest whose contract defines the group")
    a = ap.parse_args()

    rows = [read_contract(p, y) for p in PESTS for y in YEARS]
    df = pd.DataFrame(rows)
    missing = df[df.status != "ok"]
    ok = df[df.status == "ok"]

    print(f"=== multi-pest input contract ({len(PESTS)} pests x {len(YEARS)} years) ===")
    if not missing.empty:
        print(f"\n[!] {len(missing)} cell(s) have no production checkpoint -- cannot be judged:")
        for _, r in missing.iterrows():
            print(f"  MISSING  {r['pest']}/{r['year']}  {r['path']}")
    if ok.empty:
        print("\n[contract] no readable checkpoints; nothing to compare")
        return 1

    cols = ["pest", "year", "doy_start", "doy_end", "T", "d_in", "n_feature_cols",
            "feature_hash"] + ARCH_KEYS
    print()
    print(ok[cols].to_string(index=False))

    ref = ok[ok.pest == a.reference]
    if ref.empty:
        print(f"\n[contract] reference pest {a.reference} unreadable")
        return 1
    r0 = ref.iloc[0]
    key = ["doy_start", "doy_end", "T", "d_in", "feature_hash"] + ARCH_KEYS

    eligible, ineligible = [], {}
    for pest in PESTS:
        sub = ok[ok.pest == pest]
        if sub.empty:
            ineligible[pest] = ["no readable checkpoint"]
            continue
        # a pest must be self-consistent across its own years AND match the reference
        diffs = sorted({k for _, r in sub.iterrows() for k in key if r[k] != r0[k]})
        if len(sub) != len(YEARS):
            diffs.append(f"only {len(sub)}/{len(YEARS)} years readable")
        if diffs:
            ineligible[pest] = diffs
        else:
            eligible.append(pest)

    print(f"\n=== verdict (reference = {a.reference}) ===")
    print(f"  contract: DOY {r0['doy_start']}-{r0['doy_end']}  T={r0['T']}  d_in={r0['d_in']}  "
          f"feature_hash={r0['feature_hash']}  "
          + "  ".join(f"{k}={r0[k]}" for k in ARCH_KEYS))
    print(f"\n  ELIGIBLE for the shared backbone ({len(eligible)}): {eligible}")
    for pest, diffs in ineligible.items():
        print(f"  INELIGIBLE {pest}: disagrees on {diffs}")

    if a.out:
        d = Path(a.out); d.mkdir(parents=True, exist_ok=True)
        df.to_csv(d / "contract_report.csv", index=False)
        (d / "contract_verdict.json").write_text(json.dumps(
            dict(reference=a.reference, contract={k: _j(r0[k]) for k in key},
                 eligible=eligible, ineligible=ineligible), indent=2, ensure_ascii=False))
        print(f"\n  wrote {d/'contract_report.csv'} and {d/'contract_verdict.json'}")

    # Non-zero only if the eligible set is too small to pretrain on.
    return 0 if len(eligible) >= 2 else 1


def _j(v):
    return int(v) if hasattr(v, "item") or isinstance(v, bool) else v


if __name__ == "__main__":
    sys.exit(main())
