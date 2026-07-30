#!/usr/bin/env python
"""Record the Stage-1 artifacts each pest is pinned to, and its freeze policy.

  FROZEN     (7 pests) -- reuse the existing batch_rolling output untouched. The sha256 of
                          every pinned artifact is recorded here so dry_run.py can prove
                          nothing drifted between the two servers or between runs.
  REGENERATE (BPH)     -- the existing Stage-1 was fit on DOY 140-270; it is being rebuilt at
                          60-300 so BPH lands on the same axis as the other seven. Until that
                          rebuild lands, BPH's rows point at the OLD artifacts and are marked
                          so, which is why dry_run refuses to treat BPH as comparable.

Writes stage1_freeze_manifest.csv next to this file. Reads only; touches no Stage-1 output.
"""
from __future__ import annotations
import argparse, hashlib, sys
from pathlib import Path
import pandas as pd

from repo_paths import AP, CS        # roots derived from this file's location
S1 = CS / "rice/outputs/stage1/batch_rolling"
S2 = CS / "rice/outputs/stage2"
SPLIT_OF = {2022: ("split1", 2021), 2023: ("split2", 2022), 2024: ("split3", 2023)}
REGENERATE = {"BPH"}


def sha256(p: Path, chunk=1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(chunk), b""):
            h.update(b)
    return h.hexdigest()[:16]


def batch_dir(y): return "batch_2024_bestgate" if y == 2024 else f"batch_{y}_baseline"


def s1_dir(pest, year):
    sp, val = SPLIT_OF[year]
    for run in (0, 1, 2):
        d = S1 / pest / f"run{run}" / f"{sp}_v{val}_t{year}"
        if (d / "A/ckpt/event_xgb_w28_lead14-45_A.pt").exists():
            return d
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pests", nargs="+", default=None)
    a = ap.parse_args()
    pests = a.pests or [l.split()[0] for l in (AP / "pests.tsv").read_text().splitlines()
                        if l.strip() and not l.lstrip().startswith("#")]

    rows = []
    for p in pests:
        policy = "REGENERATE" if p in REGENERATE else "FROZEN"
        for y in (2022, 2023, 2024):
            d = s1_dir(p, y)
            targets = []
            if d:
                targets += [("stage1_A_ckpt", d / "A/ckpt/event_xgb_w28_lead14-45_A.pt"),
                            ("stage1_D_ckpt", d / "D/ckpt/event_xgb_w28_lead14-45_D.pt"),
                            ("group_tau_summary", d / "group_tau/group_tau_hybrid_summary.json")]
            cell = S2 / batch_dir(y) / p
            g = sorted(cell.glob("gate_*_R088_features_per_sy.csv"))
            if g:
                targets.append(("dispatch_csv", g[0]))
            targets.append(("climatology", cell / "climatology_train_stats.csv"))
            for kind, f in targets:
                rows.append(dict(pest=p, year=y, policy=policy, kind=kind,
                                 path=str(f.relative_to(CS)) if f.exists() else str(f),
                                 exists=f.exists(),
                                 sha256_recorded=sha256(f) if f.exists() else ""))
    df = pd.DataFrame(rows)
    out = AP / "stage1_freeze_manifest.csv"
    df.to_csv(out, index=False)
    miss = df[~df.exists]
    print(df.groupby(["pest", "policy"]).size().to_string())
    print(f"\n[freeze] {len(df)} artifacts, {len(miss)} missing -> {out}")
    if len(miss):
        print(miss[["pest", "year", "kind", "path"]].to_string(index=False))
    print("\nREGENERATE pests still point at their OLD artifacts until the rebuild lands: "
          f"{sorted(REGENERATE)}")


if __name__ == "__main__":
    sys.exit(main())
