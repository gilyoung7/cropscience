#!/usr/bin/env python
"""Freeze the completed E5d scratch (S0) result for WBPH into a read-only manifest + metrics.

Copies nothing out of place and OVERWRITES NOTHING: it reads the existing
rice/outputs_allpests_e5d/WBPH tree and writes a new, separate snapshot under
rice/outputs_e5d_ablation/WBPH/S0_scratch_baseline/. Refuses to run if that
snapshot already exists (use --force only to deliberately re-snapshot).

S0 is the reference arm of the curriculum ablation:
  random init, gaussian / lead_from_alert, asym_weight=25, asym_weight_early=0,
  NO warm-start chain (scripts/93 and common.sh strip it).
"""
from __future__ import annotations
import argparse, hashlib, json, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

CS = Path("/home/gpu4080/research/cropscience")
SRC = CS / "rice/outputs_allpests_e5d/WBPH"
DST = CS / "rice/outputs_e5d_ablation/WBPH/S0_scratch_baseline"
YEARS = [2022, 2023, 2024]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    if DST.exists() and not a.force:
        raise SystemExit(f"[abort] snapshot already exists: {DST} (use --force to redo)")
    if not SRC.exists():
        raise SystemExit(f"[abort] S0 source missing: {SRC}")
    DST.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    # every artifact that defines the S0 arm: checkpoints, grids, splits, eval, logs
    for p in sorted(SRC.rglob("*")):
        if not p.is_file():
            continue
        rows.append(dict(rel_path=str(p.relative_to(SRC)), size_bytes=p.stat().st_size,
                         sha256=sha256(p),
                         mtime=datetime.fromtimestamp(p.stat().st_mtime,
                                                      timezone.utc).isoformat(timespec="seconds")))
    man = pd.DataFrame(rows)
    man.insert(0, "arm", "S0_scratch_baseline")
    man["snapshot_at"] = stamp
    man["source_root"] = str(SRC)
    man.to_csv(DST / "S0_MANIFEST.csv", index=False)

    # copy the small result files so the arm is self-describing without touching the source
    for rel in ["eval/dev_fold_metrics.csv", "eval/dev_pooled_metrics.csv",
                "eval/clean_fold_metrics.csv", "eval/clean_pooled_metrics.csv",
                "eval/clean_chosen_shifts.json", "eval/clean_protocol_assertions.json",
                "eval/eval_log.txt", "clean/split_assignment.json",
                "clean/split_manifest.csv", "clean/split_overlap_checks.json"]:
        s = SRC / rel
        if s.is_file():
            d = DST / rel
            d.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(s, d)

    # the comparison table the ablation will extend
    met = []
    for proto in ("dev", "clean"):
        f = SRC / f"eval/{proto}_fold_metrics.csv"
        if not f.is_file():
            continue
        d = pd.read_csv(f)
        for _, r in d.iterrows():
            met.append(dict(arm="S0_scratch_baseline", protocol=proto,
                            calib=r.get("calib", "calibrated"),
                            eval_year=int(r["eval_year"]), n=int(r["n"]),
                            shift=int(r["shift"]),
                            IoU80_tol0=float(r["IoU80_overall_tol0"]),
                            IoU80_tol1=float(r.get("IoU80_overall_tol1", float("nan"))),
                            MAE_center=float(r.get("MAE_center", float("nan"))),
                            PI_hit=float(r.get("PI_hit", float("nan"))),
                            coverage=float(r.get("coverage", float("nan"))),
                            oracle_iou=float(r.get("oracle_iou", float("nan"))),
                            center_bias=float(r.get("center_bias", float("nan")))))
    pd.DataFrame(met).to_csv(DST / "S0_metrics.csv", index=False)

    # best epoch per cell, straight from the training logs (not re-derived)
    be = []
    import re
    for phase in ("dev", "clean"):
        for y in YEARS:
            lg = SRC / f"{phase}/ckpt/{y}/train.log"
            if not lg.is_file():
                continue
            m = re.findall(r"\[seed 0\] DONE \| best_epoch=(\d+) \| best_val_iou80=([0-9.]+)",
                           lg.read_text(errors="ignore"))
            if m:
                be.append(dict(arm="S0_scratch_baseline", phase=phase, eval_year=y,
                               best_epoch=int(m[-1][0]), best_val_iou80=float(m[-1][1]),
                               warm_start="none (scratch)"))
    pd.DataFrame(be).to_csv(DST / "S0_best_epochs.csv", index=False)

    print(f"[S0] manifest {len(man)} files -> {DST/'S0_MANIFEST.csv'}")
    print(f"[S0] metrics  {len(met)} rows  -> {DST/'S0_metrics.csv'}")
    print(f"[S0] epochs   {len(be)} rows   -> {DST/'S0_best_epochs.csv'}")
    print(f"[S0] source left untouched: {SRC}")


if __name__ == "__main__":
    sys.exit(main())
