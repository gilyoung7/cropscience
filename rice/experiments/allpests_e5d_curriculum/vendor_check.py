#!/usr/bin/env python
"""Verify the vendored dependencies against VENDOR_MANIFEST.csv.

This is the gate: if a vendored file is missing or its sha256 does not match the manifest,
nothing may train. Both servers run the same bytes or neither runs.

The external wbph_interval_perf_202607 workspace is NEVER used for execution. If it happens
to be present this reports whether it has drifted from the pin -- informational only; drift
there is not a failure, because the pinned copy is the authority.

  python vendor_check.py            # exit 0 = ok, 1 = FAIL
  python vendor_check.py --quiet
"""
from __future__ import annotations
import argparse, hashlib, sys
from pathlib import Path
import pandas as pd

CS = Path("/home/gpu4080/research/cropscience")
AP = CS / "rice/experiments/allpests_e5d"
VENDOR = AP / "vendor"
MANIFEST = VENDOR / "VENDOR_MANIFEST.csv"


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def verify(quiet: bool = False) -> tuple[bool, list[str], list[str]]:
    """Returns (ok, failures, external_notes)."""
    if not MANIFEST.exists():
        return False, [f"manifest missing: {MANIFEST}"], []
    man = pd.read_csv(MANIFEST)
    fails, notes = [], []
    for _, r in man.iterrows():
        p = VENDOR / r["vendored_path"]
        if not p.is_file():
            fails.append(f"MISSING {r['vendored_path']}")
            continue
        got = sha256(p)
        if got != r["sha256"]:
            fails.append(f"HASH MISMATCH {r['vendored_path']}: "
                         f"manifest {r['sha256'][:12]} != actual {got[:12]}")
        elif not quiet:
            print(f"  ok  {got[:12]}  {int(r['size_bytes']):>7}  {r['vendored_path']}")
        # informational only -- the external copy is never imported
        ext = Path(str(r["source_abs"]))
        if ext.is_file():
            e = sha256(ext)
            notes.append(f"{'same ' if e == r['sha256'] else 'DRIFT'}  {r['vendored_path']}"
                         + ("" if e == r["sha256"] else f"  external {e[:12]} vs pinned {r['sha256'][:12]}"))
        else:
            notes.append(f"absent {r['vendored_path']}  (external workspace not on this host)")
    return (not fails), fails, notes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args()
    print(f"=== vendor check ({MANIFEST.name}) ===")
    ok, fails, notes = verify(a.quiet)
    if fails:
        print("\n".join(f"  FAIL  {f}" for f in fails))
    print(f"\n[vendor] {'OK' if ok else 'FAIL'} — {len(fails)} problem(s)")
    print("\n=== external workspace comparison (informational; NOT used for execution) ===")
    for n in notes:
        print(f"  {n}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
