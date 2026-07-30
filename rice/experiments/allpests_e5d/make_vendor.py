#!/usr/bin/env python
"""Copy the 10 external code dependencies into vendor/ and write VENDOR_MANIFEST.csv.

Run ONCE from a machine that has the wbph_interval_perf_202607 workspace. Afterwards both
servers execute the vendored copies only -- there is no fallback to the external workspace,
by design: a fallback is exactly how two servers end up running different code.

The layout mirrors the import names so `sys.path.insert(0, vendor)` resolves them:
    vendor/src/...            -> `from src.io_utils import ...`
    vendor/src/vendor/...     -> `from src.vendor.model import ...`
    vendor/scripts/87_...py   -> loaded by path (importlib)
    vendor/_patched_train.py  -> invoked by path (clean-fold training)
"""
from __future__ import annotations
import argparse, hashlib, shutil, sys
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

from repo_paths import AP, CS, VENDOR, WS as WS_DEFAULT   # roots derived from this file's location

# (source path relative to the external workspace, destination relative to vendor/)
FILES = [
    ("scripts/87_make_shared_offset_grid.py", "scripts/87_make_shared_offset_grid.py"),
    ("src/__init__.py",                        "src/__init__.py"),
    ("src/io_utils.py",                        "src/io_utils.py"),
    ("src/selector_utils.py",                  "src/selector_utils.py"),
    ("src/diagnostics.py",                     "src/diagnostics.py"),
    ("src/eval_metrics.py",                    "src/eval_metrics.py"),
    ("src/vendor/__init__.py",                 "src/vendor/__init__.py"),
    ("src/vendor/model.py",                    "src/vendor/model.py"),
    ("src/vendor/run_train.py",                "src/vendor/run_train.py"),
    ("outputs/feature_experiments/e5d_clean_selection_3fold_20260716/_code/_patched_train.py",
     "_patched_train.py"),
]

NOTES = {
    "scripts/87_make_shared_offset_grid.py":
        "PINNED 2026-07-29 revision: adds the module-level PEST global (default 'WBPH') so the "
        "same collect() can be fanned over all 8 pests. Default value keeps every pre-existing "
        "WBPH result bit-identical. This is INTENTIONALLY not the revision that produced the "
        "original WBPH numbers, though it is behaviourally identical for pest=WBPH.",
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default=str(WS_DEFAULT),
                    help="external workspace to vendor FROM (needed only for this one-off copy)")
    a = ap.parse_args()
    src_root = Path(a.source)
    if not src_root.exists():
        raise SystemExit(f"[abort] source workspace not found: {src_root}")

    stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
    rows = []
    for rel_src, rel_dst in FILES:
        s = src_root / rel_src
        if not s.is_file():
            raise SystemExit(f"[abort] missing dependency: {s}")
        d = VENDOR / rel_dst
        d.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(s, d)
        rows.append(dict(vendored_path=rel_dst, sha256=sha256(d), size_bytes=d.stat().st_size,
                         source_abs=str(s), source_mtime=datetime.fromtimestamp(
                             s.stat().st_mtime, timezone.utc).isoformat(timespec="seconds"),
                         vendored_at=stamp, note=NOTES.get(rel_src, "")))
        print(f"  {rows[-1]['sha256'][:12]}  {rows[-1]['size_bytes']:>7}  {rel_dst}")

    df = pd.DataFrame(rows)
    df.to_csv(VENDOR / "VENDOR_MANIFEST.csv", index=False)
    print(f"\n[vendor] {len(df)} files -> {VENDOR}")
    print(f"[vendor] manifest -> {VENDOR/'VENDOR_MANIFEST.csv'}  (vendored_at={stamp})")


if __name__ == "__main__":
    sys.exit(main())
