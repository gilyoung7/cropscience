#!/usr/bin/env python
"""Root resolution for the all-pest E5d scripts. Import this instead of hard-coding a root.

Every script in this directory used to open with the first server's absolute paths baked in,
which made the tree unrunnable anywhere else. Both roots are now derived, so a checkout under
any prefix works with no edits:

  CS  the cropscience repo. This file lives at <CS>/rice/experiments/allpests_e5d/repo_paths.py,
      so the repo root is three levels up from this directory. Override with CROPSCIENCE_ROOT.

  WS  the external wbph_interval_perf_202607 workspace. It sits BESIDE the repo on the first
      server, so `CS.parent / "wbph_interval_perf_202607"` reproduces the original value there.
      Override with WBPH_WS_ROOT. It is optional and is NEVER used for execution -- all runtime
      code comes from vendor/. Only capacity_report/smoke_wandb read published WBPH result CSVs
      from it, each already guarded by .exists(), and vendor_check reports drift against it as
      information. So WS is allowed to point at a path that does not exist.

Resolution is validated against repo markers rather than trusted blindly: a stale copy of this
directory sitting outside the repo would otherwise resolve CS to the wrong tree and only fail
much later, with a confusing message.
"""
from __future__ import annotations
import os
from pathlib import Path

# markers that identify the cropscience repo root (present in every checkout, data-independent)
_MARKERS = ("rice/configs", "rice/pests", "rice/src/pest_resolver.py")


def _valid(root: Path) -> bool:
    return all((root / m).exists() for m in _MARKERS)


AP = Path(__file__).resolve().parent                      # <CS>/rice/experiments/allpests_e5d

_env = os.environ.get("CROPSCIENCE_ROOT")
CS = Path(_env).expanduser().resolve() if _env else AP.parents[2]

if not _valid(CS):
    raise SystemExit(
        f"[repo_paths] cannot locate the cropscience repo root.\n"
        f"  resolved to : {CS}  ({'CROPSCIENCE_ROOT' if _env else 'derived from ' + str(AP)})\n"
        f"  expected it to contain: {', '.join(_MARKERS)}\n"
        f"  fix: run the copy of this directory that lives inside the repo, or export "
        f"CROPSCIENCE_ROOT=/path/to/cropscience"
    )

_ws = os.environ.get("WBPH_WS_ROOT")
WS = Path(_ws).expanduser() if _ws else CS.parent / "wbph_interval_perf_202607"

VENDOR = AP / "vendor"

__all__ = ["AP", "CS", "WS", "VENDOR"]


if __name__ == "__main__":                                # `python repo_paths.py` to inspect
    print(f"AP     = {AP}")
    print(f"CS     = {CS}")
    print(f"WS     = {WS}  ({'present' if WS.exists() else 'absent -- optional'})")
    print(f"VENDOR = {VENDOR}")
