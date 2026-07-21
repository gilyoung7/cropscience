"""Request-scoped input resolution and output-collision safety.

Two operational concerns live here, kept out of the prediction code:

1. Where the request's data files come from. By default they are the fixed
   filenames inside --input-dir, exactly as before. A request may instead name
   an explicit path, so a server can point every request at one shared master
   file rather than staging a copy per request. Nothing here ever writes to an
   input path.

2. Whether it is safe to write into --output-dir. The output filenames are
   fixed (response.json / predictions.csv / run_log.txt), so two requests
   sharing an output directory silently overwrite each other. `plan_output_dir`
   makes that collision explicit instead.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path

# Written by both the single and the batch path.
OUTPUT_FILES = ("response.json", "predictions.csv", "run_log.txt")

DEFAULT_DAILY_NAME = "daily_weather.csv"
DEFAULT_OBS_NAME = "long_observation.csv"


class InputResolutionError(ValueError):
    """A request named an input path that does not exist."""


class OutputCollisionError(RuntimeError):
    """The output directory already holds results from an earlier run."""


def resolve_input_path(explicit: str | os.PathLike | None, *, input_dir: Path,
                       pkg_root: Path, default_rel: str | None = None,
                       label: str = "input") -> Path | None:
    """Resolve one request-scoped input file.

    `explicit` is the request field (e.g. daily_weather_path). Absolute paths are
    used as given; relative paths are tried against the caller's cwd, then
    --input-dir, then the package root — the same order resolve_rep_csv already
    used for the representative-site CSV, so the two behave alike.

    With no `explicit`, falls back to `input_dir / default_rel` (the historical
    fixed-filename layout). Returns None when neither is available, so the
    caller can report a missing file in its own words.
    """
    if explicit:
        p = Path(explicit).expanduser()
        candidates = ([p] if p.is_absolute()
                      else [Path.cwd() / p, input_dir / p, pkg_root / p])
        for c in candidates:
            if c.is_file():
                return c
        raise InputResolutionError(
            f"{label} not found at the requested path {str(explicit)!r}. Tried: "
            + ", ".join(str(c) for c in candidates)
        )
    if default_rel is None:
        return None
    p = input_dir / default_rel
    return p if p.is_file() else None


def resolve_daily(request: dict | None, input_dir: Path, pkg_root: Path) -> Path | None:
    """daily_weather_path from the request, else input_dir/daily_weather.csv."""
    return resolve_input_path(
        (request or {}).get("daily_weather_path"),
        input_dir=input_dir, pkg_root=pkg_root,
        default_rel=DEFAULT_DAILY_NAME, label="daily weather CSV")


def resolve_obs(request: dict | None, input_dir: Path, pkg_root: Path,
                pest: str | None) -> Path | None:
    """long_observation_path from the request, else the two historical layouts.

    Layout A (`long_observation.csv`) still wins over Layout B
    (`LONG_by_pest/RICE_LONG_<pest>.csv`) when no explicit path is given.
    """
    explicit = (request or {}).get("long_observation_path")
    if explicit:
        return resolve_input_path(explicit, input_dir=input_dir, pkg_root=pkg_root,
                                  label="LONG observation CSV")
    local = input_dir / DEFAULT_OBS_NAME
    if local.is_file():
        return local
    p = input_dir / "LONG_by_pest" / f"RICE_LONG_{pest or 'UNKNOWN'}.csv"
    return p if p.is_file() else None


def obs_path_for_message(request: dict | None, input_dir: Path,
                         pest: str | None) -> Path:
    """The path to name in a 'missing input' error when nothing resolved."""
    explicit = (request or {}).get("long_observation_path")
    if explicit:
        return Path(explicit)
    return input_dir / "LONG_by_pest" / f"RICE_LONG_{pest or 'UNKNOWN'}.csv"


# ---------------------------------------------------------------------------
# Output safety
# ---------------------------------------------------------------------------
def existing_outputs(output_dir: Path) -> list[str]:
    """Which of the fixed output filenames already exist."""
    return [n for n in OUTPUT_FILES if (Path(output_dir) / n).is_file()]


def unique_run_dir(output_dir: Path, request: dict | None) -> Path:
    """A per-request subdirectory that cannot collide with a concurrent run.

    Name carries what identifies the run (pest + the time spec) plus a UTC
    timestamp to microseconds and the PID, so two processes started in the same
    microsecond still land in different directories.
    """
    req = request or {}
    parts = [str(req.get("pest") or "run")]
    for key in ("as_of_date", "year", "start_year"):
        v = req.get(key)
        if v is not None:
            parts.append(str(v).replace("-", ""))
            break
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
    parts += [stamp, f"pid{os.getpid()}"]
    return Path(output_dir) / "_".join(parts)


CLAIM_FILE = ".run_claim"

_HELP = (
    "  Choose one:\n"
    "    * point --output-dir at a fresh directory (recommended for "
    "concurrent runs), or\n"
    "    * pass --unique-output-subdir (or \"unique_output_subdir\": true) "
    "to auto-create a per-run subfolder, or\n"
    "    * pass --overwrite (or \"overwrite\": true) to replace them."
)


def _claim(output_dir: Path) -> None:
    """Take exclusive ownership of an output directory, or raise.

    Checking `existing_outputs` alone loses a race: several runs starting
    against the same empty directory all see it empty, all proceed, and the last
    to finish wins silently — which is exactly the bug this module exists to
    prevent (measured: 3 concurrent runs, 3 exit-0, 1 surviving result set).
    O_CREAT|O_EXCL makes the claim itself atomic, so precisely one run wins.
    """
    path = Path(output_dir) / CLAIM_FILE
    payload = (f"pid={os.getpid()}\n"
               f"claimed_utc={datetime.now(timezone.utc).isoformat(timespec='seconds')}\n")
    try:
        fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
    except FileExistsError:
        holder = ""
        try:
            holder = path.read_text(encoding="utf-8").strip().replace("\n", " ")
        except OSError:
            pass
        raise OutputCollisionError(
            f"output directory is claimed by another run ({holder}).\n"
            f"  dir: {output_dir}\n"
            "  Refusing to write into a directory another request is using.\n"
            f"{_HELP}\n"
            f"  If that run died, delete {path} and retry."
        ) from None
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(payload)


def release_claim(output_dir: Path) -> None:
    """Drop this run's claim. Safe to call when no claim is held."""
    try:
        (Path(output_dir) / CLAIM_FILE).unlink()
    except (OSError, FileNotFoundError):
        pass


def plan_output_dir(output_dir: Path, request: dict | None, *,
                    overwrite: bool, unique_subdir: bool) -> tuple[Path, str]:
    """Decide where this run may write. Returns (dir, note_for_the_run_log).

    Default is refuse-on-collision, enforced two ways:
      * finished results already present -> refuse (a completed run's output);
      * directory claimed by a live run   -> refuse (a concurrent run).
    `unique_subdir` sidesteps both by giving every run its own directory;
    `overwrite` is the explicit opt-in to replace.

    The caller must call `release_claim(dir)` when the run ends.
    """
    output_dir = Path(output_dir)
    if unique_subdir:
        d = unique_run_dir(output_dir, request)
        d.mkdir(parents=True, exist_ok=True)
        _claim(d)
        return d, f"output_policy=unique_subdir dir={d}"

    output_dir.mkdir(parents=True, exist_ok=True)
    present = existing_outputs(output_dir)
    if present and not overwrite:
        raise OutputCollisionError(
            f"output directory already contains results: {', '.join(present)}\n"
            f"  dir: {output_dir}\n"
            "  Refusing to overwrite — another request may have written these.\n"
            f"{_HELP}"
        )
    if overwrite:
        # An explicit overwrite still must not collide with a run in flight.
        release_claim(output_dir)
    _claim(output_dir)
    note = f"output_policy={'overwrite' if present else 'fresh'} dir={output_dir}"
    return output_dir, note


def write_atomic(path: Path, text: str, encoding: str = "utf-8",
                 newline: str | None = None) -> None:
    """Write via a temp file in the same directory, then os.replace.

    A crashed or killed run therefore leaves the previous file intact rather
    than a half-written one; os.replace is atomic within a filesystem.
    """
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with open(tmp, "w", encoding=encoding, newline=newline) as f:
        f.write(text)
    os.replace(tmp, path)
