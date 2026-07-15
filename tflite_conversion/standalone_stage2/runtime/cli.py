"""CLI for the standalone Stage-2 LiteRT runner.

    python predict.py --pest BPH \
        --daily-csv sample_daily.csv \
        --dispatch-json sample_dispatch.json \
        --variant fp16
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .interpreter import DEFAULT_VARIANT, VARIANTS, Stage2Model, models_root
from .preprocessing import PreprocessError, load_daily_csv
from .schema import DispatchRequest, SchemaError


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="predict.py",
        description="Standalone Stage-2 pest-timing inference (LiteRT/TFLite, no PyTorch).",
    )
    p.add_argument("--pest", help="pest name; omit with --list-pests")
    p.add_argument("--daily-csv", type=Path, help="daily weather CSV (Korean schema)")
    p.add_argument("--dispatch-json", type=Path,
                   help="JSON with pest, alert_tstar, dispatch_features[, site, phenology, year]")
    p.add_argument("--variant", choices=VARIANTS, default=DEFAULT_VARIANT,
                   help=f"model precision (default: {DEFAULT_VARIANT})")
    p.add_argument("--models-dir", type=Path, default=None, help="override models/ location")
    p.add_argument("--output", type=Path, default=None, help="write JSON here instead of stdout")
    p.add_argument("--list-pests", action="store_true", help="list packaged pests and exit")
    p.add_argument("--no-verify-sha256", action="store_true",
                   help="skip re-hashing the model file (faster startup)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = args.models_dir or models_root()

    if args.list_pests:
        pests = sorted(p.name for p in Path(root).iterdir() if (p / "metadata.json").is_file())
        print(json.dumps({"pests": pests, "models_dir": str(root)}, indent=2))
        return 0

    missing = [n for n, v in (("--pest", args.pest), ("--daily-csv", args.daily_csv),
                              ("--dispatch-json", args.dispatch_json)) if not v]
    if missing:
        print(f"error: missing required argument(s): {', '.join(missing)}", file=sys.stderr)
        return 2

    try:
        req = DispatchRequest.from_json(args.dispatch_json)
        if req.pest != args.pest:
            raise SchemaError(
                f"--pest {args.pest!r} does not match dispatch JSON pest {req.pest!r}"
            )
        daily = load_daily_csv(args.daily_csv)
        model = Stage2Model(args.pest, variant=args.variant, models_dir=root,
                            verify_sha256=not args.no_verify_sha256)
        result = model.predict(daily, req)
    except (SchemaError, PreprocessError) as e:
        # Contract violations: actionable message, exit 2, never a silent fallback.
        print(f"error: {e}", file=sys.stderr)
        return 2
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    text = json.dumps(result, indent=2, ensure_ascii=False)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
