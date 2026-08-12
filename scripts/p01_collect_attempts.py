"""Validate a frozen P01 attempt ledger and emit scorer input selection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from src.utils.p01_attempt_ledger import collect_attempt_ledger, write_collection


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--ledger-sha256", required=True)
    parser.add_argument("--output", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = collect_attempt_ledger(args.ledger, args.ledger_sha256)
    output_path = Path(args.output).resolve()
    output_sha256 = write_collection(output_path, payload)
    print(
        json.dumps(
            {
                "output": str(output_path),
                "output_sha256": output_sha256,
                "selected_attempts": len(payload["selected_attempts"]),
                "failed_attempts": len(payload["failed_attempts"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
