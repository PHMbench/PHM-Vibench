"""Dataset-bundle command surface."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phmfactory data",
        description="Download and validate PHMFactory dataset bundles.",
    )
    commands = parser.add_subparsers(dest="data_command", required=True)

    download = commands.add_parser("download", help="Download one versioned bundle.")
    download.add_argument("--bundle", default="cwru-demo-v1")
    download.add_argument(
        "--source",
        choices=("huggingface", "modelscope"),
        default="huggingface",
    )
    download.add_argument("--destination", default=None)
    download.add_argument("--revision", default=None)
    download.add_argument("--force", action="store_true")

    validate = commands.add_parser("validate", help="Validate a local bundle.")
    validate.add_argument("--bundle", default="cwru-demo-v1")
    validate.add_argument("--path", required=True)

    compare = commands.add_parser(
        "compare",
        help="Require identical core-file hashes for two local bundles.",
    )
    compare.add_argument("--bundle", default="cwru-demo-v1")
    compare.add_argument("--left", required=True)
    compare.add_argument("--right", required=True)
    return parser


def run(argv: Sequence[str]) -> Any:
    from phmfactory.data_sources import (
        compare_bundle_hashes,
        download_bundle,
        load_bundle_spec,
        validate_bundle,
    )

    args = build_parser().parse_args(list(argv))
    spec = load_bundle_spec(args.bundle)

    if args.data_command == "download":
        result = download_bundle(
            args.bundle,
            source=args.source,
            destination=args.destination,
            revision=args.revision,
            force=args.force,
        )
        validation = result.validation
        print(f"bundle={validation.spec.bundle_id}")
        print(f"provider={result.provider}")
        print(f"revision={result.requested_revision}")
        print(f"path={result.directory}")
        print(f"selected_rows={validation.selected_rows}")
        print(f"corpus_present={str(validation.corpus_present).lower()}")
        return result

    if args.data_command == "validate":
        validation = validate_bundle(args.path, spec=spec)
        print(f"bundle={validation.spec.bundle_id}")
        print(f"path={validation.directory}")
        print(f"metadata_rows={validation.metadata_rows}")
        print(f"selected_rows={validation.selected_rows}")
        print(f"signal_keys={validation.signal_keys}")
        print(f"corpus_present={str(validation.corpus_present).lower()}")
        return validation

    hashes = compare_bundle_hashes(args.left, args.right, spec=spec)
    for name, digest in hashes.items():
        print(f"{name}={digest}")
    return hashes
