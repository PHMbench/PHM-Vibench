"""Public command-line interface for PHMFactory."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from typing import Any

from phmfactory.config import DEFAULT_CONFIG, resolve_config
from phmfactory.pipelines import pipeline_module_name


def build_parser() -> argparse.ArgumentParser:
    """Build the single parser shared by all supported public entrypoints."""
    parser = argparse.ArgumentParser(
        prog="phmfactory",
        description="PHMFactory task pipeline",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Configuration path or maintained preset name.",
    )
    parser.add_argument(
        "--config_path",
        type=str,
        default=None,
        help="Deprecated alias for --config.",
    )
    parser.add_argument(
        "--notes",
        type=str,
        default="",
        help="Experiment notes.",
    )
    parser.add_argument(
        "--override",
        action="append",
        help="Configuration override in key=value form; may be repeated.",
    )
    return parser


def _resolve_config_path(args: argparse.Namespace) -> str:
    if args.config is not None:
        return args.config
    if args.config_path is not None:
        return args.config_path
    return DEFAULT_CONFIG


def _resolve_pipeline(args: argparse.Namespace, config_path: str) -> str:
    """Resolve the canonical Pipeline through the public config API."""
    return resolve_config(
        config_path,
        override_values=args.override,
    ).pipeline


def run(args: argparse.Namespace) -> Any:
    """Dispatch a parsed argument namespace to the protected runtime."""
    requested_config = _resolve_config_path(args)
    resolved = resolve_config(requested_config, override_values=args.override)

    # Preserve the historical downstream contract: Pipelines receive the source
    # requested by the user rather than a generated temporary file.
    args.config_path = requested_config
    args.resolved_config_path = str(resolved.path)
    args.resolved_pipeline = resolved.pipeline

    pipeline_module = importlib.import_module(
        pipeline_module_name(resolved.pipeline, warn=False)
    )
    result = pipeline_module.pipeline(args)
    print("完成所有实验！")
    return result


def main(argv: Sequence[str] | None = None) -> Any:
    """Parse command-line arguments and execute the selected Pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)
