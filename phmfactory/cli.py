"""Public command-line interface for PHMFactory."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import yaml

from src.utils.config_utils import parse_overrides

DEFAULT_CONFIG = "configs/demo/01_cross_domain/cwru_dg.yaml"
DEFAULT_PIPELINE = "Pipeline_01_default"


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


def _pipeline_from_yaml(config_path: str) -> str:
    path = Path(config_path)
    if not path.exists():
        return DEFAULT_PIPELINE
    try:
        with path.open("r", encoding="utf-8") as handle:
            config = yaml.safe_load(handle) or {}
    except (OSError, UnicodeDecodeError, yaml.YAMLError):
        return DEFAULT_PIPELINE
    if isinstance(config, dict):
        pipeline = config.get("pipeline")
        if isinstance(pipeline, str) and pipeline.strip():
            return pipeline.strip()
    return DEFAULT_PIPELINE


def _resolve_pipeline(args: argparse.Namespace, config_path: str) -> str:
    pipeline_name = _pipeline_from_yaml(config_path)
    if args.override:
        overrides = parse_overrides(args.override)
        override_pipeline = overrides.get("pipeline")
        if override_pipeline is not None:
            if not isinstance(override_pipeline, str) or not override_pipeline.strip():
                raise ValueError("pipeline override must be a non-empty string")
            pipeline_name = override_pipeline.strip()
    return pipeline_name


def run(args: argparse.Namespace) -> Any:
    """Dispatch a parsed argument namespace to the protected runtime."""
    config_path = _resolve_config_path(args)
    args.config_path = config_path
    pipeline_name = _resolve_pipeline(args, config_path)
    pipeline_module = importlib.import_module(f"src.{pipeline_name}")
    result = pipeline_module.pipeline(args)
    print("完成所有实验！")
    return result


def main(argv: Sequence[str] | None = None) -> Any:
    """Parse command-line arguments and execute the selected Pipeline."""
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)
