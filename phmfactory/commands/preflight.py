"""Compile and validate a public run without importing or executing training code."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import importlib.util
from typing import Any

from phmfactory.commands.common import (
    add_config_arguments,
    check_writable_directory,
    requested_config,
)
from phmfactory.config import resolve_config
from phmfactory.pipelines import pipeline_module_name, require_pipeline_access
from phmfactory.runtime import CompiledRunSpec


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="phmfactory preflight",
        description="Compile and validate one PHMFactory run without training.",
    )
    add_config_arguments(parser, include_experimental=True)
    return parser


def run(argv: Sequence[str]) -> dict[str, Any]:
    args = build_parser().parse_args(list(argv))
    source = requested_config(args)
    resolved = resolve_config(source, override_values=args.override)
    compiled = CompiledRunSpec.compile(resolved)
    descriptor = require_pipeline_access(
        resolved.pipeline,
        allow_experimental=bool(args.allow_experimental),
        warn=False,
    )
    module_name = pipeline_module_name(resolved.pipeline, warn=False)
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(module_name)

    environment = compiled.config.get("environment")
    output_dir = environment.get("output_dir") if isinstance(environment, dict) else None
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("environment.output_dir is required for preflight")
    writable = check_writable_directory(output_dir)
    result = {
        "status": "passed",
        "requested_config": source,
        "resolved_config_path": str(resolved.path),
        "run_spec_sha256": compiled.sha256,
        "pipeline": compiled.pipeline,
        "pipeline_module": module_name,
        "maturity": descriptor.maturity,
        "output_dir": str(writable),
    }
    for key, value in result.items():
        print(f"{key}={value}")
    return result
