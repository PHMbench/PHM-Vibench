"""Validate one analyzed PHMFactory run without starting training."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
import importlib.util
from typing import Any

from phmfactory.commands.common import (
    add_config_arguments,
    check_writable_directory,
    requested_config,
    requested_local_config,
)
from phmfactory.config import analyze_config
from phmfactory.device import resolve_device_request
from phmfactory.pipelines import pipeline_module_name, require_pipeline_access
from phmfactory.runtime import CompiledRunSpec


def build_parser() -> argparse.ArgumentParser:
    """Build the preflight parser shared by console and module entrypoints."""

    parser = argparse.ArgumentParser(
        prog="phmfactory preflight",
        description="Analyze and validate one PHMFactory run without training.",
    )
    add_config_arguments(parser, include_experimental=True)
    return parser


def run(argv: Sequence[str]) -> dict[str, Any]:
    """Return and print the exact non-training preflight report.

    The function uses the same :func:`phmfactory.config.analyze_config` call as the real
    runtime. It does not import the Pipeline implementation, construct factories, create
    the configured output directory, or start a run. Device resolution uses the same
    lightweight function as real Trainer construction.
    """

    args = build_parser().parse_args(list(argv))
    source = requested_config(args)
    analysis = analyze_config(
        source,
        override_values=args.override,
        local_config=requested_local_config(args),
    )
    errors = [item for item in analysis.diagnostics if item.severity == "error"]
    if errors:
        detail = "; ".join(f"{item.field}: {item.message}" for item in errors)
        raise ValueError(f"configuration analysis failed: {detail}")

    compiled = CompiledRunSpec.compile(analysis.to_resolved_config())
    descriptor = require_pipeline_access(
        analysis.pipeline,
        allow_experimental=bool(args.allow_experimental),
        warn=False,
    )
    module_name = pipeline_module_name(analysis.pipeline, warn=False)
    if importlib.util.find_spec(module_name) is None:
        raise ModuleNotFoundError(module_name)

    environment = analysis.effective_config.get("environment")
    output_dir = environment.get("output_dir") if isinstance(environment, dict) else None
    if not isinstance(output_dir, str) or not output_dir.strip():
        raise ValueError("environment.output_dir is required for preflight")
    writable = check_writable_directory(output_dir)

    trainer_config = analysis.effective_config.get("trainer")
    if not isinstance(trainer_config, dict):
        raise ValueError("trainer configuration must be a mapping for preflight")
    accelerator, devices = resolve_device_request(
        argparse.Namespace(**trainer_config)
    )

    result = {
        "status": "passed",
        "requested_config": source,
        "resolved_config_path": str(analysis.path),
        "local_config_path": (
            str(analysis.local_config_path)
            if analysis.local_config_path is not None
            else "none"
        ),
        "effective_config_sha256": analysis.effective_config_sha256,
        "run_spec_sha256": compiled.sha256,
        "pipeline": analysis.pipeline,
        "pipeline_module": module_name,
        "maturity": descriptor.maturity,
        "output_dir": str(writable),
        "requested_device": str(trainer_config.get("device")),
        "resolved_accelerator": accelerator,
        "resolved_devices": devices,
    }
    for key, value in result.items():
        print(f"{key}={value}")
    return result
