"""Public command routing and process entrypoints for PHMFactory.

This module exposes a programmatic API and an operating-system process boundary. Both
consume the same :class:`phmfactory.config.ConfigAnalysis`; neither reparses YAML nor
searches for machine-local configuration after resolution.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from collections.abc import Mapping, Sequence
from typing import Any

from phmfactory.commands.common import (
    add_config_arguments,
    requested_config,
    requested_local_config,
)
from phmfactory.config import analyze_config
from phmfactory.pipelines import pipeline_module_name, require_pipeline_access
from phmfactory.runtime import ExecutionEnvelope


COMMANDS = ("data", "doctor", "demo", "preflight")
DIRECT_OUTPUT_KEYS = (
    "result_dir",
    "best_checkpoint",
    "test_metrics",
    "run_summary",
)


def build_parser() -> argparse.ArgumentParser:
    """Build the backward-compatible experiment parser."""

    parser = argparse.ArgumentParser(
        prog="phmfactory",
        description="PHMFactory task pipeline",
        epilog=(
            "Commands: doctor, demo, preflight, data. "
            "Run an experiment explicitly with phmfactory --config <yaml>."
        ),
    )
    add_config_arguments(parser, include_notes=True, include_experimental=True)
    return parser


def _resolve_config_path(args: argparse.Namespace) -> str:
    """Return the selected config while preserving the deprecated alias."""

    return requested_config(args)


def _resolve_pipeline(args: argparse.Namespace, config_path: str) -> str:
    """Resolve the canonical Pipeline through the public config authority."""

    return analyze_config(
        config_path,
        override_values=args.override,
        local_config=requested_local_config(args),
    ).pipeline


def _print_direct_outputs(result: Any) -> None:
    """Print canonical user outputs returned directly by a maintained Pipeline."""

    if not isinstance(result, Mapping):
        return
    for key in DIRECT_OUTPUT_KEYS:
        value = result.get(key)
        if value is not None:
            print(f"{key}={value}")
    primary_metrics = result.get("primary_metrics")
    if isinstance(primary_metrics, Mapping):
        print(
            "primary_metrics="
            + json.dumps(
                dict(primary_metrics),
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
        )


def run(args: argparse.Namespace) -> Any:
    """Analyze, authorize, execute, and return one Pipeline invocation.

    Configuration composition and schema validation occur exactly once in
    :func:`analyze_config`. The maintained Pipeline receives one mutable copy through
    ``args.resolved_config_data``. No second config object, digest, or YAML loader is
    involved in execution.
    """

    requested = requested_config(args)
    analysis = analyze_config(
        requested,
        override_values=args.override,
        local_config=requested_local_config(args),
    )

    args.requested_config = requested
    args.config_path = str(analysis.path)
    args.resolved_config_path = str(analysis.path)
    args.resolved_pipeline = analysis.pipeline
    args.config_analysis = analysis
    args.resolved_config_data = analysis.runtime_config()

    module_name = pipeline_module_name(analysis.pipeline, warn=False)
    envelope = ExecutionEnvelope(
        pipeline=analysis.pipeline,
        pipeline_module=module_name,
    )
    args.execution_envelope = envelope

    try:
        descriptor = require_pipeline_access(
            analysis.pipeline,
            allow_experimental=bool(getattr(args, "allow_experimental", False)),
            warn=False,
        )
    except BaseException as error:
        envelope.record_failure(error, stage="maturity")
        raise
    args.pipeline_descriptor = descriptor

    try:
        pipeline_module = importlib.import_module(module_name)
    except BaseException as error:
        envelope.record_failure(error, stage="import")
        raise

    result = envelope.execute(pipeline_module, args)
    _print_direct_outputs(result)
    print("完成所有实验！")
    return result


def _run_command(name: str, argv: Sequence[str]) -> Any:
    """Load one small command module only when selected."""

    if name == "data":
        from phmfactory.commands import data

        return data.run(argv)
    if name == "doctor":
        from phmfactory.commands import doctor

        return doctor.run(argv)
    if name == "preflight":
        from phmfactory.commands import preflight

        return preflight.run(argv)
    if name == "demo":
        from phmfactory.commands import demo

        return demo.run(argv, experiment_runner=run)
    raise ValueError(f"unknown PHMFactory command: {name!r}")


def main(argv: Sequence[str] | None = None) -> Any:
    """Return the structured result of a named command or explicit experiment."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if not arguments:
        build_parser().print_help()
        return {"status": "help"}
    if arguments[0] in COMMANDS:
        return _run_command(arguments[0], arguments[1:])
    return run(build_parser().parse_args(arguments))


def entrypoint(argv: Sequence[str] | None = None) -> int:
    """Execute the public process contract and return integer status code ``0``.

    Successful structured results are discarded at the process boundary. ``argparse``
    exits and runtime exceptions are deliberately not caught, preserving non-zero status
    and the original diagnostic.
    """

    main(argv)
    return 0
