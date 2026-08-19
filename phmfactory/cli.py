"""Public command routing and process entrypoints for PHMFactory.

This module exposes a programmatic API and an operating-system process boundary. Both
consume the same :class:`phmfactory.config.ConfigAnalysis`; neither reparses YAML or
searches for machine-local configuration after compilation.
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
from phmfactory.runtime import (
    AttestationWriteError,
    CompiledRunSpec,
    ExecutionEnvelope,
    RunAttestation,
)
from phmfactory.runtime.evidence import register_pipeline_result_evidence


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


def _record_warning(context: str, error: Exception) -> None:
    """Report a non-authoritative run-record failure without hiding the run result."""

    print(
        f"[WARNING] {context} could not be recorded: "
        f"{type(error).__name__}: {error}",
        file=sys.stderr,
    )


def _prepare_optional_attestation(
    compiled: CompiledRunSpec,
    module_name: str,
    envelope: ExecutionEnvelope,
) -> RunAttestation | None:
    """Prepare the legacy diagnostic manifest without making it a run gate."""

    try:
        return RunAttestation.prepare(compiled, module_name, envelope)
    except Exception as error:
        _record_warning("pending run manifest", error)
        return None


def _write_optional_attestation(
    attestation: RunAttestation | None,
    envelope: ExecutionEnvelope,
    *,
    context: str,
) -> bool:
    """Best-effort manifest update that never replaces the scientific outcome."""

    if attestation is None:
        return False
    try:
        attestation.write(envelope)
    except AttestationWriteError as error:
        _record_warning(context, error)
        return False
    return True


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

    Configuration composition occurs exactly once in :func:`analyze_config`. Protected
    runtime code receives a mutable copy through ``CompiledRunSpec.runtime_config()``.

    The Pipeline result and exception are authoritative. The historical run manifest and
    Pipeline-specific evidence index are retained as optional diagnostics during the v0.3
    migration; inability to prepare, enrich, or finalize them cannot convert a completed
    fit/checkpoint/evaluation path into a failed scientific run.
    """

    requested = requested_config(args)
    analysis = analyze_config(
        requested,
        override_values=args.override,
        local_config=requested_local_config(args),
    )
    resolved = analysis.to_resolved_config()
    compiled = CompiledRunSpec.compile(resolved)

    args.requested_config = requested
    args.config_path = str(analysis.path)
    args.resolved_config_path = str(analysis.path)
    args.resolved_pipeline = analysis.pipeline
    args.config_analysis = analysis
    args.compiled_run_spec = compiled
    args.resolved_config_data = compiled.runtime_config()
    args.effective_config_sha256 = analysis.effective_config_sha256
    args.run_spec_sha256 = compiled.sha256

    module_name = pipeline_module_name(analysis.pipeline, warn=False)
    envelope = ExecutionEnvelope(spec=compiled, pipeline_module=module_name)
    args.execution_envelope = envelope

    attestation = _prepare_optional_attestation(compiled, module_name, envelope)
    args.run_attestation = attestation
    args.run_id = attestation.run_id if attestation is not None else None
    args.run_manifest_path = (
        str(attestation.manifest_path) if attestation is not None else None
    )

    try:
        descriptor = require_pipeline_access(
            analysis.pipeline,
            allow_experimental=bool(getattr(args, "allow_experimental", False)),
            warn=False,
        )
    except BaseException as error:
        envelope.record_failure(error, stage="maturity")
        _write_optional_attestation(
            attestation,
            envelope,
            context="failed run manifest",
        )
        raise
    args.pipeline_descriptor = descriptor

    try:
        pipeline_module = importlib.import_module(module_name)
    except BaseException as error:
        envelope.record_failure(error, stage="import")
        _write_optional_attestation(
            attestation,
            envelope,
            context="failed run manifest",
        )
        raise

    try:
        result = envelope.execute(pipeline_module, args)
    except BaseException:
        _write_optional_attestation(
            attestation,
            envelope,
            context="failed run manifest",
        )
        raise

    if attestation is not None:
        try:
            register_pipeline_result_evidence(attestation, compiled, result)
        except Exception as error:
            _record_warning("optional Pipeline evidence", error)

    manifest_written = _write_optional_attestation(
        attestation,
        envelope,
        context="terminal run manifest",
    )
    if manifest_written and attestation is not None:
        print(f"run_manifest={attestation.manifest_path}")
    else:
        print("run_manifest=unavailable")
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
