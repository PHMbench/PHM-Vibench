"""Public command router and experiment execution boundary for PHMFactory."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from typing import Any

from phmfactory.commands.common import add_config_arguments, requested_config
from phmfactory.config import resolve_config
from phmfactory.pipelines import pipeline_module_name, require_pipeline_access
from phmfactory.runtime import (
    AttestationWriteError,
    CompiledRunSpec,
    ExecutionEnvelope,
    RunAttestation,
)
from phmfactory.runtime.evidence import register_pipeline_result_evidence


COMMANDS = ("data", "doctor", "demo", "preflight")


def build_parser() -> argparse.ArgumentParser:
    """Build the backward-compatible experiment parser."""

    parser = argparse.ArgumentParser(
        prog="phmfactory",
        description="PHMFactory task pipeline",
        epilog=(
            "Commands: doctor, demo, preflight, data. "
            "Legacy experiment form remains: phmfactory --config <yaml>."
        ),
    )
    add_config_arguments(parser, include_notes=True, include_experimental=True)
    return parser


def _resolve_config_path(args: argparse.Namespace) -> str:
    """Compatibility wrapper retained for callers and tests."""

    return requested_config(args)


def _resolve_pipeline(args: argparse.Namespace, config_path: str) -> str:
    """Resolve the canonical Pipeline through the public config API."""

    return resolve_config(
        config_path,
        override_values=args.override,
    ).pipeline


def _write_failed_attestation(
    attestation: RunAttestation,
    envelope: ExecutionEnvelope,
    original_error: BaseException,
) -> None:
    try:
        attestation.write(envelope)
    except AttestationWriteError as write_error:
        raise write_error from original_error


def run(args: argparse.Namespace) -> Any:
    """Compile, attest, authorize, execute, and index one Pipeline."""

    requested = requested_config(args)
    resolved = resolve_config(requested, override_values=args.override)
    compiled = CompiledRunSpec.compile(resolved)

    args.requested_config = requested
    args.config_path = str(resolved.path)
    args.resolved_config_path = str(resolved.path)
    args.resolved_pipeline = resolved.pipeline
    args.compiled_run_spec = compiled
    args.resolved_config_data = compiled.runtime_config()
    args.run_spec_sha256 = compiled.sha256

    module_name = pipeline_module_name(resolved.pipeline, warn=False)
    envelope = ExecutionEnvelope(spec=compiled, pipeline_module=module_name)
    args.execution_envelope = envelope

    try:
        attestation = RunAttestation.prepare(compiled, module_name, envelope)
    except BaseException as error:
        envelope.record_failure(error, stage="attestation_prepare")
        raise

    args.run_attestation = attestation
    args.run_id = attestation.run_id
    args.run_manifest_path = str(attestation.manifest_path)

    try:
        descriptor = require_pipeline_access(
            resolved.pipeline,
            allow_experimental=bool(getattr(args, "allow_experimental", False)),
            warn=False,
        )
    except BaseException as error:
        envelope.record_failure(error, stage="maturity")
        _write_failed_attestation(attestation, envelope, error)
        raise
    args.pipeline_descriptor = descriptor

    try:
        pipeline_module = importlib.import_module(module_name)
    except BaseException as error:
        envelope.record_failure(error, stage="import")
        _write_failed_attestation(attestation, envelope, error)
        raise

    try:
        result = envelope.execute(pipeline_module, args)
    except BaseException as error:
        _write_failed_attestation(attestation, envelope, error)
        raise

    try:
        register_pipeline_result_evidence(attestation, compiled, result)
    except BaseException as error:
        envelope.record_failure(error, stage="evidence_finalize")
        _write_failed_attestation(attestation, envelope, error)
        raise

    try:
        attestation.write(envelope)
    except AttestationWriteError as error:
        envelope.record_failure(error, stage="attestation_finalize")
        raise

    print(f"run_manifest={attestation.manifest_path}")
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
    """Route a named command or execute the compatible experiment form."""

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] and arguments[0] in COMMANDS:
        return _run_command(arguments[0], arguments[1:])
    return run(build_parser().parse_args(arguments))
