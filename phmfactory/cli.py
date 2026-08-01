"""Public command-line interface for PHMFactory."""

from __future__ import annotations

import argparse
import importlib
import sys
from collections.abc import Sequence
from typing import Any

from phmfactory.config import DEFAULT_CONFIG, resolve_config
from phmfactory.pipelines import pipeline_module_name
from phmfactory.runtime import (
    AttestationWriteError,
    CompiledRunSpec,
    ExecutionEnvelope,
    RunAttestation,
)


def build_parser() -> argparse.ArgumentParser:
    """Build the parser shared by the three experiment entrypoints."""
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


def build_data_parser() -> argparse.ArgumentParser:
    """Build the bounded dataset-bundle management command surface."""
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
    """Compile, attest, and execute one Pipeline through the public boundary."""
    requested_config = _resolve_config_path(args)
    resolved = resolve_config(requested_config, override_values=args.override)
    compiled = CompiledRunSpec.compile(resolved)

    # Keep the user's source for provenance, but pass the resolved file path to
    # the protected runtime so public presets never fall into legacy aliases.
    args.requested_config = requested_config
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
        attestation.write(envelope)
    except AttestationWriteError as error:
        envelope.record_failure(error, stage="attestation_finalize")
        raise

    print(f"run_manifest={attestation.manifest_path}")
    print("完成所有实验！")
    return result


def _run_data_command(argv: Sequence[str]) -> Any:
    from phmfactory.data_sources import (
        compare_bundle_hashes,
        download_bundle,
        load_bundle_spec,
        validate_bundle,
    )

    parser = build_data_parser()
    args = parser.parse_args(list(argv))
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


def main(argv: Sequence[str] | None = None) -> Any:
    """Execute a data subcommand or the backward-compatible experiment CLI."""
    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments[:1] == ["data"]:
        return _run_data_command(arguments[1:])
    parser = build_parser()
    args = parser.parse_args(arguments)
    return run(args)
