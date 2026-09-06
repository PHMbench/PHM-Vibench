#!/usr/bin/env python3
"""Dry-run or explicitly execute one frozen P07 work unit.

The default is read-only.  ``--execute`` is the sole write switch and still
requires an approved protocol binding, a new absolute output path, complete
dependency receipts, and an explicitly selected typed backend.
"""

from __future__ import annotations

import argparse
import importlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

sys.dont_write_bytecode = True


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.utils.p07_protocol.execution_plan import build_execution_plan
from src.utils.p07_protocol.work_unit_executor import (
    CWRUSourcePaths,
    DIRGSourcePaths,
    DependencyBinding,
    HardwareRequest,
    WorkUnitRequest,
    load_protocol_config,
    run_work_unit,
)


DEFAULT_CONFIG_PATH = (
    REPO_ROOT
    / "configs"
    / "experiments"
    / "p07_xoan_operator_attention"
    / "g040_protocol.yaml"
)


class _UnavailableBackend:
    """Turns a backend import failure into an executor-owned failure record."""

    def __init__(self, message: str) -> None:
        self._message = message

    def __getattr__(self, name: str) -> Any:
        def fail(_context: Any) -> Any:
            raise RuntimeError(self._message)

        return fail


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Validate or execute exactly one P07 work-unit ID. The default is "
            "a read-only dry-run; --execute is required for derived writes."
        )
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--protocol-sha256", required=True)
    parser.add_argument("--approved-protocol-sha256")
    parser.add_argument("--unit-id", required=True)
    parser.add_argument("--runtime-commit")
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument(
        "--backend",
        help="Explicit Python backend object as module:attribute; required for execution.",
    )
    parser.add_argument(
        "--dependency-manifest",
        type=Path,
        help="Strict JSON object with a dependencies array of pinned finalized stores.",
    )
    parser.add_argument(
        "--dependency",
        action="append",
        default=[],
        metavar="UNIT_ID|ABS_ROOT|INDEX_SHA256|MARKER_SHA256",
    )
    parser.add_argument("--immutable-source-root", type=Path, action="append", default=[])
    parser.add_argument(
        "--cwru-metadata-path",
        "--metadata-path",
        dest="cwru_metadata_path",
        type=Path,
    )
    parser.add_argument(
        "--cwru-raw-dir",
        "--raw-dir",
        dest="cwru_raw_dir",
        type=Path,
    )
    parser.add_argument(
        "--cwru-reader-source-path",
        "--reader-source-path",
        dest="cwru_reader_source_path",
        type=Path,
    )
    parser.add_argument(
        "--cwru-preprocessing-source-path",
        "--preprocessing-source-path",
        dest="cwru_preprocessing_source_path",
        type=Path,
    )
    parser.add_argument("--dirg-metadata-path", type=Path)
    parser.add_argument("--dirg-raw-dir", type=Path)
    parser.add_argument("--dirg-reader-source-path", type=Path)
    parser.add_argument("--dirg-preprocessing-source-path", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--physical-gpu-index", type=int)
    parser.add_argument("--world-size", type=int, default=1)
    parser.add_argument("--distributed-backend")
    return parser


def _strict_json_load(path: Path) -> Any:
    def reject_constant(value: str) -> None:
        raise ValueError(f"non-finite JSON constant {value}")

    def reject_duplicates(items: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key {key!r}")
            result[key] = value
        return result

    return json.loads(
        path.read_text(encoding="utf-8"),
        parse_constant=reject_constant,
        object_pairs_hook=reject_duplicates,
    )


def _dependency_from_mapping(value: Any) -> DependencyBinding:
    if not isinstance(value, dict) or set(value) != {
        "work_unit_id",
        "output_root",
        "artifact_index_sha256",
        "completion_marker_sha256",
    }:
        raise ValueError("Each dependency record has an invalid key set.")
    return DependencyBinding(
        work_unit_id=value["work_unit_id"],
        output_root=Path(value["output_root"]),
        artifact_index_sha256=value["artifact_index_sha256"],
        completion_marker_sha256=value["completion_marker_sha256"],
    )


def _load_dependencies(args: argparse.Namespace) -> tuple[DependencyBinding, ...]:
    result: list[DependencyBinding] = []
    if args.dependency_manifest is not None:
        payload = _strict_json_load(args.dependency_manifest.resolve())
        if not isinstance(payload, dict) or set(payload) != {"dependencies"}:
            raise ValueError("Dependency manifest must contain only 'dependencies'.")
        records = payload["dependencies"]
        if not isinstance(records, list):
            raise ValueError("Dependency manifest dependencies must be an array.")
        result.extend(_dependency_from_mapping(item) for item in records)
    for serialized in args.dependency:
        fields = serialized.split("|")
        if len(fields) != 4:
            raise ValueError(
                "--dependency requires UNIT_ID|ABS_ROOT|INDEX_SHA256|MARKER_SHA256."
            )
        result.append(
            DependencyBinding(
                work_unit_id=fields[0],
                output_root=Path(fields[1]),
                artifact_index_sha256=fields[2],
                completion_marker_sha256=fields[3],
            )
        )
    return tuple(result)


def _cwru_sources(args: argparse.Namespace) -> CWRUSourcePaths | None:
    values = (
        args.cwru_metadata_path,
        args.cwru_raw_dir,
        args.cwru_reader_source_path,
        args.cwru_preprocessing_source_path,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "CWRU execution requires metadata, raw, reader-source, and preprocessing-source paths together."
        )
    return CWRUSourcePaths(
        metadata_path=args.cwru_metadata_path.resolve(),
        raw_dir=args.cwru_raw_dir.resolve(),
        reader_source_path=args.cwru_reader_source_path.resolve(),
        preprocessing_source_path=args.cwru_preprocessing_source_path.resolve(),
    )


def _dirg_sources(args: argparse.Namespace) -> DIRGSourcePaths | None:
    values = (
        args.dirg_metadata_path,
        args.dirg_raw_dir,
        args.dirg_reader_source_path,
        args.dirg_preprocessing_source_path,
    )
    if all(value is None for value in values):
        return None
    if any(value is None for value in values):
        raise ValueError(
            "DIRG execution requires --dirg-metadata-path, --dirg-raw-dir, "
            "--dirg-reader-source-path, and --dirg-preprocessing-source-path together."
        )
    return DIRGSourcePaths(
        metadata_path=args.dirg_metadata_path.resolve(),
        raw_dir=args.dirg_raw_dir.resolve(),
        reader_source_path=args.dirg_reader_source_path.resolve(),
        preprocessing_source_path=args.dirg_preprocessing_source_path.resolve(),
    )


def _runtime_commit(value: str | None) -> str:
    if value is not None:
        return value
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _load_backend(specification: str | None, *, execute: bool) -> Any:
    if not execute or specification is None:
        return None
    try:
        module_name, separator, attribute_name = specification.partition(":")
        if not separator or not module_name or not attribute_name:
            raise ValueError("Backend must use module:attribute syntax.")
        value = getattr(importlib.import_module(module_name), attribute_name)
        return value() if isinstance(value, type) else value
    except Exception as error:
        return _UnavailableBackend(
            f"Cannot load requested backend {specification!r}: {type(error).__name__}: {error}"
        )


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    try:
        args = parser.parse_args(raw_argv)
        config_path = args.config.resolve()
        config = load_protocol_config(config_path)
        approval = config.get("approval")
        if not isinstance(approval, dict):
            raise ValueError("Config approval must be a mapping.")
        human = approval.get("experiment_protocol_approved")
        thresholds = approval.get("thresholds_approved")
        if not isinstance(human, bool) or not isinstance(thresholds, bool):
            raise ValueError("Config approval snapshots must be booleans.")
        plan = build_execution_plan(
            protocol_sha256=args.protocol_sha256,
            human_gate_snapshot=human,
            thresholds_approved_snapshot=thresholds,
        )
        request = WorkUnitRequest(
            plan=plan,
            work_unit_id=args.unit_id,
            config_path=config_path,
            approved_protocol_sha256=args.approved_protocol_sha256,
            runtime_commit=_runtime_commit(args.runtime_commit),
            command=("python", str(Path(__file__).resolve()), *raw_argv),
            execute=args.execute,
            output_root=(
                None if args.output_root is None else args.output_root.resolve()
            ),
            dependencies=_load_dependencies(args),
            immutable_source_roots=tuple(
                path.resolve() for path in args.immutable_source_root
            ),
            cwru_sources=_cwru_sources(args),
            dirg_sources=_dirg_sources(args),
            hardware=HardwareRequest(
                device=args.device,
                physical_gpu_index=args.physical_gpu_index,
                world_size=args.world_size,
                distributed_backend=args.distributed_backend,
            ),
        )
        result = run_work_unit(
            request,
            backend=_load_backend(args.backend, execute=args.execute),
        )
        print(result.canonical_json())
        return 0 if result.succeeded else 2
    except Exception as error:
        print(
            json.dumps(
                {
                    "schema_version": 1,
                    "domain": "P07-WORK-UNIT-EXECUTOR-CLI-v1",
                    "state": "cli_error",
                    "claim_evidence": False,
                    "evidence_state": "not_evidence",
                    "error_type": type(error).__name__,
                    "error_message": str(error),
                },
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                allow_nan=False,
            )
        )
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
