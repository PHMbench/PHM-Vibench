"""Run discovery, artifact inventory, and compare helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import yaml

from src.plot_factory.io import (
    discover_runs,
    find_metrics_csv,
    find_predictions,
    find_test_results,
    read_manifest,
)


@dataclass(frozen=True)
class ArtifactRecord:
    """A concrete artifact that the UI can preview."""

    label: str
    path: Path
    kind: str
    exists: bool


@dataclass(frozen=True)
class ProtocolSignature:
    """A derived fairness signature for compare guard rails."""

    summary: str
    hard_fields: Dict[str, str]
    soft_fields: Dict[str, str]


@dataclass(frozen=True)
class RunSummary:
    """A lightweight view of a discovered run."""

    run_dir: Path
    manifest_path: Path
    manifest: Dict[str, Any]
    config_snapshot: Path
    metrics_path: Path
    metrics_csv_logger: Path
    figures_dir: Path
    predictions_path: Path
    artifacts_dir: Path
    checkpoint_paths: Tuple[Path, ...]
    timestamp: str
    run_id: str
    evidence_state: str


@dataclass(frozen=True)
class RunRecord(RunSummary):
    """A hydrated run record with compare metadata."""

    protocol_signature: ProtocolSignature


@dataclass(frozen=True)
class CompareReport:
    """Compatibility report for comparing multiple runs."""

    compatible: bool
    hard_mismatches: List[Dict[str, str]]
    soft_mismatches: List[Dict[str, str]]


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


def _missing_path() -> Path:
    return repo_root() / ".phmfactory_missing"


def _existing_path(raw: str) -> Path:
    if not raw:
        return _missing_path()
    candidate = Path(raw)
    if candidate.is_absolute():
        return candidate
    return repo_root() / candidate


def _coalesce_path(raw: str, fallback: Optional[Path]) -> Path:
    path = _existing_path(raw)
    if path.exists():
        return path
    if fallback is not None:
        return fallback
    return path


@lru_cache(maxsize=512)
def _read_manifest_cached(path_str: str) -> Dict[str, Any]:
    return read_manifest(Path(path_str))


@lru_cache(maxsize=512)
def _read_yaml_cached(path_str: str) -> Dict[str, Any]:
    path = Path(path_str)
    if not path.exists():
        return {}
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    return data if isinstance(data, dict) else {}


def _parse_timestamp(raw: str, fallback_path: Path) -> str:
    if raw:
        return raw
    return datetime.fromtimestamp(fallback_path.stat().st_mtime).isoformat()


def _stringify(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def _collect_field(config: Dict[str, Any], dotted_key: str) -> str:
    current: Any = config
    for part in dotted_key.split("."):
        if not isinstance(current, dict):
            return ""
        current = current.get(part)
    return _stringify(current)


def derive_protocol_signature(config: Dict[str, Any]) -> ProtocolSignature:
    """Derive a compare signature from the resolved config snapshot."""
    hard_keys = [
        "pipeline",
        "data.factory_name",
        "data.metadata_file",
        "task.type",
        "task.name",
        "task.target_system_id",
        "task.source_domain_id",
        "task.target_domain_id",
        "task.target_domain_num",
        "task.n_way",
        "task.k_shot",
        "task.n_query",
    ]
    soft_keys = [
        "base_configs.data",
        "base_configs.task",
        "environment.project",
        "environment.output_dir",
        "data.Name",
        "data.data_dir",
    ]
    hard_fields = {key: _collect_field(config, key) for key in hard_keys}
    soft_fields = {key: _collect_field(config, key) for key in soft_keys}
    summary = " / ".join(
        part
        for part in [
            hard_fields.get("pipeline", ""),
            hard_fields.get("task.type", ""),
            hard_fields.get("task.name", ""),
        ]
        if part
    )
    return ProtocolSignature(summary=summary or "unknown", hard_fields=hard_fields, soft_fields=soft_fields)


def compare_protocols(records: Sequence[RunRecord]) -> CompareReport:
    """Return compare guard rails for the selected runs."""
    if not records:
        return CompareReport(compatible=False, hard_mismatches=[], soft_mismatches=[])
    baseline = records[0].protocol_signature
    hard_mismatches: List[Dict[str, str]] = []
    soft_mismatches: List[Dict[str, str]] = []
    for record in records[1:]:
        for field, expected in baseline.hard_fields.items():
            observed = record.protocol_signature.hard_fields.get(field, "")
            if observed != expected:
                hard_mismatches.append(
                    {
                        "field": field,
                        "expected": expected,
                        "observed": observed,
                        "run_id": record.run_id,
                    }
                )
        for field, expected in baseline.soft_fields.items():
            observed = record.protocol_signature.soft_fields.get(field, "")
            if observed != expected:
                soft_mismatches.append(
                    {
                        "field": field,
                        "expected": expected,
                        "observed": observed,
                        "run_id": record.run_id,
                    }
                )
    return CompareReport(
        compatible=not hard_mismatches,
        hard_mismatches=hard_mismatches,
        soft_mismatches=soft_mismatches,
    )


def evidence_state_for_paths(
    config_snapshot: Path,
    metrics_path: Path,
    figures_dir: Path,
    manifest_path: Path,
) -> str:
    """Classify how complete a run's evidence chain is."""
    has_config = config_snapshot.exists()
    has_metrics = metrics_path.exists()
    has_figures = figures_dir.exists() and any(figures_dir.iterdir())
    has_manifest = manifest_path.exists()
    if has_config and has_metrics and has_figures and has_manifest:
        return "complete"
    if has_manifest and (has_config or has_metrics):
        return "partial"
    return "minimal"


def _default_roots() -> List[Path]:
    roots: List[Path] = []
    for name in ["results", "save"]:
        candidate = repo_root() / name
        if candidate.exists():
            roots.append(candidate)
    return roots


@lru_cache(maxsize=16)
def _discover_all_summaries(root_keys: Tuple[str, ...]) -> Tuple[RunSummary, ...]:
    summaries: List[RunSummary] = []
    for root_key in root_keys:
        root = Path(root_key)
        if not root.exists():
            continue
        for run_ref in discover_runs(root):
            manifest_path = run_ref.manifest_path.resolve()
            manifest = _read_manifest_cached(str(manifest_path))
            config_snapshot = _existing_path(str(manifest.get("config_snapshot") or ""))
            metrics_path = _coalesce_path(
                str(manifest.get("metrics_path") or ""),
                fallback=find_test_results(run_ref.run_dir),
            )
            metrics_csv_logger = _coalesce_path(
                str(manifest.get("metrics_csv_logger") or ""),
                fallback=find_metrics_csv(run_ref.run_dir),
            )
            figures_dir = _existing_path(str(manifest.get("figures_dir") or ""))
            predictions_path = _coalesce_path(
                str(manifest.get("predictions_path") or ""),
                fallback=find_predictions(run_ref.run_dir),
            )
            artifacts_dir = run_ref.run_dir / "artifacts"
            checkpoint_paths = tuple(sorted(run_ref.run_dir.glob("*.ckpt")))
            evidence_state = evidence_state_for_paths(
                config_snapshot=config_snapshot,
                metrics_path=metrics_path,
                figures_dir=figures_dir,
                manifest_path=manifest_path,
            )
            summaries.append(
                RunSummary(
                    run_dir=run_ref.run_dir,
                    manifest_path=manifest_path,
                    manifest=manifest,
                    config_snapshot=config_snapshot,
                    metrics_path=metrics_path,
                    metrics_csv_logger=metrics_csv_logger,
                    figures_dir=figures_dir,
                    predictions_path=predictions_path,
                    artifacts_dir=artifacts_dir,
                    checkpoint_paths=checkpoint_paths,
                    timestamp=_parse_timestamp(str(manifest.get("timestamp") or ""), manifest_path),
                    run_id=str(manifest.get("run_id") or run_ref.run_dir.name),
                    evidence_state=evidence_state,
                )
            )
    summaries.sort(key=lambda item: item.timestamp, reverse=True)
    return tuple(summaries)


def discover_recent_runs(
    root_dirs: Optional[Sequence[Path]] = None,
    limit: int = 200,
) -> List[RunSummary]:
    """Discover recent runs from repo-native manifest files without hydrating configs."""
    roots = list(root_dirs or _default_roots())
    root_keys = tuple(sorted(str(path.resolve()) for path in roots if path.exists()))
    return list(_discover_all_summaries(root_keys)[:limit])


def hydrate_run_record(summary: RunSummary) -> RunRecord:
    """Hydrate a run summary into a compare-ready record."""
    config = _read_yaml_cached(str(summary.config_snapshot.resolve()))
    return RunRecord(
        run_dir=summary.run_dir,
        manifest_path=summary.manifest_path,
        manifest=summary.manifest,
        config_snapshot=summary.config_snapshot,
        metrics_path=summary.metrics_path,
        metrics_csv_logger=summary.metrics_csv_logger,
        figures_dir=summary.figures_dir,
        predictions_path=summary.predictions_path,
        artifacts_dir=summary.artifacts_dir,
        checkpoint_paths=summary.checkpoint_paths,
        timestamp=summary.timestamp,
        run_id=summary.run_id,
        evidence_state=summary.evidence_state,
        protocol_signature=derive_protocol_signature(config),
    )


def discover_run_records(root_dirs: Optional[Sequence[Path]] = None, limit: int = 200) -> List[RunRecord]:
    """Discover and hydrate recent runs from repo-native manifest files."""
    return [hydrate_run_record(summary) for summary in discover_recent_runs(root_dirs=root_dirs, limit=limit)]


def list_artifacts(record: RunSummary) -> List[ArtifactRecord]:
    """Return a stable artifact inventory for a run."""
    inventory: List[ArtifactRecord] = [
        ArtifactRecord("config_snapshot.yaml", record.config_snapshot, "yaml", record.config_snapshot.exists()),
        ArtifactRecord("test_result_*.csv", record.metrics_path, "csv", record.metrics_path.exists()),
        ArtifactRecord("artifacts/manifest.json", record.manifest_path, "json", record.manifest_path.exists()),
        ArtifactRecord("figures/", record.figures_dir, "directory", record.figures_dir.exists()),
        ArtifactRecord(
            "logs/**/metrics.csv",
            record.metrics_csv_logger,
            "csv",
            record.metrics_csv_logger.exists(),
        ),
        ArtifactRecord(
            "artifacts/predictions.npz",
            record.predictions_path,
            "npz",
            record.predictions_path.exists(),
        ),
    ]
    for checkpoint in record.checkpoint_paths:
        inventory.append(ArtifactRecord(checkpoint.name, checkpoint, "ckpt", True))
    return inventory


def load_config_snapshot(record: RunSummary) -> Dict[str, Any]:
    """Load the run's config snapshot."""
    return _read_yaml_cached(str(record.config_snapshot.resolve()))


def _load_metrics_csv(path: Path) -> Dict[str, Any]:
    if not path.exists():
        return {}
    rows = list(csv.DictReader(path.open("r", encoding="utf-8")))
    if not rows:
        return {}
    return rows[0]


def load_metrics(record: RunSummary) -> Dict[str, Any]:
    """Load best-effort metrics for compare and preview."""
    metrics_inline = record.manifest.get("metrics_inline")
    if isinstance(metrics_inline, dict) and metrics_inline:
        return {str(key): value for key, value in metrics_inline.items()}
    if record.metrics_path.exists():
        return _load_metrics_csv(record.metrics_path)
    return {}


def load_metrics_history(record: RunSummary) -> List[Dict[str, Any]]:
    """Load the optional CSV logger history."""
    if not record.metrics_csv_logger.exists():
        return []
    rows = list(csv.DictReader(record.metrics_csv_logger.open("r", encoding="utf-8")))
    return rows


def figure_files(record: RunSummary) -> List[Path]:
    """Return previewable figure files for a run."""
    if not record.figures_dir.exists():
        return []
    return [
        path
        for path in sorted(record.figures_dir.iterdir())
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".svg", ".pdf"}
    ]


def preview_text(path: Path, max_chars: int = 6000) -> str:
    """Read a text preview for a file."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8", errors="replace")[:max_chars]


def preview_predictions(path: Path) -> Dict[str, Any]:
    """Return summary information for a predictions NPZ artifact."""
    if not path.exists():
        return {}
    with np.load(path, allow_pickle=False) as payload:
        return {
            key: {"shape": list(payload[key].shape), "dtype": str(payload[key].dtype)}
            for key in payload.files
        }
