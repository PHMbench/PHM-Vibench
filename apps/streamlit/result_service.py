"""Direct PHMFactory result binding for Streamlit experiment runs.

The public CLI is the scientific result authority. This module reads the final CLI
trailer from the selected run log and only browses that exact ``result_dir`` plus the
Streamlit process directory. It never scans a shared output root or uses mtime to guess
which experiment produced a file.
"""

from __future__ import annotations

import csv
import json
import math
import os
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from .config_service import parse_yaml_text
    from .run_service import RunRecord
except ImportError:  # pragma: no cover - Streamlit executes app.py as a script.
    from config_service import parse_yaml_text  # type: ignore
    from run_service import RunRecord  # type: ignore


@dataclass(frozen=True)
class DiscoveryLimits:
    max_depth: int = 6
    max_entries: int = 3000
    max_files: int = 500
    max_metric_bytes: int = 2_000_000
    max_metric_rows: int = 500
    max_log_bytes: int = 10_000_000


@dataclass(frozen=True)
class Artifact:
    path: Path
    root: Path
    relative_path: str
    kind: str
    size_bytes: int
    modified_at: str


@dataclass(frozen=True)
class MetricTable:
    source: Path
    columns: Tuple[str, ...] = ()
    rows: Tuple[Mapping[str, Any], ...] = ()
    truncated: bool = False
    warning: str = ""


@dataclass(frozen=True)
class DirectResults:
    completed: bool = False
    result_dir: Optional[Path] = None
    best_checkpoint: Optional[Path] = None
    test_metrics: Optional[Path] = None
    run_summary: Optional[Path] = None
    primary_metrics: Mapping[str, Any] = field(default_factory=dict)
    evaluation_requested: Optional[bool] = None
    warnings: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ResultBundle:
    run_id: str
    roots: Tuple[Path, ...]
    artifacts: Tuple[Artifact, ...]
    metrics: Tuple[MetricTable, ...]
    direct: DirectResults = field(default_factory=DirectResults)
    warnings: Tuple[str, ...] = ()
    truncated: bool = False


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_CONFIG_NAMES = {"execution.yaml", "config.yaml", "config.yml", "hparams.yaml"}
_TEXT_EXTENSIONS = {".txt", ".md"}
_DATA_EXTENSIONS = {".csv", ".json", ".parquet", ".npy", ".npz"}
_DOCUMENT_EXTENSIONS = {".pdf", ".svg", ".html"}
_DIRECT_KEYS = frozenset(
    {"result_dir", "best_checkpoint", "test_metrics", "run_summary", "primary_metrics"}
)


def _classify(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if name in {"all_results.csv", "run_summary.json", "metrics.json"} or (
        name.startswith("test_result") and suffix == ".csv"
    ):
        return "metrics"
    if name in _CONFIG_NAMES or suffix in {".yaml", ".yml"}:
        return "config"
    if name.endswith(".log"):
        return "log"
    if suffix in _TEXT_EXTENSIONS:
        return "text"
    if suffix in _DATA_EXTENSIONS:
        return "data"
    if suffix in _DOCUMENT_EXTENSIONS:
        return "document"
    return "file"


def format_bytes(size: int) -> str:
    value = float(max(0, size))
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.0f} {unit}" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _resolve_cli_path(repo_root: Path, value: str) -> Path:
    raw = Path(value).expanduser()
    return raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _evaluation_requested(record: RunRecord) -> Optional[bool]:
    config_path = record.run_dir / "execution.yaml"
    if not config_path.is_file():
        return None
    try:
        config = parse_yaml_text(
            config_path.read_text(encoding="utf-8"), source=str(config_path)
        )
    except (OSError, RuntimeError):
        return None
    trainer = config.get("trainer")
    value = trainer.get("test_after_fit") if isinstance(trainer, Mapping) else None
    return value if isinstance(value, bool) else None


def _read_log(record: RunRecord, *, max_bytes: int) -> Tuple[str, str]:
    """Read a bounded log tail because the public result trailer is emitted last."""

    path = record.run_dir / "run.log"
    if not path.is_file():
        return "", "Run log is not available."
    try:
        size = path.stat().st_size
        with path.open("rb") as handle:
            if size > max_bytes:
                handle.seek(-max_bytes, os.SEEK_END)
                handle.readline()  # discard a partial first line
            data = handle.read()
    except OSError as error:
        return "", f"Could not read run log: {error}"
    return data.decode("utf-8", errors="replace"), ""


def _final_cli_trailer(text: str) -> Optional[Dict[str, str]]:
    lines = text.splitlines()
    try:
        completed_index = max(
            index for index, line in enumerate(lines) if line == "run=completed"
        )
    except ValueError:
        return None

    values: Dict[str, str] = {}
    for line in reversed(lines[:completed_index]):
        if not line:
            continue
        key, separator, value = line.partition("=")
        if not separator or key not in _DIRECT_KEYS:
            break
        values.setdefault(key, value)
    return values


def parse_direct_results(
    repo_root: Path,
    record: RunRecord,
    *,
    limits: DiscoveryLimits = DiscoveryLimits(),
) -> DirectResults:
    evaluation_requested = _evaluation_requested(record)
    if record.status != "succeeded" or record.exit_code not in {0, None}:
        return DirectResults(
            evaluation_requested=evaluation_requested,
            warnings=(
                "Direct scientific results are not accepted for a non-successful run.",
            ),
        )

    text, log_warning = _read_log(record, max_bytes=limits.max_log_bytes)
    if log_warning:
        return DirectResults(
            evaluation_requested=evaluation_requested, warnings=(log_warning,)
        )
    trailer = _final_cli_trailer(text)
    if trailer is None:
        return DirectResults(
            evaluation_requested=evaluation_requested,
            warnings=(
                "The process succeeded but no final `run=completed` CLI trailer was found.",
            ),
        )

    warnings: List[str] = []
    raw_result_dir = trailer.get("result_dir", "").strip()
    if not raw_result_dir:
        return DirectResults(
            completed=True,
            evaluation_requested=evaluation_requested,
            warnings=("The completed CLI trailer did not report result_dir.",),
        )
    result_dir = _resolve_cli_path(repo_root.resolve(), raw_result_dir)
    if not result_dir.is_dir():
        return DirectResults(
            completed=True,
            evaluation_requested=evaluation_requested,
            warnings=(f"Reported result_dir is not a directory: {result_dir}",),
        )

    resolved_files: Dict[str, Optional[Path]] = {
        "best_checkpoint": None,
        "test_metrics": None,
        "run_summary": None,
    }
    for key in resolved_files:
        raw_value = trailer.get(key, "").strip()
        if not raw_value:
            continue
        path = _resolve_cli_path(repo_root.resolve(), raw_value)
        if not _is_within(path, result_dir):
            warnings.append(f"Ignoring {key} outside reported result_dir: {path}")
            continue
        if not path.is_file():
            warnings.append(f"Reported {key} does not exist: {path}")
            continue
        resolved_files[key] = path

    primary_metrics: Mapping[str, Any] = {}
    raw_primary = trailer.get("primary_metrics")
    if raw_primary is not None:
        try:
            parsed = json.loads(raw_primary)
        except json.JSONDecodeError as error:
            warnings.append(f"Could not parse primary_metrics from the CLI trailer: {error}")
        else:
            if isinstance(parsed, dict):
                primary_metrics = parsed
            else:
                warnings.append("primary_metrics from the CLI trailer is not a mapping.")

    if evaluation_requested is True:
        for key in ("test_metrics", "run_summary"):
            if resolved_files[key] is None:
                warnings.append(
                    f"Evaluation was requested but the CLI did not provide usable {key}."
                )

    return DirectResults(
        completed=True,
        result_dir=result_dir,
        best_checkpoint=resolved_files["best_checkpoint"],
        test_metrics=resolved_files["test_metrics"],
        run_summary=resolved_files["run_summary"],
        primary_metrics=primary_metrics,
        evaluation_requested=evaluation_requested,
        warnings=tuple(warnings),
    )


def _discover_root(
    root: Path,
    *,
    limits: DiscoveryLimits,
) -> Tuple[List[Artifact], List[str], bool]:
    artifacts: List[Artifact] = []
    warnings: List[str] = []
    truncated = False
    if not root.exists():
        warnings.append(f"Result root does not exist: {root}")
        return artifacts, warnings, truncated
    if not root.is_dir():
        warnings.append(f"Result root is not a directory: {root}")
        return artifacts, warnings, truncated

    queue = deque([(root, 0)])
    entries_seen = 0
    while queue:
        directory, depth = queue.popleft()
        if depth > limits.max_depth:
            truncated = True
            continue
        try:
            entries = list(os.scandir(directory))
        except OSError as error:
            warnings.append(f"Could not scan {directory}: {error}")
            continue
        entries_seen += len(entries)
        if entries_seen > limits.max_entries:
            warnings.append(
                f"Artifact scan stopped after {limits.max_entries} directory entries."
            )
            truncated = True
            break
        for entry in entries:
            if entry.is_symlink():
                continue
            path = Path(entry.path)
            try:
                if entry.is_dir(follow_symlinks=False):
                    queue.append((path, depth + 1))
                    continue
                if not entry.is_file(follow_symlinks=False):
                    continue
                stat = entry.stat(follow_symlinks=False)
            except OSError:
                continue
            try:
                relative = path.relative_to(root).as_posix()
            except ValueError:
                continue
            artifacts.append(
                Artifact(
                    path=path,
                    root=root,
                    relative_path=relative,
                    kind=_classify(path),
                    size_bytes=stat.st_size,
                    modified_at=datetime.fromtimestamp(
                        stat.st_mtime, tz=timezone.utc
                    ).isoformat(timespec="seconds"),
                )
            )
            if len(artifacts) >= limits.max_files:
                warnings.append(f"Artifact scan stopped after {limits.max_files} files.")
                truncated = True
                return artifacts, warnings, truncated
    return artifacts, warnings, truncated


def _normalize_cell(value: Any) -> Any:
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return value


def _rows_from_json(payload: Any) -> Tuple[List[Dict[str, Any]], str]:
    if isinstance(payload, dict):
        for key in ("metrics", "results", "summary"):
            candidate = payload.get(key)
            if isinstance(candidate, dict):
                return [
                    {str(k): _normalize_cell(v) for k, v in candidate.items()}
                ], ""
            if isinstance(candidate, list) and all(
                isinstance(item, dict) for item in candidate
            ):
                return [
                    {str(k): _normalize_cell(v) for k, v in item.items()}
                    for item in candidate
                ], ""
        return [{str(k): _normalize_cell(v) for k, v in payload.items()}], ""
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return [
            {str(k): _normalize_cell(v) for k, v in item.items()} for item in payload
        ], ""
    return [], "JSON metrics must be an object or a list of objects."


def load_metric_table(
    path: Path, limits: DiscoveryLimits = DiscoveryLimits()
) -> MetricTable:
    try:
        size = path.stat().st_size
    except OSError as error:
        return MetricTable(source=path, warning=f"Could not stat metric file: {error}")
    if size > limits.max_metric_bytes:
        return MetricTable(
            source=path,
            warning=(
                f"Metric file is {format_bytes(size)}; parsing is limited to "
                f"{format_bytes(limits.max_metric_bytes)}."
            ),
        )

    rows: List[Dict[str, Any]] = []
    warning = ""
    try:
        if path.suffix.lower() == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            rows, warning = _rows_from_json(payload)
        elif path.suffix.lower() == ".csv":
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                if not reader.fieldnames:
                    return MetricTable(source=path, warning="CSV metrics have no header.")
                for row in reader:
                    rows.append({str(key): value for key, value in row.items()})
                    if len(rows) > limits.max_metric_rows:
                        break
        else:
            return MetricTable(source=path, warning="Unsupported metric format.")
    except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError) as error:
        return MetricTable(source=path, warning=f"Could not parse metrics: {error}")

    truncated = len(rows) > limits.max_metric_rows
    if truncated:
        rows = rows[: limits.max_metric_rows]
        warning = warning or f"Showing the first {limits.max_metric_rows} metric rows."
    columns: List[str] = []
    seen = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                columns.append(key)
    return MetricTable(
        source=path,
        columns=tuple(columns),
        rows=tuple(rows),
        truncated=truncated,
        warning=warning,
    )


def primary_metric_headlines(
    primary_metrics: Mapping[str, Any], *, limit: int = 4
) -> Tuple[Tuple[str, float], ...]:
    values: List[Tuple[str, float]] = []
    for name, raw in primary_metrics.items():
        candidate = raw.get("mean") if isinstance(raw, Mapping) else raw
        if isinstance(candidate, (int, float)) and not isinstance(candidate, bool):
            value = float(candidate)
            if math.isfinite(value):
                values.append((str(name), value))
                if len(values) >= limit:
                    break
    return tuple(values)


def discover_results(
    repo_root: Path,
    record: RunRecord,
    *,
    limits: DiscoveryLimits = DiscoveryLimits(),
) -> ResultBundle:
    repo = repo_root.resolve()
    direct = parse_direct_results(repo, record, limits=limits)
    roots: List[Path] = [record.run_dir.resolve()]
    if direct.result_dir is not None and direct.result_dir not in roots:
        roots.append(direct.result_dir)

    artifacts: List[Artifact] = []
    warnings: List[str] = list(direct.warnings)
    truncated = False
    for root in roots:
        found, root_warnings, root_truncated = _discover_root(root, limits=limits)
        artifacts.extend(found)
        warnings.extend(root_warnings)
        truncated = truncated or root_truncated

    unique: Dict[Path, Artifact] = {}
    for artifact in artifacts:
        try:
            key = artifact.path.resolve()
        except OSError:
            key = artifact.path.absolute()
        unique.setdefault(key, artifact)
    artifacts = sorted(
        unique.values(), key=lambda item: (item.kind, item.relative_path)
    )

    metric_paths = [
        path for path in (direct.test_metrics, direct.run_summary) if path is not None
    ]
    metrics = tuple(load_metric_table(path, limits) for path in metric_paths)
    return ResultBundle(
        run_id=record.run_id,
        roots=tuple(roots),
        artifacts=tuple(artifacts),
        metrics=metrics,
        direct=direct,
        warnings=tuple(dict.fromkeys(warnings)),
        truncated=truncated,
    )


def artifact_groups(bundle: ResultBundle) -> Mapping[str, Tuple[Artifact, ...]]:
    groups: Dict[str, List[Artifact]] = {}
    for artifact in bundle.artifacts:
        groups.setdefault(artifact.kind, []).append(artifact)
    return {key: tuple(value) for key, value in groups.items()}
