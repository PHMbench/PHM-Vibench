"""Bounded result and artifact discovery for Streamlit experiment runs."""

from __future__ import annotations

import csv
import json
import math
import os
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

try:
    from .run_service import RunRecord
except ImportError:  # pragma: no cover - Streamlit executes app.py as a script.
    from run_service import RunRecord  # type: ignore


@dataclass(frozen=True)
class DiscoveryLimits:
    max_depth: int = 6
    max_entries: int = 3000
    max_files: int = 500
    max_metric_bytes: int = 2_000_000
    max_metric_rows: int = 500


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
class ResultBundle:
    run_id: str
    roots: Tuple[Path, ...]
    artifacts: Tuple[Artifact, ...]
    metrics: Tuple[MetricTable, ...]
    warnings: Tuple[str, ...] = ()
    truncated: bool = False


_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
_METRIC_NAMES = {
    "metrics.json",
    "results.json",
    "summary.json",
    "all_results.csv",
}
_CONFIG_NAMES = {"execution.yaml", "config.yaml", "config.yml", "hparams.yaml"}
_TEXT_EXTENSIONS = {".txt", ".md"}
_DATA_EXTENSIONS = {".csv", ".json", ".parquet", ".npy", ".npz"}
_DOCUMENT_EXTENSIONS = {".pdf", ".svg", ".html"}


def _parse_time(value: str) -> Optional[float]:
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def _classify(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()
    if suffix in _IMAGE_EXTENSIONS:
        return "image"
    if name in _METRIC_NAMES or name.startswith("test_result") and suffix == ".csv":
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


def _is_dangerous_root(repo_root: Path, candidate: Path) -> bool:
    resolved = candidate.resolve()
    anchor = Path(resolved.anchor)
    if resolved == anchor or resolved == Path.home().resolve():
        return True
    repo = repo_root.resolve()
    if resolved == repo:
        return True
    try:
        repo.relative_to(resolved)
        return True  # candidate is a parent of the repository
    except ValueError:
        return False


def _output_root(repo_root: Path, value: str) -> Optional[Path]:
    if not value or "{" in value or "}" in value:
        return None
    raw = Path(value).expanduser()
    return raw.resolve() if raw.is_absolute() else (repo_root / raw).resolve()


def result_roots(repo_root: Path, record: RunRecord) -> Tuple[Path, ...]:
    roots: List[Path] = [record.run_dir.resolve()]
    output = _output_root(repo_root.resolve(), record.output_root)
    if output is not None and output not in roots:
        roots.append(output)
    return tuple(roots)


def _discover_root(
    root: Path,
    *,
    started_epoch: Optional[float],
    limits: DiscoveryLimits,
    include_old: bool,
) -> Tuple[List[Artifact], List[str], bool]:
    artifacts: List[Artifact] = []
    warnings: List[str] = []
    truncated = False
    if not root.exists():
        warnings.append(f"Result root does not exist yet: {root}")
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
        except OSError as exc:
            warnings.append(f"Could not scan {directory}: {exc}")
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
            if (
                not include_old
                and started_epoch is not None
                and stat.st_mtime < started_epoch - 5.0
            ):
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
                return [{str(k): _normalize_cell(v) for k, v in candidate.items()}], ""
            if isinstance(candidate, list) and all(isinstance(item, dict) for item in candidate):
                return [
                    {str(k): _normalize_cell(v) for k, v in item.items()}
                    for item in candidate
                ], ""
        return [{str(k): _normalize_cell(v) for k, v in payload.items()}], ""
    if isinstance(payload, list) and all(isinstance(item, dict) for item in payload):
        return [
            {str(k): _normalize_cell(v) for k, v in item.items()}
            for item in payload
        ], ""
    return [], "JSON metrics must be an object or a list of objects."


def load_metric_table(path: Path, limits: DiscoveryLimits = DiscoveryLimits()) -> MetricTable:
    try:
        size = path.stat().st_size
    except OSError as exc:
        return MetricTable(source=path, warning=f"Could not stat metric file: {exc}")
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
    except (OSError, UnicodeDecodeError, csv.Error, json.JSONDecodeError) as exc:
        return MetricTable(source=path, warning=f"Could not parse metrics: {exc}")

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


def _metric_priority(artifact: Artifact) -> Tuple[int, str]:
    name = artifact.path.name.lower()
    if name == "all_results.csv":
        return (0, artifact.relative_path)
    if name == "metrics.json":
        return (1, artifact.relative_path)
    if name.startswith("test_result"):
        return (2, artifact.relative_path)
    return (3, artifact.relative_path)


def discover_results(
    repo_root: Path,
    record: RunRecord,
    *,
    limits: DiscoveryLimits = DiscoveryLimits(),
) -> ResultBundle:
    repo = repo_root.resolve()
    roots = result_roots(repo, record)
    started_epoch = _parse_time(record.started_at)
    artifacts: List[Artifact] = []
    warnings: List[str] = []
    truncated = False
    accepted_roots: List[Path] = []

    for root in roots:
        if root != record.run_dir.resolve() and _is_dangerous_root(repo, root):
            warnings.append(f"Refusing to scan overly broad result root: {root}")
            continue
        accepted_roots.append(root)
        found, root_warnings, root_truncated = _discover_root(
            root,
            started_epoch=started_epoch,
            limits=limits,
            include_old=root == record.run_dir.resolve(),
        )
        artifacts.extend(found)
        warnings.extend(root_warnings)
        truncated = truncated or root_truncated

    # De-duplicate the same physical file when roots overlap.
    unique: Dict[Path, Artifact] = {}
    for artifact in artifacts:
        try:
            key = artifact.path.resolve()
        except OSError:
            key = artifact.path.absolute()
        unique.setdefault(key, artifact)
    artifacts = sorted(unique.values(), key=lambda item: (item.kind, item.relative_path))

    metric_artifacts = sorted(
        (
            item
            for item in artifacts
            if item.kind == "metrics"
            or item.path.suffix.lower() in {".csv", ".json"}
            and any(token in item.path.name.lower() for token in ("metric", "result", "summary"))
        ),
        key=_metric_priority,
    )
    metrics = tuple(load_metric_table(item.path, limits) for item in metric_artifacts[:12])
    return ResultBundle(
        run_id=record.run_id,
        roots=tuple(accepted_roots),
        artifacts=tuple(artifacts),
        metrics=metrics,
        warnings=tuple(dict.fromkeys(warnings)),
        truncated=truncated,
    )


def headline_metrics(
    tables: Sequence[MetricTable],
    *,
    limit: int = 4,
) -> Tuple[Tuple[str, Any], ...]:
    values: List[Tuple[str, Any]] = []
    seen = set()
    for table in tables:
        if not table.rows:
            continue
        row = table.rows[-1]
        for key, raw in row.items():
            if key in seen or raw in (None, ""):
                continue
            value: Any = raw
            if isinstance(raw, str):
                try:
                    value = float(raw)
                except ValueError:
                    continue
            if (
                isinstance(value, (int, float))
                and not isinstance(value, bool)
                and math.isfinite(float(value))
            ):
                seen.add(key)
                values.append((str(key), value))
                if len(values) >= limit:
                    return tuple(values)
    return tuple(values)


def artifact_groups(bundle: ResultBundle) -> Mapping[str, Tuple[Artifact, ...]]:
    groups: Dict[str, List[Artifact]] = {}
    for artifact in bundle.artifacts:
        groups.setdefault(artifact.kind, []).append(artifact)
    return {key: tuple(value) for key, value in groups.items()}
