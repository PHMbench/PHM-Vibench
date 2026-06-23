"""Config catalog, preflight, and CLI launch helpers."""

from __future__ import annotations

import csv
import shlex
import subprocess
import sys
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional

import yaml

from scripts.config_inspect import InspectResult, inspect_config
from src.configs.config_utils import build_experiment_name, transfer_namespace


@dataclass(frozen=True)
class ConfigCatalogEntry:
    """A runnable config surfaced in the UI."""

    entry_id: str
    category: str
    path: str
    description: str
    pipeline: str
    status: str
    outputs: str
    related_docs: str


@dataclass(frozen=True)
class PreflightResult:
    """Resolved config and traceability surfaced before launch."""

    config_path: str
    overrides: List[str]
    shell_command: str
    resolved: Dict[str, Any]
    resolved_yaml: str
    sources: List[Dict[str, str]]
    targets: Dict[str, Any]
    sanity: List[Dict[str, Any]]
    output_preview: str
    pipeline_name: str


@dataclass(frozen=True)
class LaunchRequest:
    """A CLI launch request produced by the frontend."""

    argv: List[str]
    shell_command: str
    config_path: str
    overrides: List[str]
    notes: str


@dataclass(frozen=True)
class LaunchResult:
    """The result of executing a CLI launch request."""

    returncode: int
    output: str


def repo_root() -> Path:
    """Return the repository root."""
    return Path(__file__).resolve().parents[3]


def parse_override_text(text: str) -> List[str]:
    """Parse newline-delimited CLI overrides from the UI."""
    lines: List[str] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        lines.append(line)
    return lines


def _load_csv(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8") as handle:
        yield from csv.DictReader(handle)


@lru_cache(maxsize=1)
def load_config_catalog() -> List[ConfigCatalogEntry]:
    """Return maintained config entries from the registry CSV."""
    registry_path = repo_root() / "configs" / "config_registry.csv"
    entries: List[ConfigCatalogEntry] = []
    for row in _load_csv(registry_path):
        path = (row.get("path") or "").strip()
        if not path:
            continue
        entries.append(
            ConfigCatalogEntry(
                entry_id=(row.get("id") or "").strip(),
                category=(row.get("category") or "").strip(),
                path=path,
                description=(row.get("description") or "").strip(),
                pipeline=(row.get("pipeline") or "").strip(),
                status=(row.get("status") or "").strip(),
                outputs=(row.get("outputs") or "").strip(),
                related_docs=(row.get("related_docs") or "").strip(),
            )
        )
    entries.sort(key=lambda item: (item.category != "demo", item.path))
    return entries


def _preview_output_dir(resolved: Dict[str, Any]) -> str:
    try:
        namespace_cfg = transfer_namespace(resolved)
        exp_name = build_experiment_name(namespace_cfg)
        base_dir = (
            resolved.get("environment", {}).get("output_dir")
            or resolved.get("output_dir")
            or "save"
        )
        return str(Path(str(base_dir)) / exp_name / "iter_0")
    except Exception:
        base_dir = (
            resolved.get("environment", {}).get("output_dir")
            or resolved.get("output_dir")
            or "save"
        )
        return str(Path(str(base_dir)) / "<experiment_name>" / "iter_0")


def build_launch_request(
    config_path: str,
    overrides: List[str],
    notes: str = "",
) -> LaunchRequest:
    """Build the exact CLI request the UI will run."""
    argv = [sys.executable, "main.py", "--config", config_path]
    for override in overrides:
        argv.extend(["--override", override])
    if notes.strip():
        argv.extend(["--notes", notes.strip()])
    shell_command = shlex.join(argv)
    return LaunchRequest(
        argv=argv,
        shell_command=shell_command,
        config_path=config_path,
        overrides=list(overrides),
        notes=notes.strip(),
    )


def build_preflight_result(
    config_path: str,
    overrides: List[str],
    notes: str = "",
) -> PreflightResult:
    """Resolve a config exactly as the CLI would before launch."""
    inspect_result: InspectResult = inspect_config(config_path, overrides=overrides)
    request = build_launch_request(config_path, overrides, notes=notes)
    resolved_yaml = yaml.safe_dump(
        inspect_result.resolved,
        allow_unicode=True,
        sort_keys=False,
    )
    sources = [
        {"field": key, "source": inspect_result.sources[key]}
        for key in sorted(inspect_result.sources.keys())
    ]
    pipeline_name = str(inspect_result.resolved.get("pipeline") or "Pipeline_01_default")
    return PreflightResult(
        config_path=config_path,
        overrides=list(overrides),
        shell_command=request.shell_command,
        resolved=inspect_result.resolved,
        resolved_yaml=resolved_yaml,
        sources=sources,
        targets=inspect_result.targets,
        sanity=inspect_result.sanity,
        output_preview=_preview_output_dir(inspect_result.resolved),
        pipeline_name=pipeline_name,
    )


def run_launch_request(
    request: LaunchRequest,
    on_output: Optional[Callable[[str], None]] = None,
) -> LaunchResult:
    """Execute a CLI launch request and optionally stream its output."""
    process = subprocess.Popen(
        request.argv,
        cwd=repo_root(),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    chunks: List[str] = []
    assert process.stdout is not None
    for line in process.stdout:
        chunks.append(line)
        if on_output is not None:
            on_output(line)
    returncode = process.wait()
    output = "".join(chunks)
    return LaunchResult(returncode=returncode, output=output)
