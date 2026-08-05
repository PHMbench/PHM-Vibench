"""Streamlit adapters for PHMFactory's explicit configuration policy.

The public resolver never discovers ``configs/local/local.yaml``.  Template inspection,
edited YAML inspection, CLI preflight, and real execution therefore share the same
precedence unless the user explicitly supplies a local config through the CLI.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Iterable, Tuple

try:
    from .config_service import ValidationReport, inspect_config, parse_yaml_text
except ImportError:  # pragma: no cover - Streamlit executes app.py as a script.
    from config_service import (  # type: ignore
        ValidationReport,
        inspect_config,
        parse_yaml_text,
    )


def inspect_portable_config(
    repo_root: Path,
    config_path: Path,
    overrides: Iterable[Tuple[str, Any]] = (),
    *,
    timeout: float = 90.0,
) -> ValidationReport:
    """Inspect a template through the same explicit public precedence chain."""

    return inspect_config(repo_root, config_path, overrides, timeout=timeout)


def inspect_execution_yaml(
    repo_root: Path,
    yaml_text: str,
    overrides: Iterable[Tuple[str, Any]] = (),
    *,
    timeout: float = 90.0,
) -> ValidationReport:
    """Validate edited standalone YAML without hidden machine-local inputs."""

    parse_yaml_text(yaml_text)
    with tempfile.TemporaryDirectory(prefix="phmfactory_streamlit_") as temp_dir:
        config_path = Path(temp_dir) / "execution.yaml"
        config_path.write_text(yaml_text, encoding="utf-8")
        return inspect_config(repo_root, config_path, overrides, timeout=timeout)
