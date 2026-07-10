"""Runtime configuration policy shared by the Streamlit UI layers.

PHM-Vibench applies ``configs/local/local.yaml`` inside the core loader. The UI
therefore keeps editable YAML portable (resolved without that local layer) and
lets validation/execution apply the local layer exactly once.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Iterable, Tuple

from .config_service import ValidationReport, inspect_config, parse_yaml_text


def inspect_portable_config(
    repo_root: Path,
    config_path: Path,
    overrides: Iterable[Tuple[str, Any]] = (),
    *,
    timeout: float = 90.0,
) -> ValidationReport:
    """Resolve a template while suppressing the default machine-local layer."""

    with tempfile.TemporaryDirectory(prefix="phm_vibench_streamlit_") as temp_dir:
        empty_local = Path(temp_dir) / "empty_local.yaml"
        empty_local.write_text("{}\n", encoding="utf-8")
        return inspect_config(
            repo_root,
            config_path,
            overrides,
            timeout=timeout,
            local_config_path=empty_local,
        )


def inspect_execution_yaml(
    repo_root: Path,
    yaml_text: str,
    overrides: Iterable[Tuple[str, Any]] = (),
    *,
    timeout: float = 90.0,
) -> ValidationReport:
    """Validate portable YAML through the normal core precedence chain."""

    parse_yaml_text(yaml_text)
    with tempfile.TemporaryDirectory(prefix="phm_vibench_streamlit_") as temp_dir:
        config_path = Path(temp_dir) / "execution.yaml"
        config_path.write_text(yaml_text, encoding="utf-8")
        return inspect_config(repo_root, config_path, overrides, timeout=timeout)
