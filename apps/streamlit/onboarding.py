"""First-run readiness and template guidance for the Streamlit workspace.

The module has no Streamlit dependency. It converts repository, Python-environment, and
selected-template facts into immutable reports. Configuration composition remains owned
by the public PHMFactory inspector; no machine-local YAML is discovered here.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, MutableMapping, Tuple

import yaml


class OnboardingError(RuntimeError):
    """Raised when declarative onboarding metadata is invalid."""


@dataclass(frozen=True)
class TemplateProfile:
    template_id: str
    title: str
    summary: str
    difficulty: str
    data_label: str
    device_label: str
    estimated_time: str
    requires_external_data: bool
    badges: Tuple[str, ...] = ()
    required_paths: Tuple[str, ...] = ()
    next_step: str = ""


@dataclass(frozen=True)
class ReadinessCheck:
    key: str
    label: str
    status: str
    detail: str
    action: str = ""

    @property
    def is_blocking(self) -> bool:
        return self.status == "blocked"


@dataclass(frozen=True)
class ReadinessReport:
    checks: Tuple[ReadinessCheck, ...]

    @property
    def blocked(self) -> Tuple[ReadinessCheck, ...]:
        return tuple(item for item in self.checks if item.status == "blocked")

    @property
    def warnings(self) -> Tuple[ReadinessCheck, ...]:
        return tuple(item for item in self.checks if item.status == "warning")

    @property
    def can_execute(self) -> bool:
        return not self.blocked


@dataclass(frozen=True)
class TemplateDataStatus:
    ready: bool
    detail: str
    action: str = ""
    data_root: str = ""
    metadata_path: str = ""


_DEPENDENCIES = (
    (
        "streamlit",
        "streamlit",
        "Streamlit UI",
        "Install the optional UI layer: pip install -r apps/streamlit/requirements.txt",
    ),
    (
        "yaml",
        "PyYAML",
        "YAML configuration support",
        "Install the optional UI layer: pip install -r apps/streamlit/requirements.txt",
    ),
    (
        "torch",
        "torch",
        "PyTorch runtime",
        "Install the maintained core environment before launching experiments.",
    ),
    (
        "pytorch_lightning",
        "pytorch-lightning",
        "Lightning runtime",
        "Install the maintained core environment before launching experiments.",
    ),
)


def _read_yaml_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except FileNotFoundError as error:
        raise OnboardingError(f"Template profile file does not exist: {path}") from error
    except (OSError, UnicodeDecodeError) as error:
        raise OnboardingError(f"Could not read template profile file: {path}") from error
    except yaml.YAMLError as error:
        raise OnboardingError(f"Invalid YAML in template profile file: {path}") from error
    if not isinstance(value, dict):
        raise OnboardingError("Template profile YAML root must be a mapping.")
    return value


def _string_list(value: Any, name: str) -> Tuple[str, ...]:
    if value is None:
        return ()
    if not isinstance(value, list) or not all(
        isinstance(item, str) and item.strip() for item in value
    ):
        raise OnboardingError(f"{name} must be a list of non-empty strings.")
    return tuple(item.strip() for item in value)


def _safe_relative_paths(value: Any, name: str) -> Tuple[str, ...]:
    paths = _string_list(value, name)
    for item in paths:
        path = Path(item)
        if path.is_absolute() or ".." in path.parts:
            raise OnboardingError(f"{name} must stay repository-relative: {item}")
    return paths


def load_template_profiles(path: Path) -> Mapping[str, TemplateProfile]:
    """Load declarative user-facing metadata keyed by registry template id."""

    raw = _read_yaml_mapping(path)
    version = raw.get("version", 1)
    if not isinstance(version, int) or version < 1:
        raise OnboardingError("template_profiles.yaml version must be a positive integer.")
    raw_profiles = raw.get("profiles")
    if not isinstance(raw_profiles, dict) or not raw_profiles:
        raise OnboardingError(
            "template_profiles.yaml must define a non-empty profiles mapping."
        )

    profiles: Dict[str, TemplateProfile] = {}
    for template_id, item in raw_profiles.items():
        if not isinstance(template_id, str) or not template_id.strip():
            raise OnboardingError("Template profile ids must be non-empty strings.")
        if not isinstance(item, dict):
            raise OnboardingError(f"Template profile {template_id!r} must be a mapping.")
        requires_external_data = item.get("requires_external_data", True)
        if not isinstance(requires_external_data, bool):
            raise OnboardingError(
                f"{template_id}.requires_external_data must be true or false."
            )
        profiles[template_id] = TemplateProfile(
            template_id=template_id,
            title=str(item.get("title") or template_id),
            summary=str(item.get("summary") or "Registry-backed experiment template."),
            difficulty=str(item.get("difficulty") or "Advanced"),
            data_label=str(item.get("data_label") or "Check template"),
            device_label=str(item.get("device_label") or "Check config"),
            estimated_time=str(item.get("estimated_time") or "Environment-dependent"),
            requires_external_data=requires_external_data,
            badges=_string_list(item.get("badges"), f"{template_id}.badges"),
            required_paths=_safe_relative_paths(
                item.get("required_paths"), f"{template_id}.required_paths"
            ),
            next_step=str(item.get("next_step") or ""),
        )
    return profiles


def profile_for(
    profiles: Mapping[str, TemplateProfile],
    template_id: str,
) -> TemplateProfile:
    """Return explicit metadata or a conservative generic profile."""

    profile = profiles.get(template_id)
    if profile is not None:
        return profile
    return TemplateProfile(
        template_id=template_id,
        title=template_id,
        summary="Registry-backed template. Review its contract before running.",
        difficulty="Advanced",
        data_label="Check template",
        device_label="Check config",
        estimated_time="Environment-dependent",
        requires_external_data=True,
        badges=("Registry template",),
        next_step="Inspect the resolved data and trainer settings before execution.",
    )


def apply_safe_defaults(
    state: MutableMapping[str, Any],
    default_template_id: str,
) -> None:
    """Reset only configuration UI state; never discard run history."""

    for key in tuple(state.keys()):
        if str(key).startswith("field::"):
            state.pop(key, None)
    state.update(
        {
            "ui_mode": "Quick Start",
            "template_group": "quick_start",
            "selected_template_id": default_template_id,
            "advanced_yaml_template_id": "",
            "advanced_yaml_text": "",
            "advanced_override_text": "",
            "validation_report": None,
            "validation_signature": "",
        }
    )


def _module_available(module_name: str, finder: Callable[[str], Any]) -> bool:
    try:
        return finder(module_name) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def _distribution_version(distribution: str, reader: Callable[[str], str]) -> str:
    try:
        return reader(distribution)
    except importlib.metadata.PackageNotFoundError:
        return ""
    except Exception:
        return ""


def _nearest_existing_parent(path: Path) -> Path:
    candidate = path
    while not candidate.exists() and candidate.parent != candidate:
        candidate = candidate.parent
    return candidate


def collect_environment_readiness(
    repo_root: Path,
    default_profile: TemplateProfile,
    *,
    module_finder: Callable[[str], Any] = importlib.util.find_spec,
    version_reader: Callable[[str], str] = importlib.metadata.version,
    access_checker: Callable[[os.PathLike[str], int], bool] = os.access,
) -> ReadinessReport:
    """Check whether the safest offline CPU path can be executed."""

    root = Path(repo_root).resolve()
    checks = []
    repository_ready = (root / "main.py").is_file() and (root / "configs").is_dir()
    checks.append(
        ReadinessCheck(
            "repository",
            "Repository contract",
            "ready" if repository_ready else "blocked",
            "main.py and configs/ are available."
            if repository_ready
            else "The app is not running from a PHMFactory checkout.",
            "Start Streamlit from the repository root."
            if not repository_ready
            else "",
        )
    )

    for module_name, distribution, label, action in _DEPENDENCIES:
        present = _module_available(module_name, module_finder)
        version = _distribution_version(distribution, version_reader) if present else ""
        checks.append(
            ReadinessCheck(
                f"dependency:{module_name}",
                label,
                "ready" if present else "blocked",
                f"Installed{f' ({version})' if version else ''}."
                if present
                else f"Python module {module_name!r} is unavailable.",
                "" if present else action,
            )
        )

    missing_paths = [
        path for path in default_profile.required_paths if not (root / path).exists()
    ]
    checks.append(
        ReadinessCheck(
            "offline-smoke-assets",
            "Offline smoke assets",
            "ready" if not missing_paths else "blocked",
            "Repository-shipped smoke configuration and data are present."
            if not missing_paths
            else "Missing: " + ", ".join(missing_paths),
            "Restore the repository-shipped smoke files before the first run."
            if missing_paths
            else "",
        )
    )

    output_root = root / "outputs" / "streamlit"
    writable_parent = _nearest_existing_parent(output_root)
    writable = writable_parent.is_dir() and access_checker(writable_parent, os.W_OK)
    checks.append(
        ReadinessCheck(
            "output",
            "Run workspace",
            "ready" if writable else "blocked",
            f"Runs can be written below {output_root}."
            if writable
            else f"No writable parent is available for {output_root}.",
            "Grant write access to the checkout or use a writable clone."
            if not writable
            else "",
        )
    )

    checks.append(
        ReadinessCheck(
            "config-inputs",
            "Configuration inputs",
            "ready",
            "Only the selected template, visible edits, and explicit overrides are active.",
            "",
        )
    )
    return ReadinessReport(tuple(checks))


def _expanded_path(repo_root: Path, raw: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw))
    path = Path(expanded)
    return path if path.is_absolute() else repo_root / path


def _resolved_data_paths(
    repo_root: Path,
    resolved: Mapping[str, Any],
) -> Tuple[Path, Path] | None:
    data = resolved.get("data") if isinstance(resolved.get("data"), Mapping) else {}
    data_dir = data.get("data_dir") if isinstance(data, Mapping) else None
    metadata_file = data.get("metadata_file") if isinstance(data, Mapping) else None
    if not isinstance(data_dir, str) or not data_dir.strip():
        return None
    if not isinstance(metadata_file, str) or not metadata_file.strip():
        return None

    data_root = _expanded_path(repo_root, data_dir.strip()).resolve()
    metadata_value = Path(os.path.expandvars(os.path.expanduser(metadata_file.strip())))
    metadata_path = (
        metadata_value.resolve()
        if metadata_value.is_absolute()
        else (data_root / metadata_value).resolve()
    )
    return data_root, metadata_path


def assess_template_data(
    repo_root: Path,
    resolved: Mapping[str, Any],
    profile: TemplateProfile,
) -> TemplateDataStatus:
    """Resolve selected-template data and return an actionable status."""

    root = Path(repo_root).resolve()
    missing_required = [
        path for path in profile.required_paths if not (root / path).exists()
    ]
    if missing_required:
        return TemplateDataStatus(
            False,
            "Required repository assets are missing: " + ", ".join(missing_required),
            "Restore the missing files or switch to another maintained template.",
        )

    resolved_paths = _resolved_data_paths(root, resolved)
    if resolved_paths is None:
        return TemplateDataStatus(
            False,
            "The selected configuration needs data.data_dir and data.metadata_file.",
            "Set the fields in Advanced mode or add explicit raw overrides.",
        )
    data_root, metadata_path = resolved_paths

    missing = []
    if not data_root.is_dir():
        missing.append(str(data_root))
    if not metadata_path.is_file():
        missing.append(str(metadata_path))
    if missing:
        kind = "External data" if profile.requires_external_data else "Configured smoke data"
        return TemplateDataStatus(
            False,
            f"{kind} is not ready. Missing: " + ", ".join(missing),
            "Set data.data_dir in Advanced mode or add an explicit override, then validate again.",
            data_root=str(data_root),
            metadata_path=str(metadata_path),
        )

    detail = (
        "External data and metadata were found for the selected template."
        if profile.requires_external_data
        else "Repository-shipped data and metadata are ready for the first run."
    )
    return TemplateDataStatus(
        True,
        detail,
        data_root=str(data_root),
        metadata_path=str(metadata_path),
    )
