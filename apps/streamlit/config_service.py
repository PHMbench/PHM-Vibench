"""Configuration adapter for the optional PHM-Vibench Streamlit UI.

The module has no Streamlit dependency. It treats the repository registry and
``scripts.config_inspect`` CLI as authoritative, while keeping UI aliases and
grouping declarative in ``field_catalog.yaml``.
"""

from __future__ import annotations

import copy
import csv
import json
import os
import re
import shlex
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

CONFIG_BLOCKS: Tuple[str, ...] = (
    "environment",
    "data",
    "model",
    "task",
    "trainer",
)
_OVERRIDE_KEY = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")


class ConfigServiceError(RuntimeError):
    """Base class for user-correctable configuration failures."""


class RegistryError(ConfigServiceError):
    pass


class ConfigPathError(ConfigServiceError):
    pass


class ConfigFormatError(ConfigServiceError):
    pass


class OverrideError(ConfigServiceError):
    pass


@dataclass(frozen=True)
class RegistryEntry:
    id: str
    category: str
    path: str
    description: str
    pipeline: str = ""
    status: str = ""
    minimal_run: str = ""
    common_overrides: str = ""
    outputs: str = ""
    related_docs: str = ""
    metadata: Mapping[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class FieldSpec:
    key: str
    label: str
    widget: str
    paths: Tuple[str, ...]
    help: str = ""
    default: Any = None
    minimum: Optional[float] = None
    maximum: Optional[float] = None
    step: Optional[float] = None
    options: Tuple[Any, ...] = ()
    quick_start: bool = False


@dataclass(frozen=True)
class Catalog:
    version: int
    default_template_id: str
    fields: Tuple[FieldSpec, ...]
    template_groups: Mapping[str, Mapping[str, Any]]


@dataclass(frozen=True)
class ValidationReport:
    ok: bool
    command: Tuple[str, ...]
    resolved: Mapping[str, Any] = field(default_factory=dict)
    sources: Mapping[str, str] = field(default_factory=dict)
    targets: Mapping[str, Any] = field(default_factory=dict)
    sanity: Tuple[Mapping[str, Any], ...] = ()
    stdout: str = ""
    stderr: str = ""
    error: str = ""

    @property
    def failed_checks(self) -> Tuple[Mapping[str, Any], ...]:
        return tuple(item for item in self.sanity if not bool(item.get("ok")))


def find_repo_root(start: Optional[Path] = None) -> Path:
    current = (start or Path(__file__)).resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "main.py").is_file() and (candidate / "configs").is_dir():
            return candidate
    raise ConfigPathError(
        "Could not locate the PHM-Vibench repository root. Start the app from "
        "a checkout containing main.py and configs/."
    )


def _within(root: Path, candidate: Path) -> Path:
    root = root.resolve()
    candidate = candidate.resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ConfigPathError(f"Path escapes the repository root: {candidate}") from exc
    return candidate


def resolve_repo_path(
    repo_root: Path,
    value: str,
    *,
    allowed_prefixes: Sequence[str] = ("configs",),
    must_exist: bool = True,
    yaml_only: bool = False,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ConfigPathError("Configuration path is empty.")
    raw = Path(value.strip())
    resolved = _within(repo_root, raw if raw.is_absolute() else repo_root / raw)
    allowed = any(
        _is_relative_to(resolved, _within(repo_root, repo_root / prefix))
        for prefix in allowed_prefixes
    )
    if not allowed:
        raise ConfigPathError(
            "Path must stay under one of these repository directories: "
            + ", ".join(allowed_prefixes)
            + "."
        )
    if yaml_only and resolved.suffix.lower() not in {".yaml", ".yml"}:
        raise ConfigPathError(f"Expected a YAML configuration file: {value}")
    if must_exist and not resolved.is_file():
        raise ConfigPathError(f"Configuration file does not exist: {value}")
    return resolved


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def load_registry(
    repo_root: Path,
    registry_path: str = "configs/config_registry.csv",
) -> Tuple[RegistryEntry, ...]:
    path = resolve_repo_path(repo_root, registry_path, yaml_only=False)
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise RegistryError(f"Registry has no header: {path}")
            missing = {"id", "category", "path", "description"}.difference(
                reader.fieldnames
            )
            if missing:
                raise RegistryError(
                    "Registry is missing required columns: "
                    + ", ".join(sorted(missing))
                )
            entries: List[RegistryEntry] = []
            seen = set()
            for line_no, row in enumerate(reader, start=2):
                data = {str(key): (value or "").strip() for key, value in row.items()}
                entry_id, entry_path = data.get("id", ""), data.get("path", "")
                if not entry_id or not entry_path:
                    raise RegistryError(
                        f"Registry row {line_no} must include non-empty id and path."
                    )
                if entry_id in seen:
                    raise RegistryError(f"Duplicate registry id: {entry_id}")
                seen.add(entry_id)
                entries.append(
                    RegistryEntry(
                        id=entry_id,
                        category=data.get("category", ""),
                        path=entry_path,
                        description=data.get("description", ""),
                        pipeline=data.get("pipeline", ""),
                        status=data.get("status", ""),
                        minimal_run=data.get("minimal_run", ""),
                        common_overrides=data.get("common_overrides", ""),
                        outputs=data.get("outputs", ""),
                        related_docs=data.get("related_docs", ""),
                        metadata=data,
                    )
                )
    except OSError as exc:
        raise RegistryError(f"Could not read registry: {path}") from exc
    if not entries:
        raise RegistryError(f"Registry contains no entries: {path}")
    return tuple(entries)


def load_yaml_mapping(path: Path) -> Dict[str, Any]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = yaml.safe_load(handle) or {}
    except FileNotFoundError as exc:
        raise ConfigPathError(f"Configuration file does not exist: {path}") from exc
    except UnicodeDecodeError as exc:
        raise ConfigFormatError(f"Configuration must be UTF-8 encoded: {path}") from exc
    except yaml.YAMLError as exc:
        raise ConfigFormatError(_yaml_error(exc, f"Invalid YAML: {path}")) from exc
    if not isinstance(value, dict):
        raise ConfigFormatError(f"YAML root must be a mapping: {path}")
    return value


def _yaml_error(error: yaml.YAMLError, message: str) -> str:
    mark = getattr(error, "problem_mark", None)
    if mark is None:
        return message
    return f"{message} at line {mark.line + 1}, column {mark.column + 1}"


def parse_yaml_text(text: str, *, source: str = "edited configuration") -> Dict[str, Any]:
    try:
        value = yaml.safe_load(text) or {}
    except yaml.YAMLError as exc:
        raise ConfigFormatError(_yaml_error(exc, f"Invalid YAML in {source}")) from exc
    if not isinstance(value, dict):
        raise ConfigFormatError(f"{source} must contain a YAML mapping at its root.")
    missing = [block for block in CONFIG_BLOCKS if not isinstance(value.get(block), dict)]
    if missing:
        raise ConfigFormatError(
            f"{source} is missing resolved mapping blocks: {', '.join(missing)}."
        )
    return value


def dump_yaml(config: Mapping[str, Any]) -> str:
    return yaml.safe_dump(
        dict(config), allow_unicode=True, sort_keys=False, default_flow_style=False
    )


def _number(value: Any, name: str) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigFormatError(f"{name} must be numeric.")
    return float(value)


def load_catalog(path: Path) -> Catalog:
    raw = load_yaml_mapping(path)
    version = raw.get("version", 1)
    if not isinstance(version, int) or version < 1:
        raise ConfigFormatError("field_catalog.yaml version must be a positive integer.")
    raw_fields = raw.get("fields")
    if not isinstance(raw_fields, dict) or not raw_fields:
        raise ConfigFormatError("field_catalog.yaml must define a non-empty fields mapping.")
    fields: List[FieldSpec] = []
    for key, item in raw_fields.items():
        if not isinstance(key, str) or not _OVERRIDE_KEY.fullmatch(key.replace("-", "_")):
            raise ConfigFormatError(f"Invalid logical field key: {key!r}")
        if not isinstance(item, dict):
            raise ConfigFormatError(f"Field {key!r} must be a mapping.")
        paths = item.get("paths")
        paths = [paths] if isinstance(paths, str) else paths
        if not isinstance(paths, list) or not paths or not all(
            isinstance(value, str) for value in paths
        ):
            raise ConfigFormatError(f"Field {key!r} must define one or more paths.")
        for config_key in paths:
            validate_override_key(config_key)
        options = item.get("options") or []
        if not isinstance(options, list):
            raise ConfigFormatError(f"Field {key!r} options must be a list.")
        fields.append(
            FieldSpec(
                key=key,
                label=str(item.get("label") or key),
                widget=str(item.get("widget") or "text"),
                paths=tuple(paths),
                help=str(item.get("help") or ""),
                default=item.get("default"),
                minimum=_number(item.get("min"), f"{key}.min"),
                maximum=_number(item.get("max"), f"{key}.max"),
                step=_number(item.get("step"), f"{key}.step"),
                options=tuple(options),
                quick_start=bool(item.get("quick_start", False)),
            )
        )
    groups = raw.get("template_groups") or {}
    if not isinstance(groups, dict) or not all(
        isinstance(value, dict) for value in groups.values()
    ):
        raise ConfigFormatError("template_groups must be a mapping of mappings.")
    return Catalog(
        version=version,
        default_template_id=str(raw.get("default_template_id") or ""),
        fields=tuple(fields),
        template_groups={str(key): dict(value) for key, value in groups.items()},
    )


def get_nested(mapping: Mapping[str, Any], path: str, default: Any = None) -> Any:
    current: Any = mapping
    for part in path.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return default
        current = current[part]
    return current


def _has_nested(mapping: Mapping[str, Any], path: str) -> bool:
    marker = object()
    return get_nested(mapping, path, marker) is not marker


def set_nested(mapping: Dict[str, Any], path: str, value: Any) -> None:
    validate_override_key(path)
    current = mapping
    parts = path.split(".")
    for part in parts[:-1]:
        child = current.get(part)
        if child is None:
            child = {}
            current[part] = child
        if not isinstance(child, dict):
            raise OverrideError(
                f"Cannot set {path!r}: {part!r} is already a non-mapping value."
            )
        current = child
    current[parts[-1]] = value


def select_field_path(resolved: Mapping[str, Any], spec: FieldSpec) -> str:
    return next((path for path in spec.paths if _has_nested(resolved, path)), spec.paths[0])


def field_value(resolved: Mapping[str, Any], spec: FieldSpec) -> Any:
    value = get_nested(resolved, select_field_path(resolved, spec), None)
    return spec.default if value is None else value


def validate_override_key(key: str) -> str:
    key = key.strip() if isinstance(key, str) else ""
    if not key or not _OVERRIDE_KEY.fullmatch(key):
        raise OverrideError(
            f"Invalid override key {key!r}. Use dot-delimited identifiers "
            "such as trainer.num_epochs."
        )
    if key == "base_configs" or key.startswith("base_configs."):
        raise OverrideError("base_configs cannot be changed through the UI override editor.")
    return key


def serialize_override_value(value: Any) -> str:
    if isinstance(value, float):
        if value != value or value in (float("inf"), float("-inf")):
            raise OverrideError("NaN and infinity are not valid override values.")
        return repr(value)
    if isinstance(value, int) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (str, bool, list, dict)) or value is None:
        try:
            return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        except (TypeError, ValueError) as exc:
            raise OverrideError(
                f"Override value is not JSON/YAML serializable: {value!r}"
            ) from exc
    raise OverrideError(
        f"Unsupported override value type: {type(value).__name__}. Use a "
        "string, number, boolean, list, mapping, or null."
    )


def parse_override_lines(text: str) -> Tuple[Tuple[str, Any], ...]:
    parsed: List[Tuple[str, Any]] = []
    seen = set()
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise OverrideError(
                f"Override line {line_no} must use key=value format: {raw_line!r}"
            )
        raw_key, raw_value = line.split("=", 1)
        key = validate_override_key(raw_key)
        if key in seen:
            raise OverrideError(f"Duplicate override key on line {line_no}: {key}")
        seen.add(key)
        try:
            value = yaml.safe_load(raw_value.strip())
        except yaml.YAMLError as exc:
            raise OverrideError(f"Invalid YAML value on override line {line_no}.") from exc
        serialize_override_value(value)
        parsed.append((key, value))
    return tuple(parsed)


def normalize_overrides(
    overrides: Iterable[Tuple[str, Any]],
) -> Tuple[Tuple[str, Any], ...]:
    result: List[Tuple[str, Any]] = []
    positions: Dict[str, int] = {}
    for raw_key, value in overrides:
        key = validate_override_key(raw_key)
        serialize_override_value(value)
        if key in positions:
            result[positions[key]] = (key, value)
        else:
            positions[key] = len(result)
            result.append((key, value))
    return tuple(result)


def build_field_overrides(
    resolved: Mapping[str, Any],
    catalog: Catalog,
    values: Mapping[str, Any],
    *,
    quick_start_only: bool = False,
) -> Tuple[Tuple[str, Any], ...]:
    result = []
    for spec in catalog.fields:
        if quick_start_only and not spec.quick_start:
            continue
        if spec.key not in values:
            continue
        path = select_field_path(resolved, spec)
        if values[spec.key] != get_nested(resolved, path, spec.default):
            result.append((path, values[spec.key]))
    return normalize_overrides(result)


def apply_overrides(
    config: Mapping[str, Any],
    overrides: Iterable[Tuple[str, Any]],
) -> Dict[str, Any]:
    copied = copy.deepcopy(dict(config))
    for key, value in normalize_overrides(overrides):
        set_nested(copied, key, value)
    return copied


def override_args(overrides: Iterable[Tuple[str, Any]]) -> Tuple[str, ...]:
    args: List[str] = []
    for key, value in normalize_overrides(overrides):
        args.extend(("--override", f"{key}={serialize_override_value(value)}"))
    return tuple(args)


def _display_config_path(repo_root: Path, config_path: Path) -> str:
    try:
        return config_path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return str(config_path.resolve())


def build_main_command(
    repo_root: Path,
    config_path: Path,
    overrides: Iterable[Tuple[str, Any]] = (),
    *,
    python_executable: Optional[str] = None,
) -> Tuple[str, ...]:
    return (
        python_executable or sys.executable,
        "main.py",
        "--config",
        _display_config_path(repo_root, config_path),
        *override_args(overrides),
    )


def format_command(command: Sequence[str], *, platform: Optional[str] = None) -> str:
    if (platform or os.name) == "nt":
        return subprocess.list2cmdline(list(command))
    return shlex.join(command)


def inspect_config(
    repo_root: Path,
    config_path: Path,
    overrides: Iterable[Tuple[str, Any]] = (),
    *,
    timeout: float = 90.0,
    python_executable: Optional[str] = None,
    local_config_path: Optional[Path] = None,
) -> ValidationReport:
    command = [
        python_executable or sys.executable,
        "-m",
        "scripts.config_inspect",
        "--config",
        _display_config_path(repo_root, config_path),
        "--dump",
        "all",
        "--format",
        "json",
    ]
    if local_config_path is not None:
        command.extend(("--local_config", str(local_config_path.resolve())))
    command.extend(override_args(overrides))
    try:
        completed = subprocess.run(
            command,
            cwd=str(repo_root),
            text=True,
            capture_output=True,
            timeout=timeout,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return ValidationReport(
            False,
            tuple(command),
            stdout=exc.stdout or "",
            stderr=exc.stderr or "",
            error=f"Configuration inspection timed out after {timeout:g} seconds.",
        )
    except OSError as exc:
        return ValidationReport(
            False,
            tuple(command),
            error=f"Could not start the config inspector: {exc}",
        )
    if completed.returncode != 0:
        return ValidationReport(
            False,
            tuple(command),
            stdout=completed.stdout,
            stderr=completed.stderr,
            error=(
                "The repository config inspector rejected the configuration. "
                "Review stderr and install core dependencies if an import failed."
            ),
        )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError:
        return ValidationReport(
            False,
            tuple(command),
            stdout=completed.stdout,
            stderr=completed.stderr,
            error="The config inspector did not return valid JSON.",
        )
    if not isinstance(payload, dict):
        return ValidationReport(
            False,
            tuple(command),
            stdout=completed.stdout,
            stderr=completed.stderr,
            error="The config inspector returned an unexpected payload.",
        )
    resolved, sanity_raw = payload.get("resolved") or {}, payload.get("sanity") or []
    if not isinstance(resolved, dict) or not isinstance(sanity_raw, list):
        return ValidationReport(
            False,
            tuple(command),
            stdout=completed.stdout,
            stderr=completed.stderr,
            error="The config inspector returned an unexpected payload.",
        )
    sanity = tuple(item for item in sanity_raw if isinstance(item, dict))
    passed = bool(sanity) and all(bool(item.get("ok")) for item in sanity)
    return ValidationReport(
        passed,
        tuple(command),
        resolved=resolved,
        sources=payload.get("sources") or {},
        targets=payload.get("targets") or {},
        sanity=sanity,
        stdout=completed.stdout,
        stderr=completed.stderr,
        error="" if passed else "One or more repository sanity checks failed.",
    )


def inspect_yaml_text(
    repo_root: Path,
    yaml_text: str,
    overrides: Iterable[Tuple[str, Any]] = (),
    *,
    timeout: float = 90.0,
) -> ValidationReport:
    parse_yaml_text(yaml_text)
    with tempfile.TemporaryDirectory(prefix="phm_vibench_streamlit_") as temp_dir:
        config_path = Path(temp_dir) / "edited_config.yaml"
        empty_local = Path(temp_dir) / "empty_local.yaml"
        config_path.write_text(yaml_text, encoding="utf-8")
        empty_local.write_text("{}\n", encoding="utf-8")
        return inspect_config(
            repo_root,
            config_path,
            overrides,
            timeout=timeout,
            # The YAML is already resolved. Prevent default local config from
            # being applied a second time by the core loader.
            local_config_path=empty_local,
        )


def group_entries(
    entries: Sequence[RegistryEntry],
    catalog: Catalog,
    group_key: str,
) -> Tuple[RegistryEntry, ...]:
    group = catalog.template_groups.get(group_key, {})
    include_ids = {str(value) for value in group.get("include_ids") or []}
    include_categories = {
        str(value) for value in group.get("include_categories") or []
    }
    include_statuses = {str(value) for value in group.get("include_statuses") or []}
    include_all = bool(group.get("include_all", False))
    selected = []
    for entry in entries:
        matched = include_all or entry.id in include_ids or entry.category in include_categories
        if matched and (not include_statuses or entry.status in include_statuses):
            selected.append(entry)
    return tuple(selected)


def entry_by_id(entries: Sequence[RegistryEntry], entry_id: str) -> RegistryEntry:
    for entry in entries:
        if entry.id == entry_id:
            return entry
    raise RegistryError(f"Registry entry not found: {entry_id}")
