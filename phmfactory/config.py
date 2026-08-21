"""Single public configuration authority for PHMFactory.

This module owns the complete maintained precedence chain:

``base_configs -> experiment YAML -> explicit local config -> CLI overrides``

Every public caller—run, preflight, inspection, validation, support generation, and the
optional Streamlit UI—must use :func:`analyze_config` or the compatibility wrapper
:func:`resolve_config`. The module deliberately does not import a Pipeline, model, task,
trainer, or dataset before the exact visible experiment has been composed and validated.

Machine-local configuration is never discovered implicitly. A local YAML file affects
an invocation only when the caller supplies it explicitly.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from hashlib import sha256
from importlib import resources
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml

from phmfactory.pipelines import canonical_pipeline_name

DEFAULT_CONFIG = "configs/demo/01_cross_domain/cwru_dg.yaml"
DEFAULT_PIPELINE = "Pipeline_01_Fault_Diagnosis"

# Public, maintained aliases only. Historical v0.0.9 aliases stay in the
# protected compatibility loader and are not promoted as v0.3 public API.
MAINTAINED_PRESETS: dict[str, str] = {
    "quickstart": DEFAULT_CONFIG,
    "smoke": "configs/demo/00_smoke/dummy_dg.yaml",
    "cross-domain": "configs/demo/01_cross_domain/cwru_dg.yaml",
    "cross-system": "configs/demo/02_cross_system/multi_system_cddg.yaml",
    "few-shot": "configs/demo/03_fewshot/cwru_protonet.yaml",
}


@dataclass(frozen=True)
class ResolvedConfig:
    """Backward-compatible resolved configuration payload.

    New public code should prefer :class:`ConfigAnalysis`, which also records field
    provenance, diagnostics, source files, and the effective semantic hash.
    """

    requested: str
    path: Path
    data: dict[str, Any]
    pipeline: str
    overrides: dict[str, Any]


@dataclass(frozen=True)
class ConfigDiagnostic:
    """One stable, user-correctable configuration observation."""

    code: str
    severity: str
    message: str
    field: str = ""
    suggestion: str = ""

    def as_dict(self) -> dict[str, str]:
        """Return a JSON-serializable diagnostic record."""

        return {
            "code": self.code,
            "severity": self.severity,
            "message": self.message,
            "field": self.field,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True)
class ConfigAnalysis:
    """Immutable description of the exact effective configuration.

    ``effective_config_sha256`` hashes only the canonical effective configuration. It
    excludes the requested alias, source paths, installation path, and override spelling,
    so semantically identical invocations compare equal across checkouts and entrypoints.
    """

    requested: str
    path: Path
    effective_config: dict[str, Any]
    pipeline: str
    overrides: dict[str, Any]
    local_config_path: Path | None
    source_files: tuple[Path, ...]
    sources: dict[str, str]
    diagnostics: tuple[ConfigDiagnostic, ...]
    effective_config_sha256: str

    def runtime_config(self) -> dict[str, Any]:
        """Return a mutable copy for one runtime or inspection consumer."""

        return deepcopy(self.effective_config)

    def to_resolved_config(self) -> ResolvedConfig:
        """Return the legacy payload expected by ``CompiledRunSpec`` and callers."""

        return ResolvedConfig(
            requested=self.requested,
            path=self.path,
            data=self.runtime_config(),
            pipeline=self.pipeline,
            overrides=deepcopy(self.overrides),
        )

    def as_dict(self) -> dict[str, Any]:
        """Return a portable representation for CLI and UI adapters."""

        return {
            "requested": self.requested,
            "resolved_path": str(self.path),
            "effective_config": self.runtime_config(),
            "pipeline": self.pipeline,
            "overrides": deepcopy(self.overrides),
            "local_config_path": (
                str(self.local_config_path) if self.local_config_path is not None else None
            ),
            "source_files": [str(path) for path in self.source_files],
            "sources": dict(self.sources),
            "diagnostics": [item.as_dict() for item in self.diagnostics],
            "effective_config_sha256": self.effective_config_sha256,
        }


def parse_overrides(values: Sequence[str] | None) -> dict[str, Any]:
    """Parse repeated ``key=value`` overrides into a nested dictionary.

    Raises
    ------
    ValueError
        If a token is missing ``=``, has an empty key/value, contains malformed YAML,
        or attempts to traverse a non-mapping field.
    """

    parsed: dict[str, Any] = {}
    for item in values or ():
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item!r}. Use key=value format.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Override key must be non-empty")
        value_text = raw_value.strip()
        if not value_text:
            raise ValueError(
                f"Override value for {key!r} must be non-empty. "
                "Use `null` explicitly when a null value is intended."
            )
        try:
            value = yaml.safe_load(value_text)
        except yaml.YAMLError as exc:
            raise ValueError(
                f"Invalid YAML value for override {key!r}: {raw_value!r}"
            ) from exc
        _set_dotted(parsed, key, value)
    return parsed


def semantic_config_sha256(config: Mapping[str, Any]) -> str:
    """Return the stable SHA-256 of one effective configuration mapping."""

    payload = json.dumps(
        _json_compatible(config),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(payload).hexdigest()


def validate_complete_experiment(config: Mapping[str, Any]) -> None:
    """Validate one fully composed public experiment without rewriting it.

    Base fragments remain free to contain only their owned block. This function is
    called only after the public precedence chain has produced a complete experiment.
    Pydantic is used as an acceptance boundary, not as a second configuration compiler:
    defaults are not written back and no converted ``model_dump`` replaces the visible
    mapping.
    """

    from src.config_schema import ExperimentConfig

    before = deepcopy(config)
    ExperimentConfig.model_validate(config, strict=True)
    if config != before:
        raise RuntimeError(
            "ExperimentConfig validation mutated the visible effective configuration"
        )


def analyze_config(
    source: str | Path | None = None,
    *,
    override_values: Sequence[str] | None = None,
    local_config: str | Path | None = None,
) -> ConfigAnalysis:
    """Resolve and strictly validate one public experiment.

    Parameters
    ----------
    source:
        Maintained preset name or YAML path. ``None`` selects ``DEFAULT_CONFIG``.
    override_values:
        Repeatable dotted ``key=value`` tokens applied last.
    local_config:
        Optional YAML path applied explicitly between the experiment YAML and CLI
        overrides. No default local file is searched when this argument is omitted.

    Returns
    -------
    ConfigAnalysis
        Immutable effective configuration, provenance, diagnostics, and semantic hash.

    Raises
    ------
    FileNotFoundError, TypeError, ValueError, UnicodeError, yaml.YAMLError,
    pydantic.ValidationError
        The original configuration error is propagated. There is no fallback to another
        config or Pipeline, and invalid experiments fail before Pipeline import, output
        creation, or Factory construction.
    """

    requested = str(source or DEFAULT_CONFIG)
    path = resolve_config_path(source)
    data, sources, files = _load_recursive_with_sources(
        path,
        stack=(),
        source_kind="config",
    )

    local_path: Path | None = None
    if local_config is not None:
        local_path = _resolve_explicit_local_path(local_config)
        local_data, _, local_files = _load_recursive_with_sources(
            local_path,
            stack=(),
            source_kind="local",
        )
        _deep_merge(data, local_data)
        _mark_leaf_sources(sources, local_data, f"local:{local_path}")
        files.extend(local_files)

    overrides = parse_overrides(override_values)
    _deep_merge(data, overrides)
    _mark_leaf_sources(sources, overrides, "cli:--override")

    if "pipeline" not in data:
        raise ValueError(
            "The effective configuration must declare `pipeline`. Add it to the "
            "selected YAML, an explicit --local-config file, or an explicit "
            "--override pipeline=<canonical-name>."
        )
    raw_pipeline = data["pipeline"]
    if not isinstance(raw_pipeline, str) or not raw_pipeline.strip():
        raise ValueError("pipeline must be a non-empty string")
    pipeline = canonical_pipeline_name(raw_pipeline.strip())
    data["pipeline"] = pipeline

    effective = _json_compatible(deepcopy(data))
    validate_complete_experiment(effective)
    return ConfigAnalysis(
        requested=requested,
        path=path,
        effective_config=effective,
        pipeline=pipeline,
        overrides=_json_compatible(deepcopy(overrides)),
        local_config_path=local_path,
        source_files=_unique_paths(files),
        sources=dict(sorted(sources.items())),
        diagnostics=_basic_diagnostics(effective),
        effective_config_sha256=semantic_config_sha256(effective),
    )


def resolve_config(
    source: str | Path | None = None,
    *,
    override_values: Sequence[str] | None = None,
    local_config: str | Path | None = None,
) -> ResolvedConfig:
    """Compatibility wrapper around :func:`analyze_config`."""

    return analyze_config(
        source,
        override_values=override_values,
        local_config=local_config,
    ).to_resolved_config()


def _packaged_config_path(relative_path: str) -> Path | None:
    normalized = relative_path.replace("\\", "/")
    prefix = "configs/"
    if not normalized.startswith(prefix):
        return None
    try:
        resource = resources.files("configs").joinpath(
            *Path(normalized[len(prefix) :]).parts
        )
    except (ModuleNotFoundError, TypeError):
        return None
    candidate = Path(str(resource))
    return candidate.resolve() if candidate.is_file() else None


def _resolve_existing_config_path(
    source: str | Path,
    *,
    relative_to: Path | None = None,
) -> Path:
    requested = str(source)
    candidate = Path(requested).expanduser()
    if (
        not candidate.is_absolute()
        and relative_to is not None
        and not requested.replace("\\", "/").startswith("configs/")
    ):
        candidate = relative_to / candidate
    if candidate.is_file():
        return candidate.resolve()
    packaged = _packaged_config_path(requested)
    if packaged is not None:
        return packaged
    raise FileNotFoundError(requested)


def _resolve_explicit_local_path(source: str | Path) -> Path:
    candidate = Path(source).expanduser()
    if candidate.is_file():
        return candidate.resolve()
    raise FileNotFoundError(
        f"Explicit local configuration was not found: {candidate}. "
        "Remove --local-config or provide an existing YAML file."
    )


def resolve_config_path(source: str | Path | None) -> Path:
    """Resolve a public preset or YAML path from a checkout or installed wheel."""

    requested = str(source or DEFAULT_CONFIG)
    mapped = MAINTAINED_PRESETS.get(requested, requested)
    try:
        return _resolve_existing_config_path(mapped)
    except FileNotFoundError as exc:
        known = ", ".join(sorted(MAINTAINED_PRESETS))
        raise FileNotFoundError(
            f"Configuration {requested!r} was not found. Maintained presets: {known}"
        ) from exc


def load_config_dict(path: str | Path) -> dict[str, Any]:
    """Load YAML and recursively merge its ordered ``base_configs`` entries."""

    data, _, _ = _load_recursive_with_sources(
        Path(path).resolve(),
        stack=(),
        source_kind="config",
    )
    return data


def _load_recursive_with_sources(
    path: Path,
    *,
    stack: tuple[Path, ...],
    source_kind: str,
) -> tuple[dict[str, Any], dict[str, str], list[Path]]:
    path = path.resolve()
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Cyclic base_configs reference: {chain}")

    payload = _read_yaml_mapping(path)
    merged: dict[str, Any] = {}
    sources: dict[str, str] = {}
    files: list[Path] = []

    base_configs = payload.get("base_configs") or {}
    if not isinstance(base_configs, Mapping):
        raise TypeError(f"base_configs must be a mapping: {path}")
    for base_source in base_configs.values():
        base_path = _resolve_existing_config_path(
            str(base_source),
            relative_to=path.parent,
        )
        base_data, base_sources, base_files = _load_recursive_with_sources(
            base_path,
            stack=(*stack, path),
            source_kind="base",
        )
        _deep_merge(merged, base_data)
        sources.update(base_sources)
        files.extend(base_files)

    current = {key: value for key, value in payload.items() if key != "base_configs"}
    _deep_merge(merged, current)
    _mark_leaf_sources(sources, current, f"{source_kind}:{path}")
    files.append(path)
    return merged, sources, files


def _read_yaml_mapping(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as exc:
        raise UnicodeError(
            f"Configuration file must be valid UTF-8: {path}"
        ) from exc
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Top-level YAML object must be a mapping: {path}")
    return payload


def _set_dotted(target: dict[str, Any], key: str, value: Any) -> None:
    current = target
    parts = key.split(".")
    for part in parts[:-1]:
        existing = current.get(part)
        if existing is None:
            existing = {}
            current[part] = existing
        if not isinstance(existing, dict):
            raise ValueError(f"Cannot set nested key {key!r}: {part!r} is not a mapping")
        current = existing
    current[parts[-1]] = value


def _deep_merge(target: dict[str, Any], source: Mapping[str, Any]) -> None:
    for key, value in source.items():
        if isinstance(value, Mapping) and isinstance(target.get(key), dict):
            _deep_merge(target[key], value)
        else:
            target[key] = deepcopy(value)


def _leaf_items(
    value: Mapping[str, Any],
    *,
    prefix: str = "",
) -> Iterable[tuple[str, Any]]:
    for key, item in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(item, Mapping) and item:
            yield from _leaf_items(item, prefix=path)
        else:
            yield path, item


def _mark_leaf_sources(
    target: dict[str, str],
    value: Mapping[str, Any],
    label: str,
) -> None:
    for path, _ in _leaf_items(value):
        target[path] = label


def _unique_paths(paths: Iterable[Path]) -> tuple[Path, ...]:
    result: list[Path] = []
    seen: set[Path] = set()
    for path in paths:
        resolved = path.resolve()
        if resolved not in seen:
            seen.add(resolved)
            result.append(resolved)
    return tuple(result)


def _json_compatible(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "Configuration values must be mappings, sequences, paths, or JSON scalars; "
        f"got {type(value).__name__}"
    )


def _basic_diagnostics(config: Mapping[str, Any]) -> tuple[ConfigDiagnostic, ...]:
    diagnostics: list[ConfigDiagnostic] = []
    for block in ("environment", "data", "model", "task", "trainer"):
        if not isinstance(config.get(block), Mapping):
            diagnostics.append(
                ConfigDiagnostic(
                    code="missing_required_block",
                    severity="error",
                    field=block,
                    message=f"The effective configuration has no mapping block {block!r}.",
                    suggestion=f"Add `{block}:` directly or through base_configs.",
                )
            )
    return tuple(diagnostics)
