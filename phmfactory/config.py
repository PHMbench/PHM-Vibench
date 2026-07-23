"""Public configuration resolution for PHMFactory.

This module resolves a maintained preset or YAML path, applies CLI-style dotted
key overrides, and returns a plain dictionary without importing the protected
training runtime.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    """Resolved public configuration metadata and payload."""

    requested: str
    path: Path
    data: dict[str, Any]
    pipeline: str
    overrides: dict[str, Any]


def parse_overrides(values: Sequence[str] | None) -> dict[str, Any]:
    """Parse repeated ``key=value`` overrides into a nested dictionary."""
    parsed: dict[str, Any] = {}
    for item in values or ():
        if "=" not in item:
            raise ValueError(f"Invalid override format: {item!r}. Use key=value format.")
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError("Override key must be non-empty")
        try:
            value = yaml.safe_load(raw_value.strip())
        except yaml.YAMLError:
            value = raw_value.strip()
        _set_dotted(parsed, key, value)
    return parsed


def resolve_config_path(source: str | Path | None) -> Path:
    """Resolve a public preset or YAML path without changing the working tree."""
    requested = str(source or DEFAULT_CONFIG)
    candidate = Path(MAINTAINED_PRESETS.get(requested, requested)).expanduser()
    if not candidate.is_file():
        known = ", ".join(sorted(MAINTAINED_PRESETS))
        raise FileNotFoundError(
            f"Configuration {requested!r} was not found. Maintained presets: {known}"
        )
    return candidate.resolve()


def load_config_dict(path: str | Path) -> dict[str, Any]:
    """Load YAML and recursively merge its ordered ``base_configs`` entries."""
    return _load_recursive(Path(path).resolve(), stack=())


def resolve_config(
    source: str | Path | None = None,
    *,
    override_values: Sequence[str] | None = None,
) -> ResolvedConfig:
    """Resolve path, base configs, overrides, and canonical Pipeline name."""
    requested = str(source or DEFAULT_CONFIG)
    path = resolve_config_path(source)
    data = load_config_dict(path)
    overrides = parse_overrides(override_values)
    _deep_merge(data, overrides)
    raw_pipeline = data.get("pipeline", DEFAULT_PIPELINE)
    if not isinstance(raw_pipeline, str) or not raw_pipeline.strip():
        raise ValueError("pipeline must be a non-empty string")
    pipeline = canonical_pipeline_name(raw_pipeline.strip())
    data["pipeline"] = pipeline
    return ResolvedConfig(
        requested=requested,
        path=path,
        data=data,
        pipeline=pipeline,
        overrides=overrides,
    )


def _load_recursive(path: Path, *, stack: tuple[Path, ...]) -> dict[str, Any]:
    if path in stack:
        chain = " -> ".join(str(item) for item in (*stack, path))
        raise ValueError(f"Cyclic base_configs reference: {chain}")
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except UnicodeDecodeError:
        payload = yaml.safe_load(path.read_text(encoding="gb18030", errors="ignore")) or {}
    if not isinstance(payload, dict):
        raise TypeError(f"Top-level YAML object must be a mapping: {path}")

    merged: dict[str, Any] = {}
    base_configs = payload.get("base_configs") or {}
    if not isinstance(base_configs, Mapping):
        raise TypeError(f"base_configs must be a mapping: {path}")
    for base_source in base_configs.values():
        base_path = Path(str(base_source)).expanduser()
        if not base_path.is_absolute() and not str(base_path).startswith("configs/"):
            base_path = path.parent / base_path
        else:
            base_path = base_path.resolve()
        _deep_merge(merged, _load_recursive(base_path.resolve(), stack=(*stack, path)))

    current = {key: value for key, value in payload.items() if key != "base_configs"}
    _deep_merge(merged, current)
    return merged


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
