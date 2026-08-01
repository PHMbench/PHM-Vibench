"""Compile a resolved configuration into one deterministic runtime contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from phmfactory.config import ResolvedConfig


def _plain(value: Any) -> Any:
    """Return a deterministic JSON-compatible representation."""
    if isinstance(value, Mapping):
        return {str(key): _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    raise TypeError(
        "Compiled run specs support only mappings, sequences, paths, and JSON scalars; "
        f"got {type(value).__name__}"
    )


def _digest(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        _plain(payload),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return sha256(encoded).hexdigest()


@dataclass(frozen=True)
class CompiledRunSpec:
    """Immutable metadata for the exact configuration selected by the public CLI.

    The nested configuration is copied at construction. Runtime adapters should use
    :meth:`runtime_config` to obtain their own mutable copy rather than reparsing YAML.
    The resolved absolute path is recorded for diagnostics but deliberately excluded
    from ``sha256`` so the same packaged configuration hashes identically in different
    installation directories.
    """

    schema_version: int
    requested_config: str
    resolved_config_path: str
    pipeline: str
    config: dict[str, Any]
    overrides: dict[str, Any]
    sha256: str

    @classmethod
    def compile(cls, resolved: ResolvedConfig) -> "CompiledRunSpec":
        config = _plain(deepcopy(resolved.data))
        overrides = _plain(deepcopy(resolved.overrides))
        semantic_payload = {
            "schema_version": 1,
            "requested_config": resolved.requested,
            "pipeline": resolved.pipeline,
            "config": config,
            "overrides": overrides,
        }
        return cls(
            schema_version=1,
            requested_config=resolved.requested,
            resolved_config_path=str(resolved.path),
            pipeline=resolved.pipeline,
            config=config,
            overrides=overrides,
            sha256=_digest(semantic_payload),
        )

    def runtime_config(self) -> dict[str, Any]:
        """Return a mutable copy for one protected-runtime invocation."""
        return deepcopy(self.config)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation including the semantic digest."""
        return {
            "schema_version": self.schema_version,
            "requested_config": self.requested_config,
            "resolved_config_path": self.resolved_config_path,
            "pipeline": self.pipeline,
            "config": self.runtime_config(),
            "overrides": deepcopy(self.overrides),
            "sha256": self.sha256,
        }
