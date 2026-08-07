"""Compile one analyzed configuration into the immutable runtime contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from phmfactory.config import ResolvedConfig


def _plain(value: Any) -> Any:
    """Return a JSON-compatible representation for runtime handoff."""

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


@dataclass(frozen=True)
class CompiledRunSpec:
    """Immutable configuration-to-runtime handoff for one public invocation.

    The spec freezes the already resolved configuration and request information. Runtime
    adapters consume :meth:`runtime_config` rather than reparsing YAML or rebuilding a
    second identity for the same experiment.
    """

    schema_version: int
    requested_config: str
    resolved_config_path: str
    pipeline: str
    config: dict[str, Any]
    overrides: dict[str, Any]

    @classmethod
    def compile(cls, resolved: ResolvedConfig) -> "CompiledRunSpec":
        """Compile a compatibility ``ResolvedConfig`` without changing its semantics."""

        return cls(
            schema_version=1,
            requested_config=resolved.requested,
            resolved_config_path=str(resolved.path),
            pipeline=resolved.pipeline,
            config=_plain(deepcopy(resolved.data)),
            overrides=_plain(deepcopy(resolved.overrides)),
        )

    def runtime_config(self) -> dict[str, Any]:
        """Return a mutable copy for one protected-runtime invocation."""

        return deepcopy(self.config)

    def as_dict(self) -> dict[str, Any]:
        """Return the serializable runtime handoff."""

        return {
            "schema_version": self.schema_version,
            "requested_config": self.requested_config,
            "resolved_config_path": self.resolved_config_path,
            "pipeline": self.pipeline,
            "config": self.runtime_config(),
            "overrides": deepcopy(self.overrides),
        }
