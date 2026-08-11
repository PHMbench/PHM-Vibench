"""Compile one analyzed configuration into the immutable runtime contract."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from phmfactory.config import ResolvedConfig, semantic_config_sha256


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
    """Immutable runtime contract for one public invocation.

    ``effective_config_sha256`` identifies only the canonical effective configuration
    and therefore compares equal across preset names, equivalent override spellings, and
    installation paths. ``sha256`` retains the invocation contract identity, including
    the requested source and explicit overrides. Runtime adapters use
    :meth:`runtime_config` rather than reparsing YAML.
    """

    schema_version: int
    requested_config: str
    resolved_config_path: str
    pipeline: str
    config: dict[str, Any]
    overrides: dict[str, Any]
    effective_config_sha256: str
    sha256: str

    @classmethod
    def compile(cls, resolved: ResolvedConfig) -> "CompiledRunSpec":
        """Compile a compatibility ``ResolvedConfig`` without changing its semantics."""

        config = _plain(deepcopy(resolved.data))
        overrides = _plain(deepcopy(resolved.overrides))
        effective_digest = semantic_config_sha256(config)
        invocation_payload = {
            "schema_version": 1,
            "requested_config": resolved.requested,
            "pipeline": resolved.pipeline,
            "effective_config_sha256": effective_digest,
            "overrides": overrides,
        }
        return cls(
            schema_version=1,
            requested_config=resolved.requested,
            resolved_config_path=str(resolved.path),
            pipeline=resolved.pipeline,
            config=config,
            overrides=overrides,
            effective_config_sha256=effective_digest,
            sha256=_digest(invocation_payload),
        )

    def runtime_config(self) -> dict[str, Any]:
        """Return a mutable copy for one protected-runtime invocation."""

        return deepcopy(self.config)

    def as_dict(self) -> dict[str, Any]:
        """Return a serializable representation of both configuration identities."""

        return {
            "schema_version": self.schema_version,
            "requested_config": self.requested_config,
            "resolved_config_path": self.resolved_config_path,
            "pipeline": self.pipeline,
            "config": self.runtime_config(),
            "overrides": deepcopy(self.overrides),
            "effective_config_sha256": self.effective_config_sha256,
            "sha256": self.sha256,
        }
