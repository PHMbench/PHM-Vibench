"""Hydra/OmegaConf adapter for PHM-Vibench config composition.

The public runtime still expects section-shaped configs:
``environment/data/model/task/trainer`` plus a top-level ``pipeline``. This module
keeps Hydra as a replaceable composition layer rather than leaking Hydra objects into
factories and pipelines.
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import yaml


def compose_hydra_file(config_path: Path, overrides: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Compose a config under ``configs/hydra`` and return a plain dict.

    Parameters
    ----------
    config_path:
        YAML file inside ``configs/hydra``.
    overrides:
        Optional Hydra override strings. CLI ``--override`` is still applied later by
        existing runtime code; this hook is for tooling that wants native Hydra compose.
    """

    try:
        from hydra import compose, initialize_config_dir
        from omegaconf import OmegaConf
    except ModuleNotFoundError:
        return _compose_static(config_path, overrides=overrides)

    path = config_path.resolve()
    hydra_root = _find_hydra_root(path)
    rel = path.relative_to(hydra_root).with_suffix("")

    with initialize_config_dir(config_dir=str(hydra_root), version_base=None):
        cfg = compose(config_name=rel.as_posix(), overrides=list(overrides or []))

    plain = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(plain, dict):
        raise ValueError(f"Hydra config must compose to a mapping: {config_path}")

    return plain


def _compose_static(config_path: Path, overrides: Optional[Sequence[str]] = None) -> Dict[str, Any]:
    """Minimal offline composer for this repo's Hydra-style YAMLs.

    This is intentionally narrow: it supports defaults entries like
    ``/group: name``, ``_self_``, group packages, and scalar/list/dict YAML values.
    Real Hydra remains the preferred runtime when installed.
    """

    if overrides:
        raise RuntimeError(
            "Hydra override composition requires hydra-core/omegaconf. "
            "Install with: pip install -r requirements.txt"
        )

    path = config_path.resolve()
    hydra_root = _find_hydra_root(path)
    current = _load_yaml(path)
    defaults = current.get("defaults", [])
    if defaults and not isinstance(defaults, list):
        raise ValueError(f"Hydra defaults must be a list: {config_path}")

    merged: Dict[str, Any] = {}
    apply_self_at_end = "_self_" not in defaults
    for item in defaults:
        if item == "_self_":
            _deep_merge(merged, _strip_defaults(current))
            apply_self_at_end = False
            continue
        group, name = _parse_default_item(item, config_path)
        group_path = hydra_root / group / f"{name}.yaml"
        group_payload = _load_yaml(group_path)
        _deep_merge(merged, group_payload)

    if apply_self_at_end:
        _deep_merge(merged, _strip_defaults(current))

    return _resolve_oc_env(merged)


def _parse_default_item(item: Any, config_path: Path) -> tuple[str, str]:
    if not isinstance(item, dict) or len(item) != 1:
        raise ValueError(f"Unsupported Hydra defaults entry in {config_path}: {item!r}")
    group, name = next(iter(item.items()))
    if not isinstance(group, str) or not isinstance(name, str):
        raise ValueError(f"Unsupported Hydra defaults entry in {config_path}: {item!r}")
    return group.lstrip("/"), name


def _strip_defaults(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: v for k, v in payload.items() if k != "defaults"}


def _load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Hydra config file not found: {path}")
    text = path.read_text(encoding="utf-8")
    payload = yaml.safe_load(text) or {}
    if not isinstance(payload, dict):
        raise ValueError(f"Hydra config must be a mapping: {path}")
    package = _detect_package(text)
    if package and package != "_global_":
        return {package: payload}
    return payload


def _detect_package(text: str) -> Optional[str]:
    for line in text.splitlines()[:5]:
        stripped = line.strip()
        if stripped.startswith("# @package "):
            return stripped.removeprefix("# @package ").strip()
    return None


def _deep_merge(dst: Dict[str, Any], src: Dict[str, Any]) -> Dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_merge(dst[key], value)
        else:
            dst[key] = value
    return dst


_OC_ENV_PATTERN = re.compile(r"\$\{oc\.env:([A-Za-z_][A-Za-z0-9_]*),([^}]*)\}")


def _resolve_oc_env(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _resolve_oc_env(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve_oc_env(v) for v in value]
    if not isinstance(value, str):
        return value

    def replace(match: re.Match[str]) -> str:
        env_name = match.group(1)
        default = match.group(2)
        return os.environ.get(env_name, default)

    return _OC_ENV_PATTERN.sub(replace, value)


def _find_hydra_root(path: Path) -> Path:
    parts = path.parts
    for index, part in enumerate(parts):
        if part == "configs" and index + 1 < len(parts) and parts[index + 1] == "hydra":
            return Path(*parts[: index + 2])
    raise ValueError(f"Hydra config must live under configs/hydra: {path}")
