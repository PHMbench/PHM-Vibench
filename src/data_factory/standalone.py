"""Configuration bridge for the optional ``phm-data-factory`` backend."""

from __future__ import annotations

import importlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any


EXPECTED_PROVIDER_VERSION = "0.2.0"
MISSING_BACKEND_MESSAGE = (
    "phm-data-factory is optional and is not installed. "
    "Install the approved backend and its required extra before selecting "
    "data.factory_name: phm_data."
)


def _value(config: Any, name: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(name, default)
    return getattr(config, name, default)


def _load_provider() -> Any:
    """Import the installed provider without changing import resolution."""

    try:
        provider = importlib.import_module("phm_data_factory")
    except ModuleNotFoundError as exc:
        if exc.name != "phm_data_factory":
            raise
        raise ModuleNotFoundError(MISSING_BACKEND_MESSAGE) from exc

    version = getattr(provider, "__version__", None)
    if version != EXPECTED_PROVIDER_VERSION:
        raise RuntimeError(
            "PHMFactory requires phm-data-factory 0.2.0 for the phm_data "
            f"backend; imported version {version!r}."
        )
    if not callable(getattr(provider, "connect", None)):
        raise RuntimeError("Installed phm-data-factory has no callable connect API.")
    return provider


def _configured_backend(args_data: Any) -> Mapping[str, Any] | Path:
    configured = _value(args_data, "phm_data_config")
    if not configured:
        raise ValueError(
            "data.factory_name=phm_data requires data.phm_data_config"
        )
    if isinstance(configured, Mapping):
        return dict(configured)
    if not isinstance(configured, (str, Path)):
        raise TypeError("data.phm_data_config must be a path or mapping")
    return Path(configured).expanduser().resolve()


def build_data_repository(args_data: Any) -> Any:
    """Open the explicitly configured provider; never select a fallback."""

    configured = _configured_backend(args_data)
    return _load_provider().connect(configured)
