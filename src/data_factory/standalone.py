"""Bridge PHM-Vibench configuration to the standalone data backend."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def _load_module():
    try:
        import phm_data_factory
    except ModuleNotFoundError:
        package_src = (
            Path(__file__).resolve().parents[2]
            / "packages"
            / "phm-data-factory"
            / "src"
        )
        if not package_src.exists():
            raise ModuleNotFoundError(
                "Install phm-data-factory or initialize packages/phm-data-factory"
            )
        sys.path.insert(0, str(package_src))
        import phm_data_factory
    return phm_data_factory


def _value(config: Any, name: str, default: Any = None) -> Any:
    return config.get(name, default) if isinstance(config, dict) else getattr(
        config, name, default
    )


def _configured_backend(args_data: Any):
    configured = _value(args_data, "phm_data_config")
    if not configured:
        raise ValueError("data.factory_name=phm_data requires data.phm_data_config")
    if isinstance(configured, dict):
        return configured
    path = Path(configured).expanduser()
    if path.is_absolute():
        return path.resolve()
    base = Path(_value(args_data, "data_dir", ".") or ".").expanduser().resolve()
    return (base / path).resolve()


def build_data_repository(args_data: Any, signal_path=None):
    """Return the configured repository; never silently select a fallback."""

    del signal_path
    return _load_module().connect(_configured_backend(args_data))


def build_data_backend(args_data: Any, signal_path=None):
    return build_data_repository(args_data, signal_path=signal_path)


def build_agent_data_tools(
    args_data: Any,
    signal_path=None,
    default_max_points: int | None = None,
    profile: str = "benchmark_public",
):
    del signal_path
    tools = _load_module().connect_agent(
        _configured_backend(args_data), profile=profile
    )
    max_points = default_max_points or _value(args_data, "agent_max_points")
    if max_points is not None:
        tools.default_max_points = int(max_points)
    return tools
