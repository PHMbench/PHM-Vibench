from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from phmfactory.config import ResolvedConfig
from phmfactory.runtime import CompiledRunSpec


def _resolved(*, epochs: int = 1, requested: str = "smoke") -> ResolvedConfig:
    return ResolvedConfig(
        requested=requested,
        path=Path("/tmp/installation-a/configs/demo/00_smoke/dummy_dg.yaml"),
        data={
            "pipeline": "Pipeline_01_Fault_Diagnosis",
            "environment": {"seed": 42},
            "data": {"batch_size": 8},
            "model": {"name": "M_01_ISFM"},
            "task": {"name": "classification"},
            "trainer": {"num_epochs": epochs},
        },
        pipeline="Pipeline_01_Fault_Diagnosis",
        overrides={"trainer": {"num_epochs": epochs}},
    )


def test_compiled_run_spec_is_deterministic_and_path_independent() -> None:
    first = CompiledRunSpec.compile(_resolved())
    second_resolved = _resolved()
    second_resolved = ResolvedConfig(
        requested=second_resolved.requested,
        path=Path("/opt/another-install/configs/demo/00_smoke/dummy_dg.yaml"),
        data=second_resolved.data,
        pipeline=second_resolved.pipeline,
        overrides=second_resolved.overrides,
    )
    second = CompiledRunSpec.compile(second_resolved)

    assert first.sha256 == second.sha256
    assert first.resolved_config_path != second.resolved_config_path


def test_compiled_run_spec_changes_when_execution_semantics_change() -> None:
    assert CompiledRunSpec.compile(_resolved(epochs=1)).sha256 != CompiledRunSpec.compile(
        _resolved(epochs=2)
    ).sha256


def test_runtime_config_is_isolated_from_compiled_contract() -> None:
    spec = CompiledRunSpec.compile(_resolved())
    before = deepcopy(spec.config)
    runtime = spec.runtime_config()
    runtime["trainer"]["num_epochs"] = 99

    assert spec.config == before
    assert spec.config["trainer"]["num_epochs"] == 1
