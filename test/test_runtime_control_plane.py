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


def test_compiled_run_spec_preserves_resolved_execution_inputs() -> None:
    resolved = _resolved(epochs=2, requested="experiment")
    spec = CompiledRunSpec.compile(resolved)

    assert spec.requested_config == "experiment"
    assert spec.resolved_config_path == str(resolved.path)
    assert spec.pipeline == resolved.pipeline
    assert spec.config == resolved.data
    assert spec.overrides == resolved.overrides


def test_runtime_config_is_isolated_from_compiled_contract() -> None:
    spec = CompiledRunSpec.compile(_resolved())
    before = deepcopy(spec.config)
    runtime = spec.runtime_config()
    runtime["trainer"]["num_epochs"] = 99

    assert spec.config == before
    assert spec.config["trainer"]["num_epochs"] == 1
