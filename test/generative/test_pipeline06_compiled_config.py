from __future__ import annotations

from types import SimpleNamespace

import pytest

from phmfactory.runtime import pipeline06_adapter
import src.Pipeline_06_Generative_Modeling as implementation


class _Compiled:
    pipeline = "Pipeline_06_Generative_Modeling"

    def __init__(self, payload):
        self.payload = payload

    def runtime_config(self):
        return self.payload


def _payload(mode: str = "train") -> dict:
    return {
        "pipeline": "Pipeline_06_Generative_Modeling",
        "environment": {"iterations": 1},
        "data": {},
        "model": {},
        "task": {"generative": {"mode": mode}},
        "trainer": {},
    }


def test_compiled_config_bypasses_legacy_yaml_reload(monkeypatch) -> None:
    monkeypatch.setattr(
        implementation,
        "_load_configs",
        lambda args: pytest.fail("legacy config loader must not run"),
    )
    args = SimpleNamespace(
        compiled_run_spec=_Compiled(_payload()),
        resolved_pipeline="Pipeline_06_Generative_Modeling",
    )

    configs = pipeline06_adapter._runtime_config(args, implementation)

    assert configs.task.generative.mode == "train"
    assert configs.environment.iterations == 1


def test_compiled_pipeline_mismatch_fails_closed() -> None:
    compiled = _Compiled(_payload())
    compiled.pipeline = "Pipeline_01_Fault_Diagnosis"
    args = SimpleNamespace(
        compiled_run_spec=compiled,
        resolved_pipeline="Pipeline_06_Generative_Modeling",
    )

    with pytest.raises(ValueError, match="compiled Pipeline mismatch"):
        pipeline06_adapter._runtime_config(args, implementation)


def test_adapter_dispatches_compiled_stage_without_reparse(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        implementation,
        "_load_configs",
        lambda args: pytest.fail("legacy config loader must not run"),
    )
    monkeypatch.setattr(
        implementation,
        "_run_train_stage",
        lambda args, configs, iteration: calls.append((configs, iteration))
        or {"stage": "train", "iteration": iteration},
    )
    args = SimpleNamespace(
        compiled_run_spec=_Compiled(_payload()),
        resolved_pipeline="Pipeline_06_Generative_Modeling",
    )

    result = pipeline06_adapter.pipeline(args)

    assert result == {
        "status": "succeeded",
        "stage": "train",
        "iterations": [{"stage": "train", "iteration": 0}],
    }
    assert len(calls) == 1
    assert calls[0][0].task.generative.mode == "train"
