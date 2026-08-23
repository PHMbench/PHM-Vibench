from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from src.runtime import classification


def _runtime_config(output_dir: Path) -> SimpleNamespace:
    return SimpleNamespace(
        environment=SimpleNamespace(
            project="runtime-environment-contract",
            seed=7,
            iterations=1,
            output_dir=str(output_dir),
        ),
        data=SimpleNamespace(),
        model=SimpleNamespace(),
        task=SimpleNamespace(name="classification"),
        trainer=SimpleNamespace(test_after_fit=True),
    )


def _fail_if_data_factory_runs(*args, **kwargs):
    del args, kwargs
    pytest.fail(
        "invalid environment execution fields must fail before Data Factory construction"
    )


@pytest.mark.parametrize("field", ("seed", "iterations"))
def test_classification_runtime_has_no_missing_environment_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    output_dir = tmp_path / "must-not-exist"
    configs = _runtime_config(output_dir)
    delattr(configs.environment, field)
    monkeypatch.setattr(classification, "load_runtime_config", lambda args: configs)
    monkeypatch.setattr(classification, "build_data", _fail_if_data_factory_runs)

    with pytest.raises(ValueError, match=rf"environment\.{field} is required"):
        classification.run_classification_pipeline(object())

    assert not output_dir.exists()


@pytest.mark.parametrize(
    "field,value",
    (("seed", "7"), ("iterations", "1")),
)
def test_classification_runtime_does_not_coerce_environment_strings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: str,
) -> None:
    output_dir = tmp_path / "must-not-exist"
    configs = _runtime_config(output_dir)
    setattr(configs.environment, field, value)
    monkeypatch.setattr(classification, "load_runtime_config", lambda args: configs)
    monkeypatch.setattr(classification, "build_data", _fail_if_data_factory_runs)

    with pytest.raises(TypeError, match=rf"environment\.{field} must be an integer"):
        classification.run_classification_pipeline(object())

    assert not output_dir.exists()
