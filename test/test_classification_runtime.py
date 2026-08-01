from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest

from phmfactory.config import ResolvedConfig
from phmfactory.runtime import CompiledRunSpec
from src.runtime import classification


def _config(tmp_path: Path, *, iterations: int = 1) -> dict:
    return {
        "pipeline": "Pipeline_01_Fault_Diagnosis",
        "environment": {
            "iterations": iterations,
            "seed": 7,
            "output_dir": str(tmp_path),
            "project": "runtime-test",
            "wandb": False,
            "swanlab": False,
        },
        "data": {"data_dir": str(tmp_path), "metadata_file": "dummy.csv"},
        "model": {"name": "dummy", "type": "dummy"},
        "task": {"name": "classification", "type": "DG"},
        "trainer": {"device": "cpu", "gpus": 1},
    }


def _args(tmp_path: Path, *, iterations: int = 1) -> Namespace:
    resolved = ResolvedConfig(
        requested="smoke",
        path=tmp_path / "smoke.yaml",
        data=_config(tmp_path, iterations=iterations),
        pipeline="Pipeline_01_Fault_Diagnosis",
        overrides={},
    )
    compiled = CompiledRunSpec.compile(resolved)
    return Namespace(
        config="smoke",
        config_path=str(resolved.path),
        requested_config="smoke",
        resolved_pipeline=resolved.pipeline,
        compiled_run_spec=compiled,
        override=["trainer.num_epochs=99"],
        notes="",
    )


def test_compiled_config_bypasses_legacy_reparse(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    args = _args(tmp_path)
    monkeypatch.setattr(
        classification,
        "merge_with_local_override",
        lambda *a, **k: pytest.fail("legacy loader must not run"),
    )
    configs = classification.load_runtime_config(args)
    assert configs.environment.seed == 7
    assert not hasattr(configs.trainer, "num_epochs")


def test_missing_required_section_fails_closed(tmp_path: Path) -> None:
    args = _args(tmp_path)
    data = args.compiled_run_spec.runtime_config()
    data.pop("task")
    args.compiled_run_spec = CompiledRunSpec.compile(
        ResolvedConfig(
            requested="broken",
            path=tmp_path / "broken.yaml",
            data=data,
            pipeline="Pipeline_01_Fault_Diagnosis",
            overrides={},
        )
    )
    with pytest.raises(ValueError, match="missing required section.*task"):
        classification.load_runtime_config(args)


def test_zero_iterations_fails_before_factory_construction(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="iterations must be positive"):
        classification.run_classification_pipeline(_args(tmp_path, iterations=0))


def test_runtime_closes_data_and_lab_when_training_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []

    class DataFactory:
        data = SimpleNamespace(close=lambda: events.append("data-close"))

        def get_metadata(self):
            return {0: {"Label": 0, "Domain_id": 0}}

        def get_dataloader(self, split: str):
            return split

    class Trainer:
        def fit(self, *args):
            raise RuntimeError("fit failed")

    monkeypatch.setattr(
        classification,
        "path_name",
        lambda configs, iteration: (str(tmp_path / "run"), "run"),
    )
    monkeypatch.setattr(
        classification,
        "seed_everything",
        lambda seed: events.append(f"seed:{seed}"),
    )
    monkeypatch.setattr(classification, "init_lab", lambda *args: events.append("lab-open"))
    monkeypatch.setattr(classification, "close_lab", lambda: events.append("lab-close"))
    monkeypatch.setattr(classification, "build_data", lambda *args: DataFactory())
    monkeypatch.setattr(classification, "build_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(classification, "build_task", lambda **kwargs: object())
    monkeypatch.setattr(classification, "build_trainer", lambda *args: Trainer())

    with pytest.raises(RuntimeError, match="fit failed"):
        classification.run_classification_pipeline(_args(tmp_path))

    assert events[-2:] == ["data-close", "lab-close"]


def test_pipeline_wrappers_only_select_hooks(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.Pipeline_01_Fault_Diagnosis as pipeline_01
    import src.Pipeline_05_Explainable_Fault_Diagnosis as pipeline_05

    calls: list[object] = []
    monkeypatch.setattr(
        pipeline_01,
        "run_classification_pipeline",
        lambda args: calls.append(("p01", args)) or [],
    )
    monkeypatch.setattr(
        pipeline_05,
        "run_classification_pipeline",
        lambda args, hooks: calls.append(("p05", args, type(hooks).__name__)) or [],
    )

    marker = object()
    assert pipeline_01.pipeline(marker) == []
    assert pipeline_05.pipeline(marker) == []
    assert calls == [("p01", marker), ("p05", marker, "ExplainabilityHooks")]
