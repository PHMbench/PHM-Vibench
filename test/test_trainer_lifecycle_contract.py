from __future__ import annotations

from argparse import Namespace
import importlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from phmfactory.config import ResolvedConfig
from phmfactory.runtime import CompiledRunSpec
from src.runtime import classification


def _compiled_args(tmp_path: Path, trainer: dict) -> Namespace:
    config = {
        "pipeline": "Pipeline_01_Fault_Diagnosis",
        "environment": {
            "project": "trainer-lifecycle-contract",
            "seed": 7,
            "iterations": 1,
            "output_dir": str(tmp_path / "outputs"),
            "wandb": False,
            "swanlab": False,
        },
        "data": {"data_dir": str(tmp_path), "metadata_file": "dummy.csv"},
        "model": {"type": "dummy", "name": "dummy"},
        "task": {"type": "DG", "name": "classification"},
        "trainer": dict(trainer),
    }
    resolved = ResolvedConfig(
        requested="direct-test",
        path=tmp_path / "direct-test.yaml",
        data=config,
        pipeline="Pipeline_01_Fault_Diagnosis",
        overrides={},
    )
    return Namespace(
        config="direct-test",
        config_path=str(resolved.path),
        requested_config="direct-test",
        resolved_pipeline=resolved.pipeline,
        compiled_run_spec=CompiledRunSpec.compile(resolved),
        override=None,
        notes="",
    )


def _default_trainer_module():
    return importlib.import_module("src.trainer_factory.Default_trainer")


@pytest.mark.parametrize(
    ("trainer", "error_type", "message"),
    [
        (
            {"num_epochs": 1, "device": "cpu", "gpus": 1},
            ValueError,
            "trainer.test_after_fit is required",
        ),
        (
            {
                "num_epochs": 1,
                "test_after_fit": "false",
                "device": "cpu",
                "gpus": 1,
            },
            TypeError,
            "trainer.test_after_fit must be a boolean",
        ),
    ],
)
def test_classification_requires_exact_post_fit_policy_before_side_effects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    trainer: dict,
    error_type: type[Exception],
    message: str,
) -> None:
    monkeypatch.setattr(
        classification,
        "path_name",
        lambda *args, **kwargs: pytest.fail(
            "invalid lifecycle config must fail before output-path selection"
        ),
    )
    monkeypatch.setattr(
        classification,
        "build_data",
        lambda *args, **kwargs: pytest.fail(
            "invalid lifecycle config must fail before Data Factory construction"
        ),
    )

    with pytest.raises(error_type, match=message):
        classification.run_classification_pipeline(
            _compiled_args(tmp_path, trainer)
        )

    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    ("args", "error_type", "message"),
    [
        (Namespace(), ValueError, "trainer.num_epochs is required"),
        (
            Namespace(max_epochs=3),
            ValueError,
            "trainer.max_epochs is unsupported",
        ),
        (
            Namespace(num_epochs="3"),
            TypeError,
            "trainer.num_epochs must be an integer",
        ),
        (
            Namespace(num_epochs=True),
            TypeError,
            "trainer.num_epochs must be an integer",
        ),
        (
            Namespace(num_epochs=0),
            ValueError,
            "trainer.num_epochs must be positive",
        ),
    ],
)
def test_default_trainer_has_no_epoch_alias_or_type_fallback(
    args: Namespace,
    error_type: type[Exception],
    message: str,
) -> None:
    module = _default_trainer_module()
    before = vars(args).copy()

    with pytest.raises(error_type, match=message):
        module.resolve_epoch_contract(args)

    assert vars(args) == before


def test_epoch_contract_returns_exact_visible_value_without_mutation() -> None:
    module = _default_trainer_module()
    args = Namespace(num_epochs=7)
    before = vars(args).copy()

    assert module.resolve_epoch_contract(args) == 7
    assert vars(args) == before


def test_explicit_false_skips_test_without_becoming_true(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[str] = []
    run_path = tmp_path / "run" / "iter_0"
    checkpoint = tmp_path / "best.ckpt"
    checkpoint.write_text("checkpoint\n", encoding="utf-8")

    class DataFactory:
        data = SimpleNamespace(close=lambda: events.append("data-close"))

        def get_metadata(self):
            return {0: {"Label": 0, "Domain_id": 0}}

        def get_dataloader(self, split: str):
            return split

    class Trainer:
        def fit(self, *args):
            events.append("fit")

        def test(self, *args):
            pytest.fail("trainer.test must not run when test_after_fit=false")

    monkeypatch.setattr(
        classification,
        "path_name",
        lambda configs, iteration: (str(run_path), "run"),
    )
    monkeypatch.setattr(classification, "seed_everything", lambda seed: None)
    monkeypatch.setattr(classification, "init_lab", lambda *args: None)
    monkeypatch.setattr(classification, "close_lab", lambda: None)
    monkeypatch.setattr(classification, "build_data", lambda *args: DataFactory())
    monkeypatch.setattr(classification, "build_model", lambda *args, **kwargs: object())
    monkeypatch.setattr(classification, "build_task", lambda **kwargs: object())
    monkeypatch.setattr(classification, "build_trainer", lambda *args: Trainer())
    monkeypatch.setattr(
        classification,
        "load_best_model_checkpoint",
        lambda task, trainer: task,
    )
    monkeypatch.setattr(
        classification,
        "_best_checkpoint_path",
        lambda trainer: checkpoint.resolve(),
    )

    result = classification.run_classification_pipeline(
        _compiled_args(
            tmp_path,
            {
                "num_epochs": 1,
                "test_after_fit": False,
                "device": "cpu",
                "gpus": 1,
            },
        )
    )

    assert events == ["fit", "data-close"]
    assert result["status"] == "succeeded"
    assert result["test_metrics"] is None
    assert result["run_summary"] is None
    assert result["primary_metrics"] == {}
