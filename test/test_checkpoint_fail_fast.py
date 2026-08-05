from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from pytorch_lightning.callbacks import ModelCheckpoint

from src.utils.pipeline_config.base_utils import load_pretrained_weights
from src.utils.utils import load_best_model_checkpoint

model_factory_module = importlib.import_module(
    "src.model_factory.model_factory"
)


class TinyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(2, 2)


def test_model_factory_fails_when_configured_checkpoint_is_missing(
    monkeypatch,
    tmp_path,
) -> None:
    module = SimpleNamespace(Model=lambda args, metadata: TinyModel())
    monkeypatch.setattr(
        model_factory_module.importlib,
        "import_module",
        lambda module_path: module,
    )
    args = SimpleNamespace(
        type="Dummy",
        name="Tiny",
        num_classes=2,
        weights_path=str(tmp_path / "missing.ckpt"),
    )

    with pytest.raises(
        FileNotFoundError,
        match="Configured checkpoint does not exist",
    ):
        model_factory_module.model_factory(args, metadata=None)


def test_load_ckpt_strict_loads_lightning_state_dict(tmp_path) -> None:
    source = TinyModel()
    target = TinyModel()
    for parameter in target.parameters():
        torch.nn.init.zeros_(parameter)

    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": source.state_dict()}, checkpoint)

    model_factory_module.load_ckpt(target, str(checkpoint))

    for expected, actual in zip(source.parameters(), target.parameters()):
        assert torch.equal(expected, actual)


def test_load_ckpt_non_strict_rejects_zero_matches(tmp_path) -> None:
    checkpoint = tmp_path / "wrong.ckpt"
    torch.save({"unrelated.weight": torch.ones(1)}, checkpoint)

    with pytest.raises(RuntimeError, match="matched zero model parameters"):
        model_factory_module.load_ckpt(
            TinyModel(),
            str(checkpoint),
            strict=False,
        )


def test_best_checkpoint_must_exist_before_evaluation(tmp_path) -> None:
    callback = ModelCheckpoint()
    callback.best_model_path = str(tmp_path / "missing-best.ckpt")
    trainer = SimpleNamespace(callbacks=[callback])

    with pytest.raises(
        FileNotFoundError,
        match="Best checkpoint does not exist",
    ):
        load_best_model_checkpoint(TinyModel(), trainer)


def test_best_checkpoint_path_cannot_be_empty() -> None:
    callback = ModelCheckpoint()
    callback.best_model_path = ""
    trainer = SimpleNamespace(callbacks=[callback])

    with pytest.raises(
        RuntimeError,
        match="did not produce a best checkpoint",
    ):
        load_best_model_checkpoint(TinyModel(), trainer)


def test_pretrained_loader_rejects_checkpoint_without_backbone_weights(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "no-backbone.ckpt"
    torch.save(
        {"state_dict": {"task_head.weight": torch.ones(1)}},
        checkpoint,
    )

    with pytest.raises(
        RuntimeError,
        match="no transferable 'network.' backbone",
    ):
        load_pretrained_weights(
            TinyModel(),
            str(checkpoint),
            strict=False,
        )
