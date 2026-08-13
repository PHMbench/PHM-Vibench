from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch

model_factory_module = importlib.import_module("src.model_factory.model_factory")


def _model() -> torch.nn.Module:
    return torch.nn.Sequential(torch.nn.Linear(3, 2))


@pytest.mark.parametrize("wrapped", [False, True])
def test_load_ckpt_accepts_only_a_complete_state_dict(tmp_path, wrapped: bool) -> None:
    source = _model()
    target = _model()
    state = source.state_dict()
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": state} if wrapped else state, checkpoint)

    model_factory_module.load_ckpt(target, checkpoint)

    for source_value, target_value in zip(
        source.state_dict().values(), target.state_dict().values()
    ):
        assert torch.equal(source_value, target_value)


def test_load_ckpt_rejects_missing_extra_and_wrong_shape_keys(tmp_path) -> None:
    state = _model().state_dict()
    invalid_states = {
        "missing": {key: value for key, value in state.items() if key != "0.bias"},
        "extra": {**state, "unexpected": torch.ones(1)},
        "shape": {**state, "0.weight": torch.ones(1, 1)},
    }

    for name, invalid_state in invalid_states.items():
        checkpoint = tmp_path / f"{name}.ckpt"
        torch.save(invalid_state, checkpoint)
        with pytest.raises(RuntimeError):
            model_factory_module.load_ckpt(_model(), checkpoint)


def test_model_factory_does_not_ignore_a_requested_missing_checkpoint(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Model(torch.nn.Linear):
        def __init__(self, args_model, metadata) -> None:
            super().__init__(3, 2)

    monkeypatch.setattr(
        model_factory_module.importlib,
        "import_module",
        lambda name: SimpleNamespace(Model=Model),
    )
    args_model = SimpleNamespace(
        type="X_model",
        name="Example",
        num_classes=2,
        weights_path=str(tmp_path / "missing.ckpt"),
    )

    with pytest.raises(FileNotFoundError, match="does not exist"):
        model_factory_module.model_factory(args_model, metadata={})
