from __future__ import annotations

from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from src.model_factory import model_factory as model_factory_api
from src.model_factory.model_factory import load_ckpt
from src.task_factory import task_factory as task_factory_api
from src.trainer_factory import trainer_factory as trainer_factory_api


def test_task_factory_preserves_import_failure_as_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = __import__("src.task_factory.task_factory", fromlist=["task_factory"])
    monkeypatch.setattr(module.TASK_REGISTRY, "get", lambda _key: (_ for _ in ()).throw(KeyError()))
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda _name: (_ for _ in ()).throw(ImportError("broken task module")),
    )

    with pytest.raises(RuntimeError, match="Failed to resolve task") as error:
        task_factory_api(
            Namespace(type="classification", name="missing"),
            torch.nn.Identity(),
            Namespace(),
            Namespace(),
            Namespace(),
            Namespace(),
            metadata=None,
        )

    assert isinstance(error.value.__cause__, ImportError)


def test_trainer_factory_preserves_constructor_failure_as_cause(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = __import__("src.trainer_factory.trainer_factory", fromlist=["trainer_factory"])

    def broken_trainer(**_kwargs: object) -> None:
        raise ValueError("invalid trainer configuration")

    monkeypatch.setattr(module.TRAINER_REGISTRY, "get", lambda _key: broken_trainer)

    with pytest.raises(RuntimeError, match="Failed to create trainer") as error:
        trainer_factory_api(Namespace(), Namespace(name="broken"), Namespace(), "/tmp")

    assert isinstance(error.value.__cause__, ValueError)


def test_model_factory_does_not_ignore_missing_explicit_checkpoint(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = __import__("src.model_factory.model_factory", fromlist=["model_factory"])
    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda _name: SimpleNamespace(Model=lambda _args, _metadata: torch.nn.Linear(2, 1)),
    )
    args = Namespace(
        type="CNN",
        name="Example",
        num_classes=1,
        weights_path=str(tmp_path / "missing.ckpt"),
    )

    with pytest.raises(RuntimeError, match="Failed to create model") as error:
        model_factory_api(args, metadata=None)

    assert isinstance(error.value.__cause__, FileNotFoundError)


@pytest.mark.parametrize(
    "state_dict",
    (
        {"weight": torch.zeros(1, 2)},
        {"weight": torch.zeros(1, 3), "bias": torch.zeros(1)},
        {
            "weight": torch.zeros(1, 2),
            "bias": torch.zeros(1),
            "unexpected": torch.zeros(1),
        },
    ),
    ids=("missing-key", "shape-mismatch", "unexpected-key"),
)
def test_load_ckpt_rejects_incomplete_or_incompatible_state(
    tmp_path, state_dict: dict[str, torch.Tensor]
) -> None:
    checkpoint = tmp_path / "model.ckpt"
    torch.save({"state_dict": state_dict}, checkpoint)

    with pytest.raises(RuntimeError):
        load_ckpt(torch.nn.Linear(2, 1), checkpoint)
