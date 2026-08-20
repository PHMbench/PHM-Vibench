from __future__ import annotations

from argparse import Namespace
import importlib

import pytest
import torch
import torch.nn as nn


def _default_task_module():
    return importlib.import_module("src.task_factory.Default_task")


def _default_trainer_module():
    return importlib.import_module("src.trainer_factory.Default_trainer")


def _device_module():
    return importlib.import_module("phmfactory.device")


def _model_factory_module():
    return importlib.import_module("src.model_factory.model_factory")


class TrackingNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(2, 2)
        self.cuda_calls = 0

    def cuda(self, *args, **kwargs):
        del args, kwargs
        self.cuda_calls += 1
        raise AssertionError("Task construction must not call network.cuda()")

    def forward(self, x, file_id, task_id):
        del file_id, task_id
        return self.linear(x)


def test_default_task_preserves_model_device_and_trainer_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _default_task_module()
    monkeypatch.setattr(module, "get_loss_fn", lambda name: nn.CrossEntropyLoss())
    monkeypatch.setattr(module, "get_metrics", lambda metrics, metadata: {})
    monkeypatch.setattr(
        module.torch.cuda,
        "is_available",
        lambda: pytest.fail("Task construction must not inspect CUDA availability"),
    )

    network = TrackingNetwork()
    args_trainer = Namespace(device="cuda", gpus=1)
    trainer_config_before = vars(args_trainer).copy()

    task = module.Default_task(
        network=network,
        args_data=Namespace(),
        args_model=Namespace(),
        args_task=Namespace(
            name="classification",
            loss="CE",
            metrics=[],
            optimizer="adam",
            lr=1e-3,
        ),
        args_trainer=args_trainer,
        args_environment=Namespace(),
        metadata={0: {"Name": "dummy", "Label": 0}},
    )

    assert task.network is network
    assert network.cuda_calls == 0
    assert next(task.network.parameters()).device.type == "cpu"
    assert vars(args_trainer) == trainer_config_before


@pytest.mark.parametrize(
    ("device", "devices", "expected"),
    [
        ("cpu", 1, ("cpu", 1)),
        ("cpu", 3, ("cpu", 3)),
    ],
)
def test_device_resolver_honors_explicit_cpu(
    device: str,
    devices: int,
    expected: tuple[str, int],
) -> None:
    module = _device_module()
    assert module.resolve_device_request(
        Namespace(device=device, gpus=devices)
    ) == expected


def test_cpu_resolution_does_not_import_torch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _device_module()
    monkeypatch.setattr(
        module,
        "_load_torch",
        lambda: pytest.fail("CPU resolution must not import torch"),
    )

    assert module.resolve_device_request(
        Namespace(device="cpu", devices=1)
    ) == ("cpu", 1)


def test_device_resolver_reports_actual_auto_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _device_module()
    monkeypatch.setattr(
        module,
        "_available_auto_accelerator",
        lambda: ("cpu", None),
    )

    assert module.resolve_device_request(
        Namespace(device="auto", devices=2)
    ) == ("cpu", 2)


def test_device_resolver_reports_actual_auto_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _device_module()
    monkeypatch.setattr(
        module,
        "_available_auto_accelerator",
        lambda: ("gpu", 2),
    )

    assert module.resolve_device_request(
        Namespace(device="auto", devices=2)
    ) == ("gpu", 2)


def test_device_resolver_rejects_excess_auto_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _device_module()
    monkeypatch.setattr(
        module,
        "_available_auto_accelerator",
        lambda: ("gpu", 1),
    )

    with pytest.raises(RuntimeError, match="accelerator=gpu, requested=2, available=1"):
        module.resolve_device_request(Namespace(device="auto", devices=2))


def test_device_resolver_rejects_unavailable_cuda_without_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _device_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.cuda,
        "device_count",
        lambda: pytest.fail("device_count is irrelevant when CUDA is unavailable"),
    )

    with pytest.raises(RuntimeError, match="no CPU fallback"):
        module.resolve_device_request(Namespace(device="cuda", gpus=1))


def test_device_resolver_accepts_available_cuda(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _device_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)

    assert module.resolve_device_request(
        Namespace(device="cuda", gpus=2)
    ) == ("gpu", 2)


def test_device_resolver_rejects_excess_cuda_devices(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _device_module()
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)

    with pytest.raises(RuntimeError, match="requested=2, available=1"):
        module.resolve_device_request(Namespace(device="cuda", gpus=2))


@pytest.mark.parametrize(
    "args_trainer",
    [
        Namespace(gpus=1),
        Namespace(device="gpu", gpus=1),
        Namespace(device="cuda", gpus=0),
        Namespace(device="cpu", devices=True),
    ],
)
def test_device_resolver_rejects_ambiguous_or_invalid_requests(
    args_trainer: Namespace,
) -> None:
    module = _device_module()
    with pytest.raises(ValueError):
        module.resolve_device_request(args_trainer)


def test_default_trainer_passes_resolved_cpu_request_to_lightning(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    module = _default_trainer_module()
    observed: dict[str, object] = {}

    monkeypatch.setattr(module, "call_backs", lambda args, path: [])
    monkeypatch.setattr(module, "CSVLogger", lambda *args, **kwargs: object())
    monkeypatch.setattr(
        module.pl,
        "Trainer",
        lambda **kwargs: observed.update(kwargs) or kwargs,
    )

    result = module.trainer(
        args_e=Namespace(wandb=False, swanlab=False),
        args_t=Namespace(
            device="cpu",
            gpus=1,
            num_epochs=1,
            pruning=0.0,
            monitor="val_loss",
            log_every_n_steps=1,
        ),
        args_d=Namespace(),
        path=str(tmp_path),
    )

    assert result["accelerator"] == "cpu"
    assert result["devices"] == 1
    assert result["strategy"] == "auto"
    assert observed["accelerator"] == "cpu"


def test_default_trainer_and_preflight_share_device_function() -> None:
    trainer_module = _default_trainer_module()
    preflight_module = importlib.import_module("phmfactory.commands.preflight")
    device_module = _device_module()

    assert trainer_module.resolve_device_request is device_module.resolve_device_request
    assert preflight_module.resolve_device_request is device_module.resolve_device_request


def test_model_factory_preserves_constructor_exception_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _model_factory_module()

    class ModelConstructorFailure(ValueError):
        pass

    class BrokenModel:
        def __init__(self, args_model, metadata):
            del args_model, metadata
            raise ModelConstructorFailure("invalid model dimensions")

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda path: Namespace(Model=BrokenModel),
    )
    args_model = Namespace(
        type="Broken",
        name="Model",
        num_classes=2,
        weights_path=None,
    )

    with pytest.raises(
        ModelConstructorFailure,
        match="invalid model dimensions",
    ):
        module.model_factory(args_model, metadata=None)


def test_model_factory_preserves_checkpoint_exception_type(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _model_factory_module()

    class CheckpointLoadFailure(OSError):
        pass

    class ValidModel:
        def __init__(self, args_model, metadata):
            del args_model, metadata

    def fail_checkpoint(model, path, *, strict):
        del model, path, strict
        raise CheckpointLoadFailure("checkpoint tensor layout is invalid")

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda path: Namespace(Model=ValidModel),
    )
    monkeypatch.setattr(module, "load_ckpt", fail_checkpoint)
    args_model = Namespace(
        type="Valid",
        name="Model",
        num_classes=2,
        weights_path="requested.ckpt",
        weights_strict=True,
    )

    with pytest.raises(
        CheckpointLoadFailure,
        match="checkpoint tensor layout is invalid",
    ):
        module.model_factory(args_model, metadata=None)


@pytest.mark.parametrize("strict", [True, False])
def test_model_factory_passes_explicit_checkpoint_strictness(
    monkeypatch: pytest.MonkeyPatch,
    strict: bool,
) -> None:
    module = _model_factory_module()
    observed: list[bool] = []

    class ValidModel:
        def __init__(self, args_model, metadata):
            del args_model, metadata

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda path: Namespace(Model=ValidModel),
    )
    monkeypatch.setattr(
        module,
        "load_ckpt",
        lambda model, path, *, strict: observed.append(strict),
    )
    args_model = Namespace(
        type="Valid",
        name="Model",
        num_classes=2,
        weights_path="requested.ckpt",
        weights_strict=strict,
    )

    module.model_factory(args_model, metadata=None)

    assert observed == [strict]


def test_model_factory_rejects_non_boolean_checkpoint_strictness(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = _model_factory_module()

    class ValidModel:
        def __init__(self, args_model, metadata):
            del args_model, metadata

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda path: Namespace(Model=ValidModel),
    )
    monkeypatch.setattr(
        module,
        "load_ckpt",
        lambda *args, **kwargs: pytest.fail(
            "checkpoint loading must not start for invalid weights_strict"
        ),
    )
    args_model = Namespace(
        type="Valid",
        name="Model",
        num_classes=2,
        weights_path="requested.ckpt",
        weights_strict="false",
    )

    with pytest.raises(
        TypeError,
        match="model.weights_strict must be a boolean",
    ):
        module.model_factory(args_model, metadata=None)
