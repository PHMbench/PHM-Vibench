from types import SimpleNamespace

import torch

from phmfactory.config import resolve_config
from src.model_factory import build_model, resolve_model_module
from src.task_factory import build_task


_METADATA = {
    1: {
        "Name": "Dummy_Data",
        "Dataset_id": 0,
        "Domain_id": 0,
        "Sample_Rate": 1000,
        "Label": 0,
    },
    2: {
        "Name": "Dummy_Data",
        "Dataset_id": 0,
        "Domain_id": 1,
        "Sample_Rate": 2000,
        "Label": 1,
    },
}


def _namespace(mapping):
    return SimpleNamespace(**mapping)


def test_existing_factory_supports_config_only_model_replacement_and_backward():
    resolved = resolve_config(
        "smoke",
        override_values=(
            "model.type=Baseline",
            "model.name=GlobalAverageLinear",
            "model.input_dim=2",
            "model.num_classes=2",
        ),
    )
    config = resolved.data
    args_environment = _namespace(config["environment"])
    args_data = _namespace(config["data"])
    args_model = _namespace(config["model"])
    args_task = _namespace(config["task"])
    args_trainer = _namespace(config["trainer"])

    assert resolve_model_module(args_model) == (
        "src.model_factory.Baseline.GlobalAverageLinear"
    )

    model = build_model(args_model, metadata=_METADATA)
    task = build_task(
        args_task=args_task,
        network=model,
        args_data=args_data,
        args_model=args_model,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata=_METADATA,
    )

    batch = {
        "x": torch.randn(2, 128, 2),
        "y": torch.tensor([0, 1]),
        "file_id": torch.tensor([1, 2]),
    }
    metrics = task._shared_step(batch, "train")
    loss = metrics["train_total_loss"]

    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()


def test_baseline_model_rejects_incompatible_input_without_padding_or_fallback():
    args = SimpleNamespace(input_dim=2, num_classes=2)
    model = build_model(args=SimpleNamespace(
        type="Baseline",
        name="GlobalAverageLinear",
        input_dim=2,
        num_classes=2,
    ), metadata=_METADATA)

    try:
        model(torch.randn(2, 128, 1))
    except ValueError as exc:
        assert "channel mismatch" in str(exc)
    else:
        raise AssertionError("channel mismatch must fail before classification")
