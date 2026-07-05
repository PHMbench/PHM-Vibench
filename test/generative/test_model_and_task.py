from types import SimpleNamespace

import pytest
import torch

from src.model_factory.generative_model.phm_cfm_mlp1d import Model

pytest.importorskip("pytorch_lightning")

from src.task_factory.task.generative.conditional_flow_matching import (
    ConditionalFlowMatchingTask,
)


def _args():
    args_data = SimpleNamespace(normalization="standardization", window_size=128)
    args_model = SimpleNamespace(
        type="generative_model",
        name="phm_cfm_mlp1d",
        in_channels=2,
        hidden_dim=16,
        condition_dim=8,
        num_fault_classes=2,
        num_domains=2,
    )
    args_task = SimpleNamespace(
        type="generative",
        name="conditional_flow_matching",
        lr=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        t_eps=1e-3,
    )
    args_trainer = SimpleNamespace(gpus=1, device="cpu")
    args_environment = SimpleNamespace(seed=0, project="test", output_dir="tmp", iterations=1)
    metadata = {
        1: {"Label": 0, "Domain_id": 0, "Name": "Dummy_Data"},
        2: {"Label": 1, "Domain_id": 1, "Name": "Dummy_Data"},
    }
    return args_data, args_model, args_task, args_trainer, args_environment, metadata


def test_cfm_model_forward_requires_explicit_conditions():
    _, args_model, _, _, _, metadata = _args()
    model = Model(args_model, metadata)
    x_t = torch.randn(2, 2, 32)
    t = torch.rand(2)
    condition = {
        "fault_label": torch.tensor([0, 1]),
        "domain_id": torch.tensor([0, 1]),
    }

    assert model(x_t, t, condition).shape == x_t.shape

    with pytest.raises(ValueError, match="condition missing"):
        model(x_t, t, {"fault_label": torch.tensor([0, 1])})


def test_cfm_task_shared_step_extracts_conditions_from_metadata():
    args_data, args_model, args_task, args_trainer, args_environment, metadata = _args()
    model = Model(args_model, metadata)
    task = ConditionalFlowMatchingTask(
        model,
        args_data,
        args_model,
        args_task,
        args_trainer,
        args_environment,
        metadata,
    )
    task.log = lambda *args, **kwargs: None
    batch = {
        "x": torch.randn(2, 128, 2),
        "y": torch.tensor([0, 1]),
        "file_id": torch.tensor([1, 2]),
    }

    loss = task._shared_step(batch, "train")

    assert loss.ndim == 0
    assert torch.isfinite(loss)
