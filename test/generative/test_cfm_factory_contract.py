from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from src.data_factory.data_utils import MetadataAccessor
from src.model_factory import build_model
from src.model_factory.generative_model.phm_cfm_mlp1d import Model
from src.task_factory import build_task
from src.task_factory.task.generative.conditional_flow_matching import (
    ConditionalFlowMatchingTask,
)


def _metadata() -> MetadataAccessor:
    return MetadataAccessor(
        pd.DataFrame(
            [
                {
                    "Id": 1,
                    "Dataset_id": 0,
                    "Label": 0,
                    "Domain_id": 0,
                },
                {
                    "Id": 2,
                    "Dataset_id": 0,
                    "Label": 1,
                    "Domain_id": 1,
                },
            ]
        ),
        key_column="Id",
    )


def test_cfm_model_and_task_build_through_public_factories() -> None:
    metadata = _metadata()
    args_model = SimpleNamespace(
        type="generative_model",
        name="phm_cfm_mlp1d",
        in_channels=2,
        hidden_dim=8,
        condition_dim=4,
        num_fault_classes=2,
        num_domains=2,
        weights_path=None,
    )
    args_task = SimpleNamespace(
        type="generative",
        name="conditional_flow_matching",
        target_system_id=None,
        lr=1e-4,
        weight_decay=1e-4,
        optimizer="adamw",
        t_eps=1e-3,
    )

    model = build_model(args_model, metadata=metadata)
    task = build_task(
        args_task=args_task,
        network=model,
        args_data=SimpleNamespace(normalization="standardization"),
        args_model=args_model,
        args_trainer=SimpleNamespace(),
        args_environment=SimpleNamespace(),
        metadata=metadata,
    )

    assert isinstance(model, Model)
    assert isinstance(task, ConditionalFlowMatchingTask)
    assert args_model.num_classes == 2
