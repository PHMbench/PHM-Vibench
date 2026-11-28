from __future__ import annotations

from types import SimpleNamespace

import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.Pipeline_06_Generative_Modeling import _sample_conditions, _train_reference
from src.model_factory import build_model
from src.model_factory.generative_model.phm_cfm_mlp1d import Model
from src.task_factory import build_task
from src.task_factory.task.generative.conditional_flow_matching import (
    ConditionalFlowMatchingTask,
)


class _Metadata:
    def __init__(self) -> None:
        self.df = pd.DataFrame(
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
        )
        self.df.set_index("Id", inplace=True, drop=False)

    def __getitem__(self, key):
        return self.df.loc[key].to_dict()


def test_cfm_model_and_task_build_through_public_factories() -> None:
    metadata = _Metadata()
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


@pytest.mark.parametrize("policy", ["grid", "explicit"])
def test_sampling_conditions_preserve_requested_population(policy: str) -> None:
    config = {
        "condition_sampling_policy": policy,
        "condition_grid": {
            "fault_label": [0, 1, 2], "domain_id": [0, 1, 2],
            "samples_per_condition": 1,
        },
        "explicit_conditions": [
            {"fault_label": label, "domain_id": domain}
            for label in range(3) for domain in range(3)
        ],
    }
    with pytest.raises(ValueError, match="num_samples=54 conflicts.*count=9"):
        _sample_conditions(config, None, 54, "cpu")
    conditions = _sample_conditions(config, None, 9, "cpu")
    assert conditions["fault_label"].numel() == 9
    assert conditions["domain_id"].tolist() == [0, 1, 2] * 3


@pytest.mark.parametrize("split", ["val", "test"])
def test_holdout_reference_keeps_all_windows_beyond_32_batches(split: str) -> None:
    windows = torch.arange(54 * 8, dtype=torch.float32).reshape(54, 8, 1)
    loader = DataLoader([
        {"x": window, "y": index % 3, "domain_id": index // 18}
        for index, window in enumerate(windows)
    ], batch_size=1)
    requested = []

    def get_loader(name):
        requested.append(name)
        return loader

    factory = SimpleNamespace(get_metadata=lambda: None, get_dataloader=get_loader)
    real, labels, domains = _train_reference(
        factory, 1, split=split, max_batches=None, max_samples=None,
    )
    assert requested == [split]
    torch.testing.assert_close(real, windows.transpose(1, 2))
    assert labels.tolist() == list(range(3)) * 18
    assert domains.tolist() == [0] * 18 + [1] * 18 + [2] * 18
