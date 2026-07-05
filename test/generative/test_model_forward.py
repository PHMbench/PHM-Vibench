from types import SimpleNamespace

import pytest
import torch

from src.model_factory.generative_model.phm_cfm_mlp1d import Model


def test_cfm_model_forward_requires_explicit_conditions_without_lightning():
    args_model = SimpleNamespace(
        in_channels=2,
        hidden_dim=16,
        condition_dim=8,
        num_fault_classes=2,
        num_domains=2,
    )
    model = Model(args_model, metadata=None)
    x_t = torch.randn(2, 2, 32)
    t = torch.rand(2)
    condition = {
        "fault_label": torch.tensor([0, 1]),
        "domain_id": torch.tensor([0, 1]),
    }

    assert model(x_t, t, condition).shape == x_t.shape

    with pytest.raises(ValueError, match="condition missing"):
        model(x_t, t, {"fault_label": torch.tensor([0, 1])})

