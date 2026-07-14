from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.model_factory.generative_model.phm_cfm_mlp1d import Model


ARGS = SimpleNamespace(
    in_channels=2,
    hidden_dim=8,
    condition_dim=4,
    num_fault_classes=2,
    num_domains=2,
)
CONDITION = {
    "fault_label": torch.tensor([0, 1]),
    "domain_id": torch.tensor([0, 1]),
}


def test_cfm_model_preserves_float64_when_model_is_float64() -> None:
    model = Model(ARGS, metadata=None).double()
    x_t = torch.randn(2, 2, 16, dtype=torch.float64)

    output = model(
        x_t,
        torch.tensor([0.25, 0.75], dtype=torch.float64),
        CONDITION,
    )

    assert output.dtype == torch.float64
    assert output.shape == x_t.shape


def test_cfm_model_rejects_silent_dtype_conversion() -> None:
    model = Model(ARGS, metadata=None)
    x_t = torch.randn(2, 2, 16, dtype=torch.float64)

    with pytest.raises(ValueError, match="dtype mismatch"):
        model(
            x_t,
            torch.tensor([0.25, 0.75], dtype=torch.float64),
            CONDITION,
        )
