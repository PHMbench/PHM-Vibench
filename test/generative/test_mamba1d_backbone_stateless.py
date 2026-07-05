from types import SimpleNamespace

import torch

from src.model_factory.generative_model.mamba1d_backbone import Model


def test_mamba1d_backbone_stateless_forward_contract():
    args = SimpleNamespace(
        in_channels=2,
        hidden_dim=8,
        condition_dim=4,
        num_fault_classes=2,
        num_domains=2,
    )
    model = Model(args, metadata=None)
    x = torch.randn(2, 2, 16)
    t = torch.rand(2)
    condition = {
        "fault_label": torch.tensor([0, 1]),
        "domain_id": torch.tensor([0, 1]),
    }

    out1 = model(x, t, condition)
    out2 = model(x, t, condition)

    assert out1.shape == x.shape
    assert out2.shape == x.shape
    assert model.stateless is True

