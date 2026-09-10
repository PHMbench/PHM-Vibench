import pytest
import torch
from phmfactory.finite_budget_evaluation import response_errors


class Model(torch.nn.Module):
    def forward(self, x):
        s = x.flatten(1).sum(1)
        return torch.stack([s, -s], 1)


def test_held_out_response_evaluation():
    x = torch.ones(2, 1, 4)
    generator = torch.Generator().manual_seed(7)
    vc = torch.randn(2, 8, 1, 4, generator=generator) * 0.01
    ve = torch.randn(2, 9, 1, 4, generator=generator) * 0.01
    rows = response_errors(Model().eval(), x, [0, 1], ["identity", "dct"],
        ["gradient", "path_gradient", "secant", "surrogate"], [1, 4],
        baseline=torch.zeros_like(x), evaluation_displacements=ve,
        calibration_displacements=vc, forward_budget=16, backward_budget=8, steps=8)
    assert len(rows) == 32
    assert max(row["mse"] for row in rows if row["k"] == 4) < 1e-9
    with pytest.raises(ValueError, match="must not be reused"):
        response_errors(Model().eval(), x, [0, 1], ["identity"], ["gradient"], [1],
            baseline=torch.zeros_like(x), evaluation_displacements=vc,
            calibration_displacements=vc, forward_budget=1, backward_budget=1)
