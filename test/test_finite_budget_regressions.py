import numpy as np
import pytest
import torch
from phmfactory.finite_budget import exact_sparse_risk
from phmfactory.finite_budget_evaluation import response_errors


@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_response_moment(value):
    with pytest.raises(ValueError, match="second moment must be finite"):
        exact_sparse_risk(np.eye(2), np.eye(2), np.zeros(2), value, 1)


class Model(torch.nn.Module):
    def forward(self, x):
        score = x.flatten(1).sum(1)
        return torch.stack([score, -score], 1)


def test_partial_query_overlap():
    x = torch.ones(1, 1, 4)
    cal = torch.randn(1, 2, 1, 4)
    scoring = torch.randn(1, 3, 1, 4)
    scoring[0, 2] = cal[0, 0]
    with pytest.raises(ValueError, match="overlapping"):
        response_errors(Model().eval(), x, [0], ["identity"], ["gradient"], [1],
            baseline=torch.zeros_like(x), evaluation_displacements=scoring,
            calibration_displacements=cal, forward_budget=1, backward_budget=1)
