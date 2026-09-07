import numpy as np
import pytest
import torch
from phmfactory.finite_budget import (
    orthogonal_basis, truncate, exact_sparse_risk, sparse_response_fit, coefficients,
)

@pytest.mark.parametrize("name", ["identity", "dct", "haar_packet:1", "haar_packet:3"])
def test_exact_coordinates(name):
    a = orthogonal_basis(8, name)
    np.testing.assert_allclose(a.T @ a, np.eye(8), atol=1e-12)

def test_haar_help_and_harm():
    a = orthogonal_basis(2, "haar_packet:1")
    for w in [np.array([1., 1.]) / np.sqrt(2), np.array([1., 0.])]:
        risk, _ = exact_sparse_risk(a, np.eye(2), w, w @ w, 1)
        expected = np.sum((a @ w - truncate(a @ w, 1)) ** 2)
        assert risk == pytest.approx(expected, abs=1e-12)

def test_noncentred_second_moment_and_full_budget():
    a = orthogonal_basis(2, "dct")
    m = np.array([[5., 2.], [2., 2.]])
    w = np.array([1., -2.])
    risk, coeff = exact_sparse_risk(a, m, m @ w, w @ m @ w, 2)
    assert risk < 1e-10
    np.testing.assert_allclose(a.T @ coeff, w, atol=1e-10)

def test_sparse_calibration_fit():
    v = np.eye(4)
    coeff = sparse_response_fit(v, np.array([0., 2., 0., -1.]), 2)
    np.testing.assert_allclose(coeff, [0., 2., 0., -1.])

class Model(torch.nn.Module):
    def forward(self, x):
        s = x.flatten(1).sum(1)
        return torch.stack([s, -s], 1)

@pytest.mark.parametrize("method", ["gradient", "path_gradient", "secant"])
def test_fixed_target_and_sign(method):
    x = torch.ones(1, 4, dtype=torch.float64)
    a = orthogonal_basis(4, "dct")
    raw, counts = coefficients(Model().eval(), x, 1, a, method=method,
                               baseline=torch.zeros_like(x), steps=8)
    np.testing.assert_allclose(raw @ a, -np.ones((1, 4)), atol=1e-10)
    assert counts["forward"] > 0

def test_reject_unknown_explainer():
    with pytest.raises(ValueError, match="no response-decoder"):
        coefficients(Model().eval(), torch.ones(1, 4), 0, np.eye(4), method="shap")
