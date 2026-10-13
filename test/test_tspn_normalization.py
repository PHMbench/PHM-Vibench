"""Regression tests for the native TSPN's population-standard-deviation normalization."""

import pytest
import torch

from src.model_factory.X_model.TSPN import CustomBatchNorm


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_custom_batch_norm_matches_positive_variance_formula(dtype: torch.dtype) -> None:
    x = torch.tensor([[1.0, -3.0], [2.0, 4.0], [5.0, 2.0]], dtype=dtype)
    layer = CustomBatchNorm(2).to(dtype=dtype)
    expected = (x - x.mean(dim=0)) / (x.var(dim=0, unbiased=False).sqrt() + layer.eps)

    torch.testing.assert_close(layer(x), expected)


@pytest.mark.parametrize("batch_size", [1, 3])
def test_custom_batch_norm_constant_coordinate_has_finite_backward(batch_size: int) -> None:
    x = torch.tensor([[2.0, 1.0], [2.0, 3.0], [2.0, 5.0]])[:batch_size].clone().requires_grad_()
    weights = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])[:batch_size]
    layer = CustomBatchNorm(2)

    output = layer(x)
    (output * weights).sum().backward()

    assert torch.isfinite(output).all()
    assert x.grad is not None
    assert torch.isfinite(x.grad).all()
    expected_constant_gradient = (weights[:, 0] - weights[:, 0].mean()) / layer.eps
    torch.testing.assert_close(x.grad[:, 0], expected_constant_gradient)


def test_custom_batch_norm_preserves_running_updates_and_evaluation_formula() -> None:
    layer = CustomBatchNorm(2)
    expected_mean = layer.running_mean.clone()
    expected_var = layer.running_var.clone()
    batches = (
        torch.tensor([[2.0, 1.0], [2.0, 3.0], [2.0, 5.0]]),
        torch.tensor([[1.0, 2.0], [3.0, 6.0]]),
    )
    for x in batches:
        expected_mean = (1 - layer.eps) * expected_mean + layer.eps * x.mean(dim=0)
        expected_var = (1 - layer.eps) * expected_var + layer.eps * x.var(dim=0, unbiased=False)
        layer(x)
        torch.testing.assert_close(layer.running_mean, expected_mean, rtol=0, atol=0)
        torch.testing.assert_close(layer.running_var, expected_var, rtol=0, atol=0)

    layer.eval()
    evaluation_batch = torch.tensor([[7.0, -1.0], [2.0, 8.0]])
    expected = (evaluation_batch - expected_mean) / (expected_var.sqrt() + layer.eps)
    torch.testing.assert_close(layer(evaluation_batch), expected, rtol=0, atol=0)
    torch.testing.assert_close(layer.running_mean, expected_mean, rtol=0, atol=0)
    torch.testing.assert_close(layer.running_var, expected_var, rtol=0, atol=0)
