"""Scientific-semantic checks for the feature-adapted selective baseline."""
import math
from pathlib import Path
import sys

import pytest
import torch

from src.task_factory.task.classification.selective_diagnosis.selective import SelectiveMLP


def test_constant_selector_preserves_conditional_ce_and_uses_soft_coverage():
    model = SelectiveMLP(2, 2, 3, aux_weight=.25).double()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.prediction_head.bias.copy_(torch.tensor([2., 0.]))
        model.selection_head[2].bias.fill_(math.log(.25 / .75))
    x = torch.zeros(2, 2, dtype=torch.double)
    y = torch.tensor([0, 1])
    weights = torch.tensor([1., 3.], dtype=torch.double)
    conditional_ce = .25 * math.log1p(math.exp(-2)) + .75 * math.log1p(math.exp(2))
    expected = .75 * (conditional_ce + 32 * (.9 - .25) ** 2) + .25 * math.log(2)
    torch.testing.assert_close(model.objective(x, y, weights), torch.tensor(expected, dtype=torch.double))
    torch.testing.assert_close(model.objective(x, y, weights * 100),
                               model.objective(x, y, weights))
    with torch.no_grad():
        model.selection_head[2].bias.fill_(math.log(.75 / .25))
    expected = .75 * (conditional_ce + 32 * (.9 - .75) ** 2) + .25 * math.log(2)
    torch.testing.assert_close(model.objective(x, y, weights), torch.tensor(expected, dtype=torch.double))


def test_unit_replication_preserves_objective_and_gradients():
    torch.manual_seed(8)
    model = SelectiveMLP(2, 2, 4).double()
    x = torch.tensor([[1., 0.], [0., 1.], [.5, .5]], dtype=torch.double)
    y = torch.tensor([0, 1, 0])
    # First row belongs to one unit; the final two belong to another unit.
    weights = torch.tensor([.5, .25, .25], dtype=torch.double)
    loss = model.objective(x, y, weights)
    gradients = torch.autograd.grad(loss, tuple(model.parameters()))
    repetitions = torch.tensor([1, 7, 7])
    repeated_loss = model.objective(
        x.repeat_interleave(repetitions, dim=0), y.repeat_interleave(repetitions),
        (weights / repetitions).repeat_interleave(repetitions))
    repeated_gradients = torch.autograd.grad(repeated_loss, tuple(model.parameters()))
    torch.testing.assert_close(repeated_loss, loss)
    for actual, expected in zip(repeated_gradients, gradients):
        torch.testing.assert_close(actual, expected)


def test_prediction_selection_and_auxiliary_heads_receive_gradients():
    torch.manual_seed(13)
    model = SelectiveMLP(3, 3, 8)
    x = torch.randn(12, 3)
    y = torch.arange(12) % 3
    weights = torch.ones(12)
    logits, selection, auxiliary = model.forward_components(x)
    assert logits.shape == auxiliary.shape == (12, 3)
    assert selection.shape == (12,) and ((selection > 0) & (selection < 1)).all()
    torch.testing.assert_close(model(x), logits)
    model.objective(x, y, weights).backward()
    for module in (model.body, model.prediction_head, model.selection_head, model.auxiliary_head):
        gradients = [parameter.grad for parameter in module.parameters()]
        assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
        assert sum(gradient.abs().sum() for gradient in gradients) > 0


def test_selector_is_a_distinct_function_from_maximum_class_probability():
    model = SelectiveMLP(2, 2, 2)
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.body[0].weight.copy_(torch.eye(2))
        model.selection_head[0].weight.copy_(torch.eye(2))
        model.selection_head[2].weight.copy_(torch.tensor([[1., -1.]]))
    logits, selection, _ = model.forward_components(torch.tensor([[2., 0.], [0., 2.]]))
    torch.testing.assert_close(logits.softmax(1).max(1).values, torch.full((2,), .5))
    assert selection[0] > .5 > selection[1]


@pytest.mark.parametrize('weights', [[0., 0.], [1., -1.], [1., float('nan')]])
def test_invalid_empirical_weights_fail(weights):
    model = SelectiveMLP(2, 2, 3)
    with pytest.raises(ValueError, match='weights'):
        model.objective(torch.zeros(2, 2), torch.tensor([0, 1]), torch.tensor(weights))


def test_zero_soft_acceptance_is_not_zero_selective_loss():
    model = SelectiveMLP(2, 2, 3)
    with torch.no_grad():
        model.selection_head[2].weight.zero_()
        model.selection_head[2].bias.fill_(-1000.)
    with pytest.raises(FloatingPointError, match='zero or nonfinite'):
        model.objective(torch.zeros(2, 2), torch.tensor([0, 1]), torch.ones(2))
