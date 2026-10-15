"""Shared grouped representation operations, independent of any study or arm."""
import pytest
import torch
from torch import nn
from src.task_factory.task.GFS import physical_prior_core as ops


def test_group_weights_do_not_count_duplicate_windows_as_new_units():
    y = torch.tensor([0, 0, 0, 0, 1, 1])
    groups = torch.tensor([10, 10, 10, 11, 20, 20])
    w = ops.weights(y, groups)
    assert w.sum().item() == pytest.approx(1.)
    assert w[groups == 10].sum().item() == pytest.approx(.25)
    assert w[groups == 11].sum().item() == pytest.approx(.25)
    assert w[y == 1].sum().item() == pytest.approx(.5)
    z = torch.tensor([[1., 0.], [1., 0.], [1., 0.], [0., 1.], [-1., 0.], [-1., 0.]])
    expected = ops.unit(torch.tensor([[.5, .5], [-1., 0.]]))
    assert torch.allclose(ops.prototype(z, y, groups, [0, 1]), expected)


@pytest.mark.parametrize('x', [torch.zeros(2, 3), torch.tensor([[float('nan'), 1.]])])
def test_invalid_representation_is_not_silently_normalized(x):
    with pytest.raises(ValueError):
        ops.unit(x)


def test_joint_class_score_and_frozen_state():
    model = nn.Linear(2, 2)
    state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    ops.assert_frozen(state, model)
    z = torch.tensor([[1., 0.]])
    assert torch.equal(ops.scores(z, z, torch.tensor([[0., 1.]]), 2.), torch.tensor([[2., 0.]]))
    with pytest.raises(ValueError):
        ops.scores(z, z, z, 0.)
    with torch.no_grad():
        model.weight.add_(1.)
    with pytest.raises(AssertionError):
        ops.assert_frozen(state, model)


def test_crossfit_is_differentiable_without_an_experiment_runner():
    class Encoder(nn.Module):
        def forward(self, x, p):
            return x + p
    x = torch.tensor([[2., 1.], [1.5, 1.], [1., 2.], [1., 1.5]])
    y = torch.tensor([2, 2, 3, 3]); groups = torch.tensor([10, 10, 20, 20])
    views = torch.tensor([0, 1, 0, 1]); prompt = torch.zeros(2, requires_grad=True)
    ops.validate_groups(y, groups, views, [2, 3], 1)
    loss = ops.crossfit(Encoder(), x, prompt, y, groups, views,
                        torch.tensor([[-1., 0.]]), [0], [2, 3], 2.)
    loss.backward()
    assert torch.isfinite(loss) and prompt.grad is not None
    assert torch.isfinite(prompt.grad).all() and prompt.grad.abs().sum() > 0
    assert not hasattr(ops, 'adapt')
