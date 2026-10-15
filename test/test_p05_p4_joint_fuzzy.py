"""Scientific behavior checks for the P4 model's PHMFactory relocation."""

from __future__ import annotations

from itertools import product

import numpy as np
import torch

from src.model_factory.X_model.P4JointFuzzy import Fuzzy


def _model() -> Fuzzy:
    x = np.array([[-1., -.4], [-.8, -.2], [.7, .3], [1., .5]])
    y = np.array([0, 0, 1, 1])
    return Fuzzy(x, y, rules=2, seed=42)


def test_prediction_is_the_reported_rule_mixture_and_checkpoint_roundtrips() -> None:
    model = _model()
    x = torch.tensor([[-.6, .1], [.2, .4]], dtype=torch.float32)
    w, q, _, _ = model.components(x)

    torch.testing.assert_close(model(x).exp(), w @ q)
    torch.testing.assert_close(w.sum(dim=1), torch.ones(len(x)))
    torch.testing.assert_close(q.sum(dim=1), torch.ones(2))
    assert set(model.state_dict()) == {'centers', 'log_scales', 'log_q'}

    restored = _model()
    restored.load_state_dict(model.state_dict())
    torch.testing.assert_close(restored(x), model(x), rtol=0, atol=0)


def test_surrogate_zero_radius_and_positive_margin_semantics_with_gradients() -> None:
    model = _model()
    x = torch.tensor([[0., .1], [.1, -.1], [-1., -.4], [1., .5]])
    y = torch.tensor([0, 1, 0, 1])
    radius = .2
    penalty = model.robust_penalty(x, y, radius)

    probability = model(x).exp()
    margin = probability[torch.arange(len(x)), y] - probability[
        torch.arange(len(x)), 1 - y
    ]
    torch.testing.assert_close(
        model.robust_penalty(x, y, radius=0.), (.05 - margin).clamp_min(0)
    )

    # A positive lower numerator certifies the class throughout the feature box.
    # T2 does NOT claim that a negative numerator divided by the upper firing
    # sum is a lower probability margin; do not assert that stronger property.
    positive_margin = penalty < .05
    assert positive_margin.any()
    for signs in product((-1., 1.), repeat=x.shape[1]):
        corner = x + radius * torch.tensor(signs)
        assert torch.equal(
            model(corner).argmax(dim=1)[positive_margin], y[positive_margin]
        )

    nn_loss = torch.nn.functional.nll_loss(model(x), y)
    loss = nn_loss + .1 * penalty.mean()
    loss.backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_rule_consequent_intervention_changes_prediction_without_another_head() -> None:
    model = _model()
    x = torch.tensor([[-1., -.4], [1., .5]], dtype=torch.float32)
    before = model(x).exp()
    with torch.no_grad():
        model.log_q.copy_(model.log_q.flip(1))
    torch.testing.assert_close(model(x).exp(), before.flip(1))
    assert torch.equal(model(x).argmax(dim=1), 1 - before.argmax(dim=1))
