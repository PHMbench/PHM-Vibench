from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.model_factory.X_model.TSPN_UXFD import Model as TSPNUXFD
from src.model_factory.X_model.UXFD.fuzzy import FuzzyConfig, FuzzyReasoner


def _ns(**kwargs):  # type: ignore[no-untyped-def]
    return SimpleNamespace(**kwargs)


def _model_args() -> SimpleNamespace:
    return _ns(
        device="cpu",
        num_classes=2,
        in_channels=2,
        out_channels=4,
        scale=1,
        skip_connection=True,
        internal_instance_normalization=False,
        signal_processing_configs={"layer1": ["I"]},
        feature_extractor_configs=["Mean", "Std"],
        in_dim=128,
        out_dim=128,
        uxfd=_ns(
            enable_sp2d=False,
            fuzzy=_ns(
                enable=True,
                num_fuzzy_features=4,
                num_membership_functions=3,
                num_rules=4,
                logit_scale=1.0,
            ),
            neural_residual=_ns(enable=False),
            anfis=_ns(enable=False),
            operator_attention=_ns(enable=False),
            logic=_ns(enable=False),
        ),
    )


def _reasoner() -> FuzzyReasoner:
    torch.manual_seed(20260801)
    return FuzzyReasoner(
        dim_in=3,
        num_classes=2,
        cfg=FuzzyConfig(
            num_fuzzy_features=3,
            num_membership_functions=2,
            num_rules=4,
        ),
    )


def test_f0_consumes_logged_minimum_activations_and_max_class_supports() -> None:
    reasoner = _reasoner()
    features = torch.tensor(
        [[0.1, -0.3, 0.7], [1.0, -1.0, 0.0]],
        dtype=torch.float32,
    )
    mapping = torch.tensor([0, 1, 0, 1])

    decision = reasoner.forward_f0(
        features,
        rule_to_class=mapping,
        conflict_threshold=1.0,
    )

    expected_activations = decision.antecedent_memberships.amin(dim=-1)
    assert torch.equal(decision.rule_activations, expected_activations)
    for class_id in (0, 1):
        expected_support = expected_activations.masked_fill(
            decision.rule_to_class.ne(class_id),
            0.0,
        ).max(dim=1).values
        assert torch.equal(decision.class_supports[:, class_id], expected_support)
    assert torch.equal(decision.top_support, decision.class_supports.max(dim=1).values)
    assert decision.accepted.all()
    assert (decision.issued_class >= 0).all()


def test_f0_all_rule_removal_abstains_without_fallback() -> None:
    reasoner = _reasoner()
    decision = reasoner.forward_f0(
        torch.zeros(2, 3),
        rule_to_class=torch.tensor([0, 1, 0, 1]),
        conflict_threshold=1.0,
        rule_mask=torch.zeros(4, dtype=torch.bool),
    )

    assert torch.equal(decision.rule_activations, torch.zeros_like(decision.rule_activations))
    assert torch.equal(decision.class_supports, torch.zeros_like(decision.class_supports))
    assert torch.equal(decision.conflict, torch.ones_like(decision.conflict))
    assert not decision.accepted.any()
    assert torch.equal(decision.issued_class, torch.full((2,), -1))


def test_f0_does_not_consume_neural_or_learned_vector_consequents() -> None:
    torch.manual_seed(20260801)
    model = TSPNUXFD(_model_args()).eval()
    x = torch.randn(3, 128, 2)
    mapping = torch.tensor([0, 1, 0, 1])

    with torch.no_grad():
        # The legacy TSPN backbone normalizes a weight tensor in-place on each
        # forward. Freeze its output explicitly so this intervention changes
        # only the neural and learned-vector consequent heads.
        fixed_features = model._forward_features(x)
        model._forward_features = lambda _x: fixed_features  # type: ignore[method-assign]
        logits_before = model._forward_non_fuzzy_logits(fixed_features)
        decision_before = model.forward_f0(
            x,
            rule_to_class=mapping,
            conflict_threshold=1.0,
        )
        for parameter in model.clf.parameters():
            parameter.add_(5.0 * torch.randn_like(parameter))
        assert model._uxfd_fuzzy is not None
        model._uxfd_fuzzy.rule_consequents.add_(
            5.0 * torch.randn_like(model._uxfd_fuzzy.rule_consequents)
        )
        logits_after = model._forward_non_fuzzy_logits(fixed_features)
        decision_after = model.forward_f0(
            x,
            rule_to_class=mapping,
            conflict_threshold=1.0,
        )

    assert not torch.allclose(logits_before, logits_after)
    assert torch.equal(decision_before.rule_activations, decision_after.rule_activations)
    assert torch.equal(decision_before.class_supports, decision_after.class_supports)
    assert torch.equal(decision_before.conflict, decision_after.conflict)
    assert torch.equal(decision_before.issued_class, decision_after.issued_class)


def test_f0_consequent_and_threshold_directions_are_explicit() -> None:
    reasoner = _reasoner()
    features = torch.tensor([[0.2, -0.4, 0.6]], dtype=torch.float32)
    mapping = torch.tensor([0, 1, 0, 1])
    flipped = 1 - mapping

    original = reasoner.forward_f0(
        features,
        rule_to_class=mapping,
        conflict_threshold=1.0,
    )
    intervened = reasoner.forward_f0(
        features,
        rule_to_class=mapping,
        consequent_override=flipped,
        conflict_threshold=1.0,
    )
    strict = reasoner.forward_f0(
        features,
        rule_to_class=mapping,
        conflict_threshold=0.0,
    )

    assert torch.equal(intervened.class_supports, original.class_supports.flip(dims=(1,)))
    assert torch.all(strict.accepted <= original.accepted)
    assert original.accepted.all()


@pytest.mark.parametrize(
    "mapping",
    [
        torch.tensor([0, 1, 0]),
        torch.tensor([0, 1, 0, 2]),
        torch.tensor([0.0, 1.0, 0.5, 1.0]),
    ],
)
def test_f0_rejects_missing_or_invalid_rule_mapping(mapping: torch.Tensor) -> None:
    with pytest.raises((TypeError, ValueError), match="rule_to_class"):
        _reasoner().forward_f0(
            torch.zeros(1, 3),
            rule_to_class=mapping,
            conflict_threshold=0.5,
        )


def test_p05_backbone_forward_is_non_mutating_and_gradients_are_finite() -> None:
    torch.manual_seed(20260801)
    model = TSPNUXFD(_model_args()).train()
    x = torch.randn(4, 128, 2)
    y = torch.tensor([0, 1, 0, 1])
    parameters_before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    logits = model(x)
    torch.nn.functional.cross_entropy(logits, y).backward()

    for name, parameter in model.named_parameters():
        assert torch.equal(parameter, parameters_before[name]), name
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
