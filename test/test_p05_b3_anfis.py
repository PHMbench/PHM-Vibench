from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from src.model_factory.X_model.TSPN_UXFD import Model as TSPNUXFD
from src.model_factory.X_model.UXFD.fuzzy import FuzzyConfig, FuzzyReasoner
from src.model_factory.X_model.UXFD.p05_b3_anfis import (
    P05_B3_PARAMETER_COUNT_BY_CLASSES,
    P05B3ANFISConfig,
    P05B3ANFISHead,
)


def _ns(**values):  # type: ignore[no-untyped-def]
    return SimpleNamespace(**values)


def _args(
    *,
    num_classes: int,
    enable_anfis: bool = True,
    enable_fuzzy: bool = False,
    enable_neural: bool = False,
    enable_logic: bool = False,
    out_channels: int = 4,
    anfis_overrides: dict | None = None,
) -> SimpleNamespace:
    anfis = {
        "enable": enable_anfis,
        "num_features": 8,
        "num_membership_functions": 3,
        "num_rules": 10,
        "antecedent_temperature": 1.0,
        "min_width": 1.0e-4,
        "firing_epsilon": 1.0e-12,
    }
    anfis.update(anfis_overrides or {})
    return _ns(
        device="cpu",
        num_classes=num_classes,
        in_channels=2,
        out_channels=out_channels,
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
                enable=enable_fuzzy,
                num_fuzzy_features=8,
                num_membership_functions=3,
                num_rules=10,
                logit_scale=0.5,
                antecedent_temperature=1.0,
                min_width=1.0e-4,
                firing_epsilon=1.0e-12,
            ),
            neural_residual=_ns(enable=enable_neural, hidden_dim=None),
            anfis=_ns(**anfis),
            operator_attention=_ns(enable=False),
            logic=_ns(enable=enable_logic),
        ),
    )


@pytest.mark.parametrize(
    ("num_classes", "expected_parameters"),
    [(4, 664), (2, 484)],
)
def test_b3_batch_one_reconstructs_without_non_fuzzy_or_unused_parameters(
    num_classes: int,
    expected_parameters: int,
) -> None:
    torch.manual_seed(23)
    model = TSPNUXFD(_args(num_classes=num_classes))
    head = model._uxfd_anfis

    assert isinstance(head, P05B3ANFISHead)
    assert isinstance(model.clf, nn.Identity)
    assert head.parameter_count == expected_parameters
    assert head.parameter_count == P05_B3_PARAMETER_COUNT_BY_CLASSES[num_classes]
    assert model._uxfd_fuzzy is None
    assert model._uxfd_neural_residual is None
    assert model._uxfd_logic is None
    assert not any(name.startswith("clf.") for name, _ in model.named_parameters())

    x = torch.randn(1, 128, 2)
    model.eval()
    with torch.no_grad():
        trace = model.forward_with_anfis_trace(x)
        model._forward_non_fuzzy_logits = lambda _features: pytest.fail(
            "B3 must not execute the non-fuzzy logits path"
        )
        direct_logits = model(x)

    assert trace.reduced_features.shape == (1, 8)
    assert trace.membership_values.shape == (1, 8, 3)
    assert trace.centers.shape == (8, 3)
    assert trace.widths.shape == (8, 3)
    assert trace.antecedent_probabilities.shape == (10, 8, 3)
    assert trace.antecedent_memberships.shape == (1, 10, 8)
    assert trace.normalized_rule_firing.shape == (1, 10)
    assert trace.consequent_coefficients.shape == (10, num_classes, 8)
    assert trace.consequent_bias.shape == (10, num_classes)
    assert trace.rule_outputs.shape == (1, 10, num_classes)
    assert trace.rule_contributions.shape == (1, 10, num_classes)
    assert trace.logits.shape == (1, num_classes)
    assert torch.all(trace.centers[:, 1:] > trace.centers[:, :-1])
    assert torch.allclose(
        trace.normalized_rule_firing.sum(dim=1),
        torch.ones(1),
        atol=1.0e-7,
        rtol=0.0,
    )
    assert torch.allclose(direct_logits, trace.logits, atol=0.0, rtol=0.0)
    assert torch.allclose(
        trace.reconstruct_logits(),
        trace.logits,
        atol=0.0,
        rtol=0.0,
    )
    assert torch.count_nonzero(trace.reconstruction_residual()) == 0
    manual_rule_outputs = torch.einsum(
        "bf,rkf->brk",
        trace.reduced_features,
        trace.consequent_coefficients,
    ) + trace.consequent_bias.unsqueeze(0)
    assert torch.allclose(
        trace.rule_outputs,
        manual_rule_outputs,
        atol=0.0,
        rtol=0.0,
    )
    assert not hasattr(trace, "risk_features")

    model.zero_grad(set_to_none=True)
    model.train()
    model(x).square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())

    with pytest.raises(RuntimeError, match="fuzzy.enable=true"):
        model.forward_with_fuzzy_trace(x)


@pytest.mark.parametrize("num_classes", [2, 4])
def test_b3_uses_the_same_gaussian_antecedents_as_p05_m(num_classes: int) -> None:
    torch.manual_seed(31)
    fuzzy = FuzzyReasoner(
        dim_in=8,
        num_classes=num_classes,
        cfg=FuzzyConfig(
            num_fuzzy_features=8,
            num_membership_functions=3,
            num_rules=10,
            logit_scale=0.5,
            antecedent_temperature=1.0,
            min_width=1.0e-4,
            firing_epsilon=1.0e-12,
        ),
    )
    anfis = P05B3ANFISHead(input_dim=8, num_classes=num_classes)
    features = torch.randn(3, 8)

    fuzzy.eval()
    anfis.eval()
    with torch.no_grad():
        fuzzy_trace = fuzzy.forward_with_trace(features)
        anfis_trace = anfis.forward_with_trace(features)

    for fuzzy_value, anfis_value in (
        (fuzzy_trace.reduced_features, anfis_trace.reduced_features),
        (fuzzy_trace.centers, anfis_trace.centers),
        (fuzzy_trace.widths, anfis_trace.widths),
        (fuzzy_trace.membership_values, anfis_trace.membership_values),
        (
            fuzzy_trace.antecedent_probabilities,
            anfis_trace.antecedent_probabilities,
        ),
        (fuzzy_trace.rule_firing, anfis_trace.rule_firing),
        (
            fuzzy_trace.normalized_rule_firing,
            anfis_trace.normalized_rule_firing,
        ),
    ):
        assert torch.allclose(fuzzy_value, anfis_value, atol=0.0, rtol=0.0)


@pytest.mark.parametrize(
    ("enable_fuzzy", "enable_neural"),
    [(True, False), (False, True)],
)
def test_b3_is_mutually_exclusive_with_other_heads(
    enable_fuzzy: bool,
    enable_neural: bool,
) -> None:
    with pytest.raises(ValueError, match="mutually exclusive"):
        TSPNUXFD(
            _args(
                num_classes=4,
                enable_fuzzy=enable_fuzzy,
                enable_neural=enable_neural,
            )
        )


def test_b3_rejects_logic_residual_coupling() -> None:
    with pytest.raises(ValueError, match="forbids logic"):
        TSPNUXFD(_args(num_classes=4, enable_logic=True))


def test_b3_rejects_non_eight_feature_backbone() -> None:
    with pytest.raises(ValueError, match="same eight-feature input"):
        TSPNUXFD(_args(num_classes=4, out_channels=3))


def test_b3_rejects_frozen_antecedent_drift() -> None:
    with pytest.raises(ValueError, match="requires num_rules=10"):
        P05B3ANFISHead(
            input_dim=8,
            num_classes=4,
            cfg=P05B3ANFISConfig(num_rules=9),
        )
