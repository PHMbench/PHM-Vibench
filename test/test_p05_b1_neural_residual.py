from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from src.model_factory.X_model.TSPN_UXFD import Model as TSPNUXFD
from src.model_factory.X_model.UXFD.p05_b1_neural_residual import (
    P05_B1_HIDDEN_BY_CLASSES,
    P05_B1_PARAMETER_COUNT_BY_CLASSES,
    P05B1NeuralResidual,
    P05B1NeuralResidualConfig,
)


def _ns(**values):  # type: ignore[no-untyped-def]
    return SimpleNamespace(**values)


def _args(
    *,
    num_classes: int,
    hidden_dim: int | None = None,
    enable_neural: bool = True,
    enable_fuzzy: bool = False,
    out_channels: int = 4,
) -> SimpleNamespace:
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
            neural_residual=_ns(
                enable=enable_neural,
                hidden_dim=hidden_dim,
            ),
            operator_attention=_ns(enable=False),
            logic=_ns(enable=False),
        ),
    )


@pytest.mark.parametrize(
    ("num_classes", "hidden_dim", "neural_parameters", "fuzzy_parameters"),
    [(4, 26, 342, 344), (2, 29, 321, 324)],
)
def test_b1_frozen_parameter_count_forward_batch_one_and_no_unused_parameters(
    num_classes: int,
    hidden_dim: int,
    neural_parameters: int,
    fuzzy_parameters: int,
) -> None:
    torch.manual_seed(7)
    model = TSPNUXFD(
        _args(num_classes=num_classes, hidden_dim=hidden_dim)
    )
    branch = model._uxfd_neural_residual

    assert isinstance(branch, P05B1NeuralResidual)
    assert branch.input_dim == 8
    assert branch.hidden_dim == hidden_dim
    assert branch.parameter_count == neural_parameters
    assert model._uxfd_fuzzy is None

    x = torch.randn(1, 128, 2)
    model.eval()
    with torch.no_grad():
        features = model._forward_features(x)
        non_fuzzy_logits = model._forward_non_fuzzy_logits(features)
        residual_logits = branch(features)
        direct_logits = model(x)
    assert features.shape == (1, 8)
    assert direct_logits.shape == (1, num_classes)
    assert torch.allclose(
        direct_logits,
        non_fuzzy_logits + residual_logits,
        atol=0.0,
        rtol=0.0,
    )

    model.zero_grad(set_to_none=True)
    model.train()
    model(x).square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())

    fuzzy_model = TSPNUXFD(
        _args(
            num_classes=num_classes,
            hidden_dim=None,
            enable_neural=False,
            enable_fuzzy=True,
        )
    )
    assert fuzzy_model._uxfd_fuzzy is not None
    assert sum(
        parameter.numel() for parameter in fuzzy_model._uxfd_fuzzy.parameters()
    ) == fuzzy_parameters
    assert abs(neural_parameters - fuzzy_parameters) / fuzzy_parameters < 0.05


@pytest.mark.parametrize("num_classes", [2, 4])
def test_b1_auto_selects_only_the_frozen_hidden_width(num_classes: int) -> None:
    branch = P05B1NeuralResidual(input_dim=8, num_classes=num_classes)

    assert branch.hidden_dim == P05_B1_HIDDEN_BY_CLASSES[num_classes]
    assert branch.parameter_count == P05_B1_PARAMETER_COUNT_BY_CLASSES[num_classes]


@pytest.mark.parametrize(
    ("num_classes", "wrong_hidden"),
    [(2, 26), (4, 29)],
)
def test_b1_rejects_hidden_width_drift(
    num_classes: int,
    wrong_hidden: int,
) -> None:
    with pytest.raises(ValueError, match="requires H="):
        P05B1NeuralResidual(
            input_dim=8,
            num_classes=num_classes,
            cfg=P05B1NeuralResidualConfig(hidden_dim=wrong_hidden),
        )


def test_b1_rejects_non_eight_feature_input() -> None:
    with pytest.raises(ValueError, match="same eight-feature input"):
        TSPNUXFD(_args(num_classes=4, hidden_dim=26, out_channels=3))


def test_fuzzy_and_neural_residual_are_mutually_exclusive() -> None:
    with pytest.raises(ValueError, match="forbids enabling fuzzy and neural"):
        TSPNUXFD(
            _args(
                num_classes=4,
                hidden_dim=26,
                enable_neural=True,
                enable_fuzzy=True,
            )
        )


@pytest.mark.parametrize("num_classes", [2, 4])
def test_b1_preserves_the_exact_non_fuzzy_backbone(num_classes: int) -> None:
    torch.manual_seed(19)
    b0 = TSPNUXFD(
        _args(
            num_classes=num_classes,
            enable_neural=False,
            enable_fuzzy=False,
        )
    )
    torch.manual_seed(19)
    b1 = TSPNUXFD(_args(num_classes=num_classes))

    b1_state = b1.state_dict()
    assert b0.state_dict()
    for name, value in b0.state_dict().items():
        assert "_uxfd_neural_residual" not in name
        assert name in b1_state
        assert torch.equal(value, b1_state[name])

    x = torch.randn(1, 128, 2)
    b0.eval()
    b1.eval()
    with torch.no_grad():
        b0_features = b0._forward_features(x)
        b1_features = b1._forward_features(x)
        b0_logits = b0._forward_non_fuzzy_logits(b0_features)
        b1_logits = b1._forward_non_fuzzy_logits(b1_features)
    assert torch.equal(b0_features, b1_features)
    assert torch.equal(b0_logits, b1_logits)
