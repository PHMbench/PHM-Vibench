from __future__ import annotations

import copy
from types import SimpleNamespace
from typing import Optional

import pytest
import torch
import torch.nn as nn

from src.model_factory.X_model.TSPN_UXFD import Model as TSPNUXFD


def _ns(**kwargs):  # type: ignore[no-untyped-def]
    return SimpleNamespace(**kwargs)


def _make_args(
    *,
    enable_sp2d: bool = False,
    fusion_type: str = "concat",
    enable_fuzzy: bool = False,
    fuzzy_logit_scale: float = 1.0,
    enable_operator_attention: bool = False,
    operator_list: Optional[list[str]] = None,
    enable_logic: bool = False,
    logic_logit_scale: float = 1.0,
    internal_instance_normalization: bool = True,
    num_classes: object = 3,
) -> SimpleNamespace:
    uxfd = _ns(
        enable_sp2d=enable_sp2d,
        sp2d=_ns(n_fft=128, hop_length=64),
        fusion=_ns(type=fusion_type),
        fuzzy=_ns(enable=enable_fuzzy, logit_scale=fuzzy_logit_scale),
        operator_attention=_ns(
            enable=enable_operator_attention,
            operators=operator_list or ["I", "FFT"],
            hidden_dim=32,
            temperature=1.0,
        ),
        logic=_ns(enable=enable_logic, logit_scale=logic_logit_scale, hidden_dim=32),
    )

    return _ns(
        device="cpu",
        num_classes=num_classes,
        in_channels=2,
        out_channels=4,
        scale=1,
        skip_connection=True,
        internal_instance_normalization=internal_instance_normalization,
        signal_processing_configs={"layer1": ["I"]},
        feature_extractor_configs=["Mean", "Std"],
        in_dim=128,
        out_dim=128,
        uxfd=uxfd,
    )


def _forward_once(model: TSPNUXFD, x: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        return model(x)


def test_tspn_uxfd_rejects_dict_num_classes() -> None:
    args = _make_args(num_classes={"source": 3, "target": 4})

    with pytest.raises(ValueError, match="dict-valued multi-dataset heads are out of scope"):
        TSPNUXFD(args)


def test_tspn_uxfd_rejects_complex_sp2d_output() -> None:
    args = _make_args(enable_sp2d=True)
    args.uxfd.sp2d.magnitude = False

    with pytest.raises(ValueError, match="requires uxfd.sp2d.magnitude=true"):
        TSPNUXFD(args)


def test_tspn_uxfd_preserves_internal_instance_normalization_by_default() -> None:
    args = _make_args()
    del args.internal_instance_normalization

    model = TSPNUXFD(args)

    assert all(
        isinstance(layer.norm, nn.InstanceNorm1d)
        for layer in model.signal_processing_layers
    )
    assert isinstance(model.feature_extractor_layers.pre_norm, nn.InstanceNorm1d)


def test_tspn_uxfd_disables_internal_instance_normalization_and_preserves_singleton_batch() -> None:
    torch.manual_seed(0)
    args = _make_args(internal_instance_normalization=False)
    model = TSPNUXFD(args)
    x = torch.randn(1, 128, 2)

    model.train()
    logits = model(x)
    logits.sum().backward()

    assert all(isinstance(layer.norm, nn.Identity) for layer in model.signal_processing_layers)
    assert isinstance(model.feature_extractor_layers.pre_norm, nn.Identity)
    assert logits.shape == (1, args.num_classes)
    assert torch.isfinite(logits).all()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_tspn_uxfd_sp2d_modules_follow_configured_device() -> None:
    args = _make_args(enable_sp2d=True, fusion_type="concat")
    args.device = "meta"

    model = TSPNUXFD(args)

    assert model._uxfd_2d_proj is not None
    assert model._uxfd_2d_proj.weight.device.type == "meta"
    assert model._uxfd_fusion is not None
    fusion_params = list(model._uxfd_fusion.parameters())
    assert fusion_params
    assert all(param.device.type == "meta" for param in fusion_params)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_tspn_uxfd_sp2d_forward_on_cuda() -> None:
    torch.manual_seed(0)
    args = _make_args(enable_sp2d=True, fusion_type="gated")
    args.device = "cuda"
    model = TSPNUXFD(args)
    x = torch.randn(2, 128, 2, device="cuda")

    logits = _forward_once(model, x)

    assert logits.shape == (2, args.num_classes)
    assert logits.device.type == "cuda"
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize("fusion_type", ["concat", "sum", "gated"])
def test_tspn_uxfd_sp2d_fusion_forward_shape(fusion_type: str) -> None:
    torch.manual_seed(0)
    args = _make_args(enable_sp2d=True, fusion_type=fusion_type)
    model = TSPNUXFD(args)
    x = torch.randn(2, 128, 2)

    logits = _forward_once(model, x)
    assert logits.shape == (2, args.num_classes)
    assert torch.isfinite(logits).all()


def test_tspn_uxfd_fuzzy_and_logic_residuals_forward_shape() -> None:
    torch.manual_seed(0)
    args = _make_args(enable_fuzzy=True, enable_logic=True, fuzzy_logit_scale=0.5, logic_logit_scale=0.5)
    model = TSPNUXFD(args)
    x = torch.randn(2, 128, 2)

    logits = _forward_once(model, x)
    assert logits.shape == (2, args.num_classes)
    assert torch.isfinite(logits).all()


def test_tspn_uxfd_fuzzy_trace_reconstructs_every_logit() -> None:
    torch.manual_seed(0)
    args = _make_args(
        enable_fuzzy=True,
        fuzzy_logit_scale=0.5,
        enable_logic=True,
        logic_logit_scale=0.25,
    )
    model = TSPNUXFD(args)
    model.eval()
    x = torch.randn(3, 128, 2)

    with torch.no_grad():
        direct_logits = model(x)
        output = model.forward_with_fuzzy_trace(x)

    fuzzy_trace = output.fuzzy_trace
    assert fuzzy_trace.membership_values.shape == (3, 8, 3)
    assert fuzzy_trace.antecedent_probabilities.shape == (10, 8, 3)
    assert fuzzy_trace.rule_contributions.shape == (3, 10, args.num_classes)
    assert torch.all(fuzzy_trace.centers[:, 1:] > fuzzy_trace.centers[:, :-1])
    torch.testing.assert_close(output.logits, direct_logits, atol=1.0e-7, rtol=1.0e-6)
    torch.testing.assert_close(
        fuzzy_trace.fuzzy_logits,
        fuzzy_trace.reconstruct_fuzzy_logits(),
        atol=1.0e-7,
        rtol=1.0e-6,
    )
    torch.testing.assert_close(
        output.logits,
        output.reconstruct_logits(),
        atol=1.0e-7,
        rtol=1.0e-6,
    )


def test_tspn_uxfd_rule_deletion_has_explicit_reconstructable_semantics() -> None:
    torch.manual_seed(1)
    args = _make_args(enable_fuzzy=True, fuzzy_logit_scale=0.5)
    model = TSPNUXFD(args)
    model.eval()
    x = torch.randn(4, 128, 2)

    with torch.no_grad():
        original = model.forward_with_fuzzy_trace(x)
        predicted_class = original.logits.argmax(dim=-1)
        class_index = predicted_class[:, None, None].expand(-1, 10, 1)
        contribution_scores = original.scaled_rule_contributions().gather(
            dim=2,
            index=class_index,
        ).squeeze(-1).abs()
        top_rule = contribution_scores.argmax(dim=1)
        mask = torch.ones((4, 10), dtype=torch.bool)
        mask[torch.arange(4), top_rule] = False
        deleted = model.forward_with_fuzzy_trace(x, rule_mask=mask)

    assert torch.equal(deleted.fuzzy_trace.rule_mask, mask)
    assert torch.equal(
        deleted.fuzzy_trace.normalized_rule_firing[torch.arange(4), top_rule],
        torch.zeros(4),
    )
    torch.testing.assert_close(
        deleted.fuzzy_trace.normalized_rule_firing.sum(dim=1),
        torch.ones(4),
        atol=1.0e-7,
        rtol=1.0e-7,
    )
    torch.testing.assert_close(
        deleted.logits,
        deleted.reconstruct_logits(),
        atol=1.0e-7,
        rtol=1.0e-6,
    )


def test_tspn_uxfd_consequent_shuffle_preserves_firing_and_reconstructs() -> None:
    torch.manual_seed(2)
    args = _make_args(enable_fuzzy=True)
    model = TSPNUXFD(args)
    model.eval()
    x = torch.randn(2, 128, 2)
    permutation = torch.arange(9, -1, -1)

    with torch.no_grad():
        original = model.forward_with_fuzzy_trace(x)
        shuffled = model.forward_with_fuzzy_trace(
            x,
            consequent_permutation=permutation,
        )

    torch.testing.assert_close(
        original.fuzzy_trace.normalized_rule_firing,
        shuffled.fuzzy_trace.normalized_rule_firing,
        atol=0.0,
        rtol=0.0,
    )
    assert torch.equal(shuffled.fuzzy_trace.consequent_permutation, permutation)
    torch.testing.assert_close(
        shuffled.logits,
        shuffled.reconstruct_logits(),
        atol=1.0e-7,
        rtol=1.0e-6,
    )


def test_tspn_uxfd_trace_risk_features_require_external_coefficients() -> None:
    torch.manual_seed(3)
    args = _make_args(enable_fuzzy=True)
    model = TSPNUXFD(args)
    model.eval()
    x = torch.randn(2, 128, 2)

    with torch.no_grad():
        output = model.forward_with_fuzzy_trace(x)
        features = output.risk_features()
        risk = output.calibrated_risk(torch.tensor([1.0, 0.5, -0.25]), 0.1)

    assert features.shape == (2, 3)
    assert torch.isfinite(features).all()
    assert torch.all((features >= 0.0) & (features <= 1.0))
    assert risk.shape == (2,)
    assert torch.all((risk > 0.0) & (risk < 1.0))


def test_tspn_uxfd_trace_rejects_deleting_every_rule() -> None:
    args = _make_args(enable_fuzzy=True)
    model = TSPNUXFD(args)
    x = torch.randn(2, 128, 2)

    with pytest.raises(ValueError, match="retain at least one rule"):
        model.forward_with_fuzzy_trace(x, rule_mask=torch.zeros(10, dtype=torch.bool))


def test_tspn_uxfd_operator_attention_debug_state() -> None:
    torch.manual_seed(0)
    args = _make_args(enable_operator_attention=True, operator_list=["I", "FFT"])
    model = TSPNUXFD(args)
    x = torch.randn(2, 128, 2)

    _ = _forward_once(model, x)
    state = model.get_uxfd_debug_state()
    assert state["enable_operator_attention"] is True


def test_tspn_uxfd_forward_is_repeatable_given_same_state() -> None:
    torch.manual_seed(0)
    args = _make_args(enable_sp2d=True, fusion_type="gated", enable_operator_attention=True)
    model = TSPNUXFD(args)
    x = torch.randn(2, 128, 2)

    state_before = copy.deepcopy(model.state_dict())
    out1 = _forward_once(model, x)
    model.load_state_dict(state_before)
    out2 = _forward_once(model, x)
    assert torch.allclose(out1, out2, atol=0.0, rtol=0.0)
