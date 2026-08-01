from __future__ import annotations

import torch
import pytest

from scripts.p09.g060_arms import (
    DistanceWeightNet,
    FixedGateAdapter,
    FixedPrompt,
    QueryBlindSetAttention,
    method_signature,
    prediction_signature,
    predict_b0,
    predict_b1,
    predict_b2,
    predict_b3,
    predict_b4,
    predict_b5,
    predict_b6,
    predict_b7,
    predict_b8,
    resolve_arm_predictor,
    trainable_parameters,
)
from src.task_factory.task.GFS.reliability_conditioned import (
    SupportReliabilityConditioner,
)
from src.task_factory.task.GFS.reliability_conditioned_variants import (
    condition_variant,
    predict_variant,
)


def _episode(dim: int = 16):
    torch.manual_seed(11)
    base_weights = torch.randn(2, dim)
    base_bias = torch.randn(2)
    support = torch.cat((torch.randn(5, dim) - 0.8, torch.randn(5, dim) + 0.8))
    labels = torch.tensor([2] * 5 + [3] * 5)
    query = torch.randn(12, dim)
    source_base = torch.stack((torch.randn(dim) - 1.0, torch.randn(dim) + 1.0))
    return query, support, labels, base_weights, base_bias, source_base


def test_locked_feature_arms_have_finite_four_class_probabilities() -> None:
    query, support, labels, weights, bias, source_base = _episode()
    b3 = DistanceWeightNet(8)
    b4 = QueryBlindSetAttention(16, 4)
    b6 = FixedPrompt(16)
    b8 = FixedGateAdapter(16, 4)
    predictions = [
        predict_b0(query, support, labels, weights, bias),
        predict_b1(query, support, labels, source_base, temperature=1.0),
        predict_b2(query, support, labels, weights, bias),
        predict_b3(b3, query, support, labels, weights, bias),
        predict_b4(b4, query, support, labels, weights),
        predict_b5(
            query,
            support,
            labels,
            weights,
            bias,
            novel_scale=1.0,
            novel_bias=0.0,
        ),
        predict_b6(b6, query, support, labels, weights, bias),
        predict_b7(query, support, labels, weights, bias, ridge=0.1),
        predict_b8(b8, query, support, labels, weights, bias),
    ]
    for prediction in predictions:
        assert prediction["probabilities"].shape == (12, 4)
        assert torch.isfinite(prediction["probabilities"]).all()
        torch.testing.assert_close(
            prediction["probabilities"].sum(dim=1),
            torch.ones(12),
        )


def test_b0_and_a7_are_exactly_identical_by_independent_calls() -> None:
    query, support, labels, weights, bias, _ = _episode()
    b0 = resolve_arm_predictor("B0")(query, support, labels, weights, bias)
    a7 = resolve_arm_predictor("A7")(query, support, labels, weights, bias)
    assert resolve_arm_predictor("A7") is resolve_arm_predictor("B0")
    torch.testing.assert_close(b0["joint_logits"], a7["joint_logits"], rtol=0, atol=0)
    torch.testing.assert_close(b0["probabilities"], a7["probabilities"], rtol=0, atol=0)


def test_conditioner_variants_preserve_frozen_base_logits() -> None:
    query, support, labels, weights, bias, _ = _episode()
    conditioner = SupportReliabilityConditioner(16, adapter_rank=4)
    reference_base = query @ weights.T + bias
    conditions = {}
    for variant in ("A0", "A1", "A2", "A3", "A4", "A5", "A6", "R1", "R2", "R3", "R4"):
        condition = condition_variant(
            conditioner, support, labels, weights, (2, 3), variant
        )
        prediction = predict_variant(
            conditioner,
            query,
            weights,
            (0, 1),
            condition,
            variant,
            base_bias=bias,
        )
        torch.testing.assert_close(
            prediction["base_logits"], reference_base, rtol=0, atol=0
        )
        conditions[variant] = condition
    assert bool(conditions["A4"].adapter_gate == 0)
    assert torch.all(conditions["R1"].reliability_features[:, 0] == 1)
    assert torch.all(conditions["R4"].reliability_features[:, 3] == 1)


def test_fixed_gate_adapter_obeys_relative_residual_bound() -> None:
    query, *_ = _episode()
    module = FixedGateAdapter(16, rank=4, fixed_gate=0.25, relative_bound=0.10)
    with torch.no_grad():
        module.up.weight.normal_()
    adapted = module.adapt(query)
    residual = torch.linalg.vector_norm(adapted - query, dim=1)
    allowed = 0.25 * 0.10 * torch.linalg.vector_norm(query, dim=1)
    assert torch.all(residual <= allowed + 1.0e-6)
    assert trainable_parameters(module) == 128


def test_method_signatures_bind_implementation_settings_checkpoint_and_state() -> None:
    module = FixedPrompt(16)
    first = method_signature(
        predict_b6,
        module=module,
        settings={"fixed": True},
        checkpoint_sha256="a" * 64,
    )
    assert first == method_signature(
        predict_b6,
        module=module,
        settings={"fixed": True},
        checkpoint_sha256="a" * 64,
    )
    assert first != method_signature(
        predict_b8, module=module, settings={"fixed": True}, checkpoint_sha256="a" * 64
    )
    assert first != method_signature(
        predict_b6, module=module, settings={"fixed": False}, checkpoint_sha256="a" * 64
    )
    assert first != method_signature(
        predict_b6, module=module, settings={"fixed": True}, checkpoint_sha256="b" * 64
    )
    with torch.no_grad():
        module.prompt[0] = 1.0
    assert first != method_signature(
        predict_b6,
        module=module,
        settings={"fixed": True},
        checkpoint_sha256="a" * 64,
    )
    with pytest.raises(TypeError, match="actual prediction callable"):
        method_signature("B6")  # type: ignore[arg-type]


def test_functional_signature_detects_same_behavior_across_arm_labels() -> None:
    query, support, labels, weights, bias, _ = _episode()
    b0 = resolve_arm_predictor("B0")(query, support, labels, weights, bias)
    a7 = resolve_arm_predictor("A7")(query, support, labels, weights, bias)
    assert prediction_signature(b0) == prediction_signature(a7)


def test_swapped_taxonomy_preserves_explicit_class_order() -> None:
    query, support, _, weights, bias, _ = _episode()
    labels = torch.tensor([1] * 5 + [2] * 5)
    prediction = predict_b0(
        query,
        support,
        labels,
        weights,
        bias,
        base_class_ids=(0, 3),
        novel_class_ids=(1, 2),
    )
    assert prediction["joint_class_ids"].tolist() == [0, 3, 1, 2]
    torch.testing.assert_close(
        prediction["prediction_label"],
        prediction["joint_class_ids"][prediction["prediction_index"]],
    )


def test_unexpected_support_class_is_a_hard_failure() -> None:
    query, support, labels, weights, bias, _ = _episode()
    labels = labels.clone()
    labels[0] = 0
    with pytest.raises(ValueError, match="exactly match"):
        predict_b0(query, support, labels, weights, bias)
