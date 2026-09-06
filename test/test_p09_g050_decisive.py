from __future__ import annotations

import numpy as np
import torch

from scripts.p09.run_g050_decisive import (
    cleared_control_prediction,
    evenly_spaced_starts,
    feature_vector,
    harmonic_mean,
    mean_prototype_prediction,
    split_target_records,
)


def test_feature_vector_is_fixed_width_and_finite_across_channel_counts() -> None:
    rng = np.random.default_rng(7)
    single_channel = feature_vector(rng.normal(size=(1024, 1)))
    seven_channel = feature_vector(rng.normal(size=(1024, 7, 1)))

    assert single_channel.shape == seven_channel.shape == (24,)
    assert np.isfinite(single_channel).all()
    assert np.isfinite(seven_channel).all()


def test_evenly_spaced_windows_are_unique_and_cover_endpoints() -> None:
    starts = evenly_spaced_starts(length=5000, window_size=1024, count=20)

    assert starts[0] == 0
    assert starts[-1] == 3976
    assert np.unique(starts).size == 20


def test_target_record_split_is_deterministic_and_disjoint() -> None:
    records = {class_id: list(range(100 * class_id, 100 * class_id + 6)) for class_id in range(4)}

    first = split_target_records(records, seed=42, system_id=5)
    second = split_target_records(records, seed=42, system_id=5)

    assert first == second
    for class_id in range(4):
        adaptation = set(first[class_id]["adaptation"])
        query = set(first[class_id]["query"])
        assert adaptation
        assert query
        assert adaptation.isdisjoint(query)
        assert adaptation | query == set(records[class_id])


def test_cleared_control_is_exactly_the_mean_prototype_baseline() -> None:
    torch.manual_seed(11)
    query = torch.randn(12, 6)
    support = torch.randn(10, 6)
    labels = torch.tensor([2] * 5 + [3] * 5)
    base_weights = torch.randn(2, 6)
    base_bias = torch.randn(2)

    baseline = mean_prototype_prediction(
        query, support, labels, base_weights, base_bias, (2, 3)
    )
    negative = cleared_control_prediction(
        query, support, labels, base_weights, base_bias, (2, 3)
    )

    assert torch.equal(baseline["base_logits"], negative["base_logits"])
    assert torch.equal(baseline["joint_logits"], negative["joint_logits"])
    assert torch.equal(baseline["probabilities"], negative["probabilities"])


def test_harmonic_mean_has_explicit_zero_denominator_rule() -> None:
    assert harmonic_mean(0.0, 0.0) == 0.0
    assert harmonic_mean(0.75, 0.50) == 0.60
