import numpy as np
import pytest

from src.data_factory.protocol_transforms import (
    WindowObservation,
    apply_train_channel_standardization,
    fit_train_channel_standardization,
)


def _observations():
    return [
        WindowObservation("g1-w0", "g1", np.asarray([[0.0, 0.0], [0.0, 0.0]])),
        WindowObservation("g1-w1", "g1", np.asarray([[2.0, 4.0], [2.0, 4.0]])),
        WindowObservation("g2-w0", "g2", np.asarray([[10.0, 20.0], [10.0, 20.0]])),
    ]


def test_normalization_equalizes_groups_not_raw_window_counts():
    observations = _observations()
    plan = fit_train_channel_standardization(
        lambda: iter(observations),
        dataset_id=2,
        channel_names=("horizontal", "vertical"),
        expected_window_size=2,
        expected_windows_per_group={"g1": 2, "g2": 1},
    )

    assert plan.mean == pytest.approx((5.5, 11.0))
    assert plan.std == pytest.approx((np.sqrt(20.75), np.sqrt(83.0)))
    assert plan.window_count == 3
    assert len(plan.sha256) == 64
    transformed = apply_train_channel_standardization(
        np.asarray([[5.5, 11.0], [10.0, 20.0]], dtype=np.float64),
        plan,
    )
    assert transformed.dtype == np.float32
    assert transformed[0] == pytest.approx(np.zeros(2))


def test_normalization_is_deterministic_for_identical_ordered_windows():
    observations = _observations()
    kwargs = dict(
        dataset_id=2,
        channel_names=("horizontal", "vertical"),
        expected_window_size=2,
        expected_windows_per_group={"g1": 2, "g2": 1},
    )
    first = fit_train_channel_standardization(lambda: iter(observations), **kwargs)
    second = fit_train_channel_standardization(lambda: iter(observations), **kwargs)
    assert first == second


def test_normalization_rejects_group_count_drift():
    with pytest.raises(ValueError, match="counts mismatch"):
        fit_train_channel_standardization(
            lambda: iter(_observations()),
            dataset_id=2,
            channel_names=("horizontal", "vertical"),
            expected_window_size=2,
            expected_windows_per_group={"g1": 1, "g2": 1},
        )


def test_normalization_rejects_near_constant_channel():
    observations = [
        WindowObservation("a", "g", np.ones((2, 2), dtype=np.float64)),
    ]
    with pytest.raises(ValueError, match="below 1e-8"):
        fit_train_channel_standardization(
            lambda: iter(observations),
            dataset_id=1,
            channel_names=("de", "fe"),
            expected_window_size=2,
            expected_windows_per_group={"g": 1},
        )


def test_normalization_rejects_identity_drift_between_passes():
    calls = 0

    def changing_factory():
        nonlocal calls
        calls += 1
        observations = _observations()
        if calls == 2:
            observations[0] = WindowObservation(
                "changed", observations[0].group_id, observations[0].values
            )
        return iter(observations)

    with pytest.raises(ValueError, match="identities changed"):
        fit_train_channel_standardization(
            changing_factory,
            dataset_id=2,
            channel_names=("horizontal", "vertical"),
            expected_window_size=2,
            expected_windows_per_group={"g1": 2, "g2": 1},
        )
