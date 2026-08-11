from types import SimpleNamespace

import numpy as np
import pytest

from phmfactory.config import resolve_config
from src.data_factory.dataset_task.Default_dataset import Default_dataset


_METADATA = {1: {"Label": 0}}
_TASK = SimpleNamespace(type="DG")
_DATA = {1: np.arange(1, 17, dtype=np.float32).reshape(-1, 1)}


def _args(**overrides):
    values = {
        "window_size": 8,
        "num_window": 2,
        "window_sampling_strategy": "evenly_spaced",
        "window_sampling_seed": 0,
        "train_ratio": 0.5,
        "val_ratio": 0.25,
        "test_ratio": 0.25,
        "unused_ratio": 0.0,
        "normalization": "none",
        "dtype": "float32",
        "train_noise_snr": None,
        "evaluation_noise_snr": None,
        "evaluation_noise_seed": 7,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_maintained_smoke_declares_consumed_transform_fields():
    data = resolve_config("configs/demo/00_smoke/dummy_dg.yaml").data["data"]

    assert data["window_sampling_strategy"] == "evenly_spaced"
    assert "stride" not in data
    assert data["normalization"] == "per_window_standardization"
    assert (
        data["train_ratio"]
        + data["val_ratio"]
        + data["test_ratio"]
        + data["unused_ratio"]
        == 1.0
    )


def test_stride_is_only_valid_for_sequential_sampling():
    with pytest.raises(ValueError, match="stride is only consumed"):
        Default_dataset(
            _DATA,
            _METADATA,
            _args(stride=2),
            _TASK,
            "train",
        )

    with pytest.raises(ValueError, match="stride must be a positive integer"):
        Default_dataset(
            _DATA,
            _METADATA,
            _args(window_sampling_strategy="sequential"),
            _TASK,
            "train",
        )

    dataset = Default_dataset(
        _DATA,
        _METADATA,
        _args(window_sampling_strategy="sequential", stride=2),
        _TASK,
        "train",
    )
    assert dataset.window_intervals == [(0, 8)]


def test_split_ratios_require_explicit_unused_data():
    with pytest.raises(ValueError, match="must equal 1.0"):
        Default_dataset(
            _DATA,
            _METADATA,
            _args(test_ratio=0.0),
            _TASK,
            "train",
        )

    dataset = Default_dataset(
        _DATA,
        _METADATA,
        _args(test_ratio=0.0, unused_ratio=0.25),
        _TASK,
        "train",
    )
    assert len(dataset) == 1


def test_non_finite_reader_output_fails_at_data_boundary():
    bad_data = {1: np.array([1.0, 2.0, np.nan, 4.0, 5.0, 6.0, 7.0, 8.0])}
    with pytest.raises(FloatingPointError, match="contains NaN or Inf"):
        Default_dataset(bad_data, _METADATA, _args(num_window=1), _TASK, "test")


def test_per_window_standardization_is_named_and_finite():
    dataset = Default_dataset(
        _DATA,
        _METADATA,
        _args(
            num_window=1,
            normalization="per_window_standardization",
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0,
        ),
        _TASK,
        "train",
    )
    window = dataset.processed_data[0]
    assert np.isfinite(window).all()
    assert np.allclose(window.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(window.std(axis=0), 1.0, atol=1e-6)


def test_noise_is_split_specific_and_evaluation_noise_is_deterministic():
    clean_eval = Default_dataset(
        _DATA,
        _METADATA,
        _args(num_window=1),
        _TASK,
        "test",
    )
    train_noisy = Default_dataset(
        _DATA,
        _METADATA,
        _args(
            num_window=1,
            train_noise_snr=20,
            train_ratio=1.0,
            val_ratio=0.0,
            test_ratio=0.0,
        ),
        _TASK,
        "train",
    )
    assert not np.array_equal(
        train_noisy.processed_data[0],
        clean_eval.processed_data[0],
    )

    eval_args = _args(num_window=1, evaluation_noise_snr=20)
    noisy_eval_a = Default_dataset(_DATA, _METADATA, eval_args, _TASK, "test")
    noisy_eval_b = Default_dataset(_DATA, _METADATA, eval_args, _TASK, "test")
    assert np.array_equal(
        noisy_eval_a.processed_data[0],
        noisy_eval_b.processed_data[0],
    )
    assert not np.array_equal(
        noisy_eval_a.processed_data[0],
        clean_eval.processed_data[0],
    )


def test_legacy_ambiguous_noise_setting_fails_with_repair_path():
    with pytest.raises(ValueError, match="train_noise_snr"):
        Default_dataset(
            _DATA,
            _METADATA,
            _args(noise_snr=20),
            _TASK,
            "test",
        )


def test_requested_noise_cannot_silently_skip_zero_power_signal():
    zero_data = {1: np.zeros((8, 1), dtype=np.float32)}
    with pytest.raises(ValueError, match="signal power must be finite and positive"):
        Default_dataset(
            zero_data,
            _METADATA,
            _args(
                num_window=1,
                evaluation_noise_snr=20,
                train_ratio=0.0,
                val_ratio=0.0,
                test_ratio=1.0,
            ),
            _TASK,
            "test",
        )
