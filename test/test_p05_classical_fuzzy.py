from __future__ import annotations

import inspect
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

import src.explain_factory.p05_classical_fuzzy as classical
from src.data_factory.p05_weighting import ExpectedRole, build_weight_plan
from src.explain_factory.p05_classical_fuzzy import (
    P05B4Model,
    P05B4PredictionSplit,
    P05B4TrainingSplit,
    p05_b4_extract_features,
    run_p05_b4_classical_fuzzy,
)


NORMALIZATION_HASH = "a" * 64
SPLIT_HASH = "b" * 64
CACHE_HASH = "c" * 64


def _signal_window(index: int, *, points: int = 64) -> np.ndarray:
    generator = np.random.Generator(np.random.PCG64(7000 + index))
    time = np.linspace(-np.pi, np.pi, points, dtype=np.float64)
    phase = 0.13 * index
    first = (
        (1.0 + 0.025 * index) * np.sin((1 + index % 4) * time + phase)
        + 0.11 * np.cos(2.0 * time - phase)
        + 0.025 * generator.normal(size=points)
        + 0.015 * index
    )
    second = (
        (0.8 + 0.02 * index) * np.cos((1 + index % 3) * time - phase)
        + 0.07 * np.sin(3.0 * time + phase)
        + 0.03 * generator.standard_t(df=5 + index % 4, size=points)
        - 0.01 * index
    )
    return np.stack((first, second), axis=1)


def _fixture() -> tuple[P05B4TrainingSplit, dict[str, P05B4PredictionSplit]]:
    records: list[dict[str, object]] = []
    record_id = 0
    for group_index in range(5):
        for label in (0, 1):
            repeats = 2 if (group_index, label) == (0, 0) else 1
            for _ in range(repeats):
                record_id += 1
                records.append(
                    {
                        "Id": record_id,
                        "Dataset_id": 2,
                        "Label": label,
                        "Protocol_Group": f"bearing-{group_index}",
                        "Protocol_Split": "train",
                    }
                )
    plan = build_weight_plan(
        pd.DataFrame(records),
        dataset_id=2,
        role="train",
        expected=ExpectedRole(
            row_count=11,
            group_count=5,
            class_counts={0: 6, 1: 5},
            windows_per_record=4,
        ),
    )

    sample_ids: list[str] = []
    record_ids: list[int] = []
    labels: list[int] = []
    raw_windows: list[np.ndarray] = []
    signal_index = 0
    for row in records:
        for window_index in range(plan.windows_per_record):
            sample_ids.append(f"train-{row['Id']}-{window_index}")
            record_ids.append(int(row["Id"]))
            labels.append(int(row["Label"]))
            raw_windows.append(_signal_window(signal_index))
            signal_index += 1
    raw = np.stack(raw_windows)
    point_mean = raw.reshape(-1, 2).mean(axis=0, dtype=np.float64)
    point_std = raw.reshape(-1, 2).std(axis=0, dtype=np.float64)
    windows = ((raw - point_mean) / point_std).astype(np.float32)

    prediction_raw = np.stack([_signal_window(100 + index) for index in range(6)])
    prediction_windows = ((prediction_raw - point_mean) / point_std).astype(np.float32)
    train = P05B4TrainingSplit(
        sample_ids=tuple(sample_ids),
        record_ids=tuple(record_ids),
        windows=windows,
        labels=tuple(labels),
        weight_plan=plan,
    )
    prediction = P05B4PredictionSplit(
        sample_ids=tuple(f"test-{index}" for index in range(6)),
        windows=prediction_windows,
    )
    return train, {"test": prediction}


def _cwru_fixture() -> tuple[P05B4TrainingSplit, dict[str, P05B4PredictionSplit]]:
    records = [
        {
            "Id": 200 + label,
            "Dataset_id": 1,
            "Label": label,
            "Protocol_Group": f"recording-{label}",
            "Protocol_Split": "train",
        }
        for label in range(4)
    ]
    plan = build_weight_plan(
        pd.DataFrame(records),
        dataset_id=1,
        role="train",
        expected=ExpectedRole(
            row_count=4,
            group_count=4,
            class_counts={0: 1, 1: 1, 2: 1, 3: 1},
            windows_per_record=16,
        ),
    )
    sample_ids: list[str] = []
    record_ids: list[int] = []
    labels: list[int] = []
    raw_windows: list[np.ndarray] = []
    signal_index = 300
    for row in records:
        for window_index in range(plan.windows_per_record):
            sample_ids.append(f"cwru-{row['Id']}-{window_index}")
            record_ids.append(int(row["Id"]))
            labels.append(int(row["Label"]))
            raw_windows.append(_signal_window(signal_index))
            signal_index += 1
    raw = np.stack(raw_windows)
    point_mean = raw.reshape(-1, 2).mean(axis=0, dtype=np.float64)
    point_std = raw.reshape(-1, 2).std(axis=0, dtype=np.float64)
    windows = ((raw - point_mean) / point_std).astype(np.float32)
    prediction_raw = np.stack([_signal_window(500 + index) for index in range(4)])
    prediction_windows = ((prediction_raw - point_mean) / point_std).astype(np.float32)
    return (
        P05B4TrainingSplit(
            sample_ids=tuple(sample_ids),
            record_ids=tuple(record_ids),
            windows=windows,
            labels=tuple(labels),
            weight_plan=plan,
        ),
        {
            "test": P05B4PredictionSplit(
                sample_ids=tuple(f"cwru-test-{index}" for index in range(4)),
                windows=prediction_windows,
            )
        },
    )


def _run(package, **overrides):
    train, predictions = _fixture()
    values = {
        "train": train,
        "prediction_splits": predictions,
        "channel_standardization_sha256": NORMALIZATION_HASH,
        "split_manifest_sha256": SPLIT_HASH,
        "signal_cache_manifest_sha256": CACHE_HASH,
        "expected_window_size": 64,
    }
    values.update(overrides)
    return run_p05_b4_classical_fuzzy(package, **values)


def test_b4_feature_definitions_are_per_channel_float64_and_fail_closed() -> None:
    windows = np.asarray(
        [[[-1.0, 0.0], [0.0, 1.0], [1.0, 4.0], [2.0, 9.0]]],
        dtype=np.float32,
    )
    features = p05_b4_extract_features(windows)

    expected: list[float] = []
    for channel in range(2):
        values = windows[0, :, channel].astype(np.float64)
        mean = values.mean(dtype=np.float64)
        centered = values - mean
        second = np.square(centered).mean(dtype=np.float64)
        expected.extend(
            (
                mean,
                np.sqrt(second),
                np.sqrt(np.square(values).mean(dtype=np.float64)),
                np.square(np.square(centered)).mean(dtype=np.float64)
                / np.square(second),
            )
        )

    assert features.dtype == np.float64
    np.testing.assert_allclose(features[0], expected, rtol=0.0, atol=0.0)

    constant_channel = windows.copy()
    constant_channel[0, :, 1] = 3.0
    with pytest.raises(ValueError, match="below 1e-12.*channel=1"):
        p05_b4_extract_features(constant_channel)


def test_b4_fit_is_deterministic_and_matches_every_frozen_formula(tmp_path) -> None:
    train, predictions = _fixture()
    first = _run(tmp_path / "first", train=train, prediction_splits=predictions)
    second = _run(tmp_path / "second", train=train, prediction_splits=predictions)
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))

    assert first.semantic_sha256 == second.semantic_sha256
    assert first.model_sha256 == second.model_sha256
    assert first.predictions_sha256 == second.predictions_sha256
    assert first.manifest_sha256 == second.manifest_sha256
    assert first.iterations == second.iterations
    assert 1 <= first.iterations <= 300
    assert first.final_max_center_shift <= 1.0e-5
    assert first_manifest["fit_id"] == "P05-B4-dataset-2"
    assert first_manifest["fit_contract"]["fits_per_dataset"] == 1
    assert first_manifest["fit_contract"]["validation_grid"] == "none"
    assert first_manifest["fit_contract"]["model_seed_repetition"].startswith(
        "forbidden"
    )
    assert first_manifest["provenance"]["channel_standardization_sha256"] == (
        NORMALIZATION_HASH
    )
    assert first_manifest["provenance"]["split_manifest_sha256"] == SPLIT_HASH
    assert first_manifest["provenance"]["signal_cache_manifest_sha256"] == CACHE_HASH
    assert first_manifest["model"] == second_manifest["model"]
    assert "model_seed" not in inspect.signature(
        run_p05_b4_classical_fuzzy
    ).parameters

    with np.load(first.model_path, allow_pickle=False) as first_model, np.load(
        second.model_path,
        allow_pickle=False,
    ) as second_model:
        assert set(first_model.files) == set(second_model.files)
        assert all(
            np.array_equal(first_model[name], second_model[name])
            for name in first_model.files
        )

        train_features = p05_b4_extract_features(train.windows)
        sample_weights = np.asarray(
            [train.weight_plan.record_weights[value] for value in train.record_ids],
            dtype=np.float64,
        )
        total_weight = sample_weights.sum(dtype=np.float64)
        feature_mean = (
            sample_weights[:, None] * train_features
        ).sum(axis=0, dtype=np.float64) / total_weight
        feature_std = np.sqrt(
            (
                sample_weights[:, None]
                * np.square(train_features - feature_mean)
            ).sum(axis=0, dtype=np.float64)
            / total_weight
        )
        standardized = (train_features - feature_mean) / feature_std
        np.testing.assert_allclose(first_model["feature_mean"], feature_mean)
        np.testing.assert_allclose(first_model["feature_std"], feature_std)

        generator = np.random.Generator(np.random.PCG64(20260801))
        expected_initial = generator.random((len(train.sample_ids), 10))
        expected_initial /= expected_initial.sum(axis=1, keepdims=True)
        np.testing.assert_array_equal(
            first_model["fcm_initial_memberships"],
            expected_initial,
        )

        memberships = first_model["fcm_final_memberships"]
        effective = sample_weights[:, None] * np.square(memberships)
        total_mass = effective.sum(axis=0, dtype=np.float64)
        expected_centers = effective.T @ standardized / total_mass[:, None]
        np.testing.assert_allclose(first_model["centers"], expected_centers)
        expected_widths = np.maximum(
            np.sqrt(
                (
                    effective[:, :, None]
                    * np.square(
                        standardized[:, None, :] - expected_centers[None, :, :]
                    )
                ).sum(axis=0, dtype=np.float64)
                / total_mass[:, None]
            ),
            1.0e-4,
        )
        np.testing.assert_allclose(first_model["widths"], expected_widths)
        expected_consequents = np.empty((10, 2), dtype=np.float64)
        labels = np.asarray(train.labels)
        for class_id in range(2):
            class_mass = effective[labels == class_id].sum(axis=0, dtype=np.float64)
            expected_consequents[:, class_id] = (1.0 + class_mass) / (
                2.0 + total_mass
            )
        np.testing.assert_allclose(
            first_model["consequents"],
            expected_consequents,
        )

        with np.load(first.predictions_path, allow_pickle=False) as output:
            prediction_features = p05_b4_extract_features(
                predictions["test"].windows
            )
            prediction_standardized = (
                prediction_features - feature_mean
            ) / feature_std
            scaled = (
                prediction_standardized[:, None, :]
                - expected_centers[None, :, :]
            ) / expected_widths[None, :, :]
            log_firing = (-0.5 * np.square(scaled)).mean(axis=2)
            expected_firing = np.exp(
                log_firing - log_firing.max(axis=1, keepdims=True)
            )
            expected_firing /= expected_firing.sum(axis=1, keepdims=True)
            expected_scores = expected_firing @ expected_consequents
            expected_labels = np.argmax(expected_scores, axis=1)
            np.testing.assert_allclose(
                output["test__normalized_rule_firing"],
                expected_firing,
            )
            np.testing.assert_allclose(output["test__class_scores"], expected_scores)
            np.testing.assert_array_equal(
                output["test__predicted_labels"],
                expected_labels,
            )
            np.testing.assert_allclose(
                output["test__normalized_rule_firing"].sum(axis=1),
                1.0,
                rtol=0.0,
                atol=1.0e-15,
            )


def test_b4_binds_cwru_to_four_classes_without_a_seed_axis(tmp_path) -> None:
    train, predictions = _cwru_fixture()
    result = run_p05_b4_classical_fuzzy(
        tmp_path / "cwru",
        train=train,
        prediction_splits=predictions,
        channel_standardization_sha256=NORMALIZATION_HASH,
        split_manifest_sha256=SPLIT_HASH,
        signal_cache_manifest_sha256=CACHE_HASH,
        expected_window_size=64,
    )
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert result.dataset_id == 1
    assert manifest["fit_id"] == "P05-B4-dataset-1"
    assert manifest["model"]["num_classes"] == 4
    with np.load(result.model_path, allow_pickle=False) as model:
        assert model["consequents"].shape == (10, 4)
    with np.load(result.predictions_path, allow_pickle=False) as predictions_file:
        labels = predictions_file["test__predicted_labels"]
        assert labels.dtype == np.int64
        assert np.all((0 <= labels) & (labels < 4))


def test_b4_zero_distance_splits_only_among_exact_centers() -> None:
    features = np.zeros((1, 8), dtype=np.float64)
    centers = np.ones((10, 8), dtype=np.float64)
    centers[0] = 0.0
    centers[1] = 0.0

    memberships = classical._memberships_for_centers(features, centers)

    np.testing.assert_array_equal(memberships[0, :2], (0.5, 0.5))
    np.testing.assert_array_equal(memberships[0, 2:], np.zeros(8))


def test_b4_width_floor_laplace_consequents_and_lower_class_tie() -> None:
    features = np.zeros((2, 8), dtype=np.float64)
    memberships = np.full((2, 10), 0.1, dtype=np.float64)
    weights = np.asarray([1.0, 3.0], dtype=np.float64)
    labels = np.asarray([0, 1], dtype=np.int64)
    centers = np.zeros((10, 8), dtype=np.float64)

    widths, consequents = classical._derive_widths_and_consequents(
        features,
        labels,
        weights,
        memberships,
        centers,
        num_classes=2,
    )

    np.testing.assert_array_equal(widths, np.full((10, 8), 1.0e-4))
    total_mass = 0.04
    expected = np.asarray([(1.0 + 0.01) / 2.04, (1.0 + 0.03) / 2.04])
    np.testing.assert_allclose(consequents, np.tile(expected, (10, 1)))
    assert np.allclose(consequents.sum(axis=1), 1.0)

    tied_model = P05B4Model(
        dataset_id=2,
        num_classes=2,
        feature_mean=np.zeros(8),
        feature_std=np.ones(8),
        centers=np.zeros((10, 8)),
        widths=np.ones((10, 8)),
        consequents=np.full((10, 2), 0.5),
    )
    prediction = classical._predict_from_features(
        np.zeros((1, 8), dtype=np.float64),
        tied_model,
    )
    assert prediction.predicted_labels.tolist() == [0]


def test_b4_fails_on_degenerate_features_weights_and_nonconvergence(tmp_path) -> None:
    train, predictions = _fixture()
    repeated = np.repeat(train.windows[:1], len(train.sample_ids), axis=0)
    with pytest.raises(ValueError, match="feature standard deviation is below 1e-8"):
        _run(
            tmp_path / "degenerate",
            train=replace(train, windows=repeated),
            prediction_splits=predictions,
        )
    assert not (tmp_path / "degenerate").exists()

    with pytest.raises(ValueError, match="source SHA-256"):
        _run(
            tmp_path / "bad-weight-hash",
            train=replace(
                train,
                weight_plan=replace(train.weight_plan, sha256="0" * 64),
            ),
            prediction_splits=predictions,
        )
    assert not (tmp_path / "bad-weight-hash").exists()

    features = np.arange(160, dtype=np.float64).reshape(20, 8)
    weights = np.ones(20, dtype=np.float64)
    with pytest.raises(RuntimeError, match="failed to converge"):
        classical._fit_weighted_fuzzy_c_means(
            features,
            weights,
            max_iterations=1,
            tolerance=-1.0,
        )


def test_b4_public_run_requires_registered_window_size(tmp_path) -> None:
    train, predictions = _fixture()
    with pytest.raises(ValueError, match="registered window size 4096"):
        run_p05_b4_classical_fuzzy(
            tmp_path / "short-production-window",
            train=train,
            prediction_splits=predictions,
            channel_standardization_sha256=NORMALIZATION_HASH,
            split_manifest_sha256=SPLIT_HASH,
            signal_cache_manifest_sha256=CACHE_HASH,
        )
    assert not (tmp_path / "short-production-window").exists()


def test_b4_create_only_conflict_precedes_refit(tmp_path, monkeypatch) -> None:
    package = tmp_path / "one-fit"
    first = _run(package)
    model_before = first.model_path.read_bytes()
    predictions_before = first.predictions_path.read_bytes()
    manifest_before = first.manifest_path.read_bytes()

    def fail_if_refit(*args, **kwargs):
        del args, kwargs
        raise AssertionError("existing target must fail before B4 fitting")

    monkeypatch.setattr(classical, "_prepare_training", fail_if_refit)
    with pytest.raises(FileExistsError, match="conflicts"):
        _run(package)

    assert first.model_path.read_bytes() == model_before
    assert first.predictions_path.read_bytes() == predictions_before
    assert first.manifest_path.read_bytes() == manifest_before
