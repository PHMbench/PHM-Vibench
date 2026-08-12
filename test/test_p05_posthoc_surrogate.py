from __future__ import annotations

import inspect
import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest
import torch

import src.explain_factory.p05_posthoc_surrogate as surrogate
from src.data_factory.p05_weighting import ExpectedRole, WeightPlan, build_weight_plan
from src.explain_factory.p05_posthoc_surrogate import (
    P05B2FrozenSplit,
    load_p05_b2_surrogate_checkpoint,
    p05_b2_weighted_logit_mse,
    train_p05_b2_posthoc_surrogate,
)


B0_CHECKPOINT_HASH = "a" * 64
B0_RUN_HASH = "b" * 64


def _registered_plans() -> tuple[WeightPlan, WeightPlan, list[dict], list[dict]]:
    train_records = []
    record_id = 0
    for group_index in range(5):
        for label in (0, 1):
            repeats = 2 if (group_index, label) == (0, 0) else 1
            for _ in range(repeats):
                record_id += 1
                train_records.append(
                    {
                        "Id": record_id,
                        "Dataset_id": 2,
                        "Label": label,
                        "Protocol_Group": f"train-g{group_index}",
                        "Protocol_Split": "train",
                    }
                )
    validation_records = [
        {
            "Id": 101,
            "Dataset_id": 2,
            "Label": 0,
            "Protocol_Group": "val-g0",
            "Protocol_Split": "validation",
        },
        {
            "Id": 102,
            "Dataset_id": 2,
            "Label": 1,
            "Protocol_Group": "val-g0",
            "Protocol_Split": "validation",
        },
        {
            "Id": 103,
            "Dataset_id": 2,
            "Label": 0,
            "Protocol_Group": "val-g1",
            "Protocol_Split": "validation",
        },
    ]
    train_frame = pd.DataFrame(train_records)
    validation_frame = pd.DataFrame(validation_records)
    train_plan = build_weight_plan(
        train_frame,
        dataset_id=2,
        role="train",
        expected=ExpectedRole(
            row_count=len(train_records),
            group_count=5,
            class_counts={0: 6, 1: 5},
            windows_per_record=4,
        ),
    )
    validation_plan = build_weight_plan(
        validation_frame,
        dataset_id=2,
        role="validation",
        expected=ExpectedRole(
            row_count=3,
            group_count=2,
            class_counts={0: 2, 1: 1},
            windows_per_record=4,
        ),
    )
    return train_plan, validation_plan, train_records, validation_records


def _split(
    records: list[dict],
    plan: WeightPlan,
    *,
    prefix: str,
    target: tuple[float, float],
) -> P05B2FrozenSplit:
    sample_ids = []
    record_ids = []
    group_ids = []
    features = []
    logits = []
    for row_index, row in enumerate(records):
        for window in range(plan.windows_per_record):
            sample_ids.append(f"{prefix}-{row['Id']}-{window}")
            record_ids.append(row["Id"])
            group_ids.append(row["Protocol_Group"])
            features.append([0.01 * row_index] * 8)
            logits.append(target)
    return P05B2FrozenSplit(
        sample_ids=tuple(sample_ids),
        record_ids=tuple(record_ids),
        group_ids=tuple(group_ids),
        features=torch.tensor(features, dtype=torch.float32),
        b0_logits=torch.tensor(logits, dtype=torch.float32),
        weight_plan=plan,
    )


def _fixture() -> tuple[P05B2FrozenSplit, P05B2FrozenSplit]:
    train_plan, validation_plan, train_records, validation_records = _registered_plans()
    train = _split(
        train_records,
        train_plan,
        prefix="train",
        target=(2.0, 1.0),
    )
    validation = _split(
        validation_records,
        validation_plan,
        prefix="validation",
        target=(-2.0, -1.0),
    )
    return train, validation


def _train(package, **overrides):
    train, validation = _fixture()
    values = {
        "train": train,
        "validation": validation,
        "model_seed": 42,
        "b0_checkpoint_sha256": B0_CHECKPOINT_HASH,
        "b0_run_artifact_semantic_sha256": B0_RUN_HASH,
    }
    values.update(overrides)
    return train_p05_b2_posthoc_surrogate(package, **values)


def test_weighted_logit_mse_is_mean_weight_times_per_window_mean_k() -> None:
    prediction = torch.zeros((3, 2), dtype=torch.float32)
    target = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]])
    weights = torch.tensor([1.0, 1.0, 2.0], dtype=torch.float32)

    loss = p05_b2_weighted_logit_mse(prediction, target, weights)

    assert float(loss) == pytest.approx((1.0 * 1.0 + 1.0 * 4.0 + 2.0 * 9.0) / 4.0)


def test_b2_is_deterministic_early_stopped_and_strictly_selects_minimum(tmp_path) -> None:
    first = _train(tmp_path / "first")
    second = _train(tmp_path / "second")
    first_manifest = json.loads(first.manifest_path.read_text(encoding="utf-8"))
    second_manifest = json.loads(second.manifest_path.read_text(encoding="utf-8"))

    assert first.stopped_early is True
    assert first.epochs_ran == first.best_epoch + 15
    assert first.epochs_ran < 100
    assert first.semantic_sha256 == second.semantic_sha256
    assert first_manifest["selection"] == second_manifest["selection"]
    validation_history = [
        row["validation_equal_group_equal_window_weighted_logit_mse"]
        for row in first_manifest["selection"]["history"]
    ]
    assert first.best_validation_mse == min(validation_history)
    assert first.best_epoch == validation_history.index(min(validation_history)) + 1
    assert first_manifest["training_contract"]["label_usage"] == (
        "forbidden_for_target_training_and_selection"
    )
    assert first_manifest["training_contract"]["target"] == "frozen_B0_K_logits"
    assert first_manifest["model"] == second_manifest["model"]
    with np.load(first.checkpoint_path, allow_pickle=False) as left, np.load(
        second.checkpoint_path,
        allow_pickle=False,
    ) as right:
        assert set(left.files) == set(right.files)
        assert all(np.array_equal(left[name], right[name]) for name in left.files)


def test_selected_checkpoint_reproduces_manifest_validation_mse(tmp_path) -> None:
    train, validation = _fixture()
    result = train_p05_b2_posthoc_surrogate(
        tmp_path / "checkpoint",
        train=train,
        validation=validation,
        model_seed=42,
        b0_checkpoint_sha256=B0_CHECKPOINT_HASH,
        b0_run_artifact_semantic_sha256=B0_RUN_HASH,
    )
    model = load_p05_b2_surrogate_checkpoint(result.package_dir)
    weights = torch.tensor(
        [validation.weight_plan.record_weights[record_id] for record_id in validation.record_ids],
        dtype=torch.float32,
    )
    with torch.no_grad():
        mse = p05_b2_weighted_logit_mse(
            model(validation.features),
            validation.b0_logits,
            weights,
        )

    assert float(mse) == result.best_validation_mse
    assert model.cfg.num_fuzzy_features == 8
    assert model.cfg.num_membership_functions == 3
    assert model.cfg.num_rules == 10
    assert model.cfg.logit_scale == 1.0


def test_b2_rejects_validation_weights_that_are_not_equal_group_equal_window(
    tmp_path,
) -> None:
    train, validation = _fixture()
    all_one_weights = {record_id: 1.0 for record_id in validation.weight_plan.record_weights}
    payload = {
        "schema_version": 1,
        "paper_id": "P05",
        "dataset_id": 2,
        "role": "validation",
        "windows_per_record": 4,
        "formula": "1/(n_groups*n_windows_in_group)",
        "normalization": "mean_train_or_evaluation_window_weight_equals_one",
        "record_weights": [
            {"Id": record_id, "window_weight": 1.0}
            for record_id in sorted(all_one_weights)
        ],
    }
    digest = surrogate._sha256_bytes(surrogate._canonical_json_bytes(payload))
    bad_plan = replace(
        validation.weight_plan,
        record_weights=all_one_weights,
        sha256=digest,
    )

    with pytest.raises(ValueError, match="not equal-group/equal-window"):
        train_p05_b2_posthoc_surrogate(
            tmp_path / "bad-weights",
            train=train,
            validation=replace(validation, weight_plan=bad_plan),
            model_seed=42,
            b0_checkpoint_sha256=B0_CHECKPOINT_HASH,
            b0_run_artifact_semantic_sha256=B0_RUN_HASH,
        )

    assert not (tmp_path / "bad-weights").exists()


def test_b2_public_inputs_offer_no_label_target_or_selector() -> None:
    split_fields = set(inspect.signature(P05B2FrozenSplit).parameters)
    trainer_fields = set(inspect.signature(train_p05_b2_posthoc_surrogate).parameters)

    assert "labels" not in split_fields
    assert "labels" not in trainer_fields
    assert "target" not in trainer_fields
    assert "selection" not in trainer_fields


def test_b2_create_only_conflict_fails_before_retraining(tmp_path, monkeypatch) -> None:
    package = tmp_path / "conflict"
    first = _train(package)
    manifest_before = first.manifest_path.read_bytes()

    def fail_if_retrained(*args, **kwargs):
        del args, kwargs
        raise AssertionError("existing target must fail before input preparation")

    monkeypatch.setattr(surrogate, "_prepare_split", fail_if_retrained)
    with pytest.raises(FileExistsError, match="conflicts"):
        _train(package)

    assert first.manifest_path.read_bytes() == manifest_before


def test_b2_write_failure_leaves_no_partial_artifact(tmp_path, monkeypatch) -> None:
    package = tmp_path / "failed"

    def fail_write(path, arrays):
        del path, arrays
        raise RuntimeError("synthetic checkpoint write failure")

    monkeypatch.setattr(surrogate, "_write_checkpoint", fail_write)
    with pytest.raises(RuntimeError, match="synthetic checkpoint write failure"):
        _train(package)

    assert not package.exists()
    assert not list(tmp_path.glob(".failed.*.tmp"))
