import pandas as pd
import pytest

from src.data_factory.p05_weighting import ExpectedRole, build_weight_plan


def _frame(records):
    return pd.DataFrame(records)


def test_cwru_training_equalizes_class_totals_and_normalizes_mean():
    frame = _frame(
        [
            {"Id": 1, "Dataset_id": 1, "Label": 0, "Protocol_Group": "CWRU/a", "Protocol_Split": "train"},
            {"Id": 2, "Dataset_id": 1, "Label": 1, "Protocol_Group": "CWRU/b", "Protocol_Split": "train"},
            {"Id": 3, "Dataset_id": 1, "Label": 1, "Protocol_Group": "CWRU/c", "Protocol_Split": "train"},
            {"Id": 4, "Dataset_id": 1, "Label": 2, "Protocol_Group": "CWRU/d", "Protocol_Split": "train"},
            {"Id": 5, "Dataset_id": 1, "Label": 3, "Protocol_Group": "CWRU/e", "Protocol_Split": "train"},
        ]
    )
    plan = build_weight_plan(
        frame,
        dataset_id=1,
        role="train",
        expected=ExpectedRole(5, 5, {0: 1, 1: 2, 2: 1, 3: 1}, 16),
    )

    assert plan.record_weights[1] == pytest.approx(1.25)
    assert plan.record_weights[2] == pytest.approx(0.625)
    assert plan.record_weights[3] == pytest.approx(0.625)
    assert plan.record_weights[4] == pytest.approx(1.25)
    assert plan.record_weights[5] == pytest.approx(1.25)
    assert sum(plan.record_weights.values()) / 5 == pytest.approx(1.0)
    assert len(plan.sha256) == 64


def test_xjtu_training_equalizes_every_bearing_class_cell():
    records = []
    record_id = 0
    for group_index in range(5):
        for label in (0, 1):
            repeats = 2 if group_index == 0 and label == 0 else 1
            for _ in range(repeats):
                record_id += 1
                records.append(
                    {
                        "Id": record_id,
                        "Dataset_id": 2,
                        "Label": label,
                        "Protocol_Group": f"XJTU/g{group_index}",
                        "Protocol_Split": "train",
                    }
                )
    frame = _frame(records)
    class_counts = frame["Label"].value_counts().sort_index().to_dict()
    plan = build_weight_plan(
        frame,
        dataset_id=2,
        role="train",
        expected=ExpectedRole(len(frame), 5, class_counts, 4),
    )

    totals = {}
    for row in records:
        key = (row["Protocol_Group"], row["Label"])
        totals[key] = totals.get(key, 0.0) + plan.record_weights[row["Id"]] * 4
    assert len(totals) == 10
    assert max(totals.values()) == pytest.approx(min(totals.values()))


def test_validation_weights_are_label_free_and_equalize_groups():
    records = [
        {"Id": 1, "Dataset_id": 2, "Label": 0, "Protocol_Group": "g1", "Protocol_Split": "validation"},
        {"Id": 2, "Dataset_id": 2, "Label": 1, "Protocol_Group": "g1", "Protocol_Split": "validation"},
        {"Id": 3, "Dataset_id": 2, "Label": 0, "Protocol_Group": "g2", "Protocol_Split": "validation"},
    ]
    expected = ExpectedRole(3, 2, {0: 2, 1: 1}, 4)
    first = build_weight_plan(
        _frame(records), dataset_id=2, role="validation", expected=expected
    )
    permuted = [dict(row, Label=1 - row["Label"]) for row in records]
    second = build_weight_plan(
        _frame(permuted),
        dataset_id=2,
        role="validation",
        expected=ExpectedRole(3, 2, {0: 1, 1: 2}, 4),
    )

    assert first.record_weights == second.record_weights
    assert (first.record_weights[1] + first.record_weights[2]) * 4 == pytest.approx(
        first.record_weights[3] * 4
    )


def test_weight_plan_fails_on_registered_count_drift():
    frame = _frame(
        [
            {"Id": 1, "Dataset_id": 1, "Label": 0, "Protocol_Group": "g", "Protocol_Split": "train"},
        ]
    )
    with pytest.raises(ValueError, match="row count mismatch"):
        build_weight_plan(
            frame,
            dataset_id=1,
            role="train",
            expected=ExpectedRole(2, 2, {0: 2}, 16),
        )


def test_weight_plan_fails_when_xjtu_cell_is_missing():
    records = [
        {
            "Id": index + 1,
            "Dataset_id": 2,
            "Label": 0,
            "Protocol_Group": f"g{index}",
            "Protocol_Split": "train",
        }
        for index in range(5)
    ]
    with pytest.raises(ValueError, match="five-bearing by two-class"):
        build_weight_plan(
            _frame(records),
            dataset_id=2,
            role="train",
            expected=ExpectedRole(5, 5, {0: 5}, 4),
        )
