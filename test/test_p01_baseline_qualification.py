"""Condition-wise qualification must not accept pooled accuracy or class collapse."""
import numpy as np
import pytest

from experiments.p01.baseline_qualification import qualify_source_prediction_arrays


def predictions(rows):
    """Rows are condition, physical group, acquisition, label, prediction."""
    conditions, groups, acquisitions, labels, predicted = zip(*rows)
    probabilities = np.eye(2)[predicted, :]
    return dict(domains=np.asarray(conditions), group_ids=np.asarray(groups),
                acquisition_ids=np.asarray(acquisitions), labels=np.asarray(labels),
                candidate_probs=probabilities, window_ids=np.zeros(len(rows), dtype=int))


def test_condition_accuracy_cannot_be_hidden_by_many_easy_acquisitions():
    rows = [("easy", f"g{i}", f"e{i}", i % 2, i % 2) for i in range(20)]
    rows += [("hard", f"g{i}", f"h{i}", i, 0) for i in range(2)]
    result = qualify_source_prediction_arrays(predictions(rows), ["easy", "hard"], 2)
    assert not result["passed"]
    assert [row["accuracy"] for row in result["source_conditions"]] == pytest.approx([1., .5])
    assert result["source_conditions"][1]["class_recall"] == [1., 0.]


def test_eighty_percent_with_a_never_recognized_class_is_unqualified():
    rows = [("c", f"g{i}", f"a{i}", int(i == 4), 0) for i in range(5)]
    result = qualify_source_prediction_arrays(predictions(rows), ["c"], 2)
    row = result["source_conditions"][0]
    assert row["accuracy"] == pytest.approx(.8)
    assert row["class_support"] == [4, 1]
    assert row["predicted_class_support"] == [5, 0]
    assert row["reasons"] == ["zero_class_recall"]
    assert not result["passed"]


def test_absent_true_class_cannot_qualify_at_perfect_accuracy():
    result = qualify_source_prediction_arrays(predictions([("c", "g", "a", 0, 0)]), ["c"], 2)
    assert "missing_true_class_support" in result["source_conditions"][0]["reasons"]
    assert not result["passed"]


def test_group_and_acquisition_balancing_and_window_probability_aggregation():
    rows = [("c", "g0", f"a{i}", 0, 0) for i in range(9)] + [("c", "g1", "b", 1, 0)]
    result = qualify_source_prediction_arrays(predictions(rows), ["c"], 2)
    row = result["source_conditions"][0]
    assert row["accuracy"] == pytest.approx(.5)
    np.testing.assert_allclose(row["confusion_matrix"], [[.5, 0.], [.5, 0.]])
    assert row["weighted_class_support"] == pytest.approx([.5, .5])
    arrays = predictions([("c", "g0", "a", 0, 0), ("c", "g0", "a", 0, 1),
                          ("c", "g1", "b", 1, 1)])
    arrays["window_ids"] = np.asarray([0, 1, 0])
    arrays["candidate_probs"] = np.asarray([[.99, .01], [.49, .51], [.1, .9]])
    qualified = qualify_source_prediction_arrays(arrays, ["c"], 2)
    assert qualified["passed"]
    assert qualified["source_conditions"][0]["acquisitions"] == 2


@pytest.mark.parametrize("expected", [[], ["c", "missing"], ["c", "c"], ["other"]])
def test_empty_missing_extra_and_duplicate_source_conditions_rejected(expected):
    with pytest.raises(ValueError, match="source-condition"):
        qualify_source_prediction_arrays(predictions([("c", "g", "a", 0, 0)]), expected, 2)


def test_empty_predictions_never_pass_vacuously():
    arrays = dict(domains=np.asarray([]), group_ids=np.asarray([]), acquisition_ids=np.asarray([]),
                  labels=np.asarray([]), candidate_probs=np.zeros((0, 2)))
    with pytest.raises(ValueError, match="nonempty"):
        qualify_source_prediction_arrays(arrays, ["c"], 2)


def test_requested_predictor_is_never_replaced_with_another():
    arrays = predictions([("c", "g0", "a", 0, 0), ("c", "g1", "b", 1, 1)])
    arrays["raw_probs"] = np.asarray([[1., 0.], [1., 0.]])
    assert qualify_source_prediction_arrays(arrays, ["c"], 2)["passed"]
    assert not qualify_source_prediction_arrays(arrays, ["c"], 2, predictor="raw")["passed"]
    del arrays["raw_probs"]
    with pytest.raises(ValueError, match="lack"):
        qualify_source_prediction_arrays(arrays, ["c"], 2, predictor="raw")


def test_malformed_probabilities_and_duplicate_windows_rejected():
    arrays = predictions([("c", "g0", "a", 0, 0), ("c", "g1", "b", 1, 1)])
    arrays["candidate_probs"][0] = [.9, .9]
    with pytest.raises(ValueError, match="normalized"):
        qualify_source_prediction_arrays(arrays, ["c"], 2)
    repeated = predictions([("c", "g0", "a", 0, 0), ("c", "g0", "a", 0, 0)])
    with pytest.raises(ValueError, match="duplicated"):
        qualify_source_prediction_arrays(repeated, ["c"], 2)
