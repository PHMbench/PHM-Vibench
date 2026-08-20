from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from src.runtime.classification import _result_row
from src.utils.run_summary import (
    build_run_summary,
    normalize_metric_result,
    resolved_config_sha256,
    write_run_summary,
)


def _config():
    return SimpleNamespace(
        pipeline="Pipeline_01_Fault_Diagnosis",
        environment=SimpleNamespace(seed=42, iterations=2),
        model=SimpleNamespace(type="Transformer", name="TSLTransformer"),
    )


def test_summary_records_complete_seed_statistics():
    results = [
        {"test_acc": 0.5, "test_loss": 2.0},
        {"test_acc": 0.7, "test_loss": 4.0},
    ]
    summary = build_run_summary(results, seeds=[42, 43], config=_config())

    assert summary["config_sha256"] == resolved_config_sha256(_config())
    assert summary["iterations"] == 2
    assert summary["seeds"] == [42, 43]
    assert set(summary["metrics"]) == {"test_acc", "test_loss"}
    assert summary["metrics"]["test_acc"]["count"] == 2
    assert summary["metrics"]["test_acc"]["mean"] == pytest.approx(0.6)
    assert summary["metrics"]["test_acc"]["sample_std"] == pytest.approx(2**0.5 / 10)


def test_single_run_uses_null_std_and_writes_strict_json(tmp_path):
    output = tmp_path / "run_summary.json"
    write_run_summary(output, [{"test_acc": 0.5}], [42], _config())
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["iterations"] == 1
    assert payload["metrics"]["test_acc"]["count"] == 1
    assert payload["metrics"]["test_acc"]["sample_std"] is None
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_summary_rejects_missing_seed_and_nonfinite_metrics():
    with pytest.raises(ValueError, match="one seed"):
        build_run_summary([{"value": 1.0}], [], _config())
    with pytest.raises(ValueError, match="not finite"):
        build_run_summary([{"value": float("nan")}], [42], _config())


def test_summary_rejects_metric_key_drift_across_seeds():
    with pytest.raises(
        ValueError,
        match=r"same metric set.*missing=\['test_f1'\]",
    ):
        build_run_summary(
            [
                {"test_acc": 0.5, "test_f1": 0.4},
                {"test_acc": 0.7},
            ],
            seeds=[42, 43],
            config=_config(),
        )

    with pytest.raises(
        ValueError,
        match=r"same metric set.*unexpected=\['test_f1'\]",
    ):
        build_run_summary(
            [
                {"test_acc": 0.5},
                {"test_acc": 0.7, "test_f1": 0.6},
            ],
            seeds=[42, 43],
            config=_config(),
        )


def test_metric_result_rejects_empty_nonnumeric_boolean_and_nonscalar_values():
    with pytest.raises(ValueError, match="at least one metric"):
        normalize_metric_result({})
    with pytest.raises(TypeError, match="scalar real number"):
        normalize_metric_result({"test_acc": "0.5"})
    with pytest.raises(TypeError, match="not boolean"):
        normalize_metric_result({"test_acc": True})

    class VectorLike:
        def item(self):
            raise ValueError("more than one element")

    with pytest.raises(TypeError, match="scalar numeric value"):
        normalize_metric_result({"test_acc": VectorLike()})


def test_summary_rejects_noninteger_seed_values():
    with pytest.raises(TypeError, match="seed 0 must be an integer"):
        build_run_summary([{"test_acc": 0.5}], [42.5], _config())
    with pytest.raises(TypeError, match="seed 0 must be an integer"):
        build_run_summary([{"test_acc": 0.5}], [True], _config())


def test_trainer_test_requires_exactly_one_explicit_population():
    assert _result_row([{"test_acc": 0.5}]) == {"test_acc": 0.5}

    with pytest.raises(RuntimeError, match="exactly one metric mapping"):
        _result_row([])
    with pytest.raises(RuntimeError, match="exactly one metric mapping"):
        _result_row([{"test_acc": 0.5}, {"test_acc": 0.7}])
    with pytest.raises(RuntimeError, match="result 0 must be a metric mapping"):
        _result_row([[0.5]])
    with pytest.raises(TypeError, match="scalar real number"):
        _result_row([{"test_acc": "0.5"}])
