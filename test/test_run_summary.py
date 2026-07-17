import json
from types import SimpleNamespace

import pytest

from src.utils.run_summary import (
    build_run_summary,
    resolved_config_sha256,
    write_run_summary,
)


def _config():
    return SimpleNamespace(
        pipeline="Pipeline_01_default",
        environment=SimpleNamespace(seed=42, iterations=2),
        model=SimpleNamespace(type="Transformer", name="TSLTransformer"),
    )


def test_summary_records_hash_seeds_and_sample_statistics():
    results = [
        {"test_acc": 0.5, "test_loss": 2.0, "label": "ignored"},
        {"test_acc": 0.7, "test_loss": 4.0, "label": "ignored"},
    ]
    summary = build_run_summary(results, seeds=[42, 43], config=_config())

    assert summary["config_sha256"] == resolved_config_sha256(_config())
    assert summary["seeds"] == [42, 43]
    assert summary["metrics"]["test_acc"]["mean"] == pytest.approx(0.6)
    assert summary["metrics"]["test_acc"]["sample_std"] == pytest.approx(2**0.5 / 10)
    assert "label" not in summary["metrics"]


def test_single_run_uses_null_std_and_writes_strict_json(tmp_path):
    output = tmp_path / "run_summary.json"
    write_run_summary(output, [{"test_acc": 0.5}], [42], _config())
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["iterations"] == 1
    assert payload["metrics"]["test_acc"]["sample_std"] is None
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_summary_rejects_missing_seed_and_nonfinite_metrics():
    with pytest.raises(ValueError, match="one seed"):
        build_run_summary([{"value": 1.0}], [], _config())
    with pytest.raises(ValueError, match="non-finite"):
        build_run_summary([{"value": float("nan")}], [42], _config())
