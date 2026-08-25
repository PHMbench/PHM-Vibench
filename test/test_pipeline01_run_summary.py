import json

import pandas as pd
import pytest

from src.runtime.classification import _write_aggregate_outputs


def test_aggregate_outputs_bind_all_iterations_and_seeds(tmp_path):
    run_root = tmp_path / "run"
    last_iteration = run_root / "iter_1"
    last_iteration.mkdir(parents=True)
    results = [{"test_acc": 0.5}, {"test_acc": 0.7}]

    summary = _write_aggregate_outputs(
        run_root,
        last_iteration,
        results,
        [7, 8],
    )

    assert pd.read_csv(run_root / "all_results.csv")["test_acc"].tolist() == [
        0.5,
        0.7,
    ]
    assert pd.read_csv(last_iteration / "all_results.csv")["test_acc"].tolist() == [
        0.5,
        0.7,
    ]
    stored = json.loads((run_root / "run_summary.json").read_text(encoding="utf-8"))
    assert stored == summary
    assert stored["seeds"] == [7, 8]
    assert stored["metrics"]["test_acc"]["mean"] == pytest.approx(0.6)
    assert "config_sha256" not in stored


def test_aggregate_outputs_reject_vacuous_runs(tmp_path):
    with pytest.raises(ValueError, match="at least one completed iteration"):
        _write_aggregate_outputs(tmp_path, None, [], [])
