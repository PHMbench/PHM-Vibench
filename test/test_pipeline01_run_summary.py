import json
from types import SimpleNamespace

import pandas as pd
import pytest

from src.Pipeline_01_Fault_Diagnosis import _P01DataProtocolHooks
from src.runtime.classification import _write_aggregate_outputs


def _config():
    return SimpleNamespace(
        pipeline="Pipeline_01_Fault_Diagnosis",
        environment=SimpleNamespace(seed=7, iterations=2),
        model=SimpleNamespace(type="Transformer", name="TSLTransformer"),
    )


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
        _config(),
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


def test_aggregate_outputs_reject_vacuous_runs(tmp_path):
    with pytest.raises(ValueError, match="at least one completed iteration"):
        _write_aggregate_outputs(tmp_path, None, [], [], _config())


def test_aggregate_csv_can_have_domain_rows_without_pseudoreplicating_summary(
    tmp_path,
):
    run_root = tmp_path / "run"
    last_iteration = run_root / "iter_1"
    last_iteration.mkdir(parents=True)
    domain_rows = [
        {"iteration": iteration, "target_domain": domain, "score": score}
        for iteration, values in enumerate(((0.2, 0.4, 0.3), (0.6, 0.8, 0.7)))
        for domain, score in zip((2, 3, "mean_2_3"), values)
    ]
    run_rows = [
        {"iteration": 0, "target_domain": "mean_2_3", "score": 0.3},
        {"iteration": 1, "target_domain": "mean_2_3", "score": 0.7},
    ]

    summary = _write_aggregate_outputs(
        run_root,
        last_iteration,
        domain_rows,
        [7, 8],
        _config(),
        summary_results=run_rows,
    )

    assert len(pd.read_csv(run_root / "all_results.csv")) == 6
    assert summary["iterations"] == 2
    assert summary["seeds"] == [7, 8]
    assert summary["metrics"]["score"]["count"] == 2
    assert summary["metrics"]["score"]["mean"] == pytest.approx(0.5)


def test_p01_hook_selects_only_the_cross_domain_mean_as_run_summary_row():
    context = SimpleNamespace(
        args_task=SimpleNamespace(
            grouped_evaluation=SimpleNamespace(enabled=True)
        ),
        result_rows=[
            {"target_environment": "CWRU_2_HP", "score": 0.2},
            {"target_environment": "CWRU_3_HP", "score": 0.4},
            {
                "target_environment": "mean_across_target_load_domains",
                "score": 0.3,
            },
        ],
    )

    row = _P01DataProtocolHooks().build_summary_row(context)
    assert row == {
        "target_environment": "mean_across_target_load_domains",
        "score": 0.3,
    }
