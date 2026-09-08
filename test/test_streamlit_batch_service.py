from __future__ import annotations

from copy import deepcopy

import pytest
import yaml

from apps.streamlit.batch_service import BatchPlanError, plan_grid


@pytest.fixture
def base_config():
    return {
        "pipeline": "Pipeline_01_Fault_Diagnosis",
        "environment": {"seed": 0, "iterations": 3, "output_dir": "results/demo"},
        "data": {"batch_size": 16, "num_workers": 0},
        "model": {"type": "Backbone", "name": "B_04_Dlinear"},
        "task": {"lr": 0.001, "metrics": ["acc", "f1"]},
        "trainer": {
            "name": "Default_trainer",
            "device": "cpu",
            "devices": 1,
            "num_epochs": 1,
            "test_after_fit": True,
        },
    }


ALLOWED = {
    "task.lr",
    "data.batch_size",
    "trainer.num_epochs",
    "environment.seed",
    "environment.iterations",
}


def _resolved(trial):
    return yaml.safe_load(trial.config_yaml)


def test_two_dimensional_grid_reports_trials_and_real_fit_cost(base_config):
    plan = plan_grid(
        base_config,
        {"task.lr": [0.001, 0.0005, 0.0001], "data.batch_size": [16, 32]},
        allowed_paths=ALLOWED,
    )

    assert len(plan.trials) == 6
    assert plan.total_fits == 18
    assert plan.varying_paths == ("task.lr", "data.batch_size")
    assert [_resolved(trial)["task"]["lr"] for trial in plan.trials] == [
        0.001,
        0.001,
        0.0005,
        0.0005,
        0.0001,
        0.0001,
    ]
    assert [_resolved(trial)["data"]["batch_size"] for trial in plan.trials] == [
        16,
        32,
        16,
        32,
        16,
        32,
    ]
    assert all(trial.fit_count == 3 for trial in plan.trials)


def test_iterations_are_counted_per_trial_without_seed_repair(base_config):
    plan = plan_grid(
        base_config,
        {"environment.seed": [17, 18, 19]},
        allowed_paths=ALLOWED,
    )
    assert len(plan.trials) == 3
    assert plan.total_fits == 9
    assert [_resolved(trial)["environment"]["iterations"] for trial in plan.trials] == [3, 3, 3]
    assert [_resolved(trial)["environment"]["seed"] for trial in plan.trials] == [17, 18, 19]


def test_iterations_grid_changes_visible_fit_cost(base_config):
    plan = plan_grid(
        base_config,
        {"environment.iterations": [1, 2, 4]},
        allowed_paths=ALLOWED,
    )
    assert [trial.fit_count for trial in plan.trials] == [1, 2, 4]
    assert plan.total_fits == 7


def test_planning_does_not_mutate_base_config(base_config):
    before = deepcopy(base_config)
    plan_grid(base_config, {"trainer.num_epochs": [1, 2]}, allowed_paths=ALLOWED)
    assert base_config == before


@pytest.mark.parametrize(
    ("grid", "message"),
    [
        ({}, "At least one"),
        ({"unknown.path": [1]}, "not approved"),
        ({"task.lr": []}, "cannot be empty"),
        ({"task.lr": [0.001, 0.001]}, "duplicate"),
        ({"task.lr": "0.001"}, "finite sequence"),
    ],
)
def test_invalid_grid_fails_before_execution(base_config, grid, message):
    with pytest.raises(BatchPlanError, match=message):
        plan_grid(base_config, grid, allowed_paths=ALLOWED)


def test_trial_budget_fails_before_producing_an_oversized_plan(base_config):
    with pytest.raises(BatchPlanError, match="exceeding max_trials=4"):
        plan_grid(
            base_config,
            {"task.lr": [0.001, 0.0005, 0.0001], "data.batch_size": [16, 32]},
            allowed_paths=ALLOWED,
            max_trials=4,
        )


def test_fit_budget_accounts_for_backend_iterations(base_config):
    with pytest.raises(BatchPlanError, match="exceeding max_fits=5"):
        plan_grid(
            base_config,
            {"task.lr": [0.001, 0.0005]},
            allowed_paths=ALLOWED,
            max_fits=5,
        )


@pytest.mark.parametrize("budget", [0, -1, True, 1.5])
def test_budgets_must_be_positive_integers(base_config, budget):
    with pytest.raises(BatchPlanError, match="positive integer"):
        plan_grid(
            base_config,
            {"task.lr": [0.001]},
            allowed_paths=ALLOWED,
            max_trials=budget,
        )
