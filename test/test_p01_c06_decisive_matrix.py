from __future__ import annotations

import copy
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

from phmfactory.config import resolve_config
from scripts.analyze_p01_c06_results import (
    CONDITIONS,
    DOMAINS,
    EXPECTED_MODEL_CONDITIONS,
    SEEDS,
    C06ValidationError,
    build_contrasts,
    route_decision,
    stable_positive,
    validate_matrix_contract,
)
from src.Pipeline_01_Fault_Diagnosis import (
    build_p01_forward_compute_profile,
    build_p01_grouped_result_rows,
)
from src.model_factory import build_model
from src.runtime import ClassificationContext
from src.task_factory.Default_task import Default_task


MATRIX_PATH = Path("configs/experiments/p01/p01_c06_run_matrix.csv")
CONFIGS = {
    condition: Path(f"configs/experiments/p01/p01_c06_{condition.lower()}.yaml")
    for condition in CONDITIONS
}


def _namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(**{key: _namespace(item) for key, item in value.items()})
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


@lru_cache(maxsize=None)
def _resolved(condition: str) -> dict[str, Any]:
    return resolve_config(CONFIGS[condition]).data


def _metadata() -> dict[int, dict[str, Any]]:
    return {
        1: {"Name": "CWRU", "Label": 1, "Domain_id": 2},
        2: {"Name": "CWRU", "Label": 2, "Domain_id": 2},
        3: {"Name": "CWRU", "Label": 3, "Domain_id": 2},
    }


def _batch(seed: int = 607) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(6, 256, 2, generator=generator),
        "y": torch.tensor([1, 2, 3, 1, 2, 3], dtype=torch.long),
        "file_id": torch.tensor([1, 2, 3, 1, 2, 3], dtype=torch.long),
        "task_id": torch.zeros(6, dtype=torch.long),
    }


def _records() -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for domain in DOMAINS:
        for training_label, raw_label in enumerate((1, 2, 3)):
            for replicate in range(2):
                group_id = f"label{raw_label}-group{replicate}"
                for window in range(2):
                    logits = [-4.0, -4.0, -4.0]
                    logits[training_label] = 4.0
                    records.append(
                        {
                            "file_id": f"{domain}-{group_id}",
                            "physical_group_id": group_id,
                            "domain_id": domain,
                            "raw_label": raw_label,
                            "training_label": training_label,
                            "logits": logits,
                            "window": window,
                        }
                    )
    return records


def _task(condition: str) -> Default_task:
    config = copy.deepcopy(_resolved(condition))
    return Default_task(
        network=build_model(_namespace(config["model"]), metadata=None),
        args_data=_namespace(config["data"]),
        args_model=_namespace(config["model"]),
        args_task=_namespace(config["task"]),
        args_trainer=SimpleNamespace(gpus=0, device="cpu", num_epochs=10),
        args_environment=SimpleNamespace(seed=42),
        metadata=_metadata(),
    )


@lru_cache(maxsize=None)
def _profile(condition: str) -> dict[str, Any]:
    config = _resolved(condition)
    model = build_model(_namespace(config["model"]), metadata=None)
    return build_p01_forward_compute_profile(
        model,
        _namespace(config["model"]),
        _namespace(config["data"]),
        _namespace(config["task"]["grouped_evaluation"]),
        condition_id=condition,
    )


def _context(
    condition: str,
    task: Default_task,
    *,
    seed: int,
    run_id: str,
) -> ClassificationContext:
    config = copy.deepcopy(_resolved(condition))
    task.args_task.grouped_evaluation.required_windows_per_group_domain = 2
    task.args_task.grouped_evaluation.run_id = run_id
    task._grouped_test_records = _records()
    return ClassificationContext(
        args=SimpleNamespace(),
        configs=SimpleNamespace(),
        args_environment=SimpleNamespace(seed=seed),
        args_data=_namespace(config["data"]),
        args_model=_namespace(config["model"]),
        args_task=task.args_task,
        args_trainer=_namespace(config["trainer"]),
        iteration=0,
        path=Path("unused"),
        name="c06-test",
        model=task.network,
        task=task,
        trainer=SimpleNamespace(
            callbacks=[SimpleNamespace(best_model_path="model.ckpt")]
        ),
        result={"test_acc_CWRU": 1.0},
    )


def _run_id(condition: str, seed: int) -> str:
    number = 29 + SEEDS.index(seed) * len(CONDITIONS) + CONDITIONS.index(condition)
    return f"RUN-{number:04d}"


def test_c06_matrix_contract_is_exact_and_configs_resolve() -> None:
    rows = validate_matrix_contract(MATRIX_PATH, Path("."))
    assert len(rows) == 24
    for row in rows:
        condition = row["condition_id"]
        seed = int(row["seed"])
        config = resolve_config(
            row["config_path"],
            override_values=[
                f"environment.seed={seed}",
                f"environment.output_dir={row['output_dir']}",
                f"task.grouped_evaluation.run_id={row['run_id']}",
            ],
        ).data
        grouped = config["task"]["grouped_evaluation"]
        assert config["environment"]["seed"] == seed
        assert config["environment"]["iterations"] == 1
        assert config["environment"]["output_dir"] == row["output_dir"]
        assert config["model"]["condition"] == EXPECTED_MODEL_CONDITIONS[condition]
        assert grouped["goal_id"] == "C06"
        assert grouped["run_id"] == _run_id(condition, seed)
        assert grouped["run_role"] == "matrix_cell"
        assert grouped["predeclared_seeds"] == list(SEEDS)
        assert grouped["predeclared_target_domains"] == list(DOMAINS)
        assert grouped["source_validation_tuning_trials"] == 0
        assert config["trainer"]["num_epochs"] == 10
        assert config["trainer"]["early_stopping"] is False
        assert config["trainer"]["device"] == "cuda"
        assert config["trainer"]["gpus"] == 1
        control = config["task"].get("alignment_target_control")
        if condition == "C2":
            assert control["seed"] == 31042
        else:
            assert control is None


@pytest.mark.parametrize("condition", CONDITIONS)
def test_c06_first_batch_uses_every_required_branch(condition: str) -> None:
    task = _task(condition)
    task.on_train_start()
    output = task._shared_step(_batch(), "train", batch_index=0)
    output["train_total_loss"].backward()
    task.on_after_backward()
    assert task.view_gradient_summary()["status"] == "passed"
    objective = task.training_objective_summary()
    assert objective is not None
    assert objective["observed_batches"] == 1
    assert abs(objective["objective_reconstruction_residual"]) <= 1.0e-6


@pytest.mark.parametrize("condition", CONDITIONS)
def test_c06_all_seed_run_identities_build_rows(condition: str) -> None:
    task = _task(condition)
    task.on_train_start()
    output = task._shared_step(_batch(613), "train", batch_index=0)
    output["train_total_loss"].backward()
    task.on_after_backward()
    for seed in SEEDS:
        rows = build_p01_grouped_result_rows(
            _context(condition, task, seed=seed, run_id=_run_id(condition, seed)),
            forward_compute_profile=_profile(condition),
            training_objective_summary=task.training_objective_summary(),
            view_gradient_summary=task.view_gradient_summary(),
        )
        assert [row["target_domain"] for row in rows] == [2, 3, "mean_2_3"]
        assert {row["run_id"] for row in rows} == {_run_id(condition, seed)}
        assert {row["seed"] for row in rows} == {seed}
        assert {row["run_scope"] for row in rows} == {
            "C06_three_seed_two_environment_decisive_pilot"
        }
        assert all("not mechanism evidence" in row["scientific_boundary"] for row in rows)


def _contrast_rows(value: float) -> list[dict[str, float | int]]:
    return [
        {
            "seed": seed,
            "target_domain": domain,
            **{name: value for name in (
                "alignment_gain",
                "multimodal_synergy",
                "m5_minus_c2",
                "m5_minus_c3",
                "c2_gain",
                "c3_gain",
            )},
        }
        for seed in SEEDS
        for domain in DOMAINS
    ]


def test_c06_stability_rule_treats_ties_as_nonpositive() -> None:
    positive = _contrast_rows(0.1)
    tied = _contrast_rows(0.0)
    assert stable_positive(positive, "alignment_gain") is True
    assert stable_positive(tied, "alignment_gain") is False


def test_c06_route_requires_gain_synergy_and_nonreproducing_controls() -> None:
    rows = _contrast_rows(0.1)
    for row in rows:
        row["c2_gain"] = 0.0
        row["c3_gain"] = 0.0
    assert route_decision(rows).startswith("admit_C07_C09")
    for row in rows:
        row["c3_gain"] = 0.1
    assert "negative_control_reproduces" in route_decision(rows)


def test_c06_contrasts_reject_c1_m4_drift_instead_of_winner_selection() -> None:
    domain_rows: dict[tuple[str, int, int], dict[str, str]] = {}
    for condition in CONDITIONS:
        for seed in SEEDS:
            for domain in DOMAINS:
                domain_rows[(condition, seed, domain)] = {
                    "primary_metric_value": "0.5",
                    "group_predictions_json": "[]",
                    "trainable_parameters": str(
                        49_411 if condition in {"M4", "C1"} else 47_235
                    ),
                    "learned_forward_supported_flops": str(
                        46_004_224 if condition in {"M4", "C1"} else 45_991_424
                    ),
                }
    domain_rows[("C1", 42, 2)]["primary_metric_value"] = "0.6"
    with pytest.raises(C06ValidationError, match="C1 is not an exact M4 identity"):
        build_contrasts(domain_rows)
