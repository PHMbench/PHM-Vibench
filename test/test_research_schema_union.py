from copy import deepcopy

import pytest
from pydantic import ValidationError

from src.config_schema import ExperimentConfig


def _valid_grouped_fic_config() -> dict:
    return {
        "pipeline": "Pipeline_01_Fault_Diagnosis",
        "environment": {
            "project": "schema-union",
            "output_dir": "outputs/schema-union",
        },
        "data": {
            "data_dir": "data",
            "metadata_file": "metadata.csv",
            "batch_size": 4,
            "split": {
                "strategy": "grouped_metadata",
                "group_key": "Bearing_id",
                "stratify_key": "Label",
                "seed": 17,
                "test_policy": "task_defined",
                "fractions": {"train": 0.75, "val": 0.25},
                "manifest_path": "outputs/splits/schema-union.json",
            },
        },
        "model": {"type": "Default", "name": "CNN"},
        "task": {
            "type": "DG",
            "name": "classification",
            "loss": "CE",
            "gradient_constraint": {"name": "fic", "epsilon": 2.0},
        },
        "trainer": {"name": "Default_trainer", "num_epochs": 1},
    }


def test_grouped_split_and_fic_are_valid_together() -> None:
    config = ExperimentConfig.model_validate(_valid_grouped_fic_config())

    assert config.data.split is not None
    assert config.data.split.strategy == "grouped_metadata"
    assert config.task.gradient_constraint is not None
    assert config.task.gradient_constraint.name == "fic"


@pytest.mark.parametrize("task_type", ["FS", "GFS"])
def test_grouped_split_rejects_episode_tasks(task_type: str) -> None:
    payload = _valid_grouped_fic_config()
    payload["task"]["type"] = task_type

    with pytest.raises(ValidationError, match="episode-safe"):
        ExperimentConfig.model_validate(payload)


def test_grouped_dg_rejects_partition_policy() -> None:
    payload = _valid_grouped_fic_config()
    payload["data"]["split"]["test_policy"] = "partition"
    payload["data"]["split"]["fractions"] = {
        "train": 0.6,
        "val": 0.2,
        "test": 0.2,
    }

    with pytest.raises(ValidationError, match="requires test_policy=task_defined"):
        ExperimentConfig.model_validate(payload)


def test_fic_rejects_non_ce_loss() -> None:
    payload = _valid_grouped_fic_config()
    payload["task"]["loss"] = "MSE"

    with pytest.raises(ValidationError, match="requires task.loss=CE"):
        ExperimentConfig.model_validate(payload)


def test_fic_rejects_unknown_constraint() -> None:
    payload = deepcopy(_valid_grouped_fic_config())
    payload["task"]["gradient_constraint"]["name"] = "unknown"

    with pytest.raises(ValidationError, match="gradient_constraint.name"):
        ExperimentConfig.model_validate(payload)
