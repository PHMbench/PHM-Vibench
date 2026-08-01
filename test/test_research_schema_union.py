from copy import deepcopy
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config_schema import ExperimentConfig
from src.configs.config_utils import build_experiment_name, load_config


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


def test_metadata_path_populates_legacy_metadata_file_key() -> None:
    payload = _valid_grouped_fic_config()
    payload["data"].pop("metadata_file")
    payload["data"]["metadata_path"] = "/tmp/protocol/metadata.csv"

    config = ExperimentConfig.model_validate(payload)

    assert config.data.metadata_path == "/tmp/protocol/metadata.csv"
    assert config.data.metadata_file == "/tmp/protocol/metadata.csv"


def test_metadata_path_and_metadata_file_must_agree() -> None:
    payload = _valid_grouped_fic_config()
    payload["data"]["metadata_path"] = "/tmp/protocol/metadata.csv"

    with pytest.raises(ValidationError, match="must agree"):
        ExperimentConfig.model_validate(payload)


@pytest.mark.parametrize("stage", ["fit_validate_only", "fit_validate_test"])
def test_environment_stage_accepts_registered_values(stage: str) -> None:
    payload = _valid_grouped_fic_config()
    payload["environment"]["stage"] = stage

    config = ExperimentConfig.model_validate(payload)

    assert config.environment.stage == stage


def test_environment_stage_defaults_to_legacy_fit_validate_test() -> None:
    config = ExperimentConfig.model_validate(_valid_grouped_fic_config())

    assert config.environment.stage == "fit_validate_test"


def test_environment_stage_rejects_unknown_value() -> None:
    payload = _valid_grouped_fic_config()
    payload["environment"]["stage"] = "fit_only"

    with pytest.raises(ValidationError, match="environment.stage"):
        ExperimentConfig.model_validate(payload)


def test_preassigned_metadata_split_schema_is_valid() -> None:
    payload = _valid_grouped_fic_config()
    payload["data"]["split"] = {
        "strategy": "preassigned_metadata",
        "split_key": "Protocol_Split",
        "group_key": "Protocol_Group",
        "manifest_path": "outputs/splits/p05.json",
    }

    config = ExperimentConfig.model_validate(payload)

    assert config.data.split is not None
    assert config.data.split.strategy == "preassigned_metadata"
    assert config.data.split.split_key == "Protocol_Split"


def test_flat_preassigned_split_strategy_is_typed_for_runtime_compatibility() -> None:
    payload = _valid_grouped_fic_config()
    payload["data"].pop("split")
    payload["data"]["split_strategy"] = "preassigned_metadata"

    config = ExperimentConfig.model_validate(payload)

    assert config.data.split_strategy == "preassigned_metadata"


def test_preassigned_metadata_rejects_missing_split_key() -> None:
    payload = _valid_grouped_fic_config()
    payload["data"]["split"] = {
        "strategy": "preassigned_metadata",
        "group_key": "Protocol_Group",
        "manifest_path": "outputs/splits/p05.json",
    }

    with pytest.raises(ValidationError, match="split_key is required"):
        ExperimentConfig.model_validate(payload)


def test_preassigned_metadata_rejects_random_fractions() -> None:
    payload = _valid_grouped_fic_config()
    payload["data"]["split"] = {
        "strategy": "preassigned_metadata",
        "split_key": "Protocol_Split",
        "group_key": "Protocol_Group",
        "manifest_path": "outputs/splits/p05.json",
        "fractions": {"train": 0.6, "val": 0.2, "test": 0.2},
    }

    with pytest.raises(ValidationError, match="fractions must be omitted"):
        ExperimentConfig.model_validate(payload)


def test_split_strategy_rejects_unknown_value() -> None:
    payload = _valid_grouped_fic_config()
    payload["data"]["split"]["strategy"] = "random_rows"

    with pytest.raises(ValidationError, match="data.split.strategy"):
        ExperimentConfig.model_validate(payload)


def test_lightweight_loader_canonicalizes_metadata_path_and_legacy_stage() -> None:
    payload = _valid_grouped_fic_config()
    payload["data"]["metadata_path"] = "configs/protocol/metadata.csv"

    config = load_config(payload)
    expected = str(Path("configs/protocol/metadata.csv").resolve())

    assert config.data.metadata_path == expected
    assert config.data.metadata_file == expected
    assert config.environment.stage == "fit_validate_test"
    assert build_experiment_name(config).startswith("metadata.csv/M_CNN/")


@pytest.mark.parametrize(
    ("location", "value"),
    [
        ("environment.stage", "fit_only"),
        ("data.split_strategy", "random_rows"),
    ],
)
def test_lightweight_loader_rejects_unknown_protocol_values(
    location: str,
    value: str,
) -> None:
    payload = _valid_grouped_fic_config()
    section, key = location.split(".")
    payload[section][key] = value

    with pytest.raises(ValueError, match=location):
        load_config(payload)
