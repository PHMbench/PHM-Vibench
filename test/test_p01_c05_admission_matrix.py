from __future__ import annotations

import copy
import json
from functools import lru_cache
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch

from phmfactory.config import resolve_config
from src.Pipeline_01_Fault_Diagnosis import (
    build_p01_forward_compute_profile,
    build_p01_grouped_result_rows,
)
from src.model_factory import build_model
from src.runtime import ClassificationContext
from src.task_factory.Default_task import Default_task


C05_CONFIGS = {
    condition_id: Path(f"configs/experiments/p01/p01_c05_{condition_id.lower()}.yaml")
    for condition_id in ("M1", "M2", "M3", "M4", "M5", "C1", "C2", "C3")
}
MODEL_CONDITIONS = {
    "M1": "M1",
    "M2": "M2",
    "M3": "M3",
    "M4": "M4",
    "M5": "M5",
    "C1": "M4",
    "C2": "M5",
    "C3": "C3",
}
RUN_IDS = {
    condition_id: f"RUN-{run_number:04d}"
    for condition_id, run_number in zip(MODEL_CONDITIONS, range(19, 27))
}
EXPECTED_PARAMETERS = {
    "M1": 19_587,
    "M2": 27_907,
    "M3": 47_235,
    "M4": 49_411,
    "M5": 47_235,
    "C1": 49_411,
    "C2": 47_235,
    "C3": 55_555,
}
EXPECTED_SUPPORTED_FLOPS = {
    "M1": 22_823_296,
    "M2": 23_168_512,
    "M3": 45_991_424,
    "M4": 46_004_224,
    "M5": 45_991_424,
    "C1": 46_004_224,
    "C2": 45_991_424,
    "C3": 46_336_640,
}
EXPECTED_GRADIENT_GROUPS = {
    "M1": {"waveform_1d", "classifier_head"},
    "M2": {"time_frequency_2d", "classifier_head"},
    "M3": {"waveform_1d", "time_frequency_2d", "classifier_head"},
    "M4": {
        "waveform_1d",
        "time_frequency_2d",
        "fusion_attention",
        "classifier_head",
    },
    "M5": {"waveform_1d", "time_frequency_2d", "classifier_head"},
    "C1": {
        "waveform_1d",
        "time_frequency_2d",
        "fusion_attention",
        "classifier_head",
    },
    "C2": {"waveform_1d", "time_frequency_2d", "classifier_head"},
    "C3": {
        "time_frequency_2d",
        "duplicate_time_frequency_2d",
        "classifier_head",
    },
}


def _namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(
            **{key: _namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


@lru_cache(maxsize=None)
def _resolved(condition_id: str) -> dict[str, Any]:
    return resolve_config(C05_CONFIGS[condition_id]).data


def _metadata() -> dict[int, dict[str, Any]]:
    return {
        1: {"Name": "CWRU", "Label": 1, "Domain_id": 2},
        2: {"Name": "CWRU", "Label": 2, "Domain_id": 2},
        3: {"Name": "CWRU", "Label": 3, "Domain_id": 2},
    }


def _task(condition_id: str) -> Default_task:
    config = copy.deepcopy(_resolved(condition_id))
    return Default_task(
        network=build_model(_namespace(config["model"]), metadata=None),
        args_data=_namespace(config["data"]),
        args_model=_namespace(config["model"]),
        args_task=_namespace(config["task"]),
        args_trainer=SimpleNamespace(gpus=0, device="cpu", num_epochs=10),
        args_environment=SimpleNamespace(seed=31),
        metadata=_metadata(),
    )


def _batch(seed: int = 211) -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    return {
        "x": torch.randn(6, 256, 2, generator=generator),
        "y": torch.tensor([1, 2, 3, 1, 2, 3], dtype=torch.long),
        "file_id": torch.tensor([1, 2, 3, 1, 2, 3], dtype=torch.long),
        "task_id": torch.zeros(6, dtype=torch.long),
    }


@lru_cache(maxsize=None)
def _profile(condition_id: str) -> dict[str, Any]:
    config = _resolved(condition_id)
    model = build_model(_namespace(config["model"]), metadata=None)
    return build_p01_forward_compute_profile(
        model,
        _namespace(config["model"]),
        _namespace(config["data"]),
        _namespace(config["task"]["grouped_evaluation"]),
        condition_id=condition_id,
    )


def _records() -> list[dict[str, Any]]:
    records = []
    for domain_id in (2, 3):
        for training_label, raw_label in enumerate((1, 2, 3)):
            for replicate in range(2):
                group_id = f"label{raw_label}-group{replicate}"
                for window in range(2):
                    logits = [-4.0, -4.0, -4.0]
                    logits[training_label] = 4.0
                    records.append(
                        {
                            "file_id": f"{domain_id}-{group_id}",
                            "physical_group_id": group_id,
                            "domain_id": domain_id,
                            "raw_label": raw_label,
                            "training_label": training_label,
                            "logits": logits,
                            "window": window,
                        }
                    )
    return records


def _context(condition_id: str, task: Default_task) -> ClassificationContext:
    config = copy.deepcopy(_resolved(condition_id))
    task.args_task.grouped_evaluation.required_windows_per_group_domain = 2
    task._grouped_test_records = _records()
    return ClassificationContext(
        args=SimpleNamespace(),
        configs=SimpleNamespace(),
        args_environment=_namespace(config["environment"]),
        args_data=_namespace(config["data"]),
        args_model=_namespace(config["model"]),
        args_task=task.args_task,
        args_trainer=_namespace(config["trainer"]),
        iteration=0,
        path=Path("unused"),
        name="c05-test",
        model=task.network,
        task=task,
        trainer=SimpleNamespace(
            callbacks=[SimpleNamespace(best_model_path="model.ckpt")]
        ),
        result={"test_acc_CWRU": 1.0},
    )


@pytest.mark.parametrize("condition_id", tuple(MODEL_CONDITIONS))
def test_c05_configs_freeze_one_comparable_matrix(condition_id: str) -> None:
    config = _resolved(condition_id)
    grouped = config["task"]["grouped_evaluation"]

    assert config["model"]["condition"] == MODEL_CONDITIONS[condition_id]
    assert grouped["goal_id"] == "C05"
    assert grouped["run_id"] == RUN_IDS[condition_id]
    assert grouped["run_role"] == "matrix_cell"
    assert grouped.get("reproduction_of", "") == ""
    assert grouped["condition_id"] == condition_id
    assert config["environment"]["seed"] == 31
    assert config["trainer"]["num_epochs"] == 10
    assert config["trainer"]["early_stopping"] is False
    assert config["trainer"]["device"] == "cuda"
    assert config["trainer"]["gpus"] == 1
    assert config["task"]["source_domain_id"] == [0, 1]
    assert config["task"]["target_domain_id"] == [2, 3]
    assert config["task"]["grouped_split"]["admitted_labels"] == [1, 2, 3]
    assert config["task"]["label_contract"]["raw_labels"] == [1, 2, 3]
    assert config["data"]["normalization"] == "none"
    assert config["data"]["window_size"] == 4096
    assert config["task"]["optimizer"] == "adam"
    assert config["task"]["lr"] == pytest.approx(0.001)
    assert config["task"]["weight_decay"] == pytest.approx(0.0001)
    assert config["task"]["scheduler"] is None
    assert grouped["source_validation_tuning_trials"] == 0
    assert grouped["alignment_coefficient_source_validation_trials"] == 0
    assert grouped["required_cuda_visible_devices"] == "3"
    assert grouped["forward_compute"]["reference_condition"] == "M5"
    assert grouped["forward_compute"]["comparison_condition"] == "M4"

    control = config["task"].get("alignment_target_control")
    if condition_id == "C2":
        assert control == {
            "mode": "seeded_sattolo_derangement_after_batching",
            "seed": 31042,
            "seed_key": "base_seed_plus_epoch_times_1000003_plus_batch_index",
        }
    else:
        assert control is None
    assert len({str(_resolved(key)["environment"]["output_dir"]) for key in MODEL_CONDITIONS}) == 8


def test_c05_c3_selection_is_identical_to_the_c04_frozen_control() -> None:
    c04 = resolve_config(Path("configs/experiments/p01/p01_c04_c3.yaml")).data
    assert _resolved("C3")["task"]["grouped_evaluation"]["c3_selection"] == (
        c04["task"]["grouped_evaluation"]["c3_selection"]
    )


@pytest.mark.parametrize("condition_id", tuple(MODEL_CONDITIONS))
def test_c05_first_training_batch_activates_every_required_view(
    condition_id: str,
) -> None:
    task = _task(condition_id)
    task.on_train_start()
    metrics = task._shared_step(_batch(), "train", batch_index=0)
    metrics["train_total_loss"].backward()
    task.on_after_backward()

    gradient_summary = task.view_gradient_summary()
    objective_summary = task.training_objective_summary()
    assert gradient_summary is not None
    assert gradient_summary["status"] == "passed"
    assert gradient_summary["condition_id"] == condition_id
    assert set(gradient_summary["gradient_norms"]) == EXPECTED_GRADIENT_GROUPS[
        condition_id
    ]
    assert all(value > 1.0e-12 for value in gradient_summary["gradient_norms"].values())
    assert objective_summary is not None
    assert objective_summary["observed_batches"] == 1
    assert abs(objective_summary["objective_reconstruction_residual"]) <= 1.0e-6


@pytest.mark.parametrize("condition_id", tuple(MODEL_CONDITIONS))
def test_c05_rows_bind_matrix_identity_gradient_and_compute(
    condition_id: str,
) -> None:
    task = _task(condition_id)
    task.on_train_start()
    metrics = task._shared_step(_batch(223), "train", batch_index=0)
    metrics["train_total_loss"].backward()
    task.on_after_backward()
    objective_summary = task.training_objective_summary()
    gradient_summary = task.view_gradient_summary()
    rows = build_p01_grouped_result_rows(
        _context(condition_id, task),
        forward_compute_profile=_profile(condition_id),
        training_objective_summary=objective_summary,
        view_gradient_summary=gradient_summary,
    )

    assert [row["target_domain"] for row in rows] == [2, 3, "mean_2_3"]
    for row in rows:
        assert row["run_id"] == RUN_IDS[condition_id]
        assert row["run_role"] == "matrix_cell"
        assert row["reproduction_of"] == ""
        assert row["condition_id"] == condition_id
        assert row["model_condition"] == MODEL_CONDITIONS[condition_id]
        assert row["training_epochs"] == 10
        assert json.loads(row["training_objective_summary_json"]) == objective_summary
        assert json.loads(row["view_gradient_summary_json"]) == gradient_summary
        comparison = json.loads(row["capacity_compute_comparison_json"])
        assert comparison["observed"]["trainable_parameters"] == (
            EXPECTED_PARAMETERS[condition_id]
        )
        assert comparison["observed"]["learned_forward_supported_flops"] == (
            EXPECTED_SUPPORTED_FLOPS[condition_id]
        )


@pytest.mark.parametrize(
    "condition_id,run_id,reproduction_of",
    (("M1", "RUN-0027", "RUN-0019"), ("C2", "RUN-0028", "RUN-0025")),
)
def test_c05_predeclared_fresh_process_identities_are_admitted(
    condition_id: str, run_id: str, reproduction_of: str
) -> None:
    task = _task(condition_id)
    task.args_task.grouped_evaluation.run_id = run_id
    task.args_task.grouped_evaluation.run_role = "fresh_process_reproduction"
    task.args_task.grouped_evaluation.reproduction_of = reproduction_of
    task.on_train_start()
    metrics = task._shared_step(_batch(227), "train", batch_index=0)
    metrics["train_total_loss"].backward()
    task.on_after_backward()
    rows = build_p01_grouped_result_rows(
        _context(condition_id, task),
        forward_compute_profile=_profile(condition_id),
        training_objective_summary=task.training_objective_summary(),
        view_gradient_summary=task.view_gradient_summary(),
    )
    assert {(row["run_id"], row["run_role"], row["reproduction_of"]) for row in rows} == {
        (run_id, "fresh_process_reproduction", reproduction_of)
    }


def test_c05_unregistered_run_identity_fails_closed() -> None:
    task = _task("M1")
    task.args_task.grouped_evaluation.run_id = "RUN-9999"
    with pytest.raises(ValueError, match="run identity mismatch"):
        build_p01_grouped_result_rows(_context("M1", task))
