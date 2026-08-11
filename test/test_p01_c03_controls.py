from __future__ import annotations

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


C03_CONFIGS = {
    "M3": Path("configs/experiments/p01/p01_c03_m3.yaml"),
    "M4": Path("configs/experiments/p01/p01_c03_m4.yaml"),
    "C1": Path("configs/experiments/p01/p01_c03_c1.yaml"),
}
MODEL_CONDITIONS = {"M3": "M3", "M4": "M4", "C1": "M4"}
EXPECTED_PARAMETERS = {"M3": 47_235, "M4": 49_411, "C1": 49_411}
EXPECTED_SUPPORTED_FLOPS = {
    "M3": 45_991_424,
    "M4": 46_004_224,
    "C1": 46_004_224,
}
M5_PARAMETERS = 47_235
M5_SUPPORTED_FLOPS = 45_991_424


def _namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(
            **{key: _namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _resolved(condition_id: str) -> dict[str, Any]:
    return resolve_config(C03_CONFIGS[condition_id]).data


def _model(condition_id: str):
    config = _resolved(condition_id)
    return build_model(_namespace(config["model"]), metadata=None)


def _metadata() -> dict[int, dict[str, Any]]:
    return {
        1: {"Name": "CWRU", "Label": 1, "Domain_id": 2},
        2: {"Name": "CWRU", "Label": 2, "Domain_id": 2},
        3: {"Name": "CWRU", "Label": 3, "Domain_id": 2},
    }


def _task(condition_id: str) -> Default_task:
    config = _resolved(condition_id)
    return Default_task(
        network=build_model(_namespace(config["model"]), metadata=None),
        args_data=SimpleNamespace(batch_size=3),
        args_model=_namespace(config["model"]),
        args_task=_namespace(config["task"]),
        args_trainer=SimpleNamespace(gpus=0, num_epochs=1),
        args_environment=SimpleNamespace(seed=31),
        metadata=_metadata(),
    )


def _records(*, windows: int = 2) -> list[dict[str, Any]]:
    records = []
    for domain_id in (2, 3):
        for training_label, raw_label in enumerate((1, 2, 3)):
            for replicate in range(2):
                group_id = f"label{raw_label}-group{replicate}"
                for window in range(windows):
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


def _context(
    condition_id: str,
    profile: dict[str, Any],
    *,
    windows: int = 2,
) -> ClassificationContext:
    config = _resolved(condition_id)
    task = _task(condition_id)
    task.args_task.grouped_evaluation.required_windows_per_group_domain = windows
    task._grouped_test_records = _records(windows=windows)
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
        name="c03-test",
        model=task.network,
        task=task,
        trainer=SimpleNamespace(
            callbacks=[SimpleNamespace(best_model_path="model.ckpt")]
        ),
        result={"test_acc_CWRU": 1.0},
    )


@pytest.mark.parametrize("condition_id", ["M3", "M4", "C1"])
def test_c03_configs_share_the_frozen_information_and_budget_contract(
    condition_id: str,
) -> None:
    config = _resolved(condition_id)
    grouped = config["task"]["grouped_evaluation"]

    assert config["model"]["condition"] == MODEL_CONDITIONS[condition_id]
    assert grouped["goal_id"] == "C03"
    assert grouped["condition_id"] == condition_id
    assert config["model"]["num_classes"] == 3
    assert config["task"]["label_contract"]["raw_labels"] == [1, 2, 3]
    assert config["task"]["grouped_split"]["admitted_labels"] == [1, 2, 3]
    assert config["task"]["source_domain_id"] == [0, 1]
    assert config["task"]["target_domain_id"] == [2, 3]
    assert config["data"]["normalization"] == "none"
    assert config["data"]["window_size"] == 4096
    assert config["environment"]["seed"] == 31
    assert config["trainer"]["num_epochs"] == 1
    assert config["trainer"]["early_stopping"] is False
    assert config["trainer"]["device"] == "cuda"
    assert config["trainer"]["gpus"] == 1
    assert config["task"]["optimizer"] == "adam"
    assert config["task"]["lr"] == pytest.approx(0.001)
    assert config["task"]["weight_decay"] == pytest.approx(0.0001)
    assert config["task"]["scheduler"] is None
    assert grouped["source_validation_tuning_trials"] == 0
    assert grouped["required_cuda_visible_devices"] == "3"
    assert grouped["forward_compute"] == {
        "method": (
            "torch.utils.flop_counter.FlopCounterMode_cpu_plus_"
            "explicit_two_token_attention_qk_av"
        ),
        "reference_condition": "M5",
        "batch_size": 1,
        "parameter_relative_tolerance": 0.05,
        "learned_forward_supported_flops_relative_tolerance": 0.10,
        "c1_adjustment": "none_required_existing_M4_within_tolerances",
    }


@pytest.mark.parametrize("condition_id", ["M3", "M4", "C1"])
def test_c03_objective_is_classification_only_and_reaches_declared_paths(
    condition_id: str,
) -> None:
    task = _task(condition_id)
    model = task.network

    def forbidden(*_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("C03 must not call an M5 alignment method")

    model.forward_with_alignment = forbidden
    model.compute_alignment_losses = forbidden
    model.compose_training_objective = forbidden
    batch = {
        "x": torch.randn(3, 256, 2),
        "y": torch.tensor([1, 2, 3]),
        "file_id": torch.tensor([1, 2, 3]),
        "task_id": torch.zeros(3, dtype=torch.long),
    }

    metrics = task._shared_step(batch, "train")
    assert model.uses_alignment_objective is False
    assert model.alignment_config is None
    assert model.alignment_identity() is None
    assert torch.equal(metrics["train_total_loss"], metrics["train_loss"])
    assert not any(
        token in name
        for name in metrics
        for token in ("physical", "semantic", "geometric", "weighted")
    )

    metrics["train_total_loss"].backward()
    gradients = {
        name: parameter.grad
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    required_prefixes = [
        "encoder_1d.",
        "project_1d.",
        "encoder_2d.",
        "project_2d.",
        "head.",
    ]
    if condition_id in {"M4", "C1"}:
        required_prefixes.append("attention.")
    for prefix in required_prefixes:
        assert any(
            name.startswith(prefix) and gradient is not None
            for name, gradient in gradients.items()
        )
    assert (model.attention is not None) is (condition_id in {"M4", "C1"})


def test_c03_parameter_and_supported_flop_profiles_are_exact_and_reproducible() -> None:
    profiles = {condition_id: _profile(condition_id) for condition_id in C03_CONFIGS}
    repeated_c1 = _profile("C1")

    for condition_id, profile in profiles.items():
        observed = profile["observed"]
        reference = profile["m5_reference"]
        assert profile["observed_trainable_parameters"] == EXPECTED_PARAMETERS[
            condition_id
        ]
        assert profile["m5_reference_trainable_parameters"] == M5_PARAMETERS
        assert (
            observed["learned_forward_supported_flops"]
            == EXPECTED_SUPPORTED_FLOPS[condition_id]
        )
        assert (
            reference["learned_forward_supported_flops"]
            == M5_SUPPORTED_FLOPS
        )
        assert observed["input_shape"] == [1, 4096, 2]
        assert observed["output_shape"] == [1, 3]
        assert observed["renderer_output_shape"] == [1, 2, 65, 129]
        assert "aten.convolution" in observed["by_operator"]
        assert "aten.addmm" in observed["by_operator"]
        assert profile["within_tolerances"] is True
    assert profiles["M3"]["parameter_relative_deviation"] == 0.0
    assert (
        profiles["M3"][
            "learned_forward_supported_flops_relative_deviation"
        ]
        == 0.0
    )
    assert profiles["M4"]["parameter_relative_deviation"] == pytest.approx(
        2_176 / M5_PARAMETERS
    )
    assert profiles["M4"][
        "learned_forward_supported_flops_relative_deviation"
    ] == pytest.approx(12_800 / M5_SUPPORTED_FLOPS)
    assert profiles["C1"]["observed"] == repeated_c1["observed"]
    assert profiles["C1"]["m5_reference"] == repeated_c1["m5_reference"]
    assert profiles["C1"]["observed"] == profiles["M4"]["observed"]
    assert (
        profiles["M4"]["observed"]["by_operator"][
            "explicit_two_token_attention_qk_av"
        ]
        == 512
    )


@pytest.mark.parametrize("condition_id", ["M3", "M4", "C1"])
def test_c03_grouped_rows_bind_fairness_and_control_identity(
    condition_id: str,
) -> None:
    profile = _profile(condition_id)
    rows = build_p01_grouped_result_rows(
        _context(condition_id, profile),
        forward_compute_profile=profile,
    )

    assert [row["target_domain"] for row in rows] == [2, 3, "mean_2_3"]
    assert all(row["condition_id"] == condition_id for row in rows)
    assert all(
        row["model_condition"] == MODEL_CONDITIONS[condition_id] for row in rows
    )
    assert all(
        row["run_scope"] == "C03_generic_fusion_control_exploratory"
        for row in rows
    )
    assert all(row["alignment_terms_consumed"] == "none" for row in rows)
    assert all(row["source_validation_tuning_trials"] == 0 for row in rows)
    assert all(row["scheduler"] == "none" for row in rows)
    assert all(row["early_stopping"] is False for row in rows)
    assert all(row["training_epochs"] == 1 for row in rows)
    assert all(row["optimizer"] == "adam" for row in rows)
    assert all(
        row["trainable_parameters"] == EXPECTED_PARAMETERS[condition_id]
        for row in rows
    )
    assert all(
        row["learned_forward_supported_flops"]
        == EXPECTED_SUPPORTED_FLOPS[condition_id]
        for row in rows
    )
    assert all(
        row["m5_reference_trainable_parameters"] == M5_PARAMETERS for row in rows
    )
    assert all(
        row["m5_reference_learned_forward_supported_flops"]
        == M5_SUPPORTED_FLOPS
        for row in rows
    )
    assert all(
        row["capacity_compute_match_status"] == "within_frozen_tolerances"
        for row in rows
    )
    assert all(row["status"] == "succeeded" for row in rows)
    assert all(row["primary_metric_value"] == pytest.approx(1.0) for row in rows)
    assert "not comparative evidence" in rows[0]["scientific_boundary"]


def test_c03_rejects_a_false_c1_identity_and_missing_compute_profile() -> None:
    profile = _profile("C1")
    context = _context("C1", profile)
    context.args_model.condition = "M3"
    with pytest.raises(ValueError, match="identity/model mismatch"):
        build_p01_grouped_result_rows(
            context,
            forward_compute_profile=profile,
        )

    context = _context("C1", profile)
    with pytest.raises(RuntimeError, match="forward-compute profile"):
        build_p01_grouped_result_rows(context)
