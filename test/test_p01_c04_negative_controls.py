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
    _p01_parameter_counts,
    _profile_p01_learned_forward_flops,
    build_p01_forward_compute_profile,
    build_p01_grouped_result_rows,
)
from src.model_factory import build_model
from src.runtime import ClassificationContext
from src.task_factory.Default_task import Default_task


C04_CONFIGS = {
    "M5": Path("configs/experiments/p01/p01_c04_m5.yaml"),
    "C2": Path("configs/experiments/p01/p01_c04_c2.yaml"),
    "C3": Path("configs/experiments/p01/p01_c04_c3.yaml"),
}
MODEL_CONDITIONS = {"M5": "M5", "C2": "M5", "C3": "C3"}
RUN_IDS = {"M5": "RUN-0016", "C2": "RUN-0017", "C3": "RUN-0018"}
EXPECTED_PARAMETERS = {"M5": 47_235, "C2": 47_235, "C3": 55_555}
EXPECTED_SUPPORTED_FLOPS = {
    "M5": 45_991_424,
    "C2": 45_991_424,
    "C3": 46_336_640,
}
M4_C1_PARAMETERS = 49_411
M4_C1_SUPPORTED_FLOPS = 46_004_224


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
    return resolve_config(C04_CONFIGS[condition_id]).data


def _model(condition_id: str):
    return build_model(_namespace(_resolved(condition_id)["model"]), metadata=None)


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
        args_trainer=SimpleNamespace(gpus=0, device="cpu", num_epochs=1),
        args_environment=SimpleNamespace(seed=31),
        metadata=_metadata(),
    )


def _batch(seed: int = 101, batch_size: int = 6) -> dict[str, torch.Tensor]:
    if batch_size % 3:
        raise ValueError("controlled C04 batch size must be divisible by three")
    generator = torch.Generator().manual_seed(seed)
    repeats = batch_size // 3
    return {
        "x": torch.randn(batch_size, 256, 2, generator=generator),
        "y": torch.tensor([1, 2, 3] * repeats, dtype=torch.long),
        "file_id": torch.tensor([1, 2, 3] * repeats, dtype=torch.long),
        "task_id": torch.zeros(batch_size, dtype=torch.long),
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
        name="c04-test",
        model=task.network,
        task=task,
        trainer=SimpleNamespace(
            callbacks=[SimpleNamespace(best_model_path="model.ckpt")]
        ),
        result={"test_acc_CWRU": 1.0},
    )


@pytest.mark.parametrize("condition_id", ["M5", "C2", "C3"])
def test_c04_configs_share_the_frozen_execution_contract(condition_id: str) -> None:
    config = _resolved(condition_id)
    grouped = config["task"]["grouped_evaluation"]

    assert config["model"]["condition"] == MODEL_CONDITIONS[condition_id]
    assert grouped["goal_id"] == "C04"
    assert grouped["run_id"] == RUN_IDS[condition_id]
    assert grouped["condition_id"] == condition_id
    assert config["environment"]["seed"] == 31
    assert config["trainer"]["num_epochs"] == 1
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
    assert config["model"]["alignment"] == _resolved("M5")["model"]["alignment"]

    control = config["task"].get("alignment_target_control")
    if condition_id == "C2":
        assert control == {
            "mode": "seeded_sattolo_derangement_after_batching",
            "seed": 31042,
            "seed_key": "base_seed_plus_epoch_times_1000003_plus_batch_index",
        }
    else:
        assert control is None


def test_c2_derangements_are_keyed_reproducible_and_rng_isolated() -> None:
    first = _task("C2")
    second = _task("C2")
    before = torch.random.get_rng_state().clone()
    permutation_0 = first._alignment_target_permutation(
        batch_size=8, device=torch.device("cpu"), batch_index=0
    )
    after = torch.random.get_rng_state()
    repeated_0 = second._alignment_target_permutation(
        batch_size=8, device=torch.device("cpu"), batch_index=0
    )
    permutation_1 = first._alignment_target_permutation(
        batch_size=8, device=torch.device("cpu"), batch_index=1
    )

    assert torch.equal(before, after)
    assert torch.equal(permutation_0, repeated_0)
    assert not torch.equal(permutation_0, permutation_1)
    expected = torch.arange(8)
    for permutation in (permutation_0, permutation_1):
        assert torch.equal(torch.sort(permutation).values, expected)
        assert not torch.eq(permutation, expected).any()


def test_c2_changes_only_alignment_target_slots_and_preserves_loss_scale() -> None:
    torch.manual_seed(107)
    matched = _model("M5")
    permuted = _model("C2")
    permuted.load_state_dict(copy.deepcopy(matched.state_dict()), strict=True)
    matched.eval()
    permuted.eval()
    batch = _batch(107)
    labels = batch["y"] - 1
    permutation = torch.tensor([1, 2, 3, 4, 5, 0], dtype=torch.long)

    matched_logits, matched_losses = matched.forward_with_alignment(
        batch["x"], labels
    )
    matched_state = matched.get_representation_state()
    permuted_logits, permuted_losses = permuted.forward_with_alignment(
        batch["x"], labels, alignment_target_permutation=permutation
    )
    permuted_state = permuted.get_representation_state()
    manual_losses = permuted.compute_alignment_losses(
        batch["x"],
        labels,
        permuted_state,
        target_permutation=permutation,
    )

    assert matched.trainable_parameter_count == permuted.trainable_parameter_count
    assert matched.alignment_identity() == permuted.alignment_identity()
    assert set(matched.state_dict()) == set(permuted.state_dict())
    torch.testing.assert_close(permuted_logits, matched_logits, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        permuted_state["z_1"], matched_state["z_1"], rtol=0.0, atol=0.0
    )
    torch.testing.assert_close(
        permuted_state["z_2"], matched_state["z_2"], rtol=0.0, atol=0.0
    )
    for name, value in manual_losses.items():
        torch.testing.assert_close(permuted_losses[name], value, rtol=0.0, atol=0.0)
    torch.testing.assert_close(
        permuted_losses["physical_parseval"],
        matched_losses["physical_parseval"],
        rtol=0.0,
        atol=0.0,
    )
    assert any(
        not torch.equal(permuted_losses[name], matched_losses[name])
        for name in ("physical_energy", "physical_spectral", "semantic", "geometric")
    )


@pytest.mark.parametrize(
    "permutation, message",
    (
        (torch.tensor([1, 0, 2]), "derangement"),
        (torch.tensor([1, 1, 0]), "every batch index once"),
        (torch.tensor([1.0, 2.0, 0.0]), "torch.long"),
        (torch.tensor([[1, 2, 0]]), "shape"),
    ),
)
def test_c2_invalid_target_permutations_fail_closed(
    permutation: torch.Tensor, message: str
) -> None:
    model = _model("C2")
    batch = _batch(109, batch_size=3)
    with pytest.raises(ValueError, match=message):
        model.forward_with_alignment(
            batch["x"],
            batch["y"] - 1,
            alignment_target_permutation=permutation,
        )


def test_c2_training_summary_reconstructs_objective_and_tracks_schedule() -> None:
    task = _task("C2")
    first_metrics = task._shared_step(_batch(113), "train", batch_index=0)
    second_metrics = task._shared_step(_batch(127), "train", batch_index=1)
    summary = task.training_objective_summary()

    assert summary is not None
    assert summary["observed_batches"] == 2
    assert summary["observed_samples"] == 12
    assert summary["aggregation"] == "batch_scalar_mean_weighted_by_batch_size"
    assert abs(summary["objective_reconstruction_residual"]) <= 1.0e-6
    assert summary["target_permutation_observation"] == {
        "observed_permutations": 2,
        "observed_fixed_points": 0,
        "derived_seed_min": 31042,
        "derived_seed_max": 31043,
        "unique_derived_seeds": 2,
    }
    for metrics in (first_metrics, second_metrics):
        expected = (
            metrics["train_classification_loss"]
            + metrics["train_weighted_physical_loss"]
            + metrics["train_weighted_semantic_loss"]
            + metrics["train_weighted_geometric_loss"]
        )
        torch.testing.assert_close(metrics["train_total_loss"], expected)
    first_metrics["train_total_loss"].backward()
    for prefix in ("encoder_1d.", "encoder_2d."):
        assert any(
            name.startswith(prefix) and parameter.grad is not None
            for name, parameter in task.network.named_parameters()
        )


def test_validation_never_constructs_or_consumes_c2_target_permutation() -> None:
    task = _task("C2")

    def forbidden(**_kwargs: Any) -> None:
        raise AssertionError("validation/test must not construct an alignment permutation")

    task._alignment_target_permutation = forbidden  # type: ignore[method-assign]
    metrics = task._shared_step(_batch(131), "val")
    assert "val_total_loss" in metrics
    assert not any("physical" in name for name in metrics)


def test_c3_executes_one_renderer_tensor_through_independent_2d_copies() -> None:
    torch.manual_seed(137)
    model = _model("C3")
    model.eval()
    rendered: list[torch.Tensor] = []
    branch_inputs: list[torch.Tensor] = []
    renderer_hook = model.renderer.register_forward_hook(
        lambda _module, _inputs, output: rendered.append(output)
    )
    first_hook = model.encoder_2d.register_forward_pre_hook(
        lambda _module, inputs: branch_inputs.append(inputs[0])
    )
    second_hook = model.encoder_duplicate_2d.register_forward_pre_hook(
        lambda _module, inputs: branch_inputs.append(inputs[0])
    )
    batch = _batch(137)
    try:
        logits = model(batch["x"])
    finally:
        renderer_hook.remove()
        first_hook.remove()
        second_hook.remove()

    assert logits.shape == (6, 3)
    assert len(rendered) == 1
    assert len(branch_inputs) == 2
    assert branch_inputs[0] is rendered[0]
    assert branch_inputs[1] is rendered[0]
    assert model.encoder_1d is None and model.project_1d is None
    assert model.attention is None
    assert model.uses_alignment_objective is False
    assert model.alignment_identity() is None
    assert model.encoder_2d is not model.encoder_duplicate_2d
    assert model.project_2d is not model.project_duplicate_2d
    assert model.duplicate_control_identity()["parameter_storage"] == (
        "independent_no_weight_sharing"
    )

    logits.sum().backward()
    for prefix in (
        "encoder_2d.",
        "project_2d.",
        "encoder_duplicate_2d.",
        "project_duplicate_2d.",
        "head.",
    ):
        assert any(
            name.startswith(prefix) and parameter.grad is not None
            for name, parameter in model.named_parameters()
        )


def test_c3_rejects_different_sources_and_rebuilds_reproducibly() -> None:
    torch.manual_seed(139)
    first = _model("C3").eval()
    torch.manual_seed(139)
    second = _model("C3").eval()
    batch = _batch(139)

    for name, value in first.state_dict().items():
        torch.testing.assert_close(value, second.state_dict()[name], rtol=0.0, atol=0.0)
    torch.testing.assert_close(first(batch["x"]), second(batch["x"]), rtol=0.0, atol=0.0)
    with pytest.raises(ValueError, match="identical deterministic source"):
        first.forward_paired_views(batch["x"], batch["x"] + 1.0)


def test_c04_parameter_and_compute_profiles_include_m4_c1_comparison() -> None:
    for condition_id in ("M5", "C2", "C3"):
        profile = _profile(condition_id)
        assert profile["observed_trainable_parameters"] == EXPECTED_PARAMETERS[
            condition_id
        ]
        assert profile["observed"]["learned_forward_supported_flops"] == (
            EXPECTED_SUPPORTED_FLOPS[condition_id]
        )
        assert profile["m5_reference_trainable_parameters"] == 47_235
        assert profile["m5_reference"]["learned_forward_supported_flops"] == 45_991_424
        assert profile["m4_c1_reference_trainable_parameters"] == M4_C1_PARAMETERS
        assert profile["m4_c1_reference"]["learned_forward_supported_flops"] == (
            M4_C1_SUPPORTED_FLOPS
        )
    c3 = _profile("C3")
    assert c3["within_tolerances"] is False
    assert c3["parameter_relative_deviation"] == pytest.approx(8_320 / 47_235)
    assert c3["parameter_relative_deviation_from_m4_c1"] == pytest.approx(
        6_144 / M4_C1_PARAMETERS
    )
    assert c3[
        "learned_forward_supported_flops_relative_deviation_from_m4_c1"
    ] == pytest.approx(332_416 / M4_C1_SUPPORTED_FLOPS)

    base_args = _namespace(copy.deepcopy(_resolved("M5")["model"]))
    measured: dict[str, tuple[int, int]] = {}
    for model_condition in ("M1", "M2", "M3"):
        candidate_args = copy.deepcopy(base_args)
        candidate_args.condition = model_condition
        candidate = build_model(candidate_args, metadata=None)
        candidate_profile = _profile_p01_learned_forward_flops(
            candidate, window_size=4096, batch_size=1
        )
        measured[model_condition] = (
            sum(
                parameter.numel()
                for parameter in candidate.parameters()
                if parameter.requires_grad
            ),
            candidate_profile["learned_forward_supported_flops"],
        )
    duplicate_1d = tuple(
        measured["M3"][index]
        + measured["M1"][index]
        - measured["M2"][index]
        for index in (0, 1)
    )
    duplicate_2d = tuple(
        measured["M3"][index]
        + measured["M2"][index]
        - measured["M1"][index]
        for index in (0, 1)
    )
    assert duplicate_1d == (38_915, 45_646_208)
    assert duplicate_2d == (55_555, 46_336_640)
    assert abs(duplicate_1d[0] - 47_235) == abs(duplicate_2d[0] - 47_235)
    assert abs(duplicate_1d[1] - 45_991_424) == abs(
        duplicate_2d[1] - 45_991_424
    )


def test_c02_c03_parameter_schema_is_not_changed_by_c3_components() -> None:
    m5_counts = _p01_parameter_counts(_model("M5"))
    c3_counts = _p01_parameter_counts(_model("C3"))
    assert "encoder_duplicate_2d" not in m5_counts
    assert "project_duplicate_2d" not in m5_counts
    assert c3_counts["encoder_duplicate_2d"] > 0
    assert c3_counts["project_duplicate_2d"] > 0
    assert sum(value for key, value in c3_counts.items() if key != "total") == (
        c3_counts["total"]
    )


@pytest.mark.parametrize("condition_id", ["M5", "C2", "C3"])
def test_c04_rows_are_comparable_and_reconstruct_controls(condition_id: str) -> None:
    task = _task(condition_id)
    task._shared_step(_batch(149), "train", batch_index=0)
    summary = task.training_objective_summary()
    rows = build_p01_grouped_result_rows(
        _context(condition_id, task),
        forward_compute_profile=_profile(condition_id),
        training_objective_summary=summary,
    )

    assert [row["target_domain"] for row in rows] == [2, 3, "mean_2_3"]
    for row in rows:
        assert row["run_id"] == RUN_IDS[condition_id]
        assert row["condition_id"] == condition_id
        assert row["model_condition"] == MODEL_CONDITIONS[condition_id]
        assert row["training_epochs"] == 1
        assert row["source_validation_tuning_trials"] == 0
        assert row["loss_scale_retuned"] is False
        assert row["target_control_retuned"] is False
        assert json.loads(row["training_objective_summary_json"]) == summary
        comparison = json.loads(row["capacity_compute_comparison_json"])
        assert comparison["observed"]["trainable_parameters"] == (
            EXPECTED_PARAMETERS[condition_id]
        )
        assert comparison["m4_c1_reference"]["trainable_parameters"] == (
            M4_C1_PARAMETERS
        )
        target_control = json.loads(
            row["alignment_target_control_identity_json"]
        )
        duplicate = json.loads(row["duplicate_control_identity_json"])
        if condition_id == "C2":
            assert target_control["algorithm"] == "sattolo_single_cycle"
            assert target_control["semantic_mask_basis"] == (
                "original_label_and_index_slots"
            )
            assert target_control["unaffected_terms"] == [
                "classification",
                "physical_parseval",
            ]
        elif condition_id == "M5":
            assert target_control["mode"] == "matched_no_permutation"
        else:
            assert target_control["mode"] == "not_applicable"
            assert duplicate["renderer_execution"] == (
                "single_call_shared_tensor_object"
            )
            assert row["capacity_compute_match_status"] == (
                "outside_frozen_tolerances"
            )
