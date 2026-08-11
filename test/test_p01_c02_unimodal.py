from __future__ import annotations

import copy
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Mapping

import pytest
import torch
import yaml

from phmfactory.config import resolve_config
from src.Pipeline_01_Fault_Diagnosis import build_p01_grouped_result_rows
from src.model_factory import build_model
from src.runtime import ClassificationContext
from src.task_factory.Default_task import Default_task


MODEL_CONFIG = Path("configs/base/model/p01_alignment.yaml")
C02_CONFIGS = {
    "M1": Path("configs/experiments/p01/p01_c02_m1.yaml"),
    "M2": Path("configs/experiments/p01/p01_c02_m2.yaml"),
}


def _namespace(value: Any) -> Any:
    if isinstance(value, Mapping):
        return SimpleNamespace(
            **{key: _namespace(item) for key, item in value.items()}
        )
    if isinstance(value, list):
        return [_namespace(item) for item in value]
    return value


def _model(condition: str):
    payload = yaml.safe_load(MODEL_CONFIG.read_text(encoding="utf-8"))["model"]
    payload = copy.deepcopy(payload)
    payload["condition"] = condition
    payload["num_classes"] = 3
    return build_model(_namespace(payload), metadata=None)


def _task_args(*, windows: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        type="DG",
        name="classification",
        loss="CE",
        metrics=["acc", "f1"],
        metric_average="macro",
        optimizer="adam",
        lr=0.001,
        weight_decay=0.0001,
        source_domain_id=[0, 1],
        target_domain_id=[2, 3],
        label_contract=SimpleNamespace(raw_labels=[1, 2, 3]),
        grouped_split=SimpleNamespace(
            admitted_labels=[1, 2, 3],
            group_key="documented_condition_block",
        ),
        grouped_evaluation=SimpleNamespace(
            enabled=True,
            run_stage="targeted_test",
            primary_metric="condition_block_macro_f1",
            aggregation=(
                "mean_softmax_windows_then_argmax_per_domain_condition_block"
            ),
            checkpoint_selection="lowest_validation_loss_within_fixed_budget",
            required_groups_per_domain=6,
            required_groups_per_class_domain=2,
            required_windows_per_group_domain=windows,
            required_cuda_visible_devices="3",
        ),
    )


def _metadata() -> dict[int, dict[str, Any]]:
    return {
        1: {"Name": "CWRU", "Label": 1, "Domain_id": 2},
        2: {"Name": "CWRU", "Label": 2, "Domain_id": 2},
        3: {"Name": "CWRU", "Label": 3, "Domain_id": 2},
    }


def _task(condition: str = "M1", *, windows: int = 2) -> Default_task:
    return Default_task(
        network=_model(condition),
        args_data=SimpleNamespace(),
        args_model=SimpleNamespace(condition=condition),
        args_task=_task_args(windows=windows),
        args_trainer=SimpleNamespace(gpus=0, num_epochs=10),
        args_environment=SimpleNamespace(seed=31),
        metadata=_metadata(),
    )


@pytest.mark.parametrize("condition", ["M1", "M2"])
def test_c02_configs_freeze_the_same_three_class_contract(condition: str) -> None:
    config = resolve_config(C02_CONFIGS[condition]).data

    assert config["model"]["condition"] == condition
    assert config["model"]["num_classes"] == 3
    assert config["task"]["label_contract"]["raw_labels"] == [1, 2, 3]
    assert config["task"]["grouped_split"]["admitted_labels"] == [1, 2, 3]
    assert config["task"]["metric_average"] == "macro"
    assert config["environment"]["seed"] == 31
    assert config["trainer"]["num_epochs"] == 10
    assert config["trainer"]["early_stopping"] is False
    assert config["trainer"]["gpus"] == 1
    assert config["trainer"]["device"] == "cuda"
    assert (
        config["task"]["grouped_evaluation"]["required_cuda_visible_devices"]
        == "3"
    )


def test_label_contract_maps_raw_labels_and_rejects_unknowns() -> None:
    task = _task()
    raw = torch.tensor([1, 2, 3, 1])

    encoded = task.encode_raw_labels(raw)

    assert encoded.tolist() == [0, 1, 2, 0]
    assert task.decode_training_indices(encoded).tolist() == raw.tolist()
    assert task.label_contract_identity() == {
        "raw_labels": [1, 2, 3],
        "training_indices": [0, 1, 2],
        "raw_to_training_index": {1: 0, 2: 1, 3: 2},
    }
    assert task.metrics["CWRU"]["test_f1"].average == "macro"
    with pytest.raises(ValueError, match="outside task.label_contract"):
        task.encode_raw_labels(torch.tensor([0, 1]))


@pytest.mark.parametrize(
    ("raw_labels", "num_classes", "message"),
    [
        ([1, 1, 3], 3, "must not contain duplicates"),
        ([1, 2, 3], 4, "num_classes must equal"),
        ([1, 3, 2], 3, "must exactly match"),
    ],
)
def test_label_contract_fails_closed(
    raw_labels: list[int], num_classes: int, message: str
) -> None:
    model = _model("M1")
    model.num_classes = num_classes
    args_task = _task_args()
    args_task.label_contract.raw_labels = raw_labels

    with pytest.raises(ValueError, match=message):
        Default_task(
            network=model,
            args_data=SimpleNamespace(),
            args_model=SimpleNamespace(condition="M1"),
            args_task=args_task,
            args_trainer=SimpleNamespace(gpus=0, num_epochs=10),
            args_environment=SimpleNamespace(seed=31),
            metadata=_metadata(),
        )


def test_three_logit_objective_backpropagates_after_mapping() -> None:
    task = _task()
    batch = {
        "x": torch.randn(3, 256, 2),
        "y": torch.tensor([1, 2, 3]),
        "file_id": torch.tensor([1, 2, 3]),
    }

    metrics = task._shared_step(batch, "train")
    metrics["train_total_loss"].backward()

    assert torch.isfinite(metrics["train_total_loss"])
    assert any(
        parameter.grad is not None
        for parameter in task.network.parameters()
        if parameter.requires_grad
    )


def test_m1_m2_consume_only_the_declared_view_and_match_heads() -> None:
    torch.manual_seed(17)
    m1 = _model("M1").eval()
    m2 = _model("M2").eval()
    waveform = torch.randn(2, 256, 2)
    renderer_source = torch.randn(2, 256, 2)

    assert m1.renderer is None
    assert m1.encoder_2d is None
    assert m1.project_2d is None
    assert m2.encoder_1d is None
    assert m2.project_1d is None

    with torch.no_grad():
        m1_reference = m1.forward_paired_views(waveform, renderer_source)
        m1_irrelevant_changed = m1.forward_paired_views(
            waveform, renderer_source + 100.0
        )
        m1_relevant_changed = m1.forward_paired_views(
            waveform + 1.0, renderer_source
        )
        m2_reference = m2.forward_paired_views(waveform, renderer_source)
        m2_irrelevant_changed = m2.forward_paired_views(
            waveform + 100.0, renderer_source
        )
        m2_relevant_changed = m2.forward_paired_views(
            waveform, renderer_source + 1.0
        )

    assert torch.equal(m1_reference, m1_irrelevant_changed)
    assert not torch.equal(m1_reference, m1_relevant_changed)
    assert torch.equal(m2_reference, m2_irrelevant_changed)
    assert not torch.equal(m2_reference, m2_relevant_changed)

    first = waveform.detach().clone().requires_grad_(True)
    second = renderer_source.detach().clone().requires_grad_(True)
    m1.forward_paired_views(first, second).sum().backward()
    assert first.grad is not None
    assert second.grad is None

    first = waveform.detach().clone().requires_grad_(True)
    second = renderer_source.detach().clone().requires_grad_(True)
    m2.forward_paired_views(first, second).sum().backward()
    assert first.grad is None
    assert second.grad is not None

    m1_head = [(name, tuple(value.shape)) for name, value in m1.head.named_parameters()]
    m2_head = [(name, tuple(value.shape)) for name, value in m2.head.named_parameters()]
    assert m1_head == m2_head
    assert sum(value.numel() for value in m1.head.parameters()) == 2_307
    assert sum(value.numel() for value in m2.head.parameters()) == 2_307
    assert m1.trainable_parameter_count == 19_587
    assert m2.trainable_parameter_count == 27_907


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


def _context(condition: str = "M1", *, windows: int = 2) -> ClassificationContext:
    task = _task(condition, windows=windows)
    task._grouped_test_records = _records(windows=windows)
    model = task.network
    args_task = task.args_task
    args_model = SimpleNamespace(condition=condition)
    args_trainer = SimpleNamespace(num_epochs=10)
    return ClassificationContext(
        args=SimpleNamespace(),
        configs=SimpleNamespace(),
        args_environment=SimpleNamespace(seed=31),
        args_data=SimpleNamespace(),
        args_model=args_model,
        args_task=args_task,
        args_trainer=args_trainer,
        iteration=0,
        path=Path("unused"),
        name="c02-test",
        model=model,
        task=task,
        trainer=SimpleNamespace(
            callbacks=[SimpleNamespace(best_model_path="model.ckpt")]
        ),
        result={"test_acc_CWRU": 1.0},
    )


@pytest.mark.parametrize(
    ("condition", "expected_total"), [("M1", 19_587), ("M2", 27_907)]
)
def test_grouped_rows_are_domain_specific_and_report_capacity(
    condition: str, expected_total: int
) -> None:
    rows = build_p01_grouped_result_rows(_context(condition))

    assert [row["target_domain"] for row in rows] == [2, 3, "mean_2_3"]
    assert all(row["status"] == "completed" for row in rows)
    assert all(row["seed"] == 31 for row in rows)
    assert all(
        row["primary_metric_name"] == "condition_block_macro_f1"
        for row in rows
    )
    assert all(row["primary_metric_value"] == pytest.approx(1.0) for row in rows)
    assert all(row["classifier_head_parameters"] == 2_307 for row in rows)
    assert all(row["trainable_parameters"] == expected_total for row in rows)
    assert rows[0]["group_count"] == 6
    assert rows[0]["class_group_support"] == "1:2|2:2|3:2"
    assert rows[0]["window_count"] == 12
    assert rows[2]["group_count"] == 6
    assert rows[2]["window_count"] == 24
    assert rows[2]["evaluated_domain_count"] == 2
    assert "not independent repetitions" in rows[2]["scientific_boundary"]


def test_grouped_rows_reject_mixed_labels_and_missing_class_support() -> None:
    context = _context()
    context.task._grouped_test_records[0]["raw_label"] = 2
    with pytest.raises(ValueError, match="raw/training labels"):
        build_p01_grouped_result_rows(context)

    context = _context()
    context.task._grouped_test_records = [
        record
        for record in context.task._grouped_test_records
        if record["physical_group_id"] != "label3-group1"
    ]
    with pytest.raises(ValueError, match="has 5 groups"):
        build_p01_grouped_result_rows(context)
