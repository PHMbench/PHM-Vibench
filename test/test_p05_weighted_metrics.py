from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.task_factory.Components.loss import get_loss_fn
from src.task_factory.Default_task import Default_task
from src.task_factory.p05_epoch_metrics import (
    WeightedEpochConfusionMatrix,
    WeightedEpochLoss,
    weighted_mean_loss,
)


class _LogitNetwork(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(()))

    def forward(self, x, file_id=None, task_id=None):
        del file_id, task_id
        return x + self.anchor * 0.0

    def cuda(self, device=None):
        del device
        return self


def _build_task(**task_overrides) -> Default_task:
    task_args = {
        "loss": "CE",
        "metrics": ["f1_macro"],
        "p05_evidence_mode": True,
        "train_sample_weight_key": "train_weight",
        "val_sample_weight_key": "val_weight",
    }
    task_args.update(task_overrides)
    return Default_task(
        network=_LogitNetwork(),
        args_data=SimpleNamespace(),
        args_model=SimpleNamespace(num_classes=3, device="cuda"),
        args_task=SimpleNamespace(**task_args),
        args_trainer=SimpleNamespace(
            gpus=1,
            p05_runtime_identity={"evidence_mode": True},
        ),
        args_environment=SimpleNamespace(),
        metadata={1: {"Name": "Dummy_Data", "Label": 2}},
    )


def test_ce_legacy_default_is_mean_and_p05_can_request_unreduced() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.5, 1.5]])
    targets = torch.tensor([0, 1])

    legacy = get_loss_fn("CE")(logits, targets)
    unreduced = get_loss_fn("CE", reduction="none")(logits, targets)

    assert legacy.ndim == 0
    assert unreduced.shape == (2,)
    torch.testing.assert_close(legacy, unreduced.mean())


def test_registered_ce_weighted_spelling_requires_unreduced_external_weights() -> None:
    logits = torch.tensor([[2.0, 0.0], [0.5, 1.5]])
    targets = torch.tensor([0, 1])

    with pytest.raises(ValueError, match="explicit unreduced"):
        get_loss_fn("CE_weighted")

    weighted_spelling = get_loss_fn("CE_weighted", reduction="none")(
        logits,
        targets,
    )
    canonical = get_loss_fn("CE", reduction="none")(logits, targets)
    torch.testing.assert_close(weighted_spelling, canonical)


def test_default_task_accepts_registered_ce_weighted_spelling() -> None:
    task = _build_task(loss="CE_weighted")
    result = task._compute_loss(
        torch.tensor([[2.0, 0.0], [0.5, 1.5]]),
        torch.tensor([0, 1]),
        sample_weight=torch.tensor([1.0, 3.0]),
    )

    expected = (
        F.cross_entropy(
            torch.tensor([[2.0, 0.0], [0.5, 1.5]]),
            torch.tensor([0, 1]),
            reduction="none",
        )
        * torch.tensor([1.0, 3.0])
    ).sum() / 4.0
    torch.testing.assert_close(result, expected)


def test_weighted_mean_loss_is_explicit_and_differentiable() -> None:
    per_sample = torch.tensor([1.0, 2.0, 5.0], requires_grad=True)
    weights = torch.tensor([1.0, 3.0, 2.0], dtype=torch.float64)

    result = weighted_mean_loss(per_sample, weights)
    result.backward()

    assert result.item() == pytest.approx(17.0 / 6.0)
    torch.testing.assert_close(
        per_sample.grad,
        torch.tensor([1.0 / 6.0, 3.0 / 6.0, 2.0 / 6.0]),
    )


@pytest.mark.parametrize(
    ("weights", "error", "match"),
    [
        (None, KeyError, "requires sample_weight"),
        (torch.tensor([1.0, float("nan")]), FloatingPointError, "non-finite"),
        (torch.tensor([0.0, 0.0]), ValueError, "positive total"),
        (torch.tensor([1.0, -0.5]), ValueError, "non-negative"),
    ],
)
def test_weighted_mean_loss_fails_closed_on_invalid_weights(weights, error, match) -> None:
    with pytest.raises(error, match=match):
        weighted_mean_loss(torch.tensor([1.0, 2.0]), weights)


def test_epoch_macro_f1_uses_one_float64_weighted_confusion_matrix() -> None:
    metric = WeightedEpochConfusionMatrix(num_classes=3)
    metric.update(
        torch.tensor([0, 0, 1]),
        torch.tensor([0, 1, 1]),
        torch.tensor([1.0, 2.0, 1.0]),
    )
    metric.update(
        torch.tensor([1, 2]),
        torch.tensor([0, 2]),
        torch.tensor([3.0, 4.0]),
    )

    expected_matrix = torch.tensor(
        [[1.0, 3.0, 0.0], [2.0, 1.0, 0.0], [0.0, 0.0, 4.0]],
        dtype=torch.float64,
    )
    torch.testing.assert_close(metric.matrix, expected_matrix)
    assert metric.matrix.dtype == torch.float64
    assert metric.compute_macro_f1().item() == pytest.approx(11.0 / 21.0)
    assert metric.compute_macro_f1().item() != pytest.approx(1.0 / 3.0)


def test_epoch_macro_f1_defines_absent_class_division_as_zero() -> None:
    metric = WeightedEpochConfusionMatrix(num_classes=3)
    metric.update(
        torch.tensor([0, 1]),
        torch.tensor([0, 1]),
        torch.ones(2),
    )

    assert metric.compute_macro_f1().item() == pytest.approx(2.0 / 3.0)


def test_epoch_loss_aggregates_weighted_numerator_not_batch_means() -> None:
    metric = WeightedEpochLoss()
    metric.update(
        torch.tensor([1.0, 3.0]),
        torch.tensor([1.0, 9.0]),
    )
    metric.update(
        torch.tensor([10.0]),
        torch.tensor([2.0]),
    )

    assert metric.compute().item() == pytest.approx((1.0 + 27.0 + 20.0) / 12.0)
    assert metric.compute().item() != pytest.approx(((28.0 / 10.0) + 10.0) / 2.0)


def test_default_task_binds_train_and_validation_weights_separately() -> None:
    task = _build_task()
    train_logits = torch.tensor([[3.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    train_targets = torch.tensor([0, 1])
    train_weights = torch.tensor([1.0, 4.0])
    val_logits = torch.tensor([[0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
    val_targets = torch.tensor([0, 1])
    val_weights = torch.tensor([2.0, 1.0])

    train_result = task._shared_step(
        {
            "x": train_logits,
            "y": train_targets,
            "file_id": torch.tensor([1, 1]),
            "train_weight": train_weights,
        },
        "train",
    )
    val_result = task._shared_step(
        {
            "x": val_logits,
            "y": val_targets,
            "file_id": torch.tensor([1, 1]),
            "val_weight": val_weights,
        },
        "val",
    )

    expected_train = (
        F.cross_entropy(train_logits, train_targets, reduction="none") * train_weights
    ).sum() / train_weights.sum()
    expected_val = (
        F.cross_entropy(val_logits, val_targets, reduction="none") * val_weights
    ).sum() / val_weights.sum()
    torch.testing.assert_close(train_result["train_loss"], expected_train)
    torch.testing.assert_close(val_result["val_loss"], expected_val)
    assert task.p05_epoch_metrics["train_epoch"].matrix.sum().item() == pytest.approx(5.0)
    assert task.p05_epoch_metrics["val_epoch"].matrix.sum().item() == pytest.approx(3.0)


def test_default_task_p05_mode_fails_when_stage_weight_is_missing() -> None:
    task = _build_task()
    with pytest.raises(KeyError, match="train.*missing.*train_weight"):
        task._shared_step(
            {
                "x": torch.tensor([[1.0, 0.0, 0.0]]),
                "y": torch.tensor([0]),
                "file_id": torch.tensor([1]),
            },
            "train",
        )


def test_default_task_logs_one_exact_epoch_macro_f1(monkeypatch) -> None:
    task = _build_task()
    task.on_validation_epoch_start()

    first = task._shared_step(
        {
            "x": torch.tensor(
                [[3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
            ),
            "y": torch.tensor([0, 1, 1]),
            "file_id": torch.tensor([1, 1, 1]),
            "val_weight": torch.tensor([1.0, 2.0, 1.0]),
        },
        "val",
    )
    second = task._shared_step(
        {
            "x": torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),
            "y": torch.tensor([0, 2]),
            "file_id": torch.tensor([1, 1]),
            "val_weight": torch.tensor([3.0, 4.0]),
        },
        "val",
    )
    logged = {}
    monkeypatch.setattr(task, "log", lambda name, value, **kwargs: logged.setdefault(name, value))

    task.on_validation_epoch_end()

    assert not any("f1" in name for name in first)
    assert not any("f1" in name for name in second)
    assert logged["val_f1_macro"].item() == pytest.approx(11.0 / 21.0)
    assert logged["val_acc"].item() == pytest.approx(6.0 / 11.0)
    per_sample = torch.cat(
        (
            F.cross_entropy(
                torch.tensor(
                    [[3.0, 0.0, 0.0], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]
                ),
                torch.tensor([0, 1, 1]),
                reduction="none",
            ),
            F.cross_entropy(
                torch.tensor([[0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]),
                torch.tensor([0, 2]),
                reduction="none",
            ),
        )
    )
    weights = torch.tensor([1.0, 2.0, 1.0, 3.0, 4.0])
    expected_loss = (per_sample * weights).sum() / weights.sum()
    torch.testing.assert_close(logged["val_loss"].float(), expected_loss)


def test_default_task_legacy_mode_does_not_require_sample_weight() -> None:
    task = _build_task(p05_evidence_mode=False, metrics=[])
    logits = torch.tensor([[2.0, 0.0, 0.0], [0.0, 2.0, 0.0]])
    targets = torch.tensor([0, 1])

    result = task._compute_loss(logits, targets)

    torch.testing.assert_close(result, F.cross_entropy(logits, targets))


@pytest.mark.parametrize(
    ("model_device", "runtime_identity", "message"),
    [
        ("cpu", {"evidence_mode": True}, "model.device='cuda'"),
        ("cuda", None, "runtime preflight"),
    ],
)
def test_default_task_p05_mode_forbids_unverified_cpu_fallback(
    model_device, runtime_identity, message
) -> None:
    with pytest.raises(RuntimeError, match=message):
        Default_task(
            network=_LogitNetwork(),
            args_data=SimpleNamespace(),
            args_model=SimpleNamespace(num_classes=3, device=model_device),
            args_task=SimpleNamespace(
                loss="CE", metrics=["f1_macro"], p05_evidence_mode=True
            ),
            args_trainer=SimpleNamespace(
                gpus=1,
                p05_runtime_identity=runtime_identity,
            ),
            args_environment=SimpleNamespace(),
            metadata={1: {"Name": "Dummy_Data", "Label": 2}},
        )
