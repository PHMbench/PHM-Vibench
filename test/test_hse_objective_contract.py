import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


hse_module = importlib.import_module(
    "src.task_factory.task.pretrain.hse_contrastive"
)
HSETask = hse_module.task


def _stub_parent_init(self, network, *args, **kwargs):
    torch.nn.Module.__init__(self)
    self.network = network


def _task_args(**overrides):
    values = {
        "contrast_weight": 1.0,
        "classification_weight": 0.0,
        "contrast_loss": "INFONCE",
        "temperature": 0.07,
        "augmentation_type": "none",
        "augmentation_noise_std": 0.1,
        "augmentation_dropout_p": 0.1,
        "augmentation_scale_std": 0.1,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _new_task_for_methods():
    instance = HSETask.__new__(HSETask)
    torch.nn.Module.__init__(instance)
    instance.args_task = _task_args()
    instance.ce_loss_fn = F.cross_entropy
    return instance


def test_hse_requires_at_least_one_enabled_objective(monkeypatch):
    monkeypatch.setattr(hse_module.Default_task, "__init__", _stub_parent_init)

    with pytest.raises(ValueError, match="at least one positive objective weight"):
        HSETask(
            torch.nn.Identity(),
            SimpleNamespace(),
            SimpleNamespace(),
            _task_args(contrast_weight=0.0, classification_weight=0.0),
            SimpleNamespace(),
            SimpleNamespace(),
            {},
        )


def test_hse_strategy_initialization_preserves_original_cause(monkeypatch):
    monkeypatch.setattr(hse_module.Default_task, "__init__", _stub_parent_init)

    def fail_strategy(config):
        raise KeyError("unknown loss")

    monkeypatch.setattr(hse_module, "create_contrastive_strategy", fail_strategy)

    with pytest.raises(RuntimeError, match="Unable to initialize") as exc_info:
        HSETask(
            torch.nn.Identity(),
            SimpleNamespace(),
            SimpleNamespace(),
            _task_args(),
            SimpleNamespace(),
            SimpleNamespace(),
            {},
        )

    assert isinstance(exc_info.value.__cause__, KeyError)


def test_classification_objective_rejects_invalid_inputs():
    instance = _new_task_for_methods()

    with pytest.raises(ValueError, match="batch size mismatch"):
        instance._run_classification_flow(
            torch.randn(2, 3, requires_grad=True),
            torch.tensor([0]),
        )

    logits = torch.tensor(
        [[float("nan"), 0.0], [0.0, 1.0]],
        requires_grad=True,
    )
    with pytest.raises(FloatingPointError, match="logits contain NaN or Inf"):
        instance._run_classification_flow(logits, torch.tensor([0, 1]))

    with pytest.raises(ValueError, match="outside the logits class range"):
        instance._run_classification_flow(
            torch.randn(2, 2, requires_grad=True),
            torch.tensor([0, 2]),
        )


def test_contrastive_objective_backpropagates():
    class Strategy:
        requires_labels = False

        @staticmethod
        def compute_loss(**kwargs):
            return {"loss": kwargs["projections"].square().mean()}

    instance = _new_task_for_methods()
    instance.strategy_manager = Strategy()
    features = torch.randn(3, 4, requires_grad=True)

    loss = instance._run_contrastive_flow(features, torch.tensor([0, 1, 0]))
    instance._require_valid_loss(loss, "contrastive objective", "train")
    loss.backward()

    assert features.grad is not None
    assert torch.isfinite(features.grad).all()


def test_contrastive_strategy_errors_are_not_converted_to_zero_loss():
    class BrokenStrategy:
        requires_labels = False

        @staticmethod
        def compute_loss(**kwargs):
            raise RuntimeError("contrastive computation failed")

    instance = _new_task_for_methods()
    instance.strategy_manager = BrokenStrategy()

    with pytest.raises(RuntimeError, match="contrastive computation failed"):
        instance._run_contrastive_flow(
            torch.randn(2, 4, requires_grad=True),
            torch.tensor([0, 1]),
        )


def test_enabled_training_loss_cannot_be_placeholder_zero():
    with pytest.raises(RuntimeError, match="does not require gradients"):
        HSETask._require_valid_loss(
            torch.tensor(0.0),
            "contrastive objective",
            "train",
        )

    with pytest.raises(FloatingPointError, match="non-finite"):
        HSETask._require_valid_loss(
            torch.tensor(float("nan"), requires_grad=True),
            "contrastive objective",
            "train",
        )


def test_invalid_augmentation_is_not_silently_rewritten():
    instance = _new_task_for_methods()
    features = torch.randn(2, 4)

    instance.args_task = _task_args(augmentation_type="unknown")
    with pytest.raises(ValueError, match="Unknown task.augmentation_type"):
        instance._create_augmented_view(features)

    instance.args_task = _task_args(augmentation_dropout_p=1.2)
    with pytest.raises(ValueError, match="0 <= p < 1"):
        instance._create_augmented_view(features)
