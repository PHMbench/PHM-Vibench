import importlib
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from src.model_factory.ISFM.embedding.E_01_HSE import E_01_HSE


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
    instance.args_environment = SimpleNamespace(seed=17)
    instance.ce_loss_fn = F.cross_entropy
    return instance


def _embedding_args(**overrides):
    values = {
        "patch_size_L": 4,
        "patch_size_C": 2,
        "num_patches": 5,
        "output_dim": 8,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


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


def test_hse_embedding_rejects_oversized_time_patch_without_repetition():
    model = E_01_HSE(_embedding_args(patch_size_L=9))
    x = torch.randn(2, 8, 3)

    with pytest.raises(ValueError, match="does not repeat or pad time"):
        model(x, fs=torch.tensor([12_000.0, 48_000.0]))


def test_hse_embedding_rejects_oversized_channel_patch_without_duplication():
    model = E_01_HSE(_embedding_args(patch_size_C=4))
    x = torch.randn(2, 8, 3)

    with pytest.raises(ValueError, match="does not duplicate or pad channels"):
        model(x, fs=torch.tensor([12_000.0, 48_000.0]))


def test_hse_embedding_eval_is_deterministic_and_does_not_sample_random_starts(
    monkeypatch,
):
    model = E_01_HSE(_embedding_args())
    model.eval()
    x = torch.arange(2 * 12 * 3, dtype=torch.float32).reshape(2, 12, 3)
    fs = torch.tensor([12_000.0, 48_000.0])

    def fail_random_start(*args, **kwargs):
        pytest.fail("evaluation must not call torch.randint for HSE patch starts")

    monkeypatch.setattr(torch, "randint", fail_random_start)
    first = model(x, fs=fs)
    torch.manual_seed(999)
    second = model(x, fs=fs)

    assert torch.equal(first, second)


def test_hse_embedding_training_keeps_random_patch_sampling(monkeypatch):
    model = E_01_HSE(_embedding_args())
    model.train()
    x = torch.randn(2, 12, 3)
    calls = []

    def fixed_random_start(low, high, size, *, device):
        calls.append((low, high, size, device))
        return torch.zeros(size, dtype=torch.long, device=device)

    monkeypatch.setattr(torch, "randint", fixed_random_start)
    output = model(x, fs=12_000.0)

    assert output.shape == (2, 5, 8)
    assert len(calls) == 2


def test_hse_eval_augmentation_is_deterministic_and_rng_independent():
    instance = _new_task_for_methods()
    instance.args_task = _task_args(
        augmentation_type="noise",
        augmentation_noise_std=0.2,
    )
    features = torch.arange(24, dtype=torch.float32).reshape(4, 6)

    torch.manual_seed(123)
    rng_before = torch.random.get_rng_state()
    first = instance._create_augmented_view(
        features,
        stage="val",
        batch_idx=3,
    )
    rng_after = torch.random.get_rng_state()
    assert torch.equal(rng_before, rng_after)

    torch.manual_seed(999)
    second = instance._create_augmented_view(
        features,
        stage="val",
        batch_idx=3,
    )
    test_view = instance._create_augmented_view(
        features,
        stage="test",
        batch_idx=3,
    )

    assert torch.equal(first, second)
    assert not torch.equal(first, test_view)


def test_hse_training_augmentation_remains_stochastic():
    instance = _new_task_for_methods()
    instance.args_task = _task_args(
        augmentation_type="noise",
        augmentation_noise_std=0.2,
    )
    features = torch.arange(24, dtype=torch.float32).reshape(4, 6)

    torch.manual_seed(1)
    first = instance._create_augmented_view(
        features,
        stage="train",
        batch_idx=0,
    )
    torch.manual_seed(2)
    second = instance._create_augmented_view(
        features,
        stage="train",
        batch_idx=0,
    )

    assert not torch.equal(first, second)
