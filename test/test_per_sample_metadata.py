from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from phmfactory.config import resolve_config
from src.model_factory import build_model, resolve_model_module
from src.model_factory.ISFM.system_utils import normalize_fs, resolve_batch_metadata
from src.model_factory.ISFM.task_head.H_01_Linear_cla import H_01_Linear_cla
from src.task_factory import build_task
from src.task_factory.Components.metrics import get_metrics
from src.task_factory.Components.regularization import calculate_regularization
from src.task_factory.Default_task import Default_task
from src.utils.label_ontology import validate_zero_based_contiguous_labels


_METADATA = {
    101: {
        "Name": "Dummy",
        "Dataset_id": 1,
        "Domain_id": 0,
        "Sample_Rate": 12_000,
    },
    102: {
        "Name": "Dummy",
        "Dataset_id": 1,
        "Domain_id": 1,
        "Sample_Rate": 48_000,
    },
}

_EXTENSION_METADATA = {
    1: {
        "Name": "Dummy_Data",
        "Dataset_id": 0,
        "Domain_id": 0,
        "Sample_Rate": 1000,
        "Label": 0,
    },
    2: {
        "Name": "Dummy_Data",
        "Dataset_id": 0,
        "Domain_id": 1,
        "Sample_Rate": 2000,
        "Label": 1,
    },
}


def _task_args(**overrides):
    values = {
        "name": "classification",
        "loss": "CE",
        "metrics": ["acc"],
        "optimizer": "adam",
        "lr": 1e-3,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


class _TaskHarness:
    metadata = _METADATA
    _file_id_values = staticmethod(Default_task._file_id_values)

    def __init__(self):
        self.network = SimpleNamespace()
        self.seen_file_ids = None
        self.seen_sample_rates = None

    def forward(self, batch):
        self.seen_file_ids = batch["file_id"].clone()
        _, sample_rates = resolve_batch_metadata(
            self.metadata,
            batch["file_id"],
            device=batch["x"].device,
        )
        self.seen_sample_rates = sample_rates
        return torch.tensor(
            [[2.0, -1.0], [-1.0, 2.0]],
            dtype=torch.float32,
            requires_grad=True,
        )

    @staticmethod
    def _compute_loss(y_hat, y):
        return F.cross_entropy(y_hat, y)

    @staticmethod
    def _compute_metrics(y_hat, y, data_name, stage):
        return {}

    @staticmethod
    def _compute_regularization():
        return {"total": torch.tensor(0.0)}


def test_default_task_preserves_per_sample_file_ids_and_sampling_rates():
    task = _TaskHarness()
    batch = {
        "x": torch.zeros(2, 8, 1),
        "y": torch.tensor([0, 1]),
        "file_id": torch.tensor([101, 102]),
    }

    metrics = Default_task._shared_step(task, batch, "train")

    assert "task_id" not in batch
    assert torch.equal(batch["file_id"], torch.tensor([101, 102]))
    assert torch.equal(task.seen_file_ids, torch.tensor([101, 102]))
    assert torch.equal(task.seen_sample_rates, torch.tensor([12_000.0, 48_000.0]))
    assert metrics["train_total_loss"].requires_grad


def test_default_task_rejects_partial_file_id_vectors():
    task = _TaskHarness()
    batch = {
        "x": torch.zeros(3, 8, 1),
        "y": torch.tensor([0, 1, 0]),
        "file_id": torch.tensor([101, 102]),
    }

    with pytest.raises(ValueError, match="one ID or one ID per sample"):
        Default_task._shared_step(task, batch, "train")


def test_default_task_rejects_mixed_dataset_identity_before_forward():
    task = _TaskHarness()
    task.metadata = {
        101: {"Name": "Dataset_A", "Dataset_id": 1},
        102: {"Name": "Dataset_B", "Dataset_id": 2},
    }
    batch = {
        "x": torch.zeros(2, 8, 1),
        "y": torch.tensor([0, 1]),
        "file_id": torch.tensor([101, 102]),
    }

    with pytest.raises(ValueError, match="cannot mix dataset identities"):
        Default_task._shared_step(task, batch, "train")


def test_normalize_fs_preserves_vectors_and_rejects_wrong_lengths():
    fs = normalize_fs(
        torch.tensor([12_000.0, 48_000.0]),
        batch_size=2,
        device=torch.device("cpu"),
    )
    assert torch.equal(fs, torch.tensor([12_000.0, 48_000.0]))

    with pytest.raises(ValueError, match="received 2 values for batch_size=3"):
        normalize_fs(
            torch.tensor([12_000.0, 48_000.0]),
            batch_size=3,
            device=torch.device("cpu"),
        )


def test_linear_head_requires_one_system_per_batch():
    head = H_01_Linear_cla(
        SimpleNamespace(num_classes={1: 2, 2: 2}, output_dim=4)
    )
    features = torch.randn(2, 4)

    logits = head(features, system_id=torch.tensor([1, 1]))
    assert logits.shape == (2, 2)

    with pytest.raises(ValueError, match="single Dataset_id per batch"):
        head(features, system_id=torch.tensor([1, 2]))


def test_existing_factory_supports_config_only_model_replacement_and_backward():
    resolved = resolve_config(
        "smoke",
        override_values=(
            "model.type=Baseline",
            "model.name=GlobalAverageLinear",
            "model.input_dim=2",
            "model.num_classes=2",
        ),
    )
    config = resolved.data
    args_environment = SimpleNamespace(**config["environment"])
    args_data = SimpleNamespace(**config["data"])
    args_model = SimpleNamespace(**config["model"])
    args_task = SimpleNamespace(**config["task"])
    args_trainer = SimpleNamespace(**config["trainer"])

    assert resolve_model_module(args_model) == (
        "src.model_factory.Baseline.GlobalAverageLinear"
    )

    model = build_model(args_model, metadata=_EXTENSION_METADATA)
    task = build_task(
        args_task=args_task,
        network=model,
        args_data=args_data,
        args_model=args_model,
        args_trainer=args_trainer,
        args_environment=args_environment,
        metadata=_EXTENSION_METADATA,
    )

    batch = {
        "x": torch.randn(2, 128, 2),
        "y": torch.tensor([0, 1]),
        "file_id": torch.tensor([1, 2]),
    }
    loss = task._shared_step(batch, "train")["train_total_loss"]

    assert "task_id" not in batch
    assert loss.shape == ()
    assert torch.isfinite(loss)
    assert loss.requires_grad
    loss.backward()
    assert model.classifier.weight.grad is not None
    assert torch.isfinite(model.classifier.weight.grad).all()


def test_baseline_model_rejects_incompatible_input_without_padding_or_fallback():
    model = build_model(
        SimpleNamespace(
            type="Baseline",
            name="GlobalAverageLinear",
            input_dim=2,
            num_classes=2,
        ),
        metadata=_EXTENSION_METADATA,
    )

    with pytest.raises(ValueError, match="channel mismatch"):
        model(torch.randn(2, 128, 1))


def test_default_task_uses_configured_task_identity_without_mutating_batch():
    class Network(torch.nn.Module):
        requires_physical_metadata = True

        def __init__(self):
            super().__init__()
            self.received = None

        def forward(self, x, file_id, task_id, *, physical_metadata=None):
            self.received = (x, file_id, task_id, physical_metadata)
            return x

    network = Network()
    task = Default_task(
        network=network,
        args_data=SimpleNamespace(normalization="none"),
        args_model=SimpleNamespace(type="test", name="physical"),
        args_task=_task_args(),
        args_trainer=SimpleNamespace(device="cpu", gpus=1),
        args_environment=SimpleNamespace(project="task_identity"),
        metadata=_EXTENSION_METADATA,
    )
    batch = {
        "x": torch.zeros(2, 16, 1),
        "file_id": torch.tensor([1, 2]),
        "sample_rate_hz": torch.tensor([1000.0, 2000.0]),
        "rotation_speed_rpm": torch.tensor([1797.0, 1772.0]),
        "load_hp": torch.tensor([0.0, 1.0]),
    }

    task.forward(batch)

    assert "task_id" not in batch
    assert network.received is not None
    _, received_ids, received_task, physical = network.received
    assert torch.equal(received_ids, batch["file_id"])
    assert received_task == "classification"
    assert set(physical) == {
        "sample_rate_hz",
        "rotation_speed_rpm",
        "load_hp",
    }

    with pytest.raises(ValueError, match="conflicts with the configured"):
        task.forward({**batch, "task_id": "prediction"})


def test_default_task_delegates_cuda_validation_to_trainer(monkeypatch):
    monkeypatch.setattr(
        torch.cuda,
        "is_available",
        lambda: pytest.fail("Task construction must not inspect CUDA availability"),
    )
    network = torch.nn.Linear(2, 2)
    args_trainer = SimpleNamespace(device="cuda", gpus=1)

    task = Default_task(
        network=network,
        args_data=SimpleNamespace(normalization="none"),
        args_model=SimpleNamespace(type="test", name="cuda_required"),
        args_task=_task_args(),
        args_trainer=args_trainer,
        args_environment=SimpleNamespace(project="cuda_fail_fast"),
        metadata=_EXTENSION_METADATA,
    )

    assert task.network is network
    assert next(task.network.parameters()).device.type == "cpu"
    assert vars(args_trainer) == {"device": "cuda", "gpus": 1}


def test_metric_requests_and_label_ontology_fail_closed():
    with pytest.raises(ValueError, match="Unknown task metric"):
        get_metrics(["acc", "not_a_metric"], _EXTENSION_METADATA)

    nonzero_labels = {
        1: {"Name": "bad", "Dataset_id": 1, "Label": 1},
        2: {"Name": "bad", "Dataset_id": 1, "Label": 2},
    }
    with pytest.raises(ValueError, match="zero-based and contiguous"):
        get_metrics(["acc"], nonzero_labels)

    gapped_labels = {
        1: {"Name": "bad", "Dataset_id": 1, "Label": 0},
        2: {"Name": "bad", "Dataset_id": 1, "Label": 2},
    }
    with pytest.raises(ValueError, match="zero-based and contiguous"):
        build_model(
            SimpleNamespace(
                type="Baseline",
                name="GlobalAverageLinear",
                input_dim=1,
                num_classes=3,
            ),
            metadata=gapped_labels,
        )

    assert validate_zero_based_contiguous_labels(
        [0, 1, 1], context="test"
    ) == 2


def test_regularization_consumes_the_complete_parameter_set():
    first = torch.nn.Parameter(torch.tensor([1.0]))
    second = torch.nn.Parameter(torch.tensor([2.0]))

    result = calculate_regularization(
        {"l2": 1.0},
        iter([first, second]),
    )

    assert torch.equal(result["l2"], torch.tensor(5.0))
    assert torch.equal(result["total"], torch.tensor(5.0))

    with pytest.raises(ValueError, match="Unknown regularization method"):
        calculate_regularization({"elastic": 1.0}, iter([first, second]))
