from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

from phmfactory import cli
from phmfactory.config import resolve_config
from src.data_factory import build_data
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


def _write_decoupling_signal(path: Path, reader_name: str, offset: float) -> None:
    if reader_name == "Dummy_Data":
        header = "index,ch1,ch2"
    else:
        header = "time,sensor_a,sensor_b"
    rows = [header]
    for index in range(64):
        rows.append(
            f"{index},{offset + 0.01 * index},{offset + 0.02 * index}"
        )
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")


def _decoupling_data(
    tmp_path: Path,
    reader_name: str,
    dataset_id: int,
) -> tuple[SimpleNamespace, SimpleNamespace]:
    raw_dir = tmp_path / "raw" / reader_name
    raw_dir.mkdir(parents=True)
    files = (
        (1, "source_normal.csv", 0, 0, 0.0),
        (2, "source_fault.csv", 0, 1, 1.0),
        (3, "target_normal.csv", 1, 0, 2.0),
        (4, "target_fault.csv", 1, 1, 3.0),
    )
    for _, filename, _, _, offset in files:
        _write_decoupling_signal(raw_dir / filename, reader_name, offset)

    metadata_lines = [
        "Id,Name,File,Dataset_id,Domain_id,Label,Sample_Rate"
    ]
    for file_id, filename, domain_id, label, _ in files:
        metadata_lines.append(
            f"{file_id},{reader_name},{filename},{dataset_id},"
            f"{domain_id},{label},1000"
        )
    (tmp_path / "metadata.csv").write_text(
        "\n".join(metadata_lines) + "\n",
        encoding="utf-8",
    )

    data_values = {
        "factory_name": "default",
        "data_dir": str(tmp_path),
        "metadata_file": "metadata.csv",
        "batch_size": 2,
        "num_workers": 0,
        "train_ratio": 0.5,
        "val_ratio": 0.5,
        "test_ratio": 0.0,
        "unused_ratio": 0.0,
        "normalization": "none",
        "window_size": 16,
        "window_sampling_strategy": "evenly_spaced",
        "num_window": 4,
        "window_sampling_seed": 0,
        "dtype": "float32",
        "pin_memory": False,
    }
    if reader_name == "CSV_Signal":
        data_values["csv_signal_columns"] = ["sensor_a", "sensor_b"]
        data_values["csv_delimiter"] = ","

    args_task = SimpleNamespace(
        type="DG",
        name="classification",
        target_system_id=[dataset_id],
        source_domain_id=[0],
        target_domain_id=[1],
        loss="CE",
        metrics=["acc"],
        optimizer="adam",
        lr=1e-3,
        weight_decay=0.0,
    )
    return SimpleNamespace(**data_values), args_task


def _decoupling_model(model_kind: str) -> SimpleNamespace:
    if model_kind == "linear":
        return SimpleNamespace(
            type="Baseline",
            name="GlobalAverageLinear",
            input_dim=2,
        )
    return SimpleNamespace(
        type="ISFM",
        name="M_01_ISFM",
        embedding="E_01_HSE",
        backbone="B_04_Dlinear",
        task_head="H_01_Linear_cla",
        input_dim=2,
        d_model=16,
        output_dim=8,
        num_heads=2,
        num_layers=1,
        e_layers=1,
        d_ff=16,
        dropout=0.0,
        activation="relu",
        patch_size_L=4,
        patch_size_C=1,
        num_patches=4,
        use_prompt=False,
        prompt_dim=0,
    )


@pytest.mark.parametrize(
    ("reader_name", "dataset_id"),
    (("Dummy_Data", 21), ("CSV_Signal", 22)),
)
@pytest.mark.parametrize("model_kind", ("linear", "isfm"))
def test_two_by_two_data_model_factory_matrix_backpropagates(
    tmp_path: Path,
    reader_name: str,
    dataset_id: int,
    model_kind: str,
) -> None:
    case_root = tmp_path / f"{reader_name}_{model_kind}"
    args_data, args_task = _decoupling_data(
        case_root,
        reader_name,
        dataset_id,
    )
    data_factory = build_data(args_data, args_task)
    try:
        metadata = data_factory.get_metadata()
        args_model = _decoupling_model(model_kind)
        model = build_model(args_model, metadata=metadata)
        task = build_task(
            args_task=args_task,
            network=model,
            args_data=args_data,
            args_model=args_model,
            args_trainer=SimpleNamespace(device="cpu", gpus=1),
            args_environment=SimpleNamespace(seed=0, project="factory_matrix"),
            metadata=metadata,
        )

        batch = next(iter(data_factory.get_dataloader("train")))
        loss = task._shared_step(batch, "train")["train_total_loss"]

        assert loss.shape == ()
        assert torch.isfinite(loss)
        assert loss.requires_grad
        loss.backward()
        gradients = [
            parameter.grad
            for parameter in model.parameters()
            if parameter.requires_grad and parameter.grad is not None
        ]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert data_factory.split_summary["file_overlap"]["train_test"] == []
    finally:
        data_factory.data.close()


def test_transparent_dummy_config_runs_full_cpu_lifecycle(tmp_path: Path) -> None:
    result = cli.main(
        [
            "--config",
            "configs/demo/00_smoke/dummy_global_average_linear.yaml",
            "--override",
            f"environment.output_dir={tmp_path / 'outputs'}",
            "--override",
            "environment.iterations=1",
            "--override",
            "trainer.num_epochs=1",
        ]
    )

    assert isinstance(result, dict)
    assert result["status"] == "succeeded"
    assert len(result["iterations"]) == 1

    result_dir = Path(result["result_dir"])
    best_checkpoint = Path(result["best_checkpoint"])
    test_metrics = Path(result["test_metrics"])
    run_summary = Path(result["run_summary"])

    assert result_dir.is_dir()
    assert best_checkpoint.is_file()
    assert test_metrics.is_file()
    assert run_summary.is_file()
    assert best_checkpoint in {
        Path(path) for path in result["best_checkpoints"]
    }

    numeric = [
        float(value)
        for value in result["iterations"][0].values()
        if not isinstance(value, bool) and isinstance(value, (int, float))
    ]
    assert numeric
    assert all(torch.isfinite(torch.tensor(value)) for value in numeric)

    primary_metrics = result["primary_metrics"]
    assert primary_metrics
    for metric in primary_metrics.values():
        assert metric["count"] == 1
        assert torch.isfinite(torch.tensor(float(metric["mean"])))
