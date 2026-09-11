from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader

from phmfactory.config import ResolvedConfig, analyze_config
from phmfactory.runtime import CompiledRunSpec
from src.data_factory.dataset_task.Dataset_cluster import IdIncludedDataset
from src.runtime import classification
from src.runtime.classification import _result_row
from src.task_factory.Default_task import Default_task
from src.utils.run_summary import (
    build_run_summary,
    normalize_metric_result,
    write_run_summary,
)


def _config():
    return SimpleNamespace(
        pipeline="Pipeline_01_Fault_Diagnosis",
        environment=SimpleNamespace(seed=42, iterations=2),
        model=SimpleNamespace(type="Transformer", name="TSLTransformer"),
    )


def _runtime_config(tmp_path: Path, *, iterations: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
        pipeline="Pipeline_01_Fault_Diagnosis",
        environment=SimpleNamespace(
            project="runtime-test",
            seed=7,
            iterations=iterations,
            output_dir=str(tmp_path),
        ),
        data=SimpleNamespace(data_dir=str(tmp_path), metadata_file="dummy.csv"),
        model=SimpleNamespace(type="dummy", name="dummy"),
        task=SimpleNamespace(type="DG", name="classification", loss="CE", metrics=["acc"]),
        trainer=SimpleNamespace(test_after_fit=True),
    )


def test_summary_records_complete_seed_statistics():
    results = [
        {"test_acc": 0.5, "test_loss": 2.0},
        {"test_acc": 0.7, "test_loss": 4.0},
    ]
    summary = build_run_summary(results, seeds=[42, 43], config=_config())

    assert "config_sha256" not in summary
    assert summary["iterations"] == 2
    assert summary["seeds"] == [42, 43]
    assert set(summary["metrics"]) == {"test_acc", "test_loss"}
    assert summary["metrics"]["test_acc"]["count"] == 2
    assert summary["metrics"]["test_acc"]["mean"] == pytest.approx(0.6)
    assert summary["metrics"]["test_acc"]["sample_std"] == pytest.approx(2**0.5 / 10)


def test_single_run_uses_null_std_and_writes_strict_json(tmp_path):
    output = tmp_path / "run_summary.json"
    write_run_summary(output, [{"test_acc": 0.5}], [42], _config())
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["iterations"] == 1
    assert payload["metrics"]["test_acc"]["count"] == 1
    assert payload["metrics"]["test_acc"]["sample_std"] is None
    assert "config_sha256" not in payload
    assert output.read_text(encoding="utf-8").endswith("\n")


def test_summary_rejects_missing_seed_and_nonfinite_metrics():
    with pytest.raises(ValueError, match="one seed"):
        build_run_summary([{"value": 1.0}], [], _config())
    with pytest.raises(ValueError, match="not finite"):
        build_run_summary([{"value": float("nan")}], [42], _config())


def test_summary_rejects_metric_key_drift_across_seeds():
    with pytest.raises(
        ValueError,
        match=r"same metric set.*missing=\['test_f1'\]",
    ):
        build_run_summary(
            [
                {"test_acc": 0.5, "test_f1": 0.4},
                {"test_acc": 0.7},
            ],
            seeds=[42, 43],
            config=_config(),
        )

    with pytest.raises(
        ValueError,
        match=r"same metric set.*unexpected=\['test_f1'\]",
    ):
        build_run_summary(
            [
                {"test_acc": 0.5},
                {"test_acc": 0.7, "test_f1": 0.6},
            ],
            seeds=[42, 43],
            config=_config(),
        )


def test_metric_result_rejects_empty_nonnumeric_boolean_and_nonscalar_values():
    with pytest.raises(ValueError, match="at least one metric"):
        normalize_metric_result({})
    with pytest.raises(TypeError, match="scalar real number"):
        normalize_metric_result({"test_acc": "0.5"})
    with pytest.raises(TypeError, match="not boolean"):
        normalize_metric_result({"test_acc": True})

    class VectorLike:
        def item(self):
            raise ValueError("more than one element")

    with pytest.raises(TypeError, match="scalar numeric value"):
        normalize_metric_result({"test_acc": VectorLike()})


def test_summary_rejects_noninteger_seed_values():
    with pytest.raises(ValueError, match="seed 0 must be an integer"):
        build_run_summary([{"test_acc": 0.5}], [42.5], _config())
    with pytest.raises(ValueError, match="seed 0 must be an integer"):
        build_run_summary([{"test_acc": 0.5}], [True], _config())


def test_trainer_test_requires_exactly_one_explicit_population():
    assert _result_row([{"test_acc": 0.5}]) == {"test_acc": 0.5}

    with pytest.raises(RuntimeError, match="exactly one metric mapping"):
        _result_row([])
    with pytest.raises(RuntimeError, match="exactly one metric mapping"):
        _result_row([{"test_acc": 0.5}, {"test_acc": 0.7}])
    with pytest.raises(RuntimeError, match="result 0 must be a metric mapping"):
        _result_row([[0.5]])
    with pytest.raises(TypeError, match="scalar real number"):
        _result_row([{"test_acc": "0.5"}])


def test_invocation_root_is_unique_and_owns_iteration_paths(tmp_path):
    config = _runtime_config(tmp_path)

    first_root, first_name = classification._create_invocation_root(config)
    second_root, second_name = classification._create_invocation_root(config)

    assert first_root.is_dir()
    assert second_root.is_dir()
    assert first_root != second_root
    assert first_name != second_name
    assert {
        classification._iteration_path(first_root, index).parent
        for index in range(3)
    } == {first_root}


def test_pipeline_creates_one_root_for_all_iterations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _runtime_config(tmp_path)
    run_root = tmp_path / "one-run"
    root_calls = 0
    trainer_calls = 0
    metadata = {
        0: {"Name": "A", "Dataset_id": 0, "Label": 0, "Domain_id": 0},
        1: {"Name": "A", "Dataset_id": 0, "Label": 1, "Domain_id": 0},
    }

    class DataFactory:
        data = SimpleNamespace(close=lambda: None)

        def get_metadata(self):
            return metadata

        def get_dataloader(self, split: str):
            return DataLoader(IdIncludedDataset({0: [0], 1: [0]}, metadata))

    class Trainer:
        def __init__(self, path: str, value: float):
            self.value = value
            self.checkpoint = Path(path) / "best.ckpt"
            self.checkpoint.write_text("checkpoint\n", encoding="utf-8")

        def fit(self, *args):
            return None

        def test(self, *args):
            return [{"test_acc_A": self.value}]

    def create_root(configs):
        nonlocal root_calls
        assert configs is config
        root_calls += 1
        run_root.mkdir()
        return run_root, "one-run"

    def build_trainer(*args):
        nonlocal trainer_calls
        trainer_calls += 1
        return Trainer(args[-1], float(trainer_calls))

    monkeypatch.setattr(classification, "load_runtime_config", lambda args: config)
    monkeypatch.setattr(classification, "_create_invocation_root", create_root)
    monkeypatch.setattr(classification, "seed_everything", lambda seed: None)
    monkeypatch.setattr(classification, "init_lab", lambda *args: None)
    monkeypatch.setattr(classification, "close_lab", lambda: None)
    monkeypatch.setattr(classification, "build_data", lambda *args: DataFactory())
    monkeypatch.setattr(
        classification,
        "build_model",
        lambda *args, **kwargs: torch.nn.Linear(1, 2),
    )
    monkeypatch.setattr(classification, "build_task", lambda **kwargs: Default_task(**kwargs))
    monkeypatch.setattr(classification, "build_trainer", build_trainer)
    monkeypatch.setattr(
        classification,
        "load_best_model_checkpoint",
        lambda task, trainer: task,
    )
    monkeypatch.setattr(
        classification,
        "_best_checkpoint_path",
        lambda trainer: trainer.checkpoint.resolve(),
    )

    result = classification.run_classification_pipeline(object())

    assert root_calls == 1
    assert trainer_calls == 3
    assert result["result_dir"] == str(run_root.resolve())
    assert [path.name for path in sorted(run_root.glob("iter_*"))] == [
        "iter_0",
        "iter_1",
        "iter_2",
    ]
    assert (run_root / "all_results.csv").is_file()
    assert (run_root / "run_summary.json").is_file()
    for checkpoint in result["best_checkpoints"]:
        Path(checkpoint).resolve().relative_to(run_root.resolve())


class _UnreadSamples:
    def __len__(self):
        return 1

    def __getitem__(self, index):
        raise AssertionError("Population declaration must not consume a sample.")


@pytest.fixture
def publication_case(tmp_path, monkeypatch):
    """Real config, dataset identities, Task and writer; only training is isolated."""
    metadata = {
        0: {"Name": "A", "Dataset_id": 7, "Label": 0},
        1: {"Name": "A", "Dataset_id": 8, "Label": 1},
        2: {"Name": "B", "Dataset_id": 9, "Label": 0},
        3: {"Name": "B", "Dataset_id": 9, "Label": 1},
        4: {"Name": "Train_only", "Dataset_id": 10, "Label": 0},
        5: {"Name": "Train_only", "Dataset_id": 10, "Label": 1},
    }
    state = SimpleNamespace(tests=0, closed=0, fits=0, tasks=[])

    def run(payloads, *, names=("A",), metrics=("acc", "f1"), evaluated=True):
        analysis = analyze_config("smoke", override_values=[
            f"environment.output_dir={tmp_path.as_posix()}",
            f"environment.iterations={len(payloads)}",
            "environment.seed=17",
            f"task.metrics={list(metrics)!r}",
            f"trainer.test_after_fit={str(evaluated).lower()}",
        ])
        resolved = ResolvedConfig(
            requested="smoke", path=tmp_path / "smoke.yaml",
            data=analysis.runtime_config(), pipeline="Pipeline_01_Fault_Diagnosis",
            overrides={},
        )
        args = SimpleNamespace(
            compiled_run_spec=CompiledRunSpec.compile(resolved),
            resolved_pipeline=resolved.pipeline, config_path=str(resolved.path), notes="",
        )
        selected = {
            file_id: _UnreadSamples() for file_id, row in metadata.items()
            if row["Name"] in names
        }
        dataset = IdIncludedDataset(selected, metadata)
        state.dataset = dataset

        class DataFactory:
            def __init__(self):
                self.data = SimpleNamespace(close=self.close)

            def close(self):
                state.closed += 1

            def get_metadata(self):
                return metadata

            def get_dataloader(self, split):
                if split == "test" and not evaluated:
                    pytest.fail("training-only must not request a test loader")
                return DataLoader(dataset)

        class Trainer:
            def __init__(self, path):
                self.checkpoint = Path(path) / "best.ckpt"

            def fit(self, *args):
                state.fits += 1
                self.checkpoint.write_text("isolated-training checkpoint\n")

            def test(self, task, loader):
                state.tests += 1
                return [dict(payloads[state.tests - 1])]

        def build_task(**kwargs):
            task = Default_task(**kwargs)
            state.tasks.append(task)
            return task

        monkeypatch.setattr(classification, "init_lab", lambda *args: None)
        monkeypatch.setattr(classification, "close_lab", lambda: None)
        monkeypatch.setattr(classification, "seed_everything", lambda seed: None)
        monkeypatch.setattr(classification, "build_data", lambda *args: DataFactory())
        monkeypatch.setattr(classification, "build_model", lambda *a, **k: torch.nn.Linear(1, 2))
        monkeypatch.setattr(classification, "build_task", build_task)
        monkeypatch.setattr(classification, "build_trainer", lambda *args: Trainer(args[-1]))
        monkeypatch.setattr(classification, "load_best_model_checkpoint", lambda task, trainer: task)
        monkeypatch.setattr(classification, "_best_checkpoint_path", lambda trainer: trainer.checkpoint.resolve())
        return classification.run_classification_pipeline(args)

    return run, state


def test_declared_metrics_missing_from_all_seeds_block_first_publication(publication_case, tmp_path, capsys):
    run, state = publication_case
    with pytest.raises(RuntimeError, match="missing_metric_keys=.*test_f1_A") as error:
        run([{"test_acc_A": 0.5}, {"test_acc_A": 0.5}])
    assert "seed=17" in str(error.value)
    assert "configured_metrics" in str(error.value)
    assert "expected_test_datasets=['A']" in str(error.value)
    assert state.tests == state.closed == 1
    assert not list(tmp_path.rglob("test_result_*.csv"))
    assert not list(tmp_path.rglob("all_results.csv"))
    assert not list(tmp_path.rglob("run_summary.json"))
    assert "所有实验已完成" not in capsys.readouterr().out


def test_later_missing_metric_preserves_prior_diagnostics_only(publication_case, tmp_path):
    run, state = publication_case
    with pytest.raises(RuntimeError, match="seed=18.*test_f1_A"):
        run([{"test_acc_A": 0.5, "test_f1_A": 0.4}, {"test_acc_A": 0.5}])
    assert state.tests == state.closed == 2
    assert len(list(tmp_path.rglob("test_result_0.csv"))) == 1
    assert not list(tmp_path.rglob("test_result_1.csv"))
    assert not list(tmp_path.rglob("all_results.csv"))
    assert not list(tmp_path.rglob("run_summary.json"))


def test_missing_test_name_cannot_disappear_from_expectations(publication_case, tmp_path):
    run, state = publication_case
    with pytest.raises(RuntimeError, match="test_acc_B.*test_f1_B"):
        run([{"test_acc_A": 0.5, "test_f1_A": 0.4}], names=("A", "B"))
    assert state.tests == 1
    assert not list(tmp_path.rglob("*.csv"))
    assert not list(tmp_path.rglob("run_summary.json"))


@pytest.mark.parametrize("metrics", [("acc", "f1"), ("accuracy", "F1")])
def test_complete_publication_preserves_aliases_extra_loss_and_name_pooling(publication_case, metrics):
    run, state = publication_case
    payload = {"test_acc_A": 0.5, "test_f1_A": 0.4, "test_loss": 1.0}
    result = run([payload], metrics=metrics)
    assert result["status"] == "succeeded"
    assert Path(result["test_metrics"]).is_file()
    assert Path(result["run_summary"]).is_file()
    assert set(result["primary_metrics"]) == set(payload)
    assert result["primary_metrics"]["test_f1_A"]["sample_std"] is None
    # Distinct Dataset_id values share Name A by existing protocol; train-only Names
    # must not be added to expected test keys merely because they occur in metadata.
    assert state.dataset.expected_dataset_names() == ("A",)
    task = state.tasks[0]
    expected = task.expected_evaluation_metric_keys(("A",))
    logged = task._compute_metrics(torch.tensor([[1., 0.], [0., 1.]]), torch.tensor([0, 1]), "A", "test")
    assert expected == set(logged) == {"test_acc_A", "test_f1_A"}


def test_complete_multiple_names_and_seeds_publish(publication_case):
    run, state = publication_case
    payload = {"test_acc_A": 0.5, "test_f1_A": 0.4, "test_acc_B": 0.6, "test_f1_B": 0.5}
    result = run([payload, payload], names=("A", "B"))
    assert state.tests == state.closed == 2
    summary = json.loads(Path(result["run_summary"]).read_text())
    assert summary["seeds"] == [17, 18]
    assert all(item["count"] == 2 for item in summary["metrics"].values())


def test_acc_only_request_does_not_acquire_an_f1_requirement(publication_case):
    run, _ = publication_case
    result = run([{"test_acc_A": 0.5}], metrics=("acc",))
    assert set(result["primary_metrics"]) == {"test_acc_A"}


def test_training_only_skips_declared_test_closure(publication_case, tmp_path):
    run, state = publication_case
    result = run([{}], evaluated=False)
    assert state.tests == 0
    assert result["status"] == "succeeded"
    assert result["test_metrics"] is result["run_summary"] is None
    assert not list(tmp_path.rglob("*.csv"))


@pytest.mark.parametrize("row", [{}, {"Name": ""}, {"Name": " A"}, {"Name": None}])
def test_expected_populations_reject_missing_or_invalid_names(row):
    dataset = IdIncludedDataset({7: _UnreadSamples()}, {7: row})
    with pytest.raises((KeyError, ValueError), match="Name"):
        dataset.expected_dataset_names()


def test_expected_populations_reject_missing_id_and_empty_selection():
    dataset = IdIncludedDataset({7: _UnreadSamples()}, {})
    with pytest.raises(KeyError, match="file_id=7"):
        dataset.expected_dataset_names()
    dataset.dataset_dict.clear()
    with pytest.raises(ValueError, match="non-empty selected test population"):
        dataset.expected_dataset_names()
