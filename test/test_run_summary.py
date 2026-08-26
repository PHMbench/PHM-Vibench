from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from src.runtime import classification
from src.runtime.classification import _result_row
from src.utils.run_summary import (
    build_run_summary,
    normalize_metric_result,
    resolved_config_sha256,
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
        task=SimpleNamespace(type="DG", name="classification"),
        trainer=SimpleNamespace(test_after_fit=True),
    )


def test_summary_records_complete_seed_statistics():
    results = [
        {"test_acc": 0.5, "test_loss": 2.0},
        {"test_acc": 0.7, "test_loss": 4.0},
    ]
    summary = build_run_summary(results, seeds=[42, 43], config=_config())

    assert summary["config_sha256"] == resolved_config_sha256(_config())
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
    with pytest.raises(TypeError, match="seed 0 must be an integer"):
        build_run_summary([{"test_acc": 0.5}], [42.5], _config())
    with pytest.raises(TypeError, match="seed 0 must be an integer"):
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

    class DataFactory:
        data = SimpleNamespace(close=lambda: None)

        def get_metadata(self):
            return {0: {"Label": 0, "Domain_id": 0}}

        def get_dataloader(self, split: str):
            return split

    class Trainer:
        def __init__(self, path: str, value: float):
            self.value = value
            self.checkpoint = Path(path) / "best.ckpt"
            self.checkpoint.write_text("checkpoint\n", encoding="utf-8")

        def fit(self, *args):
            return None

        def test(self, *args):
            return [{"test_acc": self.value}]

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
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(classification, "build_task", lambda **kwargs: object())
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
