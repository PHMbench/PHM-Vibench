"""CPU fixtures validate scientific operators, not PHM performance claims."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

from phmfactory.config import analyze_config
from src.model_factory.X_model.P07OperatorPath import OperatorNet
from src.task_factory.task.DG.operator_path import execute
from src.task_factory.task.DG.operator_path import operators, runtime


@pytest.fixture
def fixture(tmp_path):
    torch.set_num_threads(1)
    rng = np.random.default_rng(17)
    rows = [(domain, label, window) for domain in range(4) for label in range(2) for window in range(2)]
    x = np.stack([rng.normal(size=(32, 1)) + label for domain, label, window in rows]).astype(np.float32)
    data = dict(x=x, y=np.asarray([r[1] for r in rows], dtype=np.int64),
                unit_id=np.asarray([f"d{d}-c{c}" for d, c, w in rows]),
                domain=np.asarray([str(d) for d, c, w in rows]),
                split=np.asarray(["train" if d < 2 else "val" if d == 2 else "test" for d, c, w in rows]),
                sampling_rate=np.asarray(512.0))
    full = tmp_path / "full.npz"
    source = tmp_path / "source.npz"
    np.savez(full, **data)
    source_data = operators.subset(data, np.flatnonzero(data["split"] != "test"))
    np.savez(source, **source_data)
    config = analyze_config("configs/experiments/p07/operator_path.yaml").runtime_config()
    settings = config["task"]["operator_path"]
    settings.update(arms=["sparse", "routing_concentration", "proposed"], seeds=[7], tuning_seeds=[11],
                    search=[{"lr": 0.001}], intervention_limit=3)
    settings["training"].update(epochs=1, batch_size=3)
    return config, data, full, source


def checkpoint(tmp_path, data, *, arm="proposed", accepted=True):
    torch.manual_seed(31)
    model = operators.build(arm, 1, 2).eval()
    with torch.no_grad():
        model.head.weight.mul_(0.01 if accepted else 0)
        model.head.bias.copy_(torch.tensor([5.0, 0.0]) if accepted else torch.zeros(2))
    _, mean, scale = operators.normalize(data)
    path = tmp_path / f"{arm}-{accepted}.pt"
    torch.save(dict(state_dict=model.state_dict(), arm=arm, seed=7, mean=mean.tolist(), scale=scale.tolist(),
                    normalization_dtype=str(data["x"].dtype), source_partitions=operators.source_partitions(data),
                    channels=1, classes=2, selected_epoch=0), path)
    return path


@pytest.mark.parametrize("arm", ["sparse", "proposed", "routing_concentration"])
def test_objective_and_gradients_match_declared_loss(arm):
    torch.manual_seed(3)
    model = operators.build(arm, 1, 2).double()
    other = copy.deepcopy(model)
    x = torch.linspace(-2, 2, 96, dtype=torch.double).reshape(3, 32, 1)
    labels = torch.tensor([0, 1, 0])
    logits, trace = model(x)
    actual = operators.objective(model, logits, labels, trace, arm, 0.17, 0.23)
    ref_logits, ref_trace = other(x)
    expected = torch.nn.functional.cross_entropy(ref_logits, labels)
    if arm == "proposed":
        expected = expected + 0.17 * other.residual_loss(ref_trace)
    elif arm == "routing_concentration":
        expected = expected + 0.23 * other.routing_concentration_loss(ref_trace)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    actual.backward()
    expected.backward()
    for parameter, reference in zip(model.parameters(), other.parameters()):
        torch.testing.assert_close(parameter.grad, reference.grad, rtol=0, atol=0)


def test_source_tuning_training_checkpoint_and_target_access(fixture, tmp_path, monkeypatch):
    config, data, full, source = fixture
    with pytest.raises(ValueError, match="train/val-only"):
        execute(config, "tune", tmp_path / "forbidden", data=full)
    assert not (tmp_path / "forbidden").exists()
    original_train = operators.train
    observed = []

    def source_only(data, *args, **kwargs):
        assert set(data["split"]) == {"train", "val"}
        assert set(data["domain"]) == {"0", "1", "2"}
        observed.append(data["x"].copy())
        return original_train(data, *args, **kwargs)

    monkeypatch.setattr(operators, "train", source_only)
    tuned = execute(config, "tune", tmp_path / "tune", data=source)
    trained = execute(config, "train", tmp_path / "train", data=source, selection=Path(tuned["selection"]))
    assert len(observed) == 6
    assert len(trained["runs"]) == 3
    assert "test_metrics" not in trained
    path = Path(trained["runs"][-1]["best_checkpoint"])
    saved, model, normalized = runtime.load_checkpoint(path, data, torch.device("cpu"))
    np.testing.assert_array_equal(np.asarray(saved["mean"], dtype=np.float32), operators.normalize(data)[1])
    with (path.parent / "validation_predictions.csv").open() as stream:
        validation = list(csv.DictReader(stream))
    with torch.no_grad():
        actual = model(normalized[data["split"] == "val"])[0].argmax(-1).tolist()
    assert actual == [int(row["prediction"]) for row in validation]
    assert saved["selected_epoch"] == 0


def test_competitive_and_mechanism_selections_are_distinct(fixture):
    config, *_ = fixture
    settings = config["task"]["operator_path"]
    settings["search"] = [{"lr": 0.001}, {"lr": 0.003}]
    chosen = dict(candidate_count=2, tuning_seeds=settings["tuning_seeds"], training=settings["training"],
                  search=settings["search"], arms={
                      "sparse": dict(candidate=0, training=settings["training"] | settings["search"][0]),
                      "proposed": dict(candidate=1, training=settings["training"] | settings["search"][1])})
    assert runtime.selected_training(settings, chosen, "sparse")["lr"] == 0.001
    settings["comparison"] = "mechanism"
    assert runtime.selected_training(settings, chosen, "sparse")["lr"] == 0.003
    settings["training"] = settings["training"] | {"epochs": 2}
    with pytest.raises(ValueError, match="configuration differs"):
        runtime.selected_training(settings, chosen, "sparse")


def test_evaluate_independent_replay_without_reextracting(fixture, tmp_path, monkeypatch):
    config, data, full, _ = fixture
    path = checkpoint(tmp_path, data)
    evaluated = execute(config, "evaluate", tmp_path / "evaluate", data=full, checkpoint=path)
    assert evaluated["test_metrics"]["evaluated_windows"] == 4
    assert evaluated["test_metrics"]["cohort_windows"] == 3
    assert evaluated["test_metrics"]["record_mean_accuracy"] == 0.5
    for summary in evaluated["independent_replay"].values():
        assert summary["prediction_fidelity_given_accepted"] == 1
        assert summary["returned_path_accuracy"] < 1
        assert summary["cohort_windows"] == 3

    def forbidden(*args, **kwargs):
        raise AssertionError("Independent saved-path replay must not perform extraction")

    monkeypatch.setattr(OperatorNet, "extract", forbidden)
    replayed = execute(config, "replay", tmp_path / "replayed", data=full, checkpoint=path,
                       selection=tmp_path / "evaluate/extractions.csv")
    assert evaluated["independent_replay"] == replayed["independent_replay"]


def test_zero_margin_reports_abstention_and_undefined_conditional_rates(fixture, tmp_path):
    config, data, full, _ = fixture
    result = execute(config, "evaluate", tmp_path / "none", data=full,
                     checkpoint=checkpoint(tmp_path, data, accepted=False))
    for summary in result["independent_replay"].values():
        assert summary["explanation_coverage"] == 0
        assert summary["prediction_fidelity_given_accepted"] is None
        assert summary["analytic_sufficient_given_accepted"] is None
        assert summary["analytic_sufficient_rate"] == 0


@pytest.mark.parametrize("arm", ["learned", "unbounded"])
def test_analytical_bound_inapplicable_for_controls(fixture, tmp_path, arm):
    config, data, full, _ = fixture
    config["task"]["operator_path"]["arms"] = [arm]
    result = execute(config, "evaluate", tmp_path / arm, data=full,
                     checkpoint=checkpoint(tmp_path, data, arm=arm))
    assert result["test_metrics"]["analytic_sufficient_rate"] is None
    assert result["independent_replay"]["cost"]["analytic_sufficient_rate"] is None


def test_corrupted_saved_path_and_duplicate_identity_fail(fixture, tmp_path):
    config, data, full, _ = fixture
    path = checkpoint(tmp_path, data)
    execute(config, "evaluate", tmp_path / "original", data=full, checkpoint=path)
    with (tmp_path / "original/extractions.csv").open() as stream:
        rows = list(csv.DictReader(stream))
    corrupted = tmp_path / "corrupted.csv"
    changed = copy.deepcopy(rows)
    changed[0]["path"] = '["UNKNOWN", "I", "I"]'
    operators.write_csv(corrupted, changed)
    with pytest.raises(ValueError, match="declared operator"):
        execute(config, "replay", tmp_path / "bad-path", data=full, checkpoint=path, selection=corrupted)
    operators.write_csv(corrupted, rows + rows[:1])
    with pytest.raises(ValueError, match="unique strategy/sample"):
        execute(config, "replay", tmp_path / "duplicate", data=full, checkpoint=path, selection=corrupted)
    operators.write_csv(corrupted, rows[:-1])
    with pytest.raises(ValueError, match="complete prespecified cohort"):
        execute(config, "replay", tmp_path / "partial", data=full, checkpoint=path, selection=corrupted)


def test_checkpoint_preserves_float64_normalizer_and_source_identity(fixture, tmp_path):
    _, data, _, _ = fixture
    data["x"] = data["x"].astype(np.float64) * 100 + 1e10
    path = checkpoint(tmp_path, data)
    _, _, normalized = runtime.load_checkpoint(path, data, torch.device("cpu"))
    torch.testing.assert_close(normalized, operators.normalize(data)[0], rtol=0, atol=0)
    changed = copy.deepcopy(data)
    # A different export can be internally disjoint yet exchange a source and test unit.
    source = changed["unit_id"] == "d0-c0"
    target = changed["unit_id"] == "d3-c0"
    changed["unit_id"][source] = "d3-c0"
    changed["unit_id"][target] = "d0-c0"
    operators.validate_data(changed, True)
    with pytest.raises(ValueError, match="source unit/label/domain"):
        runtime.load_checkpoint(path, changed, torch.device("cpu"))


def test_physical_unit_and_device_violations_fail(fixture, tmp_path, monkeypatch):
    config, data, full, _ = fixture
    changed = copy.deepcopy(data)
    changed["unit_id"][-1] = changed["unit_id"][0]
    with pytest.raises(ValueError, match="crosses partitions"):
        operators.validate_data(changed, True)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    with pytest.raises(ValueError, match="GPU 2"):
        execute(config, "data-check", tmp_path / "bad-device", data=full, device="cuda")
    monkeypatch.setenv("WORLD_SIZE", "2")
    with pytest.raises(ValueError, match="DDP"):
        execute(config, "data-check", tmp_path / "bad-ddp", data=full)


def test_public_source_export_and_check_use_installed_data_owner(fixture, tmp_path):
    from phmfactory.commands.research import run
    config, data, _, _ = fixture
    records = []
    for index in np.flatnonzero(data["split"] != "test"):
        path = tmp_path / f"record-{index}.npy"
        np.save(path, data["x"][index])
        records.append(dict(path=str(path), label=int(data["y"][index]), unit_id=str(data["unit_id"][index]),
                            domain=str(data["domain"][index]), split=str(data["split"][index])))
    spec = tmp_path / "source-records.json"
    spec.write_text(json.dumps(dict(folds=[dict(dataset="fixture", fold="source", reader="npy", window_size=32,
        stride=32, channels=[0], sampling_rate=512, source_version="test-generated", provenance_notes="CPU fixture only",
        records=records)])))
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config))
    exported = run(["export-source", "--config", str(config_path), "--data", str(spec), "--output", str(tmp_path / "exported")])
    assert exported["source_only"] and exported["status"] == "completed"
    checked = run(["source-check", "--config", str(config_path), "--data", str(tmp_path / "exported/fixture/source.npz"),
                   "--output", str(tmp_path / "checked")])
    assert set(checked["splits"]) == {"train", "val"}
    assert checked["splits"]["train"]["windows"] == 8
    assert checked["splits"]["val"]["units"] == 2
