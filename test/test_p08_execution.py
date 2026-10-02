"""Real HSE on explicit synthetic data: protocol/implementation checks only."""
from __future__ import annotations

import copy
import csv
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch
import yaml

from src.task_factory.task.DG import p08_physical as runner


def _csv(path: Path, records: list[dict]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


@pytest.fixture
def config(tmp_path):
    torch.set_num_threads(1)
    records, ontology = [], []
    for system in ("1", "13", "19"):
        for label, name in enumerate(("healthy", "inner race", "outer race")):
            ontology.append(dict(system_id=system, raw_label=str(label), label=label,
                                 definition=name, source="Synthetic test ontology"))
            for unit in range(4):
                record_id = f"{system}-{unit}-{label}"
                x = np.sin(np.linspace(0, (label + 1) * 7, 64)).astype("float32")
                np.save(tmp_path / f"{record_id}.npy", x)
                records.append(dict(record_id=record_id, system_id=system,
                    physical_unit_id=f"unit-{unit}", raw_label=str(label), sampling_rate=100 + int(system),
                    signal_path=f"{record_id}.npy", signal_key="fixture", channel=0,
                    speed_rpm=1000 + 100 * unit + 3 * label, material="steel"))
    _csv(tmp_path / "records.csv", records)
    _csv(tmp_path / "ontology.csv", ontology)
    return dict(pipeline="Pipeline_01_Fault_Diagnosis",
        environment=dict(project="p08_fixture", seed=42, iterations=1, output_dir=str(tmp_path / "results")),
        data=dict(data_dir=str(tmp_path), metadata_file="records.csv", evidence_kind="tensor_fixture",
            record_inventory=str(tmp_path / "records.csv"), ontology_file=str(tmp_path / "ontology.csv"),
            qualification_file=None, continuous_fields=["speed_rpm"], categorical_fields=["material"],
            window_points=32, stride_points=32, normalization="per_window_standardize", validation_fraction=.25),
        model=dict(type="ISFM", name="M_P08_PhysicalConditioning", embedding="E_01_HSE", backbone="TransformerEncoderLayer",
            task_head="SharedLinear", output_dim=8, nhead=2, patch_size_L=8,
            num_patches=2, num_classes=3, condition_dim=None, coordinates="physical", fusion="film"),
        task=dict(type="DG", name="p08_physical", loss="CE", systems=["1", "13", "19"],
            seeds=[42, 123], split_seed=42, tuning_seed=42, arms=["B1", "P0"], comparison="tuned",
            search_learning_rates=[.0003, .001, .003], search_weight_decays=[0., .0001], bootstrap_replicates=10),
        trainer=dict(name="p08_source_only", device="cpu", devices=1, num_epochs=1, updates_per_epoch=2, batch_size=2, test_after_fit=True))


def _run(config, command, path, **kwargs):
    return runner.execute(config, command, path, target="19", seed=42, **kwargs)


def test_source_tuning_never_loads_target_and_reuses_only_fixed_tuning_seed(config, tmp_path, monkeypatch):
    accessed = []
    actual = runner.windows
    def guarded(record, cfg):
        assert record["system_id"] != "19", "Target signal read during source-only tuning"
        accessed.append(record["record_id"])
        return actual(record, cfg)
    monkeypatch.setattr(runner, "windows", guarded)
    result = _run(config, "tune", tmp_path / "tune", arms=["B1"])
    assert accessed and len(result["results"]) == 1
    selection = json.loads((tmp_path / "tune/target-19/seed-42/arm-B1/selection.json").read_text())
    assert len(selection["trials"]) == 6
    assert all(t["updates_executed"] == 2 for t in selection["trials"])
    assert "target_metrics" not in json.dumps(selection)
    assert selection["source_validation_brier"] == min(t["source_validation_brier"] for t in selection["trials"])
    assert runner._selected(tmp_path / "tune", config, "19", 123, "B1")["seed"] == 42
    with pytest.raises(ValueError, match="tuning_seed"):
        runner.execute(config, "tune", tmp_path / "badseed", target="19", seed=123, arms=["B1"])


def test_selected_checkpoint_reproduces_predictions_and_ablation_never_retrains(config, tmp_path, monkeypatch):
    config["task"]["search_learning_rates"] = [.001]
    config["task"]["search_weight_decays"] = [0.]
    _run(config, "tune", tmp_path / "tune")
    output = _run(config, "compare", tmp_path / "compare", selection=tmp_path / "tune")
    assert len(output["results"]) == 2
    path = tmp_path / "compare/target-19/seed-42/arm-P0"
    checkpoint, model, encoder = runner._load_checkpoint(path / "checkpoint.pt", "19", 42, "P0", torch.device("cpu"))
    train, val, target = runner._split(config, runner.load_records(config["data"]), "19", 42)
    restored = runner.predict(model, target, encoder, config, torch.device("cpu"))
    assert restored == json.loads((path / "predictions.json").read_text())
    assert checkpoint["train_record_ids"] == [r["record_id"] for r in train]
    assert (path / "metrics.json").is_file()
    assert json.loads((tmp_path / "compare/contrasts.json").read_text())[0]["contrast"] == "P0_minus_B1"
    def forbid_fit(*args, **kwargs):
        raise AssertionError("An intervention must not retrain")
    monkeypatch.setattr(runner, "fit", forbid_fit)
    _run(config, "ablate", tmp_path / "ablate", arms=["P0"], selection=tmp_path / "compare")
    ablation = tmp_path / "ablate/target-19/seed-42/arm-P0"
    assert json.loads((ablation / "predictions_correct.json").read_text()) == restored
    assert json.loads((ablation / "wrong.json").read_text())["status"] == "not_identifiable"
    assert json.loads((ablation / "metrics_correct.json").read_text())["mean_representation_l2_from_correct"] == 0
    with pytest.raises(ValueError, match="target, seed, or arm"):
        runner._load_checkpoint(path / "checkpoint.pt", "19", 123, "P0", torch.device("cpu"))


def test_fixed_split_across_training_seeds_and_matched_hyperparameters(config, tmp_path):
    records = runner.load_records(config["data"])
    assert runner._split(config, records, "19", 42) == runner._split(config, records, "19", 123)
    config["task"]["search_learning_rates"] = [.001]
    config["task"]["search_weight_decays"] = [0.]
    _run(config, "tune", tmp_path / "tune", arms=["B1"])
    matched = copy.deepcopy(config)
    matched["task"]["comparison"] = "matched"
    for arm in ("B0", "B1", "F01", "P0"):
        assert runner._selected(tmp_path / "tune", matched, "19", 123, arm)["arm"] == "B1"
    result = runner.execute(matched, "compare", tmp_path / "matched", target="19", seed=123,
        arms=["B0", "B1", "F01", "P0"], selection=tmp_path / "tune")
    assert len(result["results"]) == 4
    contrasts = json.loads((tmp_path / "matched/contrasts.json").read_text())
    assert {r["contrast"] for r in contrasts} == {"delta_H", "delta_P", "delta_H_at_neutral",
        "delta_H_at_physical", "delta_P_at_index", "delta_P_at_physical", "interaction"}


def test_matched_marginal_effects_paired_ci_and_complete_seed_means(config, tmp_path):
    config["task"]["comparison"] = "matched"
    # Every physical unit has the same three-label confusion pattern. Resampling
    # units therefore yields an exact point interval for each fitted-model contrast.
    predictions = {"B0": [0, 0, 0], "B1": [0, 1, 1], "F01": [1, 2, 0], "P0": [0, 1, 2]}
    expected = {"delta_H": 25 / 36, "delta_P": 5 / 36,
                "delta_H_at_neutral": 7 / 18, "delta_H_at_physical": 1.,
                "delta_P_at_index": -1 / 6, "delta_P_at_physical": 4 / 9,
                "interaction": 11 / 18}
    results = []
    for seed in (42, 123):
        for arm, predicted in predictions.items():
            rows = []
            for unit in range(3):
                for label in range(3):
                    chosen = predicted[label] if seed == 42 else label
                    rows.append(dict(record_id=f"{unit}-{label}", system_id="19",
                        physical_unit_id=str(unit), label=label,
                        probabilities=[float(i == chosen) for i in range(3)]))
            directory = tmp_path / f"{seed}-{arm}"
            directory.mkdir()
            runner.write_json(directory / "predictions.json", rows)
            results.append(dict(target="19", seed=seed, arm=arm, directory=directory.name,
                                metrics=runner.scores(rows, 3)))
    runner._contrasts(results[:4], config, tmp_path)
    for filename in ("per_system_contrasts.csv", "per_system_summary.csv"):
        with (tmp_path / filename).open() as stream:
            assert list(csv.DictReader(stream)) == []
    runner._contrasts(results, config, tmp_path)
    contrasts = json.loads((tmp_path / "contrasts.json").read_text())
    for row in contrasts:
        value = expected[row["contrast"]] if row["seed"] == 42 else 0.
        assert row["delta_macro_f1"] == pytest.approx(value)
        assert row["ci95_conditional_on_fitted_models"] == pytest.approx([value, value])
    with (tmp_path / "per_system_contrasts.csv").open() as stream:
        means = list(csv.DictReader(stream))
    assert len(means) == 7
    for row in means:
        assert int(row["seed_count"]) == 2
        assert float(row["mean_delta_macro_f1"]) == pytest.approx(expected[row["contrast"]] / 2)
        assert float(row["seed_std"]) == pytest.approx(abs(expected[row["contrast"]]) / np.sqrt(2))
    with (tmp_path / "per_system_summary.csv").open() as stream:
        summaries = {r["arm"]: r for r in csv.DictReader(stream)}
    for arm, mean in {"B0": 7 / 12, "B1": 7 / 9, "F01": .5, "P0": 1.}.items():
        assert float(summaries[arm]["mean_record_macro_f1"]) == pytest.approx(mean)
        assert int(summaries[arm]["seed_count"]) == 2


def test_failed_selection_and_bad_seed_are_preserved(config, tmp_path):
    with pytest.raises(FileNotFoundError):
        _run(config, "compare", tmp_path / "missing", selection=tmp_path / "absent", arms=["P0"])
    assert "FileNotFoundError" in (tmp_path / "missing/failed.json").read_text()
    with pytest.raises(ValueError, match="outside the fixed seed"):
        runner.execute(config, "smoke", tmp_path / "seed", target="19", seed=77, arms=["B1"])
    assert (tmp_path / "seed/failed.json").is_file()


def test_missing_comparator_fails_before_any_target_read(config, tmp_path, monkeypatch):
    config["task"]["search_learning_rates"] = [.001]
    config["task"]["search_weight_decays"] = [0.]
    _run(config, "tune", tmp_path / "tune", arms=["B1"])
    actual = runner.windows
    def guarded(record, cfg):
        assert record["system_id"] != "19"
        return actual(record, cfg)
    monkeypatch.setattr(runner, "windows", guarded)
    with pytest.raises(FileNotFoundError):
        _run(config, "compare", tmp_path / "compare", arms=["B1", "P0"], selection=tmp_path / "tune")
    assert not (tmp_path / "compare/summary.json").exists()


def test_full_matrix_and_empty_or_partial_benchmark_fail_closed(config, tmp_path):
    config["task"]["seeds"] = [42]
    config["task"]["search_learning_rates"] = [.001]
    config["task"]["search_weight_decays"] = [0.]
    runner.execute(config, "tune", tmp_path / "tune")
    complete = runner.execute(config, "benchmark", tmp_path / "benchmark", selection=tmp_path / "tune")
    assert len(complete["results"]) == 6
    interventions = runner.execute(config, "ablate", tmp_path / "ablate", arms=["P0"], selection=tmp_path / "benchmark")
    assert len(interventions["results"]) == 12
    with pytest.raises(ValueError, match="complete declared"):
        runner.execute(config, "benchmark", tmp_path / "partial", arms=["B1"], selection=tmp_path / "tune")
    config["task"]["seeds"] = []
    with pytest.raises(ValueError, match="nonempty"):
        runner.execute(config, "benchmark", tmp_path / "empty", selection=tmp_path / "tune")
    assert not (tmp_path / "empty/summary.json").exists()


def test_target_split_injected_into_source_is_rejected(config, monkeypatch):
    records = runner.load_records(config["data"])
    train, val, target = runner._split(config, records, "19", 42)
    monkeypatch.setattr(runner, "split_records", lambda *args: (train, val + target[:1], target[1:]))
    with pytest.raises(ValueError, match="Target record leaked"):
        runner._split(config, records, "19", 42)


def test_data_check_visits_each_fixed_split_once(config, tmp_path):
    result = runner.execute(config, "data-check", tmp_path / "sanity")
    assert len(result["results"]) == 3
    assert {r["seed"] for r in result["results"]} == {42}
    artifact = json.loads((tmp_path / "sanity/target-19/seed-42/data_check.json").read_text())
    assert artifact["target_reads_are_sanity_only"] is True
    assert len(artifact["records"]["test_sanity_only"]) == 12


def test_overfit_failure_is_explicit_and_source_only(config, tmp_path, monkeypatch):
    config["trainer"]["overfit_steps"] = 1
    actual = runner.windows
    def guarded(record, cfg):
        assert record["system_id"] != "19"
        return actual(record, cfg)
    monkeypatch.setattr(runner, "windows", guarded)
    with pytest.raises(RuntimeError, match="Small-data overfit failed"):
        _run(config, "overfit", tmp_path / "overfit", arms=["B1"])
    result = json.loads((tmp_path / "overfit/target-19/seed-42/arm-B1/overfit.json").read_text())
    assert result["status"] == "failed"
    assert result["updates_executed"] == 1
    assert not (tmp_path / "overfit/summary.json").exists()


def test_wrong_condition_noop_is_not_reported_as_an_intervention(config, tmp_path):
    config["task"]["search_learning_rates"] = [.001]
    config["task"]["search_weight_decays"] = [0.]
    records = runner.load_records(config["data"])
    train, _, targets = runner._split(config, records, "19", 42)
    donor_rows = []
    for record in targets:
        matching = [r for r in train if r["speed_rpm"] == record["speed_rpm"]]
        donor = matching[0] if matching else train[0]
        donor_rows.append(dict(record_id=record["record_id"], donor_record_id=donor["record_id"],
                               compatible="true", compatibility_basis="Synthetic compatible fixture"))
    manifest = tmp_path / "donors.csv"
    _csv(manifest, donor_rows)
    config["data"]["wrong_condition_donors"] = str(manifest)
    _run(config, "tune", tmp_path / "tune", arms=["P0"])
    _run(config, "compare", tmp_path / "compare", arms=["P0"], selection=tmp_path / "tune")
    _run(config, "ablate", tmp_path / "ablate", arms=["P0"], selection=tmp_path / "compare")
    path = tmp_path / "ablate/target-19/seed-42/arm-P0"
    assert json.loads((path / "wrong.json").read_text())["status"] == "not_identifiable"
    assert not (path / "predictions_wrong.json").exists()


def test_failed_comparison_residual_checkpoint_cannot_enter_ablation(config, tmp_path, monkeypatch):
    config["task"]["search_learning_rates"] = [.001]
    config["task"]["search_weight_decays"] = [0.]
    config["trainer"].update(num_epochs=2, updates_per_epoch=1)
    _run(config, "tune", tmp_path / "tune", arms=["P0"])
    original_step = torch.optim.AdamW.step
    steps = 0
    def second_step_poison(self, *args, **kwargs):
        nonlocal steps
        steps += 1
        result = original_step(self, *args, **kwargs)
        if steps == 2:
            with torch.no_grad():
                self.param_groups[0]["params"][0].fill_(float("nan"))
        return result
    monkeypatch.setattr(torch.optim.AdamW, "step", second_step_poison)
    with pytest.raises(FloatingPointError, match="optimizer parameters"):
        _run(config, "compare", tmp_path / "compare", arms=["P0"], selection=tmp_path / "tune")
    residual = tmp_path / "compare/target-19/seed-42/arm-P0"
    assert (residual / "checkpoint.pt").is_file()
    assert not (residual / "result.json").exists()
    actual_windows = runner.windows
    def guarded(record, cfg):
        assert record["system_id"] != "19", "Residual checkpoint reached a target forward"
        return actual_windows(record, cfg)
    monkeypatch.setattr(runner, "windows", guarded)
    with pytest.raises(ValueError, match="residual checkpoint"):
        _run(config, "ablate", tmp_path / "ablate", arms=["P0"], selection=tmp_path / "compare")
    assert not (tmp_path / "ablate/summary.json").exists()


def test_source_fit_rejects_target_and_nonfinite_optimizer(config, tmp_path, monkeypatch):
    records = runner.load_records(config["data"])
    train, val, target = runner._split(config, records, "19", 42)
    with pytest.raises(ValueError, match="target record"):
        runner.fit(config, train + target[:1], val, target="19", seed=42, arm="P0", lr=.001,
            weight_decay=0., output=tmp_path / "leak", device=torch.device("cpu"), epochs=1, updates=1)
    def poison(self, *args, **kwargs):
        with torch.no_grad():
            self.param_groups[0]["params"][0].fill_(float("nan"))
    monkeypatch.setattr(torch.optim.AdamW, "step", poison)
    with pytest.raises(FloatingPointError, match="optimizer parameters"):
        _run(config, "smoke", tmp_path / "nan", arms=["B1"])
    assert "FloatingPointError" in (tmp_path / "nan/failed.json").read_text()


def test_device_request_has_no_fallback(config, monkeypatch):
    config["trainer"]["device"] = "cuda"
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    with pytest.raises(RuntimeError, match="physical GPU 0"):
        runner.requested_device(config)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="no CPU fallback"):
        runner.requested_device(config)


def test_source_brier_balances_units_and_systems_not_record_counts():
    rows = [dict(record_id="a", system_id="1", physical_unit_id="u1", label=0, probabilities=[1, 0, 0]),
            dict(record_id="b", system_id="1", physical_unit_id="u2", label=0, probabilities=[0, 1, 0]),
            dict(record_id="c", system_id="2", physical_unit_id="u3", label=0, probabilities=[1, 0, 0])]
    rows += [{**rows[0], "record_id": str(i)} for i in range(9)]
    metric = runner.scores(rows, 3)
    assert metric["source_equal_system_equal_unit_record_brier"] == .25
    assert metric["record_balanced_accuracy"] == pytest.approx((11 / 12) / 3)


def test_cli_uses_public_config_precedence_and_preserves_existing_runs(config, tmp_path):
    from scripts.p08_physical import main
    base = tmp_path / "experiment.yaml"
    local = tmp_path / "local.yaml"
    base.write_text(yaml.safe_dump(config))
    local.write_text(yaml.safe_dump({"model": {"output_dim": 16}}))
    output = tmp_path / "cli"
    assert main(["smoke", "--config", str(base), "--local-config", str(local),
                 "--override", "model.output_dim=8", "--output", str(output),
                 "--target", "19", "--seed", "42", "--arms", "B1"]) == 0
    assert json.loads((output / "config.json").read_text())["model"]["output_dim"] == 8
    _run(config, "smoke", output, arms=["P0"])
    assert len(json.loads((output / "summary.json").read_text())["results"]) == 2
    with pytest.raises(FileExistsError):
        _run(config, "smoke", output, arms=["B1"])


def test_retired_alternate_entrypoint_does_not_run_another_protocol():
    from scripts.p08_experiments import main
    with pytest.raises(SystemExit, match="alternate P08 native protocol is retired"):
        main()
