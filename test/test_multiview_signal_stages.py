"""Mocked native exports test stage contracts; no PHM/GPU benchmark evidence."""
from __future__ import annotations

from copy import deepcopy
import csv
import json
from pathlib import Path

import numpy as np
from scipy.special import softmax

from src.task_factory.task.classification.multiview_signal import native, runtime
from src.task_factory.task.classification.multiview_signal.analysis import analyze
from src.task_factory.task.classification.multiview_signal.features import ProgramSpec, program_features
from src.task_factory.task.classification.multiview_signal.metrics import aggregate_metrics
from src.task_factory.task.classification.multiview_signal.readouts import ReadoutState


def test_complete_source_freeze_compare_target_evaluation_and_analysis(tmp_path, monkeypatch):
    """Exercise actual numerical stages on a complete synthetic three-seed family."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
    seeds, class_names = [11, 23, 37], ["healthy", "inner", "outer"]
    roles = ("fit", "tune", "grow", "select", "test")
    program = ProgramSpec.from_dict(dict(
        sample_rate_hz=4096., bands_hz=[[256., 48.], [512., 72.], [768., 96.], [1024., 128.]],
        modulation_bands_hz=[[16., 48.], [48., 96.], [96., 192.]], lags_samples=[4, 16, 64]))
    role_records = {}
    for role_index, role in enumerate(roles):
        record_ids = [10 * role_index + index + 1 for index in range(6)]
        role_records[role] = {"record_ids": record_ids,
                             "unit_by_record": {str(record_id): f"{role}-bearing-{index}"
                                                for index, record_id in enumerate(record_ids)}}

    def bank(role, seed, model):
        # The observations and fixed signal features are identical across fitted seeds/models.
        rng = np.random.default_rng(101 + roles.index(role))
        windows, labels, units, items = [], [], [], []
        time = np.arange(512) / program.sample_rate_hz
        for entity, record_id in enumerate(role_records[role]["record_ids"]):
            label = entity % 3
            for window in range(2 + entity % 2):
                carrier = np.cos(2 * np.pi * 256 * (label + 1) * time + .2 * window)
                modulation = 1. + (.15 + .12 * label) * np.cos(2 * np.pi * 32 * time)
                observation = carrier * modulation + rng.normal(scale=.12, size=len(time))
                observation = (observation - observation.mean()) / observation.std()
                windows.append(observation[:, None])
                labels.append(label)
                units.append(role_records[role]["unit_by_record"][str(record_id)])
                items.append(f"record-{record_id:03d}:window-{window:02d}")
        windows, labels = np.stack(windows), np.array(labels)
        features, names = program_features(windows, program)
        native_features = np.abs(np.fft.rfft(windows[:, :, 0], axis=1))[:, :64]
        # Deterministic frozen-logit fixtures stand in for independently trained checkpoints.
        rng = np.random.default_rng(seed + roles.index(role) * 100 + (500 if model != "TSPN" else 0))
        logits = .8 * np.eye(3)[labels] + rng.normal(scale=.4, size=(len(labels), 3))
        return dict(logits=logits, y=labels, unit=np.asarray(units), item=np.asarray(items),
                    role=np.full(len(labels), role), split=np.full(len(labels), role),
                    program_features=features, native_features=native_features,
                    feature_names=np.asarray(names), class_names=np.asarray(class_names),
                    training_seed=np.asarray(seed))

    preparation = tmp_path / "prepared"
    preparation.mkdir()
    config = {"environment": {"seed": 11}, "model": {"name": "TSPN", "device": "cpu", "num_classes": 3},
              "trainer": {"device": "cpu", "test_after_fit": False}}
    prepared = {"preparation": str(preparation), "roles": role_records,
                "source_config": deepcopy(config), "evaluation_config": deepcopy(config),
                "study": {"protocol": "condition_dg", "training_seeds": seeds, "class_names": class_names,
                          "evaluation_role": "exploratory", "program": program.to_dict(),
                          "baseline_models": {"TSPN": {}, "ResNet1D": {}}}}
    prepared_path = preparation / "prepared.json"
    prepared_path.write_text(json.dumps(prepared))
    selection = {"prepared": str(prepared_path), "models": {}}
    for model in prepared["study"]["baseline_models"]:
        records = {}
        for seed in seeds:
            directory = tmp_path / "native_fixtures" / model / f"seed_{seed}"
            directory.mkdir(parents=True)
            tune_path = directory / "tune_reference.npz"
            np.savez_compressed(tune_path, **bank("tune", seed, model))
            checkpoint = directory / "checkpoint.fixture"
            checkpoint.write_text("Synthetic export boundary stub; not a trained neural checkpoint.\n")
            native_config = deepcopy(config)
            native_config["model"]["name"] = model
            native_config["environment"]["seed"] = seed
            records[str(seed)] = {"native_config": native_config, "checkpoint": str(checkpoint),
                                  "tune_reference": str(tune_path)}
        selection["models"][model] = {"runs": records}

    source_calls, target_calls = [], []
    target_banks = {}

    def source_export(native_config, checkpoint, requested_roles, program_spec, output, device, *, reference=None):
        assert device == native_config["model"]["device"] == "cpu"
        assert set(requested_roles) == set(roles[:-1])
        assert all(requested_roles[role] == role_records[role] for role in roles[:-1])
        assert Path(checkpoint).is_file() and Path(reference).is_file()
        assert program_spec["class_names"] == class_names
        seed, model = native_config["environment"]["seed"], native_config["model"]["name"]
        assert model == "TSPN"
        source_calls.append((model, seed))
        output.mkdir(parents=True, exist_ok=False)
        arrays = {role: bank(role, seed, model) for role in roles[:-1]}
        with np.load(reference, allow_pickle=False) as saved:
            np.testing.assert_array_equal(arrays["tune"]["logits"], saved["logits"])
            np.testing.assert_array_equal(arrays["tune"]["item"], saved["item"])
        fixed = {"feature_names", "class_names", "training_seed"}
        combined = {key: next(iter(arrays.values()))[key] if key in fixed else
                    np.concatenate([values[key] for values in arrays.values()])
                    for key in next(iter(arrays.values()))}
        for role, values in arrays.items():
            np.savez_compressed(output / f"{role}.npz", **values)
        path = output / "operator_inputs.npz"
        np.savez_compressed(path, **combined)
        return {"operator_inputs": str(path)}

    def target_export(native_config, checkpoint, requested_role, program_spec, output, device, frozen_state):
        assert device == native_config["model"]["device"] == "cpu"
        assert requested_role == role_records["test"]
        frozen = json.loads(Path(frozen_state).read_text())
        assert frozen["target_scored"] is False
        assert frozen["checkpoint"] == str(checkpoint)
        assert frozen["states"] and np.isfinite(frozen["temperature"]) and frozen["temperature"] > 0
        assert program_spec["class_names"] == frozen["class_names"] == class_names
        model, seed = native_config["model"]["name"], native_config["environment"]["seed"]
        target_calls.append((model, seed))
        output.mkdir(parents=True, exist_ok=False)
        values = bank("test", seed, model)
        target_banks[(model, seed)] = values
        path = output / "operator_inputs.npz"
        np.savez_compressed(path, **values)
        return {"operator_inputs": str(path)}

    monkeypatch.setattr(native, "export_native", source_export)
    monkeypatch.setattr(native, "export_target_native", target_export)
    directories = {phase: tmp_path / phase for phase in ("fit", "source_compare", "evaluate", "analyze")}
    for directory in directories.values():
        directory.mkdir()
    frozen = runtime._fit(prepared, selection, directories["fit"])
    assert frozen["status"] == "completed" and len(frozen["freezes"]) == 6
    assert source_calls == [("TSPN", seed) for seed in seeds]
    assert not target_calls and frozen["target_scored"] is False
    snapshot = {path: Path(path).read_bytes() for path in frozen["freezes"]}
    for path in frozen["freezes"]:
        state = json.loads(Path(path).read_text())
        if state["model"] == "TSPN":
            assert len(state["states"]["states"]) == 8
        assert state["seed"] in seeds and state["target_scored"] is False

    source_result = runtime._source_compare(frozen, directories["source_compare"])
    assert source_result["target_scored"] is False and not target_calls
    with Path(source_result["metrics"]).open(newline="") as stream:
        source_rows = list(csv.DictReader(stream))
    assert len(source_rows) == 3 * 25

    result = runtime._evaluate(prepared, frozen, directories["evaluate"])
    evaluation = json.loads(Path(result["selection"]).read_text())
    assert evaluation["status"] == "completed" and len(evaluation["runs"]) == 6
    assert set(target_calls) == {(model, seed) for model in selection["models"] for seed in seeds}
    for row in evaluation["runs"]:
        model, seed = row["model"], row["seed"]
        values = target_banks[(model, seed)]
        frozen_path = directories["fit"] / model / f"seed_{seed}" / "frozen.json"
        saved = json.loads(frozen_path.read_text())
        offset = values["logits"] / saved["temperature"]
        expected = (ReadoutState.from_dict(saved["states"]).predict(
                    values["program_features"], values["native_features"], offset)
                    if model == "TSPN" else {model: softmax(offset, axis=1)})
        with np.load(row["predictions"], allow_pickle=False) as archive:
            assert archive["predictions"].shape == (len(values["y"]), 25 if model == "TSPN" else 1, 3)
            assert archive["methods"].tolist() == list(expected)
            for index, (method, probability) in enumerate(expected.items()):
                np.testing.assert_array_equal(archive["predictions"][:, index], probability)
                metric = aggregate_metrics(probability, archive["y"], archive["unit"])
                assert np.isfinite([metric["brier"], metric["accuracy"], metric["macro_f1"]]).all()
        if model == "TSPN":
            explanation = json.loads((Path(row["predictions"]).parent / "explanation_check.json").read_text())
            assert explanation["raw_logit_max_abs"] <= 1e-10

    analyzed = analyze(prepared, evaluation, directories["analyze"])
    summary = json.loads(Path(analyzed["summary"]).read_text())
    assert len(summary["methods"]) == 26 and summary["methods"][-1] == "ResNet1D"
    assert summary["training_seeds"] == seeds and summary["independent_units"] == 6
    assert summary["bootstrap_seed"] == 20261013
    for key in ("summary_metrics", "paired_effects", "seed_metrics", "contribution_results"):
        assert Path(analyzed[key]).is_file()
    assert {path: Path(path).read_bytes() for path in frozen["freezes"]} == snapshot
