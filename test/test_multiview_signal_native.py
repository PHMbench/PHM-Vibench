"""Native synthetic caches verify source boundaries; no PHM data or GPU is used."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import subprocess

import h5py
import numpy as np
import pandas as pd
import pytest
import torch

from src.task_factory.task.classification.multiview_signal import native
from src.data_factory.splitting import _sha256_file
from src.runtime import classification


PROGRAM = dict(program=dict(sample_rate_hz=6400.0,
                           bands_hz=[[400, 200], [800, 200], [1600, 200], [2400, 200]],
                           modulation_bands_hz=[[50, 150], [150, 500], [500, 1500]],
                           lags_samples=[1, 4, 8]), class_names=["class_0", "class_1"])


def _fixture(tmp_path: Path, *, target: bool = False) -> tuple[dict, dict]:
    data_root = tmp_path / ("evaluation" if target else "source")
    data_root.mkdir()
    roles = [*native.SOURCE_ROLES, "test"] if target else list(native.SOURCE_ROLES)
    rows, partitions = [], {}
    for index, role in enumerate(roles):
        ids = [2 * index + 1, 2 * index + 2]
        partitions[role] = dict(ids=ids, groups=[f"unit_{record_id}" for record_id in ids])
        rows.extend(dict(Id=record_id, Dataset_id=1, Name="Synthetic", File=f"record_{record_id}.npy",
                         Label=record_id % 2, Domain_id=index, Sample_rate=6400,
                         Physical_unit=f"unit_{record_id}", Sample_lenth=256, Channel=1)
                    for record_id in ids)
    metadata = data_root / "metadata.csv"
    pd.DataFrame(rows).to_csv(metadata, index=False)
    manifest = dict(schema_version=1, group_key="Physical_unit", stratify_key="Label",
                    metadata_file_sha256=_sha256_file(metadata), partitions=partitions)
    manifest_path = data_root / "split.json"
    manifest_path.write_text(json.dumps(manifest))
    rng = np.random.default_rng(19)
    with h5py.File(data_root / "Synthetic.h5", "w") as cache:
        for row in rows:
            t = np.arange(256) / 6400
            signal = np.sin(2 * np.pi * (400 + 300 * row["Label"]) * t)
            signal += rng.normal(scale=0.1, size=len(t))
            cache.create_dataset(str(row["Id"]), data=signal[:, None, None])
    config = dict(pipeline="Pipeline_01_Fault_Diagnosis",
                  environment=dict(project="p19-synthetic-native", iterations=1, seed=7,
                                   output_dir=str(tmp_path / "unused"), wandb=False, swanlab=False),
                  data=dict(factory_name="default", data_dir=str(data_root), metadata_file="metadata.csv",
                            cache_dir=str(data_root), use_cache=True, batch_size=4, num_workers=0,
                            window_size=128, num_window=2, window_sampling_strategy="evenly_spaced",
                            normalization="per_window_standardization", dtype="float32",
                            train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, unused_ratio=0.0,
                            split=dict(strategy="grouped_metadata", group_key="Physical_unit",
                                       stratify_key="Label", test_policy="partition", manifest_mode="read_only",
                                       manifest_path=str(manifest_path), seed=7,
                                       partition_map=dict(train="fit", val="tune", test="test" if target else "grow"))),
                  model=dict(type="X_model", name="TSPN", device="cpu", num_classes=2,
                             in_dim=128, out_dim=128, in_channels=1, out_channels=2,
                             scale=1, skip_connection=True, internal_instance_normalization=False,
                             signal_processing_configs=dict(layer1=["I"]), feature_extractor_configs=["Mean", "Std"]),
                  task=dict(type="Default_task", name="Default_task", target_system_id=[1], loss="CE",
                            metrics=["acc"], optimizer="adam", lr=0.005, weight_decay=0.0, epochs=2),
                  trainer=dict(name="Default_trainer", num_epochs=2, device="cpu", devices=1,
                               test_after_fit=False, monitor="val_loss", monitor_mode="min",
                               deterministic=True, early_stopping=False, num_sanity_val_steps=0,
                               log_every_n_steps=1))
    requested_roles = {role: dict(record_ids=entry["ids"],
                                 unit_by_record={str(record_id): f"unit_{record_id}" for record_id in entry["ids"]})
                       for role, entry in partitions.items()}
    return config, requested_roles


def test_native_train_checkpoint_same_pass_frozen_export_and_replay(tmp_path, monkeypatch):
    initial, gradients, model_calls, feature_calls = {}, [], [], []
    original_build = classification.build_model

    def observed_model(*args, **kwargs):
        model = original_build(*args, **kwargs)
        initial.update({name: value.detach().clone() for name, value in model.state_dict().items()})
        model.signal_processing_layers[0].weight_connection.bias.register_hook(
            lambda grad: gradients.append(grad.detach().clone()))
        return model

    monkeypatch.setattr(classification, "build_model", observed_model)
    config, roles = _fixture(tmp_path)
    before_config = deepcopy(config)
    trained = native.train_native(config, tmp_path / "training")
    assert config == before_config
    assert Path(trained["checkpoint"]).is_file()
    assert gradients and all(torch.isfinite(value).all() for value in gradients)
    assert any(torch.count_nonzero(value) > 0 for value in gradients)
    saved = torch.load(trained["checkpoint"], map_location="cpu", weights_only=False)["state_dict"]
    assert any(not torch.equal(value, saved[f"network.{name}"])
               for name, value in initial.items() if "weight" in name or "bias" in name)
    assert trained["result"]["test_metrics"] is None
    assert Path(trained["stdout"]).is_file() and Path(trained["stderr"]).is_file()
    assert trained["reference_scope"] == "best validation-selected checkpoint restored by the native runtime"
    assert Path(trained["fit_metrics_path"]).is_file()
    with np.load(trained["fit_reference"], allow_pickle=False) as fit:
        from scipy.special import softmax
        from src.task_factory.task.classification.multiview_signal.metrics import aggregate_metrics

        assert set(fit["record_id"]) == {"1", "2"}
        assert len(fit["item"]) == 4 and len(np.unique(fit["item"])) == 4
        expected_fit = aggregate_metrics(softmax(fit["logits"], axis=1), fit["y"], fit["unit"])
        assert trained["fit_metrics"] == expected_fit
        assert trained["fit_accuracy"] == expected_fit["accuracy"]

    from src.model_factory import build_model as export_build
    import src.model_factory

    def counted_model(*args, **kwargs):
        model = export_build(*args, **kwargs)
        model.register_forward_hook(lambda *args: model_calls.append(1))
        model.feature_extractor_layers.register_forward_hook(lambda *args: feature_calls.append(1))
        return model

    monkeypatch.setattr(src.model_factory, "build_model", counted_model)
    exported = native.export_native(trained["native_config"], Path(trained["checkpoint"]), roles,
                                    PROGRAM, tmp_path / "export", "cpu",
                                    reference=Path(trained["tune_reference"]))
    assert model_calls and len(model_calls) == len(feature_calls) == 4
    assert exported["native_reference_max_abs"] == 0
    assert exported["frozen_parameters"] and exported["frozen_buffers"]
    with np.load(trained["fit_reference"], allow_pickle=False) as fit, np.load(exported["roles"]["fit"], allow_pickle=False) as replayed_fit:
        for key in ("item", "logits", "y", "unit", "native_features"):
            np.testing.assert_array_equal(fit[key], replayed_fit[key])
    with np.load(exported["operator_inputs"], allow_pickle=False) as bank:
        assert bank["logits"].shape == (16, 2)
        assert bank["native_features"].shape == (16, 4)
        assert bank["program_features"].shape == (16, 42)
        assert bank["feature_names"].shape == (42,)
        assert bank["y"].dtype.kind in "iu"
        assert len(np.unique(bank["item"])) == 16
        assert set(bank["role"]) == set(native.SOURCE_ROLES)
        assert np.isfinite(bank["logits"]).all() and np.isfinite(bank["program_features"]).all()
    replay = native.export_native(trained["native_config"], Path(trained["checkpoint"]), roles,
                                 PROGRAM, tmp_path / "replay", "cpu")
    with np.load(exported["operator_inputs"]) as first, np.load(replay["operator_inputs"]) as second:
        for key in first.files:
            np.testing.assert_array_equal(first[key], second[key])


def test_source_population_and_target_role_are_rejected_before_data_loading(tmp_path, monkeypatch):
    import src.data_factory

    monkeypatch.setattr(src.data_factory, "build_data", lambda *args: pytest.fail("Forbidden population reached native readers"))
    config, roles = _fixture(tmp_path, target=True)
    config["data"]["split"]["partition_map"]["test"] = "grow"
    with pytest.raises(ValueError, match="source-only.*manifest"):
        native.export_native(config, tmp_path / "never.ckpt", {"test": roles["test"]}, PROGRAM,
                             tmp_path / "forbidden", "cpu")
    source_config, source_roles = _fixture(tmp_path)
    with pytest.raises(ValueError, match="permits only roles"):
        native.export_native(source_config, tmp_path / "never.ckpt", {"test": source_roles["fit"]},
                             PROGRAM, tmp_path / "forbidden", "cpu")
    assert not (tmp_path / "forbidden").exists()


def test_target_requires_existing_matching_source_freeze_before_data_loading(tmp_path, monkeypatch):
    import src.data_factory

    monkeypatch.setattr(src.data_factory, "build_data", lambda *args: pytest.fail("Unfrozen target reached native readers"))
    config, roles = _fixture(tmp_path, target=True)
    frozen_path = tmp_path / "frozen.json"
    frozen_path.write_text(json.dumps(dict(schema_version=1, checkpoint="wrong.ckpt", temperature=1,
                                          states={"signal": {}}, class_names=PROGRAM["class_names"])))
    with pytest.raises(ValueError, match="checkpoint differs"):
        native.export_target_native(config, tmp_path / "requested.ckpt", roles["test"],
                                    PROGRAM, tmp_path / "forbidden", "cpu", frozen_path)
    frozen_path.write_text(json.dumps(dict(schema_version=1, checkpoint="wrong.ckpt", temperature=1,
                                          states={}, class_names=PROGRAM["class_names"])))
    with pytest.raises(ValueError, match="frozen source readout"):
        native.export_target_native(config, tmp_path / "requested.ckpt", roles["test"],
                                    PROGRAM, tmp_path / "forbidden", "cpu", frozen_path)
    frozen_path.write_text(json.dumps(dict(schema_version=1, checkpoint=str(tmp_path / "requested.ckpt"),
                                          temperature=1, states={"native_reference": True},
                                          class_names=PROGRAM["class_names"], native_config=deepcopy(config))))
    config["data"]["normalization"] = "none"
    with pytest.raises(ValueError, match="Target preprocessing"):
        native.export_target_native(config, tmp_path / "requested.ckpt", roles["test"],
                                    PROGRAM, tmp_path / "forbidden", "cpu", frozen_path)


def test_device_request_cannot_fall_back_or_change(tmp_path):
    config, _ = _fixture(tmp_path)
    with pytest.raises(ValueError, match="differs from native device"):
        native._device(config, "cuda:0")
    config["trainer"]["device"] = "auto"
    with pytest.raises(ValueError, match="explicit trainer.device"):
        native._device(config)


def test_real_population_requires_derived_cache_outside_source_root(tmp_path, monkeypatch):
    monkeypatch.setattr(classification, "build_data", lambda *args: pytest.fail("Immutable source reached cache construction"))
    config, _ = _fixture(tmp_path)
    metadata_path = Path(config["data"]["data_dir"]) / config["data"]["metadata_file"]
    frame = pd.read_csv(metadata_path)
    frame["Name"] = "RM_027_PU"
    frame.to_csv(metadata_path, index=False)
    config["data"].pop("cache_dir")
    with pytest.raises(ValueError, match="explicit data.cache_dir outside"):
        native.train_native(config, tmp_path / "forbidden")
    config["data"]["cache_dir"] = str(Path(config["data"]["data_dir"]) / "derived_cache")
    with pytest.raises(ValueError, match="data.cache_dir must be outside"):
        native.train_native(config, tmp_path / "forbidden")
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.parametrize("visible", ["0", "2", "GPU-physical-gpu-2", ""])
def test_cuda_rejects_ambiguous_or_forbidden_mapping_before_cuda_calls(tmp_path, monkeypatch, visible):
    config, _ = _fixture(tmp_path)
    config["trainer"]["device"] = "cuda"
    config["model"]["device"] = "cuda:0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", visible)

    def gpu0_uuid(command, **kwargs):
        assert command == ["nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader"]
        assert kwargs == {"text": True}
        return "GPU-physical-gpu-0\n"

    monkeypatch.setattr(native.subprocess, "check_output", gpu0_uuid)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Unverified mapping reached CUDA"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: pytest.fail("Unverified mapping reached CUDA"))
    with pytest.raises(RuntimeError, match="physical GPU 0 UUID"):
        native._device(config)


def test_cuda_requires_successful_gpu0_uuid_query_before_cuda_calls(tmp_path, monkeypatch):
    config, _ = _fixture(tmp_path)
    config["trainer"]["device"] = "cuda"
    config["model"]["device"] = "cuda:0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-physical-gpu-0")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Failed GPU0 query reached CUDA"))
    monkeypatch.setattr(torch.cuda, "device_count", lambda: pytest.fail("Failed GPU0 query reached CUDA"))

    def failed_query(command, **kwargs):
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(native.subprocess, "check_output", failed_query)
    with pytest.raises(RuntimeError, match="Cannot verify physical GPU 0 UUID"):
        native._device(config)


def test_cuda_verified_gpu0_uuid_maps_to_exactly_one_logical_device(tmp_path, monkeypatch):
    config, _ = _fixture(tmp_path)
    config["trainer"]["device"] = "cuda"
    config["model"]["device"] = "cuda:0"
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-physical-gpu-0")
    calls = []

    def gpu0_uuid(command, **kwargs):
        assert command == ["nvidia-smi", "--id=0", "--query-gpu=uuid", "--format=csv,noheader"]
        calls.append("verified_gpu0_uuid")
        return "GPU-physical-gpu-0\n"

    def available():
        calls.append("cuda_available")
        return True

    monkeypatch.setattr(native.subprocess, "check_output", gpu0_uuid)
    monkeypatch.setattr(torch.cuda, "is_available", available)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    assert native._device(config) == "cuda:0"
    assert calls[0] == "verified_gpu0_uuid"
    assert "cuda_available" in calls
    config["environment"]["CUDA_VISIBLE_DEVICES"] = "0"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("Environment reset would expose ambiguous GPU"))
    with pytest.raises(RuntimeError, match="would replace the verified"):
        native._device(config)


@pytest.mark.parametrize("model", [
    dict(type="CNN", name="ResNet1D", device="cpu", num_classes=2, input_dim=1,
         block_type="basic", layers=[1, 1, 1, 1], initial_channels=2),
    dict(type="CNN", name="TCN", device="cpu", num_classes=2, input_dim=1,
         num_channels=[4, 4], kernel_size=3, dropout=0.0),
])
def test_native_baseline_source_fit_and_frozen_target_export(tmp_path, model):
    config, source_roles = _fixture(tmp_path)
    config["model"] = model
    config["data"]["eval_batch_size"] = 1
    trained = native.train_native(config, tmp_path / "training")
    exported = native.export_native(trained["native_config"], Path(trained["checkpoint"]),
                                    {"tune": source_roles["tune"]}, PROGRAM, tmp_path / "source_export",
                                    "cpu", reference=Path(trained["tune_reference"]))
    assert exported["native_feature_kind"] == "unavailable"
    assert exported["native_reference_max_abs"] == 0
    with np.load(exported["operator_inputs"]) as bank:
        assert bank["logits"].shape == (4, 2)
        assert bank["native_features"].shape == (4, 0)
    target_config, roles = _fixture(tmp_path, target=True)
    target_config["model"] = model
    target_config["data"]["eval_batch_size"] = 1
    frozen = tmp_path / "frozen.json"
    frozen.write_text(json.dumps(dict(schema_version=1, checkpoint=trained["checkpoint"], temperature=1,
                                     states={"native_reference": True}, class_names=PROGRAM["class_names"],
                                     native_config=trained["native_config"])))
    target = native.export_target_native(target_config, Path(trained["checkpoint"]),
                                         roles["test"], PROGRAM, tmp_path / "target_export", "cpu", frozen)
    assert target["target_accessed"] and target["frozen_buffers"]
    with np.load(target["operator_inputs"]) as bank:
        assert set(bank["role"]) == {"test"}
        assert set(bank["record_id"]) == {"9", "10"}
        assert bank["logits"].shape == (4, 2)
