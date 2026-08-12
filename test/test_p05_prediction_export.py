from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.explain_factory.p05_prediction_runner import export_p05_window_predictions
from src.explain_factory.p05_trace_runner import model_state_sha256
from src.model_factory.X_model.TSPN_UXFD import (
    Model as TSPNUXFD,
    P05FeatureLogitOutput,
)


CONFIG_HASH = "1" * 64
CODE_HASH = "2" * 64
CHECKPOINT_HASH = "3" * 64
RUN_CONTRACT_HASH = "4" * 64


class _PredictionNetwork(torch.nn.Module):
    def __init__(self, *, mutate: bool = False) -> None:
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float32))
        self.mutate = mutate
        self.calls = 0

    def forward_with_features(self, x: torch.Tensor) -> P05FeatureLogitOutput:
        self.calls += 1
        if self.mutate:
            self.scale.add_(0.01)
        channel_means = x.mean(dim=1)
        features = channel_means.repeat(1, 4)
        logits = torch.stack(
            (features[:, 0] * self.scale[0], features[:, 1]),
            dim=-1,
        )
        return P05FeatureLogitOutput(
            reduced_features=features,
            logits=logits,
        )


def _batch(split: str, *, order: tuple[int, int] = (1, 0)) -> dict:
    record_id = {"train": "101", "val": "201", "test": "301"}[split]
    group_id = f"group-{split}"
    starts = [0, 4]
    windows = torch.tensor(
        [
            [[0.1, 0.2], [0.2, 0.3], [0.3, 0.4], [0.4, 0.5]],
            [[0.5, 0.4], [0.4, 0.3], [0.3, 0.2], [0.2, 0.1]],
        ],
        dtype=torch.float32,
    )
    return {
        "x": windows[list(order)],
        "y": torch.tensor([0, 0], dtype=torch.int64)[list(order)],
        "sample_weight": torch.ones(2, dtype=torch.float64)[list(order)],
        "sample_id": [f"{record_id}:{starts[index]}:{starts[index] + 4}" for index in order],
        "record_id": [record_id for _ in order],
        "group_id": [group_id for _ in order],
        "window_index": torch.tensor(order, dtype=torch.int64),
        "window_start": torch.tensor([starts[index] for index in order], dtype=torch.int64),
        "window_end": torch.tensor(
            [starts[index] + 4 for index in order],
            dtype=torch.int64,
        ),
    }


def _loaders() -> dict[str, list[dict]]:
    return {split: [_batch(split)] for split in ("train", "val", "test")}


def _expected_records() -> dict[str, tuple[str, ...]]:
    return {"train": ("101",), "val": ("201",), "test": ("301",)}


def _export(tmp_path, *, network=None, loaders=None, **overrides):
    active_network = network or _PredictionNetwork()
    values = {
        "network": active_network,
        "split_dataloaders": _loaders() if loaders is None else loaders,
        "expected_record_ids_by_split": _expected_records(),
        "expected_windows_per_record": 2,
        "expected_window_size": 4,
        "config_sha256": CONFIG_HASH,
        "code_sha256": CODE_HASH,
        "checkpoint_sha256": CHECKPOINT_HASH,
        "model_sha256": model_state_sha256(active_network),
        "run_contract_sha256": RUN_CONTRACT_HASH,
        "require_cuda": False,
    }
    values.update(overrides)
    return export_p05_window_predictions(tmp_path / "predictions", **values)


def _tspn_args() -> SimpleNamespace:
    return SimpleNamespace(
        device="cpu",
        num_classes=2,
        in_channels=2,
        out_channels=4,
        scale=1,
        skip_connection=True,
        internal_instance_normalization=False,
        signal_processing_configs={"layer1": ["I"]},
        feature_extractor_configs=["Mean", "Std"],
        in_dim=128,
        out_dim=128,
        uxfd=SimpleNamespace(
            enable_sp2d=False,
            fuzzy=SimpleNamespace(enable=False),
            neural_residual=SimpleNamespace(enable=False),
            anfis=SimpleNamespace(enable=False),
            operator_attention=SimpleNamespace(enable=False),
            logic=SimpleNamespace(enable=False),
        ),
    )


def test_tspn_public_interface_returns_same_forward_eight_features_and_logits() -> None:
    torch.manual_seed(17)
    model = TSPNUXFD(_tspn_args())
    model.eval()
    state_hash = model_state_sha256(model)
    calls = 0
    original = model._forward_features

    def counted_forward_features(x):
        nonlocal calls
        calls += 1
        return original(x)

    model._forward_features = counted_forward_features
    with torch.no_grad():
        output = model.forward_with_features(torch.randn(2, 128, 2))

    assert calls == 1
    assert output.reduced_features.shape == (2, 8)
    assert output.logits.shape == (2, 2)
    assert torch.equal(
        output.logits,
        model._forward_non_fuzzy_logits(output.reduced_features),
    )
    assert model_state_sha256(model) == state_hash


def test_export_is_sorted_hashed_create_only_reusable_and_unadjudicated(tmp_path) -> None:
    network = _PredictionNetwork()
    network.train()
    state_hash = model_state_sha256(network)

    created = _export(tmp_path, network=network)
    bytes_before = {path.name: path.read_bytes() for path in created.package_dir.iterdir()}
    reused = _export(tmp_path, network=network)

    assert created.status == "created"
    assert reused.status == "reused"
    assert network.training is True
    assert network.calls == 6
    assert model_state_sha256(network) == state_hash
    assert {path.name: path.read_bytes() for path in created.package_dir.iterdir()} == bytes_before

    manifest = json.loads(created.manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_name"] == "p05.window_predictions"
    assert manifest["evidence_status"] == "unadjudicated"
    assert manifest["conclusion_control"] == {
        "claim_decisions": "not_performed",
        "decisive": False,
        "status": "unadjudicated",
    }
    assert manifest["provenance"] == {
        "checkpoint_sha256": CHECKPOINT_HASH,
        "code_sha256": CODE_HASH,
        "config_sha256": CONFIG_HASH,
        "model_sha256": state_hash,
        "run_contract_sha256": RUN_CONTRACT_HASH,
    }
    assert manifest["contract"]["record_coverage"] == (
        "exact_expected_record_ids_and_windows"
    )
    assert manifest["splits"]["train"]["sample_count"] == 2
    assert manifest["content"]["arrays_sha256"] == hashlib.sha256(
        created.arrays_path.read_bytes()
    ).hexdigest()

    with np.load(created.arrays_path, allow_pickle=False) as arrays:
        assert arrays["split"].tolist() == [
            "train",
            "train",
            "val",
            "val",
            "test",
            "test",
        ]
        assert arrays["sample_id"].tolist() == [
            "101:0:4",
            "101:4:8",
            "201:0:4",
            "201:4:8",
            "301:0:4",
            "301:4:8",
        ]
        assert arrays["window_index"].tolist() == [0, 1, 0, 1, 0, 1]
        assert arrays["reduced_features"].dtype == np.dtype("<f4")
        assert arrays["logits"].dtype == np.dtype("<f4")
        assert arrays["sample_weight"].dtype == np.dtype("<f8")
        assert all(not arrays[name].dtype.hasobject for name in arrays.files)


def test_export_rejects_incomplete_duplicate_and_overlapping_record_windows(
    tmp_path,
) -> None:
    incomplete = _loaders()
    for name, value in list(incomplete["train"][0].items()):
        incomplete["train"][0][name] = value[:1]
    with pytest.raises(ValueError, match="exactly 2 windows"):
        _export(tmp_path / "incomplete", loaders=incomplete)
    assert not (tmp_path / "incomplete" / "predictions").exists()

    with pytest.raises(ValueError, match="record coverage mismatch"):
        _export(
            tmp_path / "missing-record",
            expected_record_ids_by_split={
                "train": ("101", "102"),
                "val": ("201",),
                "test": ("301",),
            },
        )

    duplicate = _loaders()
    duplicate_batch = duplicate["train"][0]
    duplicate_batch["sample_id"] = ["101:0:4", "101:0:4"]
    duplicate_batch["window_start"] = torch.tensor([0, 0], dtype=torch.int64)
    duplicate_batch["window_end"] = torch.tensor([4, 4], dtype=torch.int64)
    with pytest.raises(ValueError, match="duplicate sample_id"):
        _export(tmp_path / "duplicate", loaders=duplicate)

    overlap = _loaders()
    overlap_batch = overlap["train"][0]
    overlap_batch["sample_id"] = ["101:2:6", "101:0:4"]
    overlap_batch["window_start"] = torch.tensor([2, 0], dtype=torch.int64)
    overlap_batch["window_end"] = torch.tensor([6, 4], dtype=torch.int64)
    with pytest.raises(ValueError, match="overlapping windows"):
        _export(tmp_path / "overlap", loaders=overlap)


def test_export_rejects_cross_split_record_or_group_overlap(tmp_path) -> None:
    with pytest.raises(ValueError, match="record IDs overlap"):
        _export(
            tmp_path / "record-overlap",
            expected_record_ids_by_split={
                "train": ("101",),
                "val": ("101",),
                "test": ("301",),
            },
        )

    loaders = _loaders()
    loaders["val"][0]["group_id"] = ["group-train", "group-train"]
    with pytest.raises(ValueError, match="overlaps splits"):
        _export(tmp_path / "group-overlap", loaders=loaders)


def test_export_rejects_model_mutation_before_writing(tmp_path) -> None:
    network = _PredictionNetwork(mutate=True)
    network.train()

    with pytest.raises(RuntimeError, match="mutated"):
        _export(tmp_path, network=network)

    assert network.training is True
    assert not (tmp_path / "predictions").exists()


def test_create_only_conflict_preserves_existing_package(tmp_path) -> None:
    network = _PredictionNetwork()
    created = _export(tmp_path, network=network)
    bytes_before = {path.name: path.read_bytes() for path in created.package_dir.iterdir()}

    with pytest.raises(FileExistsError, match="provenance or contract conflicts"):
        _export(tmp_path, network=network, code_sha256="f" * 64)

    assert {path.name: path.read_bytes() for path in created.package_dir.iterdir()} == bytes_before


def test_runner_requires_float32_cuda_and_all_provenance(tmp_path) -> None:
    wrong_dtype = _loaders()
    wrong_dtype["train"][0]["x"] = wrong_dtype["train"][0]["x"].double()
    with pytest.raises(TypeError, match="float32"):
        _export(tmp_path / "dtype", loaders=wrong_dtype)

    with pytest.raises(RuntimeError, match="CUDA-resident"):
        _export(tmp_path / "cuda", require_cuda=True)

    with pytest.raises(ValueError, match="code_sha256"):
        _export(tmp_path / "hash", code_sha256="missing")


def test_model_hash_must_match_before_inference(tmp_path) -> None:
    network = _PredictionNetwork()

    with pytest.raises(ValueError, match="model state"):
        _export(tmp_path, network=network, model_sha256="f" * 64)

    assert network.calls == 0
    assert not (tmp_path / "predictions").exists()
