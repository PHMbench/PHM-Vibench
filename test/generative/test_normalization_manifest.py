from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from src.Pipeline_06_generative import (
    _attach_normalization_artifacts,
    _build_normalization_params,
)
from src.task_factory.Components.generative.manifests.synthetic_data_manifest import (
    build_synthetic_data_manifest,
)


class _DataFactory:
    def __init__(self, batches: list[dict[str, torch.Tensor]]) -> None:
        self._batches = batches

    def get_dataloader(self, split: str):
        assert split == "train"
        return iter(self._batches)


def _batch() -> dict[str, torch.Tensor]:
    return {
        "x": torch.tensor(
            [
                [[1.0, 2.0, 3.0], [10.0, 12.0, 14.0]],
                [[2.0, 4.0, 6.0], [20.0, 24.0, 28.0]],
            ]
        )
    }


def test_normalization_artifact_records_standardization_params(tmp_path: Path) -> None:
    args_data = SimpleNamespace(normalization="standardization")
    task = SimpleNamespace(args_data=args_data)
    data_factory = _DataFactory([_batch()])

    params_path, params_hash = _attach_normalization_artifacts(
        tmp_path,
        data_factory,
        args_data,
        task,
        channels=2,
    )

    payload = json.loads(Path(params_path).read_text(encoding="utf-8"))
    assert payload["method"] == "standardization"
    assert payload["scope"] == "per_channel"
    assert payload["source_split"] == "train"
    assert set(payload["channels"]) == {"0", "1"}
    assert {"mean", "std", "epsilon"} <= set(payload["channels"]["0"])
    assert getattr(args_data, "normalization_params_path") == params_path
    assert getattr(task.args_data, "normalization_params_hash") == params_hash
    sha_text = (tmp_path / "normalization_params.sha256").read_text(encoding="utf-8")
    assert sha_text.startswith(params_hash)


def test_robust_scaler_normalization_params_are_supported() -> None:
    params = _build_normalization_params(
        _DataFactory([_batch()]),
        SimpleNamespace(normalization="robust_scaler"),
        channels=2,
    )

    assert params["method"] == "robust_scaler"
    assert {"median", "q1", "q3", "iqr", "epsilon"} <= set(params["channels"]["0"])


def test_minmax_is_not_silently_used_for_generative_evidence() -> None:
    with pytest.raises(ValueError, match="standardization or robust_scaler"):
        _build_normalization_params(
            _DataFactory([_batch()]),
            SimpleNamespace(normalization="minmax"),
            channels=2,
        )


def test_manifest_marks_normalization_params_recorded(tmp_path: Path) -> None:
    args_data = SimpleNamespace(normalization="standardization")
    params_path, params_hash = _attach_normalization_artifacts(
        tmp_path,
        _DataFactory([_batch()]),
        args_data,
        SimpleNamespace(args_data=args_data),
        channels=2,
    )

    manifest = build_synthetic_data_manifest(
        synthetic_dataset_id="synthetic-smoke",
        model_type="generative_model",
        model_name="phm_cfm_mlp1d",
        loss_id="conditional_flow_matching",
        checkpoint_path="checkpoint.ckpt",
        generator_run_id="run-001",
        source_split="train",
        domain_map_path="configs/domain_maps/dummy_domain_map.csv",
        domain_map_hash="domain-hash",
        normalization={
            "method": "standardization",
            "scope": "per_channel",
            "params_artifact": params_path,
            "params_hash": params_hash,
        },
        sampler_id="euler_ode",
        num_steps=8,
        seed=0,
        num_samples=4,
        shape=[4, 2, 128],
    )

    assert manifest["normalization"]["params_recorded"] is True
