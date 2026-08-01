from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data_factory.data_factory import data_factory
from src.data_factory.dataset_task.Default_dataset import Default_dataset
from src.data_factory.p05_weighting import ExpectedRole, build_weight_plan
from src.data_factory.protocol_transforms import (
    WindowObservation,
    fit_train_channel_standardization,
)


def _dataset_args(weight_plan, normalization_plan):
    return SimpleNamespace(
        split=SimpleNamespace(strategy="preassigned_metadata"),
        p05_evidence_mode=True,
        p05_weight_plans={"val": weight_plan},
        p05_normalization_plan=normalization_plan,
        window_size=4,
        stride=4,
        train_ratio=0.8,
        num_window=2,
        window_sampling_strategy="evenly_spaced",
        dtype="float32",
        normalization="train_channel_standardization",
        noise_snr=None,
    )


def test_dataset_binds_frozen_normalization_weight_and_provenance():
    metadata_frame = pd.DataFrame(
        [
            {
                "Id": 7,
                "Dataset_id": 2,
                "Label": 0,
                "Protocol_Group": "XJTU/g0",
                "Protocol_Split": "validation",
            },
            {
                "Id": 8,
                "Dataset_id": 2,
                "Label": 1,
                "Protocol_Group": "XJTU/g1",
                "Protocol_Split": "validation",
            },
        ]
    )
    weight_plan = build_weight_plan(
        metadata_frame,
        dataset_id=2,
        role="validation",
        expected=ExpectedRole(2, 2, {0: 1, 1: 1}, 2),
    )
    observations = [
        WindowObservation("a", "train-g0", np.zeros((4, 2))),
        WindowObservation("b", "train-g1", np.asarray([[2.0, 4.0]] * 4)),
    ]
    normalization_plan = fit_train_channel_standardization(
        lambda: iter(observations),
        dataset_id=2,
        channel_names=("horizontal", "vertical"),
        expected_window_size=4,
        expected_windows_per_group={"train-g0": 1, "train-g1": 1},
    )
    args = _dataset_args(weight_plan, normalization_plan)
    args.p05_weight_plans = {"val": weight_plan}
    raw = np.arange(20, dtype=np.float64).reshape(10, 2, 1)
    metadata = {
        7: {
            "Dataset_id": 2,
            "Label": 0,
            "Protocol_Group": "XJTU/g0",
        }
    }

    dataset = Default_dataset({7: raw}, metadata, args, SimpleNamespace(), mode="val")
    first = dataset[0]

    assert len(dataset) == 2
    assert first["x"].dtype == np.float32
    assert first["sample_weight"] == pytest.approx(weight_plan.record_weights[7])
    assert first["sample_id"] == "7:0:4"
    assert first["record_id"] == "7"
    assert first["group_id"] == "XJTU/g0"


def test_p05_data_contract_is_exact_and_fails_on_batch_drift():
    factory = data_factory.__new__(data_factory)
    factory.args_data = SimpleNamespace(
        cache_mode="read_only_verified",
        allow_download=False,
        batch_size=64,
        window_size=4096,
        window_sampling_strategy="evenly_spaced",
        normalization="train_channel_standardization",
        dtype="float32",
        num_workers=0,
        drop_last_train=False,
        num_window=4,
        noise_snr=None,
        split=SimpleNamespace(
            strategy="preassigned_metadata",
            split_key="Protocol_Split",
            group_key="Protocol_Group",
        ),
    )
    factory.target_metadata = SimpleNamespace(
        df=pd.DataFrame({"Dataset_id": [2]})
    )

    assert factory._validate_p05_data_contract() == (2, 4)
    factory.args_data.batch_size = 32
    with pytest.raises(ValueError, match="batch_size=64"):
        factory._validate_p05_data_contract()


def test_dataset_rejects_missing_weight_or_normalization_plan():
    raw = np.arange(20, dtype=np.float64).reshape(10, 2, 1)
    metadata = {
        7: {
            "Dataset_id": 2,
            "Label": 0,
            "Protocol_Group": "XJTU/g0",
        }
    }
    args = SimpleNamespace(
        split=SimpleNamespace(strategy="preassigned_metadata"),
        p05_evidence_mode=True,
        window_size=4,
        stride=4,
        train_ratio=0.8,
        num_window=2,
        window_sampling_strategy="evenly_spaced",
        dtype="float32",
        normalization="train_channel_standardization",
    )
    with pytest.raises(ValueError, match="weight plan"):
        Default_dataset({7: raw}, metadata, args, SimpleNamespace(), mode="train")
