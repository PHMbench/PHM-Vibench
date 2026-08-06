from types import SimpleNamespace

import numpy as np

from src.data_factory.dataset_task.Default_dataset import Default_dataset
from src.data_factory.explicit_data_factory import _summarize_split_assignments


def _data_args():
    return SimpleNamespace(
        window_size=8,
        stride=1,
        num_window=5,
        window_sampling_strategy="evenly_spaced",
        window_sampling_seed=0,
        train_ratio=0.4,
        val_ratio=0.2,
        test_ratio=0.4,
        normalization="none",
        noise_snr=None,
    )


def test_same_file_split_reports_raw_sample_overlap():
    metadata = {
        1: {
            "Label": 0,
            "Domain_id": 0,
        }
    }
    data = {1: np.arange(20, dtype=np.float32).reshape(-1, 1)}
    task = SimpleNamespace(type="FS")

    train = Default_dataset(data, metadata, _data_args(), task, "train")
    val = Default_dataset(data, metadata, _data_args(), task, "val")
    test = Default_dataset(data, metadata, _data_args(), task, "test")

    assert train.window_intervals == [(0, 8), (3, 11)]
    assert val.window_intervals == [(6, 14)]
    assert test.window_intervals == [(9, 17), (12, 20)]

    summary = _summarize_split_assignments(
        {"train": {1: train}, "val": {1: val}, "test": {1: test}},
        metadata,
    )

    assert summary["raw_interval_overlap"] == {
        "train_val": True,
        "train_test": True,
        "val_test": True,
    }
    assert summary["file_overlap"]["train_test"] == [1]
    assert summary["domain_overlap"]["train_test"] == [0]
    assert summary["test_classes_seen_in_train"] is True


def test_distinct_files_and_unseen_test_class_are_reported():
    metadata = {
        1: {"Label": 0, "Domain_id": 0},
        2: {"Label": 0, "Domain_id": 1},
        3: {"Label": 1, "Domain_id": 2},
    }
    dataset = lambda interval: SimpleNamespace(window_intervals=[interval])

    summary = _summarize_split_assignments(
        {
            "train": {1: dataset((0, 8))},
            "val": {2: dataset((0, 8))},
            "test": {3: dataset((0, 8))},
        },
        metadata,
    )

    assert summary["raw_interval_overlap"] == {
        "train_val": False,
        "train_test": False,
        "val_test": False,
    }
    assert summary["file_overlap"]["train_val"] == []
    assert summary["domain_overlap"]["train_test"] == []
    assert summary["classes"] == {
        "train": [0],
        "val": [0],
        "test": [1],
    }
    assert summary["test_classes_seen_in_train"] is False
