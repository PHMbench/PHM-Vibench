import json
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.data_factory.data_utils import MetadataAccessor
from src.data_factory.data_factory import data_factory
from src.data_factory.dataset_task.Default_dataset import Default_dataset
from src.data_factory.explicit_data_factory import ExplicitDataFactory
from src.data_factory.id_data_factory import id_data_factory
from src.data_factory.splitting import resolve_data_splits


def _metadata(group_count: int = 30, ids_per_group: int = 2) -> MetadataAccessor:
    records = []
    for group_index in range(group_count):
        for sample_index in range(ids_per_group):
            records.append(
                {
                    "Id": f"id-{group_index:02d}-{sample_index}",
                    "Bearing_id": f"bearing-{group_index:02d}",
                    "Label": group_index % 2,
                    "Dataset_id": 1,
                    "Domain_id": 1,
                }
            )
    return MetadataAccessor(pd.DataFrame(records), key_column="Id")


def _split_args(path, **overrides):
    split = {
        "strategy": "grouped_metadata",
        "group_key": "Bearing_id",
        "stratify_key": "Label",
        "seed": 17,
        "test_policy": "partition",
        "fractions": {"train": 0.6, "val": 0.2, "test": 0.2},
        "manifest_path": str(path),
    }
    split.update(overrides)
    return SimpleNamespace(
        split=SimpleNamespace(**split),
        normalization="standardization",
    )


def _groups(metadata, ids):
    return {metadata[sample_id]["Bearing_id"] for sample_id in ids}


def test_grouped_partition_is_disjoint_deterministic_and_manifested(tmp_path):
    metadata = _metadata()
    ids = metadata.keys()
    manifest = tmp_path / "split.json"

    first = resolve_data_splits(
        metadata,
        _split_args(manifest),
        SimpleNamespace(type="Default_task"),
        ids,
        [],
    )
    first_bytes = manifest.read_bytes()
    second = resolve_data_splits(
        metadata,
        _split_args(manifest),
        SimpleNamespace(type="Default_task"),
        list(reversed(ids)),
        [],
    )

    assert first == second
    assert manifest.read_bytes() == first_bytes
    train_groups = _groups(metadata, first.train_ids)
    val_groups = _groups(metadata, first.val_ids)
    test_groups = _groups(metadata, first.test_ids)
    assert train_groups.isdisjoint(val_groups | test_groups)
    assert val_groups.isdisjoint(test_groups)
    assert len(first.train_ids) + len(first.val_ids) + len(first.test_ids) == len(ids)

    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["metadata_sha256"]
    assert payload["normalization"] == {
        "method": "standardization",
        "scope": "per_window",
    }


def test_task_defined_keeps_target_groups_as_test(tmp_path):
    metadata = _metadata(group_count=20)
    source_ids = metadata.keys()[:32]
    target_ids = metadata.keys()[32:]
    args = _split_args(
        tmp_path / "dg.json",
        test_policy="task_defined",
        fractions={"train": 0.75, "val": 0.25},
    )

    result = resolve_data_splits(
        metadata,
        args,
        SimpleNamespace(type="DG"),
        source_ids,
        target_ids,
    )

    assert set(result.test_ids) == set(target_ids)
    assert _groups(metadata, result.train_ids).isdisjoint(_groups(metadata, result.test_ids))
    assert _groups(metadata, result.val_ids).isdisjoint(_groups(metadata, result.test_ids))


def test_task_defined_rejects_source_target_group_overlap(tmp_path):
    metadata = _metadata(group_count=12)
    args = _split_args(
        tmp_path / "overlap.json",
        test_policy="task_defined",
        fractions={"train": 0.75, "val": 0.25},
    )

    with pytest.raises(ValueError, match="group leakage"):
        resolve_data_splits(
            metadata,
            args,
            SimpleNamespace(type="DG"),
            metadata.keys()[:20],
            metadata.keys()[18:],
        )


def test_task_defined_rejects_empty_target_partition(tmp_path):
    metadata = _metadata(group_count=12)
    args = _split_args(
        tmp_path / "empty-target.json",
        test_policy="task_defined",
        fractions={"train": 0.75, "val": 0.25},
    )

    with pytest.raises(ValueError, match="non-empty task test IDs"):
        resolve_data_splits(
            metadata,
            args,
            SimpleNamespace(type="DG"),
            metadata.keys(),
            [],
        )


@pytest.mark.parametrize("task_type", ["FS", "GFS"])
def test_grouped_split_rejects_episodic_task(tmp_path, task_type):
    metadata = _metadata()
    with pytest.raises(ValueError, match="episode-safe"):
        resolve_data_splits(
            metadata,
            _split_args(tmp_path / f"{task_type.lower()}.json"),
            SimpleNamespace(type=task_type),
            metadata.keys(),
            [],
        )


def test_legacy_split_preserves_id_contract():
    metadata = _metadata(group_count=4)
    train_ids = metadata.keys()[:4]
    test_ids = metadata.keys()[4:]
    result = resolve_data_splits(
        metadata,
        SimpleNamespace(),
        SimpleNamespace(type="DG"),
        train_ids,
        test_ids,
    )
    assert result.train_ids == tuple(train_ids)
    assert result.val_ids == tuple(train_ids)
    assert result.test_ids == tuple(test_ids)


def _factory_args(manifest_path):
    args_data = _split_args(manifest_path)
    args_data.window_size = 2
    args_data.stride = 2
    args_data.train_ratio = 0.6
    args_data.num_window = 4
    args_data.window_sampling_strategy = "sequential"
    args_data.dtype = "float32"
    args_data.num_workers = 0
    return args_data


def _task_args():
    return SimpleNamespace(
        type="Default_task",
        name="Default_task",
        target_system_id=[1],
    )


def _id_task_args():
    return SimpleNamespace(
        type="pretrain",
        name="classification",
        target_system_id=None,
    )


def _dataset_ids(dataset):
    return tuple(dataset.dataset_dict)


def test_default_dataset_uses_group_split_instead_of_window_split():
    data = {"id-0": np.arange(20, dtype=np.float32).reshape(20, 1)}
    metadata = {"id-0": {"Label": 0}}
    common = {
        "window_size": 2,
        "stride": 2,
        "train_ratio": 0.6,
        "num_window": 10,
        "window_sampling_strategy": "sequential",
        "dtype": "float32",
        "normalization": "none",
    }
    legacy = SimpleNamespace(**common)
    grouped = SimpleNamespace(
        **common,
        split=SimpleNamespace(strategy="grouped_metadata"),
    )

    assert len(Default_dataset(data, metadata, legacy, SimpleNamespace(), "train")) == 6
    assert len(Default_dataset(data, metadata, legacy, SimpleNamespace(), "val")) == 4
    assert len(Default_dataset(data, metadata, grouped, SimpleNamespace(), "train")) == 10
    assert len(Default_dataset(data, metadata, grouped, SimpleNamespace(), "val")) == 10

    invalid = SimpleNamespace(
        **common,
        split=SimpleNamespace(strategy="unknown"),
    )
    with pytest.raises(ValueError, match="Unknown data.split.strategy"):
        Default_dataset(data, metadata, invalid, SimpleNamespace(), "train")


def test_runtime_factories_split_groups_before_dataset_construction(tmp_path):
    metadata = _metadata(group_count=20)
    args_data = _factory_args(tmp_path / "runtime-split.json")
    args_task = _task_args()
    raw_data = {
        sample_id: np.arange(20, dtype=np.float32).reshape(20, 1)
        for sample_id in metadata.keys()
    }

    first = data_factory.__new__(data_factory)
    first.args_data = args_data
    first.args_task = args_task
    first.target_metadata = metadata
    first.data = raw_data
    candidate_ids = metadata.keys()
    first.search_id = lambda: (candidate_ids, [])
    train, val, test = first._init_dataset()

    assert _dataset_ids(train) == first.split_result.train_ids
    assert _dataset_ids(val) == first.split_result.val_ids
    assert _dataset_ids(test) == first.split_result.test_ids
    assert all(
        len(per_id_dataset) == args_data.num_window
        for split_dataset in (train, val, test)
        for per_id_dataset in split_dataset.dataset_dict.values()
    )
    train_groups = _groups(metadata, _dataset_ids(train))
    val_groups = _groups(metadata, _dataset_ids(val))
    test_groups = _groups(metadata, _dataset_ids(test))
    assert train_groups.isdisjoint(val_groups | test_groups)
    assert val_groups.isdisjoint(test_groups)

    second = data_factory.__new__(data_factory)
    second.args_data = args_data
    second.args_task = args_task
    second.target_metadata = metadata
    second.data = raw_data
    second.search_id = lambda: (list(reversed(candidate_ids)), [])
    second._init_dataset()
    assert second.split_result == first.split_result

    explicit = ExplicitDataFactory.__new__(ExplicitDataFactory)
    explicit.args_data = args_data
    explicit.args_task = args_task
    explicit.target_metadata = metadata
    explicit.data = raw_data
    explicit.search_id = lambda: (candidate_ids, [])
    explicit_train, explicit_val, explicit_test = explicit._init_dataset()
    assert _dataset_ids(explicit_train) == explicit.split_result.train_ids
    assert _dataset_ids(explicit_val) == explicit.split_result.val_ids
    assert _dataset_ids(explicit_test) == explicit.split_result.test_ids
    assert all(
        len(per_id_dataset) == args_data.num_window
        for split_dataset in (explicit_train, explicit_val, explicit_test)
        for per_id_dataset in split_dataset.dataset_dict.values()
    )
    assert explicit.split_summary["file_overlap"] == {
        "train_val": [],
        "train_test": [],
        "val_test": [],
    }

    id_factory = id_data_factory.__new__(id_data_factory)
    id_factory.args_data = args_data
    id_factory.args_task = _id_task_args()
    id_factory.metadata = metadata
    id_train, id_val, id_test = id_factory._init_dataset()
    assert _dataset_ids(id_train) == id_factory.split_result.train_ids
    assert _dataset_ids(id_val) == id_factory.split_result.val_ids
    assert _dataset_ids(id_test) == id_factory.split_result.test_ids
    assert all(
        child.ids == [sample_id]
        for split_dataset in (id_train, id_val, id_test)
        for sample_id, child in split_dataset.dataset_dict.items()
    )
