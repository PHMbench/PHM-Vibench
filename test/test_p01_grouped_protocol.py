from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
import h5py
from torch.utils.data import Dataset

from src.data_factory.ID.Id_searcher import search_target_dataset_metadata
from src.data_factory.data_utils import MetadataAccessor
from src.data_factory.dataset_task.Dataset_cluster import (
    FrozenClassPairDataset,
    IdIncludedDataset,
)
from src.data_factory.grouped_split import build_grouped_split, write_frozen_json
from src.data_factory.samplers.Get_sampler import Get_sampler


data_factory_module = importlib.import_module("src.data_factory.data_factory")


def _ns(**kwargs):  # type: ignore[no-untyped-def]
    return SimpleNamespace(**kwargs)


def test_grouped_split_is_stratified_disjoint_and_reproducible(tmp_path) -> None:
    rows = []
    identifier = 1
    for label in range(3):
        for group in range(5):
            rows.append(
                {
                    "Id": identifier,
                    "Dataset_id": 1,
                    "File": f"label{label}_group{group}.mat",
                    "Label": label,
                    "Domain_id": group,
                }
            )
            identifier += 1
    metadata = MetadataAccessor(pd.DataFrame(rows), key_column="Id")
    config = _ns(
        strategy="grouped_metadata",
        group_key="File",
        stratify_key="Label",
        seed=20260801,
        fractions=_ns(train=0.6, val=0.2, test=0.2),
    )

    first = build_grouped_split(metadata, config)
    second = build_grouped_split(metadata, config)
    assert first.manifest == second.manifest
    assert first.manifest["overlap_audit"] == {"group_overlap": 0, "id_overlap": 0}
    assert first.manifest["counts"] == {
        "train": {"groups": 9, "ids": 9},
        "val": {"groups": 3, "ids": 3},
        "test": {"groups": 3, "ids": 3},
    }
    target = tmp_path / "split.json"
    write_frozen_json(first.manifest, target)
    write_frozen_json(second.manifest, target)
    assert target.is_file()


def test_frozen_manifest_write_is_atomic_under_concurrency(tmp_path) -> None:
    target = tmp_path / "split.json"
    payload = {"schema_version": 1, "manifest_payload_sha256": "approved"}
    with ThreadPoolExecutor(max_workers=8) as executor:
        paths = list(
            executor.map(lambda _: write_frozen_json(payload, target), range(32))
        )
    assert paths == [target] * 32
    write_frozen_json(payload, target)
    assert not list(tmp_path.glob(".*.tmp"))


def test_concurrent_frozen_manifest_drift_fails_closed(tmp_path) -> None:
    target = tmp_path / "split.json"

    def _write(payload):  # type: ignore[no-untyped-def]
        try:
            write_frozen_json(payload, target)
        except RuntimeError as exc:
            assert "Frozen manifest drift" in str(exc)
            return "drift"
        return "written"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(
            executor.map(
                _write,
                [
                    {"schema_version": 1, "value": "first"},
                    {"schema_version": 1, "value": "second"},
                ],
            )
        )
    assert sorted(outcomes) == ["drift", "written"]
    assert not list(tmp_path.glob(".*.tmp"))


def test_file_parent_grouping_keeps_bearing_identity_intact() -> None:
    rows = []
    identifier = 1
    for domain in range(3):
        for bearing in range(5):
            parent = f"condition{domain}/Bearing{domain}_{bearing}"
            for step in range(2):
                rows.append(
                    {
                        "Id": identifier,
                        "Dataset_id": 2,
                        "File": f"{parent}/{step}.csv",
                        "Label": int(step > 0),
                        "Domain_id": domain,
                    }
                )
                identifier += 1
    metadata = MetadataAccessor(pd.DataFrame(rows), key_column="Id")
    split = build_grouped_split(
        metadata,
        _ns(
            strategy="grouped_metadata",
            group_key="FileParent",
            stratify_key="Domain_id",
            seed=20260801,
            fractions=_ns(train=0.6, val=0.2, test=0.2),
        ),
    )
    assert split.manifest["counts"]["train"]["groups"] == 9
    assert split.manifest["counts"]["val"]["groups"] == 3
    assert split.manifest["counts"]["test"]["groups"] == 3


def test_grouped_kfold_gives_each_group_one_outer_test_assignment() -> None:
    rows = []
    identifier = 1
    for label in range(2):
        for group in range(4):
            rows.append(
                {
                    "Id": identifier,
                    "Dataset_id": 1,
                    "File": f"label{label}_group{group}.mat",
                    "Label": label,
                }
            )
            identifier += 1
    metadata = MetadataAccessor(pd.DataFrame(rows), key_column="Id")
    test_groups = []
    for outer_fold in range(4):
        split = build_grouped_split(
            metadata,
            _ns(
                strategy="grouped_kfold",
                group_key="File",
                stratify_key="Label",
                seed=20260801,
                outer_folds=4,
                outer_fold=outer_fold,
                validation_offset=1,
            ),
        )
        assert split.manifest["overlap_audit"] == {
            "group_overlap": 0,
            "id_overlap": 0,
        }
        assert split.manifest["counts"] == {
            "train": {"groups": 4, "ids": 4},
            "val": {"groups": 2, "ids": 2},
            "test": {"groups": 2, "ids": 2},
        }
        test_groups.extend(split.manifest["split_groups"]["test"])
    assert sorted(test_groups) == sorted(row["File"] for row in rows)


def test_binary_fault_label_policy_is_explicit() -> None:
    frame = pd.DataFrame(
        [
            {"Id": 1, "Dataset_id": 2, "File": "a.csv", "Label": 0},
            {"Id": 2, "Dataset_id": 2, "File": "b.csv", "Label": 7},
            {"Id": 3, "Dataset_id": 2, "File": "c.csv", "Label": -1},
        ]
    )
    metadata = MetadataAccessor(frame, key_column="Id")
    mapped = search_target_dataset_metadata(
        metadata, _ns(target_system_id=[2], label_policy="binary_fault")
    )
    assert sorted(mapped.df["Label"].unique().tolist()) == [0, 1]
    assert set(mapped.df["Id"].tolist()) == {1, 2}


class _Windows(Dataset):
    def __init__(self, label: int, offset: float) -> None:
        self.label = label
        self.samples = [torch.full((16, 1), offset + index) for index in range(3)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):  # type: ignore[no-untyped-def]
        return {"x": self.samples[index], "y": self.label}


class _FactoryWindows(Dataset):
    def __init__(self, data, metadata, args_data, args_task, mode):  # type: ignore[no-untyped-def]
        del args_data, args_task, mode
        self.file_id = next(iter(data))
        self.label = int(metadata[self.file_id]["Label"])
        self.samples = [
            torch.full((16, 1), float(self.file_id * 10 + index))
            for index in range(3)
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int):  # type: ignore[no-untyped-def]
        return {"x": self.samples[index], "y": self.label}


def _synthetic_factory(
    tmp_path,
    monkeypatch,
    *,
    expected_sha: str | None = None,
    pairing_group_key: str = "File",
):  # type: ignore[no-untyped-def]
    rows = [
        {
            "Id": label * 4 + group + 1,
            "Dataset_id": 1,
            "File": f"label{label}_group{group}.mat",
            "Label": label,
        }
        for label in range(2)
        for group in range(4)
    ]
    metadata = MetadataAccessor(pd.DataFrame(rows), key_column="Id")
    split_cfg = _ns(
        strategy="grouped_kfold",
        group_key="File",
        stratify_key="Label",
        seed=20260801,
        outer_folds=4,
        outer_fold=0,
        validation_offset=1,
        test_policy="partition",
        manifest_path=str(tmp_path / "split.json"),
    )
    approved_sha = build_grouped_split(
        metadata, split_cfg
    ).manifest["manifest_payload_sha256"]
    split_cfg.expected_manifest_payload_sha256 = expected_sha or approved_sha
    args_data = _ns(
        split=split_cfg,
        pairing=_ns(
            mode="frozen_within_group_class_derangement",
            seed=20260801,
            splits=["train"],
            group_key=pairing_group_key,
            manifest_dir=str(tmp_path / "pairing"),
            protocol_id="P01-G040-v1",
        ),
    )
    factory = data_factory_module.data_factory.__new__(
        data_factory_module.data_factory
    )
    factory.args_data = args_data
    factory.args_task = _ns(name="Classification", type="DG")
    factory.target_metadata = metadata
    factory.data = {
        row["Id"]: np.full((8, 2), float(row["Id"]), dtype=np.float32)
        for row in rows
    }
    factory._data_fingerprint_records = {}
    monkeypatch.setattr(
        data_factory_module,
        "importlib",
        _ns(import_module=lambda _: _ns(set_dataset=_FactoryWindows)),
    )
    return factory, approved_sha


def test_grouped_factory_rejects_unapproved_payload_before_manifest_write(
    tmp_path, monkeypatch
) -> None:
    factory, _ = _synthetic_factory(
        tmp_path, monkeypatch, expected_sha="0" * 64
    )
    with pytest.raises(RuntimeError, match="does not match the approved protocol"):
        factory._init_dataset()
    assert not (tmp_path / "split.json").exists()
    assert not (tmp_path / "pairing").exists()


def test_grouped_factory_requires_an_expected_payload_hash(
    tmp_path, monkeypatch
) -> None:
    factory, _ = _synthetic_factory(tmp_path, monkeypatch)
    del factory.args_data.split.expected_manifest_payload_sha256
    with pytest.raises(
        ValueError,
        match="grouped_kfold requires data.split.expected_manifest_payload_sha256",
    ):
        factory._init_dataset()
    assert not (tmp_path / "split.json").exists()
    assert not (tmp_path / "pairing").exists()


def test_pairing_group_key_must_equal_split_group_key(tmp_path, monkeypatch) -> None:
    factory, _ = _synthetic_factory(
        tmp_path, monkeypatch, pairing_group_key="Id"
    )
    with pytest.raises(
        ValueError, match="data.pairing.group_key must equal data.split.group_key"
    ):
        factory._init_dataset()


def test_grouped_factory_deranges_training_only(tmp_path, monkeypatch) -> None:
    factory, approved_sha = _synthetic_factory(tmp_path, monkeypatch)
    train, val, test = factory._init_dataset()
    assert isinstance(train, FrozenClassPairDataset)
    assert isinstance(val, IdIncludedDataset)
    assert not isinstance(val, FrozenClassPairDataset)
    assert isinstance(test, IdIncludedDataset)
    assert not isinstance(test, FrozenClassPairDataset)
    assert train.split_manifest_sha256 == approved_sha
    assert train._manifest()["class_preserved_fraction"] == 1.0
    assert train._manifest()["group_preserved_fraction"] == 1.0
    assert (tmp_path / "pairing" / "train.json").is_file()
    assert not (tmp_path / "pairing" / "val.json").exists()
    assert not (tmp_path / "pairing" / "test.json").exists()


def test_consumed_data_fingerprint_is_complete_and_content_sensitive(tmp_path) -> None:
    cache_path = tmp_path / "cache.h5"
    cache_path.write_bytes(b"structural-placeholder")
    metadata = MetadataAccessor(
        pd.DataFrame(
            [
                {"Id": 1, "Dataset_id": 1, "File": "a", "Label": 0},
                {"Id": 2, "Dataset_id": 1, "File": "b", "Label": 1},
            ]
        ),
        key_column="Id",
    )
    factory = data_factory_module.data_factory.__new__(
        data_factory_module.data_factory
    )
    factory.target_metadata = metadata
    factory.data = _ns(h5_file=str(cache_path))
    factory._data_fingerprint_records = {}
    factory._record_data_fingerprint(1, np.arange(6, dtype=np.float32))
    with pytest.raises(RuntimeError, match="coverage mismatch"):
        factory.get_data_fingerprint()
    factory._record_data_fingerprint(2, np.arange(6, dtype=np.float32) + 1)
    first = factory.get_data_fingerprint()
    second = factory.get_data_fingerprint()
    assert first == second
    assert first["eligible_ids"] == 2

    changed = data_factory_module.data_factory.__new__(
        data_factory_module.data_factory
    )
    changed.target_metadata = metadata
    changed.data = _ns(h5_file=str(cache_path))
    changed._data_fingerprint_records = {}
    changed._record_data_fingerprint(1, np.arange(6, dtype=np.float32))
    changed._record_data_fingerprint(2, np.arange(6, dtype=np.float32) + 2)
    assert changed.get_data_fingerprint()["data_payload_sha256"] != first[
        "data_payload_sha256"
    ]


def test_read_only_evidence_cache_fails_without_mutating_missing_ids(tmp_path) -> None:
    metadata = MetadataAccessor(
        pd.DataFrame(
            [
                {"Id": 1, "Dataset_id": 1, "File": "a", "Label": 0},
                {"Id": 2, "Dataset_id": 1, "File": "b", "Label": 1},
            ]
        ),
        key_column="Id",
    )
    cache_path = tmp_path / "cache.h5"
    with h5py.File(cache_path, "w") as handle:
        handle.create_dataset("1", data=np.ones((4, 2), dtype=np.float32))
    factory = data_factory_module.data_factory.__new__(
        data_factory_module.data_factory
    )
    factory.search_dataset_id = lambda: metadata
    args_data = _ns(data_dir=str(tmp_path), read_only_cache_required=True)
    with pytest.raises(RuntimeError, match="refusing mutation"):
        factory._init_data(args_data)
    with h5py.File(cache_path, "r") as handle:
        assert set(handle.keys()) == {"1"}

    with h5py.File(cache_path, "a") as handle:
        handle.create_dataset("2", data=np.ones((4, 2), dtype=np.float32))
    data = factory._init_data(args_data)
    assert data.h5_file == str(cache_path)
    data.close()


def test_frozen_pairing_preserves_class_group_and_partner_marginal(tmp_path) -> None:
    metadata = {
        1: {"Label": 0, "Dataset_id": 1, "File": "normal_a.mat"},
        2: {"Label": 0, "Dataset_id": 1, "File": "normal_b.mat"},
        3: {"Label": 1, "Dataset_id": 1, "File": "fault_a.mat"},
        4: {"Label": 1, "Dataset_id": 1, "File": "fault_b.mat"},
    }
    base = IdIncludedDataset(
        {
            1: _Windows(0, 10.0),
            2: _Windows(0, 20.0),
            3: _Windows(1, 30.0),
            4: _Windows(1, 40.0),
        },
        metadata,
    )
    first = FrozenClassPairDataset(
        base, seed=20260801, split_name="train", manifest_dir=str(tmp_path), group_key="File",
        protocol_id="P01-G040-v1", split_manifest_sha256="split-sha",
    )
    second = FrozenClassPairDataset(
        base, seed=20260801, split_name="train", manifest_dir=str(tmp_path), group_key="File",
        protocol_id="P01-G040-v1", split_manifest_sha256="split-sha",
    )
    reordered_base = IdIncludedDataset(
        {
            4: _Windows(1, 40.0),
            3: _Windows(1, 30.0),
            2: _Windows(0, 20.0),
            1: _Windows(0, 10.0),
        },
        metadata,
    )
    reordered = FrozenClassPairDataset(
        reordered_base,
        seed=20260801,
        split_name="train",
        manifest_dir=str(tmp_path),
        group_key="File",
        protocol_id="P01-G040-v1",
        split_manifest_sha256="split-sha",
    )
    assert first.mapping == second.mapping
    assert first._manifest() == reordered._manifest()
    assert first._manifest()["mapping_sha256"] == reordered._manifest()[
        "mapping_sha256"
    ]
    assert all(index != partner for index, partner in first.mapping.items())
    assert all(first._identity(index) == first._identity(partner) for index, partner in first.mapping.items())
    assert set(first.mapping) == set(first.mapping.values())
    assert first._manifest()["partner_bijection"] is True
    assert first._manifest()["maximum_partner_reuse"] == 1
    assert first._manifest()["class_preserved_fraction"] == 1.0
    assert first._manifest()["group_preserved_fraction"] == 1.0
    for index in range(len(first)):
        sample = first[index]
        assert int(sample["y"]) == first._label(first.mapping[index])
        assert not torch.equal(sample["x"], sample["x_2d"])


def test_frozen_pairing_supports_one_group_with_multiple_windows(tmp_path) -> None:
    metadata = {1: {"Label": 0, "Dataset_id": 1, "File": "normal_only.mat"}}
    base = IdIncludedDataset({1: _Windows(0, 10.0)}, metadata)
    paired = FrozenClassPairDataset(
        base, seed=20260801, split_name="val", manifest_dir=str(tmp_path), group_key="File",
        protocol_id="P01-G040-v1", split_manifest_sha256="split-sha",
    )
    manifest = paired._manifest()
    assert all(index != partner for index, partner in paired.mapping.items())
    assert manifest["group_preserved_pairs"] == len(paired)
    assert manifest["class_preserved_fraction"] == 1.0
    assert manifest["group_preserved_fraction"] == 1.0
    assert manifest["partner_bijection"] is True
    assert manifest["maximum_partner_reuse"] == 1


def test_default_eval_sampler_keeps_the_final_partial_batch() -> None:
    metadata = {1: {"Label": 0, "Dataset_id": 1, "File": "normal.mat"}}
    base = IdIncludedDataset({1: _Windows(0, 10.0)}, metadata)
    sampler = Get_sampler(
        _ns(type="Default_task"), _ns(batch_size=2), base, mode="test"
    )
    batches = list(sampler)
    assert sampler.drop_last is False
    assert sorted(index for batch in batches for index in batch) == list(range(len(base)))
