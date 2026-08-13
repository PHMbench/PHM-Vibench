import hashlib
import importlib
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src.data_factory.data_factory import _cache_directory, data_factory
from src.data_factory.data_utils import MetadataAccessor


def _inventory(root):
    return {
        path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in root.rglob("*")
        if path.is_file()
    }


def test_cache_dir_keeps_hash_ledgered_source_tree_immutable(tmp_path, monkeypatch):
    source = tmp_path / "immutable_source"
    cache = tmp_path / "runtime_cache"
    raw = source / "raw" / "P04_Synthetic"
    raw.mkdir(parents=True)
    (raw / "sample.npy").write_bytes(b"reader fixture")
    frame = pd.DataFrame(
        [
            {
                "Id": 904000000,
                "Dataset_id": 904,
                "Domain_id": 0,
                "Label": 0,
                "Name": "P04_Synthetic",
                "File": "sample.npy",
            }
        ]
    )
    frame.to_csv(source / "metadata.csv", index=False)
    metadata = MetadataAccessor(frame, key_column="Id")
    args = SimpleNamespace(
        data_dir=str(source),
        cache_dir=str(cache),
        metadata_file="metadata.csv",
    )

    reader = SimpleNamespace(
        read=lambda path, args_data: np.arange(16, dtype=np.float32).reshape(8, 2)
    )
    module = importlib.import_module("src.data_factory.data_factory")
    monkeypatch.setattr(module.importlib, "import_module", lambda module_name: reader)
    factory = data_factory.__new__(data_factory)
    factory.metadata = metadata
    factory.args_task = SimpleNamespace(type="generative", target_system_id=None)
    before = _inventory(source)

    data = factory._init_data(args, use_cache=True, max_workers=1)
    try:
        np.testing.assert_array_equal(
            data[904000000], np.arange(16, dtype=np.float32).reshape(8, 2, 1)
        )
    finally:
        data.close()

    assert _inventory(source) == before
    assert sorted(path.name for path in cache.iterdir()) == [
        "P04_Synthetic.h5",
        "cache.h5",
    ]

    monkeypatch.setattr(
        module.importlib,
        "import_module",
        lambda module_name: (_ for _ in ()).throw(AssertionError("reader reused")),
    )
    reused = factory._init_data(args, use_cache=True, max_workers=1)
    try:
        assert reused[904000000].shape == (8, 2, 1)
    finally:
        reused.close()
    assert _inventory(source) == before


def test_cache_dir_defaults_to_data_dir_for_existing_configs(tmp_path):
    args = SimpleNamespace(data_dir=str(tmp_path))
    assert _cache_directory(args) == str(tmp_path)
