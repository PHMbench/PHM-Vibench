from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from scripts.p09.g060_representation import (
    HSEDLinearGlobalHead,
    WindowBank,
    manifest_patch_starts,
    model_state_sha256,
)
from scripts.p09.run_g060_source_fit import epoch_keys, grouped_batches


def _bank(path: Path) -> None:
    with h5py.File(path, "w") as handle:
        for system in (1, 2):
            for class_id in range(4):
                for record_index in range(2):
                    record_id = system * 100 + class_id * 10 + record_index
                    dataset = handle.create_dataset(
                        str(record_id),
                        data=np.zeros((4, 32, 1), dtype=np.float32),
                    )
                    dataset.attrs["system_id"] = system
                    dataset.attrs["canonical_label"] = class_id
                    dataset.attrs["sample_rate"] = 12000.0


def test_window_bank_balanced_keys_never_include_novel(tmp_path: Path) -> None:
    path = tmp_path / "bank.h5"
    _bank(path)
    with WindowBank(path) as bank:
        keys = epoch_keys(
            bank,
            [1, 2],
            seed=42,
            epoch=1,
            per_system_class=4,
        )
        assert len(keys) == 16
        assert {bank.records[record_id].canonical_label for record_id, _ in keys} == {
            0,
            1,
        }
        batches = grouped_batches(
            bank, keys, seed=42, epoch=1, batch_size=5
        )
        assert sum(map(len, batches)) == len(keys)
        assert all(
            len({bank.records[record_id].channels for record_id, _ in batch}) == 1
            for batch in batches
        )


def test_manifest_patch_starts_are_keyed_and_in_range(tmp_path: Path) -> None:
    path = tmp_path / "bank.h5"
    _bank(path)
    with WindowBank(path) as bank:
        keys = [(100, 0), (100, 1)]
        first = manifest_patch_starts(
            keys,
            bank.records,
            length=32,
            patch_size_L=8,
            patch_size_C=1,
            num_patches=4,
            sampling_seed=19,
        )
        second = manifest_patch_starts(
            keys,
            bank.records,
            length=32,
            patch_size_L=8,
            patch_size_C=1,
            num_patches=4,
            sampling_seed=19,
        )
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        assert first[0].min() >= 0 and first[0].max() <= 24
        assert first[1].min() == first[1].max() == 0


def test_representation_has_one_global_head_and_stable_state_digest() -> None:
    config = {
        "patch_size_L": 8,
        "patch_size_C": 1,
        "num_patches": 4,
        "output_dim": 16,
    }
    torch.manual_seed(5)
    model = HSEDLinearGlobalHead(config)
    assert model.global_base_head.out_features == 2
    assert model.global_base_head.in_features == 16
    first = model_state_sha256(model)
    second = model_state_sha256(model)
    assert first == second
    with torch.no_grad():
        model.global_base_head.bias[0] += 1.0
    assert model_state_sha256(model) != first
