from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.utils.evaluation_artifacts import export_classification_artifacts


class _Dataset(Dataset):
    def __len__(self) -> int:
        return 3

    def __getitem__(self, index: int):  # type: ignore[no-untyped-def]
        return {
            "x": torch.tensor([[float(index)], [float(index + 1)]]),
            "y": index % 2,
            "file_id": index + 1,
            "window_id": 0,
        }


class _Network(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(1, 2)
        self._state = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pooled = x.mean(dim=1)
        self._state = {"shared_1d": pooled}
        return self.linear(pooled)

    def get_representation_state(self, *, detach: bool = True):  # type: ignore[no-untyped-def]
        assert self._state is not None
        if detach:
            return {key: value.detach() for key, value in self._state.items()}
        return dict(self._state)


class _Task(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.network = _Network()

    def forward(self, batch):  # type: ignore[no-untyped-def]
        return self.network(batch["x"])


class _NonFiniteTask(_Task):
    def forward(self, batch):  # type: ignore[no-untyped-def]
        logits = super().forward(batch).clone()
        logits[0, 0] = torch.nan
        return logits


def test_export_is_complete_grouped_and_fail_closed(tmp_path: Path) -> None:
    task = _Task()
    loader = DataLoader(_Dataset(), batch_size=2, shuffle=False, drop_last=False)
    metadata = {
        1: {"File": "bearing_a/one.csv"},
        2: {"File": "bearing_b/two.csv"},
        3: {"File": "bearing_c/three.csv"},
    }
    target = tmp_path / "predictions.npz"
    manifest = export_classification_artifacts(
        task, loader, target, metadata=metadata, group_key="FileParent",
        outer_fold=2, training_seed=123,
        expected_file_ids=[1, 2, 3],
        expected_group_ids=["bearing_a", "bearing_b", "bearing_c"],
        required_representation_names=["shared_1d"],
        provenance={"protocol_id": "P01-G040-v1", "arm_id": "FULL"},
    )
    assert manifest["samples"] == 3
    assert manifest["groups"] == 3
    assert manifest["outer_fold"] == 2
    assert manifest["training_seed"] == 123
    assert manifest["coverage_audit"]["duplicate_sample_keys"] == 0
    assert manifest["required_representation_arrays"] == ["repr__shared_1d"]
    assert manifest["provenance"]["protocol_id"] == "P01-G040-v1"
    with np.load(target, allow_pickle=False) as artifact:
        assert artifact["logits"].shape == (3, 2)
        assert artifact["repr__shared_1d"].shape == (3, 1)
        assert artifact["group_id"].tolist() == [
            "bearing_a", "bearing_b", "bearing_c"
        ]
        assert artifact["outer_fold"].tolist() == [2, 2, 2]
        assert artifact["training_seed"].tolist() == [123, 123, 123]
    with pytest.raises(FileExistsError, match="overwrite"):
        export_classification_artifacts(
            task, loader, target, metadata=metadata, group_key="FileParent",
            outer_fold=2, training_seed=123,
            expected_file_ids=[1, 2, 3],
            expected_group_ids=["bearing_a", "bearing_b", "bearing_c"],
        )


def test_export_rejects_missing_required_representation(tmp_path: Path) -> None:
    task = _Task()
    loader = DataLoader(_Dataset(), batch_size=2, shuffle=False, drop_last=False)
    metadata = {
        1: {"File": "bearing_a/one.csv"},
        2: {"File": "bearing_b/two.csv"},
        3: {"File": "bearing_c/three.csv"},
    }
    with pytest.raises(ValueError, match="private_1d"):
        export_classification_artifacts(
            task,
            loader,
            tmp_path / "missing.npz",
            metadata=metadata,
            group_key="FileParent",
            outer_fold=0,
            training_seed=42,
            expected_file_ids=[1, 2, 3],
            expected_group_ids=["bearing_a", "bearing_b", "bearing_c"],
            required_representation_names=["private_1d"],
        )


@pytest.mark.parametrize(
    ("expected_files", "expected_groups", "message"),
    [
        ([1, 2], ["bearing_a", "bearing_b", "bearing_c"], "file coverage"),
        ([1, 2, 3], ["bearing_a", "bearing_b"], "group coverage"),
    ],
)
def test_export_rejects_incomplete_split_coverage(
    tmp_path: Path,
    expected_files,
    expected_groups,
    message: str,
) -> None:
    task = _Task()
    loader = DataLoader(_Dataset(), batch_size=2, shuffle=False, drop_last=False)
    metadata = {
        1: {"File": "bearing_a/one.csv"},
        2: {"File": "bearing_b/two.csv"},
        3: {"File": "bearing_c/three.csv"},
    }
    with pytest.raises(AssertionError, match=message):
        export_classification_artifacts(
            task,
            loader,
            tmp_path / f"{message.replace(' ', '_')}.npz",
            metadata=metadata,
            group_key="FileParent",
            outer_fold=0,
            training_seed=42,
            expected_file_ids=expected_files,
            expected_group_ids=expected_groups,
        )


def test_export_rejects_non_finite_logits(tmp_path: Path) -> None:
    loader = DataLoader(_Dataset(), batch_size=2, shuffle=False, drop_last=False)
    metadata = {
        1: {"File": "bearing_a/one.csv"},
        2: {"File": "bearing_b/two.csv"},
        3: {"File": "bearing_c/three.csv"},
    }
    with pytest.raises(ValueError, match="non-finite logits"):
        export_classification_artifacts(
            _NonFiniteTask(),
            loader,
            tmp_path / "non_finite.npz",
            metadata=metadata,
            group_key="FileParent",
            outer_fold=0,
            training_seed=42,
            expected_file_ids=[1, 2, 3],
            expected_group_ids=["bearing_a", "bearing_b", "bearing_c"],
        )


@pytest.mark.parametrize(
    ("expected_file_ids", "expected_group_ids", "message"),
    [
        ([1, 2, 4], ["bearing_a", "bearing_b", "bearing_c"], "file coverage"),
        ([1, 2, 3], ["bearing_a", "bearing_b", "bearing_z"], "group coverage"),
    ],
)
def test_export_rejects_inexact_test_identity_coverage(
    tmp_path: Path,
    expected_file_ids,
    expected_group_ids,
    message: str,
) -> None:  # type: ignore[no-untyped-def]
    task = _Task()
    loader = DataLoader(_Dataset(), batch_size=2, shuffle=False, drop_last=False)
    metadata = {
        1: {"File": "bearing_a/one.csv"},
        2: {"File": "bearing_b/two.csv"},
        3: {"File": "bearing_c/three.csv"},
    }
    target = tmp_path / "predictions.npz"
    with pytest.raises(AssertionError, match=message):
        export_classification_artifacts(
            task,
            loader,
            target,
            metadata=metadata,
            group_key="FileParent",
            outer_fold=2,
            training_seed=123,
            expected_file_ids=expected_file_ids,
            expected_group_ids=expected_group_ids,
        )
    assert not target.exists()
    assert not target.with_suffix(".manifest.json").exists()
