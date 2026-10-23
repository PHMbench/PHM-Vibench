"""Native TII data boundaries using constructed source fixtures."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
import torch

from phmfactory.config import analyze_config
from scripts.tii_make_fixture import main as make_fixture
from src.data_factory import build_data


@pytest.fixture
def native_fixture(tmp_path: Path):
    root = tmp_path / "fixture"
    with patch("sys.argv", ["tii_make_fixture", "--output", str(root)]):
        make_fixture()
    resolved = analyze_config(root / "native.yaml")
    args = {
        key: SimpleNamespace(**value)
        for key, value in resolved.runtime_config().items()
        if isinstance(value, dict)
    }
    return root, args["data"], args["task"]


@pytest.mark.parametrize(
    "filename,column,message",
    [
        ("metadata.csv", "Label", "labels must be finite integers"),
        ("records.csv", "channel", "channel must be a nonnegative integer"),
    ],
)
def test_fractional_label_or_channel_fails_before_signal_read(
    native_fixture, filename: str, column: str, message: str
) -> None:
    root, args_data, args_task = native_fixture
    path = root / filename
    table = pd.read_csv(path)
    table.loc[0, column] = 0.5
    table.to_csv(path, index=False)

    with (
        patch("src.data_factory.tii_data.h5py.File") as open_h5,
        patch("src.data_factory.tii_data.project_window") as project,
        pytest.raises(ValueError, match=message),
    ):
        build_data(args_data, args_task)
    open_h5.assert_not_called()
    project.assert_not_called()
    assert not (root / "scratch").exists()


def test_native_data_builds_complete_source_rounds_and_validation(native_fixture) -> None:
    _, args_data, args_task = native_fixture
    factory = build_data(args_data, args_task)

    # Analytic fixture energy: four equally weighted train groups/source;
    # only source 1 has the orthogonal 0.3-amplitude incremental sinusoid.
    train_amplitudes = 1.0 + 0.03 * np.arange(4)
    expected_rms = np.sqrt(((1.0 + 0.3**2) / 2 + 1.0 / 2) / 2
                           * np.mean(train_amplitudes**2))
    assert factory.source_rms == pytest.approx(expected_rms, abs=1e-7)
    assert len(factory.get_dataloader("train")) == 20
    round_batch = next(iter(factory.get_dataloader("train")))
    assert set(round_batch) == {1, 2}
    for source, batch in round_batch.items():
        assert batch["x"].shape == batch["incremental"].shape == (32, 4, 16)
        assert batch["y"].shape == batch["availability"].shape == (32,)
        assert batch["y"].dtype == torch.long
        assert set(batch["group"]) <= {0, 1, 2, 3}
        assert set(batch["role"]) == {"source_train"}
        assert (batch["availability"] == (source == 1)).all()

    validation = list(factory.get_dataloader("val"))
    assert {batch["source"] for batch in validation} == {1, 2}
    for batch in validation:
        assert batch["x"].shape == batch["incremental"].shape == (8, 4, 16)
        assert set(batch["group"]) == {4, 5, 6, 7}
        assert set(batch["role"]) == {"source_val"}
        assert set(batch["y"].tolist()) == {0, 1}
        assert torch.isfinite(batch["x"]).all()
    assert factory.get_dataloader("test") is None
    assert len(factory.window_inventory) == 32
    zero_id = [row for row in factory.window_inventory if row["file_id"] == 0]
    assert len(zero_id) == 2
    assert all(row["recording_id"] == row["group"] == row["channel"] == 0
               for row in zero_id)
