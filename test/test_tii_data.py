"""Native TII data boundaries using constructed source fixtures."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import h5py
import pandas as pd
import pytest
import torch

from phmfactory.config import analyze_config
from scripts.tii_make_fixture import main as make_fixture
from src.data_factory import build_data
from src.data_factory.tii_data import load_tii_target_data
from src.model_factory import build_model
from src.task_factory import build_task


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


def test_target_episode_uses_same_projection_and_excludes_source_groups(native_fixture) -> None:
    root, args_data, args_task = native_fixture
    factory = build_data(args_data, args_task)
    metadata, inventory = [], []
    with h5py.File(root / 'target_fixture.h5', 'w') as h5:
        for label in range(2):
            for i in range(14):
                fid = 100 + 20 * label + i
                signal = np.sin(2 * np.pi * (3 + label) * np.arange(64) / 64)
                h5.create_dataset(str(fid), data=signal.astype('float32')[:, None, None])
                metadata.append(dict(Id=fid, Name='target_fixture', Dataset_id=3, Label=label,
                                     Sample_rate=64, Channel=1, File=f'target_{fid}', Domain_id=fid))
                inventory.append(dict(Id=fid, dataset_id=3, recording_id=fid, group=fid, role='target',
                    native_rate_hz=64, effective_rate_hz=64, channel=0, input_unit='m/s^2', unit_scale=1.,
                    common_support='fully_usable', increment_support='fully_usable',
                    support_basis='tensor_fixture', support_evidence='constructed test'))
    pd.DataFrame(metadata).to_csv(root / 'target_metadata.csv', index=False)
    pd.DataFrame(inventory).to_csv(root / 'target_records.csv', index=False)
    arguments = dict(target_dataset_id=3, source_dataset_ids=[1, 2],
                     source_group_ids=[r['group'] for r in factory.window_inventory])
    collections, windows, episode = load_tii_target_data(
        args_data, root / 'target_metadata.csv', root / 'target_records.csv', **arguments)
    assert len(episode) == len(windows) == 28
    assert len(collections['support'][3]['y']) == 10
    assert len(collections['query'][3]['y']) == 18
    assert set(collections['support'][3]['group']).isdisjoint(collections['query'][3]['group'])
    assert collections['support'][3]['x'].shape[1:] == (4, 16)
    with pytest.raises(ValueError, match='source and target physical groups'):
        load_tii_target_data(args_data, root / 'target_metadata.csv', root / 'target_records.csv',
                             **dict(arguments, source_group_ids=[100]))
    with pytest.raises(ValueError, match='excluded'):
        load_tii_target_data(args_data, root / 'target_metadata.csv', root / 'target_records.csv',
                             **dict(arguments, source_dataset_ids=[1, 3]))


@pytest.mark.parametrize('embedding', ['E_01_HSE', 'E_03_Patch'])
def test_projected_baselines_share_the_source_task(native_fixture, embedding) -> None:
    root, args_data, args_task = native_fixture
    factory = build_data(args_data, args_task)
    resolved = analyze_config(root / 'native.yaml').runtime_config()
    model_args = SimpleNamespace(**resolved['model'])
    model_args.embedding = embedding
    model_args.source_rms = factory.source_rms
    model_args.window_size = 64
    model_args.input_dim = 1
    model_args.patch_size_C = 1
    network = build_model(model_args, metadata=factory.get_metadata())
    task = build_task(args_task=args_task, network=network, args_data=args_data, args_model=model_args,
                      args_trainer=SimpleNamespace(**resolved['trainer']),
                      args_environment=SimpleNamespace(**resolved['environment']),
                      metadata=factory.get_metadata())
    loss = task.joint_loss(next(iter(factory.get_dataloader('train'))))
    loss.backward()
    assert torch.isfinite(loss)
    assert any(p.grad is not None and torch.isfinite(p.grad).all() for p in network.embedding.parameters())
    with patch.object(network, '_head', side_effect=AssertionError('source head accessed')):
        features = task.frozen_features(factory.train_dataset.sources[2], effective_rate_hz=16)
    assert features.shape == (8, 8)
    assert torch.isfinite(features).all()
