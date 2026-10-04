"""Condition-only controls disclose input permissions and executed capacity."""
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

from test.test_p08_execution import config
from test.test_p08_model import args
from src.model_factory.ISFM.M_P08_PhysicalConditioning import Model
from src.task_factory.task.DG import p08_physical as runner


def test_condition_control_never_constructs_hse_and_ignores_signal_and_rate():
    with patch('src.model_factory.ISFM.M_P08_PhysicalConditioning.E_01_HSE',
               side_effect=AssertionError('C_ONLY must not instantiate HSE')):
        model = Model(args(fusion='condition_only')).eval()
    p = torch.randn(4, 8)
    a = model(torch.randn(4, 32, 1), fs=1., condition=p)
    b = model(torch.randn(4, 64, 1), fs=10000., condition=p)
    assert torch.equal(a, b)
    assert not torch.allclose(a, model(None, condition=p + 2.))
    assert not hasattr(model, 'embedding') and not hasattr(model, 'backbone')
    counts = model.parameter_counts()
    assert counts['active_parameters'] == counts['stored_parameters'] == (8 * 8 + 8) + (8 * 3 + 3)
    with pytest.raises(ValueError, match='no unconditioned branch'):
        model(None, condition=p, detach_condition=True)


def test_condition_control_source_search_and_checkpoint_replay(config, tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('C_ONLY must not evaluate HSE patches')
    monkeypatch.setattr(runner, '_patches', forbidden)
    result = runner.execute(config, 'tune', tmp_path / 'tune', target='19', seed=42,
                            arms=['C_ONLY'])
    assert result['status'] == 'completed'
    selection = runner._selected(tmp_path / 'tune', config, '19', 42, 'C_ONLY')
    assert len(selection['trials']) == 6
    assert all(t['updates_executed'] == 2 for t in selection['trials'])
    output = runner.execute(config, 'compare', tmp_path / 'compare', target='19', seed=42,
                            arms=['C_ONLY'], selection=tmp_path / 'tune')
    checkpoint_path = tmp_path / 'compare/target-19/seed-42/arm-C_ONLY/checkpoint.pt'
    _, model, encoder = runner._load_checkpoint(checkpoint_path, '19', 42, 'C_ONLY', torch.device('cpu'))
    _, _, target = runner._split(config, runner.load_records(config['data']), '19', 42)
    import json
    saved = json.loads((checkpoint_path.parent / 'predictions.json').read_text())
    assert runner.predict(model, target, encoder, config, torch.device('cpu')) == saved
    assert output['results'][0]['parameter_counts']['active_parameters'] == sum(p.numel() for p in model.parameters())


def test_provenance_does_not_require_git_or_repository(config, monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    with patch('subprocess.run', side_effect=AssertionError('No Git at installed runtime')):
        value = runner._provenance(config)
    assert value['package_version']
    assert value['declared_source_revision'] is None
    assert 'working_tree_status' not in value


def test_declared_h5_singleton_axis_preserves_selected_channel(config, tmp_path):
    import h5py
    import numpy as np
    from src.data_factory.p08_data import windows
    path = tmp_path / 'signal.h5'
    raw = np.stack([np.arange(64), np.arange(64) ** 2], axis=1).astype('float32')
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('record', data=raw[:, :, None])
    record = {'record_id': 'fixture', 'signal_path': str(path), 'signal_key': 'record', 'channel': 1}
    with pytest.raises(ValueError, match='cannot supply channel'):
        windows(record, config['data'])
    configured = {**config['data'], 'signal_layout': 'points_channels_singleton'}
    actual = windows(record, configured)
    with h5py.File(path, 'w') as handle:
        handle.create_dataset('record', data=raw)
    expected = windows(record, config['data'])
    assert torch.equal(actual, expected)
    with pytest.raises(ValueError, match='requires'):
        windows(record, configured)


def test_research_preflight_and_plain_cli_rejection(config, tmp_path, monkeypatch):
    import yaml
    from phmfactory.cli import main
    config['task']['execution'] = 'research'
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    with patch.object(runner, 'execute', side_effect=AssertionError('Preflight must not execute')):
        result = main(['research', 'preflight', '--config', str(path)])
        assert result['execution_verified'] is False
        with pytest.raises(ValueError, match='requires phmfactory research'):
            main(['--config', str(path)])
    assert not Path(config['environment']['output_dir']).exists()
    with pytest.raises(ValueError, match='task.execution only supports'):
        main(['preflight', '--config', str(path), '--override', 'task.execution=training'])
    del config['task']['execution']
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(ValueError, match='requires task.execution=research'):
        main(['preflight', '--config', str(path)])


def test_research_cli_runs_configured_task_and_rejects_unsupported_input(config, tmp_path):
    import yaml
    from phmfactory.cli import main
    config['task']['execution'] = 'research'
    path = tmp_path / 'config.yaml'
    path.write_text(yaml.safe_dump(config))
    out = tmp_path / 'smoke'
    result = main(['research', 'smoke', '--config', str(path), '--output', str(out),
                   '--target', '19', '--seed', '42', '--arms', 'C_ONLY'])
    assert result['status'] == 'completed' and (out / 'summary.json').is_file()
    with pytest.raises(ValueError, match='explicit checkpoint requires ablate'):
        main(['research', 'smoke', '--config', str(path), '--output', str(tmp_path / 'bad'),
              '--checkpoint', str(tmp_path / 'unused.pt')])
    assert not (tmp_path / 'bad').exists()
