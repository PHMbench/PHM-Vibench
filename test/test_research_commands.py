"""Public research commands preserve the requested task and failure status."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from phmfactory.cli import main
from phmfactory.config import analyze_config


def config_path(tmp_path):
    path = tmp_path / 'research.yaml'
    path.write_text(yaml.safe_dump(dict(
        pipeline='Pipeline_01_Fault_Diagnosis',
        environment={'output_dir': str(tmp_path / 'results')},
        data={'kind': 'declared_fixture'}, model={'object': 'saved affine head'},
        task={'type': 'classification', 'name': 'symbolic_diagnosis', 'execution': 'research'},
        trainer={'device': 'cpu'})))
    return path


def test_non_gradient_task_has_no_fake_lightning_configuration(tmp_path):
    analysis = analyze_config(config_path(tmp_path))
    assert analysis.runtime_config()['model'] == {'object': 'saved affine head'}
    assert analysis.runtime_config()['task']['type'] == 'classification'
    with pytest.raises(Exception):
        analyze_config(config_path(tmp_path), override_values=['task.execution=training'])


def test_both_preflight_commands_never_call_execute_or_write_results(tmp_path, monkeypatch):
    from phmfactory.commands import research
    execute = Mock(side_effect=AssertionError('Must not execute'))
    module = SimpleNamespace(__name__='declared scientific task', execute=execute, PHASES=('synthetic',))
    monkeypatch.setattr(research, 'task_module', lambda config: module)
    path = config_path(tmp_path)
    for prefix in (['preflight'], ['research', 'preflight']):
        result = main([*prefix, '--config', str(path)])
        assert not result['execution_verified'] and not result['data_qualified']
    execute.assert_not_called()
    assert not (tmp_path / 'results').exists()


def test_failure_mapping_cannot_be_announced_as_completion(tmp_path, monkeypatch):
    from phmfactory.commands import research
    def execute(config, phase, output):
        return {'status': 'failed', 'reason': 'fixture failure'}
    monkeypatch.setattr(research, 'task_module', lambda config: SimpleNamespace(execute=execute))
    with pytest.raises(RuntimeError, match='fixture failure'):
        main(['research', 'synthetic', '--config', str(config_path(tmp_path))])


def test_unexpected_checkpoint_is_rejected_before_operator_execution(tmp_path, monkeypatch):
    from phmfactory.commands import research
    def execute(config, phase, output):
        pytest.fail('Must fail at the public input boundary')
    monkeypatch.setattr(research, 'task_module', lambda config: SimpleNamespace(execute=execute))
    with pytest.raises(TypeError, match='unexpected keyword'):
        main(['research', 'synthetic', '--config', str(config_path(tmp_path)),
              '--checkpoint', str(tmp_path / 'missing.pt')])


def test_factory_preflight_does_not_import_training_stack(monkeypatch):
    from phmfactory.commands import research
    monkeypatch.setattr(research.importlib, 'import_module',
                        lambda name: pytest.fail('Factory preflight must not import a task'))
    research.reject_scientific_factory_task({'task': {'type': 'DG', 'name': 'classification'}})
