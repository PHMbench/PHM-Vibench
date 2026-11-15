"""Public research commands preserve the requested task and failure status."""
from types import SimpleNamespace
from pathlib import Path
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
    monkeypatch.setattr(research, 'task_module', lambda config: SimpleNamespace(execute=execute, PHASES=('synthetic',)))
    with pytest.raises(RuntimeError, match='fixture failure'):
        main(['research', 'synthetic', '--config', str(config_path(tmp_path))])


def test_unexpected_checkpoint_is_rejected_before_operator_execution(tmp_path, monkeypatch):
    from phmfactory.commands import research
    def execute(config, phase, output):
        pytest.fail('Must fail at the public input boundary')
    monkeypatch.setattr(research, 'task_module', lambda config: SimpleNamespace(execute=execute, PHASES=('synthetic',)))
    with pytest.raises(TypeError, match='unexpected keyword'):
        main(['research', 'synthetic', '--config', str(config_path(tmp_path)),
              '--checkpoint', str(tmp_path / 'missing.pt')])


def test_factory_preflight_does_not_import_training_stack(monkeypatch):
    from phmfactory.commands import research
    monkeypatch.setattr(research.importlib, 'import_module',
                        lambda name: pytest.fail('Factory preflight must not import a task'))
    research.reject_scientific_factory_task({'task': {'type': 'DG', 'name': 'classification'}})


def _declared_phase_config(tmp_path, *, phase='synthetic', options=None, pipeline=None):
    path = config_path(tmp_path)
    config = yaml.safe_load(path.read_text())
    config['task']['phase'] = phase
    if options is not None:
        config['task']['options'] = options
    if pipeline is not None:
        config['pipeline'] = pipeline
    path.write_text(yaml.safe_dump(config))
    return path


def test_plain_and_research_commands_share_one_resolved_execution(tmp_path, monkeypatch):
    from phmfactory import cli
    from phmfactory.commands import research
    analyses = []
    calls = []

    def analyze(*args, **kwargs):
        result = analyze_config(*args, **kwargs)
        analyses.append(result)
        return result

    def execute(config, phase, output, *, data, checkpoint, seed, device):
        calls.append((config, phase, output, data, checkpoint, seed, device))
        return {'status': 'completed', 'result_dir': str(output)}

    monkeypatch.setattr(cli, 'analyze_config', analyze)
    monkeypatch.setattr(research, 'analyze_config', analyze)
    monkeypatch.setattr(research, 'task_module', lambda config: SimpleNamespace(
        execute=execute, PHASES=('synthetic',)))
    monkeypatch.setattr(cli, 'pipeline_module_name',
                        lambda *a, **kw: pytest.fail('Must not enter a training Pipeline'))
    path = _declared_phase_config(tmp_path, options={
        'data': 'inputs.npz', 'checkpoint': 'state.pt', 'seed': 17, 'device': 'cpu'})
    before = yaml.safe_load(path.read_text())
    for argv in (['--config', str(path)], ['research', 'synthetic', '--config', str(path)]):
        previous = len(analyses)
        result = main(argv)
        assert len(analyses) == previous + 1
        assert result['status'] == 'completed'
    assert calls[0] == calls[1]
    assert calls[0][1] == 'synthetic'
    assert calls[0][3:5] == (Path('inputs.npz'), Path('state.pt'))
    assert calls[0][5:] == (17, 'cpu')
    assert analyses[0].runtime_config() == before
    assert not (tmp_path / 'results').exists()  # Only the selected task owns its output.


def test_research_command_explicit_options_override_config_options(tmp_path, monkeypatch):
    from phmfactory.commands import research
    observed = {}

    def execute(config, phase, output, *, seed, device, checkpoint):
        observed.update(seed=seed, device=device, checkpoint=checkpoint, output=output)
        return {'status': 'completed'}

    monkeypatch.setattr(research, 'task_module', lambda config: SimpleNamespace(
        execute=execute, PHASES=('synthetic',)))
    path = _declared_phase_config(tmp_path, options={
        'seed': 17, 'device': 'cpu', 'checkpoint': 'from-config.pt'})
    main(['research', 'synthetic', '--config', str(path), '--seed', '19',
          '--device', 'cuda:2', '--checkpoint', 'explicit.pt', '--output', str(tmp_path / 'out')])
    assert observed == {'seed': 19, 'device': 'cuda:2',
                        'checkpoint': Path('explicit.pt'), 'output': tmp_path / 'out'}


def test_plain_research_requires_phase_before_import(tmp_path, monkeypatch):
    from phmfactory.commands import research
    monkeypatch.setattr(research, 'task_module', lambda c: pytest.fail('No phase; no import'))
    with pytest.raises(ValueError, match='explicit task.phase'):
        main(['--config', str(config_path(tmp_path))])


def test_conflicting_phase_does_not_execute(tmp_path, monkeypatch):
    from phmfactory.commands import research
    monkeypatch.setattr(research, 'task_module', lambda c: pytest.fail('Conflicting phases'))
    with pytest.raises(ValueError, match='conflicts with task.phase'):
        main(['research', 'tune', '--config', str(_declared_phase_config(tmp_path))])


def test_unsupported_phase_is_rejected_before_task_runs(tmp_path, monkeypatch):
    from phmfactory.commands import research
    execute = Mock(side_effect=AssertionError('Must not execute'))
    monkeypatch.setattr(research, 'task_module', lambda c: SimpleNamespace(
        execute=execute, PHASES=('synthetic',)))
    with pytest.raises(ValueError, match='supports'):
        main(['--config', str(_declared_phase_config(tmp_path, phase='fit'))])
    execute.assert_not_called()
    assert not (tmp_path / 'results').exists()


@pytest.mark.parametrize('phase', ['', ' fit ', 1, None])
def test_invalid_config_phase_rejected_without_mutation(tmp_path, phase):
    path = _declared_phase_config(tmp_path, phase=phase)
    with pytest.raises(ValueError, match='task.phase'):
        analyze_config(path)


@pytest.mark.parametrize('options', [[], 'seed=17', None])
def test_invalid_execute_options_rejected(tmp_path, options):
    path = _declared_phase_config(tmp_path)
    config = yaml.safe_load(path.read_text())
    config['task']['options'] = options
    path.write_text(yaml.safe_dump(config))
    with pytest.raises(TypeError, match='task.options'):
        analyze_config(path)


@pytest.mark.parametrize('options', [{'phase': 'tune'}, {'config': {}}, {'output': 'other'}])
def test_reserved_execute_options_cannot_replace_request(tmp_path, monkeypatch, options):
    from phmfactory.commands import research
    execute = Mock(side_effect=AssertionError('Must not execute'))
    monkeypatch.setattr(research, 'task_module', lambda c: SimpleNamespace(
        execute=execute, PHASES=('synthetic',)))
    with pytest.raises(ValueError, match='reserved'):
        main(['--config', str(_declared_phase_config(tmp_path, options=options))])
    execute.assert_not_called()


def test_plain_research_keeps_maturity_opt_in(tmp_path, monkeypatch):
    from phmfactory.commands import research
    from phmfactory.pipelines import PipelineMaturityError
    execute = Mock(return_value={'status': 'completed'})
    module = Mock(return_value=SimpleNamespace(execute=execute, PHASES=('synthetic',)))
    monkeypatch.setattr(research, 'task_module', module)
    path = _declared_phase_config(tmp_path, pipeline='Pipeline_03_Multitask_Pretraining_Finetuning')
    with pytest.raises(PipelineMaturityError, match='--allow-experimental'):
        main(['--config', str(path)])
    module.assert_not_called()
    main(['--config', str(path), '--allow-experimental'])
    execute.assert_called_once()


@pytest.mark.parametrize('bad_result', [None, [], {'status': 'error', 'reason': 'operator failed'}])
def test_plain_research_cannot_publish_failed_completion(tmp_path, monkeypatch, capsys, bad_result):
    from phmfactory.commands import research
    monkeypatch.setattr(research, 'task_module', lambda c: SimpleNamespace(
        execute=lambda **kwargs: bad_result, PHASES=('synthetic',)))
    with pytest.raises((TypeError, RuntimeError)):
        main(['--config', str(_declared_phase_config(tmp_path))])
    assert 'run=completed' not in capsys.readouterr().out


def test_plain_research_preserves_operator_exception(tmp_path, monkeypatch):
    from phmfactory.commands import research
    error = RuntimeError('original task failure')
    execute = Mock(side_effect=error)
    monkeypatch.setattr(research, 'task_module', lambda c: SimpleNamespace(
        execute=execute, PHASES=('synthetic',)))
    with pytest.raises(RuntimeError) as caught:
        main(['--config', str(_declared_phase_config(tmp_path))])
    assert caught.value is error
