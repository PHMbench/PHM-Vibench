"""Real Streamlit rendering with the current public inspector.

Run this separately from the optional-import stub tests.
"""
from pathlib import Path

import yaml
import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]


def _button(app, label):
    return next(button for button in app.button if button.label == label)


def test_real_page_accepts_current_inspection_and_invalidates_edits():
    app = AppTest.from_file(str(ROOT / 'apps/streamlit/app.py'), default_timeout=90).run()
    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    assert _button(app, 'Run experiment').disabled
    _button(app, 'Validate configuration').click().run()
    assert not app.exception
    assert not app.error, [error.value for error in app.error]
    assert not _button(app, 'Run experiment').disabled
    assert app.session_state['validation_report'].ok
    checked = app.session_state['validated_inputs']
    app.number_input(key='field::demo_00_smoke_dummy_dg::epochs').set_value(2).run()
    assert not app.exception
    assert _button(app, 'Run experiment').disabled
    assert app.session_state['validated_inputs'] == checked
    _button(app, 'Validate configuration').click().run()
    assert not app.exception
    assert not _button(app, 'Run experiment').disabled
    assert app.session_state['validation_report'].resolved['trainer']['num_epochs'] == 2
    assert app.session_state['validated_inputs'] != checked


def test_advanced_yaml_resets_common_field_values_before_validation():
    app = AppTest.from_file(str(ROOT / 'apps/streamlit/app.py'), default_timeout=90).run()
    app.radio(key='ui_mode').set_value('Advanced').run()
    assert not app.exception

    config = yaml.safe_load(app.session_state['advanced_yaml_text'])
    config['trainer']['num_epochs'] = 7
    edited = yaml.safe_dump(config, sort_keys=False)
    app.text_area(key='advanced_yaml_text').set_value(edited).run()
    assert not app.exception

    revision = app.session_state['advanced_field_revision::demo_00_smoke_dummy_dg']
    key = f'field::demo_00_smoke_dummy_dg::advanced-{revision}::epochs'
    assert app.number_input(key=key).value == 7

    app.number_input(key=key).set_value(8).run()
    _button(app, 'Validate configuration').click().run()
    assert not app.exception
    assert app.session_state['validation_report'].ok
    assert app.session_state['validation_report'].resolved['trainer']['num_epochs'] == 8

    config['trainer']['num_epochs'] = 9
    app.text_area(key='advanced_yaml_text').set_value(
        yaml.safe_dump(config, sort_keys=False)
    ).run()
    new_revision = app.session_state['advanced_field_revision::demo_00_smoke_dummy_dg']
    new_key = f'field::demo_00_smoke_dummy_dg::advanced-{new_revision}::epochs'
    assert new_revision > revision
    assert app.number_input(key=new_key).value == 9
    assert _button(app, 'Run experiment').disabled


def test_advanced_yaml_invalid_integer_is_not_repaired_by_a_widget():
    app = AppTest.from_file(str(ROOT / 'apps/streamlit/app.py'), default_timeout=90).run()
    app.radio(key='ui_mode').set_value('Advanced').run()
    assert not app.exception

    config = yaml.safe_load(app.session_state['advanced_yaml_text'])
    config['trainer']['num_epochs'] = 1.5
    app.text_area(key='advanced_yaml_text').set_value(
        yaml.safe_dump(config, sort_keys=False)
    ).run()
    assert not app.exception

    _button(app, 'Validate configuration').click().run()
    assert not app.exception
    report = app.session_state['validation_report']
    assert not report.ok
    assert not report.resolved
    assert 'num_epochs' in report.stderr
    assert _button(app, 'Run experiment').disabled


def test_batch_preview_counts_and_invalidates_edits_without_launching(monkeypatch):
    from apps.streamlit import ui_batch

    def forbidden(*args, **kwargs):
        raise AssertionError('Preview or rerun must not start a process')
    monkeypatch.setattr(ui_batch, 'start_batch', forbidden)
    app = AppTest.from_file(str(ROOT / 'apps/streamlit/app.py'), default_timeout=90).run()
    assert not app.exception
    assert _button(app, 'Preview batch').disabled
    assert _button(app, 'Run batch').disabled
    _button(app, 'Validate configuration').click().run()
    assert not app.exception
    _button(app, 'Preview batch').click().run()
    assert not app.exception
    assert len(app.session_state['batch_plan'].trials) == 2
    assert app.session_state['batch_plan'].total_fits == 2
    assert not _button(app, 'Run batch').disabled
    app.run()
    assert not app.exception
    assert not _button(app, 'Run batch').disabled
    app.text_area(key='batch_grid_text').set_value('task.lr: [0.001, 0.0005, 0.0001]').run()
    assert _button(app, 'Run batch').disabled
    _button(app, 'Preview batch').click().run()
    assert not app.exception
    assert len(app.session_state['batch_plan'].trials) == 3
    app.number_input(key='batch_max_fits').set_value(2).run()
    _button(app, 'Preview batch').click().run()
    assert not app.exception
    assert app.session_state['batch_plan'] is None
    assert _button(app, 'Run batch').disabled
    assert any('max_fits' in error.value for error in app.error)


def test_batch_launch_requires_click_and_locks_the_submitted_preview(monkeypatch):
    from types import SimpleNamespace
    from apps.streamlit import ui_batch

    submitted = []
    def submit(request, plan):
        submitted.append((request, plan))
        return SimpleNamespace(batch_id='submitted-batch')
    monkeypatch.setattr(ui_batch, 'start_batch', submit)
    monkeypatch.setattr(ui_batch, 'list_batches', lambda root: ())
    app = AppTest.from_file(str(ROOT / 'apps/streamlit/app.py'), default_timeout=90).run()
    _button(app, 'Validate configuration').click().run()
    _button(app, 'Preview batch').click().run()
    assert not submitted
    _button(app, 'Run batch').click().run()
    assert not app.exception
    assert len(submitted) == 1
    assert submitted[0][1].total_fits == 2
    assert app.session_state['batch_plan'].total_fits == 2
    assert app.session_state['batch_plan_submitted'] is True
    assert _button(app, 'Run batch').disabled
    app.run()
    assert not app.exception
    assert len(submitted) == 1


def test_detached_release_requires_confirmation_in_the_page(monkeypatch, tmp_path):
    from apps.streamlit import ui_runtime

    released = []
    monkeypatch.setattr(ui_runtime, 'release_detached_run',
                        lambda root, run_id, *, confirmed_stopped:
                        released.append((root, run_id, confirmed_stopped)))
    app = AppTest.from_string('''
from pathlib import Path
from apps.streamlit.run_service import RunRecord
from apps.streamlit.ui_runtime import _render_detached_controls
root = Path(".")
record = RunRecord("lost-run", "detached", root, (), error="Process outcome unknown.")
_render_detached_controls(root, record)
''').run()
    assert not app.exception
    assert _button(app, 'Release finished run').disabled
    assert released == []
    app.checkbox(key='stopped-confirmed::lost-run').set_value(True).run()
    assert not _button(app, 'Release finished run').disabled
    assert released == []
    _button(app, 'Release finished run').click().run()
    assert not app.exception
    assert released == [(Path('.'), 'lost-run', True)]


@pytest.mark.parametrize('failure', ['catalogue', 'empty_group', 'template'])
def test_editor_failure_keeps_existing_run_logs_and_cancel(monkeypatch, tmp_path, failure):
    from dataclasses import replace
    from apps.streamlit import workspace, ui_runtime, ui_batch
    from apps.streamlit.config_service import ConfigServiceError, ValidationReport
    from apps.streamlit.run_service import RunRecord

    (tmp_path / 'run.log').write_text('existing-run-log', encoding='utf-8')
    state = {'record': RunRecord('existing-run', 'running', tmp_path, ('python', 'main.py'))}
    cancelled = []
    monkeypatch.setattr(ui_runtime, 'list_runs', lambda root, limit: (state['record'],))
    monkeypatch.setattr(ui_runtime, 'get_run', lambda root, run_id: state['record'])
    monkeypatch.setattr(ui_batch, 'list_batches', lambda root: ())
    def cancel(root, run_id):
        cancelled.append(run_id)
        state['record'] = replace(state['record'], status='cancelled', cancel_requested=True)
        return state['record']
    monkeypatch.setattr(ui_runtime, 'cancel_run', cancel)
    def forbidden(*args, **kwargs):
        raise AssertionError('An editor error or page rerun must never submit a run.')
    monkeypatch.setattr(workspace, 'start_run', forbidden)
    monkeypatch.setattr(ui_batch, 'start_batch', forbidden)
    if failure == 'catalogue':
        def broken(path):
            raise ConfigServiceError('Broken catalogue fixture')
        monkeypatch.setattr(workspace, '_cached_catalog', broken)
    elif failure == 'template':
        monkeypatch.setattr(workspace, '_cached_inspection', lambda *args: ValidationReport(
            False, (), error='Rejected template fixture'))
    app = AppTest.from_file(str(ROOT / 'apps/streamlit/app.py'), default_timeout=90)
    if failure == 'empty_group':
        app.session_state['template_group'] = 'generative_models'
    app.run()
    assert not app.exception
    assert any('existing-run-log' in item.value for item in app.code)
    assert not _button(app, 'Cancel run').disabled
    assert not any(button.label == 'Run experiment' for button in app.button)
    _button(app, 'Cancel run').click().run()
    assert not app.exception
    assert cancelled == ['existing-run']
    assert state['record'].status == 'cancelled'
    app.run()
    assert cancelled == ['existing-run']


def test_broken_editor_keeps_paused_batch_controls(monkeypatch):
    from types import SimpleNamespace
    from apps.streamlit import workspace, ui_batch, ui_runtime
    from apps.streamlit.config_service import ConfigServiceError

    record = SimpleNamespace(batch_id='paused-batch', status='paused', is_terminal=False,
                             total_fits=1, error='Trial failed', trials=(dict(
                                 index=1, status='pending', fit_count=1, run_id='', error=''),))
    cancelled = []
    monkeypatch.setattr(ui_runtime, 'list_runs', lambda root, limit: ())
    monkeypatch.setattr(ui_batch, 'list_batches', lambda root: (record,))
    monkeypatch.setattr(ui_batch, 'get_batch', lambda root, batch_id: record)
    def broken(path):
        raise ConfigServiceError('Broken catalogue fixture')
    monkeypatch.setattr(workspace, '_cached_catalog', broken)
    def cancel(root, batch_id):
        cancelled.append(batch_id)
        record.status = 'cancelled'
        record.is_terminal = True
    monkeypatch.setattr(ui_batch, 'cancel_batch', cancel)
    app = AppTest.from_file(str(ROOT / 'apps/streamlit/app.py'), default_timeout=90).run()
    assert not app.exception
    assert not _button(app, 'Continue remaining trials').disabled
    assert not _button(app, 'Cancel batch').disabled
    _button(app, 'Cancel batch').click().run()
    assert not app.exception
    assert cancelled == ['paused-batch']
    assert _button(app, 'Cancel batch').disabled


def test_unreadable_historical_log_keeps_batch_controls_and_editor(monkeypatch, tmp_path):
    from types import SimpleNamespace
    from apps.streamlit import workspace, ui_batch, ui_runtime
    from apps.streamlit.run_service import RunRecord

    record = RunRecord('old-run', 'succeeded', tmp_path, ('python', 'main.py'))
    batch = SimpleNamespace(batch_id='paused-batch', status='paused', is_terminal=False,
                            total_fits=1, error='Trial failed', trials=(dict(
                                index=1, status='pending', fit_count=1, run_id='', error=''),))
    monkeypatch.setattr(ui_runtime, 'list_runs', lambda root, limit: (record,))
    monkeypatch.setattr(ui_runtime, 'get_run', lambda root, run_id: record)
    monkeypatch.setattr(ui_batch, 'list_batches', lambda root: (batch,))
    monkeypatch.setattr(ui_batch, 'get_batch', lambda root, batch_id: batch)
    def unreadable(*args, **kwargs):
        raise PermissionError('Cannot read old-run/run.log')
    monkeypatch.setattr(ui_runtime, 'read_log_tail', unreadable)
    def forbidden(*args, **kwargs):
        raise AssertionError('Displaying a damaged historical run must not submit work.')
    monkeypatch.setattr(workspace, 'start_run', forbidden)
    monkeypatch.setattr(ui_batch, 'start_batch', forbidden)
    app = AppTest.from_file(str(ROOT / 'apps/streamlit/app.py'), default_timeout=90).run()
    assert not app.exception
    assert not _button(app, 'Continue remaining trials').disabled
    assert not _button(app, 'Cancel batch').disabled
    assert not _button(app, 'Validate configuration').disabled
    assert _button(app, 'Run experiment').disabled
    assert app.session_state['selected_run_id'] == 'old-run'
    assert any('Cannot read old-run/run.log' in item.value for item in app.error)
    _button(app, 'Validate configuration').click().run()
    assert not app.exception
    assert app.session_state['validation_report'].ok
    assert app.session_state['selected_run_id'] == 'old-run'


def test_damaged_batch_record_keeps_editor_and_single_run_view(monkeypatch, tmp_path):
    from apps.streamlit import workspace, ui_runtime, ui_batch, run_service

    directory = tmp_path / 'batches' / 'damaged'
    directory.mkdir(parents=True)
    path = directory / 'batch.json'
    path.write_text('{}', encoding='utf-8')
    monkeypatch.setattr(run_service, '_batch_root', lambda root: directory.parent)
    monkeypatch.setattr(ui_runtime, 'list_runs', lambda root, limit: ())
    def forbidden(*args, **kwargs):
        raise AssertionError('A malformed history must not trigger any submission.')
    monkeypatch.setattr(workspace, 'start_run', forbidden)
    monkeypatch.setattr(ui_batch, 'start_batch', forbidden)
    app = AppTest.from_file(str(ROOT / 'apps/streamlit/app.py'), default_timeout=90).run()
    assert not app.exception
    assert not _button(app, 'Validate configuration').disabled
    assert any(str(path) in error.value for error in app.error)
    _button(app, 'Validate configuration').click().run()
    assert not app.exception
    assert app.session_state['validation_report'].ok
    assert not _button(app, 'Run experiment').disabled
    assert not any(button.label == 'Run batch' for button in app.button)
    assert path.read_text() == '{}'
