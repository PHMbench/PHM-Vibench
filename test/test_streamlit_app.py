"""Real Streamlit rendering with the current public inspector.

Run this separately from the optional-import stub tests.
"""
from pathlib import Path

import yaml
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
