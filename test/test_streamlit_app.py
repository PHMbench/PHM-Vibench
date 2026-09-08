"""Real Streamlit rendering with the current public inspector.

Run this separately from the optional-import stub tests.
"""
from pathlib import Path

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
