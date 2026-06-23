from __future__ import annotations

import pytest


def test_streamlit_workbench_smoke() -> None:
    testing = pytest.importorskip("streamlit.testing.v1")
    AppTest = testing.AppTest

    app = AppTest.from_file("frontend/streamlit_app.py")
    app.run()
    assert len(app.exception) == 0
