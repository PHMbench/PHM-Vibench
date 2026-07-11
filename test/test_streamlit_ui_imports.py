from __future__ import annotations

import importlib
import sys
import types


class _Decorator:
    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda function: function


def test_ui_modules_import_with_optional_streamlit_stub(monkeypatch):
    """Keep module wiring testable even when the optional UI is not installed."""

    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.cache_data = _Decorator()
    fake_streamlit.fragment = _Decorator()
    fake_streamlit.session_state = {}
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)

    modules = (
        "apps.streamlit.ui_theme",
        "apps.streamlit.ui_runtime",
        "apps.streamlit.workspace",
        "apps.streamlit.app",
        "streamlit_app",
    )
    for name in modules:
        sys.modules.pop(name, None)
        imported = importlib.import_module(name)
        assert imported is not None
