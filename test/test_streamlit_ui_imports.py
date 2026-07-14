from __future__ import annotations

import importlib
import sys
import types


class _Decorator:
    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda function: function


def _install_streamlit_stub(monkeypatch) -> None:
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.cache_data = _Decorator()
    fake_streamlit.fragment = _Decorator()
    fake_streamlit.session_state = {}
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)


def test_ui_modules_import_with_optional_streamlit_stub(monkeypatch):
    """Keep module wiring testable even when the optional UI is not installed."""

    _install_streamlit_stub(monkeypatch)
    modules = (
        "apps.streamlit.onboarding",
        "apps.streamlit.ui_onboarding",
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


def test_local_config_fingerprint_changes_with_file(monkeypatch, tmp_path):
    _install_streamlit_stub(monkeypatch)
    sys.modules.pop("apps.streamlit.workspace", None)
    workspace = importlib.import_module("apps.streamlit.workspace")

    assert workspace._local_config_fingerprint(tmp_path) == "missing"
    local_dir = tmp_path / "configs" / "local"
    local_dir.mkdir(parents=True)
    local_path = local_dir / "local.yaml"
    local_path.write_text("trainer:\n  device: cpu\n", encoding="utf-8")
    first = workspace._local_config_fingerprint(tmp_path)
    local_path.write_text("trainer:\n  device: cuda\n", encoding="utf-8")
    second = workspace._local_config_fingerprint(tmp_path)
    assert first != second
