from __future__ import annotations

import importlib
from importlib.util import find_spec
import sys
import types

import pytest
import yaml


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


def _advanced_yaml(epochs: int) -> str:
    return yaml.safe_dump(
        {
            "environment": {"seed": 0, "iterations": 1, "output_dir": "results/demo"},
            "data": {"batch_size": 4, "num_workers": 0},
            "model": {"type": "Backbone", "name": "B_04_Dlinear"},
            "task": {"lr": 0.001},
            "trainer": {
                "name": "Default_trainer",
                "num_epochs": epochs,
                "device": "cpu",
                "devices": 1,
                "test_after_fit": True,
            },
        },
        sort_keys=False,
    )


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
    )
    for name in modules:
        sys.modules.pop(name, None)
        imported = importlib.import_module(name)
        assert imported is not None


def test_legacy_root_streamlit_launcher_is_removed() -> None:
    """The maintained UI has one import and deployment entrypoint."""

    assert find_spec("streamlit_app") is None


def test_validation_snapshot_uses_only_visible_ui_inputs(monkeypatch, tmp_path):
    """An unmentioned local YAML must not invalidate or alter UI validation."""

    _install_streamlit_stub(monkeypatch)
    sys.modules.pop("apps.streamlit.workspace", None)
    workspace = importlib.import_module("apps.streamlit.workspace")

    visible = workspace._validation_inputs(
        "Quick Start",
        "effective yaml",
        (("trainer.device", "cpu"),),
    )
    local_dir = tmp_path / "configs" / "local"
    local_dir.mkdir(parents=True)
    local_path = local_dir / "local.yaml"
    local_path.write_text("trainer:\n  device: cuda\n", encoding="utf-8")
    after_hidden_file = workspace._validation_inputs(
        "Quick Start",
        "effective yaml",
        (("trainer.device", "cpu"),),
    )
    changed_visible_input = workspace._validation_inputs(
        "Quick Start",
        "effective yaml",
        (("trainer.device", "cuda"),),
    )

    assert visible == after_hidden_file
    assert visible != changed_visible_input


def test_validation_snapshot_preserves_types_and_copies_nested_values(monkeypatch):
    _install_streamlit_stub(monkeypatch)
    sys.modules.pop("apps.streamlit.workspace", None)
    workspace = importlib.import_module("apps.streamlit.workspace")
    capture = workspace._validation_inputs
    assert capture("Advanced", "yaml", (("value", True),)) != capture("Advanced", "yaml", (("value", 1),))
    assert capture("Advanced", "yaml", (("value", 1),)) != capture("Advanced", "yaml", (("value", "1"),))
    values = [1, 2]
    checked = capture("Advanced", "yaml", (("values", values),))
    values.append(3)
    assert checked != capture("Advanced", "yaml", (("values", values),))
    assert checked == capture("Advanced", "yaml", (("values", [1, 2]),))


def test_advanced_common_fields_follow_the_current_yaml_draft(monkeypatch):
    _install_streamlit_stub(monkeypatch)
    sys.modules.pop("apps.streamlit.ui_theme", None)
    ui_theme = importlib.import_module("apps.streamlit.ui_theme")
    fallback = yaml.safe_load(_advanced_yaml(1))
    state = {"ui_mode": "Advanced", "advanced_yaml_text": _advanced_yaml(2)}

    resolved, first_scope = ui_theme._advanced_field_context(state, fallback, "demo")
    assert resolved["trainer"]["num_epochs"] == 2
    assert first_scope == "advanced-1"

    _, unchanged_scope = ui_theme._advanced_field_context(state, fallback, "demo")
    assert unchanged_scope == first_scope

    state["advanced_yaml_text"] = _advanced_yaml(3)
    resolved, changed_scope = ui_theme._advanced_field_context(state, fallback, "demo")
    assert resolved["trainer"]["num_epochs"] == 3
    assert changed_scope == "advanced-2"

    state["ui_mode"] = "Quick Start"
    resolved, quick_scope = ui_theme._advanced_field_context(state, fallback, "demo")
    assert resolved is fallback
    assert quick_scope == ""


def test_advanced_common_fields_reject_an_invalid_yaml_draft(monkeypatch):
    _install_streamlit_stub(monkeypatch)
    sys.modules.pop("apps.streamlit.ui_theme", None)
    ui_theme = importlib.import_module("apps.streamlit.ui_theme")
    state = {"ui_mode": "Advanced", "advanced_yaml_text": "trainer: ["}

    with pytest.raises(ui_theme.ConfigServiceError):
        ui_theme._advanced_field_context(state, {}, "demo")
