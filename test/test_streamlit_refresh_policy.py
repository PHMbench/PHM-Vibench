from __future__ import annotations

from contextlib import nullcontext
import importlib
import sys
import types


class _Decorator:
    def __call__(self, *args, **kwargs):
        if len(args) == 1 and callable(args[0]) and not kwargs:
            return args[0]
        return lambda function: function


def _load_runtime(monkeypatch):
    fake_streamlit = types.ModuleType("streamlit")
    fake_streamlit.fragment = _Decorator()
    fake_streamlit.session_state = types.SimpleNamespace(
        selected_run_id=None,
        active_run_id=None,
    )
    monkeypatch.setitem(sys.modules, "streamlit", fake_streamlit)
    sys.modules.pop("apps.streamlit.ui_runtime", None)
    return importlib.import_module("apps.streamlit.ui_runtime")


def _tabs(monkeypatch, runtime, count: int) -> None:
    monkeypatch.setattr(
        runtime.st,
        "tabs",
        lambda labels: tuple(nullcontext() for _ in range(count)),
        raising=False,
    )


def test_active_run_never_discovers_results(monkeypatch) -> None:
    runtime = _load_runtime(monkeypatch)
    record = types.SimpleNamespace(is_terminal=False)
    monkeypatch.setattr(runtime, "get_run", lambda repo_root, run_id: record)
    monkeypatch.setattr(runtime, "_render_status", lambda *args: None)
    monkeypatch.setattr(runtime, "_render_active_overview", lambda *args: None)
    monkeypatch.setattr(runtime, "_render_logs", lambda *args: None)
    monkeypatch.setattr(
        runtime,
        "discover_results",
        lambda *args: (_ for _ in ()).throw(AssertionError("active run scanned results")),
    )
    _tabs(monkeypatch, runtime, 2)

    runtime._render_active_run("/tmp/repo", "run-active")


def test_active_fragment_leaves_timer_when_run_becomes_terminal(monkeypatch) -> None:
    runtime = _load_runtime(monkeypatch)
    record = types.SimpleNamespace(is_terminal=True)
    reruns = []
    monkeypatch.setattr(runtime, "get_run", lambda repo_root, run_id: record)
    monkeypatch.setattr(runtime.st, "rerun", lambda: reruns.append(True), raising=False)
    monkeypatch.setattr(
        runtime,
        "discover_results",
        lambda *args: (_ for _ in ()).throw(AssertionError("transition scanned results")),
    )

    runtime._render_active_run("/tmp/repo", "run-finished")

    assert reruns == [True]


def test_outer_renderer_uses_timer_only_for_active_runs(monkeypatch) -> None:
    runtime = _load_runtime(monkeypatch)
    active = types.SimpleNamespace(is_active=True)
    terminal = types.SimpleNamespace(is_active=False)
    active_calls = []
    terminal_calls = []
    monkeypatch.setattr(runtime, "_render_active_run", lambda *args: active_calls.append(args))
    monkeypatch.setattr(runtime, "_render_terminal_run", lambda *args: terminal_calls.append(args))

    monkeypatch.setattr(runtime, "get_run", lambda repo_root, run_id: active)
    runtime._render_live_run("/tmp/repo", "active")
    assert len(active_calls) == 1
    assert terminal_calls == []

    monkeypatch.setattr(runtime, "get_run", lambda repo_root, run_id: terminal)
    runtime._render_live_run("/tmp/repo", "terminal")
    assert len(terminal_calls) == 1
    assert len(active_calls) == 1


def test_terminal_renderer_discovers_results_once(monkeypatch) -> None:
    runtime = _load_runtime(monkeypatch)
    record = types.SimpleNamespace(run_id="terminal")
    bundle = object()
    discovered = []
    monkeypatch.setattr(runtime, "_render_status", lambda *args: None)
    monkeypatch.setattr(
        runtime,
        "discover_results",
        lambda repo_root, selected: discovered.append((repo_root, selected)) or bundle,
    )
    monkeypatch.setattr(runtime, "_render_overview", lambda *args: None)
    monkeypatch.setattr(runtime, "_render_metrics", lambda *args: None)
    monkeypatch.setattr(runtime, "_render_artifacts", lambda *args: None)
    monkeypatch.setattr(runtime, "_render_logs", lambda *args: None)
    _tabs(monkeypatch, runtime, 4)

    runtime._render_terminal_run(types.SimpleNamespace(), record)

    assert len(discovered) == 1
    assert discovered[0][1] is record
