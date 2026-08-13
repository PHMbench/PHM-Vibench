from __future__ import annotations

from argparse import Namespace

import pytest

from src import Pipeline_04_Unified_Evaluation as pipeline04


def test_adapter_failure_is_terminal_and_cleanup_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleaned: list[bool] = []
    monkeypatch.setattr(
        pipeline04,
        "adapt_p04",
        lambda *_args: (_ for _ in ()).throw(ValueError("invalid unified config")),
    )
    monkeypatch.setattr(pipeline04, "close_lab", lambda: cleaned.append(True))

    with pytest.raises(ValueError, match="invalid unified config"):
        pipeline04.pipeline(Namespace(config_path="config.yaml", local_config=None))

    assert cleaned == [True]


def test_orchestrator_failure_is_terminal_and_cleanup_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleaned: list[bool] = []
    monkeypatch.setattr(pipeline04, "adapt_p04", lambda *_args: {"stages": []})

    class BrokenOrchestrator:
        def __init__(self, config: object) -> None:
            assert config == {"stages": []}

        def run_complete(self) -> None:
            raise RuntimeError("iteration failed")

    monkeypatch.setattr(pipeline04, "TwoStageOrchestrator", BrokenOrchestrator)
    monkeypatch.setattr(pipeline04, "close_lab", lambda: cleaned.append(True))

    with pytest.raises(RuntimeError, match="iteration failed"):
        pipeline04.pipeline(Namespace(config_path="config.yaml", local_config=None))

    assert cleaned == [True]


def test_success_returns_only_unified_result_and_cleanup_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cleaned: list[bool] = []
    monkeypatch.setattr(pipeline04, "adapt_p04", lambda *_args: {"stage": "canonical"})

    class SuccessfulOrchestrator:
        def __init__(self, config: object) -> None:
            assert config == {"stage": "canonical"}

        def run_complete(self) -> dict[str, bool]:
            return {"completed": True}

    monkeypatch.setattr(pipeline04, "TwoStageOrchestrator", SuccessfulOrchestrator)
    monkeypatch.setattr(pipeline04, "close_lab", lambda: cleaned.append(True))

    result = pipeline04.pipeline(
        Namespace(config_path="config.yaml", local_config="local.yaml")
    )

    assert result == {"results": {"completed": True}, "unified": True}
    assert cleaned == [True]
