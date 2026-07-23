from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.Pipeline_06_Generative_Modeling as pipeline06


def _configs(mode: str, iterations: int = 2, **generative_values) -> SimpleNamespace:
    return SimpleNamespace(
        environment=SimpleNamespace(iterations=iterations),
        data=SimpleNamespace(),
        model=SimpleNamespace(),
        task=SimpleNamespace(
            generative=SimpleNamespace(mode=mode, **generative_values)
        ),
        trainer=SimpleNamespace(),
    )


@pytest.mark.parametrize(
    ("mode", "handler_name", "stage_values"),
    [
        ("train", "_run_train_stage", {}),
        (
            "sample",
            "_run_sample_stage",
            {
                "checkpoint_path": "checkpoint.ckpt",
                "normalization_path": "normalization_params.json",
                "normalization_sha256": "digest",
            },
        ),
        (
            "eval",
            "_run_eval_stage",
            {"generated_path": "samples.pt"},
        ),
    ],
)
def test_pipeline_dispatches_only_the_selected_stage(
    monkeypatch,
    mode: str,
    handler_name: str,
    stage_values: dict[str, str],
) -> None:
    configs = _configs(mode, iterations=2, **stage_values)
    calls: list[tuple[str, int]] = []

    monkeypatch.setattr(pipeline06, "_load_configs", lambda args: configs)

    for candidate in ["_run_train_stage", "_run_sample_stage", "_run_eval_stage"]:
        if candidate == handler_name:
            monkeypatch.setattr(
                pipeline06,
                candidate,
                lambda args, current_configs, iteration, selected=mode: (
                    calls.append((selected, iteration))
                    or {"mode": selected, "iteration": iteration}
                ),
            )
        else:
            monkeypatch.setattr(
                pipeline06,
                candidate,
                lambda *args, unexpected=candidate, **kwargs: pytest.fail(
                    f"unexpected handler called: {unexpected}"
                ),
            )

    results = pipeline06.pipeline(SimpleNamespace(config_path="unused.yaml"))

    assert calls == [(mode, 0), (mode, 1)]
    assert results == [
        {"mode": mode, "iteration": 0},
        {"mode": mode, "iteration": 1},
    ]


def test_stage_failure_is_recorded_and_reraised(monkeypatch) -> None:
    configs = _configs("train", iterations=1)
    records = []
    monkeypatch.setattr(pipeline06, "_load_configs", lambda args: configs)
    monkeypatch.setattr(
        pipeline06,
        "_run_train_stage",
        lambda *args: (_ for _ in ()).throw(RuntimeError("stage exploded")),
    )
    monkeypatch.setattr(
        pipeline06,
        "_record_stage",
        lambda configs, stage, **values: records.append((stage, values)),
    )

    with pytest.raises(RuntimeError, match="stage exploded"):
        pipeline06.pipeline(SimpleNamespace(config_path="unused.yaml"))

    assert records == [
        (
            "train",
            {
                "status": "failed",
                "iteration": 0,
                "error_type": "RuntimeError",
                "error": "stage exploded",
            },
        )
    ]


def test_ledger_failure_does_not_replace_stage_failure(monkeypatch) -> None:
    configs = _configs("train", iterations=1)
    monkeypatch.setattr(pipeline06, "_load_configs", lambda args: configs)
    monkeypatch.setattr(
        pipeline06,
        "_run_train_stage",
        lambda *args: (_ for _ in ()).throw(RuntimeError("stage exploded")),
    )
    monkeypatch.setattr(
        pipeline06,
        "_record_stage",
        lambda *args, **kwargs: (_ for _ in ()).throw(OSError("ledger unavailable")),
    )

    with pytest.raises(RuntimeError, match="stage exploded") as captured:
        pipeline06.pipeline(SimpleNamespace(config_path="unused.yaml"))

    assert isinstance(captured.value.__cause__, OSError)
    assert str(captured.value.__cause__) == "ledger unavailable"
