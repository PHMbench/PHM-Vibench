from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.Pipeline_06_generative as pipeline06


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
            {"checkpoint_path": "checkpoint.ckpt"},
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


def test_g1_placeholder_fails_explicitly() -> None:
    with pytest.raises(RuntimeError, match="not integrated in the G1 shell"):
        pipeline06._run_train_stage(None, None, 0)
