from __future__ import annotations

from argparse import Namespace

import pytest

from src.task_factory.task.pretrain import hse_contrastive
from src.utils.training.two_stage_orchestrator import _stage_seed


def test_contrastive_strategy_error_is_not_converted_to_zero_weight(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        hse_contrastive.Default_task,
        "__init__",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(hse_contrastive, "get_loss_fn", lambda _name: object())

    def fail_strategy(_config: dict[str, object]) -> None:
        raise ValueError("unsupported contrastive strategy")

    monkeypatch.setattr(
        hse_contrastive,
        "create_contrastive_strategy",
        fail_strategy,
    )

    with pytest.raises(ValueError, match="unsupported contrastive strategy"):
        hse_contrastive.task(
            network=object(),
            args_data=Namespace(),
            args_model=Namespace(),
            args_task=Namespace(contrast_weight=1.0),
            args_trainer=Namespace(),
            args_environment=Namespace(),
            metadata=None,
        )


def test_training_stage_requires_explicit_seed() -> None:
    with pytest.raises(ValueError, match="environment.seed is required"):
        _stage_seed(Namespace(), iteration=0)


@pytest.mark.parametrize("seed", (None, "42", 42.0, True))
def test_training_stage_rejects_non_integer_seed(seed: object) -> None:
    with pytest.raises(TypeError, match="environment.seed must be an integer"):
        _stage_seed(Namespace(seed=seed), iteration=0)


def test_training_stage_offsets_explicit_seed_by_iteration() -> None:
    assert _stage_seed(Namespace(seed=42), iteration=3) == 45
