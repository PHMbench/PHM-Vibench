from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.Pipeline_06_generative as pipeline06


def _configs(
    *,
    mode: str = "train",
    iterations: int = 1,
    **generative_values,
) -> SimpleNamespace:
    generative = SimpleNamespace(mode=mode, **generative_values)
    return SimpleNamespace(
        environment=SimpleNamespace(iterations=iterations),
        data=SimpleNamespace(),
        model=SimpleNamespace(),
        task=SimpleNamespace(generative=generative),
        trainer=SimpleNamespace(),
    )


def test_required_five_block_config_is_enforced() -> None:
    incomplete = SimpleNamespace(
        environment=SimpleNamespace(),
        data=SimpleNamespace(),
        model=SimpleNamespace(),
        task=SimpleNamespace(),
    )

    with pytest.raises(ValueError, match="trainer"):
        pipeline06._validate_required_sections(incomplete)


def test_unknown_mode_is_rejected_before_runtime_dispatch() -> None:
    with pytest.raises(ValueError, match="unsupported generative mode"):
        pipeline06._resolve_mode(_configs(mode="paperpack"))


def test_sample_requires_checkpoint_by_default() -> None:
    config = _configs(mode="sample")

    with pytest.raises(ValueError, match="checkpoint_path"):
        pipeline06._validate_stage_inputs(
            "sample",
            pipeline06._generative_cfg(config),
        )


def test_trained_sample_requires_explicit_normalization_evidence() -> None:
    config = _configs(mode="sample", checkpoint_path="model.ckpt")

    with pytest.raises(ValueError, match="normalization_path"):
        pipeline06._validate_stage_inputs(
            "sample",
            pipeline06._generative_cfg(config),
        )


def test_trained_sample_requires_expected_normalization_hash() -> None:
    config = _configs(
        mode="sample",
        checkpoint_path="model.ckpt",
        normalization_path="normalization_params.json",
    )

    with pytest.raises(ValueError, match="normalization_sha256"):
        pipeline06._validate_stage_inputs(
            "sample",
            pipeline06._generative_cfg(config),
        )


def test_explicit_untrained_sample_smoke_is_allowed() -> None:
    config = _configs(mode="sample", allow_untrained_smoke=True)

    pipeline06._validate_stage_inputs(
        "sample",
        pipeline06._generative_cfg(config),
    )


def test_eval_requires_generated_sample_path() -> None:
    config = _configs(mode="eval")

    with pytest.raises(ValueError, match="generated_path"):
        pipeline06._validate_stage_inputs(
            "eval",
            pipeline06._generative_cfg(config),
        )


def test_iterations_must_be_positive() -> None:
    with pytest.raises(ValueError, match="must be positive"):
        pipeline06._resolve_iterations(_configs(iterations=0))
