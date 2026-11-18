from __future__ import annotations

from types import SimpleNamespace

import pytest

import src.Pipeline_06_Generative_Modeling as pipeline06


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


def test_population_bandwidths_are_read_from_task_config() -> None:
    args_task = SimpleNamespace(
        population_regularization=SimpleNamespace(
            rbf_bandwidths=[7, 1.5],
        )
    )

    assert pipeline06._population_rbf_bandwidths(args_task) == (7.0, 1.5)


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


def test_public_generative_config_accepts_explicit_factory_extension() -> None:
    from phmfactory.config import analyze_config

    analysis = analyze_config(
        "configs/demo/10_generative/dummy_generative_cfm.yaml",
        override_values=["task.factory_module=example_extension.factory"],
    )
    assert analysis.effective_config["task"]["factory_module"] == "example_extension.factory"
    assert analysis.pipeline == "Pipeline_06_Generative_Modeling"


def test_factory_extension_keeps_other_pipeline_boundaries() -> None:
    from phmfactory.config import analyze_config

    with pytest.raises(ValueError, match="task.factory_module requires ordinary"):
        analyze_config(
            "configs/demo/10_generative/dummy_generative_cfm.yaml",
            override_values=[
                "task.factory_module=example_extension.factory",
                "pipeline=Pipeline_02_Pretraining_Few_Shot",
            ],
        )


def test_lightning_schema_rejects_research_factory_extension() -> None:
    from phmfactory.config import analyze_config
    from src.config_schema import ExperimentConfig

    config = analyze_config(
        "configs/demo/10_generative/dummy_generative_cfm.yaml",
        override_values=["task.factory_module=example_extension.factory"],
    ).runtime_config()
    config["task"]["execution"] = "research"
    with pytest.raises(ValueError, match="task.factory_module requires ordinary"):
        ExperimentConfig.model_validate(config, strict=True)
