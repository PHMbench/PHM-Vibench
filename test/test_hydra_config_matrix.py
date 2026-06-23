from __future__ import annotations

from pathlib import Path

from src.config_schema import ExperimentConfig
from src.configs.config_utils import load_config


HYDRA_EXPERIMENTS = [
    "configs/hydra/experiments/00_smoke/dummy_dg.yaml",
    "configs/hydra/experiments/01_cross_domain/cwru_dg.yaml",
    "configs/hydra/experiments/02_cross_system/multi_system_cddg.yaml",
    "configs/hydra/experiments/03_fewshot/cwru_protonet.yaml",
    "configs/hydra/experiments/04_cross_system_fewshot/cross_system_tspn.yaml",
    "configs/hydra/experiments/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml",
    "configs/hydra/experiments/06_pretrain_cddg/pretrain_hse_cddg.yaml",
]


def _namespace_to_dict(value):
    if hasattr(value, "__dict__") and not isinstance(value, dict):
        return {k: _namespace_to_dict(v) for k, v in value.__dict__.items()}
    if isinstance(value, dict):
        return {k: _namespace_to_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_namespace_to_dict(v) for v in value]
    return value


def test_hydra_experiment_matrix_composes_to_runtime_shape() -> None:
    for config_path in HYDRA_EXPERIMENTS:
        assert Path(config_path).exists(), config_path
        cfg = load_config(config_path)
        resolved = _namespace_to_dict(cfg)
        ExperimentConfig.model_validate(resolved)

        assert resolved["pipeline"].startswith("Pipeline_")
        for section in ["environment", "data", "model", "task", "trainer"]:
            assert isinstance(resolved[section], dict), config_path


def test_hydra_p02_experiment_uses_explicit_single_mode() -> None:
    cfg = load_config("configs/hydra/experiments/05_pretrain_fewshot/pretrain_hse_then_fewshot.yaml")

    assert cfg.pipeline == "Pipeline_02_pretrain_fewshot"
    assert cfg.pipeline_mode == "single"
    assert cfg.task.name == "hse_contrastive"
    assert cfg.task.contrastive_pairing == "simclr_2view"
