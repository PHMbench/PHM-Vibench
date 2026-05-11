from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
import torch
from pydantic import ValidationError

from src.config_schema.models import ExperimentConfig
from src.task_factory.Components.generative.losses.score_sde import ScoreSDEResearchLoss
from src.task_factory.Components.generative.manifests.synthetic_data_manifest import (
    build_synthetic_data_manifest,
)
from src.task_factory.Components.generative.samplers.score_sde import (
    sample_score_sde_annealed_langevin,
)


REPO = Path(__file__).resolve().parents[2]


class _ZeroScore(torch.nn.Module):
    def forward(self, x: torch.Tensor, t: torch.Tensor, condition: dict[str, torch.Tensor]) -> torch.Tensor:
        assert t.min().item() > 0.0
        return torch.zeros_like(x)


def _condition(n: int = 2) -> dict[str, torch.Tensor]:
    return {
        "fault_label": torch.zeros(n, dtype=torch.long),
        "domain_id": torch.zeros(n, dtype=torch.long),
    }


def _base_config(generative: dict) -> dict:
    return {
        "pipeline": "Pipeline_06_generative",
        "environment": {"project": "x", "output_dir": "results", "seed": 0, "iterations": 1},
        "data": {"data_dir": "data", "metadata_file": "metadata_dummy.csv"},
        "model": {"type": "generative_model", "name": "phm_cfm_mlp1d"},
        "task": {
            "type": "generative",
            "name": "score_sde",
            "generative": generative,
        },
        "trainer": {"name": "Default_trainer", "device": "cpu", "gpus": 0, "num_epochs": 1},
    }


def test_score_sde_loss_and_sampler_are_finite() -> None:
    loss_fn = ScoreSDEResearchLoss()
    target = torch.randn(2, 2, 16)
    loss = loss_fn(target, target)
    sample = sample_score_sde_annealed_langevin(
        _ZeroScore(),
        torch.randn(2, 2, 16),
        _condition(),
        num_steps=4,
        sigma_min=0.01,
        sigma_max=1.0,
        step_size=0.001,
        seed=0,
    )

    assert loss["loss"].item() == 0.0
    assert sample.shape == (2, 2, 16)
    assert torch.isfinite(sample).all()


def test_score_sde_schema_rejects_incomplete_stochastic_settings() -> None:
    with pytest.raises(ValidationError, match="stochastic_sampler requires"):
        ExperimentConfig.model_validate(
            _base_config({"mode": "train", "stochastic_sampler": "annealed_langevin"})
        )


def test_score_sde_manifest_records_stochastic_sampler_metadata() -> None:
    manifest = build_synthetic_data_manifest(
        synthetic_dataset_id="score-sde-smoke",
        model_type="generative_model",
        model_name="phm_cfm_mlp1d",
        loss_id="score_sde_dsm",
        checkpoint_path="checkpoint.ckpt",
        generator_run_id="run-001",
        source_split="train",
        domain_map_path="configs/domain_maps/dummy_domain_map.csv",
        domain_map_hash="domain-hash",
        normalization={
            "method": "standardization",
            "scope": "per_channel",
            "params_artifact": "normalization_params.json",
            "params_hash": "normalization-hash",
        },
        sampler_id="score_sde_annealed_langevin",
        num_steps=8,
        seed=0,
        num_samples=2,
        shape=[2, 2, 128],
        sampler_metadata={
            "stochastic": True,
            "stochastic_sampler": "annealed_langevin",
            "sigma_min": 0.01,
            "sigma_max": 1.0,
            "stochastic_step_size": 0.001,
        },
    )

    assert manifest["sampling"]["sampler_metadata"]["stochastic"] is True
    assert manifest["sampling"]["seed"] == 0


def test_score_sde_demo_preflight_passes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--config",
            "configs/demo/10_generative/dummy_generative_score_sde.yaml",
            "--preflight-only",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Pipeline_06_generative" in result.stdout
