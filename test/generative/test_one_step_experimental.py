from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.config_schema.models import ExperimentConfig
from src.task_factory.Components.generative.manifests.synthetic_data_manifest import (
    build_synthetic_data_manifest,
)


REPO = Path(__file__).resolve().parents[2]
METHOD_CONFIGS = {
    "meanflow": "configs/demo/10_generative/dummy_generative_meanflow.yaml",
    "drifting_flow": "configs/demo/10_generative/dummy_generative_drifting_flow.yaml",
    "transition_flow_matching": "configs/demo/10_generative/dummy_generative_transition_flow_matching.yaml",
    "ot_nfm": "configs/demo/10_generative/dummy_generative_ot_nfm.yaml",
}


def _base_config(method: str, generative: dict) -> dict:
    return {
        "pipeline": "Pipeline_06_generative",
        "environment": {"project": "x", "output_dir": "results", "seed": 0, "iterations": 1},
        "data": {"data_dir": "data", "metadata_file": "metadata_dummy.csv"},
        "model": {"type": "generative_model", "name": "phm_cfm_mlp1d"},
        "task": {"type": "generative", "name": method, "generative": generative},
        "trainer": {"name": "Default_trainer", "device": "cpu", "gpus": 0, "num_epochs": 1},
    }


@pytest.mark.parametrize("config_path", METHOD_CONFIGS.values(), ids=METHOD_CONFIGS.keys())
def test_one_step_experimental_demo_preflight_passes(config_path: str) -> None:
    result = subprocess.run(
        [sys.executable, "main.py", "--config", config_path, "--preflight-only"],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Pipeline_06_generative" in result.stdout


@pytest.mark.parametrize("method", METHOD_CONFIGS.keys())
def test_one_step_experimental_blocks_benchmark_valid(method: str) -> None:
    with pytest.raises(ValidationError, match="cannot be benchmark-valid"):
        ExperimentConfig.model_validate(
            _base_config(
                method,
                {
                    "mode": "train",
                    "num_steps": 1,
                    "experimental": True,
                    "validity_status": "benchmark-valid",
                },
            )
        )


@pytest.mark.parametrize("method", METHOD_CONFIGS.keys())
def test_one_step_experimental_requires_one_step(method: str) -> None:
    with pytest.raises(ValidationError, match="num_steps=1"):
        ExperimentConfig.model_validate(
            _base_config(
                method,
                {
                    "mode": "train",
                    "num_steps": 2,
                    "experimental": True,
                    "validity_status": "exploratory",
                },
            )
        )


def test_one_step_manifest_metadata_remains_compatible() -> None:
    manifest = build_synthetic_data_manifest(
        synthetic_dataset_id="meanflow-smoke",
        model_type="generative_model",
        model_name="phm_cfm_mlp1d",
        loss_id="meanflow_imf_experimental",
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
        sampler_id="one_step_euler",
        num_steps=1,
        seed=0,
        num_samples=2,
        shape=[2, 2, 128],
        sampler_metadata={
            "experimental": True,
            "method_id": "meanflow_imf_experimental",
            "one_step": True,
        },
    )

    assert manifest["sampling"]["sampler_id"] == "one_step_euler"
    assert manifest["sampling"]["sampler_metadata"]["experimental"] is True
