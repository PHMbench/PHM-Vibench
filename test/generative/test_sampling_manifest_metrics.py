import json

import pytest
import torch

from src.configs.config_utils import load_config
from src.task_factory.Components.generative.manifests.synthetic_data_manifest import (
    build_synthetic_data_manifest, write_synthetic_data_manifest)
from src.task_factory.Components.generative.samplers.euler_ode import \
    sample_euler_ode
from src.task_factory.task.generative.generative_eval import \
    evaluate_generated_windows


class ZeroVelocityModel(torch.nn.Module):
    def forward(self, x_t, t, condition):
        return torch.zeros_like(x_t)


def test_euler_ode_sampler_preserves_noise_for_zero_velocity():
    noise = torch.randn(2, 2, 16)
    condition = {
        "fault_label": torch.tensor([0, 1]),
        "domain_id": torch.tensor([0, 1]),
    }

    out = sample_euler_ode(ZeroVelocityModel(), noise, condition, num_steps=4)

    assert torch.allclose(out, noise)


def test_synthetic_manifest_write_contract(tmp_path):
    manifest = build_synthetic_data_manifest(
        synthetic_dataset_id="test_synth",
        model_type="generative_model",
        model_name="phm_cfm_mlp1d",
        loss_id="conditional_flow_matching",
        checkpoint_path="checkpoint.ckpt",
        generator_run_id="run0",
        source_split="train",
        domain_map_path="configs/domain_maps/dummy_domain_map.csv",
        domain_map_hash="abc123",
        normalization={"method": "standardization", "scope": "window"},
        sampler_id="euler_ode",
        num_steps=8,
        seed=0,
        num_samples=2,
        shape=[2, 2, 128],
    )
    out_path = tmp_path / "synthetic_data_manifest.json"

    write_synthetic_data_manifest(out_path, manifest)
    loaded = json.loads(out_path.read_text(encoding="utf-8"))

    assert loaded["generator"]["loss_id"] == "conditional_flow_matching"
    assert loaded["validity"]["benchmark_valid"] is False
    assert loaded["validity"]["status"] == "exploratory"
    assert loaded["normalization"]["params_recorded"] is False


def test_manifest_downgrades_benchmark_valid_instead_of_accepting_missing_evidence():
    # Missing evidence must produce an explicit exploratory downgrade, not a
    # silently accepted benchmark-valid manifest.
    manifest = build_synthetic_data_manifest(
        synthetic_dataset_id="test_synth",
        model_type="generative_model",
        model_name="phm_cfm_mlp1d",
        loss_id="conditional_flow_matching",
        checkpoint_path="checkpoint.ckpt",
        generator_run_id="run0",
        source_split="train",
        domain_map_path="configs/domain_maps/dummy_domain_map.csv",
        domain_map_hash="abc123",
        normalization={"method": "standardization", "scope": "window"},
        sampler_id="euler_ode",
        num_steps=8,
        seed=0,
        num_samples=2,
        shape=[2, 2, 128],
        status="benchmark-valid",
    )

    assert manifest["validity"]["status"] == "exploratory"
    assert manifest["validity"]["benchmark_valid"] is False


def test_manifest_rejects_forbidden_source_split():
    with pytest.raises(ValueError, match="source_split"):
        build_synthetic_data_manifest(
            synthetic_dataset_id="test_synth",
            model_type="generative_model",
            model_name="phm_cfm_mlp1d",
            loss_id="conditional_flow_matching",
            checkpoint_path="checkpoint.ckpt",
            generator_run_id="run0",
            source_split="test",
            domain_map_path="configs/domain_maps/dummy_domain_map.csv",
            domain_map_hash="abc123",
            normalization={"method": "standardization", "scope": "window"},
            sampler_id="euler_ode",
            num_steps=8,
            seed=0,
            num_samples=2,
            shape=[2, 2, 128],
        )


def test_generative_metric_bundle_returns_v0_keys():
    real = torch.randn(2, 2, 16)
    fake = real + 0.1 * torch.randn_like(real)
    labels = torch.tensor([0, 1])
    domains = torch.tensor([0, 1])

    metrics = evaluate_generated_windows(
        real,
        fake,
        real_labels=labels,
        fake_labels=labels,
        real_domains=domains,
        fake_domains=domains,
    )

    assert "temporal_l1" in metrics
    assert "temporal_autocorr_rmse" in metrics
    assert "spectral_fft_l1" in metrics
    assert "spectral_psd_l2" in metrics
    assert "distribution_mean_distance" in metrics
    assert "distribution_mmd_rbf" in metrics
    assert "leakage_nearest_neighbor_l2" in metrics
    assert "diversity_prdc_precision" in metrics
    assert "tstr_accuracy" in metrics
    assert metrics["tstr_status_code"] == 1.0
    assert metrics["eval_has_fault_labels"] == 1.0


def test_generative_demo_config_composes():
    cfg = load_config("configs/demo/10_generative/dummy_generative_cfm.yaml")

    assert cfg.pipeline == "Pipeline_06_generative"
    assert cfg.task.type == "generative"
    assert cfg.task.generative.mode == "train"
    assert cfg.data.test_ratio == 0.0
    assert cfg.trainer.monitor == "val_loss"
