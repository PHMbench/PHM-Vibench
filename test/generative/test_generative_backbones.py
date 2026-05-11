from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest
import torch
from pydantic import ValidationError

from src.config_schema.models import ExperimentConfig


def _args(name: str, **overrides):
    values = {
        "type": "generative_model",
        "name": name,
        "in_channels": 2,
        "hidden_dim": 32,
        "condition_dim": 16,
        "num_fault_classes": 2,
        "num_domains": 2,
        "patch_size": 8,
        "num_layers": 1,
        "num_heads": 4,
        "use_true_mamba": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _condition(n: int = 3) -> dict[str, torch.Tensor]:
    return {
        "fault_label": torch.tensor([0, 1, 0], dtype=torch.long)[:n],
        "domain_id": torch.tensor([0, 1, 0], dtype=torch.long)[:n],
    }


@pytest.mark.parametrize(
    "module_name,args",
    [
        ("phm_unet1d", _args("phm_unet1d")),
        ("phm_dit1d", _args("phm_dit1d", condition_dim=32)),
        ("mamba1d_backbone", _args("mamba1d_backbone")),
    ],
)
def test_generative_backbone_forward_contract(module_name: str, args: SimpleNamespace) -> None:
    module = importlib.import_module(f"src.model_factory.generative_model.{module_name}")
    model = module.Model(args, metadata=None)
    x = torch.randn(3, 2, 31)
    t = torch.rand(3)

    out = model(x, t, _condition())

    assert out.shape == x.shape
    assert out.dtype == torch.float32
    assert getattr(model, "stateless", False) is True


def test_generative_schema_rejects_ambiguous_backbone_field() -> None:
    with pytest.raises(ValidationError, match="do not set model.backbone"):
        ExperimentConfig.model_validate(
            {
                "pipeline": "Pipeline_06_generative",
                "environment": {"project": "x", "output_dir": "results", "seed": 0, "iterations": 1},
                "data": {"data_dir": "data", "metadata_file": "metadata_dummy.csv"},
                "model": {
                    "type": "generative_model",
                    "name": "phm_unet1d",
                    "backbone": "B_04_Dlinear",
                },
                "task": {
                    "type": "generative",
                    "name": "conditional_flow_matching",
                    "generative": {"mode": "train"},
                },
                "trainer": {"name": "Default_trainer", "device": "cpu", "gpus": 0, "num_epochs": 1},
            }
        )


def test_optional_true_mamba_backend_is_guarded() -> None:
    module = importlib.import_module("src.model_factory.generative_model.mamba1d_backbone")
    try:
        import mamba_ssm  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="optional dependency mamba_ssm"):
            module.Model(_args("mamba1d_backbone", use_true_mamba=True), metadata=None)
