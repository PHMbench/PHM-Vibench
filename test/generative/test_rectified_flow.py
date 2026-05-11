from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import torch

from src.task_factory.Components.generative.losses.rectified_flow import RectifiedFlowLoss


REPO = Path(__file__).resolve().parents[2]


def test_rectified_flow_loss_matches_velocity_contract() -> None:
    loss_fn = RectifiedFlowLoss(eps=1e-3)
    x1 = torch.randn(3, 2, 16)
    z = torch.randn_like(x1)
    t = loss_fn.sample_t(x1.shape[0], x1.device)
    x_t = loss_fn.sample_xt(x1, z, t)
    pred_velocity = x1 - z

    loss = loss_fn(pred_velocity, x1, z, t)

    assert x_t.shape == x1.shape
    assert loss["loss"].item() == 0.0
    assert torch.isfinite(loss["mse_v"])


def test_rectified_flow_demo_preflight_passes() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "main.py",
            "--config",
            "configs/demo/10_generative/dummy_generative_rectified_flow.yaml",
            "--preflight-only",
        ],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert "Pipeline_06_generative" in result.stdout
