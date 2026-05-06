from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ScoreSDEResearchLoss(nn.Module):
    """Research-only Score SDE denoising score matching skeleton.

    Contract:
    - score prediction shape: `[N, C, L]`
    - denoising target shape: `[N, C, L]`
    - conditions remain `fault_label` and `domain_id`
    - no predictor-corrector sampler is provided in core V0
    - status is research-only until protocol, manifest, leakage, and metrics mature
    """

    research_status = "research-only"
    condition_keys = ("fault_label", "domain_id")

    def forward(
        self,
        pred_score: torch.Tensor,
        target_score: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        if pred_score.shape != target_score.shape:
            raise ValueError(
                f"pred_score and target_score shape mismatch: "
                f"{tuple(pred_score.shape)} vs {tuple(target_score.shape)}"
            )
        if not torch.isfinite(pred_score).all() or not torch.isfinite(target_score).all():
            raise ValueError("Score SDE skeleton received NaN/Inf tensors")
        loss = F.mse_loss(pred_score, target_score)
        return {"loss": loss, "mse_score": loss.detach(), "status": self.research_status}

