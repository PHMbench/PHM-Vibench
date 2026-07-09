from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class OTNFMPairing:
    target_indices: torch.Tensor
    cost_mean: torch.Tensor
    method: str


def _check_ncl(name: str, tensor: torch.Tensor) -> None:
    if tensor.ndim != 3:
        raise ValueError(f"{name} must be [N, C, L], got shape={tuple(tensor.shape)}")
    if tensor.shape[0] < 2:
        raise ValueError(f"{name} requires batch size >=2 for minibatch OT pairing")
    if not torch.isfinite(tensor).all():
        raise ValueError(f"{name} contains NaN/Inf; shape={tuple(tensor.shape)}")


def _linear_sum_assignment(cost: torch.Tensor) -> tuple[torch.Tensor, str]:
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise ImportError(
            "OT-NFM minibatch pairing requires scipy.optimize.linear_sum_assignment; "
            "install scipy to use OTNFMLoss."
        ) from exc

    rows, cols = linear_sum_assignment(cost.detach().cpu().numpy())
    order = torch.as_tensor(rows, device=cost.device, dtype=torch.long).argsort()
    cols_tensor = torch.as_tensor(cols, device=cost.device, dtype=torch.long)
    return cols_tensor[order], "scipy_linear_sum_assignment"


class OTNFMLoss(nn.Module):
    """Minibatch OT one-step map loss for exploratory OT-NFM PHM pilots."""

    def pair(self, z: torch.Tensor, x1: torch.Tensor) -> OTNFMPairing:
        _check_ncl("z", z)
        _check_ncl("x1", x1)
        if z.shape != x1.shape:
            raise ValueError(f"z and x1 shape mismatch: {tuple(z.shape)} vs {tuple(x1.shape)}")
        with torch.no_grad():
            z_flat = z.detach().flatten(1).float()
            x_flat = x1.detach().flatten(1).float()
            cost = torch.cdist(z_flat, x_flat, p=2)
            if not torch.isfinite(cost).all():
                raise ValueError("OT-NFM cost matrix contains NaN/Inf")
            target_indices, method = _linear_sum_assignment(cost)
            if target_indices.numel() != z.shape[0] or target_indices.unique().numel() != z.shape[0]:
                raise ValueError("OT-NFM pairing must be a full minibatch permutation")
            row_indices = torch.arange(z.shape[0], device=z.device)
            cost_mean = cost[row_indices, target_indices].mean()
        return OTNFMPairing(target_indices=target_indices, cost_mean=cost_mean, method=method)

    def forward(
        self,
        pred_map: torch.Tensor,
        x1: torch.Tensor,
        z: torch.Tensor,
    ) -> dict[str, torch.Tensor | str]:
        _check_ncl("pred_map", pred_map)
        pairing = self.pair(z, x1)
        if pred_map.shape != x1.shape:
            raise ValueError(
                f"pred_map and x1 shape mismatch: {tuple(pred_map.shape)} vs {tuple(x1.shape)}"
            )
        paired_target = x1[pairing.target_indices].detach()
        loss = F.mse_loss(pred_map, paired_target)
        if not torch.isfinite(loss):
            raise ValueError("OT-NFM loss contains NaN/Inf")
        return {
            "loss": loss,
            "mse_map": loss.detach(),
            "pairing_cost": pairing.cost_mean.detach(),
            "pairing_method": pairing.method,
            "pairing_indices": pairing.target_indices.detach(),
        }
