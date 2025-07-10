from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PretrainLossCfg:
    """Configuration for :class:`PretrainHierarchicalLoss`."""

    target_radius: float = 1.0
    margin: float = 0.1
    lambda_recon: float = 1.0
    lambda_flow: float = 1.0
    lambda_h_struct: float = 1.0
    lambda_reg: float = 1e-2
    lambda_class: float = 1.0
    lambda_cohesion: float = 1.0
    lambda_separation: float = 1.0
    lambda_hier_margin: float = 1.0


class PretrainHierarchicalLoss(nn.Module):
    """Loss for hierarchical pretraining with domain and system labels.

    Parameters
    ----------
    cfg : PretrainLossCfg
        Hyper-parameters controlling each loss term.

    Examples
    --------
    >>> cfg = PretrainLossCfg()
    >>> crit = PretrainHierarchicalLoss(cfg)
    >>> loss, stats = crit(model, batch)
    """

    def __init__(self, cfg: PretrainLossCfg) -> None:
        super().__init__()
        self.cfg = cfg

    # --------------------------------------------------------------
    def forward(self, model: nn.Module, batch: Dict[str, Any]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Compute the total pretraining loss.

        Parameters
        ----------
        model : nn.Module
            Model expected to return ``(x_recon, h, v_pred, y_pred)`` when called
            with ``(x, domain, dataset)``.
        batch : dict
            A collated batch from :class:`ID_dataset`::

                {
                    'data': List[np.ndarray],
                    'metadata': List[dict],
                    'id': List[str]
                }

            ``metadata`` entries must contain ``'domain'``, ``'dataset'`` and
            ``'label'`` fields.

        Returns
        -------
        Tuple[torch.Tensor, Dict[str, float]]
            Total loss tensor and a dictionary of individual loss components.
        """
        device = next(model.parameters()).device

        x_batch = torch.stack([torch.as_tensor(d, dtype=torch.float32) for d in batch['data']]).to(device)
        meta: List[dict] = batch['metadata']
        d_batch = torch.tensor([m['domain'] for m in meta], dtype=torch.long, device=device)
        s_batch = torch.tensor([m['dataset'] for m in meta], dtype=torch.long, device=device)
        y_batch = torch.tensor([m.get('label', -1) for m in meta], dtype=torch.long, device=device)

        x_recon, h, v_pred, y_pred = model(x_batch, d_batch, s_batch)

        cfg = self.cfg

        loss_r = F.mse_loss(x_recon, x_batch)
        loss_reg = ((torch.linalg.norm(h, dim=1) - cfg.target_radius) ** 2).mean() + h.mean(dim=0).pow(2).sum()

        target = h.detach() - torch.randn_like(h)
        negative_target = target[torch.randperm(target.size(0))]
        loss_f = 10 * F.mse_loss(v_pred, target) - 0.1 * F.mse_loss(v_pred, negative_target)

        combo_labels = d_batch * 2 + s_batch
        loss_cohesion = torch.tensor(0.0, device=device)
        unique_combo = combo_labels.unique()
        for g in unique_combo:
            mask = combo_labels == g
            if mask.sum() > 1:
                class_h = h[mask]
                loss_cohesion += class_h.var(dim=0).sum()
        loss_cohesion = loss_cohesion / len(unique_combo)

        # Domain separation
        dom_means = [h[d_batch == d].mean(dim=0) for d in d_batch.unique()]
        if len(dom_means) >= 2:
            dist_domain = sum(F.mse_loss(a, b) for i, a in enumerate(dom_means) for b in dom_means[i+1:]) / (len(dom_means) - 1)
        else:
            dist_domain = torch.tensor(0.0, device=device)
        loss_domain_sep = -dist_domain

        # System separation
        sys_means = [h[s_batch == s].mean(dim=0) for s in s_batch.unique()]
        if len(sys_means) >= 2:
            dist_system = sum(F.mse_loss(a, b) for i, a in enumerate(sys_means) for b in sys_means[i+1:]) / (len(sys_means) - 1)
        else:
            dist_system = torch.tensor(0.0, device=device)
        loss_sys_sep = -dist_system

        loss_hier = F.relu(dist_domain - dist_system + cfg.margin)

        loss_h_struct = (
            cfg.lambda_cohesion * loss_cohesion
            + cfg.lambda_separation * (loss_domain_sep + loss_sys_sep)
            + cfg.lambda_hier_margin * loss_hier
        )

        loss_class = F.cross_entropy(y_pred, y_batch) if y_pred is not None else torch.tensor(0.0, device=device)

        total_loss = (
            cfg.lambda_recon * loss_r
            + cfg.lambda_flow * loss_f
            + cfg.lambda_h_struct * loss_h_struct
            + cfg.lambda_reg * loss_reg
            + cfg.lambda_class * loss_class
        )

        stats = {
            'loss_recon': loss_r.item(),
            'loss_flow': loss_f.item(),
            'loss_h_struct': loss_h_struct.item(),
            'loss_reg': loss_reg.item(),
            'loss_class': loss_class.item(),
        }
        return total_loss, stats
