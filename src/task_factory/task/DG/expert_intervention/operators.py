"""Loss and paired fixed-route interventions for an already selected MoE."""
from __future__ import annotations

import numpy as np
import torch
from torch import nn

from src.model_factory.MoE.M_05_FixedRouteMoE import Model


def training_objective(logits: torch.Tensor, gates: torch.Tensor,
                       labels: torch.Tensor, balance: float, arm: str) -> torch.Tensor:
    """The prospective P04 CE + K ||mean(g)-1/K||² objective."""
    if not np.isfinite(balance) or balance < 0:
        raise ValueError('balance must be finite and nonnegative')
    k = gates.shape[1]
    penalty = k * (gates.mean(0) - 1 / k).square().sum()
    return nn.functional.cross_entropy(logits, labels) + (0 if arm == 'no_balance' else balance) * penalty


def fixed_route_deletions(model: Model, views: torch.Tensor, raw: torch.Tensor,
                          clean_gates: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Renormalized deletion z[-k]=(z-g[k]h[k])/(1-g[k]); no gate floor.

    This is a prospective sensitivity operator, distinct from historical M04
    deletion. A route of mass one makes deletion undefined and is rejected.
    """
    features = model.encode(views, raw)
    if clean_gates.shape != features.shape[:2]:
        raise ValueError('Clean gates must have [N,K] axes matching expert features')
    if (not torch.isfinite(clean_gates).all() or torch.any(clean_gates < 0)
            or not torch.allclose(clean_gates.sum(-1), torch.ones_like(clean_gates[:, 0]))):
        raise ValueError('Clean gates must be finite, nonnegative, and sum to one')
    if torch.any(clean_gates >= 1):
        raise ValueError('Renormalized deletion is undefined for a gate of mass one')
    mixture = (clean_gates[:, :, None] * features).sum(1)
    deleted = (mixture[:, None, :] - clean_gates[:, :, None] * features) / (1 - clean_gates[:, :, None])
    return model.head(mixture).softmax(-1), model.head(deleted).softmax(-1)


@torch.no_grad()
def paired_interventions(model: Model, data: dict[str, np.ndarray], ids: np.ndarray,
                         device: torch.device, *, intervention: str = 'replacement') -> dict[str, np.ndarray]:
    """Use the identical clean route on both spectral edits for every role.

    Role fitting receives only feature-change signatures. Neither labels nor
    replacement/deletion losses enter the role assignment.
    """
    if intervention not in {'replacement', 'deletion'}:
        raise ValueError('intervention must be replacement or deletion')

    def tensor(key: str) -> torch.Tensor:
        return torch.as_tensor(data[key][ids], device=device, dtype=torch.float32)

    logits, gates, reference = model(tensor('views'), tensor('raw'), tensor('compatibility'))
    bases, replacements, responses = [], [], []
    for role in range(model.k):
        base_pair, replacement_pair, response_pair = [], [], []
        for prefix in ('probe', 'control'):
            views, raw = tensor(f'{prefix}_views')[:, role], tensor(f'{prefix}_raw')[:, role]
            base, replacement, response = model.fixed_route_replacements(views, raw, gates, reference)
            if intervention == 'deletion':
                base, replacement = fixed_route_deletions(model, views, raw, gates)
            base_pair.append(base)
            replacement_pair.append(replacement)
            response_pair.append(response)
        bases.append(torch.stack(base_pair, dim=1))
        replacements.append(torch.stack(replacement_pair, dim=2))
        responses.append(response_pair[0] - response_pair[1])
    return dict(base=torch.stack(bases, dim=1).cpu().numpy(),
                replaced=torch.stack(replacements, dim=1).cpu().numpy(),
                response=torch.stack(responses, dim=1).cpu().numpy(),
                clean_probability=logits.softmax(-1).cpu().numpy())
