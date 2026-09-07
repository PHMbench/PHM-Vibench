"""P01 losses. Sum path/class terms as in the core equations; average units."""
from __future__ import annotations
import torch
import torch.nn.functional as F


def unit_mean(values: torch.Tensor, unit_ids: torch.Tensor) -> torch.Tensor:
    """Equal weight for each experimental unit present in this batch.

    The runner samples units uniformly with the same windows-per-unit, making
    this an unbiased unit-level batch objective. Arbitrary window sampling would
    not be repaired by this reduction alone.
    """
    if values.ndim != 1 or unit_ids.ndim != 1 or len(values) != len(unit_ids):
        raise ValueError('values and unit_ids must be equally sized vectors')
    if len(values) == 0:
        raise ValueError('empty unit batch')
    return torch.stack([values[unit_ids == u].mean() for u in torch.unique(unit_ids)]).mean()


def path_margins(contributions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if contributions.ndim != 3 or labels.shape != (contributions.shape[0],):
        raise ValueError('expected contributions B,K+1,C and labels B')
    true = contributions.gather(-1, labels[:, None, None].expand(-1, contributions.shape[1], 1))
    return true - contributions  # the true-class entry is exactly zero


def covariance_loss(first, second, unit_ids, *, global_alignment=False):
    if first.shape != second.shape:
        raise ValueError('paired contribution shapes differ')
    a, b = first[:, 1:], second[:, 1:]  # bias is input independent
    if global_alignment:
        # Deliberate negative control: forces distinct operator paths to agree.
        target = torch.cat((a, b), dim=1).mean(dim=1, keepdim=True)
        per_sample = ((a-target).square().sum((1,2)) + (b-target).square().sum((1,2))) / 2
    else:
        per_sample = (a-b).square().sum((1,2))
    return unit_mean(per_sample, unit_ids)


def retention_loss(current, teacher, labels, unit_ids, mode='path_one_sided'):
    if current.shape != teacher.shape:
        raise ValueError('teacher path count/order must match the student')
    teacher = teacher.detach()
    if mode == 'path_symmetric':
        difference = path_margins(teacher, labels) - path_margins(current, labels)
    elif mode == 'total_one_sided':
        difference = path_margins(teacher.sum(1, keepdim=True), labels) - path_margins(current.sum(1, keepdim=True), labels)
    elif mode in {'path_one_sided', 'wrong_path'}:
        if mode == 'wrong_path':
            teacher = torch.cat((teacher[:, :1], teacher[:, 1:].roll(1, 1)), dim=1)
        difference = path_margins(teacher, labels) - path_margins(current, labels)
    else:
        raise ValueError(f'unknown retention mode: {mode}')
    if mode != 'path_symmetric':
        difference = difference.relu()
    return unit_mean(difference.square().sum((1,2)), unit_ids)


def replay_der_losses(logits, historical_logits, labels, unit_ids):
    """DER++ supervised replay and historical-logit MSE (targets do not refresh)."""
    ce = unit_mean(F.cross_entropy(logits, labels, reduction='none'), unit_ids)
    mse = unit_mean((logits-historical_logits.detach()).square().mean(-1), unit_ids)
    return ce, mse


def vrex_loss(per_sample_ce, domain_ids):
    risks = torch.stack([per_sample_ce[domain_ids == d].mean() for d in torch.unique(domain_ids)])
    if len(risks) < 2:
        raise ValueError('V-REx requires at least two source domains in every batch')
    return risks.mean(), risks.var(unbiased=False)
