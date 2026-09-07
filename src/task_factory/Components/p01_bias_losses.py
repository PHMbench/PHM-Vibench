"""P01 losses. Average independent units, not windows.

Contribution order for the additive P01 model is:
``bias, condition context, signal path 1, ...``.
Same-path consistency and path-wise retention operate only on signal paths (and,
for retention, the classifier bias). The separate condition contribution is not
treated as an operator path and is not frozen by path retention.
"""
from __future__ import annotations
import torch
import torch.nn.functional as F


def unit_mean(values: torch.Tensor, unit_ids: torch.Tensor) -> torch.Tensor:
    """Equal weight for each experimental unit present in this batch.

    This is unit-equal only if the sampler itself selects units with the declared
    probability. Arbitrary window-level sampling is not repaired by this reducer.
    """
    if values.ndim != 1 or unit_ids.ndim != 1 or len(values) != len(unit_ids):
        raise ValueError("values and unit_ids must be equally sized vectors")
    if len(values) == 0:
        raise ValueError("empty unit batch")
    return torch.stack([values[unit_ids == u].mean() for u in torch.unique(unit_ids)]).mean()


def path_margins(contributions: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    if contributions.ndim != 3 or labels.shape != (contributions.shape[0],):
        raise ValueError("expected contributions B,K,C and labels B")
    true = contributions.gather(
        -1, labels[:, None, None].expand(-1, contributions.shape[1], 1)
    )
    return true - contributions


def _bias_and_signal_paths(contributions: torch.Tensor) -> torch.Tensor:
    if contributions.ndim != 3 or contributions.shape[1] < 3:
        raise ValueError(
            "expected contributions ordered as bias, condition, one or more signal paths"
        )
    return torch.cat((contributions[:, :1], contributions[:, 2:]), dim=1)


def covariance_loss(first, second, unit_ids, *, global_alignment=False):
    """Compare only signal-operator contributions across a declared pair."""
    if first.shape != second.shape:
        raise ValueError("paired contribution shapes differ")
    a, b = first[:, 2:], second[:, 2:]
    if global_alignment:
        target = torch.cat((a, b), dim=1).mean(dim=1, keepdim=True)
        per_sample = (
            (a - target).square().sum((1, 2))
            + (b - target).square().sum((1, 2))
        ) / 2
    else:
        per_sample = (a - b).square().sum((1, 2))
    return unit_mean(per_sample, unit_ids)


def retention_loss(current, teacher, labels, unit_ids, mode="path_one_sided"):
    if current.shape != teacher.shape:
        raise ValueError("teacher path count/order must match the student")
    teacher = teacher.detach()

    if mode == "total_one_sided":
        difference = (
            path_margins(teacher.sum(1, keepdim=True), labels)
            - path_margins(current.sum(1, keepdim=True), labels)
        ).relu()
        return unit_mean(difference.square().sum((1, 2)), unit_ids)

    current_paths = _bias_and_signal_paths(current)
    teacher_paths = _bias_and_signal_paths(teacher)
    if mode == "path_symmetric":
        difference = path_margins(teacher_paths, labels) - path_margins(current_paths, labels)
    elif mode in {"path_one_sided", "wrong_path"}:
        if mode == "wrong_path":
            teacher_paths = torch.cat(
                (teacher_paths[:, :1], teacher_paths[:, 1:].roll(1, 1)), dim=1
            )
        difference = (
            path_margins(teacher_paths, labels) - path_margins(current_paths, labels)
        ).relu()
    else:
        raise ValueError(f"unknown retention mode: {mode}")
    return unit_mean(difference.square().sum((1, 2)), unit_ids)


def replay_der_losses(logits, historical_logits, labels, unit_ids):
    """DER-style supervised replay plus a supplied frozen-logit target."""
    ce = unit_mean(F.cross_entropy(logits, labels, reduction="none"), unit_ids)
    mse = unit_mean((logits - historical_logits.detach()).square().mean(-1), unit_ids)
    return ce, mse


def vrex_loss(per_sample_ce, domain_ids):
    risks = torch.stack([
        per_sample_ce[domain_ids == d].mean() for d in torch.unique(domain_ids)
    ])
    if len(risks) < 2:
        raise ValueError("V-REx requires at least two source domains in every batch")
    return risks.mean(), risks.var(unbiased=False)
