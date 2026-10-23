"""Reusable physical-prior GFS operations. The supplied encoder implements forward(x, p)."""
from __future__ import annotations
from typing import Sequence
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def unit(x: Tensor) -> Tensor:
    n = torch.linalg.vector_norm(x, dim=-1, keepdim=True)
    if torch.any(n <= 1e-10) or not torch.isfinite(x).all() or not torch.isfinite(n).all():
        raise ValueError('Cannot normalize zero or non-finite scientific representations.')
    return x / n


def validate_groups(y: Tensor, groups: Tensor, views: Tensor, classes: Sequence[int], shots: int) -> None:
    if set(y.tolist()) != set(classes):
        raise ValueError('Support must contain exactly the declared novel classes.')
    if set(views.tolist()) != {0, 1}:
        raise ValueError('Support requires two explicitly defined, nonempty views.')
    for g in groups.unique():
        ix = groups == g
        if y[ix].unique().numel() != 1 or set(views[ix].tolist()) != {0, 1}:
            raise ValueError('Each physical group needs one label and both support views.')
    for c in classes:
        if groups[y == c].unique().numel() != shots:
            raise ValueError(f'Class {c} does not contain {shots} independent support groups.')


def prototype(z: Tensor, y: Tensor, groups: Tensor, classes: Sequence[int]) -> Tensor:
    means = []
    for c in classes:
        gs = groups[y == c].unique()
        if gs.numel() == 0:
            raise ValueError(f'No support group for novel class {c}.')
        means.append(torch.stack([z[(groups == g) & (y == c)].mean(0) for g in gs]).mean(0))
    return unit(torch.stack(means))


def weights(y: Tensor, groups: Tensor) -> Tensor:
    """Each class has equal weight; within a class, each acquisition has equal weight."""
    out = torch.zeros(y.shape, dtype=torch.float32, device=y.device)
    classes = y.unique()
    for c in classes:
        gs = groups[y == c].unique()
        for g in gs:
            ix = (y == c) & (groups == g)
            out[ix] = 1.0 / (classes.numel() * gs.numel() * int(ix.sum()))
    return out


def scores(z: Tensor, base: Tensor, novel: Tensor, scale: float) -> Tensor:
    if scale <= 0:
        raise ValueError('The source-selected common logit scale must be positive.')
    return scale * z @ torch.cat((base, novel)).T


def crossfit(model: nn.Module, x: Tensor, p: Tensor, y: Tensor, groups: Tensor,
             views: Tensor, base: Tensor, base_classes: Sequence[int], novel_classes: Sequence[int], scale: float) -> Tensor:
    z = unit(model(x, p))
    order = list(base_classes) + list(novel_classes)
    target = torch.tensor([order.index(int(v)) for v in y], device=y.device)
    losses = []
    for v in (0, 1):
        ref, qry = views == v, views != v
        q = prototype(z[ref], y[ref], groups[ref], novel_classes)
        ce = F.cross_entropy(scores(z[qry], base, q, scale), target[qry], reduction='none')
        losses.append((weights(y[qry], groups[qry]) * ce).sum())
    return torch.stack(losses).mean()


def offset(model: nn.Module, x: Tensor, groups: Tensor, metadata: Tensor,
           prompt_dim: int, injection: Tensor, magnitude: float, wrong: bool = False) -> Tensor:
    """Use paired feature-metadata cross-moments, not a permutation-invariant metadata mean."""
    gs = groups.unique(sorted=True)
    with torch.no_grad():
        z = unit(model(x, x.new_zeros(prompt_dim)))
        h = torch.stack([z[groups == g].mean(0) for g in gs])
        m = []
        for g in gs:
            values = metadata[groups == g]
            if not torch.allclose(values, values[0].expand_as(values)):
                raise ValueError('The supplied metadata are not constant at the declared group granularity.')
            m.append(values[0])
        m = torch.stack(m)
        if wrong:
            m = torch.roll(m, shifts=1, dims=0)
        joint = torch.einsum('gd,gm->dm', h, m).reshape(-1) / len(gs)
        raw = injection @ joint
        return magnitude * unit(raw)


def assert_frozen(before: dict[str, Tensor], model: nn.Module) -> None:
    after = model.state_dict()
    if before.keys() != after.keys():
        raise AssertionError('The non-prompt source state changed its structure.')
    for k, v in before.items():
        if not torch.equal(v, after[k]):
            raise AssertionError(f'Non-prompt state changed: {k}')
