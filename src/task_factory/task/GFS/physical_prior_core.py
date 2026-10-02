"""Canonical PHMFactory physical-prior GFS operations. The supplied encoder must implement forward(x, p)."""
from __future__ import annotations
import copy
import math
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


def adapt(model: nn.Module, x: Tensor, y: Tensor, groups: Tensor, views: Tensor,
          base: Tensor, base_classes: Sequence[int], novel_classes: Sequence[int],
          *, arm: str, prior: Tensor, scale: float, lr: float, steps: int,
          lam: float, radius: float, checkpoints: Sequence[int], subset: Sequence[str] = (),
          initial_prompt: Tensor | None = None) -> list[tuple[int, nn.Module, Tensor]]:
    """Adapt with a declared regularization center and a separate total-prompt start.

    ``prior`` centers the penalty; ``initial_prompt`` only chooses p at step zero.
    Omitting the latter preserves the original A6/A7/A8 initialization p=prior.
    No query argument is accepted. Checkpoints are fixed before this call.
    """
    if arm not in {'A2', 'A3', 'A4', 'A6', 'A7', 'A8'}:
        raise ValueError(f'Unsupported arm: {arm}')
    if not all(math.isfinite(v) for v in (radius, lam, lr)) or radius <= 0 or lam < 0 or lr <= 0 or type(steps) is not int or steps < 0:
        raise ValueError('Invalid source-selected adaptation settings.')
    if not torch.isfinite(prior).all():
        raise ValueError('The regularization center must be finite.')
    if float(prior.norm()) > radius + 1e-7:
        raise ValueError('The prior lies outside the common total-prompt ball.')
    if not checkpoints or min(checkpoints) < 0 or max(checkpoints) > steps:
        raise ValueError('Checkpoint schedule must be predeclared within the update budget.')
    net = copy.deepcopy(model).eval()
    original = {k: v.detach().clone() for k, v in net.state_dict().items()}
    for value in net.parameters():
        # ScriptModule deepcopy can yield non-leaf clones. Detach this private copy
        # before selecting trainable parameters; never detach the source module.
        value.detach_()
        value.requires_grad_(False)
    if initial_prompt is not None and arm not in {'A6', 'A7', 'A8'}:
        raise ValueError('An explicit prompt start is only valid for prompt-adaptation arms.')
    start = prior if initial_prompt is None else initial_prompt
    if start.shape != prior.shape or start.device != prior.device or start.dtype != prior.dtype:
        raise ValueError('Initial total prompt and center must have identical shape, device and dtype.')
    if not torch.isfinite(start).all() or float(start.norm()) > radius + 1e-7:
        raise ValueError('Initial total prompt must be finite and within the shared radius.')
    # phi is displacement from the regularization center, not from the optimizer start.
    prior = prior.detach().clone()
    phi = nn.Parameter(start.detach().clone() - prior)
    if arm == 'A4':
        return [(0, net, torch.zeros_like(prior))]
    if arm in {'A2', 'A3'}:
        prior = torch.zeros_like(prior)
        params = dict(net.named_parameters())
        names = list(params) if arm == 'A2' else list(subset)
        if not names or any(n not in params for n in names):
            raise ValueError('A2/A3 requires exact exported non-prompt parameter names.')
        selected = [params[n] for n in names]
        if arm == 'A3' and abs(sum(t.numel() for t in selected) - phi.numel()) > .05 * phi.numel():
            raise ValueError('A3 is not within five percent of the prompt parameter count.')
        for t in selected:
            t.requires_grad_(True)
        initial = [t.detach().clone() for t in selected]
    else:
        selected = [phi]
        initial = [torch.zeros_like(phi)]
    optimizer = torch.optim.SGD(selected, lr=lr)
    snapshots = []
    for step in range(steps + 1):
        p = prior + phi if arm in {'A6', 'A7', 'A8'} else prior
        if step in checkpoints:
            snapshots.append((step, copy.deepcopy(net).eval(), p.detach().clone()))
        if step == steps:
            break
        optimizer.zero_grad()
        loss = crossfit(net, x, p, y, groups, views, base, base_classes, novel_classes, scale)
        penalty = sum((t - t0).square().sum() for t, t0 in zip(selected, initial))
        loss = loss + .5 * lam * penalty
        if not torch.isfinite(loss):
            raise FloatingPointError('Non-finite support objective; no fallback is used.')
        loss.backward()
        optimizer.step()
        if arm in {'A6', 'A7', 'A8'}:
            with torch.no_grad():
                total = prior + phi
                total *= min(1.0, radius / max(float(total.norm()), 1e-12))
                phi.copy_(total - prior)
    if arm in {'A6', 'A7', 'A8'}:
        assert_frozen(original, net)
    return snapshots
