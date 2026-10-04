"""Source-only fitting and explicit-checkpoint intervention evaluation."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
from torch import nn

from src.model_factory import build_model
from .audit import audit
from .operators import paired_interventions, training_objective

ARMS = ('aligned', 'generic', 'shuffled', 'learned_physics', 'uniform', 'no_balance')
NORMALIZATION_KEYS = ('raw_mean', 'raw_scale', 'views_mean', 'views_scale')


def make_model(*, dim: int, experts: int, classes: int, width: int, arm: str,
               device: torch.device) -> nn.Module:
    return build_model(SimpleNamespace(type='MoE', name='M_05_FixedRouteMoE', input_dim=dim,
        num_experts=experts, num_classes=classes, width=width, arm=arm)).to(device)


def prior_validation_ce(data: dict[str, np.ndarray]) -> float:
    """Training-class prior evaluated with the same source-group validation CE."""
    labels, split = data['labels'], data['split']
    counts = np.bincount(labels[split == 'train'], minlength=int(labels.max()) + 1)
    probabilities = counts / counts.sum()
    val = split == 'val'
    losses = -np.log(probabilities[labels[val]])
    groups = data['group'][val]
    return float(np.mean([losses[groups == group].mean() for group in np.unique(groups)]))


def fit(data: dict[str, np.ndarray], arm: str, seed: int, settings: dict[str, Any],
        output: Path, device: torch.device) -> dict[str, Any]:
    """Fit only train/val. Save exactly the group-CE-selected checkpoint."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    _, k, d = data['views'].shape
    classes = int(data['labels'][data['split'] == 'train'].max()) + 1
    model = make_model(dim=d, experts=k, classes=classes, width=settings['width'], arm=arm, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings['lr'], weight_decay=settings['weight_decay'])
    train = np.flatnonzero(data['split'] == 'train')
    val = np.flatnonzero(data['split'] == 'val')

    def tensor(key: str, ids: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(data[key][ids], device=device,
                               dtype=torch.long if key == 'labels' else torch.float32)

    def forward(ids: np.ndarray):
        return model(tensor('views', ids), tensor('raw', ids), tensor('compatibility', ids))

    best, best_state, best_epoch = float('inf'), None, None
    history = []
    for epoch in range(settings['epochs']):
        model.train()
        for ids in np.array_split(np.random.permutation(train), max(1, int(np.ceil(len(train)/settings['batch_size'])))):
            logits, gates, _ = forward(ids)
            loss = training_objective(logits, gates, tensor('labels', ids), settings['balance'], arm)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'{arm}/{seed}: nonfinite training objective')
            optimizer.zero_grad()
            loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError(f'{arm}/{seed}: nonfinite objective gradient')
            optimizer.step()
        model.eval()
        with torch.no_grad():
            losses, groups = [], []
            for ids in np.array_split(val, max(1, int(np.ceil(len(val)/settings['batch_size'])))):
                losses.extend(nn.functional.cross_entropy(forward(ids)[0], tensor('labels', ids), reduction='none').cpu().tolist())
                groups.extend(data['group'][ids].tolist())
            losses, groups = np.asarray(losses), np.asarray(groups)
            score = float(np.mean([losses[groups == group].mean() for group in np.unique(groups)]))
        if not np.isfinite(score):
            raise FloatingPointError('Nonfinite validation loss')
        history.append(dict(epoch=epoch, validation_ce=score))
        if score < best:
            best, best_epoch = score, epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    if best_state is None:
        raise ValueError('At least one training epoch is required')
    checkpoint = dict(state_dict=best_state, arm=arm, seed=seed, width=settings['width'],
                      classes=classes, input_dim=d, num_experts=k, best_epoch=best_epoch,
                      selected_validation_ce=best,
                      settings=settings,
                      normalization={key: torch.as_tensor(data[key]) for key in NORMALIZATION_KEYS})
    source = np.isin(data['split'], ('train', 'val'))
    checkpoint['source_groups'] = sorted(set(data['group'][source].tolist()))
    checkpoint['source_domains'] = sorted(set(data['domain'][data['split'] == 'train'].tolist()))
    checkpoint['source_specimens'] = sorted(set(data['specimen'][source].tolist())) if 'specimen' in data else None
    checkpoint['role_names'] = data['role_names'].tolist() if 'role_names' in data else None
    checkpoint['class_names'] = data['class_names'].tolist() if 'class_names' in data else None
    torch.save(checkpoint, output/'model.pt')
    report = dict(parameters=sum(p.numel() for p in model.parameters()), history=history,
                  best_epoch=best_epoch, selected_validation_ce=best,
                  evaluation_requested=False, settings=settings, arm=arm, seed=seed)
    (output/'training.json').write_text(json.dumps(report, indent=2, allow_nan=False)+'\n')
    return report


def load_checkpoint(path: Path, device: torch.device) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location='cpu', weights_only=True)
    model = make_model(dim=int(checkpoint['input_dim']), experts=int(checkpoint['num_experts']),
        classes=int(checkpoint['classes']), width=int(checkpoint['width']), arm=checkpoint['arm'], device=device)
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    model.eval()
    return model, checkpoint


def evaluate(model: nn.Module, checkpoint: dict[str, Any], data: dict[str, np.ndarray],
             output: Path, device: torch.device, *, batch_size: int, alpha: float,
             independent_groups: bool, intervention: str) -> dict[str, float]:
    if data['views'].shape[1:] != (checkpoint['num_experts'], checkpoint['input_dim']):
        raise ValueError('Checkpoint expert/feature axes differ from evaluation data')
    if int(data['labels'].max()) >= checkpoint['classes']:
        raise ValueError('Checkpoint class axis differs from evaluation data')
    ids_all = np.flatnonzero(np.isin(data['split'], ('match', 'test')))
    batches = [paired_interventions(model, data, ids, device, intervention=intervention)
               for ids in np.array_split(ids_all, max(1, int(np.ceil(len(ids_all)/batch_size))))]
    exported = {key: np.concatenate([batch[key] for batch in batches]) for key in batches[0]}
    metadata = {key: data[key][ids_all] for key in ('labels', 'group', 'split', 'domain')}
    if 'specimen' in data:
        metadata['specimen'] = data['specimen'][ids_all]
    np.savez_compressed(output/'audit_input.npz', **exported, **metadata,
                        model=checkpoint['arm'], seed=checkpoint['seed'], intervention=intervention)
    rows = [dict(group=data['group'][index], domain=data['domain'][index], split=data['split'][index],
                 label=int(data['labels'][index]), **{f'p{c}': float(p) for c,p in enumerate(probability)})
            for index, probability in zip(ids_all, exported['clean_probability'])]
    with (output/'clean_predictions.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    contrasts_path, _ = audit(output/'audit_input.npz', output/'audit', alpha, independent_groups)
    with contrasts_path.open() as stream:
        contrasts = list(csv.DictReader(stream))
    mean = float(np.mean([float(row['contrast']) for row in contrasts]))
    # Group weighting follows the source protocol, rather than counting all
    # windows as independently observed diagnoses.
    test = metadata['split'] == 'test'
    correct = exported['clean_probability'].argmax(-1) == metadata['labels']
    accuracy = float(np.mean([correct[test & (metadata['group'] == group)].mean()
                             for group in np.unique(metadata['group'][test])]))
    metrics = dict(signed_source_role_contrast=mean, group_averaged_test_accuracy=accuracy)
    (output/'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False)+'\n')
    return metrics
