"""Prospective expert-role interventions; no historical G050 execution path.

``execute`` consumes an already resolved public configuration. The CLI owns
configuration loading; this package owns the exact scientific computation.
"""
from __future__ import annotations

import itertools
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .audit import audit, bounded_radius, brier, fit_roles
from .data import load_pack, validate_dg
from .operators import fixed_route_deletions, paired_interventions, training_objective
from .preparation import components, paired_edits, prepare
from .training import ARMS, NORMALIZATION_KEYS, evaluate, fit, load_checkpoint, prior_validation_ce

PHASES = ('prepare', 'sanity', 'train', 'tune', 'evaluate')


def _get(value: Any, name: str, default: Any = None) -> Any:
    return value.get(name, default) if isinstance(value, Mapping) else getattr(value, name, default)


def _options(config: Any) -> dict[str, Any]:
    task = _get(config, 'task')
    options = _get(task, 'expert_intervention')
    if options is None:
        raise ValueError('task.expert_intervention must declare the scientific run settings')
    return dict(options) if isinstance(options, Mapping) else vars(options).copy()


def _training_settings(options: dict[str, Any]) -> dict[str, Any]:
    settings = {name: options[name] for name in ('width', 'epochs', 'batch_size', 'lr', 'weight_decay', 'balance')}
    for name in ('width', 'epochs', 'batch_size'):
        if isinstance(settings[name], bool) or int(settings[name]) != settings[name] or settings[name] < 1:
            raise ValueError(f'{name} must be a positive integer')
        settings[name] = int(settings[name])
    for name in ('lr', 'weight_decay', 'balance'):
        if not np.isfinite(settings[name]) or settings[name] < 0 or (name == 'lr' and settings[name] == 0):
            raise ValueError(f'{name} must be finite and in the declared nonnegative range (lr > 0)')
        settings[name] = float(settings[name])
    return settings


def _device(config: Any, requested: str | None) -> torch.device:
    trainer = _get(config, 'trainer')
    if _get(trainer, 'devices', 1) != 1 or _get(trainer, 'strategy', 'auto') not in ('auto', None):
        raise ValueError('Expert-intervention research requires one device and no distributed strategy')
    name = requested if requested is not None else _get(_get(config, 'trainer'), 'device')
    if not name:
        raise ValueError('An explicit device is required; no device fallback is permitted')
    device = torch.device(name)
    if device.type not in {'cpu', 'cuda'}:
        raise ValueError('Expert intervention supports explicitly selected cpu or cuda devices')
    if device.type == 'cuda':
        if device.index is None:
            raise ValueError('CUDA requires an explicit index, e.g. cuda:0')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is None:
            physical = device.index
        else:
            mapping = [entry.strip() for entry in visible.split(',')]
            if device.index >= len(mapping) or not all(entry.isdecimal() for entry in mapping):
                raise ValueError('Cannot establish the physical GPU from CUDA_VISIBLE_DEVICES; supply explicit numeric visibility')
            physical = int(mapping[device.index])
        if physical == 2:
            raise ValueError('Physical GPU 2 is excluded from this research protocol')
        if not torch.cuda.is_available() or device.index >= torch.cuda.device_count():
            raise RuntimeError(f'Requested CUDA device {name} is unavailable; no fallback')
    return device


def execute(config: Any, *, phase: str, output: str | Path,
            data: str | Path | None = None, checkpoint: str | Path | None = None,
            device: str | None = None) -> dict[str, Any]:
    """Run prepare, protocol sanity, source training/search, or checkpoint audit.

    ``train`` and ``tune`` never export matching/test model responses.
    ``evaluate`` uses the specified checkpoint's scaling and weights without
    refitting. Bounds require both an explicit independent-group assertion and
    specimen identities; neither is inferred from recording/window counts.
    """
    if phase not in PHASES:
        raise ValueError(f'Unknown expert-intervention phase {phase!r}; expected {PHASES}')
    options = _options(config)
    configured_model = _get(config, 'model')
    if configured_model is not None:
        if (_get(configured_model, 'type') != 'MoE'
                or _get(configured_model, 'name') != 'M_05_FixedRouteMoE'):
            raise ValueError('Expert interventions require the declared MoE.M_05_FixedRouteMoE model')
    destination = Path(output).expanduser().resolve()
    if destination.exists():
        raise FileExistsError(f'Output must be a new path: {destination}')
    if phase == 'prepare':
        if checkpoint is not None or data is not None:
            raise ValueError('prepare uses declared manifest/raw_root/roles, not a feature pack or checkpoint')
        manifest, raw_root, roles = (Path(options[key]).expanduser().resolve()
                                    for key in ('manifest', 'raw_root', 'roles'))
        for path in (manifest, raw_root, roles):
            if not path.exists():
                raise FileNotFoundError(path)
        prepare(manifest, raw_root, roles, destination/'feature_pack.npz',
                int(options['window']), float(options['strength']),
                partitions=tuple(options['partitions']), class_names=options['class_names'])
        result = dict(result_dir=str(destination), best_checkpoint=None, test_metrics={},
                      feature_pack=str(destination/'feature_pack.npz'), phase=phase)
    else:
        if data is None:
            raise ValueError('An explicit --data feature-pack path is required')
        if (phase == 'evaluate') != (checkpoint is not None):
            raise ValueError('An explicit checkpoint is required only for evaluate')
        pack_path = Path(data).expanduser().resolve()
        model = saved = None
        requested_device = _device(config, device)
        normalization = None
        if phase == 'evaluate':
            model, saved = load_checkpoint(Path(checkpoint).expanduser().resolve(), requested_device)
            normalization = {key: saved['normalization'][key].cpu().numpy() for key in NORMALIZATION_KEYS}
        partitions = (('train', 'val') if phase in {'train','tune'} else
                      ('match', 'test') if phase == 'evaluate' else ('train','val','match','test'))
        pack = load_pack(pack_path, normalization=normalization, partitions=partitions)
        if phase != 'sanity' and any(key not in pack for key in ('specimen', 'role_names', 'class_names')):
            raise ValueError('Public scientific execution requires specimen, role_names and class_names in the feature pack')
        if phase == 'evaluate':
            for name in ('role_names', 'class_names'):
                if saved.get(name) != pack[name].tolist():
                    raise ValueError(f'Checkpoint {name} differ from evaluation feature-pack semantics')
            if saved.get('source_specimens') is None:
                raise ValueError('Checkpoint must record the training/validation specimen identities')
            if set(saved['source_specimens']) & set(pack['specimen']):
                raise ValueError('Evaluation specimen overlaps the checkpoint training/validation population')
            if set(saved['source_groups']) & set(pack['group']):
                raise ValueError('Evaluation group overlaps the checkpoint training/validation population')
        if configured_model is not None:
            expected_axes = dict(input_dim=pack['raw'].shape[-1], num_experts=pack['views'].shape[1],
                                 num_classes=len(pack['class_names']) if 'class_names' in pack else int(pack['labels'].max()) + 1)
            for name, expected in expected_axes.items():
                declared = _get(configured_model, name)
                if declared is not None and declared != expected:
                    raise ValueError(f'model.{name}={declared} differs from feature-pack value {expected}')
        if options.get('dg', False):
            validate_dg(pack, source_domains=set(saved['source_domains']) if saved else None)
        partitions = {part: dict(windows=int((pack['split'] == part).sum()),
            aggregation_groups=len(np.unique(pack['group'][pack['split'] == part])),
            specimens=(len(np.unique(pack['specimen'][pack['split'] == part])) if 'specimen' in pack else None),
            domains=sorted(set(pack['domain'][pack['split'] == part])))
            for part in ('train', 'val', 'match', 'test')}
        settings = _training_settings(options)
        arm = options['arm']
        if arm not in ARMS:
            raise ValueError(f'Unknown expert control {arm!r}')
        if configured_model is not None:
            for name, expected in (('width', settings['width']), ('arm', arm)):
                declared = _get(configured_model, name)
                if declared is not None and declared != expected:
                    raise ValueError(f'model.{name} differs from task.expert_intervention.{name}; declare one consistent value')
        seed = options['seed']
        if isinstance(seed, bool) or int(seed) != seed or not 0 <= seed < 2**32:
            raise ValueError('seed must be an integer in [0,2**32)')
        alpha = float(options.get('alpha', .05))
        if not 0 < alpha < 1:
            raise ValueError('alpha must be in (0,1); divide by compared arms/seeds before a family audit')
        intervention = options.get('intervention', 'replacement')
        if intervention not in {'replacement', 'deletion'}:
            raise ValueError('intervention must be replacement or deletion')
        if phase == 'evaluate' and options.get('independent_groups', False) and 'specimen' not in pack:
            raise ValueError('Independent-group bounds require explicit specimen identities')
        if phase == 'tune':
            arms = options['search_arms']
            if list(arms) != ['generic', 'aligned', 'shuffled']:
                raise ValueError('The existing competitive search uses generic, aligned, shuffled in that order')
            rates, decays = options['learning_rates'], options['weight_decays']
            if not rates or not decays:
                raise ValueError('The declared candidate grid must be nonempty')
            for rate, decay in itertools.product(rates, decays):
                _training_settings(dict(settings, lr=rate, weight_decay=decay))
        destination.mkdir(parents=True, exist_ok=False)
        result = dict(result_dir=str(destination), best_checkpoint=None, test_metrics={},
                      phase=phase, data=str(pack_path), device=str(requested_device),
                      partitions=partitions, scientific_validation=False)
        if phase == 'train':
            report = fit(pack, arm, int(seed), settings, destination, requested_device)
            result.update(best_checkpoint=str(destination/'model.pt'),
                          selected_validation_ce=report['selected_validation_ce'])
        elif phase == 'evaluate':
            result.update(best_checkpoint=str(Path(checkpoint).expanduser().resolve()),
                test_metrics=evaluate(model, saved, pack, destination, requested_device,
                    batch_size=settings['batch_size'], alpha=alpha,
                    independent_groups=bool(options.get('independent_groups', False)), intervention=intervention),
                intervention=intervention, alpha=alpha)
            result['checkpoint_model'] = dict(arm=saved['arm'], width=saved['width'],
                seed=saved['seed'], input_dim=saved['input_dim'], num_experts=saved['num_experts'], classes=saved['classes'])
        elif phase == 'tune':
            prior = prior_validation_ce(pack)
            selected, searches = {}, {}
            for search_arm in arms:
                candidates = []
                for index, (rate, decay) in enumerate(itertools.product(rates, decays)):
                    candidate = _training_settings(dict(settings, lr=rate, weight_decay=decay))
                    run = destination/f'{search_arm}_candidate{index}'
                    run.mkdir()
                    report = fit(pack, search_arm, int(seed), candidate, run, requested_device)
                    candidates.append(dict(candidate=index, lr=float(rate), weight_decay=float(decay),
                        source_validation_ce=report['selected_validation_ce'], checkpoint=str(run/'model.pt')))
                best = min(candidates, key=lambda candidate: candidate['source_validation_ce'])
                searches[search_arm] = candidates
                selected[search_arm] = best
                (destination/f'{search_arm}_search.json').write_text(json.dumps(dict(candidates=candidates,
                    selected=best, prior_validation_ce=prior), indent=2, allow_nan=False)+'\n')
                if search_arm == 'generic' and best['source_validation_ce'] >= prior:
                    raise RuntimeError('Generic baseline did not beat the training-class-prior validation predictor')
            selection = dict(arms=selected, prior_validation_ce=prior,
                selection='minimum source-group-averaged validation CE; first candidate wins ties',
                data=str(pack_path), seed=int(seed), settings=settings,
                learning_rates=rates, weight_decays=decays, searches=searches)
            (destination/'selected_configs.json').write_text(json.dumps(selection, indent=2, allow_nan=False)+'\n')
            result['selected_configs'] = str(destination/'selected_configs.json')
    summary_path = destination/'run_summary.json'
    result['run_summary'] = str(summary_path)
    summary_path.write_text(json.dumps(result, indent=2, allow_nan=False)+'\n')
    return result


__all__ = ['PHASES', 'execute', 'audit', 'fit_roles', 'brier', 'bounded_radius', 'components',
           'paired_edits', 'prepare', 'load_pack', 'validate_dg', 'training_objective',
           'fixed_route_deletions', 'paired_interventions']
