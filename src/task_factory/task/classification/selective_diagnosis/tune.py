"""Bounded source-only search. No calibration/test prediction or scoring path."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from . import run
from .core import unit_weights
from .selective import SelectiveMLP


def search(args: argparse.Namespace, protocol: dict | None = None) -> dict:
    run.reserve_output(args.output)
    if protocol is None:
        protocol = json.loads(args.config.read_text(encoding='utf-8'))
    if protocol['selection_metric'] != 'unit_balanced_tune_nll':
        raise ValueError('unsupported model-selection rule')
    data = run.load_data(args.data, roles=('train', 'tune'))
    if data['kind'] != 'real' and not args.allow_synthetic:
        raise ValueError('synthetic tuning requires explicit --allow-synthetic')
    train = data['split'] == 'train'
    tune = data['split'] == 'tune'
    weights = unit_weights(data['unit'][train])
    tune_weights = unit_weights(data['unit'][tune])
    mu = np.average(data['x'][train], axis=0, weights=weights)
    sd = np.sqrt(np.average((data['x'][train] - mu) ** 2, axis=0, weights=weights))
    if np.any(sd <= 1e-12):
        raise ValueError('constant training feature; do not silently drop it')
    x, y = (data['x'] - mu) / sd, data['y']
    classes = int(y[train].max()) + 1
    if not set(protocol['candidates']) <= {'fuzzy', 'mlp', 'selective'}:
        raise ValueError('search candidates must name existing fuzzy, mlp or selective models')
    methods = args.methods or list(protocol['candidates'])
    if not set(methods) <= set(protocol['candidates']) or len(set(methods)) != len(methods):
        raise ValueError('unknown or duplicate tuning method')
    torch.set_num_threads(1)
    selected, trials = {}, []
    resolved = dict(data=str(args.data.resolve()), data_kind=data['kind'], methods=methods,
                    seed=args.seed, device=args.device, protocol=protocol,
                    diagnostic=args.diagnostic, epochs_override=args.epochs,
                    max_trials=args.max_trials, test_consumed=False, calibration_consumed=False)
    (args.output / 'resolved_config.json').write_text(json.dumps(resolved, indent=2), encoding='utf-8')
    for method in methods:
        candidates = protocol['candidates'][method]
        limit = len(candidates) if args.max_trials is None else args.max_trials
        if limit > len(candidates) or limit < 1:
            raise ValueError('max-trials must not exceed the frozen candidate list')
        for index, candidate in enumerate(candidates[:limit]):
            config = dict(protocol['training'], **candidate, device=args.device)
            if args.epochs is not None:
                config['epochs'] = args.epochs
            if config['epochs'] < 1 or config['batch_size'] < 1 or config['learning_rate'] <= 0:
                raise ValueError('epochs/batch size/learning rate must be positive')
            if not args.diagnostic and (args.epochs is not None or args.max_trials is not None):
                raise ValueError('budget overrides require --diagnostic; final search uses frozen config')
            torch.manual_seed(args.seed)
            if method == 'fuzzy':
                model = run.Fuzzy(x[train], y[train], config['rules'], args.seed)
            elif method == 'mlp':
                model = nn.Sequential(nn.Linear(x.shape[1], config['mlp_width']), nn.ReLU(),
                                      nn.Linear(config['mlp_width'], classes))
            else:
                config.update(target_coverage=protocol['target_coverage'],
                              coverage_penalty=protocol['selectivenet']['coverage_penalty'],
                              aux_weight=protocol['selectivenet']['aux_weight'])
                model = SelectiveMLP(x.shape[1], classes, config['selector_width'],
                    coverage=config['target_coverage'], coverage_penalty=config['coverage_penalty'],
                    aux_weight=config['aux_weight'])
            model, state, seconds = run.train_torch(
                model, x[train], y[train], weights, SimpleNamespace(**config),
                robust=method == 'fuzzy', validation=(x[tune], y[tune], tune_weights))
            result = dict(method=method, trial=index, config=config, seconds=seconds,
                          parameters=sum(p.numel() for p in model.parameters()), **model.fit_report)
            trials.append(result)
            torch.save(dict(state=state, mean=mu, std=sd, config=config),
                       args.output / f'{method}_trial_{index}.pt')
            (args.output / 'trials.json').write_text(json.dumps(trials, indent=2, allow_nan=False), encoding='utf-8')
        best = min((t for t in trials if t['method'] == method), key=lambda t: (t['tune_nll'], t['trial']))
        selected[method] = dict(best['config'], epochs=best['selected_epoch'])
        selected[method].pop('device')
    selection = dict(status='diagnostic' if args.diagnostic else 'completed',
                     selection_metric=protocol['selection_metric'], selected_configs=selected,
                     data=str(args.data.resolve()), data_kind=data['kind'], seed=args.seed,
                     feature_names=data['feature_names'].tolist(),
                     train_units=np.unique(data['unit'][train]).tolist(),
                     tune_units=np.unique(data['unit'][tune]).tolist(),
                     tie_rule='lower candidate index', test_consumed=False,
                     calibration_consumed=False, baseline_qualified=False)
    (args.output / 'selection.json').write_text(json.dumps(selection, indent=2), encoding='utf-8')
    run.write_state('completed', scientific_effect='not_evaluated', baseline_qualified=False)
    return selection


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', choices=['fuzzy', 'mlp', 'selective'])
    parser.add_argument('--seed', type=int, default=20261003)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--diagnostic', action='store_true')
    parser.add_argument('--allow-synthetic', action='store_true')
    parser.add_argument('--epochs', type=int)
    parser.add_argument('--max-trials', type=int)
    args = parser.parse_args(argv)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')
    try:
        search(args)
    except KeyboardInterrupt:
        run.write_state('cancelled', scientific_effect='not_evaluated')
        raise
    except Exception as exc:
        run.write_state('failed', error_type=type(exc).__name__, error=str(exc), scientific_effect='not_evaluated')
        raise


if __name__ == '__main__':
    main()
