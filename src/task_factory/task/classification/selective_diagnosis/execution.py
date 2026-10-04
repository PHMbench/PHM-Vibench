"""Thin adapter for the resolved public configuration and existing operators."""
from __future__ import annotations

from argparse import Namespace
from collections.abc import Mapping
from pathlib import Path
import os
from typing import Any

PHASES = ('sanity', 'capacity', 'fit', 'overfit', 'tune', 'predict', 'summarize', 'plot', 'export')


def _validate_device(config: Mapping[str, Any], requested: str) -> None:
    """Enforce the approved single-device research budget before creating output."""
    trainer = config.get('trainer', {})
    devices = trainer.get('devices', 1)
    if (type(devices) is not int or devices != 1
            or trainer.get('strategy', 'auto') not in {'auto', None}
            or os.environ.get('WORLD_SIZE', '1') != '1'):
        raise ValueError('selective diagnosis requires one device; DDP is forbidden')
    if requested not in {'cpu', 'cuda'}:
        raise ValueError('selective diagnosis device must be cpu or cuda; no device fallback')
    if requested == 'cpu':
        return
    if os.environ.get('CONDA_DEFAULT_ENV') != 'LQ_signal':
        raise ValueError('CUDA selective diagnosis requires the LQ_signal environment')
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in {'0', '1'}:
        raise ValueError('set exactly one numeric CUDA_VISIBLE_DEVICES index, 0 or 1; physical GPU 2 is forbidden')
    import torch
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('requested CUDA requires exactly one available visible device; no CPU fallback')


def execute(*, config: Mapping[str, Any], phase: str, output: Path | str,
            data: Path | str | None = None, checkpoint: Path | str | None = None,
            selection: Path | str | None = None, device: str | None = None,
            seed: int | None = None, arms: list[str] | None = None) -> dict:
    """Execute explicit feature/archive inputs; no paper checkout is consulted.

    ``data`` is a feature NPZ for fitting/prediction, an existing run directory
    for summary/plot, or a .npy feature matrix for export. ``checkpoint`` is the
    exact joint-policy NPZ for prediction. Source-only search uses the frozen
    candidate mapping under ``task.selective_diagnosis.search``.
    """
    if phase not in PHASES:
        raise ValueError(f'unsupported selective-diagnosis phase: {phase!r}')
    options = dict(config['task']['selective_diagnosis'])
    known_options = {
        'suite', 'epochs', 'rules', 'radius', 'alpha', 'delta', 'seeds', 'device',
        'learning_rate', 'weight_decay', 'batch_size', 'mlp_width', 'selector_width',
        'target_coverage', 'coverage_penalty', 'aux_weight', 'selection', 'methods',
        'overfit_model', 'unseen_domain', 'calibration', 'diagnostic_tune',
        'calibration_units', 'tune_methods', 'tuning_seed', 'diagnostic',
        'allow_synthetic', 'tuning_epochs', 'max_trials', 'search', 'summary_methods',
        'bootstrap_draws', 'bootstrap_seed', 'allow_diagnostic', 'manifest',
        'feature_names', 'producer', 'kind',
    }
    unknown = set(options) - known_options
    if unknown:
        raise ValueError(f'unknown selective-diagnosis options: {sorted(unknown)}')
    if selection is not None:
        if phase != 'fit':
            raise ValueError('selection is consumed only by fit')
        options['selection'] = str(selection)
    if device is not None:
        if phase not in {'fit', 'overfit', 'tune'}:
            raise ValueError('device is consumed only by fit, overfit or tune')
        options['device'] = device
    if seed is not None:
        if phase == 'tune':
            options['tuning_seed'] = seed
        elif phase in {'fit', 'overfit'}:
            options['seeds'] = [seed]
        else:
            raise ValueError('seed is consumed only by fit, overfit or tune')
    if arms is not None:
        if phase == 'fit':
            options['methods'] = arms
        elif phase == 'tune':
            options['tune_methods'] = arms
        else:
            raise ValueError('arms is consumed only by fit or tune')
    _validate_device(config, options.get('device', 'cpu'))
    output = Path(output)
    data = Path(data) if data is not None else None
    if checkpoint is not None and phase != 'predict':
        raise ValueError('checkpoint is consumed only by predict; it must not be silently ignored')
    if data is None and phase != 'capacity':
        raise ValueError(f'explicit data path is required for {phase}')
    if phase in {'fit', 'sanity', 'capacity', 'overfit'}:
        from . import run
        suite = options.get('suite', 'all') if phase in {'fit', 'capacity'} else phase
        if suite == 'smoke':
            raise ValueError('the public adapter requires explicit fixture data; use fit with kind=synthetic')
        argv = [suite, '--output', str(output)]
        if data is not None:
            argv += ['--data', str(data)]
        keys = ('epochs', 'rules', 'radius', 'alpha', 'delta', 'seeds', 'device',
                'learning_rate', 'weight_decay', 'batch_size', 'mlp_width', 'selector_width',
                'target_coverage', 'coverage_penalty', 'aux_weight', 'selection', 'methods',
                'overfit_model', 'unseen_domain', 'calibration')
        for key in keys:
            if key in options and options[key] is not None:
                value = options[key]
                argv += ['--' + key.replace('_', '-')]
                argv += [str(v) for v in value] if isinstance(value, (list, tuple)) else [str(value)]
        if options.get('diagnostic_tune', False):
            argv.append('--diagnostic-tune')
        if phase == 'capacity':
            argv.append('--check-calibration')
            if data is None:
                argv += ['--calibration-units', str(options['calibration_units'])]
        try:
            run.main(argv)
        except SystemExit as exc:
            if phase != 'capacity' or exc.code != 0:
                raise
    elif phase == 'predict':
        from .predict import predict
        if checkpoint is None:
            raise ValueError('predict requires an explicit joint-policy checkpoint')
        predict(Path(checkpoint), data, output)
    elif phase == 'tune':
        from . import run, tune
        device = options.get('device', 'cpu')
        args = Namespace(data=data, output=output, device=device, config=None,
                         methods=options.get('tune_methods'), seed=options.get('tuning_seed', 20261003),
                         diagnostic=options.get('diagnostic', False),
                         allow_synthetic=options.get('allow_synthetic', False),
                         epochs=options.get('tuning_epochs'), max_trials=options.get('max_trials'))
        try:
            tune.search(args, protocol=options['search'])
        except KeyboardInterrupt:
            run.write_state('cancelled', scientific_effect='not_evaluated')
            raise
        except Exception as exc:
            run.write_state('failed', error_type=type(exc).__name__, error=str(exc), scientific_effect='not_evaluated')
            raise
    elif phase == 'summarize':
        from .summarize import DEFAULT_METHODS, summarize
        summarize(data, output, tuple(options.get('summary_methods', DEFAULT_METHODS)),
                  options.get('bootstrap_draws', 1000), options.get('bootstrap_seed', 20261003),
                  options.get('allow_diagnostic', False))
    elif phase == 'plot':
        from .plot import main
        argv = ['--run', str(data), '--output', str(output)]
        if options.get('allow_diagnostic', False):
            argv.append('--allow-diagnostic')
        main(argv)
    else:
        from .export import export_archive
        export_archive(data, Path(options['manifest']), Path(options['feature_names']),
                       Path(options['producer']), output, options.get('kind', 'real'))
    return {'result_dir': str(output.resolve() if output.is_dir() else output.parent.resolve()),
            'output': str(output.resolve()), 'phase': phase, 'scientific_conclusion': 'not_adjudicated'}
