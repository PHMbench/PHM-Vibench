"""Execute the existing symbolic protocol from an already resolved configuration."""
from __future__ import annotations

from collections.abc import Mapping
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import baselines
from .analysis import diagnose_band, paired_effects
from .operators import diagnostic_comparison, evaluate_symbols
from .output import run_output, write_json
from .panel import inspect_panel
from .population import generate, summarize, write_csv

PHASES = ('synthetic', 'tune', 'evaluate', 'panel')


def _baseline_archive(archive, options: dict, *, tune: bool,
                      fit_dir: Path | None) -> dict[str, np.ndarray]:
    """Check permissions and frozen source identities before materializing signals."""
    required = {'x', 'y', 'unit', 'split', 'speed_hz', 'fs'}
    if required - set(archive.files):
        raise ValueError(f'Missing fields: {sorted(required - set(archive.files))}')
    # NpzFile is lazy: inspecting files and these metadata arrays does not load x.
    split = archive['split']
    if split.ndim != 1 or split.dtype.kind not in 'US':
        raise ValueError('split must contain one explicit string per acquisition')
    if tune and set(split) != {'train', 'val'}:
        raise ValueError('Public tune requires a source-only train/val NPZ; test rows are forbidden')
    source = options['source']
    key = 'condition' if 'condition' in source else 'speed_hz'
    if key not in archive.files:
        raise ValueError(f'Missing source selection field: {key}')
    condition = archive[key]
    if condition.shape != split.shape:
        raise ValueError(f'{key} must match the split metadata population')
    selected = condition == source[key]
    if tune and not np.all(selected):
        raise ValueError('Public tune requires only the declared source condition before loading signals')
    metadata_keys = ('y', 'unit', 'speed_hz', 'fs', 'nominal_speed_hz', 'condition', 'acquisition')
    metadata = {'split': split}
    metadata.update({key: archive[key] for key in metadata_keys if key in archive.files})
    if any(value.shape != split.shape for key, value in metadata.items() if key != 'fs'):
        raise ValueError('One metadata value per acquisition is required')
    if tune:
        if metadata['y'].dtype.kind not in 'iu':
            raise ValueError('Source labels must be integers')
        if not np.isfinite(metadata['speed_hz']).all() or np.any(metadata['speed_hz'] <= 0):
            raise ValueError('Source speed_hz must be finite and positive')
        for key in ('unit', 'condition', 'acquisition'):
            if key in metadata and (metadata[key].dtype.kind not in 'US' or
                                    np.any(np.char.strip(metadata[key]) == '')):
                raise ValueError(f'{key} must contain nonempty source identifiers')
        for unit in np.unique(metadata['unit']):
            mask = metadata['unit'] == unit
            if len(set(split[mask])) != 1 or len(set(metadata['y'][mask])) != 1:
                raise ValueError(f'Identity leakage or contradictory label: {unit}')
        labels = set(range(len(options['orders']) + 1))
        if any(set(metadata['y'][split == part]) != labels for part in ('train', 'val')):
            raise ValueError('Source train and validation must contain every declared class')
        if 'condition' in source:
            if not {'nominal_speed_hz', 'acquisition'} <= set(metadata):
                raise ValueError('Source condition requires nominal_speed_hz and independent acquisition IDs')
            nominal = metadata['nominal_speed_hz']
            if not np.isfinite(nominal).all() or np.any(nominal <= 0) or len(np.unique(nominal)) != 1:
                raise ValueError('Source condition must declare one positive nominal speed')
        elif 'condition' in metadata:
            raise ValueError('Condition-bearing data require explicit source condition')
        if 'acquisition' in metadata and len(np.unique(metadata['acquisition'])) != len(split):
            raise ValueError('Duplicate acquisition identity')
    else:
        state = json.loads((fit_dir / 'fit_state.json').read_text())
        if state.get('input_population') != 'source_trainval_only':
            raise ValueError('Public evaluation requires a source-only tune checkpoint')
        if 'source_membership' not in state:
            raise ValueError('Checkpoint lacks the source membership required for separate-pack evaluation')
        membership = baselines.source_membership(metadata,
            selected & (split == 'train'), selected & (split == 'val'))
        if membership != state['source_membership']:
            raise ValueError('Source acquisition membership differs from the frozen tuning population')
    return {**metadata, 'x': archive['x']}


def execute(config: Mapping[str, Any], phase: str, output: str | Path, *,
            data: str | Path | None = None, checkpoint: str | Path | None = None) -> dict:
    """Run on CPU; data, checkpoint and output locations are always explicit.

    ``synthetic`` is a small software fixture, not real-data evidence. ``tune``
    rejects test/non-source rows before loading signal arrays. ``evaluate`` requires a completed tune output
    and performs zero fitting. ``panel`` requires independent train/val
    acquisitions and reports unpaired descriptive statistics only.
    """
    if phase not in PHASES:
        raise ValueError(f'Unknown symbolic-diagnosis phase {phase!r}; expected {PHASES}')
    trainer = config.get('trainer', {})
    if trainer.get('device', 'cpu') != 'cpu':
        raise ValueError('Symbolic prototype/SVM execution requires explicit CPU; no device fallback')
    if (type(trainer.get('devices', 1)) is not int or trainer.get('devices', 1) != 1 or
            trainer.get('strategy') not in (None, 'auto', 'single_device')):
        raise ValueError('Symbolic prototype/SVM execution requires one CPU device; no DDP')
    options = dict(config['task']['symbolic_diagnosis'])
    destination = Path(output).resolve()
    data_path = Path(data).resolve() if data is not None else None
    fit_dir = Path(checkpoint).resolve() if checkpoint is not None else None
    if phase == 'synthetic':
        if data_path is not None or fit_dir is not None:
            raise ValueError('synthetic creates the declared fixture; data/checkpoint are not accepted')
        if set(options) != {'counts', 'orders', 'seed'}:
            raise ValueError('synthetic requires exactly counts, orders and seed')
        counts = options.get('counts')
        if (not isinstance(counts, (list, tuple)) or len(counts) != 3 or
                any(isinstance(n, bool) or not isinstance(n, int) or n < 1 for n in counts)):
            raise ValueError('synthetic requires positive train/val/test counts per class')
        if isinstance(options.get('seed'), bool) or not isinstance(options.get('seed'), int):
            raise ValueError('synthetic requires an explicit integer seed')
        if options.get('orders') != [3.2, 4.8, 6.4]:
            raise ValueError('The existing synthetic generator has fixed orders [3.2, 4.8, 6.4]')
    else:
        if data_path is None or not data_path.is_file():
            raise FileNotFoundError(f'{phase} requires an explicit accessible data NPZ: {data_path}')
        if phase == 'evaluate' and fit_dir is None:
            raise ValueError('evaluate requires an explicit completed tune checkpoint directory')
        if phase != 'evaluate' and fit_dir is not None:
            raise ValueError(f'{phase} does not accept a checkpoint')
        if phase in ('tune', 'evaluate'):
            baselines.validate_config(options)
        elif set(options) != {'orders', 'source_condition', 'target_condition'}:
            raise ValueError('panel requires exactly orders, source_condition and target_condition')
    arguments = {'configuration': options, 'phase': phase,
                 'data': str(data_path) if data_path else None,
                 'fit_dir': str(fit_dir) if fit_dir else None}
    with run_output(destination, arguments) as out:
        if phase == 'synthetic':
            seed = options['seed']
            population = generate(tuple(options['counts']), seed)
            population['pairing_semantics'] = np.asarray('P4-lifted-state-synthetic-v1')
            population['generator_seed'] = np.asarray(seed)
            population['generator_profile'] = np.asarray('explicit_fixture_counts')
            np.savez_compressed(out / 'data.npz', **population)
            result = evaluate_symbols(population, np.asarray(options['orders']), paired_synthetic=True)
            for name in ('rows', 'utility', 'predictions'):
                write_csv(out / f'{name}.csv', result[name])
            write_json(out / 'fit_state.json', result['fit_state'])
            write_json(out / 'representation_comparison.json', diagnostic_comparison(result['rows']))
            write_json(out / 'paired_effects.json', paired_effects(result['rows'], seed + 2))
            write_json(out / 'activation.json', diagnose_band(population, result['rows']))
            summary = {'protocol': 'P4-v1-not-RUN-0002', 'phase': phase,
                       'evidence_status': 'synthetic software fixture; not scientific validation',
                       'seed': seed, 'counts_per_class': list(options['counts']),
                       'utility': result['utility'], 'summary': summarize(result['rows'], seed + 2)}
            write_json(out / 'summary.json', summary)
        elif phase == 'panel':
            with np.load(data_path, allow_pickle=False) as archive:
                summary, rows = inspect_panel(dict(archive), options['orders'],
                    options['source_condition'], options['target_condition'])
            write_csv(out / 'features.csv', rows)
            write_json(out / 'summary.json', summary)
        else:
            with np.load(data_path, allow_pickle=False) as archive:
                population = _baseline_archive(archive, options, tune=phase == 'tune', fit_dir=fit_dir)
            summary = baselines.evaluate(population, options, out,
                tune_only=phase == 'tune', fit_dir=fit_dir, source_only=phase == 'tune')
    return {'result_dir': str(destination),
            'best_checkpoint': str(destination) if phase == 'tune' else
                str(fit_dir) if fit_dir else
                str(destination / 'fit_state.json') if phase == 'synthetic' else None,
            'test_metrics': summary.get('metrics', summary.get('utility', [])),
            'run_summary': str(destination / 'summary.json')}
