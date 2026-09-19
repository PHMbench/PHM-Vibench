"""Verify the actual O/UO/RO/RC source comparison from saved source artifacts only.

No waveform or prediction model is opened. Initial candidate tensors, observed
sampler rows, fixed fitting controls and frozen-reference source vectors must
agree; a matching YAML alone is not evidence of matching execution.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import random
from typing import Mapping

import numpy as np
import pandas as pd
import torch
import yaml


ARMS = ('O', 'UO', 'RO', 'RC')
RECIPE = dict(epochs=20, steps_per_epoch=50, units_per_domain=2, lr=.001,
              pair_shift=16, selection_brier_weight=.25, selection_predictor='candidate')
LOSS_SWITCHES = {
    'O': ('mean_source', 'relative', 'correction', 0.),
    'UO': ('worst_source', 'absolute', 'candidate', .1),
    'RO': ('worst_source', 'relative', 'candidate', .1),
    'RC': ('worst_source', 'relative', 'correction', .1),
}
IDENTITIES = ('labels', 'group_ids', 'acquisition_ids', 'window_ids', 'domains',
              'raw_class_names', 'candidate_class_names')


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _load_run(path: Path, arm: str, seed: int, recipe: Mapping) -> dict:
    command = json.loads((path/'command.json').read_text())
    model = yaml.safe_load((path/'model_config.yaml').read_text())
    data = yaml.safe_load((path/'data_config.yaml').read_text())
    scope = json.loads((path/'result_scope.json').read_text())
    for key, value in recipe.items():
        _require(command[key] == value, f'{arm}/{seed}: fitting or selector setting {key} differs from the fixed recipe.')
    _require(command['seed'] == seed and command['pair_seed'] == seed+10000,
             f'{arm}/{seed}: fitting or independent pair seed changed.')
    _require(not command.get('evaluate_test', False) and not scope['permanent_test_predicted'],
             f'{arm}/{seed}: this source fit also predicted permanent test.')
    _require(scope['reference_state_unchanged'] and scope['deployment_alpha'] == 0,
             f'{arm}/{seed}: reference state or pre-assessment deployment changed.')
    loss = model['loss']
    observed_switches = (loss['reduction'], loss.get('risk_reference', 'relative'),
                        loss.get('consistency_target', 'correction'), loss['lambda_delta'])
    _require(observed_switches == LOSS_SWITCHES[arm], f'{arm}/{seed}: wrong loss intervention.')
    for key, value in dict(tau=1., brier_weight=.25, domain_temperature=.25).items():
        _require(loss[key] == value, f'{arm}/{seed}: fixed loss setting {key} changed.')
    candidate = model['model']
    _require(candidate.get('head_type', 'linear') == 'linear' and candidate.get('use_reference_features', True)
             and bool(candidate['branches']) and candidate.get('head_frobenius_cap', 5.) == 5.,
             f'{arm}/{seed}: loss comparison no longer uses the fixed reference-plus-operators linear candidate.')
    sampling = pd.read_csv(path/'sampling.csv', dtype=str, keep_default_na=False)
    batches = pd.read_csv(path/'training_batches.csv')
    epochs = pd.read_csv(path/'training.csv')
    expected_steps = {(epoch, step) for epoch in range(recipe['epochs'])
                      for step in range(recipe['steps_per_epoch'])}
    actual_steps = list(zip(batches['epoch'], batches['step']))
    _require(len(actual_steps) == len(expected_steps) and set(actual_steps) == expected_steps,
             f'{arm}/{seed}: actual optimization steps differ from the fixed budget.')
    _require(epochs['epoch'].tolist() == list(range(recipe['epochs'])),
             f'{arm}/{seed}: one source-selection pass per epoch was not completed.')
    scores=epochs['worst_validation_relative_score'].to_numpy()
    _require(np.isfinite(scores).all() and scope['selected_epoch'] == int(np.argmin(scores)),
             f'{arm}/{seed}: selected checkpoint epoch does not use the earliest minimum source score.')
    _require(scope['optimizer_steps'] == len(expected_steps) and scope['supervised_endpoints'] == 2,
             f'{arm}/{seed}: optimization budget or two-endpoint supervision changed.')
    dataset = next(d for d in data['datasets'] if d['name'] == command['dataset'])
    sources = list(map(str, dataset['source_domains']))
    sampling_steps = set(zip(sampling['epoch'].astype(int), sampling['step'].astype(int)))
    _require(sampling_steps == expected_steps, f'{arm}/{seed}: missing sampler steps.')
    pair_rng=random.Random(seed+10000)
    for (epoch, step), selected in sampling.groupby(['epoch', 'step'], sort=False):
        _require(set(selected['domain']) == set(sources) and
                 len(selected) == len(sources)*recipe['units_per_domain'],
                 f'{arm}/{seed}, {epoch}/{step}: unequal source/group sampling budget.')
        _require(selected['draw'].astype(int).tolist() == list(range(len(selected))),
                 f'{arm}/{seed}, {epoch}/{step}: sampler draw order changed.')
        shifts = selected['pair_shift'].astype(int)
        _require(shifts.nunique() == 1 and 1 <= shifts.iloc[0] <= recipe['pair_shift'],
                 f'{arm}/{seed}, {epoch}/{step}: invalid batch-level paired transformation.')
        _require(shifts.iloc[0] == pair_rng.randint(1, recipe['pair_shift']),
                 f'{arm}/{seed}, {epoch}/{step}: pair schedule does not match its independent frozen RNG.')
        for domain, part in selected.groupby('domain', sort=False):
            _require(part['group_id'].nunique() == recipe['units_per_domain'],
                     f'{arm}/{seed}, {epoch}/{step}, {domain}: repeated physical group within one source batch.')
        for row in selected.itertuples(index=False):
            _require(int(row.windows) == data['data']['windows_per_unit'] and
                     row.window_ids == ';'.join(map(str, range(int(row.windows)))),
                     f'{arm}/{seed}: sampled window identities disagree with declared deterministic windows.')
    initial = torch.load(path/'initial_candidate_state.pt', map_location='cpu', weights_only=True)
    _require(bool(initial) and all(isinstance(v, torch.Tensor) for v in initial.values()),
             f'{arm}/{seed}: initial candidate tensor state is missing.')
    with np.load(path/'selected_source_validation_windows.npz', allow_pickle=False) as archive:
        vectors = {key: archive[key] for key in (*IDENTITIES, 'raw_probs', 'raw_log_probs')}
    for key in ('raw_probs', 'raw_log_probs'):
        _require(np.isfinite(vectors[key]).all(), f'{arm}/{seed}: nonfinite frozen-reference source vectors.')
    n = len(vectors['labels'])
    _require(vectors['raw_probs'].shape == vectors['raw_log_probs'].shape == (n, len(vectors['raw_class_names'])),
             f'{arm}/{seed}: source vector/class shapes disagree.')
    return dict(path=path, command=command, model=model, data=data, scope=scope,
                sampling=sampling, initial=initial, vectors=vectors, steps=len(expected_steps))


def verify_runs(runs: Mapping[tuple[str, int], Path], *, recipe: Mapping = RECIPE) -> dict:
    """Return a concrete comparison result; raise on any missing or changed run."""
    seeds = sorted({seed for _, seed in runs})
    _require(bool(seeds) and set(runs) == {(arm, seed) for seed in seeds for arm in ARMS},
             'Supply exactly O, UO, RO and RC for every declared seed.')
    summaries = []
    for seed in seeds:
        loaded = {arm: _load_run(Path(runs[(arm, seed)]), arm, seed, recipe) for arm in ARMS}
        baseline = loaded['O']
        for arm in ARMS[1:]:
            current = loaded[arm]
            _require(current['model']['model'] == baseline['model']['model'] and current['data'] == baseline['data'],
                     f'{arm}/{seed}: candidate structure, reference binding or data configuration differs from O.')
            for key in ('dataset', 'device', 'benchmark_commit', 'pair_rng'):
                _require(current['command'][key] == baseline['command'][key],
                         f'{arm}/{seed}: execution setting {key} differs from O.')
            a, b = baseline['initial'], current['initial']
            _require(a.keys() == b.keys() and all(torch.equal(a[key], b[key]) for key in a),
                     f'{arm}/{seed}: actual initial candidate state differs from O.')
            _require(current['sampling'].equals(baseline['sampling']),
                     f'{arm}/{seed}: actual group/acquisition/window/pair schedule differs from O.')
            for key in IDENTITIES:
                _require(np.array_equal(current['vectors'][key], baseline['vectors'][key]),
                         f'{arm}/{seed}: source-vector identity or class order {key} differs from O.')
            for key in ('raw_probs', 'raw_log_probs'):
                _require(np.array_equal(current['vectors'][key], baseline['vectors'][key]),
                         f'{arm}/{seed}: frozen-reference {key} differs from O.')
            for key in ('fit_access', 'selection_access'):
                _require(current['scope'][key] == baseline['scope'][key],
                         f'{arm}/{seed}: source access count {key} differs from O.')
        summaries.append(dict(seed=seed, arms=list(ARMS), optimizer_steps_per_arm=baseline['steps'],
                              sampled_acquisitions_per_arm=len(baseline['sampling']),
                              source_validation_windows_per_arm=len(baseline['vectors']['labels']),
                              initial_candidate_state_equal=True, actual_sampler_and_pair_schedule_equal=True,
                              source_reference_vectors_exactly_equal=True, source_access_and_selector_equal=True))
    return dict(status='matched_source_comparison', seeds=summaries,
                scope='Saved source execution only; no H5, assessment/test predictions, checkpoint inference or scientific performance claim.')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    paths = parser.add_mutually_exclusive_group(required=True)
    paths.add_argument('--root', help='Study directory containing ARM/seed_SEED.')
    paths.add_argument('--run', action='append', metavar='ARM:SEED:PATH')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456])
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.root:
        runs = {(arm, seed): Path(args.root)/arm/f'seed_{seed}' for seed in args.seeds for arm in ARMS}
    else:
        runs = {}
        for item in args.run:
            arm, seed, path = item.split(':', 2)
            key = (arm, int(seed))
            _require(key not in runs, f'Duplicate run {key}.')
            runs[key] = Path(path)
        _require({seed for _, seed in runs} == set(args.seeds), 'Explicit runs must cover all --seeds.')
    result = verify_runs(runs)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
