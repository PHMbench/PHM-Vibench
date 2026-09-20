"""G07: read-only validity gate and post-hoc source-fixed constant controls.

Only saved JSON/YAML/CSV/NPZ artifacts are read. No model, H5, checkpoint, or
training module is imported. Outputs must be outside the original run directory.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from contextlib import redirect_stderr, redirect_stdout
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Any

import numpy as np
import yaml

from experiments.p01.analyze_d1 import (
    Artifact, SEEDS, METRICS, ANALYSIS_SEED, BOOTSTRAPS, load_artifact,
    source_run_descriptors, acquisition_estimates, group_estimates,
    bootstrap_counts, condition_metrics, _write_csv,
)

SOURCES = ('0', '2', '3')
CLASSES = ['healthy', 'inner_race_or_IR_dominant_mixed', 'outer_race_or_OR_dominant_mixed']
OBSERVATION = dict(layout='LC', squeeze_axes=[2], channel_indices=[2],
                   window_size=8192, windows_per_unit=2)
FROZEN_COMMIT = '72874750be3e53cf8f78f27e1991ed4749c9a4f9'
IDENTITY = ('labels', 'domains', 'group_ids', 'acquisition_ids', 'window_ids')


def protect_inputs(root: Path) -> None:
    """Enforce this task's no-checkpoint/H5 and immutable-input boundaries."""
    def guard(event: str, args: tuple) -> None:
        if event != 'open' or not isinstance(args[0], (str, bytes)):
            return
        path = Path(os.fsdecode(args[0])).resolve()
        if path.suffix.lower() in ('.h5', '.hdf5', '.pt', '.pth', '.ckpt'):
            raise PermissionError(f'G07 forbids opening waveform/checkpoint: {path}')
        mode, flags = args[1:3]
        writing = (isinstance(mode, str) and any(c in mode for c in 'wax+')) or (
            isinstance(flags, int) and flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC))
        if writing and (path == root or path.is_relative_to(root)):
            raise PermissionError(f'G07 input directory is read-only: {path}')
    sys.addaudithook(guard)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding='utf-8'))


def read_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding='utf-8'))


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def csv_rows(path: Path) -> list[dict]:
    with path.open(newline='', encoding='utf-8') as handle:
        return list(csv.DictReader(handle))


def align(a: Artifact, b: Artifact, fields: tuple[str, ...] = IDENTITY) -> bool:
    return a.classes == b.classes and all(np.array_equal(a.arrays[k], b.arrays[k]) for k in fields)


def window_weights(arrays: dict, domains: tuple[str, ...]) -> np.ndarray:
    """Equal conditions/groups/acquisitions, then equal windows; no labels used."""
    weights = np.zeros(len(arrays['labels']), dtype=float)
    for domain in domains:
        dg = sorted(set(arrays['group_ids'][arrays['domains'] == domain]))
        if not dg:
            raise ValueError(f'No observations in declared condition {domain}')
        for group in dg:
            mask = (arrays['domains'] == domain) & (arrays['group_ids'] == group)
            acquisitions = sorted(set(arrays['acquisition_ids'][mask]))
            for acquisition in acquisitions:
                indices = np.flatnonzero(mask & (arrays['acquisition_ids'] == acquisition))
                weights[indices] = 1 / (len(domains) * len(dg) * len(acquisitions) * len(indices))
    np.testing.assert_allclose(weights.sum(), 1, atol=1e-14, rtol=0)
    return weights


def source_constants(artifacts: dict[str, Artifact]) -> dict[str, list[float]]:
    result = {}
    for name, artifact in artifacts.items():
        predictor = 'raw' if name == 'p0' else 'candidate'
        weights = window_weights(artifact.arrays, SOURCES)
        result[name] = (weights @ artifact.arrays[predictor + '_probs']).tolist()
    p0 = artifacts['p0']
    result['pi_S'] = (window_weights(p0.arrays, SOURCES) @ np.eye(3)[p0.arrays['labels']]).tolist()
    return result


def constant_arrays(arrays: dict, constant: list[float]) -> dict:
    vector = np.asarray(constant, dtype=float)
    if vector.shape != (3,) or np.any(vector < 0) or not np.isfinite(vector).all():
        raise ValueError('Invalid source constant.')
    np.testing.assert_allclose(vector.sum(), 1, atol=1e-6, rtol=0)
    # Zero mass remains exactly zero and yields -inf, never epsilon clipping.
    with np.errstate(divide='ignore'):
        lp = np.log(vector)
    result = {k: arrays[k] for k in IDENTITY}
    for predictor in ('raw', 'candidate', 'deployed'):
        result[predictor + '_probs'] = np.broadcast_to(vector, (len(arrays['labels']), 3))
        result[predictor + '_log_probs'] = np.broadcast_to(lp, (len(arrays['labels']), 3))
    return result


def scopes(split: str) -> dict[str, tuple[str, ...]]:
    domains = SOURCES if split == 'validation' else ('0', '1', '2', '3')
    result = {'condition:' + d: (d,) for d in domains}
    result['source_selection' if split == 'validation' else 'held_out_source'] = SOURCES
    if split == 'test':
        result['unseen'] = ('1',)
    return result


def identity_meta(name: str) -> dict:
    if name == 'p0':
        return dict(arm='p0', seed='fixed/reference')
    arm, seed = name.split('_seed_')
    return dict(arm=arm, seed=int(seed))


def summaries(artifact: Artifact, name: str) -> list[dict]:
    predictor = 'raw' if name == 'p0' else 'candidate'
    a = artifact.arrays
    acq = [row for row in artifact.acquisitions if row['predictor'] == predictor]
    groups = [row for row in artifact.groups if row['predictor'] == predictor]
    entropy = -np.sum(a[predictor + '_probs'] * a[predictor + '_log_probs'], axis=1)
    rows = []
    for scope, domains in scopes(artifact.spec['split']).items():
        weights = window_weights(a, domains)
        part = [r for r in acq if r['domain'] in domains]
        confusion = np.zeros((3, 3), dtype=int)
        for r in part:
            confusion[r['label'], r['prediction']] += 1
        estimates = []
        for domain in domains:
            cell = [r for r in groups if r['domain'] == domain]
            ids = [r['unit_id'] for r in cell]
            estimates.append(condition_metrics(cell, ids, np.ones((1, len(ids)), dtype=int)))
        rows.append(dict(row_type='summary', name=name, **identity_meta(name),
                         split=artifact.spec['split'], scope=scope, status='observed',
                         groups=len(set(r['unit_id'] for r in part)), acquisitions=len(part),
                         true_class_support=confusion.sum(1).tolist(),
                         predicted_class_histogram=confusion.sum(0).tolist(),
                         confusion_matrix=confusion.flatten().tolist(),
                         **{k: float(np.mean([e[k][0] for e in estimates])) for k in METRICS},
                         entropy=float(weights @ entropy),
                         mean_probability=(weights @ a[predictor + '_probs']).tolist(),
                         post_hoc=True))
    for row in groups:
        rows.append(dict(row_type='per_bearing', name=name, **identity_meta(name),
                         split=artifact.spec['split'], scope='condition:' + row['domain'],
                         bearing_id=row['unit_id'], status='observed', brier=row['brier'],
                         accuracy=float(np.trace(row['confusion_matrix'])),
                         acquisitions=row['acquisitions'], post_hoc=True))
    return rows


def audit(root: Path, recorded_root: Path) -> tuple[list[dict], dict, bool]:
    checks: list[dict] = []
    artifacts: dict[str, dict[str, Artifact]] = {'validation': {}, 'test': {}}

    def check(name: str, ok: bool, evidence: str, *, status: str = 'protocol_invalid',
              critical: bool = True) -> None:
        checks.append(dict(row_type='protocol_check', check=name,
                           status='pass' if ok else status, critical=critical,
                           evidence=evidence, post_hoc=True))

    def rebound(path: str) -> Path:
        return root / Path(path).relative_to(recorded_root)

    f1, f2 = read_json(root/'frozen/F1.json'), read_json(root/'frozen/F2.json')
    data = read_yaml(root/'configs/data.yaml')
    check('class_order_and_identity_label_map', data['model']['class_names'] == CLASSES
          and data['model']['num_classes'] == 3 and not data['datasets'][0].get('label_map')
          and f1['class_names'] == CLASSES, 'configs/data.yaml; F1; original labels 0/1/2 unchanged')
    check('observation_and_sample_rate', data['data'] == OBSERVATION
          and read_yaml(root/'frozen/data_config.yaml') == data
          and f1['sample_rate_hz'] == 64000 and f1['source_domains'] == list(SOURCES),
          'data configs: LC, squeeze axis2, vibration channel2, two start/end 8192 windows, 64000Hz')
    ref_config = read_yaml(root/'reference/model_config.yaml')['model']
    ref_core = {k: v for k, v in ref_config.items() if k not in ('type', 'name', 'device')}
    check('reference_preprocessing', ref_core['internal_instance_normalization'] is False
          and ref_core['in_channels'] == 1 and ref_core['in_dim'] == 8192,
          'reference model_config; input_binding.md: no resampling or added normalization')
    check('reference_training_temperature', read_json(root/'reference/command.json')['reference_temperature'] == 1,
          'reference/command.json; reference_temperature=1.0')
    check('frozen_export_revision', f1['code_commit'] == FROZEN_COMMIT,
          'F1 recorded implementation; state_dict guards and strict restore inspected at this exact revision')
    known = {s['name']: s for s in f1['predictors']}
    final = {s['name']: s for s in f2['predictors']}
    slots = [f'{arm}_seed_{seed}' for arm in ('MLP16', 'O') for seed in SEEDS]
    for split, directory in (('validation', 'source_validation'), ('test', 'test')):
        exports = {s['name']: s for s in read_json(root/f'frozen/{directory}/exports.json')}
        for name in ['raw_reference', *slots]:
            spec = dict(exports[name]); spec['path'] = str(rebound(spec['path']))
            if Path(spec['path']).suffix != '.npz':
                raise ValueError('Prediction input must be NPZ, never a checkpoint or H5.')
            loaded = load_artifact(spec)
            key = 'p0' if name == 'raw_reference' else name
            artifacts[split][key] = loaded
            check(f'{split}/{name}/class_and_recipe', loaded.classes == CLASSES
                  and json.loads(loaded.arrays['config_snapshot'].item()) == f1['plan']
                  and loaded.spec['code_commit'] == FROZEN_COMMIT,
                  f'frozen/{directory}/{name}.npz; unchanged F1 plan/class space')
            check(f'{split}/{name}/checkpoint_role', known[name] == final[name]
                  and loaded.spec['checkpoint'] == known[name]['checkpoint']
                  and known[name]['temperature'] == 1.0,
                  'F1/F2 descriptor and saved checkpoint role; no checkpoint opened')
        ref = artifacts[split]['p0']
        for name in slots:
            loaded = artifacts[split][name]
            check(f'{split}/{name}/raw_identity', align(ref, loaded)
                  and all(np.array_equal(ref.arrays[k], loaded.arrays[k])
                          for k in ('raw_probs', 'raw_log_probs')),
                  'Exact identity-aligned raw vectors across all MLP16/O seeds')
    source = artifacts['validation']
    original_p0 = load_artifact(source_run_descriptors([f'p0:42:{root}/reference'])[0])
    check('original_p0_to_frozen_source', align(original_p0, source['p0'])
          and np.array_equal(original_p0.arrays['raw_probs'], source['p0'].arrays['raw_probs'])
          and np.array_equal(original_p0.arrays['raw_log_probs'], source['p0'].arrays['raw_log_probs']),
          'reference/selected_source_validation_windows.npz vs frozen raw reference: exact arrays')
    for name in slots:
        arm, seed = name.split('_seed_'); directory = root/f'core/{arm}/seed_{seed}'
        training = load_artifact(source_run_descriptors([f'{arm}:{seed}:{directory}'])[0])
        frozen = source[name]
        check(f'{name}/selected_to_export', align(training, frozen)
              and all(np.array_equal(training.arrays[k], frozen.arrays[k]) for k in
                      ('raw_probs', 'raw_log_probs', 'candidate_probs', 'candidate_log_probs')),
              'Original selected source NPZ matches frozen source NPZ exactly')
        cfg = read_yaml(directory/'model_config.yaml')['model']
        history = read_json(directory/'result_scope.json')
        check(f'{name}/reference_and_observation_config', read_yaml(directory/'data_config.yaml') == data
              and cfg['reference_config'] == ref_core and cfg['reference_temperature'] == 1
              and cfg['checkpoint_kind'] == 'reference'
              and rebound(cfg['checkpoint_path']) == root/'reference/selected_candidate.pt',
              'Actual training config snapshot: common p0 path, architecture, temperature, observation')
        check(f'{name}/frozen_training_state', history['reference_state_unchanged'] is True,
              'result_scope.json; exact all-reference-state_dict comparison includes parameters and buffers')
    # Existing separate-process prediction artifacts, not a new restore/inference.
    verified = read_json(root/'frozen/source_verified.json')
    check('separate_process_restore_record', verified['separate_process'] is True
          and verified['predictors'] == [s['name'] for s in f2['predictors']],
          'source_verified.json; all declared predictors restored before original test release')
    for name in ['raw_reference', *slots]:
        original = source['p0' if name == 'raw_reference' else name]
        spec = dict(original.spec, path=str(root/f'frozen/restored_source_validation/{name}.npz'))
        restored = load_artifact(spec)
        check(f'{name}/saved_restore_vectors', align(original, restored)
              and all(np.array_equal(original.arrays[k], restored.arrays[k]) for k in
                      ('raw_probs', 'candidate_probs', 'deployed_probs', 'raw_log_probs',
                       'candidate_log_probs', 'deployed_log_probs')),
              'Existing separate-process source predictions: exact probability and stable-log vectors')
    for stage in ('prepare', 'assess', 'verify-source', 'test'):
        log = read_json(root/f'attempt_logs/frozen_{stage}.exit.json')
        check(f'{stage}/historical_state_guard_execution', log['exit_status'] == 0
              and log['benchmark_commit'] == FROZEN_COMMIT,
              'Existing exit log; frozen_predict checked eval and all state_dict tensors before/after; strict load')
    check('standalone_restore_parameter_tensor_comparison', False,
          'No standalone pre/post-restore parameter/buffer dump. Functional restore supported by strict '
          'state loading, exact saved source vectors, and executed full-state invariance guards; '
          'no claim of a newly performed tensor-level restore audit.', status='unresolved', critical=False)
    records = read_json(root/'frozen/records.json')
    record_map = {(r['domain'], r['acquisition_id']): r for r in records}
    owners: dict[str, str] = {}
    consistent = len(record_map) == len(records)
    for r in records:
        consistent &= owners.setdefault(r['unit_id'], r['split']) == r['split']
        consistent &= r['sample_rate_hz'] == 64000
    protocol = {r['Id']: r for r in csv_rows(root/'protocol.csv')}
    consistent &= all(protocol[r['id']]['unit_id'] == r['unit_id']
                      and protocol[r['id']]['split'] == r['split'] for r in records)
    check('global_split_and_protocol', bool(consistent), 'frozen/records.json vs original protocol.csv; no repartition')
    for split, bank in artifacts.items():
        a = bank['p0'].arrays
        expected = {(r['domain'], r['unit_id'], r['acquisition_id'], str(w), r['label'])
                    for r in records if r['split'] == split and (split == 'test' or r['domain'] in SOURCES)
                    for w in range(2)}
        actual = set(zip(a['domains'], a['group_ids'], a['acquisition_ids'], a['window_ids'], a['labels']))
        check(f'{split}/literal_observation_labels', expected == actual,
              'All saved windows/labels exactly match frozen records and unchanged two-window rule')
        count = 5 if split == 'validation' else 7
        by_domain = [set(a['group_ids'][a['domains'] == d]) for d in sorted(set(a['domains']))]
        check(f'{split}/physical_groups', len(set(a['group_ids'])) == count
              and all(g == by_domain[0] for g in by_domain),
              f'{count} shared physical bearings across all conditions in this partition')
    for directory in [root/'reference', *[root/f'core/{n.split("_seed_")[0]}/seed_{n.split("_seed_")[1]}' for n in slots]]:
        history = {r['group_id'] for r in csv_rows(directory/'development_groups.csv')}
        expected = {g for g, split in owners.items() if split in ('update', 'validation')}
        check(f'{directory.relative_to(root)}/development_history', history == expected,
              'Actual development groups equal fit+selection only; disjoint assessment/test')
    assessment = []
    for name in ('MLP16_seed_42', 'O_seed_42', 'RC_seed_42', 'temperature_seed42'):
        spec = dict(known[name], split='assessment', path=str(root/f'frozen/assessment/{name}.npz'))
        # Bundle is descriptor-only; load_artifact opens only spec[path] NPZ.
        a = load_artifact(spec); assessment.append(a)
        expected = {(r['domain'], r['unit_id'], r['acquisition_id'], str(w), r['label'])
                    for r in records if r['split'] == 'assessment' and r['domain'] in SOURCES for w in range(2)}
        check(f'{name}/assessment_semantics', a.classes == CLASSES
              and expected == set(zip(*(a.arrays[k] for k in ('domains','group_ids','acquisition_ids','window_ids','labels')))),
              'Existing assessment NPZ only: same class order and reserved observations')
    check('assessment_raw_identity', all(align(assessment[0], a)
          and np.array_equal(assessment[0].arrays['raw_probs'], a.arrays['raw_probs'])
          and np.array_equal(assessment[0].arrays['raw_log_probs'], a.arrays['raw_log_probs']) for a in assessment[1:]),
          'K4 raw reference exactly equal on independent assessment observations')
    summaries_out = [row for bank in artifacts.values() for name, a in bank.items() for row in summaries(a, name)]
    passed = all(row['status'] == 'pass' or not row['critical'] for row in checks)
    for row in summaries_out:
        row['protocol_verdict'] = 'pass_with_limitations' if passed else 'protocol_invalid'
    return checks + summaries_out, artifacts, passed


def estimates(arrays: dict, predictor: str, domains: tuple[str, ...], groups: list[str], counts: np.ndarray) -> dict:
    rows = group_estimates(acquisition_estimates(arrays), 3)
    by_domain = []
    for domain in domains:
        cell = [r for r in rows if r['domain'] == domain and r['predictor'] == predictor]
        # Avoid 0*inf contamination when a zero-mass constant incurs infinite CE.
        values = condition_metrics([{**r, 'ce': 0.0} for r in cell], groups, counts)
        weights = counts[:, [groups.index(r['unit_id']) for r in cell]]
        ce = np.asarray([r['ce'] for r in cell])
        with np.errstate(invalid='ignore'):
            terms = np.where(weights > 0, weights * ce, 0)
        values['ce'] = terms.sum(1) / weights.sum(1)
        by_domain.append(values)
    return {k: np.mean([d[k] for d in by_domain], axis=0) for k in METRICS}


def interval(values: np.ndarray) -> tuple[float, float]:
    if np.isposinf(values).all():
        return float('inf'), float('inf')
    # Empirical nearest-rank handles genuine infinities without interpolating inf-inf.
    if not np.isfinite(values).all():
        ordered = np.sort(values)
        return float(ordered[int(.025 * (len(ordered)-1))]), float(ordered[int(.975 * (len(ordered)-1))])
    return tuple(float(x) for x in np.quantile(values, [.025, .975]))


def controls(root: Path, output: Path, artifacts: dict) -> dict:
    constants = source_constants(artifacts['validation'])
    write_json(output/'source_constants.json', dict(post_hoc=True, constants=constants,
               class_order=CLASSES, source_groups=5, source_conditions=list(SOURCES),
               weights='equal conditions, bearings within condition, acquisitions within bearing, windows within acquisition',
               inputs={k: v.spec['path'] for k, v in artifacts['validation'].items()},
               constant_estimation='source-selection only; fixed in every test bootstrap',
               zero_components={k: [i for i, x in enumerate(v) if x == 0] for k, v in constants.items()}))
    metrics, contrasts, decomposition = [], [], []
    all_estimates = {}
    for split, bank in artifacts.items():
        reference = bank['p0'].arrays
        incidence = defaultdict(set)
        for g, d in zip(reference['group_ids'], reference['domains']):
            incidence[str(g)].add(str(d))
        groups, boot = bootstrap_counts(incidence)
        counts = np.vstack([np.ones(len(groups), dtype=int), boot])
        for scope, domains in scopes(split).items():
            scope_estimates = {}
            meta = dict(split=split, scope=scope, groups=len(groups), post_hoc=True,
                        interval_scope='post-hoc descriptive; fixed predictors and source constants; no source-estimation uncertainty',
                        source_in_sample=split == 'validation', bootstrap_repeats=BOOTSTRAPS, analysis_seed=ANALYSIS_SEED)
            for name, artifact in bank.items():
                q = estimates(artifact.arrays, 'raw' if name == 'p0' else 'candidate', domains, groups, counts)
                m = estimates(constant_arrays(artifact.arrays, constants[name]), 'candidate', domains, groups, counts)
                pi = estimates(constant_arrays(artifact.arrays, constants['pi_S']), 'candidate', domains, groups, counts)
                scope_estimates[name] = dict(q=q, m=m, pi=pi)
                for kind, values in (('q', q), ('m_q', m), ('pi_S', pi)):
                    for metric in METRICS:
                        lo, hi = interval(values[metric][1:])
                        metrics.append(dict(meta, name=name, **identity_meta(name), predictor=kind,
                                            metric=metric, value=float(values[metric][0]), lower=lo, upper=hi))
                for contrast, comparator in (('Delta_input', m), ('Delta_pi', pi)):
                    for metric in METRICS:
                        delta = q[metric] - comparator[metric]
                        lo, hi = interval(delta[1:])
                        contrasts.append(dict(meta, name=name, **identity_meta(name), contrast=contrast,
                                              metric=metric, value=float(delta[0]), lower=lo, upper=hi,
                                              seed_sd='', summary='fixed_reference' if name == 'p0' else 'individual_seed'))
            for arm in ('MLP16', 'O'):
                for kind in ('q', 'm', 'pi'):
                    for metric in METRICS:
                        values = np.stack([scope_estimates[f'{arm}_seed_{s}'][kind][metric] for s in SEEDS])
                        mean = values.mean(0); lo, hi = interval(mean[1:])
                        metrics.append(dict(meta, name=arm, arm=arm, seed='finite_mean_42_123_456',
                                            predictor={'q': 'q', 'm': 'm_q', 'pi': 'pi_S'}[kind], metric=metric,
                                            value=float(mean[0]), lower=lo, upper=hi,
                                            seed_sd=float(values[:, 0].std(ddof=1))))
                for contrast, comparator in (('Delta_input', 'm'), ('Delta_pi', 'pi')):
                    for metric in METRICS:
                        delta = np.stack([scope_estimates[f'{arm}_seed_{s}']['q'][metric]
                                          - scope_estimates[f'{arm}_seed_{s}'][comparator][metric] for s in SEEDS])
                        mean = delta.mean(0); lo, hi = interval(mean[1:])
                        contrasts.append(dict(meta, name=arm, arm=arm, seed='finite_mean_42_123_456',
                                              contrast=contrast, metric=metric, value=float(mean[0]),
                                              lower=lo, upper=hi, seed_sd=float(delta[:, 0].std(ddof=1)), summary='finite_seed_mean'))
            terms = []
            for seed in SEEDS:
                o, m = scope_estimates[f'O_seed_{seed}'], scope_estimates[f'MLP16_seed_{seed}']
                term = dict(Delta_pipe=o['q']['brier']-m['q']['brier'],
                            O_input=o['q']['brier']-o['m']['brier'],
                            MLP16_input=m['q']['brier']-m['m']['brier'],
                            mean_level=o['m']['brier']-m['m']['brier'])
                term['residual'] = term['Delta_pipe'] - (term['O_input']-term['MLP16_input']+term['mean_level'])
                np.testing.assert_allclose(term['residual'], 0, atol=1e-12, rtol=0)
                terms.append(term)
                for key, value in term.items():
                    lo, hi = interval(value[1:])
                    decomposition.append(dict(meta, seed=seed, term=key, value=float(value[0]), lower=lo, upper=hi, seed_sd=''))
            for key in terms[0]:
                values = np.stack([t[key] for t in terms]); mean = values.mean(0); lo, hi = interval(mean[1:])
                decomposition.append(dict(meta, seed='finite_mean_42_123_456', term=key,
                                          value=float(mean[0]), lower=lo, upper=hi, seed_sd=float(values[:, 0].std(ddof=1))))
            all_estimates[(split, scope)] = scope_estimates
    # Recover the original saved Delta_pipe, including its paired bootstrap intervals.
    for split, dirname in (('validation', 'analysis_source_frozen'), ('test', 'analysis_test')):
        for old in csv_rows(root/dirname/'paired_contrasts.csv'):
            if old['contrast'] != 'Delta_pipe' or old['metric'] != 'brier':
                continue
            scope = {'source': 'source_selection' if split == 'validation' else 'held_out_source'}.get(old['scope'], old['scope'])
            found = next(r for r in decomposition if r['split'] == split and r['scope'] == scope
                         and str(r['seed']) == old['seed'] and r['term'] == 'Delta_pipe')
            for key in ('value', 'lower', 'upper'):
                np.testing.assert_allclose(found[key], float(old[key]), atol=1e-12, rtol=0)
    for filename, rows in (('constant_controls_metrics.csv', metrics),
                           ('input_dependence_contrasts.csv', contrasts), ('pipeline_decomposition.csv', decomposition)):
        _write_csv(output/filename, rows)
    return dict(post_hoc=True, status='completed', protocol_verdict='pass_with_limitations',
                permanent_test_sealed=True, sealed_groups=7,
                restriction='No further model/loss/operator/routing/calibration/seed/baseline/decision design using these bearings.',
                original_pipeline_recovered=True, constants_fixed_in_bootstrap=True,
                interpretation='Exploratory input-dependence accounting only, not a causal or information-theoretic decomposition.')


def run(root: Path, recorded_root: Path, output: Path) -> int:
    rows: list[dict] = []
    try:
        rows, artifacts, passed = audit(root, recorded_root)
        _write_csv(output/'reference_validity.csv', rows)
        if not passed:
            write_json(output/'failure.json', dict(post_hoc=True, status='protocol_invalid',
                       step='A', step_b_executed=False, failures=[r for r in rows
                       if r.get('critical') and r['status'] != 'pass'], permanent_test_sealed=True))
            return 2
        print('Step A: no known semantic mismatch; pass_with_limitations. Starting fixed source constants.', flush=True)
        result = controls(root, output, artifacts)
        write_json(output/'result.json', result)
        print(json.dumps(result), flush=True)
        return 0
    except (ValueError, KeyError, AssertionError, FileNotFoundError) as error:
        rows.append(dict(row_type='protocol_check', check='execution_failure', status='unresolved',
                         critical=True, evidence=f'{type(error).__name__}: {error}', post_hoc=True))
        _write_csv(output/'reference_validity.csv', rows)
        write_json(output/'failure.json', dict(post_hoc=True, status='blocked', error=f'{type(error).__name__}: {error}',
                   permanent_test_sealed=True, instruction='Do not repair predictions or reinterpret current test.'))
        traceback.print_exc()
        return 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--recorded-root', type=Path, required=True, help='Exact original absolute run root in saved descriptors')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    root, output = args.run_root.resolve(), args.output.resolve()
    if os.environ.get('CUDA_VISIBLE_DEVICES') != '':
        parser.error("Set CUDA_VISIBLE_DEVICES='' explicitly; this analysis is CPU-only.")
    if output == root or output.is_relative_to(root):
        parser.error('Output must be outside the read-only original run root.')
    output.mkdir(parents=True, exist_ok=False)
    protect_inputs(root)
    write_json(output/'command.json', dict(post_hoc=True, argv=sys.argv, python=sys.executable,
               run_root=str(root), recorded_root=str(args.recorded_root), output=str(output),
               benchmark_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
               device='CPU', CUDA_VISIBLE_DEVICES='', bootstrap_repeats=BOOTSTRAPS, analysis_seed=ANALYSIS_SEED))
    start = time.monotonic()
    with (output/'stdout.log').open('w') as stdout, (output/'stderr.log').open('w') as stderr:
        with redirect_stdout(stdout), redirect_stderr(stderr):
            code = run(root, args.recorded_root, output)
    forbidden = [name for name in sys.modules if name == 'torch' or name == 'h5py'
                 or name.startswith(('src.model_factory', 'experiments.p01.frozen_export'))]
    if forbidden:
        raise RuntimeError(f'Forbidden runtime import: {forbidden}')
    write_json(output/'exit_status.json', dict(post_hoc=True, exit_status=code,
               seconds=time.monotonic()-start, forbidden_runtime_imports=forbidden))
    print(f'G07 exit={code}: {output}')
    raise SystemExit(code)


if __name__ == '__main__':
    main()
