"""Describe saved P4 predictions with paired whole-unit bootstrap intervals.

No fitting, threshold calibration, or parameter selection occurs here. Intervals
condition on the fixed trained seeds and describe test-unit resampling only.
They are unadjusted descriptive intervals, not a contribution acceptance test.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from .core import matched_risk, risk_coverage, unit_weights


DEFAULT_METHODS = ('joint', 'fuzzy_rule_score', 'fuzzy_msp', 'class_only',
                   'mlp_msp', 'selectivenet_feature')
TARGETS = (.9, .5, .7)
SHARED_PREDICTOR = {'fuzzy_rule_score', 'fuzzy_msp', 'class_only'}


@dataclass(frozen=True)
class Prediction:
    error: np.ndarray
    score: np.ndarray
    eligible: np.ndarray
    accepted: np.ndarray


def resample_units(unit: np.ndarray, sampled: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Copy complete units, assigning each draw occurrence its own new ID."""
    groups = [np.flatnonzero(unit == value) for value in np.unique(unit)]
    indices = np.concatenate([groups[index] for index in sampled])
    new_units = np.concatenate([np.full(len(groups[index]), occurrence, dtype=int)
                                for occurrence, index in enumerate(sampled)])
    return indices, new_units


def complete_mean(values: list[float | None]) -> float | None:
    return None if any(value is None for value in values) else float(np.mean(values))


def seed_sd(values: list[float | None]) -> float | None:
    return (float(np.std(values, ddof=1)) if len(values) > 1 and
            all(value is not None for value in values) else None)


def load_predictions(run: Path, methods: tuple[str, ...], allow_diagnostic: bool
                     ) -> tuple[dict, dict, dict, np.ndarray, dict]:
    config = json.loads((run / 'config.json').read_text(encoding='utf-8'))
    state = json.loads((run / 'run_state.json').read_text(encoding='utf-8'))
    if state.get('status') != 'completed':
        raise ValueError('input run must be completed; failed/partial results are not silently pooled')
    for field in ('evaluation_role', 'data_kind'):
        if field in state and state[field] != config.get(field):
            raise ValueError(f'input run state and config disagree on {field}')
    role = config.get('evaluation_role')
    diagnostic = (config.get('data_kind') != 'real' or role != 'test' or
                  config.get('evidence_eligible') is not True or config.get('diagnostic_tune', False))
    if diagnostic and not allow_diagnostic:
        raise ValueError('synthetic/tune/ineligible inputs require explicit --allow-diagnostic')
    if role not in {'test', 'tune'}:
        raise ValueError('input config must declare test or tune evaluation_role')
    if config.get('data_kind') not in {'real', 'synthetic'}:
        raise ValueError('input config must declare real or synthetic data_kind')
    seeds = config.get('seeds', [])
    if not seeds or len(set(seeds)) != len(seeds) or not all(type(seed) is int for seed in seeds):
        raise ValueError('input config must declare distinct integer training seeds')
    if 'joint' not in methods or len(set(methods)) != len(methods) or len(methods) < 2:
        raise ValueError('choose joint and at least one distinct control')
    if not set(methods) <= set(config.get('methods', [])):
        raise ValueError('requested methods are absent from the recorded run')
    predictions, reference, reference_probabilities, files = {}, None, {}, []
    classes = None
    for method in methods:
        for seed in seeds:
            path = run / f'predictions_{method}_{seed}.npz'
            with np.load(path, allow_pickle=False) as archive:
                required = {'probabilities', 'score', 'eligible', 'accepted', 'y', 'unit', 'split', 'domain'}
                if not required <= set(archive.files):
                    raise ValueError(f'missing prediction arrays in {path.name}')
                arrays = {key: archive[key] for key in required}
            y, p = arrays['y'], arrays['probabilities']
            n = len(y)
            if y.shape != (n,) or not np.issubdtype(y.dtype, np.integer):
                raise ValueError('prediction labels must be an integer vector')
            if p.ndim != 2 or p.shape[0] != n or p.shape[1] < 2:
                raise ValueError('probabilities must have shape N x C with C >= 2')
            if classes is None:
                classes = p.shape[1]
            elif p.shape[1] != classes:
                raise ValueError('class dimension differs across methods or seeds')
            if not np.isfinite(p).all() or (p < 0).any() or not np.allclose(p.sum(1), 1., atol=1e-5):
                raise ValueError('prediction probabilities must be finite normalized probabilities')
            if (y < 0).any() or (y >= p.shape[1]).any():
                raise ValueError('prediction labels fall outside the class dimension')
            for key in required - {'probabilities'}:
                if arrays[key].shape != (n,):
                    raise ValueError(f'{key} must be an N-vector')
            for key in ('unit', 'split', 'domain'):
                if arrays[key].dtype.kind != 'U' or any(not value.strip() for value in arrays[key]):
                    raise ValueError(f'{key} must contain nonempty Unicode string identifiers')
            if arrays['eligible'].dtype != bool or arrays['accepted'].dtype != bool:
                raise ValueError('eligibility and acceptance arrays must be boolean')
            if (arrays['accepted'] & ~arrays['eligible']).any():
                raise ValueError('saved acceptance violates eligibility')
            if np.isnan(arrays['score']).any() or not np.isfinite(arrays['score'][arrays['eligible']]).all():
                raise ValueError('eligible scores must be finite; all scores must be non-NaN')
            metadata = {key: arrays[key] for key in ('y', 'unit', 'split', 'domain')}
            if reference is None:
                reference = metadata
                for value in np.unique(reference['unit']):
                    if len(np.unique(reference['split'][reference['unit'] == value])) != 1:
                        raise ValueError(f'physical unit crosses split roles: {value}')
            elif any(not np.array_equal(metadata[key], reference[key]) for key in metadata):
                raise ValueError('method/seed metadata order differs; paired rows cannot be inferred')
            mask = arrays['split'] == role
            if not mask.any():
                raise ValueError(f'input predictions contain no {role} rows')
            predictions[method, seed] = Prediction(
                (p[mask].argmax(1) != y[mask]).astype(float), arrays['score'][mask],
                arrays['eligible'][mask], arrays['accepted'][mask])
            reference_probabilities[method, seed] = p[mask]
            files.append(str(path.resolve()))
    for seed in seeds:
        for method in SHARED_PREDICTOR.intersection(methods):
            if not np.array_equal(reference_probabilities[method, seed], reference_probabilities['joint', seed]):
                raise ValueError(f'{method} and joint do not share the same saved predictor for seed {seed}')
    unit = reference['unit'][reference['split'] == role]
    if any(not str(value).strip() for value in unit):
        raise ValueError('evaluation unit identifiers must be nonempty')
    context = dict(diagnostic=bool(diagnostic), evaluation_role=role, seeds=seeds,
                   prediction_files=files, units=int(len(np.unique(unit))), rows=int(len(unit)),
                   domain_counts={str(d): int(np.sum(reference['domain'][reference['split'] == role] == d))
                                  for d in np.unique(reference['domain'][reference['split'] == role])})
    return config, state, context, unit, predictions


def paired_analysis(predictions: dict[tuple[str, int], Prediction], unit: np.ndarray,
                    methods: tuple[str, ...], seeds: list[int], draws: int = 1000,
                    bootstrap_seed: int = 20261003) -> tuple[list[dict], list[dict], list[dict]]:
    if draws < 1 or len(unit) == 0:
        raise ValueError('positive bootstrap draws and a nonempty evaluation cohort are required')
    controls = [method for method in methods if method != 'joint']
    point, per_seed = {}, []
    for target in TARGETS:
        for method in methods:
            for seed in seeds:
                prediction = predictions[method, seed]
                risk = matched_risk(prediction.error, prediction.score, prediction.eligible, unit, target)
                deployed_coverage, deployed_risk = risk_coverage(prediction.error, prediction.accepted, unit)
                point[method, seed, target] = risk
                per_seed.append(dict(method=method, seed=seed, target_coverage=target,
                    matched_risk=risk, attainable=risk is not None,
                    eligible_coverage=float(unit_weights(unit) @ prediction.eligible),
                    deployed_coverage=deployed_coverage, deployed_risk=deployed_risk,
                    full_risk=float(unit_weights(unit) @ prediction.error)))
    rng = np.random.default_rng(bootstrap_seed)
    n_units = len(np.unique(unit))
    bootstrap = []
    for draw in range(draws):
        indices, draw_units = resample_units(unit, rng.integers(0, n_units, size=n_units))
        for target in TARGETS:
            risks = {}
            for method in methods:
                risks[method] = []
                for seed in seeds:
                    prediction = predictions[method, seed]
                    risks[method].append(matched_risk(prediction.error[indices], prediction.score[indices],
                        prediction.eligible[indices], draw_units, target))
            joint_mean = complete_mean(risks['joint'])
            for control in controls:
                control_mean = complete_mean(risks[control])
                delta = (control_mean - joint_mean if joint_mean is not None and control_mean is not None else None)
                bootstrap.append(dict(draw=draw, control=control, target_coverage=target,
                    joint_risk=joint_mean, control_risk=control_mean, delta=delta,
                    undefined_seed_pairs=sum(j is None or c is None for j, c in zip(risks['joint'], risks[control]))))
    rows = []
    for target in TARGETS:
        for control in controls:
            joint_values = [point['joint', seed, target] for seed in seeds]
            control_values = [point[control, seed, target] for seed in seeds]
            deltas = [None if j is None or c is None else c - j for j, c in zip(joint_values, control_values)]
            values = [row['delta'] for row in bootstrap if row['control'] == control and row['target_coverage'] == target]
            undefined = sum(value is None for value in values)
            estimate = complete_mean(deltas)
            reason = ('point_estimate_undefined' if estimate is None else
                      'fewer_than_two_evaluation_units' if n_units < 2 else
                      'undefined_bootstrap_draws' if undefined else None)
            interval = np.quantile(values, [.025, .975]) if reason is None else (None, None)
            rows.append(dict(control=control, target_coverage=target,
                endpoint='primary' if target == .9 else 'secondary', joint_risk_mean=complete_mean(joint_values),
                control_risk_mean=complete_mean(control_values), delta=estimate,
                joint_seed_sd=seed_sd(joint_values), control_seed_sd=seed_sd(control_values),
                delta_seed_sd=seed_sd(deltas), seeds=len(seeds), undefined_seed_pairs=sum(d is None for d in deltas),
                ci_low=None if interval[0] is None else float(interval[0]),
                ci_high=None if interval[1] is None else float(interval[1]), ci_unavailable_reason=reason,
                bootstrap_valid_draws=draws - undefined, bootstrap_undefined_draws=undefined,
                bootstrap_total_draws=draws, evaluation_units=n_units,
                interpretation='descriptive control-minus-joint; no claim adjudication'))
    return rows, per_seed, bootstrap


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(run: Path, output: Path, methods: tuple[str, ...] = DEFAULT_METHODS,
              draws: int = 1000, bootstrap_seed: int = 20261003,
              allow_diagnostic: bool = False) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'output is not empty: {output}; use a new analysis directory')
    output.mkdir(parents=True, exist_ok=True)
    state_path = output / 'run_state.json'
    state_path.write_text(json.dumps(dict(status='running')), encoding='utf-8')
    try:
        config, input_state, context, unit, predictions = load_predictions(run, methods, allow_diagnostic)
        analysis_config = dict(input_run=str(run.resolve()), input_config=config, input_state=input_state,
            methods=list(methods), **context, bootstrap_draws=draws, bootstrap_seed=bootstrap_seed,
            coverage_targets=list(TARGETS), primary_coverage=.9, confidence_level=.95,
            bootstrap='whole units, with replacement; occurrence-specific unit IDs; common draws across methods/seeds',
            estimand='mean of per-training-seed unit-balanced matched risks; control minus joint',
            missing_policy='no omitted seeds or draws; any undefined draw makes that comparison CI unavailable',
            evidence_eligible=not context['diagnostic'], allow_diagnostic=allow_diagnostic,
            limits=['Fixed trained seeds; seed variation is reported separately and not bootstrapped.',
                    'Intervals are descriptive, unadjusted for multiple comparisons, conditional on saved fitting.',
                    'Independent physical unit identity must be justified by acquisition records.',
                    'Ordinary unit bootstrap does not hold domain counts fixed.',
                    'Metadata alignment cannot detect swapping indistinguishable rows within a unit.',
                    'Retrospective coverage matching is not the saved deployed calibrated policy.'])
        (output / 'analysis_config.json').write_text(json.dumps(analysis_config, indent=2, allow_nan=False), encoding='utf-8')
        paired, per_seed, bootstrap = paired_analysis(predictions, unit, methods, context['seeds'], draws, bootstrap_seed)
        write_csv(output / 'paired_matched_risk.csv', paired)
        write_csv(output / 'per_seed_metrics.csv', per_seed)
        write_csv(output / 'bootstrap_effects.csv', bootstrap)
        contribution_rows = []
        for row in paired:
            contribution_rows.append(dict(contribution='P4-C2 (empirical obligation of K1)', research_question='RQ2',
                experiment='shared-predictor scoring' if row['control'] in SHARED_PREDICTOR else 'separately fitted classifier',
                control=row['control'], metric='matched conditional error: control minus joint',
                target_coverage=row['target_coverage'], endpoint=row['endpoint'], delta=row['delta'],
                ci_low=row['ci_low'], ci_high=row['ci_high'],
                status='undefined' if row['delta'] is None else 'descriptive_observation',
                scientific_conclusion='not_adjudicated', diagnostic=context['diagnostic'],
                limitation=row['ci_unavailable_reason'] or 'unadjusted fixed-fit whole-unit percentile interval'))
        write_csv(output / 'contribution_experiment_result.csv', contribution_rows)
        result = dict(status='completed', scientific_conclusion='not_adjudicated',
            evidence_eligible=not context['diagnostic'], evaluation_role=context['evaluation_role'],
            input_run=str(run.resolve()), methods=list(methods), seeds=context['seeds'],
            bootstrap_draws=draws, bootstrap_seed=bootstrap_seed, paired_effects=paired,
            unassessed=['P4-C1 theorem assumptions and saved-component replay',
                        'baseline qualification and adequacy of the tuning budget',
                        'calibration/test sampling assumptions and physical independence'])
        (output / 'summary.json').write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
        state_path.write_text(json.dumps(dict(status='completed', scientific_conclusion='not_adjudicated',
                                             evidence_eligible=not context['diagnostic'])), encoding='utf-8')
        return result
    except BaseException as exc:
        state_path.write_text(json.dumps(dict(status='cancelled' if isinstance(exc, KeyboardInterrupt) else 'failed',
            error_type=type(exc).__name__, error=str(exc), scientific_conclusion='not_adjudicated')), encoding='utf-8')
        raise


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--methods', nargs='+', default=list(DEFAULT_METHODS))
    parser.add_argument('--bootstrap-draws', type=int, default=1000)
    parser.add_argument('--bootstrap-seed', type=int, default=20261003)
    parser.add_argument('--allow-diagnostic', action='store_true')
    args = parser.parse_args(argv)
    result = summarize(args.run, args.output, tuple(args.methods), args.bootstrap_draws,
                       args.bootstrap_seed, args.allow_diagnostic)
    print(json.dumps(dict(status=result['status'], output=str(args.output),
                         scientific_conclusion=result['scientific_conclusion']), indent=2))


if __name__ == '__main__':
    main()
