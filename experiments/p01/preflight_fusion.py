"""D1 declarations and source support; never load assessment/test waveforms.

Read existing group labels and H5 keys via fusion_data. History files are supplied
by the experimenter: passing an overlap check does not prove history completeness.
This command does not fit a candidate, choose a risk rule or release permanent test.
"""
from __future__ import annotations
import argparse
import csv
import math
from pathlib import Path
import sys
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))


def write_csv(path, rows, fields=None):
    with Path(path).open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def group_counts(records, classes, windows):
    rows = []
    for domain in sorted({r['domain'] for r in records}):
        for label in range(classes):
            part = [r for r in records if r['domain'] == domain and r['label'] == label]
            row = dict(condition=domain, label=label)
            for split in ('update', 'validation', 'assessment', 'test'):
                selected = [r for r in part if r['split'] == split]
                row[split + '_groups'] = len({r['unit_id'] for r in selected})
                row[split + '_acquisitions'] = len(selected)
                row[split + '_planned_windows'] = len(selected) * windows
            rows.append(row)
    return rows


def history_groups(paths):
    groups = set()
    for path in paths:
        frame = pd.read_csv(path, dtype=str, keep_default_na=False)
        if frame.empty or 'group_id' not in frame or (frame['group_id'] == '').any():
            raise ValueError(f'{path}: nonempty reference history with group_id required.')
        groups.update(frame['group_id'])
    return groups


def support_rows(config, length, fs):
    """Use actual configured analysis objects, not an invented STFT/CWT formula."""
    from src.model_factory.X_model.TSPN_fusion import EnvelopeBranch
    from src.model_factory.X_model.TSPN_tf_operators import TimeFrequencyBranch
    rows = []
    channels = int(config['reference_config']['in_channels'])
    for spec in config['branches']:
        kind, name = spec['type'], spec['name']
        if kind == 'envelope':
            b = EnvelopeBranch(channels, spec['carrier'], spec['modulation'])
            for component in ('carrier', 'modulation'):
                bands = getattr(b, component); bands.check_grid(length)
                centers, widths = bands.parameters_in_frequency()
                for i in range(bands.count):
                    rows.append(dict(branch=name, component=component, row=i,
                        support_samples=length, valid_positions=length, duration_s=length/fs,
                        frequency_hz=float(centers[i].detach())*fs,
                        width_hz=float(widths[i].detach())*fs, window_enbw_hz='',
                        grid_hz=fs/length, convention='periodic FFT; three-width support'))
        else:
            b = TimeFrequencyBranch(kind, channels, spec['transform'], spec['readout'])
            t = b.transform; positions = len(t.times(length))
            if positions < b.readout.time_bins:
                raise ValueError(f'{name}: insufficient measured support for the declared time bins.')
            support = t.win_length if kind == 'stft' else 2*t.radius+1
            for i, frequency in enumerate(t.frequency.tolist()):
                rows.append(dict(branch=name, component=kind, row=i,
                    support_samples=support, valid_positions=positions, duration_s=support/fs,
                    frequency_hz=frequency*fs, width_hz='',
                    window_enbw_hz=t.enbw()*fs if kind == 'stft' else '',
                    grid_hz=fs/t.n_fft if kind == 'stft' else '',
                    convention=t.coordinate_kind + '; valid measured centers'))
    return rows


def run(plan_path, model_path, output):
    from experiments.p01.fusion_data import read_records
    plan_path = Path(plan_path).resolve(); base = plan_path.parent
    plan = yaml.safe_load(plan_path.read_text())
    model = yaml.safe_load(Path(model_path).read_text())
    def local(value):
        p = Path(value).expanduser()
        return p if p.is_absolute() else base/p
    data = yaml.safe_load(local(plan['data_config']).read_text())
    dataset = next(d for d in data['datasets'] if d['name'] == plan['dataset'])
    mode = plan['mode']
    if mode not in {'independent', 'empirical'}:
        raise ValueError('Choose mode independent or empirical before fitting; no inferred route.')
    controls = plan['d1_controls']
    if controls['tau'] != model['loss']['tau'] or not 0 < controls['tau'] <= 1:
        raise ValueError('One declared nonzero tau must match the source model configuration.')
    if controls['selection_predictor'] not in {'candidate', 'training_tau_mixture'}:
        raise ValueError('Declare the source-selection predictor.')
    if not math.isfinite(controls['selection_brier_weight']) or controls['selection_brier_weight'] < 0:
        raise ValueError('Declare a nonnegative source-selection Brier weight.')
    if controls['nonlinear_hidden_dim'] < 1 or type(controls['nonlinear_hidden_dim']) is not int:
        raise ValueError('Declare one positive integer nonlinear width.')
    if model['model'].get('head_hidden_dim', 16) != controls['nonlinear_hidden_dim']:
        raise ValueError('Declared nonlinear width differs from the readout-study configuration.')
    names = [c['name'] for c in plan['candidates']]
    if not names or len(names) != len(set(names)):
        raise ValueError('The assessment bank must have unique fixed candidates.')
    if plan['rule'] not in {'moments', 'paired'}:
        raise ValueError('Declare one primary rule; moments_fixed is diagnostic only.')
    records = read_records(dataset, data)
    sources = list(map(str, dataset['source_domains']))
    classes = int(data['model']['num_classes'])
    if (int(model['model']['num_classes']) != classes or
            int(model['model']['reference_config']['num_classes']) != classes):
        raise ValueError('Data, reference and candidate class counts must agree.')
    if len(plan['class_names']) != classes or len(set(plan['class_names'])) != classes:
        raise ValueError('Declare the exact ordered class space.')
    length = int(data['data']['window_size'])
    if length != int(model['model']['reference_config']['in_dim']):
        raise ValueError('The actual observation interval must match the frozen reference.')
    # Match the actual trainer's fixed-frequency contract. This inspects declared
    # acquisition rates only, without using held-out signals to choose parameters.
    rates = {r['sample_rate_hz'] for r in records}
    if len(rates) != 1:
        raise ValueError('This normalized-frequency model needs one sampling convention across the declared task.')
    root = Path(output); root.mkdir(parents=True, exist_ok=False)
    write_csv(root/'group_counts.csv', group_counts(records, classes, int(data['data']['windows_per_unit'])))
    paths = [local(p) for p in plan.get('reference_development_group_files', [])]
    ref = history_groups(paths)
    current = {r['unit_id'] for r in records if r['domain'] in sources and r['split'] in {'update', 'validation'}}
    overlaps = []
    for owner, groups in [('reference', ref), ('candidate_declared_fit_and_selection', current)]:
        for split in ('assessment', 'test'):
            target = {r['unit_id'] for r in records if r['split'] == split}
            overlap = sorted(groups & target)
            overlaps.append(dict(owner=owner, partition=split, supplied_history=bool(paths) if owner == 'reference' else True,
                                 overlap_count=len(overlap), overlapping_groups=';'.join(overlap)))
    write_csv(root/'reference_overlap.csv', overlaps)
    supports = support_rows(model['model'], length, next(iter(rates)))
    write_csv(root/'operator_support.csv', supports, fields=['branch','component','row','support_samples','valid_positions',
        'duration_s','frequency_hz','width_hz','window_enbw_hz','grid_hz','convention'])
    reasons = []
    if any(r['overlap_count'] and (mode == 'independent' or r['partition'] == 'test') for r in overlaps):
        reasons.append('Known development overlap with protected observations.')
    counts = {domain: len({r['unit_id'] for r in records
                          if r['domain'] == domain and r['split'] == 'assessment'})
              for domain in sources}
    design = None
    design_note = 'requires independent moments and at least two groups per condition'
    missing_allocation = sorted({'delta_total', 'delta_shift'} - plan.keys())
    if mode == 'independent':
        if not paths:
            reasons.append('Complete reference history has not been supplied; choose empirical explicitly or retrain cleanly.')
        for domain, n in counts.items():
            # Both radius families feed the same sample-moment estimator.
            if n < 2:
                reasons.append(f'{domain}: insufficient independent assessment groups for the moment estimator (need at least two).')
        if plan['rule'] == 'moments' and missing_allocation:
            # Source-only preflight plans need not yet allocate assessment
            # error. Do not invent defaults or mask existing history errors.
            design_note = 'assessment allocation not fully declared: ' + ', '.join(missing_allocation)
        if plan['rule'] == 'moments' and not missing_allocation and all(n >= 2 for n in counts.values()):
            from experiments.p01.fusion_assessment import moment_design
            design = moment_design(counts, candidate_count=len(names),
                                   failure=float(plan['delta_total'])-float(plan['delta_shift']),
                                   method=plan['bound'])
            # The full physical group is counted once even if it has multiple
            # labels/acquisitions. These rows contain no observed probabilities.
            rows = [dict(**row, candidate_count=design['candidate_count'],
                         condition_count=design['condition_count'], method=design['method'],
                         rule_failure_budget=design['rule_failure_budget'],
                         event_failure_budget=design['event_failure_budget'])
                    for row in design['by_domain']]
            write_csv(root/'assessment_design.csv', rows)
    text = ['# D1 pre-test decision', '', f"Mode: {mode}. Primary rule: {plan['rule']}.",
            f"Assessment K: {len(names)}; candidates: {', '.join(names)}.",
            f"Training tau: {controls['tau']}; selection predictor: {controls['selection_predictor']}; selection Brier weight: {controls['selection_brier_weight']}.",
            f"Nonlinear width: {controls['nonlinear_hidden_dim']}; source-screening seed: {controls['seed']}.",
            'No assessment or test waveform was loaded; counts come from declared records and windows are planned counts.',
            'History completeness is a scientific obligation, not established by this overlap check.',
            'Passing counts does not imply enough power for a nonzero correction.',
            'Next: source-only candidate fitting. Candidate and rule definitions must be frozen before test prediction.', '']
    if design is not None:
        outcome = ('ruled out by the declared counts' if design['nonzero_ruled_out_by_count']
                   else 'not ruled out by the count-only check; candidate quality and power remain unknown')
        text += [f'Nonzero moment-rule adoption: {outcome}.',
                 'See assessment_design.csv. This diagnostic does not select a rule, change coefficients, or block valid source fitting.', '']
    else:
        text += [f'Count-only moment-design diagnostic: not computed ({design_note}).', '']
    text += ['Status: BLOCKED', *reasons] if reasons else ['Status: source fitting may proceed under the declared scope.']
    (root/'decision.md').write_text('\n'.join(text)+'\n')
    if reasons:
        raise ValueError(' '.join(reasons))
    print(root)
    return root


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--plan', required=True); p.add_argument('--model-config', required=True); p.add_argument('--output', required=True)
    a = p.parse_args(); run(a.plan, a.model_config, a.output)


if __name__ == '__main__': main()
