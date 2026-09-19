"""D1 frozen all-candidate inference: prepare -> assess -> verify-source -> test.

The existing D1 plan owns scientific choices. ``final_candidates`` adds the 15
completed arm/seed checkpoints; ``candidates`` remains the four-member bank.
Only ``test`` reads permanent-test waveforms. Interrupted test export can resume
missing files without changing any bundle, coefficient, or observation rule.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.p01.fusion_data import read_records
from experiments.p01.fusion_assessment import same_reference
from experiments.p01.fusion_deployment import (
    load_model, local_path, predict_records, save_bundle,
    selection_excess, selection_risk, verify_vectors, write_csv,
)

CORE_ARMS = ('MLP16', 'O', 'UO', 'RO', 'RC')
SEEDS = (42, 123, 456)


def write_json(path: Path, value: object) -> None:
    with path.open('x', encoding='utf-8') as handle:
        json.dump(value, handle, indent=2, allow_nan=False)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding='utf-8'))


def validate_plan(plan: dict) -> None:
    final = plan['final_candidates']
    slots = [(item['arm'], int(item['seed'])) for item in final if item['arm'] in CORE_ARMS]
    required = {(arm, seed) for arm in CORE_ARMS for seed in SEEDS}
    if len(slots) != len(required) or set(slots) != required:
        raise ValueError('F1 requires exactly the 15 completed MLP16/O/UO/RO/RC x 42/123/456 slots.')
    names = [item['name'] for item in final]
    if len(set(names)) != len(names):
        raise ValueError('Final candidate names must be unique.')
    if any(item['arm'] not in (*CORE_ARMS, 'ResNet1D', 'FIRNet') for item in final):
        raise ValueError('Only prespecified core arms and completed ResNet1D/FIRNet baselines may enter test.')
    bank = plan['candidates']
    if len(bank) != 4 or len({item['name'] for item in bank}) != 4:
        raise ValueError('The primary D1 adoption bank has K=4.')
    if {item.get('arm', item['name']) for item in bank} != {'temperature', 'MLP16', 'O', 'RC'} or any(int(item.get('seed', 42)) != 42 for item in bank):
        raise ValueError('The primary bank is temperature/MLP16/O/RC at seed 42 only.')
    if plan['mode'] not in {'independent', 'empirical'}:
        raise ValueError('Freeze independent or empirical mode before assessment.')
    if (plan['rule'], plan['bound'], plan['scope'], float(plan['delta_total']), float(plan['delta_shift'])) != (
            'moments', 'bernstein', 'source_mixture', .05, 0.):
        raise ValueError('D1 primary rule is source-mixture moments/Bernstein, delta_total=.05, delta_shift=0.')
    if not np.array_equal(np.asarray(plan['selection_alpha_grid'], float), np.arange(11) / 10):
        raise ValueError('Freeze the D1 source coefficient grid 0,.1,...,1.')
    for item in bank:
        if item.get('arm', item['name']) == 'temperature' and list(map(float, item['temperature_grid'])) != [.5, .75, 1., 1.5, 2.]:
            raise ValueError('Freeze the D1 temperature grid .5,.75,1,1.5,2.')


def context(root: Path) -> tuple[dict, dict, dict, list[dict]]:
    f1 = read_json(root / 'F1.json')
    data = yaml.safe_load((root / 'data_config.yaml').read_text())
    dataset = next(item for item in data['datasets'] if item['name'] == f1['plan']['dataset'])
    records = json.loads((root / 'records.json').read_text())
    return f1, data, dataset, records


def select_records(records: list[dict], dataset: dict, split: str) -> list[dict]:
    sources = set(map(str, dataset['source_domains']))
    selected = [r for r in records if r['split'] == split and (split == 'test' or r['domain'] in sources)]
    if not selected:
        raise ValueError(f'No frozen observations for {split}.')
    return selected


def frozen_predict(model, records, dataset, data, classes, device, *, kind, temperature, alpha):
    if model.training or any(module.training for module in model.modules()):
        raise ValueError('Predict-only export requires all modules in eval mode.')
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}
    with torch.inference_mode():
        result = predict_records(model, records, dataset, data, classes, device, kind, temperature, alpha)
    if any(not torch.equal(value, model.state_dict()[key]) for key, value in before.items()):
        raise RuntimeError('Predict-only inference changed a parameter or buffer; release is invalid.')
    return result


def annotate(predictions: dict, spec: dict, f1: dict, split: str) -> dict:
    return dict(predictions, class_names=predictions['raw_class_names'],
                arm=np.asarray(spec['arm']), seed=np.asarray(spec['seed']),
                split=np.asarray(split), role=np.asarray(spec['role']),
                alpha=np.asarray(spec['alpha']), checkpoint=np.asarray(spec['checkpoint']),
                code_commit=np.asarray(f1['code_commit']),
                config_snapshot=np.asarray(json.dumps(f1['plan'], sort_keys=True)))


def save_predictions(path: Path, predictions: dict) -> None:
    # The final pathname appears only after the complete NPZ is closed. A crash
    # cannot make a partial output look like a completed permanent-test export.
    partial = path.with_suffix('.partial.npz')
    if path.exists():
        raise FileExistsError(f'Completed predictions cannot be overwritten: {path}')
    if partial.exists():
        partial.rename(path.with_suffix(f'.interrupted-{time.time_ns()}.npz'))
    with partial.open('xb') as handle:
        np.savez(handle, **predictions)
    os.replace(partial, path)


def prepare(plan_path: str, output: str, device: str) -> None:
    plan_path = Path(plan_path).resolve()
    plan = yaml.safe_load(plan_path.read_text())
    validate_plan(plan)
    for item in plan['candidates']:
        item.setdefault('arm', item['name'])
        item.setdefault('seed', 42)
    base = plan_path.parent
    for item in plan['final_candidates'] + plan['candidates']:
        item['checkpoint'] = str(local_path(item['checkpoint'], base))
        if not Path(item['checkpoint']).is_file():
            raise FileNotFoundError(item['checkpoint'])
    source_runs = []
    for item in plan['final_candidates']:
        run = Path(item['checkpoint']).parent
        source_runs.append(dict(arm=item['arm'], seed=int(item['seed']),
                                command=read_json(run / 'command.json'),
                                model_config=yaml.safe_load((run / 'model_config.yaml').read_text()),
                                data_config=yaml.safe_load((run / 'data_config.yaml').read_text()),
                                result_scope=read_json(run / 'result_scope.json')))
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=False)
    data_path = local_path(plan['data_config'], base)
    data = yaml.safe_load(data_path.read_text())
    dataset = next(d for d in data['datasets'] if d['name'] == plan['dataset'])
    records = read_records(dataset, data)
    classes = plan['class_names']
    rates = {r['sample_rate_hz'] for r in records}
    if len(rates) != 1:
        raise ValueError('Frozen D1 needs one declared sampling convention.')
    val = select_records(records, dataset, 'validation')
    (root / 'data_config.yaml').write_text(yaml.safe_dump(data, sort_keys=False))
    write_json(root / 'records.json', records)
    write_json(root / 'command.json', dict(argv=sys.argv, executable=sys.executable, device=device))
    history = set()
    for value in plan.get('development_group_files', []) + plan.get('reference_development_group_files', []):
        import csv
        with local_path(value, base).open(newline='') as handle:
            rows = csv.DictReader(handle)
            if 'group_id' not in (rows.fieldnames or []):
                raise ValueError('Development histories require literal group_id.')
            history.update(row['group_id'] for row in rows)
    history.update(r['unit_id'] for r in records if r['domain'] in set(map(str, dataset['source_domains'])) and r['split'] in {'update', 'validation'})
    if history.intersection(r['unit_id'] for r in records if r['split'] in {'assessment', 'test'}):
        raise ValueError('A development group overlaps assessment or permanent test.')
    if plan['mode'] == 'independent' and not plan.get('reference_development_group_files'):
        raise ValueError('Independent mode requires actual reference development history.')
    write_csv(root / 'development_groups.csv', [dict(group_id=g) for g in sorted(history)])
    code = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
    f1 = dict(plan=plan, code_commit=code, device=device, predictors=[], bank=[], source_runs=source_runs,
              dataset=plan['dataset'], source_domains=list(map(str, dataset['source_domains'])),
              class_names=classes, sample_rate_hz=next(iter(rates)))
    (root / 'bundles').mkdir()
    (root / 'source_validation').mkdir()
    torch.set_num_threads(1)
    start = time.perf_counter()
    reference_vectors = None

    def freeze(item, *, name, role, alpha=1., temperature=1., predictions=None):
        nonlocal reference_vectors
        if not name or Path(name).name != name:
            raise ValueError('Predictor names must be nonempty single path components.')
        model, saved = load_model(item['checkpoint'], device)
        reference_config = saved['reference_model'] if saved.get('kind') == 'classifier' else saved['model']['reference_config']
        if int(reference_config['in_dim']) != int(data['data']['window_size']):
            raise ValueError('Frozen observation window differs from the complete reference model interval.')
        kind = item.get('kind', 'model')
        spec = dict(name=name, arm=item['arm'], seed=int(item['seed']), role=role,
                    kind=kind, temperature=float(temperature), alpha=float(alpha),
                    checkpoint=item['checkpoint'], bundle=str(root / 'bundles' / f'{name}.pt'))
        if predictions is None:
            predictions = frozen_predict(model, val, dataset, data, classes, device, kind=kind, temperature=temperature, alpha=alpha)
        if reference_vectors is None:
            reference_vectors = predictions
        else:
            for key in ('labels', 'group_ids', 'acquisition_ids', 'window_ids', 'domains'):
                if not np.array_equal(reference_vectors[key], predictions[key]):
                    raise ValueError(f'Frozen predictors saw different {key}.')
            same_reference(reference_vectors['raw_probs'], classes, predictions['raw_probs'], classes)
        save_bundle(model, saved['model'], spec['bundle'], kind=kind, temperature=temperature, alpha=alpha,
                    classes=classes, input_data=data['data'], sampling_rate=next(iter(rates)),
                    scope=dict(mode=plan['mode'], role=role, assessment_scope='source_mixture'))
        save_predictions(root / 'source_validation' / f'{name}.npz', annotate(predictions, spec, f1, 'validation'))
        f1['predictors'].append(spec)
        return spec

    direct = {}
    for item in plan['final_candidates']:
        direct[(item['arm'], int(item['seed']))] = freeze(item, name=item['name'], role='direct')
    selection_rows = []
    for item in plan['candidates']:
        model, _ = load_model(item['checkpoint'], device)
        trials = []
        for temperature in item.get('temperature_grid', [1.]):
            p = frozen_predict(model, val, dataset, data, classes, device, kind=item.get('kind', 'model'), temperature=float(temperature), alpha=1.)
            trials.append((selection_risk(p), float(temperature), p))
        _, temperature, p = min(trials, key=lambda row: (row[0], row[1]))
        score, alpha = min((selection_excess(p, float(a)), float(a)) for a in plan['selection_alpha_grid'])
        if item['arm'] == 'temperature':
            frozen_direct = freeze(item, name='temperature_seed42', role='direct', temperature=temperature, predictions=p)
        else:
            frozen_direct = direct[(item['arm'], 42)]
            if frozen_direct['checkpoint'] != item['checkpoint']:
                raise ValueError('Bank candidate must be exactly the frozen seed-42 final checkpoint.')
        selected = freeze(item, name=f"{item['arm']}_seed42_source_selected", role='source_selected', temperature=temperature, alpha=alpha)
        f1['bank'].append(dict(name=item['name'], direct=frozen_direct, source_selected=selected))
        selection_rows.append(dict(candidate=item['name'], arm=item['arm'], seed=42, temperature=temperature,
                                   alpha=alpha, worst_source_brier=selection_risk(p), worst_source_brier_excess=score))
    raw_item = next(item for item in plan['final_candidates'] if item['arm'] == 'O' and int(item['seed']) == 42)
    freeze(dict(raw_item, arm='raw'), name='raw_reference', role='raw', alpha=0.)
    write_csv(root / 'source_selection.csv', selection_rows)
    f1['source_selection_seconds'] = time.perf_counter() - start
    # This is written last: incomplete preparation never opens assessment.
    write_json(root / 'F1.json', f1)


def predict_spec(root: Path, spec: dict, split: str, device: str) -> dict:
    f1, data, dataset, records = context(root)
    model, saved = load_model(spec['bundle'], device)
    frozen = saved['deployment']
    for key in ('kind', 'temperature', 'alpha'):
        if frozen[key] != spec[key]:
            raise ValueError(f'Bundle disagrees with frozen {key}.')
    if frozen['data'] != data['data'] or frozen['class_names'] != f1['class_names']:
        raise ValueError('Bundle observation rule/class order differs from F1.')
    p = frozen_predict(model, select_records(records, dataset, split), dataset, data, f1['class_names'], device,
                       kind=spec['kind'], temperature=spec['temperature'], alpha=spec['alpha'])
    return annotate(p, spec, f1, split)


def assess_frozen(output: str, device: str) -> None:
    from experiments.p01.calibrate_tspn_fusion import assess_plan
    root = Path(output).resolve()
    f1, data, dataset, records = context(root)
    if (root / 'F2.json').exists() or (root / 'assessment').exists():
        raise FileExistsError('Assessment is single-use; preserve any partial result for diagnosis.')
    mode = f1['plan']['mode']
    start = time.perf_counter()
    bank = []
    if mode == 'independent':
        (root / 'assessment').mkdir()
        for item in f1['bank']:
            spec = item['direct']
            p = predict_spec(root, spec, 'assessment', device)
            path = root / 'assessment' / f"{spec['name']}.npz"
            save_predictions(path, p)
            bank.append(dict(name=item['name'], kind='model', predictions=str(path), fixed_alpha=item['source_selected']['alpha']))
        plan = dict(scope='source_mixture', bound='bernstein', delta_total=.05, delta_shift=0.,
                    rule_budgets={'moments': .05}, candidates=bank, conditions=f1['source_domains'],
                    development_group_files=[str(root / 'development_groups.csv')])
        path = root / 'assessment' / 'plan.yaml'
        path.write_text(yaml.safe_dump(plan, sort_keys=False))
        result = assess_plan(path, root / 'decision.json')
        chosen = result['rules']['moments']['selected']
        index, alpha = int(chosen['candidate']), float(chosen['alpha'])
    else:
        import csv
        with (root / 'source_selection.csv').open(newline='') as handle:
            selections = list(csv.DictReader(handle))
        index = min(range(len(selections)), key=lambda i: (float(selections[i]['worst_source_brier_excess']), float(selections[i]['alpha'])))
        alpha = float(selections[index]['alpha'])
        write_json(root / 'decision.json', dict(mode=mode, selected=index, alpha=alpha, independent_coverage=None))
    selected = dict(f1['bank'][index]['direct'], name='adopted', role='adopted', alpha=alpha,
                    bundle=str(root / 'bundles' / 'adopted.pt'))
    source = f1['bank'][index]['direct']
    model, saved = load_model(source['bundle'], device)
    save_bundle(model, saved['model'], selected['bundle'], kind=selected['kind'], temperature=selected['temperature'],
                alpha=alpha, classes=f1['class_names'], input_data=data['data'], sampling_rate=f1['sample_rate_hz'],
                scope=dict(mode=mode, assessment_scope='source_mixture' if mode == 'independent' else None,
                           external_test='empirical: new physical conditions are outside source-mixture coverage'))
    p = predict_spec(root, selected, 'validation', device)
    save_predictions(root / 'source_validation' / 'adopted.npz', p)
    write_json(root / 'source_validation' / 'exports.json', [
        dict(name=spec['name'], arm=spec['arm'], seed=spec['seed'], role=spec['role'], split='validation',
             path=str(root / 'source_validation' / f"{spec['name']}.npz"), alpha=spec['alpha'])
        for spec in f1['predictors'] + [selected]])
    write_json(root / 'F2.json', dict(selected=selected, assessment_seconds=time.perf_counter() - start,
                                   predictors=f1['predictors'] + [selected], mode=mode))


def worker(output: str, name: str, split: str, destination: str, device: str) -> None:
    root = Path(output).resolve()
    f2 = read_json(root / 'F2.json')
    spec = next(item for item in f2['predictors'] if item['name'] == name)
    if split == 'test' and not (root / 'test_release.json').is_file():
        raise ValueError('Permanent test has not been released.')
    save_predictions(Path(destination), predict_spec(root, spec, split, device))


def verify_source(output: str, device: str) -> None:
    root = Path(output).resolve()
    f2 = read_json(root / 'F2.json')
    directory = root / 'restored_source_validation'
    directory.mkdir(exist_ok=True)
    for spec in f2['predictors']:
        path = directory / f"{spec['name']}.npz"
        if not path.exists():
            subprocess.run([sys.executable, str(Path(__file__).resolve()), '_predict', '--output', str(root),
                            '--name', spec['name'], '--split', 'validation', '--destination', str(path), '--device', device], check=True)
        with np.load(root / 'source_validation' / f"{spec['name']}.npz", allow_pickle=False) as expected, np.load(path, allow_pickle=False) as actual:
            verify_vectors(expected, actual)
    if not (root / 'source_verified.json').exists():
        write_json(root / 'source_verified.json', dict(predictors=[spec['name'] for spec in f2['predictors']],
                                                     executable=sys.executable, device=device, separate_process=True))


def release_test(output: str, device: str) -> None:
    root = Path(output).resolve()
    f1 = read_json(root / 'F1.json')
    validate_plan(f1['plan'])
    f2 = read_json(root / 'F2.json')
    verified = read_json(root / 'source_verified.json')
    names = [spec['name'] for spec in f2['predictors']]
    if verified['predictors'] != names or not verified['separate_process']:
        raise ValueError('Every planned predictor needs separate-process source verification before test.')
    release = root / 'test_release.json'
    if release.exists():
        if read_json(release)['predictors'] != names:
            raise ValueError('A permanent release cannot change the frozen predictor list.')
    else:
        write_json(release, dict(predictors=names, code_commit=f1['code_commit'], mode=f2['mode'],
                                instruction='One permanent-test release; resume missing frozen exports only.'))
    directory = root / 'test'
    directory.mkdir(exist_ok=True)
    exports = []
    start = time.perf_counter()
    for spec in f2['predictors']:
        path = directory / f"{spec['name']}.npz"
        if not path.exists():
            worker(str(root), spec['name'], 'test', str(path), device)
        with np.load(path, allow_pickle=False) as p:
            if str(p['arm']) != spec['arm'] or int(p['seed']) != spec['seed'] or float(p['alpha']) != spec['alpha']:
                raise ValueError('An existing test output disagrees with its frozen predictor.')
        exports.append(dict(name=spec['name'], arm=spec['arm'], seed=spec['seed'], role=spec['role'],
                            split='test', path=str(path), alpha=spec['alpha']))
    if not (directory / 'exports.json').exists():
        write_json(directory / 'exports.json', exports)
        write_json(directory / 'runtime.json', dict(inference_seconds=time.perf_counter() - start,
                                                  note='Wall time for this invocation; interrupted earlier invocations retain their logs.'))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='phase', required=True)
    for phase in ('prepare', 'assess', 'verify-source', 'test', '_predict'):
        command = sub.add_parser(phase)
        command.add_argument('--output', required=True)
        command.add_argument('--device', default='cpu')
        if phase == 'prepare':
            command.add_argument('--plan', required=True)
        if phase == '_predict':
            command.add_argument('--name', required=True)
            command.add_argument('--split', choices=['validation', 'test'], required=True)
            command.add_argument('--destination', required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    if args.phase == 'prepare':
        prepare(args.plan, args.output, args.device)
    elif args.phase == 'assess':
        assess_frozen(args.output, args.device)
    elif args.phase == 'verify-source':
        verify_source(args.output, args.device)
    elif args.phase == 'test':
        release_test(args.output, args.device)
    else:
        worker(args.output, args.name, args.split, args.destination, args.device)


if __name__ == '__main__':
    main()
