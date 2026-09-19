"""Run declared candidate comparisons through the existing H5 trainer.

Replication preserves fitting, pairing and selection, not only the model YAML.
Expected comparisons are written before training; completed runs do not define
which methods or seeds were requested. This is not final-alpha deployment.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys
import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULTS = dict(epochs=20, steps_per_epoch=50, units_per_domain=2, lr=.001,
                pair_shift=0, selection_brier_weight=.25,
                selection_predictor='candidate', device='cpu')


def make_arms(config: dict, study: str) -> dict[str, dict]:
    branches = config['model']['branches']
    names = [branch['name'] for branch in branches]
    if len(set(names)) != len(names) or any(not re.fullmatch(r'[A-Za-z0-9_-]+', name) for name in names):
        raise ValueError('Branch names must be unique path-safe strings.')
    if not branches and (study == 'operators' or not config['model'].get('use_reference_features', True)):
        raise ValueError('Operator enumeration needs branches; a branchless model must retain reference features.')
    arms = {}
    if study == 'readouts':
        # Same frozen features, loss, observation and selection budget. The two
        # parameter counts are reported rather than asserted to be identical.
        for name, kind in [('linear_head', 'linear'), ('nonlinear_head', 'mlp')]:
            cfg = copy.deepcopy(config)
            cfg['model'].update(branches=[], use_reference_features=True, head_type=kind,
                                head_hidden_dim=config['model'].get('head_hidden_dim', 16))
            cfg['loss'].update(reduction='mean_source', lambda_delta=0.)
            arms[name] = cfg
    elif study == 'operators':
        groups = {'head_only': [], **{f'only_{b["name"]}': [b] for b in branches}, 'all_operators': branches}
        for name, group in groups.items():
            cfg = copy.deepcopy(config)
            cfg['model']['branches'] = copy.deepcopy(group)
            cfg['model']['use_reference_features'] = True
            cfg['loss'].update(reduction='mean_source', lambda_delta=0.)
            arms[name] = cfg
    elif study == 'losses':
        if config['loss']['brier_weight'] <= 0 or config['loss']['lambda_delta'] <= 0:
            raise ValueError('Loss comparisons require positive Brier and consistency weights.')
        for name in ('ce_mean', 'score_mean', 'score_worst', 'full'):
            cfg = copy.deepcopy(config)
            cfg['loss']['reduction'] = 'mean_source' if name in {'ce_mean', 'score_mean'} else 'worst_source'
            if name == 'ce_mean': cfg['loss']['brier_weight'] = 0.
            if name != 'full': cfg['loss']['lambda_delta'] = 0.
            arms[name] = cfg
    elif study == 'd1_losses':
        if not branches or config['model'].get('head_type','linear')!='linear' or not config['model'].get('use_reference_features',True):
            raise ValueError('D1 loss comparison fixes the reference-plus-operators linear candidate.')
        for name, reduction, prior, target, weight in (
                ('O','mean_source','relative','correction',0.),
                ('UO','worst_source','absolute','candidate',.1),
                ('RO','worst_source','relative','candidate',.1),
                ('RC','worst_source','relative','correction',.1)):
            cfg=copy.deepcopy(config)
            cfg['loss'].update(reduction=reduction,risk_reference=prior,consistency_target=target,lambda_delta=weight)
            arms[name]=cfg
    elif study == 'replicate':
        arms['frozen_configuration'] = copy.deepcopy(config)
    else:
        raise ValueError(f'Unknown study: {study}')
    return arms


def resolve_recipe(args):
    if args.recipe:
        if args.study != 'replicate':
            raise ValueError('--recipe is for an already fitted arm.')
        path = Path(args.recipe).expanduser().resolve()
        saved = json.loads(path.read_text())
        missing = set(DEFAULTS) - saved.keys()
        if missing:
            raise ValueError(f'Legacy command lacks explicit settings {sorted(missing)}; declare a complete new recipe instead.')
        for key in DEFAULTS:
            supplied = getattr(args, key)
            if supplied is not None and supplied != saved[key]:
                raise ValueError(f'Replication changes {key}; run a separately named study instead.')
            setattr(args, key, saved[key])
        for key, name in [('model_config', 'model_config.yaml'), ('data_config', 'data_config.yaml')]:
            if getattr(args, key) is not None:
                raise ValueError('A saved recipe owns model/data snapshots; omit extra config arguments.')
            value = path.parent / name
            if not value.is_file(): raise FileNotFoundError(value)
            setattr(args, key, str(value))
        if args.dataset is not None and args.dataset != saved['dataset']:
            raise ValueError('Replication cannot change dataset.')
        args.dataset = saved['dataset']
    else:
        if args.study == 'replicate' and any(getattr(args, k) is None for k in DEFAULTS):
            raise ValueError('replicate needs --recipe or every fitting/selection/device setting explicitly.')
        for key, default in DEFAULTS.items():
            if getattr(args, key) is None: setattr(args, key, default)
    if any(getattr(args, k) is None for k in ('model_config', 'data_config', 'dataset')):
        raise ValueError('Supply model-config, data-config and dataset, or a complete saved recipe.')
    return args


def training_command(args, config_path: Path, output: Path, seed: int) -> list[str]:
    cmd = [sys.executable, str(ROOT / 'experiments/p01/train_tspn_fusion_v2.py'),
           '--model-config', str(config_path), '--data-config', str(Path(args.data_config).resolve()),
           '--dataset', args.dataset, '--output', str(output), '--seed', str(seed)]
    for key in DEFAULTS:
        cmd += ['--'+key.replace('_', '-'), str(getattr(args, key))]
    if args.evaluate_test: cmd.append('--evaluate-test')
    return cmd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--study', choices=['readouts', 'operators', 'losses', 'd1_losses', 'replicate'], required=True)
    parser.add_argument('--model-config'); parser.add_argument('--data-config'); parser.add_argument('--dataset')
    parser.add_argument('--recipe', help='Prior run command.json with adjacent model/data snapshots.')
    parser.add_argument('--output', required=True); parser.add_argument('--arms', nargs='+')
    parser.add_argument('--seeds', type=int, nargs='+', default=[42])
    for key, default in DEFAULTS.items():
        parser.add_argument('--'+key.replace('_', '-'), type=type(default), default=None)
    parser.add_argument('--evaluate-test', action='store_true', help='Explicit test release; does not change fitting.')
    parser.add_argument('--dry-run', action='store_true')
    args = resolve_recipe(parser.parse_args())
    if len(set(args.seeds)) != len(args.seeds): raise ValueError('Duplicate seed values.')
    config = yaml.safe_load(Path(args.model_config).read_text())
    data = yaml.safe_load(Path(args.data_config).read_text())
    dataset = next(d for d in data['datasets'] if d['name'] == args.dataset)
    if config['model']['name'] != 'TSPN_fusion': raise ValueError('Use the existing TSPN_fusion model.')
    arms = make_arms(config, args.study)
    if args.arms:
        if len(set(args.arms)) != len(args.arms) or set(args.arms)-arms.keys():
            raise ValueError('Unknown or duplicate arm.')
        arms = {name: arms[name] for name in args.arms}
    if (args.study == 'losses' or any(c['loss']['lambda_delta'] > 0 for c in arms.values())) and args.pair_shift <= 0:
        raise ValueError('Declare the same justified nonzero pair shift for the comparison.')
    output = Path(args.output).resolve()
    if output.exists(): raise FileExistsError('Use a new output directory.')
    if not args.dry_run:
        (output/'configs').mkdir(parents=True)
        saved = vars(args).copy(); saved['arms'] = list(arms)
        saved['domains'] = {'validation': list(map(str, dataset['source_domains'])),
                            'test': list(map(str, dataset['source_domains']+dataset['domain_sequence']))}
        (output/'command.json').write_text(json.dumps(saved, indent=2))
    paths = {}
    for name, cfg in arms.items():
        checkpoint = Path(cfg['model']['checkpoint_path']).expanduser()
        cfg['model']['checkpoint_path'] = str(checkpoint if checkpoint.is_absolute() else ROOT/checkpoint)
        paths[name] = output/'configs'/f'{name}.yaml'
        if not args.dry_run: paths[name].write_text(yaml.safe_dump(cfg, sort_keys=False))
    for name in arms:
        for seed in args.seeds:
            cmd = training_command(args, paths[name], output/name/f'seed_{seed}', seed)
            print(shlex.join(cmd), flush=True)
            if not args.dry_run: subprocess.run(cmd, cwd=ROOT, check=True)


if __name__ == '__main__': main()
