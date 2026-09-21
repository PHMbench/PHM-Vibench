"""Run one native TII shared model on two to five qualified industrial sources.

No-config mode reads the existing qualification report, not the H5 collection.
It never invents acquisition parameters or switches to a tensor fixture.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
LOCAL_CONFIG = ROOT / 'configs/experiments/tii/one_model.local.yaml'
QUALIFICATION = ROOT / 'reports/tii_local_j1/DATASET_QUALIFICATION.csv'
CANDIDATES = ('RM_001_CWRU', 'RM_007_MFPT', 'RM_020_DIRG', 'RM_027_PU', 'RM_031_HUST24')


def constraints(config: dict[str, Any]) -> list[int]:
    """Check the requested run count, not just the number of parallel jobs."""
    required = {
        'pipeline': 'Pipeline_01_Fault_Diagnosis',
        'environment.iterations': 1, 'environment.seed': 0,
        'environment.wandb': False, 'environment.swanlab': False,
        'data.evidence_kind': 'natural_acquisition',
        'data.use_cache': True, 'data.normalization': 'source_rms',
        'data.source_batch_size': 32, 'data.seed': 0,
        'model.type': 'ISFM', 'model.name': 'M_01_ISFM',
        'model.embedding': 'SupportConditionedTokenizer',
        'model.token_organization': 'support',
        'model.backbone': 'B_04_Dlinear', 'model.task_head': 'H_01_Linear_cla',
        'task.type': 'DG', 'task.name': 'tii_joint',
        'task.sampling_seed': 0, 'task.lambda_common': 0, 'task.lambda_private': 0,
        'trainer.num_epochs': 1, 'trainer.devices': 1, 'trainer.device': 'cuda',
        'trainer.test_after_fit': False, 'trainer.early_stopping': False,
    }
    for key, expected in required.items():
        value: Any = config
        for part in key.split('.'):
            value = value.get(part) if isinstance(value, dict) else None
        # In particular, a boolean must not pass as an integer seed/count.
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f'{key}: expected {expected!r}, got {value!r}')
    sources = config['task'].get('source_system_ids')
    if (not isinstance(sources, list) or not 2 <= len(sources) <= 5
            or any(type(x) is not int or x < 0 for x in sources)
            or len(set(sources)) != len(sources)):
        raise ValueError('source_system_ids must contain 2..5 distinct integer dataset IDs')
    # This native field selects source metadata; it is NOT a held-out-target loop.
    if config['task'].get('target_system_id') != sources:
        raise ValueError('native target_system_id must equal source_system_ids; no extra datasets')
    rounds = config['data'].get('rounds')
    if type(rounds) is not int or not 1 <= rounds <= 10000:
        raise ValueError('data.rounds must be an explicit integer in [1, 10000]')
    if config['model'].get('pretrained') or config['model'].get('weights_path'):
        raise ValueError('this single-model run starts from scratch, not external pretrained weights')
    return sources


def qualification_rows(path: Path) -> list[dict[str, str]]:
    # The retained report stores whole-record ID lists in cells exceeding csv's
    # 128 KiB default. Use the actual file size, without truncating those lists.
    previous_limit = csv.field_size_limit()
    csv.field_size_limit(max(previous_limit, path.stat().st_size))
    try:
        with path.open(newline='', encoding='utf-8') as stream:
            reader = csv.DictReader(stream)
            if not {'dataset', 'dataset_id', 'file', 'eligible', 'exclusion_reason'} <= set(reader.fieldnames or []):
                raise ValueError(f'{path}: expected the existing dataset qualification columns')
            return list(reader)
    finally:
        csv.field_size_limit(previous_limit)


def require_sources(config: dict[str, Any], sources: list[int]) -> list[dict[str, str]]:
    """Reuse the existing admission decision; do not recompute or waive it."""
    data = config['data']
    rows = qualification_rows(Path(data['qualification_file']))
    selected = []
    for source in sources:
        matches = [r for r in rows if r['dataset_id'] == str(source)]
        if len(matches) != 1:
            raise ValueError(f'dataset {source}: expected one qualification row, got {len(matches)}')
        row = matches[0]
        if row['eligible'].strip().lower() != 'true':
            raise ValueError(f"dataset {source} ({row['dataset']}) is ineligible: {row['exclusion_reason']}")
        selected.append(row)
    # Two label spaces in the same container (e.g. SEU) are not two sources here.
    if len({r['file'] for r in selected}) != len(selected):
        raise ValueError('selected dataset IDs share an H5 container; independent sources not established')
    root = Path(data['data_dir']).resolve()
    needed = [root / data['metadata_file'], Path(data['record_inventory'])]
    needed += [root / r['file'] for r in selected]
    for path in needed:
        if not path.is_file():
            raise FileNotFoundError(path)
    cache = Path(data['cache_dir']).resolve()
    if cache == root or root in cache.parents:
        raise ValueError('cache_dir must be outside the immutable source directory')
    return selected


def current_blocker() -> dict[str, Any]:
    """Report only the five original candidates, without inspecting raw signals."""
    rows = qualification_rows(QUALIFICATION)
    selected = [r for name in CANDIDATES for r in rows if r['dataset'] == name]
    return {
        'status': 'blocked', 'native_invocations': 0,
        'reason': f'No reviewed runtime configuration at {LOCAL_CONFIG}',
        'qualification_file': str(QUALIFICATION),
        'candidates': [{k: r[k] for k in ('dataset', 'dataset_id', 'eligible', 'exclusion_reason')}
                       for r in selected],
        'next_input': 'A native YAML with 2..5 qualified sources, reviewed record inventory and physical parameters. Do not generate fixture data.',
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Reviewed native YAML; default: one_model.local.yaml')
    parser.add_argument('--output', type=Path, default=Path('results/tii_one_model'), help='New run directory; never overwrite')
    parser.add_argument('--check-only', action='store_true', help='Check config/admission/files only; no waveform read or training')
    args = parser.parse_args(argv)
    config_path = (args.config or LOCAL_CONFIG).expanduser().resolve()
    output = args.output.expanduser().resolve()
    os.chdir(ROOT)  # Native relative data/config paths have one documented meaning.
    if args.config is None and not config_path.is_file():
        print(json.dumps(current_blocker(), ensure_ascii=False, indent=2))
        return 2

    from phmfactory.config import analyze_config
    analysis = analyze_config(config_path)
    config = analysis.runtime_config()
    sources = constraints(config)
    selected = require_sources(config, sources)
    data_root = Path(config['data']['data_dir']).resolve()
    if output == data_root or data_root in output.parents or output in data_root.parents:
        raise ValueError('output must neither contain nor lie inside the original data directory')
    summary = {
        'status': 'input_check_passed', 'native_invocations': 0,
        'datasets': [r['dataset'] for r in selected], 'source_system_ids': sources,
        'models_requested': 1, 'seed': 0, 'rounds': config['data']['rounds'],
        'scope': 'single shared support model; source validation only; not J2 or a transfer effect',
        'waveforms_checked': False,
    }
    if args.check_only:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0

    # The local launcher is explicitly for physical GPU0, not visible device 0
    # after an unknown caller remapping. No CPU/DDP fallback is exposed.
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '0'):
        raise ValueError('this launcher requests physical GPU0; remove the conflicting CUDA_VISIBLE_DEVICES mapping')
    output.mkdir(parents=True, exist_ok=False)
    config['environment']['output_dir'] = str(output / 'native')
    import yaml
    snapshot = output / 'runtime.yaml'
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding='utf-8')
    command = [sys.executable, '-m', 'phmfactory', '--config', str(snapshot)]
    env = dict(os.environ, CUDA_DEVICE_ORDER='PCI_BUS_ID', CUDA_VISIBLE_DEVICES='0')
    env.pop('PYTHONPATH', None)
    summary.update(status='failed_or_interrupted', native_invocations=1, command=command)
    started = time.monotonic()
    try:
        with (output / 'run.log').open('w', encoding='utf-8') as log:
            result = subprocess.run(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
        summary['exit_code'] = result.returncode
        if result.returncode != 0:
            print(f"Training failed; retained log: {output / 'run.log'}", file=sys.stderr)
            return result.returncode if result.returncode > 0 else 1
        # Consume the native result path, not a newest-checkpoint filesystem guess.
        lines = (output / 'run.log').read_text(encoding='utf-8').splitlines()
        checkpoints = [x.split('=', 1)[1] for x in lines if x.startswith('best_checkpoint=')]
        if len(checkpoints) != 1 or not Path(checkpoints[0]).is_file():
            raise ValueError('native run did not return exactly one existing best_checkpoint')
        summary.update(status='completed_training_only', best_checkpoint=checkpoints[0])
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    finally:
        summary['wall_seconds'] = time.monotonic() - started
        (output / 'run.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    raise SystemExit(main())
