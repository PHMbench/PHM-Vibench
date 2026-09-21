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
    # Native runtime applies uppercase environment fields after process startup.
    # Forbid YAML overrides of the physical-device mapping owned by this command.
    for key in ('CUDA_VISIBLE_DEVICES', 'CUDA_DEVICE_ORDER'):
        if key in config['environment']:
            raise ValueError(f'environment.{key} must be absent; this command owns the GPU0 mapping')
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


def acceptance_constraints(config):
    """First real acceptance is one 20-round fit, not two successive fits."""
    constraints(config)
    for section, key, expected in (
        ('data', 'rounds', 20), ('trainer', 'val_check_interval', 10),
        ('trainer', 'num_sanity_val_steps', 0),
    ):
        value = config[section].get(key)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f'acceptance requires {section}.{key}={expected!r}, got {value!r}')


def _gpu0():
    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '0'):
        raise ValueError('acceptance explicitly requests physical GPU0')
    os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable; no CPU or fixture substitution')
    return torch


def _save_report(output, report):
    (output / 'run.json').write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def _expected_source_windows(factory):
    import pandas as pd
    metadata = factory.get_metadata()
    rows = [dict(row, true_label=int(metadata[row['file_id']]['Label'])) for row in factory.window_inventory]
    return pd.DataFrame(rows)


def _finish_analysis(output, report):
    """Complete saved-artifact checks without acquiring data or loading a model."""
    import math
    from src.task_factory.Components.tii_evaluation import analyze_source_acceptance
    if not report.get('restore', {}).get('rms_equal'):
        raise ValueError('successful checkpoint/RMS restoration must precede acceptance')
    score = report.get('native_selection_nll')
    if not isinstance(score, (int, float)) or not math.isfinite(score):
        raise ValueError('missing finite native checkpoint selection score')
    analysis = analyze_source_acceptance(output)
    if abs(score-analysis['source_validation_nll']) > 1e-8:
        raise ValueError('offline source NLL disagrees with the native selection score')
    if 'error' in report:
        report['recovered_export_error'] = report.pop('error')
    report.update(analysis=analysis, selection_score_recomputed=True,
                  status='completed_source_acceptance_not_transfer')
    _save_report(output, report)


def _export_selected(context, output, report):
    """Only inference through the selected native Task; no optimizer or head fit."""
    import numpy as np
    import torch
    task, factory = context.task, context.data_factory
    task.eval()
    checkpoint = report['best_checkpoint']
    saved = torch.load(checkpoint, map_location=task.device, weights_only=False)
    if int(saved['global_step']) not in (10, 20):
        raise ValueError('selected checkpoint is outside this acceptance budget')
    report['selected_global_step'] = int(saved['global_step'])
    scores = [value['best_model_score'] for value in saved['callbacks'].values()
              if isinstance(value, dict) and value.get('monitor') == 'val_group_nll']
    if len(scores) != 1:
        raise ValueError('checkpoint must contain exactly one native source selection score')
    report['native_selection_nll'] = float(scores[0])
    before_rms = task.network.embedding.source_rms.detach().cpu().clone()
    if before_rms.item() != factory.source_rms:
        raise ValueError('source RMS in selected encoder disagrees with the native source fit')
    # Preserve full predictions even if a later restore/mask check fails.
    full = task.source_validation_predictions(factory, checkpoint)
    full.to_csv(output/'source_validation_predictions.csv', index=False)
    task.load_state_dict(saved['state_dict'], strict=True)
    restored = task.source_validation_predictions(factory, checkpoint)
    key = ['dataset', 'recording_id', 'channel', 'window_start', 'window_end']
    if not full[key].equals(restored[key]):
        raise ValueError('checkpoint restore changed the predicted population')
    left = [np.asarray(json.loads(x), dtype=float) for x in full.logits]
    right = [np.asarray(json.loads(x), dtype=float) for x in restored.logits]
    delta = max(float(np.max(np.abs(a-b))) for a, b in zip(left, right))
    close = all(np.allclose(a, b, rtol=1e-6, atol=1e-7) for a, b in zip(left, right))
    if not close or not torch.equal(before_rms, task.network.embedding.source_rms.detach().cpu()):
        raise ValueError('selected-checkpoint restore changed logits or normalization')
    report['restore'] = dict(max_abs_logit_delta=delta, rtol=1e-6, atol=1e-7, rms_equal=True)
    masked = task.source_validation_predictions(factory, checkpoint, mask_increment=True)
    masked.to_csv(output/'source_validation_masked_predictions.csv', index=False)
    report['export_completed'] = True
    _save_report(output, report)
    _finish_analysis(output, report)


def execute_acceptance(config, output):
    """Use the existing classification lifecycle once; capture its live context."""
    import contextlib
    from types import SimpleNamespace
    import traceback
    import yaml
    output.mkdir(parents=True, exist_ok=False)
    config['environment']['output_dir'] = str(output/'native')
    config['task']['acceptance_audit'] = True
    config['task']['acceptance_output'] = str(output/'audit')
    snapshot = output/'runtime.yaml'
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    report = dict(status='failed_or_interrupted', mode='one_model_acceptance_20',
                  native_invocations=0, model_fits_completed=0, source_system_ids=config['task']['source_system_ids'],
                  seed=0, rounds=20, command=sys.orig_argv, industrial_transfer_delta=None, pooling_interaction=None)
    started = time.monotonic()
    _save_report(output, report)
    try:
        torch = _gpu0()
        torch.cuda.reset_peak_memory_stats(0)
        from phmfactory.config import analyze_config
        from src.runtime.classification import ClassificationHooks, run_classification_pipeline
        analysis = analyze_config(snapshot)

        class Capture(ClassificationHooks):
            context = None

            def after_stack_built(self, context):
                if self.context is not None or context.task.network is not context.model:
                    raise ValueError('acceptance requires exactly one shared native model')
                self.context = context
                if context.trainer.accumulate_grad_batches != 1 or context.trainer.precision != '32-true':
                    raise ValueError('acceptance requires native accumulation=1 and 32-true precision')
                expected = _expected_source_windows(context.data_factory)
                expected.to_csv(output/'expected_source_windows.csv', index=False)
                classes = {str(s): list(range(context.task.network.task_head.mutiple_fc[str(s)].out_features))
                           for s in context.task.sources}
                (output/'local_class_map.json').write_text(json.dumps(classes, indent=2)+'\n')

        capture = Capture()
        with (output/'run.log').open('w', buffering=1) as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
            report['native_invocations'] = 1
            result = run_classification_pipeline(SimpleNamespace(
                config_path=str(snapshot), compiled_run_spec=analysis, resolved_pipeline=analysis.pipeline), hooks=capture)
            if result['status'] != 'succeeded' or len(result['best_checkpoints']) != 1:
                raise ValueError('native lifecycle did not return one selected checkpoint')
            context = capture.context
            if context is None or context.trainer.global_step != 20:
                raise ValueError('native lifecycle did not complete exactly 20 shared updates')
            report.update(model_fits_completed=1, best_checkpoint=result['best_checkpoints'][0],
                          native_result=result, status='training_completed_export_pending',
                          parameter_count=sum(p.numel() for p in context.task.network.parameters()),
                          training_wall_seconds=time.monotonic()-started,
                          training_peak_cuda_bytes=torch.cuda.max_memory_allocated(0))
            _save_report(output, report)
            context.task.to('cuda:0')
            _export_selected(context, output, report)
        return 0
    finally:
        error = sys.exc_info()[1]
        if error is not None:
            report['status'] = 'export_or_analysis_failed' if report['model_fits_completed'] else 'failed_or_interrupted'
            report['error'] = repr(error)
            with (output/'run.log').open('a') as log:
                traceback.print_exception(*sys.exc_info(), file=log)
        report['wall_seconds_including_export'] = time.monotonic()-started
        _save_report(output, report)


def export_completed_acceptance(output):
    """Recover only export from a completed fit; never repeat Trainer.fit."""
    from types import SimpleNamespace
    import pandas as pd
    report = json.loads((output/'run.json').read_text())
    if report.get('mode') != 'one_model_acceptance_20' or report.get('model_fits_completed') != 1:
        raise ValueError('export-only requires this acceptance run with one completed fit')
    # Saved predictions never reopen H5 or repeat inference for a plotting failure.
    if report.get('export_completed'):
        try:
            _finish_analysis(output, report)
        except Exception as exc:
            report.update(status='export_or_analysis_failed', error=repr(exc))
            _save_report(output, report)
            raise
        print('Saved acceptance artifacts validated; no data reload, inference or fit.')
        return 0
    torch = _gpu0()
    from phmfactory.config import analyze_config
    from src.data_factory import build_data
    from src.model_factory import build_model
    from src.task_factory import build_task
    config = analyze_config(output/'runtime.yaml').runtime_config()
    acceptance_constraints(config)
    require_sources(config, config['task']['source_system_ids'])
    args = {k: SimpleNamespace(**v) for k, v in config.items() if isinstance(v, dict)}
    factory = build_data(args['data'], args['task'])
    current = _expected_source_windows(factory).astype(str)
    expected = pd.read_csv(output/'expected_source_windows.csv', dtype=str)
    pd.testing.assert_frame_equal(current, expected, check_dtype=False)
    args['model'].source_rms = factory.source_rms
    network = build_model(args['model'], metadata=factory.get_metadata())
    task = build_task(args_task=args['task'], network=network, args_data=args['data'],
                      args_model=args['model'], args_trainer=args['trainer'],
                      args_environment=args['environment'], metadata=factory.get_metadata())
    saved = torch.load(report['best_checkpoint'], map_location='cpu', weights_only=False)
    if factory.source_rms != saved['hyper_parameters']['model']['source_rms']:
        raise ValueError('source RMS differs from the completed fit; refuse export')
    task.load_state_dict(saved['state_dict'], strict=True)
    try:
        task.to('cuda:0')
        _export_selected(SimpleNamespace(task=task, data_factory=factory), output, report)
    except Exception as exc:
        report.update(status='export_or_analysis_failed', error=repr(exc))
        _save_report(output, report)
        raise
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Reviewed native YAML; default: one_model.local.yaml')
    parser.add_argument('--output', type=Path, default=Path('results/tii_one_model'), help='New run directory; never overwrite')
    parser.add_argument('--check-only', action='store_true', help='Check config/admission/files only; no waveform read or training')
    parser.add_argument('--acceptance', action='store_true', help='One real 20-round fit, native audit, restore and source export')
    parser.add_argument('--export-only', action='store_true', help='Recover a failed export from one completed acceptance; never fit')
    parser.add_argument('--analyze-only', action='store_true', help='Read saved source artifacts only; no inference or training')
    args = parser.parse_args(argv)
    if sum((args.acceptance, args.export_only, args.analyze_only)) > 1:
        parser.error('choose one of acceptance, export-only, analyze-only')
    if (args.export_only or args.analyze_only) and (args.config or args.check_only):
        parser.error('artifact recovery uses only --output, not a new config')
    if args.export_only:
        output = args.output.expanduser().resolve()
        os.chdir(ROOT)
        return export_completed_acceptance(output)
    if args.analyze_only:
        from src.task_factory.Components.tii_evaluation import analyze_source_acceptance
        print(json.dumps(analyze_source_acceptance(args.output.expanduser().resolve()), indent=2))
        return 0
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
    if args.acceptance:
        acceptance_constraints(config)
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

    if os.environ.get('CUDA_VISIBLE_DEVICES') not in (None, '', '0'):
        raise ValueError('this launcher requests physical GPU0; remove the conflicting CUDA_VISIBLE_DEVICES mapping')
    if args.acceptance:
        return execute_acceptance(config, output)
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
