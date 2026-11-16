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
QUALIFICATION = Path(os.environ.get('TII_QUALIFICATION_CSV')
                     or ROOT / 'reports/tii_local_j1/DATASET_QUALIFICATION.csv')
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
    if not QUALIFICATION.is_file():
        return {
            'status': 'qualification_records_relocated', 'native_invocations': 0,
            'reason': f'Qualification records are not stored in this repository; {QUALIFICATION} is absent.',
            'qualification_file': str(QUALIFICATION), 'candidates': [],
            'next_input': 'Set TII_QUALIFICATION_CSV to the retained DATASET_QUALIFICATION.csv in the paper '
                          'repository, or pass --config whose data.qualification_file points there. '
                          'Do not regenerate fixture data.',
        }
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


def fit95_constraints(config):
    """User-requested fitting diagnostic; not the confirmatory S/U experiment."""
    constraints(config)
    for section, key, expected in (
        ('data', 'rounds', 10000), ('trainer', 'val_check_interval', 100),
        ('trainer', 'num_sanity_val_steps', 0),
    ):
        value = config[section].get(key)
        if type(value) is not type(expected) or value != expected:
            raise ValueError(f'fit95 requires {section}.{key}={expected!r}, got {value!r}')


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
    status = ('completed_source_fit95_diagnostics_not_transfer'
              if report.get('mode') == 'one_model_fit95' else 'completed_source_acceptance_not_transfer')
    report.update(analysis=analysis, selection_score_recomputed=True, status=status)
    _save_report(output, report)


def _export_selected(context, output, report):
    """Only inference through the selected native Task; no optimizer or head fit."""
    import numpy as np
    import torch
    task, factory = context.task, context.data_factory
    task.eval()
    checkpoint = report['best_checkpoint']
    saved = torch.load(checkpoint, map_location=task.device, weights_only=False)
    interval = 100 if report.get('mode') == 'one_model_fit95' else 10
    limit = 10000 if report.get('mode') == 'one_model_fit95' else 20
    if int(saved['global_step']) not in range(interval, limit+1, interval):
        raise ValueError('selected checkpoint is outside the declared validation schedule')
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
    from src.task_factory.Components.tii_fit95 import retain_predictions
    full = task.source_validation_predictions(factory, checkpoint)
    retain_predictions(output/'source_validation_predictions.csv', full)
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
    retain_predictions(output/'source_validation_masked_predictions.csv', masked)
    if report.get('mode') == 'one_model_fit95':
        train = task.source_predictions(factory, checkpoint, role='source_train')
        retain_predictions(output/'source_training_predictions.csv', train)
    report['export_completed'] = True
    _save_report(output, report)
    _finish_analysis(output, report)


def execute_acceptance(config, output, *, fit95=False):
    """Use the existing classification lifecycle once; capture its live context."""
    import contextlib
    from types import SimpleNamespace
    import traceback
    import yaml
    output.mkdir(parents=True, exist_ok=False)
    config['environment']['output_dir'] = str(output/'native')
    config['task']['acceptance_audit'] = True
    config['task']['acceptance_output'] = str(output/'audit')
    # First 20 real updates are audited; later fitting is not charged repeated VJPs.
    config['task']['acceptance_audit_rounds'] = 20
    snapshot = output/'runtime.yaml'
    snapshot.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True))
    report = dict(status='failed_or_interrupted',
                  mode='one_model_fit95' if fit95 else 'one_model_acceptance_20',
                  native_invocations=0, model_fits_completed=0, source_system_ids=config['task']['source_system_ids'],
                  seed=0, rounds=config['data']['rounds'], command=sys.orig_argv,
                  industrial_transfer_delta=None, pooling_interaction=None)
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
                if fit95:
                    from src.task_factory.Components.tii_fit95 import SourceFit95
                    context.trainer.callbacks.append(SourceFit95(context.data_factory, output))
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
            if context is None or context.trainer.global_step != config['data']['rounds']:
                raise ValueError('native lifecycle did not complete the predeclared shared-update budget')
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
    if report.get('mode') not in ('one_model_acceptance_20', 'one_model_fit95') or report.get('model_fits_completed') != 1:
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
    if report['mode'] == 'one_model_fit95':
        fit95_constraints(config)
    else:
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


def frozen_features_with_common_tokens(task, windows, *, effective_rate_hz):
    """Observe a frozen tokenizer's first branch without changing its forward path."""
    import torch
    if task.network.args_m.embedding != 'SupportConditionedTokenizer':
        raise ValueError('common-token export requires the matched S/U tokenizer')
    observed = []

    def capture(_module, _inputs, output):
        if output.ndim != 3 or not torch.isfinite(output).all():
            raise ValueError('common branch must return finite [windows,K,D] tokens')
        observed.append(output.detach().cpu())

    hook = task.network.embedding.common_branch.register_forward_hook(capture)
    try:
        pooled = task.frozen_features(windows, effective_rate_hz=effective_rate_hz)
    finally:
        hook.remove()
    if not observed or sum(len(batch) for batch in observed) != len(pooled):
        raise ValueError('common-token export must cover each frozen feature exactly once')
    return pooled, torch.cat(observed, dim=0)


def export_frozen_episode(config_path, checkpoint, metadata_file, inventory_file, data_dir,
                          feature_spec, selection_file, output, *, fisher=False):
    """Export an explicitly isolated frozen or deterministic-feature episode."""
    from types import SimpleNamespace
    import numpy as np
    import torch
    from phmfactory.config import analyze_config
    from src.data_factory import build_data
    from src.data_factory.tii_data import load_tii_target_data
    from src.data_factory.data_utils import read_metadata_table
    from src.model_factory import build_model
    from src.model_factory.ISFM.physical_features import squared_envelope_spectrum
    from src.task_factory import build_task
    from src.task_factory.Components.tii_target_head import select_source_regularization

    spec = json.loads(feature_spec.read_text(encoding='utf-8'))
    if spec['kind'] not in ('source_inner', 'target'):
        raise ValueError('feature spec must identify a source_inner or target episode')
    save_common = spec.get('save_common_tokens', False)
    if not isinstance(save_common, bool):
        raise ValueError('save_common_tokens must be a boolean')
    if save_common and checkpoint is None:
        raise ValueError('common-token export requires a frozen checkpoint')
    if fisher and (spec.get('representation') != 'official_fisher' or save_common):
        raise ValueError('FISHER export requires its declared representation and cannot export S/U branches')
    config = analyze_config(config_path).runtime_config()
    args = {k: SimpleNamespace(**v) for k, v in config.items() if isinstance(v, dict)}
    sources = list(args['task'].source_system_ids)
    episode_id = spec['episode_dataset']
    outer_target = spec['outer_target']
    if episode_id in sources or outer_target in sources:
        raise ValueError('episode and outer target must be excluded from encoder sources')
    if spec['kind'] == 'source_inner':
        if selection_file or episode_id == outer_target:
            raise ValueError('source-inner episode must not use outer-target selection or records')
    else:
        if episode_id != outer_target or selection_file is None:
            raise ValueError('target episode requires a previously saved source-only selection')
        selection = json.loads(selection_file.read_text(encoding='utf-8'))
        select_source_regularization(selection, encoder_sources=sources, target_dataset=outer_target,
            source_tasks=spec['source_tasks'], seeds=spec['seeds'], selection_arms=spec['selection_arms'],
            source_pool=spec['source_pool'])
    factory = build_data(args['data'], args['task'])
    saved = None
    fisher_geometry = None
    if fisher:
        from src.model_factory.ISFM.third_party.fisher.fisher import FISHER
        from src.model_factory.ISFM.third_party.fisher.preprocessing import log_spectrogram

        checkpoint_kind = spec.get('fisher_checkpoint_kind')
        if checkpoint_kind not in ('random_fixture', 'external_pretrained'):
            raise ValueError('FISHER checkpoint kind must be random_fixture or external_pretrained')
        if spec['kind'] == 'target' and any(
            row.get('fisher_checkpoint') != str(checkpoint)
            or row.get('fisher_checkpoint_kind') != checkpoint_kind
            for mode in ('raw', 'source_rms') for row in selection[mode]
        ):
            raise ValueError('FISHER source selection must use the same declared frozen checkpoint path')
        exclusions = spec.get('pretraining_excluded_dataset_ids', [])
        if checkpoint_kind == 'external_pretrained':
            if not spec.get('pretraining_provenance') or not {episode_id, outer_target} <= set(exclusions):
                raise ValueError('external FISHER requires documented pretraining provenance and excluded episode/target IDs')
            if spec['kind'] == 'target' and any(
                row.get('fisher_pretraining_provenance') != spec['pretraining_provenance']
                or not {row['task'], outer_target} <= set(row.get('fisher_pretraining_excluded_dataset_ids', []))
                for mode in ('raw', 'source_rms') for row in selection[mode]
            ):
                raise ValueError('FISHER source selection lacks matching pretraining provenance and exclusions')
        model = FISHER.from_pretrained(str(checkpoint)).cpu().eval()

        def encode(windows, rate, *, capture_common=False):
            nonlocal fisher_geometry
            result = []
            with torch.inference_mode():
                for start in range(0, len(windows['x']), args['data'].source_batch_size):
                    stop = start + args['data'].source_batch_size
                    spec_input, geometry = log_spectrogram(
                        windows['x'][start:stop], windows['incremental'][start:stop],
                        windows['availability'][start:stop], source_rms=factory.source_rms,
                        sample_rate_hz=args['data'].grid_rate_hz, band_width=model.band_width,
                        patch_size=model.patch_size, allow_time_frame_drop=spec['allow_time_frame_drop'],
                        allow_frequency_bin_drop=spec['allow_frequency_bin_drop'])
                    if fisher_geometry is not None and geometry != fisher_geometry:
                        raise ValueError('FISHER source and episode preprocessing geometries differ')
                    fisher_geometry = geometry
                    result.append(model.extract_features(spec_input).cpu().numpy())
            return np.concatenate(result), None
    elif checkpoint is not None:
        saved = torch.load(checkpoint, map_location='cpu', weights_only=False)
        for section in ('data', 'task', 'model'):
            for key, value in config[section].items():
                if section == 'model' and key == 'source_rms':
                    continue
                if saved['hyper_parameters'][section].get(key) != value:
                    raise ValueError(f'checkpoint {section}.{key} disagrees with the supplied source configuration')
        if saved['hyper_parameters']['model']['source_rms'] != factory.source_rms:
            raise ValueError('restored checkpoint and source-only signal RMS disagree')
        args['model'].source_rms = factory.source_rms
        network = build_model(args['model'], metadata=factory.get_metadata())
        task = build_task(args_task=args['task'], network=network, args_data=args['data'],
                          args_model=args['model'], args_trainer=args['trainer'],
                          args_environment=args['environment'], metadata=factory.get_metadata())
        task.load_state_dict(saved['state_dict'], strict=True)
        task.cpu().eval()

        def encode(windows, rate, *, capture_common=False):
            if capture_common:
                pooled, common = frozen_features_with_common_tokens(task, windows, effective_rate_hz=rate)
                return pooled.numpy(), common.numpy()
            return task.frozen_features(windows, effective_rate_hz=rate).numpy(), None
    else:
        if spec.get('representation') != 'squared_envelope_spectrum':
            raise ValueError('checkpoint-free export requires an explicit fixed envelope spectrum')

        def encode(windows, rate, *, capture_common=False):
            with torch.inference_mode():
                return squared_envelope_spectrum(windows['x'], windows['incremental'], windows['availability'],
                    source_rms=factory.source_rms, sample_rate_hz=args['data'].grid_rate_hz,
                    frequency_band_hz=spec['envelope_band_hz']).cpu().numpy(), None
    source_features, source_roles, source_datasets, source_groups = [], [], [], []
    for source, windows in sorted(factory.train_dataset.sources.items()):
        rates = {float(factory.get_metadata()[fid]['Sample_rate']) for fid in windows['file_id']}
        if len(rates) != 1:
            raise ValueError('source feature export requires homogeneous physical sample rate')
        source_features.append(encode(windows, next(iter(rates)))[0])
        source_roles.extend(windows['role'])
        source_datasets.extend([source] * len(windows['y']))
        source_groups.extend(windows['group'])
    episode_data = SimpleNamespace(**vars(args['data']))
    episode_data.data_dir = str(data_dir)
    collections, inventory, episode = load_tii_target_data(
        episode_data, metadata_file, inventory_file, target_dataset_id=episode_id,
        source_dataset_ids=sources, source_group_ids=[r['group'] for r in factory.window_inventory])
    metadata = read_metadata_table(metadata_file).set_index('Id')
    target_features, target_common, labels, groups, recording_ids, roles = [], [], [], [], [], []
    for role in ('support', 'query'):
        windows = collections[role][episode_id]
        rates = {float(metadata.loc[fid, 'Sample_rate']) for fid in windows['file_id']}
        if len(rates) != 1:
            raise ValueError('episode feature export requires homogeneous physical sample rate')
        pooled, common = encode(windows, next(iter(rates)), capture_common=save_common)
        target_features.append(pooled)
        if save_common:
            target_common.append(common)
        labels.extend(windows['y'].tolist())
        groups.extend(windows['group'])
        recording_ids.extend(windows['recording_id'])
        roles.extend(windows['role'])
    output.mkdir(parents=True, exist_ok=False)
    extra = {'target_common_tokens': np.concatenate(target_common)} if save_common else {}
    if fisher:
        extra.update(fisher_checkpoint=np.asarray(str(checkpoint)),
                     fisher_checkpoint_kind=np.asarray(checkpoint_kind),
                     fisher_pretraining_provenance=np.asarray(spec.get('pretraining_provenance') or ''),
                     fisher_pretraining_excluded_dataset_ids=np.asarray(spec.get('pretraining_excluded_dataset_ids', [])))
    np.savez_compressed(output/'features.npz', source_features=np.concatenate(source_features),
        source_roles=source_roles, source_datasets=source_datasets, source_groups=source_groups,
        target_features=np.concatenate(target_features), target_labels=labels, target_groups=groups,
        target_recording_ids=recording_ids, target_roles=roles, target_datasets=[episode_id]*len(labels),
        **extra)
    (output/'episode.json').write_text(json.dumps(episode, indent=2)+'\n', encoding='utf-8')
    report = dict(kind=spec['kind'], checkpoint=str(checkpoint) if checkpoint else None,
                  representation=spec.get('representation', 'restored_encoder'), source_signal_rms=factory.source_rms,
                  source_datasets=sources, episode_dataset=episode_id, source_train_windows=len(source_roles),
                  episode_windows=len(inventory), checkpoint_global_step=int(saved['global_step']) if saved else (None if fisher else 0),
                  episode_features_used_for_selection_by_this_export=False,
                  target_common_tokens_before_backbone=save_common)
    if fisher:
        report.update(fisher_checkpoint_kind=checkpoint_kind, fisher_preprocessing=fisher_geometry,
                      pretraining_provenance=spec.get('pretraining_provenance'),
                      pretraining_excluded_dataset_ids=spec.get('pretraining_excluded_dataset_ids', []),
                      encoder_fit_during_export=False,
                      yaml_model_section_used=False,
                      provenance_note='caller-declared pretraining exclusions are not independently verified')
    (output/'features.json').write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(json.dumps(report, indent=2))
    return 0


def score_saved_source_episode(feature_file, episode_file, spec_file, output):
    """Select candidate alphas from an excluded source, before outer-target access."""
    import numpy as np
    from src.task_factory.Components.tii_target_head import score_source_episode
    episode = json.loads(episode_file.read_text(encoding='utf-8'))
    spec = json.loads(spec_file.read_text(encoding='utf-8'))
    with np.load(feature_file, allow_pickle=False) as saved:
        rows = score_source_episode(
            episode, saved['target_features'], saved['target_labels'], saved['target_groups'],
            saved['target_recording_ids'], saved['target_roles'], saved['source_features'],
            saved['source_roles'], saved['source_datasets'], saved['source_groups'],
            episode_datasets=saved['target_datasets'], encoder_sources=spec['encoder_sources'],
            pseudo_target=spec['pseudo_target'], outer_target=spec['outer_target'],
            seed=spec['seed'], arm=spec['arm'])
        if 'fisher_checkpoint' in saved.files:
            for mode in ('raw', 'source_rms'):
                for row in rows[mode]:
                    row.update(fisher_checkpoint=str(saved['fisher_checkpoint'].item()),
                               fisher_checkpoint_kind=str(saved['fisher_checkpoint_kind'].item()),
                               fisher_pretraining_provenance=str(saved['fisher_pretraining_provenance'].item()),
                               fisher_pretraining_excluded_dataset_ids=saved['fisher_pretraining_excluded_dataset_ids'].tolist())
    output.mkdir(parents=True, exist_ok=False)
    (output/'source_scores.json').write_text(json.dumps(rows, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(rows, indent=2, allow_nan=False))
    return 0


def evaluate_saved_target_features(feature_file, episode_file, selection_file, spec_file, output):
    """Evaluate saved frozen features; this path never reads H5 or fits an encoder."""
    import numpy as np
    from src.task_factory.Components.tii_target_head import evaluate_target_readouts

    episode = json.loads(episode_file.read_text(encoding='utf-8'))
    selection = json.loads(selection_file.read_text(encoding='utf-8'))
    spec = json.loads(spec_file.read_text(encoding='utf-8'))
    with np.load(feature_file, allow_pickle=False) as saved:
        fisher_metadata = None
        has_fisher_selection = any('fisher_checkpoint' in row
            for mode in ('raw', 'source_rms') for row in selection[mode])
        if ('fisher_checkpoint' in saved.files) != has_fisher_selection:
            raise ValueError('FISHER feature and source-selection provenance must agree')
        if has_fisher_selection:
            fisher_metadata = dict(fisher_checkpoint=str(saved['fisher_checkpoint'].item()),
                                   fisher_checkpoint_kind=str(saved['fisher_checkpoint_kind'].item()),
                                   fisher_pretraining_provenance=str(saved['fisher_pretraining_provenance'].item()))
            if any(any(row.get(key) != value for key, value in fisher_metadata.items())
                   for mode in ('raw', 'source_rms') for row in selection[mode]):
                raise ValueError('FISHER source selection and target features declare different checkpoint paths')
        result = evaluate_target_readouts(
            episode, saved['target_features'], saved['target_labels'], saved['target_groups'],
            saved['target_recording_ids'], saved['target_roles'], saved['source_features'],
            saved['source_roles'], saved['source_datasets'], saved['source_groups'],
            episode_datasets=saved['target_datasets'], selection_rows=selection,
            encoder_sources=spec['encoder_sources'], target_dataset=spec['target_dataset'],
            source_tasks=spec['source_tasks'], seeds=spec['seeds'],
            selection_arms=spec['selection_arms'], source_pool=spec.get('source_pool'))
    output.mkdir(parents=True, exist_ok=False)
    summary = {'source_feature_rms': result['source_feature_rms'],
               'metric_weighting': 'equal physical group, equal window within group', 'modes': {}}
    if fisher_metadata:
        summary.update(fisher_metadata)
        summary['fisher_scope'] = ('random fixture; software integration only, not baseline performance'
                                   if fisher_metadata['fisher_checkpoint_kind'] == 'random_fixture'
                                   else 'external pretrained checkpoint; caller provenance not independently verified')
    arrays = {}
    for mode, value in result['modes'].items():
        summary['modes'][mode] = {'alpha': value['alpha'], 'selection_scores': value['selection_scores'],
                                  **value['metrics']}
        for field in ('probabilities', 'labels', 'groups', 'recording_ids'):
            arrays[f'{mode}_{field}'] = value[field]
        arrays[f'{mode}_weight'] = value['head'].weight
        arrays[f'{mode}_bias'] = value['head'].bias
        path = output / f'{mode}_query_predictions.csv'
        with path.open('x', encoding='utf-8', newline='') as stream:
            names = ['recording_id', 'group', 'label', 'prediction'] + [
                f'probability_{i}' for i in range(value['probabilities'].shape[1])]
            writer = csv.writer(stream)
            writer.writerow(names)
            for rid, group, label, probability in zip(value['recording_ids'], value['groups'],
                                                       value['labels'], value['probabilities']):
                writer.writerow([rid, group, int(label), int(np.argmax(probability)), *probability.tolist()])
    np.savez_compressed(output / 'query_predictions.npz', **arrays)
    (output / 'metrics.json').write_text(json.dumps(summary, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, help='Reviewed native YAML; default: one_model.local.yaml')
    parser.add_argument('--output', type=Path, default=Path('results/tii_one_model'), help='New run directory; never overwrite')
    parser.add_argument('--check-only', action='store_true', help='Check config/admission/files only; no waveform read or training')
    parser.add_argument('--acceptance', action='store_true', help='One real 20-round fit, native audit, restore and source export')
    parser.add_argument('--fit95', action='store_true', help='One bounded 10000-update fit; report every-source 95% train target')
    parser.add_argument('--export-only', action='store_true', help='Recover a failed export from one completed acceptance; never fit')
    parser.add_argument('--analyze-only', action='store_true', help='Read saved source artifacts only; no inference or training')
    parser.add_argument('--target-features', type=Path, help='Saved frozen source/target feature arrays (NPZ)')
    parser.add_argument('--target-episode', type=Path, help='Complete fixed target-record episode (JSON)')
    parser.add_argument('--source-selection', type=Path, help='Source-only inner-task alpha scores (JSON)')
    parser.add_argument('--readout-spec', type=Path, help='Declared source, target, seed and arm identities (JSON)')
    parser.add_argument('--output_dir', type=Path, help='New directory for frozen target readout outputs')
    parser.add_argument('--frozen-episode', action='store_true', help='Restore a source fit and export episode features')
    parser.add_argument('--envelope-episode', action='store_true', help='Export fixed envelope-spectrum episode features')
    parser.add_argument('--fisher-episode', action='store_true', help='Export a frozen official FISHER inference episode')
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--episode-metadata', type=Path)
    parser.add_argument('--episode-inventory', type=Path)
    parser.add_argument('--episode-data-dir', type=Path)
    parser.add_argument('--feature-spec', type=Path)
    parser.add_argument('--score-source-episode', action='store_true', help='Score alpha grid on frozen source-only features')
    args = parser.parse_args(argv)
    if sum((args.frozen_episode, args.envelope_episode, args.fisher_episode, args.score_source_episode)) > 1:
        parser.error('choose one episode export or source scoring mode')
    if args.frozen_episode or args.envelope_episode or args.fisher_episode:
        required = (args.config, args.episode_metadata, args.episode_inventory,
                    args.episode_data_dir, args.feature_spec, args.output_dir)
        if (not all(required) or bool(args.checkpoint) != (args.frozen_episode or args.fisher_episode)
                or args.target_features or args.target_episode or args.readout_spec
                or args.check_only or args.acceptance or args.fit95 or args.export_only or args.analyze_only
                or args.output != Path('results/tii_one_model')):
            parser.error('episode export requires its config, episode inputs, spec, --output_dir and applicable checkpoint')
        return export_frozen_episode(args.config.expanduser().resolve(),
            args.checkpoint.expanduser().resolve() if args.checkpoint else None,
            *(p.expanduser().resolve() for p in required[1:5]),
            args.source_selection.expanduser().resolve() if args.source_selection else None,
            args.output_dir.expanduser().resolve(), fisher=args.fisher_episode)
    if args.score_source_episode:
        required = (args.target_features, args.target_episode, args.readout_spec, args.output_dir)
        if (not all(required) or args.source_selection or args.config or args.check_only or args.acceptance
                or args.fit95 or args.export_only or args.analyze_only or args.output != Path('results/tii_one_model')):
            parser.error('source episode scoring requires features, episode, spec and --output_dir')
        return score_saved_source_episode(*(p.expanduser().resolve() for p in required))
    target_arguments = (args.target_features, args.target_episode, args.source_selection,
                        args.readout_spec, args.output_dir)
    if any(target_arguments):
        if (not all(target_arguments) or args.config or args.check_only or args.acceptance or args.fit95
                or args.export_only or args.analyze_only or args.output != Path('results/tii_one_model')):
            parser.error('target feature evaluation requires exactly its four inputs and --output_dir')
        return evaluate_saved_target_features(*(p.expanduser().resolve() for p in target_arguments))
    if sum((args.acceptance, args.fit95, args.export_only, args.analyze_only)) > 1:
        parser.error('choose one of acceptance, fit95, export-only, analyze-only')
    if (args.export_only or args.analyze_only) and (args.config or args.check_only):
        parser.error('artifact recovery uses only --output, not a new config')
    if args.export_only:
        output = args.output.expanduser().resolve()
        os.chdir(ROOT)
        return export_completed_acceptance(output)
    if args.analyze_only:
        from src.task_factory.Components.tii_evaluation import analyze_source_acceptance
        result = analyze_source_acceptance(args.output.expanduser().resolve())
        print(json.dumps(result, indent=2))
        return 3 if result.get('source_fit_target_met') is False else 0
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
    if args.fit95:
        fit95_constraints(config)
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
    if args.acceptance or args.fit95:
        return execute_acceptance(config, output, fit95=args.fit95)
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
