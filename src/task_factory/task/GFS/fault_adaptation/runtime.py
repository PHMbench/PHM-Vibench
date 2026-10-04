"""Run acquisition-group GFS through the single P4 adaptation core.

Source fitting and physical acquisition provenance are supplied explicitly. Target
query labels are opened only after every requested adaptation state is fixed.
"""
from __future__ import annotations
import copy
import csv
import json
import os
import platform
import time
import traceback
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from phmfactory import __version__
from src.task_factory.task.GFS.physical_prior_core import (
    unit, validate_groups, prototype, weights, scores, offset, adapt, crossfit,
)

ARMS = {'E1': ['A2', 'A3', 'A4', 'A6', 'A7'],
        'E2': ['A2', 'A3', 'A4', 'A7'],
        'E3': ['A4', 'A6', 'A7', 'A8', 'A7_prior_only', 'A7_init_only']}


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def integer_array(values: np.ndarray, name: str) -> np.ndarray:
    if values.ndim != 1 or values.dtype.kind not in 'iu':
        raise ValueError(f'{name} must be a one-dimensional integer array; no label/ID coercion is allowed.')
    if values.dtype.kind == 'u' and values.size and int(values.max()) > np.iinfo(np.int64).max:
        raise ValueError(f'{name} exceeds signed int64 identity range; no overflow is allowed.')
    return values


def read_support(path: Path, device: str) -> dict[str, torch.Tensor]:
    with np.load(path, allow_pickle=False) as data:
        result = {}
        for name in ('x', 'y', 'group', 'view', 'metadata', 'raw_start', 'raw_stop'):
            value = data[name]
            if name in {'y', 'group', 'view', 'raw_start', 'raw_stop'}:
                integer_array(value, f'support.{name}')
            elif not np.isfinite(value).all():
                raise ValueError(f'Nonfinite support.{name}.')
            result[name] = torch.as_tensor(value, device=device,
                dtype=torch.long if name in {'y', 'group', 'view', 'raw_start', 'raw_stop'} else torch.float32)
    size = len(result['y'])
    if not size or any(len(v) != size for v in result.values()) or result['metadata'].ndim != 2:
        raise ValueError('Support arrays have incompatible shapes or are empty.')
    if torch.any(result['raw_start'] < 0) or torch.any(result['raw_stop'] <= result['raw_start']):
        raise ValueError('Raw signal intervals must be positive-length half-open sample intervals.')
    for group in result['group'].unique():
        indexes = result['group'] == group
        left = indexes & (result['view'] == 0)
        right = indexes & (result['view'] == 1)
        a, b = result['raw_start'][left], result['raw_stop'][left]
        c, d = result['raw_start'][right], result['raw_stop'][right]
        if torch.any((a[:, None] < d[None, :]) & (c[None, :] < b[:, None])):
            raise ValueError('Cross-view raw signal intervals overlap within a physical acquisition.')
    return result


def resolved_settings(spec: dict, cfg: dict, experiment: str) -> dict:
    settings = copy.deepcopy(spec['adaptation'])
    for field, per_arm in cfg.get('adaptation_overrides', {}).items():
        if field not in {'steps', 'learning_rate', 'regularization'}:
            raise ValueError(f'Unsupported adaptation override: {field}')
        for arm, value in per_arm.items():
            if arm not in {'A2', 'A3', 'A6', 'A7', 'A8'}:
                raise ValueError(f'Unsupported tuning arm: {arm}')
            settings[field][arm] = value
    if experiment == 'E3':
        for field in ('steps', 'learning_rate', 'regularization'):
            if len({settings[field][a] for a in ('A6', 'A7', 'A8')}) != 1:
                raise ValueError(f'Matched E3 A6/A7/A8 must have identical {field}.')
    if not np.isfinite(settings['offset_norm']) or settings['offset_norm'] <= 0:
        raise ValueError('Physical-prior norm must be positive and finite.')
    for arm in ARMS[experiment]:
        family = 'A7' if arm.startswith('A7_') else arm
        step = settings['steps'][family]
        if isinstance(step, bool) or not isinstance(step, int) or step < 0:
            raise ValueError('Adaptation steps must be nonnegative integers.')
        for field in ('learning_rate', 'regularization'):
            if not np.isfinite(settings[field][family]):
                raise ValueError('Adaptation settings must be finite.')
    return settings


def scalar_metrics(logits, y, groups, base_classes, novel_classes, old, z, z0, scale):
    order = list(base_classes) + list(novel_classes)
    truth = torch.tensor([order.index(int(v)) for v in y], device=y.device)
    nb = len(base_classes)
    pred, pred_b = logits.argmax(1), logits[:, :nb].argmax(1)
    is_b = truth < nb
    if not is_b.any() or is_b.all():
        raise ValueError('Generalized query must contain base and novel observations.')
    wb, wn = weights(y[is_b], groups[is_b]), weights(y[~is_b], groups[~is_b])
    ab = float((wb * (pred[is_b] == truth[is_b])).sum())
    an = float((wn * (pred[~is_b] == truth[~is_b])).sum())
    base_only = float((wb * (pred_b[is_b] == truth[is_b])).sum())
    intrusion = float((wb * ((pred_b[is_b] == truth[is_b]) & (pred[is_b] >= nb))).sum())
    w = weights(y, groups)
    nll = float((w * F.cross_entropy(logits, truth, reduction='none')).sum())
    conf = logits.softmax(1).max(1).values
    correct = (pred == truth).float()
    ece = 0.
    for b in range(15):
        ix = (conf >= b / 15) & ((conf < (b + 1) / 15) if b < 14 else (conf <= 1))
        if float(w[ix].sum()) > 0:
            ece += float(torch.abs((w[ix] * (conf[ix] - correct[ix])).sum()))
    a, b = z0[is_b], z[is_b]
    a, b = a - a.mean(0), b - b.mean(0)
    denom = torch.linalg.norm(a.T @ a) * torch.linalg.norm(b.T @ b)
    cka = float(torch.linalg.norm(a.T @ b) ** 2 / denom) if float(denom) > 1e-12 else float('nan')
    return dict(base_acc=ab, novel_acc=an, joint_acc=(len(base_classes)*ab + len(novel_classes)*an)/len(order), harmonic=2*ab*an/(ab+an) if ab+an else 0.,
                base_only_acc=base_only, intrusion=intrusion, nll=nll, ece=ece,
                geometry_drift=1-cka, anchor_flip=float((wb * (pred_b[is_b] != old[is_b])).sum()))


def _execute(cfg: dict, root: Path, out: Path, experiment: str,
             selected_arms: Sequence[str], evaluate_query: bool) -> None:
    def path(value: str) -> Path:
        if not isinstance(value, str) or not value:
            raise ValueError('Bind an explicit source/data path; null is not runnable.')
        p = Path(value).expanduser()
        return p if p.is_absolute() else root / p

    metric_writer = prediction_writer = None
    with (out/'metrics.csv').open('x', newline='') as mf, (out/'predictions.csv').open('x', newline='') as pf:
        for episode_index, ep in enumerate(cfg['episodes']):
            torch.manual_seed(ep['seed'])
            source = path(ep['source'])
            spec = json.loads((source/'source.json').read_text())
            group_ids = spec.get('fitted_or_selected_group_ids')
            if not isinstance(group_ids, list) or not group_ids or any(type(g) is not int for g in group_ids) or len(set(group_ids)) != len(group_ids):
                raise ValueError('Source fitted_or_selected_group_ids must be nonempty unique integer IDs; no coercion is allowed.')
            fitted_folds = spec.get('fitted_or_selected_folds')
            excluded_folds = spec.get('excluded_target_folds')
            for name, values in [('fitted_or_selected_folds', fitted_folds), ('excluded_target_folds', excluded_folds)]:
                if not isinstance(values, list) or not values or any(type(f) is not str or not f for f in values) or len(set(values)) != len(values):
                    raise ValueError(f'Source {name} must be a nonempty unique list of fold names.')
            if set(fitted_folds) & set(excluded_folds):
                raise ValueError('Source fitted and excluded fold declarations contradict each other.')
            if cfg.get('information_regime') != 'synthetic_interface' and spec.get('source_training_seed') != ep['seed']:
                raise ValueError('Episode seed must name the actual independent source fit; reloading a checkpoint is not a new seed.')
            if ep['fold'] not in spec['excluded_target_folds']:
                raise ValueError('Source export does not declare exclusion of this target fold.')
            if not spec['physical_fields'] or not spec['group_definition'] or not spec['source_fit_description']:
                raise ValueError('Source population, physical fields and acquisition definition must be documented.')
            if len(set(spec['base_classes'])) != len(spec['base_classes']) or len(set(ep['novel_classes'])) != len(ep['novel_classes']):
                raise ValueError('Class lists must contain unique labels.')
            if not set(ep['novel_classes']).issubset(set(spec['excluded_class_ids'])):
                raise ValueError('Novel classes were not explicitly excluded from all source fitting/selection.')
            qualification = spec.get('source_qualification', {})
            qualification_flags = ('forward_equivalent', 'prompt_gradient_nonzero', 'nonprompt_gradient_nonzero',
                                   'a3_gradient_nonzero', 'native_state_preserved')
            if cfg.get('information_regime') != 'synthetic_interface' and not all(qualification.get(k) is True for k in qualification_flags):
                raise ValueError('Actual source/export forward and gradient qualification is required.')
            model = torch.jit.load(str(source/'encoder.pt'), map_location=cfg['device']).eval()
            with np.load(source/'constants.npz', allow_pickle=False) as arrays:
                base = torch.as_tensor(arrays['base_anchors'], dtype=torch.float32, device=cfg['device'])
                injection = torch.as_tensor(arrays['physical_injection'], dtype=torch.float32, device=cfg['device'])
            if base.ndim != 2 or not torch.isfinite(base).all() or not torch.isfinite(injection).all():
                raise ValueError('Source constants must be finite matrices.')
            if not torch.allclose(base.norm(dim=1), torch.ones(base.shape[0], device=cfg['device']), atol=1e-5):
                raise ValueError('Base anchors must already be normalized by the admitted source export.')
            bc, nc = spec['base_classes'], ep['novel_classes']
            if set(bc) & set(nc) or len(bc) != len(base):
                raise ValueError('Base/novel ontology or anchor count is inconsistent.')
            support = read_support(path(ep['support']), cfg['device'])
            validate_groups(support['y'], support['group'], support['view'], nc, ep['shots'])
            # No query observations or labels are inspected in preflight/adaptation.
            with np.load(path(ep['query']), allow_pickle=False) as query:
                qgroups = set(map(int, integer_array(query['group'], 'query.group')))
            sgroups = set(map(int, support['group'].tolist()))
            used = set(spec['fitted_or_selected_group_ids'])
            if sgroups & qgroups or (sgroups | qgroups) & used:
                raise ValueError('Physical-group overlap across source, support or query.')
            if evaluate_query and cfg.get('information_regime', 'target_gfs') == 'target_gfs' and cfg.get('adaptation_overrides'):
                selection = cfg.get('source_selection', {})
                if selection.get('information_regime') != 'source_validation':
                    raise ValueError('Target adaptation overrides need the completed source-validation selection record.')
                if ep['fold'] not in selection.get('forbidden_target_folds', []) or not (sgroups | qgroups) <= set(selection.get('forbidden_target_group_ids', [])):
                    raise ValueError('Final target folds/groups were not excluded from the recorded tuning population.')
            settings = resolved_settings(spec, cfg, experiment)
            folder = out / f'episode-{episode_index:04d}'
            folder.mkdir()
            write_json(folder/'source.settings.json', spec)
            write_json(folder/'adaptation.settings.json', settings)
            p0 = support['x'].new_zeros(spec['prompt_dim'])
            needs_prior = any(arm in {'A7', 'A8', 'A7_prior_only', 'A7_init_only'} for arm in selected_arms)
            d7 = offset(model, support['x'], support['group'], support['metadata'], len(p0), injection, settings['offset_norm']) if needs_prior else p0
            d8 = p0
            if 'A8' in selected_arms:
                if not ep.get('wrong_metadata_type_compatible', False):
                    raise ValueError('E3 needs a predeclared physically type-compatible metadata reassignment.')
                d8 = offset(model, support['x'], support['group'], support['metadata'], len(p0), injection, settings['offset_norm'], True)
                if torch.allclose(d7, d8, atol=1e-6, rtol=1e-5):
                    raise ValueError('A8 has no effective physical intervention; E1 remains a separate valid comparison.')
            fitted = []
            for arm in selected_arms:
                family = 'A7' if arm.startswith('A7_') else arm
                prior = d7 if arm in {'A7', 'A7_prior_only'} else d8 if arm == 'A8' else p0
                initial = p0 if arm == 'A7_prior_only' else d7 if arm == 'A7_init_only' else None
                steps = settings['steps'][family] if arm != 'A4' else 0
                checkpoints = settings['trajectory_steps'][arm] if experiment == 'E2' and arm in {'A2', 'A3'} else [steps]
                if cfg['device'].startswith('cuda'):
                    torch.cuda.synchronize(device=cfg['device'])
                begin = time.perf_counter()
                states = adapt(model, support['x'], support['y'], support['group'], support['view'], base, bc, nc,
                               arm=family, prior=prior, scale=spec['logit_scale'], lr=settings['learning_rate'][family],
                               steps=steps, lam=settings['regularization'][family], radius=settings['prompt_radius'],
                               checkpoints=checkpoints, subset=spec['a3_parameter_names'], initial_prompt=initial)
                if cfg['device'].startswith('cuda'):
                    torch.cuda.synchronize(device=cfg['device'])
                elapsed = time.perf_counter() - begin
                for step, net, prompt in states:
                    # Native state and prompt can reproduce the fitted endpoint without query selection.
                    torch.jit.save(net, str(folder/f'{arm}-{step}.encoder.pt'))
                    torch.save({'prompt': prompt.cpu(), 'prior': prior.cpu(), 'arm': arm, 'step': step}, folder/f'{arm}-{step}.prompt.pt')
                    with torch.no_grad():
                        loss = float(crossfit(net, support['x'], prompt, support['y'], support['group'], support['view'], base, bc, nc, spec['logit_scale']))
                    write_json(folder/f'{arm}-{step}.fit.json', {'support_crossfit_loss': loss, 'trajectory_seconds': elapsed})
                    fitted.append((arm, step, net, prompt, elapsed))
            if not evaluate_query:
                continue
            # All requested arm endpoints are fixed before opening query labels.
            with np.load(path(ep['query']), allow_pickle=False) as query:
                yq_array = integer_array(query['y'], 'query.y')
                gq_array = integer_array(query['group'], 'query.group')
                if len(query['x']) != len(yq_array) or len(gq_array) != len(yq_array) or not np.isfinite(query['x']).all():
                    raise ValueError('Query shapes or finite-signal check failed.')
                xq = torch.as_tensor(query['x'], dtype=torch.float32, device=cfg['device'])
                yq = torch.as_tensor(yq_array, dtype=torch.long, device=cfg['device'])
                gq = torch.as_tensor(gq_array, dtype=torch.long, device=cfg['device'])
            if set(yq.tolist()) != set(bc + nc):
                raise ValueError('Query labels do not match the declared joint label space.')
            for group in gq.unique():
                if yq[gq == group].unique().numel() != 1:
                    raise ValueError('A query acquisition cannot have multiple class labels.')
            with torch.no_grad():
                z0 = unit(model(xq, p0))
                source_base_logits = z0 @ base.T
                old = source_base_logits.argmax(1)
                is_b = torch.tensor([int(y) in bc for y in yq], device=yq.device)
                old_truth = torch.tensor([bc.index(int(y)) for y in yq[is_b]], device=yq.device)
                a0 = float((weights(yq[is_b], gq[is_b]) * (old[is_b] == old_truth)).sum())
                for arm, step, net, prompt, elapsed in fitted:
                    proto = prototype(unit(net(support['x'], prompt)), support['y'], support['group'], nc)
                    zq = unit(net(xq, prompt))
                    logits = scores(zq, base, proto, spec['logit_scale'])
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError('Nonfinite query logits.')
                    named = dict(net.named_parameters())
                    ntrain = sum(v.numel() for v in named.values()) if arm == 'A2' else sum(named[n].numel() for n in spec['a3_parameter_names']) if arm == 'A3' else 0 if arm == 'A4' else len(prompt)
                    row = dict(trainable_parameters=ntrain, support_groups=len(sgroups), support_windows=len(support['y']),
                               dataset=ep.get('dataset', 'UNSPECIFIED'), fold=ep['fold'], shots=ep['shots'], seed=ep['seed'], draw=ep['draw'], arm=arm, step=step,
                               a0_base_acc=a0, trajectory_seconds=elapsed,
                               **scalar_metrics(logits, yq, gq, bc, nc, old, zq, z0, spec['logit_scale']))
                    row['signed_base_accuracy_loss'] = a0 - row['base_acc']
                    if metric_writer is None:
                        metric_writer = csv.DictWriter(mf, fieldnames=list(row)); metric_writer.writeheader()
                    metric_writer.writerow(row); mf.flush()
                    np.savez_compressed(folder/f'{arm}-{step}.evaluation.npz', logits=logits.cpu().numpy(),
                                        representation=zq.cpu().numpy(), source_representation=z0.cpu().numpy(), source_base_logits=source_base_logits.cpu().numpy(),
                                        label=yq.cpu().numpy(), group=gq.cpu().numpy(), class_order=np.asarray(bc+nc))
                    for index in range(len(yq)):
                        prediction = dict(dataset=ep.get('dataset', 'UNSPECIFIED'), fold=ep['fold'], shots=ep['shots'], seed=ep['seed'], draw=ep['draw'], arm=arm, step=step,
                                          observation=index, group=int(gq[index]), label=int(yq[index]),
                                          joint_prediction=(bc+nc)[int(logits[index].argmax())],
                                          base_prediction=bc[int(logits[index, :len(bc)].argmax())],
                                          source_base_prediction=bc[int(old[index])])
                        if prediction_writer is None:
                            prediction_writer = csv.DictWriter(pf, fieldnames=list(prediction)); prediction_writer.writeheader()
                        prediction_writer.writerow(prediction)
                    pf.flush()


def run(cfg: dict, root: Path, out: Path, experiment: str, *,
        arms: Sequence[str] | None = None, evaluate_query: bool = True) -> Path:
    """Execute the declared comparison with the existing, query-free adaptation core."""
    episodes = cfg['episodes']
    if not episodes:
        raise ValueError('No physical-group episodes are bound; synthetic fixture results are not PHM evidence.')
    keys = [(e['fold'], e['shots'], e['seed'], e['draw']) for e in episodes]
    if len(keys) != len(set(keys)):
        raise ValueError('Duplicated fold/shot/seed/draw episode.')
    if experiment not in ARMS:
        raise ValueError(f'Unknown experiment: {experiment}')
    selected = list(arms) if arms is not None else cfg.get('arms', ARMS[experiment])
    if arms is None and 'arms' in cfg and cfg.get('information_regime') != 'source_validation':
        raise ValueError('Selective config arms are allowed only for source-validation tuning.')
    if not selected or len(set(selected)) != len(selected) or not set(selected).issubset(ARMS[experiment]):
        raise ValueError('Invalid or duplicate requested arms.')
    if cfg.get('information_regime') not in {'target_gfs', 'source_validation', 'synthetic_interface'}:
        raise ValueError('Declare target_gfs, source_validation or synthetic_interface; target-support adaptation is not strict DG.')
    requested = torch.device(cfg['device'])
    if requested.type not in {'cpu', 'cuda'}:
        raise ValueError('Only an explicitly requested cpu or cuda device is supported.')
    if requested.type == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('Requested CUDA is unavailable; no CPU fallback is performed.')
        if requested.index is None:
            raise ValueError('Specify an explicit CUDA device index.')
        visible = os.environ.get('CUDA_VISIBLE_DEVICES')
        if visible is None:
            physical = str(requested.index)
        else:
            visible_ids = [value.strip() for value in visible.split(',')]
            if not all(value.isdigit() for value in visible_ids):
                raise ValueError('Use numeric CUDA_VISIBLE_DEVICES identities so the GPU2 exclusion is verifiable.')
            if requested.index >= len(visible_ids):
                raise ValueError('Requested CUDA index is outside CUDA_VISIBLE_DEVICES.')
            physical = visible_ids[requested.index]
        if int(physical) == 2:
            raise ValueError('Physical GPU2 is excluded from this comparison.')
    out.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    write_json(out/'config.resolved.json', {**cfg, 'actual_arms': selected, 'query_evaluated': evaluate_query})
    try:
        write_json(out/'environment.json', {'package_version': __version__, 'python': platform.python_version(),
                   'torch': torch.__version__, 'numpy': np.__version__, 'device': cfg['device'],
                   'source_revision': cfg.get('source_revision'),
                   'conda_environment': os.environ.get('CONDA_DEFAULT_ENV'),
                   'cuda_visible_devices': os.environ.get('CUDA_VISIBLE_DEVICES')})
        write_json(out/'status.json', {'status': 'running', 'experiment': experiment})
        _execute(cfg, root, out, experiment, selected, evaluate_query)
    except BaseException as error:
        (out/'error.txt').write_text(traceback.format_exc())
        status = 'cancelled' if isinstance(error, (KeyboardInterrupt, SystemExit)) else 'invalid' if isinstance(error, (ValueError, KeyError, AssertionError)) else 'failed'
        write_json(out/'status.json', {'status': status, 'error_type': type(error).__name__, 'message': str(error),
                                      'elapsed_seconds': time.perf_counter()-started})
        raise
    write_json(out/'status.json', {'status': 'completed', 'experiment': experiment,
                                  'elapsed_seconds': time.perf_counter()-started, 'query_evaluated': evaluate_query})
    return out
