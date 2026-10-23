"""Train feature-matched controls and a no-bypass fuzzy model on explicit splits."""
from __future__ import annotations
import argparse
import csv
import json
import sys
import platform
from importlib.metadata import version
from pathlib import Path
import time
from types import SimpleNamespace
import numpy as np
import torch
from torch import nn
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from .core import (unit_weights, box_bounds, class_certificate, max_box_average,
                  calibrate, calibration_capacity, risk_coverage, matched_risk)

from src.model_factory.X_model.P4JointFuzzy import Fuzzy
from .selective import SelectiveMLP

ROLES = ('train', 'tune', 'cal', 'test')
ACTIVE_OUTPUT: Path | None = None


def reserve_output(path: Path) -> None:
    """Never overwrite a previous attempt, including failed attempts."""
    global ACTIVE_OUTPUT
    ACTIVE_OUTPUT = None
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f'output is not empty: {path}; use a new run directory')
    path.mkdir(parents=True, exist_ok=True)
    ACTIVE_OUTPUT = path
    write_state('running')
    from phmfactory import __version__
    provenance = dict(python=platform.python_version(), platform=platform.platform(),
        packages={name: version(name) for name in ('numpy', 'scipy', 'torch', 'scikit-learn')},
        phmfactory_version=__version__, source_revision=None,
        source_revision_note='not embedded in this package; no Git checkout required',
        command=sys.argv)
    (path / 'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


def write_state(status: str, **fields) -> None:
    if ACTIVE_OUTPUT is not None:
        (ACTIVE_OUTPUT / 'run_state.json').write_text(
            json.dumps(dict(status=status, **fields), indent=2, allow_nan=False), encoding='utf-8')


def model_settings(args, name: str):
    values = vars(args).copy()
    values.update(getattr(args, 'selected_configs', {}).get(name, {}))
    return SimpleNamespace(**values)


def load_data(path: Path, roles: tuple[str, ...] | None = None) -> dict:
    with np.load(path, allow_pickle=False) as f:
        required = {'x', 'y', 'unit', 'split', 'domain', 'feature_names', 'kind'}
        if not required <= set(f.files):
            raise ValueError(f'missing arrays: {sorted(required - set(f.files))}')
        data = {key: f[key] for key in required}
    for key in ('unit', 'split', 'domain'):
        if data[key].ndim != 1 or data[key].shape != (len(data['x']),):
            raise ValueError(f'{key} must have shape (N,)')
        if data[key].dtype.kind not in {'U', 'S'}:
            raise ValueError(f'{key} identifiers must be strings, not implicit numeric IDs')
        data[key] = data[key].astype(str)
        if any(not value.strip() for value in data[key]):
            raise ValueError(f'{key} identifiers must be nonempty')
    # A tuning caller consumes only train/tune arrays. It never validates,
    # standardizes, predicts or scores calibration/test x/y.
    for u in np.unique(data['unit']):
        if len(np.unique(data['split'][data['unit'] == u])) != 1:
            raise ValueError(f'physical unit crosses split roles: {u}')
    if roles is not None:
        mask = np.isin(data['split'], roles)
        for key in ('x', 'y', 'unit', 'split', 'domain'):
            data[key] = data[key][mask]
    x, y = data['x'], data['y']
    if x.ndim != 2 or y.shape != (len(x),) or not np.isfinite(x).all():
        raise ValueError('x must be a finite N x D matrix and y an N-vector')
    if not np.issubdtype(y.dtype, np.integer) or len(data['feature_names']) != x.shape[1]:
        raise ValueError('use integer labels and one name per feature')
    for key in ('unit', 'split', 'domain'):
        if data[key].shape != (len(x),):
            raise ValueError(f'{key} must have shape (N,)')
    if set(data['split']) != set(roles or ROLES):
        raise ValueError(f'split must contain exactly {roles or ROLES}')
    if len(set(data['feature_names'].astype(str))) != x.shape[1]:
        raise ValueError('feature names must be unique')
    if any(not str(v).strip() for v in data['feature_names']):
        raise ValueError('feature names must be nonempty')
    for u in np.unique(data['unit']):
        if len(np.unique(data['split'][data['unit'] == u])) != 1:
            raise ValueError(f'physical unit crosses split roles: {u}')
    train = data['split'] == 'train'
    classes = np.unique(y[train])
    if not np.array_equal(classes, np.arange(len(classes))) or not set(y) <= set(classes):
        raise ValueError('closed-set labels must be 0..C-1 and all present in train')
    if len(classes) < 2:
        raise ValueError('classification requires at least two classes')
    data['kind'] = str(data['kind'].item())
    if data['kind'] not in {'real', 'synthetic'}:
        raise ValueError('kind must be real or synthetic')
    return data


def synthetic(path: Path) -> None:
    """Explicit software smoke data, not a substitute for industrial observations."""
    rng = np.random.default_rng(2026)
    xs, ys, units, splits = [], [], [], []
    for role, n in zip(ROLES, (480, 240, 800, 800)):
        y = rng.integers(0, 2, n)
        x = rng.normal(0, .45, (n, 3))
        x[:, 0] += 2 * y - 1
        x[:, 1] += .4 * (2 * y - 1)
        xs.append(x); ys.append(y)
        units.extend([f'{role}_{i}' for i in range(n)])
        splits.extend([role] * n)
    np.savez_compressed(path, x=np.concatenate(xs), y=np.concatenate(ys),
                        unit=np.array(units), split=np.array(splits),
                        domain=np.array(['synthetic_iid'] * len(units)),
                        feature_names=np.array(['concept_1', 'concept_2', 'nuisance']),
                        kind=np.array('synthetic'))


def train_torch(model: nn.Module, x: np.ndarray, y: np.ndarray, weights: np.ndarray,
                args, robust: bool = False, validation=None) -> tuple[nn.Module, dict, float]:
    device = torch.device(args.device)
    model.to(device)
    tx = torch.tensor(x, dtype=torch.float32, device=device)
    ty = torch.tensor(y, dtype=torch.long, device=device)
    tw = torch.tensor(weights * len(weights), dtype=torch.float32, device=device)
    opt = torch.optim.Adam(model.parameters(), lr=getattr(args, 'learning_rate', .003),
                           weight_decay=getattr(args, 'weight_decay', 0.0))
    start = time.perf_counter()
    history, best_state, best_loss, best_epoch, stale, steps = [], None, float('inf'), 0, 0, 0
    for epoch in range(args.epochs):
        model.train()
        training_objective = 0.
        order = torch.randperm(len(x), device=device)
        for idx in order.split(getattr(args, 'batch_size', 256)):
            opt.zero_grad()
            if isinstance(model, SelectiveMLP):
                objective = model.objective(tx[idx], ty[idx], tw[idx])
            else:
                loss = nn.functional.cross_entropy(model(tx[idx]), ty[idx], reduction='none')
                if robust:
                    loss = loss + .1 * model.robust_penalty(tx[idx], ty[idx], args.radius)
                objective = (loss * tw[idx]).mean()
            if not torch.isfinite(objective):
                raise FloatingPointError('nonfinite training objective')
            training_objective += float(objective.detach()) * len(idx) / len(x)
            objective.backward()
            for parameter in model.parameters():
                if parameter.grad is None or not torch.isfinite(parameter.grad).all():
                    raise FloatingPointError('missing or nonfinite gradient')
            opt.step(); steps += 1
        history.append(dict(epoch=epoch + 1, training_batch_objective=training_objective))
        if validation is not None:
            vx, vy, vw = validation
            model.eval()
            with torch.no_grad():
                vl = nn.functional.cross_entropy(
                    model(torch.as_tensor(vx, dtype=torch.float32, device=device)),
                    torch.as_tensor(vy, dtype=torch.long, device=device), reduction='none')
                value = float(vl.cpu().numpy() @ vw)
            if not np.isfinite(value):
                raise FloatingPointError('nonfinite tuning NLL')
            history[-1]['tune_nll'] = value
            if value < best_loss:
                best_loss, best_epoch, stale = value, epoch + 1, 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if stale >= getattr(args, 'patience', 10):
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    elapsed = time.perf_counter() - start
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    model.fit_report = dict(epochs_completed=epoch + 1, optimizer_steps=steps,
                            selected_epoch=best_epoch or epoch + 1,
                            checkpoint_rule='minimum tune NLL' if validation is not None else 'fixed final epoch',
                            tune_nll=best_loss if validation is not None else None, history=history)
    return model, state, elapsed


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite', choices=['sanity', 'overfit', 'smoke', 'main', 'ablation', 'all'])
    parser.add_argument('--data', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--rules', type=int, default=16)
    parser.add_argument('--radius', type=float, default=.10)
    parser.add_argument('--alpha', type=float, default=.05)
    parser.add_argument('--delta', type=float, default=.05)
    parser.add_argument('--seeds', type=int, nargs='+', default=[42, 123, 456, 789, 1024])
    parser.add_argument('--device', default='cpu', choices=['cpu', 'cuda'])
    parser.add_argument('--learning-rate', type=float, default=.003)
    parser.add_argument('--weight-decay', type=float, default=0.)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--mlp-width', type=int, default=64)
    parser.add_argument('--selector-width', type=int, default=64)
    parser.add_argument('--target-coverage', type=float, default=.9)
    parser.add_argument('--coverage-penalty', type=float, default=32.)
    parser.add_argument('--aux-weight', type=float, default=.5)
    parser.add_argument('--selection', type=Path, help='frozen train/tune-only selection.json')
    parser.add_argument('--methods', nargs='+', help='explicit subset; calibration family is corrected accordingly')
    parser.add_argument('--overfit-model', choices=['fuzzy', 'mlp', 'selective'], default='mlp')
    parser.add_argument('--unseen-domain', help='optional DG audit: this domain occurs only in test')
    parser.add_argument('--diagnostic-tune', action='store_true',
                        help='fit/compare using train+tune only; no risk certificate or test outputs')
    parser.add_argument('--calibration', choices=['iid', 'cluster'], default='cluster')
    parser.add_argument('--check-calibration', action='store_true',
                        help='write a count-based feasibility report and exit before training')
    parser.add_argument('--calibration-units', type=int,
                        help='declared unit count for a data-free feasibility check, not a dataset audit')
    args = parser.parse_args(argv)
    args.selected_configs = {}
    if args.selection is not None:
        selection = json.loads(args.selection.read_text(encoding='utf-8'))
        if selection.get('status') != 'completed' or selection.get('selection_metric') != 'unit_balanced_tune_nll':
            raise ValueError('selection must be a completed train/tune-only search')
        args.selected_configs = selection['selected_configs']
        args.selection = str(args.selection.resolve())
    if args.learning_rate <= 0 or args.weight_decay < 0 or args.batch_size < 1:
        raise ValueError('learning rate/batch size must be positive and weight decay nonnegative')
    if args.mlp_width < 2 or args.selector_width < 2:
        raise ValueError('MLP/selector width must be at least two')
    if args.calibration_units is not None and (not args.check_calibration or args.data is not None):
        raise ValueError('--calibration-units requires --check-calibration and no --data')
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError('seeds must be distinct; repeated seeds would overwrite run outputs')
    grid = np.linspace(0, 1, 21)  # fixed before calibration labels are used
    names = ['fuzzy_rule_score', 'joint']
    if args.suite in {'main', 'all', 'smoke'}:
        names += ['logistic_msp', 'forest_msp', 'mlp_msp', 'selectivenet_feature']
    if args.suite in {'ablation', 'all'}:
        names += ['fuzzy_msp', 'class_only', 'joint_no_robust_training',
                  'joint_permuted_cost', 'joint_zero_radius']
    if args.methods is not None:
        unknown = set(args.methods) - set(names)
        if unknown or len(set(args.methods)) != len(args.methods):
            raise ValueError(f'methods must be a unique subset of the suite: {unknown}')
        names = args.methods
        if 'joint' not in names:
            raise ValueError('evaluation subsets must include joint; use tune.py for isolated fit diagnostics')
    if args.selection is not None:
        required = {'fuzzy'}
        if 'mlp_msp' in names:
            required.add('mlp')
        if 'selectivenet_feature' in names:
            required.add('selective')
        if not required <= set(args.selected_configs):
            raise ValueError(f'selection lacks required model searches: {sorted(required - set(args.selected_configs))}')
    delta_family = args.delta / (len(names) * len(args.seeds))
    if args.check_calibration and args.calibration_units is not None:
        if args.suite == 'smoke':
            raise ValueError('count-only checks use main, ablation, or all; not synthetic smoke')
        report = calibration_capacity(args.calibration_units, len(grid), args.alpha,
                                      delta_family, args.calibration)
        report.update(input_kind='declared count; no dataset inspected',
                      methods=names, seeds=args.seeds, training_executed=False)
        reserve_output(args.output)
        (args.output / 'resolved_config.json').write_text(
            json.dumps(vars(args), default=str, indent=2), encoding='utf-8')
        (args.output / 'calibration_capacity.json').write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding='utf-8')
        print(json.dumps(report, indent=2))
        write_state('completed' if report['structurally_feasible'] else 'calibration_infeasible',
                    training_executed=False)
        raise SystemExit(0 if report['structurally_feasible'] else 2)
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA was requested but is unavailable')
    if args.epochs < 1 or args.rules < 2 or args.radius < 0:
        raise ValueError('epochs>=1, rules>=2, radius>=0 are required')
    reserve_output(args.output)
    (args.output / 'resolved_config.json').write_text(
        json.dumps(vars(args), default=str, indent=2), encoding='utf-8')
    if args.suite == 'smoke':
        if args.data is not None:
            raise ValueError('smoke creates synthetic data; use main for a supplied dataset')
        args.data = args.output / 'synthetic.npz'; synthetic(args.data)
    elif args.data is None:
        raise ValueError('--data is required; real experiments never generate fallback data')
    if args.diagnostic_tune and (args.check_calibration or args.unseen_domain):
        raise ValueError('tune diagnostic cannot calibrate or evaluate an unseen test domain')
    data = load_data(args.data, roles=('train', 'tune') if args.suite == 'overfit' or args.diagnostic_tune else None)
    if args.unseen_domain is not None:
        heldout = data['domain'] == args.unseen_domain
        test_role = data['split'] == 'test'
        if not test_role.any() or not np.all(heldout == test_role):
            raise ValueError('unseen-domain must identify every test row and no train/tune/cal row')
    if args.selection is not None:
        if (selection['data'] != str(args.data.resolve()) or
                selection['data_kind'] != data['kind'] or
                selection['feature_names'] != data['feature_names'].tolist()):
            raise ValueError('selected configuration belongs to a different archive or feature contract')
        for role in ('train', 'tune'):
            if selection[f'{role}_units'] != np.unique(data['unit'][data['split'] == role]).tolist():
                raise ValueError(f'selection {role} units differ from the supplied archive')
        if any(c['radius'] != args.radius for c in args.selected_configs.values()):
            raise ValueError('evaluation radius differs from the source-only selection protocol')
    if args.suite == 'sanity':
        report = dict(kind=data['kind'], features=data['feature_names'].tolist(), roles={
            role: dict(rows=int(np.sum(data['split'] == role)),
                       units=int(len(np.unique(data['unit'][data['split'] == role]))),
                       classes=np.unique(data['y'][data['split'] == role]).tolist(),
                       domains=np.unique(data['domain'][data['split'] == role]).tolist()) for role in ROLES})
        (args.output / 'data_sanity.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        write_state('completed', training_executed=False, independence='requires acquisition record')
        return
    split, y, unit = data['split'], data['y'], data['unit']
    train, tune, cal, test = [split == r for r in ROLES]
    if args.diagnostic_tune:
        test = tune.copy()  # reporting mask only; archive split remains explicitly tune
    if args.suite == 'overfit':
        weights = unit_weights(unit[train])
        mu = np.average(data['x'][train], axis=0, weights=weights)
        sd = np.sqrt(np.average((data['x'][train] - mu) ** 2, axis=0, weights=weights))
        if np.any(sd <= 1e-12):
            raise ValueError('constant training feature')
        xx, yy = (data['x'][train] - mu) / sd, y[train]
        chosen = np.unique(np.r_[np.arange(min(128, len(yy))),
                                [np.flatnonzero(yy == c)[0] for c in np.unique(yy)]])
        xx, yy = xx[chosen], yy[chosen]
        torch.manual_seed(args.seeds[0]); torch.set_num_threads(1)
        classes = int(yy.max()) + 1
        if args.overfit_model == 'fuzzy':
            model = Fuzzy(xx, yy, min(args.rules, len(yy)), args.seeds[0])
        elif args.overfit_model == 'selective':
            model = SelectiveMLP(xx.shape[1], classes, args.selector_width,
                coverage=args.target_coverage, coverage_penalty=args.coverage_penalty,
                aux_weight=args.aux_weight)
        else:
            model = nn.Sequential(nn.Linear(xx.shape[1], args.mlp_width), nn.ReLU(), nn.Linear(args.mlp_width, classes))
        model, state, _ = train_torch(model, xx, yy, np.ones(len(yy)) / len(yy), args)
        with torch.no_grad():
            accuracy = float((model(torch.tensor(xx, dtype=torch.float32, device=args.device)).argmax(1).cpu().numpy() == yy).mean())
        torch.save(dict(state=state, mean=mu, std=sd), args.output / 'overfit.pt')
        (args.output / 'overfit.json').write_text(json.dumps(dict(train_accuracy=accuracy,
            required_accuracy=.95, passed=accuracy >= .95, rows=len(yy), **model.fit_report), indent=2), encoding='utf-8')
        write_state('completed' if accuracy >= .95 else 'overfit_failed', evidence_eligible=False)
        if accuracy < .95:
            raise SystemExit(3)
        return
    if args.calibration == 'iid' and len(np.unique(unit[cal])) != int(cal.sum()):
        raise ValueError('iid calibration requires one observation per independent unit')
    capacity = (dict(structurally_feasible=None, status='not_attempted_tuning_diagnostic')
                if args.diagnostic_tune else calibration_capacity(
                    len(np.unique(unit[cal])), len(grid), args.alpha, delta_family, args.calibration))
    capacity.update(input_kind='archive unit-count check; independence not established',
                    data_kind=data['kind'], methods=names, seeds=args.seeds,
                    training_executed=False)
    (args.output / 'calibration_capacity.json').write_text(
        json.dumps(capacity, indent=2, allow_nan=False), encoding='utf-8')
    if args.check_calibration:
        print(json.dumps(capacity, indent=2))
        write_state('completed' if capacity['structurally_feasible'] else 'calibration_infeasible', training_executed=False)
        raise SystemExit(0 if capacity['structurally_feasible'] else 2)
    if not args.diagnostic_tune and not capacity['structurally_feasible']:
        print('The declared calibration bound cannot certify any policy at this unit count. '
              'Continuing the requested empirical comparisons without changing the bound; '
              'see calibration_capacity.json.', file=sys.stderr)
    weights = unit_weights(unit[train])
    mu = np.average(data['x'][train], axis=0, weights=weights)
    sd = np.sqrt(np.average((data['x'][train] - mu) ** 2, axis=0, weights=weights))
    if np.any(sd <= 1e-12):
        raise ValueError('constant training feature: remove it explicitly before running')
    x = (data['x'] - mu) / sd
    if model_settings(args, 'fuzzy').rules > train.sum():
        raise ValueError('rules exceeds training observations')
    classes = int(y.max()) + 1
    config = vars(args).copy()
    config.update(data=str(args.data), output=str(args.output), data_kind=data['kind'],
                  methods=names, family_delta=delta_family,
                  evaluation_role='tune' if args.diagnostic_tune else 'test',
                  evidence_eligible=not args.diagnostic_tune and data['kind'] == 'real',
                  estimand='uniform physical unit then uniform window; conditional on acceptance')
    (args.output / 'config.json').write_text(json.dumps(config, default=str, indent=2), encoding='utf-8')
    metrics, curves, matched, per_unit = [], [], [], []
    torch.set_num_threads(1)
    for seed in args.seeds:
        torch.manual_seed(seed)
        fitted = {}
        fuzzy_args = model_settings(args, 'fuzzy')
        nf = Fuzzy(x[train], y[train], fuzzy_args.rules, seed)
        nf, state, elapsed = train_torch(nf, x[train], y[train], weights, fuzzy_args, robust=True)
        torch.save(dict(state=state, mean=mu, std=sd), args.output / f'fuzzy_{seed}.pt')
        (args.output / f'fit_fuzzy_{seed}.json').write_text(json.dumps(nf.fit_report, indent=2), encoding='utf-8')
        c = nf.centers.detach().cpu().numpy()
        s = nf.log_scales.exp().clamp(.1, 10).detach().cpu().numpy()
        q = nf.log_q.softmax(dim=1).detach().cpu().numpy()
        lo, hi, w = box_bounds(x, c, s, args.radius)
        prob = w @ q; pred = prob.argmax(axis=1)
        wt = unit_weights(unit[tune])
        err = (pred[tune] != y[tune]).astype(float)
        # Smoothed rule errors are a ranking score, not calibrated probabilities.
        mass = wt[:, None] * w[tune]
        cost = ((mass * err[:, None]).sum(axis=0) + .01) / (mass.sum(axis=0) + .02)
        stable = class_certificate(lo, hi, q, pred)
        upper_cost = max_box_average(lo, hi, cost)
        parameters = sum(p.numel() for p in nf.parameters())
        fitted['fuzzy_rule_score'] = (prob, w @ cost, np.ones(len(x), bool), parameters, elapsed)
        fitted['joint'] = (prob, upper_cost, stable, parameters, elapsed)
        np.savez_compressed(args.output / f'trace_{seed}.npz', probabilities=prob, weights=w,
                            q=q, rule_cost=cost, lower=lo, upper=hi, centers=c, scales=s,
                            mean=mu, std=sd, y=y, unit=unit, split=split, domain=data['domain'])
        if args.suite in {'main', 'all', 'smoke'}:
            controls = {
                'logistic_msp': LogisticRegression(max_iter=1000, random_state=seed),
                'forest_msp': RandomForestClassifier(n_estimators=200, min_samples_leaf=2,
                                                     random_state=seed, n_jobs=1)}
            for name, model in controls.items():
                if name not in names:
                    continue
                start = time.perf_counter()
                model.fit(x[train], y[train], sample_weight=weights * train.sum())
                duration = time.perf_counter() - start
                p = model.predict_proba(x)
                fitted[name] = (p, 1 - p.max(axis=1), np.ones(len(x), bool), None, duration)
            if 'mlp_msp' in names:
                mlp_args = model_settings(args, 'mlp')
                width = mlp_args.mlp_width
                torch.manual_seed(seed)
                mlp = nn.Sequential(nn.Linear(x.shape[1], width), nn.ReLU(), nn.Linear(width, classes))
                mlp, state, duration = train_torch(mlp, x[train], y[train], weights, mlp_args)
                torch.save(dict(state=state, mean=mu, std=sd), args.output / f'mlp_{seed}.pt')
                (args.output / f'fit_mlp_{seed}.json').write_text(json.dumps(mlp.fit_report, indent=2), encoding='utf-8')
                with torch.no_grad():
                    p = mlp(torch.tensor(x, dtype=torch.float32, device=args.device)).softmax(dim=1).cpu().numpy()
                fitted['mlp_msp'] = (p, 1-p.max(axis=1), np.ones(len(x), bool),
                                     sum(p.numel() for p in mlp.parameters()), duration)
            if 'selectivenet_feature' in names:
                selector_args = model_settings(args, 'selective')
                torch.manual_seed(seed)
                selector = SelectiveMLP(x.shape[1], classes, selector_args.selector_width,
                    coverage=selector_args.target_coverage, coverage_penalty=selector_args.coverage_penalty,
                    aux_weight=selector_args.aux_weight)
                selector, state, duration = train_torch(selector, x[train], y[train], weights, selector_args)
                torch.save(dict(state=state, mean=mu, std=sd, width=selector_args.selector_width),
                           args.output / f'selective_{seed}.pt')
                (args.output / f'fit_selective_{seed}.json').write_text(json.dumps(selector.fit_report, indent=2), encoding='utf-8')
                with torch.no_grad():
                    logits, acceptance, _ = selector.forward_components(torch.tensor(x, dtype=torch.float32, device=args.device))
                    p = logits.softmax(1).cpu().numpy()
                    score = 1 - acceptance.cpu().numpy()
                fitted['selectivenet_feature'] = (p, score, np.ones(len(x), bool),
                    sum(p.numel() for p in selector.parameters()), duration)
        if args.suite in {'ablation', 'all'}:
            fitted['fuzzy_msp'] = (prob, 1-prob.max(axis=1), np.ones(len(x), bool), parameters, elapsed)
            fitted['class_only'] = (prob, w @ cost, stable, parameters, elapsed)
            perm = np.random.default_rng(seed).permutation(cost)
            fitted['joint_permuted_cost'] = (prob, max_box_average(lo, hi, perm), stable, parameters, elapsed)
            l0, h0, _ = box_bounds(x, c, s, 0.)
            fitted['joint_zero_radius'] = (prob, max_box_average(l0, h0, cost),
                                           class_certificate(l0, h0, q, pred), parameters, elapsed)
            if 'joint_no_robust_training' in names:
                torch.manual_seed(seed)
                plain = Fuzzy(x[train], y[train], fuzzy_args.rules, seed)
                plain, pstate, ptime = train_torch(plain, x[train], y[train], weights, fuzzy_args)
                (args.output / f'fit_fuzzy_plain_{seed}.json').write_text(
                    json.dumps(plain.fit_report, indent=2), encoding='utf-8')
                torch.save(dict(state=pstate, mean=mu, std=sd), args.output / f'fuzzy_plain_{seed}.pt')
                cc = plain.centers.detach().cpu().numpy()
                ss = plain.log_scales.exp().clamp(.1, 10).detach().cpu().numpy()
                qq = plain.log_q.softmax(dim=1).detach().cpu().numpy()
                ll, hh, ww = box_bounds(x, cc, ss, args.radius)
                pp = ww @ qq; yy = pp.argmax(axis=1)
                mm = wt[:, None] * ww[tune]
                rr = ((mm * (yy[tune] != y[tune])[:, None]).sum(axis=0)+.01)/(mm.sum(axis=0)+.02)
                fitted['joint_no_robust_training'] = (pp, max_box_average(ll, hh, rr),
                                                     class_certificate(ll, hh, qq, yy), parameters, ptime)
        for name in names:
            p, score, eligible, size, duration = fitted[name]
            pred = p.argmax(axis=1); error = (pred != y).astype(float)
            cert = (dict(threshold=None, upper=None, calibration_coverage=0., certified=False,
                         calibration_units=0, mode='not_attempted_tuning_diagnostic')
                    if args.diagnostic_tune else calibrate(error[cal], score[cal], eligible[cal],
                        unit[cal], grid, args.alpha, delta_family, args.calibration))
            if name == 'joint':
                np.savez_compressed(args.output / f'model_{seed}.npz', centers=c, scales=s, q=q,
                                    rule_cost=cost, mean=mu, std=sd, radius=args.radius,
                                    threshold=cert['threshold'] if cert['certified'] else 0.,
                                    certified=cert['certified'], feature_names=data['feature_names'])
            accepted = np.zeros(len(x), bool) if cert['threshold'] is None else eligible & (score <= cert['threshold'])
            coverage, risk = risk_coverage(error[test], accepted[test], unit[test])
            row = dict(method=name, seed=seed, data_kind=data['kind'], coverage=coverage, risk=risk,
                       full_accuracy=1-float(unit_weights(unit[test]) @ error[test]),
                       parameters=size, train_seconds=duration, **cert)
            metrics.append(row)
            np.savez_compressed(args.output / f'predictions_{name}_{seed}.npz',
                                probabilities=p, score=score, eligible=eligible, accepted=accepted,
                                y=y, unit=unit, split=split, domain=data['domain'])
            for t in grid:
                cv, rr = risk_coverage(error[test], eligible[test] & (score[test] <= t), unit[test])
                curves.append(dict(method=name, seed=seed, data_kind=data['kind'], threshold=t, coverage=cv, risk=rr))
            for target in (.5, .7, .9):
                rr = matched_risk(error[test], score[test], eligible[test], unit[test], target)
                matched.append(dict(method=name, seed=seed, data_kind=data['kind'], target_coverage=target,
                                    risk=rr, diagnostic='empirical fractional-boundary ranking; not deployment'))
            for u in np.unique(unit[test]):
                idx = test & (unit == u)
                per_unit.append(dict(method=name, seed=seed, unit=u, domain=data['domain'][idx][0],
                                     accepted_fraction=float(accepted[idx].mean()),
                                     accepted_error_fraction=float((error[idx]*accepted[idx]).mean()),
                                     full_error=float(error[idx].mean())))
    (args.output / 'metrics.json').write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding='utf-8')
    write_csv(args.output / 'metrics.csv', metrics)
    write_csv(args.output / 'risk_coverage.csv', curves)
    write_csv(args.output / 'matched_coverage.csv', matched)
    write_csv(args.output / 'per_unit.csv', per_unit)
    write_state('completed', data_kind=data['kind'], calibration_feasible=capacity['structurally_feasible'],
                scientific_effect='not_evaluated' if args.diagnostic_tune else 'unadjudicated',
                evaluation_role='tune' if args.diagnostic_tune else 'test', baseline_qualified=False)
    print(json.dumps(dict(output=str(args.output), data_kind=data['kind'], methods=len(names),
                          runs=len(metrics), certified=sum(r['certified'] for r in metrics)), indent=2))


def main(argv: list[str] | None = None) -> None:
    global ACTIVE_OUTPUT
    ACTIVE_OUTPUT = None
    try:
        _main(argv)
    except KeyboardInterrupt:
        write_state('cancelled', scientific_effect='not_adjudicated')
        raise
    except Exception as exc:
        write_state('failed', error_type=type(exc).__name__, error=str(exc), scientific_effect='not_adjudicated')
        raise


if __name__ == '__main__':
    main()
