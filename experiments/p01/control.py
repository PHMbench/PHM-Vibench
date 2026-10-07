"""Plan and run the EXISTING P01 DG commands; no models, metrics or new protocol.

The public paper entry is `fusion.sh dg plan|run|status`. Configurations remain
study.yaml and the existing task files. Local control logs record process state,
not dataset admission, baseline qualification or scientific success.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import json
from importlib.metadata import version, PackageNotFoundError
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

import yaml

ROOT = Path(__file__).resolve().parents[2]
STAGES = ('bind', 'condition-audit', 'preflight', 'smoke-baseline', 'calibrate',
          'tune-reference', 'tune-baselines', 'fit-baselines', 'qualify',
          'smoke-method', 'tune-method', 'fit-method', 'freeze', 'test', 'analyze')
CORE = ('I', 'I-F', 'MLP16', 'I-single', 'Dense', 'Dense-matched', 'I-base')
EXPERIMENT_ARMS = {'E1': ('I',), 'E2': ('I', 'I-F', 'MLP16'),
                   'E3': ('I', 'I-single'), 'E4': ('I', 'Dense', 'Dense-matched'),
                   'E5': ('I', 'I-base')}
MARKERS = {'bind': 'tasks.json', 'condition-audit': 'condition_audit.csv',
           'preflight': 'preflight.json', 'smoke-baseline': 'smoke_baseline.json',
           'calibrate': 'budget_selection.json', 'qualify': 'baseline_qualification.json',
           'smoke-method': 'smoke_method.json', 'freeze': 'frozen.json',
           'test': 'test_complete.json', 'analyze': 'summary/scope.json'}
REENTER = {'condition-audit', 'tune-reference', 'tune-baselines', 'fit-baselines',
           'qualify', 'tune-method', 'fit-method', 'test'}


def read(path):
    with Path(path).open(encoding='utf-8') as stream:
        return json.load(stream) if Path(path).suffix == '.json' else yaml.safe_load(stream)


def atomic_json(path, data):
    temporary = path.with_name(path.name + f'.{os.getpid()}.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(data, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def csv_values(value):
    values = value.split(',')
    if not all(values) or len(set(values)) != len(values):
        raise argparse.ArgumentTypeError('Use nonempty unique comma-separated values.')
    return values


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=('plan', 'run', 'status'))
    p.add_argument('--root', required=True, type=Path, help='Existing/new numerical result root; logs go to ROOT.control.')
    p.add_argument('--study', type=Path, help='Existing study YAML; no CLI scientific overrides.')
    p.add_argument('--task', action='append', type=Path, help='All prospective task YAMLs, repeated; never a final-test subset.')
    p.add_argument('--only', type=csv_values, help='Only named stages, executed in the fixed protocol order.')
    p.add_argument('--from-stage', choices=STAGES, default='bind')
    p.add_argument('--through', choices=STAGES, help='Default freeze; default fit-method when using filters.')
    p.add_argument('--experiments', type=csv_values, help='Select E1..E6 development comparisons; remaining required comparisons still block final freeze.')
    p.add_argument('--arms', type=csv_values, help='Existing arm names in selected tune/fit families.')
    p.add_argument('--task-names', type=csv_values, help='Execute selected bound tasks in tune/fit; the study binding stays complete.')
    p.add_argument('--seeds', type=csv_values, help='Execute selected FINAL seeds in fit; never change the HPO seed or final matrix.')
    p.add_argument('--device', default='cuda:0', help='Passed unchanged to the existing runtime; physical GPU0 remains required.')
    p.add_argument('--resume', action='store_true', help='Reuse recorded completed stages; trainers validate completed exact trials.')
    p.add_argument('--retry-interrupted', action='store_true', help='Explicit same-trial/seed restart in calibration/tune/fit after interruption; preserves old files.')
    p.add_argument('--allow-target-read', action='store_true', help='Explicitly authorize this invocation to reach the existing final test stage.')
    p.add_argument('--fixture', action='store_true', help='Constructed integration tests only; never use for real-data admission.')
    return p


def control_dir(root):
    return root.with_name(root.name + '.control')


def make_plan(args):
    root = args.root.expanduser().resolve()
    if args.only and (args.from_stage != 'bind' or args.through):
        raise ValueError('--only cannot be combined with --from-stage/--through.')
    if args.experiments and args.arms:
        raise ValueError('Select --experiments or --arms, not both.')
    if args.retry_interrupted and not args.resume:
        raise ValueError('--retry-interrupted requires --resume.')
    filtered = any((args.experiments, args.arms, args.task_names, args.seeds))
    end = args.through or ('fit-method' if filtered else 'freeze')
    stages = ([s for s in STAGES if s in args.only] if args.only else
              list(STAGES[STAGES.index(args.from_stage):STAGES.index(end) + 1]))
    if not stages or (args.only and set(args.only) - set(STAGES)):
        raise ValueError(f'Invalid/empty stage range; available={STAGES}')
    if filtered and set(stages) & {'freeze', 'test', 'analyze'}:
        raise ValueError('Partial selections cannot freeze/test/analyze. Complete the unchanged full study first.')
    if args.action == 'run' and 'test' in stages and not args.allow_target_read:
        raise ValueError('Final target access requires --allow-target-read; default execution stops at freeze.')
    recorded_path = control_dir(root)/'state.json'
    original_study = read(recorded_path)['inputs']['study_path'] if args.resume and recorded_path.exists() else None
    study_path = args.study.expanduser().resolve() if args.study else Path(original_study) if original_study else root/'study.yaml'
    study = read(study_path)
    if not isinstance(study, dict): raise ValueError('Study must be the existing mapping YAML.')
    baselines = list(study['baselines'])
    views = [v['name'] for v in study['fusion']['model']['branches']]
    excluded = [f'I-minus-{v}' for v in views] if study.get('leave_one_view_out') else []
    available = [*baselines, *CORE, *excluded]
    arms = args.arms
    if args.experiments:
        if set(args.experiments) - {*EXPERIMENT_ARMS, 'E6'}:
            raise ValueError('E7–E12 reuse full frozen outputs; select stages test/analyze after completing the matrix.')
        if 'E6' in args.experiments and not excluded:
            raise ValueError('The bound study has no refitted view exclusions.')
        chosen = set()
        for experiment in args.experiments:
            chosen.update(['I', *excluded] if experiment == 'E6' else EXPERIMENT_ARMS[experiment])
        arms = [a for a in available if a in chosen]
    if arms and set(arms) - set(available): raise ValueError(f'Unknown arms: {set(arms) - set(available)}')
    seeds = [int(s) for s in args.seeds] if args.seeds else None
    if seeds and (len(set(seeds)) != len(seeds) or set(seeds) - set(study['seeds'])): raise ValueError('Selected seeds are not in the frozen final seed list.')
    task_paths = [p.expanduser().resolve() for p in (args.task or [])]
    if 'bind' in stages and not task_paths:
        # A full resume uses the originally declared input paths, not inferred H5 data.
        state_path = control_dir(root)/'state.json'
        if args.resume and state_path.exists():
            task_paths = [Path(p) for p in read(state_path)['inputs']['tasks']]
        else:
            raise ValueError('bind requires repeated --task with the complete prospective suite.')
    if task_paths:
        names = [read(p)['name'] for p in task_paths]
    elif (root/'tasks.json').exists():
        names = [t['name'] for t in read(root/'tasks.json')['tasks']]
    else:
        names = []
    if args.task_names and set(args.task_names) - set(names): raise ValueError('Unknown bound task name.')
    commands = []
    for stage in stages:
        command = [sys.executable, '-u', '-m', 'experiments.p01.multiview_dg']
        if stage == 'bind':
            command += ['bind', '--study', str(study_path), '--output', str(root)]
            for path in task_paths: command += ['--task', str(path)]
            if args.fixture: command.append('--fixture')
        else:
            base, _, family = stage.partition('-') if stage.startswith(('tune-', 'fit-', 'smoke-')) else (stage, '', '')
            command += [base, '--root', str(root)]
            if family: command += ['--kind' if base == 'smoke' else '--family', family]
            if base in {'smoke', 'calibrate', 'tune', 'fit', 'freeze', 'test'}:
                command += ['--device', args.device]
            if base in {'tune', 'fit'}:
                for name in args.task_names or []: command += ['--task-name', name]
                if arms and family != 'reference':
                    pool = baselines if family == 'baselines' else [*CORE, *excluded]
                    picked = [a for a in pool if a in arms]
                    if args.experiments and family == 'baselines': picked = baselines
                    if base == 'tune':
                        picked = list(dict.fromkeys('I' if a in excluded else a for a in picked))
                    if not picked: raise ValueError(f'No selected arm belongs to stage {stage}. Use --only to choose the intended family.')
                    for arm in picked: command += ['--arm', arm]
                if seeds and base == 'fit':
                    for seed in seeds: command += ['--seed', str(seed)]
                if args.retry_interrupted: command.append('--retry-interrupted')
        if stage == 'calibrate' and args.retry_interrupted: command.append('--retry-interrupted')
        commands.append((stage, command))
    return root, study_path, task_paths, commands


def code_context():
    paths = {'runtime': ROOT}
    if os.environ.get('P01_PAPER_ROOT'): paths['paper'] = Path(os.environ['P01_PAPER_ROOT'])
    result = {}
    for role, path in paths.items():
        revision = subprocess.run(['git','-C',str(path),'rev-parse','HEAD'],capture_output=True,text=True)
        dirty = subprocess.run(['git','-C',str(path),'diff','--name-only','HEAD','--','.'],capture_output=True,text=True)
        result[role] = dict(path=str(path), revision=revision.stdout.strip() if revision.returncode == 0 else None,
                            modified_files=dirty.stdout.splitlines())
    packages = {}
    for package in ('torch','numpy','pandas','h5py','PyYAML','pytorch-wavelets'):
        try: packages[package] = version(package)
        except PackageNotFoundError: packages[package] = None
    result['environment'] = dict(python=sys.version, executable=sys.executable, packages=packages,
                                CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'))
    return result


def group_alive(pgid):
    if not pgid: return False
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


@contextmanager
def lock(path):
    with path.open('a') as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another controller owns this result root; no concurrent writes.') from error
        yield


def call_stage(command, log_path, announce):
    """Signal the whole stage process group, including the existing trainer child."""
    child = None
    received = [None]
    old = {}
    def stop(signum, _frame):
        received[0] = signum
        raise KeyboardInterrupt
    for signum in (signal.SIGINT, signal.SIGTERM):
        old[signum] = signal.signal(signum, stop)
    try:
        env = dict(os.environ, PYTHONUNBUFFERED='1', PYTHONPATH=str(ROOT) + os.pathsep + os.environ.get('PYTHONPATH',''))
        with log_path.open('x', encoding='utf-8') as log:
            child = subprocess.Popen(command,cwd=ROOT,env=env,start_new_session=True,
                                     stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,bufsize=1)
            announce(child.pid)
            for line in child.stdout:
                log.write(line); log.flush()
                print(line,end='',flush=True)
            return child.wait(), None
    except KeyboardInterrupt:
        signum = received[0] or signal.SIGINT
        for sig in (signal.SIGINT, signal.SIGTERM): signal.signal(sig, signal.SIG_IGN)
        if child is not None:
            if group_alive(child.pid): os.killpg(child.pid, signum)
            try:
                child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL); child.wait()
        return 128 + signum, signum
    finally:
        if child is not None and child.stdout is not None: child.stdout.close()
        for signum, handler in old.items(): signal.signal(signum, handler)


def run(args, root, study_path, task_paths, commands):
    folder = control_dir(root)
    folder.mkdir(parents=True, exist_ok=True)
    with lock(folder/'lock'):
        path = folder/'state.json'
        state = read(path) if path.exists() else dict(inputs=None, code=None, steps={})
        if group_alive(state.get('active_pgid')):
            raise RuntimeError(f'Previous stage process group {state["active_pgid"]} is still alive. Inspect it before resuming.')
        context = code_context()
        if not args.fixture and any(not c['revision'] or c['modified_files'] for role,c in context.items() if role != 'environment'):
            raise ValueError('Formal execution requires committed paper/runtime sources. Local task files may remain outside tracked code.')
        if state['code'] is not None and state['code'] != context:
            raise ValueError('Code revision or tracked modifications changed; do not resume this study under different code.')
        inputs = dict(study_path=str(study_path), study=read(study_path), tasks={str(p):p.read_text() for p in task_paths})
        if state['inputs'] is not None:
            if not inputs['tasks']:
                inputs['tasks'] = {p:Path(p).read_text() for p in state['inputs']['tasks']}
            if state['inputs'] != inputs:
                raise ValueError('Study/task configuration changed; use a new prospective result root.')
        state.update(inputs=inputs, code=context, active_pgid=None)
        for stage, command in commands:
            # Retry authorization is not part of the scientific command identity.
            key = shlex.join([v for v in command if v != '--retry-interrupted'])
            previous = state['steps'].get(key)
            if previous and not args.resume:
                raise ValueError(f'{stage} was already attempted. Inspect status and use --resume explicitly.')
            marker = root / MARKERS[stage] if stage in MARKERS else None
            frozen_source = (root/'frozen.json').exists() and stage not in {'test', 'analyze'}
            if previous and previous['status']=='completed' and (stage not in REENTER or frozen_source):
                if (marker is not None and not marker.exists()) or (marker is None and not frozen_source):
                    raise ValueError(f'Completed {stage} marker is missing; controller state is not scientific authority.')
                if marker is not None and marker.suffix=='.json': read(marker)
                print(f'SKIP {stage}: previously completed; no new execution.')
                continue
            if previous and previous['status'] in {'running','interrupted'} and stage not in REENTER | {'calibrate'}:
                raise ValueError(f'{stage} was interrupted outside resumable calibration/tune/fit/test stages. Preserve partial outputs and inspect before proceeding; no automatic deletion.')
            logfile = folder/f'{time.time_ns()}_{stage}.log'
            row = dict(stage=stage,command=command,status='running',started=datetime.now(timezone.utc).isoformat(),
                       log=str(logfile),returncode=None)
            state['steps'][key] = row
            atomic_json(path,state)
            def announce(pgid):
                state['active_pgid'] = pgid
                atomic_json(path,state)
            print('EXEC '+shlex.join(command),flush=True)
            try:
                returncode, interrupted = call_stage(command,logfile,announce)
            except OSError as error:
                row.update(status='failed', returncode=None, error=str(error), finished=datetime.now(timezone.utc).isoformat())
                state['active_pgid']=None
                atomic_json(path,state)
                raise
            row.update(status='interrupted' if interrupted else 'completed' if returncode==0 else 'failed',
                       returncode=returncode,finished=datetime.now(timezone.utc).isoformat())
            state['active_pgid']=None
            atomic_json(path,state)
            with (folder/'commands.jsonl').open('a',encoding='utf-8') as stream:
                stream.write(json.dumps(row,ensure_ascii=False)+'\n')
            if returncode:
                print(f'STOP {stage}: {row["status"]}; original output: {logfile}',file=sys.stderr)
                return returncode
    return 0


def main(argv=None):
    args=parser().parse_args(argv)
    try:
        if args.action=='status':
            state=read(control_dir(args.root.expanduser().resolve())/'state.json')
            for row in state['steps'].values():
                print(f'{row["status"]:12s} {row["stage"]:18s} exit={row["returncode"]} log={row["log"]}')
            print('Process status only; original admission/qualification/evaluation artifacts remain authoritative.')
            return 0
        root,study,tasks,commands=make_plan(args)
        for stage,command in commands: print(f'{stage:18s} {shlex.join(command)}')
        if args.action=='plan':
            print('PLAN ONLY: no data admission, model execution, target read or output directory creation.')
            return 0
        return run(args,root,study,tasks,commands)
    except (OSError,ValueError,RuntimeError) as error:
        print(f'ERROR: {error}',file=sys.stderr)
        return 2


if __name__=='__main__':
    sys.exit(main())
