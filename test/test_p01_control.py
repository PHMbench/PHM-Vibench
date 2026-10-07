"""Controller execution semantics; no industrial dataset or accuracy claim."""
from pathlib import Path
import copy
import json
import os
import signal
import subprocess
import sys
import threading
import time

import pytest
import yaml

from experiments.p01 import control as ctl
from experiments.p01 import multiview_dg as dg
from test.test_p01_multiview_dg import task_source, fixture_study  # Existing constructed task, not another benchmark.


@pytest.fixture
def setup(tmp_path):
    study=fixture_study(); study['leave_one_view_out']=True
    path=tmp_path/'study with spaces.yaml'; path.write_text(yaml.safe_dump(study))
    task=tmp_path/'task.yaml'; task.write_text(yaml.safe_dump({'name':'one'}))
    args=['--root',str(tmp_path/'run with spaces'),'--study',str(path),'--task',str(task),'--fixture','--device','cpu']
    return args,study,path,task


def parsed(setup, *extra, action='plan'):
    return ctl.parser().parse_args([action,*setup[0],*extra])


def test_plan_is_read_only_and_default_stops_before_target(setup):
    args=parsed(setup)
    root,_,_,commands=ctl.make_plan(args)
    assert [s for s,_ in commands]==list(ctl.STAGES[:ctl.STAGES.index('freeze')+1])
    assert all('test' not in c for _,c in commands)
    assert not root.exists() and not ctl.control_dir(root).exists()
    assert ctl.main(['plan',*setup[0]])==0
    assert not root.exists() and not ctl.control_dir(root).exists()
    assert commands[0][1][-2:] == [str(setup[3]),'--fixture']


def test_full_run_requires_explicit_target_permission(setup):
    with pytest.raises(ValueError,match='allow-target-read'):
        ctl.make_plan(parsed(setup,'--through','analyze',action='run'))
    _,_,_,commands=ctl.make_plan(parsed(setup,'--through','analyze','--allow-target-read',action='run'))
    assert [s for s,_ in commands][-3:]==['freeze','test','analyze']


@pytest.mark.parametrize('options', [('--arms','invented'),('--seeds','7'),('--task-names','missing'),
    ('--only','unknown'),('--experiments','E7'),('--arms','I','--through','test'),
    ('--only','fit-method','--from-stage','fit-method'),('--retry-interrupted',)])
def test_invalid_controls_do_not_create_outputs(setup,options):
    with pytest.raises(ValueError):ctl.make_plan(parsed(setup,*options))
    root=Path(setup[0][1]);assert not root.exists() and not ctl.control_dir(root).exists()


def test_experiment_selection_reuses_existing_family_commands(setup):
    args=parsed(setup,'--only','tune-method,fit-method','--experiments','E1,E5,E6','--seeds','42','--task-names','one')
    _,_,_,commands=ctl.make_plan(args)
    tune,fit=[c for _,c in commands]
    assert tune[:4]==[sys.executable,'-u','-m','experiments.p01.multiview_dg']
    assert '--seed' not in tune
    def flags(command,flag): return [command[i+1] for i,v in enumerate(command) if v==flag]
    assert flags(tune,'--arm')==['I','I-base']
    assert flags(fit,'--arm')==['I','I-base','I-minus-stft_short','I-minus-envelope']
    assert flags(fit,'--seed')==['42']
    assert flags(fit,'--task-name')==['one']
    assert '--fixture' not in fit  # The already bound fixture flag belongs to bind only.


def test_partial_experiments_keep_all_required_baselines(setup):
    _,_,_,commands=ctl.make_plan(parsed(setup,'--experiments','E1'))
    for stage,c in commands:
        if stage in {'fit-baselines','tune-baselines'}:
            assert [c[i+1] for i,v in enumerate(c) if v=='--arm']==list(setup[1]['baselines'])
    assert commands[-1][0]=='fit-method'


def test_run_failure_stops_and_resume_reenters_original_command(setup,monkeypatch):
    args=parsed(setup,'--only','tune-baselines,fit-baselines',action='run')
    root,study,tasks,commands=ctl.make_plan(args)
    monkeypatch.setattr(ctl,'code_context',lambda: {'runtime':{'revision':'same','modified_files':[]}})
    calls=[]
    def failing(command,log,announce):
        calls.append(command);announce(None);log.write_text('original trainer failure')
        return 17,None
    monkeypatch.setattr(ctl,'call_stage',failing)
    assert ctl.run(args,root,study,tasks,commands)==17 and len(calls)==1
    state=ctl.read(ctl.control_dir(root)/'state.json')
    assert list(state['steps'].values())[0]['status']=='failed'
    def succeeds(command,log,announce):
        calls.append(command);announce(None);log.write_text('original command ran');return 0,None
    monkeypatch.setattr(ctl,'call_stage',succeeds)
    args.resume=True
    assert ctl.run(args,root,study,tasks,commands)==0
    assert calls[0]==calls[1] and len(calls)==3
    setup[2].write_text(setup[2].read_text()+'\n# formatting only\n')  # Semantic study values unchanged.
    assert ctl.run(args,root,study,tasks,commands)==0
    setup[3].write_text('name: changed\n')
    with pytest.raises(ValueError,match='configuration changed'):ctl.run(args,root,study,tasks,commands)


def test_clean_completion_marker_is_not_a_scientific_bypass(setup,monkeypatch):
    args=parsed(setup,'--only','preflight',action='run')
    root,study,tasks,commands=ctl.make_plan(args)
    monkeypatch.setattr(ctl,'code_context',lambda: {})
    def succeeds(command,log,announce):
        root.mkdir(exist_ok=True);(root/'preflight.json').write_text('{"status":"passed"}')
        log.write_text('done');return 0,None
    monkeypatch.setattr(ctl,'call_stage',succeeds)
    assert ctl.run(args,root,study,tasks,commands)==0
    args.resume=True
    monkeypatch.setattr(ctl,'call_stage',lambda *_: pytest.fail('Do not rerun non-idempotent completed preflight'))
    assert ctl.run(args,root,study,tasks,commands)==0
    (root/'preflight.json').unlink()
    with pytest.raises(ValueError,match='marker is missing'):ctl.run(args,root,study,tasks,commands)


def test_exclusive_controller_lock(tmp_path):
    with ctl.lock(tmp_path/'lock'):
        with pytest.raises(RuntimeError,match='Another controller'):
            with ctl.lock(tmp_path/'lock'):pass


def test_process_log_and_exit_code(tmp_path):
    command=[sys.executable,'-c','print("retained stdout"); raise SystemExit(9)']
    ids=[]
    rc,interrupted=ctl.call_stage(command,tmp_path/'stage.log',ids.append)
    assert (rc,interrupted)==(9,None) and ids
    assert (tmp_path/'stage.log').read_text().strip()=='retained stdout'


def test_signal_reaches_stage_and_its_trainer_child(tmp_path):
    # Use a separate controller process so pytest itself cannot be interrupted.
    script=tmp_path/'interrupt.py'
    ready=tmp_path/'ready';child_done=tmp_path/'child_done'
    child_code=f'import time,signal; from pathlib import Path; signal.signal(signal.SIGTERM,lambda *a:(Path({str(child_done)!r}).write_text("term"),exit(0))); Path({str(ready)!r}).write_text("ready"); time.sleep(60)'
    stage_code=f'import subprocess,sys; subprocess.run([sys.executable,"-c",{child_code!r}])'
    script.write_text('from experiments.p01.control import call_stage\nimport sys\nfrom pathlib import Path\n'
                     +f'rc,_=call_stage([sys.executable,"-c",{stage_code!r}],Path({str(tmp_path/"signals.log")!r}),lambda pid:None)\nsys.exit(rc)\n')
    proc=subprocess.Popen([sys.executable,str(script)],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    try:
        deadline=time.monotonic()+10
        while not ready.exists() and time.monotonic()<deadline:time.sleep(.02)
        assert ready.exists()
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=12)==143
        assert child_done.read_text()=='term'
    finally:
        if proc.poll() is None:proc.kill();proc.wait()


def test_controller_bind_preflight_uses_real_runtime_and_preserves_target_embargo(task_source,tmp_path):
    task_path,_=task_source
    study=fixture_study();study_path=tmp_path/'study.yaml';study_path.write_text(yaml.safe_dump(study))
    root=tmp_path/'actual'
    argv=['run','--root',str(root),'--study',str(study_path),'--task',str(task_path),'--fixture',
          '--device','cpu','--through','preflight']
    assert ctl.main(argv)==0
    assert ctl.read(root/'preflight.json')['status']=='passed'
    assert not (root/'fixture_to_2'/'test.csv').exists()
    assert not (root/'frozen.json').exists()
    assert ctl.main([*argv,'--resume'])==0


def test_filtered_fit_never_changes_complete_study_or_baseline_gate(tmp_path,monkeypatch):
    study=fixture_study();study['leave_one_view_out']=True
    task=dict(name='a',path=str(tmp_path/'a'));other=dict(name='b',path=str(tmp_path/'b'))
    info=dict(tasks=[task,other]);before=copy.deepcopy(study);gate=[];calls=[]
    monkeypatch.setattr(dg,'_development_guard',lambda *a:(tmp_path,study,info))
    monkeypatch.setattr(dg,'_training_study',lambda root,study,info:study)
    monkeypatch.setattr(dg,'_require_qualified_baselines',lambda root,study,info:gate.append(info))
    monkeypatch.setattr(dg,'_reference',lambda t:Path(t['path'])/'p0')
    selected=tmp_path/'a'/'hpo'/'I';selected.mkdir(parents=True)
    dg.dump(selected/'selection.json',dict(trial=study['trials'][0],view=None))
    monkeypatch.setattr(dg,'_execute_or_resume',lambda *a,**kw:calls.append((a,kw)))
    dg.fit(tmp_path,'cpu','method',task_names=['a'],selected_arms=['I','I-minus-envelope'],selected_seeds=[42])
    assert len(calls)==2 and all(c[0][4]==42 for c in calls)
    assert [c[0][2] for c in calls]==['I','I-minus-envelope']
    assert gate==[info] and study==before and len(info['tasks'])==2
    with pytest.raises(ValueError,match='arms'):
        dg.fit(tmp_path,'cpu','method',selected_arms=['invented'])


def interrupted_run(tmp_path,monkeypatch,status='running',failure=None):
    task=dict(name='a',path=str(tmp_path/'a'));study=fixture_study();trial=study['trials'][0]
    out=Path(task['path'])/'fits'/'I'/'42';out.mkdir(parents=True)
    cfg={'model':{'name':'actual'}}
    monkeypatch.setattr(dg,'candidate_config',lambda *a:cfg)
    (out.parent/'42.yaml').write_text(yaml.safe_dump(cfg));(out.parent/'42.log').write_text('preserve this attempt')
    dg.dump(out/'result_scope.json',dict(status=status))
    dg.dump(out/'command.json',dict(trial,seed=42,epochs=study['epochs'],steps_per_epoch=study['steps_per_epoch'],units_per_domain=study['units_per_domain'],dg=True))
    if failure:dg.dump(out/'failure.json',failure)
    return study,task,trial,out


def test_interrupted_retry_preserves_original_and_restarts_same_recipe(tmp_path,monkeypatch):
    study,task,trial,out=interrupted_run(tmp_path,monkeypatch)
    calls=[]
    monkeypatch.setattr(dg,'_execute',lambda *args:calls.append(args))
    with pytest.raises(ValueError):dg._execute_or_resume(study,task,'I',trial,42,out,'cpu',None,None)
    dg._execute_or_resume(study,task,'I',trial,42,out,'cpu',None,None,retry_interrupted=True)
    assert len(calls)==1 and calls[0][2:6]==('I',trial,42,out)
    archives=list((Path(task['path'])/'interrupted').glob('*'))
    assert len(archives)==1 and (archives[0]/'42.log').read_text()=='preserve this attempt'
    assert (archives[0]/'42'/'command.json').exists() and (archives[0]/'restart.json').exists()


@pytest.mark.parametrize('change', ['failure','recipe'])
def test_failed_or_changed_trial_is_never_automatically_retried(tmp_path,monkeypatch,change):
    study,task,trial,out=interrupted_run(tmp_path,monkeypatch,failure={'error':'nonfinite loss'} if change=='failure' else None)
    if change=='recipe':
        command=dg.read(out/'command.json');command['seed']=123
        (out/'command.json').write_text(json.dumps(command))
    monkeypatch.setattr(dg,'_execute',lambda *a:pytest.fail('No alternate or numerical-failure rerun'))
    with pytest.raises(ValueError):dg._execute_or_resume(study,task,'I',trial,42,out,'cpu',None,None,retry_interrupted=True)
    assert (out/'command.json').exists()


def test_resume_after_freeze_never_restarts_source_development(setup,monkeypatch):
    args=parsed(setup,action='run')
    root,study,tasks,commands=ctl.make_plan(args)
    monkeypatch.setattr(ctl,'code_context',lambda: {})
    calls=[]
    def execute(command,log,announce):
        stage=next(s for s,c in commands if c==command)
        calls.append(stage);root.mkdir(exist_ok=True);log.write_text(stage)
        if stage in ctl.MARKERS:
            path=root/ctl.MARKERS[stage];path.parent.mkdir(parents=True,exist_ok=True)
            path.write_text('{}' if path.suffix=='.json' else 'condition')
        return 0,None
    monkeypatch.setattr(ctl,'call_stage',execute)
    assert ctl.run(args,root,study,tasks,commands)==0
    original=list(calls)
    args=parsed(setup,'--through','analyze','--allow-target-read','--resume',action='run')
    root,study,tasks,commands=ctl.make_plan(args)
    assert ctl.run(args,root,study,tasks,commands)==0
    assert calls==original+['test','analyze']


def test_controller_core_matches_runtime():
    assert ctl.CORE==dg.CORE
