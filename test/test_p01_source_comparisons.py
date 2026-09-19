"""Actual saved schedule/state checks; synthetic source artifacts only."""
import json
import random
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.p01.verify_source_comparisons import ARMS, LOSS_SWITCHES, RECIPE, verify_runs


@pytest.fixture
def comparison(tmp_path):
    recipe=dict(RECIPE,epochs=2,steps_per_epoch=2)
    seed=42;pair_rng=random.Random(seed+10000);sampling=[]
    for epoch in range(2):
        for step in range(2):
            shift=pair_rng.randint(1,16)
            for draw,(domain,group) in enumerate((('d0','g0'),('d0','g1'),('d1','g0'),('d1','g1'))):
                sampling.append(dict(epoch=epoch,step=step,draw=draw,domain=domain,group_id=group,
                    acquisition_id=domain+group,windows=2,window_ids='0;1',pair_shift=shift))
    data=dict(data=dict(windows_per_unit=2),datasets=[dict(name='constructed',source_domains=['d0','d1'])])
    model=dict(branches=[dict(name='observed',type='envelope')],head_type='linear',use_reference_features=True,head_frobenius_cap=5.)
    access=dict(groups=2,acquisitions=4,windows=8,labelled_acquisitions=4)
    scope=dict(permanent_test_predicted=False,reference_state_unchanged=True,deployment_alpha=0.,
               optimizer_steps=4,supervised_endpoints=2,selected_epoch=0,fit_access=access,selection_access=access)
    vectors=dict(labels=np.array([0,1]),group_ids=np.array(['g0','g1']),acquisition_ids=np.array(['a0','a1']),
                 window_ids=np.array(['0','0']),domains=np.array(['d0','d1']),
                 raw_class_names=np.array(['normal','fault']),candidate_class_names=np.array(['normal','fault']),
                 raw_probs=np.array([[.7,.3],[.4,.6]]),raw_log_probs=np.log([[.7,.3],[.4,.6]]))
    runs={}
    for arm in ARMS:
        path=tmp_path/arm;path.mkdir();runs[(arm,seed)]=path
        command=dict(recipe,seed=seed,pair_seed=seed+10000,dataset='constructed',device='cpu',benchmark_commit='fixture',
                     pair_rng='random.Random; randint(1,pair_shift) once per batch',evaluate_test=False)
        reduction,prior,target,weight=LOSS_SWITCHES[arm]
        config=dict(model=model,loss=dict(reduction=reduction,risk_reference=prior,consistency_target=target,
            lambda_delta=weight,tau=1.,brier_weight=.25,domain_temperature=.25))
        (path/'command.json').write_text(json.dumps(command))
        (path/'model_config.yaml').write_text(yaml.safe_dump(config))
        (path/'data_config.yaml').write_text(yaml.safe_dump(data))
        (path/'result_scope.json').write_text(json.dumps(scope))
        pd.DataFrame(sampling).to_csv(path/'sampling.csv',index=False)
        pd.DataFrame([dict(epoch=e,step=s) for e in range(2) for s in range(2)]).to_csv(path/'training_batches.csv',index=False)
        pd.DataFrame(dict(epoch=[0,1],worst_validation_relative_score=[.1,.1])).to_csv(path/'training.csv',index=False)
        torch.save(dict(weight=torch.arange(6).reshape(2,3).float()),path/'initial_candidate_state.pt')
        np.savez(path/'selected_source_validation_windows.npz',**vectors)
    return runs,recipe


def test_matching_actual_source_execution(comparison):
    runs,recipe=comparison
    report=verify_runs(runs,recipe=recipe)
    assert report['status']=='matched_source_comparison'
    assert report['seeds'][0]['optimizer_steps_per_arm']==4
    assert report['seeds'][0]['sampled_acquisitions_per_arm']==16


@pytest.mark.parametrize('mutation,match',[
    ('state','initial candidate state'),('sampling','schedule differs'),('raw','frozen-reference raw_probs'),
    ('steps','actual optimization steps'),('selector','earliest minimum'),('pair_rng','independent frozen RNG'),
    ('arm','wrong loss intervention'),('access','source access count'),('early_test','permanent test'),
])
def test_real_execution_difference_is_rejected(comparison,mutation,match):
    runs,recipe=comparison;path=runs[('RC',42)]
    if mutation=='state':
        torch.save(dict(weight=torch.ones(2,3)),path/'initial_candidate_state.pt')
    elif mutation=='sampling':
        frame=pd.read_csv(path/'sampling.csv');frame.loc[0,'acquisition_id']='different';frame.to_csv(path/'sampling.csv',index=False)
    elif mutation=='raw':
        file=path/'selected_source_validation_windows.npz'
        with np.load(file) as archive:arrays=dict(archive)
        arrays['raw_probs']=arrays['raw_probs'][:,::-1]
        np.savez(file,**arrays)
    elif mutation=='steps':
        frame=pd.read_csv(path/'training_batches.csv');frame.iloc[:-1].to_csv(path/'training_batches.csv',index=False)
    elif mutation=='pair_rng':
        frame=pd.read_csv(path/'sampling.csv');frame.loc[:3,'pair_shift']=1;frame.to_csv(path/'sampling.csv',index=False)
    elif mutation=='arm':
        file=path/'model_config.yaml';config=yaml.safe_load(file.read_text());config['loss']['consistency_target']='candidate';file.write_text(yaml.safe_dump(config))
    else:
        file=path/'result_scope.json';scope=json.loads(file.read_text())
        if mutation=='selector':scope['selected_epoch']=1
        elif mutation=='access':scope['fit_access']=dict(scope['fit_access'],groups=3)
        elif mutation=='early_test':scope['permanent_test_predicted']=True
        file.write_text(json.dumps(scope))
    with pytest.raises(ValueError,match=match):verify_runs(runs,recipe=recipe)


def test_missing_arm_is_not_a_matched_comparison(comparison):
    runs,recipe=comparison;runs.pop(('UO',42))
    with pytest.raises(ValueError,match='exactly O, UO, RO and RC'):verify_runs(runs,recipe=recipe)
