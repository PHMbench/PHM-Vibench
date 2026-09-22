"""Installed Model Factory and source-only H5 execution of the new opt-in mode.

All signals/checkpoints here are constructed fixtures, never industrial results.
"""
import json
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from src.model_factory.model_factory import model_factory
from experiments.p01.fusion_deployment import load_model, predict_records, verify_vectors
from experiments.p01 import train_tspn_fusion_v2 as trainer
from experiments.p01.preflight_fusion import support_rows


def envelope():
    return dict(name='envelope',type='envelope',
        carrier=dict(centers=[.2],widths=[.03],center_bounds=[[.18,.22]],width_bounds=[[.02,.04]]),
        modulation=dict(centers=[.08],widths=[.01],center_bounds=[[.07,.09]],width_bounds=[[.008,.014]]),
        diagnostics=dict(lags=[1,4,8],epsilon=1e-8))


@pytest.fixture
def runtime(tmp_path):
    ref=dict(type='X_model',name='TSPN',device='cpu',num_classes=2,
             in_channels=1,out_channels=12,scale=1,in_dim=128,out_dim=128,
             skip_connection=False,internal_instance_normalization=False,
             signal_processing_configs={'layer1':['I','HT','WF']},
             feature_extractor_configs=['RMS','Std','AbsMean'],
             f_c_mu=.2,f_c_sigma=.01,f_b_mu=.05,f_b_sigma=.002)
    torch.manual_seed(10)
    reference=model_factory(SimpleNamespace(**ref),metadata=None)
    # Constant equal logits: the software qualification test must reject a
    # balanced 50%-accuracy reference. No real trained baseline is fabricated.
    for module in reference.clf.modules():
        if isinstance(module,torch.nn.Linear):
            torch.nn.init.zeros_(module.weight)
            if module.bias is not None:torch.nn.init.zeros_(module.bias)
    checkpoint=tmp_path/'reference.pt'
    torch.save({'model':ref,'state_dict':reference.state_dict()},checkpoint)
    cfg=dict(model=dict(type='X_model',name='TSPN_fusion',device='cpu',num_classes=2,
             checkpoint_kind='reference',checkpoint_path=str(checkpoint),reference_config=ref,
             reference_temperature=1.3,head_frobenius_cap=5.,head_type='operator_residual',
             head_hidden_dim=16,use_reference_features=True,branches=[envelope()]),
             loss=dict(tau=1.,brier_weight=.25,domain_temperature=.25,lambda_delta=0.,
                       reduction='mean_source',risk_reference='relative',consistency_target='correction'))
    h5=tmp_path/'signals.h5';rows=[]
    with h5py.File(h5,'w') as signals:
        for domain in ('0','1','2'):
            for split in ('update','validation','assessment','test'):
                for label in range(2):
                    key=str(len(rows)+1)
                    x=np.sin(np.arange(256)*(.06+.11*label)).astype(np.float32)[:,None]
                    if split in {'assessment','test'}: x[:]=np.nan
                    signals.create_dataset(key,data=x)
                    rows.append(dict(Id=key,group_id=f'{split}_{label}',split=split,
                                     label=label,domain=domain,fs=64000,rpm=1500))
    metadata=tmp_path/'metadata.csv';pd.DataFrame(rows).to_csv(metadata,index=False)
    data=dict(model=dict(num_classes=2,class_names=['healthy','fault']),
              data=dict(layout='LC',window_size=128,windows_per_unit=2),
              datasets=[dict(name='constructed',format='vibench_h5',metadata_file=str(metadata),
                  h5_file=str(h5),source_domains=['0','1'],domain_sequence=['2'],columns=dict(
                  id='Id',unit_id='group_id',split='split',label='label',domain='domain',
                  sample_rate_hz='fs',rotation_speed_rpm='rpm'))])
    for name,value in [('model',cfg),('data',data)]:
        (tmp_path/f'{name}.yaml').write_text(yaml.safe_dump(value))
    return cfg,data


def run_args(tmp_path,output,extra=()):
    return ['train','--model-config',str(tmp_path/'model.yaml'),'--data-config',str(tmp_path/'data.yaml'),
            '--dataset','constructed','--output',str(tmp_path/output),'--device','cpu','--seed','42',
            '--epochs','1','--steps-per-epoch','3','--units-per-domain','2','--pair-shift','4',*extra]


def test_real_factory_train_export_and_strict_reload(runtime,tmp_path,monkeypatch):
    cfg,data=runtime
    monkeypatch.setattr(sys,'argv',run_args(tmp_path,'fit'))
    trainer.main()
    output=tmp_path/'fit'
    scope=json.loads((output/'result_scope.json').read_text())
    assert scope['reference_state_unchanged'] and not scope['permanent_test_predicted']
    assert scope['head_type']=='operator_residual' and scope['optimizer_steps']==3
    saved=torch.load(output/'selected_candidate.pt',weights_only=True)
    assert saved['model']['head_type']=='operator_residual'
    model,_=load_model(output/'selected_candidate.pt','cpu')
    restored=model_factory(SimpleNamespace(**dict(saved['model'],checkpoint_kind='fusion',
                                checkpoint_path=str(output/'selected_candidate.pt'),device='cpu')),metadata=None).eval()
    x=torch.randn(3,128,1)
    for key in ('raw_logits','candidate_logits'):
        torch.testing.assert_close(model.forward_details(x)[key],restored.forward_details(x)[key],atol=0,rtol=0)
    with np.load(output/'selected_source_validation_windows.npz',allow_pickle=False) as arr:
        names=[k for k in arr.files if k.startswith('logit_contribution__')]
        assert names==['logit_contribution__reference','logit_contribution__envelope']
        correction=sum(arr[k] for k in names)
        raw=arr['raw_log_probs'];candidate=arr['candidate_log_probs']
        np.testing.assert_allclose((candidate[:,0]-candidate[:,1])-(raw[:,0]-raw[:,1]),
                                    correction[:,0]-correction[:,1],atol=3e-7,rtol=2e-5)
        assert set(arr['domains'])=={'0','1'}
        assert all(g.startswith('validation_') for g in arr['group_ids'])
    # Same actual waveform access through canonical export; protected arrays are
    # NaN and would fail immediately if the source operation read them.
    records=trainer.read_records(data['datasets'][0],data)
    val=[r for r in records if r['domain'] in {'0','1'} and r['split']=='validation']
    one=predict_records(model,val,data['datasets'][0],data,['healthy','fault'],'cpu',alpha=1.)
    two=predict_records(restored,val,data['datasets'][0],data,['healthy','fault'],'cpu',alpha=1.)
    verify_vectors(one,two)
    assert np.isfinite(one['candidate_probs']).all()


def test_reference_competence_gate_blocks_before_optimizer(runtime,tmp_path,monkeypatch):
    def no_optimizer(*_,**__):
        raise AssertionError('An unqualified reference must not start candidate optimization.')
    monkeypatch.setattr(trainer.torch.optim,'Adam',no_optimizer)
    monkeypatch.setattr(sys,'argv',run_args(tmp_path,'blocked',['--reference-min-accuracy','.8']))
    with pytest.raises(ValueError,match='No candidate was trained'):
        trainer.main()
    result=json.loads((tmp_path/'blocked/reference_qualification.json').read_text())
    assert not result['passed'] and not result['independent_test_guarantee']
    assert all(row['accuracy']==.5 for row in result['source_conditions'])
    assert not (tmp_path/'blocked/selected_candidate.pt').exists()


def test_preflight_validates_diagnostic_lags(runtime):
    cfg,_=runtime
    rows=support_rows(cfg['model'],128,64000.)
    lags=[r for r in rows if r['component']=='envelope_correlation']
    assert len(lags)==3 and lags[-1]['valid_positions']==120
    cfg['model']['branches'][0]['diagnostics']['lags']=[127]
    with pytest.raises(ValueError,match='two observed pairs'):
        support_rows(cfg['model'],128,64000.)


def test_initial_incumbent_retained_on_adverse_source_scores(runtime,tmp_path,monkeypatch):
    original=trainer.evaluate
    calls=[]
    def adverse_after_initial(*args,**kwargs):
        score,rows,summary=original(*args,**kwargs)
        calls.append(score)
        return (score if len(calls)==1 else 10.),rows,summary
    monkeypatch.setattr(trainer,'evaluate',adverse_after_initial)
    monkeypatch.setattr(sys,'argv',run_args(tmp_path,'incumbent'))
    trainer.main()
    scope=json.loads((tmp_path/'incumbent/result_scope.json').read_text())
    assert scope['selected_epoch']==-1 and scope['reference_incumbent_retained']
    with np.load(tmp_path/'incumbent/selected_source_validation_windows.npz') as p:
        np.testing.assert_array_equal(p['candidate_probs'],p['raw_probs'])
