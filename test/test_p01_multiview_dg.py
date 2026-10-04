"""DG boundaries and the actual fixed P01 implementation; no private observations."""
from pathlib import Path
import copy
import json
from types import SimpleNamespace

import h5py
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

from experiments.p01 import multiview_dg as dg
from experiments.p01.fusion_data import read_records
from experiments.p01 import analyze_d1
from src.model_factory.model_factory import model_factory


@pytest.fixture
def task_source(tmp_path):
    rows=[];assign=[]
    with h5py.File(tmp_path/'signals.h5','w') as h5:
        index=0
        for split in ('update','validation','test'):
            for domain in ('0','1','2'):
                for label in (0,1):
                    index+=1
                    time=np.arange(1024,dtype=np.float32)
                    signal=(1+4*label)*np.sin(2*np.pi*(.08+.01*int(domain))*time)
                    h5[str(index)]=signal[:,None]
                    rows.append(dict(Id=str(index),Label=str(label),Domain=str(domain),Fs='12000',RPM='1000'))
                    assign.append(dict(Id=str(index),Unit=f'{split}_{label}',Split=split))
    pd.DataFrame(rows).to_csv(tmp_path/'metadata.csv',index=False)
    pd.DataFrame(assign).to_csv(tmp_path/'protocol.csv',index=False)
    task=dict(name='fixture_to_2',dataset_id='constructed',physical_group_basis='Generated independent source/test units',domain_basis='Generated frequency',
              model=dict(num_classes=2,class_names=['healthy','fault']),
              data=dict(layout='LC',window_size=512,windows_per_unit=1,channel_indices=[0]),
              dataset=dict(name='fixture',format='vibench_h5',metadata_file='metadata.csv',h5_file='signals.h5',protocol_file='protocol.csv',
                  columns=dict(id='Id',label='Label',domain='Domain',unit_id='Unit',split='Split',sample_rate_hz='Fs',rotation_speed_rpm='RPM'),
                  source_domains=['0','1'],domain_sequence=['2']))
    (tmp_path/'task.yaml').write_text(yaml.safe_dump(task))
    return tmp_path/'task.yaml',task


def fixture_study():
    ref=dict(in_channels=1,in_dim=512,out_dim=512,out_channels=4,scale=1,skip_connection=False,
             internal_instance_normalization=False,signal_processing_configs={'layer1':['I']},
             feature_extractor_configs=['RMS','Std','AbsMean'],f_c_mu=.18,f_c_sigma=.01,f_b_mu=.04,f_b_sigma=.001)
    return dict(schema_version=1,min_datasets=3,min_splits_per_dataset=2,seeds=[42,123],hpo_seed=20261003,
       epochs=20,steps_per_epoch=10,units_per_domain=2,pair_shift=8,overfit_steps=150,reference_min_accuracy=.8,
       trials=[dict(lr=.003,weight_decay=0.,scheduler='none'),dict(lr=.001,weight_decay=0.,scheduler='none')],
       reference=ref,baselines={'TCN':dict(type='CNN',name='TCN',num_channels=[8,8],kernel_size=3,dropout=0.)},
       fusion=dict(model=dict(type='X_model',name='TSPN_fusion',head_type='operator_residual',head_hidden_dim=16,
            use_reference_features=True,reference_temperature=1.,head_frobenius_cap=5.,
            branches=[dict(name='stft_short',type='stft',transform=dict(win_length=64,n_fft=64,hop_length=16),readout=dict(row_groups=4,time_bins=2,log_floor=.001)),
                      dict(name='envelope',type='envelope',carrier=dict(centers=[.19],widths=[.025],center_bounds=[[.16,.22]],width_bounds=[[.015,.035]]),
                           modulation=dict(centers=[.06],widths=[.0045],center_bounds=[[.056,.064]],width_bounds=[[.003,.006]]),diagnostics=dict(lags=[16],epsilon=1e-8))]),
            loss=dict(tau=1.,brier_weight=.25,domain_temperature=.25,lambda_delta=0.,reduction='mean_source',risk_reference='relative',consistency_target='correction')))


def bind_fixture(tmp_path, task_source):
    study=fixture_study();(tmp_path/'study.yaml').write_text(yaml.safe_dump(study))
    return dg.bind(tmp_path/'study.yaml',[task_source[0]],tmp_path/'suite',fixture=True)


def test_bind_never_reads_target_labels_or_payloads(task_source,tmp_path):
    path,task=task_source
    frame=pd.read_csv(tmp_path/'metadata.csv',dtype=str)
    frame.loc[frame.Domain=='2','Label']='DO_NOT_PARSE_TARGET_LABEL'
    frame.to_csv(tmp_path/'metadata.csv',index=False)
    root=bind_fixture(tmp_path,task_source)
    _,_,info=dg.suite(root)
    _,dataset,records=dg.source_data(info['tasks'][0])
    assert set(r['domain'] for r in records)=={'0','1'}
    assert set(r['split'] for r in records)=={'update','validation'}
    assert 'DO_NOT_PARSE' not in Path(dataset['metadata_file']).read_text()
    assert 'Label' not in pd.read_csv(root/'fixture_to_2'/'test_structure.csv')
    dg.preflight(root)
    assert not (root/'fixture_to_2'/'test.csv').exists()


def test_physical_units_cannot_cross_partitions(task_source,tmp_path):
    protocol=pd.read_csv(tmp_path/'protocol.csv',dtype=str)
    protocol.loc[0,'Split']='test';protocol.to_csv(tmp_path/'protocol.csv',index=False)
    with pytest.raises(ValueError,match='crosses partitions'):bind_fixture(tmp_path,task_source)


def test_true_dg_requires_two_sources(task_source,tmp_path):
    path,task=task_source;task['dataset']['source_domains']=['0'];path.write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError,match='>=2'):bind_fixture(tmp_path,task_source)


def test_source_only_loader_rejects_holdout_before_h5(task_source,tmp_path,monkeypatch):
    root=bind_fixture(tmp_path,task_source);_,_,info=dg.suite(root)
    data,dataset,_=dg.source_data(info['tasks'][0]);frame=pd.read_csv(dataset['metadata_file'],dtype=str)
    frame.loc[0,'Split']='test';frame.loc[0,'Label']='poison';frame.to_csv(dataset['metadata_file'],index=False)
    dataset['h5_file']=str(tmp_path/'DO_NOT_OPEN.h5')
    with pytest.raises(ValueError,match='before labels or H5'):read_records(dataset,data)


def test_no_preprocessing_silent_ignore(task_source,tmp_path):
    path,task=task_source;task['data']['normalization']='target_zscore';path.write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError,match='normalization'):bind_fixture(tmp_path,task_source)


def test_study_mutation_and_test_before_freeze_are_blocked(task_source,tmp_path):
    root=bind_fixture(tmp_path,task_source)
    with pytest.raises(FileNotFoundError):dg.test(root,'cpu')
    study=dg.read(root/'study.yaml');study['epochs']+=1;(root/'study.yaml').write_text(yaml.safe_dump(study))
    with pytest.raises(ValueError,match='changed after binding'):dg.suite(root)


@pytest.mark.parametrize('name,kind,options',[
 ('ResNet1D','CNN',dict(input_dim=1,layers=[2,2,2,2],initial_channels=64,block_type='basic')),
 ('TCN','CNN',dict(input_dim=1,num_channels=[32,32,32],kernel_size=3,dropout=.1)),
 ('PatchTST','Transformer',dict(input_dim=1,d_model=64,n_heads=4,num_layers=2,d_ff=128,patch_size=16,stride=8)),
 ('BASE_ExplainableCNN','X_model',dict(in_channels=1,width=32,dropout=.1)),
])
def test_baseline_factory_gradient_strict_restore(name,kind,options,tmp_path):
    torch.set_num_threads(1);torch.manual_seed(42)
    config=SimpleNamespace(type=kind,name=name,num_classes=3,device='cpu',**options)
    model=model_factory(config,None);x=torch.randn(4,512,1);target=torch.tensor([0,1,2,1])
    logits=model(x);assert logits.shape==(4,3)
    torch.nn.functional.cross_entropy(logits,target).backward()
    assert any(p.grad is not None and p.grad.abs().sum()>0 for p in model.parameters())
    model.eval();expected=model(x).detach();torch.save(model.state_dict(),tmp_path/'state.pt')
    restored=model_factory(config,None);restored.load_state_dict(torch.load(tmp_path/'state.pt',weights_only=True),strict=True);restored.eval()
    torch.testing.assert_close(expected,restored(x),rtol=1e-6,atol=1e-7)


def test_small_suite_end_to_end(task_source,tmp_path):
    root=bind_fixture(tmp_path,task_source)
    dg.preflight(root);dg.smoke(root,'baseline','cpu')
    dg.tune(root,'reference','cpu');dg.smoke(root,'method','cpu')
    dg.tune(root,'baselines','cpu');dg.tune(root,'method','cpu');dg.fit(root,'cpu')
    assert not (root/'fixture_to_2'/'test.csv').exists()
    dg.freeze(root,'cpu')
    with pytest.raises(ValueError,match='frozen'):dg.fit(root,'cpu')
    dg.test(root,'cpu');dg.analyze(root)
    contrasts=pd.read_csv(root/'summary'/'contrasts.csv')
    assert set(contrasts.contrast)==set(dg.CONTRASTS)|{'I-TCN'}
    assert set(contrasts.status)=={'complete'}
    rec=pd.read_csv(root/'summary'/'reconstruction.csv')
    assert (rec.max_abs<rec.tolerance).all()
    assert dg.read(root/'summary'/'scope.json')['fixture'] is True


def test_missing_physical_assignment_is_not_silently_dropped(task_source,tmp_path):
    protocol=pd.read_csv(tmp_path/'protocol.csv',dtype=str)
    protocol.iloc[1:].to_csv(tmp_path/'protocol.csv',index=False)
    with pytest.raises(ValueError,match='Every selected acquisition'):bind_fixture(tmp_path,task_source)


def test_changed_source_metadata_is_rejected(task_source,tmp_path):
    root=bind_fixture(tmp_path,task_source)
    path=root/'fixture_to_2'/'source.csv'
    path.write_text(path.read_text()+'\n')
    with pytest.raises(ValueError,match='Bound source/task assignment'):dg.suite(root)


def test_different_target_sampling_convention_is_rejected(task_source,tmp_path):
    path=tmp_path/'metadata.csv';data=pd.read_csv(path,dtype=str)
    data.loc[data.Domain=='2','Fs']='48000';data.to_csv(path,index=False)
    with pytest.raises(ValueError,match='sampling convention'):bind_fixture(tmp_path,task_source)


def test_core_alias_cannot_be_shadowed():
    study=fixture_study();study['baselines']['I']=study['baselines']['TCN']
    with pytest.raises(ValueError,match='shadow'):dg.validate_study(study)


def test_duplicate_split_is_not_an_additional_experiment(task_source,tmp_path):
    path,task=task_source;other=copy.deepcopy(task);other['name']='renamed_same_split'
    second=tmp_path/'second.yaml';second.write_text(yaml.safe_dump(other))
    config=tmp_path/'study.yaml';config.write_text(yaml.safe_dump(fixture_study()))
    with pytest.raises(ValueError,match='Duplicate domain splits'):
        dg.bind(config,[path,second],tmp_path/'suite',fixture=True)


def test_same_observations_cannot_count_as_two_datasets(task_source,tmp_path):
    path,task=task_source;other=copy.deepcopy(task);other.update(name='alias_task',dataset_id='not_a_new_dataset')
    second=tmp_path/'second.yaml';second.write_text(yaml.safe_dump(other))
    config=tmp_path/'study.yaml';config.write_text(yaml.safe_dump(fixture_study()))
    with pytest.raises(ValueError,match='relabeled as multiple datasets'):
        dg.bind(config,[path,second],tmp_path/'suite',fixture=True)


def test_source_class_profile_uses_weighted_physical_groups():
    p=np.array([[.9,.1],[.2,.8],[.8,.2]])
    arrays=dict(raw_probs=p,candidate_probs=p,raw_log_probs=np.log(p),candidate_log_probs=np.log(p),
       labels=np.array([0,1,1]),group_ids=np.array(['g0','g1','g1']),
       acquisition_ids=np.array(['a0','a1','a2']),window_ids=np.array(['0','0','0']),domains=np.array(['0','0','0']))
    profile=dg.class_profile(arrays)['0']
    assert profile['weighted_support']==[.5,.5]
    assert profile['recall']==[1.,.5]
    assert not profile['class_collapse']


def test_source_selector_retains_literal_domain_ids(tmp_path):
    pd.DataFrame(dict(domain=['01','1','1','01'],predictor=['candidate','candidate','raw','raw'],
                      ce=[.1,.4,.1,.4],brier=[0.,0.,0.,0.])).to_csv(tmp_path/'selected_source_validation.csv',index=False)
    assert dg._score(tmp_path)==pytest.approx(.3)
