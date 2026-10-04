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
from src.data_factory.H5DataDict import H5DataDict
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
                    # The same physical specimen across conditions has one split.
                    rows.append(dict(Id=str(index),Label=str(label),Domain=str(domain),Fs='12000',
                                     RPM=str({'0':1500,'1':1200,'2':900}[domain])))
                    assign.append(dict(Id=str(index),Unit=f'{split}_{label}',Split=split))
    pd.DataFrame(rows).to_csv(tmp_path/'metadata.csv',index=False)
    pd.DataFrame(assign).to_csv(tmp_path/'protocol.csv',index=False)
    evidence=dict(source='test/test_p01_multiview_dg.py',location='task_source fixture',
                  statement='Synthetic independent specimen identities and RPM settings are assigned by this fixture; no real-data scientific claim.')
    physical_conditions=dict(
        variables=dict(rpm=dict(unit='rpm',kind='setpoint',quantity='shaft rotational speed')),
        mapping=dict(kind='documented_metadata',source_column='Domain',evidence=copy.deepcopy(evidence)),
        conditions={cid:dict(values=dict(rpm=rpm),evidence=copy.deepcopy(evidence))
                    for cid,rpm in [('0',1500),('1',1200),('2',900)]})
    task=dict(name='fixture_to_2',dataset_id='constructed',
              protocol='specimen_disjoint_condition_dg',physical_conditions=physical_conditions,
              specimen_basis=dict(unit='physical_specimen',definition='One generated physical specimen per partition and class',evidence=evidence),
              physical_group_basis='Generated independent source/test specimens',domain_basis='Generated documented RPM settings',
              model=dict(num_classes=2,class_names=['healthy','fault']),
              data=dict(layout='LC',window_size=512,windows_per_unit=1,channel_indices=[0]),
              dataset=dict(name='fixture',format='vibench_h5',metadata_file='metadata.csv',h5_file='signals.h5',protocol_file='protocol.csv',
                  rotation_speed_kind='setpoint',
                  columns=dict(id='Id',label='Label',domain='Domain',unit_id='Unit',split='Split',sample_rate_hz='Fs',rotation_speed_rpm='RPM'),
                  source_domains=['0','1'],domain_sequence=['2']))
    (tmp_path/'task.yaml').write_text(yaml.safe_dump(task))
    return tmp_path/'task.yaml',task


def fixture_study():
    ref=dict(in_channels=1,in_dim=512,out_dim=512,out_channels=4,scale=1,skip_connection=False,
             internal_instance_normalization=False,signal_processing_configs={'layer1':['I']},
             feature_extractor_configs=['RMS','Std','AbsMean'],f_c_mu=.18,f_c_sigma=.01,f_b_mu=.04,f_b_sigma=.001)
    return dict(schema_version=1,min_datasets=2,min_splits_per_dataset=2,seeds=[42,123,456],hpo_seed=20261003,
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


def test_bind_and_condition_audit_never_read_holdout_labels_or_payloads(task_source,tmp_path,monkeypatch):
    path,task=task_source
    frame=pd.read_csv(tmp_path/'metadata.csv',dtype=str)
    protocol=pd.read_csv(tmp_path/'protocol.csv',dtype=str)
    forbidden=set(protocol.loc[protocol.Split=='test','Id']) | set(frame.loc[frame.Domain=='2','Id'])
    frame.loc[frame.Id.isin(forbidden),'Label']='DO_NOT_PARSE_TARGET_LABEL'
    frame.to_csv(tmp_path/'metadata.csv',index=False)
    with monkeypatch.context() as metadata_only:
        def forbid_open(*args,**kwargs):
            raise AssertionError('Condition binding/audit must not open waveform storage')
        metadata_only.setattr(h5py,'File',forbid_open)
        root=bind_fixture(tmp_path,task_source)
        dg.condition_audit(root)
    original_getitem=H5DataDict.__getitem__
    def source_only(self,key):
        assert str(key) not in forbidden, 'Development accessed a held-out specimen or target condition'
        return original_getitem(self,key)
    monkeypatch.setattr(H5DataDict,'__getitem__',source_only)
    _,_,info=dg.suite(root)
    _,dataset,records=dg.source_data(info['tasks'][0])
    assert set(r['domain'] for r in records)=={'0','1'}
    assert set(r['split'] for r in records)=={'update','validation'}
    assert 'DO_NOT_PARSE' not in Path(dataset['metadata_file']).read_text()
    assert 'Label' not in pd.read_csv(root/'fixture_to_2'/'test_structure.csv')
    audited=pd.read_csv(root/'condition_audit.csv',dtype=str,keep_default_na=False).set_index('condition_id')
    assert audited.loc['0','class_support']==audited.loc['1','class_support']=='update:0,1;validation:0,1'
    assert audited.loc['2','class_support']==''
    dg.preflight(root)
    assert not (root/'fixture_to_2'/'test.csv').exists()


def test_physical_units_cannot_cross_partitions(task_source,tmp_path):
    protocol=pd.read_csv(tmp_path/'protocol.csv',dtype=str)
    protocol.loc[0,'Split']='test';protocol.to_csv(tmp_path/'protocol.csv',index=False)
    with pytest.raises(ValueError,match='crosses partitions'):bind_fixture(tmp_path,task_source)


def test_true_dg_requires_two_sources(task_source,tmp_path):
    path,task=task_source;task['dataset']['source_domains']=['0'];path.write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError,match='>=2'):bind_fixture(tmp_path,task_source)


@pytest.mark.parametrize('invalid', ['unknown_rpm','duplicate_tuple','missing_evidence','run_identity'])
def test_physical_admission_is_enforced_by_binding(task_source,tmp_path,invalid):
    path,task=task_source
    if invalid=='unknown_rpm':
        task['physical_conditions']['conditions']['2']['values']['rpm']='unknown'
    elif invalid=='duplicate_tuple':
        task['physical_conditions']['conditions']['2']['values']['rpm']=1500
    elif invalid=='missing_evidence':
        task['physical_conditions']['mapping']['evidence']['source']='UNVERIFIED'
    else:
        task['specimen_basis']['unit']='acquisition'
    path.write_text(yaml.safe_dump(task))
    with pytest.raises(ValueError,match='CONDITION_UNVERIFIED|SPECIMEN_UNVERIFIED'):
        bind_fixture(tmp_path,task_source)


def test_direct_source_reader_revalidates_despite_old_passed_audit(task_source,tmp_path,monkeypatch):
    root=bind_fixture(tmp_path,task_source)
    dg.condition_audit(root)
    data=dg.read(root/'fixture_to_2'/'source.yaml')
    dataset=data['datasets'][0]
    dataset['physical_conditions']['conditions']['0']['values']['rpm']='unknown'
    dataset['condition_audit']={'status':'passed'}
    def forbid_open(*args,**kwargs):
        raise AssertionError('An unverified physical condition reached waveform storage')
    monkeypatch.setattr(h5py,'File',forbid_open)
    with pytest.raises(ValueError,match='CONDITION_UNVERIFIED'):
        read_records(dataset,data)


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
    dg.condition_audit(root);dg.preflight(root);dg.smoke(root,'baseline','cpu')
    assert {row['arm'] for row in dg.read(root/'smoke_baseline.json')['checks']} == {'p0', 'TCN'}
    dg.tune(root,'reference','cpu');dg.tune(root,'baselines','cpu')
    dg.fit(root,'cpu',family='baselines');dg.qualify(root,'cpu')
    # A cached PASS cannot hide one final seed collapsing a source condition.
    predictions=root/'fixture_to_2'/'fits'/'TCN'/'456'/'selected_source_validation_windows.npz'
    with np.load(predictions,allow_pickle=False) as saved:
        original={key:saved[key] for key in saved.files}
    damaged={key:value.copy() for key,value in original.items()}
    damaged['candidate_probs'][damaged['domains']=='1']=np.array([1.,0.])
    np.savez_compressed(predictions,**damaged)
    with pytest.raises(ValueError,match='BASELINE_UNQUALIFIED'):
        dg.smoke(root,'method','cpu')
    with pytest.raises(ValueError,match='BASELINE_UNQUALIFIED'):
        dg.fit(root,'cpu',family='method')
    assert not (root/'fixture_to_2'/'smoke_method').exists()
    np.savez_compressed(predictions,**original)
    dg.smoke(root,'method','cpu');dg.tune(root,'method','cpu')
    dg.fit(root,'cpu',family='method')
    assert not (root/'fixture_to_2'/'test.csv').exists()
    dg.freeze(root,'cpu')
    frozen=dg.read(root/'frozen.json')
    study=frozen['study']
    arms=set(dg.CORE) | set(study['baselines']) | set(dg.ablation_arms(study))
    expected_fits={(arm,seed) for arm in arms for seed in study['seeds']}
    assert {(item['arm'],item['seed']) for item in frozen['models']}==expected_fits
    for arm,seed in expected_fits:
        restored=root/'fixture_to_2'/'frozen'/f'{arm}_{seed}_source_restore'
        assert (restored/'scope.json').is_file()
        assert (restored/'conditions.csv').is_file()
        with np.load(restored/'predictions.npz',allow_pickle=False) as arrays:
            assert set(arrays['domains'])=={'0','1'}
            assert set(arrays['group_ids'])=={'validation_0','validation_1'}
            np.testing.assert_allclose(arrays['deployed_probs'],arrays['candidate_probs'])
    with pytest.raises(ValueError,match='frozen'):dg.fit(root,'cpu')
    dg.test(root,'cpu');dg.analyze(root)
    contrasts=pd.read_csv(root/'summary'/'contrasts.csv')
    assert set(contrasts.contrast)==set(dg.CONTRASTS)|{'I-TCN'}
    assert set(contrasts.status)=={'complete'}
    rec=pd.read_csv(root/'summary'/'reconstruction.csv')
    assert (rec.max_abs<rec.tolerance).all()
    paired=pd.read_csv(root/'summary'/'paired_condition_differences.csv',dtype={'source_condition':str,'target_condition':str})
    assert set(paired.estimand)=={'paired_specimen_equal_class'}
    assert set(paired.status)=={'estimable'}
    assert set(paired.target_condition)=={'2'}
    assert set(paired.paired_groups)=={2}
    metrics={'accuracy','macro_f1','ce','brier'}
    expected_rows={(arm,seed,'candidate',source,metric)
                   for arm,seed in expected_fits for source in ('0','1') for metric in metrics}
    expected_rows |= {('p0',study['hpo_seed'],'raw',source,metric)
                     for source in ('0','1') for metric in metrics}
    identity=list(paired[['arm','seed','predictor','source_condition','metric']].itertuples(index=False,name=None))
    assert len(identity)==len(expected_rows)
    assert set(identity)==expected_rows
    np.testing.assert_allclose(paired.target_minus_source,paired.target_estimate-paired.source_estimate,atol=1e-12)
    assert dg.read(root/'summary'/'scope.json')['fixture'] is True


def test_execute_preserves_trainer_failure_when_recording_process_exit(task_source,tmp_path,monkeypatch):
    root=bind_fixture(tmp_path,task_source)
    _,study,info=dg.suite(root)
    output=root/'simulated_trainer_failure'
    failure=dict(status='failed',error_type='ValueError',error='constructed invalid source batch')
    commands=[]
    def trainer_failure(command,**kwargs):
        commands.append(command)
        run_dir=Path(command[command.index('--output')+1])
        run_dir.mkdir()
        (run_dir/'failure.json').write_text(json.dumps(failure))
        kwargs['stdout'].write('constructed invalid source batch\n')
        return SimpleNamespace(returncode=7)
    monkeypatch.setattr(dg.subprocess,'run',trainer_failure)
    with pytest.raises(RuntimeError,match='Training failed without changing recipe'):
        dg._execute(study,info['tasks'][0],'p0',study['trials'][0],42,output,'cpu')
    assert dg.read(output/'failure.json')==failure
    process_failure=dg.read(output/'process_failure.json')
    assert process_failure['returncode']==7
    assert process_failure['command']==commands[0]
    assert Path(process_failure['log']).read_text()=='constructed invalid source batch\n'
    assert len(commands)==1


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


def test_hpo_budget_cannot_be_expanded_or_padded_with_duplicates():
    study=fixture_study()
    study['trials']=[dict(lr=.0001*(index+1),weight_decay=0.,scheduler='none') for index in range(13)]
    with pytest.raises(ValueError,match='twelve'):
        dg.validate_study(study)
    study['trials']=[study['trials'][0],study['trials'][0]]
    with pytest.raises(ValueError,match='Duplicate HPO'):
        dg.validate_study(study)


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
