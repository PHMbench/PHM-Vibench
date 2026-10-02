"""P08 record/split/selection tests using explicit CSV fixtures, never PHM claims."""
import json
from types import SimpleNamespace
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import yaml
from src.data_factory.p08_data import fit_conditions, encode_conditions, validate_inventory
from src.task_factory.task.DG.p08 import record_metrics
from scripts.p08_experiments import preflight, fit_once, evaluate


def args():
    return SimpleNamespace(condition_fields=[dict(name='rpm',kind='continuous',unit='rpm',
        meaning='Fixture speed',availability='Constructed test input')],label_names=['normal','fault'],
        source_system_ids=[0,1],target_system_id=2)


def frame():
    rows=[]
    for system in (0,1):
        for role in ('source_train','source_val'):
            for label in (0,1):
                rid=f'{system}-{role}-{label}'
                rows.append(dict(Id=len(rows)+1,Name='Dummy_Data',File=rid+'.csv',Dataset_id=system,
                    Label=label,LabelName=['normal','fault'][label],Group=rid,record_id=rid,role=role,
                    Channel=0,Sample_rate=100.,rpm=1000+100*system+label))
    return pd.DataFrame(rows)


def test_source_fit_rejects_validation_and_target_information():
    data=frame(); a=args()
    with pytest.raises(ValueError,match='source_train'): fit_conditions(data,a.condition_fields)
    schema=fit_conditions(data[data.role=='source_train'],a.condition_fields)
    validation=data[data.role=='source_val'].copy(); validation.rpm=100000.
    encoded=encode_conditions(validation,schema)
    assert (encoded[:,3] == 1).all() and schema[0]['upper'] < 2000
    altered=data.copy(); altered.loc[0,'Group']=altered.loc[2,'Group']
    with pytest.raises(ValueError,match='leakage'): validate_inventory(altered,a)
    altered=data.copy(); altered.loc[0,'Dataset_id']=2
    with pytest.raises(ValueError,match='target'): validate_inventory(altered,a)


def test_record_metric_averages_probabilities_and_checks_population():
    rows=[dict(record_id='a',unit='u',system='s',label=0,window_start=0,p0=.9,p1=.1),
          dict(record_id='a',unit='u',system='s',label=0,window_start=1,p0=.3,p1=.7),
          dict(record_id='b',unit='v',system='s',label=1,window_start=0,p0=.2,p1=.8)]
    records,metrics=record_metrics(rows,2,{'a':2,'b':1})
    assert records.set_index('record_id').loc['a','p0'] == pytest.approx(.6)
    assert metrics['unit_balanced_brier'] == pytest.approx((.16+.04)/2)
    assert metrics['systems'][0]['record_macro_f1'] == 1.
    with pytest.raises(ValueError,match='Duplicate'): record_metrics(rows+[rows[0]],2,{'a':3,'b':1})
    with pytest.raises(ValueError,match='population'): record_metrics(rows,2,{'a':2})


def test_native_source_pipeline_then_frozen_target_evaluation(tmp_path):
    source=frame(); raw=tmp_path/'raw'/'Dummy_Data'; raw.mkdir(parents=True)
    for row in source.to_dict('records'):
        signal=np.linspace(0,1,64)+row['Label']
        pd.DataFrame({'ch1':signal,'ch2':-signal}).to_csv(raw/row['File'],index=False)
    source.to_csv(tmp_path/'source.csv',index=False)
    root=Path(__file__).parents[1]
    cfg=yaml.safe_load((root/'configs/experiments/p08/source_only.yaml').read_text())
    cfg['environment'].update(output_dir=str(tmp_path/'native_results'))
    cfg['data'].update(vars(args()),data_dir=str(tmp_path),metadata_file='source.csv',storage='raw',
        evidence_kind='software_fixture',window_size=32,stride=32,batch_size=4,train_batches_per_epoch=2)
    cfg['model'].update(output_dim=8,num_classes=2,condition_dim=4,patch_size_L=8,num_patches=4,nhead=2)
    cfg['trainer'].update(device='cpu',num_epochs=1,early_stopping=False)
    path=tmp_path/'source.yaml'; path.write_text(yaml.safe_dump(cfg))
    request=SimpleNamespace(config=path,local_config=None,override=[])
    check=tmp_path/'check'; check.mkdir()
    assert preflight(request,check)['target_opened'] is False
    run=tmp_path/'fit'; run.mkdir()
    fitted=fit_once(request,run)
    assert Path(fitted['checkpoint']).is_file() and np.isfinite(fitted['source_unit_brier'])
    # Target records are created only after fitting completed: they cannot be read by fitting.
    target=source.iloc[:2].copy(); target.Dataset_id=2; target.role='target_test'
    target.Id=[101,102]; target.Group=['test-a','test-b']; target.record_id=['test-a','test-b']
    target.File=['test-a.csv','test-b.csv']
    for row in target.to_dict('records'):
        pd.DataFrame({'ch1':np.linspace(0,1,64)+row['Label'],'ch2':np.zeros(64)}).to_csv(raw/row['File'],index=False)
    target_path=tmp_path/'target.csv'; target.to_csv(target_path,index=False)
    output=tmp_path/'evaluation'; output.mkdir()
    e=SimpleNamespace(checkpoint=Path(fitted['checkpoint']),inventory=target_path,data_root=tmp_path,
                      conditions=None,detach=False,device='cpu')
    metrics=evaluate(e,output)
    assert metrics['systems'][0]['record_count']==2
    assert (output/'predictions.csv').is_file()


def test_default_missing_and_categorical_unknown_encoding():
    data=frame(); data['housing']=['a','b']*4
    fields=args().condition_fields+[dict(name='housing',kind='categorical',unit='category',
        meaning='Fixture housing',availability='Constructed test input')]
    schema=fit_conditions(data[data.role=='source_train'],fields)
    data=data.iloc[:2].copy(); data.loc[0,'rpm']=np.nan; data.loc[1,'rpm__defaulted']=True
    data.loc[0,'rpm__defaulted']=False; data.housing=['new',None]
    values=encode_conditions(data,schema).numpy()
    assert list(values[0,:4]) == [0.,0.,0.,0.]
    assert list(values[1,:4]) == [0.,0.,1.,0.]
    assert list(values[:,4+2]) == [1.,1.]  # UNKNOWN coordinate


def test_same_source_sampling_probability_despite_window_counts(tmp_path):
    from src.data_factory.p08_data import P08Windows
    import torch
    data=frame(); a=args(); a.window_size=8; a.stride=8; a.normalization='none'
    signals=[torch.zeros(16+8*i,1) for i in range(len(data))]
    dataset=P08Windows(data,signals,torch.zeros(len(data),4),a)
    weights=dataset.sampling_weights().numpy()
    totals={s:sum(w for w,(i,_,_) in zip(weights,dataset.windows) if data.iloc[i].Dataset_id==s) for s in (0,1)}
    assert totals[0] == pytest.approx(totals[1])


def test_target_group_overlap_is_rejected_after_string_normalization():
    data=frame().iloc[:2].copy(); data.Dataset_id=2; data.role='target_test'; data.Group=[11,12]
    with pytest.raises(ValueError,match='Target physical unit'):
        validate_inventory(data,args(),evaluation=True,source_groups=['11'])


def test_native_h5_reader_contract(tmp_path):
    import h5py
    from src.data_factory.p08_data import read_records
    data=frame().iloc[:1].copy(); a=args()
    a.data_dir=str(tmp_path); a.storage='h5'; a.h5_layout='sample_channel'; a.window_size=8
    with h5py.File(tmp_path/'Dummy_Data.h5','w') as f: f.create_dataset(str(data.Id.iloc[0]),data=np.ones((16,2)))
    signal=read_records(data,a)
    assert tuple(signal[0].shape)==(16,1)


def test_native_inventory_preserves_unit_and_category_spellings(tmp_path):
    from src.data_factory.p08_data import read_inventory
    path=tmp_path/'ids.csv'
    path.write_text('Group,record_id,housing,rpm\n01,001,02,1000\n1,1,2,1100\n')
    fields=args().condition_fields+[dict(name='housing',kind='categorical',unit='category',meaning='Fixture',availability='Fixture')]
    rows=read_inventory(path,fields)
    assert list(rows.Group)==['01','1'] and list(rows.record_id)==['001','1']
    assert list(rows.housing)==['02','2']


def test_source_statistics_exclude_declared_default_values():
    data=frame(); train=data[data.role=='source_train'].copy()
    train['rpm__defaulted']=False
    train.loc[train.index[0],['rpm','rpm__defaulted']]=[1000000.,True]
    schema=fit_conditions(train,args().condition_fields)
    assert schema[0]['upper'] < 2000
    fields=[dict(name='record_id',kind='categorical',unit='id',meaning='Not physics',availability='File')]
    with pytest.raises(ValueError,match='identity'): fit_conditions(train,fields)


def test_existing_prediction_comparison_and_plot_never_invoke_models(tmp_path):
    from scripts.p08_experiments import compare_existing, plot_existing
    rows=pd.DataFrame(dict(record_id=['01','1','2','3'],unit=['01','1','2','3'],system=['2']*4,
        label=[0,1,0,1],prediction=[0,0,0,1],p0=[.9,.6,.9,.2],p1=[.1,.4,.1,.8]))
    left=tmp_path/'baseline.csv'; right=tmp_path/'method.csv'
    rows.to_csv(left,index=False); rows.to_csv(right,index=False)
    a=SimpleNamespace(predictions=[left,right],bootstrap_repeats=20,analysis_seed=0)
    result=compare_existing(a,tmp_path)
    assert result['comparisons'][0]['interval_95'] == [0.,0.]
    assert result['comparisons'][0]['physical_units'] == 4
    plot_existing(a,tmp_path)
    assert (tmp_path/'system_0.svg').is_file()
    rows.loc[0,'unit']='different'; rows.to_csv(right,index=False)
    with pytest.raises(ValueError,match='identical'): compare_existing(a,tmp_path)
