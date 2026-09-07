"""Scientific protocol checks, independent of a real dataset download."""
import copy
import importlib.util
from pathlib import Path
import random
import sys
import numpy as np
import pandas as pd
import pytest
import torch
import yaml

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('p01_runner',ROOT/'scripts/p01/run_experiments.py')
r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r)


def fixture(tmp_path):
    config=yaml.safe_load((ROOT/'configs/experiments/p01/demo.yaml').read_text())
    return config,r.generate_demo(tmp_path)


def test_permanent_unit_split(tmp_path):
    c,d=fixture(tmp_path); table=pd.read_csv(d['metadata_file'])
    table.loc[table['split']=='test','unit_id']=table.iloc[0]['unit_id']
    table.to_csv(d['metadata_file'],index=False)
    with pytest.raises(ValueError,match='crosses partitions'): r.read_records(d,c)


def test_no_fractional_label_coercion(tmp_path):
    c,d=fixture(tmp_path); table=pd.read_csv(d['metadata_file'])
    table['label']=table['label'].astype(float);table.loc[0,'label']=1.2
    table.to_csv(d['metadata_file'],index=False)
    with pytest.raises(ValueError,match='explicit integers'): r.read_records(d,c)


def test_native_reader_name_is_explicit(tmp_path):
    c,d=fixture(tmp_path);d['format']='phmfactory_reader'
    with pytest.raises(ValueError,match='reader'):r.read_records(d,c)


def test_no_unit_reuse_across_domains_and_splits(tmp_path):
    c,d=fixture(tmp_path); records=r.read_records(d,c)
    assert len(records)==48
    for part in ['update','validation','test']:
        assert {x['unit_id'] for x in records if x['split']==part}.isdisjoint(
            {x['unit_id'] for x in records if x['split']!=part})


def test_der_reservoir_keeps_historical_targets():
    units=[{'domain':'d0','unit_id':str(i),'x':torch.ones(2,8,1)} for i in range(4)]
    history={('d0',str(i)):torch.full((2,3),float(i)) for i in range(4)}
    memory=[]
    seen=r.update_memory(memory,units,history,0,2,random.Random(8))
    saved=[x['historical_logits'].clone() for x in memory]
    # Empty stages do not refresh targets from a newer network.
    assert r.update_memory(memory,[],{},seen,2,random.Random(8))==4
    assert all(torch.equal(a,b['historical_logits']) for a,b in zip(saved,memory))


def test_replay_without_logits_does_not_allocate_targets():
    u={'domain':'d0','unit_id':'a','x':torch.ones(2,8,1)}
    memory=[];r.update_memory(memory,[u],{},0,2,random.Random(8))
    assert 'historical_logits' not in memory[0]


def test_false_replay_does_not_change_initial_training(tmp_path):
    c,d=fixture(tmp_path);c['training']['epochs']=1;c['training']['steps_per_epoch']=6
    records=r.read_records(d,c)
    train=r.materialize([u for u in records if u['domain'] in d['source_domains'] and u['split']=='update'],d,c)
    val=r.materialize([u for u in records if u['domain'] in d['source_domains'] and u['split']=='validation'],d,c)
    torch.manual_seed(15);a=r.build_model(c['model'],c['engine']);b=copy.deepcopy(a)
    # Recording history must not alter optimization or sampling.
    r.train_stage(a,train,val,c,torch.device('cpu'),random.Random(3),r.OFFLINE['S4_covariance'],store_history=False,stage_seed=7)
    r.train_stage(b,train,val,c,torch.device('cpu'),random.Random(3),r.OFFLINE['S4_covariance'],store_history=True,stage_seed=7)
    assert all(torch.equal(a.state_dict()[k],b.state_dict()[k]) for k in a.state_dict())
