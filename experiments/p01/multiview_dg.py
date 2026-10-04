"""Source-only DG orchestration over the existing P01 model, trainers and evaluator.

Task assignment is administrative: it contains physical IDs and domains, not a
learned split. Only the final `test` command reads held-out labels or waveforms.
This module neither defines another network nor reimplements the risk estimator.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.p01.fusion_data import read_records, summarize_rows
from experiments.p01.fusion_deployment import (
    FrozenClassifier, load_model, predict_records, save_bundle, verify_vectors,
    acquisition_rows, write_csv,
)
from experiments.p01.window_io import window_record, _h5_metadata_frame, _vibench_id
from experiments.p01 import analyze_d1
from src.model_factory.model_factory import model_factory

ROOT = Path(__file__).resolve().parents[2]
CORE = ('I', 'I-F', 'MLP16', 'I-single', 'Dense', 'I-base')
CONTRASTS = {f'I-{arm}': ('I', arm) for arm in ('p0', 'I-F', 'MLP16', 'I-single', 'Dense', 'I-base')}
SUPPORTED = {('CNN','ResNet1D'), ('CNN','TCN'), ('Transformer','PatchTST'),
             ('X_model','BASE_ExplainableCNN')}


def read(path):
    path=Path(path)
    text=path.read_text(encoding='utf-8')
    return json.loads(text) if path.suffix=='.json' else yaml.safe_load(text)


def dump(path, obj):
    path = Path(path)
    with path.open('x', encoding='utf-8') as stream:
        json.dump(obj, stream, indent=2, allow_nan=False)


def resolve(value, base):
    value = Path(os.path.expandvars(str(value))).expanduser()
    return value.resolve() if value.is_absolute() else (base/value).resolve()


def validate_study(study):
    if study['schema_version'] != 1:
        raise ValueError('Unsupported DG study schema.')
    seeds = study['seeds']
    if study['min_datasets']<2 or study['min_splits_per_dataset']<2:
        raise ValueError('The industrial suite needs multiple datasets and multiple splits per dataset.')
    if len(seeds)<2 or any(type(x) is not int for x in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('At least two unique explicit final seeds are required.')
    if study['hpo_seed'] in seeds:
        raise ValueError('HPO seed and final reporting seeds must be distinct.')
    if study['reference_min_accuracy'] != .8:
        raise ValueError('The current study retains its prespecified .8 reference criterion.')
    if min(study[k] for k in ('epochs','steps_per_epoch','units_per_domain','overfit_steps')) < 1:
        raise ValueError('Positive training and smoke budgets are required.')
    if len(study['trials']) < 2:
        raise ValueError('Freeze more than one source-only HPO trial.')
    for trial in study['trials']:
        if set(trial)!={'lr','weight_decay','scheduler'} or not math.isfinite(trial['lr']) or trial['lr']<=0:
            raise ValueError('Each trial declares lr, weight_decay and scheduler.')
        if trial['scheduler'] not in {'none','cosine'} or not math.isfinite(trial['weight_decay']) or trial['weight_decay']<0:
            raise ValueError('Invalid optimizer trial.')
    if set(study['baselines']) & set((*CORE, 'p0')):
        raise ValueError('Baseline aliases may not shadow a core model identity.')
    if not study['baselines'] or any((m['type'],m['name']) not in SUPPORTED for m in study['baselines'].values()):
        raise ValueError('Use only inspected classification models at the recorded runtime revision.')
    names = [b['name'] for b in study['fusion']['model']['branches']]
    if len(names)<2 or len(names)!=len(set(names)) or len(study['trials'])<len(names):
        raise ValueError('Require multiple unique views and enough total single-view trials to cover them.')
    loss=study['fusion']['loss']
    if (loss['tau']!=1 or loss['brier_weight']!=.25 or loss['lambda_delta']!=0 or loss['reduction']!='mean_source'):
        raise ValueError('The primary DG model retains ordinary paired CE+.25 Brier, no new alignment or routing loss.')
    if study['fusion']['model']['head_type']!='operator_residual':
        raise ValueError('Do not replace the current proposed method.')


def _metadata_rows(dataset, selected_ids):
    """Read labels only for explicitly allowed acquisition rows (CSV or XLSX).

    The first pass requests the Id column only; skipped rows are not returned to
    fitting or selection. Original target labels are never copied to source.csv.
    """
    path=Path(dataset['metadata_file']); idcol=dataset['columns']['id']
    reader=pd.read_excel if path.suffix.lower()=='.xlsx' else pd.read_csv
    ids=reader(path,usecols=[idcol],dtype=str,keep_default_na=False)[idcol].map(_vibench_id)
    if ids.duplicated().any(): raise ValueError('Duplicate acquisition ID in metadata.')
    keep={i+1 for i,x in enumerate(ids) if x in selected_ids}
    result=reader(path,skiprows=lambda i: i!=0 and i not in keep,dtype=str,keep_default_na=False)
    result[idcol]=result[idcol].map(_vibench_id)
    if set(result[idcol])!=set(selected_ids): raise ValueError('Assignment IDs are missing from original metadata.')
    return result


def bind_task(task_path, out):
    task=read(task_path)
    if set(task['data'])-{'layout','window_size','windows_per_unit','channel_indices','squeeze_axes'}:
        raise ValueError('Undeclared preprocessing/normalization is forbidden; no target-fitted statistics.')
    names=task['model']['class_names']
    if len(names)!=task['model']['num_classes'] or len(set(names))!=len(names) or len(names)<2:
        raise ValueError('Declare the complete ordered class space once.')
    base=Path(task_path).resolve().parent
    dataset=copy.deepcopy(task['dataset']); mapping=dataset['columns']
    for key in ('metadata_file','h5_file','protocol_file'):
        dataset[key]=str(resolve(dataset[key],base))
    # A custodian assigns a physical specimen/run globally, never a window.
    if not task.get('physical_group_basis') or not task.get('domain_basis'):
        raise ValueError('State how physical IDs and operating domains were established.')
    if task.get('dataset_id')=='RM_027_PU':
        raise ValueError('PU D1 test exposure is sealed; no automatic reuse in the new study.')
    sources=list(map(str,dataset['source_domains'])); targets=list(map(str,dataset['domain_sequence']))
    if len(sources)<2 or len(targets)!=1 or len(set(sources+targets))!=len(sources)+1:
        raise ValueError('DG requires >=2 distinct sources and exactly one disjoint unseen target.')
    protocol=pd.read_csv(dataset['protocol_file'],dtype=str,keep_default_na=False)
    required={mapping['id'],mapping['unit_id'],mapping['split']}
    protocol[mapping['id']]=protocol[mapping['id']].map(_vibench_id)
    if set(protocol)!=required or protocol[mapping['id']].duplicated().any():
        raise ValueError('Protocol must provide exactly unique Id, physical unit and partition columns.')
    if not set(protocol[mapping['split']]).issubset({'update','validation','test','exclude'}):
        raise ValueError('Assign update/validation/test/exclude explicitly.')
    if protocol.groupby(mapping['unit_id'])[mapping['split']].nunique().max()!=1:
        raise ValueError('A physical specimen crosses partitions, including other operating conditions.')
    path=Path(dataset['metadata_file']); reader=pd.read_excel if path.suffix.lower()=='.xlsx' else pd.read_csv
    structural_cols={mapping[k] for k in ('id','domain','sample_rate_hz','rotation_speed_rpm')} | set(dataset.get('select',{}))
    if mapping['label'] in structural_cols:
        raise ValueError('Class labels may not define condition selection.')
    frame=reader(path,usecols=list(structural_cols),dtype=str,keep_default_na=False)
    selection=dict(dataset);selection.pop('protocol_file')
    frame=_h5_metadata_frame(selection,frame,mapping)
    required_ids=set(frame[frame[mapping['domain']].isin(sources+targets)][mapping['id']])
    if not required_ids.issubset(set(protocol[mapping['id']])):
        raise ValueError('Every selected acquisition needs an explicit physical partition; use exclude rather than omit rows.')
    frame=frame.merge(protocol,on=mapping['id'],how='left',validate='one_to_one')
    if not len(frame) or not frame[mapping['unit_id']].str.len().min():
        raise ValueError('Missing assigned physical identities.')
    frame=frame[frame[mapping['domain']].isin(sources+targets)]
    src=frame[frame[mapping['domain']].isin(sources)&frame[mapping['split']].isin(['update','validation'])]
    tst=frame[frame[mapping['split']]=='test']
    if not set(targets).issubset(set(tst[mapping['domain']])) or not len(src):
        raise ValueError('Assigned task lacks source development or target test groups.')
    if set(src[mapping['unit_id']])&set(tst[mapping['unit_id']]):
        raise ValueError('Source and test physical units overlap.')
    rates=pd.to_numeric(frame[mapping['sample_rate_hz']],errors='raise')
    if not np.isfinite(rates).all() or (rates<=0).any() or rates.nunique()!=1:
        raise ValueError('One sampling convention per task, including held-out structure; do not silently resample.')
    source=_metadata_rows(dataset,set(src[mapping['id']]))
    if any(c in source for c in (mapping['unit_id'],mapping['split'])):
        raise ValueError('Existing physical partitions must not be overwritten.')
    source=source.merge(src[list(required)],on=mapping['id'],validate='one_to_one')
    out.mkdir(parents=True,exist_ok=False)
    source.to_csv(out/'source.csv',index=False)
    # Only IDs/domains/partitions/physical units/measurement conventions; no test labels.
    tst.to_csv(out/'test_structure.csv',index=False)
    src.to_csv(out/'source_structure.csv',index=False)
    source_dataset=copy.deepcopy(dataset)
    source_dataset.pop('protocol_file');source_dataset.pop('select',None)
    source_dataset.update(metadata_file=str(out/'source.csv'),source_domains=sources,domain_sequence=[],access_scope='source')
    data=dict(model=task['model'],data=task['data'],datasets=[source_dataset])
    (out/'source.yaml').write_text(yaml.safe_dump(data,sort_keys=False))
    task['dataset']=dataset
    dump(out/'task.json',task)
    h5_stat=Path(dataset['h5_file']).stat()
    return dict(name=task['name'],dataset_id=task['dataset_id'],path=str(out),target=targets[0],
                binding={name:(out/name).read_text() for name in ('task.json','source.yaml','source.csv','source_structure.csv','test_structure.csv')},
                h5_stat=[h5_stat.st_size,h5_stat.st_mtime_ns])


def bind(study_path, task_paths, output, fixture=False):
    study=read(study_path);validate_study(study)
    tasks=[read(p) for p in task_paths]
    names=[t['name'] for t in tasks]
    if len(names)!=len(set(names)) or any('/' in n or n in {'.','..'} for n in names):
        raise ValueError('Unique path-safe task names required.')
    identities=[(t['dataset_id'],tuple(sorted(map(str,t['dataset']['source_domains']))),tuple(map(str,t['dataset']['domain_sequence']))) for t in tasks]
    if len(set(identities))!=len(identities):
        raise ValueError('Duplicate domain splits must not be counted as distinct experiments.')
    if not fixture:
        by_dataset={t['dataset_id'] for t in tasks}
        if len(by_dataset)<study['min_datasets'] or any(sum(t['dataset_id']==d for t in tasks)<study['min_splits_per_dataset'] for d in by_dataset):
            raise ValueError('Bind the complete prospective multi-dataset/multi-split suite before tuning.')
    root=Path(output).resolve();root.mkdir(parents=True,exist_ok=False)
    (root/'study.yaml').write_text(yaml.safe_dump(study,sort_keys=False))
    descriptions=[bind_task(p,root/read(p)['name']) for p in task_paths]
    physical_partitions={};observation_owners={};label_spaces={}
    for task in descriptions:
        spec=read(Path(task['path'])/'task.json');m=spec['dataset']['columns']
        labels=tuple(spec['model']['class_names'])
        if label_spaces.setdefault(task['dataset_id'],labels)!=labels:
            raise ValueError('One dataset must retain its ordered class space across domain splits.')
        for filename in ('source_structure.csv','test_structure.csv'):
            frame=pd.read_csv(Path(task['path'])/filename,dtype=str,keep_default_na=False)
            for identifier in frame[m['id']]:
                key=(spec['dataset']['h5_file'],identifier)
                if observation_owners.setdefault(key,task['dataset_id'])!=task['dataset_id']:
                    raise ValueError('The same H5 observations were relabeled as multiple datasets.')
        protocol=pd.read_csv(spec['dataset']['protocol_file'],dtype=str,keep_default_na=False)
        for unit,split in protocol[[m['unit_id'],m['split']]].drop_duplicates().itertuples(index=False,name=None):
            key=(task['dataset_id'],unit)
            if key in physical_partitions and physical_partitions[key]!=split:
                raise ValueError('A physical group changes partition across DG splits; preserve the global specimen assignment.')
            physical_partitions[key]=split
    dump(root/'tasks.json',dict(tasks=descriptions,fixture=fixture))
    dump(root/'study_lock.json',study)
    return root


def suite(root):
    root=Path(root).resolve()
    study=read(root/'study.yaml')
    if study!=read(root/'study_lock.json'):
        raise ValueError('Study configuration changed after binding; use a new prospective study.')
    info=read(root/'tasks.json')
    for task in info['tasks']:
        for name,text in task['binding'].items():
            if (Path(task['path'])/name).read_text()!=text:
                raise ValueError(f'Bound source/task assignment changed: {task["name"]}/{name}')
        h5=Path(read(Path(task['path'])/'task.json')['dataset']['h5_file']).stat()
        if [h5.st_size,h5.st_mtime_ns]!=task['h5_stat']:
            raise ValueError('Read-only H5 changed after task binding.')
    return root,study,info


def source_data(task):
    data=read(Path(task['path'])/'source.yaml');dataset=data['datasets'][0]
    if dataset.get('access_scope')!='source' or dataset['domain_sequence']:
        raise ValueError('Only isolated source data may enter development.')
    return data,dataset,read_records(dataset,data)


def candidate_config(study,task,arm,reference=None,view=None):
    data=read(Path(task['path'])/'source.yaml')
    classes=data['model']['num_classes'];length=data['data']['window_size']
    channels=len(data['data']['channel_indices'])
    if arm=='p0':
        model=copy.deepcopy(study['reference'])
        model.update(type='X_model',name='TSPN',num_classes=classes,in_channels=channels,in_dim=length,out_dim=length,device='cpu')
        return dict(model=model)
    if arm in study['baselines']:
        model=copy.deepcopy(study['baselines'][arm])
        model.update(num_classes=classes,device='cpu')
        model['in_channels' if model['type']=='X_model' else 'input_dim']=channels
        return dict(model=model)
    cfg=copy.deepcopy(study['fusion']);m=cfg['model']
    ref=read(Path(reference)/'model_config.yaml')['model']
    m.update(num_classes=classes,reference_config=ref,checkpoint_kind='reference',
             checkpoint_path=str(Path(reference)/'selected_candidate.pt'),device='cpu')
    if arm in {'I-F','MLP16'}: m['branches']=[]
    elif arm=='I-single':
        m['branches']=[b for b in m['branches'] if b['name']==view]
        if len(m['branches'])!=1: raise ValueError('Unknown source-selected view.')
    elif arm=='I-base':
        for b in m['branches']:
            if b['type']=='envelope': b.pop('diagnostics',None)
    elif arm not in {'I','Dense'}: raise ValueError('Unknown proposed/control arm.')
    if arm in {'Dense','MLP16'}: m['head_type']='mlp'
    return cfg


def preflight(root):
    root,study,info=suite(root)
    if (root/'preflight.json').exists(): raise FileExistsError('Preflight already recorded; use a fresh suite after protocol edits.')
    checks=[]
    for task in info['tasks']:
        data,dataset,records=source_data(task)
        classes=set(range(data['model']['num_classes']))
        for d in dataset['source_domains']:
            for split in ('update','validation'):
                rows=[r for r in records if r['domain']==str(d) and r['split']==split]
                if {r['label'] for r in rows}!=classes:
                    raise ValueError(f'{task["name"]}/{d}/{split} lacks closed-set source class coverage.')
                if len({r['unit_id'] for r in rows})<study['units_per_domain']:
                    raise ValueError('Too few physical groups for the declared batch sampling.')
        for record in records:
            x=window_record(record,dataset,data['data'])
            if x.shape[-1]!=len(data['data']['channel_indices']) or not torch.isfinite(x).all():
                raise ValueError('Invalid observed channels/tensor.')
            if torch.any(x.std(dim=1)==0): raise ValueError('Constant source window/channel; inspect reader and signal.')
        checks.append(dict(task=task['name'],groups=len({r['unit_id'] for r in records}),
                           acquisitions=len(records),source_domains=dataset['source_domains'],target_read=False))
    dump(root/'preflight.json',dict(status='passed',tasks=checks))


def _development_guard(root, device):
    root,study,info=suite(root)
    if (root/'frozen.json').exists(): raise ValueError('The complete suite is frozen; no further fitting or selection.')
    if not (root/'preflight.json').exists(): raise ValueError('Run source-only preflight first.')
    if device=='cpu' and not info['fixture']:
        raise ValueError('CPU training is allowed only for explicit constructed fixtures; use physical GPU0 locally.')
    if device!='cpu' and (device!='cuda:0' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or not torch.cuda.is_available()):
        raise ValueError('Training requires CUDA_VISIBLE_DEVICES=0 and available cuda:0; no fallback.')
    return root,study,info


def _reference(task):
    record=read(Path(task['path'])/'hpo'/'p0'/'selection.json')
    return Path(record['run'])


def _execute(study,task,arm,trial,seed,output,device,reference=None,view=None):
    output=Path(output)
    if output.exists(): raise FileExistsError(output)
    output.parent.mkdir(parents=True,exist_ok=True)
    cfg=candidate_config(study,task,arm,reference,view)
    cfgpath=output.parent/(output.name+'.yaml')
    cfgpath.write_text(yaml.safe_dump(cfg,sort_keys=False))
    flags=['--model-config',str(cfgpath),'--data-config',str(Path(task['path'])/'source.yaml'),
           '--dataset',read(Path(task['path'])/'source.yaml')['datasets'][0]['name'],
           '--output',str(output),'--seed',str(seed),'--device',device,'--dg',
           '--epochs',str(study['epochs']),'--steps-per-epoch',str(study['steps_per_epoch']),
           '--units-per-domain',str(study['units_per_domain']),
           '--lr',str(trial['lr']),'--weight-decay',str(trial['weight_decay']),'--scheduler',trial['scheduler']]
    if arm=='p0' or arm in study['baselines']:
        module='experiments.p01.train_source_classifier'
        flags+=['--role','reference' if arm=='p0' else 'baseline']
        if arm!='p0':flags+=['--reference-config',str(reference/'model_config.yaml'),
                             '--reference-checkpoint',str(reference/'selected_candidate.pt'),
                             '--pair-shift',str(study['pair_shift'])]
    else:
        module='experiments.p01.train_tspn_fusion_v2'
        flags+=['--pair-shift',str(study['pair_shift']),'--arm',arm,
                '--reference-min-accuracy',str(study['reference_min_accuracy'])]
    command=[sys.executable,'-m',module,*flags]
    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''))
    logpath=output.parent/(output.name+'.log')
    with logpath.open('x') as stream:
        run=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
