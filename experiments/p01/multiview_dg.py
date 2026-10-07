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
import shutil
import time
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import yaml

from experiments.p01.fusion_data import read_records, summarize_rows
from experiments.p01.condition_contract import validate_condition_task
from experiments.p01.baseline_qualification import qualify_source_prediction_arrays
from experiments.p01.condition_comparison import paired_condition_rows
from experiments.p01.diagnostic_analysis import profile_predictor, select_explanation_cases
from experiments.p01.fusion_deployment import (
    FrozenClassifier, load_model, predict_records, save_bundle, verify_vectors,
    acquisition_rows, write_csv,
)
from experiments.p01.window_io import window_record, _h5_metadata_frame, _vibench_id
from experiments.p01 import analyze_d1
from src.model_factory.model_factory import model_factory

ROOT = Path(__file__).resolve().parents[2]
CORE = ('I', 'I-F', 'MLP16', 'I-single', 'Dense', 'Dense-matched', 'I-base')
CONTRASTS = {f'I-{arm}': ('I', arm) for arm in ('p0', 'I-F', 'MLP16', 'I-single', 'Dense', 'Dense-matched', 'I-base')}
SUPPORTED = {('CNN','ResNet1D'), ('CNN','TCN'), ('Transformer','PatchTST'),
             ('X_model','BASE_ExplainableCNN'), ('X_model','MWA_CNN'), ('X_model','TSPN')}


def ablation_arms(study):
    if not study.get('leave_one_view_out', False):
        return ()
    return tuple(f"I-minus-{b['name']}" for b in study['fusion']['model']['branches'])


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


def validate_study(study, *, fixture=False):
    if study['schema_version'] != 1:
        raise ValueError('Unsupported DG study schema.')
    seeds = study['seeds']
    if study['min_datasets']<2 or study['min_splits_per_dataset']<2:
        raise ValueError('The industrial suite requires at least two datasets and two prospectively frozen condition splits per dataset.')
    if len(seeds)<2 or any(type(x) is not int for x in seeds) or len(set(seeds))!=len(seeds):
        raise ValueError('At least two unique explicit final seeds are required.')
    if study['hpo_seed'] in seeds:
        raise ValueError('HPO seed and final reporting seeds must be distinct.')
    if 42 not in seeds:
        raise ValueError('The frozen explanation-case rule requires final seed 42, including in constructed fixtures.')
    if not fixture and (seeds != [42, 123, 456] or study['hpo_seed'] != 20261003):
        raise ValueError('Formal DG retains final seeds 42/123/456 and HPO seed 20261003.')
    if study['reference_min_accuracy'] != .8:
        raise ValueError('The current study retains its prespecified .8 reference criterion.')
    if min(study[k] for k in ('epochs','steps_per_epoch','units_per_domain','overfit_steps')) < 1:
        raise ValueError('Positive training and smoke budgets are required.')
    if type(study['pair_shift']) is not int or study['pair_shift'] < 1:
        raise ValueError('All formal DG models share a positive declared circular-pair shift budget.')
    if not 2 <= len(study['trials']) <= 12:
        raise ValueError('Freeze between two and twelve source-only HPO trials per family.')
    if len({json.dumps(t, sort_keys=True) for t in study['trials']}) != len(study['trials']):
        raise ValueError('Duplicate HPO trials do not add tuning evidence.')
    if 'update_budgets' in study and study['update_budgets'] != [1000, 2500, 5000, 7500]:
        raise ValueError('Freeze the declared common source-convergence budget ladder.')
    for trial in study['trials']:
        if set(trial)!={'lr','weight_decay','scheduler'} or not math.isfinite(trial['lr']) or trial['lr']<=0:
            raise ValueError('Each trial declares lr, weight_decay and scheduler.')
        if trial['scheduler'] not in {'none','cosine'} or not math.isfinite(trial['weight_decay']) or trial['weight_decay']<0:
            raise ValueError('Invalid optimizer trial.')
    if not fixture and {(t['lr'], t['weight_decay'], t['scheduler']) for t in study['trials']} != {
            (lr, wd, 'none') for lr in (1e-4, 3e-4, 1e-3, 3e-3) for wd in (0., 1e-4, 1e-3)}:
        raise ValueError('Formal DG requires the frozen twelve-trial 4-by-3 Adam grid for every family.')
    if set(study['baselines']) & set((*CORE, 'p0')):
        raise ValueError('Baseline aliases may not shadow a core model identity.')
    if not study['baselines'] or any((m['type'],m['name']) not in SUPPORTED for m in study['baselines'].values()):
        raise ValueError('Use only inspected classification models at the recorded runtime revision.')
    for arm, model in study['baselines'].items():
        if arm == 'TSPN_TON' and (model['type'], model['name']) != ('X_model', 'TSPN'):
            raise ValueError('TSPN_TON identifies a TSPN configuration, not a replacement architecture.')
        if (model['type'], model['name']) == ('X_model', 'TSPN'):
            if arm != 'TSPN_TON' or study.get('baseline_training', {}).get(arm) != {'objective': 'ce'}:
                raise ValueError('The independent TSPN_TON comparator requires its explicit native CE objective.')
            provenance = study.get('baseline_provenance', {}).get(arm, {})
            if provenance.get('status') not in {'implementation_unverified', 'documented_adaptation'}:
                raise ValueError('TSPN_TON requires explicit adaptation provenance, not a faithful-reproduction claim.')
            if provenance['status'] == 'documented_adaptation' and not provenance.get('differences'):
                raise ValueError('TSPN_TON documented_adaptation must disclose its implemented differences.')
    if set(study.get('baseline_training', {})) - set(study['baselines']):
        raise ValueError('Baseline training recipes must name an existing baseline.')
    for arm, recipe in study.get('baseline_training', {}).items():
        expected = 'ce' if arm == 'TSPN_TON' else 'ce_plus_0.25_brier'
        if recipe != {'objective': expected}:
            raise ValueError(f'{arm} must retain its declared training objective {expected}.')
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
    if task.get('protocol') != 'specimen_disjoint_condition_dg':
        raise ValueError('CONDITION_UNVERIFIED: bind the explicit specimen-disjoint physical-condition protocol.')
    validate_condition_task(task)
    if set(task['data'])-{'layout','window_size','windows_per_unit','channel_indices','squeeze_axes'}:
        raise ValueError('Undeclared preprocessing/normalization is forbidden; no target-fitted statistics.')
    names=task['model']['class_names']
    if len(names)!=task['model']['num_classes'] or len(set(names))!=len(names) or len(names)<2:
        raise ValueError('Declare the complete ordered class space once.')
    base=Path(task_path).resolve().parent
    dataset=copy.deepcopy(task['dataset']); mapping=dataset['columns']
    for key in ('metadata_file','h5_file','protocol_file'):
        dataset[key]=str(resolve(dataset[key],base))
    # Identity and condition evidence were checked before metadata was opened.
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
    admitted = pd.concat([src, tst])
    validate_condition_task(task, admitted.to_dict('records'))
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
    frame.loc[~frame[mapping['id']].isin(admitted[mapping['id']])].to_csv(out/'embargo_structure.csv', index=False)
    source_dataset=copy.deepcopy(dataset)
    source_dataset.pop('protocol_file');source_dataset.pop('select',None)
    for key in ('protocol', 'specimen_basis', 'physical_conditions'):
        source_dataset[key] = copy.deepcopy(task[key])
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
    study=read(study_path);validate_study(study, fixture=fixture)
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
        # Reports are not authority: recheck the live scientific declarations.
        validate_condition_task(read(Path(task['path'])/'task.json'))
    return root,study,info


def condition_audit(root):
    """Verify declared physical semantics without reading any signal array."""
    root, study, info = suite(root)
    rows = []
    for task in info['tasks']:
        directory = Path(task['path'])
        spec = read(directory/'task.json')
        structure = pd.concat([pd.read_csv(directory/name, dtype=str, keep_default_na=False)
                               for name in ('source_structure.csv', 'test_structure.csv')])
        # Source labels are permitted here; held-out A/B labels stay sealed.
        source_spec = copy.deepcopy(spec)
        source_spec['dataset'] = read(directory/'source.yaml')['datasets'][0]
        source_labels = pd.read_csv(directory/'source.csv', dtype=str, keep_default_na=False)
        source_support = {r['condition_id']: r['class_support'] for r in
                          validate_condition_task(source_spec, source_labels.to_dict('records'))}
        for row in validate_condition_task(spec, structure.to_dict('records')):
            row['class_support'] = source_support.get(row['condition_id'], '')
            rows.append(dict(dataset=task['dataset_id'], task=task['name'], **row))
    output = root/'condition_audit.csv'
    # Revalidation is deliberate; this table is a readable result, not a bypass token.
    pd.DataFrame(rows).to_csv(output, index=False)
    return rows


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
        if (model['type'], model['name']) == ('X_model', 'TSPN'):
            model.update(in_dim=length, out_dim=length)
        cfg = dict(model=model, arm=arm)
        if arm in study.get('baseline_training', {}):
            cfg['training'] = copy.deepcopy(study['baseline_training'][arm])
        if arm in study.get('baseline_provenance', {}):
            cfg['provenance'] = copy.deepcopy(study['baseline_provenance'][arm])
        return cfg
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
    elif arm.startswith('I-minus-'):
        removed=arm.removeprefix('I-minus-')
        kept=[b for b in m['branches'] if b['name']!=removed]
        if len(kept)!=len(m['branches'])-1:
            raise ValueError('Unknown leave-one-view-out branch.')
        m['branches']=kept
    elif arm not in {'I','Dense','Dense-matched'}: raise ValueError('Unknown proposed/control arm.')
    if arm == 'Dense-matched':
        # Compare readouts on exactly the same learned operator family. Count
        # actual branch dimensions; equal hidden width is not equal capacity.
        with torch.random.fork_rng(devices=[]):
            proposed = model_factory(SimpleNamespace(**m), metadata=None)
        dimension = sum(proposed.feature_dims.values())
        target = sum(p.numel() for p in proposed.operator_heads.parameters())
        slope = dimension + 1 + classes
        width = max(1, int(round((target - classes) / slope)))
        m['head_hidden_dim'] = min({max(1,width-1), width, width+1},
                                  key=lambda h: (abs(h*slope+classes-target), h))
    if arm in {'Dense','Dense-matched','MLP16'}: m['head_type']='mlp'
    return cfg


def preflight(root):
    root,study,info=suite(root)
    condition_audit(root)
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
    condition_audit(root)
    if device=='cpu' and not info['fixture']:
        raise ValueError('CPU training is allowed only for explicit constructed fixtures; use physical GPU0 locally.')
    if device!='cpu' and (device!='cuda:0' or os.environ.get('CUDA_VISIBLE_DEVICES')!='0' or not torch.cuda.is_available()):
        raise ValueError('Training requires CUDA_VISIBLE_DEVICES=0 and available cuda:0; no fallback.')
    return root,study,info


def _training_study(root, study, info):
    """Apply the source-calibrated common update cap, never target outcomes."""
    if info['fixture']:
        return study
    selection = read(Path(root)/'budget_selection.json')
    if selection.get('status') != 'passed' or selection['updates'] not in study['update_budgets']:
        raise ValueError('BUDGET_INSUFFICIENT: finish source-only convergence calibration before HPO.')
    result = copy.deepcopy(study)
    if selection['updates'] % result['steps_per_epoch']:
        raise ValueError('Common update budget must end at a declared source checkpoint.')
    result['epochs'] = selection['updates'] // result['steps_per_epoch']
    return result


def calibrate(root, device, *, retry_interrupted=False):
    """Baseline-only prospective budget calibration; no proposed/target fitting."""
    root, study, info = _development_guard(root, device)
    if not (root/'smoke_baseline.json').exists():
        raise ValueError('Baseline smoke must pass before source convergence calibration.')
    if (root/'budget_selection.json').exists():
        raise FileExistsError('A common budget decision already exists; do not overwrite it.')
    if any((Path(t['path'])/'hpo').exists() for t in info['tasks']):
        raise ValueError('Freeze the common budget before any HPO trial.')
    trial = dict(lr=.001, weight_decay=0., scheduler='none')
    records = []
    budgets = study.get('update_budgets', [study['epochs']*study['steps_per_epoch']])
    for cap in budgets:
        recipe = copy.deepcopy(study)
        if cap % recipe['steps_per_epoch']:
            raise ValueError('Calibration cap must be an integer number of source-checkpoint intervals.')
        recipe['epochs'] = cap // recipe['steps_per_epoch']
        improving = []
        for task in info['tasks']:
            reference = None
            for arm in ['p0', *study['baselines']]:
                out = Path(task['path'])/'calibration'/str(cap)/arm
                if retry_interrupted:
                    _execute_or_resume(recipe, task, arm, trial, study['hpo_seed'], out, device, reference, None, True)
                elif out.exists():
                    _completed_run(recipe, task, arm, trial, study['hpo_seed'], out, reference, None)
                else:
                    _execute(recipe, task, arm, trial, study['hpo_seed'], out, device, reference)
                gain = _recent_source_improvement(out)
                still_improving = gain >= .001
                improving.append(still_improving)
                records.append(dict(task=task['name'], arm=arm, updates=cap,
                                    recent_source_improvement=gain, still_improving=still_improving,
                                    run=str(out)))
                if arm == 'p0': reference = out
        write_csv(root/'calibration.csv', records)
        if not any(improving):
            dump(root/'budget_selection.json', dict(status='passed', updates=cap,
                 rule='last_five_checkpoints_improvement_below_0.001_for_all_baselines', target_read=False))
            return
    dump(root/'budget_selection.json', dict(status='budget_insufficient', updates=budgets[-1], target_read=False))
    raise ValueError('BUDGET_INSUFFICIENT: a baseline still improves at the maximum common source budget.')


def _recent_source_improvement(run):
    history = pd.read_csv(Path(run)/'training.csv')['worst_validation_score'].to_numpy(float)
    if len(history) < 5 or not np.isfinite(history).all():
        raise ValueError('BUDGET_INSUFFICIENT: five finite source checkpoints are required to assess recent improvement.')
    return float(history[-5] - min(history[-4:]))


def _reference(task):
    record=read(Path(task['path'])/'hpo'/'p0'/'selection.json')
    if not record.get('qualification', {}).get('passed', False):
        raise ValueError('BASELINE_UNQUALIFIED: the complete reference must qualify on every source condition.')
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
        flags+=['--role','reference' if arm=='p0' else 'baseline', '--pair-shift',str(study['pair_shift'])]
        if arm!='p0':flags+=['--reference-config',str(reference/'model_config.yaml'),
                             '--reference-checkpoint',str(reference/'selected_candidate.pt')]
    else:
        module='experiments.p01.train_tspn_fusion_v2'
        flags+=['--pair-shift',str(study['pair_shift']),'--arm',arm,
                '--reference-min-accuracy',str(study['reference_min_accuracy'])]
    command=[sys.executable,'-m',module,*flags]
    env=dict(os.environ,PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH',''))
    logpath=output.parent/(output.name+'.log')
    with logpath.open('x') as stream:
        run=subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT)
    if run.returncode:
        output.mkdir(exist_ok=True)
        failure = output/('process_failure.json' if (output/'failure.json').exists() else 'failure.json')
        dump(failure,dict(command=command,returncode=run.returncode,log=str(logpath)))
        raise RuntimeError(f'Training failed without changing recipe; inspect {logpath}')
    return output


def _completed_run(study,task,arm,trial,seed,out,reference,view):
    out=Path(out)
    if (out/'failure.json').exists() or not (out/'result_scope.json').exists():
        raise ValueError('Interrupted/failed run cannot be silently omitted or overwritten.')
    if read(out/'result_scope.json').get('status') not in {'classifier_fitted', 'candidate_fitted'}:
        raise ValueError('Only a completed training run can be selected or resumed.')
    command=read(out/'command.json')
    expected=dict(trial,seed=seed,epochs=study['epochs'],steps_per_epoch=study['steps_per_epoch'],
                  units_per_domain=study['units_per_domain'],dg=True)
    if any(command.get(k)!=v for k,v in expected.items()):
        raise ValueError('Completed run recipe differs from the frozen trial/seed/budget.')
    # Saved training models may add derived fields; the original submitted YAML
    # is retained separately and must exactly match the study-derived recipe.
    submitted=read(out.parent/(out.name+'.yaml'))
    if submitted!=candidate_config(study,task,arm,reference,view):
        raise ValueError('Completed run inputs, model or selected view changed.')


def _selected(available, requested, label):
    """Filter execution, never the bound scientific population or freeze matrix."""
    if requested is None:
        return list(available)
    if not requested or len(set(requested)) != len(requested) or set(requested) - set(available):
        raise ValueError(f'Unknown, empty or duplicate {label} selection: {requested}; available={list(available)}')
    return [item for item in available if item in requested]


def _execute_or_resume(study, task, arm, trial, seed, out, device, reference, view, retry_interrupted=False):
    """Keep exact completed fits; explicitly restart only an interrupted attempt.

    Selected checkpoints are not optimizer/RNG snapshots. Restart the SAME trial
    and seed from initialization, retaining the incomplete directory and log. A
    numerical/qualification failure or changed recipe is never automatically retried.
    """
    out = Path(out)
    log = out.parent / (out.name + '.log')
    if not out.exists() and not log.exists():
        return _execute(study, task, arm, trial, seed, out, device, reference, view)
    if out.exists() and not retry_interrupted:
        _completed_run(study, task, arm, trial, seed, out, reference, view)
        return out
    scope = read(out/'result_scope.json') if (out/'result_scope.json').exists() else {}
    if scope.get('status') in {'classifier_fitted', 'candidate_fitted'}:
        _completed_run(study, task, arm, trial, seed, out, reference, view)
        return out
    if not retry_interrupted:
        raise ValueError(f'Incomplete run: {out}. Preserve it; --retry-interrupted explicitly restarts only a signal-interrupted trial.')
    submitted = read(out.parent/(out.name+'.yaml'))
    if submitted != candidate_config(study, task, arm, reference, view):
        raise ValueError('Interrupted model/configuration changed; use a new study.')
    if (out/'command.json').exists():
        command = read(out/'command.json')
        expected = dict(trial, seed=seed, epochs=study['epochs'], steps_per_epoch=study['steps_per_epoch'],
                        units_per_domain=study['units_per_domain'], dg=True)
        if any(command.get(k) != v for k, v in expected.items()):
            raise ValueError('Interrupted trial/seed/budget differs from the frozen recipe.')
    failures = [read(out/n) for n in ('failure.json', 'process_failure.json') if (out/n).exists()]
    interrupted_failure = lambda f: (f.get('returncode') in {-2, -15, 130, 143}
        or (f.get('status') == 'interrupted' and f.get('error_type') == 'KeyboardInterrupt'))
    if any(not interrupted_failure(f) for f in failures) or scope.get('status') not in {None, 'running', 'interrupted'}:
        raise ValueError('A failed numerical experiment cannot be retried as an interruption.')
    archive = Path(task['path'])/'interrupted'/f'{arm}_{seed}_{time.time_ns()}'
    archive.mkdir(parents=True, exist_ok=False)
    for path in (out, log, out.parent/(out.name+'.yaml')):
        if path.exists():
            shutil.move(str(path), str(archive/path.name))
    dump(archive/'restart.json', dict(arm=arm, seed=seed, trial=trial, original_run=str(out),
                                    reason='explicit_restart_from_initialization', target_read=False))
    return _execute(study, task, arm, trial, seed, out, device, reference, view)


def _score(run, reference=False):
    table=pd.read_csv(Path(run)/'selected_source_validation.csv',dtype={'domain':str})
    q=table[table.predictor=='candidate'].set_index('domain')
    p=table[table.predictor=='raw'].set_index('domain')
    # The inherited reference uses ordinary CE; all downstream candidates share
    # the exact same CE+.25 Brier excess selector and source observation set.
    score=float(q.ce.max()) if reference else float(((q.ce-p.ce)+.25*(q.brier-p.brier)).max())
    if not math.isfinite(score): raise ValueError('Nonfinite source selection score.')
    return score


def class_profile(arrays):
    """Use the existing acquisition/group estimator for source class diagnostics."""
    p=dict(arrays)
    p.setdefault('deployed_probs',p['candidate_probs'])
    p.setdefault('deployed_log_probs',p['candidate_log_probs'])
    grouped=analyze_d1.group_estimates(analyze_d1.acquisition_estimates(p),p['candidate_probs'].shape[1])
    output={}
    for domain in sorted(set(p['domains'])):
        rows=[g for g in grouped if g['domain']==domain and g['predictor']=='candidate']
        cm=np.mean(np.stack([g['confusion_matrix'] for g in rows]),axis=0)
        support=cm.sum(axis=1)
        recall=[float(cm[c,c]/v) if v else None for c,v in enumerate(support)]
        output[str(domain)]=dict(weighted_support=support.tolist(),recall=recall,
                                 confusion=cm.tolist(),class_collapse=any(x==0 for x in recall if x is not None))
    return output


def _qualify_run(task, run, threshold=.8):
    data, dataset, _ = source_data(task)
    with np.load(Path(run)/'selected_source_validation_windows.npz', allow_pickle=False) as archive:
        arrays = {key: archive[key] for key in archive.files}
    return qualify_source_prediction_arrays(arrays, list(map(str, dataset['source_domains'])),
                                            data['model']['num_classes'], threshold)


def _baseline_qualifications(study, info):
    reports = []
    for task in info['tasks']:
        reference = _reference(task)
        reports.append(dict(task=task['name'], arm='p0', seed=study['hpo_seed'], shared_reference=True,run=str(reference),
                            **_qualify_run(task, reference, study['reference_min_accuracy'])))
        for arm in study['baselines']:
            selected = read(Path(task['path'])/'hpo'/arm/'selection.json')
            for seed in study['seeds']:
                run = Path(task['path'])/'fits'/arm/str(seed)
                _completed_run(study, task, arm, selected['trial'], seed, run, reference, None)
                reports.append(dict(task=task['name'], arm=arm, seed=seed, shared_reference=False,run=str(run),
                                    **_qualify_run(task, run, study['reference_min_accuracy'])))
    for row in reports:
        row['performance_passed'] = row['passed']
        if row['arm'] in study.get('baseline_provenance', {}):
            row['provenance'] = copy.deepcopy(study['baseline_provenance'][row['arm']])
        if not info['fixture']:
            if row.get('provenance', {}).get('status') == 'implementation_unverified':
                row['implementation_passed'] = False
                row['passed'] = False
                row['reasons'].append('BASELINE_IMPLEMENTATION_UNVERIFIED: unresolved paper-to-configuration semantics; source diagnostics are not faithful baseline evidence.')
            gain = _recent_source_improvement(row['run'])
            row['recent_source_improvement'] = gain
            row['convergence_passed'] = gain < .001
            if not row['convergence_passed']:
                row['passed'] = False
                row['reasons'].append('BUDGET_INSUFFICIENT: source criterion still improves at the common cap.')
    return reports


def qualify(root, device='cpu'):
    """Qualify the actual selected baseline fits, not seed/condition averages."""
    root, study, info = suite(root)
    condition_audit(root)
    study = _training_study(root, study, info)
    reports = _baseline_qualifications(study, info)
    result = dict(passed=all(row['passed'] for row in reports), runs=reports, target_read=False)
    path = root/'baseline_qualification.json'
    if path.exists():
        if read(path) != result:
            raise ValueError('Baseline qualification changed; preserve the failed study and diagnose before rerunning.')
    else:
        dump(path, result)
    if not result['passed']:
        if any(row.get('implementation_passed') is False for row in reports):
            raise ValueError('BASELINE_IMPLEMENTATION_UNVERIFIED: resolve the declared baseline semantics before proposed development or target release.')
        if any(row.get('convergence_passed') is False for row in reports):
            raise ValueError('BUDGET_INSUFFICIENT: a selected baseline still improves; preserve this study and revise the shared budget before target access.')
        raise ValueError('BASELINE_UNQUALIFIED: at least one fixed baseline seed/source condition failed; no proposed comparison.')
    return result


def _require_qualified_baselines(root, study, info):
    if not (Path(root)/'baseline_qualification.json').exists():
        raise ValueError('BASELINE_UNQUALIFIED: fit all fixed baseline seeds and run qualify before proposed development.')
    # Recompute from the selected source predictions; a stale PASS file cannot
    # override missing conditions/classes or a failed fixed seed.
    reports = _baseline_qualifications(study, info)
    if any(row.get('implementation_passed') is False for row in reports):
        raise ValueError('BASELINE_IMPLEMENTATION_UNVERIFIED: source qualification cannot certify an unresolved literature mapping.')
    if any(row.get('convergence_passed') is False for row in reports):
        raise ValueError('BUDGET_INSUFFICIENT: current baseline source trajectories fail the frozen recent-improvement rule.')
    if not reports or not all(row['passed'] for row in reports):
        raise ValueError('BASELINE_UNQUALIFIED: current baseline source predictions fail qualification.')


def tune(root, family, device, *, task_names=None, selected_arms=None, retry_interrupted=False):
    root,study,info=_development_guard(root,device)
    study = _training_study(root, study, info)
    if not (root/'smoke_baseline.json').exists(): raise ValueError('Baseline smoke must pass before HPO.')
    arms=['p0'] if family=='reference' else list(study['baselines']) if family=='baselines' else list(CORE)
    if family=='method' and not (root/'smoke_method.json').exists(): raise ValueError('Method smoke must pass before proposed HPO.')
    if family=='method': _require_qualified_baselines(root, study, info)
    arms = _selected(arms, selected_arms, 'arms')
    tasks = _selected([t['name'] for t in info['tasks']], task_names, 'tasks')
    for task in info['tasks']:
        if task['name'] not in tasks: continue
        reference=None if family=='reference' else _reference(task)
        for arm in arms:
            directory=Path(task['path'])/'hpo'/arm
            # A recorded selection does not bypass verification of its complete
            # original search; rerunning this stage adds no search opportunities.
            recorded = read(directory/'selection.json') if (directory/'selection.json').exists() else None
            trials=[]
            views=[b['name'] for b in study['fusion']['model']['branches']]
            for index,trial in enumerate(study['trials']):
                view=views[index%len(views)] if arm=='I-single' else None
                out=directory/f'trial_{index:02d}'
                _execute_or_resume(study,task,arm,trial,study['hpo_seed'],out,device,reference,view,
                                   retry_interrupted=retry_interrupted)
                trials.append(dict(index=index,score=_score(out,arm=='p0'),run=str(out),trial=trial,view=view))
            best=min(trials,key=lambda t:(t['score'],t['index']))
            with np.load(Path(best['run'])/'selected_source_validation_windows.npz',allow_pickle=False) as saved:
                profile=class_profile({k:saved[k] for k in saved.files})
            qualification = _qualify_run(task, best['run']) if arm == 'p0' or arm in study['baselines'] else None
            selection = dict(**best,all_trials=trials,source_class_profile=profile,
                 qualification=qualification, criterion='worst_source_CE' if arm=='p0' else 'worst_source_CE_plus_0.25_Brier_excess',target_read=False)
            if recorded is not None:
                if recorded != selection:
                    raise ValueError('Recorded HPO selection differs from the complete frozen search.')
            else:
                dump(directory/'selection.json', selection)
            if arm=='p0' and not qualification['passed']:
                raise ValueError('BASELINE_UNQUALIFIED: reference failed after the complete source-only HPO budget.')


def fit(root,device,family='all', *, task_names=None, selected_arms=None, selected_seeds=None, retry_interrupted=False):
    root,study,info=_development_guard(root,device)
    study = _training_study(root, study, info)
    if family not in {'all', 'baselines', 'method'}:
        raise ValueError('Choose baselines or method for final fitting.')
    if family == 'all':
        if any(v is not None for v in (task_names, selected_arms, selected_seeds)) or retry_interrupted:
            raise ValueError('Filtered fitting requires an explicit baselines or method family.')
        fit(root, device, 'baselines')
        qualify(root, device)
        return fit(root, device, 'method')
    if family == 'method': _require_qualified_baselines(root, study, info)
    arms = list(study['baselines']) if family == 'baselines' else [*CORE, *ablation_arms(study)]
    arms = _selected(arms, selected_arms, 'arms')
    seeds = _selected(study['seeds'], selected_seeds, 'seeds')
    tasks = _selected([t['name'] for t in info['tasks']], task_names, 'tasks')
    for task in info['tasks']:
        if task['name'] not in tasks: continue
        reference=_reference(task)
        for arm in arms:
            selector='I' if arm.startswith('I-minus-') else arm
            selected=read(Path(task['path'])/'hpo'/selector/'selection.json')
            for seed in seeds:
                out=Path(task['path'])/'fits'/arm/str(seed)
                _execute_or_resume(study,task,arm,selected['trial'],seed,out,device,reference,selected['view'],
                                   retry_interrupted=retry_interrupted)


def _reference_wrapper(path,device):
    saved=torch.load(path,map_location='cpu',weights_only=True)
    saved=dict(saved,reference_model=saved['model'],reference_state_dict=saved['state_dict'],reference_temperature=1.)
    return FrozenClassifier(saved,device)


def smoke(root,kind,device):
    root,study,info=_development_guard(root,device)
    if kind == 'method':
        study = _training_study(root, study, info)
        _require_qualified_baselines(root, study, info)
    torch.set_num_threads(1);torch.manual_seed(study['hpo_seed'])
    reports=[]
    for task in info['tasks']:
        data,dataset,records=source_data(task)
        chosen=[]
        for label in range(data['model']['num_classes']):
            chosen.append(next(r for r in records if r['split']=='update' and r['label']==label))
        xs=[window_record(r,dataset,data['data'])[:1] for r in chosen]
        x=torch.cat(xs).to(device);y=torch.tensor([r['label'] for r in chosen],device=device)
        # Tiny-source memorization diagnoses optimization; these weights never
        # enter HPO, final fitting or test evaluation.
        temporary=Path(task['path'])/f'smoke_{kind}';temporary.mkdir(exist_ok=False)
        reference=None
        if kind=='method':
            reference=_reference(task)
            names=['I']
        else:names=['p0', *study['baselines']]
        for name in names:
            cfg=candidate_config(study,task,name,reference)
            cfg['model']['device']=device
            model=model_factory(SimpleNamespace(**cfg['model']),metadata=None).to(device)
            def logits():
                return model.forward_details(x)['candidate_logits'] if name=='I' else model(x)
            model.eval()
            with torch.no_grad():
                initial=float(torch.nn.functional.cross_entropy(logits(),y))
                if name=='I':
                    out=model.forward_details(x)
                    torch.testing.assert_close(out['candidate_probs'],out['raw_probs'],atol=1e-7,rtol=1e-6)
            reference_state={k:v.clone() for k,v in model.reference.state_dict().items()} if name=='I' else None
            optimizer=torch.optim.Adam([p for p in model.parameters() if p.requires_grad],lr=.003)
            for step in range(study['overfit_steps']):
                model.train();value=logits()
                if value.shape!=(len(y),data['model']['num_classes']):raise ValueError('Logit shape/class mismatch.')
                loss=torch.nn.functional.cross_entropy(value,y)
                if not torch.isfinite(loss):raise FloatingPointError('Nonfinite tiny-source loss.')
                optimizer.zero_grad(set_to_none=True);loss.backward()
                if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                    raise FloatingPointError('Nonfinite gradient.')
                if not any(p.grad is not None and bool(p.grad.abs().sum()>0) for p in model.parameters() if p.requires_grad):
                    raise ValueError('All trainable gradients vanished.')
                optimizer.step()
            model.eval()
            with torch.no_grad():
                final=float(torch.nn.functional.cross_entropy(logits(),y));accuracy=float((logits().argmax(-1)==y).float().mean())
                before=logits().clone()
            checkpoint=temporary/(name+'.pt');torch.save(model.state_dict(),checkpoint)
            model.load_state_dict(torch.load(checkpoint,map_location=device,weights_only=True),strict=True)
            with torch.no_grad():torch.testing.assert_close(before,logits(),rtol=1e-6,atol=1e-7)
            if name=='I':
                if any(not torch.equal(v,model.reference.state_dict()[k]) for k,v in reference_state.items()):raise ValueError('Frozen reference changed.')
                out=model.forward_details(x);parts=torch.stack(list(out['branch_logit_contributions'].values())).sum(0)
                raw=out['raw_logits']/float(model.reference_temperature)
                torch.testing.assert_close(parts,out['candidate_logits']-raw,atol=2e-6,rtol=1e-6)
            report=dict(task=task['name'],arm=name,initial_CE=initial,final_CE=final,tiny_source_accuracy=accuracy,
                        trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad),
                        passed=final<initial and accuracy>=.95,target_read=False)
            dump(temporary/(name+'.json'),report);reports.append(report)
            if not report['passed']:raise ValueError(f'Tiny-source memorization failed: {task["name"]}/{name}. Diagnose; do not start formal fitting.')
    dump(root/f'smoke_{kind}.json',dict(status='passed',checks=reports))


def reconstruction(arrays):
    keys=sorted(k for k in arrays if k.startswith('logit_contribution__'))
    if not keys:return None
    total=sum(arrays[k].astype(np.float64) for k in keys)
    # Subtract a fixed class-0 contrast, never unstable log(prob) in low precision.
    direct=arrays['candidate_log_probs'].astype(np.float64)-arrays['raw_log_probs'].astype(np.float64)
    error=(total-total[:,:1])-(direct-direct[:,:1])
    maximum=float(np.abs(error).max())
    scale=1+float(np.abs(direct).max())
    if maximum>1e-5*scale:raise ValueError('Actual branch contributions do not reconstruct direct class contrasts.')
    return dict(max_abs=maximum,mean_abs=float(np.abs(error).mean()),tolerance=1e-5*scale,reference_branch_separate=True)


def freeze(root,device):
    root,study,info=_development_guard(root,device)
    study = _training_study(root, study, info)
    _require_qualified_baselines(root, study, info)
    frozen=[];quality=[];task_snapshots=[];inference_costs=[]
    for task in info['tasks']:
        reference=_reference(task)
        for arm in [*study['baselines'],*CORE,*ablation_arms(study)]:
            selector='I' if arm.startswith('I-minus-') else arm
            selected=read(Path(task['path'])/'hpo'/selector/'selection.json')
            for seed in study['seeds']:
                _completed_run(study,task,arm,selected['trial'],seed,Path(task['path'])/'fits'/arm/str(seed),reference,selected['view'])
    for task in info['tasks']:
        data,dataset,records=source_data(task)
        validation=[r for r in records if r['split']=='validation']
        reference=_reference(task);ref=torch.load(reference/'selected_candidate.pt',map_location='cpu',weights_only=True)
        destination=Path(task['path'])/'frozen';destination.mkdir(exist_ok=False)
        # Reuse a fixed, source-validation-only batch for all predictors. Timing
        # neither opens a target nor changes a model/configuration selection.
        timing_cap=min(32,len(dataset['source_domains'])*study['units_per_domain'])
        timing_records=validation[:timing_cap]
        timing_x=torch.cat([window_record(r,dataset,data['data'])[:1] for r in timing_records]).to(device)
        def record_inference(arm, seed, predictor):
            profile=profile_predictor(predictor,timing_x,device)
            profile.update(task=task['name'],arm=arm,seed=seed,target_read=False,
                           acquisition_ids=[str(r['acquisition_id']) for r in timing_records])
            dump(destination/f'{arm}_{seed}_inference.json',profile)
            reference_cost=profile['reference_only'] or {}
            inference_costs.append(dict(task=task['name'],arm=arm,seed=seed,
                input_shape=json.dumps(profile['input_shape']),dtype=profile['dtype'],device=profile['device'],
                hardware=profile['hardware'],warmup=profile['warmup'],repeats=profile['repeats'],
                full_median_ms=profile['full']['median_ms'],full_p95_ms=profile['full']['p95_ms'],
                full_forward_peak_increment_bytes=profile['full']['forward_peak_increment_bytes'],
                reference_median_ms=reference_cost.get('median_ms'),
                incremental_median_ms=profile['incremental_median_ms'],
                timing_boundary=profile['timing_boundary'],memory_boundary=profile['memory_boundary'],target_read=False))
        reference_predictor=_reference_wrapper(reference/'selected_candidate.pt',device).candidate
        record_inference('p0',study['hpo_seed'],reference_predictor)
        del reference_predictor
        spec=read(Path(task['path'])/'task.json')
        task_snapshots.append(dict(**task,spec=spec,
            test_structure=pd.read_csv(Path(task['path'])/'test_structure.csv',dtype=str,keep_default_na=False).to_dict('records'),
            source_groups=sorted({r['unit_id'] for r in records})))
        for arm in [*study['baselines'],*CORE,*ablation_arms(study)]:
            for seed in study['seeds']:
                directory=Path(task['path'])/'fits'/arm/str(seed)
                if not (directory/'result_scope.json').exists() or (directory/'failure.json').exists():
                    raise ValueError('All declared final fits must complete before any target release.')
                model,saved=load_model(directory/'selected_candidate.pt',device)
                if any(not torch.equal(v.cpu(),model.reference.state_dict()[k].cpu()) for k,v in ref['state_dict'].items()):
                    raise ValueError('Candidates do not share the selected complete reference.')
                classes=data['model']['class_names']
                predictions=predict_records(model,validation,dataset,data,classes,device,alpha=1.)
                reconstruction(predictions)
                with np.load(directory/'selected_source_validation_windows.npz',allow_pickle=False) as old:
                    expected={k:old[k] for k in old.files}
                expected['deployed_probs']=expected['candidate_probs'];expected['deployed_log_probs']=expected['candidate_log_probs']
                verify_vectors(expected,predictions)
                path=destination/f'{arm}_{seed}.pt'
                save_bundle(model,saved['model'],path,kind='model',temperature=1.,alpha=1.,classes=classes,
                            input_data=data['data'],sampling_rate=records[0]['sample_rate_hz'],scope='prospective_DG_no_target_selection')
                restored = destination/f'{arm}_{seed}_source_restore'
                command = [sys.executable, '-m', 'experiments.p01.fusion_deployment', 'predict',
                           '--bundle', str(path), '--data-config', str(Path(task['path'])/'source.yaml'),
                           '--dataset', dataset['name'], '--split', 'validation', '--sources-only',
                           '--device', device, '--output', str(restored)]
                env = dict(os.environ, PYTHONPATH=str(ROOT)+os.pathsep+os.environ.get('PYTHONPATH', ''),
                           OMP_NUM_THREADS='1', MKL_NUM_THREADS='1')
                with (destination/f'{arm}_{seed}_source_restore.log').open('x') as stream:
                    subprocess.run(command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, check=True)
                with np.load(restored/'predictions.npz', allow_pickle=False) as actual:
                    verify_vectors(predictions, actual)
                # Reload the deployed alpha=1 model; selected fusion training
                # checkpoints retain alpha=0, which would time only p0.
                del model
                deployed,_=load_model(path,device)
                if isinstance(deployed,FrozenClassifier):deployed=deployed.candidate
                record_inference(arm,seed,deployed)
                del deployed
                selector='I' if arm.startswith('I-minus-') else arm
                chosen=read(Path(task['path'])/'hpo'/selector/'selection.json')
                frozen.append(dict(task=task['name'],arm=arm,seed=seed,checkpoint=str(path),view=chosen['view']))
                profile=class_profile(predictions)
                for row in summarize_rows(acquisition_rows(predictions),len(classes)):
                    if row['predictor']=='candidate':
                        quality.append(dict(task=task['name'],arm=arm,seed=seed,**row,
                            source_class_profile=json.dumps(profile[str(row['domain'])],allow_nan=False)))
    write_csv(root/'source_quality.csv',quality)
    write_csv(root/'inference_costs.csv',inference_costs)
    # A single suite barrier: all splits and datasets are selected before any target.
    dump(root/'frozen.json',dict(tasks=task_snapshots,study=study,models=frozen,fixture=info['fixture'],target_read=False))


def test(root,device):
    root, original_study, info = suite(root)
    condition_audit(root)
    frozen=read(root/'frozen.json')
    for task in frozen['tasks']:
        directory=Path(task['path']);spec=task['spec'];dataset=copy.deepcopy(spec['dataset']);m=dataset['columns']
        stat=Path(dataset['h5_file']).stat()
        if [stat.st_size,stat.st_mtime_ns]!=task['h5_stat']:
            raise ValueError('Read-only H5 changed after freezing.')
        frame=pd.DataFrame(task['test_structure'])
        labels=_metadata_rows(dataset,set(frame[m['id']]))
        structural=[c for c in frame.columns if c not in {m['unit_id'],m['split']}]
        pd.testing.assert_frame_equal(
            labels[structural].sort_values(m['id']).reset_index(drop=True),
            frame[structural].sort_values(m['id']).reset_index(drop=True))
        labels=labels.merge(frame[[m['id'],m['unit_id'],m['split']]],on=m['id'],validate='one_to_one')
        test_csv=directory/'test.csv'
        if test_csv.exists():
            pd.testing.assert_frame_equal(pd.read_csv(test_csv,dtype=str,keep_default_na=False),labels.reset_index(drop=True))
        else: labels.to_csv(test_csv,index=False)
        dataset.pop('protocol_file');dataset.pop('select',None)
        for key in ('protocol', 'specimen_basis', 'physical_conditions'):
            dataset[key] = copy.deepcopy(spec[key])
        dataset.update(metadata_file=str(test_csv),access_scope='test')
        data=dict(model=spec['model'],data=spec['data'],datasets=[dataset]);records=read_records(dataset,data)
        evaluated_spec=dict(spec,dataset=dataset)
        pd.DataFrame(validate_condition_task(evaluated_spec, labels.to_dict('records'))).to_csv(
            directory/'test_condition_audit.csv', index=False)
        source_ids=set(task['source_groups'])
        if source_ids&{r['unit_id'] for r in records}:raise ValueError('Target records cross the source physical partition.')
        exports=[]
        out=directory/'predictions';out.mkdir(exist_ok=True)
        for item in [x for x in frozen['models'] if x['task']==task['name']]:
            path=out/f'{item["arm"]}_{item["seed"]}.npz'
            entry=dict(name=path.stem,arm=item['arm'],seed=item['seed'],split='test',path=str(path),alpha=1.,role='direct')
            if not path.exists():
                model,_=load_model(item['checkpoint'],device)
                arrays=predict_records(model,records,dataset,data,data['model']['class_names'],device,alpha=1.)
                reconstruction(arrays)
                np.savez_compressed(path,**arrays)
            analyze_d1.load_artifact(entry)  # Validate every completed or resumed export.
            exports.append(entry)
        filename=directory/'exports.json'
        if filename.exists():
            if read(filename)!=exports:raise ValueError('Frozen export identities changed.')
        else:dump(filename,exports)
    if not (root/'test_complete.json').exists():dump(root/'test_complete.json',dict(status='complete',frozen_models=len(frozen['models'])))


def contribution_rows(arrays,task,arm,seed):
    """Mean WINDOW logit contributions, never additive acquisition log-odds."""
    rows=[]
    keys=[k for k in arrays if k.startswith('logit_contribution__')]
    for acquisition in np.unique(arrays['acquisition_ids']):
        mask=arrays['acquisition_ids']==acquisition
        label_values=np.unique(arrays['labels'][mask])
        if len(label_values)!=1: raise ValueError('An acquisition has multiple labels.')
        label=int(label_values[0]);prediction=int(arrays['candidate_probs'][mask].mean(0).argmax())
        domain=str(arrays['domains'][mask][0]);group=str(arrays['group_ids'][mask][0])
        for key in keys:
            values=arrays[key][mask].mean(0)
            for c in range(1,len(values)):
                rows.append(dict(task=task,arm=arm,seed=seed,domain=domain,group_id=group,
                                 acquisition_id=acquisition,label=label,correct=int(prediction==label),
                                 branch=key.split('__',1)[1],class_c=c,class_reference=0,
                                 mean_window_logit_contribution=float(values[c]-values[0])))
    return rows


def plot_contrasts(frame,output):
    """Plot actual per-task contrasts and intervals without pooling tasks."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for (task,scope,metric),part in frame.groupby(['task','scope','metric'],sort=True):
        if metric not in {'accuracy','brier'} or scope not in {'target','heldout_source'}: continue
        if not (part.status=='complete').all(): raise ValueError('Incomplete comparison cannot become a paper figure.')
        values=part[['finite_seed_mean','lower','upper']].to_numpy(float)
        if not np.isfinite(values).all(): raise ValueError('Nonfinite contrast/interval.')
        y=np.arange(len(part));fig,ax=plt.subplots(figsize=(8,max(3,len(part)*.4)))
        ax.hlines(y,values[:,1],values[:,2]);ax.plot(values[:,0],y,'o');ax.axvline(0,linestyle=':')
        ax.set_yticks(y,part.contrast);ax.set_xlabel(f'Proposed minus comparator: {metric}')
        ax.set_title(f'{task} / {scope}');fig.tight_layout()
        fig.savefig(output/f'{task}_{scope}_{metric}.pdf');plt.close(fig)


def analyze(root):
    root=Path(root).resolve();frozen=read(root/'frozen.json')
    if not (root/'test_complete.json').exists():raise ValueError('Finish the complete frozen target release first.')
    all_metrics=[];all_contrasts=[];explanations=[];contributions=[];costs=[];reference_metrics=[];condition_differences=[]
    explanation_cases=[];explanation_strata=[]
    ablations=list(ablation_arms(frozen['study']))
    core=[*frozen['study']['baselines'],*CORE,*ablations]
    contrasts={**CONTRASTS,**{f'I-{b}':('I',b) for b in frozen['study']['baselines']},
               **{f'I-vs-no-{a.removeprefix("I-minus-")}':('I',a) for a in ablations}}
    for task in frozen['tasks']:
        directory=Path(task['path']);exports=read(directory/'exports.json')
        expected={(a,s) for a in core for s in frozen['study']['seeds']}
        actual=[(x['arm'],x['seed']) for x in exports]
        if len(actual)!=len(expected) or set(actual)!=expected:
            raise ValueError('The explicit current matrix is incomplete or duplicated; do not analyze a favorable subset.')
        structural=pd.DataFrame(task['test_structure']);spec=task['spec']
        domains=set(structural[spec['dataset']['columns']['domain']]);target=[str(x) for x in spec['dataset']['domain_sequence']]
        source=sorted(domains-set(target));sets={'target':target}
        if source:sets['heldout_source']=source
        mapping=spec['dataset']['columns']
        declared_sources=list(map(str,spec['dataset']['source_domains']))
        target_groups=set(structural.loc[structural[mapping['domain']]==target[0],mapping['unit_id']])
        paired_groups={d: sorted(target_groups & set(structural.loc[structural[mapping['domain']]==d,mapping['unit_id']]))
                       for d in declared_sources}
        if not (directory/'analysis').exists():
            analyze_d1.run(exports,directory/'analysis',condition_sets={'test':sets},core_arms=core,
                           contrasts=contrasts,seeds=frozen['study']['seeds'])
        recorded=read(directory/'analysis'/'analysis.json')
        if (recorded['status']!='complete' or recorded['declared_core_arms']!=core or
                recorded['declared_seeds']!=frozen['study']['seeds'] or
                recorded['declared_contrasts']!={k:list(v) for k,v in contrasts.items()}):
            raise ValueError('Existing analysis does not describe the complete frozen comparison matrix.')
        for filename,output in [('seed_summary.csv',all_metrics),('contrast_seed_summary.csv',all_contrasts)]:
            rows=pd.read_csv(directory/'analysis'/filename).to_dict('records')
            output.extend(dict(dataset=task['dataset_id'],task=task['name'],**r) for r in rows)
        for entry_index, entry in enumerate(exports):
            with np.load(entry['path'],allow_pickle=False) as archive:arrays={k:archive[k] for k in archive.files}
            for row in paired_condition_rows(arrays, declared_sources, target[0], eligible_groups=paired_groups):
                if row['predictor']=='raw' and entry_index:
                    continue  # Shared p0 is one fitted function, not repeated reference trials.
                condition_differences.append(dict(task=task['name'],dataset=task['dataset_id'],
                    arm='p0' if row['predictor']=='raw' else entry['arm'],
                    seed=frozen['study']['hpo_seed'] if row['predictor']=='raw' else entry['seed'],**row))
            check=reconstruction(arrays)
            if check:contributions.extend(contribution_rows(arrays,task['name'],entry['arm'],entry['seed']))
            if check:explanations.append(dict(task=task['name'],arm=entry['arm'],seed=entry['seed'],**check))
            if entry['arm']=='I' and entry['seed']==42:
                cases=select_explanation_cases(arrays,conditions=[*declared_sources,*target],
                                                class_names=spec['model']['class_names'],seed=42)
                dump(directory/'analysis'/'explanation_cases.json',cases)
                for row in cases['cases']:
                    flattened=dict(row,signed_branch_terms=json.dumps(row['signed_branch_terms'],sort_keys=True),
                                   descriptors=json.dumps(row['descriptors'],sort_keys=True))
                    explanation_cases.append(dict(task=task['name'],dataset=task['dataset_id'],seed=42,**flattened))
                explanation_strata.extend(dict(task=task['name'],dataset=task['dataset_id'],seed=42,**row)
                                          for row in cases['strata'])
        # The reference is shared: report it once per task, not as three fits.
        first=exports[0]
        with np.load(first['path'],allow_pickle=False) as archive: raw={k:archive[k] for k in archive.files}
        for row in summarize_rows(acquisition_rows(raw),spec['model']['num_classes']):
            if row['predictor']=='raw':reference_metrics.append(dict(task=task['name'],**row))
        for command in (sorted((directory/'hpo').glob('*/trial_*/command.json'))+
                        sorted((directory/'fits').glob('*/*/command.json'))+
                        sorted((directory/'calibration').glob('*/*/command.json'))):
            scope=read(command.parent/'result_scope.json')
            config=read(command)
            calibration='calibration' in command.parts
            endpoints=scope.get('supervised_window_endpoints')
            if endpoints is None and scope.get('sampled_windows') is not None and scope.get('supervised_endpoints') is not None:
                endpoints=scope['sampled_windows']*scope['supervised_endpoints']
            costs.append(dict(task=task['name'],stage='calibration' if calibration else 'hpo' if 'hpo' in command.parts else 'fit',
                arm=command.parent.name if calibration else command.parent.parent.name,seed=config['seed'],run=str(command.parent),
                status=scope['status'],optimizer_steps=scope.get('optimizer_steps'),
                trainable_parameters=scope.get('trainable_parameters'),total_parameters=scope.get('total_parameters'),
                reference_parameters=scope.get('reference_parameters'),training_seconds=scope.get('training_seconds'),
                supervised_endpoints=scope.get('supervised_endpoints'),
                sampled_windows=scope.get('sampled_windows'),
                supervised_window_endpoints=endpoints,
                selection_seconds=scope.get('source_selection_seconds',scope.get('selection_seconds')),
                export_seconds=scope.get('export_seconds'),total_run_seconds=scope.get('total_run_seconds'),
                peak_allocated_gpu_bytes=scope.get('peak_allocated_gpu_bytes')))
    destination=root/'summary';destination.mkdir(exist_ok=False)
    write_csv(destination/'metrics.csv',all_metrics);write_csv(destination/'contrasts.csv',all_contrasts)
    write_csv(destination/'reconstruction.csv',explanations)
    write_csv(destination/'reference_metrics.csv',reference_metrics)
    write_csv(destination/'costs.csv',costs)
    write_csv(destination/'inference_costs.csv',pd.read_csv(root/'inference_costs.csv').to_dict('records'))
    write_csv(destination/'explanation_cases.csv',explanation_cases)
    write_csv(destination/'explanation_strata.csv',explanation_strata)
    write_csv(destination/'paired_condition_differences.csv',condition_differences)
    write_csv(destination/'window_contributions_by_acquisition.csv',contributions)
    plot_contrasts(pd.DataFrame(all_contrasts),destination)
    dump(destination/'scope.json',dict(fixture=frozen['fixture'],data_scope='constructed' if frozen['fixture'] else 'declared_industrial_requires_custodian_validation',
         independent_unit_validity='Verified physical specimen declarations are required; software checks their consistency, not the truth of cited evidence.',
         aggregation='Per task; no pooling of windows/seeds/datasets as independent physical units.',
         claims='Report signed contrasts and valid failures; qualification is not a proof of strong target performance.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('command',choices=['bind','condition-audit','preflight','smoke','calibrate','tune','fit','qualify','freeze','test','analyze'])
    p.add_argument('--root');p.add_argument('--study');p.add_argument('--task',action='append');p.add_argument('--output')
    p.add_argument('--fixture',action='store_true');p.add_argument('--device',default='cuda:0')
    p.add_argument('--kind',choices=['baseline','method']);p.add_argument('--family',choices=['reference','baselines','method'])
    p.add_argument('--task-name', action='append', help='Execute a bound task subset in tune/fit only; does not redefine the suite.')
    p.add_argument('--arm', action='append', help='Execute an existing arm subset in tune/fit only.')
    p.add_argument('--seed', action='append', type=int, help='Execute a final seed subset in fit only; HPO remains unchanged.')
    p.add_argument('--retry-interrupted', action='store_true', help='Explicit same-trial restart for incomplete calibration/tune/fit outputs; never retry numerical failures.')
    args=p.parse_args()
    if (args.task_name or args.arm) and args.command not in {'tune','fit'}:
        p.error('Task/arm filtering is limited to tune/fit.')
    if args.retry_interrupted and args.command not in {'calibrate','tune','fit'}:
        p.error('Interrupted-trial retry is limited to calibrate/tune/fit.')
    if args.seed and args.command != 'fit': p.error('--seed filters final fitting only.')
    if args.command=='bind':
        if not args.study or not args.task or not args.output:p.error('bind requires --study, repeated --task and --output')
        bind(args.study,args.task,args.output,args.fixture)
    else:
        if not args.root:p.error('--root required')
        if args.command in {'smoke','tune'} and not (args.kind if args.command=='smoke' else args.family):p.error('Declare --kind or --family')
        try:
            if args.command=='condition-audit':condition_audit(args.root)
            elif args.command=='preflight':preflight(args.root)
            elif args.command=='smoke':smoke(args.root,args.kind,args.device)
            elif args.command=='calibrate':calibrate(args.root,args.device,retry_interrupted=args.retry_interrupted)
            elif args.command=='tune':tune(args.root,args.family,args.device,task_names=args.task_name,selected_arms=args.arm,retry_interrupted=args.retry_interrupted)
            elif args.command=='fit':fit(args.root,args.device,args.family or 'all',task_names=args.task_name,selected_arms=args.arm,selected_seeds=args.seed,retry_interrupted=args.retry_interrupted)
            elif args.command=='qualify':qualify(args.root,args.device)
            elif args.command=='freeze':freeze(args.root,args.device)
            elif args.command=='test':test(args.root,args.device)
            elif args.command=='analyze':analyze(args.root)
        except Exception as exc:
            root=Path(args.root)
            if root.is_dir():
                import time
                dump(root/f'failure_{args.command}_{time.time_ns()}.json',dict(command=vars(args),error_type=type(exc).__name__,error=str(exc)))
            raise

if __name__=='__main__':main()
