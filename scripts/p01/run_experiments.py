"""Bounded P01 experiment driver: existing Model Factory + native readers.

This paper-level driver owns experimental-unit sampling and domain sequencing.
It does not replace PHMFactory's general Pipeline, Trainer or Data Factory.
Real mode requires an installed PHMFactory checkout. Standalone mode is explicit
and restricted to generated demo data; it is never a fallback for a failed run.
"""
from __future__ import annotations
import argparse
import copy
import csv
import importlib
import json
from pathlib import Path
import random
import sys
import time
from types import SimpleNamespace

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from src.task_factory.Components.p01_bias_losses import (
    covariance_loss, retention_loss, unit_mean, replay_der_losses, vrex_loss)

OFFLINE = {
 'raw_cnn': {'reference_kind':'raw_cnn'},
 'stft_cnn': {'reference_kind':'stft_cnn'},
 'tspn_metadata': {'reference_kind':'tspn_metadata'},
 'S1_additive': {'binding':'fixed'},
 'S2_physical': {'binding':'physical'},
 'S3_augmented': {'binding':'physical','augmentation':True},
 'S4_covariance': {'binding':'physical','augmentation':True,'covariance':True},
 'N_wrong_physics': {'binding':'wrong_resonance','augmentation':True,'covariance':True},
 'N_global_alignment': {'binding':'physical','augmentation':True,'covariance':True,'global_alignment':True},
 'A_mlp': {'binding':'physical','head':'mlp'},
 'A_raw_only': {'binding':'physical','paths':['raw']},
 'A_periodic_only': {'binding':'physical','paths':['periodic']},
 'A_envelope_only': {'binding':'physical','paths':['envelope']},
 'A_stft_only': {'binding':'physical','paths':['stft']},
 'vrex': {'binding':'fixed','dg':'vrex'},
 'groupdro': {'binding':'fixed','dg':'groupdro'},
}
CONTINUAL = {'FT':None, 'ER':None, 'DERPP':'derpp',
 'TOTAL_MARGIN':'total_one_sided', 'PATH_SYMMETRIC':'path_symmetric',
 'OURS':'path_one_sided', 'N_WRONG_PATH':'wrong_path'}


def build_model(settings, engine):
    args = SimpleNamespace(**settings)
    if engine == 'phmfactory':
        # No fallback: a missing or broken installed factory fails here.
        from src.model_factory.model_factory import model_factory
        return model_factory(args, metadata=None)
    if engine != 'standalone_demo':
        raise ValueError('engine must be phmfactory or standalone_demo')
    module = importlib.import_module(f'src.model_factory.{args.type}.{args.name}')
    return module.Model(args, metadata=None)


def read_records(dataset, config):
    path = Path(dataset['metadata_file']).expanduser().resolve()
    frame = pd.read_excel(path) if path.suffix.lower() == '.xlsx' else pd.read_csv(path)
    mapping = dataset['columns']  # canonical field -> actual metadata column
    required = ['file','unit_id','label','domain','split','sample_rate_hz','rotation_speed_rpm']
    missing = [key for key in required if key not in mapping or mapping[key] not in frame.columns]
    if missing:
        raise ValueError(f'{path}: missing explicitly mapped columns {missing}')
    if dataset['format'] not in {'npz','phmfactory_reader'}:
        raise ValueError('data format must be npz or phmfactory_reader')
    if dataset['format'] == 'phmfactory_reader' and ('reader' not in mapping or mapping['reader'] not in frame):
        raise ValueError('native reader format requires an explicit reader/Name column')
    records=[]
    for row in frame.to_dict('records'):
        record={key:row[column] for key,column in mapping.items()}
        if any(pd.isna(record[k]) for k in required):
            raise ValueError('null label, physical metadata, split or identity')
        record['unit_id']=str(record['unit_id']); record['domain']=str(record['domain'])
        record['split']=str(record['split'])
        if record['split'] not in {'update','validation','test'}:
            raise ValueError('split must be update, validation or test; never inferred')
        label_map=dataset.get('label_map')
        mapped=label_map[str(record['label'])] if label_map is not None else record['label']
        if isinstance(mapped, bool) or float(mapped) != int(mapped):
            raise ValueError('class labels must be explicit integers or use label_map')
        record['label']=int(mapped)
        if not 0 <= record['label'] < config['model']['num_classes']:
            raise ValueError('label outside configured class range')
        record['sample_rate_hz']=float(record['sample_rate_hz'])
        record['rotation_speed_rpm']=float(record['rotation_speed_rpm'])
        if not all(np.isfinite(record[k]) and record[k]>0 for k in ['sample_rate_hz','rotation_speed_rpm']):
            raise ValueError('positive measured fs/rpm required')
        if dataset['format']=='phmfactory_reader':
            record['path']=str(Path(dataset['data_dir'])/'raw'/str(record['reader'])/str(record['file']))
        else:
            record['path']=str(Path(dataset['data_dir'])/str(record['file']))
        if not Path(record['path']).is_file():
            raise FileNotFoundError(record['path'])
        records.append(record)
    # A physical unit never changes partition, even across operating domains.
    partitions={}; files={}; domain_units=set()
    for r in records:
        u=r['unit_id']
        if u in partitions and partitions[u] != r['split']:
            raise ValueError(f'unit {u} crosses partitions')
        partitions[u]=r['split']
        source=str(Path(r['path']).resolve())
        if source in files:
            raise ValueError(f'duplicate raw file in metadata: {source}')
        files[source]=u
        key=(r['domain'],u)
        if key in domain_units:
            raise ValueError('first implementation requires one acquisition per unit/domain; combine recordings explicitly first')
        domain_units.add(key)
    source=list(map(str,dataset['source_domains'])); sequence=list(map(str,dataset['domain_sequence']))
    if len(set(source+sequence)) != len(source+sequence):
        raise ValueError('source domains and new-domain sequence must be unique and disjoint')
    present={r['domain'] for r in records}
    if set(source+sequence) != present:
        raise ValueError('declare every metadata domain exactly once as source or sequence')
    for d in source+sequence:
        for part in ['update','validation','test']:
            if not any(r['domain']==d and r['split']==part for r in records):
                raise ValueError(f'domain {d} lacks {part} units')
    source_classes={r['label'] for r in records if r['domain'] in source and r['split']=='update'}
    if source_classes != set(range(config['model']['num_classes'])):
        raise ValueError('source update domains must cover the fixed label space')
    return records


def window_record(record, dataset, data_cfg):
    if dataset['format']=='npz':
        with np.load(record['path'],allow_pickle=False) as archive:
            signal=archive[data_cfg.get('array_key','x')]
    else:
        reader=importlib.import_module('src.data_factory.reader.'+str(record['reader']))
        signal=reader.read(record['path'],SimpleNamespace(**dataset.get('reader_args',{})))
    for axis in sorted(data_cfg.get('squeeze_axes',[]),reverse=True):
        signal=np.squeeze(signal,axis=int(axis))
    layout=data_cfg['layout']
    if layout=='L':
        if signal.ndim != 1: raise ValueError(f'expected L, got {signal.shape}')
        signal=signal[:,None]
    elif layout=='CL':
        if signal.ndim != 2: raise ValueError(f'expected CL, got {signal.shape}')
        signal=signal.T
    elif layout!='LC' or signal.ndim!=2:
        raise ValueError(f'expected declared layout {layout}, got {signal.shape}')
    signal=np.asarray(signal,dtype=np.float32)
    if not np.isfinite(signal).all(): raise ValueError('nonfinite raw signal')
    length=int(data_cfg['window_size']); count=int(data_cfg['windows_per_unit'])
    if len(signal)<length or count<1: raise ValueError('record too short or invalid window count')
    starts=np.linspace(0,len(signal)-length,count,dtype=np.int64)
    if len(np.unique(starts))!=count: raise ValueError('requested duplicate windows; reduce windows_per_unit')
    # No train/test transform or normalization is fitted on these windows.
    return torch.from_numpy(np.stack([signal[s:s+length] for s in starts]))


def materialize(records,dataset,cfg):
    return [dict(r,x=window_record(r,dataset,cfg['data'])) for r in records]


def pack_batch(units,device):
    x=torch.cat([u['x'] for u in units]).to(device)
    counts=[len(u['x']) for u in units]
    y=torch.cat([torch.full((n,),u['label'],dtype=torch.long) for u,n in zip(units,counts)]).to(device)
    unit_ids=torch.cat([torch.full((n,),i,dtype=torch.long) for i,n in enumerate(counts)]).to(device)
    meta={key:torch.cat([torch.full((n,),u[key]) for u,n in zip(units,counts)]).to(device)
          for key in ['sample_rate_hz','rotation_speed_rpm']}
    return x,y,unit_ids,meta


def macro_f1(labels,predictions,classes):
    scores=[]
    for c in range(classes):
        tp=sum(y==c and p==c for y,p in zip(labels,predictions))
        fp=sum(y!=c and p==c for y,p in zip(labels,predictions))
        fn=sum(y==c and p!=c for y,p in zip(labels,predictions))
        scores.append(2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.0)
    return float(np.mean(scores))


def evaluate(model,units,device,classes):
    model.eval(); rows=[]
    with torch.no_grad():
        for unit in units:
            x,y,ids,meta=pack_batch([unit],device)
            out=model.forward_details(x,physical_metadata=meta)
            logits=out['logits']; prob=logits.softmax(-1); mean_prob=prob.mean(0)
            brier=(prob-F.one_hot(y,classes)).square().sum(-1).mean()
            rows.append({'unit_id':unit['unit_id'],'domain':unit['domain'],'label':unit['label'],
                         'prediction':int(mean_prob.argmax()),'ce':float(F.cross_entropy(logits,y)),
                         'brier':float(brier),'probabilities':mean_prob.cpu().tolist()})
    represented=sorted({r['label'] for r in rows})
    balanced=float(np.mean([np.mean([r['brier'] for r in rows if r['label']==c]) for c in represented]))
    summary={'brier_class_balanced':balanced,'represented_classes':len(represented),'ce':float(np.mean([r['ce'] for r in rows])),
             'brier':float(np.mean([r['brier'] for r in rows])),
             'accuracy':float(np.mean([r['label']==r['prediction'] for r in rows])),
             'macro_f1':macro_f1([r['label'] for r in rows],[r['prediction'] for r in rows],classes)}
    return summary,rows


def train_stage(model,units,validation,cfg,device,rng,arm,*,memory=(),teacher=None,retention=None,store_history=False,stage_seed=0):
    optimizer=torch.optim.Adam(model.parameters(),lr=float(cfg['training']['lr']))
    best=float('inf'); best_state=None; log=[]; historical={}
    # Separate streams preserve current-unit ordering across replay/augmentation arms.
    augmentation_rng=random.Random(stage_seed+2000)
    replay_rng=random.Random(stage_seed+3000)
    domains=sorted({u['domain'] for u in units}); dro=torch.full((len(domains),),1/len(domains),device=device)
    if arm.get('dg') in {'vrex','groupdro'} and len(domains)<2:
        raise ValueError('DG penalty requires at least two source training domains')
    pools={d:[u for u in units if u['domain']==d] for d in domains}
    if int(cfg['training']['steps_per_epoch']) < max(map(len,pools.values())):
        raise ValueError('steps_per_epoch must cover every update unit once; increase it before training')
    for epoch in range(int(cfg['training']['epochs'])):
        model.train(); totals=[]
        for pool in pools.values(): rng.shuffle(pool)
        for step in range(int(cfg['training']['steps_per_epoch'])):
            current=[pools[d][step % len(pools[d])] for d in domains]
            x,y,ids,meta=pack_batch(current,device)
            out=model.forward_details(x,physical_metadata=meta)
            # Capture the first-seen pre-update logits; do not refresh DER targets.
            offset=0
            for unit in current:
                key=(unit['domain'],unit['unit_id']); n=len(unit['x'])
                if store_history and key not in historical:
                    historical[key]=out['logits'][offset:offset+n].detach().cpu().clone()
                offset+=n
            ce=F.cross_entropy(out['logits'],y,reduction='none')
            loss=unit_mean(ce,ids)
            if arm.get('dg'):
                dids=torch.cat([torch.full((len(u['x']),),domains.index(u['domain']),device=device) for u in current])
                if arm['dg']=='vrex':
                    mean,var=vrex_loss(ce,dids)
                    loss=mean+float(cfg['training']['dg_weight'])*var
                else:
                    risks=torch.stack([ce[dids==d].mean() for d in range(len(domains))])
                    with torch.no_grad():
                        dro*=torch.exp(float(cfg['training']['dro_eta'])*risks.detach()); dro/=dro.sum()
                    loss=(dro*risks).sum()
            if arm.get('augmentation'):
                # Declared circular time-origin shift, not fabricated acceleration.
                shift=augmentation_rng.randint(1,int(cfg['training']['shift_samples']))
                paired=model.forward_details(x.roll(shift,dims=1),physical_metadata=meta)
                loss=(loss+unit_mean(F.cross_entropy(paired['logits'],y,reduction='none'),ids))/2
                if arm.get('covariance'):
                    loss=loss+float(cfg['training']['cov_weight'])*covariance_loss(
                        out['contributions'],paired['contributions'],ids,
                        global_alignment=bool(arm.get('global_alignment',False)))
            if memory:
                replay=replay_rng.sample(list(memory),min(len(current),len(memory)))
                rx,ry,rids,rmeta=pack_batch(replay,device)
                replay_out=model.forward_details(rx,physical_metadata=rmeta)
                if retention=='derpp':
                    targets=torch.cat([u['historical_logits'] for u in replay]).to(device)
                    rce,mse=replay_der_losses(replay_out['logits'],targets,ry,rids)
                    loss=loss+float(cfg['training']['replay_weight'])*rce+float(cfg['training']['der_weight'])*mse
                else:
                    loss=loss+float(cfg['training']['replay_weight'])*unit_mean(
                        F.cross_entropy(replay_out['logits'],ry,reduction='none'),rids)
                    if retention is not None:
                        if teacher is None: raise ValueError('retention requires a frozen previous-stage teacher')
                        with torch.no_grad(): old=teacher.forward_details(rx,physical_metadata=rmeta)
                        if old['path_names']!=replay_out['path_names']: raise ValueError('teacher/student path ordering mismatch')
                        loss=loss+float(cfg['training']['keep_weight'])*retention_loss(
                            replay_out['contributions'],old['contributions'],ry,rids,retention)
            if not torch.isfinite(loss): raise FloatingPointError('nonfinite total objective')
            optimizer.zero_grad(set_to_none=True); loss.backward()
            if any(p.grad is not None and not torch.isfinite(p.grad).all() for p in model.parameters()):
                raise FloatingPointError('nonfinite gradient')
            optimizer.step(); totals.append(float(loss.detach()))
        score,_=evaluate(model,validation,device,int(cfg['model']['num_classes']))
        log.append({'epoch':epoch,'train_total':float(np.mean(totals)),'validation_ce':score['ce']})
        if score['ce']<best:
            best=score['ce']; best_state=copy.deepcopy(model.state_dict())
    if best_state is None: raise RuntimeError('no selected checkpoint')
    model.load_state_dict(best_state)
    if store_history and len(historical) != len(units):
        raise RuntimeError('a unit lacks its first-seen trajectory target')
    return log,historical


def update_memory(memory,new_units,history,seen,capacity,rng):
    # Unit-reservoir, shared across methods. Stored logits never get recomputed.
    for unit in sorted(new_units,key=lambda u:(u['domain'],u['unit_id'])):
        seen+=1
        item=dict(unit)
        if history:
            item['historical_logits']=history[(unit['domain'],unit['unit_id'])]
        if len(memory)<capacity: memory.append(item)
        else:
            slot=rng.randrange(seen)
            if slot<capacity: memory[slot]=item
    return seen


def dump_csv(path,rows):
    if not rows: raise ValueError('no results to write')
    with path.open('w',newline='',encoding='utf8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def report(model,records,dataset,cfg,device,output,stage,timing,domains):
    result=[]; predictions=[]
    for domain in domains:
        selected=[r for r in records if r['domain']==domain and r['split']=='test']
        summary,rows=evaluate(model,materialize(selected,dataset,cfg),device,cfg['model']['num_classes'])
        result.append({'stage':stage,'timing':timing,'domain':domain,'units':len(rows),**summary})
        for row in rows:
            row.update(stage=stage,timing=timing)
            row['probabilities']=json.dumps(row['probabilities'])
            predictions.append(row)
    return result,predictions


def run_one(cfg,dataset,mode,arm_name,seed,output,device):
    if output.exists(): raise FileExistsError(f'{output}: use a new output root; runs are not silently overwritten')
    output.mkdir(parents=True)
    torch.manual_seed(seed); np.random.seed(seed); random.seed(seed)
    rng=random.Random(seed); memory_rng=random.Random(seed+1000)
    records=read_records(dataset,cfg)
    sources=list(map(str,dataset['source_domains'])); future=list(map(str,dataset['domain_sequence']))
    arm=copy.deepcopy(OFFLINE[arm_name]) if mode=='offline' else copy.deepcopy(OFFLINE['S4_covariance'])
    settings=copy.deepcopy(cfg['model'])
    for key in ['binding','paths','head','reference_kind']:
        if key in arm: settings[key]=arm[key]
    settings['type']='X_model'; settings['name']='P01Reference' if 'reference_kind' in arm else 'P01OperatorBias'
    model=build_model(settings,cfg['engine']).to(device)
    initial=[r for r in records if r['domain'] in sources and r['split']=='update']
    validation=[r for r in records if r['domain'] in sources and r['split']=='validation']
    units=materialize(initial,dataset,cfg); val=materialize(validation,dataset,cfg)
    start=time.perf_counter()
    store_history=mode=='continual' and arm_name=='DERPP'
    use_memory=mode=='continual' and arm_name!='FT'
    logs,history=train_stage(model,units,val,cfg,device,rng,arm,store_history=store_history,stage_seed=seed)
    dump_csv(output/'train_initial.csv',logs)
    memory=[];seen=0
    if use_memory:
        seen=update_memory(memory,units,history,seen,cfg['continual']['memory_units'],memory_rng)
    torch.save({'state_dict':model.state_dict(),'model':settings},output/'stage_0.pt')
    results,predictions=report(model,records,dataset,cfg,device,output,0,'post_initial',sources)
    if mode=='offline':
        r,p=report(model,records,dataset,cfg,device,output,0,'unseen',future);results+=r;predictions+=p
    else:
        for stage,domain in enumerate(future,1):
            # Predictions precede use of any new-domain update/validation signal.
            r,p=report(model,records,dataset,cfg,device,output,stage,'pre_update',[domain]);results+=r;predictions+=p
            retention=CONTINUAL[arm_name]
            teacher=copy.deepcopy(model).eval().requires_grad_(False) if retention not in {None,'derpp'} else None
            update=materialize([u for u in records if u['domain']==domain and u['split']=='update'],dataset,cfg)
            val=materialize([u for u in records if u['domain']==domain and u['split']=='validation'],dataset,cfg)
            log,history=train_stage(model,update,val,cfg,device,rng,arm,
                memory=memory,teacher=teacher,retention=retention,
                store_history=store_history,stage_seed=seed+stage)
            dump_csv(output/f'train_stage_{stage}.csv',log)
            if use_memory:
                seen=update_memory(memory,update,history,seen,cfg['continual']['memory_units'],memory_rng)
            r,p=report(model,records,dataset,cfg,device,output,stage,'post_update',sources+future[:stage]);results+=r;predictions+=p
            torch.save({'state_dict':model.state_dict(),'model':settings},output/f'stage_{stage}.pt')
    torch.save({'state_dict':model.state_dict(),'model':settings},output/'selected_model.pt')
    dump_csv(output/'metrics.csv',results);dump_csv(output/'predictions.csv',predictions)
    stats={'status':'completed','run_kind':'synthetic_demo' if cfg['engine']=='standalone_demo' else 'research_unvalidated',
           'dataset':dataset['name'],'arm':arm_name,'mode':mode,'seed':seed,
           'parameters':sum(p.numel() for p in model.parameters()),'seconds':time.perf_counter()-start,
           'memory_records':len(memory),'memory_unique_units':len({u['unit_id'] for u in memory}),
           'memory_raw_bytes':sum(u['x'].numel()*u['x'].element_size() for u in memory),
           'stored_logit_bytes':sum(u['historical_logits'].numel()*u['historical_logits'].element_size() for u in memory if 'historical_logits' in u),
           'teacher_parameter_bytes':sum(p.numel()*p.element_size() for p in model.parameters()) if mode=='continual' and CONTINUAL[arm_name] not in {None,'derpp'} else 0,
           'der_protocol':'historical-logit objective with unit reservoir and domain-batched updates; not original online benchmark'}
    (output/'summary.json').write_text(json.dumps(stats,indent=2),encoding='utf8')
    (output/'config.yaml').write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf8')
    print(json.dumps(stats),flush=True)


def generate_demo(output):
    data=output/'demo_data';data.mkdir(parents=True,exist_ok=False)
    rows=[]; fs=2048.; rng=np.random.default_rng(712)
    for di,speed in enumerate([8.,10.,14.,18.]):
        for split,count in [('update',6),('validation',3),('test',3)]:
            for i in range(count):
                label=i%3; n=4096; t=np.arange(n)/fs; phase=rng.uniform(0,2*np.pi)
                order=[3.,5.,7.][label]
                env=1+.5*np.cos(2*np.pi*order*speed*t+phase)
                signal=.35*np.cos(2*np.pi*order*speed*t+phase)+.3*env*np.cos(2*np.pi*320*t)+.05*rng.normal(size=n)
                name=f'd{di}_{split}_{i}.npz';np.savez(data/name,x=signal[:,None].astype(np.float32))
                rows.append({'file':name,'unit_id':name[:-4],'label':label,'domain':f'd{di}','split':split,
                             'sample_rate_hz':fs,'rotation_speed_rpm':speed*60})
    dump_csv(data/'records.csv',rows)
    return {'name':'synthetic','format':'npz','data_dir':str(data),'metadata_file':str(data/'records.csv'),
            'columns':{k:k for k in rows[0]},'source_domains':['d0','d1'],'domain_sequence':['d2','d3']}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',required=True);parser.add_argument('--mode',choices=['offline','continual'],default='offline')
    parser.add_argument('--arms',nargs='+');parser.add_argument('--seeds',type=int,nargs='+')
    parser.add_argument('--output',required=True);parser.add_argument('--device',default='cpu')
    parser.add_argument('--demo',action='store_true');parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    cfg=yaml.safe_load(Path(args.config).read_text(encoding='utf8'))
    if args.device.startswith('cuda'):
        visible=__import__('os').environ.get('CUDA_VISIBLE_DEVICES')
        if visible is None or ',' in visible or visible=='2':
            raise ValueError('set exactly one physical CUDA_VISIBLE_DEVICES, not GPU 2')
        if not torch.cuda.is_available(): raise RuntimeError('requested CUDA is unavailable')
    torch.set_num_threads(int(cfg['training'].get('cpu_threads',1)))
    output=Path(args.output).resolve()
    if args.demo:
        if cfg['engine']!='standalone_demo': raise ValueError('--demo requires engine=standalone_demo')
        if output.exists(): raise FileExistsError(output)
        output.mkdir(parents=True)
        cfg['datasets']=[generate_demo(output)]
    elif cfg['engine']!='phmfactory':
        raise ValueError('real data require engine=phmfactory; standalone is not a fallback')
    arms=args.arms or cfg[args.mode+'_arms']; allowed=OFFLINE if args.mode=='offline' else CONTINUAL
    if set(arms)-set(allowed): raise ValueError(f'unknown arm(s): {set(arms)-set(allowed)}')
    if not cfg.get('datasets'): raise ValueError('configure at least one real dataset')
    for dataset in cfg['datasets']:
        for seed in args.seeds or cfg['seeds']:
            for arm in arms:
                path=output/dataset['name']/args.mode/arm/f'seed_{seed}'
                if args.dry_run:
                    read_records(dataset,cfg)
                    print(f'planned {path}')
                    continue
                try:
                    run_one(cfg,dataset,args.mode,arm,seed,path,torch.device(args.device))
                except Exception as error:
                    # Preserve the failure and re-raise; no substitute model/data/run.
                    if path.exists() and not (path/'summary.json').exists():
                        (path/'failure.txt').write_text(f'{type(error).__name__}: {error}\n',encoding='utf8')
                    raise


if __name__=='__main__':
    main()
