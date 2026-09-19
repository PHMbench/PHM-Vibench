"""P01 physical-group sampling and estimates over the existing H5/window access.

A row is one constant-label acquisition. Multiple rows may belong to one physical
specimen. Independent assessment is an explicit partition, never validation.
"""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
import re
import numpy as np
import pandas as pd


def read_records(dataset, config):
    from experiments.p01.window_io import _h5_metadata_frame, _vibench_id
    from src.data_factory.H5DataDict import H5DataDict
    if dataset['format'] != 'vibench_h5':
        raise ValueError('Fusion uses the existing Vibench H5, not a second waveform format.')
    path=Path(dataset['metadata_file']).expanduser().resolve()
    read=pd.read_excel if path.suffix.lower()=='.xlsx' else pd.read_csv
    frame=read(path,dtype=str,keep_default_na=False)
    mapping=dataset['columns']; idcol=mapping['id']
    speed_pattern = dataset.get('rotation_speed_pattern')
    if speed_pattern is not None:
        speed_pattern = re.compile(speed_pattern)
        if 'rpm' not in speed_pattern.groupindex:
            raise ValueError('rotation_speed_pattern must explicitly capture the named rpm group.')
    selection=dict(dataset); selection.pop('protocol_file',None)
    selection['select']={k:[_vibench_id(v) if k==idcol else str(v) for v in vals]
                         for k,vals in dataset.get('select',{}).items()}
    frame=_h5_metadata_frame(selection,frame,mapping)
    if dataset.get('protocol_file'):
        protocol=pd.read_csv(Path(dataset['protocol_file']).expanduser(),dtype=str,keep_default_na=False)
        fields=[mapping['unit_id'],mapping['split']]
        if set(protocol)!=set([idcol,*fields]) or any(k in frame for k in fields):
            raise ValueError('Grouping table supplies only previously absent Id/unit/split columns.')
        protocol[idcol]=protocol[idcol].map(_vibench_id)
        frame=frame.merge(protocol,on=idcol,how='left',validate='one_to_one')
    required=('id','unit_id','label','domain','split','sample_rate_hz','rotation_speed_rpm')
    if any(k not in mapping or mapping[k] not in frame for k in required):
        raise ValueError('Map the existing Id, physical group, label, condition, partition and physical units.')
    h5=Path(dataset['h5_file']).expanduser().resolve();records=[];partitions={};seen=set()
    with H5DataDict(str(h5)) as signals:
        for row in frame.to_dict('records'):
            r={k:row[col] for k,col in mapping.items()}
            if any(pd.isna(r[k]) or str(r[k])=='' for k in required):
                raise ValueError('Missing physical identity, partition, label or measurement.')
            r['unit_id']=str(r['unit_id']);r['domain']=str(r['domain']);r['split']=str(r['split'])
            if r['split'] not in {'update','validation','assessment','test'}:
                raise ValueError('Use update, validation, assessment or test explicitly.')
            u=r['unit_id']
            if u in partitions and partitions[u]!=r['split']:
                raise ValueError(f'Physical group {u} crosses partitions.')
            partitions[u]=r['split']
            value=dataset['label_map'][str(r['label'])] if dataset.get('label_map') else r['label']
            if isinstance(value,bool) or float(value)!=int(value) or not 0<=int(value)<int(config['model']['num_classes']):
                raise ValueError('Invalid explicit class mapping.')
            r['label']=int(value)
            if speed_pattern is not None:
                measured = speed_pattern.fullmatch(str(r['rotation_speed_rpm']))
                if measured is None:
                    raise ValueError(f"Observed RPM metadata does not match the declared unit pattern: {r['rotation_speed_rpm']!r}")
                r['rotation_speed_rpm'] = measured.group('rpm')
            for k in ('sample_rate_hz','rotation_speed_rpm'):
                r[k]=float(r[k])
                if not np.isfinite(r[k]) or r[k]<=0:raise ValueError(f'Invalid measured {k}.')
            r['h5_key']=_vibench_id(r['id']);r['path']=str(h5)
            r['acquisition_id']=r['h5_key']
            if r['h5_key'] in seen:raise ValueError('Repeated H5 acquisition.')
            seen.add(r['h5_key'])
            if r['h5_key'] not in signals:raise KeyError(f"Missing H5 Id {r['h5_key']}")
            records.append(r)
    sources=list(map(str,dataset['source_domains']));future=list(map(str,dataset['domain_sequence']))
    declared=sources+future
    if not sources or len(set(declared))!=len(declared):
        raise ValueError('Declare unique, disjoint source and future conditions.')
    if set(declared)!={r['domain'] for r in records}:
        raise ValueError('Declare every selected condition.')
    for d in sources:
        for split in ('update','validation'):
            if not any(r['domain']==d and r['split']==split for r in records):
                raise ValueError(f'{d} lacks {split} acquisitions.')
    for d in declared:
        if not any(r['domain']==d and r['split']=='test' for r in records):
            raise ValueError(f'{d} lacks permanent test acquisitions; declared conditions cannot disappear from evaluation.')
    if {r['label'] for r in records if r['domain'] in sources and r['split']=='update'}!=set(range(int(config['model']['num_classes']))):
        raise ValueError('Source training must cover the declared classes.')
    return records


def group_pools(units, domains):
    pools={d:defaultdict(list) for d in domains}
    for r in units:pools[r['domain']][r['unit_id']].append(r)
    return pools


def sample_units(pools,count,rng):
    """Uniform groups, then uniform acquisition; no extra weight for repeat files."""
    return [rng.choice(pools[d][g]) for d in pools
            for g in rng.sample(sorted(pools[d]),count)]


def summarize_rows(rows, classes):
    """Group-balanced window risk and acquisition-level class confusion.

    Older tables with one acquisition/group use unit_id as their acquisition key.
    A subject with multiple labels is never assigned one averaged class label.
    """
    if not rows:raise ValueError('No acquisition predictions.')
    buckets=defaultdict(list);seen=set()
    for r in rows:
        key=(r['domain'],r['predictor'],str(r['unit_id']),str(r.get('acquisition_id',r['unit_id'])))
        if key in seen:raise ValueError('Duplicate classification-unit estimates.')
        seen.add(key);buckets[key[:2]].append(r)
        if any(not 0<=int(r[k])<classes or float(r[k])!=int(r[k]) for k in ('label','prediction')):
            raise ValueError('Invalid class in acquisition estimate.')
        if not np.isfinite([r['ce'],r['brier']]).all() or r['ce']<0 or not 0<=r['brier']<=2+1e-6:
            raise ValueError('Invalid acquisition risk.')
    result=[]
    for (d,predictor),part in sorted(buckets.items()):
        groups=defaultdict(list)
        for r in part:groups[str(r['unit_id'])].append(r)
        cm=np.zeros((classes,classes),float)
        for group in groups.values():
            for r in group:cm[int(r['label']),int(r['prediction'])]+=1/len(group)
        den=cm.sum(0)+cm.sum(1)
        result.append(dict(domain=d,predictor=predictor,units=len(groups),
                           ce=float(np.mean([np.mean([r['ce'] for r in g]) for g in groups.values()])),
                           brier=float(np.mean([np.mean([r['brier'] for r in g]) for g in groups.values()])),
                           accuracy=float(np.trace(cm)/cm.sum()),
                           macro_f1=float(np.divide(2*np.diag(cm),den,out=np.zeros(classes),where=den!=0).mean())))
    return result
